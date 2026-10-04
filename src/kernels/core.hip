// Correctness-first GPU kernels for single-token decode (P2). Each mirrors an
// operation of the CPU reference (src/model/*.cpp) and is tested against it;
// speed comes in P3. Wave32 (gfx1151). FP32 accumulation throughout.
//
// ABI (src/rt/device.hpp): pointer parameters are buffers, everything else is
// a by-value scalar. Kernels are extern "C" so names are stable.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/step_params.h"

namespace {

constexpr int kWarp = 32;

__device__ inline float WarpSum(float v) {
  for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, kWarp);
  return v;
}

// Block-wide sum for up to 1024 threads; `scratch` holds 32 floats.
__device__ inline float BlockSum(float v, float* scratch) {
  v = WarpSum(v);
  const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + kWarp - 1) / kWarp;
  v = threadIdx.x < warps ? scratch[threadIdx.x] : 0.0f;
  if (warp == 0) v = WarpSum(v);
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  const float r = scratch[0];
  __syncthreads();
  return r;
}

__device__ inline float BlockMax(float v, float* scratch) {
  for (int o = kWarp / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, kWarp));
  const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + kWarp - 1) / kWarp;
  v = threadIdx.x < warps ? scratch[threadIdx.x] : -INFINITY;
  if (warp == 0)
    for (int o = kWarp / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, kWarp));
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  const float r = scratch[0];
  __syncthreads();
  return r;
}

__device__ inline float Bf16(std::uint16_t h) { return __uint_as_float(static_cast<std::uint32_t>(h) << 16); }
__device__ inline float F16(std::uint16_t h) { return static_cast<float>(*reinterpret_cast<const _Float16*>(&h)); }
__device__ inline float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float Silu(float x) { return x * Sigmoid(x); }
__device__ inline float Softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// Element loaders: value of column c in a row starting at `row`.
struct LoadF32 {
  __device__ static float At(const std::uint8_t* row, std::uint32_t c) { return reinterpret_cast<const float*>(row)[c]; }
};
struct LoadBf16 {
  __device__ static float At(const std::uint8_t* row, std::uint32_t c) {
    return Bf16(reinterpret_cast<const std::uint16_t*>(row)[c]);
  }
};
struct LoadF16 {
  __device__ static float At(const std::uint8_t* row, std::uint32_t c) {
    return F16(reinterpret_cast<const std::uint16_t*>(row)[c]);
  }
};
struct LoadQ8_0 {  // 34-byte blocks: fp16 scale + 32 int8
  __device__ static float At(const std::uint8_t* row, std::uint32_t c) {
    const std::uint8_t* b = row + (c / 32) * 34;
    const std::uint16_t d = static_cast<std::uint16_t>(b[0] | (b[1] << 8));
    return F16(d) * static_cast<float>(static_cast<std::int8_t>(b[2 + c % 32]));
  }
};

__device__ inline std::uint16_t U16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

// K-quant 6-bit scale/min unpacking (ggml get_scale_min_k4).
__device__ inline void ScaleMinK4(std::uint32_t j, const std::uint8_t* q, std::uint32_t& sc, std::uint32_t& m) {
  if (j < 4) {
    sc = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// The same unpacking from the header's three scale words (bytes 4..15 of a
// K-quant block) with shifts and selects: indexing a local byte array by a
// runtime j forces it into scratch memory.
__device__ inline std::uint32_t ScaleByte(std::uint32_t i, std::uint32_t w0, std::uint32_t w1, std::uint32_t w2) {
  const std::uint32_t w = i < 4 ? w0 : (i < 8 ? w1 : w2);
  return (w >> (8 * (i & 3))) & 0xFF;
}
__device__ inline void ScaleMinK4Words(std::uint32_t j, std::uint32_t w0, std::uint32_t w1, std::uint32_t w2,
                                       std::uint32_t& sc, std::uint32_t& m) {
  if (j < 4) {
    sc = ScaleByte(j, w0, w1, w2) & 63;
    m = ScaleByte(j + 4, w0, w1, w2) & 63;
  } else {
    sc = (ScaleByte(j + 4, w0, w1, w2) & 0xF) | ((ScaleByte(j - 4, w0, w1, w2) >> 6) << 4);
    m = (ScaleByte(j + 4, w0, w1, w2) >> 4) | ((ScaleByte(j, w0, w1, w2) >> 6) << 4);
  }
}

struct LoadQ4_K {  // 144-byte blocks of 256: d, dmin, scales[12], qs[128]
  __device__ static float At(const std::uint8_t* row, std::uint32_t c) {
    const std::uint8_t* b = row + (c / 256) * 144;
    const std::uint32_t e = c % 256, chunk = e / 64, r = e % 64, hi = r / 32, l = r % 32;
    std::uint32_t sc, m;
    ScaleMinK4(2 * chunk + hi, b + 4, sc, m);
    const std::uint8_t qb = b[16 + chunk * 32 + l];
    const std::uint32_t q = hi ? (qb >> 4) : (qb & 0xF);
    return F16(U16(b)) * static_cast<float>(sc) * static_cast<float>(q) - F16(U16(b + 2)) * static_cast<float>(m);
  }
};
struct LoadQ5_K {  // 176-byte blocks: d, dmin, scales[12], qh[32], qs[128]
  __device__ static float At(const std::uint8_t* row, std::uint32_t c) {
    const std::uint8_t* b = row + (c / 256) * 176;
    const std::uint32_t e = c % 256, chunk = e / 64, r = e % 64, hi = r / 32, l = r % 32;
    std::uint32_t sc, m;
    ScaleMinK4(2 * chunk + hi, b + 4, sc, m);
    const std::uint8_t qb = b[48 + chunk * 32 + l];
    std::uint32_t q = hi ? (qb >> 4) : (qb & 0xF);
    if (b[16 + l] & (1u << (2 * chunk + hi))) q += 16;
    return F16(U16(b)) * static_cast<float>(sc) * static_cast<float>(q) - F16(U16(b + 2)) * static_cast<float>(m);
  }
};
struct LoadQ6_K {  // 210-byte blocks: ql[128], qh[64], scales[16] (int8), d
  __device__ static float At(const std::uint8_t* row, std::uint32_t c) {
    const std::uint8_t* b = row + (c / 256) * 210;
    const std::uint32_t e = c % 256, n = e / 128, r = e % 128, quarter = r / 32, l = r % 32, is = l / 16;
    const std::uint8_t* ql = b + n * 64;
    const std::uint8_t* qh = b + 128 + n * 32;
    const auto* sc = reinterpret_cast<const std::int8_t*>(b + 192 + n * 8);
    const std::uint8_t lo = quarter == 0 ? (ql[l] & 0xF) : quarter == 1 ? (ql[l + 32] & 0xF)
                            : quarter == 2 ? (ql[l] >> 4) : (ql[l + 32] >> 4);
    const int q = static_cast<int>(lo | (((qh[l] >> (2 * quarter)) & 3) << 4)) - 32;
    return F16(U16(b + 208)) * static_cast<float>(sc[is + 2 * quarter]) * static_cast<float>(q);
  }
};
struct LoadQ5_1 {  // 24-byte blocks of 32: d, m, qh (u32), qs[16]
  __device__ static float At(const std::uint8_t* row, std::uint32_t c) {
    const std::uint8_t* b = row + (c / 32) * 24;
    const std::uint32_t j = c % 32;
    const std::uint32_t qh = static_cast<std::uint32_t>(b[4]) | (static_cast<std::uint32_t>(b[5]) << 8) |
                             (static_cast<std::uint32_t>(b[6]) << 16) | (static_cast<std::uint32_t>(b[7]) << 24);
    const std::uint32_t q = j < 16 ? ((b[8 + j] & 0xF) | (((qh >> j) & 1) << 4))
                                   : ((b[8 + j - 16] >> 4) | (((qh >> j) & 1) << 4));
    return static_cast<float>(q) * F16(U16(b)) + F16(U16(b + 2));
  }
};
__constant__ std::int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
struct LoadIQ4_NL {  // 18-byte blocks of 32: d, qs[16]
  __device__ static float At(const std::uint8_t* row, std::uint32_t c) {
    const std::uint8_t* b = row + (c / 32) * 18;
    const std::uint32_t j = c % 32;
    const std::uint8_t qb = b[2 + (j % 16)];
    return F16(U16(b)) * static_cast<float>(kIq4Nl[j < 16 ? (qb & 0xF) : (qb >> 4)]);
  }
};

// y[r] = sum_c W[r][c] x[c]; one wave per row, 8 rows per 256-thread block.
template <typename L>
__device__ void MatVec(const std::uint8_t* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols,
                       std::uint64_t row_bytes) {
  const std::uint32_t r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
  if (r >= rows) return;
  const std::uint8_t* row = w + r * row_bytes;
  float acc = 0.0f;
  for (std::uint32_t c = threadIdx.x % kWarp; c < cols; c += kWarp) acc += L::At(row, c) * x[c];
  acc = WarpSum(acc);
  if (threadIdx.x % kWarp == 0) y[r] = acc;
}

template <typename L>
__device__ void Embed(const std::uint8_t* w, float* out, std::uint32_t token, std::uint32_t cols, std::uint64_t row_bytes) {
  const std::uint8_t* row = w + token * row_bytes;
  for (std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x)
    out[c] = L::At(row, c);
}

}  // namespace

#define HALO_MATVEC(NAME, L)                                                                                   \
  extern "C" __global__ void NAME(const std::uint8_t* w, const float* x, float* y, std::uint32_t rows,         \
                                  std::uint32_t cols, std::uint64_t row_bytes) {                               \
    MatVec<L>(w, x, y, rows, cols, row_bytes);                                                                 \
  }                                                                                                            \
  extern "C" __global__ void NAME##_expert(const std::uint8_t* w, const std::uint32_t* ids, const float* x, float* y,  \
                                           std::uint32_t slot, std::uint64_t expert_bytes, std::uint32_t rows,      \
                                           std::uint32_t cols, std::uint64_t row_bytes) {                           \
    MatVec<L>(w + ids[slot] * expert_bytes, x, y, rows, cols, row_bytes);                                         \
  }                                                                                                            \
  extern "C" __global__ void NAME##_embed_p(const std::uint8_t* w, float* out, const std::uint32_t* params,       \
                                            std::uint32_t cols, std::uint64_t row_bytes) {                          \
    Embed<L>(w, out, params[kHaloToken], cols, row_bytes);                                                        \
  }                                                                                                            \
  extern "C" __global__ void NAME##_embed(const std::uint8_t* w, float* out, std::uint32_t token,              \
                                          std::uint32_t cols, std::uint64_t row_bytes) {                       \
    Embed<L>(w, out, token, cols, row_bytes);                                                                  \
  }

HALO_MATVEC(halo_matvec_f32, LoadF32)
HALO_MATVEC(halo_matvec_bf16, LoadBf16)
HALO_MATVEC(halo_matvec_f16, LoadF16)
HALO_MATVEC(halo_matvec_q8_0, LoadQ8_0)
HALO_MATVEC(halo_matvec_q4_k, LoadQ4_K)
HALO_MATVEC(halo_matvec_q5_k, LoadQ5_K)
HALO_MATVEC(halo_matvec_q6_k, LoadQ6_K)
HALO_MATVEC(halo_matvec_q5_1, LoadQ5_1)
HALO_MATVEC(halo_matvec_iq4_nl, LoadIQ4_NL)

// RMSNorm over n_vec vectors of length d (one block each): x * w / rms.
// Vector b uses weights w + b * w_stride (0 = shared gamma; d = grouped norm
// with one gamma per hyper-connection stream).
extern "C" __global__ void halo_rmsnorm(const float* x, const float* w_all, float* y, std::uint32_t d, float eps,
                                        std::uint32_t w_stride) {
  __shared__ float scratch[32];
  const float* xv = x + blockIdx.x * d;
  const float* w = w_all + blockIdx.x * w_stride;
  float* yv = y + blockIdx.x * d;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += xv[i] * xv[i];
  ss = BlockSum(ss, scratch);
  const float scale = 1.0f / sqrtf(ss / static_cast<float>(d) + eps);
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) yv[i] = xv[i] * scale * w[i];
}

// In-place x / sqrt(sum x^2 + eps) per vector (DeltaNet q/k).
extern "C" __global__ void halo_l2norm(float* x, std::uint32_t d, float eps) {
  __shared__ float scratch[32];
  float* xv = x + blockIdx.x * d;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += xv[i] * xv[i];
  ss = BlockSum(ss, scratch);
  const float scale = 1.0f / sqrtf(ss + eps);
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) xv[i] *= scale;
}

// NEOX partial rotary embedding, in place; one block per head.
__device__ inline void RopeNeox(float* x, std::uint32_t head_dim, std::uint32_t rotary_dim, std::uint32_t pos,
                                float theta) {
  float* v = x + blockIdx.x * head_dim;
  const std::uint32_t half = rotary_dim / 2;
  for (std::uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
    const float freq = powf(theta, -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim));
    const float angle = static_cast<float>(pos) * freq;
    const float c = cosf(angle), s = sinf(angle);
    const float a = v[i], b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = a * s + b * c;
  }
}
extern "C" __global__ void halo_rope_neox(float* x, std::uint32_t head_dim, std::uint32_t rotary_dim,
                                          std::uint32_t pos, float theta) {
  RopeNeox(x, head_dim, rotary_dim, pos, theta);
}
// Position from the step parameters: params[slot] * scale.
extern "C" __global__ void halo_rope_neox_p(float* x, const std::uint32_t* params, std::uint32_t head_dim,
                                            std::uint32_t rotary_dim, std::uint32_t slot, std::uint32_t scale,
                                            float theta) {
  RopeNeox(x, head_dim, rotary_dim, params[slot] * scale, theta);
}

extern "C" __global__ void halo_add(float* y, const float* x, std::uint32_t n) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] += x[i];
}

extern "C" __global__ void halo_axpy(float* y, const float* x, float a, std::uint32_t n) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] += a * x[i];
}

// g = silu(g) * up
extern "C" __global__ void halo_silu_mul(float* g, const float* up, std::uint32_t n) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    g[i] = Silu(g[i]) * up[i];
}

// x *= sigmoid(gate)
extern "C" __global__ void halo_sigmoid_gate(float* x, const float* gate, std::uint32_t n) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    x[i] *= Sigmoid(gate[i]);
}

// dst[params[slot] * stride + i] = src[i]: appends one row to a cache.
extern "C" __global__ void halo_copy_at(float* dst, const float* src, const std::uint32_t* params, std::uint32_t slot,
                                        std::uint32_t stride, std::uint32_t n) {
  float* d = dst + static_cast<std::size_t>(params[slot]) * stride;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) d[i] = src[i];
}

extern "C" __global__ void halo_copy(float* dst, const float* src, std::uint32_t n) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

// Splits [heads][2*hd] (q|gate per head) into q[heads][hd] and gate[heads][hd].
extern "C" __global__ void halo_split_qgate(const float* qg, float* q, float* gate, std::uint32_t hd) {
  const float* src = qg + blockIdx.x * 2 * hd;
  for (std::uint32_t i = threadIdx.x; i < hd; i += blockDim.x) {
    q[blockIdx.x * hd + i] = src[i];
    gate[blockIdx.x * hd + i] = src[hd + i];
  }
}

// Causal depthwise conv over the last `kern` projections, then SiLU; shifts
// the history (state holds kern-1 rows, oldest first).
extern "C" __global__ void halo_conv_step(const float* in, float* state, const float* w, float* out,
                                          std::uint32_t channels, std::uint32_t kern) {
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= channels) return;
  float acc = w[ch * kern + kern - 1] * in[ch];
  for (std::uint32_t k = 0; k + 1 < kern; ++k) acc += w[ch * kern + k] * state[k * channels + ch];
  for (std::uint32_t k = 0; k + 2 < kern; ++k) state[k * channels + ch] = state[(k + 1) * channels + ch];
  if (kern > 1) state[(kern - 2) * channels + ch] = in[ch];
  out[ch] = Silu(acc);
}

// One Gated DeltaNet decode step. One block per value head, one thread per
// value index j (blockDim == d). The state is stored key-major,
// S[h][i][j] (i over key, j over value), so at every step of the loops the
// threads of a wave touch consecutive addresses. (Row-major S[h][j][i] made
// each thread walk its own row: 512-byte strides across lanes.)
extern "C" __global__ void halo_gdn_step(const float* q, const float* k, const float* v, const float* alpha,
                                         const float* beta, const float* a, const float* dt, float* state, float* out,
                                         std::uint32_t num_k_heads, std::uint32_t d) {
  __shared__ float ks[256], qs[256];
  const std::uint32_t h = blockIdx.x, j = threadIdx.x;
  const float* qh = q + (h % num_k_heads) * d;
  const float* kh = k + (h % num_k_heads) * d;
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
    ks[i] = kh[i];
    qs[i] = qh[i];
  }
  __syncthreads();
  if (j >= d) return;
  const float decay = expf(a[h] * Softplus(alpha[h] + dt[h]));
  const float b = Sigmoid(beta[h]);
  float* S = state + static_cast<std::size_t>(h) * d * d + j;
  float u = 0.0f;
  for (std::uint32_t i = 0; i < d; ++i) {
    const float s = S[i * d] * decay;
    S[i * d] = s;
    u += s * ks[i];
  }
  const float delta = (v[h * d + j] - u) * b;
  float o = 0.0f;
  for (std::uint32_t i = 0; i < d; ++i) {
    const float s = S[i * d] + delta * ks[i];
    S[i * d] = s;
    o += s * qs[i];
  }
  out[h * d + j] = o / sqrtf(static_cast<float>(d));
}

// Per-head RMSNorm with a plain weight, then an output gate: silu (mode 0)
// or sigmoid (mode 1) of z. One block per head.
extern "C" __global__ void halo_gated_norm(float* o, const float* w, const float* z, std::uint32_t d, float eps,
                                           std::uint32_t mode) {
  __shared__ float scratch[32];
  float* ov = o + blockIdx.x * d;
  const float* zv = z + blockIdx.x * d;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += ov[i] * ov[i];
  ss = BlockSum(ss, scratch);
  const float scale = 1.0f / sqrtf(ss / static_cast<float>(d) + eps);
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
    const float g = mode == 0 ? Silu(zv[i]) : Sigmoid(zv[i]);
    ov[i] = ov[i] * scale * w[i] * g;
  }
}

// Decode attention over positions [0, n) for one query head per block, GQA
// grouping, output gated by sigmoid(gate). `scores` is scratch [heads][max_n].
extern "C" __global__ void halo_attention_decode(const float* q, const float* kc, const float* vc, const float* gate,
                                                 float* out, float* scores, std::uint32_t num_heads,
                                                 std::uint32_t num_kv_heads, std::uint32_t hd, std::uint32_t n,
                                                 std::uint32_t max_n) {
  __shared__ float scratch[32];
  const std::uint32_t h = blockIdx.x, kvh = h / (num_heads / num_kv_heads);
  const float* qh = q + h * hd;
  float* sc = scores + static_cast<std::size_t>(h) * max_n;
  const float scale = 1.0f / sqrtf(static_cast<float>(hd));
  float mx = -INFINITY;
  for (std::uint32_t p = threadIdx.x; p < n; p += blockDim.x) {
    const float* kp = kc + (static_cast<std::size_t>(p) * num_kv_heads + kvh) * hd;
    float dot = 0.0f;
    for (std::uint32_t i = 0; i < hd; ++i) dot += qh[i] * kp[i];
    sc[p] = dot * scale;
    mx = fmaxf(mx, sc[p]);
  }
  mx = BlockMax(mx, scratch);
  float den = 0.0f;
  for (std::uint32_t p = threadIdx.x; p < n; p += blockDim.x) {
    sc[p] = expf(sc[p] - mx);
    den += sc[p];
  }
  den = BlockSum(den, scratch);
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < hd; i += blockDim.x) {
    float acc = 0.0f;
    for (std::uint32_t p = 0; p < n; ++p) acc += sc[p] * vc[(static_cast<std::size_t>(p) * num_kv_heads + kvh) * hd + i];
    out[h * hd + i] = acc / den * Sigmoid(gate[h * hd + i]);
  }
}

// ---- Qwen3.8-Flash-Next specifics ------------------------------------------

// x = silu(x * scale) (hyper-connection low-rank mixer: silu(down(xn) / hc)).
extern "C" __global__ void halo_scale_silu(float* x, float scale, std::uint32_t n) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    x[i] = Silu(x[i] * scale);
}

// mixed[i] = mean_s xn[s*H+i] * sigmoid(gate[s*H+i]).
extern "C" __global__ void halo_hc_mix(const float* xn, const float* gate, float* mixed, std::uint32_t hc,
                                       std::uint32_t h) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += gridDim.x * blockDim.x) {
    float acc = 0.0f;
    for (std::uint32_t s = 0; s < hc; ++s) acc += xn[s * h + i] * Sigmoid(gate[s * h + i]);
    mixed[i] = acc / static_cast<float>(hc);
  }
}

// res[s*H+i] += out[i] * 2 * sigmoid(inject[s] / hc).
extern "C" __global__ void halo_hc_combine(float* res, const float* out, const float* inject, std::uint32_t hc,
                                           std::uint32_t h) {
  for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < hc * h; idx += gridDim.x * blockDim.x) {
    const std::uint32_t s = idx / h;
    res[idx] += out[idx % h] * 2.0f * Sigmoid(inject[s] / static_cast<float>(hc));
  }
}

// Replicates the embedding into hc residual streams.
extern "C" __global__ void halo_broadcast(float* res, const float* x, std::uint32_t hc, std::uint32_t h) {
  for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < hc * h; idx += gridDim.x * blockDim.x)
    res[idx] = x[idx % h];
}

// PLE gate: per stream s, g = <key_s, query_s>/sqrt(H), signed sqrt, then
// gated[s*H+i] = value[i] * sigmoid(sign(g) * sqrt(max(|g|, 1e-6))).
extern "C" __global__ void halo_ple_gate(const float* key, const float* query, const float* value, float* gated,
                                         std::uint32_t h) {
  __shared__ float scratch[32];
  const std::uint32_t s = blockIdx.x;
  float dot = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < h; i += blockDim.x) dot += key[s * h + i] * query[s * h + i];
  dot = BlockSum(dot, scratch);
  const float sc = dot / sqrtf(static_cast<float>(h));
  const float mag = sqrtf(fmaxf(fabsf(sc), 1e-6f));
  const float gate = Sigmoid((sc < 0.0f ? -1.0f : (sc > 0.0f ? 1.0f : 0.0f)) * mag);
  for (std::uint32_t i = threadIdx.x; i < h; i += blockDim.x) gated[s * h + i] = value[i] * gate;
}

// PLE dilated depthwise causal conv: tap k reads (kern-1-k)*dil tokens back.
// history: [hist][dim] oldest first (hist = (kern-1)*dil), shifted in place.
// res += gated + silu(conv).
extern "C" __global__ void halo_ple_conv(const float* normalized, const float* gated, float* history, const float* w,
                                         float* res, std::uint32_t dim, std::uint32_t kern, std::uint32_t dil) {
  const std::uint32_t hist = (kern - 1) * dil;
  for (std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x; ch < dim; ch += gridDim.x * blockDim.x) {
    float acc = 0.0f;
    for (std::uint32_t k = 0; k < kern; ++k) {
      const std::uint32_t back = (kern - 1 - k) * dil;
      const float src = back == 0 ? normalized[ch] : history[(hist - back) * dim + ch];
      acc += w[ch * kern + k] * src;
    }
    for (std::uint32_t t = 0; t + 1 < hist; ++t) history[t * dim + ch] = history[(t + 1) * dim + ch];
    if (hist > 0) history[(hist - 1) * dim + ch] = normalized[ch];
    res[ch] += gated[ch] + Silu(acc);
  }
}

// QSA: pooled[i] = mean over `ratio` raw indexer keys of one block.
// The block is the one holding this step's token: pooled every step, so the
// value written when the block completes is final, and an incomplete
// block's slot is never read before then.
extern "C" __global__ void halo_pool_block(const float* raw, float* pooled, const std::uint32_t* params,
                                           std::uint32_t ratio, std::uint32_t dim) {
  const std::uint32_t block = params[kHaloBlock];
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float acc = 0.0f;
    for (std::uint32_t t = 0; t < ratio; ++t) acc += raw[(block * ratio + t) * dim + i] / static_cast<float>(ratio);
    pooled[i] = acc;
  }
}

// QSA block scores: score[b] = sum_heads relu(<q_h, k_b>). One thread per block.
extern "C" __global__ void halo_qsa_scores(const float* q, const float* blocks, float* score,
                                           const std::uint32_t* params, std::uint32_t heads, std::uint32_t dim) {
  const std::uint32_t n_blocks = params[kHaloComplete];
  for (std::uint32_t b = blockIdx.x * blockDim.x + threadIdx.x; b < n_blocks; b += gridDim.x * blockDim.x) {
    float acc = 0.0f;
    for (std::uint32_t h = 0; h < heads; ++h) {
      float dot = 0.0f;
      for (std::uint32_t i = 0; i < dim; ++i) dot += q[h * dim + i] * blocks[b * dim + i];
      acc += fmaxf(0.0f, dot);
    }
    score[b] = acc;
  }
}

// Attention over a list of visible positions (QSA selection), GQA, gated.
extern "C" __global__ void halo_attention_select(const float* q, const float* kc, const float* vc, const float* gate,
                                                 const std::uint32_t* visible, float* out, float* scores,
                                                 const std::uint32_t* params, std::uint32_t num_heads,
                                                 std::uint32_t num_kv_heads, std::uint32_t hd, std::uint32_t max_n) {
  __shared__ float scratch[32];
  const std::uint32_t n = params[kHaloNVisible];
  const std::uint32_t h = blockIdx.x, kvh = h / (num_heads / num_kv_heads);
  const float* qh = q + h * hd;
  float* sc = scores + static_cast<std::size_t>(h) * max_n;
  const float scale = 1.0f / sqrtf(static_cast<float>(hd));
  float mx = -INFINITY;
  for (std::uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
    const float* kp = kc + (static_cast<std::size_t>(visible[j]) * num_kv_heads + kvh) * hd;
    float dot = 0.0f;
    for (std::uint32_t i = 0; i < hd; ++i) dot += qh[i] * kp[i];
    sc[j] = dot * scale;
    mx = fmaxf(mx, sc[j]);
  }
  mx = BlockMax(mx, scratch);
  float den = 0.0f;
  for (std::uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
    sc[j] = expf(sc[j] - mx);
    den += sc[j];
  }
  den = BlockSum(den, scratch);
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < hd; i += blockDim.x) {
    float acc = 0.0f;
    for (std::uint32_t j = 0; j < n; ++j)
      acc += sc[j] * vc[(static_cast<std::size_t>(visible[j]) * num_kv_heads + kvh) * hd + i];
    out[h * hd + i] = acc / den * Sigmoid(gate[h * hd + i]);
  }
}

// ---- Device-side routing and selection (no host round trip) ----------------

// MoE router: softmax over all experts, top-k of the probabilities (ties
// keep the lower expert index), renormalized with the reference's floor.
// One block; ids/weights receive k entries in descending probability.
extern "C" __global__ void halo_router_topk(const float* logits, std::uint32_t* ids, float* weights, std::uint32_t n,
                                            std::uint32_t k) {
  __shared__ float scratch[32];
  __shared__ float prob[1024];
  __shared__ float best_v[32];
  __shared__ std::uint32_t best_i[32];
  float mx = -INFINITY;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, logits[i]);
  mx = BlockMax(mx, scratch);
  float den = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    prob[i] = expf(logits[i] - mx);
    den += prob[i];
  }
  den = BlockSum(den, scratch);
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) prob[i] /= den;
  __syncthreads();
  float sum = 0.0f;
  for (std::uint32_t slot = 0; slot < k; ++slot) {
    float v = -1.0f;
    std::uint32_t at = 0xFFFFFFFFu;
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x)
      if (prob[i] > v) {  // strided ascending scan: first max per thread has the lowest index
        v = prob[i];
        at = i;
      }
    for (int o = kWarp / 2; o > 0; o >>= 1) {
      const float ov = __shfl_xor(v, o, kWarp);
      const std::uint32_t oi = __shfl_xor(at, o, kWarp);
      if (ov > v || (ov == v && oi < at)) {
        v = ov;
        at = oi;
      }
    }
    if (threadIdx.x % kWarp == 0) {
      best_v[threadIdx.x / kWarp] = v;
      best_i[threadIdx.x / kWarp] = at;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (std::uint32_t w = 1; w < (blockDim.x + kWarp - 1) / kWarp; ++w)
        if (best_v[w] > best_v[0] || (best_v[w] == best_v[0] && best_i[w] < best_i[0])) {
          best_v[0] = best_v[w];
          best_i[0] = best_i[w];
        }
      ids[slot] = best_i[0];
      weights[slot] = best_v[0];
      sum += best_v[0];
      prob[best_i[0]] = -1.0f;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sum = fmaxf(sum, 6.103515625e-5f);
    for (std::uint32_t slot = 0; slot < k; ++slot) weights[slot] /= sum;
  }
}

// y += weights[slot] * x
extern "C" __global__ void halo_axpy_slot(float* y, const float* x, const float* weights, std::uint32_t slot,
                                          std::uint32_t n) {
  const float a = weights[slot];
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] += a * x[i];
}

// y += sigmoid(g[0]) * x (shared-expert gate)
extern "C" __global__ void halo_axpy_sigmoid(float* y, const float* x, const float* g, std::uint32_t n) {
  const float a = Sigmoid(g[0]);
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] += a * x[i];
}

namespace {
// Order-preserving key of a non-negative score (QSA scores are sums of ReLU).
__device__ inline std::uint32_t ScoreKey(float v) { return v > 0.0f ? __float_as_uint(v) : 0u; }

// Exclusive prefix sum of one flag per thread across the block; returns the
// thread's offset and writes the block total to *total.
__device__ inline std::uint32_t BlockScan(std::uint32_t flag, std::uint32_t* buf, std::uint32_t* total) {
  buf[threadIdx.x] = flag;
  __syncthreads();
  for (std::uint32_t o = 1; o < blockDim.x; o <<= 1) {
    const std::uint32_t add = threadIdx.x >= o ? buf[threadIdx.x - o] : 0u;
    __syncthreads();
    buf[threadIdx.x] += add;
    __syncthreads();
  }
  const std::uint32_t inclusive = buf[threadIdx.x];
  *total = buf[blockDim.x - 1];
  __syncthreads();
  return inclusive - flag;
}
}  // namespace

// QSA block selection: keep the k highest-scoring of n complete blocks
// (equal scores: lower block first, i.e. the reference's stable sort), in
// position order, expanded to their `ratio` token positions, then the
// incomplete tail [n*ratio, n_kv). One block; blockDim a power of two.
extern "C" __global__ void halo_qsa_select(const float* score, std::uint32_t* visible, const std::uint32_t* params,
                                           std::uint32_t k, std::uint32_t ratio) {
  const std::uint32_t n = params[kHaloComplete], n_kv = params[kHaloNkv];
  __shared__ std::uint32_t hist[256];
  __shared__ std::uint32_t buf[1024];
  __shared__ std::uint32_t prefix, prefix_mask, remaining;
  if (k >= n) {  // under budget: every position is visible
    for (std::uint32_t p = threadIdx.x; p < n_kv; p += blockDim.x) visible[p] = p;
    return;
  }
  if (threadIdx.x == 0) {
    prefix = 0;
    prefix_mask = 0;
    remaining = k;  // rank (1-based) of the threshold among keys matching the prefix
  }
  // MSB-first radix select of the k-th largest key, 8 bits per pass.
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (std::uint32_t i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    for (std::uint32_t b = threadIdx.x; b < n; b += blockDim.x) {
      const std::uint32_t key = ScoreKey(score[b]);
      if ((key & prefix_mask) == prefix) atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      std::uint32_t digit = 255, acc = 0;
      for (;; --digit) {
        if (acc + hist[digit] >= remaining || digit == 0) break;
        acc += hist[digit];
      }
      remaining -= acc;
      prefix |= digit << shift;
      prefix_mask |= 0xFFu << shift;
    }
    __syncthreads();
  }
  const std::uint32_t threshold = prefix, take_equal = remaining;
  // One pass in block order: every key above the threshold, and the first
  // `take_equal` keys equal to it.
  std::uint32_t written = 0, equal_seen = 0;
  for (std::uint32_t base = 0; base < n; base += blockDim.x) {
    const std::uint32_t b = base + threadIdx.x;
    const std::uint32_t key = b < n ? ScoreKey(score[b]) : 0u;
    const std::uint32_t eq = b < n && key == threshold;
    std::uint32_t eq_total;
    const std::uint32_t eq_rank = equal_seen + BlockScan(eq, buf, &eq_total);
    const std::uint32_t keep = b < n && (key > threshold || (eq && eq_rank < take_equal));
    std::uint32_t kept;
    const std::uint32_t at = written + BlockScan(keep, buf, &kept);
    if (keep)
      for (std::uint32_t t = 0; t < ratio; ++t) visible[at * ratio + t] = b * ratio + t;
    written += kept;
    equal_seen += eq_total;
  }
  for (std::uint32_t p = n * ratio + threadIdx.x; p < n_kv; p += blockDim.x) visible[k * ratio + p - n * ratio] = p;
}

// ---- Device-side sampling (core/sampling semantics) --------------------------

#include "kernels/sample_params.h"

namespace {

constexpr std::uint32_t kIdBits = 18;  // vocab <= 2^18
constexpr std::uint32_t kIdMax = (1u << kIdBits) - 1;

// Order-preserving key of a double; +0 and -0 compare equal, as on the CPU.
__device__ inline std::uint64_t OrderedKey(double d) {
  if (d == 0.0) d = 0.0;
  const std::uint64_t b = static_cast<std::uint64_t>(__double_as_longlong(d));
  return (b >> 63) ? ~b : (b | (1ull << 63));
}

__device__ inline double Penalize(double logit, const HaloSampleParams& p, const HaloPenalty& q) {
  if (q.repeated && p.repeat_penalty != 1.0f) logit = logit <= 0 ? logit * p.repeat_penalty : logit / p.repeat_penalty;
  logit -= static_cast<double>(p.frequency_penalty) * q.generated_count;
  if (q.generated_count != 0) logit -= p.presence_penalty;
  return logit;
}

// Position in the sampling order: logit descending, then token ascending.
struct Cut {
  std::uint64_t key;
  std::uint32_t rid;  // kIdMax - token: larger sorts first
};
__device__ inline bool AtOrAbove(std::uint64_t key, std::uint32_t rid, const Cut& c) {
  return key > c.key || (key == c.key && rid >= c.rid);
}

struct SampleShared {
  unsigned long long hist[512];
  std::uint32_t bitmap[HALO_SAMPLE_MAX_VOCAB / 32];
  double max_v;
  std::uint32_t argmax, count, valid, nonzero, pred;
  unsigned long long mass;
  std::uint64_t pre_key, pre_key_mask, rem;
  std::uint32_t pre_rid, pre_rid_mask;
  double red_d[32];
  std::uint32_t red_u[32];
};

// The item at which the inclusive cumulative weight first reaches `target`
// (>= target, or > target when `strict`); weights null counts items. The
// order is the sampling order (logit descending, token ascending), or token
// order alone when `by_token`. MSB-first radix over the composite (key,
// rid): eight 8-bit key digits, then two 9-bit rid digits. Integer
// histograms make the answer independent of thread order.
__device__ Cut RadixSelect(const std::uint64_t* key, const std::uint32_t* rid, const std::uint64_t* w, std::uint32_t n,
                           std::uint64_t target, bool strict, SampleShared& s, bool by_token = false) {
  if (threadIdx.x == 0) {
    s.pre_key = s.pre_key_mask = 0;
    s.pre_rid = s.pre_rid_mask = 0;
    s.rem = target;
  }
  for (int pass = by_token ? 8 : 0; pass < 10; ++pass) {
    const bool on_key = pass < 8;
    const std::uint32_t shift = on_key ? 56 - 8 * pass : (pass == 8 ? 9 : 0);
    const std::uint32_t bins = on_key ? 256 : 512;
    for (std::uint32_t i = threadIdx.x; i < bins; i += blockDim.x) s.hist[i] = 0;
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      if ((key[i] & s.pre_key_mask) != s.pre_key || (rid[i] & s.pre_rid_mask) != s.pre_rid) continue;
      const std::uint32_t d = on_key ? static_cast<std::uint32_t>(key[i] >> shift) & 0xFF : (rid[i] >> shift) & 0x1FF;
      atomicAdd(&s.hist[d], static_cast<unsigned long long>(w ? w[i] : 1));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      std::uint64_t acc = 0;
      std::uint32_t d = bins - 1;
      for (;; --d) {
        const std::uint64_t h = s.hist[d];
        if (h > 0 && (strict ? acc + h > s.rem : acc + h >= s.rem)) break;
        acc += h;
        if (d == 0) break;  // unreachable for a satisfiable target
      }
      s.rem -= acc;
      if (on_key) {
        s.pre_key |= static_cast<std::uint64_t>(d) << shift;
        s.pre_key_mask |= 0xFFull << shift;
      } else {
        s.pre_rid |= d << shift;
        s.pre_rid_mask |= 0x1FFu << shift;
      }
    }
    __syncthreads();
  }
  return {s.pre_key, s.pre_rid};
}

// Weight (or count) of the items at or above `c`.
__device__ std::uint64_t MassAbove(const std::uint64_t* key, const std::uint32_t* rid, const std::uint64_t* w,
                                   std::uint32_t n, const Cut& c, SampleShared& s) {
  if (threadIdx.x == 0) s.mass = 0;
  __syncthreads();
  unsigned long long local = 0;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x)
    if (AtOrAbove(key[i], rid[i], c)) local += w ? w[i] : 1;
  atomicAdd(&s.mass, local);
  __syncthreads();
  const std::uint64_t r = s.mass;
  __syncthreads();
  return r;
}

}  // namespace

// One draw from the request's distribution over `logits` (vocab floats).
// One block of 1024 threads. `key`, `rid`, `w` are scratch lists of
// capacity vocab.
extern "C" __global__ void __launch_bounds__(1024)
    halo_sample(const float* logits, const HaloSampleParams* params, const HaloPenalty* pen, std::uint64_t* key,
                std::uint32_t* rid, std::uint64_t* w, HaloSampleOut* out) {
  __shared__ SampleShared s;
  const HaloSampleParams p = *params;
  // Fallback mode: run only when the fast sampler asked for the exact one.
  if (p.fallback_only && out->token != HALO_SAMPLE_FALLBACK) return;
  __syncthreads();
  const std::uint32_t V = p.vocab, NP = p.n_penalties;
  for (std::uint32_t i = threadIdx.x; i < HALO_SAMPLE_MAX_VOCAB / 32; i += blockDim.x) s.bitmap[i] = 0;
  if (threadIdx.x == 0) s.count = 0;
  __syncthreads();
  for (std::uint32_t j = threadIdx.x; j < NP; j += blockDim.x)
    if (pen[j].token < V) atomicOr(&s.bitmap[pen[j].token / 32], 1u << (pen[j].token % 32));
  __syncthreads();
  const auto penalized = [&](std::uint32_t t) { return (s.bitmap[t / 32] >> (t % 32)) & 1u; };

  // Max and argmax (ties to the lower token) of the penalized logits.
  double best = -INFINITY;
  std::uint32_t best_t = 0xFFFFFFFFu, valid = 0;
  const auto consider = [&](double v, std::uint32_t t) {
    ++valid;
    if (v > best || (v == best && t < best_t)) {
      best = v;
      best_t = t;
    }
  };
  for (std::uint32_t i = threadIdx.x; i < V; i += blockDim.x)
    if (!penalized(i) && isfinite(logits[i])) consider(logits[i], i);
  for (std::uint32_t j = threadIdx.x; j < NP; j += blockDim.x)
    if (pen[j].token < V && isfinite(logits[pen[j].token])) consider(Penalize(logits[pen[j].token], p, pen[j]), pen[j].token);
  for (int o = kWarp / 2; o > 0; o >>= 1) {
    const double ov = __shfl_xor(best, o, kWarp);
    const std::uint32_t ot = __shfl_xor(best_t, o, kWarp);
    if (ov > best || (ov == best && ot < best_t)) {
      best = ov;
      best_t = ot;
    }
    valid += __shfl_xor(valid, o, kWarp);
  }
  if (threadIdx.x % kWarp == 0) {
    s.red_d[threadIdx.x / kWarp] = best;
    s.red_u[threadIdx.x / kWarp] = best_t;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double b = -INFINITY;
    std::uint32_t bt = 0xFFFFFFFFu;
    for (std::uint32_t k = 0; k < blockDim.x / kWarp; ++k)
      if (s.red_d[k] > b || (s.red_d[k] == b && s.red_u[k] < bt)) {
        b = s.red_d[k];
        bt = s.red_u[k];
      }
    s.max_v = b;
    s.argmax = bt;
    s.valid = 0;
  }
  __syncthreads();
  if (threadIdx.x % kWarp == 0) atomicAdd(&s.valid, valid);
  __syncthreads();
  const double mx = s.max_v;
  const std::uint32_t n_valid = s.valid;
  if (n_valid == 0 || p.temperature == 0.0f) {  // greedy: no RNG draw
    if (threadIdx.x == 0) *out = {n_valid ? s.argmax : 0xFFFFFFFFu, n_valid ? 1u : 0u, p.rng};
    return;
  }

  // Live candidates: fixed-point weight exp((l - max) / T) * 2^40 >= 1.
  const double T = p.temperature, scale = static_cast<double>(1ull << HALO_SAMPLE_FIXED_BITS);
  // core/sampling's SamplerState has two paths with one distribution but
  // different draw orders: with no top-k, no top-p and no min-p floor it
  // draws in token order and applies min-p as zero weight; otherwise it
  // draws in sampling order. Both test min-p as v >= max + T * log(min_p).
  const bool linear = p.top_k == 0 && p.top_p >= 1.0f && !(p.min_p > 0.0f && p.min_keep > 1);
  const double min_p_cut = p.min_p > 0.0f ? mx + T * log(static_cast<double>(p.min_p)) : -INFINITY;
  if (threadIdx.x == 0) s.nonzero = s.pred = 0;
  __syncthreads();
  std::uint32_t nonzero = 0, pred = 0;
  const auto push = [&](double v, std::uint32_t t) {
    const double e = exp((v - mx) / T);
    const bool passes = v >= min_p_cut;
    pred += passes;
    if (linear && !passes) return;
    nonzero += e > 0.0;  // the CPU keeps every token with a nonzero probability
    const auto wq = static_cast<std::uint64_t>(e * scale + 0.5);
    if (wq == 0) return;
    const std::uint32_t at = atomicAdd(&s.count, 1u);
    key[at] = OrderedKey(v);
    rid[at] = kIdMax - t;
    w[at] = wq;
  };
  for (std::uint32_t i = threadIdx.x; i < V; i += blockDim.x)
    if (!penalized(i) && isfinite(logits[i])) push(logits[i], i);
  for (std::uint32_t j = threadIdx.x; j < NP; j += blockDim.x)
    if (pen[j].token < V && isfinite(logits[pen[j].token])) push(Penalize(logits[pen[j].token], p, pen[j]), pen[j].token);
  atomicAdd(&s.nonzero, nonzero);
  atomicAdd(&s.pred, pred);
  __syncthreads();
  const std::uint32_t L = s.count, n_nonzero = s.nonzero, n_pred = s.pred;
  const auto floor_of = [&](std::uint32_t size) { return min(size, max(p.min_keep, 1u)); };

  // Every filter keeps a prefix of the order: track it as a cut plus the
  // number of live items at or above it, and the CPU-side set size.
  Cut cut{0, 0};  // everything
  std::uint32_t kept = L, size = linear ? n_nonzero : n_valid;
  const auto cut_at_count = [&](std::uint32_t c) {  // first c live items
    if (c < kept) {
      cut = RadixSelect(key, rid, nullptr, L, c, false, s);
      kept = c;
    }
  };
  if (!linear && p.top_k > 0) {
    size = min(n_valid, max(p.top_k, floor_of(n_valid)));
    cut_at_count(size);
  }
  if (!linear && p.top_p < 1.0f && size > 1) {
    const std::uint64_t z = MassAbove(key, rid, w, L, cut, s);
    const auto target = static_cast<std::uint64_t>(static_cast<double>(p.top_p) * static_cast<double>(z));
    const Cut cross = RadixSelect(key, rid, w, L, target, false, s);
    const auto through = static_cast<std::uint32_t>(MassAbove(key, rid, nullptr, L, cross, s));
    const std::uint32_t minimum = floor_of(size);
    if (through >= minimum) {
      cut = cross;
      kept = through;
      size = through;
    } else {
      size = minimum;
      cut_at_count(minimum);
    }
  }
  if (!linear && p.min_p > 0.0f && size > 1) {
    // Tokens passing min-p form a prefix of the order; within the kept
    // prefix there are min(n_pred, size) of them, floored at min_keep.
    size = min(max(min(n_pred, size), floor_of(size)), size);
    cut_at_count(size);
  }

  // Draw: the first item whose cumulative weight exceeds u * Z.
  const std::uint64_t z = MassAbove(key, rid, w, L, cut, s);
  std::uint64_t rng = p.rng;
  std::uint32_t token;
  // The CPU draws a random number whenever more than one kept token has a
  // nonzero probability, including tokens too unlikely to be drawn here.
  if (min(size, n_nonzero) <= 1) {
    token = kIdMax - RadixSelect(key, rid, nullptr, L, 1, false, s, linear).rid;
  } else {
    std::uint64_t v = rng == 0 ? 0x9e3779b97f4a7c15ull : rng;
    v ^= v >> 12;
    v ^= v << 25;
    v ^= v >> 27;
    rng = v;
    const double u = static_cast<double>((v * 0x2545f4914f6cdd1dull) >> 11) * 0x1.0p-53;
    std::uint64_t target = static_cast<std::uint64_t>(u * static_cast<double>(z));
    if (target >= z) target = z - 1;
    token = kIdMax - RadixSelect(key, rid, w, L, target, true, s, linear).rid;
  }
  if (threadIdx.x == 0) *out = {token, kept, rng};
}

// ---- Fast sampler: distribution-exact, multi-block ---------------------------

namespace {

__device__ inline std::uint32_t OrderedKey32(float f) {
  if (f == 0.0f) f = 0.0f;
  const std::uint32_t b = __float_as_uint(f);
  return (b >> 31) ? ~b : (b | 0x80000000u);
}
__device__ inline float FromOrderedKey32(std::uint32_t k) {
  return __uint_as_float((k >> 31) ? (k & 0x7FFFFFFFu) : ~k);
}

// Gumbel(0, 1) noise for `token`, keyed by the request's RNG state: a
// counter-based hash, so every block draws its tokens' noise independently
// and the result does not depend on the launch shape.
// Single precision: u has 24 bits, so the noise is capped near 17.3; a token
// would need a logit gap beyond ~20 temperatures to be affected, which has
// probability around e^-20.
__device__ inline float GumbelNoise(std::uint64_t key, std::uint32_t token) {
  std::uint64_t z = key ^ (0x9e3779b97f4a7c15ull * (static_cast<std::uint64_t>(token) + 1));
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  const float u = (static_cast<float>(z >> 40) + 0.5f) * 0x1.0p-24f;  // (0, 1)
  return -__logf(-__logf(u));
}

__device__ inline double PenalizedLogit(const float* logits, std::uint32_t i, const HaloSampleParams& p,
                                        const HaloPenalty* pen) {
  double v = logits[i];
  if (p.n_penalties == 0) return v;
  std::uint32_t lo = 0, hi = p.n_penalties;  // the list is sorted by token
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) / 2;
    if (pen[mid].token < i) lo = mid + 1;
    else hi = mid;
  }
  return lo < p.n_penalties && pen[lo].token == i ? Penalize(v, p, pen[lo]) : v;
}

// Candidates per block: top-k needs top_k (floored by min_keep) of them;
// other filters use the full HALO_SAMPLE_TOPK.
__device__ inline std::uint32_t BlockKeep(const HaloSampleParams& p) {
  const std::uint32_t k = p.top_k > 0 ? max(p.top_k, max(p.min_keep, 1u)) : HALO_SAMPLE_TOPK;
  return min(k, HALO_SAMPLE_TOPK);
}

}  // namespace

// Stage 1: one block of HALO_SAMPLE_BLOCK threads per HALO_SAMPLE_BLOCK
// tokens. Per block: max and argmax, softmax mass relative to the block max,
// the best Gumbel score (plain temperature sampling), and the block's top
// candidates in sampling order (bitonic sort of (value, token) composites).
extern "C" __global__ void __launch_bounds__(1024)
    halo_sample_blocks(const float* logits, const HaloSampleParams* params, const HaloPenalty* pen,
                       HaloSampleBlock* blocks, std::uint64_t* candidates) {
  __shared__ std::uint64_t sorted[HALO_SAMPLE_BLOCK];
  __shared__ double red_d[32];
  __shared__ float red_f[32];
  __shared__ std::uint32_t red_u[32], red_g[32];
  const HaloSampleParams p = *params;
  const std::uint32_t i = blockIdx.x * HALO_SAMPLE_BLOCK + threadIdx.x;
  const bool valid = i < p.vocab && isfinite(logits[i]);
  const float v = valid ? static_cast<float>(PenalizedLogit(logits, i, p, pen)) : -INFINITY;
  const std::uint32_t lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp, warps = blockDim.x / kWarp;

  // Max and argmax (ties to the lower token); valid count.
  float bv = v;
  std::uint32_t bt = valid ? i : 0xFFFFFFFFu, nvalid = valid;
  for (int o = kWarp / 2; o > 0; o >>= 1) {
    const float ov = __shfl_xor(bv, o, kWarp);
    const std::uint32_t ot = __shfl_xor(bt, o, kWarp);
    if (ov > bv || (ov == bv && ot < bt)) {
      bv = ov;
      bt = ot;
    }
    nvalid += __shfl_xor(nvalid, o, kWarp);
  }
  if (lane == 0) {
    red_f[warp] = bv;
    red_u[warp] = bt;
    red_g[warp] = nvalid;
  }
  __syncthreads();
  float mx = -INFINITY;
  std::uint32_t amx = 0xFFFFFFFFu, total_valid = 0;
  for (std::uint32_t w = 0; w < warps; ++w) {
    if (red_f[w] > mx || (red_f[w] == mx && red_u[w] < amx)) {
      mx = red_f[w];
      amx = red_u[w];
    }
    total_valid += red_g[w];
  }
  __syncthreads();

  // Mass and Gumbel score (temperature > 0), deterministic reductions.
  double mass = 0, gscore = -INFINITY;
  std::uint32_t gid = 0xFFFFFFFFu;
  if (p.temperature > 0.0f && total_valid > 0) {
    const float inv_t = 1.0f / p.temperature;
    double e = valid ? static_cast<double>(__expf((v - mx) * inv_t)) : 0.0;
    double g = valid ? static_cast<double>(v * inv_t + GumbelNoise(p.rng, i)) : -INFINITY;
    std::uint32_t gi = valid ? i : 0xFFFFFFFFu;
    for (int o = kWarp / 2; o > 0; o >>= 1) {
      e += __shfl_xor(e, o, kWarp);
      const double og = __shfl_xor(g, o, kWarp);
      const std::uint32_t oi = __shfl_xor(gi, o, kWarp);
      if (og > g || (og == g && oi < gi)) {
        g = og;
        gi = oi;
      }
    }
    if (lane == 0) {
      red_d[warp] = e;
      red_u[warp] = gi;
    }
    __syncthreads();
    for (std::uint32_t w = 0; w < warps; ++w) mass += red_d[w];
    __syncthreads();
    if (lane == 0) red_d[warp] = g;
    __syncthreads();
    for (std::uint32_t w = 0; w < warps; ++w)
      if (red_d[w] > gscore || (red_d[w] == gscore && red_u[w] < gid)) {
        gscore = red_d[w];
        gid = red_u[w];
      }
    __syncthreads();
  }

  // Candidates (only when a filter needs them): bitonic sort of composites,
  // descending.
  const bool filters = p.temperature > 0.0f && (p.top_k > 0 || p.top_p < 1.0f || p.min_p > 0.0f);
  if (!filters) {
    if (threadIdx.x == 0) blocks[blockIdx.x] = {mass, gscore, mx, amx, gid, total_valid, 0, 0, 0};
    return;
  }
  sorted[threadIdx.x] = valid ? (static_cast<std::uint64_t>(OrderedKey32(v)) << 32) | (kIdMax - i) : 0;
  __syncthreads();
  for (std::uint32_t k = 2; k <= HALO_SAMPLE_BLOCK; k <<= 1)
    for (std::uint32_t j = k >> 1; j > 0; j >>= 1) {
      const std::uint32_t t = threadIdx.x, partner = t ^ j;
      if (partner > t) {
        const bool descending = (t & k) == 0;
        const std::uint64_t a = sorted[t], b = sorted[partner];
        if (descending ? a < b : a > b) {
          sorted[t] = b;
          sorted[partner] = a;
        }
      }
      __syncthreads();
    }
  const std::uint32_t keep = min(BlockKeep(p), total_valid);
  if (threadIdx.x < keep) candidates[blockIdx.x * HALO_SAMPLE_TOPK + threadIdx.x] = sorted[threadIdx.x];
  if (threadIdx.x == 0)
    blocks[blockIdx.x] = {mass, gscore, mx, amx, gid, total_valid, keep, 0,
                          keep < HALO_SAMPLE_BLOCK ? sorted[keep] : 0};
}

// Stage 2: one block of 1024 threads. Greedy and plain temperature finish
// from the block results; otherwise the filters run over the candidates
// with core/sampling's semantics and a Gumbel-max draw picks the token.
// Every filter decision must be provably confined to the candidates (a
// top-k they fully contain, or a cut above every excluded token);
// otherwise the token is HALO_SAMPLE_FALLBACK and the exact kernel runs.
extern "C" __global__ void __launch_bounds__(1024)
    halo_sample_final(const HaloSampleParams* params, const HaloSampleBlock* blocks, const std::uint64_t* candidates,
                      std::uint32_t n_blocks, std::uint64_t* key, std::uint32_t* rid, std::uint64_t* w,
                      HaloSampleOut* out) {
  __shared__ SampleShared s;
  __shared__ std::uint32_t offset[1025];  // candidate prefix offsets per block
  const HaloSampleParams p = *params;
  // Block summaries, reduced in parallel (one thread per block): global
  // max/argmax, valid count, best Gumbel score and the excluded maximum.
  {
    float mx = -INFINITY;
    std::uint32_t amx = 0xFFFFFFFFu, valid = 0, gid = 0xFFFFFFFFu, kept = 0;
    double g = -INFINITY;
    std::uint64_t excluded = 0;
    for (std::uint32_t b = threadIdx.x; b < n_blocks; b += blockDim.x) {
      const HaloSampleBlock k = blocks[b];
      kept += k.kept;
      if (k.valid == 0) continue;
      valid += k.valid;
      if (k.max_v > mx || (k.max_v == mx && k.argmax < amx)) {
        mx = k.max_v;
        amx = k.argmax;
      }
      if (k.gumbel > g || (k.gumbel == g && k.gumbel_id < gid)) {
        g = k.gumbel;
        gid = k.gumbel_id;
      }
      excluded = max(excluded, k.excluded);
    }
    // Candidate offsets: blocks are <= 1024, so one per thread.
    std::uint32_t total;
    const std::uint32_t mine = threadIdx.x < n_blocks ? blocks[threadIdx.x].kept : 0;
    const std::uint32_t at = BlockScan(mine, s.bitmap, &total);
    if (threadIdx.x < n_blocks) offset[threadIdx.x] = at;
    if (threadIdx.x == 0) offset[n_blocks] = total;
    for (int o = kWarp / 2; o > 0; o >>= 1) {
      const float omx = __shfl_xor(mx, o, kWarp);
      const std::uint32_t oam = __shfl_xor(amx, o, kWarp);
      if (omx > mx || (omx == mx && oam < amx)) {
        mx = omx;
        amx = oam;
      }
      const double og = __shfl_xor(g, o, kWarp);
      const std::uint32_t ogi = __shfl_xor(gid, o, kWarp);
      if (og > g || (og == g && ogi < gid)) {
        g = og;
        gid = ogi;
      }
      valid += __shfl_xor(valid, o, kWarp);
      excluded = max(excluded, static_cast<std::uint64_t>(__shfl_xor(static_cast<unsigned long long>(excluded), o, kWarp)));
    }
    __shared__ float w_mx[32];
    __shared__ std::uint32_t w_amx[32], w_valid[32], w_gid[32];
    __shared__ double w_g[32];
    __shared__ std::uint64_t w_ex[32];
    const std::uint32_t lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    if (lane == 0) {
      w_mx[warp] = mx;
      w_amx[warp] = amx;
      w_valid[warp] = valid;
      w_gid[warp] = gid;
      w_g[warp] = g;
      w_ex[warp] = excluded;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bm = -INFINITY;
      std::uint32_t ba = 0xFFFFFFFFu, bv = 0, bgi = 0xFFFFFFFFu;
      double bg = -INFINITY;
      std::uint64_t be = 0;
      for (std::uint32_t k = 0; k < blockDim.x / kWarp; ++k) {
        if (w_mx[k] > bm || (w_mx[k] == bm && w_amx[k] < ba)) {
          bm = w_mx[k];
          ba = w_amx[k];
        }
        if (w_g[k] > bg || (w_g[k] == bg && w_gid[k] < bgi)) {
          bg = w_g[k];
          bgi = w_gid[k];
        }
        bv += w_valid[k];
        be = max(be, w_ex[k]);
      }
      s.max_v = bm;
      s.argmax = ba;
      s.valid = bv;
      s.count = total;
      s.nonzero = bgi;
      s.pre_key = be;
    }
    (void)kept;
  }
  __syncthreads();
  const double mx = s.max_v;
  const std::uint32_t n_valid = s.valid, L_total = s.count;
  const std::uint64_t excluded32 = s.pre_key;
  const std::uint64_t next_rng = [&] {
    std::uint64_t v = p.rng == 0 ? 0x9e3779b97f4a7c15ull : p.rng;
    v ^= v >> 12;
    v ^= v << 25;
    v ^= v >> 27;
    return v;
  }();
  if (n_valid == 0 || p.temperature == 0.0f) {
    if (threadIdx.x == 0) *out = {n_valid ? s.argmax : 0xFFFFFFFFu, n_valid ? 1u : 0u, p.rng};
    return;
  }
  if (p.top_k == 0 && p.top_p >= 1.0f && p.min_p <= 0.0f) {  // plain temperature
    if (threadIdx.x == 0) *out = {s.nonzero, n_valid, next_rng};
    return;
  }
  __syncthreads();

  // Gather candidates (all, including zero-weight ones: they count for
  // top-k and floors). Keys in the double domain, as the exact kernel.
  const double T = p.temperature, scale = static_cast<double>(1ull << HALO_SAMPLE_FIXED_BITS);
  const std::uint32_t L = L_total;
  for (std::uint32_t j = threadIdx.x; j < L; j += blockDim.x) {
    std::uint32_t lo = 0, hi = n_blocks;  // the block holding flat index j
    while (hi - lo > 1) {
      const std::uint32_t mid = (lo + hi) / 2;
      if (offset[mid] <= j) lo = mid;
      else hi = mid;
    }
    const std::uint64_t c = candidates[lo * HALO_SAMPLE_TOPK + (j - offset[lo])];
    const float v = FromOrderedKey32(static_cast<std::uint32_t>(c >> 32));
    key[j] = OrderedKey(static_cast<double>(v));
    rid[j] = static_cast<std::uint32_t>(c);
    w[j] = static_cast<std::uint64_t>(exp((static_cast<double>(v) - mx) / T) * scale + 0.5);
  }
  __syncthreads();
  // Every token above `excluded` (in sampling order) is a candidate.
  const Cut ex = excluded32 == 0 ? Cut{0, 0}
                                 : Cut{OrderedKey(static_cast<double>(FromOrderedKey32(
                                           static_cast<std::uint32_t>(excluded32 >> 32)))),
                                       static_cast<std::uint32_t>(excluded32)};
  const auto above_excluded = [&](const Cut& c) {
    return excluded32 == 0 || c.key > ex.key || (c.key == ex.key && c.rid > ex.rid);
  };
  const auto floor_of = [&](std::uint32_t size) { return min(size, max(p.min_keep, 1u)); };
  Cut cut{0, 0};
  std::uint32_t kept = L, size = n_valid;
  bool prefix_ok = excluded32 == 0;  // the kept prefix consists of candidates
  bool exact = true;
  const auto cut_at_count = [&](std::uint32_t c) {
    if (c < kept) {
      cut = RadixSelect(key, rid, nullptr, L, c, false, s);
      kept = c;
      prefix_ok = prefix_ok || above_excluded(cut);
    } else if (!prefix_ok) {
      exact = false;  // the cut lies past the candidates
    }
  };
  if (p.top_k > 0) {
    size = min(n_valid, max(p.top_k, floor_of(n_valid)));
    if (size <= BlockKeep(p)) {  // every block holds its share of the global top `size`
      cut_at_count(size);
      prefix_ok = true;
    } else {
      exact = false;
    }
  }
  if (exact && p.top_p < 1.0f && size > 1) {
    std::uint64_t z;
    if (p.top_k > 0) {
      z = MassAbove(key, rid, w, L, cut, s);
    } else {  // all valid tokens: rescale each block's mass to the global max
      double zd = 0;
      for (std::uint32_t b = 0; b < n_blocks; ++b)
        if (blocks[b].valid) zd += blocks[b].mass * exp((static_cast<double>(blocks[b].max_v) - mx) / T);
      z = static_cast<std::uint64_t>(zd * scale + 0.5);
    }
    const auto target = static_cast<std::uint64_t>(static_cast<double>(p.top_p) * static_cast<double>(z));
    if (MassAbove(key, rid, w, L, Cut{0, 0}, s) < target) {
      exact = false;  // the crossing lies past the candidates
    } else {
      const Cut cross = RadixSelect(key, rid, w, L, target, false, s);
      if (!prefix_ok && !above_excluded(cross)) {
        exact = false;
      } else {
        const auto through = static_cast<std::uint32_t>(MassAbove(key, rid, nullptr, L, cross, s));
        const std::uint32_t minimum = floor_of(size);
        if (through >= minimum) {
          cut = cross;
          kept = through;
          size = through;
          prefix_ok = true;
        } else {
          size = minimum;
          cut_at_count(minimum);
        }
      }
    }
  }
  if (exact && p.min_p > 0.0f && size > 1) {
    const double min_p_cut = mx + T * log(static_cast<double>(p.min_p));
    // All tokens passing min-p are candidates when the excluded maximum fails it.
    const bool all_passing_known =
        excluded32 == 0 || static_cast<double>(FromOrderedKey32(static_cast<std::uint32_t>(excluded32 >> 32))) < min_p_cut;
    if (!prefix_ok && !all_passing_known) {
      exact = false;
    } else {
      if (threadIdx.x == 0) s.mass = 0;
      __syncthreads();
      unsigned long long local = 0;
      for (std::uint32_t i = threadIdx.x; i < L; i += blockDim.x) {
        if (!AtOrAbove(key[i], rid[i], cut)) continue;
        const std::uint64_t k = key[i];
        const double v = __longlong_as_double(static_cast<long long>((k >> 63) ? (k & ~(1ull << 63)) : ~k));
        local += v >= min_p_cut;
      }
      atomicAdd(&s.mass, local);
      __syncthreads();
      const auto pass = static_cast<std::uint32_t>(s.mass);
      __syncthreads();
      const std::uint32_t passing = min(pass, size);
      size = min(max(passing, floor_of(size)), size);
      // A cut at the last passing token lies above every excluded token
      // when all passing tokens are known; a min_keep floor below it does
      // not, so cut_at_count checks that one against the excluded maximum.
      if (all_passing_known && size == passing) {
        if (size < kept) {
          cut = RadixSelect(key, rid, nullptr, L, size, false, s);
          kept = size;
        }
        prefix_ok = true;
      } else {
        cut_at_count(size);
      }
    }
  }
  if (!exact || !prefix_ok) {
    if (threadIdx.x == 0) *out = {HALO_SAMPLE_FALLBACK, 0, p.rng};
    return;
  }

  // Gumbel-max over the kept candidates: an exact draw from the softmax
  // restricted to them.
  double g = -INFINITY;
  std::uint32_t gi = 0xFFFFFFFFu;
  for (std::uint32_t i = threadIdx.x; i < L; i += blockDim.x) {
    if (!AtOrAbove(key[i], rid[i], cut)) continue;
    const std::uint32_t t = kIdMax - rid[i];
    const std::uint64_t k = key[i];
    const double v = __longlong_as_double(static_cast<long long>((k >> 63) ? (k & ~(1ull << 63)) : ~k));
    const double sc = static_cast<double>(static_cast<float>(v) * (1.0f / p.temperature) + GumbelNoise(p.rng, t));
    if (sc > g || (sc == g && t < gi)) {
      g = sc;
      gi = t;
    }
  }
  for (int o = kWarp / 2; o > 0; o >>= 1) {
    const double og = __shfl_xor(g, o, kWarp);
    const std::uint32_t oi = __shfl_xor(gi, o, kWarp);
    if (og > g || (og == g && oi < gi)) {
      g = og;
      gi = oi;
    }
  }
  if (threadIdx.x % kWarp == 0) {
    s.red_d[threadIdx.x / kWarp] = g;
    s.red_u[threadIdx.x / kWarp] = gi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double bg = -INFINITY;
    std::uint32_t bi = 0xFFFFFFFFu;
    for (std::uint32_t k = 0; k < blockDim.x / kWarp; ++k)
      if (s.red_d[k] > bg || (s.red_d[k] == bg && s.red_u[k] < bi)) {
        bg = s.red_d[k];
        bi = s.red_u[k];
      }
    *out = {bi, kept, next_rng};
  }
}

// ---- P3: bandwidth-oriented GEMV, Q8_0 repacked ------------------------------
//
// Q8_0 stores 34-byte blocks (fp16 scale + 32 int8), so rows are not even
// 2-byte aligned and the P2 kernel decodes one element per load. At load
// time each Q8_0 matrix is repacked once into int8 rows padded to 16 bytes
// (`qs`, row stride `qstride`) and a separate fp16 scale array (`d`, one per
// 32 columns, row stride cols / 32). The GEMV then streams 16 bytes per lane
// per load.

extern "C" __global__ void halo_repack_q8_0(const std::uint8_t* raw, std::int8_t* qs, std::uint16_t* d,
                                            std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride) {
  const std::uint32_t blocks = cols / 32;
  const std::uint64_t total = static_cast<std::uint64_t>(rows) * blocks;
  for (std::uint64_t b = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; b < total;
       b += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    const std::uint64_t r = b / blocks, k = b % blocks;
    const std::uint8_t* src = raw + (r * blocks + k) * 34;
    d[r * blocks + k] = U16(src);
    std::int8_t* dst = qs + r * qstride + k * 32;
    for (int i = 0; i < 32; ++i) dst[i] = static_cast<std::int8_t>(src[2 + i]);
  }
}

namespace {

constexpr std::uint32_t kGemvMaxCols = 8192;

// One wave per row; the block's x lives in LDS (cols <= kGemvMaxCols). Each lane
// loads 16 int8 at a time (half a Q8_0 block), so a wave covers 512 columns
// per iteration.
template <int kRowsPerBlock, std::uint32_t kMaxCols = kGemvMaxCols>
__device__ void GemvQ8r(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y, std::uint32_t rows,
                        std::uint32_t cols, std::uint32_t qstride) {
  __shared__ float xs[kMaxCols];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const std::uint32_t r = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
  if (r >= rows) return;
  const std::uint32_t lane = threadIdx.x % kWarp;
  const std::int8_t* row = qs + static_cast<std::size_t>(r) * qstride;
  const std::uint16_t* scales = d + static_cast<std::size_t>(r) * (cols / 32);
  float acc = 0.0f;
  for (std::uint32_t c = lane * 16; c < cols; c += kWarp * 16) {
    const int4 q = *reinterpret_cast<const int4*>(row + c);
    const float s = F16(scales[c / 32]);
    const std::int8_t* v = reinterpret_cast<const std::int8_t*>(&q);
    const float* xv = xs + c;
    float part = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) part += static_cast<float>(v[i]) * xv[i];
    acc += s * part;
  }
  acc = WarpSum(acc);
  if (lane == 0) y[r] = acc;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256)
    halo_gemv_q8r(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y, std::uint32_t rows,
                  std::uint32_t cols, std::uint32_t qstride) {
  GemvQ8r<8>(qs, d, x, y, rows, cols, qstride);
}

// Inputs wider than kGemvMaxCols (the hyper-connection down projections read
// all hc streams: 4 x 2560 = 10240 columns). A separate kernel, so the
// common case keeps its smaller LDS footprint and occupancy.
extern "C" __global__ void __launch_bounds__(256)
    halo_gemv_q8r_wide(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y, std::uint32_t rows,
                       std::uint32_t cols, std::uint32_t qstride) {
  GemvQ8r<8, 12288>(qs, d, x, y, rows, cols, qstride);
}

// Few output rows over long inputs (router 512, DeltaNet alpha/beta 48,
// hyper-connection inject 4): one 256-thread block per row instead of one
// wave, so the launch has enough blocks to fill the GPU.
extern "C" __global__ void __launch_bounds__(256)
    halo_matvec_f32_rowblock(const float* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols) {
  __shared__ float scratch[32];
  const std::uint32_t r = blockIdx.x;
  if (r >= rows) return;
  const float* row = w + static_cast<std::size_t>(r) * cols;
  float acc = 0.0f;
  for (std::uint32_t c = threadIdx.x; c < cols; c += blockDim.x) acc += row[c] * x[c];
  acc = BlockSum(acc, scratch);
  if (threadIdx.x == 0) y[r] = acc;
}

// Stacked experts: matrix ids[slot] of a [experts][rows][cols] repacked tensor.
extern "C" __global__ void __launch_bounds__(256)
    halo_gemv_q8r_expert(const std::int8_t* qs, const std::uint16_t* d, const std::uint32_t* ids, const float* x,
                         float* y, std::uint32_t slot, std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride) {
  const std::size_t e = ids[slot];
  GemvQ8r<8>(qs + e * rows * qstride, d + e * rows * (cols / 32), x, y, rows, cols, qstride);
}

// ---- P3: bandwidth GEMV for the expert formats (Q4_K, Q5_1) ------------------
//
// One wave per row, x in LDS. Q4_K: a lane takes one 64-value chunk of a
// 256-value super-block (32 bytes of nibbles, two 6-bit scale/min pairs);
// Q5_1: a lane takes one 32-value block (24 bytes). The block minimum is
// applied once per block: d*sc*sum(q*x) - dmin*m*sum(x).

namespace {

struct DotQ4K {
  static constexpr std::uint32_t kItemValues = 64;
  __device__ static float Item(const std::uint8_t* row, const float* xs, std::uint32_t item) {
    const std::uint32_t sb = item / 4, c = item % 4;
    const std::uint8_t* b = row + sb * 144;
    // d, dmin and the 12 scale bytes in one 16-byte load.
    const int4 hdr = *reinterpret_cast<const int4*>(b);
    const auto w0 = static_cast<std::uint32_t>(hdr.y), w1 = static_cast<std::uint32_t>(hdr.z),
               w2 = static_cast<std::uint32_t>(hdr.w);
    std::uint32_t sc0, m0, sc1, m1;
    ScaleMinK4Words(2 * c, w0, w1, w2, sc0, m0);
    ScaleMinK4Words(2 * c + 1, w0, w1, w2, sc1, m1);
    const int4 qa = *reinterpret_cast<const int4*>(b + 16 + c * 32);
    const int4 qb = *reinterpret_cast<const int4*>(b + 16 + c * 32 + 16);
    const std::uint8_t* q0 = reinterpret_cast<const std::uint8_t*>(&qa);
    const std::uint8_t* q1 = reinterpret_cast<const std::uint8_t*>(&qb);
    const float* x = xs + sb * 256 + c * 64;
    float lo = 0, hi = 0, xlo = 0, xhi = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      lo += static_cast<float>(q0[j] & 0xF) * x[j];
      hi += static_cast<float>(q0[j] >> 4) * x[32 + j];
      xlo += x[j];
      xhi += x[32 + j];
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      lo += static_cast<float>(q1[j] & 0xF) * x[16 + j];
      hi += static_cast<float>(q1[j] >> 4) * x[48 + j];
      xlo += x[16 + j];
      xhi += x[48 + j];
    }
    const float d = F16(static_cast<std::uint16_t>(hdr.x & 0xFFFF)),
                dmin = F16(static_cast<std::uint16_t>(static_cast<std::uint32_t>(hdr.x) >> 16));
    return d * (static_cast<float>(sc0) * lo + static_cast<float>(sc1) * hi) -
           dmin * (static_cast<float>(m0) * xlo + static_cast<float>(m1) * xhi);
  }
  static constexpr std::uint32_t kRowBytesPer256 = 144;
};

struct DotQ5_1 {
  static constexpr std::uint32_t kItemValues = 32;
  __device__ static float Item(const std::uint8_t* row, const float* xs, std::uint32_t item) {
    const std::uint8_t* b = row + item * 24;
    // The 24-byte block in three 8-byte loads: d, m, qh, then 16 nibble bytes.
    const uint2 hd = *reinterpret_cast<const uint2*>(b);
    const uint2 qa = *reinterpret_cast<const uint2*>(b + 8);
    const uint2 qb = *reinterpret_cast<const uint2*>(b + 16);
    const std::uint32_t qh = hd.y;
    std::uint8_t q[16];
    *reinterpret_cast<uint2*>(q) = qa;
    *reinterpret_cast<uint2*>(q + 8) = qb;
    const float* x = xs + item * 32;
    float s = 0, sx = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float lo = static_cast<float>((q[j] & 0xF) | (((qh >> j) & 1) << 4));
      const float hi = static_cast<float>((q[j] >> 4) | (((qh >> (j + 16)) & 1) << 4));
      s += lo * x[j] + hi * x[16 + j];
      sx += x[j] + x[16 + j];
    }
    return F16(static_cast<std::uint16_t>(hd.x & 0xFFFF)) * s + F16(static_cast<std::uint16_t>(hd.x >> 16)) * sx;
  }
};

template <typename Fmt>
__device__ void GemvBlocks(const std::uint8_t* w, std::uint64_t row_bytes, const float* x, float* y, std::uint32_t rows,
                           std::uint32_t cols) {
  __shared__ float xs[kGemvMaxCols];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const std::uint32_t r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
  if (r >= rows) return;
  const std::uint32_t lane = threadIdx.x % kWarp, items = cols / Fmt::kItemValues;
  const std::uint8_t* row = w + r * row_bytes;
  float acc = 0;
  for (std::uint32_t i = lane; i < items; i += kWarp) acc += Fmt::Item(row, xs, i);
  acc = WarpSum(acc);
  if (lane == 0) y[r] = acc;
}

}  // namespace

#define HALO_GEMV_BLOCKS(NAME, FMT)                                                                            \
  extern "C" __global__ void __launch_bounds__(256)                                                            \
      NAME(const std::uint8_t* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols,            \
           std::uint64_t row_bytes) {                                                                          \
    GemvBlocks<FMT>(w, row_bytes, x, y, rows, cols);                                                           \
  }                                                                                                            \
  extern "C" __global__ void __launch_bounds__(256)                                                            \
      NAME##_expert(const std::uint8_t* w, const std::uint32_t* ids, const float* x, float* y,                 \
                    std::uint32_t slot, std::uint64_t expert_bytes, std::uint32_t rows, std::uint32_t cols,    \
                    std::uint64_t row_bytes) {                                                                 \
    GemvBlocks<FMT>(w + ids[slot] * expert_bytes, row_bytes, x, y, rows, cols);                                \
  }

HALO_GEMV_BLOCKS(halo_gemv_q4_k, DotQ4K)
HALO_GEMV_BLOCKS(halo_gemv_q5_1, DotQ5_1)

// ---- P3: fused routed-expert MoE --------------------------------------------
//
// All routed experts in three launches instead of five per expert:
//   gate_up: h[slot][r] = silu(<gate_e[r], x>) * <up_e[r], x>   (grid.y = slot)
//   down:    part[slot][r] = <down_e[r], h[slot]>                (grid.y = slot)
//   combine: out[r] = sum_slot weights[slot] * part[slot][r]     (fixed order)
// e = ids[slot]. gate/up are Q4_K; down is Q5_1 or repacked Q8_0.

namespace {

__device__ inline float RowDotQ4K(const std::uint8_t* row, const float* xs, std::uint32_t cols, std::uint32_t lane) {
  float acc = 0;
  for (std::uint32_t i = lane; i < cols / 64; i += kWarp) acc += DotQ4K::Item(row, xs, i);
  return WarpSum(acc);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256)
    halo_moe_gate_up_q4_k(const std::uint8_t* gate, const std::uint8_t* up, const std::uint32_t* ids, const float* x,
                          float* h, std::uint64_t expert_bytes, std::uint32_t rows, std::uint32_t cols,
                          std::uint64_t row_bytes) {
  __shared__ float xs[kGemvMaxCols];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const std::uint32_t slot = blockIdx.y, r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
  if (r >= rows) return;
  const std::uint32_t lane = threadIdx.x % kWarp;
  const std::size_t e = ids[slot];
  const float g = RowDotQ4K(gate + e * expert_bytes + r * row_bytes, xs, cols, lane);
  const float u = RowDotQ4K(up + e * expert_bytes + r * row_bytes, xs, cols, lane);
  if (lane == 0) h[static_cast<std::size_t>(slot) * rows + r] = Silu(g) * u;
}

extern "C" __global__ void __launch_bounds__(256)
    halo_moe_down_q5_1(const std::uint8_t* down, const std::uint32_t* ids, const float* h, float* part,
                       std::uint64_t expert_bytes, std::uint32_t rows, std::uint32_t cols, std::uint64_t row_bytes) {
  const std::uint32_t slot = blockIdx.y;
  GemvBlocks<DotQ5_1>(down + ids[slot] * expert_bytes, row_bytes, h + static_cast<std::size_t>(slot) * cols,
                      part + static_cast<std::size_t>(slot) * rows, rows, cols);
}

extern "C" __global__ void __launch_bounds__(256)
    halo_moe_down_q8r(const std::int8_t* qs, const std::uint16_t* d, const std::uint32_t* ids, const float* h,
                      float* part, std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride) {
  const std::uint32_t slot = blockIdx.y;
  const std::size_t e = ids[slot];
  GemvQ8r<8>(qs + e * rows * qstride, d + e * rows * (cols / 32), h + static_cast<std::size_t>(slot) * cols,
             part + static_cast<std::size_t>(slot) * rows, rows, cols, qstride);
}

extern "C" __global__ void halo_moe_combine(const float* part, const float* weights, float* out, std::uint32_t slots,
                                            std::uint32_t n) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    float acc = 0;
    for (std::uint32_t s = 0; s < slots; ++s) acc += weights[s] * part[static_cast<std::size_t>(s) * n + i];
    out[i] = acc;
  }
}

// ---- P3: multi-row Q8 GEMV variants (tuning) ---------------------------------
// Each wave computes kWaveRows rows: every x chunk read from LDS feeds all of
// them and each lane keeps kWaveRows 16-byte loads in flight; a block of 8
// waves stages x once for 8 * kWaveRows rows.

namespace {

template <int kWaveRows, std::uint32_t kMaxCols>
__device__ void GemvQ8rMulti(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y,
                             std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride) {
  __shared__ float xs[kMaxCols];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kWaveRows;
  if (r0 >= rows) return;
  const std::uint32_t lane = threadIdx.x % kWarp, nblk = cols / 32;
  float acc[kWaveRows] = {};
  for (std::uint32_t c = lane * 16; c < cols; c += kWarp * 16) {
    int4 q[kWaveRows];
    float s[kWaveRows];
#pragma unroll
    for (int k = 0; k < kWaveRows; ++k) {
      const std::uint32_t r = min(r0 + k, rows - 1);
      q[k] = *reinterpret_cast<const int4*>(qs + static_cast<std::size_t>(r) * qstride + c);
      s[k] = F16(d[static_cast<std::size_t>(r) * nblk + c / 32]);
    }
    const float* xv = xs + c;
    float xr[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) xr[i] = xv[i];
#pragma unroll
    for (int k = 0; k < kWaveRows; ++k) {
      const std::int8_t* v = reinterpret_cast<const std::int8_t*>(&q[k]);
      float part = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) part += static_cast<float>(v[i]) * xr[i];
      acc[k] += s[k] * part;
    }
  }
#pragma unroll
  for (int k = 0; k < kWaveRows; ++k) {
    const float a = WarpSum(acc[k]);
    if (lane == 0 && r0 + k < rows) y[r0 + k] = a;
  }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256)
    halo_gemv_q8r_r2(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y, std::uint32_t rows,
                     std::uint32_t cols, std::uint32_t qstride) {
  GemvQ8rMulti<2, 12288>(qs, d, x, y, rows, cols, qstride);
}
extern "C" __global__ void __launch_bounds__(256)
    halo_gemv_q8r_r4(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y, std::uint32_t rows,
                     std::uint32_t cols, std::uint32_t qstride) {
  GemvQ8rMulti<4, 12288>(qs, d, x, y, rows, cols, qstride);
}
extern "C" __global__ void __launch_bounds__(256)
    halo_gemv_q8r_r4n(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y, std::uint32_t rows,
                      std::uint32_t cols, std::uint32_t qstride) {
  GemvQ8rMulti<4, 6144>(qs, d, x, y, rows, cols, qstride);
}

// Multi-row expert and fused-MoE down variants (cols <= 6144).
extern "C" __global__ void __launch_bounds__(256)
    halo_gemv_q8r_expert_r4(const std::int8_t* qs, const std::uint16_t* d, const std::uint32_t* ids, const float* x,
                            float* y, std::uint32_t slot, std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride) {
  const std::size_t e = ids[slot];
  GemvQ8rMulti<4, 6144>(qs + e * rows * qstride, d + e * rows * (cols / 32), x, y, rows, cols, qstride);
}
extern "C" __global__ void __launch_bounds__(256)
    halo_moe_down_q8r_r4(const std::int8_t* qs, const std::uint16_t* d, const std::uint32_t* ids, const float* h,
                         float* part, std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride) {
  const std::uint32_t slot = blockIdx.y;
  const std::size_t e = ids[slot];
  GemvQ8rMulti<4, 6144>(qs + e * rows * qstride, d + e * rows * (cols / 32), h + static_cast<std::size_t>(slot) * cols,
                        part + static_cast<std::size_t>(slot) * rows, rows, cols, qstride);
}

// ---- P3: multi-row block-format GEMVs (tuning) --------------------------------
// Lanes take (row, block) pairs over kRows rows, so short rows (Q5_1 down:
// 20 blocks of 32) still keep every lane busy and several loads in flight.

namespace {

template <typename Fmt, int kRows, std::uint32_t kMaxCols>
__device__ void GemvBlocksMulti(const std::uint8_t* w, std::uint64_t row_bytes, const float* x, float* y,
                                std::uint32_t rows, std::uint32_t cols) {
  __shared__ float xs[kMaxCols];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  if (r0 >= rows) return;
  const std::uint32_t lane = threadIdx.x % kWarp, per_row = cols / Fmt::kItemValues;
  float acc[kRows] = {};
  for (std::uint32_t t = lane; t < kRows * per_row; t += kWarp) {
    const std::uint32_t k = t / per_row, i = t % per_row;
    const std::uint32_t r = min(r0 + k, rows - 1);
    const float v = Fmt::Item(w + r * row_bytes, xs, i);
#pragma unroll
    for (int kk = 0; kk < kRows; ++kk)
      if (kk == static_cast<int>(k)) acc[kk] += v;
  }
#pragma unroll
  for (int kk = 0; kk < kRows; ++kk) {
    const float a = WarpSum(acc[kk]);
    if (lane == 0 && r0 + kk < rows) y[r0 + kk] = a;
  }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256)
    halo_moe_down_q5_1_r4(const std::uint8_t* down, const std::uint32_t* ids, const float* h, float* part,
                          std::uint64_t expert_bytes, std::uint32_t rows, std::uint32_t cols, std::uint64_t row_bytes) {
  const std::uint32_t slot = blockIdx.y;
  GemvBlocksMulti<DotQ5_1, 4, 2048>(down + ids[slot] * expert_bytes, row_bytes, h + static_cast<std::size_t>(slot) * cols,
                                    part + static_cast<std::size_t>(slot) * rows, rows, cols);
}

// gate and up rows r0..r0+1 per wave: lanes take (matrix, row, chunk) items.
extern "C" __global__ void __launch_bounds__(256)
    halo_moe_gate_up_q4_k_r2(const std::uint8_t* gate, const std::uint8_t* up, const std::uint32_t* ids,
                             const float* x, float* h, std::uint64_t expert_bytes, std::uint32_t rows,
                             std::uint32_t cols, std::uint64_t row_bytes) {
  constexpr int kRows = 2;
  __shared__ float xs[4096];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const std::uint32_t slot = blockIdx.y;
  const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  if (r0 >= rows) return;
  const std::uint32_t lane = threadIdx.x % kWarp, per_row = cols / 64;
  const std::size_t e = ids[slot];
  float g[kRows] = {}, u[kRows] = {};
  for (std::uint32_t t = lane; t < 2 * kRows * per_row; t += kWarp) {
    const std::uint32_t m = t / (kRows * per_row), rem = t % (kRows * per_row), k = rem / per_row, i = rem % per_row;
    const std::uint32_t r = min(r0 + k, rows - 1);
    const float v = DotQ4K::Item((m == 0 ? gate : up) + e * expert_bytes + r * row_bytes, xs, i);
#pragma unroll
    for (int kk = 0; kk < kRows; ++kk) {
      if (kk == static_cast<int>(k) && m == 0) g[kk] += v;
      if (kk == static_cast<int>(k) && m == 1) u[kk] += v;
    }
  }
#pragma unroll
  for (int kk = 0; kk < kRows; ++kk) {
    const float gs = WarpSum(g[kk]), us = WarpSum(u[kk]);
    if (lane == 0 && r0 + kk < rows) h[static_cast<std::size_t>(slot) * rows + r0 + kk] = Silu(gs) * us;
  }
}

// Q4_K with x sub-block sums precomputed: the min term needs sum(x) over each
// 32-value sub-block, which does not depend on the row.
namespace {

__device__ inline float Q4KItemSums(const std::uint8_t* row, const float* xs, const float* xsum, std::uint32_t item) {
  const std::uint32_t sb = item / 4, c = item % 4;
  const std::uint8_t* b = row + sb * 144;
  const int4 hdr = *reinterpret_cast<const int4*>(b);
  const int4 qa = *reinterpret_cast<const int4*>(b + 16 + c * 32);
  const int4 qb = *reinterpret_cast<const int4*>(b + 16 + c * 32 + 16);
  const auto w0 = static_cast<std::uint32_t>(hdr.y), w1 = static_cast<std::uint32_t>(hdr.z),
             w2 = static_cast<std::uint32_t>(hdr.w);
  std::uint32_t sc0, m0, sc1, m1;
  ScaleMinK4Words(2 * c, w0, w1, w2, sc0, m0);
  ScaleMinK4Words(2 * c + 1, w0, w1, w2, sc1, m1);
  const std::uint8_t* q0 = reinterpret_cast<const std::uint8_t*>(&qa);
  const std::uint8_t* q1 = reinterpret_cast<const std::uint8_t*>(&qb);
  const float* x = xs + sb * 256 + c * 64;
  float lo = 0, hi = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    lo += static_cast<float>(q0[j] & 0xF) * x[j] + static_cast<float>(q1[j] & 0xF) * x[16 + j];
    hi += static_cast<float>(q0[j] >> 4) * x[32 + j] + static_cast<float>(q1[j] >> 4) * x[48 + j];
  }
  const std::uint32_t s0 = (sb * 256 + c * 64) / 32;
  return F16(static_cast<std::uint16_t>(hdr.x & 0xFFFF)) * (static_cast<float>(sc0) * lo + static_cast<float>(sc1) * hi) -
         F16(static_cast<std::uint16_t>(static_cast<std::uint32_t>(hdr.x) >> 16)) *
             (static_cast<float>(m0) * xsum[s0] + static_cast<float>(m1) * xsum[s0 + 1]);
}

template <int kRows>
__device__ void MoeGateUpQ4K(const std::uint8_t* gate, const std::uint8_t* up, const std::uint32_t* ids,
                             const float* x, float* h, std::uint64_t expert_bytes, std::uint32_t rows,
                             std::uint32_t cols, std::uint64_t row_bytes) {
  __shared__ float xs[4096], xsum[128];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  for (std::uint32_t j = threadIdx.x; j < cols / 32; j += blockDim.x) {
    float a = 0;
    for (int i = 0; i < 32; ++i) a += xs[j * 32 + i];
    xsum[j] = a;
  }
  __syncthreads();
  const std::uint32_t slot = blockIdx.y;
  const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  if (r0 >= rows) return;
  const std::uint32_t lane = threadIdx.x % kWarp, per_row = cols / 64;
  const std::size_t e = ids[slot];
  float g[kRows] = {}, u[kRows] = {};
  for (std::uint32_t t = lane; t < 2 * kRows * per_row; t += kWarp) {
    const std::uint32_t m = t / (kRows * per_row), rem = t % (kRows * per_row), k = rem / per_row, i = rem % per_row;
    const std::uint32_t r = min(r0 + k, rows - 1);
    const float v = Q4KItemSums((m == 0 ? gate : up) + e * expert_bytes + r * row_bytes, xs, xsum, i);
#pragma unroll
    for (int kk = 0; kk < kRows; ++kk) {
      if (kk == static_cast<int>(k) && m == 0) g[kk] += v;
      if (kk == static_cast<int>(k) && m == 1) u[kk] += v;
    }
  }
#pragma unroll
  for (int kk = 0; kk < kRows; ++kk) {
    const float gs = WarpSum(g[kk]), us = WarpSum(u[kk]);
    if (lane == 0 && r0 + kk < rows) h[static_cast<std::size_t>(slot) * rows + r0 + kk] = Silu(gs) * us;
  }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256)
    halo_moe_gate_up_q4_k_s2(const std::uint8_t* gate, const std::uint8_t* up, const std::uint32_t* ids,
                             const float* x, float* h, std::uint64_t expert_bytes, std::uint32_t rows,
                             std::uint32_t cols, std::uint64_t row_bytes) {
  MoeGateUpQ4K<2>(gate, up, ids, x, h, expert_bytes, rows, cols, row_bytes);
}
extern "C" __global__ void __launch_bounds__(256)
    halo_moe_gate_up_q4_k_s4(const std::uint8_t* gate, const std::uint8_t* up, const std::uint32_t* ids,
                             const float* x, float* h, std::uint64_t expert_bytes, std::uint32_t rows,
                             std::uint32_t cols, std::uint64_t row_bytes) {
  MoeGateUpQ4K<4>(gate, up, ids, x, h, expert_bytes, rows, cols, row_bytes);
}

// ---- P3: single-pass Gated DeltaNet step (head dim 128) -----------------------
// With A = S k and B = S q over the old state, the update and the output are
//   u = decay A, delta = (v - u) b, S' = decay S + delta k^T, o = decay B + delta (k.q)
// so each thread reads its state column once into registers and writes it
// once (the two-loop kernel reads it twice and writes it twice).
extern "C" __global__ void __launch_bounds__(128)
    halo_gdn_step_d128(const float* q, const float* k, const float* v, const float* alpha, const float* beta,
                       const float* a, const float* dt, float* state, float* out, std::uint32_t num_k_heads) {
  constexpr std::uint32_t d = 128;
  __shared__ float ks[d], qs[d], red[32];
  const std::uint32_t h = blockIdx.x, j = threadIdx.x;
  const float* qh = q + (h % num_k_heads) * d;
  const float* kh = k + (h % num_k_heads) * d;
  ks[j] = kh[j];
  qs[j] = qh[j];
  const float kq = BlockSum(kh[j] * qh[j], red);  // syncs, so ks/qs are visible
  const float decay = expf(a[h] * Softplus(alpha[h] + dt[h]));
  const float b = Sigmoid(beta[h]);
  float* S = state + static_cast<std::size_t>(h) * d * d + j;
  float s[d];
  float A = 0.0f, B = 0.0f;
#pragma unroll
  for (std::uint32_t i = 0; i < d; ++i) {
    s[i] = S[i * d];
    A += s[i] * ks[i];
    B += s[i] * qs[i];
  }
  const float delta = (v[h * d + j] - decay * A) * b;
#pragma unroll
  for (std::uint32_t i = 0; i < d; ++i) S[i * d] = decay * s[i] + delta * ks[i];
  out[h * d + j] = (decay * B + delta * kq) / sqrtf(static_cast<float>(d));
}

// ---- P3: Q4_K gate+up with int8 activations and integer dot products --------
// x is quantized once per block to int8 with one scale per 32 values (as
// llama.cpp's Q8_1 activations); nibbles multiply them with v_dot4 (4
// multiply-adds per instruction) instead of 64 float conversions and FMAs
// per chunk. The minimum term keeps exact float sums of x.
//   value = d * (sc0 * sx0 * <q_lo, xq0> + sc1 * sx1 * <q_hi, xq1>) - dmin * (m0 * sum0 + m1 * sum1)

namespace {

struct Q8Act {
  std::uint32_t q[1024];  // int8 x, 4 per word, 4096 values
  float scale[128];       // per 32 values
  float sum[128];         // exact float sums per 32 values
};

// Fills `a` from x (cols <= 4096, cols % 32 == 0). One warp per 32 values.
__device__ inline void QuantizeActivations(const float* x, std::uint32_t cols, Q8Act& a) {
  const std::uint32_t lane = threadIdx.x % kWarp;
  for (std::uint32_t blk = threadIdx.x / kWarp; blk < cols / 32; blk += blockDim.x / kWarp) {
    const float v = x[blk * 32 + lane];
    float amax = fabsf(v), sum = v;
    for (int o = kWarp / 2; o > 0; o >>= 1) {
      amax = fmaxf(amax, __shfl_xor(amax, o, kWarp));
      sum += __shfl_xor(sum, o, kWarp);
    }
    const float scale = amax / 127.0f;
    const int qv = scale > 0.0f ? static_cast<int>(rintf(v / scale)) : 0;
    // Pack 4 lanes' bytes into one word.
    const std::uint32_t byte = static_cast<std::uint32_t>(qv & 0xFF) << (8 * (lane & 3));
    std::uint32_t word = byte;
    word |= __shfl_xor(word, 1, kWarp);
    word |= __shfl_xor(word, 2, kWarp);
    if ((lane & 3) == 0) a.q[blk * 8 + lane / 4] = word;
    if (lane == 0) {
      a.scale[blk] = scale;
      a.sum[blk] = sum;
    }
  }
}

__device__ inline float Q4KItemDot(const std::uint8_t* row, const Q8Act& a, std::uint32_t item) {
  const std::uint32_t sb = item / 4, c = item % 4;
  const std::uint8_t* b = row + sb * 144;
  const int4 hdr = *reinterpret_cast<const int4*>(b);
  const int4 qa = *reinterpret_cast<const int4*>(b + 16 + c * 32);
  const int4 qb = *reinterpret_cast<const int4*>(b + 16 + c * 32 + 16);
  const auto w0 = static_cast<std::uint32_t>(hdr.y), w1 = static_cast<std::uint32_t>(hdr.z),
             w2 = static_cast<std::uint32_t>(hdr.w);
  std::uint32_t sc0, m0, sc1, m1;
  ScaleMinK4Words(2 * c, w0, w1, w2, sc0, m0);
  ScaleMinK4Words(2 * c + 1, w0, w1, w2, sc1, m1);
  const std::uint32_t s0 = sb * 8 + 2 * c, s1 = s0 + 1;
  const std::uint32_t* x0 = a.q + s0 * 8;  // values 0..31 of sub-block s0 (low nibbles)
  const std::uint32_t* x1 = a.q + s1 * 8;  // values 0..31 of sub-block s1 (high nibbles)
  const std::uint32_t qw[8] = {static_cast<std::uint32_t>(qa.x), static_cast<std::uint32_t>(qa.y),
                               static_cast<std::uint32_t>(qa.z), static_cast<std::uint32_t>(qa.w),
                               static_cast<std::uint32_t>(qb.x), static_cast<std::uint32_t>(qb.y),
                               static_cast<std::uint32_t>(qb.z), static_cast<std::uint32_t>(qb.w)};
  int lo = 0, hi = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    lo = __builtin_amdgcn_sudot4(false, static_cast<int>(qw[k] & 0x0F0F0F0Fu), true, static_cast<int>(x0[k]), lo, false);
    hi = __builtin_amdgcn_sudot4(false, static_cast<int>((qw[k] >> 4) & 0x0F0F0F0Fu), true, static_cast<int>(x1[k]), hi,
                                 false);
  }
  const float d = F16(static_cast<std::uint16_t>(hdr.x & 0xFFFF));
  const float dmin = F16(static_cast<std::uint16_t>(static_cast<std::uint32_t>(hdr.x) >> 16));
  return d * (static_cast<float>(sc0) * a.scale[s0] * static_cast<float>(lo) +
              static_cast<float>(sc1) * a.scale[s1] * static_cast<float>(hi)) -
         dmin * (static_cast<float>(m0) * a.sum[s0] + static_cast<float>(m1) * a.sum[s1]);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256)
    halo_moe_gate_up_q4_k_dot(const std::uint8_t* gate, const std::uint8_t* up, const std::uint32_t* ids,
                              const float* x, float* h, std::uint64_t expert_bytes, std::uint32_t rows,
                              std::uint32_t cols, std::uint64_t row_bytes) {
  constexpr int kRows = 2;
  __shared__ Q8Act act;
  QuantizeActivations(x, cols, act);
  __syncthreads();
  const std::uint32_t slot = blockIdx.y;
  const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  if (r0 >= rows) return;
  const std::uint32_t lane = threadIdx.x % kWarp, per_row = cols / 64;
  const std::size_t e = ids[slot];
  float g[kRows] = {}, u[kRows] = {};
  for (std::uint32_t t = lane; t < 2 * kRows * per_row; t += kWarp) {
    const std::uint32_t m = t / (kRows * per_row), rem = t % (kRows * per_row), k = rem / per_row, i = rem % per_row;
    const std::uint32_t r = min(r0 + k, rows - 1);
    const float v = Q4KItemDot((m == 0 ? gate : up) + e * expert_bytes + r * row_bytes, act, i);
#pragma unroll
    for (int kk = 0; kk < kRows; ++kk) {
      if (kk == static_cast<int>(k) && m == 0) g[kk] += v;
      if (kk == static_cast<int>(k) && m == 1) u[kk] += v;
    }
  }
#pragma unroll
  for (int kk = 0; kk < kRows; ++kk) {
    const float gs = WarpSum(g[kk]), us = WarpSum(u[kk]);
    if (lane == 0 && r0 + kk < rows) h[static_cast<std::size_t>(slot) * rows + r0 + kk] = Silu(gs) * us;
  }
}

// ---- P3: fused hyper-connection mixer (6 launches -> 2) ------------------------
//
// A: grouped RMSNorm of the residual (staged in LDS) -> Q8 down projection
//    with the silu(y / hc) epilogue -> lo; extra blocks compute the F32
//    inject rows from the same normalized input; block 0 writes xn.
// B: Q8 up projection, one wave per hidden index i over rows s*H + i
//    (s < hc), with the hc_mix epilogue: mixed[i] = mean_s xn[s*H+i] * sigmoid(g_s).

namespace {
constexpr std::uint32_t kHcMaxDim = 12288;  // hc * hidden
constexpr std::uint32_t kHcMaxLow = 1024;   // low rank
}  // namespace

// Per-stream RMS scales of the residual for the fused mixer: one block per
// stream, scale[s] = 1 / sqrt(mean(res_s^2) + eps).
extern "C" __global__ void halo_hc_norm_scales(const float* res, float* scale, std::uint32_t h, float eps) {
  __shared__ float scratch[32];
  const float* x = res + blockIdx.x * h;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < h; i += blockDim.x) ss += x[i] * x[i];
  ss = BlockSum(ss, scratch);
  if (threadIdx.x == 0) scale[blockIdx.x] = 1.0f / sqrtf(ss / static_cast<float>(h) + eps);
}

extern "C" __global__ void __launch_bounds__(256)
    halo_hc_mix_a(const float* res, const float* norm_w, const float* scale, const std::int8_t* dqs,
                  const std::uint16_t* dd, const float* inject_w, float* xn, float* lo_part, float* inject_part,
                  std::uint32_t hc, std::uint32_t h, std::uint32_t low_rank, std::uint32_t down_blocks,
                  std::uint32_t qstride, std::uint32_t chunk) {
  // Split over columns: block (bx, ky) normalizes and covers columns
  // [ky * chunk, ...) of its rows and writes partial sums; halo_hc_mix_b
  // finishes them. The stream scales come from halo_hc_norm_scales.
  __shared__ float xs[kHcMaxDim];
  __shared__ float scratch[32];
  const std::uint32_t D = hc * h, ky = blockIdx.y, c0 = ky * chunk, c1 = min(D, c0 + chunk);
  for (std::uint32_t i = c0 + threadIdx.x; i < c1; i += blockDim.x) xs[i] = res[i] * scale[i / h] * norm_w[i];
  __syncthreads();
  if (blockIdx.x == 0)
    for (std::uint32_t i = c0 + threadIdx.x; i < c1; i += blockDim.x) xn[i] = xs[i];
  const std::uint32_t lane = threadIdx.x % kWarp;
  if (blockIdx.x >= down_blocks) {  // inject row: the whole block on one row's chunk
    const std::uint32_t r = blockIdx.x - down_blocks;
    const float* w = inject_w + static_cast<std::size_t>(r) * D;
    float acc = 0.0f;
    for (std::uint32_t c = c0 + threadIdx.x; c < c1; c += blockDim.x) acc += w[c] * xs[c];
    acc = BlockSum(acc, scratch);
    if (threadIdx.x == 0) inject_part[ky * hc + r] = acc;
    return;
  }
  constexpr int kRows = 2;
  const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  if (r0 >= low_rank) return;
  const std::uint32_t nblk = D / 32;
  float acc[kRows] = {};
  for (std::uint32_t c = c0 + lane * 16; c < c1; c += kWarp * 16) {
    float xr[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) xr[i] = xs[c + i];
#pragma unroll
    for (int k = 0; k < kRows; ++k) {
      const std::uint32_t r = min(r0 + k, low_rank - 1);
      const int4 q = *reinterpret_cast<const int4*>(dqs + static_cast<std::size_t>(r) * qstride + c);
      const float sc = F16(dd[static_cast<std::size_t>(r) * nblk + c / 32]);
      const std::int8_t* v = reinterpret_cast<const std::int8_t*>(&q);
      float part = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) part += static_cast<float>(v[i]) * xr[i];
      acc[k] += sc * part;
    }
  }
#pragma unroll
  for (int k = 0; k < kRows; ++k) {
    const float a = WarpSum(acc[k]);
    if (lane == 0 && r0 + k < low_rank) lo_part[ky * low_rank + r0 + k] = a;
  }
}

extern "C" __global__ void __launch_bounds__(256)
    halo_hc_mix_b(const float* lo_part, std::uint32_t parts, const float* inject_part, float* inject,
                  std::uint32_t inject_rows, const std::int8_t* uqs, const std::uint16_t* ud, const float* xn,
                  float* mixed, std::uint32_t hc, std::uint32_t h, std::uint32_t low_rank, std::uint32_t qstride) {
  // Finishes halo_hc_mix_a: lo = silu(sum of partials / hc), inject = sum
  // of partials (block 0), in a fixed order.
  __shared__ float ls[kHcMaxLow];
  for (std::uint32_t i = threadIdx.x; i < low_rank; i += blockDim.x) {
    float a = 0.0f;
    for (std::uint32_t p = 0; p < parts; ++p) a += lo_part[p * low_rank + i];
    ls[i] = Silu(a / static_cast<float>(hc));
  }
  if (blockIdx.x == 0 && threadIdx.x < inject_rows) {
    float a = 0.0f;
    for (std::uint32_t p = 0; p < parts; ++p) a += inject_part[p * hc + threadIdx.x];
    inject[threadIdx.x] = a;
  }
  __syncthreads();
  const std::uint32_t i = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
  if (i >= h) return;
  const std::uint32_t lane = threadIdx.x % kWarp, nblk = low_rank / 32;
  float acc[4] = {};
  for (std::uint32_t c = lane * 16; c < low_rank; c += kWarp * 16) {
    float xr[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) xr[j] = ls[c + j];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      if (static_cast<std::uint32_t>(s) >= hc) break;
      const std::size_t r = static_cast<std::size_t>(s) * h + i;
      const int4 q = *reinterpret_cast<const int4*>(uqs + r * qstride + c);
      const float sc = F16(ud[r * nblk + c / 32]);
      const std::int8_t* v = reinterpret_cast<const std::int8_t*>(&q);
      float part = 0.0f;
#pragma unroll
      for (int j = 0; j < 16; ++j) part += static_cast<float>(v[j]) * xr[j];
      acc[s] += sc * part;
    }
  }
  float m = 0.0f;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const float g = WarpSum(acc[s]);
    if (static_cast<std::uint32_t>(s) < hc) m += xn[s * h + i] * Sigmoid(g);
  }
  if (lane == 0) mixed[i] = m / static_cast<float>(hc);
}

// ---- P3: fused shared expert (6 launches -> 2) --------------------------------
//   gate_up:  h = silu(<gate_r, x>) * <up_r, x>; block 0 also writes
//             sgate = <gate_inp, x> (the shared expert's scalar gate logit)
//   down_add: out[r] += sigmoid(sgate) * <down_r, h>   (out holds the routed sum)

namespace {

// Dot products of kRows consecutive repacked-Q8 rows with xs (LDS), reduced
// over the wave; every lane gets the sums.
template <int kRows>
__device__ inline void Q8RowsDot(const std::int8_t* qs, const std::uint16_t* d, const float* xs, std::uint32_t r0,
                                 std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride, float (&out)[kRows]) {
  const std::uint32_t lane = threadIdx.x % kWarp, nblk = cols / 32;
  float acc[kRows] = {};
  for (std::uint32_t c = lane * 16; c < cols; c += kWarp * 16) {
    float xr[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) xr[i] = xs[c + i];
#pragma unroll
    for (int k = 0; k < kRows; ++k) {
      const std::uint32_t r = min(r0 + k, rows - 1);
      const int4 q = *reinterpret_cast<const int4*>(qs + static_cast<std::size_t>(r) * qstride + c);
      const float sc = F16(d[static_cast<std::size_t>(r) * nblk + c / 32]);
      const std::int8_t* v = reinterpret_cast<const std::int8_t*>(&q);
      float part = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) part += static_cast<float>(v[i]) * xr[i];
      acc[k] += sc * part;
    }
  }
#pragma unroll
  for (int k = 0; k < kRows; ++k) out[k] = WarpSum(acc[k]);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256)
    halo_shexp_gate_up(const std::int8_t* gq, const std::uint16_t* gd, const std::int8_t* uq, const std::uint16_t* ud,
                       const float* gate_inp, const float* x, float* h, float* sgate, std::uint32_t rows,
                       std::uint32_t cols, std::uint32_t qstride) {
  __shared__ float xs[kGemvMaxCols];
  __shared__ float scratch[32];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  if (blockIdx.x == 0) {
    float a = 0.0f;
    for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) a += gate_inp[i] * xs[i];
    a = BlockSum(a, scratch);
    if (threadIdx.x == 0) sgate[0] = a;
  }
  constexpr int kRows = 2;
  const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  if (r0 >= rows) return;
  float g[kRows], u[kRows];
  Q8RowsDot<kRows>(gq, gd, xs, r0, rows, cols, qstride, g);
  Q8RowsDot<kRows>(uq, ud, xs, r0, rows, cols, qstride, u);
  if (threadIdx.x % kWarp == 0)
#pragma unroll
    for (int k = 0; k < kRows; ++k)
      if (r0 + k < rows) h[r0 + k] = Silu(g[k]) * u[k];
}

extern "C" __global__ void __launch_bounds__(256)
    halo_shexp_down_add(const std::int8_t* dq, const std::uint16_t* dd, const float* hin, const float* sgate, float* out,
                        std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride) {
  __shared__ float hs[kGemvMaxCols];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) hs[i] = hin[i];
  __syncthreads();
  constexpr int kRows = 4;
  const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  if (r0 >= rows) return;
  float y[kRows];
  Q8RowsDot<kRows>(dq, dd, hs, r0, rows, cols, qstride, y);
  const float s = Sigmoid(sgate[0]);
  if (threadIdx.x % kWarp == 0)
#pragma unroll
    for (int k = 0; k < kRows; ++k)
      if (r0 + k < rows) out[r0 + k] += s * y[k];
}

// ---- P3: fused DeltaNet block pieces -------------------------------------------

// Two repacked-Q8 matrices with the same input in one launch (4 rows per
// wave): blocks [0, blocks1) take matrix 1, the rest matrix 2.
extern "C" __global__ void __launch_bounds__(256)
    halo_gemv_q8r_dual(const std::int8_t* q1, const std::uint16_t* d1, float* y1, std::uint32_t rows1,
                       std::uint32_t qstride1, const std::int8_t* q2, const std::uint16_t* d2, float* y2,
                       std::uint32_t rows2, std::uint32_t qstride2, const float* x, std::uint32_t cols,
                       std::uint32_t blocks1) {
  __shared__ float xs[12288];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const bool first = blockIdx.x < blocks1;
  const std::uint32_t b = first ? blockIdx.x : blockIdx.x - blocks1;
  const std::int8_t* qs = first ? q1 : q2;
  const std::uint16_t* d = first ? d1 : d2;
  float* y = first ? y1 : y2;
  const std::uint32_t rows = first ? rows1 : rows2, qstride = first ? qstride1 : qstride2;
  constexpr int kRows = 4;
  const std::uint32_t r0 = (b * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  if (r0 >= rows) return;
  float acc[kRows];
  Q8RowsDot<kRows>(qs, d, xs, r0, rows, cols, qstride, acc);
  if (threadIdx.x % kWarp == 0)
#pragma unroll
    for (int k = 0; k < kRows; ++k)
      if (r0 + k < rows) y[r0 + k] = acc[k];
}

// Two few-row F32 matvecs with the same input (DeltaNet alpha and beta): a
// block per row, blocks [0, rows) for wa, [rows, 2 rows) for wb.
extern "C" __global__ void __launch_bounds__(256)
    halo_matvec_f32_rowblock2(const float* wa, const float* wb, const float* x, float* ya, float* yb,
                              std::uint32_t rows, std::uint32_t cols) {
  __shared__ float scratch[32];
  const bool a = blockIdx.x < rows;
  const std::uint32_t r = a ? blockIdx.x : blockIdx.x - rows;
  const float* row = (a ? wa : wb) + static_cast<std::size_t>(r) * cols;
  float acc = 0.0f;
  for (std::uint32_t c = threadIdx.x; c < cols; c += blockDim.x) acc += row[c] * x[c];
  acc = BlockSum(acc, scratch);
  if (threadIdx.x == 0) (a ? ya : yb)[r] = acc;
}

// Causal conv step (as halo_conv_step) and, for the first l2_channels
// channels (DeltaNet q and k), the per-head L2 norm. One block per head of d
// channels (blockDim == d).
extern "C" __global__ void halo_conv_l2norm(const float* in, float* state, const float* w, float* out,
                                            std::uint32_t channels, std::uint32_t kern, std::uint32_t l2_channels,
                                            float eps) {
  __shared__ float scratch[32];
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  float v = 0.0f;
  if (ch < channels) {
    float acc = w[ch * kern + kern - 1] * in[ch];
    for (std::uint32_t k = 0; k + 1 < kern; ++k) acc += w[ch * kern + k] * state[k * channels + ch];
    for (std::uint32_t k = 0; k + 2 < kern; ++k) state[k * channels + ch] = state[(k + 1) * channels + ch];
    if (kern > 1) state[(kern - 2) * channels + ch] = in[ch];
    v = Silu(acc);
  }
  if (blockIdx.x * blockDim.x < l2_channels) {  // block-uniform
    const float ss = BlockSum(v * v, scratch);
    v *= 1.0f / sqrtf(ss + eps);
  }
  if (ch < channels) out[ch] = v;
}

// halo_gdn_step_d128 with the per-head gated RMSNorm in the epilogue
// (mode 0: silu(z), mode 1: sigmoid(z)), replacing halo_gated_norm.
extern "C" __global__ void __launch_bounds__(128)
    halo_gdn_step_d128_norm(const float* q, const float* k, const float* v, const float* alpha, const float* beta,
                            const float* a, const float* dt, float* state, const float* norm_w, const float* z,
                            float* out, std::uint32_t num_k_heads, float eps, std::uint32_t mode) {
  constexpr std::uint32_t d = 128;
  __shared__ float ks[d], qs[d], red[32];
  const std::uint32_t h = blockIdx.x, j = threadIdx.x;
  const float* qh = q + (h % num_k_heads) * d;
  const float* kh = k + (h % num_k_heads) * d;
  ks[j] = kh[j];
  qs[j] = qh[j];
  const float kq = BlockSum(kh[j] * qh[j], red);
  const float decay = expf(a[h] * Softplus(alpha[h] + dt[h]));
  const float b = Sigmoid(beta[h]);
  float* S = state + static_cast<std::size_t>(h) * d * d + j;
  float s[d];
  float A = 0.0f, B = 0.0f;
#pragma unroll
  for (std::uint32_t i = 0; i < d; ++i) {
    s[i] = S[i * d];
    A += s[i] * ks[i];
    B += s[i] * qs[i];
  }
  const float delta = (v[h * d + j] - decay * A) * b;
#pragma unroll
  for (std::uint32_t i = 0; i < d; ++i) S[i * d] = decay * s[i] + delta * ks[i];
  const float o = (decay * B + delta * kq) / sqrtf(static_cast<float>(d));
  const float ss = BlockSum(o * o, red);
  const float scale = 1.0f / sqrtf(ss / static_cast<float>(d) + eps);
  const float zv = z[h * d + j];
  out[h * d + j] = o * scale * norm_w[j] * (mode == 0 ? Silu(zv) : Sigmoid(zv));
}

// ---- P4: prefill GEMMs on RDNA3 WMMA ------------------------------------------
//
// gfx11 wave32 WMMA, 16x16x16 int8 -> int32. Operand layout (checked by
// GpuKernels.WmmaInt8LayoutProbe): lane l holds row (l % 16) of A and row
// (l % 16) of B (B given N-major, i.e. one token's K values per lane), 16
// int8 each, with lanes 16..31 repeating lanes 0..15; result element j of
// lane l is C[2j + l / 16][l % 16].

typedef int halo_v4i __attribute__((ext_vector_type(4)));
typedef int halo_v8i __attribute__((ext_vector_type(8)));

extern "C" __global__ void halo_wmma_probe_i8(const std::int8_t* a, const std::int8_t* b, int* c) {
  const std::uint32_t lane = threadIdx.x % 32;
  const halo_v4i av = *reinterpret_cast<const halo_v4i*>(a + (lane % 16) * 16);
  const halo_v4i bv = *reinterpret_cast<const halo_v4i*>(b + (lane % 16) * 16);
  halo_v8i acc = {};
  acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av, true, bv, acc, false);
  for (int j = 0; j < 8; ++j) c[(2 * j + lane / 16) * 16 + lane % 16] = acc[j];
}

// Activations for the int8 GEMMs: per token row, one scale per 32 values
// (amax / 127, round to nearest), as the decode int8 path quantizes them.
//   grid (T), block 256; x [T][cols] -> xq [T][cols], xd [T][cols / 32]
extern "C" __global__ void __launch_bounds__(256)
    halo_quantize_q8_rows(const float* x, std::int8_t* xq, float* xd, std::uint32_t cols) {
  const std::size_t t = blockIdx.x;
  const std::uint32_t lane = threadIdx.x % kWarp;
  for (std::uint32_t blk = threadIdx.x / kWarp; blk < cols / 32; blk += blockDim.x / kWarp) {
    const float v = x[t * cols + blk * 32 + lane];
    float amax = fabsf(v);
    for (int o = kWarp / 2; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, kWarp));
    const float scale = amax / 127.0f;
    xq[t * cols + blk * 32 + lane] = static_cast<std::int8_t>(scale > 0.0f ? static_cast<int>(rintf(v / scale)) : 0);
    if (lane == 0) xd[t * (cols / 32) + blk] = scale;
  }
}

// y[t][r] = sum_c W[r][c] x[t][c] for repacked Q8_0 W (int8 rows, fp16 scale
// per 32) and quantized x (halo_quantize_q8_rows). A wave computes 16 rows x
// 64 tokens; a block of 4 waves 64 rows. Grid (ceil(T / 64), ceil(rows / 64)):
// token tiles vary fastest, so neighbouring blocks share weight rows in L2.
// Requires cols % 32 == 0 and 16-byte aligned rows (qstride % 16 == 0).
extern "C" __global__ void __launch_bounds__(128)
    halo_gemm_q8r_wmma(const std::int8_t* qs, const std::uint16_t* d, const std::int8_t* xq, const float* xd, float* y,
                       std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride, std::uint32_t tokens) {
  constexpr int kSub = 4;  // 16-token subtiles per wave
  const std::uint32_t lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  const std::uint32_t r0 = (blockIdx.y * 4 + wave) * 16, t0 = blockIdx.x * 64;
  if (r0 >= rows) return;
  const std::uint32_t nblk = cols / 32;
  const std::uint32_t my_row = min(r0 + lane % 16, rows - 1);
  const std::int8_t* wrow = qs + static_cast<std::size_t>(my_row) * qstride;
  std::uint32_t tok[kSub];
#pragma unroll
  for (int s = 0; s < kSub; ++s) tok[s] = min(t0 + 16 * s + lane % 16, tokens - 1);
  float acc[kSub][8] = {};
  for (std::uint32_t b = 0; b < nblk; ++b) {
    const halo_v4i a0 = *reinterpret_cast<const halo_v4i*>(wrow + b * 32);
    const halo_v4i a1 = *reinterpret_cast<const halo_v4i*>(wrow + b * 32 + 16);
    // Row scales of this block for the 8 result rows this lane holds.
    const float dw_mine = F16(d[static_cast<std::size_t>(my_row) * nblk + b]);
    float dw[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) dw[j] = __shfl(dw_mine, 2 * j + lane / 16, 32);
#pragma unroll
    for (int s = 0; s < kSub; ++s) {
      const std::int8_t* xrow = xq + static_cast<std::size_t>(tok[s]) * cols + b * 32;
      const halo_v4i b0 = *reinterpret_cast<const halo_v4i*>(xrow);
      const halo_v4i b1 = *reinterpret_cast<const halo_v4i*>(xrow + 16);
      halo_v8i c = {};
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, c, false);
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, c, false);
      const float dx = xd[static_cast<std::size_t>(tok[s]) * nblk + b];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[s][j] += static_cast<float>(c[j]) * (dw[j] * dx);
    }
  }
#pragma unroll
  for (int s = 0; s < kSub; ++s) {
    const std::uint32_t t = t0 + 16 * s + lane % 16;
    if (t >= tokens) continue;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t r = r0 + 2 * j + lane / 16;
      if (r < rows) y[static_cast<std::size_t>(t) * rows + r] = acc[s][j];
    }
  }
}

// FP16 WMMA, 16x16x16 -> fp32 (layout checked by GpuKernels.WmmaF16LayoutProbe):
// lane l holds 16 halves of row (l % 16) of A and of B (N-major), lanes
// 16..31 repeating 0..15; result element j of lane l is C[2j + l / 16][l % 16].
typedef _Float16 halo_v16h __attribute__((ext_vector_type(16)));
typedef float halo_v8f __attribute__((ext_vector_type(8)));

extern "C" __global__ void halo_wmma_probe_f16(const _Float16* a, const _Float16* b, float* c) {
  const std::uint32_t lane = threadIdx.x % 32;
  const halo_v16h av = *reinterpret_cast<const halo_v16h*>(a + (lane % 16) * 16);
  const halo_v16h bv = *reinterpret_cast<const halo_v16h*>(b + (lane % 16) * 16);
  halo_v8f acc = {};
  acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av, bv, acc);
  for (int j = 0; j < 8; ++j) c[(2 * j + lane / 16) * 16 + lane % 16] = acc[j];
}

// Peak probe: `iters` rounds of 4 independent int8 (or fp16) WMMAs on
// registers per wave; no memory traffic except one store.
extern "C" __global__ void __launch_bounds__(256) halo_wmma_peak(int* out, std::uint32_t iters, std::uint32_t fp16) {
  const std::uint32_t lane = threadIdx.x % 32;
  if (fp16) {
    halo_v16h a = {}, b = {};
    for (int i = 0; i < 16; ++i) {
      a[i] = static_cast<_Float16>(static_cast<float>((lane + i) % 7) * 0.01f);
      b[i] = static_cast<_Float16>(static_cast<float>((lane * 3 + i) % 5) * 0.01f);
    }
    halo_v8f c0 = {}, c1 = {}, c2 = {}, c3 = {};
    for (std::uint32_t it = 0; it < iters; ++it) {
      c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c0);
      c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, a, c1);
      c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, a, c2);
      c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, b, c3);
    }
    if (c0[0] + c1[1] + c2[2] + c3[3] == 12345.0f) out[0] = 1;
  } else {
    halo_v4i a = {static_cast<int>(lane), 3, 5, 7}, b = {1, static_cast<int>(lane), 2, 9};
    halo_v8i c0 = {}, c1 = {}, c2 = {}, c3 = {};
    for (std::uint32_t it = 0; it < iters; ++it) {
      c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c0, false);
      c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, b, true, a, c1, false);
      c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, a, c2, false);
      c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, b, true, b, c3, false);
    }
    if (c0[0] + c1[1] + c2[2] + c3[3] == 12345) out[0] = 1;
  }
}

// The production prefill GEMM: the block tile staged in LDS (rows padded
// against bank conflicts, double-buffered over 32-value K blocks), WI x WK
// WMMA tiles per wave (each fragment feeds WI or WK WMMAs, independent ones
// issued back to back), GR x GT waves per block. Scales are staged as floats,
// row scales de-interleaved so a lane's 8 rows (2j + lane / 16) are 8
// consecutive floats. Measured (docs/BENCHMARKS.md): ~20 TOPS of the 56
// TOPS int8 WMMA peak at 4 x 2 tiles and 2 x 4 waves; skipping the scale
// epilogue entirely reaches only ~23, so staging, not math, is the limit.
template <int WI, int WK, int GR, int GT>
__device__ void GemmQ8rWmma(const std::int8_t* qs, const std::uint16_t* d, const std::int8_t* xq, const float* xd,
                            float* y, std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride,
                            std::uint32_t tokens) {
  // Block tile BR x BT; wave (wr, wt) owns WI x WK 16 x 16 tiles.
  constexpr std::uint32_t BR = 16 * WI * GR, BT = 16 * WK * GT, NT = 32 * GR * GT;
  // Rows padded to 48 bytes: 16-byte reads by lanes at a 32-byte stride hit
  // the same banks 4 ways; at 48 they spread over all 32.
  constexpr std::uint32_t RS = 48;
  __shared__ __attribute__((aligned(16))) std::int8_t As[2][BR * RS];
  __shared__ __attribute__((aligned(16))) std::int8_t Bs[2][BT * RS];
  __shared__ __attribute__((aligned(16))) float Ds[2][BR];
  __shared__ float Xs[2][BT];
  const std::uint32_t tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t r0 = blockIdx.y * BR, t0 = blockIdx.x * BT;
  const std::uint32_t nblk = cols / 32;
  const std::uint32_t wr = wave % GR, wt = wave / GR;
  // Staging: 16-byte halves of A rows, B rows, row scales and token scales,
  // spread over the block's threads.
  const auto stage = [&](std::uint32_t b, std::uint32_t p) {
    for (std::uint32_t i = tid; i < BR * 2; i += NT) {
      const std::uint32_t row = i / 2, half = i % 2;
      *reinterpret_cast<halo_v4i*>(&As[p][row * RS + half * 16]) = *reinterpret_cast<const halo_v4i*>(
          qs + static_cast<std::size_t>(min(r0 + row, rows - 1)) * qstride + b * 32 + half * 16);
    }
    for (std::uint32_t i = tid; i < BT * 2; i += NT) {
      const std::uint32_t tok = i / 2, half = i % 2;
      *reinterpret_cast<halo_v4i*>(&Bs[p][tok * RS + half * 16]) = *reinterpret_cast<const halo_v4i*>(
          xq + static_cast<std::size_t>(min(t0 + tok, tokens - 1)) * cols + b * 32 + half * 16);
    }
    for (std::uint32_t row = tid; row < BR; row += NT)
      Ds[p][(row / 16) * 16 + (row % 2) * 8 + (row % 16) / 2] = F16(d[static_cast<std::size_t>(min(r0 + row, rows - 1)) * nblk + b]);
    for (std::uint32_t tok = tid; tok < BT; tok += NT) Xs[p][tok] = xd[static_cast<std::size_t>(min(t0 + tok, tokens - 1)) * nblk + b];
  };
  float acc[WI][WK][8] = {};
  stage(0, 0);
  __syncthreads();
  for (std::uint32_t b = 0; b < nblk; ++b) {
    const std::uint32_t p = b & 1;
    if (b + 1 < nblk) stage(b + 1, p ^ 1);
    halo_v4i a[WI][2], bv[WK][2];
#pragma unroll
    for (int i = 0; i < WI; ++i) {
      const std::uint32_t ar = (wr * 16 * WI + i * 16 + lane % 16) * RS;
      a[i][0] = *reinterpret_cast<const halo_v4i*>(&As[p][ar]);
      a[i][1] = *reinterpret_cast<const halo_v4i*>(&As[p][ar + 16]);
    }
#pragma unroll
    for (int k = 0; k < WK; ++k) {
      const std::uint32_t bt = (wt * 16 * WK + k * 16 + lane % 16) * RS;
      bv[k][0] = *reinterpret_cast<const halo_v4i*>(&Bs[p][bt]);
      bv[k][1] = *reinterpret_cast<const halo_v4i*>(&Bs[p][bt + 16]);
    }
    // Four independent WMMAs per K half, so the second half does not wait
    // on a result still in flight.
    halo_v8i c[WI][WK];
#pragma unroll
    for (int i = 0; i < WI; ++i)
#pragma unroll
      for (int k = 0; k < WK; ++k)
        c[i][k] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a[i][0], true, bv[k][0], halo_v8i{}, false);
#pragma unroll
    for (int i = 0; i < WI; ++i)
#pragma unroll
      for (int k = 0; k < WK; ++k)
        c[i][k] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a[i][1], true, bv[k][1], c[i][k], false);
#pragma unroll
    for (int i = 0; i < WI; ++i) {
      const float* dsp = &Ds[p][(wr * 16 * WI + i * 16) + (lane / 16) * 8];
      float dw[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) dw[j] = dsp[j];
#pragma unroll
      for (int k = 0; k < WK; ++k) {
        const float dx = Xs[p][wt * 16 * WK + k * 16 + lane % 16];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][k][j] += static_cast<float>(c[i][k][j]) * (dw[j] * dx);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < WI; ++i)
#pragma unroll
    for (int k = 0; k < WK; ++k) {
      const std::uint32_t t = t0 + wt * 16 * WK + k * 16 + lane % 16;
      if (t >= tokens) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t r = r0 + wr * 16 * WI + i * 16 + 2 * j + lane / 16;
        if (r < rows) y[static_cast<std::size_t>(t) * rows + r] = acc[i][k][j];
      }
    }
}


// 128 x 128 per block, 64 x 32 per wave.
//   grid (ceil(T / 128), ceil(rows / 128)), block 256
extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_q8r_wmma_t(const std::int8_t* qs, const std::uint16_t* d, const std::int8_t* xq, const float* xd,
                         float* y, std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride, std::uint32_t tokens) {
  GemmQ8rWmma<4, 2, 2, 4>(qs, d, xq, xd, y, rows, cols, qstride, tokens);
}

// ---- P4: DeltaNet over a prompt chunk ----------------------------------------
//
// halo_conv_l2norm for T tokens at once: token t's window is the previous
// kern - 1 inputs, taken from the chunk or, before it, from the conv state;
// the sums run in the decode kernel's order, so each token's output is
// bit-identical to stepping. The state is not touched (blocks of other
// tokens still read it); halo_conv_state_advance moves it afterwards.
//   in, out [T][channels]; grid (ceil(channels / d), T), block d
extern "C" __global__ void halo_conv_l2norm_prefill(const float* in, const float* state, const float* w, float* out,
                                                    std::uint32_t channels, std::uint32_t kern,
                                                    std::uint32_t l2_channels, float eps) {
  __shared__ float scratch[32];
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
  float v = 0.0f;
  if (ch < channels) {
    float acc = w[ch * kern + kern - 1] * in[static_cast<std::size_t>(t) * channels + ch];
    for (std::uint32_t k = 0; k + 1 < kern; ++k) {
      // Window slot k holds input t - (kern - 1) + k.
      const std::int64_t i = static_cast<std::int64_t>(t) - static_cast<std::int64_t>(kern - 1) + k;
      const float x = i >= 0 ? in[static_cast<std::size_t>(i) * channels + ch]
                             : state[static_cast<std::size_t>(i + kern - 1) * channels + ch];
      acc += w[ch * kern + k] * x;
    }
    v = Silu(acc);
  }
  if (blockIdx.x * blockDim.x < l2_channels) {  // block-uniform
    const float ss = BlockSum(v * v, scratch);
    v *= 1.0f / sqrtf(ss + eps);
  }
  if (ch < channels) out[static_cast<std::size_t>(t) * channels + ch] = v;
}

// The conv state after a chunk of T inputs: the last kern - 1 inputs, the
// older ones from the previous state when T < kern - 1.
//   grid (ceil(channels / 256)), block 256
extern "C" __global__ void halo_conv_state_advance(const float* in, float* state, std::uint32_t channels,
                                                   std::uint32_t kern, std::uint32_t tokens) {
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= channels || kern < 2) return;
  float next[8];  // kern <= 9
  for (std::uint32_t k = 0; k + 1 < kern; ++k) {
    const std::int64_t i = static_cast<std::int64_t>(tokens) - static_cast<std::int64_t>(kern - 1) + k;
    next[k] = i >= 0 ? in[static_cast<std::size_t>(i) * channels + ch]
                     : state[static_cast<std::size_t>(i + kern - 1) * channels + ch];
  }
  for (std::uint32_t k = 0; k + 1 < kern; ++k) state[static_cast<std::size_t>(k) * channels + ch] = next[k];
}

// halo_gdn_step_d128_norm over T tokens in one launch: the head's 128 x 128
// state stays in registers (a column per thread) across the chunk instead of
// a round trip to memory per token. Same arithmetic, same order.
//   row t of q and k at t * qk_stride, of v at t * v_stride, of z and out at
//   t * zo_stride; alpha, beta [T][heads]. grid (heads), block 128
extern "C" __global__ void __launch_bounds__(128)
    halo_gdn_prefill_d128_norm(const float* q, const float* k, const float* v, const float* alpha,
                               const float* beta, const float* a, const float* dt, float* state,
                               const float* norm_w, const float* z, float* out, std::uint32_t num_k_heads, float eps,
                               std::uint32_t mode, std::uint32_t tokens, std::uint32_t qk_stride,
                               std::uint32_t v_stride, std::uint32_t zo_stride) {
  constexpr std::uint32_t d = 128;
  __shared__ float ks[d], qs[d], red[32];
  const std::uint32_t h = blockIdx.x, j = threadIdx.x, heads = gridDim.x;
  float* S = state + static_cast<std::size_t>(h) * d * d + j;
  float s[d];
#pragma unroll
  for (std::uint32_t i = 0; i < d; ++i) s[i] = S[i * d];
  const float ah = a[h], dth = dt[h], wj = norm_w[j];
  for (std::uint32_t t = 0; t < tokens; ++t) {
    const float* qh = q + static_cast<std::size_t>(t) * qk_stride + (h % num_k_heads) * d;
    const float* kh = k + static_cast<std::size_t>(t) * qk_stride + (h % num_k_heads) * d;
    // No barrier needed here: the previous token's last BlockSum ends with
    // barriers after every read of ks / qs.
    ks[j] = kh[j];
    qs[j] = qh[j];
    const float kq = BlockSum(kh[j] * qh[j], red);
    const float decay = expf(ah * Softplus(alpha[static_cast<std::size_t>(t) * heads + h] + dth));
    const float b = Sigmoid(beta[static_cast<std::size_t>(t) * heads + h]);
    float A = 0.0f, B = 0.0f;
#pragma unroll
    for (std::uint32_t i = 0; i < d; ++i) {
      A += s[i] * ks[i];
      B += s[i] * qs[i];
    }
    const std::size_t vo = static_cast<std::size_t>(t) * v_stride + h * d + j;
    const std::size_t zo = static_cast<std::size_t>(t) * zo_stride + h * d + j;
    const float delta = (v[vo] - decay * A) * b;
#pragma unroll
    for (std::uint32_t i = 0; i < d; ++i) s[i] = decay * s[i] + delta * ks[i];
    const float o = (decay * B + delta * kq) / sqrtf(static_cast<float>(d));
    const float ss = BlockSum(o * o, red);
    const float scale = 1.0f / sqrtf(ss / static_cast<float>(d) + eps);
    const float zv = z[zo];
    out[zo] = o * scale * wj * (mode == 0 ? Silu(zv) : Sigmoid(zv));
  }
#pragma unroll
  for (std::uint32_t i = 0; i < d; ++i) S[i * d] = s[i];
}
