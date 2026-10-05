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
// Rows of `heads` heads per token, token t (= blockIdx.x / heads) at
// position pos0 + t. grid (T * heads)
extern "C" __global__ void halo_rope_neox_rows(float* x, std::uint32_t head_dim, std::uint32_t rotary_dim,
                                               std::uint32_t pos0, std::uint32_t heads, float theta) {
  RopeNeox(x, head_dim, rotary_dim, pos0 + blockIdx.x / heads, theta);
}
// Position from the step parameters: params[slot] * scale.
extern "C" __global__ void halo_rope_neox_p(float* x, const std::uint32_t* params, std::uint32_t head_dim,
                                            std::uint32_t rotary_dim, std::uint32_t slot, std::uint32_t scale,
                                            float theta) {
  RopeNeox(x, head_dim, rotary_dim, params[slot] * scale, theta);
}

// P4: x holds pooled blocks b0, b0 + 1, ...; block b at position b * ratio.
// grid (nblocks)
extern "C" __global__ void halo_rope_neox_blocks(float* x, std::uint32_t head_dim, std::uint32_t rotary_dim,
                                                 std::uint32_t b0, std::uint32_t ratio, float theta) {
  RopeNeox(x, head_dim, rotary_dim, (b0 + blockIdx.x) * ratio, theta);
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

// halo_attention_decode for a chunk of T queries in one launch: query t (at
// position pos0 + t) attends to positions [0, pos0 + t] of the cache, which
// already holds the whole chunk. Same arithmetic per query as the decode
// kernel. q, gate, out [T][heads][hd]; scores [T][heads][max_n] scratch.
//   grid (heads, T), block 256
extern "C" __global__ void halo_attention_prefill(const float* q, const float* kc, const float* vc, const float* gate,
                                                  float* out, float* scores, std::uint32_t num_heads,
                                                  std::uint32_t num_kv_heads, std::uint32_t hd, std::uint32_t pos0,
                                                  std::uint32_t max_n) {
  __shared__ float scratch[32];
  const std::uint32_t h = blockIdx.x, t = blockIdx.y, kvh = h / (num_heads / num_kv_heads), n = pos0 + t + 1;
  const std::size_t row = (static_cast<std::size_t>(t) * num_heads + h) * hd;
  const float* qh = q + row;
  float* sc = scores + (static_cast<std::size_t>(t) * num_heads + h) * max_n;
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
    out[row + i] = acc / den * Sigmoid(gate[row + i]);
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

// P4: halo_ple_gate over T tokens: vector t * hc + s of key / gated, row t
// of value. grid (T * hc)
extern "C" __global__ void halo_ple_gate_rows(const float* key, const float* query, const float* value, float* gated,
                                              std::uint32_t h, std::uint32_t hc) {
  __shared__ float scratch[32];
  const std::size_t v = blockIdx.x, t = v / hc;
  float dot = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < h; i += blockDim.x) dot += key[v * h + i] * query[v * h + i];
  dot = BlockSum(dot, scratch);
  const float sc = dot / sqrtf(static_cast<float>(h));
  const float mag = sqrtf(fmaxf(fabsf(sc), 1e-6f));
  const float gate = Sigmoid((sc < 0.0f ? -1.0f : (sc > 0.0f ? 1.0f : 0.0f)) * mag);
  for (std::uint32_t i = threadIdx.x; i < h; i += blockDim.x) gated[v * h + i] = value[t * h + i] * gate;
}

// P4: halo_ple_conv for T tokens at once (same sums, same order): token t's
// tap `back` tokens back reads the chunk or, before it, the history; the
// history is not touched (halo_ple_history_advance moves it afterwards).
//   rows [T][dim]; grid (ceil(dim / 256), T)
extern "C" __global__ void halo_ple_conv_prefill(const float* normalized, const float* gated, const float* history,
                                                 const float* w, float* res, std::uint32_t dim, std::uint32_t kern,
                                                 std::uint32_t dil) {
  const std::uint32_t hist = (kern - 1) * dil, t = blockIdx.y;
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= dim) return;
  const std::size_t row = static_cast<std::size_t>(t) * dim;
  float acc = 0.0f;
  for (std::uint32_t k = 0; k < kern; ++k) {
    const std::uint32_t back = (kern - 1 - k) * dil;
    const float src = back == 0 ? normalized[row + ch]
                      : back <= t ? normalized[static_cast<std::size_t>(t - back) * dim + ch]
                                  : history[static_cast<std::size_t>(hist + t - back) * dim + ch];
    acc += w[ch * kern + k] * src;
  }
  res[row + ch] += gated[row + ch] + Silu(acc);
}

// The PLE conv history after T tokens: the last `hist` of (history, rows).
//   grid (ceil(dim / 256))
extern "C" __global__ void halo_ple_history_advance(const float* normalized, float* history, std::uint32_t dim,
                                                    std::uint32_t hist, std::uint32_t tokens) {
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= dim) return;
  for (std::uint32_t i = 0; i < hist; ++i) {  // reads index tokens + i > i: not yet overwritten
    const std::uint32_t j = tokens + i;
    history[static_cast<std::size_t>(i) * dim + ch] =
        j < hist ? history[static_cast<std::size_t>(j) * dim + ch] : normalized[static_cast<std::size_t>(j - hist) * dim + ch];
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

// P4: pools blocks b0 .. b0 + gridDim.x - 1 straight into their cache rows
// (raw rows past the chunk's last position are whatever the cache holds,
// as when stepping pools an incomplete block). grid (nblocks)
extern "C" __global__ void halo_pool_blocks(const float* raw, float* block_k, std::uint32_t b0, std::uint32_t ratio,
                                            std::uint32_t dim) {
  const std::uint32_t block = b0 + blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float acc = 0.0f;
    for (std::uint32_t t = 0; t < ratio; ++t) acc += raw[(block * ratio + t) * dim + i] / static_cast<float>(ratio);
    block_k[static_cast<std::size_t>(block) * dim + i] = acc;
  }
}

// QSA block scores: score[b] = sum_heads relu(<q_h, k_b>). One thread per block.
__device__ void QsaScores(const float* q, const float* blocks, float* score,
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
extern "C" __global__ void halo_qsa_scores(const float* q, const float* blocks, float* score,
                                           const std::uint32_t* params, std::uint32_t heads, std::uint32_t dim) {
  QsaScores(q, blocks, score, params, heads, dim);
}
// P4: token t = blockIdx.y of a chunk, its step parameters 16 words apart.
extern "C" __global__ void halo_qsa_scores_rows(const float* q, const float* blocks, float* score,
                                                const std::uint32_t* params, std::uint32_t heads, std::uint32_t dim,
                                                std::uint32_t score_stride) {
  const std::size_t t = blockIdx.y;
  QsaScores(q + t * heads * dim, blocks, score + t * score_stride, params + t * 16, heads, dim);
}

// Attention over a list of visible positions (QSA selection), GQA, gated.
__device__ void AttentionSelect(const float* q, const float* kc, const float* vc, const float* gate,
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
extern "C" __global__ void halo_attention_select(const float* q, const float* kc, const float* vc, const float* gate,
                                                 const std::uint32_t* visible, float* out, float* scores,
                                                 const std::uint32_t* params, std::uint32_t num_heads,
                                                 std::uint32_t num_kv_heads, std::uint32_t hd, std::uint32_t max_n) {
  AttentionSelect(q, kc, vc, gate, visible, out, scores, params, num_heads, num_kv_heads, hd, max_n);
}
// P4: token t = blockIdx.y; visible lists `max_n` apart, scores
// num_heads * max_n apart. grid (num_heads, T)
extern "C" __global__ void halo_attention_select_rows(const float* q, const float* kc, const float* vc,
                                                      const float* gate, const std::uint32_t* visible, float* out,
                                                      float* scores, const std::uint32_t* params,
                                                      std::uint32_t num_heads, std::uint32_t num_kv_heads,
                                                      std::uint32_t hd, std::uint32_t max_n) {
  const std::size_t t = blockIdx.y, row = static_cast<std::size_t>(num_heads) * hd;
  AttentionSelect(q + t * row, kc, vc, gate + t * row, visible + t * max_n, out + t * row,
                  scores + t * num_heads * max_n, params + t * 16, num_heads, num_kv_heads, hd, max_n);
}

// ---- Device-side routing and selection (no host round trip) ----------------

// MoE router: softmax over all experts, top-k of the probabilities (ties
// keep the lower expert index), renormalized with the reference's floor.
// One block; ids/weights receive k entries in descending probability.
__device__ void RouterTopk(const float* logits, std::uint32_t* ids, float* weights, std::uint32_t n, std::uint32_t k) {
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

extern "C" __global__ void halo_router_topk(const float* logits, std::uint32_t* ids, float* weights, std::uint32_t n,
                                            std::uint32_t k) {
  RouterTopk(logits, ids, weights, n, k);
}

// P4: one block per token row: logits [T][n] -> ids, weights [T][k].
extern "C" __global__ void halo_router_topk_rows(const float* logits, std::uint32_t* ids, float* weights,
                                                 std::uint32_t n, std::uint32_t k) {
  const std::size_t t = blockIdx.x;
  RouterTopk(logits + t * n, ids + t * k, weights + t * k, n, k);
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
__device__ void QsaSelect(const float* score, std::uint32_t* visible, const std::uint32_t* params,
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
extern "C" __global__ void halo_qsa_select(const float* score, std::uint32_t* visible, const std::uint32_t* params,
                                           std::uint32_t k, std::uint32_t ratio) {
  QsaSelect(score, visible, params, k, ratio);
}
// P4: one block per token t = blockIdx.x of a chunk. grid (T)
extern "C" __global__ void halo_qsa_select_rows(const float* score, std::uint32_t* visible, const std::uint32_t* params,
                                                std::uint32_t k, std::uint32_t ratio, std::uint32_t score_stride,
                                                std::uint32_t visible_stride) {
  const std::size_t t = blockIdx.x;
  QsaSelect(score + t * score_stride, visible + t * visible_stride, params + t * 16, k, ratio);
}

// P4: the union of 4 consecutive tokens' QSA selections, for attention that
// shares K/V reads between them (halo_attention_wmma_qsel). Token q selects
// its k chosen blocks (all complete blocks when under budget) and its
// incomplete tail block; the union, in ascending block order, goes to
// ublk[group][*] with umask's bit q set where token q selected the block.
// Token q sees position p exactly when bit q is set for p / ratio and
// p < n_kv(q): the same set halo_qsa_select_rows lists.
//   grid (ceil(T / 4)), block 1024; max_blocks >= every n_kv / ratio + 1
//   (4 bitmaps of max_blocks bits in LDS); max_u >= 4 k + 4.
extern "C" __global__ void __launch_bounds__(1024)
    halo_qsa_union4(const std::uint32_t* visible, const std::uint32_t* params, std::uint32_t k, std::uint32_t ratio,
                    std::uint32_t visible_stride, std::uint32_t max_blocks, std::uint32_t tokens, std::uint32_t* ublk,
                    std::uint8_t* umask, std::uint32_t* ucount, std::uint32_t max_u) {
  constexpr std::uint32_t QT = 4, kMaxWords = 65537 / 32 + 1;  // 262144-token context
  __shared__ std::uint32_t bits[QT][kMaxWords];
  __shared__ std::uint32_t buf[1024];
  const std::uint32_t grp = blockIdx.x, t0 = grp * QT, words = (max_blocks + 31) / 32;
  for (std::uint32_t i = threadIdx.x; i < QT * words; i += blockDim.x) bits[i / words][i % words] = 0;
  __syncthreads();
  std::uint32_t last = 0;  // highest block any token sees
  for (std::uint32_t q = 0; q < QT && t0 + q < tokens; ++q) {
    const std::uint32_t* pp = params + static_cast<std::size_t>(t0 + q) * 16;
    const std::uint32_t n = pp[kHaloComplete], n_kv = pp[kHaloNkv];
    const std::uint32_t* vis = visible + static_cast<std::size_t>(t0 + q) * visible_stride;
    if (k >= n) {
      for (std::uint32_t b = threadIdx.x; b < n; b += blockDim.x) atomicOr(&bits[q][b / 32], 1u << (b % 32));
    } else {
      for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
        const std::uint32_t b = vis[i * ratio] / ratio;
        atomicOr(&bits[q][b / 32], 1u << (b % 32));
      }
    }
    if (threadIdx.x == 0 && n_kv > n * ratio) atomicOr(&bits[q][n / 32], 1u << (n % 32));
    last = max(last, (n_kv + ratio - 1) / ratio);
  }
  __syncthreads();
  std::uint32_t written = 0;
  for (std::uint32_t base = 0; base < last; base += blockDim.x) {
    const std::uint32_t b = base + threadIdx.x;
    std::uint32_t m = 0;
    if (b < last)
      for (std::uint32_t q = 0; q < QT; ++q) m |= ((bits[q][b / 32] >> (b % 32)) & 1u) << q;
    std::uint32_t kept;
    const std::uint32_t at = written + BlockScan(m != 0, buf, &kept);
    if (m != 0 && at < max_u) {
      ublk[static_cast<std::size_t>(grp) * max_u + at] = b;
      umask[static_cast<std::size_t>(grp) * max_u + at] = static_cast<std::uint8_t>(m);
    }
    written += kept;
  }
  if (threadIdx.x == 0) ucount[grp] = min(written, max_u);
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

// ---- P4: prefill GEMMs for BF16 / F32 weights (Qwen3.5 torso) ----------------
//
// y[t][r] = sum_c W[r][c] x[t][c], W BF16 [rows][cols], x fp32 [T][cols], on
// FP16 WMMA: weights are converted to fp16 (exact unless below fp16's range)
// and activations rounded to fp16 while staging into LDS; products
// accumulate in fp32. Block tile 128 rows x 64 tokens, 32 x 32 per wave, K
// staged 32 at a time and double-buffered.
//   grid (ceil(T / 64), ceil(rows / 128)), block 256; cols % 32 == 0.
extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_bf16_f16(const std::uint16_t* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols,
                       std::uint32_t tokens) {
  constexpr std::uint32_t BR = 128, BT = 64, BK = 32, RS = BK + 8;  // rows padded against bank conflicts
  __shared__ __attribute__((aligned(16))) _Float16 As[2][BR * RS];
  __shared__ __attribute__((aligned(16))) _Float16 Bs[2][BT * RS];
  const std::uint32_t tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t r0 = blockIdx.y * BR, t0 = blockIdx.x * BT;
  const std::uint32_t nblk = cols / BK;
  const std::uint32_t wr = wave % 4, wt = wave / 4;
  // Staging: A 128 rows x 32 bf16 = 256 threads x 16 values; B 64 tokens x 32
  // floats = 256 threads x 8 values.
  const std::uint32_t a_row = tid / 2, a_half = tid % 2;
  const std::uint16_t* a_src = w + static_cast<std::size_t>(min(r0 + a_row, rows - 1)) * cols + a_half * 16;
  const std::uint32_t b_tok = tid / 4, b_q = tid % 4;
  const float* b_src = x + static_cast<std::size_t>(min(t0 + b_tok, tokens - 1)) * cols + b_q * 8;
  typedef _Float16 v8h __attribute__((ext_vector_type(8)));
  struct Regs {
    halo_v4i a0, a1;  // 16 bf16
    float4 b0, b1;
  };
  const auto load = [&](std::uint32_t b, Regs& r) {
    r.a0 = *reinterpret_cast<const halo_v4i*>(a_src + b * BK);
    r.a1 = *reinterpret_cast<const halo_v4i*>(a_src + b * BK + 8);
    r.b0 = *reinterpret_cast<const float4*>(b_src + b * BK);
    r.b1 = *reinterpret_cast<const float4*>(b_src + b * BK + 4);
  };
  const auto bf = [](std::uint32_t bits16) { return __uint_as_float(bits16 << 16); };
  const auto store = [&](const Regs& r, std::uint32_t p) {
    const std::uint32_t* aw = reinterpret_cast<const std::uint32_t*>(&r.a0);
    const std::uint32_t* aw1 = reinterpret_cast<const std::uint32_t*>(&r.a1);
    v8h h0, h1;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h0[2 * i] = static_cast<_Float16>(bf(aw[i] & 0xFFFFu));
      h0[2 * i + 1] = static_cast<_Float16>(bf(aw[i] >> 16));
      h1[2 * i] = static_cast<_Float16>(bf(aw1[i] & 0xFFFFu));
      h1[2 * i + 1] = static_cast<_Float16>(bf(aw1[i] >> 16));
    }
    *reinterpret_cast<v8h*>(&As[p][a_row * RS + a_half * 16]) = h0;
    *reinterpret_cast<v8h*>(&As[p][a_row * RS + a_half * 16 + 8]) = h1;
    const v8h bh = {static_cast<_Float16>(r.b0.x), static_cast<_Float16>(r.b0.y), static_cast<_Float16>(r.b0.z),
                    static_cast<_Float16>(r.b0.w), static_cast<_Float16>(r.b1.x), static_cast<_Float16>(r.b1.y),
                    static_cast<_Float16>(r.b1.z), static_cast<_Float16>(r.b1.w)};
    *reinterpret_cast<v8h*>(&Bs[p][b_tok * RS + b_q * 8]) = bh;
  };
  halo_v8f acc[2][2] = {};
  {
    Regs r;
    load(0, r);
    store(r, 0);
  }
  __syncthreads();
  for (std::uint32_t b = 0, p = 0; b < nblk; ++b, p ^= 1) {
    Regs next;
    const bool more = b + 1 < nblk;
    if (more) load(b + 1, next);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      halo_v16h a[2], bv[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const _Float16* ap = &As[p][(wr * 32 + i * 16 + lane % 16) * RS + h * 16];
        const _Float16* bp = &Bs[p][(wt * 32 + i * 16 + lane % 16) * RS + h * 16];
        const v8h a_lo = *reinterpret_cast<const v8h*>(ap), a_hi = *reinterpret_cast<const v8h*>(ap + 8);
        const v8h b_lo = *reinterpret_cast<const v8h*>(bp), b_hi = *reinterpret_cast<const v8h*>(bp + 8);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          a[i][e] = a_lo[e];
          a[i][e + 8] = a_hi[e];
          bv[i][e] = b_lo[e];
          bv[i][e + 8] = b_hi[e];
        }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int k = 0; k < 2; ++k) acc[i][k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], bv[k], acc[i][k]);
    }
    if (more) store(next, p ^ 1);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const std::uint32_t t = t0 + wt * 32 + k * 16 + lane % 16;
      if (t >= tokens) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t r = r0 + wr * 32 + i * 16 + 2 * j + lane / 16;
        if (r < rows) y[static_cast<std::size_t>(t) * rows + r] = acc[i][k][j];
      }
    }
}

// y[t][r] = sum_c W[r][c] x[t][c] for F32 W with few columns (LoRA B
// factors): one thread per output, x row in LDS.
//   grid (ceil(rows / 256), T), block 256; cols <= 8192
extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_f32_narrow(const float* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols) {
  __shared__ float xs[8192];
  const std::uint32_t t = blockIdx.y;
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[static_cast<std::size_t>(t) * cols + i];
  __syncthreads();
  const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* wr = w + static_cast<std::size_t>(r) * cols;
  float acc = 0.0f;
  for (std::uint32_t c = 0; c < cols; ++c) acc += wr[c] * xs[c];
  y[static_cast<std::size_t>(t) * rows + r] = acc;
}

// The same for wider W (LoRA A factors, small projections): one warp per
// output, lanes across columns, W and x both read coalesced (x stays in
// cache across the block's rows).
//   grid (ceil(rows / 8), T), block 256
extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_f32_rows(const float* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t lane = threadIdx.x % kWarp, r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
  if (r >= rows) return;
  const float* wr = w + static_cast<std::size_t>(r) * cols;
  const float* xr = x + static_cast<std::size_t>(t) * cols;
  float acc = 0.0f;
  for (std::uint32_t c = lane; c < cols; c += kWarp) acc += wr[c] * xr[c];
  acc = WarpSum(acc);
  if (lane == 0) y[static_cast<std::size_t>(t) * rows + r] = acc;
}

// The BF16 GEMM shaped for weights streamed from DRAM: the tile's rows are
// read 128 values (256 bytes) at a time instead of 32, and the next stage's
// operands (12 16-byte loads per thread) are in flight while the current one
// is multiplied out of LDS. Block tile 64 rows x 64 tokens, 4 waves of
// 32 x 32 (2 x 2 WMMA tiles), K 128 per stage.
//   grid (ceil(T / 64), ceil(rows / 64)), block 128; cols % 128 == 0.
extern "C" __global__ void __launch_bounds__(128)
    halo_gemm_bf16_f16_k128(const std::uint16_t* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols,
                            std::uint32_t tokens) {
  constexpr std::uint32_t BR = 64, BT = 64, BK = 128, RS = BK + 8;
  __shared__ __attribute__((aligned(16))) _Float16 As[BR * RS];
  __shared__ __attribute__((aligned(16))) _Float16 Bs[BT * RS];
  typedef _Float16 v8h __attribute__((ext_vector_type(8)));
  const std::uint32_t tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t r0 = blockIdx.y * BR, t0 = blockIdx.x * BT;
  const std::uint32_t wr = wave % 2, wt = wave / 2;
  // A: 64 rows x 128 bf16 = 1024 16-byte pieces, 8 per thread: thread tid takes
  // row (tid / 2) and pieces (tid % 2) * 8 .. + 8 (contiguous 128 bytes).
  const std::uint32_t a_row = tid / 2, a_part = (tid % 2) * 8;
  const std::uint16_t* a_src = w + static_cast<std::size_t>(min(r0 + a_row, rows - 1)) * cols + a_part * 8;
  // B: 64 tokens x 128 floats = 2048 16-byte pieces, 16 per thread: token
  // tid / 2, floats (tid % 2) * 64 .. + 64.
  const std::uint32_t b_tok = tid / 2, b_part = (tid % 2) * 64;
  const float* b_src = x + static_cast<std::size_t>(min(t0 + b_tok, tokens - 1)) * cols + b_part;
  halo_v4i ra[8];
  float4 rb[16];
  const auto load = [&](std::uint32_t k0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) ra[i] = *reinterpret_cast<const halo_v4i*>(a_src + k0 + i * 8);
#pragma unroll
    for (int i = 0; i < 16; ++i) rb[i] = *reinterpret_cast<const float4*>(b_src + k0 + i * 4);
  };
  const auto bf = [](std::uint32_t bits16) { return __uint_as_float(bits16 << 16); };
  const auto store = [&]() {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const std::uint32_t* aw = reinterpret_cast<const std::uint32_t*>(&ra[i]);
      v8h h;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        h[2 * j] = static_cast<_Float16>(bf(aw[j] & 0xFFFFu));
        h[2 * j + 1] = static_cast<_Float16>(bf(aw[j] >> 16));
      }
      *reinterpret_cast<v8h*>(&As[a_row * RS + a_part * 8 + i * 8]) = h;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const v8h h = {static_cast<_Float16>(rb[2 * i].x), static_cast<_Float16>(rb[2 * i].y),
                     static_cast<_Float16>(rb[2 * i].z), static_cast<_Float16>(rb[2 * i].w),
                     static_cast<_Float16>(rb[2 * i + 1].x), static_cast<_Float16>(rb[2 * i + 1].y),
                     static_cast<_Float16>(rb[2 * i + 1].z), static_cast<_Float16>(rb[2 * i + 1].w)};
      *reinterpret_cast<v8h*>(&Bs[b_tok * RS + b_part + i * 8]) = h;
    }
  };
  halo_v8f acc[2][2] = {};
  load(0);
  for (std::uint32_t k0 = 0; k0 < cols; k0 += BK) {
    __syncthreads();  // the previous stage's LDS reads are done
    store();
    __syncthreads();
    if (k0 + BK < cols) load(k0 + BK);  // in flight during the math below
#pragma unroll
    for (std::uint32_t h = 0; h < BK / 16; ++h) {
      halo_v16h a[2], bv[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const _Float16* ap = &As[(wr * 32 + i * 16 + lane % 16) * RS + h * 16];
        const _Float16* bp = &Bs[(wt * 32 + i * 16 + lane % 16) * RS + h * 16];
        const v8h a_lo = *reinterpret_cast<const v8h*>(ap), a_hi = *reinterpret_cast<const v8h*>(ap + 8);
        const v8h b_lo = *reinterpret_cast<const v8h*>(bp), b_hi = *reinterpret_cast<const v8h*>(bp + 8);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          a[i][e] = a_lo[e];
          a[i][e + 8] = a_hi[e];
          bv[i][e] = b_lo[e];
          bv[i][e + 8] = b_hi[e];
        }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int k = 0; k < 2; ++k) acc[i][k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], bv[k], acc[i][k]);
    }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const std::uint32_t t = t0 + wt * 32 + k * 16 + lane % 16;
      if (t >= tokens) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t r = r0 + wr * 32 + i * 16 + 2 * j + lane / 16;
        if (r < rows) y[static_cast<std::size_t>(t) * rows + r] = acc[i][k][j];
      }
    }
}

// ---- P4: Flash-Next chunk prefill, row-batched elementwise kernels -------------
// Each is the decode kernel applied to T token rows at once (same arithmetic).

// halo_rmsnorm over T * hc vectors whose gammas repeat every w_period
// vectors (grouped hyper-connection norms: vector t*hc + s uses gamma s).
extern "C" __global__ void halo_rmsnorm_rows(const float* x, const float* w_all, float* y, std::uint32_t d, float eps,
                                             std::uint32_t w_stride, std::uint32_t w_period) {
  __shared__ float scratch[32];
  const float* xv = x + static_cast<std::size_t>(blockIdx.x) * d;
  const float* w = w_all + (blockIdx.x % w_period) * w_stride;
  float* yv = y + static_cast<std::size_t>(blockIdx.x) * d;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += xv[i] * xv[i];
  ss = BlockSum(ss, scratch);
  const float scale = 1.0f / sqrtf(ss / static_cast<float>(d) + eps);
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) yv[i] = xv[i] * scale * w[i];
}

// halo_hc_combine_rows fused with the next mixer's grouped RMSNorm (and,
// with y16, its fp16 copy): one pass over the residual instead of a write
// and a re-read. Same arithmetic as the two kernels: each thread combines
// and squares the values at i = threadIdx.x + k * blockDim.x, as the norm
// kernel reads them. grid (T * hc), block 256; h <= 16 * 256.
// With iw (the next mixer's F32 inject weights, irows <= 8 rows of hc * h),
// each block also writes this stream's partial dot products of its normalized
// row with them, ipart[(t * hc + s) * irows + r]; halo_inject_reduce sums the
// streams (replacing that projection's GEMM, which re-read every row).
// y may be null when its only reader rebuilds it (the mixer's up/mix
// epilogue, from res, rscale and the gammas: the same (x * scale) * w); then
// rscale[v] receives the row's scale instead of h floats of y.
extern "C" __global__ void __launch_bounds__(256)
    halo_hc_combine_norm_rows(float* res, const float* out, const float* inject, const float* w_all, float* y,
                              _Float16* y16, std::uint32_t hc, std::uint32_t h, float eps, const float* iw,
                              float* ipart, std::uint32_t irows, float* rscale) {
  __shared__ float scratch[32];
  const std::size_t v = blockIdx.x, t = v / hc, row = v * h;
  const std::uint32_t st = static_cast<std::uint32_t>(v % hc);
  const float a = 2.0f * Sigmoid(inject[t * hc + st] / static_cast<float>(hc));
  const float* w = w_all + static_cast<std::size_t>(st) * h;
  // All of a thread's loads are issued before any is used. Combined with the
  // arithmetic in one loop, each k waited for its own loads (a thread trace
  // showed every load pair followed by vmcnt(0)): no memory parallelism.
  float vals[16], outs[16];
#pragma unroll
  for (std::uint32_t k = 0; k < 16; ++k) {
    const std::uint32_t i = threadIdx.x + k * blockDim.x;
    vals[k] = i < h ? res[row + i] : 0.0f;
    outs[k] = i < h ? out[t * h + i] : 0.0f;
  }
  float ss = 0.0f;
#pragma unroll
  for (std::uint32_t k = 0; k < 16; ++k) {
    const std::uint32_t i = threadIdx.x + k * blockDim.x;
    if (i < h) {
      const float x = vals[k] + outs[k] * a;
      res[row + i] = x;
      vals[k] = x;
      ss += x * x;
    }
  }
  ss = BlockSum(ss, scratch);
  const float scale = 1.0f / sqrtf(ss / static_cast<float>(h) + eps);
  if (rscale && threadIdx.x == 0) rscale[v] = scale;
  float ip[8] = {};
#pragma unroll
  for (std::uint32_t k = 0; k < 16; ++k) {
    const std::uint32_t i = threadIdx.x + k * blockDim.x;
    if (i < h) {
      const float yv = vals[k] * scale * w[i];
      if (y) y[row + i] = yv;
      if (y16) y16[row + i] = static_cast<_Float16>(yv);
      if (iw) {
#pragma unroll
        for (std::uint32_t r = 0; r < 8; ++r)
          if (r < irows) ip[r] += yv * iw[static_cast<std::size_t>(r) * hc * h + static_cast<std::size_t>(st) * h + i];
      }
    }
  }
  if (iw) {
#pragma unroll
    for (std::uint32_t r = 0; r < 8; ++r) {
      if (r >= irows) break;
      const float sum = BlockSum(ip[r], scratch);
      if (threadIdx.x == 0) ipart[v * irows + r] = sum;
    }
  }
}

// inject[t][r] = sum over streams s (in order) of ipart[(t * hc + s) * irows + r].
//   grid (ceil(T * irows / 256))
extern "C" __global__ void halo_inject_reduce(const float* ipart, float* inject, std::uint32_t hc, std::uint32_t irows,
                                              std::uint32_t tokens) {
  const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tokens * irows) return;
  const std::uint32_t t = idx / irows, r = idx % irows;
  float acc = 0.0f;
  for (std::uint32_t st = 0; st < hc; ++st) acc += ipart[(static_cast<std::size_t>(t) * hc + st) * irows + r];
  inject[idx] = acc;
}

// halo_rmsnorm_rows that also writes an fp16 copy (the next GEMM's input on
// the FP16 route), sparing a conversion pass.
// iw / ipart / irows: the next projection's partial products, as in
// halo_hc_combine_norm_rows (w_period = hc streams; iw null to skip).
extern "C" __global__ void halo_rmsnorm_rows_h(const float* x, const float* w_all, float* y, _Float16* y16,
                                               std::uint32_t d, float eps, std::uint32_t w_stride,
                                               std::uint32_t w_period, const float* iw, float* ipart,
                                               std::uint32_t irows) {
  __shared__ float scratch[32];
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * d;
  const std::uint32_t st = blockIdx.x % w_period;
  const float* w = w_all + st * w_stride;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += x[row + i] * x[row + i];
  ss = BlockSum(ss, scratch);
  const float scale = 1.0f / sqrtf(ss / static_cast<float>(d) + eps);
  float ip[8] = {};
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
    const float v = x[row + i] * scale * w[i];
    y[row + i] = v;
    y16[row + i] = static_cast<_Float16>(v);
    if (iw) {
#pragma unroll
      for (std::uint32_t r = 0; r < 8; ++r)
        if (r < irows) ip[r] += v * iw[static_cast<std::size_t>(r) * w_period * d + static_cast<std::size_t>(st) * d + i];
    }
  }
  if (iw) {
#pragma unroll
    for (std::uint32_t r = 0; r < 8; ++r) {
      if (r >= irows) break;
      const float sum = BlockSum(ip[r], scratch);
      if (threadIdx.x == 0) ipart[static_cast<std::size_t>(blockIdx.x) * irows + r] = sum;
    }
  }
}

// halo_gated_norm writing its result in fp16 only (the out projection's
// input on the FP16 route). grid (rows), rows of d.
extern "C" __global__ void halo_gated_norm_h(const float* o, const float* w, const float* z, _Float16* y16,
                                             std::uint32_t d, float eps, std::uint32_t mode) {
  __shared__ float scratch[32];
  const float* ov = o + static_cast<std::size_t>(blockIdx.x) * d;
  const float* zv = z + static_cast<std::size_t>(blockIdx.x) * d;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += ov[i] * ov[i];
  ss = BlockSum(ss, scratch);
  const float scale = 1.0f / sqrtf(ss / static_cast<float>(d) + eps);
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
    const float g = mode == 0 ? Silu(zv[i]) : Sigmoid(zv[i]);
    y16[static_cast<std::size_t>(blockIdx.x) * d + i] = static_cast<_Float16>(ov[i] * scale * w[i] * g);
  }
}

// halo_hc_mix for T rows: xn, gate [T][hc][h] -> mixed [T][h].
extern "C" __global__ void halo_hc_mix_rows(const float* xn, const float* gate, float* mixed, std::uint32_t hc,
                                            std::uint32_t h, std::uint32_t tokens) {
  const std::size_t n = static_cast<std::size_t>(tokens) * h;
  for (std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t t = idx / h, i = idx % h;
    const float* xr = xn + t * hc * h;
    const float* gr = gate + t * hc * h;
    float acc = 0.0f;
    for (std::uint32_t s = 0; s < hc; ++s) acc += xr[s * h + i] * Sigmoid(gr[s * h + i]);
    mixed[idx] = acc / static_cast<float>(hc);
  }
}

// halo_hc_combine for T rows: res [T][hc][h] += out [T][h] * 2 sigmoid(inject [T][hc] / hc).
extern "C" __global__ void halo_hc_combine_rows(float* res, const float* out, const float* inject, std::uint32_t hc,
                                                std::uint32_t h, std::uint32_t tokens) {
  const std::size_t n = static_cast<std::size_t>(tokens) * hc * h;
  for (std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t t = idx / (static_cast<std::size_t>(hc) * h), r = idx % (static_cast<std::size_t>(hc) * h);
    const std::uint32_t s = static_cast<std::uint32_t>(r / h);
    res[idx] += out[t * h + r % h] * 2.0f * Sigmoid(inject[t * hc + s] / static_cast<float>(hc));
  }
}

// halo_broadcast for T rows: res [T][hc][h] = x [T][h].
extern "C" __global__ void halo_broadcast_rows(float* res, const float* x, std::uint32_t hc, std::uint32_t h,
                                               std::uint32_t tokens) {
  const std::size_t n = static_cast<std::size_t>(tokens) * hc * h;
  for (std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t t = idx / (static_cast<std::size_t>(hc) * h);
    res[idx] = x[t * h + idx % h];
  }
}

// halo_axpy_sigmoid for T rows: y [T][n] += x [T][n] * sigmoid(g [T]).
extern "C" __global__ void halo_axpy_sigmoid_rows(float* y, const float* x, const float* g, std::uint32_t n,
                                                  std::uint32_t tokens) {
  const std::size_t total = static_cast<std::size_t>(tokens) * n;
  for (std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    y[idx] += Sigmoid(g[idx / n]) * x[idx];
}

// ---- P4 stage B: routed experts over a prompt chunk, grouped by expert -------
//
// The chunk's T * k (token, slot) pairs, pair p = t * k + slot, are bucketed
// by expert in pair order (deterministic), then each expert's tokens run as
// one GEMM against its weights: gate and up with SwiGLU fused into the
// epilogue, then down, each into a per-pair row; the combine sums a token's
// k rows in slot order with its routing weights. Weights are dequantized
// into fp16 WMMA fragments as they are read (each weight row belongs to one
// wave, so they bypass LDS); activations are staged in LDS as fp16.

// Bucketing in one block: per-expert counts and offsets, pairs sorted by
// expert, each pair's sorted position, and work tiles of up to kMoeTile
// pairs of one expert: tiles[3 i .. 3 i + 2] = (expert, first, count).
constexpr std::uint32_t kMoeTile = 64;
extern "C" __global__ void __launch_bounds__(1024)
    halo_moe_bucket(const std::uint32_t* ids, std::uint32_t pairs, std::uint32_t experts, std::uint32_t* sorted,
                    std::uint32_t* pos, std::uint32_t* tiles, std::uint32_t* ntiles) {
  __shared__ std::uint32_t cnt[1024], poff[1024], toff[1024];
  __shared__ std::uint32_t chunk[4096];
  const std::uint32_t tid = threadIdx.x;
  for (std::uint32_t e = tid; e < experts; e += blockDim.x) cnt[e] = 0;
  // Pass 1: counts. Each thread owns experts tid, tid + blockDim, ...
  for (std::uint32_t base = 0; base < pairs; base += 4096) {
    const std::uint32_t n = min(4096u, pairs - base);
    __syncthreads();
    for (std::uint32_t i = tid; i < n; i += blockDim.x) chunk[i] = ids[base + i];
    __syncthreads();
    for (std::uint32_t e = tid; e < experts; e += blockDim.x) {
      std::uint32_t c = 0;
      for (std::uint32_t i = 0; i < n; ++i) c += chunk[i] == e;
      cnt[e] += c;
    }
  }
  __syncthreads();
  if (tid == 0) {
    std::uint32_t p = 0, t = 0;
    for (std::uint32_t e = 0; e < experts; ++e) {
      poff[e] = p;
      toff[e] = t;
      p += cnt[e];
      t += (cnt[e] + kMoeTile - 1) / kMoeTile;
    }
    *ntiles = t;
  }
  __syncthreads();
  for (std::uint32_t e = tid; e < experts; e += blockDim.x)
    for (std::uint32_t i = 0; i * kMoeTile < cnt[e]; ++i) {
      tiles[3 * (toff[e] + i)] = e;
      tiles[3 * (toff[e] + i) + 1] = poff[e] + i * kMoeTile;
      tiles[3 * (toff[e] + i) + 2] = min(kMoeTile, cnt[e] - i * kMoeTile);
    }
  // Pass 2: scatter in pair order.
  for (std::uint32_t e = tid; e < experts; e += blockDim.x) cnt[e] = poff[e];
  for (std::uint32_t base = 0; base < pairs; base += 4096) {
    const std::uint32_t n = min(4096u, pairs - base);
    __syncthreads();
    for (std::uint32_t i = tid; i < n; i += blockDim.x) chunk[i] = ids[base + i];
    __syncthreads();
    for (std::uint32_t e = tid; e < experts; e += blockDim.x) {
      std::uint32_t c = cnt[e];
      for (std::uint32_t i = 0; i < n; ++i)
        if (chunk[i] == e) {
          sorted[c] = base + i;
          pos[base + i] = c;
          ++c;
        }
      cnt[e] = c;
    }
  }
}

// The same bucketing in three parallel passes, for large chunks; output is
// identical (pairs in pair order within an expert, same tiles).
//   1. halo_moe_bucket_count: per chunk of 1024 pairs, per-expert counts
//      [chunks][experts]. grid (chunks), block 1024
//   2. halo_moe_bucket_scan: per expert, chunk offsets (exclusive scan over
//      chunks) and expert offsets, tiles, ntiles. grid (1), block 1024
//   3. halo_moe_bucket_scatter: each pair's rank among earlier pairs of its
//      expert in its chunk (an LDS scan) gives its sorted position.
//      grid (chunks), block 1024
constexpr std::uint32_t kBucketChunk = 1024;
extern "C" __global__ void __launch_bounds__(1024)
    halo_moe_bucket_count(const std::uint32_t* ids, std::uint32_t pairs, std::uint32_t experts, std::uint32_t* counts) {
  __shared__ std::uint32_t c[1024];
  const std::uint32_t chunk = blockIdx.x, tid = threadIdx.x, p = chunk * kBucketChunk + tid;
  for (std::uint32_t e = tid; e < experts; e += blockDim.x) c[e] = 0;
  __syncthreads();
  if (p < pairs) atomicAdd(&c[ids[p]], 1u);
  __syncthreads();
  for (std::uint32_t e = tid; e < experts; e += blockDim.x) counts[static_cast<std::size_t>(chunk) * experts + e] = c[e];
}
extern "C" __global__ void __launch_bounds__(1024)
    halo_moe_bucket_scan(std::uint32_t* counts, std::uint32_t chunks, std::uint32_t experts, std::uint32_t* tiles,
                         std::uint32_t* ntiles, std::uint32_t* base) {
  __shared__ std::uint32_t total[1024], poff[1024], toff[1024];
  const std::uint32_t tid = threadIdx.x;
  for (std::uint32_t e = tid; e < experts; e += blockDim.x) {
    std::uint32_t run = 0;
    for (std::uint32_t ch = 0; ch < chunks; ++ch) {  // counts become exclusive chunk offsets
      const std::uint32_t v = counts[static_cast<std::size_t>(ch) * experts + e];
      counts[static_cast<std::size_t>(ch) * experts + e] = run;
      run += v;
    }
    total[e] = run;
  }
  __syncthreads();
  if (tid == 0) {
    std::uint32_t p = 0, t = 0;
    for (std::uint32_t e = 0; e < experts; ++e) {
      poff[e] = p;
      toff[e] = t;
      p += total[e];
      t += (total[e] + kMoeTile - 1) / kMoeTile;
    }
    *ntiles = t;
  }
  __syncthreads();
  for (std::uint32_t e = tid; e < experts; e += blockDim.x) {
    base[e] = poff[e];
    for (std::uint32_t i = 0; i * kMoeTile < total[e]; ++i) {
      tiles[3 * (toff[e] + i)] = e;
      tiles[3 * (toff[e] + i) + 1] = poff[e] + i * kMoeTile;
      tiles[3 * (toff[e] + i) + 2] = min(kMoeTile, total[e] - i * kMoeTile);
    }
  }
}
extern "C" __global__ void __launch_bounds__(1024)
    halo_moe_bucket_scatter(const std::uint32_t* ids, std::uint32_t pairs, std::uint32_t experts,
                            const std::uint32_t* chunk_off, const std::uint32_t* base, std::uint32_t* sorted,
                            std::uint32_t* pos) {
  __shared__ std::uint32_t chunk_ids[kBucketChunk];
  const std::uint32_t chunk = blockIdx.x, tid = threadIdx.x, p = chunk * kBucketChunk + tid;
  const std::uint32_t e = p < pairs ? ids[p] : 0xFFFFFFFFu;
  chunk_ids[tid] = e;
  __syncthreads();
  if (p >= pairs) return;
  std::uint32_t rank = 0;
  for (std::uint32_t i = 0; i < tid; ++i) rank += chunk_ids[i] == e;
  const std::uint32_t at = base[e] + chunk_off[static_cast<std::size_t>(chunk) * experts + e] + rank;
  sorted[at] = p;
  pos[p] = at;
}

namespace {

// 16 weights (k0 % 16 == 0) of one row as a WMMA A fragment, in the
// K order kMoePerm: within every group of 4, positions (0, 1, 2, 3) hold
// k (0, 2, 1, 3). Activation fragments are staged in the same order, so
// the dot products are unchanged, and the specialized loaders can build
// pairs (byte 0, byte 2) and (byte 1, byte 3) of a word with masks alone:
// a value v < 1024 ORed into fp16 1024 (0x6400) is exactly 1024 + v, so
// one subtraction gives v exactly and one packed multiply-add scales it.
__device__ inline std::uint32_t MoePerm(std::uint32_t c) { return (c & ~3u) | ((c & 1u) << 1) | ((c >> 1) & 1u); }

typedef _Float16 halo_h2 __attribute__((ext_vector_type(2)));
__device__ inline halo_h2 AsH2(std::uint32_t u) { return __builtin_bit_cast(halo_h2, u); }

// Bytes of `w` (each < 1024 after `bias` is removed) times scale plus
// offset, written to v[4 j .. 4 j + 3] in kMoePerm order.
__device__ inline void Unpack4(halo_v16h& v, int j, std::uint32_t bytes, halo_h2 bias, halo_h2 scale, halo_h2 offset) {
  const halo_h2 p0 = (AsH2((bytes & 0x00FF00FFu) | 0x64006400u) - bias) * scale + offset;
  const halo_h2 p1 = (AsH2(((bytes >> 8) & 0x00FF00FFu) | 0x64006400u) - bias) * scale + offset;
  v[4 * j] = p0.x;
  v[4 * j + 1] = p0.y;
  v[4 * j + 2] = p1.x;
  v[4 * j + 3] = p1.y;
}

// Per-stage interface used by the grouped gate/up: LoadHdr(row, k0) once
// per kMoeKs-value stage (k0 % kMoeKs == 0), then GetH(row, k0 + kk, hdr)
// per 16 values. Formats whose scales sit beside their quants keep Get.
template <typename L>
struct Frag16 {
  struct Hdr {};
  __device__ static Hdr LoadHdr(const std::uint8_t*, std::uint32_t) { return {}; }
  __device__ static halo_v16h GetH(const std::uint8_t* row, std::uint32_t k, const Hdr&) { return Get(row, k); }
  // Stage prefetch (kMoeKs values from k0): Q4_K loads its quant bytes up
  // front; other formats load per fragment as before.
  struct Stage {};
  __device__ static Stage LoadStage(const std::uint8_t*, std::uint32_t, std::uint32_t = 128) { return {}; }
  __device__ static halo_v16h GetS(const std::uint8_t* row, std::uint32_t k0, std::uint32_t kk, const Hdr& h,
                                   const Stage&) {
    return GetH(row, k0 + kk, h);
  }
  __device__ static halo_v16h Get(const std::uint8_t* row, std::uint32_t k0) {
    halo_v16h a;
#pragma unroll
    for (std::uint32_t i = 0; i < 16; ++i) a[i] = static_cast<_Float16>(L::At(row, k0 + MoePerm(i)));
    return a;
  }
};
// K-quants: a 128-value stage is half a super-block, sub-blocks j0 .. j0 + 3;
// the header (d, dmin, 12 scale bytes) is one 16-byte load.
struct KQuantHdr {
  _Float16 a[4], c[4];
};
__device__ inline KQuantHdr LoadKQuantHdr(const std::uint8_t* b, std::uint32_t j0) {
  const uint4 h = *reinterpret_cast<const uint4*>(b);
  const float d = F16(static_cast<std::uint16_t>(h.x & 0xFFFF)), dmin = F16(static_cast<std::uint16_t>(h.x >> 16));
  KQuantHdr r;
#pragma unroll
  for (std::uint32_t i = 0; i < 4; ++i) {
    std::uint32_t sc, m;
    ScaleMinK4Words(j0 + i, h.y, h.z, h.w, sc, m);
    r.a[i] = static_cast<_Float16>(d * static_cast<float>(sc));
    r.c[i] = static_cast<_Float16>(-dmin * static_cast<float>(m));
  }
  return r;
}
template <>
struct Frag16<LoadQ4_K> {
  using Hdr = KQuantHdr;
  __device__ static Hdr LoadHdr(const std::uint8_t* row, std::uint32_t k0) {
    return LoadKQuantHdr(row + (k0 / 256) * 144, (k0 % 256) / 32);
  }
  // k % kMoeKs selects the sub-block; constant once the stage loop unrolls.
  __device__ static halo_v16h GetH(const std::uint8_t* row, std::uint32_t k, const Hdr& hd) {
    const std::uint8_t* b = row + (k / 256) * 144;
    const std::uint32_t e = k % 256, chunk = e / 64, hi = (e % 64) / 32, l0 = e % 32, i = (k % 128) / 32;
    const uint4 q = *reinterpret_cast<const uint4*>(b + 16 + chunk * 32 + l0);
    const std::uint32_t w[4] = {q.x, q.y, q.z, q.w};
    halo_v16h v;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      Unpack4(v, j, (w[j] >> (4 * hi)) & 0x0F0F0F0Fu, halo_h2{1024, 1024}, halo_h2{hd.a[i], hd.a[i]},
              halo_h2{hd.c[i], hd.c[i]});
    return v;
  }
  // A 128-value stage reads 64 quant bytes per row (two 32-byte chunks; the
  // high-nibble half of each chunk reuses its bytes): 4 loads, issued before
  // they are needed.
  struct Stage {
    uint4 q[4];
  };
  __device__ static Stage LoadStage(const std::uint8_t* row, std::uint32_t k0, std::uint32_t = 128) {
    const std::uint8_t* qb = row + (k0 / 256) * 144 + 16 + ((k0 % 256) / 64) * 32;
    Stage st;
#pragma unroll
    for (int i = 0; i < 4; ++i) st.q[i] = *reinterpret_cast<const uint4*>(qb + 16 * i);
    return st;
  }
  // kk in 0 .. 127 (constant once unrolled): chunk kk / 64, nibble (kk % 64) / 32.
  __device__ static halo_v16h GetS(const std::uint8_t*, std::uint32_t, std::uint32_t kk, const Hdr& hd,
                                   const Stage& st) {
    const std::uint32_t hi = (kk % 64) / 32, i = kk / 32;
    const uint4 q = st.q[(kk / 64) * 2 + (kk % 32) / 16];
    const std::uint32_t w[4] = {q.x, q.y, q.z, q.w};
    halo_v16h v;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      Unpack4(v, j, (w[j] >> (4 * hi)) & 0x0F0F0F0Fu, halo_h2{1024, 1024}, halo_h2{hd.a[i], hd.a[i]},
              halo_h2{hd.c[i], hd.c[i]});
    return v;
  }
  __device__ static halo_v16h Get(const std::uint8_t* row, std::uint32_t k0) {
    const std::uint8_t* b = row + (k0 / 256) * 144;
    const std::uint32_t e = k0 % 256, chunk = e / 64, hi = (e % 64) / 32, l0 = e % 32;
    std::uint32_t sc, m;
    ScaleMinK4(2 * chunk + hi, b + 4, sc, m);
    const _Float16 a = static_cast<_Float16>(F16(U16(b)) * static_cast<float>(sc));
    const _Float16 c = static_cast<_Float16>(-F16(U16(b + 2)) * static_cast<float>(m));
    const uint4 q = *reinterpret_cast<const uint4*>(b + 16 + chunk * 32 + l0);
    const std::uint32_t w[4] = {q.x, q.y, q.z, q.w};
    halo_v16h v;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      Unpack4(v, j, (w[j] >> (4 * hi)) & 0x0F0F0F0Fu, halo_h2{1024, 1024}, halo_h2{a, a}, halo_h2{c, c});
    return v;
  }
};
template <>
struct Frag16<LoadQ5_K> {
  struct Stage {};
  __device__ static Stage LoadStage(const std::uint8_t*, std::uint32_t, std::uint32_t = 128) { return {}; }
  template <typename H>
  __device__ static halo_v16h GetS(const std::uint8_t* row, std::uint32_t k0, std::uint32_t kk, const H& h,
                                   const Stage&) {
    return GetH(row, k0 + kk, h);
  }
  using Hdr = KQuantHdr;
  __device__ static Hdr LoadHdr(const std::uint8_t* row, std::uint32_t k0) {
    return LoadKQuantHdr(row + (k0 / 256) * 176, (k0 % 256) / 32);
  }
  __device__ static halo_v16h GetH(const std::uint8_t* row, std::uint32_t k, const Hdr& hd) {
    const std::uint8_t* b = row + (k / 256) * 176;
    const std::uint32_t e = k % 256, chunk = e / 64, hi = (e % 64) / 32, l0 = e % 32, bit = 2 * chunk + hi,
                        i = (k % 128) / 32;
    const uint4 q = *reinterpret_cast<const uint4*>(b + 48 + chunk * 32 + l0);
    const uint4 h = *reinterpret_cast<const uint4*>(b + 16 + l0);
    const std::uint32_t w[4] = {q.x, q.y, q.z, q.w}, hw[4] = {h.x, h.y, h.z, h.w};
    halo_v16h v;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      Unpack4(v, j, ((w[j] >> (4 * hi)) & 0x0F0F0F0Fu) | (((hw[j] >> bit) & 0x01010101u) << 4), halo_h2{1024, 1024},
              halo_h2{hd.a[i], hd.a[i]}, halo_h2{hd.c[i], hd.c[i]});
    return v;
  }
  __device__ static halo_v16h Get(const std::uint8_t* row, std::uint32_t k0) {
    const std::uint8_t* b = row + (k0 / 256) * 176;
    const std::uint32_t e = k0 % 256, chunk = e / 64, hi = (e % 64) / 32, l0 = e % 32, bit = 2 * chunk + hi;
    std::uint32_t sc, m;
    ScaleMinK4(bit, b + 4, sc, m);
    const _Float16 a = static_cast<_Float16>(F16(U16(b)) * static_cast<float>(sc));
    const _Float16 c = static_cast<_Float16>(-F16(U16(b + 2)) * static_cast<float>(m));
    const uint4 q = *reinterpret_cast<const uint4*>(b + 48 + chunk * 32 + l0);
    const uint4 h = *reinterpret_cast<const uint4*>(b + 16 + l0);
    const std::uint32_t w[4] = {q.x, q.y, q.z, q.w}, hw[4] = {h.x, h.y, h.z, h.w};
    halo_v16h v;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      Unpack4(v, j, ((w[j] >> (4 * hi)) & 0x0F0F0F0Fu) | (((hw[j] >> bit) & 0x01010101u) << 4), halo_h2{1024, 1024},
              halo_h2{a, a}, halo_h2{c, c});
    return v;
  }
};
template <>
struct Frag16<LoadQ5_1> {
  // A 128-value stage is 4 blocks of 24 bytes (8-byte aligned): 12 loads
  // up front, the blocks past `kn` remaining values left unloaded.
  struct Stage {
    uint2 r[12];
  };
  __device__ static Stage LoadStage(const std::uint8_t* row, std::uint32_t k0, std::uint32_t kn = 128) {
    const std::uint8_t* b = row + (k0 / 32) * 24;
    Stage st;
#pragma unroll
    for (std::uint32_t i = 0; i < 12; ++i)
      st.r[i] = (i / 3) * 32 < kn ? *reinterpret_cast<const uint2*>(b + 8 * i) : make_uint2(0, 0);
    return st;
  }
  template <typename H>
  __device__ static halo_v16h GetS(const std::uint8_t*, std::uint32_t, std::uint32_t kk, const H&, const Stage& st) {
    const std::uint32_t blk = kk / 32;
    return Decode(st.r[3 * blk], st.r[3 * blk + 1], st.r[3 * blk + 2], kk % 32);
  }
  __device__ static halo_v16h Decode(uint2 hd, uint2 q0, uint2 q1, std::uint32_t j0) {
    const _Float16 d = AsH2(hd.x).x, m = AsH2(hd.x).y;
    const std::uint32_t qh = hd.y >> j0;
    const std::uint32_t w[4] = {q0.x, q0.y, q1.x, q1.y};
    halo_v16h v;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const std::uint32_t x = (qh >> (4 * j)) & 0xFu;
      const std::uint32_t bits = (x & 1u) | ((x & 2u) << 7) | ((x & 4u) << 14) | ((x & 8u) << 21);
      Unpack4(v, j, ((w[j] >> (j0 ? 4 : 0)) & 0x0F0F0F0Fu) | (bits << 4), halo_h2{1024, 1024}, halo_h2{d, d},
              halo_h2{m, m});
    }
    return v;
  }
  struct Hdr {};
  __device__ static Hdr LoadHdr(const std::uint8_t*, std::uint32_t) { return {}; }
  __device__ static halo_v16h GetH(const std::uint8_t* row, std::uint32_t k, const Hdr&) { return Get(row, k); }
  __device__ static halo_v16h Get(const std::uint8_t* row, std::uint32_t k0) {
    const std::uint8_t* b = row + (k0 / 32) * 24;  // 8-byte aligned
    const std::uint32_t j0 = k0 % 32;
    const uint2 hd = *reinterpret_cast<const uint2*>(b);
    const _Float16 d = AsH2(hd.x).x, m = AsH2(hd.x).y;
    const std::uint32_t qh = hd.y >> j0;
    const uint2 q0 = *reinterpret_cast<const uint2*>(b + 8), q1 = *reinterpret_cast<const uint2*>(b + 16);
    const std::uint32_t w[4] = {q0.x, q0.y, q1.x, q1.y};
    halo_v16h v;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const std::uint32_t x = (qh >> (4 * j)) & 0xFu;
      const std::uint32_t bits = (x & 1u) | ((x & 2u) << 7) | ((x & 4u) << 14) | ((x & 8u) << 21);
      Unpack4(v, j, ((w[j] >> (j0 ? 4 : 0)) & 0x0F0F0F0Fu) | (bits << 4), halo_h2{1024, 1024}, halo_h2{d, d},
              halo_h2{m, m});
    }
    return v;
  }
};
// Signed bytes: x ^ 0x80 = x + 128 < 256.
__device__ inline halo_v16h Int8Frag(const std::uint32_t w[4], _Float16 d) {
  halo_v16h v;
#pragma unroll
  for (int j = 0; j < 4; ++j) Unpack4(v, j, w[j] ^ 0x80808080u, halo_h2{1152, 1152}, halo_h2{d, d}, halo_h2{0, 0});
  return v;
}
template <>
struct Frag16<LoadQ8_0> {
  struct Stage {};
  __device__ static Stage LoadStage(const std::uint8_t*, std::uint32_t, std::uint32_t = 128) { return {}; }
  template <typename H>
  __device__ static halo_v16h GetS(const std::uint8_t* row, std::uint32_t k0, std::uint32_t kk, const H& h,
                                   const Stage&) {
    return GetH(row, k0 + kk, h);
  }
  struct Hdr {};
  __device__ static Hdr LoadHdr(const std::uint8_t*, std::uint32_t) { return {}; }
  __device__ static halo_v16h GetH(const std::uint8_t* row, std::uint32_t k, const Hdr&) { return Get(row, k); }
  __device__ static halo_v16h Get(const std::uint8_t* row, std::uint32_t k0) {
    const std::uint8_t* b = row + (k0 / 32) * 34;  // 2-byte aligned
    const std::uint16_t* q = reinterpret_cast<const std::uint16_t*>(b + 2 + k0 % 32);
    std::uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = static_cast<std::uint32_t>(q[2 * j]) | (static_cast<std::uint32_t>(q[2 * j + 1]) << 16);
    return Int8Frag(w, AsH2(U16(b)).x);
  }
};

__device__ inline halo_v16h LdsFrag(const _Float16* p) {
  typedef _Float16 v8h __attribute__((ext_vector_type(8)));
  const v8h lo = *reinterpret_cast<const v8h*>(p), hi = *reinterpret_cast<const v8h*>(p + 8);
  halo_v16h b;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    b[e] = lo[e];
    b[e + 8] = hi[e];
  }
  return b;
}

// K staged 128 at a time for up to kMoeTile token rows.
constexpr std::uint32_t kMoeKs = 128, kMoeRs = kMoeKs + 8, kMoeSub = kMoeTile / 16;
// Row tiles per wave: a block covers 8 * 16 * kMoeRowTiles weight rows, so a
// tile's activation rows are staged ceil(rows / that) times. 2 halves that
// traffic but costs occupancy (160 -> 228 VGPRs in gate/up): measured +31 ms
// per 2048-token prefill (within noise), so 1.
constexpr std::uint32_t kMoeRowTiles = 1, kMoeBlockRows = 8 * 16 * kMoeRowTiles;
static_assert(kMoeKs == 128, "the K-quant stage headers (KQuantHdr) cover 128 values");

// gate/up: hp[pair][r] = silu(gate_e[r] . x_t) * (up_e[r] . x_t). Each wave
// owns kMoeRowTiles x 16 rows; every weight fragment feeds the tile's
// 16-token sub-tiles. x and hp are fp16 rows in kMoePerm order
// (halo_f16_perm_rows), so staging is a plain copy.
//   grid (ceil(ff / kMoeBlockRows), max tiles), block 256; H % 256 == 0.
template <typename L>
__device__ void MoeGateUp(const std::uint8_t* gate, const std::uint8_t* up, std::uint64_t expert_bytes,
                          std::uint64_t row_bytes, const _Float16* x, const std::uint32_t* sorted,
                          const std::uint32_t* tiles, const std::uint32_t* ntiles, _Float16* hp, std::uint32_t ff,
                          std::uint32_t H, std::uint32_t k) {
  __shared__ __attribute__((aligned(16))) _Float16 Bs[kMoeTile * kMoeRs];
  __shared__ std::uint32_t tok[kMoeTile];
  const std::uint32_t tile = blockIdx.y;
  if (tile >= *ntiles) return;
  const std::uint32_t e = tiles[3 * tile], first = tiles[3 * tile + 1], n = tiles[3 * tile + 2];
  const std::uint32_t subs = (n + 15) / 16;  // block-uniform
  const std::uint32_t lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  const std::uint32_t r0 = blockIdx.x * kMoeBlockRows + wave * 16 * kMoeRowTiles;
  const std::uint8_t* g[kMoeRowTiles];
  const std::uint8_t* u[kMoeRowTiles];
#pragma unroll
  for (std::uint32_t rt = 0; rt < kMoeRowTiles; ++rt) {
    const std::uint32_t row = min(r0 + rt * 16 + lane % 16, ff - 1);
    g[rt] = gate + e * expert_bytes + row * row_bytes;
    u[rt] = up + e * expert_bytes + row * row_bytes;
  }
  for (std::uint32_t i = threadIdx.x; i < kMoeTile; i += blockDim.x) tok[i] = sorted[first + (i < n ? i : 0)] / k;
  halo_v8f ag[kMoeRowTiles][kMoeSub] = {}, au[kMoeRowTiles][kMoeSub] = {};
  static_assert(kMoeRowTiles == 1, "stage prefetch holds one row tile's weights");
  // Software pipeline over stages: while a stage is multiplied out, the next
  // stage's weights (header and quant bytes) and activation pieces load into
  // registers; staging into LDS then only stores registers. (Thread trace:
  // per-fragment weight loads were 74% of the time, then the synchronous
  // activation staging the largest remaining wait.)
  constexpr std::uint32_t kXp = kMoeTile * (kMoeKs / 8) / 256;  // activation pieces per thread
  __syncthreads();  // tok[] written
  // Four named registers, unconditional (all kMoeTile rows: tok[] repeats
  // the first token past n, and those rows are never written out). As an
  // array the compiler kept xr in scratch and indexed it with a runtime
  // counter in the store loop (kernel_report: 80 B), unroll pragma or not.
  static_assert(kXp == 4, "four activation pieces per thread per stage");
  uint4 xr0, xr1, xr2, xr3;
  const auto x_at = [&](std::uint32_t p, std::uint32_t k0) {
    const std::uint32_t i = threadIdx.x + p * 256, r = i / (kMoeKs / 8), c = (i % (kMoeKs / 8)) * 8;
    return *reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(tok[r]) * H + k0 + c);
  };
  const auto bs_at = [&](std::uint32_t p) {
    const std::uint32_t i = threadIdx.x + p * 256, r = i / (kMoeKs / 8), c = (i % (kMoeKs / 8)) * 8;
    return reinterpret_cast<uint4*>(Bs + r * kMoeRs + c);
  };
#define HALO_MOE_LOAD_X(K0) \
  xr0 = x_at(0, (K0));      \
  xr1 = x_at(1, (K0));      \
  xr2 = x_at(2, (K0));      \
  xr3 = x_at(3, (K0));
  auto hg = Frag16<L>::LoadHdr(g[0], 0), hu = Frag16<L>::LoadHdr(u[0], 0);
  auto sg = Frag16<L>::LoadStage(g[0], 0), su = Frag16<L>::LoadStage(u[0], 0);
  HALO_MOE_LOAD_X(0)
  for (std::uint32_t k0 = 0; k0 < H; k0 += kMoeKs) {
    __syncthreads();  // the previous stage's LDS reads are done
    *bs_at(0) = xr0;
    *bs_at(1) = xr1;
    *bs_at(2) = xr2;
    *bs_at(3) = xr3;
    __syncthreads();
    const auto cg = hg, cu = hu;
    const auto csg = sg, csu = su;
    if (k0 + kMoeKs < H) {  // next stage, in flight during the math below
      hg = Frag16<L>::LoadHdr(g[0], k0 + kMoeKs);
      hu = Frag16<L>::LoadHdr(u[0], k0 + kMoeKs);
      sg = Frag16<L>::LoadStage(g[0], k0 + kMoeKs);
      su = Frag16<L>::LoadStage(u[0], k0 + kMoeKs);
      HALO_MOE_LOAD_X(k0 + kMoeKs)
    }
#pragma unroll
    for (std::uint32_t rt = 0; rt < kMoeRowTiles; ++rt) {
#pragma unroll
      for (std::uint32_t kk = 0; kk < kMoeKs; kk += 16) {
        const halo_v16h fg = Frag16<L>::GetS(g[rt], k0, kk, cg, csg), fu = Frag16<L>::GetS(u[rt], k0, kk, cu, csu);
#pragma unroll
        for (std::uint32_t h = 0; h < kMoeSub; ++h) {
          if (h >= subs) break;
          const halo_v16h b = LdsFrag(&Bs[(h * 16 + lane % 16) * kMoeRs + kk]);
          ag[rt][h] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fg, b, ag[rt][h]);
          au[rt][h] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fu, b, au[rt][h]);
        }
      }
    }
  }
#pragma unroll
  for (std::uint32_t rt = 0; rt < kMoeRowTiles; ++rt)
#pragma unroll
    for (std::uint32_t h = 0; h < kMoeSub; ++h) {
      const std::uint32_t t = h * 16 + lane % 16;
      if (t >= n) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t r = r0 + rt * 16 + 2 * j + lane / 16;
        if (r < ff)
          hp[static_cast<std::size_t>(first + t) * ff + MoePerm(r)] = static_cast<_Float16>(Silu(ag[rt][h][j]) * au[rt][h][j]);
      }
    }
}

#undef HALO_MOE_LOAD_X

// down: yp[pair][r] = down_e[r] . hp[pair]; the tile's pairs are contiguous.
// load(e, row, k0, ks) returns a stage of weights (loaded before the
// activation staging), get(stage, e, row, k0, kk) 16 weights of it.
//   grid (ceil(H / kMoeBlockRows), max tiles), block 256; ff % 16 == 0.
template <typename SL, typename SG>
__device__ void MoeDown(SL load, SG get, const _Float16* hp, const std::uint32_t* tiles, const std::uint32_t* ntiles,
                        _Float16* yp, std::uint32_t H, std::uint32_t ff) {
  __shared__ __attribute__((aligned(16))) _Float16 Bs[kMoeTile * kMoeRs];
  const std::uint32_t tile = blockIdx.y;
  if (tile >= *ntiles) return;
  const std::uint32_t e = tiles[3 * tile], first = tiles[3 * tile + 1], n = tiles[3 * tile + 2];
  const std::uint32_t subs = (n + 15) / 16;
  const std::uint32_t lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  const std::uint32_t r0 = blockIdx.x * kMoeBlockRows + wave * 16 * kMoeRowTiles;
  halo_v8f acc[kMoeRowTiles][kMoeSub] = {};
  const std::uint32_t row = min(r0 + lane % 16, H - 1);
  for (std::uint32_t k0 = 0; k0 < ff; k0 += kMoeKs) {
    const std::uint32_t ks = min(kMoeKs, ff - k0);
    const auto st = load(e, row, k0, ks);  // in flight during the staging below
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < subs * 16 * ks / 8; i += blockDim.x) {
      const std::uint32_t r = i / (ks / 8), c = (i % (ks / 8)) * 8;
      *reinterpret_cast<uint4*>(Bs + r * kMoeRs + c) =
          *reinterpret_cast<const uint4*>(hp + static_cast<std::size_t>(first + (r < n ? r : 0)) * ff + k0 + c);
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t kk = 0; kk < kMoeKs; kk += 16) {  // fully unrolled: the stage stays in registers
      if (kk >= ks) break;
#pragma unroll
      for (std::uint32_t rt = 0; rt < kMoeRowTiles; ++rt) {
        const halo_v16h fw = get(st, e, row, k0, kk);
#pragma unroll
        for (std::uint32_t h = 0; h < kMoeSub; ++h) {
          if (h >= subs) break;
          acc[rt][h] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fw, LdsFrag(&Bs[(h * 16 + lane % 16) * kMoeRs + kk]), acc[rt][h]);
        }
      }
    }
  }
#pragma unroll
  for (std::uint32_t rt = 0; rt < kMoeRowTiles; ++rt)
#pragma unroll
    for (std::uint32_t h = 0; h < kMoeSub; ++h) {
      const std::uint32_t t = h * 16 + lane % 16;
      if (t >= n) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t r = r0 + rt * 16 + 2 * j + lane / 16;
        if (r < H) yp[static_cast<std::size_t>(first + t) * H + r] = static_cast<_Float16>(acc[rt][h][j]);
      }
    }
}

}  // namespace
#define HALO_MOE_GROUPED(SUFFIX, L)                                                                                   \
  extern "C" __global__ void __launch_bounds__(256) halo_moe_gate_up_grouped_##SUFFIX(                               \
      const std::uint8_t* gate, const std::uint8_t* up, std::uint64_t expert_bytes, std::uint64_t row_bytes,          \
      const _Float16* x, const std::uint32_t* sorted, const std::uint32_t* tiles, const std::uint32_t* ntiles,        \
      _Float16* hp, std::uint32_t ff, std::uint32_t H, std::uint32_t k) {                                             \
    MoeGateUp<L>(gate, up, expert_bytes, row_bytes, x, sorted, tiles, ntiles, hp, ff, H, k);                          \
  }                                                                                                                   \
  extern "C" __global__ void __launch_bounds__(256) halo_moe_down_grouped_##SUFFIX(                                  \
      const std::uint8_t* down, std::uint64_t expert_bytes, std::uint64_t row_bytes, const _Float16* hp,              \
      const std::uint32_t* tiles, const std::uint32_t* ntiles, _Float16* yp, std::uint32_t H, std::uint32_t ff) {     \
    MoeDown(                                                                                                          \
        [=](std::uint32_t e, std::uint32_t row, std::uint32_t k0, std::uint32_t ks) {                                 \
          return Frag16<L>::LoadStage(down + e * expert_bytes + row * row_bytes, k0, ks);                             \
        },                                                                                                            \
        [=](const auto& st, std::uint32_t e, std::uint32_t row, std::uint32_t k0, std::uint32_t kk) {                 \
          return Frag16<L>::GetS(down + e * expert_bytes + row * row_bytes, k0, kk, typename Frag16<L>::Hdr{}, st);  \
        },                                                                                                            \
        hp, tiles, ntiles, yp, H, ff);                                                                                \
  }
HALO_MOE_GROUPED(q4_k, LoadQ4_K)
HALO_MOE_GROUPED(q5_k, LoadQ5_K)
HALO_MOE_GROUPED(q6_k, LoadQ6_K)
HALO_MOE_GROUPED(q5_1, LoadQ5_1)
HALO_MOE_GROUPED(q8_0, LoadQ8_0)
HALO_MOE_GROUPED(f32, LoadF32)
#undef HALO_MOE_GROUPED

// Down projection over repacked Q8_0 (int8 rows qstride apart, fp16 scales
// per 32 values; experts rows * qstride and rows * cols / 32 apart).
extern "C" __global__ void __launch_bounds__(256)
    halo_moe_down_grouped_q8r(const std::int8_t* qs, const std::uint16_t* d, std::uint32_t qstride, const _Float16* hp,
                              const std::uint32_t* tiles, const std::uint32_t* ntiles, _Float16* yp, std::uint32_t H,
                              std::uint32_t ff) {
  MoeDown([](std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) { return 0; },
          [=](int, std::uint32_t e, std::uint32_t row, std::uint32_t k0, std::uint32_t kk) {
    k0 += kk;
    const std::size_t r = static_cast<std::size_t>(e) * H + row;
    const int4 q = *reinterpret_cast<const int4*>(qs + r * qstride + k0);
    const std::uint32_t w[4] = {static_cast<std::uint32_t>(q.x), static_cast<std::uint32_t>(q.y),
                                static_cast<std::uint32_t>(q.z), static_cast<std::uint32_t>(q.w)};
    return Int8Frag(w, AsH2(d[r * (ff / 32) + k0 / 32]).x);
  }, hp, tiles, ntiles, yp, H, ff);
}

// fp16 copy of n values (n % 4 == 0) in kMoePerm order, for the grouped
// gate/up's activation staging. grid (ceil(n / 256))
extern "C" __global__ void halo_f16_perm_rows(const float* x, _Float16* y, std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[MoePerm(i)] = static_cast<_Float16>(x[i]);
}

// out[t][i] = sum_s w[t][s] * yp[pos[t * k + s]][i], slots in order.
//   grid (ceil(n / 256), T), block 256
// yp is fp16 (the down kernels' per-pair rows; halves their traffic), the
// weighted sum fp32.
extern "C" __global__ void halo_moe_combine_grouped(const _Float16* yp, const std::uint32_t* pos, const float* w,
                                                    float* out, std::uint32_t k, std::uint32_t n) {
  const std::uint32_t t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < k; ++s)
    acc += w[t * k + s] * static_cast<float>(yp[static_cast<std::size_t>(pos[t * k + s]) * n + i]);
  out[static_cast<std::size_t>(t) * n + i] = acc;
}

// ---- P4: chunk attention on FP16 WMMA ---------------------------------------
//
// halo_attention_select_rows' result for head_dim 256 by tiles: one block
// per (KV head g, token t); the G <= 16 query heads sharing g are the rows
// of a 16-row tile (padding rows zero). The token's visible keys run 64 at
// a time with an online softmax: each wave scores 16 keys (Q . K^T, K read
// from the cache as fp16 fragments), the block rescales and exponentiates
// in LDS, then each wave adds P . V for its 64 of the 256 output dims (V
// staged in LDS). Inputs are rounded to fp16, scores and sums stay fp32.
//   grid (num_kv_heads, T), block 128; visible lists max_n apart.
extern "C" __global__ void __launch_bounds__(128)
    halo_attention_wmma_rows(const float* q, const float* kc, const float* vc, const float* gate,
                             const std::uint32_t* visible, float* out, const std::uint32_t* params,
                             std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t max_n) {
  constexpr std::uint32_t HD = 256, QR = HD + 8, KC = 64, VR = HD + 8, PR = KC + 8;
  __shared__ __attribute__((aligned(16))) _Float16 Qs[16 * QR];
  __shared__ __attribute__((aligned(16))) _Float16 Vs[KC * VR];
  __shared__ __attribute__((aligned(16))) _Float16 Ps[16 * PR];
  __shared__ float Ss[16 * KC];
  __shared__ float mrow[16], lrow[16], alpha[16];
  const std::uint32_t g = blockIdx.x, t = blockIdx.y, G = num_heads / num_kv_heads;
  const std::uint32_t tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t n = params[static_cast<std::size_t>(t) * 16 + kHaloNVisible];
  const std::uint32_t* vis = visible + static_cast<std::size_t>(t) * max_n;
  const float* qrow = q + static_cast<std::size_t>(t) * num_heads * HD;
  const float scale = 1.0f / sqrtf(static_cast<float>(HD));
  for (std::uint32_t i = tid; i < 16 * HD; i += blockDim.x) {
    const std::uint32_t h = i / HD, d = i % HD;
    Qs[h * QR + d] = static_cast<_Float16>(h < G ? qrow[(g * G + h) * HD + d] : 0.0f);
  }
  if (tid < 16) {
    mrow[tid] = -INFINITY;
    lrow[tid] = 0.0f;
  }
  halo_v8f acc[4] = {};
  for (std::uint32_t c0 = 0; c0 < n; c0 += KC) {
    __syncthreads();  // previous chunk's Vs / Ps reads done (and Qs written)
    for (std::uint32_t i = tid; i < KC * HD / 4; i += blockDim.x) {
      const std::uint32_t j = i / (HD / 4), d = (i % (HD / 4)) * 4;
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (c0 + j < n)
        v = *reinterpret_cast<const float4*>(vc + (static_cast<std::size_t>(vis[c0 + j]) * num_kv_heads + g) * HD + d);
      _Float16* dst = Vs + j * VR + d;
      dst[0] = static_cast<_Float16>(v.x);
      dst[1] = static_cast<_Float16>(v.y);
      dst[2] = static_cast<_Float16>(v.z);
      dst[3] = static_cast<_Float16>(v.w);
    }
    // Scores of this wave's 16 keys.
    const std::uint32_t kj = c0 + wave * 16 + lane % 16;
    const float* krow = kc + (static_cast<std::size_t>(vis[min(kj, n - 1)]) * num_kv_heads + g) * HD;
    halo_v8f s = {};
    for (std::uint32_t k0 = 0; k0 < HD; k0 += 16) {
      const halo_v16h a = LdsFrag(&Qs[(lane % 16) * QR + k0]);
      halo_v16h b;
#pragma unroll
      for (int i = 0; i < 16; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(krow + k0 + i);
        b[i] = static_cast<_Float16>(v.x);
        b[i + 1] = static_cast<_Float16>(v.y);
        b[i + 2] = static_cast<_Float16>(v.z);
        b[i + 3] = static_cast<_Float16>(v.w);
      }
      s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
      Ss[(2 * j + lane / 16) * KC + wave * 16 + lane % 16] = kj < n ? s[j] * scale : -INFINITY;
    __syncthreads();
    // Online softmax: 8 threads per head row, 8 keys each.
    {
      const std::uint32_t row = tid / 8, part = tid % 8;
      float mx = -INFINITY;
      for (std::uint32_t c = part * 8; c < part * 8 + 8; ++c) mx = fmaxf(mx, Ss[row * KC + c]);
      for (int o = 4; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
      const float m_old = mrow[row], m_new = fmaxf(m_old, mx);
      float sum = 0.0f;
      for (std::uint32_t c = part * 8; c < part * 8 + 8; ++c) {
        const float p = expf(Ss[row * KC + c] - m_new);
        Ps[row * PR + c] = static_cast<_Float16>(p);
        sum += p;
      }
      for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
      __syncthreads();  // every lane of the row has read mrow
      if (part == 0) {
        const float a = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
        alpha[row] = a;
        lrow[row] = lrow[row] * a + sum;
        mrow[row] = m_new;
      }
    }
    __syncthreads();
#pragma unroll
    for (int tile = 0; tile < 4; ++tile)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[tile][j] *= alpha[2 * j + lane / 16];
    for (std::uint32_t kk = 0; kk < KC; kk += 16) {
      const halo_v16h a = LdsFrag(&Ps[(lane % 16) * PR + kk]);
#pragma unroll
      for (int tile = 0; tile < 4; ++tile) {
        const std::uint32_t d = wave * 64 + tile * 16 + lane % 16;
        halo_v16h b;
#pragma unroll
        for (int i = 0; i < 16; ++i) b[i] = Vs[(kk + i) * VR + d];
        acc[tile] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[tile]);
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int tile = 0; tile < 4; ++tile)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t h = 2 * j + lane / 16, d = wave * 64 + tile * 16 + lane % 16;
      if (h >= G) continue;
      const std::size_t idx = static_cast<std::size_t>(t) * num_heads * HD + (g * G + h) * HD + d;
      out[idx] = acc[tile][j] / lrow[h] * Sigmoid(gate[idx]);
    }
}

// ---- P4: tiled FP32 GEMM for the F32 projections (router, DeltaNet
// alpha/beta, shared-expert gate) ---------------------------------------------
// y[t][r] = sum_c W[r][c] x[t][c], fp32 throughout: 64 rows x 64 tokens per
// block, K staged 32 at a time in LDS, 4 x 4 outputs per thread, products
// added in ascending k. The next stage's operands load into registers while
// the current one is multiplied out (a thread trace of the unprefetched
// version: 52% of cycles waiting on these loads, 33% at barriers).
//   grid (ceil(T / 64), ceil(rows / 64)), block 256; cols % 16 == 0 (a
//   trailing half stage is zero-filled).
extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_f32_tiled(const float* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols,
                        std::uint32_t tokens) {
  constexpr std::uint32_t BM = 64, BN = 64, BK = 32;
  __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
  const std::uint32_t tid = threadIdx.x, r0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
  const std::uint32_t tr = (tid % 16) * 4, tt = (tid / 16) * 4;
  // Loads: 64 x 32 floats per operand, two float4 per thread (rows lrow and
  // lrow + 32).
  const std::uint32_t lrow = tid / 8, lk = (tid % 8) * 4;
  const float* wp0 = w + static_cast<std::size_t>(min(r0 + lrow, rows - 1)) * cols + lk;
  const float* wp1 = w + static_cast<std::size_t>(min(r0 + lrow + 32, rows - 1)) * cols + lk;
  const float* xp0 = x + static_cast<std::size_t>(min(t0 + lrow, tokens - 1)) * cols + lk;
  const float* xp1 = x + static_cast<std::size_t>(min(t0 + lrow + 32, tokens - 1)) * cols + lk;
  const auto ld = [&](const float* p, std::uint32_t k0) {
    return k0 + lk < cols ? *reinterpret_cast<const float4*>(p + k0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  };
  float4 a0 = ld(wp0, 0), a1 = ld(wp1, 0), b0 = ld(xp0, 0), b1 = ld(xp1, 0);
  float acc[4][4] = {};
  for (std::uint32_t k0 = 0; k0 < cols; k0 += BK) {
    __syncthreads();  // the previous stage's reads are done
    As[lk][lrow] = a0.x;
    As[lk + 1][lrow] = a0.y;
    As[lk + 2][lrow] = a0.z;
    As[lk + 3][lrow] = a0.w;
    As[lk][lrow + 32] = a1.x;
    As[lk + 1][lrow + 32] = a1.y;
    As[lk + 2][lrow + 32] = a1.z;
    As[lk + 3][lrow + 32] = a1.w;
    Bs[lk][lrow] = b0.x;
    Bs[lk + 1][lrow] = b0.y;
    Bs[lk + 2][lrow] = b0.z;
    Bs[lk + 3][lrow] = b0.w;
    Bs[lk][lrow + 32] = b1.x;
    Bs[lk + 1][lrow + 32] = b1.y;
    Bs[lk + 2][lrow + 32] = b1.z;
    Bs[lk + 3][lrow + 32] = b1.w;
    __syncthreads();
    if (k0 + BK < cols) {  // next stage, in flight during the math below
      a0 = ld(wp0, k0 + BK);
      a1 = ld(wp1, k0 + BK);
      b0 = ld(xp0, k0 + BK);
      b1 = ld(xp1, k0 + BK);
    }
    const std::uint32_t kn = min(BK, cols - k0);  // 16 or 32: a zero-filled half adds nothing
#pragma unroll
    for (std::uint32_t k = 0; k < BK; ++k) {
      if (k >= kn) break;
      float av[4], bv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        av[i] = As[k][tr + i];
        bv[i] = Bs[k][tt + i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const std::uint32_t t = t0 + tt + j;
    if (t >= tokens) continue;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (r0 + tr + i < rows) y[static_cast<std::size_t>(t) * rows + r0 + tr + i] = acc[i][j];
  }
}

// ---- P4: FP32 GEMM on bf16 WMMA with hi/lo splitting (the router) -----------
// y[t][r] = sum_k W[r][k] x[t][k] for fp32 W and x. Each operand is split
// while staging into bf16 halves, v = hi + lo (hi = bf16(v), lo = bf16(v -
// hi)), and each product taken as hi*hi + hi*lo + lo*hi with fp32
// accumulation: about 2^-15 relative per product (the dropped lo*lo and the
// rounding of lo), against fp32 FMA's 2^-24, at WMMA rate instead of VALU
// (halo_gemm_f32_tiled: 1.28 ms for the 512 x 2560 router at 2048 tokens).
//   grid (ceil(T / 64), ceil(rows / 64)), block 256; cols % 32 == 0.
typedef short halo_v16s __attribute__((ext_vector_type(16)));
__device__ inline void SplitBf16(float v, short& hi, short& lo) {
  const auto rne = [](float f) {
    const std::uint32_t u = __float_as_uint(f);
    return static_cast<std::uint32_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
  };
  const std::uint32_t h = rne(v);
  hi = static_cast<short>(h);
  lo = static_cast<short>(rne(v - __uint_as_float(h << 16)));
}
__device__ inline halo_v16s LdsFragS(const short* p) {
  typedef short v8s __attribute__((ext_vector_type(8)));
  const v8s a = *reinterpret_cast<const v8s*>(p), b = *reinterpret_cast<const v8s*>(p + 8);
  halo_v16s r;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    r[e] = a[e];
    r[e + 8] = b[e];
  }
  return r;
}
extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_f32_bf16x3(const float* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols,
                         std::uint32_t tokens) {
  constexpr std::uint32_t BM = 64, BN = 64, KS = 32, RS = KS + 8, WM = 4, WN = 2, TN = BN / WN / 16;
  __shared__ __attribute__((aligned(16))) short Ah[BM * RS], Al[BM * RS], Bh[BN * RS], Bl[BN * RS];
  const std::uint32_t tid = threadIdx.x, lane = tid % 32, wave = tid / 32, wr = wave % WM, wt = wave / WM;
  const std::uint32_t r0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
  // Staging: 64 x 32 floats per operand = 512 float4, two per thread.
  const float* wp[2];
  const float* xp[2];
  std::uint32_t srow[2], sk[2];
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const std::uint32_t i = tid + q * 256;
    srow[q] = i / 8;
    sk[q] = (i % 8) * 4;
    wp[q] = w + static_cast<std::size_t>(min(r0 + srow[q], rows - 1)) * cols + sk[q];
    xp[q] = x + static_cast<std::size_t>(min(t0 + srow[q], tokens - 1)) * cols + sk[q];
  }
  float4 ra[2], rb[2];
  const auto load = [&](std::uint32_t k0) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      ra[q] = *reinterpret_cast<const float4*>(wp[q] + k0);
      rb[q] = *reinterpret_cast<const float4*>(xp[q] + k0);
    }
  };
  const auto put = [&](short* hi, short* lo, std::uint32_t row, std::uint32_t k, const float4& v) {
    const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) SplitBf16(f[e], hi[row * RS + k + e], lo[row * RS + k + e]);
  };
  halo_v8f acc[TN] = {};
  load(0);
  for (std::uint32_t k0 = 0; k0 < cols; k0 += KS) {
    __syncthreads();  // the previous stage's fragment reads are done
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      put(Ah, Al, srow[q], sk[q], ra[q]);
      put(Bh, Bl, srow[q], sk[q], rb[q]);
    }
    __syncthreads();
    if (k0 + KS < cols) load(k0 + KS);  // in flight during the math below
#pragma unroll
    for (std::uint32_t kk = 0; kk < KS; kk += 16) {
      const std::uint32_t ar = (wr * 16 + lane % 16) * RS + kk;
      const halo_v16s ah = LdsFragS(Ah + ar), al = LdsFragS(Al + ar);
#pragma unroll
      for (std::uint32_t n = 0; n < TN; ++n) {
        const std::uint32_t br = ((wt * TN + n) * 16 + lane % 16) * RS + kk;
        const halo_v16s bh = LdsFragS(Bh + br), bl = LdsFragS(Bl + br);
        acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al, bh, acc[n]);
        acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bl, acc[n]);
        acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bh, acc[n]);
      }
    }
  }
#pragma unroll
  for (std::uint32_t n = 0; n < TN; ++n) {
    const std::uint32_t t = t0 + (wt * TN + n) * 16 + lane % 16;
    if (t >= tokens) continue;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t r = r0 + wr * 16 + 2 * j + lane / 16;
      if (r < rows) y[static_cast<std::size_t>(t) * rows + r] = acc[n][j];
    }
  }
}

// ---- P4: QSA block scores for a chunk on bf16 WMMA (hi/lo split) -----------
// halo_qsa_scores_rows for 4 indexer heads as one GEMM: rows are (token,
// head) pairs, columns key blocks, K = dim; score[t][b] = sum_h relu(q_th .
// k_b). Operands split into bf16 hi/lo as in halo_gemm_f32_bf16x3 (~2^-15
// relative per product), so the top-k selection sees near-fp32 scores. Each
// wave: 16 rows (4 tokens) x 64 blocks, fragments built from global memory
// (K = 128 needs no staging); the pooled keys of a layer (~1 MB) stay in L2.
// Scores are written for every block < cols; the selection reads only the
// first n_complete(t).
//   grid (ceil(cols / 64), ceil(tokens / 16)), block 128; heads == 4,
//   dim % 16 == 0.
extern "C" __global__ void __launch_bounds__(128)
    halo_qsa_scores_wmma(const float* q, const float* blocks, float* score, std::uint32_t tokens, std::uint32_t cols,
                         std::uint32_t dim, std::uint32_t score_stride) {
  constexpr std::uint32_t HEADS = 4, TN = 4;
  const std::uint32_t lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  const std::uint32_t r0 = (blockIdx.y * 4 + wave) * 16;  // 16 rows = 4 tokens x 4 heads
  const std::uint32_t c0 = blockIdx.x * 64;
  const std::uint32_t rows = tokens * HEADS;
  if (r0 >= rows) return;
  const float* qa = q + static_cast<std::size_t>(min(r0 + lane % 16, rows - 1)) * dim;
  const float* kb[TN];
#pragma unroll
  for (std::uint32_t n = 0; n < TN; ++n) kb[n] = blocks + static_cast<std::size_t>(min(c0 + n * 16 + lane % 16, cols - 1)) * dim;
  const auto frag = [](const float* p, halo_v16s& hi, halo_v16s& lo) {
#pragma unroll
    for (int e = 0; e < 16; e += 4) {
      const float4 v = *reinterpret_cast<const float4*>(p + e);
      const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        short h, l;
        SplitBf16(f[i], h, l);
        hi[e + i] = h;
        lo[e + i] = l;
      }
    }
  };
  halo_v8f acc[TN] = {};
  for (std::uint32_t k0 = 0; k0 < dim; k0 += 16) {
    halo_v16s ah, al;
    frag(qa + k0, ah, al);
#pragma unroll
    for (std::uint32_t n = 0; n < TN; ++n) {
      halo_v16s bh, bl;
      frag(kb[n] + k0, bh, bl);
      acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al, bh, acc[n]);
      acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bl, acc[n]);
      acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bh, acc[n]);
    }
  }
  // C layout: lane holds column lane % 16 and rows 2 j + lane / 16. Row
  // 4 tq + h: this lane holds heads lane / 16 and lane / 16 + 2 of token tq
  // (elements 2 tq, 2 tq + 1); the other half-wave the other two heads.
#pragma unroll
  for (std::uint32_t n = 0; n < TN; ++n) {
    const std::uint32_t b = c0 + n * 16 + lane % 16;
#pragma unroll
    for (std::uint32_t tq = 0; tq < 4; ++tq) {
      float part = fmaxf(0.0f, acc[n][2 * tq]) + fmaxf(0.0f, acc[n][2 * tq + 1]);
      part += __shfl_xor(part, 16, 32);
      const std::uint32_t t = r0 / HEADS + tq;
      if (lane < 16 && b < cols && t < tokens) score[static_cast<std::size_t>(t) * score_stride + b] = part;
    }
  }
}

// Lane exchange within aligned groups of 4 lanes by DPP quad permutes:
// QuadXor1 reads lane ^ 1, QuadXor2 lane ^ 2 (quad_perm [1,0,3,2], [2,3,0,1]).
__device__ inline float QuadXor1(float v) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, false));
}
__device__ inline float QuadXor2(float v) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, false));
}

// ---- P4: DeltaNet recurrence over a chunk, state split across lanes ----------
//
// halo_gdn_prefill_d128_norm's recurrence with each state column (value
// dim j) spread over 4 lanes of 32 keys: partial dots combine with two lane
// shuffles, so a thread does ~100 FMAs per token instead of ~384, and a head
// runs as 2 blocks of 256 threads (64 columns each) instead of one block of
// 128. Same arithmetic up to summation order. Writes the ungated output;
// halo_gated_norm over T * heads rows applies the gated RMSNorm.
//   q, k rows at t * qk_stride, v at t * v_stride, out at t * o_stride;
//   alpha, beta [T][heads]. grid (2, heads), block 256
extern "C" __global__ void __launch_bounds__(256)
    halo_gdn_prefill_split(const float* q, const float* k, const float* v, const float* alpha, const float* beta,
                           const float* a, const float* dt, float* state, float* out, std::uint32_t num_k_heads,
                           std::uint32_t tokens, std::uint32_t qk_stride, std::uint32_t v_stride,
                           std::uint32_t o_stride) {
  constexpr std::uint32_t d = 128, kStage = 8;
  __shared__ __attribute__((aligned(16))) float ks[kStage][d], qs[kStage][d];  // float4 reads
  __shared__ float vs[kStage][64], decay_s[kStage], beta_s[kStage], kq_s[kStage];
  const std::uint32_t h = blockIdx.y, heads = gridDim.y, tid = threadIdx.x;
  const std::uint32_t j = blockIdx.x * 64 + tid / 4, part = tid % 4, i0 = part * 32;
  float* S = state + static_cast<std::size_t>(h) * d * d + j;  // S[i][j] at S[i * d]
  float s[32];
#pragma unroll
  for (std::uint32_t i = 0; i < 32; ++i) s[i] = S[(i0 + i) * d];
  const float ah = a[h], dth = dt[h];
  const std::uint32_t kh = h % num_k_heads;
  // The next stage's inputs are loaded into registers while this stage
  // computes; each thread holds its share: k and q (4 + 4 of the stage's
  // 8 x 128), v (2 of 8 x 64) and, for tid < 8, alpha and beta.
  float rk[4], rq[4], rv[2], ra = 0.0f, rb = 0.0f;
  const auto load = [&](std::uint32_t t0, std::uint32_t n) {
#pragma unroll
    for (std::uint32_t p = 0; p < 4; ++p) {
      const std::uint32_t x = tid + p * 256, tt = x / d, c = x % d;
      const std::size_t row = static_cast<std::size_t>(t0 + min(tt, n - 1)) * qk_stride + kh * d + c;
      rk[p] = k[row];
      rq[p] = q[row];
    }
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p) {
      const std::uint32_t x = tid + p * 256, tt = x / 64, c = x % 64;
      rv[p] = v[static_cast<std::size_t>(t0 + min(tt, n - 1)) * v_stride + h * d + blockIdx.x * 64 + c];
    }
    if (tid < kStage) {
      ra = alpha[static_cast<std::size_t>(t0 + min(tid, n - 1)) * heads + h];
      rb = beta[static_cast<std::size_t>(t0 + min(tid, n - 1)) * heads + h];
    }
  };
  if (tokens > 0) load(0, min(kStage, tokens));
  for (std::uint32_t t0 = 0; t0 < tokens; t0 += kStage) {
    const std::uint32_t n = min(kStage, tokens - t0);
    __syncthreads();  // the previous stage's LDS reads are done
#pragma unroll
    for (std::uint32_t p = 0; p < 4; ++p) {
      const std::uint32_t x = tid + p * 256;
      ks[x / d][x % d] = rk[p];
      qs[x / d][x % d] = rq[p];
    }
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p) {
      const std::uint32_t x = tid + p * 256;
      vs[x / 64][x % 64] = rv[p];
    }
    if (tid < kStage) {
      decay_s[tid] = expf(ah * Softplus(ra + dth));
      beta_s[tid] = Sigmoid(rb);
    }
    __syncthreads();
    // k . q is the same for every column of the head: one wave per token
    // computes it once (it was 32 FMAs and two exchanges per token in every
    // thread).
    {
      const std::uint32_t tt = tid / 32, l = tid % 32;
      float p = 0.0f;
#pragma unroll
      for (std::uint32_t i = 0; i < d; i += 32) p += ks[tt][i + l] * qs[tt][i + l];
      p = WarpSum(p);
      if (l == 0) kq_s[tt] = p;
    }
    __syncthreads();
    if (t0 + kStage < tokens) load(t0 + kStage, min(kStage, tokens - t0 - kStage));  // in flight below
    for (std::uint32_t tt = 0; tt < n; ++tt) {
      const std::uint32_t t = t0 + tt;
      // This lane's 32 k and q values in registers, read as float4s once and
      // used by both the dot products and the state update. Left to the
      // compiler, ROCm clang 22 split these into narrow LDS reads and re-read
      // k for the update (553 vs 308 ms per prefill against clang 21).
      float kr[32], qr[32];
#pragma unroll
      for (std::uint32_t i = 0; i < 32; i += 4) {
        const float4 k4 = *reinterpret_cast<const float4*>(&ks[tt][i0 + i]);
        const float4 q4 = *reinterpret_cast<const float4*>(&qs[tt][i0 + i]);
        kr[i] = k4.x, kr[i + 1] = k4.y, kr[i + 2] = k4.z, kr[i + 3] = k4.w;
        qr[i] = q4.x, qr[i + 1] = q4.y, qr[i + 2] = q4.z, qr[i + 3] = q4.w;
      }
      float A = 0.0f, B = 0.0f;
#pragma unroll
      for (std::uint32_t i = 0; i < 32; ++i) {
        A += s[i] * kr[i];
        B += s[i] * qr[i];
      }
      // Sum over the 4 lanes of this column with DPP quad permutes (register
      // to register): __shfl_xor compiles to ds_bpermute here, an LDS round
      // trip on the serial per-token path (thread trace: 73% of the kernel's
      // time was LDS waits). Same additions in the same order as before.
      A += QuadXor1(A);
      B += QuadXor1(B);
      A += QuadXor2(A);
      B += QuadXor2(B);
      const float decay = decay_s[tt], b = beta_s[tt], kq = kq_s[tt];
      const float delta = (vs[tt][tid / 4] - decay * A) * b;
#pragma unroll
      for (std::uint32_t i = 0; i < 32; ++i) s[i] = decay * s[i] + delta * kr[i];
      if (part == 0)
        out[static_cast<std::size_t>(t) * o_stride + h * d + j] = (decay * B + delta * kq) / sqrtf(static_cast<float>(d));
    }
  }
#pragma unroll
  for (std::uint32_t i = 0; i < 32; ++i) S[(i0 + i) * d] = s[i];
}

// ---- P4: chunked gated DeltaNet ---------------------------------------------
//
// halo_gdn_prefill_split's recurrence per v-head (S key x value, decay
// g = exp(a softplus(alpha + dt)), b = sigmoid(beta), delta = b (v - g S^T k),
// S = g S + k delta^T, o = S^T q / sqrt d) over chunks of C tokens. With
// G_t the cumulative log decay within a chunk and S0 its starting state:
//   (I + A) Delta = diag(b) V - diag(b e^G) K S0,  A[t][s] = b_t (k_t.k_s) e^(G_t - G_s), s < t
//   o_t = (e^G_t q_t^T S0 + sum_{s<=t} (q_t.k_s) e^(G_t - G_s) delta_s) / sqrt d
//   S_C = e^G_C S0 + sum_s e^(G_C - G_s) k_s delta_s^T
// (verified exact against stepping in double precision). The prep kernel
// computes each chunk's A and masked QK in parallel; the scan walks the
// chunks of a head in order, once per chunk instead of once per token.
constexpr std::uint32_t kGdnChunk = 16;  // 16: ~23 KB LDS in the scan, all 96 blocks resident
// Per (chunk, v-head): G [C], b [C], A [C][C] (strictly lower), QK [C][C]
// (lower incl. diagonal, decay applied); prep layout [head][chunk][kGdnPrep].
constexpr std::uint32_t kGdnPrep = 2 * kGdnChunk + 2 * kGdnChunk * kGdnChunk;
//   grid (ceil(T / C), heads), block 256
extern "C" __global__ void __launch_bounds__(256)
    halo_gdn_chunk_prep(const float* q, const float* k, const float* alpha, const float* beta, const float* a,
                        const float* dt, float* prep, std::uint32_t num_k_heads, std::uint32_t tokens,
                        std::uint32_t qk_stride) {
  constexpr std::uint32_t d = 128, C = kGdnChunk;
  __shared__ __attribute__((aligned(16))) float ks[C][d + 4], qs[C][d + 4];
  __shared__ float G[C], bs[C];
  const std::uint32_t c = blockIdx.x, h = blockIdx.y, heads = gridDim.y, tid = threadIdx.x;
  const std::uint32_t t0 = c * C, n = min(C, tokens - t0), kh = h % num_k_heads;
  for (std::uint32_t x = tid; x < C * d / 4; x += blockDim.x) {
    const std::uint32_t tt = x / (d / 4), i = (x % (d / 4)) * 4;
    const std::size_t row = static_cast<std::size_t>(t0 + min(tt, n - 1)) * qk_stride + kh * d + i;
    *reinterpret_cast<float4*>(&ks[tt][i]) = *reinterpret_cast<const float4*>(k + row);
    *reinterpret_cast<float4*>(&qs[tt][i]) = *reinterpret_cast<const float4*>(q + row);
  }
  if (tid < 32) {
    const std::uint32_t tt = tid;
    const float lg = tt < n && tt < C ? a[h] * Softplus(alpha[static_cast<std::size_t>(t0 + tt) * heads + h] + dt[h]) : 0.0f;
    float g = lg;  // inclusive prefix sum over the chunk's 32 lanes
    for (int o = 1; o < 32; o <<= 1) {
      const float up = __shfl_up(g, o, 32);
      if (tt >= static_cast<std::uint32_t>(o)) g += up;
    }
    if (tt < C) {
      G[tt] = g;
      bs[tt] = tt < n ? Sigmoid(beta[static_cast<std::size_t>(t0 + tt) * heads + h]) : 0.0f;
    }
  }
  __syncthreads();
  float* out = prep + (static_cast<std::size_t>(h) * gridDim.x + c) * kGdnPrep;
  if (tid < C) {
    out[tid] = G[tid];
    out[C + tid] = bs[tid];
  }
  float* A = out + 2 * C;
  float* QK = A + C * C;
  for (std::uint32_t x = tid; x < 2 * C * C; x += blockDim.x) {
    const std::uint32_t which = x / (C * C), t = (x % (C * C)) / C, sidx = x % C;
    const bool live = t < n && sidx < n && (which == 0 ? sidx < t : sidx <= t);
    float v = 0.0f;
    if (live) {
      const float* a0 = which == 0 ? ks[t] : qs[t];
      float acc = 0.0f;
#pragma unroll 8
      for (std::uint32_t i = 0; i < d; i += 4) {
        const float4 x4 = *reinterpret_cast<const float4*>(a0 + i), y4 = *reinterpret_cast<const float4*>(&ks[sidx][i]);
        acc += x4.x * y4.x + x4.y * y4.y + x4.z * y4.z + x4.w * y4.w;
      }
      v = acc * expf(G[t] - G[sidx]) * (which == 0 ? bs[t] : 1.0f);
    }
    (which == 0 ? A : QK)[t * C + sidx] = v;
  }
}

// The chunk scan: state columns spread over 4 lanes of 32 keys as in
// halo_gdn_prefill_split (grid (2, heads), block 256: 64 columns a block).
// Per chunk, every token's S0^T k and S0^T q are independent dot products;
// the deltas follow by forward substitution; the state updates once.
extern "C" __global__ void __launch_bounds__(256)
    halo_gdn_chunk_scan(const float* q, const float* k, const float* v, const float* prep, float* state, float* out,
                        std::uint32_t num_k_heads, std::uint32_t tokens, std::uint32_t qk_stride, std::uint32_t v_stride,
                        std::uint32_t o_stride) {
  constexpr std::uint32_t d = 128, C = kGdnChunk;
  __shared__ __attribute__((aligned(16))) float ks[C][d + 4], qs[C][d + 4];
  __shared__ float vs[C][64];
  __shared__ float P[kGdnPrep];
  const std::uint32_t h = blockIdx.y, tid = threadIdx.x;
  const std::uint32_t j = blockIdx.x * 64 + tid / 4, part = tid % 4, i0 = part * 32;
  const std::uint32_t kh = h % num_k_heads, nchunks = (tokens + C - 1) / C;
  float* S = state + static_cast<std::size_t>(h) * d * d + j;  // S[i][j] at S[i * d]
  float s[32];
#pragma unroll
  for (std::uint32_t i = 0; i < 32; ++i) s[i] = S[(i0 + i) * d];
  const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(d));
  for (std::uint32_t c = 0; c < nchunks; ++c) {
    const std::uint32_t t0 = c * C, n = min(C, tokens - t0);
    __syncthreads();  // the previous chunk's LDS reads are done
    for (std::uint32_t x = tid; x < C * d / 4; x += blockDim.x) {
      const std::uint32_t tt = x / (d / 4), i = (x % (d / 4)) * 4;
      const std::size_t row = static_cast<std::size_t>(t0 + min(tt, n - 1)) * qk_stride + kh * d + i;
      *reinterpret_cast<float4*>(&ks[tt][i]) = *reinterpret_cast<const float4*>(k + row);
      *reinterpret_cast<float4*>(&qs[tt][i]) = *reinterpret_cast<const float4*>(q + row);
    }
    for (std::uint32_t x = tid; x < C * 64; x += blockDim.x) {
      const std::uint32_t tt = x / 64, cc = x % 64;
      vs[tt][cc] = v[static_cast<std::size_t>(t0 + min(tt, n - 1)) * v_stride + h * d + blockIdx.x * 64 + cc];
    }
    const float* pp = prep + (static_cast<std::size_t>(h) * nchunks + c) * kGdnPrep;
    for (std::uint32_t x = tid; x < kGdnPrep; x += blockDim.x) P[x] = pp[x];
    __syncthreads();
    const float* G = P;
    const float* bs = P + C;
    const float* A = P + 2 * C;
    const float* QK = A + C * C;
    // S0^T k_t and S0^T q_t for every token: partial over this lane's keys,
    // then summed over the column's 4 lanes.
    float kd[C], qd[C];
#pragma unroll
    for (std::uint32_t t = 0; t < C; ++t) {
      float pk = 0.0f, pq = 0.0f;
#pragma unroll
      for (std::uint32_t i = 0; i < 32; i += 4) {
        const float4 k4 = *reinterpret_cast<const float4*>(&ks[t][i0 + i]);
        const float4 q4 = *reinterpret_cast<const float4*>(&qs[t][i0 + i]);
        pk += s[i] * k4.x + s[i + 1] * k4.y + s[i + 2] * k4.z + s[i + 3] * k4.w;
        pq += s[i] * q4.x + s[i + 1] * q4.y + s[i + 2] * q4.z + s[i + 3] * q4.w;
      }
      pk += QuadXor1(pk);
      pq += QuadXor1(pq);
      pk += QuadXor2(pk);
      pq += QuadXor2(pq);
      kd[t] = pk;
      qd[t] = pq;
    }
    // Deltas by forward substitution (every lane of the column alike).
    float dl[C];
#pragma unroll
    for (std::uint32_t t = 0; t < C; ++t) {
      float r = bs[t] * (vs[t][tid / 4] - expf(G[t]) * kd[t]);
#pragma unroll
      for (std::uint32_t u = 0; u < t; ++u) r -= A[t * C + u] * dl[u];
      dl[t] = t < n ? r : 0.0f;
    }
    // Outputs.
    if (part == 0) {
#pragma unroll
      for (std::uint32_t t = 0; t < C; ++t) {
        if (t >= n) break;
        float o = expf(G[t]) * qd[t];
#pragma unroll
        for (std::uint32_t u = 0; u <= t; ++u) o += QK[t * C + u] * dl[u];
        out[static_cast<std::size_t>(t0 + t) * o_stride + h * d + j] = o * inv_sqrt_d;
      }
    }
    // State: S = e^G_n S0 + sum_t e^(G_n - G_t) k_t delta_t^T.
    const float gn = G[n - 1];
    float w[C];
#pragma unroll
    for (std::uint32_t t = 0; t < C; ++t) w[t] = t < n ? expf(gn - G[t]) * dl[t] : 0.0f;
    const float en = expf(gn);
#pragma unroll
    for (std::uint32_t i = 0; i < 32; ++i) {
      float acc = en * s[i];
#pragma unroll
      for (std::uint32_t t = 0; t < C; ++t) acc += ks[t][i0 + i] * w[t];
      s[i] = acc;
    }
  }
#pragma unroll
  for (std::uint32_t i = 0; i < 32; ++i) S[(i0 + i) * d] = s[i];
}

// ---- P4: chunked gated DeltaNet on WMMA ---------------------------------------
// The chunked form of halo_gdn_chunk_scan (see there) with every product on
// bf16 WMMA, operands split hi/lo (three WMMAs per product, ~2^-15 relative):
// per 16-token chunk K S0 and Q S0 (16 x 128 . 128 x cols), A = K K^T and
// Q K^T (16 x 16), Delta = T RHS with T = (I + A)^-1, O = e^G Q S0 + QK Delta,
// S += Kw^T Delta. The state stays in fp32 accumulator tiles (keys x the
// wave's 16 value columns) and doubles as the B operand of K S0 / Q S0: in
// the 16 x 16 accumulator layout lane l holds column l % 16, rows 2 j + l / 16,
// so one exchange with lane l ^ 16 gives the 16 rows a B fragment needs.
//   grid (4, heads), block 128: 4 waves = 2 column groups x 2 key halves, 32 columns.
namespace {
__device__ inline void SplitFrag16(const float (&f)[16], halo_v16s& hi, halo_v16s& lo) {
#pragma unroll
  for (int e = 0; e < 16; ++e) {
    short h, l;
    SplitBf16(f[e], h, l);
    hi[e] = h;
    lo[e] = l;
  }
}
__device__ inline halo_v8f Wmma3(const halo_v16s& ah, const halo_v16s& al, const halo_v16s& bh, const halo_v16s& bl,
                                 halo_v8f c) {
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al, bh, c);
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bl, c);
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bh, c);
}
// Accumulator layout -> the 16 K-values of this lane's column (B operand),
// split straight into bf16 hi/lo (no fp32 staging array: register pressure).
__device__ inline void AccToBSplit(const halo_v8f& c, halo_v16s& hi, halo_v16s& lo) {
  const bool upper = (threadIdx.x % 32) >= 16;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float own = c[j], other = __shfl_xor(c[j], 16, 32);
    short h0, l0, h1, l1;
    SplitBf16(upper ? other : own, h0, l0);
    SplitBf16(upper ? own : other, h1, l1);
    hi[2 * j] = h0;
    lo[2 * j] = l0;
    hi[2 * j + 1] = h1;
    lo[2 * j + 1] = l1;
  }
}
// 16 consecutive floats (16-byte aligned) -> bf16 hi/lo.
__device__ inline void RowSplit(const float* p, halo_v16s& hi, halo_v16s& lo) {
#pragma unroll
  for (int e = 0; e < 16; e += 4) {
    const float4 x = *reinterpret_cast<const float4*>(p + e);
    const float f[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      short h, l;
      SplitBf16(f[i], h, l);
      hi[e + i] = h;
      lo[e + i] = l;
    }
  }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(128)
    halo_gdn_chunk_wmma(const float* q, const float* k, const float* v, const float* alpha, const float* beta,
                        const float* a, const float* dt, float* state, float* out, std::uint32_t num_k_heads,
                        std::uint32_t tokens, std::uint32_t qk_stride, std::uint32_t v_stride, std::uint32_t o_stride) {
  constexpr std::uint32_t d = 128, C = 16, KR = d + 4, NC = 32;  // value columns per block
  __shared__ __attribute__((aligned(16))) float ks[C][KR], qs[C][KR];
  __shared__ float vs[C][NC];
  __shared__ float part[4][2][8][32];  // per wave: its K S0 and Q S0 partials (accumulator layout)
  __shared__ float Tm[C][C + 1], QKm[C][C + 1], Am[C][C + 1];
  __shared__ float G[C], bs[C];
  const std::uint32_t h = blockIdx.y, heads = gridDim.y, tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  // Wave = (column group cg, key half kh2): columns cg * 16 .. + 15 of the
  // block's 32, keys kh2 * 64 .. + 63 of the state (4 tiles of 16 keys).
  const std::uint32_t cg = wave % 2, kh2 = wave / 2;
  const std::uint32_t kh = h % num_k_heads, c0 = blockIdx.x * NC + cg * 16;  // this wave's first column
  const std::uint32_t col = c0 + lane % 16, hi = lane / 16;
  float* S = state + static_cast<std::size_t>(h) * d * d;
  // State tiles: acc[kt] holds keys (kh2 * 4 + kt) * 16 + 2 j + hi of column `col`.
  halo_v8f acc[4];
#pragma unroll
  for (int kt = 0; kt < 4; ++kt)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[kt][j] = S[((kh2 * 4 + kt) * 16 + 2 * j + hi) * d + col];
  const float ah = a[h], dth = dt[h], inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(d));
  const std::uint32_t nchunks = (tokens + C - 1) / C;
  for (std::uint32_t ch = 0; ch < nchunks; ++ch) {
    const std::uint32_t t0 = ch * C, n = min(C, tokens - t0);
    __syncthreads();  // the previous chunk's LDS reads are done
    for (std::uint32_t x = tid; x < C * d / 4; x += blockDim.x) {
      const std::uint32_t tt = x / (d / 4), i = (x % (d / 4)) * 4;
      float4 kk = make_float4(0.0f, 0.0f, 0.0f, 0.0f), qq = kk;
      if (tt < n) {
        const std::size_t row = static_cast<std::size_t>(t0 + tt) * qk_stride + kh * d + i;
        kk = *reinterpret_cast<const float4*>(k + row);
        qq = *reinterpret_cast<const float4*>(q + row);
      }
      *reinterpret_cast<float4*>(&ks[tt][i]) = kk;
      *reinterpret_cast<float4*>(&qs[tt][i]) = qq;
    }
    for (std::uint32_t x = tid; x < C * NC; x += blockDim.x) {
      const std::uint32_t tt = x / NC, cc = x % NC;
      vs[tt][cc] = tt < n ? v[static_cast<std::size_t>(t0 + tt) * v_stride + h * d + blockIdx.x * NC + cc] : 0.0f;
    }
    if (tid < 32) {  // cumulative log decays and gates; padding tokens: no decay, no gate
      const float lg = tid < n ? ah * Softplus(alpha[static_cast<std::size_t>(t0 + tid) * heads + h] + dth) : 0.0f;
      float gsum = lg;
      for (int o = 1; o < 32; o <<= 1) {
        const float up = __shfl_up(gsum, o, 32);
        if (tid >= static_cast<std::uint32_t>(o)) gsum += up;
      }
      if (tid < C) {
        G[tid] = gsum;
        bs[tid] = tid < n ? Sigmoid(beta[static_cast<std::size_t>(t0 + tid) * heads + h]) : 0.0f;
      }
    }
    __syncthreads();
    // Waves 0 and 1: K K^T and Q K^T (16 x 16, K = 128) for A and the masked QK.
    if (wave < 2) {
      const float(&L)[C][KR] = wave == 0 ? ks : qs;
      halo_v8f m = {};
      for (std::uint32_t k0 = 0; k0 < d; k0 += 16) {
        halo_v16s ahh, all, bhh, bll;
        RowSplit(&L[lane % 16][k0], ahh, all);
        RowSplit(&ks[lane % 16][k0], bhh, bll);
        m = Wmma3(ahh, all, bhh, bll, m);
      }
      // m[j]: row t = 2 j + hi, column s = lane % 16.
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t t = 2 * j + hi, sx = lane % 16;
        const float dec = expf(G[t] - G[sx]);
        if (wave == 0)
          Am[t][sx] = sx < t ? bs[t] * m[j] * dec : 0.0f;
        else
          QKm[t][sx] = sx <= t ? m[j] * dec : 0.0f;
      }
    }
    // Every wave: its key half's part of K S0 and Q S0 for its 16 columns.
    {
      halo_v8f pk = {}, pq = {};
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) {
        const std::uint32_t key0 = (kh2 * 4 + kt) * 16;
        halo_v16s bhh, bll, xh, xl;
        AccToBSplit(acc[kt], bhh, bll);  // S0 rows key0 .. + 15 of this column
        RowSplit(&ks[lane % 16][key0], xh, xl);
        pk = Wmma3(xh, xl, bhh, bll, pk);
        RowSplit(&qs[lane % 16][key0], xh, xl);
        pq = Wmma3(xh, xl, bhh, bll, pq);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        part[wave][0][j][lane] = pk[j];
        part[wave][1][j][lane] = pq[j];
      }
    }
    __syncthreads();  // Am, QKm and the partials written
    // Full sums, low key half first (the same order in both waves of a group).
    halo_v8f kd, qd;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      kd[j] = part[cg][0][j][lane] + part[cg + 2][0][j][lane];
      qd[j] = part[cg][1][j][lane] + part[cg + 2][1][j][lane];
    }
    if (tid < C) {  // T = (I + A)^-1 by forward substitution; lane = column
      const std::uint32_t sx = tid;
      float tcol[C];
#pragma unroll
      for (std::uint32_t t = 0; t < C; ++t) {
        float r = t == sx ? 1.0f : 0.0f;
#pragma unroll
        for (std::uint32_t u = 0; u < t; ++u) r -= Am[t][u] * tcol[u];
        tcol[t] = r;
      }
#pragma unroll
      for (std::uint32_t t = 0; t < C; ++t) Tm[t][sx] = tcol[t];
    }
    __syncthreads();
    // RHS = b (V - e^G (K S0)) in accumulator layout: rows tokens 2 j + hi.
    halo_v8f rhs;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t t = 2 * j + hi;
      rhs[j] = bs[t] * (vs[t][cg * 16 + lane % 16] - expf(G[t]) * kd[j]);
    }
    // Delta = T RHS: A operand T (row t = lane % 16), B operand RHS.
    halo_v16s rhh, rll, thh, tll;
    AccToBSplit(rhs, rhh, rll);
    {
      float ft[16];
#pragma unroll
      for (int e = 0; e < 16; ++e) ft[e] = Tm[lane % 16][e];
      SplitFrag16(ft, thh, tll);
    }
    halo_v8f dl = {};
    dl = Wmma3(thh, tll, rhh, rll, dl);
    halo_v16s dhh, dll;
    AccToBSplit(dl, dhh, dll);  // Delta's 16 tokens of this column
    // Outputs: e^G (Q S0) + QK Delta.
    halo_v16s qkh, qkl;
    {
      float fqk[16];
#pragma unroll
      for (int e = 0; e < 16; ++e) fqk[e] = QKm[lane % 16][e];
      SplitFrag16(fqk, qkh, qkl);
    }
    halo_v8f o = {};
    o = Wmma3(qkh, qkl, dhh, dll, o);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t t = 2 * j + hi;
      if (kh2 == 0 && t < n)
        out[static_cast<std::size_t>(t0 + t) * o_stride + h * d + col] = (expf(G[t]) * qd[j] + o[j]) * inv_sqrt_d;
    }
    // State: S = e^G_n S0 + Kw^T Delta, Kw[t][i] = e^(G_n - G_t) k_t[i].
    const float gn = G[n - 1], en = expf(gn);
    float wt[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) wt[e] = static_cast<std::uint32_t>(e) < n ? expf(gn - G[e]) : 0.0f;
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const std::uint32_t key0 = (kh2 * 4 + kt) * 16;
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[kt][j] *= en;
      halo_v16s ahh, all;  // A operand: key key0 + lane % 16, the chunk's 16 tokens
#pragma unroll
      for (int e = 0; e < 16; ++e) {
        short hh, ll;
        SplitBf16(ks[e][key0 + lane % 16] * wt[e], hh, ll);
        ahh[e] = hh;
        all[e] = ll;
      }
      acc[kt] = Wmma3(ahh, all, dhh, dll, acc[kt]);
    }
  }
#pragma unroll
  for (int kt = 0; kt < 4; ++kt)
#pragma unroll
    for (int j = 0; j < 8; ++j) S[((kh2 * 4 + kt) * 16 + 2 * j + hi) * d + col] = acc[kt][j];
}

// ---- P4: chunked gated DeltaNet, prep + WMMA scan ----------------------------
// halo_gdn_chunk_wmma's math with everything that does not depend on the
// state moved to a prep kernel that runs over all chunks at once: per (v-head,
// 16-token chunk) T' = (I + A)^-1 diag(b), P' = masked decayed Q K^T / sqrt d,
// e^G / sqrt d, e^G, the update weights e^(G_n - G_s) and e^G_n. The scan then
// is, per chunk, Delta = T' (V - e^G (K S0)), O = e^G/sqrt d (Q S0) + P' Delta,
// S = e^G_n S + Kw^T Delta: bf16 hi/lo WMMAs only, no serial solve. A wave
// owns 16 value columns over all 128 keys (state: 8 accumulator tiles), so
// the waves share nothing but the staged chunk; 4 waves a block, 2 blocks a
// head, ~19 KB of LDS: every block of a layer (96) is resident at once.
// Prep layout [head][chunk][kGdnP16]: T' [16][16], P' [16][16], e^G/sqrt d [16],
// e^G [16], weights [16], e^G_n, zero padding.
namespace {
typedef std::uint32_t halo_v8u __attribute__((ext_vector_type(8)));
// Two floats -> packed bf16 hi pair and lo pair (element 2 i in the low half).
__device__ inline void SplitPair(float a, float b, std::uint32_t& hi, std::uint32_t& lo) {
  const auto rne = [](float f) {
    const std::uint32_t u = __float_as_uint(f);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
  };
  const std::uint32_t ha = rne(a), hb = rne(b);
  hi = (ha & 0xFFFFu) | (hb << 16);
  lo = (rne(a - __uint_as_float(ha << 16)) & 0xFFFFu) | (rne(b - __uint_as_float(hb << 16)) << 16);
}
__device__ inline halo_v16s AsFrag(const halo_v8u& x) { return __builtin_bit_cast(halo_v16s, x); }
// RowSplit and AccToBSplit with the halves packed as they are made.
__device__ inline void RowSplitP(const float* p, halo_v16s& hi, halo_v16s& lo) {
  halo_v8u h, l;
#pragma unroll
  for (int e = 0; e < 16; e += 4) {
    const float4 x = *reinterpret_cast<const float4*>(p + e);
    std::uint32_t a, b;
    SplitPair(x.x, x.y, a, b);
    h[e / 2] = a;
    l[e / 2] = b;
    SplitPair(x.z, x.w, a, b);
    h[e / 2 + 1] = a;
    l[e / 2 + 1] = b;
  }
  hi = AsFrag(h);
  lo = AsFrag(l);
}
__device__ inline void AccToBSplitP(const halo_v8f& c, halo_v16s& hi, halo_v16s& lo) {
  const bool upper = (threadIdx.x % 32) >= 16;
  halo_v8u h, l;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float own = c[j], other = __shfl_xor(c[j], 16, 32);
    std::uint32_t a, b;
    SplitPair(upper ? other : own, upper ? own : other, a, b);
    h[j] = a;
    l[j] = b;
  }
  hi = AsFrag(h);
  lo = AsFrag(l);
}
// Three-way split (hi + mid + lo, ~24 bits): for products whose result
// cancels, e.g. V - S^T k once the state predicts the values well.
__device__ inline void SplitPair3(float a, float b, std::uint32_t& hi, std::uint32_t& mi, std::uint32_t& lo) {
  const auto rne = [](float f) {
    const std::uint32_t u = __float_as_uint(f);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
  };
  const std::uint32_t ha = rne(a), hb = rne(b);
  const float ra = a - __uint_as_float(ha << 16), rb = b - __uint_as_float(hb << 16);
  const std::uint32_t ma = rne(ra), mb = rne(rb);
  hi = (ha & 0xFFFFu) | (hb << 16);
  mi = (ma & 0xFFFFu) | (mb << 16);
  lo = (rne(ra - __uint_as_float(ma << 16)) & 0xFFFFu) | (rne(rb - __uint_as_float(mb << 16)) << 16);
}
__device__ inline void RowSplit3P(const float* p, halo_v16s& hi, halo_v16s& mi, halo_v16s& lo) {
  halo_v8u h, m, l;
#pragma unroll
  for (int e = 0; e < 16; e += 2) {
    std::uint32_t a, b, c;
    SplitPair3(p[e], p[e + 1], a, b, c);
    h[e / 2] = a;
    m[e / 2] = b;
    l[e / 2] = c;
  }
  hi = AsFrag(h);
  mi = AsFrag(m);
  lo = AsFrag(l);
}
__device__ inline void AccToBSplit3P(const halo_v8f& c, halo_v16s& hi, halo_v16s& mi, halo_v16s& lo) {
  const bool upper = (threadIdx.x % 32) >= 16;
  halo_v8u h, m, l;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float own = c[j], other = __shfl_xor(c[j], 16, 32);
    std::uint32_t a, b, x;
    SplitPair3(upper ? other : own, upper ? own : other, a, b, x);
    h[j] = a;
    m[j] = b;
    l[j] = x;
  }
  hi = AsFrag(h);
  mi = AsFrag(m);
  lo = AsFrag(l);
}
// The six terms down to 2^-16 x 2^-16 relative, smallest first.
__device__ inline halo_v8f Wmma6(const halo_v16s& ah, const halo_v16s& am, const halo_v16s& al, const halo_v16s& bh,
                                 const halo_v16s& bm, const halo_v16s& bl, halo_v8f c) {
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al, bh, c);
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bl, c);
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(am, bm, c);
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(am, bh, c);
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bm, c);
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bh, c);
}
}  // namespace
constexpr std::uint32_t kGdnP16 = 576;
// One block per (k-head, chunk) serves the k-head's v-heads kh + r nk
// (r < heads / nk <= 8): K and Q are loaded and K K^T, Q K^T formed once,
// then the per-head gates, decays and T solves run side by side.
//   grid (ceil(T / 16), num_k_heads), block 256
extern "C" __global__ void __launch_bounds__(256)
    halo_gdn_c16_prep(const float* q, const float* k, const float* alpha, const float* beta, const float* a,
                      const float* dt, float* prep, std::uint32_t heads, std::uint32_t tokens,
                      std::uint32_t qk_stride) {
  constexpr std::uint32_t d = 128, C = 16, RM = 8, NT = 256;
  __shared__ __attribute__((aligned(16))) float ks[C][d + 4], qs[C][d + 4];
  __shared__ float KK[C][C + 1], QK[C][C + 1], G[RM][C], bs[RM][C], Am[RM][C][C + 1];
  const std::uint32_t c = blockIdx.x, kh = blockIdx.y, nk = gridDim.y, nchunks = gridDim.x, tid = threadIdx.x;
  const std::uint32_t R = heads / nk, t0 = c * C, n = min(C, tokens - t0), lane = tid % 32, wave = tid / 32;
  const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(d));
  {  // every load in flight before the first store
    float4 rk[2], rq[2];
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p) {
      const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
      const std::size_t row = static_cast<std::size_t>(t0 + min(tt, n - 1)) * qk_stride + kh * d + i;
      rk[p] = *reinterpret_cast<const float4*>(k + row);
      rq[p] = *reinterpret_cast<const float4*>(q + row);
    }
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p) {
      const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
      *reinterpret_cast<float4*>(&ks[tt][i]) = rk[p];
      *reinterpret_cast<float4*>(&qs[tt][i]) = rq[p];
    }
  }
  if (wave < R) {  // head kh + wave nk: cumulative log decays and gates; padding: no decay, no gate
    const std::uint32_t h = kh + wave * nk;
    const float lg =
        lane < n ? a[h] * Softplus(alpha[static_cast<std::size_t>(t0 + lane) * heads + h] + dt[h]) : 0.0f;
    float g = lg;
    for (int o = 1; o < 32; o <<= 1) {
      const float up = __shfl_up(g, o, 32);
      if (lane >= static_cast<std::uint32_t>(o)) g += up;
    }
    if (lane < C) {
      G[wave][lane] = g;
      bs[wave][lane] = lane < n ? Sigmoid(beta[static_cast<std::size_t>(t0 + lane) * heads + h]) : 0.0f;
    }
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t p = 0; p < 2; ++p) {  // the 512 raw dot products, 2 a thread
    const std::uint32_t x = tid + p * NT, which = x / (C * C), t = (x % (C * C)) / C, sx = x % C;
    const float* a0 = which == 0 ? ks[t] : qs[t];
    float acc = 0.0f;
#pragma unroll 8
    for (std::uint32_t i = 0; i < d; i += 4) {
      const float4 x4 = *reinterpret_cast<const float4*>(a0 + i), y4 = *reinterpret_cast<const float4*>(&ks[sx][i]);
      acc += x4.x * y4.x + x4.y * y4.y + x4.z * y4.z + x4.w * y4.w;
    }
    (which == 0 ? KK : QK)[t][sx] = acc;
  }
  __syncthreads();
  const auto out_of = [&](std::uint32_t r) { return prep + (static_cast<std::size_t>(kh + r * nk) * nchunks + c) * kGdnP16; };
  for (std::uint32_t x = tid; x < R * 2 * C * C; x += NT) {
    const std::uint32_t r = x / (2 * C * C), which = (x / (C * C)) % 2, t = (x % (C * C)) / C, sx = x % C;
    const bool live = t < n && sx < n && (which == 0 ? sx < t : sx <= t);
    const float v = live ? (which == 0 ? KK : QK)[t][sx] * expf(G[r][t] - G[r][sx]) : 0.0f;
    if (which == 0)
      Am[r][t][sx] = v * bs[r][t];
    else
      out_of(r)[C * C + t * C + sx] = v * inv_sqrt_d;
  }
  for (std::uint32_t x = tid; x < R * C; x += NT) {
    const std::uint32_t r = x / C, e = x % C;
    float* out = out_of(r);
    const float eg = expf(G[r][e]);
    out[2 * C * C + e] = eg * inv_sqrt_d;
    out[2 * C * C + C + e] = eg;
    out[2 * C * C + 2 * C + e] = e < n ? expf(G[r][C - 1] - G[r][e]) : 0.0f;  // G[C - 1] = G[n - 1]
    out[2 * C * C + 3 * C + e] = e == 0 ? expf(G[r][C - 1]) : 0.0f;
  }
  __syncthreads();
  for (std::uint32_t x = tid; x < R * C; x += NT) {  // column sx of T = (I + A)^-1, scaled by b_sx
    const std::uint32_t r = x / C, sx = x % C;
    float tcol[C];
#pragma unroll
    for (std::uint32_t t = 0; t < C; ++t) {
      float v = t == sx ? 1.0f : 0.0f;
#pragma unroll
      for (std::uint32_t u = 0; u < t; ++u) v -= Am[r][t][u] * tcol[u];
      tcol[t] = v;
    }
    float* out = out_of(r);
#pragma unroll
    for (std::uint32_t t = 0; t < C; ++t) out[t * C + sx] = tcol[t] * bs[r][sx];
  }
}

// halo_gdn_c16_prep over cpb consecutive chunks a block, the next chunk's
// K, Q and gates prefetched into registers behind LDS-only barriers (a trace
// of the one-chunk version: 55% of cycles at barriers, 22% waiting on loads).
//   grid (ceil(ceil(T / 16) / cpb), num_k_heads), block 256
extern "C" __global__ void __launch_bounds__(256)
    halo_gdn_c16_prep2(const float* q, const float* k, const float* alpha, const float* beta, const float* a,
                       const float* dt, float* prep, std::uint32_t heads, std::uint32_t tokens, std::uint32_t qk_stride,
                       std::uint32_t cpb) {
  constexpr std::uint32_t d = 128, C = 16, RM = 8, NT = 256;
  __shared__ __attribute__((aligned(16))) float ks[C][d + 4], qs[C][d + 4];
  __shared__ float KK[C][C + 1], QK[C][C + 1], G[RM][C], bs[RM][C], Am[RM][C][C + 1];
  const std::uint32_t kh = blockIdx.y, nk = gridDim.y, tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t R = heads / nk, nchunks = (tokens + C - 1) / C, c_begin = blockIdx.x * cpb,
                      c_end = min(c_begin + cpb, nchunks);
  if (c_begin >= nchunks) return;
  const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(d));
  const auto barrier = [] {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
  };
  const std::uint32_t h_mine = kh + min(wave, R - 1) * nk;  // the head whose gates this wave loads
  const float ah = a[h_mine], dth = dt[h_mine];
  float4 rk[2], rq[2];
  float ral, rbe;
  const auto load = [&](std::uint32_t c) {
    const std::uint32_t t0 = c * C, n = min(C, tokens - t0);
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p) {
      const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
      const std::uint32_t row = (t0 + min(tt, n - 1)) * qk_stride + kh * d + i;
      rk[p] = *reinterpret_cast<const float4*>(k + row);
      rq[p] = *reinterpret_cast<const float4*>(q + row);
    }
    const std::uint32_t t = t0 + min(lane, n - 1);
    ral = alpha[t * heads + h_mine];
    rbe = beta[t * heads + h_mine];
  };
  load(c_begin);
  for (std::uint32_t c = c_begin; c < c_end; ++c) {
    const std::uint32_t t0 = c * C, n = min(C, tokens - t0);
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p) {
      const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
      *reinterpret_cast<float4*>(&ks[tt][i]) = rk[p];
      *reinterpret_cast<float4*>(&qs[tt][i]) = rq[p];
    }
    const float al = ral, be = rbe;
    load(min(c + 1, c_end - 1));
    if (wave < R) {  // cumulative log decays and gates; padding: no decay, no gate
      float g = lane < n ? ah * Softplus(al + dth) : 0.0f;
      for (int o = 1; o < 32; o <<= 1) {
        const float up = __shfl_up(g, o, 32);
        if (lane >= static_cast<std::uint32_t>(o)) g += up;
      }
      if (lane < C) {
        G[wave][lane] = g;
        bs[wave][lane] = lane < n ? Sigmoid(be) : 0.0f;
      }
    }
    barrier();
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p) {
      const std::uint32_t x = tid + p * NT, which = x / (C * C), t = (x % (C * C)) / C, sx = x % C;
      const float* a0 = which == 0 ? ks[t] : qs[t];
      float acc = 0.0f;
#pragma unroll 8
      for (std::uint32_t i = 0; i < d; i += 4) {
        const float4 x4 = *reinterpret_cast<const float4*>(a0 + i), y4 = *reinterpret_cast<const float4*>(&ks[sx][i]);
        acc += x4.x * y4.x + x4.y * y4.y + x4.z * y4.z + x4.w * y4.w;
      }
      (which == 0 ? KK : QK)[t][sx] = acc;
    }
    barrier();
    const auto out_of = [&](std::uint32_t r) {
      return prep + (static_cast<std::size_t>(kh + r * nk) * nchunks + c) * kGdnP16;
    };
    for (std::uint32_t x = tid; x < R * 2 * C * C; x += NT) {
      const std::uint32_t r = x / (2 * C * C), which = (x / (C * C)) % 2, t = (x % (C * C)) / C, sx = x % C;
      const bool live = t < n && sx < n && (which == 0 ? sx < t : sx <= t);
      const float v = live ? (which == 0 ? KK : QK)[t][sx] * expf(G[r][t] - G[r][sx]) : 0.0f;
      if (which == 0)
        Am[r][t][sx] = v * bs[r][t];
      else
        out_of(r)[C * C + t * C + sx] = v * inv_sqrt_d;
    }
    for (std::uint32_t x = tid; x < R * C; x += NT) {
      const std::uint32_t r = x / C, e = x % C;
      float* out = out_of(r);
      const float eg = expf(G[r][e]);
      out[2 * C * C + e] = eg * inv_sqrt_d;
      out[2 * C * C + C + e] = eg;
      out[2 * C * C + 2 * C + e] = e < n ? expf(G[r][C - 1] - G[r][e]) : 0.0f;
      out[2 * C * C + 3 * C + e] = e == 0 ? expf(G[r][C - 1]) : 0.0f;
    }
    barrier();
    for (std::uint32_t x = tid; x < R * C; x += NT) {  // column sx of T = (I + A)^-1, scaled by b_sx
      const std::uint32_t r = x / C, sx = x % C;
      float tcol[C];
#pragma unroll
      for (std::uint32_t t = 0; t < C; ++t) {
        float v = t == sx ? 1.0f : 0.0f;
#pragma unroll
        for (std::uint32_t u = 0; u < t; ++u) v -= Am[r][t][u] * tcol[u];
        tcol[t] = v;
      }
      float* out = out_of(r);
#pragma unroll
      for (std::uint32_t t = 0; t < C; ++t) out[t * C + sx] = tcol[t] * bs[r][sx];
    }
    barrier();  // LDS reads done before the next chunk is staged
  }
}

// KS3: K S0 with the three-way split (6 WMMAs a tile).
template <bool KS3>
__device__ void GdnC16Scan(const float* q, const float* k, const float* v, const float* prep, float* state, float* out,
                           std::uint32_t num_k_heads, std::uint32_t tokens, std::uint32_t qk_stride,
                           std::uint32_t v_stride, std::uint32_t o_stride) {
  constexpr std::uint32_t d = 128, C = 16, KR = d + 4, NT = 128, KP = C * d / 4 / NT, PP = kGdnP16 / 4;
  __shared__ __attribute__((aligned(16))) float ks[C][KR], qs[C][KR], P[kGdnP16];
  const std::uint32_t h = blockIdx.y, tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t kh = h % num_k_heads, col = blockIdx.x * 64 + wave * 16 + lane % 16, hi = lane / 16;
  const std::uint32_t nchunks = (tokens + C - 1) / C;
  float* S = state + static_cast<std::size_t>(h) * d * d;
  // State tiles: acc[kt] holds keys kt * 16 + 2 j + hi of column `col`:
  // S row (kt * 16 + 2 j) (uniform) at this lane's offset so.
  std::uint32_t so = hi * d + col;
  halo_v8f acc[8];
#pragma unroll
  for (int kt = 0; kt < 8; ++kt)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[kt][j] = (S + (kt * 16 + 2 * j) * d)[so];
  // Only LDS is shared: LDS-only barriers keep the next chunk's loads (into
  // registers, unconditional: clamped to the last chunk) in flight.
  const auto barrier = [] {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
  };
  // Offsets in 32 bits (< 2^32 elements): a uniform base plus one VGPR per
  // load, not a 64-bit address per load kept live across the loop.
  // Vectors, not arrays: arrays carried around the loop stayed in scratch.
  typedef float v16f __attribute__((ext_vector_type(16)));
  static_assert(KP == 4, "rk/rq hold 4 float4");
  v16f rk, rq;
  halo_v8f rp, rv;
  const auto load_v = [&](std::uint32_t ch) __attribute__((always_inline)) {
    const std::uint32_t t0 = ch * C, n = min(C, tokens - t0);
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j)
      rv[j] = v[(t0 + min(2 * j + hi, n - 1)) * v_stride + h * d + col];
  };
  const auto load = [&](std::uint32_t ch) __attribute__((always_inline)) {
    const std::uint32_t t0 = ch * C, n = min(C, tokens - t0);
#pragma unroll
    for (std::uint32_t p = 0; p < KP; ++p) {
      const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
      const std::uint32_t row = (t0 + min(tt, n - 1)) * qk_stride + kh * d + i;
      const float4 k4 = *reinterpret_cast<const float4*>(k + row), q4 = *reinterpret_cast<const float4*>(q + row);
      rk[4 * p] = k4.x, rk[4 * p + 1] = k4.y, rk[4 * p + 2] = k4.z, rk[4 * p + 3] = k4.w;
      rq[4 * p] = q4.x, rq[4 * p + 1] = q4.y, rq[4 * p + 2] = q4.z, rq[4 * p + 3] = q4.w;
    }
    const float4* pp = reinterpret_cast<const float4*>(prep);
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p) {
      const float4 x = pp[(h * nchunks + ch) * PP + min(tid + p * NT, PP - 1)];
      rp[4 * p] = x.x, rp[4 * p + 1] = x.y, rp[4 * p + 2] = x.z, rp[4 * p + 3] = x.w;
    }
  };
  load(0);
  load_v(0);
  for (std::uint32_t ch = 0; ch < nchunks; ++ch) {
    const std::uint32_t t0 = ch * C, n = min(C, tokens - t0);
#pragma unroll
    for (std::uint32_t p = 0; p < KP; ++p) {
      const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
      *reinterpret_cast<float4*>(&ks[tt][i]) = make_float4(rk[4 * p], rk[4 * p + 1], rk[4 * p + 2], rk[4 * p + 3]);
      *reinterpret_cast<float4*>(&qs[tt][i]) = make_float4(rq[4 * p], rq[4 * p + 1], rq[4 * p + 2], rq[4 * p + 3]);
    }
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p)
      if (tid + p * NT < PP)
        reinterpret_cast<float4*>(P)[tid + p * NT] = make_float4(rp[4 * p], rp[4 * p + 1], rp[4 * p + 2], rp[4 * p + 3]);
    const std::uint32_t nx = min(ch + 1, nchunks - 1);
    load(nx);
    barrier();
    // K S0 and Q S0 for the chunk's 16 tokens and this wave's 16 columns.
    halo_v8f pk = {}, pq = {};
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      halo_v16s bhh, bll, xh, xl;
      if constexpr (KS3) {
        halo_v16s bmm, xm;
        AccToBSplit3P(acc[kt], bhh, bmm, bll);
        RowSplit3P(&ks[lane % 16][kt * 16], xh, xm, xl);
        pk = Wmma6(xh, xm, xl, bhh, bmm, bll, pk);
        // Q S0 keeps hi/lo: the pair's lo, rne(x - hi), is the triple's mid.
        bll = bmm;
      } else {
        AccToBSplitP(acc[kt], bhh, bll);
        RowSplitP(&ks[lane % 16][kt * 16], xh, xl);
        pk = Wmma3(xh, xl, bhh, bll, pk);
      }
      RowSplitP(&qs[lane % 16][kt * 16], xh, xl);
      pq = Wmma3(xh, xl, bhh, bll, pq);
      __builtin_amdgcn_sched_barrier(0);
    }
    // Delta = T' (V - e^G (K S0)); accumulator rows are tokens 2 j + hi.
    halo_v8f r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = rv[j] - P[2 * C * C + C + 2 * j + hi] * pk[j];
    load_v(nx);  // V is used once, above: its next loads fly during the rest of the chunk
    halo_v16s rh, rl, th, tl;
    AccToBSplitP(r, rh, rl);
    RowSplitP(&P[(lane % 16) * C], th, tl);
    halo_v8f dl = {};
    dl = Wmma3(th, tl, rh, rl, dl);
    halo_v16s dh, dlo;
    AccToBSplitP(dl, dh, dlo);
    // O = e^G / sqrt d (Q S0) + P' Delta.
    RowSplitP(&P[C * C + (lane % 16) * C], th, tl);
    halo_v8f o = {};
    o = Wmma3(th, tl, dh, dlo, o);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t t = 2 * j + hi;
      if (t < n) out[static_cast<std::size_t>(t0 + t) * o_stride + h * d + col] = P[2 * C * C + t] * pq[j] + o[j];
    }
    // S = e^G_n S0 + Kw^T Delta.
    const float en = P[2 * C * C + 3 * C];
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[kt][j] *= en;
      halo_v8u ah, al;  // A operand: key kt * 16 + lane % 16 over the chunk's tokens
#pragma unroll
      for (int e = 0; e < 16; e += 2) {
        std::uint32_t hh, ll;
        SplitPair(ks[e][kt * 16 + lane % 16] * P[2 * C * C + 2 * C + e],
                  ks[e + 1][kt * 16 + lane % 16] * P[2 * C * C + 2 * C + e + 1], hh, ll);
        ah[e / 2] = hh;
        al[e / 2] = ll;
      }
      acc[kt] = Wmma3(AsFrag(ah), AsFrag(al), dh, dlo, acc[kt]);
      __builtin_amdgcn_sched_barrier(0);
    }
    barrier();  // the chunk's LDS reads are done before the next is staged
  }
  // Recompute the store addresses (an opaque offset): reusing the loads'
  // 64 addresses kept them live across the loop and spilled.
  asm volatile("" : "+v"(so));
#pragma unroll
  for (int kt = 0; kt < 8; ++kt)
#pragma unroll
    for (int j = 0; j < 8; ++j) (S + (kt * 16 + 2 * j) * d)[so] = acc[kt][j];
}

#define HALO_GDN_C16_SCAN(NAME, KS3)                                                                             \
  extern "C" __global__ void __launch_bounds__(128)                                                              \
      NAME(const float* q, const float* k, const float* v, const float* prep, float* state, float* out,          \
           std::uint32_t num_k_heads, std::uint32_t tokens, std::uint32_t qk_stride, std::uint32_t v_stride,      \
           std::uint32_t o_stride) {                                                                             \
    GdnC16Scan<KS3>(q, k, v, prep, state, out, num_k_heads, tokens, qk_stride, v_stride, o_stride);              \
  }
//   grid (2, heads), block 128
HALO_GDN_C16_SCAN(halo_gdn_c16_scan, false)
HALO_GDN_C16_SCAN(halo_gdn_c16_scan_ks3, true)
#undef HALO_GDN_C16_SCAN

// halo_gdn_c16_scan with K S0 on the three-way split (a two-way split there
// shifted long-prompt NLL by +0.002: V - S^T k cancels once the state
// predicts the values) and the splitting done once a block, at staging:
// K is staged as bf16 hi / mid / lo and Q as hi / lo, not fp32 split again
// by every wave. The update folds the decay weights into Delta,
// S = e^G_n S0 + K^T (w Delta), so its A operand is K's hi / mid gathered
// as they are. ~24 KB of LDS: every block of a layer still resident.
// U3: the update's K three-way too (hi / mid / lo against w Delta's hi / lo).
template <bool U3>
__device__ void GdnC16Scan3(const float* q, const float* k, const float* v, const float* prep, float* state,
                            float* out, std::uint32_t num_k_heads, std::uint32_t tokens, std::uint32_t qk_stride,
                            std::uint32_t v_stride, std::uint32_t o_stride) {
  constexpr std::uint32_t d = 128, C = 16, KR = d + 8, NT = 128, KP = C * d / 4 / NT, PP = kGdnP16 / 4;
  __shared__ __attribute__((aligned(16))) short kh[C][KR], km[C][KR], kl[C][KR], qh[C][KR], ql[C][KR];
  __shared__ __attribute__((aligned(16))) float P[kGdnP16];
  const std::uint32_t h = blockIdx.y, tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t kh_ = h % num_k_heads, col = blockIdx.x * 64 + wave * 16 + lane % 16, hi = lane / 16;
  const std::uint32_t nchunks = (tokens + C - 1) / C;
  float* S = state + static_cast<std::size_t>(h) * d * d;
  std::uint32_t so = hi * d + col;
  halo_v8f acc[8];
#pragma unroll
  for (int kt = 0; kt < 8; ++kt)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[kt][j] = (S + (kt * 16 + 2 * j) * d)[so];
  const auto barrier = [] {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
  };
  typedef float v16f __attribute__((ext_vector_type(16)));
  static_assert(KP == 4, "rk/rq hold 4 float4");
  v16f rk, rq;
  halo_v8f rp, rv;
  const auto load_v = [&](std::uint32_t ch) {
    const std::uint32_t t0 = ch * C, n = min(C, tokens - t0);
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j) rv[j] = v[(t0 + min(2 * j + hi, n - 1)) * v_stride + h * d + col];
  };
  const auto load = [&](std::uint32_t ch) {
    const std::uint32_t t0 = ch * C, n = min(C, tokens - t0);
#pragma unroll
    for (std::uint32_t p = 0; p < KP; ++p) {
      const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
      const std::uint32_t row = (t0 + min(tt, n - 1)) * qk_stride + kh_ * d + i;
      const float4 k4 = *reinterpret_cast<const float4*>(k + row), q4 = *reinterpret_cast<const float4*>(q + row);
      rk[4 * p] = k4.x, rk[4 * p + 1] = k4.y, rk[4 * p + 2] = k4.z, rk[4 * p + 3] = k4.w;
      rq[4 * p] = q4.x, rq[4 * p + 1] = q4.y, rq[4 * p + 2] = q4.z, rq[4 * p + 3] = q4.w;
    }
    const float4* pp = reinterpret_cast<const float4*>(prep);
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p) {
      const float4 x = pp[(h * nchunks + ch) * PP + min(tid + p * NT, PP - 1)];
      rp[4 * p] = x.x, rp[4 * p + 1] = x.y, rp[4 * p + 2] = x.z, rp[4 * p + 3] = x.w;
    }
  };
  load(0);
  load_v(0);
  for (std::uint32_t ch = 0; ch < nchunks; ++ch) {
    const std::uint32_t t0 = ch * C, n = min(C, tokens - t0);
#pragma unroll
    for (std::uint32_t p = 0; p < KP; ++p) {  // split once, for every wave
      const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
      uint2 a, b, c;
      SplitPair3(rk[4 * p], rk[4 * p + 1], a.x, b.x, c.x);
      SplitPair3(rk[4 * p + 2], rk[4 * p + 3], a.y, b.y, c.y);
      *reinterpret_cast<uint2*>(&kh[tt][i]) = a;
      *reinterpret_cast<uint2*>(&km[tt][i]) = b;
      *reinterpret_cast<uint2*>(&kl[tt][i]) = c;
      SplitPair(rq[4 * p], rq[4 * p + 1], a.x, b.x);
      SplitPair(rq[4 * p + 2], rq[4 * p + 3], a.y, b.y);
      *reinterpret_cast<uint2*>(&qh[tt][i]) = a;
      *reinterpret_cast<uint2*>(&ql[tt][i]) = b;
    }
#pragma unroll
    for (std::uint32_t p = 0; p < 2; ++p)
      if (tid + p * NT < PP)
        reinterpret_cast<float4*>(P)[tid + p * NT] = make_float4(rp[4 * p], rp[4 * p + 1], rp[4 * p + 2], rp[4 * p + 3]);
    const std::uint32_t nx = min(ch + 1, nchunks - 1);
    load(nx);
    barrier();
    // K S0 (three-way) and Q S0 (two-way: the pair's lo is the triple's mid).
    halo_v8f pk = {}, pq = {};
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      halo_v16s bh, bm, bl;
      AccToBSplit3P(acc[kt], bh, bm, bl);
      pk = Wmma6(LdsFragS(&kh[lane % 16][kt * 16]), LdsFragS(&km[lane % 16][kt * 16]),
                 LdsFragS(&kl[lane % 16][kt * 16]), bh, bm, bl, pk);
      pq = Wmma3(LdsFragS(&qh[lane % 16][kt * 16]), LdsFragS(&ql[lane % 16][kt * 16]), bh, bm, pq);
      __builtin_amdgcn_sched_barrier(0);
    }
    // Delta = T' (V - e^G (K S0)); accumulator rows are tokens 2 j + hi.
    halo_v8f r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = rv[j] - P[2 * C * C + C + 2 * j + hi] * pk[j];
    load_v(nx);
    halo_v16s rh, rl, th, tl;
    AccToBSplitP(r, rh, rl);
    RowSplitP(&P[(lane % 16) * C], th, tl);
    halo_v8f dl = {};
    dl = Wmma3(th, tl, rh, rl, dl);
    halo_v16s dh, dlo;
    AccToBSplitP(dl, dh, dlo);
    // O = e^G / sqrt d (Q S0) + P' Delta.
    RowSplitP(&P[C * C + (lane % 16) * C], th, tl);
    halo_v8f o = {};
    o = Wmma3(th, tl, dh, dlo, o);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t t = 2 * j + hi;
      if (t < n) out[static_cast<std::size_t>(t0 + t) * o_stride + h * d + col] = P[2 * C * C + t] * pq[j] + o[j];
    }
    // S = e^G_n S0 + K^T (w Delta): B operand w Delta, A operand K^T (hi, mid).
#pragma unroll
    for (int j = 0; j < 8; ++j) dl[j] *= P[2 * C * C + 2 * C + 2 * j + hi];
    AccToBSplitP(dl, dh, dlo);
    const float en = P[2 * C * C + 3 * C];
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[kt][j] *= en;
      halo_v16s ah, am;  // key kt * 16 + lane % 16 over the chunk's tokens
#pragma unroll
      for (int e = 0; e < 16; ++e) {
        ah[e] = kh[e][kt * 16 + lane % 16];
        am[e] = km[e][kt * 16 + lane % 16];
      }
      if constexpr (U3) {
        halo_v16s al;
#pragma unroll
        for (int e = 0; e < 16; ++e) al[e] = kl[e][kt * 16 + lane % 16];
        acc[kt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al, dh, acc[kt]);
        acc[kt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(am, dlo, acc[kt]);
      }
      acc[kt] = Wmma3(ah, am, dh, dlo, acc[kt]);
      __builtin_amdgcn_sched_barrier(0);
    }
    barrier();
  }
  asm volatile("" : "+v"(so));
#pragma unroll
  for (int kt = 0; kt < 8; ++kt)
#pragma unroll
    for (int j = 0; j < 8; ++j) (S + (kt * 16 + 2 * j) * d)[so] = acc[kt][j];
}

#define HALO_GDN_C16_SCAN3(NAME, U3)                                                                             \
  extern "C" __global__ void __launch_bounds__(128)                                                              \
      NAME(const float* q, const float* k, const float* v, const float* prep, float* state, float* out,          \
           std::uint32_t num_k_heads, std::uint32_t tokens, std::uint32_t qk_stride, std::uint32_t v_stride,      \
           std::uint32_t o_stride) {                                                                             \
    GdnC16Scan3<U3>(q, k, v, prep, state, out, num_k_heads, tokens, qk_stride, v_stride, o_stride);              \
  }
//   grid (2, heads), block 128
HALO_GDN_C16_SCAN3(halo_gdn_c16_scan3, false)
HALO_GDN_C16_SCAN3(halo_gdn_c16_scan3u, true)
#undef HALO_GDN_C16_SCAN3

// ---- Study: structural floor of a chunked DeltaNet scan ---------------------
// The loads, LDS staging, barriers and state traffic of a chunked scan with
// no math: per chunk of C tokens, K and Q (C x 128 fp32) and the block's NC
// value columns of V into LDS, the block's 128 x NC fp32 state slice read
// and written once in LDS, NB barriers. PF prefetches the next chunk's K/Q/V
// into registers during the chunk. A checksum keeps the work live.
//   grid (128 / NC, heads), block NT
template <std::uint32_t C, std::uint32_t NC, std::uint32_t NT, std::uint32_t NB, bool PF, bool SEQ, bool ST = true>
__device__ void GdnSkeleton(const float* q, const float* k, const float* v, const float* alpha, float* state,
                            float* out, std::uint32_t num_k_heads, std::uint32_t tokens, std::uint32_t qk_stride,
                            std::uint32_t v_stride) {
  constexpr std::uint32_t d = 128, KR = d + 4, KP = C * d / 4 / NT, VP = (C * NC + NT - 1) / NT;
  static_assert(C * d / 4 % NT == 0, "whole float4 per thread");
  // SEQ: Q is staged into K's buffer once K is used (two more barriers).
  __shared__ __attribute__((aligned(16))) float ks[C][KR], qx[SEQ ? 1 : C][KR];
  float(*qs)[KR] = SEQ ? ks : qx;
  __shared__ float vs[C * NC];
  __shared__ float st[d * NC];
  __shared__ float gs[C];
  const std::uint32_t h = blockIdx.y, heads = gridDim.y, tid = threadIdx.x, kh = h % num_k_heads;
  float* S = state + static_cast<std::size_t>(h) * d * d + blockIdx.x * NC;
  for (std::uint32_t x = tid; x < d * NC; x += NT) st[x] = S[(x / NC) * d + x % NC];
  float4 rk[KP], rq[KP];
  float rv[VP];
  const auto load = [&](std::uint32_t t0) {
    const std::uint32_t n = min(C, tokens - t0);
#pragma unroll
    for (std::uint32_t p = 0; p < KP; ++p) {
      const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
      const std::size_t row = static_cast<std::size_t>(t0 + min(tt, n - 1)) * qk_stride + kh * d + i;
      rk[p] = *reinterpret_cast<const float4*>(k + row);
      rq[p] = *reinterpret_cast<const float4*>(q + row);
    }
#pragma unroll
    for (std::uint32_t p = 0; p < VP; ++p) {
      const std::uint32_t x = min(tid + p * NT, C * NC - 1), tt = x / NC, cc = x % NC;
      rv[p] = v[static_cast<std::size_t>(t0 + min(tt, n - 1)) * v_stride + h * d + blockIdx.x * NC + cc];
    }
  };
  float sum = 0.0f;
  load(0);
  for (std::uint32_t t0 = 0; t0 < tokens; t0 += C) {
    __syncthreads();
#pragma unroll
    for (std::uint32_t p = 0; p < KP; ++p) {
      const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
      *reinterpret_cast<float4*>(&ks[tt][i]) = rk[p];
      if constexpr (!SEQ) *reinterpret_cast<float4*>(&qs[tt][i]) = rq[p];
    }
#pragma unroll
    for (std::uint32_t p = 0; p < VP; ++p)
      if (tid + p * NT < C * NC) vs[tid + p * NT] = rv[p];
    if (tid < C) gs[tid] = alpha[static_cast<std::size_t>(min(t0 + tid, tokens - 1)) * heads + h];
    if constexpr (PF) {
      if (t0 + C < tokens) load(t0 + C);
    }
    __syncthreads();
    // The state slice read and written once (the update), and a touch of
    // the staged chunk so nothing is dead.
    if constexpr (ST)
      for (std::uint32_t x = tid; x < d * NC; x += NT) st[x] = st[x] * 0.999f + ks[x % C][x / NC] * vs[x % (C * NC)];
    else
      sum += vs[tid % (C * NC)];
    if constexpr (SEQ) {
      __syncthreads();
#pragma unroll
      for (std::uint32_t p = 0; p < KP; ++p) {
        const std::uint32_t x = tid + p * NT, tt = x / (d / 4), i = (x % (d / 4)) * 4;
        *reinterpret_cast<float4*>(&qs[tt][i]) = rq[p];
      }
      __syncthreads();
    }
    sum += qs[tid % C][tid % d] + gs[tid % C];
#pragma unroll
    for (std::uint32_t b = 2; b < NB; ++b) __syncthreads();
    if constexpr (!PF) {
      if (t0 + C < tokens) load(t0 + C);
    }
  }
  __syncthreads();
  for (std::uint32_t x = tid; x < d * NC; x += NT) S[(x / NC) * d + x % NC] = st[x];
  if (sum == 12345.678f) out[0] = sum;
}
#define HALO_GDN_SKEL(NAME, C, NC, NT, NB, PF, SEQ, ...)                                                                   \
  extern "C" __global__ void __launch_bounds__(NT)                                                               \
      NAME(const float* q, const float* k, const float* v, const float* alpha, float* state, float* out,         \
           std::uint32_t num_k_heads, std::uint32_t tokens, std::uint32_t qk_stride, std::uint32_t v_stride) {   \
    GdnSkeleton<C, NC, NT, NB, PF, SEQ __VA_OPT__(,) __VA_ARGS__>(q, k, v, alpha, state, out, num_k_heads, tokens, qk_stride, v_stride);       \
  }
HALO_GDN_SKEL(halo_gdnskel_c16_n32, 16, 32, 128, 4, false, false)
HALO_GDN_SKEL(halo_gdnskel_c16_n32_pf, 16, 32, 128, 4, true, false)
HALO_GDN_SKEL(halo_gdnskel_c16_n64_pf, 16, 64, 256, 4, true, false)
HALO_GDN_SKEL(halo_gdnskel_c32_n32, 32, 32, 128, 4, false, false)
HALO_GDN_SKEL(halo_gdnskel_c32_n32_pf, 32, 32, 128, 4, true, false)
HALO_GDN_SKEL(halo_gdnskel_c32_n64_pf_seq, 32, 64, 256, 4, true, true)
HALO_GDN_SKEL(halo_gdnskel_c16_n64_pf_nost, 16, 64, 256, 4, true, false, false)
HALO_GDN_SKEL(halo_gdnskel_c32_n64_pf_seq_nost, 32, 64, 256, 4, true, true, false)
HALO_GDN_SKEL(halo_gdnskel_c64_n32_seq, 64, 32, 256, 4, false, true)
HALO_GDN_SKEL(halo_gdnskel_c64_n32_pf_seq, 64, 32, 256, 4, true, true)
#undef HALO_GDN_SKEL

// ---- P4: dense GEMM over repacked Q8_0 on FP16 WMMA ---------------------------
//
// y[t][r] = sum_c W[r][c] x[t][c]: W repacked Q8_0 (int8 rows qstride apart,
// fp16 scales per 32), x fp16 rows. Weights are dequantized to fp16 while
// staging into LDS (signed byte ^ 0x80 ORed into fp16 1024: one exact
// subtraction and one multiply by the scale), so the WMMAs need no scale
// epilogue; products accumulate in fp32. A K stage is 64 values (two Q8
// blocks); the next stage's raw operands are loaded into registers while
// the current one is multiplied out of LDS. Each wave owns TM x TN 16 x 16
// tiles; results leave through LDS so a wave's stores run along rows.
//   grid (ceil(T / BN), ceil(rows / BM)), block 256; cols % KS == 0
//   (KS 64 for the large tile, 128 for the small one: long-K shapes such
//   as the mixer's 320 x 10240 pay per stage).
// kMix (the mixer's up projection, rows permuted so row 4 i + s is stream s
// of hidden index i; hc = 4): instead of y, the epilogue writes
// mixed[t][i] = sum_s xn[t][s h + i] * sigmoid(gate) / 4, in fp32 and fp16.
// With rs and nw, xn is the residual and xn[t][s h + i] is rebuilt as
// (res * rs[t * 4 + s]) * nw[s h + i] (combine_norm's arithmetic).
template <std::uint32_t BM, std::uint32_t BN, std::uint32_t WM, std::uint32_t WN, std::uint32_t KS, bool kMix = false>
__device__ void GemmQ8rF16(const std::int8_t* qs, const std::uint16_t* d, const _Float16* x, float* y,
                           std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride, std::uint32_t tokens,
                           const float* xn = nullptr, _Float16* y16 = nullptr, const float* rs = nullptr,
                           const float* nw = nullptr) {
  constexpr std::uint32_t RS = KS + 8, NT = 256;
  constexpr std::uint32_t TM = BM / WM / 16, TN = BN / WN / 16;
  constexpr std::uint32_t APR = KS / 16, BPR = KS / 8;  // 16-byte pieces per row: int8 weights, fp16 tokens
  constexpr std::uint32_t AP = BM * APR / NT, BP = BN * BPR / NT;  // per thread
  static_assert(WM * WN == 8 && AP >= 1 && BP >= 1, "8 waves; every thread stages A and B");
  static_assert(BM * APR % NT == 0 && BN * BPR % NT == 0,
                "the stage splits evenly over the threads (else rows go unstaged: <320, 64, .., 32> faulted)");
  // One buffer: the A and B stages, later the epilogue's per-wave slices.
  constexpr std::uint32_t kStage = (BM + BN) * RS, kSlices = 8 * TM * 16 * 16 * 2;
  __shared__ __attribute__((aligned(16))) _Float16 smem[kStage > kSlices ? kStage : kSlices];
  _Float16* As = smem;
  _Float16* Bs = smem + BM * RS;
  const std::uint32_t tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t r0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
  const std::uint32_t wr = wave % WM, wt = wave / WM;
  const std::uint32_t nblk = cols / 32;
  // Native vectors, not HIP's int4/uint4: those are structs, and LLVM's
  // promote-alloca leaves arrays of them in scratch (here rb, carried across
  // the K loop), which made the prefetch synchronous.
  halo_v4i ra[AP];
  std::uint16_t rd[AP];
  halo_v4i rb[BP];
  const auto load = [&](std::uint32_t k0) {
#pragma unroll
    for (std::uint32_t p = 0; p < AP; ++p) {
      const std::uint32_t i = tid + p * NT, row = i / APR, part = i % APR;  // part: 16 values
      const std::size_t r = min(r0 + row, rows - 1);
      ra[p] = *reinterpret_cast<const halo_v4i*>(qs + r * qstride + k0 + part * 16);
      rd[p] = d[r * nblk + (k0 + part * 16) / 32];
    }
#pragma unroll
    for (std::uint32_t p = 0; p < BP; ++p) {
      const std::uint32_t i = tid + p * NT, tok = i / BPR, part = i % BPR;  // part: 8 values
      rb[p] = *reinterpret_cast<const halo_v4i*>(x + static_cast<std::size_t>(min(t0 + tok, tokens - 1)) * cols + k0 + part * 8);
    }
  };
  const auto store = [&] {
#pragma unroll
    for (std::uint32_t p = 0; p < AP; ++p) {
      const std::uint32_t i = tid + p * NT, row = i / APR, part = i % APR;
      const std::uint32_t w[4] = {static_cast<std::uint32_t>(ra[p][0]), static_cast<std::uint32_t>(ra[p][1]),
                                  static_cast<std::uint32_t>(ra[p][2]), static_cast<std::uint32_t>(ra[p][3])};
      const _Float16 sc = AsH2(rd[p]).x;
      const halo_h2 s2 = {sc, sc}, bias = {1152, 1152};
      _Float16* dst = As + row * RS + part * 16;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const std::uint32_t u = w[j] ^ 0x80808080u;
        const halo_h2 p0 = (AsH2((u & 0x00FF00FFu) | 0x64006400u) - bias) * s2;         // bytes 0, 2
        const halo_h2 p1 = (AsH2(((u >> 8) & 0x00FF00FFu) | 0x64006400u) - bias) * s2;  // bytes 1, 3
        dst[4 * j] = p0.x;
        dst[4 * j + 1] = p1.x;
        dst[4 * j + 2] = p0.y;
        dst[4 * j + 3] = p1.y;
      }
    }
#pragma unroll
    for (std::uint32_t p = 0; p < BP; ++p) {
      const std::uint32_t i = tid + p * NT, tok = i / BPR, part = i % BPR;
      *reinterpret_cast<halo_v4i*>(Bs + tok * RS + part * 8) = rb[p];
    }
  };
  halo_v8f acc[TM][TN] = {};
  load(0);
  for (std::uint32_t k0 = 0; k0 < cols; k0 += KS) {
    __syncthreads();  // the previous stage's fragment reads are done
    store();
    __syncthreads();
    if (k0 + KS < cols) load(k0 + KS);  // in flight during the math below
#pragma unroll
    for (std::uint32_t kk = 0; kk < KS; kk += 16) {
      halo_v16h a[TM];
#pragma unroll
      for (std::uint32_t m = 0; m < TM; ++m) a[m] = LdsFrag(&As[((wr * TM + m) * 16 + lane % 16) * RS + kk]);
#pragma unroll
      for (std::uint32_t n = 0; n < TN; ++n) {
        const halo_v16h b = LdsFrag(&Bs[((wt * TN + n) * 16 + lane % 16) * RS + kk]);
#pragma unroll
        for (std::uint32_t m = 0; m < TM; ++m) acc[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[m], b, acc[m][n]);
      }
    }
  }
  // Epilogue: per 16-token tile, the wave's TM * 16 rows go through its own
  // LDS slice token-major, then lanes store consecutive rows of one token.
  __syncthreads();
  float* slice = reinterpret_cast<float*>(smem) + wave * (TM * 16 * 16);
#pragma unroll
  for (std::uint32_t n = 0; n < TN; ++n) {
#pragma unroll
    for (std::uint32_t m = 0; m < TM; ++m)
#pragma unroll
      for (int j = 0; j < 8; ++j) slice[(lane % 16) * (TM * 16) + m * 16 + 2 * j + lane / 16] = acc[m][n][j];
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    if constexpr (kMix) {
      const std::uint32_t h = rows / 4;
      for (std::uint32_t idx = lane; idx < 16 * TM * 4; idx += 32) {
        const std::uint32_t tt = idx / (TM * 4), ii = idx % (TM * 4);
        const std::uint32_t t = t0 + (wt * TN + n) * 16 + tt, i = (r0 + wr * TM * 16) / 4 + ii;
        if (t >= tokens || i >= h) continue;
        const float* xr = xn + static_cast<std::size_t>(t) * rows;
        const float* g = slice + tt * (TM * 16) + ii * 4;
        float acc = 0.0f;
#pragma unroll
        for (std::uint32_t st = 0; st < 4; ++st) {
          // With rs, xn is rebuilt from the residual as combine_norm computed it.
          const float xv = rs ? xr[st * h + i] * rs[t * 4 + st] * nw[st * h + i] : xr[st * h + i];
          acc += xv * Sigmoid(g[st]);
        }
        const float m = acc / 4.0f;
        y[static_cast<std::size_t>(t) * h + i] = m;
        y16[static_cast<std::size_t>(t) * h + i] = static_cast<_Float16>(m);
      }
    } else {
      for (std::uint32_t idx = lane; idx < 16 * TM * 16; idx += 32) {
        const std::uint32_t tt = idx / (TM * 16), rr = idx % (TM * 16);
        const std::uint32_t t = t0 + (wt * TN + n) * 16 + tt, r = r0 + wr * TM * 16 + rr;
        if (t < tokens && r < rows) y[static_cast<std::size_t>(t) * rows + r] = slice[tt * (TM * 16) + rr];
      }
    }
    __builtin_amdgcn_wave_barrier();
  }
}

extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_q8r_f16_l(const std::int8_t* qs, const std::uint16_t* d, const _Float16* x, float* y, std::uint32_t rows,
                        std::uint32_t cols, std::uint32_t qstride, std::uint32_t tokens) {
  GemmQ8rF16<256, 128, 8, 1, 64>(qs, d, x, y, rows, cols, qstride, tokens);
}
extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_q8r_f16_s(const std::int8_t* qs, const std::uint16_t* d, const _Float16* x, float* y, std::uint32_t rows,
                        std::uint32_t cols, std::uint32_t qstride, std::uint32_t tokens) {
  GemmQ8rF16<64, 128, 2, 4, 128>(qs, d, x, y, rows, cols, qstride, tokens);
}

// The mixer's up projection fused with the stream mix (see kMix): rows =
// 4 h, permuted by halo_permute_hc_rows; y [T][h] fp32, y16 fp16.
extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_q8r_f16_mix(const std::int8_t* qs, const std::uint16_t* d, const _Float16* x, float* y,
                          std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride, std::uint32_t tokens,
                          const float* xn, _Float16* y16, const float* rs, const float* nw) {
  GemmQ8rF16<256, 128, 8, 1, 32, true>(qs, d, x, y, rows, cols, qstride, tokens, xn, y16, rs, nw);
}

// Repacked Q8_0 rows reordered stream-major -> index-major: destination row
// 4 i + s is source row s h + i (hc = 4). grid (rows), block 64
extern "C" __global__ void halo_permute_hc_rows(const std::int8_t* qs, const std::uint16_t* d, std::int8_t* pqs,
                                                std::uint16_t* pd, std::uint32_t h, std::uint32_t qstride,
                                                std::uint32_t nblk) {
  const std::uint32_t dst = blockIdx.x, i = dst / 4, st = dst % 4, src = st * h + i;
  for (std::uint32_t c = threadIdx.x; c < qstride; c += blockDim.x)
    pqs[static_cast<std::size_t>(dst) * qstride + c] = qs[static_cast<std::size_t>(src) * qstride + c];
  for (std::uint32_t b = threadIdx.x; b < nblk; b += blockDim.x)
    pd[static_cast<std::size_t>(dst) * nblk + b] = d[static_cast<std::size_t>(src) * nblk + b];
}

// Rows <= 320 (the mixer's down projection, 320 x 10240): one block covers
// every row, so the long fp16 activation rows are read once instead of once
// per 64-row block; the 3.3 MB of weights are re-read per token tile but
// stay in the last-level cache.
extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_q8r_f16_m(const std::int8_t* qs, const std::uint16_t* d, const _Float16* x, float* y, std::uint32_t rows,
                        std::uint32_t cols, std::uint32_t qstride, std::uint32_t tokens) {
  GemmQ8rF16<320, 64, 4, 2, 64>(qs, d, x, y, rows, cols, qstride, tokens);
}

// fp32 -> fp16 rows (n values). grid (ceil(n / 256))
extern "C" __global__ void halo_f16_rows(const float* x, _Float16* y, std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = static_cast<_Float16>(x[i]);
}

// ---- P4: chunk attention, several query tokens per block ----------------------
//
// halo_attention_wmma_rows for QT = 4 consecutive tokens at once, when each
// of them still sees every earlier position (dense causal: QSA has selected
// nothing away). The block's rows are (token, head) pairs of one KV head:
// QT * G <= 48 rows in up to 3 tiles of 16. Every K fragment and staged V
// chunk serves all QT tokens, so K/V traffic falls by QT; keys past a
// token's own position are masked. 32 keys per chunk: each of 8 waves
// scores one (row tile, key tile) of the up to 3 x 2, then adds P . V for
// its row tiles and 32 of the 256 output dims.
//   grid (num_kv_heads, ceil(T / 4)), block 256; head_dim 256, G <= 12.
extern "C" __global__ void __launch_bounds__(256)
    halo_attention_wmma_mq(const float* q, const float* kc, const float* vc, const float* gate, float* out,
                           const std::uint32_t* params, std::uint32_t num_heads, std::uint32_t num_kv_heads,
                           std::uint32_t tokens) {
  constexpr std::uint32_t HD = 256, QT = 4, KC = 32, QR = HD + 8, VR = HD + 8, PR = KC + 8, MR = 48;
  __shared__ __attribute__((aligned(16))) _Float16 Qs[MR * QR];
  __shared__ __attribute__((aligned(16))) _Float16 Vs[KC * VR];
  __shared__ __attribute__((aligned(16))) _Float16 Ps[MR * PR];
  __shared__ float Ss[MR * KC];
  __shared__ float mrow[MR], lrow[MR], alpha[MR];
  __shared__ std::uint32_t nkv[QT];
  const std::uint32_t g = blockIdx.x, t0 = blockIdx.y * QT, G = num_heads / num_kv_heads;
  const std::uint32_t tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t rows = QT * G, rtiles = (rows + 15) / 16;
  const float scale = 1.0f / sqrtf(static_cast<float>(HD));
  if (tid < QT) nkv[tid] = t0 + tid < tokens ? params[static_cast<std::size_t>(t0 + tid) * 16 + kHaloNVisible] : 0;
  for (std::uint32_t i = tid; i < MR * HD; i += blockDim.x) {
    const std::uint32_t r = i / HD, d = i % HD, tq = r / G, h = r % G;
    const bool live = r < rows && t0 + tq < tokens;
    Qs[r * QR + d] = static_cast<_Float16>(live ? q[(static_cast<std::size_t>(t0 + tq) * num_heads + g * G + h) * HD + d] : 0.0f);
  }
  if (tid < MR) {
    mrow[tid] = -INFINITY;
    lrow[tid] = 0.0f;
  }
  __syncthreads();
  // Keys 0 .. n - 1, n the longest list in the block (the last live token's).
  std::uint32_t n = 0;
  for (std::uint32_t i = 0; i < QT; ++i) n = max(n, nkv[i]);
  // P . V work: wave owns dim tiles 2 * (wave % 8) .. + 1 for every row tile.
  halo_v8f acc[3][2] = {};
  for (std::uint32_t c0 = 0; c0 < n; c0 += KC) {
    __syncthreads();  // previous chunk's Vs / Ps reads done
    for (std::uint32_t i = tid; i < KC * HD / 4; i += blockDim.x) {
      const std::uint32_t j = i / (HD / 4), d = (i % (HD / 4)) * 4;
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (c0 + j < n) v = *reinterpret_cast<const float4*>(vc + (static_cast<std::size_t>(c0 + j) * num_kv_heads + g) * HD + d);
      _Float16* dst = Vs + j * VR + d;
      dst[0] = static_cast<_Float16>(v.x);
      dst[1] = static_cast<_Float16>(v.y);
      dst[2] = static_cast<_Float16>(v.z);
      dst[3] = static_cast<_Float16>(v.w);
    }
    // Scores: wave -> (row tile wave / 2, key tile wave % 2), up to 6 waves.
    const std::uint32_t rt = wave / 2, kt = wave % 2;
    if (rt < rtiles) {
      const std::uint32_t kj = c0 + kt * 16 + lane % 16;
      const float* krow = kc + (static_cast<std::size_t>(min(kj, n - 1)) * num_kv_heads + g) * HD;
      halo_v8f s = {};
      for (std::uint32_t k0 = 0; k0 < HD; k0 += 16) {
        const halo_v16h a = LdsFrag(&Qs[(rt * 16 + lane % 16) * QR + k0]);
        halo_v16h b;
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
          const float4 v = *reinterpret_cast<const float4*>(krow + k0 + i);
          b[i] = static_cast<_Float16>(v.x);
          b[i + 1] = static_cast<_Float16>(v.y);
          b[i + 2] = static_cast<_Float16>(v.z);
          b[i + 3] = static_cast<_Float16>(v.w);
        }
        s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t r = rt * 16 + 2 * j + lane / 16, tq = r / G;
        const bool ok = r < rows && kj < nkv[min(tq, QT - 1)];
        Ss[r * KC + kt * 16 + lane % 16] = ok ? s[j] * scale : -INFINITY;
      }
    }
    __syncthreads();
    // Online softmax: 4 threads per row (192 of 256), 8 keys each.
    if (tid < MR * 4) {
      const std::uint32_t row = tid / 4, part = tid % 4;
      float mx = -INFINITY;
      if (row < rows)
        for (std::uint32_t c = part * 8; c < part * 8 + 8; ++c) mx = fmaxf(mx, Ss[row * KC + c]);
      for (int o = 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
      const float m_old = mrow[row], m_new = fmaxf(m_old, mx);
      float sum = 0.0f;
      for (std::uint32_t c = part * 8; c < part * 8 + 8; ++c) {
        const float p = row < rows && m_new != -INFINITY ? expf(Ss[row * KC + c] - m_new) : 0.0f;
        Ps[row * PR + c] = static_cast<_Float16>(p);
        sum += p;
      }
      for (int o = 2; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
      if (part == 0) {
        const float a = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
        alpha[row] = a;
        lrow[row] = lrow[row] * a + sum;
        mrow[row] = m_new;
      }
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t r = 0; r < 3; ++r) {  // fully unrolled: acc stays in registers
      if (r >= rtiles) break;
#pragma unroll
      for (int dt = 0; dt < 2; ++dt)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][dt][j] *= alpha[r * 16 + 2 * j + lane / 16];
      for (std::uint32_t kk = 0; kk < KC; kk += 16) {
        const halo_v16h a = LdsFrag(&Ps[(r * 16 + lane % 16) * PR + kk]);
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          const std::uint32_t d = (wave * 2 + dt) * 16 + lane % 16;
          halo_v16h b;
#pragma unroll
          for (int i = 0; i < 16; ++i) b[i] = Vs[(kk + i) * VR + d];
          acc[r][dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[r][dt]);
        }
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t r = 0; r < 3; ++r)
#pragma unroll
    for (int dt = 0; dt < 2; ++dt)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t row = r * 16 + 2 * j + lane / 16, tq = row / G, h = row % G;
        if (r >= rtiles || row >= rows || t0 + tq >= tokens) continue;
        const std::uint32_t d = (wave * 2 + dt) * 16 + lane % 16;
        const std::size_t idx = (static_cast<std::size_t>(t0 + tq) * num_heads + g * G + h) * HD + d;
        out[idx] = acc[r][dt][j] / lrow[row] * Sigmoid(gate[idx]);
      }
}

// halo_attention_wmma_mq's structure for QSA-selected keys: the 4 tokens of
// a group walk the union of their selections (halo_qsa_union4), so each K
// fragment and V chunk is read once for all of them (neighbouring tokens
// share ~60-70% of their selections: the union is ~1.6x one set instead of
// 4x). A token's scores outside its own selection are masked.
//   grid (num_kv_heads, ceil(T / 4)), block 256; head_dim 256, G <= 12.
extern "C" __global__ void __launch_bounds__(256)
    halo_attention_wmma_qsel(const float* q, const float* kc, const float* vc, const float* gate, float* out,
                             const std::uint32_t* params, const std::uint32_t* ublk, const std::uint8_t* umask,
                             const std::uint32_t* ucount, std::uint32_t max_u, std::uint32_t ratio,
                             std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t tokens) {
  constexpr std::uint32_t HD = 256, QT = 4, KC = 32, QR = HD + 8, VR = HD + 8, PR = KC + 8, MR = 48;
  __shared__ __attribute__((aligned(16))) _Float16 Qs[MR * QR];
  __shared__ __attribute__((aligned(16))) _Float16 Vs[KC * VR];
  __shared__ __attribute__((aligned(16))) _Float16 Ps[MR * PR];
  __shared__ float Ss[MR * KC];
  __shared__ float mrow[MR], lrow[MR], alpha[MR];
  __shared__ std::uint32_t nkv[QT];
  const std::uint32_t g = blockIdx.x, t0 = blockIdx.y * QT, G = num_heads / num_kv_heads;
  const std::uint32_t tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t rows = QT * G, rtiles = (rows + 15) / 16;
  const float scale = 1.0f / sqrtf(static_cast<float>(HD));
  if (tid < QT) nkv[tid] = t0 + tid < tokens ? params[static_cast<std::size_t>(t0 + tid) * 16 + kHaloNkv] : 0;
  const std::uint32_t* ub = ublk + static_cast<std::size_t>(blockIdx.y) * max_u;
  const std::uint8_t* um = umask + static_cast<std::size_t>(blockIdx.y) * max_u;
  // Key j of the union: position ub[j / ratio] * ratio + j % ratio.
  const auto pos_of = [&](std::uint32_t j) { return ub[j / ratio] * ratio + j % ratio; };
  for (std::uint32_t i = tid; i < MR * HD; i += blockDim.x) {
    const std::uint32_t r = i / HD, d = i % HD, tq = r / G, h = r % G;
    const bool live = r < rows && t0 + tq < tokens;
    Qs[r * QR + d] = static_cast<_Float16>(live ? q[(static_cast<std::size_t>(t0 + tq) * num_heads + g * G + h) * HD + d] : 0.0f);
  }
  if (tid < MR) {
    mrow[tid] = -INFINITY;
    lrow[tid] = 0.0f;
  }
  __syncthreads();
  // The union's keys; token q sees key j when its bit is set and the
  // position is before its own causal limit.
  const std::uint32_t n = ucount[blockIdx.y] * ratio;
  // Positions past the group's last causal limit (the tail block's unused
  // slots) are masked for every token, but their K/V must not be read: past
  // the cache's end, or stale and possibly non-finite (0 * NaN poisons P . V).
  std::uint32_t pend = 0;
  for (std::uint32_t i = 0; i < QT; ++i) pend = max(pend, nkv[i]);
  const auto pos_in = [&](std::uint32_t j) { return min(pos_of(min(j, n - 1)), pend - 1); };
  const auto live_at = [&](std::uint32_t j) { return j < n && pos_of(min(j, n - 1)) < pend; };
  // P . V work: wave owns dim tiles 2 * (wave % 8) .. + 1 for every row tile.
  halo_v8f acc[3][2] = {};
  // Only LDS is shared between the waves: LDS-only barriers keep the next
  // chunk's V rows (prefetched into registers, unconditionally: clamped to
  // the last key) in flight while this chunk is scored and multiplied (a
  // trace of the synchronous version: 51% of cycles waiting on gathers).
  const auto barrier = [] {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
  };
  constexpr std::uint32_t VP = KC * HD / 4 / 256;  // float4 of V per thread per chunk
  float4 rv[VP];
  const auto load_v = [&](std::uint32_t c) {
#pragma unroll
    for (std::uint32_t p = 0; p < VP; ++p) {
      const std::uint32_t i = tid + p * 256, j = i / (HD / 4), d = (i % (HD / 4)) * 4;
      rv[p] = *reinterpret_cast<const float4*>(
          vc + (static_cast<std::size_t>(pos_in(c + j)) * num_kv_heads + g) * HD + d);
    }
  };
  if (n > 0) load_v(0);
  for (std::uint32_t c0 = 0; c0 < n; c0 += KC) {
    barrier();  // previous chunk's Vs / Ps reads done
#pragma unroll
    for (std::uint32_t p = 0; p < VP; ++p) {
      const std::uint32_t i = tid + p * 256, j = i / (HD / 4), d = (i % (HD / 4)) * 4;
      const bool live = live_at(c0 + j);
      _Float16* dst = Vs + j * VR + d;
      dst[0] = static_cast<_Float16>(live ? rv[p].x : 0.0f);
      dst[1] = static_cast<_Float16>(live ? rv[p].y : 0.0f);
      dst[2] = static_cast<_Float16>(live ? rv[p].z : 0.0f);
      dst[3] = static_cast<_Float16>(live ? rv[p].w : 0.0f);
    }
    load_v(min(c0 + KC, n - 1));  // next chunk, in flight during the work below
    // Scores: wave -> (row tile wave / 2, key tile wave % 2), up to 6 waves.
    const std::uint32_t rt = wave / 2, kt = wave % 2;
    if (rt < rtiles) {
      const std::uint32_t kj = c0 + kt * 16 + lane % 16;
      const std::uint32_t kp = pos_of(min(kj, n - 1)), km = um[min(kj, n - 1) / ratio];
      const float* krow = kc + (static_cast<std::size_t>(pos_in(kj)) * num_kv_heads + g) * HD;
      halo_v8f s = {};
      for (std::uint32_t k0 = 0; k0 < HD; k0 += 16) {
        const halo_v16h a = LdsFrag(&Qs[(rt * 16 + lane % 16) * QR + k0]);
        halo_v16h b;
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
          const float4 v = *reinterpret_cast<const float4*>(krow + k0 + i);
          b[i] = static_cast<_Float16>(v.x);
          b[i + 1] = static_cast<_Float16>(v.y);
          b[i + 2] = static_cast<_Float16>(v.z);
          b[i + 3] = static_cast<_Float16>(v.w);
        }
        s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t r = rt * 16 + 2 * j + lane / 16, tq = r / G;
        const std::uint32_t qq = min(tq, QT - 1);
        const bool ok = r < rows && kj < n && ((km >> qq) & 1u) && kp < nkv[qq];
        Ss[r * KC + kt * 16 + lane % 16] = ok ? s[j] * scale : -INFINITY;
      }
    }
    barrier();
    // Online softmax: 4 threads per row (192 of 256), 8 keys each.
    if (tid < MR * 4) {
      const std::uint32_t row = tid / 4, part = tid % 4;
      float mx = -INFINITY;
      if (row < rows)
        for (std::uint32_t c = part * 8; c < part * 8 + 8; ++c) mx = fmaxf(mx, Ss[row * KC + c]);
      for (int o = 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
      const float m_old = mrow[row], m_new = fmaxf(m_old, mx);
      float sum = 0.0f;
      for (std::uint32_t c = part * 8; c < part * 8 + 8; ++c) {
        const float p = row < rows && m_new != -INFINITY ? expf(Ss[row * KC + c] - m_new) : 0.0f;
        Ps[row * PR + c] = static_cast<_Float16>(p);
        sum += p;
      }
      for (int o = 2; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
      if (part == 0) {
        const float a = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
        alpha[row] = a;
        lrow[row] = lrow[row] * a + sum;
        mrow[row] = m_new;
      }
    }
    barrier();
#pragma unroll
    for (std::uint32_t r = 0; r < 3; ++r) {  // fully unrolled: acc stays in registers
      if (r >= rtiles) break;
#pragma unroll
      for (int dt = 0; dt < 2; ++dt)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][dt][j] *= alpha[r * 16 + 2 * j + lane / 16];
      for (std::uint32_t kk = 0; kk < KC; kk += 16) {
        const halo_v16h a = LdsFrag(&Ps[(r * 16 + lane % 16) * PR + kk]);
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          const std::uint32_t d = (wave * 2 + dt) * 16 + lane % 16;
          halo_v16h b;
#pragma unroll
          for (int i = 0; i < 16; ++i) b[i] = Vs[(kk + i) * VR + d];
          acc[r][dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[r][dt]);
        }
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t r = 0; r < 3; ++r)
#pragma unroll
    for (int dt = 0; dt < 2; ++dt)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t row = r * 16 + 2 * j + lane / 16, tq = row / G, h = row % G;
        if (r >= rtiles || row >= rows || t0 + tq >= tokens) continue;
        const std::uint32_t d = (wave * 2 + dt) * 16 + lane % 16;
        const std::size_t idx = (static_cast<std::size_t>(t0 + tq) * num_heads + g * G + h) * HD + d;
        out[idx] = acc[r][dt][j] / lrow[row] * Sigmoid(gate[idx]);
      }
}

// halo_attention_wmma_qsel over the fp16 copy of the K/V cache (the WMMA
// path rounds K/V to fp16 anyway, so the result is the same): K and V of
// the next chunk prefetched together, K staged in LDS for the scoring waves.
extern "C" __global__ void __launch_bounds__(256)
    halo_attention_wmma_qsel_h(const float* q, const _Float16* kc, const _Float16* vc, const float* gate, float* out,
                             const std::uint32_t* params, const std::uint32_t* ublk, const std::uint8_t* umask,
                             const std::uint32_t* ucount, std::uint32_t max_u, std::uint32_t ratio,
                             std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t tokens) {
  constexpr std::uint32_t HD = 256, QT = 4, KC = 32, QR = HD + 8, VR = HD + 8, PR = KC + 8, MR = 48;
  __shared__ __attribute__((aligned(16))) _Float16 Qs[MR * QR];
  __shared__ __attribute__((aligned(16))) _Float16 Vs[KC * VR];
  __shared__ __attribute__((aligned(16))) _Float16 Ps[MR * PR];
  // The chunk's K, staged once for every scoring wave; its space is reused
  // for the scores once all waves have read it.
  __shared__ __attribute__((aligned(16))) _Float16 Ks[KC * VR];
  float* const Ss = reinterpret_cast<float*>(Ks);
  static_assert(sizeof(float) * MR * KC <= sizeof(_Float16) * KC * VR, "scores fit the K tile");
  __shared__ float mrow[MR], lrow[MR], alpha[MR];
  __shared__ std::uint32_t nkv[QT];
  const std::uint32_t g = blockIdx.x, t0 = blockIdx.y * QT, G = num_heads / num_kv_heads;
  const std::uint32_t tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t rows = QT * G, rtiles = (rows + 15) / 16;
  const float scale = 1.0f / sqrtf(static_cast<float>(HD));
  if (tid < QT) nkv[tid] = t0 + tid < tokens ? params[static_cast<std::size_t>(t0 + tid) * 16 + kHaloNkv] : 0;
  const std::uint32_t* ub = ublk + static_cast<std::size_t>(blockIdx.y) * max_u;
  const std::uint8_t* um = umask + static_cast<std::size_t>(blockIdx.y) * max_u;
  // Key j of the union: position ub[j / ratio] * ratio + j % ratio.
  const auto pos_of = [&](std::uint32_t j) { return ub[j / ratio] * ratio + j % ratio; };
  for (std::uint32_t i = tid; i < MR * HD; i += blockDim.x) {
    const std::uint32_t r = i / HD, d = i % HD, tq = r / G, h = r % G;
    const bool live = r < rows && t0 + tq < tokens;
    Qs[r * QR + d] = static_cast<_Float16>(live ? q[(static_cast<std::size_t>(t0 + tq) * num_heads + g * G + h) * HD + d] : 0.0f);
  }
  if (tid < MR) {
    mrow[tid] = -INFINITY;
    lrow[tid] = 0.0f;
  }
  __syncthreads();
  // The union's keys; token q sees key j when its bit is set and the
  // position is before its own causal limit.
  const std::uint32_t n = ucount[blockIdx.y] * ratio;
  // Positions past the group's last causal limit (the tail block's unused
  // slots) are masked for every token, but their K/V must not be read: past
  // the cache's end, or stale and possibly non-finite (0 * NaN poisons P . V).
  std::uint32_t pend = 0;
  for (std::uint32_t i = 0; i < QT; ++i) pend = max(pend, nkv[i]);
  const auto pos_in = [&](std::uint32_t j) { return min(pos_of(min(j, n - 1)), pend - 1); };
  const auto live_at = [&](std::uint32_t j) { return j < n && pos_of(min(j, n - 1)) < pend; };
  // P . V work: wave owns dim tiles 2 * (wave % 8) .. + 1 for every row tile.
  halo_v8f acc[3][2] = {};
  // Only LDS is shared between the waves: LDS-only barriers keep the next
  // chunk's V rows (prefetched into registers, unconditionally: clamped to
  // the last key) in flight while this chunk is scored and multiplied (a
  // trace of the synchronous version: 51% of cycles waiting on gathers).
  const auto barrier = [] {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
  };
  // K and V of the next chunk in flight as fp16 (the fp16 cache copy: half
  // the bytes and registers of fp32), 16 bytes = 8 values per load.
  constexpr std::uint32_t KP = KC * HD / 8 / 256;  // uint4 per thread per chunk, each of K and V
  uint4 rk[KP], rv[KP];
  const auto load_kv = [&](std::uint32_t c) {
#pragma unroll
    for (std::uint32_t p = 0; p < KP; ++p) {
      const std::uint32_t i = tid + p * 256, j = i / (HD / 8), d = (i % (HD / 8)) * 8;
      const std::size_t at = (static_cast<std::size_t>(pos_in(c + j)) * num_kv_heads + g) * HD + d;
      rk[p] = *reinterpret_cast<const uint4*>(kc + at);
      rv[p] = *reinterpret_cast<const uint4*>(vc + at);
    }
  };
  if (n > 0) load_kv(0);
  for (std::uint32_t c0 = 0; c0 < n; c0 += KC) {
    barrier();  // previous chunk's Ks (scores) / Vs / Ps reads done
#pragma unroll
    for (std::uint32_t p = 0; p < KP; ++p) {
      const std::uint32_t i = tid + p * 256, j = i / (HD / 8), d = (i % (HD / 8)) * 8;
      const bool live = live_at(c0 + j);
      *reinterpret_cast<uint4*>(Ks + j * VR + d) = live ? rk[p] : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(Vs + j * VR + d) = live ? rv[p] : make_uint4(0, 0, 0, 0);
    }
    load_kv(min(c0 + KC, n - 1));  // next chunk, in flight during the work below
    barrier();                     // Ks, Vs complete
    // Scores: wave -> (row tile wave / 2, key tile wave % 2), up to 6 waves.
    const std::uint32_t rt = wave / 2, kt = wave % 2;
    const std::uint32_t kj = c0 + kt * 16 + lane % 16;
    const std::uint32_t kp = pos_of(min(kj, n - 1)), km = um[min(kj, n - 1) / ratio];
    halo_v8f s = {};
    if (rt < rtiles)
      for (std::uint32_t k0 = 0; k0 < HD; k0 += 16)
        s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(LdsFrag(&Qs[(rt * 16 + lane % 16) * QR + k0]),
                                                       LdsFrag(&Ks[(kt * 16 + lane % 16) * VR + k0]), s);
    barrier();  // every wave's Ks reads done: the scores overwrite it
    if (rt < rtiles) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t r = rt * 16 + 2 * j + lane / 16, tq = r / G;
        const std::uint32_t qq = min(tq, QT - 1);
        const bool ok = r < rows && kj < n && ((km >> qq) & 1u) && kp < nkv[qq];
        Ss[r * KC + kt * 16 + lane % 16] = ok ? s[j] * scale : -INFINITY;
      }
    }
    barrier();
    // Online softmax: 4 threads per row (192 of 256), 8 keys each.
    if (tid < MR * 4) {
      const std::uint32_t row = tid / 4, part = tid % 4;
      float mx = -INFINITY;
      if (row < rows)
        for (std::uint32_t c = part * 8; c < part * 8 + 8; ++c) mx = fmaxf(mx, Ss[row * KC + c]);
      for (int o = 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
      const float m_old = mrow[row], m_new = fmaxf(m_old, mx);
      float sum = 0.0f;
      for (std::uint32_t c = part * 8; c < part * 8 + 8; ++c) {
        const float p = row < rows && m_new != -INFINITY ? expf(Ss[row * KC + c] - m_new) : 0.0f;
        Ps[row * PR + c] = static_cast<_Float16>(p);
        sum += p;
      }
      for (int o = 2; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
      if (part == 0) {
        const float a = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
        alpha[row] = a;
        lrow[row] = lrow[row] * a + sum;
        mrow[row] = m_new;
      }
    }
    barrier();
#pragma unroll
    for (std::uint32_t r = 0; r < 3; ++r) {  // fully unrolled: acc stays in registers
      if (r >= rtiles) break;
#pragma unroll
      for (int dt = 0; dt < 2; ++dt)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][dt][j] *= alpha[r * 16 + 2 * j + lane / 16];
      for (std::uint32_t kk = 0; kk < KC; kk += 16) {
        const halo_v16h a = LdsFrag(&Ps[(r * 16 + lane % 16) * PR + kk]);
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          const std::uint32_t d = (wave * 2 + dt) * 16 + lane % 16;
          halo_v16h b;
#pragma unroll
          for (int i = 0; i < 16; ++i) b[i] = Vs[(kk + i) * VR + d];
          acc[r][dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[r][dt]);
        }
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t r = 0; r < 3; ++r)
#pragma unroll
    for (int dt = 0; dt < 2; ++dt)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t row = r * 16 + 2 * j + lane / 16, tq = row / G, h = row % G;
        if (r >= rtiles || row >= rows || t0 + tq >= tokens) continue;
        const std::uint32_t d = (wave * 2 + dt) * 16 + lane % 16;
        const std::size_t idx = (static_cast<std::size_t>(t0 + tq) * num_heads + g * G + h) * HD + d;
        out[idx] = acc[r][dt][j] / lrow[row] * Sigmoid(gate[idx]);
      }
}

// F32 GEMM for few rows (<= 16: the mixer's 4-row inject, the shared
// expert's 1-row gate): one warp per token reads the token's row once,
// coalesced, against every weight row (which stay in cache); lane sums
// reduce across the warp. fp32 throughout.
//   grid (ceil(T / 8)), block 256; rows <= 16.
extern "C" __global__ void __launch_bounds__(256)
    halo_gemm_f32_skinny(const float* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols,
                         std::uint32_t tokens) {
  const std::uint32_t lane = threadIdx.x % 32, t = blockIdx.x * 8 + threadIdx.x / 32;
  if (t >= tokens) return;
  const float* xr = x + static_cast<std::size_t>(t) * cols;
  float acc[16] = {};
  for (std::uint32_t c = lane; c < cols; c += 32) {
    const float xv = xr[c];
#pragma unroll
    for (std::uint32_t r = 0; r < 16; ++r)
      if (r < rows) acc[r] = fmaf(w[static_cast<std::size_t>(r) * cols + c], xv, acc[r]);
  }
#pragma unroll
  for (std::uint32_t r = 0; r < 16; ++r) {
    if (r >= rows) break;
    const float s = WarpSum(acc[r]);
    if (lane == 0) y[static_cast<std::size_t>(t) * rows + r] = s;
  }
}

// Profiling: the GPU's constant-rate counter into stamps[slot] (gfx11 reads
// it with s_sendmsg_rtn MSG_RTN_GET_REALTIME, as HIP's wall_clock64 does).
// Launched after each kernel while kernel timing is on; kernels on the stream
// run in order, so consecutive stamps bracket each kernel without any host
// synchronization. grid (1), block 32
extern "C" __global__ void halo_timestamp(unsigned long long* stamps, std::uint32_t slot) {
  if (threadIdx.x == 0) stamps[slot] = __builtin_amdgcn_s_sendmsg_rtnl(0x83);  // MSG_RTN_GET_REALTIME
}

// ---- Bench: memory hierarchy (halo_gemm_bench memory) ------------------------
// Streaming read of n float4 (summed so the loads stay), `reps` passes.
//   grid (blocks), block 256
extern "C" __global__ void halo_bench_read(const float4* p, std::uint64_t n, std::uint32_t reps, float* out) {
  float acc = 0.0f;
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint32_t r = 0; r < reps; ++r)
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; i < n; i += stride) {
      const float4 v = p[i];
      acc += v.x + v.y + v.z + v.w;
    }
  if (acc == 12345.678f) out[0] = acc;  // keeps the loads
}
// Dependent loads: i = next[i] for `steps` steps from 0 (one thread).
extern "C" __global__ void halo_bench_chase(const std::uint32_t* next, std::uint32_t steps, std::uint32_t* out) {
  std::uint32_t i = 0;
  for (std::uint32_t s = 0; s < steps; ++s) i = next[i];
  out[0] = i;
}
