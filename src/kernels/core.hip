// Correctness-first GPU kernels for single-token decode (P2). Each mirrors an
// operation of the CPU reference (src/model/*.cpp) and is tested against it;
// speed comes in P3. Wave32 (gfx1151). FP32 accumulation throughout.
//
// ABI (src/rt/device.hpp): pointer parameters are buffers, everything else is
// a by-value scalar. Kernels are extern "C" so names are stable.

#include <hip/hip_runtime.h>

#include <cstdint>

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
extern "C" __global__ void halo_rope_neox(float* x, std::uint32_t head_dim, std::uint32_t rotary_dim,
                                          std::uint32_t pos, float theta) {
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
// state row j (blockDim == d). State S[h][j][i]: j over value dim, i over key.
extern "C" __global__ void halo_gdn_step(const float* q, const float* k, const float* v, const float* alpha,
                                         const float* beta, const float* a, const float* dt, float* state, float* out,
                                         std::uint32_t num_k_heads, std::uint32_t d) {
  const std::uint32_t h = blockIdx.x, j = threadIdx.x;
  if (j >= d) return;
  const float* qh = q + (h % num_k_heads) * d;
  const float* kh = k + (h % num_k_heads) * d;
  const float decay = expf(a[h] * Softplus(alpha[h] + dt[h]));
  const float b = Sigmoid(beta[h]);
  float* row = state + (static_cast<std::size_t>(h) * d + j) * d;
  float u = 0.0f;
  for (std::uint32_t i = 0; i < d; ++i) {
    row[i] *= decay;
    u += row[i] * kh[i];
  }
  const float delta = (v[h * d + j] - u) * b;
  float o = 0.0f;
  for (std::uint32_t i = 0; i < d; ++i) {
    row[i] += delta * kh[i];
    o += row[i] * qh[i];
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
extern "C" __global__ void halo_pool_block(const float* raw, float* pooled, std::uint32_t block, std::uint32_t ratio,
                                           std::uint32_t dim) {
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float acc = 0.0f;
    for (std::uint32_t t = 0; t < ratio; ++t) acc += raw[(block * ratio + t) * dim + i] / static_cast<float>(ratio);
    pooled[i] = acc;
  }
}

// QSA block scores: score[b] = sum_heads relu(<q_h, k_b>). One thread per block.
extern "C" __global__ void halo_qsa_scores(const float* q, const float* blocks, float* score, std::uint32_t heads,
                                           std::uint32_t dim, std::uint32_t n_blocks) {
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
                                                 std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t hd,
                                                 std::uint32_t n, std::uint32_t max_n) {
  __shared__ float scratch[32];
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
extern "C" __global__ void halo_qsa_select(const float* score, std::uint32_t* visible, std::uint32_t n, std::uint32_t k,
                                           std::uint32_t ratio, std::uint32_t n_kv) {
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
