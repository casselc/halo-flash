// ---- P6: batched decode (docs/SESSIONS.md) -------------------------------------------------------------------
// Included at the end of core.hip. Every kernel here computes NB independent decode columns (tokens of
// different sessions, or the positions of one MTP verify) in one launch, reading each weight once. The rule
// that makes a batched run byte-identical to NB single-token runs: every column's value goes through the
// arithmetic of the kernel it replaces in the same order (same lane owning the same terms, same per-lane
// accumulation order, same cross-lane reduction); only the loads are shared. A property test per kernel
// compares each column bitwise against the single-token kernel (tests/gpu/batch_gpu_test.cpp).
//
// Column layout: column b's input at x + b * xstride and output at y + b * ystride (floats).

namespace {

// Q8RowsDot / GemvQ8rMulti over NB columns. x is staged through LDS in chunks of `chunk` columns (a multiple of
// kWarp * 16, so a lane's sequence of 16-value pieces is the single kernel's), for all columns at once.
// Every thread of the block must call it (it synchronizes); `active` waves compute, the others only stage.
template <int kRows, int NB>
__device__ inline void Q8RowsDotB(const std::int8_t* qs, const std::uint16_t* d, const float* x, std::uint32_t xstride,
                                  float* lds, std::uint32_t chunk, std::uint32_t r0, bool active, std::uint32_t rows,
                                  std::uint32_t cols, std::uint32_t qstride, float (&out)[NB][kRows]) {
  const std::uint32_t lane = threadIdx.x % kWarp, nblk = cols / 32;
  float acc[NB][kRows] = {};
  for (std::uint32_t k0 = 0; k0 < cols; k0 += chunk) {
    const std::uint32_t n = min(chunk, cols - k0);
    __syncthreads();
#pragma unroll
    for (int b = 0; b < NB; ++b)
      for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) lds[b * chunk + i] = x[b * xstride + k0 + i];
    __syncthreads();
    if (!active) continue;
    for (std::uint32_t c = k0 + lane * 16; c < k0 + n; c += kWarp * 16) {
      int4 q[kRows];
      float sc[kRows];
#pragma unroll
      for (int k = 0; k < kRows; ++k) {
        const std::uint32_t r = min(r0 + k, rows - 1);
        q[k] = *reinterpret_cast<const int4*>(qs + static_cast<std::size_t>(r) * qstride + c);
        sc[k] = F16(d[static_cast<std::size_t>(r) * nblk + c / 32]);
      }
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        float xr[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) xr[i] = lds[b * chunk + (c - k0) + i];
#pragma unroll
        for (int k = 0; k < kRows; ++k) {
          const std::int8_t* v = reinterpret_cast<const std::int8_t*>(&q[k]);
          float part = 0.0f;
#pragma unroll
          for (int i = 0; i < 16; ++i) part += static_cast<float>(v[i]) * xr[i];
          acc[b][k] += sc[k] * part;
        }
      }
    }
  }
#pragma unroll
  for (int b = 0; b < NB; ++b)
#pragma unroll
    for (int k = 0; k < kRows; ++k) out[b][k] = WarpSum(acc[b][k]);
}

// LDS budget of the batched Q8 kernels: 12288 floats (48 KB), as halo_gemv_q8r_r4's staging.
constexpr std::uint32_t kBatchLds = 12288;
template <int NB>
constexpr std::uint32_t BatchChunk() {
  return (kBatchLds / NB) / (kWarp * 16) * (kWarp * 16);
}

template <int kRows, int NB>
__device__ inline void GemvQ8rB(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y,
                                std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride, std::uint32_t xstride,
                                std::uint32_t ystride) {
  __shared__ float lds[kBatchLds];
  const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  float out[NB][kRows];
  Q8RowsDotB<kRows, NB>(qs, d, x, xstride, lds, BatchChunk<NB>(), r0, r0 < rows, rows, cols, qstride, out);
  if (r0 >= rows || threadIdx.x % kWarp != 0) return;
#pragma unroll
  for (int b = 0; b < NB; ++b)
#pragma unroll
    for (int k = 0; k < kRows; ++k)
      if (r0 + k < rows) y[b * ystride + r0 + k] = out[b][k];
}

// RowDotHoisted for NB columns: the row's loads once, each column's FMA chain in the single kernel's order.
template <int NB>
__device__ inline void RowDotHoistedB(const float* row, const float* x, std::uint32_t xstride, std::uint32_t cols,
                                      float (&acc)[NB]) {
  constexpr std::uint32_t kMax = 16;
  float wv[kMax];
#pragma unroll
  for (std::uint32_t j = 0; j < kMax; ++j) {
    const std::uint32_t c = threadIdx.x + j * blockDim.x;
    wv[j] = c < cols ? row[c] : 0.0f;
  }
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    float xv[kMax];
#pragma unroll
    for (std::uint32_t j = 0; j < kMax; ++j) {
      const std::uint32_t c = threadIdx.x + j * blockDim.x;
      xv[j] = c < cols ? x[b * xstride + c] : 0.0f;
    }
    float a = 0.0f;
#pragma unroll
    for (std::uint32_t j = 0; j < kMax; ++j)
      if (threadIdx.x + j * blockDim.x < cols) a = __builtin_fmaf(wv[j], xv[j], a);
    acc[b] = a;
  }
}

}  // namespace

// The batched kernels, one per batch size (NB = 2..8). HALO_BATCH_KERNELS(name, body) instantiates them.
#define HALO_FOR_NB(X) X(2) X(3) X(4) X(5) X(6) X(7) X(8)

// halo_gemv_q8r_r4 / _r2.
#define HALO_GEMV_B(NB)                                                                                              \
  extern "C" __global__ void __launch_bounds__(256)                                                                  \
      halo_gemv_q8r_r4_b##NB(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y,                \
                             std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride, std::uint32_t xstride,   \
                             std::uint32_t ystride) {                                                                \
    GemvQ8rB<4, NB>(qs, d, x, y, rows, cols, qstride, xstride, ystride);                                             \
  }                                                                                                                  \
  extern "C" __global__ void __launch_bounds__(256)                                                                  \
      halo_gemv_q8r_r2_b##NB(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y,                \
                             std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride, std::uint32_t xstride,   \
                             std::uint32_t ystride) {                                                                \
    GemvQ8rB<2, NB>(qs, d, x, y, rows, cols, qstride, xstride, ystride);                                             \
  }
HALO_FOR_NB(HALO_GEMV_B)
#undef HALO_GEMV_B

// halo_gemv_q8r_dual (DeltaNet qkv + z).
#define HALO_DUAL_B(NB)                                                                                              \
  extern "C" __global__ void __launch_bounds__(256) halo_gemv_q8r_dual_b##NB(                                       \
      const std::int8_t* q1, const std::uint16_t* d1, float* y1, std::uint32_t rows1, std::uint32_t qstride1,        \
      const std::int8_t* q2, const std::uint16_t* d2, float* y2, std::uint32_t rows2, std::uint32_t qstride2,        \
      const float* x, std::uint32_t cols, std::uint32_t blocks1, std::uint32_t xstride, std::uint32_t ystride1,      \
      std::uint32_t ystride2) {                                                                                      \
    __shared__ float lds[kBatchLds];                                                                                 \
    const bool first = blockIdx.x < blocks1;                                                                         \
    const std::uint32_t b = first ? blockIdx.x : blockIdx.x - blocks1;                                               \
    const std::int8_t* qs = first ? q1 : q2;                                                                         \
    const std::uint16_t* d = first ? d1 : d2;                                                                        \
    float* y = first ? y1 : y2;                                                                                      \
    const std::uint32_t rows = first ? rows1 : rows2, qstride = first ? qstride1 : qstride2;                         \
    const std::uint32_t ystride = first ? ystride1 : ystride2;                                                       \
    constexpr int kRows = 4;                                                                                         \
    const std::uint32_t r0 = (b * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                               \
    float acc[NB][kRows];                                                                                            \
    Q8RowsDotB<kRows, NB>(qs, d, x, xstride, lds, BatchChunk<NB>(), r0, r0 < rows, rows, cols, qstride, acc);       \
    if (r0 >= rows || threadIdx.x % kWarp != 0) return;                                                              \
    _Pragma("unroll") for (int c = 0; c < NB; ++c) _Pragma("unroll") for (int k = 0; k < kRows; ++k) if (r0 + k <   \
                                                                                                          rows)      \
        y[c * ystride + r0 + k] = acc[c][k];                                                                         \
  }
HALO_FOR_NB(HALO_DUAL_B)
#undef HALO_DUAL_B

// halo_matvec_f32_rowblock2_u (DeltaNet alpha and beta): a block per row; y strides are the outputs' strides.
#define HALO_ROWBLOCK2_B(NB)                                                                                         \
  extern "C" __global__ void __launch_bounds__(256) halo_matvec_f32_rowblock2_u_b##NB(                              \
      const float* wa, const float* wb, const float* x, float* ya, float* yb, std::uint32_t rows,                    \
      std::uint32_t cols, std::uint32_t xstride, std::uint32_t ystride) {                                            \
    __shared__ float scratch[32];                                                                                    \
    const bool a = blockIdx.x < rows;                                                                                \
    const std::uint32_t r = a ? blockIdx.x : blockIdx.x - rows;                                                      \
    float acc[NB];                                                                                                   \
    RowDotHoistedB<NB>((a ? wa : wb) + static_cast<std::size_t>(r) * cols, x, xstride, cols, acc);                   \
    _Pragma("unroll") for (int c = 0; c < NB; ++c) {                                                                 \
      const float s = BlockSum(acc[c], scratch);                                                                     \
      if (threadIdx.x == 0) (a ? ya : yb)[c * ystride + r] = s;                                                      \
    }                                                                                                                \
  }
HALO_FOR_NB(HALO_ROWBLOCK2_B)
#undef HALO_ROWBLOCK2_B

// halo_router_rowblock_q8act: router logits and, in the first cols / 32 blocks, column b's int8 activations into
// act[b] (QuantizeActivations' values for a 32-value block do not depend on the column next to it).
#define HALO_ROUTER_B(NB)                                                                                            \
  extern "C" __global__ void __launch_bounds__(256) halo_router_rowblock_q8act_b##NB(                               \
      const float* w, const float* x, float* y, std::uint32_t rows, std::uint32_t cols, Q8Act* act,                  \
      std::uint32_t xstride, std::uint32_t ystride) {                                                                \
    __shared__ float scratch[32];                                                                                    \
    if (blockIdx.x < cols / 32 && threadIdx.x < kWarp) {                                                             \
      const std::uint32_t lane = threadIdx.x, blk = blockIdx.x;                                                      \
      _Pragma("unroll") for (int c = 0; c < NB; ++c) {                                                               \
        const float v = x[c * xstride + blk * 32 + lane];                                                            \
        float amax = fabsf(v), sum = v;                                                                              \
        for (int o = kWarp / 2; o > 0; o >>= 1) {                                                                    \
          amax = fmaxf(amax, __shfl_xor(amax, o, kWarp));                                                            \
          sum += __shfl_xor(sum, o, kWarp);                                                                          \
        }                                                                                                            \
        const float scale = amax / 127.0f;                                                                           \
        const int qv = scale > 0.0f ? static_cast<int>(rintf(v / scale)) : 0;                                        \
        const std::uint32_t byte = static_cast<std::uint32_t>(qv & 0xFF) << (8 * (lane & 3));                        \
        std::uint32_t word = byte;                                                                                   \
        word |= __shfl_xor(word, 1, kWarp);                                                                          \
        word |= __shfl_xor(word, 2, kWarp);                                                                          \
        if ((lane & 3) == 0) act[c].q[blk * 8 + lane / 4] = word;                                                    \
        if (lane == 0) {                                                                                             \
          act[c].scale[blk] = scale;                                                                                 \
          act[c].sum[blk] = sum;                                                                                     \
        }                                                                                                            \
      }                                                                                                              \
    }                                                                                                                \
    const std::uint32_t r = blockIdx.x;                                                                              \
    if (r >= rows) return;                                                                                           \
    float acc[NB];                                                                                                   \
    RowDotHoistedB<NB>(w + static_cast<std::size_t>(r) * cols, x, xstride, cols, acc);                               \
    _Pragma("unroll") for (int c = 0; c < NB; ++c) {                                                                 \
      const float s = BlockSum(acc[c], scratch);                                                                     \
      if (threadIdx.x == 0) y[c * ystride + r] = s;                                                                  \
    }                                                                                                                \
  }
HALO_FOR_NB(HALO_ROUTER_B)
#undef HALO_ROUTER_B

// ---- Routed experts across columns, grouped by expert ---------------------------------------------------------
// halo_moe_bucket_b: ids [n = NB * k] (column b's slot s at b * k + s; an expert appears at most once a column)
// -> ucount, uexp[u], unum[u], upair[u * 8 + j] (pairs in column order). One block of >= n threads.
extern "C" __global__ void __launch_bounds__(128)
    halo_moe_bucket_b(const std::uint32_t* ids, std::uint32_t n, std::uint32_t* ucount, std::uint32_t* uexp,
                      std::uint32_t* unum, std::uint32_t* upair) {
  const std::uint32_t p = threadIdx.x;
  bool first = false;
  std::uint32_t e = 0;
  if (p < n) {
    e = ids[p];
    first = true;
    for (std::uint32_t q = 0; q < p; ++q)
      if (ids[q] == e) first = false;
  }
  __shared__ std::uint32_t flag[128];
  flag[p] = first ? 1u : 0u;
  __syncthreads();
  if (first) {
    std::uint32_t u = 0;
    for (std::uint32_t q = 0; q < p; ++q) u += flag[q];
    uexp[u] = e;
    std::uint32_t m = 0;
    for (std::uint32_t q = p; q < n; ++q)
      if (ids[q] == e) upair[u * 8 + m++] = q;
    unum[u] = m;
  }
  if (p == 0) {
    std::uint32_t u = 0;
    for (std::uint32_t q = 0; q < n; ++q) u += flag[q];
    *ucount = u;
  }
}

namespace {

// Q4KItemDot split: the weight loads (shared by every pair of an expert) and the arithmetic (per pair), the
// arithmetic copied from Q4KItemDot.
struct Q4KPiece {
  int4 hdr, qa, qb;
};
__device__ inline Q4KPiece Q4KLoad(const std::uint8_t* row, std::uint32_t item) {
  const std::uint32_t sb = item / 4, c = item % 4;
  const std::uint8_t* b = row + sb * 144;
  return {*reinterpret_cast<const int4*>(b), *reinterpret_cast<const int4*>(b + 16 + c * 32),
          *reinterpret_cast<const int4*>(b + 16 + c * 32 + 16)};
}
__device__ inline float Q4KDot(const Q4KPiece& w, const Q8Act& a, std::uint32_t item) {
  const std::uint32_t sb = item / 4, c = item % 4;
  const int4 hdr = w.hdr, qa = w.qa, qb = w.qb;
  const auto w0 = static_cast<std::uint32_t>(hdr.y), w1 = static_cast<std::uint32_t>(hdr.z),
             w2 = static_cast<std::uint32_t>(hdr.w);
  std::uint32_t sc0, m0, sc1, m1;
  ScaleMinK4Words(2 * c, w0, w1, w2, sc0, m0);
  ScaleMinK4Words(2 * c + 1, w0, w1, w2, sc1, m1);
  const std::uint32_t s0 = sb * 8 + 2 * c, s1 = s0 + 1;
  const std::uint32_t* x0 = a.q + s0 * 8;
  const std::uint32_t* x1 = a.q + s1 * 8;
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

// DotQ5_1::Item split the same way.
struct Q5_1Piece {
  uint2 hd, qa, qb;
};
__device__ inline Q5_1Piece Q5_1Load(const std::uint8_t* row, std::uint32_t item) {
  const std::uint8_t* b = row + item * 24;
  return {*reinterpret_cast<const uint2*>(b), *reinterpret_cast<const uint2*>(b + 8),
          *reinterpret_cast<const uint2*>(b + 16)};
}
__device__ inline float Q5_1Dot(const Q5_1Piece& w, const float* xs, std::uint32_t item) {
  const std::uint32_t qh = w.hd.y;
  std::uint8_t q[16];
  *reinterpret_cast<uint2*>(q) = w.qa;
  *reinterpret_cast<uint2*>(q + 8) = w.qb;
  const float* x = xs + item * 32;
  float s = 0, sx = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float lo = static_cast<float>((q[j] & 0xF) | (((qh >> j) & 1) << 4));
    const float hi = static_cast<float>((q[j] >> 4) | (((qh >> (j + 16)) & 1) << 4));
    s += lo * x[j] + hi * x[16 + j];
    sx += x[j] + x[16 + j];
  }
  return F16(static_cast<std::uint16_t>(w.hd.x & 0xFFFF)) * s + F16(static_cast<std::uint16_t>(w.hd.x >> 16)) * sx;
}

}  // namespace

// halo_moe_gate_up_q4_k_dot_g over the experts of NB columns: block (row group, u) reads expert uexp[u]'s rows
// once and applies them to each of its pairs' activations (act[column]); h[pair][rows] (pair = column * k + slot).
#define HALO_MOE_GU_B(NB)                                                                                            \
  extern "C" __global__ void __launch_bounds__(256) halo_moe_gate_up_q4_k_dot_g_b##NB(                              \
      const std::uint8_t* gate, const std::uint8_t* up, const std::uint32_t* ucount, const std::uint32_t* uexp,      \
      const std::uint32_t* unum, const std::uint32_t* upair, const Q8Act* gact, float* h, std::uint64_t expert_bytes, \
      std::uint32_t rows, std::uint32_t cols, std::uint64_t row_bytes, std::uint32_t k) {                            \
    constexpr int kRows = 2;                                                                                         \
    __shared__ Q8Act act[NB];                                                                                        \
    const std::uint32_t u = blockIdx.y;                                                                              \
    if (u >= *ucount) return;                                                                                        \
    const std::uint32_t m = unum[u];                                                                                 \
    std::uint32_t pair[NB];                                                                                          \
    _Pragma("unroll") for (int j = 0; j < NB; ++j) pair[j] = j < static_cast<int>(m) ? upair[u * 8 + j] : 0;         \
    {                                                                                                                \
      constexpr std::uint32_t n16 = sizeof(Q8Act) / 16;                                                              \
      for (std::uint32_t j = 0; j < m; ++j) {                                                                        \
        const uint4* src = reinterpret_cast<const uint4*>(gact + pair[j] / k);                                       \
        uint4* dst = reinterpret_cast<uint4*>(&act[j]);                                                              \
        for (std::uint32_t i = threadIdx.x; i < n16; i += blockDim.x) dst[i] = src[i];                               \
      }                                                                                                              \
    }                                                                                                                \
    __syncthreads();                                                                                                 \
    const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                      \
    if (r0 >= rows) return;                                                                                          \
    const std::uint32_t lane = threadIdx.x % kWarp, per_row = cols / 64;                                             \
    const std::size_t e = uexp[u];                                                                                   \
    float g[NB][kRows] = {}, uu[NB][kRows] = {};                                                                     \
    for (std::uint32_t t = lane; t < 2 * kRows * per_row; t += kWarp) {                                              \
      const std::uint32_t mm = t / (kRows * per_row), rem = t % (kRows * per_row), kk0 = rem / per_row,              \
                          i = rem % per_row;                                                                         \
      const std::uint32_t r = min(r0 + kk0, rows - 1);                                                               \
      const Q4KPiece w = Q4KLoad((mm == 0 ? gate : up) + e * expert_bytes + r * row_bytes, i);                       \
      _Pragma("unroll") for (int j = 0; j < NB; ++j) {                                                               \
        if (j >= static_cast<int>(m)) break;                                                                         \
        const float v = Q4KDot(w, act[j], i);                                                                        \
        _Pragma("unroll") for (int kk = 0; kk < kRows; ++kk) {                                                       \
          if (kk == static_cast<int>(kk0) && mm == 0) g[j][kk] += v;                                                 \
          if (kk == static_cast<int>(kk0) && mm == 1) uu[j][kk] += v;                                                \
        }                                                                                                            \
      }                                                                                                              \
    }                                                                                                                \
    _Pragma("unroll") for (int j = 0; j < NB; ++j) {                                                                 \
      if (j >= static_cast<int>(m)) break;                                                                           \
      _Pragma("unroll") for (int kk = 0; kk < kRows; ++kk) {                                                         \
        const float gs = WarpSum(g[j][kk]), us = WarpSum(uu[j][kk]);                                                 \
        if (lane == 0 && r0 + kk < rows) h[static_cast<std::size_t>(pair[j]) * rows + r0 + kk] = Silu(gs) * us;     \
      }                                                                                                              \
    }                                                                                                                \
  }
HALO_FOR_NB(HALO_MOE_GU_B)
#undef HALO_MOE_GU_B

// halo_moe_down_q5_1_r4 grouped: block (row group, u); h[pair][cols] in, part[pair][rows] out.
#define HALO_MOE_DN51_B(NB)                                                                                          \
  extern "C" __global__ void __launch_bounds__(256) halo_moe_down_q5_1_r4_b##NB(                                    \
      const std::uint8_t* down, const std::uint32_t* ucount, const std::uint32_t* uexp, const std::uint32_t* unum,   \
      const std::uint32_t* upair, const float* h, float* part, std::uint64_t expert_bytes, std::uint32_t rows,       \
      std::uint32_t cols, std::uint64_t row_bytes) {                                                                 \
    constexpr int kRows = 4, kG = NB < 4 ? NB : 4; /* pairs a pass: more spill (weights re-read from cache) */       \
    __shared__ float xs[NB][1024];                                                                                   \
    const std::uint32_t u = blockIdx.y;                                                                              \
    if (u >= *ucount) return;                                                                                        \
    const std::uint32_t m = unum[u];                                                                                 \
    for (std::uint32_t j = 0; j < m; ++j) {                                                                          \
      const std::uint32_t pj = upair[u * 8 + j];                                                                     \
      for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[j][i] = h[static_cast<std::size_t>(pj) * cols + i]; \
    }                                                                                                                \
    __syncthreads();                                                                                                 \
    const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                      \
    if (r0 >= rows) return;                                                                                          \
    const std::uint32_t lane = threadIdx.x % kWarp, per_row = cols / 32;                                             \
    const std::uint8_t* w = down + uexp[u] * expert_bytes;                                                           \
    for (std::uint32_t g0 = 0; g0 < m; g0 += kG) {                                                                   \
      float acc[kG][kRows] = {};                                                                                     \
      for (std::uint32_t t = lane; t < kRows * per_row; t += kWarp) {                                                \
        const std::uint32_t k = t / per_row, i = t % per_row;                                                        \
        const std::uint32_t r = min(r0 + k, rows - 1);                                                               \
        const Q5_1Piece p = Q5_1Load(w + r * row_bytes, i);                                                          \
        _Pragma("unroll") for (int j = 0; j < kG; ++j) {                                                             \
          if (g0 + j >= m) break;                                                                                    \
          const float v = Q5_1Dot(p, xs[g0 + j], i);                                                                 \
          _Pragma("unroll") for (int kk = 0; kk < kRows; ++kk) if (kk == static_cast<int>(k)) acc[j][kk] += v;      \
        }                                                                                                            \
      }                                                                                                              \
      _Pragma("unroll") for (int j = 0; j < kG; ++j) {                                                               \
        if (g0 + j >= m) break;                                                                                      \
        const std::uint32_t pj = upair[u * 8 + g0 + j];                                                             \
        _Pragma("unroll") for (int kk = 0; kk < kRows; ++kk) {                                                       \
          const float a = WarpSum(acc[j][kk]);                                                                       \
          if (lane == 0 && r0 + kk < rows) part[static_cast<std::size_t>(pj) * rows + r0 + kk] = a;                 \
        }                                                                                                            \
      }                                                                                                              \
    }                                                                                                                \
  }
HALO_FOR_NB(HALO_MOE_DN51_B)
#undef HALO_MOE_DN51_B

// halo_moe_down_q8r_r4 grouped (GemvQ8rMulti<4, 6144>'s arithmetic; cols <= 1024).
#define HALO_MOE_DNQ8_B(NB)                                                                                          \
  extern "C" __global__ void __launch_bounds__(256) halo_moe_down_q8r_r4_b##NB(                                     \
      const std::int8_t* qs, const std::uint16_t* d, const std::uint32_t* ucount, const std::uint32_t* uexp,         \
      const std::uint32_t* unum, const std::uint32_t* upair, const float* h, float* part, std::uint32_t rows,         \
      std::uint32_t cols, std::uint32_t qstride) {                                                                   \
    constexpr int kRows = 4;                                                                                         \
    __shared__ float xs[NB][1024];                                                                                   \
    const std::uint32_t u = blockIdx.y;                                                                              \
    if (u >= *ucount) return;                                                                                        \
    const std::uint32_t m = unum[u];                                                                                 \
    std::uint32_t pair[NB];                                                                                          \
    _Pragma("unroll") for (int j = 0; j < NB; ++j) pair[j] = j < static_cast<int>(m) ? upair[u * 8 + j] : 0;         \
    for (std::uint32_t j = 0; j < m; ++j)                                                                            \
      for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[j][i] = h[static_cast<std::size_t>(pair[j]) * cols + i]; \
    __syncthreads();                                                                                                 \
    const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                      \
    if (r0 >= rows) return;                                                                                          \
    const std::size_t e = uexp[u];                                                                                   \
    const std::int8_t* q8 = qs + e * rows * qstride;                                                                 \
    const std::uint16_t* dd = d + e * rows * (cols / 32);                                                            \
    const std::uint32_t lane = threadIdx.x % kWarp, nblk = cols / 32;                                                \
    float acc[NB][kRows] = {};                                                                                       \
    for (std::uint32_t c = lane * 16; c < cols; c += kWarp * 16) {                                                   \
      int4 q[kRows];                                                                                                 \
      float s[kRows];                                                                                                \
      _Pragma("unroll") for (int k = 0; k < kRows; ++k) {                                                            \
        const std::uint32_t r = min(r0 + k, rows - 1);                                                               \
        q[k] = *reinterpret_cast<const int4*>(q8 + static_cast<std::size_t>(r) * qstride + c);                       \
        s[k] = F16(dd[static_cast<std::size_t>(r) * nblk + c / 32]);                                                 \
      }                                                                                                              \
      _Pragma("unroll") for (int j = 0; j < NB; ++j) {                                                               \
        if (j >= static_cast<int>(m)) break;                                                                         \
        float xr[16];                                                                                                \
        _Pragma("unroll") for (int i = 0; i < 16; ++i) xr[i] = xs[j][c + i];                                         \
        _Pragma("unroll") for (int k = 0; k < kRows; ++k) {                                                          \
          const std::int8_t* v = reinterpret_cast<const std::int8_t*>(&q[k]);                                       \
          float pp = 0.0f;                                                                                           \
          _Pragma("unroll") for (int i = 0; i < 16; ++i) pp += static_cast<float>(v[i]) * xr[i];                     \
          acc[j][k] += s[k] * pp;                                                                                    \
        }                                                                                                            \
      }                                                                                                              \
    }                                                                                                                \
    _Pragma("unroll") for (int j = 0; j < NB; ++j) {                                                                 \
      if (j >= static_cast<int>(m)) break;                                                                           \
      _Pragma("unroll") for (int k = 0; k < kRows; ++k) {                                                            \
        const float a = WarpSum(acc[j][k]);                                                                          \
        if (lane == 0 && r0 + k < rows) part[static_cast<std::size_t>(pair[j]) * rows + r0 + k] = a;                \
      }                                                                                                              \
    }                                                                                                                \
  }
HALO_FOR_NB(HALO_MOE_DNQ8_B)
#undef HALO_MOE_DNQ8_B

// halo_shexp_gate_up_u over NB columns: block 0 also writes each column's gate logit sgate[b].
#define HALO_SHEXP_GU_B(NB)                                                                                          \
  extern "C" __global__ void __launch_bounds__(256) halo_shexp_gate_up_u_b##NB(                                     \
      const std::int8_t* gq, const std::uint16_t* gd, const std::int8_t* uq, const std::uint16_t* ud,                \
      const float* gate_inp, const float* x, float* h, float* sgate, std::uint32_t rows, std::uint32_t cols,         \
      std::uint32_t qstride, std::uint32_t xstride, std::uint32_t hstride) {                                         \
    __shared__ float lds[kBatchLds];                                                                                 \
    __shared__ float scratch[32];                                                                                    \
    if (blockIdx.x == 0) {                                                                                           \
      float a[NB];                                                                                                   \
      RowDotHoistedB<NB>(gate_inp, x, xstride, cols, a);                                                             \
      _Pragma("unroll") for (int c = 0; c < NB; ++c) {                                                               \
        const float s = BlockSum(a[c], scratch);                                                                     \
        if (threadIdx.x == 0) sgate[c] = s;                                                                          \
      }                                                                                                              \
    }                                                                                                                \
    constexpr int kRows = 2;                                                                                         \
    const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                      \
    float g[NB][kRows], u[NB][kRows];                                                                                \
    Q8RowsDotB<kRows, NB>(gq, gd, x, xstride, lds, BatchChunk<NB>(), r0, r0 < rows, rows, cols, qstride, g);        \
    Q8RowsDotB<kRows, NB>(uq, ud, x, xstride, lds, BatchChunk<NB>(), r0, r0 < rows, rows, cols, qstride, u);        \
    if (r0 >= rows || threadIdx.x % kWarp != 0) return;                                                              \
    _Pragma("unroll") for (int c = 0; c < NB; ++c) _Pragma("unroll") for (int k = 0; k < kRows; ++k) if (r0 + k <   \
                                                                                                          rows)      \
        h[c * hstride + r0 + k] = Silu(g[c][k]) * u[c][k];                                                           \
  }
HALO_FOR_NB(HALO_SHEXP_GU_B)
#undef HALO_SHEXP_GU_B

// halo_shexp_down_combine_u over NB columns: column b's routed partials part[b * slots + s][rows], weights
// [b * slots + s], gate logit sgate[b], shared hidden hin[b * hstride], output out[b * ostride].
#define HALO_SHEXP_DC_B(NB)                                                                                          \
  extern "C" __global__ void __launch_bounds__(256) halo_shexp_down_combine_u_b##NB(                                \
      const std::int8_t* dq, const std::uint16_t* dd, const float* hin, const float* sgate, const float* part,       \
      const float* weights, std::uint32_t slots, float* out, std::uint32_t rows, std::uint32_t cols,                 \
      std::uint32_t qstride, std::uint32_t hstride, std::uint32_t ostride) {                                         \
    __shared__ float lds[kBatchLds];                                                                                 \
    constexpr int kRows = 4;                                                                                         \
    const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                      \
    const std::uint32_t lane = threadIdx.x % kWarp;                                                                  \
    float y[NB][kRows];                                                                                              \
    Q8RowsDotB<kRows, NB>(dq, dd, hin, hstride, lds, BatchChunk<NB>(), r0, r0 < rows, rows, cols, qstride, y);      \
    if (r0 >= rows) return;                                                                                          \
    _Pragma("unroll") for (int c = 0; c < NB; ++c) {                                                                 \
      const float* pc = part + static_cast<std::size_t>(c) * slots * rows;                                           \
      const float wl = lane < slots ? weights[c * slots + lane] : 0.0f;                                              \
      float pv[kRows];                                                                                               \
      _Pragma("unroll") for (int k = 0; k < kRows; ++k) pv[k] =                                                      \
          lane < slots && r0 + k < rows ? pc[static_cast<std::size_t>(lane) * rows + r0 + k] : 0.0f;                 \
      const float s = Sigmoid(sgate[c]);                                                                             \
      _Pragma("unroll") for (int k = 0; k < kRows; ++k) {                                                            \
        float acc = 0.0f;                                                                                            \
        for (std::uint32_t sl = 0; sl < slots; ++sl)                                                                 \
          acc = __builtin_fmaf(__shfl(wl, sl, kWarp), __shfl(pv[k], sl, kWarp), acc);                                \
        if (lane == 0 && r0 + k < rows) out[c * ostride + r0 + k] = acc + s * y[c][k];                               \
      }                                                                                                              \
    }                                                                                                                \
  }
HALO_FOR_NB(HALO_SHEXP_DC_B)
#undef HALO_SHEXP_DC_B

// ---- Hyper-connection mixers over NB columns -------------------------------------------------------------------
// halo_hc_mix_a_cu: column b's residual res[b * D], norm scales scale[b * hc], outputs xn[b * D],
// lo_part[b * parts * low_rank], inject_part[b * parts * hc]. The chunk of every column is staged (NB * chunk floats).
#define HALO_MIX_A_B(NB)                                                                                             \
  extern "C" __global__ void __launch_bounds__(256) halo_hc_mix_a_cu_b##NB(                                         \
      const float* res, const float* norm_w, const float* scale, const std::int8_t* dqs, const std::uint16_t* dd,    \
      const float* inject_w, float* xn, float* lo_part, float* inject_part, std::uint32_t hc, std::uint32_t h,       \
      std::uint32_t low_rank, std::uint32_t down_blocks, std::uint32_t qstride, std::uint32_t chunk,                 \
      std::uint32_t parts) {                                                                                         \
    __shared__ float xsc[NB * 1536];                                                                                 \
    __shared__ float scratch[32];                                                                                    \
    const std::uint32_t D = hc * h, ky = blockIdx.y, c0 = ky * chunk, c1 = min(D, c0 + chunk);                       \
    constexpr std::uint32_t kMax = 8;                                                                                \
    _Pragma("unroll") for (int b = 0; b < NB; ++b) {                                                                 \
      float rv[kMax], sv[kMax], nv[kMax];                                                                            \
      _Pragma("unroll") for (std::uint32_t j = 0; j < kMax; ++j) {                                                   \
        const std::uint32_t i = c0 + threadIdx.x + j * blockDim.x;                                                   \
        const bool in = i < c1;                                                                                      \
        rv[j] = in ? res[b * D + i] : 0.0f;                                                                          \
        sv[j] = in ? scale[b * hc + i / h] : 0.0f;                                                                   \
        nv[j] = in ? norm_w[i] : 0.0f;                                                                               \
      }                                                                                                              \
      _Pragma("unroll") for (std::uint32_t j = 0; j < kMax; ++j) {                                                   \
        const std::uint32_t i = c0 + threadIdx.x + j * blockDim.x;                                                   \
        if (i < c1) xsc[b * chunk + i - c0] = rv[j] * sv[j] * nv[j];                                                 \
      }                                                                                                              \
    }                                                                                                                \
    __syncthreads();                                                                                                 \
    if (blockIdx.x == 0)                                                                                             \
      _Pragma("unroll") for (int b = 0; b < NB; ++b) for (std::uint32_t i = c0 + threadIdx.x; i < c1;              \
                                                          i += blockDim.x) xn[b * D + i] = xsc[b * chunk + i - c0];  \
    const std::uint32_t lane = threadIdx.x % kWarp;                                                                  \
    if (blockIdx.x >= down_blocks) {                                                                                 \
      const std::uint32_t r = blockIdx.x - down_blocks;                                                              \
      const float* w = inject_w + static_cast<std::size_t>(r) * D;                                                   \
      float wv[kMax];                                                                                                \
      _Pragma("unroll") for (std::uint32_t j = 0; j < kMax; ++j) {                                                   \
        const std::uint32_t cc = c0 + threadIdx.x + j * blockDim.x;                                                  \
        wv[j] = cc < c1 ? w[cc] : 0.0f;                                                                              \
      }                                                                                                              \
      _Pragma("unroll") for (int b = 0; b < NB; ++b) {                                                               \
        float acc = 0.0f;                                                                                            \
        _Pragma("unroll") for (std::uint32_t j = 0; j < kMax; ++j) {                                                 \
          const std::uint32_t cc = c0 + threadIdx.x + j * blockDim.x;                                                \
          if (cc < c1) acc = __builtin_fmaf(wv[j], xsc[b * chunk + cc - c0], acc);                                   \
        }                                                                                                            \
        acc = BlockSum(acc, scratch);                                                                                \
        if (threadIdx.x == 0) inject_part[b * parts * hc + ky * hc + r] = acc;                                       \
      }                                                                                                              \
      return;                                                                                                        \
    }                                                                                                                \
    constexpr int kRows = 2;                                                                                         \
    const std::uint32_t r0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                      \
    if (r0 >= low_rank) return;                                                                                      \
    const std::uint32_t nblk = D / 32;                                                                               \
    float acc[NB][kRows] = {};                                                                                       \
    for (std::uint32_t c = c0 + lane * 16; c < c1; c += kWarp * 16) {                                                \
      int4 q[kRows];                                                                                                 \
      float sc[kRows];                                                                                               \
      _Pragma("unroll") for (int k = 0; k < kRows; ++k) {                                                            \
        const std::uint32_t r = min(r0 + k, low_rank - 1);                                                           \
        q[k] = *reinterpret_cast<const int4*>(dqs + static_cast<std::size_t>(r) * qstride + c);                      \
        sc[k] = F16(dd[static_cast<std::size_t>(r) * nblk + c / 32]);                                                \
      }                                                                                                              \
      _Pragma("unroll") for (int b = 0; b < NB; ++b) {                                                               \
        float xr[16];                                                                                                \
        _Pragma("unroll") for (int i = 0; i < 16; ++i) xr[i] = xsc[b * chunk + c - c0 + i];                          \
        _Pragma("unroll") for (int k = 0; k < kRows; ++k) {                                                          \
          const std::int8_t* v = reinterpret_cast<const std::int8_t*>(&q[k]);                                       \
          float part = 0.0f;                                                                                         \
          _Pragma("unroll") for (int i = 0; i < 16; ++i) part += static_cast<float>(v[i]) * xr[i];                   \
          acc[b][k] += sc[k] * part;                                                                                 \
        }                                                                                                            \
      }                                                                                                              \
    }                                                                                                                \
    _Pragma("unroll") for (int b = 0; b < NB; ++b) _Pragma("unroll") for (int k = 0; k < kRows; ++k) {             \
      const float a = WarpSum(acc[b][k]);                                                                            \
      if (lane == 0 && r0 + k < low_rank) lo_part[b * parts * low_rank + ky * low_rank + r0 + k] = a;               \
    }                                                                                                                \
  }
HALO_FOR_NB(HALO_MIX_A_B)
#undef HALO_MIX_A_B

// halo_hc_mix_b_u over NB columns: lo_part / inject_part / inject / xn / mixed of column b at their per-column
// strides (parts * low_rank, parts * hc, hc, hc * h, h).
#define HALO_MIX_B_B(NB)                                                                                             \
  extern "C" __global__ void __launch_bounds__(256) halo_hc_mix_b_u_b##NB(                                          \
      const float* lo_part, std::uint32_t parts, const float* inject_part, float* inject, std::uint32_t inject_rows, \
      const std::int8_t* uqs, const std::uint16_t* ud, const float* xn, float* mixed, std::uint32_t hc,              \
      std::uint32_t h, std::uint32_t low_rank, std::uint32_t qstride) {                                              \
    __shared__ float ls[NB][512];                                                                                    \
    const std::uint32_t i = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;                                 \
    const std::uint32_t lane = threadIdx.x % kWarp, nblk = low_rank / 32, c = lane * 16;                             \
    const bool live = i < h && c < low_rank;                                                                         \
    constexpr std::uint32_t kMaxParts = 16;                                                                          \
    _Pragma("unroll") for (int b = 0; b < NB; ++b) {                                                                 \
      const float* lp = lo_part + static_cast<std::size_t>(b) * parts * low_rank;                                    \
      _Pragma("unroll") for (std::uint32_t jj = 0; jj < 2; ++jj) {                                                   \
        const std::uint32_t j = threadIdx.x + jj * blockDim.x;                                                       \
        float pv[kMaxParts];                                                                                         \
        _Pragma("unroll") for (std::uint32_t p = 0; p < kMaxParts; ++p) pv[p] =                                      \
            j < low_rank && p < parts ? lp[p * low_rank + j] : 0.0f;                                                 \
        if (j < low_rank) {                                                                                          \
          float a = 0.0f;                                                                                            \
          _Pragma("unroll") for (std::uint32_t p = 0; p < kMaxParts; ++p) if (p < parts) a += pv[p];                 \
          ls[b][j] = Silu(a / static_cast<float>(hc));                                                               \
        }                                                                                                            \
      }                                                                                                              \
      if (blockIdx.x == 0 && threadIdx.x < inject_rows) {                                                            \
        const float* ip = inject_part + static_cast<std::size_t>(b) * parts * hc;                                    \
        float a = 0.0f;                                                                                              \
        for (std::uint32_t p = 0; p < parts; ++p) a += ip[p * hc + threadIdx.x];                                     \
        inject[b * hc + threadIdx.x] = a;                                                                            \
      }                                                                                                              \
    }                                                                                                                \
    __syncthreads();                                                                                                 \
    if (i >= h) return;                                                                                              \
    int4 q[4];                                                                                                       \
    float sc[4];                                                                                                     \
    _Pragma("unroll") for (int s = 0; s < 4; ++s) {                                                                  \
      const std::size_t r = static_cast<std::size_t>(min(static_cast<std::uint32_t>(s), hc - 1)) * h + min(i, h - 1); \
      const std::uint32_t cc = live ? c : 0;                                                                         \
      q[s] = *reinterpret_cast<const int4*>(uqs + r * qstride + cc);                                                 \
      sc[s] = F16(ud[r * nblk + cc / 32]);                                                                           \
    }                                                                                                                \
    _Pragma("unroll") for (int b = 0; b < NB; ++b) {                                                                 \
      float acc[4] = {};                                                                                             \
      if (c < low_rank) {                                                                                            \
        float xr[16];                                                                                                \
        _Pragma("unroll") for (int j = 0; j < 16; ++j) xr[j] = ls[b][c + j];                                         \
        _Pragma("unroll") for (int s = 0; s < 4; ++s) {                                                              \
          if (static_cast<std::uint32_t>(s) >= hc) break;                                                            \
          const std::int8_t* v = reinterpret_cast<const std::int8_t*>(&q[s]);                                       \
          float part = 0.0f;                                                                                         \
          _Pragma("unroll") for (int j = 0; j < 16; ++j) part += static_cast<float>(v[j]) * xr[j];                   \
          acc[s] += sc[s] * part;                                                                                    \
        }                                                                                                            \
      }                                                                                                              \
      float m = 0.0f;                                                                                                \
      _Pragma("unroll") for (int s = 0; s < 4; ++s) {                                                                \
        const float g = WarpSum(acc[s]);                                                                             \
        if (static_cast<std::uint32_t>(s) < hc) m += xn[static_cast<std::size_t>(b) * hc * h + s * h + i] * Sigmoid(g); \
      }                                                                                                              \
      if (lane == 0) mixed[b * h + i] = m / static_cast<float>(hc);                                                  \
    }                                                                                                                \
  }
HALO_FOR_NB(HALO_MIX_B_B)
#undef HALO_MIX_B_B
