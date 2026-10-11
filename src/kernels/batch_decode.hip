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
template <int kRows, int NB, bool kPrefetch = false>
__device__ inline void Q8RowsDotB(const std::int8_t* qs, const std::uint16_t* d, const float* x, std::uint32_t xstride,
                                  float* lds, std::uint32_t chunk, std::uint32_t r0, bool active, std::uint32_t rows,
                                  std::uint32_t cols, std::uint32_t qstride, float (&out)[NB][kRows]) {
  const std::uint32_t lane = threadIdx.x % kWarp, nblk = cols / 32;
  float acc[NB][kRows] = {};
  // kPrefetch: the next piece's weights are loaded before the current piece's arithmetic (one piece in flight a
  // lane while computing); the arithmetic and its order are unchanged.
  int4 qn[kRows];
  float sn[kRows];
  const auto load = [&](std::uint32_t c, int4 (&q)[kRows], float (&sc)[kRows]) {
#pragma unroll
    for (int k = 0; k < kRows; ++k) {
      const std::uint32_t r = min(r0 + k, rows - 1);
      q[k] = *reinterpret_cast<const int4*>(qs + static_cast<std::size_t>(r) * qstride + c);
      sc[k] = F16(d[static_cast<std::size_t>(r) * nblk + c / 32]);
    }
  };
  if (kPrefetch && active && lane * 16 < cols) load(lane * 16, qn, sn);
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
      if (kPrefetch) {
#pragma unroll
        for (int k = 0; k < kRows; ++k) q[k] = qn[k], sc[k] = sn[k];
        if (c + kWarp * 16 < cols) load(c + kWarp * 16, qn, sn);
      } else {
        load(c, q, sc);
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
template <int NB, std::uint32_t kLds = kBatchLds>
constexpr std::uint32_t BatchChunk() {
  return (kLds / NB) / (kWarp * 16) * (kWarp * 16) > 0 ? (kLds / NB) / (kWarp * 16) * (kWarp * 16) : kWarp * 16;
}

template <int kRows, int NB, bool kPrefetch = false, std::uint32_t kLds = kBatchLds>
__device__ inline void GemvQ8rB(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y,
                                std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride, std::uint32_t xstride,
                                std::uint32_t ystride, std::uint32_t groups) {
  __shared__ float lds[kLds];
  // Column groups of NB interleaved with the row blocks: neighbouring blocks read the same rows (cache hits).
  const std::uint32_t bx = blockIdx.x / groups, g = blockIdx.x % groups;
  x += g * NB * xstride;
  y += g * NB * ystride;
  const std::uint32_t r0 = (bx * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  float out[NB][kRows];
  Q8RowsDotB<kRows, NB, kPrefetch>(qs, d, x, xstride, lds, BatchChunk<NB, kLds>(), r0, r0 < rows, rows, cols, qstride,
                                   out);
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
#define HALO_FOR_NB(X) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8)

// halo_gemv_q8r_r4 / _r2.
#define HALO_GEMV_BV(NB, SUF, PF, LDS)                                                                               \
  extern "C" __global__ void __launch_bounds__(256)                                                                  \
      halo_gemv_q8r_r4_b##NB##SUF(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y,           \
                                  std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride,                     \
                                  std::uint32_t xstride, std::uint32_t ystride, std::uint32_t groups) {              \
    GemvQ8rB<4, NB, PF, LDS>(qs, d, x, y, rows, cols, qstride, xstride, ystride, groups);                            \
  }                                                                                                                  \
  extern "C" __global__ void __launch_bounds__(256)                                                                  \
      halo_gemv_q8r_r2_b##NB##SUF(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* y,           \
                                  std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride,                     \
                                  std::uint32_t xstride, std::uint32_t ystride, std::uint32_t groups) {              \
    GemvQ8rB<2, NB, PF, LDS>(qs, d, x, y, rows, cols, qstride, xstride, ystride, groups);                            \
  }
// Variants: plain; _p weights prefetched a piece ahead; _s 16 KB of LDS (more resident blocks); _ps both.
#define HALO_GEMV_B(NB)                                                                                              \
  HALO_GEMV_BV(NB, , false, kBatchLds) HALO_GEMV_BV(NB, _p, true, kBatchLds) HALO_GEMV_BV(NB, _s, false, 4096u)     \
      HALO_GEMV_BV(NB, _ps, true, 4096u)
HALO_FOR_NB(HALO_GEMV_B)
#undef HALO_GEMV_B
#undef HALO_GEMV_BV

// halo_gemv_q8r_dual (DeltaNet qkv + z).
#define HALO_DUAL_BV(NB, SUF, PF, LDS)                                                                                              \
  extern "C" __global__ void __launch_bounds__(256) halo_gemv_q8r_dual_b##NB##SUF(                                       \
      const std::int8_t* q1, const std::uint16_t* d1, float* y1, std::uint32_t rows1, std::uint32_t qstride1,        \
      const std::int8_t* q2, const std::uint16_t* d2, float* y2, std::uint32_t rows2, std::uint32_t qstride2,        \
      const float* x, std::uint32_t cols, std::uint32_t blocks1, std::uint32_t xstride, std::uint32_t ystride1,      \
      std::uint32_t ystride2, std::uint32_t groups) {                                                                \
    __shared__ float lds[LDS];                                                                                 \
    const std::uint32_t bx = blockIdx.x / groups, grp = blockIdx.x % groups;                                         \
    x += grp * NB * xstride;                                                                                         \
    y1 += grp * NB * ystride1;                                                                                       \
    y2 += grp * NB * ystride2;                                                                                       \
    const bool first = bx < blocks1;                                                                                 \
    const std::uint32_t b = first ? bx : bx - blocks1;                                                               \
    const std::int8_t* qs = first ? q1 : q2;                                                                         \
    const std::uint16_t* d = first ? d1 : d2;                                                                        \
    float* y = first ? y1 : y2;                                                                                      \
    const std::uint32_t rows = first ? rows1 : rows2, qstride = first ? qstride1 : qstride2;                         \
    const std::uint32_t ystride = first ? ystride1 : ystride2;                                                       \
    constexpr int kRows = 4;                                                                                         \
    const std::uint32_t r0 = (b * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                               \
    float acc[NB][kRows];                                                                                            \
    Q8RowsDotB<kRows, NB, PF>(qs, d, x, xstride, lds, BatchChunk<NB, LDS>(), r0, r0 < rows, rows, cols, qstride, acc);       \
    if (r0 >= rows || threadIdx.x % kWarp != 0) return;                                                              \
    _Pragma("unroll") for (int c = 0; c < NB; ++c) _Pragma("unroll") for (int k = 0; k < kRows; ++k) if (r0 + k <   \
                                                                                                          rows)      \
        y[c * ystride + r0 + k] = acc[c][k];                                                                         \
  }
#define HALO_DUAL_B(NB)                                                                                              \
  HALO_DUAL_BV(NB, , false, kBatchLds) HALO_DUAL_BV(NB, _p, true, kBatchLds) HALO_DUAL_BV(NB, _s, false, 4096u)     \
      HALO_DUAL_BV(NB, _ps, true, 4096u)
HALO_FOR_NB(HALO_DUAL_B)
#undef HALO_DUAL_B
#undef HALO_DUAL_BV

// ---- Relaxed batched dense kernels (B >= 2, docs/SESSIONS.md 3 "Relaxing exactness") ------------------------------
// The columns' activations quantized to int8 per 32 values (as the routed experts' Q8Act); each lane's 16 weights of a
// row one v_dot4 chain per column, scaled per 32-block by weight scale x activation scale. Not byte-identical to the
// single-token kernels: gated by halo_score --batch against the single-token scores.

// Quantizes B columns of x (column stride xstride floats; cols % 32 == 0) to int8 xq (column stride cols bytes) and
// per-32 scales xs (column stride cols / 32). Grid (ceil(cols / 256), B), 256 threads: a warp per 32 values.
// With xq2 / xs2 (the residual plane, same layout): the residual x - q * s quantized again, so that
// x ~ q * s + q2 * s2 (two int8 levels, ~15 bits).
extern "C" __global__ void __launch_bounds__(256)
    halo_q8x_cols(const float* x, std::uint32_t xstride, std::uint32_t cols, std::int8_t* xq, float* xs,
                  std::int8_t* xq2, float* xs2) {
  const std::uint32_t b = blockIdx.y, lane = threadIdx.x % kWarp, nblk = cols / 32;
  const std::uint32_t blk = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
  if (blk >= nblk) return;
  const float v = x[static_cast<std::size_t>(b) * xstride + blk * 32 + lane];
  float amax = fabsf(v);
  for (int o = kWarp / 2; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, kWarp));
  const float scale = amax / 127.0f;
  const int qv = scale > 0.0f ? static_cast<int>(rintf(v / scale)) : 0;
  xq[static_cast<std::size_t>(b) * cols + blk * 32 + lane] = static_cast<std::int8_t>(qv);
  if (lane == 0) xs[static_cast<std::size_t>(b) * nblk + blk] = scale;
  if (!xq2) return;
  const float r = v - static_cast<float>(qv) * scale;
  float rmax = fabsf(r);
  for (int o = kWarp / 2; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor(rmax, o, kWarp));
  const float s2 = rmax / 127.0f;
  const int q2 = s2 > 0.0f ? static_cast<int>(rintf(r / s2)) : 0;
  xq2[static_cast<std::size_t>(b) * cols + blk * 32 + lane] = static_cast<std::int8_t>(q2);
  if (lane == 0) xs2[static_cast<std::size_t>(b) * nblk + blk] = s2;
}

namespace {
template <int kRows, int NB, bool kResid = false>
__device__ inline void Q8RowsDotI8B(const std::int8_t* qs, const std::uint16_t* d, const std::int8_t* xq,
                                    const float* xs, const std::int8_t* xq2, const float* xs2, std::uint32_t r0,
                                    bool active, std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride,
                                    float (&out)[NB][kRows]) {
  const std::uint32_t lane = threadIdx.x % kWarp, nblk = cols / 32;
  float acc[NB][kRows] = {};
  if (active)
    for (std::uint32_t c = lane * 16; c < cols; c += kWarp * 16) {
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
        const int4 xv = *reinterpret_cast<const int4*>(xq + static_cast<std::size_t>(b) * cols + c);
        const float s = xs[static_cast<std::size_t>(b) * nblk + c / 32];
#pragma unroll
        for (int k = 0; k < kRows; ++k) {
          int i = __builtin_amdgcn_sudot4(true, q[k].x, true, xv.x, 0, false);
          i = __builtin_amdgcn_sudot4(true, q[k].y, true, xv.y, i, false);
          i = __builtin_amdgcn_sudot4(true, q[k].z, true, xv.z, i, false);
          i = __builtin_amdgcn_sudot4(true, q[k].w, true, xv.w, i, false);
          float part = s * static_cast<float>(i);
          if constexpr (kResid) {
            const int4 rv = *reinterpret_cast<const int4*>(xq2 + static_cast<std::size_t>(b) * cols + c);
            int j = __builtin_amdgcn_sudot4(true, q[k].x, true, rv.x, 0, false);
            j = __builtin_amdgcn_sudot4(true, q[k].y, true, rv.y, j, false);
            j = __builtin_amdgcn_sudot4(true, q[k].z, true, rv.z, j, false);
            j = __builtin_amdgcn_sudot4(true, q[k].w, true, rv.w, j, false);
            part += xs2[static_cast<std::size_t>(b) * nblk + c / 32] * static_cast<float>(j);
          }
          acc[b][k] += sc[k] * part;
        }
      }
    }
#pragma unroll
  for (int b = 0; b < NB; ++b)
#pragma unroll
    for (int k = 0; k < kRows; ++k) out[b][k] = WarpSum(acc[b][k]);
}

// `groups` column groups of NB interleaved with the row blocks (more blocks for few rows).
template <int kRows, int NB, bool kResid>
__device__ inline void GemvQ8rI8B(const std::int8_t* qs, const std::uint16_t* d, const std::int8_t* xq, const float* xs,
                                  const std::int8_t* xq2, const float* xs2, float* y, std::uint32_t rows,
                                  std::uint32_t cols, std::uint32_t qstride, std::uint32_t ystride,
                                  std::uint32_t groups) {
  const std::uint32_t bx = blockIdx.x / groups, g = blockIdx.x % groups;
  xq += static_cast<std::size_t>(g) * NB * cols;
  xs += static_cast<std::size_t>(g) * NB * (cols / 32);
  if constexpr (kResid) {
    xq2 += static_cast<std::size_t>(g) * NB * cols;
    xs2 += static_cast<std::size_t>(g) * NB * (cols / 32);
  }
  y += static_cast<std::size_t>(g) * NB * ystride;
  const std::uint32_t r0 = (bx * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;
  float out[NB][kRows];
  Q8RowsDotI8B<kRows, NB, kResid>(qs, d, xq, xs, xq2, xs2, r0, r0 < rows, rows, cols, qstride, out);
  if (r0 >= rows || threadIdx.x % kWarp != 0) return;
#pragma unroll
  for (int b = 0; b < NB; ++b)
#pragma unroll
    for (int k = 0; k < kRows; ++k)
      if (r0 + k < rows) y[b * ystride + r0 + k] = out[b][k];
}
}  // namespace

#define HALO_GEMV_I8_ROWS_V(NB, R, SUF, RES)                                                                        \
  extern "C" __global__ void __launch_bounds__(256)                                                                  \
      halo_gemv_q8r_i8##SUF##_r##R##_b##NB(const std::int8_t* qs, const std::uint16_t* d, const std::int8_t* xq,    \
                                           const float* xs, const std::int8_t* xq2, const float* xs2, float* y,      \
                                           std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride,            \
                                           std::uint32_t ystride, std::uint32_t groups) {                            \
    GemvQ8rI8B<R, NB, RES>(qs, d, xq, xs, xq2, xs2, y, rows, cols, qstride, ystride, groups);                        \
  }
#define HALO_GEMV_I8_ROWS(NB, R) HALO_GEMV_I8_ROWS_V(NB, R, , false) HALO_GEMV_I8_ROWS_V(NB, R, x, true)
#define HALO_GEMV_I8_B(NB)                                                                                           \
  HALO_GEMV_I8_ROWS(NB, 4) HALO_GEMV_I8_ROWS(NB, 2) HALO_GEMV_I8_ROWS(NB, 1)                                         \
  HALO_GEMV_I8_DUAL(NB, 4) HALO_GEMV_I8_DUAL(NB, 2) HALO_GEMV_I8_DUAL(NB, 1)
#define HALO_GEMV_I8_DUAL(NB, R) HALO_GEMV_I8_DUAL_V(NB, R, , false) HALO_GEMV_I8_DUAL_V(NB, R, x, true)
#define HALO_GEMV_I8_DUAL_V(NB, R, SUF, RES)                                                                        \
  extern "C" __global__ void __launch_bounds__(256) halo_gemv_q8r_i8##SUF##_dual_r##R##_b##NB(                       \
      const std::int8_t* q1, const std::uint16_t* d1, float* y1, std::uint32_t rows1, std::uint32_t qstride1,        \
      const std::int8_t* q2, const std::uint16_t* d2, float* y2, std::uint32_t rows2, std::uint32_t qstride2,        \
      const std::int8_t* xq, const float* xs, const std::int8_t* xq2, const float* xs2, std::uint32_t cols,          \
      std::uint32_t blocks1, std::uint32_t ystride1, std::uint32_t ystride2, std::uint32_t groups) {                 \
    const std::uint32_t bxg = blockIdx.x / groups, g = blockIdx.x % groups;                                          \
    xq += static_cast<std::size_t>(g) * NB * cols;                                                                   \
    xs += static_cast<std::size_t>(g) * NB * (cols / 32);                                                            \
    if (RES) {                                                                                                       \
      xq2 += static_cast<std::size_t>(g) * NB * cols;                                                                \
      xs2 += static_cast<std::size_t>(g) * NB * (cols / 32);                                                         \
    }                                                                                                                \
    const bool first = bxg < blocks1;                                                                                \
    const std::uint32_t bx = first ? bxg : bxg - blocks1;                                                            \
    const std::uint32_t rows = first ? rows1 : rows2;                                                                \
    const std::uint32_t r0 = (bx * (blockDim.x / kWarp) + threadIdx.x / kWarp) * R;                                  \
    float out[NB][R];                                                                                                \
    Q8RowsDotI8B<R, NB, RES>(first ? q1 : q2, first ? d1 : d2, xq, xs, xq2, xs2, r0, r0 < rows, rows, cols,        \
                        first ? qstride1 : qstride2, out);                                                           \
    if (r0 >= rows || threadIdx.x % kWarp != 0) return;                                                              \
    float* y = (first ? y1 : y2) + static_cast<std::size_t>(g) * NB * (first ? ystride1 : ystride2);                \
    const std::uint32_t ystride = first ? ystride1 : ystride2;                                                       \
    _Pragma("unroll") for (int b = 0; b < NB; ++b) _Pragma("unroll") for (int k = 0; k < R; ++k) if (r0 + k < rows) \
        y[b * ystride + r0 + k] = out[b][k];                                                                         \
  }
HALO_FOR_NB(HALO_GEMV_I8_B)
#undef HALO_GEMV_I8_B
#undef HALO_GEMV_I8_DUAL
#undef HALO_GEMV_I8_DUAL_V
#undef HALO_GEMV_I8_ROWS
#undef HALO_GEMV_I8_ROWS_V

// halo_matvec_f32_rowblock2_u (DeltaNet alpha and beta): a block per row; y strides are the outputs' strides.
#define HALO_ROWBLOCK2_B(NB)                                                                                         \
  extern "C" __global__ void __launch_bounds__(256) halo_matvec_f32_rowblock2_u_b##NB(                              \
      const float* wa, const float* wb, const float* x, float* ya, float* yb, std::uint32_t rows,                    \
      std::uint32_t cols, std::uint32_t xstride, std::uint32_t ystride, std::uint32_t groups) {                      \
    __shared__ float scratch[32];                                                                                    \
    const std::uint32_t bx = blockIdx.x / groups, grp = blockIdx.x % groups;                                         \
    x += grp * NB * xstride;                                                                                         \
    ya += grp * NB * ystride;                                                                                        \
    yb += grp * NB * ystride;                                                                                        \
    const bool a = bx < rows;                                                                                        \
    const std::uint32_t r = a ? bx : bx - rows;                                                                      \
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
      std::uint32_t xstride, std::uint32_t ystride, std::uint32_t groups) {                                          \
    __shared__ float scratch[32];                                                                                    \
    const std::uint32_t bx = blockIdx.x / groups, grp = blockIdx.x % groups;                                         \
    x += grp * NB * xstride;                                                                                         \
    y += grp * NB * ystride;                                                                                         \
    act += grp * NB;                                                                                                 \
    if (bx < cols / 32 && threadIdx.x < kWarp) {                                                                     \
      const std::uint32_t lane = threadIdx.x, blk = bx;                                                              \
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
    const std::uint32_t r = bx;                                                                                      \
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
// The same with the 32 weights converted once (shared by an expert's pairs); identical values and sums.
struct Q5_1Vals {
  float lo[16], hi[16], d, m;
};
__device__ inline Q5_1Vals Q5_1Deq(const Q5_1Piece& w) {
  Q5_1Vals v;
  const std::uint32_t qh = w.hd.y;
  std::uint8_t q[16];
  *reinterpret_cast<uint2*>(q) = w.qa;
  *reinterpret_cast<uint2*>(q + 8) = w.qb;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    v.lo[j] = static_cast<float>((q[j] & 0xF) | (((qh >> j) & 1) << 4));
    v.hi[j] = static_cast<float>((q[j] >> 4) | (((qh >> (j + 16)) & 1) << 4));
  }
  v.d = F16(static_cast<std::uint16_t>(w.hd.x & 0xFFFF));
  v.m = F16(static_cast<std::uint16_t>(w.hd.x >> 16));
  return v;
}
__device__ inline float Q5_1DotV(const Q5_1Vals& v, const float* xs, std::uint32_t item) {
  const float* x = xs + item * 32;
  float s = 0, sx = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    s += v.lo[j] * x[j] + v.hi[j] * x[16 + j];
    sx += x[j] + x[16 + j];
  }
  return v.d * s + v.m * sx;
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
        const Q5_1Vals p = Q5_1Deq(Q5_1Load(w + r * row_bytes, i));                                                 \
        _Pragma("unroll") for (int j = 0; j < kG; ++j) {                                                             \
          if (g0 + j >= m) break;                                                                                    \
          const float v = Q5_1DotV(p, xs[g0 + j], i);                                                                 \
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
      std::uint32_t qstride, std::uint32_t xstride, std::uint32_t hstride, std::uint32_t groups) {                   \
    __shared__ float lds[kBatchLds];                                                                                 \
    __shared__ float scratch[32];                                                                                    \
    const std::uint32_t bx = blockIdx.x / groups, grp = blockIdx.x % groups;                                         \
    x += grp * NB * xstride;                                                                                         \
    h += grp * NB * hstride;                                                                                         \
    sgate += grp * NB;                                                                                               \
    if (bx == 0) {                                                                                                   \
      float a[NB];                                                                                                   \
      RowDotHoistedB<NB>(gate_inp, x, xstride, cols, a);                                                             \
      _Pragma("unroll") for (int c = 0; c < NB; ++c) {                                                               \
        const float s = BlockSum(a[c], scratch);                                                                     \
        if (threadIdx.x == 0) sgate[c] = s;                                                                          \
      }                                                                                                              \
    }                                                                                                                \
    constexpr int kRows = 2;                                                                                         \
    const std::uint32_t r0 = (bx * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                              \
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
      std::uint32_t qstride, std::uint32_t hstride, std::uint32_t ostride, std::uint32_t groups) {                   \
    __shared__ float lds[kBatchLds];                                                                                 \
    constexpr int kRows = 4;                                                                                         \
    const std::uint32_t bx = blockIdx.x / groups, grp = blockIdx.x % groups;                                         \
    hin += grp * NB * hstride;                                                                                       \
    sgate += grp * NB;                                                                                               \
    part += static_cast<std::size_t>(grp) * NB * slots * rows;                                                       \
    weights += grp * NB * slots;                                                                                     \
    out += grp * NB * ostride;                                                                                       \
    const std::uint32_t r0 = (bx * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                              \
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
      std::uint32_t parts, std::uint32_t groups) {                                                                                         \
    __shared__ float xsc[NB * 1536];                                                                                 \
    {                                                                                                                \
      const std::uint32_t c0z = (blockIdx.x % groups) * NB; /* column group, interleaved with x */                                                  \
      res += static_cast<std::size_t>(c0z) * hc * h;                                                                 \
      scale += c0z * hc;                                                                                             \
      xn += static_cast<std::size_t>(c0z) * hc * h;                                                                  \
      lo_part += static_cast<std::size_t>(c0z) * parts * low_rank;                                                   \
      inject_part += static_cast<std::size_t>(c0z) * parts * hc;                                                     \
    }                                                                                 \
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
    if (blockIdx.x / groups == 0)                                                                                             \
      _Pragma("unroll") for (int b = 0; b < NB; ++b) for (std::uint32_t i = c0 + threadIdx.x; i < c1;              \
                                                          i += blockDim.x) xn[b * D + i] = xsc[b * chunk + i - c0];  \
    const std::uint32_t lane = threadIdx.x % kWarp;                                                                  \
    if (blockIdx.x / groups >= down_blocks) {                                                                                 \
      const std::uint32_t r = blockIdx.x / groups - down_blocks;                                                              \
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
    const std::uint32_t r0 = (blockIdx.x / groups * (blockDim.x / kWarp) + threadIdx.x / kWarp) * kRows;                      \
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
      std::uint32_t h, std::uint32_t low_rank, std::uint32_t qstride, std::uint32_t groups) {                                              \
    __shared__ float ls[NB][512];                                                                                    \
    {                                                                                                                \
      const std::uint32_t c0z = (blockIdx.x % groups) * NB; /* column group, interleaved with x */                                                  \
      lo_part += static_cast<std::size_t>(c0z) * parts * low_rank;                                                   \
      inject_part += static_cast<std::size_t>(c0z) * parts * hc;                                                     \
      inject += c0z * hc;                                                                                            \
      xn += static_cast<std::size_t>(c0z) * hc * h;                                                                  \
      mixed += c0z * h;                                                                                              \
    }                                                                                    \
    const std::uint32_t i = blockIdx.x / groups * (blockDim.x / kWarp) + threadIdx.x / kWarp;                                 \
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
      if (blockIdx.x / groups == 0 && threadIdx.x < inject_rows) {                                                            \
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

// ---- Draft-block catch-up over prompt rows (src/engine/mtp_catchup.cpp) -----------------------------------------
// halo_matvec_bf16_dual with the column on grid y: column b reads x + b * xstride and writes y1 + b * y1stride,
// y2 + b * y2stride; each row's arithmetic is the single kernel's (lane l: columns l, l + 32, ..., kBatch loads a
// lane at a time, one FMA chain, WarpSum). grid (blocks1 + ceil(rows2 / 8), columns), block 256.
extern "C" __global__ void __launch_bounds__(256)
    halo_matvec_bf16_dual_y(const std::uint16_t* w1, float* y1, std::uint32_t rows1, const std::uint16_t* w2,
                            float* y2, std::uint32_t rows2, const float* x, std::uint32_t cols, std::uint32_t blocks1,
                            std::uint32_t xstride, std::uint32_t y1stride, std::uint32_t y2stride) {
  x += static_cast<std::size_t>(blockIdx.y) * xstride;
  y1 += static_cast<std::size_t>(blockIdx.y) * y1stride;
  y2 += static_cast<std::size_t>(blockIdx.y) * y2stride;
  const bool first = blockIdx.x < blocks1;
  const std::uint32_t b = first ? blockIdx.x : blockIdx.x - blocks1;
  const std::uint32_t r = b * (blockDim.x / kWarp) + threadIdx.x / kWarp, rows = first ? rows1 : rows2;
  if (r >= rows) return;
  const std::uint16_t* row = (first ? w1 : w2) + static_cast<std::size_t>(r) * cols;
  float* y = first ? y1 : y2;
  const std::uint32_t lane = threadIdx.x % kWarp, n = cols / kWarp;
  constexpr std::uint32_t kBatch = 16;
  float acc = 0.0f;
  for (std::uint32_t j0 = 0; j0 < n; j0 += kBatch) {
    std::uint16_t wv[kBatch];
    float xv[kBatch];
#pragma unroll
    for (std::uint32_t k = 0; k < kBatch; ++k) {
      const std::uint32_t c = min(j0 + k, n - 1) * kWarp + lane;
      wv[k] = row[c];
      xv[k] = x[c];
    }
#pragma unroll
    for (std::uint32_t k = 0; k < kBatch; ++k)
      if (j0 + k < n) acc = __builtin_fmaf(Bf16(wv[k]), xv[k], acc);
  }
  acc = WarpSum(acc);
  if (lane == 0) y[r] = acc;
}
// The draft block's fused input rows: column t * hc + s = [en[t] ; hn[t * hc + s]] (2h floats each), the vector
// MtpRun builds in mtp_cat_ for stream s. grid (ceil(2h / 256), T * hc).
extern "C" __global__ void halo_mtp_cat_rows(const float* en, const float* hn, float* cat, std::uint32_t h,
                                             std::uint32_t hc) {
  const std::uint32_t col = blockIdx.y, t = col / hc;
  float* out = cat + static_cast<std::size_t>(col) * 2 * h;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < 2 * h; i += gridDim.x * blockDim.x)
    out[i] = i < h ? en[static_cast<std::size_t>(t) * h + i] : hn[static_cast<std::size_t>(col) * h + (i - h)];
}

// ---- Per-session small kernels for NB entries in one launch (blockIdx.y = entry): the single kernels' bodies
// verbatim, pointers offset to the entry's slice. ---------------------------------------------------------------
extern "C" __global__ void halo_hc_norm_scales_u_y(const float* res, float* scale, std::uint32_t h, float eps, std::uint32_t hc) {
  res += static_cast<std::size_t>(blockIdx.y) * hc * h;
  scale += blockIdx.y * hc;

  __shared__ float scratch[32];
  const float* x = res + blockIdx.x * h;
  constexpr std::uint32_t kMax = 16;  // h <= 16 * blockDim
  float v[kMax];
#pragma unroll
  for (std::uint32_t j = 0; j < kMax; ++j) {
    const std::uint32_t i = threadIdx.x + j * blockDim.x;
    v[j] = i < h ? x[i] : 0.0f;
  }
  float ss = 0.0f;
#pragma unroll
  for (std::uint32_t j = 0; j < kMax; ++j)
    if (threadIdx.x + j * blockDim.x < h) ss = __builtin_fmaf(v[j], v[j], ss);
  ss = BlockSum(ss, scratch);
  if (threadIdx.x == 0) scale[blockIdx.x] = 1.0f / sqrtf(ss / static_cast<float>(h) + eps);
}
extern "C" __global__ void halo_hc_combine_y(float* res, const float* out, const float* inject, std::uint32_t hc,
                                           std::uint32_t h) {
  res += static_cast<std::size_t>(blockIdx.y) * hc * h;
  out += static_cast<std::size_t>(blockIdx.y) * h;
  inject += blockIdx.y * hc;

  for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < hc * h; idx += gridDim.x * blockDim.x) {
    const std::uint32_t s = idx / h;
    res[idx] += out[idx % h] * 2.0f * Sigmoid(inject[s] / static_cast<float>(hc));
  }
}
extern "C" __global__ void __launch_bounds__(256)
    halo_router_topk_w_y(const float* logits, std::uint32_t* ids, float* weights, std::uint32_t n, std::uint32_t k, std::uint32_t lstride) {
  logits += static_cast<std::size_t>(blockIdx.y) * lstride;
  ids += blockIdx.y * k;
  weights += blockIdx.y * k;

  __shared__ float scratch[32];
  __shared__ float prob[1024];
  __shared__ float cv[8 * 16];
  __shared__ std::uint32_t ci[8 * 16];
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
  const std::uint32_t lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp, warps = blockDim.x / kWarp;
  const auto better = [](float ov, std::uint32_t oi, float v, std::uint32_t at) {
    return ov > v || (ov == v && oi < at);
  };
  // Stage 1: warp w's slice [w * seg, (w + 1) * seg); lane l holds slice[l], slice[l + 32], ... (<= 4 values).
  const std::uint32_t seg = (n + warps - 1) / warps;
  constexpr std::uint32_t kPer = 4;  // 1024 / 8 warps / 32 lanes
  float lv[kPer];
  std::uint32_t li[kPer];
#pragma unroll
  for (std::uint32_t m = 0; m < kPer; ++m) {
    const std::uint32_t i = warp * seg + lane + m * kWarp;
    const bool in = lane + m * kWarp < seg && i < n;
    lv[m] = in ? prob[i] : -1.0f;
    li[m] = in ? i : 0xFFFFFFFFu;
  }
  for (std::uint32_t r = 0; r < k; ++r) {
    float v = -1.0f;
    std::uint32_t at = 0xFFFFFFFFu;
#pragma unroll
    for (std::uint32_t m = 0; m < kPer; ++m)
      if (lv[m] > v) {  // ascending index within the lane: the first max has the lowest index
        v = lv[m];
        at = li[m];
      }
    for (int o = kWarp / 2; o > 0; o >>= 1) {
      const float ov = __shfl_xor(v, o, kWarp);
      const std::uint32_t oi = __shfl_xor(at, o, kWarp);
      if (better(ov, oi, v, at)) {
        v = ov;
        at = oi;
      }
    }
#pragma unroll
    for (std::uint32_t m = 0; m < kPer; ++m)
      if (li[m] == at && at != 0xFFFFFFFFu) lv[m] = -1.0f;
    if (lane == 0) {
      cv[warp * k + r] = v;
      ci[warp * k + r] = at;
    }
  }
  __syncthreads();
  if (warp != 0) return;
  // Stage 2: warp 0 over the warps' candidates (lane l holds candidates l, l + 32, ...: <= 4).
  const std::uint32_t nc = warps * k;
  float cvl[kPer];
  std::uint32_t cil[kPer];
#pragma unroll
  for (std::uint32_t m = 0; m < kPer; ++m) {
    const std::uint32_t c = lane + m * kWarp;
    cvl[m] = c < nc ? cv[c] : -1.0f;
    cil[m] = c < nc ? ci[c] : 0xFFFFFFFFu;
  }
  float sum = 0.0f;
  for (std::uint32_t slot = 0; slot < k; ++slot) {
    float v = -1.0f;
    std::uint32_t at = 0xFFFFFFFFu;
#pragma unroll
    for (std::uint32_t m = 0; m < kPer; ++m)
      if (better(cvl[m], cil[m], v, at)) {
        v = cvl[m];
        at = cil[m];
      }
    for (int o = kWarp / 2; o > 0; o >>= 1) {
      const float ov = __shfl_xor(v, o, kWarp);
      const std::uint32_t oi = __shfl_xor(at, o, kWarp);
      if (better(ov, oi, v, at)) {
        v = ov;
        at = oi;
      }
    }
#pragma unroll
    for (std::uint32_t m = 0; m < kPer; ++m)
      if (cil[m] == at && at != 0xFFFFFFFFu) cvl[m] = -1.0f, cil[m] = 0xFFFFFFFFu;
    if (lane == 0) {
      ids[slot] = at;
      weights[slot] = v;
      sum += v;
    }
  }
  if (lane == 0) {
    sum = fmaxf(sum, 6.103515625e-5f);
    for (std::uint32_t slot = 0; slot < k; ++slot) weights[slot] /= sum;
  }
}

// ---- PROBE ONLY (not used by any decode path): batch-shaped WMMA GEMVs for the X measurement ------------------
// Weights dequantized to fp16 per 32-value block (fp16(q * scale) for Q8; fp16(d sc q - dmin m) for Q4_K;
// fp16(d q + m) for Q5_1), activations rounded to fp16, FP16 WMMA 16x16x16 with fp32 accumulation, split-K
// partials summed in split order: a column's result does not depend on how many columns run (batch-invariant),
// but it differs from the fp32 single-token kernels (fp16 activations and weights, WMMA sum order, split-K).
// A wave owns 16 rows; columns 0..B-1 of the 16-wide tile are live.
namespace {
__device__ inline void ProbeStore(halo_v8f acc, float* out, std::uint32_t r0, std::uint32_t rows, std::uint32_t ncols,
                                  std::size_t col_stride) {
  const std::uint32_t lane = threadIdx.x % kWarp, col = lane % 16;
  if (col >= ncols) return;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const std::uint32_t r = r0 + 2 * j + lane / 16;
    if (r < rows) out[col * col_stride + r] = acc[j];
  }
}
}  // namespace

// part[s][b][rows] = W[rows, k-split s] . x[b]; grid (ceil(rows / 128), S), 256 threads.
extern "C" __global__ void __launch_bounds__(256)
    halo_probe_q8r_wmma(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* part, std::uint32_t rows,
                        std::uint32_t cols, std::uint32_t qstride, std::uint32_t B, std::uint32_t kspan,
                        std::uint32_t xstride) {
  __shared__ _Float16 xs[16 * 1024];
  const std::uint32_t s = blockIdx.y, k0 = s * kspan, k1 = min(cols, k0 + kspan), n = k1 - k0;
  for (std::uint32_t b = 0; b < B; ++b)
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) xs[b * kspan + i] = static_cast<_Float16>(x[b * xstride + k0 + i]);
  __syncthreads();
  const std::uint32_t lane = threadIdx.x % kWarp, r0 = (blockIdx.x * 8 + threadIdx.x / kWarp) * 16;
  if (r0 >= rows) return;
  const std::uint32_t row = min(r0 + lane % 16, rows - 1), col = lane % 16, nblk = cols / 32;
  halo_v8f acc = {};
  for (std::uint32_t k = 0; k < n; k += 32) {
    const int4 q0 = *reinterpret_cast<const int4*>(qs + static_cast<std::size_t>(row) * qstride + k0 + k);
    const int4 q1 = *reinterpret_cast<const int4*>(qs + static_cast<std::size_t>(row) * qstride + k0 + k + 16);
    const float sc = F16(d[static_cast<std::size_t>(row) * nblk + (k0 + k) / 32]);
    const std::int8_t* v0 = reinterpret_cast<const std::int8_t*>(&q0);
    const std::int8_t* v1 = reinterpret_cast<const std::int8_t*>(&q1);
    halo_v16h a0, a1, b0 = {}, b1 = {};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      a0[i] = static_cast<_Float16>(static_cast<float>(v0[i]) * sc);
      a1[i] = static_cast<_Float16>(static_cast<float>(v1[i]) * sc);
    }
    if (col < B) {
      b0 = *reinterpret_cast<const halo_v16h*>(xs + col * kspan + k);
      b1 = *reinterpret_cast<const halo_v16h*>(xs + col * kspan + k + 16);
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc);
  }
  ProbeStore(acc, part + static_cast<std::size_t>(s) * B * rows, r0, rows, B, rows);
}

// y[b][r] = sum over s of part[s][b][r], in split order.
extern "C" __global__ void halo_probe_reduce(const float* part, float* y, std::uint32_t rows, std::uint32_t B,
                                             std::uint32_t S) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * B) return;
  float a = 0.0f;
  for (std::uint32_t s = 0; s < S; ++s) a += part[static_cast<std::size_t>(s) * rows * B + i];
  y[i] = a;
}

// Routed gate/up, Q4_K, grouped by expert: block (row group of 128, u); pair activations x[pair / k] (fp32 [*][cols]).
extern "C" __global__ void __launch_bounds__(256)
    halo_probe_q4k_gu_wmma(const std::uint8_t* gate, const std::uint8_t* up, const std::uint32_t* ucount,
                           const std::uint32_t* uexp, const std::uint32_t* unum, const std::uint32_t* upair,
                           const float* x, float* h, std::uint64_t expert_bytes, std::uint32_t rows, std::uint32_t cols,
                           std::uint64_t row_bytes, std::uint32_t k) {
  __shared__ _Float16 xs[8 * 2560];
  const std::uint32_t u = blockIdx.y;
  if (u >= *ucount) return;
  const std::uint32_t m = unum[u];
  for (std::uint32_t j = 0; j < m; ++j) {
    const std::uint32_t col = upair[u * 8 + j] / k;
    for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[j * cols + i] = static_cast<_Float16>(x[col * cols + i]);
  }
  __syncthreads();
  const std::uint32_t lane = threadIdx.x % kWarp, r0 = (blockIdx.x * 8 + threadIdx.x / kWarp) * 16;
  if (r0 >= rows) return;
  const std::uint32_t row = min(r0 + lane % 16, rows - 1), col = lane % 16;
  const std::size_t e = uexp[u];
  const std::uint8_t* gr = gate + e * expert_bytes + row * row_bytes;
  const std::uint8_t* ur = up + e * expert_bytes + row * row_bytes;
  halo_v8f ag = {}, au = {};
  for (std::uint32_t kk = 0; kk < cols; kk += 64) {  // one Q4_K item: 64 values, sub-blocks 2c (low) and 2c+1 (high)
    const std::uint32_t sb = kk / 256, c = (kk % 256) / 64;
    halo_v16h bl0 = {}, bl1 = {}, bh0 = {}, bh1 = {};
    if (col < m) {
      bl0 = *reinterpret_cast<const halo_v16h*>(xs + col * cols + kk);
      bl1 = *reinterpret_cast<const halo_v16h*>(xs + col * cols + kk + 16);
      bh0 = *reinterpret_cast<const halo_v16h*>(xs + col * cols + kk + 32);
      bh1 = *reinterpret_cast<const halo_v16h*>(xs + col * cols + kk + 48);
    }
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      const std::uint8_t* b = (g == 0 ? gr : ur) + sb * 144;
      const int4 hdr = *reinterpret_cast<const int4*>(b);
      const int4 qa = *reinterpret_cast<const int4*>(b + 16 + c * 32);
      const int4 qb = *reinterpret_cast<const int4*>(b + 16 + c * 32 + 16);
      std::uint32_t sc0, m0, sc1, m1;
      ScaleMinK4Words(2 * c, static_cast<std::uint32_t>(hdr.y), static_cast<std::uint32_t>(hdr.z),
                      static_cast<std::uint32_t>(hdr.w), sc0, m0);
      ScaleMinK4Words(2 * c + 1, static_cast<std::uint32_t>(hdr.y), static_cast<std::uint32_t>(hdr.z),
                      static_cast<std::uint32_t>(hdr.w), sc1, m1);
      const float dd = F16(static_cast<std::uint16_t>(hdr.x & 0xFFFF));
      const float dm = F16(static_cast<std::uint16_t>(static_cast<std::uint32_t>(hdr.x) >> 16));
      const float s0 = dd * static_cast<float>(sc0), o0 = dm * static_cast<float>(m0);
      const float s1 = dd * static_cast<float>(sc1), o1 = dm * static_cast<float>(m1);
      const std::uint8_t* q0 = reinterpret_cast<const std::uint8_t*>(&qa);
      const std::uint8_t* q1 = reinterpret_cast<const std::uint8_t*>(&qb);
      halo_v16h l0, l1, h0, h1;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        l0[i] = static_cast<_Float16>(s0 * static_cast<float>(q0[i] & 0xF) - o0);
        l1[i] = static_cast<_Float16>(s0 * static_cast<float>(q1[i] & 0xF) - o0);
        h0[i] = static_cast<_Float16>(s1 * static_cast<float>(q0[i] >> 4) - o1);
        h1[i] = static_cast<_Float16>(s1 * static_cast<float>(q1[i] >> 4) - o1);
      }
      halo_v8f& acc = g == 0 ? ag : au;
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(l0, bl0, acc);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(l1, bl1, acc);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(h0, bh0, acc);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(h1, bh1, acc);
    }
  }
  if (col >= m) return;
  const std::uint32_t pair = upair[u * 8 + col];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const std::uint32_t r = r0 + 2 * j + lane / 16;
    if (r < rows) h[static_cast<std::size_t>(pair) * rows + r] = Silu(ag[j]) * au[j];
  }
}

// Routed down, Q5_1, grouped by expert: block (row group of 128, u); h[pair][cols] in, part[pair][rows] out.
extern "C" __global__ void __launch_bounds__(256)
    halo_probe_q51_down_wmma(const std::uint8_t* down, const std::uint32_t* ucount, const std::uint32_t* uexp,
                             const std::uint32_t* unum, const std::uint32_t* upair, const float* h, float* part,
                             std::uint64_t expert_bytes, std::uint32_t rows, std::uint32_t cols, std::uint64_t row_bytes) {
  __shared__ _Float16 xs[8 * 1024];
  const std::uint32_t u = blockIdx.y;
  if (u >= *ucount) return;
  const std::uint32_t m = unum[u];
  for (std::uint32_t j = 0; j < m; ++j) {
    const std::uint32_t p = upair[u * 8 + j];
    for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) xs[j * cols + i] = static_cast<_Float16>(h[p * cols + i]);
  }
  __syncthreads();
  const std::uint32_t lane = threadIdx.x % kWarp, r0 = (blockIdx.x * 8 + threadIdx.x / kWarp) * 16;
  if (r0 >= rows) return;
  const std::uint32_t row = min(r0 + lane % 16, rows - 1), col = lane % 16;
  const std::uint8_t* wr = down + uexp[u] * expert_bytes + row * row_bytes;
  halo_v8f acc = {};
  for (std::uint32_t kk = 0; kk < cols; kk += 32) {
    const std::uint8_t* b = wr + (kk / 32) * 24;
    const uint2 hd = *reinterpret_cast<const uint2*>(b);
    const uint2 qa = *reinterpret_cast<const uint2*>(b + 8);
    const uint2 qb = *reinterpret_cast<const uint2*>(b + 16);
    std::uint8_t q[16];
    *reinterpret_cast<uint2*>(q) = qa;
    *reinterpret_cast<uint2*>(q + 8) = qb;
    const float dd = F16(static_cast<std::uint16_t>(hd.x & 0xFFFF)), mm = F16(static_cast<std::uint16_t>(hd.x >> 16));
    const std::uint32_t qh = hd.y;
    halo_v16h a0, a1, b0 = {}, b1 = {};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      a0[j] = static_cast<_Float16>(dd * static_cast<float>((q[j] & 0xF) | (((qh >> j) & 1) << 4)) + mm);
      a1[j] = static_cast<_Float16>(dd * static_cast<float>((q[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) + mm);
    }
    if (col < m) {
      b0 = *reinterpret_cast<const halo_v16h*>(xs + col * cols + kk);
      b1 = *reinterpret_cast<const halo_v16h*>(xs + col * cols + kk + 16);
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc);
  }
  if (col >= m) return;
  const std::uint32_t pair = upair[u * 8 + col];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const std::uint32_t r = r0 + 2 * j + lane / 16;
    if (r < rows) part[static_cast<std::size_t>(pair) * rows + r] = acc[j];
  }
}

// PROBE ONLY, v2: the weight tile goes through LDS so that global loads stay row-contiguous (each load instruction
// reads 256 contiguous bytes of one row, as the decode GEMVs do) instead of 16 rows x 16 bytes. Per wave: 16 rows,
// K tiles of 256; the next tile's loads are issued before the current tile's WMMAs. Same numerics as v1.
extern "C" __global__ void __launch_bounds__(256)
    halo_probe_q8r_wmma2(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* part, std::uint32_t rows,
                         std::uint32_t cols, std::uint32_t qstride, std::uint32_t B, std::uint32_t kspan,
                         std::uint32_t xstride) {
  __shared__ _Float16 xs[8 * 1024];
  __shared__ int2 wt[8][16][32];  // per wave: 16 rows x 256 bytes
  const std::uint32_t s = blockIdx.y, k0 = s * kspan, k1 = min(cols, k0 + kspan), n = k1 - k0;
  for (std::uint32_t b = 0; b < B; ++b)
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) xs[b * kspan + i] = static_cast<_Float16>(x[b * xstride + k0 + i]);
  __syncthreads();
  const std::uint32_t lane = threadIdx.x % kWarp, w = threadIdx.x / kWarp, r0 = (blockIdx.x * 8 + w) * 16;
  if (r0 >= rows) return;
  const std::uint32_t col = lane % 16, nblk = cols / 32, myrow = min(r0 + lane % 16, rows - 1);
  int2 cur[16], nxt[16];
  const auto load = [&](std::uint32_t kt, int2 (&v)[16]) {
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const std::uint32_t row = min(r0 + r, rows - 1);
      const std::uint32_t c = k0 + kt + lane * 8;
      v[r] = c < k1 ? *reinterpret_cast<const int2*>(qs + static_cast<std::size_t>(row) * qstride + c) : int2{0, 0};
    }
  };
  halo_v8f acc = {};
  load(0, cur);
  for (std::uint32_t kt = 0; kt < n; kt += 256) {
    if (kt + 256 < n) load(kt + 256, nxt);
#pragma unroll
    for (int r = 0; r < 16; ++r) wt[w][r][lane] = cur[r];
    __builtin_amdgcn_wave_barrier();
    const std::int8_t* rowt = reinterpret_cast<const std::int8_t*>(&wt[w][lane % 16][0]);
    const std::uint32_t kn = min(256u, n - kt);
    for (std::uint32_t kk = 0; kk < kn; kk += 32) {
      const int4 q0 = *reinterpret_cast<const int4*>(rowt + kk);
      const int4 q1 = *reinterpret_cast<const int4*>(rowt + kk + 16);
      const float sc = F16(d[static_cast<std::size_t>(myrow) * nblk + (k0 + kt + kk) / 32]);
      const std::int8_t* v0 = reinterpret_cast<const std::int8_t*>(&q0);
      const std::int8_t* v1 = reinterpret_cast<const std::int8_t*>(&q1);
      halo_v16h a0, a1, b0 = {}, b1 = {};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        a0[i] = static_cast<_Float16>(static_cast<float>(v0[i]) * sc);
        a1[i] = static_cast<_Float16>(static_cast<float>(v1[i]) * sc);
      }
      if (col < B) {
        b0 = *reinterpret_cast<const halo_v16h*>(xs + col * kspan + kt + kk);
        b1 = *reinterpret_cast<const halo_v16h*>(xs + col * kspan + kt + kk + 16);
      }
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc);
    }
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int r = 0; r < 16; ++r) cur[r] = nxt[r];
  }
  ProbeStore(acc, part + static_cast<std::size_t>(s) * B * rows, r0, rows, B, rows);
}

// PROBE ONLY, v3: v2 with the tile's scales loaded with the tile (v2 loaded one per 32 values inside the WMMA loop).
extern "C" __global__ void __launch_bounds__(256)
    halo_probe_q8r_wmma3(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* part, std::uint32_t rows,
                         std::uint32_t cols, std::uint32_t qstride, std::uint32_t B, std::uint32_t kspan,
                         std::uint32_t xstride) {
  __shared__ _Float16 xs[8 * 1024];
  __shared__ int2 wt[8][16][32];  // per wave: 16 rows x 256 bytes
  const std::uint32_t s = blockIdx.y, k0 = s * kspan, k1 = min(cols, k0 + kspan), n = k1 - k0;
  for (std::uint32_t b = 0; b < B; ++b)
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) xs[b * kspan + i] = static_cast<_Float16>(x[b * xstride + k0 + i]);
  __syncthreads();
  const std::uint32_t lane = threadIdx.x % kWarp, w = threadIdx.x / kWarp, r0 = (blockIdx.x * 8 + w) * 16;
  if (r0 >= rows) return;
  const std::uint32_t col = lane % 16, nblk = cols / 32, myrow = min(r0 + lane % 16, rows - 1);
  int2 cur[16], nxt[16];
  float scur[8], snxt[8];  // this lane's row's 8 scales of a tile, loaded with the tile (no load in the inner loop)
  const auto load = [&](std::uint32_t kt, int2 (&v)[16], float (&sc)[8]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t blk = (k0 + kt) / 32 + j;
      sc[j] = blk < nblk ? F16(d[static_cast<std::size_t>(myrow) * nblk + blk]) : 0.0f;
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const std::uint32_t row = min(r0 + r, rows - 1);
      const std::uint32_t c = k0 + kt + lane * 8;
      v[r] = c < k1 ? *reinterpret_cast<const int2*>(qs + static_cast<std::size_t>(row) * qstride + c) : int2{0, 0};
    }
  };
  halo_v8f acc = {};
  load(0, cur, scur);
  for (std::uint32_t kt = 0; kt < n; kt += 256) {
    if (kt + 256 < n) load(kt + 256, nxt, snxt);
#pragma unroll
    for (int r = 0; r < 16; ++r) wt[w][r][lane] = cur[r];
    __builtin_amdgcn_wave_barrier();
    const std::int8_t* rowt = reinterpret_cast<const std::int8_t*>(&wt[w][lane % 16][0]);
    const std::uint32_t kn = min(256u, n - kt);
#pragma unroll
    for (std::uint32_t kk = 0; kk < 256; kk += 32) {
      if (kk >= kn) break;
      const int4 q0 = *reinterpret_cast<const int4*>(rowt + kk);
      const int4 q1 = *reinterpret_cast<const int4*>(rowt + kk + 16);
      const float sc = scur[kk / 32];
      const std::int8_t* v0 = reinterpret_cast<const std::int8_t*>(&q0);
      const std::int8_t* v1 = reinterpret_cast<const std::int8_t*>(&q1);
      halo_v16h a0, a1, b0 = {}, b1 = {};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        a0[i] = static_cast<_Float16>(static_cast<float>(v0[i]) * sc);
        a1[i] = static_cast<_Float16>(static_cast<float>(v1[i]) * sc);
      }
      if (col < B) {
        b0 = *reinterpret_cast<const halo_v16h*>(xs + col * kspan + kt + kk);
        b1 = *reinterpret_cast<const halo_v16h*>(xs + col * kspan + kt + kk + 16);
      }
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc);
    }
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int r = 0; r < 16; ++r) cur[r] = nxt[r];
#pragma unroll
    for (int j = 0; j < 8; ++j) scur[j] = snxt[j];
  }
  ProbeStore(acc, part + static_cast<std::size_t>(s) * B * rows, r0, rows, B, rows);
}

// PROBE ONLY, v4: W8A8. Weights int8 straight into the iu8 WMMA (no dequantization: the fp16 probes spend ~4 VALU
// per weight, twice, since both half-waves hold the A tile), activations quantized per column and 32 values to int8
// (amax / 127, as the routed experts' Q8Act), int32 sums per 32-value block scaled by row scale x column scale into
// fp32. Numerics: int8 activations for the dense projections (a precision change), block sums exact in int32, the
// fp32 accumulation order over blocks fixed (independent of the column count).
extern "C" __global__ void __launch_bounds__(256)
    halo_probe_q8r_wmma4(const std::int8_t* qs, const std::uint16_t* d, const float* x, float* part, std::uint32_t rows,
                         std::uint32_t cols, std::uint32_t qstride, std::uint32_t B, std::uint32_t kspan,
                         std::uint32_t xstride) {
  __shared__ std::int8_t xq[8 * 1024];
  __shared__ float xsc[8 * 32];
  __shared__ int2 wt[8][16][32];
  __shared__ float wsc[8][16][8];
  const std::uint32_t s = blockIdx.y, k0 = s * kspan, k1 = min(cols, k0 + kspan), n = k1 - k0;
  // Quantize: thread t handles one (column, 32-value block).
  for (std::uint32_t t = threadIdx.x; t < B * ((n + 31) / 32); t += blockDim.x) {
    const std::uint32_t b = t / ((n + 31) / 32), blk = t % ((n + 31) / 32);
    float v[32], amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const std::uint32_t k = blk * 32 + i;
      v[i] = k < n ? x[b * xstride + k0 + k] : 0.0f;
      amax = fmaxf(amax, fabsf(v[i]));
    }
    const float sc = amax / 127.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i)
      xq[b * kspan + blk * 32 + i] = static_cast<std::int8_t>(sc > 0.0f ? static_cast<int>(rintf(v[i] / sc)) : 0);
    xsc[b * 32 + blk] = sc;
  }
  __syncthreads();
  const std::uint32_t lane = threadIdx.x % kWarp, w = threadIdx.x / kWarp, r0 = (blockIdx.x * 8 + w) * 16;
  if (r0 >= rows) return;
  const std::uint32_t col = lane % 16, nblk = cols / 32;
  int2 cur[16], nxt[16];
  float scur[4], snxt[4];  // lane l holds scales of row l % 16, blocks 4 (l / 16) .. +3 of the tile
  const auto load = [&](std::uint32_t kt, int2 (&v)[16], float (&sc)[4]) {
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const std::uint32_t row = min(r0 + r, rows - 1);
      const std::uint32_t c = k0 + kt + lane * 8;
      v[r] = c < k1 ? *reinterpret_cast<const int2*>(qs + static_cast<std::size_t>(row) * qstride + c) : int2{0, 0};
    }
    const std::uint32_t row = min(r0 + lane % 16, rows - 1);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const std::uint32_t blk = (k0 + kt) / 32 + 4 * (lane / 16) + j;
      sc[j] = blk < nblk ? F16(d[static_cast<std::size_t>(row) * nblk + blk]) : 0.0f;
    }
  };
  halo_v8f acc = {};
  load(0, cur, scur);
  for (std::uint32_t kt = 0; kt < n; kt += 256) {
    if (kt + 256 < n) load(kt + 256, nxt, snxt);
#pragma unroll
    for (int r = 0; r < 16; ++r) wt[w][r][lane] = cur[r];
#pragma unroll
    for (int j = 0; j < 4; ++j) wsc[w][lane % 16][4 * (lane / 16) + j] = scur[j];
    __builtin_amdgcn_wave_barrier();
    const std::int8_t* rowt = reinterpret_cast<const std::int8_t*>(&wt[w][lane % 16][0]);
    const std::uint32_t kn = min(256u, n - kt);
#pragma unroll
    for (std::uint32_t kk = 0; kk < 256; kk += 32) {
      if (kk >= kn) break;
      const halo_v4i a0 = *reinterpret_cast<const halo_v4i*>(rowt + kk);
      const halo_v4i a1 = *reinterpret_cast<const halo_v4i*>(rowt + kk + 16);
      halo_v4i b0 = {}, b1 = {};
      float xs = 0.0f;
      if (col < B) {
        b0 = *reinterpret_cast<const halo_v4i*>(xq + col * kspan + kt + kk);
        b1 = *reinterpret_cast<const halo_v4i*>(xq + col * kspan + kt + kk + 16);
        xs = xsc[col * 32 + (kt + kk) / 32];
      }
      halo_v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, halo_v8i{}, false);
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, c, false);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += static_cast<float>(c[j]) * (wsc[w][2 * j + lane / 16][kk / 32] * xs);
    }
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int r = 0; r < 16; ++r) cur[r] = nxt[r];
#pragma unroll
    for (int j = 0; j < 4; ++j) scur[j] = snxt[j];
  }
  ProbeStore(acc, part + static_cast<std::size_t>(s) * B * rows, r0, rows, B, rows);
}
