// halo_hwprobe: measured facts about the GPU (gfx1151 Strix Halo) that the
// performance work relies on. Standalone HIP; each subcommand prints one table.
//
//   halo_hwprobe clock           shader clock (in-kernel counters) vs load
//   halo_hwprobe dram            bandwidth vs segment size, dispatch order, read/write
//   halo_hwprobe wmma            f16 / bf16 / iu8 WMMA rate, chain latency, VALU overlap
//
// Clock truth comes from inside kernels: SHADER_CYCLES (20 bits, wraps every
// ~0.35 ms at 2.9 GHz) against the 100 MHz realtime counter. sysfs reports the
// requested level, not the delivered clock.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

#define CHECK(x)                                                                    \
  do {                                                                              \
    const hipError_t e_ = (x);                                                      \
    if (e_ != hipSuccess) {                                                         \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      std::exit(1);                                                                 \
    }                                                                               \
  } while (0)

namespace {

// ---- clock -------------------------------------------------------------------
__global__ void ClockKernel(unsigned long long* o, int windows) {
  float a = threadIdx.x * 1e-3f;
  unsigned long long cyc = 0;
  const unsigned long long r0 = wall_clock64();
  for (int w = 0; w < windows; ++w) {
    const unsigned c0 = static_cast<unsigned>(clock64()) & 0xFFFFF;
    for (int i = 0; i < 2000; ++i) a = __builtin_fmaf(a, 1.0001f, 0.5f);
    const unsigned c1 = static_cast<unsigned>(clock64()) & 0xFFFFF;
    cyc += (c1 - c0) & 0xFFFFF;
  }
  const unsigned long long r1 = wall_clock64();
  if (threadIdx.x == 0) o[2 * blockIdx.x] = cyc, o[2 * blockIdx.x + 1] = r1 - r0;
  if (a == 12345.f) o[0] = 1;
}

void Clock() {
  std::printf("shader clock under VALU load (dependent FMA chains; MHz from in-kernel counters)\n");
  for (const auto& [blocks, threads] : std::vector<std::pair<int, int>>{{1, 32}, {80, 32}, {80, 256}, {640, 256}}) {
    unsigned long long* o;
    CHECK(hipMalloc(&o, 16ull * blocks));
    hipLaunchKernelGGL(ClockKernel, blocks, threads, 0, 0, o, 50);
    CHECK(hipDeviceSynchronize());
    hipLaunchKernelGGL(ClockKernel, blocks, threads, 0, 0, o, 1500);
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned long long> h(2 * blocks);
    CHECK(hipMemcpy(h.data(), o, 16ull * blocks, hipMemcpyDeviceToHost));
    double sum = 0, lo = 1e9, hi = 0;
    for (int b = 0; b < blocks; ++b) {
      const double m = h[2 * b] / (h[2 * b + 1] / 100.0);
      sum += m, lo = std::min(lo, m), hi = std::max(hi, m);
    }
    std::printf("  %4d blocks x %3d threads: %5.0f MHz (min %.0f, max %.0f), %.2f cycles per dependent FMA per wave\n",
                blocks, threads, sum / blocks, lo, hi, h[0] / (1500.0 * 2000.0));
    CHECK(hipFree(o));
  }
}

// ---- dram --------------------------------------------------------------------
// The buffer is rows x row_bytes. A block reads (or writes) a tile: `seg`
// contiguous bytes from each of `trows` consecutive rows. Tiles are numbered
// (row group, column); col_fast: consecutive blocks take consecutive columns
// of the same rows (concurrent blocks cover whole rows); otherwise
// consecutive blocks take the next row group in the same column.
template <int MODE>  // 0 read, 1 write, 2 read + write (copy to a second buffer)
__global__ void __launch_bounds__(256)
    TileKernel(const float4* src, float4* dst, std::uint32_t row_f4, std::uint32_t seg_f4, std::uint32_t trows,
               std::uint32_t cols, std::uint32_t groups, int col_fast, float* sink) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t col = col_fast ? b % cols : b / groups, grp = col_fast ? b / cols : b % groups;
  const std::uint32_t per = trows * seg_f4;
  float acc = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < per; i += blockDim.x) {
    const std::uint32_t r = grp * trows + i / seg_f4, c = col * seg_f4 + i % seg_f4;
    const std::size_t at = static_cast<std::size_t>(r) * row_f4 + c;
    if constexpr (MODE == 1) {
      dst[at] = make_float4(1.0f, 2.0f, 3.0f, static_cast<float>(i));
    } else {
      const float4 v = src[at];
      if constexpr (MODE == 2) dst[at] = v;
      else acc += v.x + v.y + v.z + v.w;
    }
  }
  if (acc == 12345.f) sink[0] = acc;
}

void Dram() {
  const std::uint32_t row_bytes = 40960, rows = 12288;  // 480 MB: past the 32 MB MALL
  const std::size_t bytes = std::size_t{row_bytes} * rows;
  float4 *a, *b;
  float* sink;
  CHECK(hipMalloc(&a, bytes));
  CHECK(hipMalloc(&b, bytes));
  CHECK(hipMalloc(&sink, 64));
  CHECK(hipMemset(a, 1, bytes));
  CHECK(hipMemset(b, 1, bytes));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::printf("DRAM: 480 MB as rows of 40 KB; a block takes `seg` bytes from each of 128 rows (256 threads, 16-byte accesses)\n");
  std::printf("  %6s  %-9s  %8s  %8s  %8s\n", "seg B", "order", "read", "write", "copy");
  for (const std::uint32_t seg : {32u, 64u, 128u, 256u, 512u, 1024u, 4096u, 40960u}) {
    const std::uint32_t seg_f4 = seg / 16, trows = seg == 40960 ? 4 : 128, cols = row_bytes / seg, groups = rows / trows;
    for (const int col_fast : {1, 0}) {
      if (seg == 40960 && !col_fast) continue;
      double gbs[3];
      for (int mode = 0; mode < 3; ++mode) {
        const auto run = [&] {
          const dim3 grid(cols * groups);
          if (mode == 0)
            hipLaunchKernelGGL(TileKernel<0>, grid, 256, 0, 0, a, b, row_bytes / 16, seg_f4, trows, cols, groups, col_fast, sink);
          else if (mode == 1)
            hipLaunchKernelGGL(TileKernel<1>, grid, 256, 0, 0, a, b, row_bytes / 16, seg_f4, trows, cols, groups, col_fast, sink);
          else
            hipLaunchKernelGGL(TileKernel<2>, grid, 256, 0, 0, a, b, row_bytes / 16, seg_f4, trows, cols, groups, col_fast, sink);
        };
        run();
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(e0));
        for (int r = 0; r < 5; ++r) run();
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        gbs[mode] = (mode == 2 ? 2.0 : 1.0) * bytes * 5 / (ms * 1e-3) / 1e9;
      }
      std::printf("  %6u  %-9s  %8.1f  %8.1f  %8.1f GB/s\n", seg, col_fast ? "cols-fast" : "rows-fast", gbs[0], gbs[1], gbs[2]);
    }
  }
  CHECK(hipFree(a));
  CHECK(hipFree(b));
}

// ---- wmma --------------------------------------------------------------------
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));

// T: 0 f16, 1 bf16, 2 iu8. ACC independent accumulators per wave; VALU
// independent FMAs per WMMA interleaved (0 = none). Each wave runs `iters`
// rounds of ACC WMMAs.
template <int T, int ACC, int VALU>
__global__ void WmmaKernel(int iters, float* out) {
  const int lane = threadIdx.x % 32;
  v16h ah, bh;
  v16s as, bs;
  v4i ai = {lane, 3, 5, 7}, bi = {1, lane, 2, 9};
  for (int i = 0; i < 16; ++i) {
    ah[i] = static_cast<_Float16>((lane + i) % 7 * 0.01f);
    bh[i] = static_cast<_Float16>((lane * 3 + i) % 5 * 0.01f);
    as[i] = static_cast<short>(0x3c00 + ((lane + i) & 7));
    bs[i] = static_cast<short>(0x3c00 + ((lane * 3 + i) & 7));
  }
  v8f cf[ACC] = {};
  v8i ci[ACC] = {};
  float f[VALU > 0 ? VALU : 1];
  for (int v = 0; v < (VALU > 0 ? VALU : 1); ++v) f[v] = lane + v;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int a = 0; a < ACC; ++a) {
      if constexpr (T == 0) cf[a] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bh, cf[a]);
      else if constexpr (T == 1) cf[a] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as, bs, cf[a]);
      else ci[a] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ai, true, bi, ci[a], false);
#pragma unroll
      for (int v = 0; v < VALU; ++v) f[v] = __builtin_fmaf(f[v], 1.0001f, 0.5f);
    }
  }
  float s = 0;
  for (int a = 0; a < ACC; ++a) s += cf[a][0] + static_cast<float>(ci[a][0]);
  for (int v = 0; v < (VALU > 0 ? VALU : 1); ++v) s += f[v];
  if (s == 12345.f) out[0] = s;
}

template <int T, int ACC, int VALU>
double WmmaTime(int blocks, int threads, int iters, float* out) {
  hipLaunchKernelGGL((WmmaKernel<T, ACC, VALU>), blocks, threads, 0, 0, 10, out);
  CHECK(hipDeviceSynchronize());
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  CHECK(hipEventRecord(e0));
  hipLaunchKernelGGL((WmmaKernel<T, ACC, VALU>), blocks, threads, 0, 0, iters, out);
  CHECK(hipEventRecord(e1));
  CHECK(hipEventSynchronize(e1));
  float ms;
  CHECK(hipEventElapsedTime(&ms, e0, e1));
  return ms;
}

void Wmma() {
  constexpr double kClockGHz = 2.88, kSimds = 80;  // 20 WGPs x 4 SIMD32 (clock: halo_hwprobe clock)
  float* out;
  CHECK(hipMalloc(&out, 64));
  const char* names[3] = {"f16", "bf16", "iu8"};
  std::printf("WMMA 16x16x16 (one wave-instruction = 8192 FLOP/OP); cycles per WMMA per SIMD at %.2f GHz\n", kClockGHz);
  const auto row = [&](const char* label, int t, int waves_per_simd, int acc, double ms, int iters) {
    const double wmmas = kSimds * waves_per_simd * double(iters) * acc;
    const double cyc = ms * 1e-3 * kClockGHz * 1e9 / (wmmas / kSimds);
    std::printf("  %-4s %-28s %6.2f cycles/WMMA/SIMD  %6.1f T%s/s\n", names[t], label, cyc,
                wmmas * 8192 / (ms * 1e-3) / 1e12, t == 2 ? "OP" : "FLOP");
  };
  const int iters = 20000;
  (void)WmmaTime<0, 4, 0>(80, 128, iters, out);  // a full-length throwaway first: the clock ramps from idle
#define WMMA_ROWS(T)                                                                                               \
  row("1 wave/SIMD, 1 chain (latency)", T, 1, 1, WmmaTime<T, 1, 0>(80, 32, iters, out), iters);                  \
  row("1 wave/SIMD, 4 chains", T, 1, 4, WmmaTime<T, 4, 0>(80, 32, iters, out), iters);                          \
  row("1 wave/SIMD, 8 chains", T, 1, 8, WmmaTime<T, 8, 0>(80, 32, iters, out), iters);                          \
  row("4 waves/SIMD, 1 chain", T, 4, 1, WmmaTime<T, 1, 0>(80, 128, iters, out), iters);                         \
  row("4 waves/SIMD, 4 chains", T, 4, 4, WmmaTime<T, 4, 0>(80, 128, iters, out), iters);                        \
  row("8 waves/SIMD, 4 chains", T, 8, 4, WmmaTime<T, 4, 0>(160, 128, iters, out), iters);                       \
  row("4 waves/SIMD, 4 ch + 4 FMA/WMMA", T, 4, 4, WmmaTime<T, 4, 4>(80, 128, iters, out), iters);               \
  row("4 waves/SIMD, 4 ch + 16 FMA/WMMA", T, 4, 4, WmmaTime<T, 4, 16>(80, 128, iters, out), iters);
  WMMA_ROWS(0)
  WMMA_ROWS(1)
  WMMA_ROWS(2)
  row("1 wave/SIMD, 1 chain, again", 0, 1, 1, WmmaTime<0, 1, 0>(80, 32, iters, out), iters);
#undef WMMA_ROWS
  CHECK(hipFree(out));
}

// ---- lds ---------------------------------------------------------------------
// Each wave reads (or writes) LDS `iters` times; lane l touches element
// (l * stride + it) of width W bytes. Reports bytes per cycle per CU.
template <int W, bool STORE>
__global__ void __launch_bounds__(256) LdsKernel(int iters, int stride, float* out) {
  // Index math is a shift and a mask (a modulo per access swamped the first
  // version); a compiler barrier after each store keeps every one.
  __shared__ __attribute__((aligned(16))) unsigned char lds[65536];
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  for (int i = threadIdx.x; i < 65536 / 4; i += blockDim.x) reinterpret_cast<unsigned*>(lds)[i] = i;
  __syncthreads();
  unsigned acc = 0;
  constexpr unsigned mask = 65536 / W - 1;
  const unsigned base = static_cast<unsigned>(lane * stride + wave * 977);
#pragma unroll 8
  for (int it = 0; it < iters; ++it) {
    const unsigned idx = ((base + static_cast<unsigned>(it) * 64u) & mask) * W;
    if constexpr (STORE) {
      if constexpr (W == 2) *reinterpret_cast<unsigned short*>(lds + idx) = static_cast<unsigned short>(it);
      else if constexpr (W == 4) *reinterpret_cast<unsigned*>(lds + idx) = it;
      else if constexpr (W == 8) *reinterpret_cast<uint2*>(lds + idx) = make_uint2(it, lane);
      else *reinterpret_cast<uint4*>(lds + idx) = make_uint4(it, lane, it, lane);
      asm volatile("" ::: "memory");  // keep every store
    } else {
      if constexpr (W == 2) acc += *reinterpret_cast<const unsigned short*>(lds + idx);
      else if constexpr (W == 4) acc += *reinterpret_cast<const unsigned*>(lds + idx);
      else if constexpr (W == 8) { const uint2 v = *reinterpret_cast<const uint2*>(lds + idx); acc += v.x ^ v.y; }
      else { const uint4 v = *reinterpret_cast<const uint4*>(lds + idx); acc += v.x ^ v.y ^ v.z ^ v.w; }
    }
  }
  __syncthreads();
  if (STORE) acc = reinterpret_cast<const unsigned*>(lds)[threadIdx.x * 7];  // a word: a byte could never equal the sentinel, and the kernel folded away
  if (acc == 0xdeadbeef) out[0] = acc;
}

template <int W, bool STORE>
double LdsTime(int stride, int iters, float* out) {
  hipLaunchKernelGGL((LdsKernel<W, STORE>), 40, 256, 0, 0, 10, stride, out);
  CHECK(hipDeviceSynchronize());
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  CHECK(hipEventRecord(e0));
  hipLaunchKernelGGL((LdsKernel<W, STORE>), 40, 256, 0, 0, iters, stride, out);
  CHECK(hipEventRecord(e1));
  CHECK(hipEventSynchronize(e1));
  float ms;
  CHECK(hipEventElapsedTime(&ms, e0, e1));
  return ms;
}

void Lds() {
  constexpr double kClockGHz = 2.88;
  float* out;
  CHECK(hipMalloc(&out, 64));
  const int iters = 20000;
  std::printf("LDS: 40 blocks x 8 waves (one block per WGP, 64 KB each), lane l at element l * stride; bytes/cycle per WGP\n");
  std::printf("  %-6s %6s %10s %10s\n", "width", "stride", "load", "store");
  const auto row = [&](int w, int stride, double ld, double st) {
    const double bytes = 40.0 * 256 * iters * w;
    const auto bpc = [&](double ms) { return bytes / (ms * 1e-3 * kClockGHz * 1e9) / 20; };  // 20 WGPs
    std::printf("  %4d B %6d %10.1f %10.1f\n", w, stride, bpc(ld), bpc(st));
  };
  for (const int stride : {1, 2, 4, 8, 16, 32, 33, 64}) {
    row(2, stride, LdsTime<2, false>(stride, iters, out), LdsTime<2, true>(stride, iters, out));
    row(4, stride, LdsTime<4, false>(stride, iters, out), LdsTime<4, true>(stride, iters, out));
    row(8, stride, LdsTime<8, false>(stride, iters, out), LdsTime<8, true>(stride, iters, out));
    row(16, stride, LdsTime<16, false>(stride, iters, out), LdsTime<16, true>(stride, iters, out));
  }
  CHECK(hipFree(out));
}

// ---- latency -------------------------------------------------------------------
// Dependent pointer chase over a working set: latency per load by footprint.
__global__ void ChaseKernel(const unsigned* next, int steps, unsigned* out) {
  unsigned p = 0;
  const unsigned long long t0 = wall_clock64();
  for (int i = 0; i < steps; ++i) p = next[p];
  const unsigned long long t1 = wall_clock64();
  out[0] = p;
  reinterpret_cast<unsigned long long*>(out)[1] = t1 - t0;
}

// Memory-level parallelism: each lane keeps K independent 16-byte loads in
// flight over a 256 MB buffer (random 4 KB-aligned pages, contiguous within).
template <int K>
__global__ void __launch_bounds__(256) MlpKernel(const float4* buf, std::size_t n_f4, int iters, float* out) {
  float acc = 0.0f;
  unsigned x = blockIdx.x * 2654435761u + threadIdx.x / 32 * 40503u;
  const unsigned lane = threadIdx.x % 32;
  for (int it = 0; it < iters; ++it) {
    float4 v[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
      x = x * 1664525u + 1013904223u;
      const std::size_t page = (x >> 8) % (n_f4 / 256);
      v[k] = buf[page * 256 + lane * 8 + k % 8];
    }
#pragma unroll
    for (int k = 0; k < K; ++k) acc += v[k].x;
  }
  if (acc == 12345.f) out[0] = acc;
}

template <int K>
double MlpGBs(int blocks, int threads, const float4* buf, std::size_t n_f4, float* out) {
  const int iters = 2000;
  hipLaunchKernelGGL((MlpKernel<K>), blocks, threads, 0, 0, buf, n_f4, 10, out);
  CHECK(hipDeviceSynchronize());
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  CHECK(hipEventRecord(e0));
  hipLaunchKernelGGL((MlpKernel<K>), blocks, threads, 0, 0, buf, n_f4, iters, out);
  CHECK(hipEventRecord(e1));
  CHECK(hipEventSynchronize(e1));
  float ms;
  CHECK(hipEventElapsedTime(&ms, e0, e1));
  return double(blocks) * threads * iters * K * 16 / (ms * 1e-3) / 1e9;
}

void Latency() {
  std::printf("dependent load latency (one lane, random cycle over the footprint)\n");
  unsigned* out;
  CHECK(hipMalloc(&out, 64));
  for (const std::size_t kb : {16ul, 256ul, 2048ul, 8192ul, 24576ul, 65536ul, 262144ul}) {
    const std::size_t n = kb * 1024 / 64;  // one entry per 64-byte line
    std::vector<unsigned> perm(n), next(n * 16, 0);
    for (std::size_t i = 0; i < n; ++i) perm[i] = static_cast<unsigned>(i);
    unsigned s = 12345;
    for (std::size_t i = n - 1; i > 0; --i) {
      s = s * 1664525u + 1013904223u;
      std::swap(perm[i], perm[s % (i + 1)]);
    }
    for (std::size_t i = 0; i < n; ++i) next[perm[i] * 16] = perm[(i + 1) % n] * 16;
    unsigned* d;
    CHECK(hipMalloc(&d, next.size() * 4));
    CHECK(hipMemcpy(d, next.data(), next.size() * 4, hipMemcpyHostToDevice));
    const int steps = 20000;
    hipLaunchKernelGGL(ChaseKernel, 1, 1, 0, 0, d, steps, out);  // warm
    hipLaunchKernelGGL(ChaseKernel, 1, 1, 0, 0, d, steps, out);
    CHECK(hipDeviceSynchronize());
    unsigned long long h[2];
    CHECK(hipMemcpy(h, out, 16, hipMemcpyDeviceToHost));
    std::printf("  footprint %7zu KB: %6.0f ns per dependent load\n", kb, h[1] * 10.0 / steps);
    CHECK(hipFree(d));
  }
  std::printf("random-page read bandwidth vs loads in flight per lane and waves (256 MB buffer, 16 B per load)\n");
  const std::size_t bytes = 256ull << 20, n_f4 = bytes / 16;
  float4* buf;
  float* fo;
  CHECK(hipMalloc(&buf, bytes));
  CHECK(hipMalloc(&fo, 64));
  CHECK(hipMemset(buf, 1, bytes));
  std::printf("  %-22s %8s %8s %8s %8s\n", "waves", "K=1", "K=2", "K=4", "K=8");
  for (const auto& [blocks, threads, label] :
       std::vector<std::tuple<int, int, const char*>>{{80, 32, "1 per SIMD"}, {80, 128, "4 per SIMD"},
                                                      {160, 256, "16 per SIMD"}}) {
    std::printf("  %-22s %8.1f %8.1f %8.1f %8.1f GB/s\n", label, MlpGBs<1>(blocks, threads, buf, n_f4, fo),
                MlpGBs<2>(blocks, threads, buf, n_f4, fo), MlpGBs<4>(blocks, threads, buf, n_f4, fo),
                MlpGBs<8>(blocks, threads, buf, n_f4, fo));
  }
  CHECK(hipFree(buf));
  CHECK(hipFree(out));
}

// ---- sync / launch -------------------------------------------------------------
// MODE 0: __syncthreads with no memory in flight; 1: __syncthreads with a
// global load issued just before (it waits for vmcnt(0)); 2: an LDS-only
// fence + s_barrier with the same load in flight (consumed after the loop).
template <int MODE>
__global__ void SyncKernel(const float* g, int iters, float* out) {
  __shared__ float s[1024];
  // The load of iteration it is consumed in iteration it + 1, so only a
  // barrier that waits for all memory (vmcnt(0)) has to wait for it.
  float acc = 0.0f, pending = 0.0f, next = 0.0f;
  for (int it = 0; it < iters; ++it) {
    pending += next;
    if constexpr (MODE >= 1) next = g[(it * 977 + threadIdx.x * 31 + blockIdx.x * 4099) & ((1 << 24) - 1)];
    s[threadIdx.x] = acc;
    if constexpr (MODE == 2) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    } else {
      __syncthreads();
    }
    acc += s[(threadIdx.x + 1) % blockDim.x];
  }
  if (acc + pending + next == 12345.f) out[0] = acc;
}

__global__ void EmptyKernel(float* out) {
  if (threadIdx.x == 12345) out[0] = 1.0f;
}

template <int MODE>
double SyncNs(int blocks, int threads, const float* g, float* out) {
  const int iters = 20000;
  hipLaunchKernelGGL(SyncKernel<MODE>, blocks, threads, 0, 0, g, 100, out);
  CHECK(hipDeviceSynchronize());
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  CHECK(hipEventRecord(e0));
  hipLaunchKernelGGL(SyncKernel<MODE>, blocks, threads, 0, 0, g, iters, out);
  CHECK(hipEventRecord(e1));
  CHECK(hipEventSynchronize(e1));
  float ms;
  CHECK(hipEventElapsedTime(&ms, e0, e1));
  return ms * 1e6 / iters;
}

void Sync() {
  float *g, *out;
  CHECK(hipMalloc(&g, (1ull << 24) * 4));
  CHECK(hipMemset(g, 0, (1ull << 24) * 4));
  CHECK(hipMalloc(&out, 64));
  std::printf("barrier round (LDS write, barrier, LDS read) per iteration, ns; 80 blocks (one per SIMD)\n");
  std::printf("  %-14s %12s %22s %22s\n", "block", "syncthreads", "syncthreads+load", "LDS-only fence+load");
  for (const int threads : {32, 64, 128, 256, 512, 1024}) {
    std::printf("  %4d threads   %12.1f %22.1f %22.1f\n", threads, SyncNs<0>(80, threads, g, out),
                SyncNs<1>(80, threads, g, out), SyncNs<2>(80, threads, g, out));
  }
  // Launch overhead: back-to-back empty kernels on one stream.
  for (int i = 0; i < 100; ++i) hipLaunchKernelGGL(EmptyKernel, 1, 64, 0, 0, out);
  CHECK(hipDeviceSynchronize());
  for (const int blocks : {1, 1024}) {
    const int n = 5000;
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    CHECK(hipEventRecord(e0));
    for (int i = 0; i < n; ++i) hipLaunchKernelGGL(EmptyKernel, blocks, 64, 0, 0, out);
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    std::printf("back-to-back empty kernels, %4d blocks: %.2f us per launch\n", blocks, ms * 1e3 / n);
  }
  CHECK(hipFree(g));
  CHECK(hipFree(out));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "clock") Clock();
  else if (cmd == "dram") Dram();
  else if (cmd == "wmma") Wmma();
  else if (cmd == "lds") Lds();
  else if (cmd == "latency") Latency();
  else if (cmd == "sync") Sync();
  else {
    std::fprintf(stderr, "usage: halo_hwprobe clock|dram|wmma|lds|latency|sync\n");
    return 2;
  }
  return 0;
}
