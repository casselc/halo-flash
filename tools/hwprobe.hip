// halo_hwprobe: measured facts about the GPU (gfx1151 Strix Halo) that the
// performance work relies on. Standalone HIP; each subcommand prints one table.
//
//   halo_hwprobe keepalive [ms] [s] an empty marker every ms: keeps rocprofv3 instruction counters armed
//   halo_hwprobe sweep [custom]   staged-read patterns as kernels issue them (feeds perf_model)
//   halo_hwprobe rounds          dependent load rounds vs concurrency (feeds perf_model)
//   halo_hwprobe empty           cost of blocks that exit at once (grids sized for the most tiles)
//   halo_hwprobe stream          the hyper-connection residual's DRAM traffic patterns
//   halo_hwprobe clock           shader clock (in-kernel counters) vs load
//   halo_hwprobe dram [MB]       bandwidth vs segment size, dispatch order, read/write
//   halo_hwprobe wmma            f16 / bf16 / iu8 / iu4 WMMA rate, chain latency, VALU overlap
//
// Clock truth comes from inside kernels: SHADER_CYCLES (20 bits, wraps every
// ~0.35 ms at 2.9 GHz) against the 100 MHz realtime counter. sysfs reports the
// requested level, not the delivered clock.
#include <hip/hip_runtime.h>

#include <atomic>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

void Dram(std::uint32_t mb) {
  // Default 480 MB: past the 32 MB MALL. A footprint inside the MALL (e.g. 16)
  // tells DRAM row locality from cache / translation effects.
  const std::uint32_t row_bytes = 40960, rows = std::max<std::uint32_t>(128, (mb << 20) / row_bytes / 128 * 128);
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
  std::printf("memory: %zu MB as rows of 40 KB; a block takes `seg` bytes from each of 128 rows (256 threads, 16-byte accesses)\n", bytes >> 20);
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

// ---- stream: the hyper-connection residual's traffic ---------------------------
// res is 4 streams x T x h fp32 (84 MB at T = 2048, h = 2560), past the MALL.
// Modes, grid-stride with 16-byte accesses: 0 read res; 1 res += 1 in place;
// 2 the combine's pattern: res[t][s] += out[t] (out shared by the 4 streams)
// in place, plus a half-size fp16 copy written.
template <int MODE>
__global__ void __launch_bounds__(256) StreamKernel(float4* res, const float4* out, uint2* y16, std::size_t n_f4,
                                                    std::uint32_t h_f4, float* sink) {
  float acc = 0.0f;
  for (std::size_t i = blockIdx.x * 256ull + threadIdx.x; i < n_f4; i += gridDim.x * 256ull) {
    float4 v = res[i];
    if constexpr (MODE == 0) {
      acc += v.x;
    } else {
      if constexpr (MODE == 2) {
        const float4 o = out[(i / (4ull * h_f4)) * h_f4 + i % h_f4];
        v.x += o.x, v.y += o.y, v.z += o.z, v.w += o.w;
        const _Float16 hv[4] = {_Float16(v.x), _Float16(v.y), _Float16(v.z), _Float16(v.w)};
        y16[i] = *reinterpret_cast<const uint2*>(hv);
      } else {
        v.x += 1.0f;
      }
      res[i] = v;
    }
  }
  if (acc == 12345.f) sink[0] = acc;
}

// The fused mixer's memory side: block b owns tokens [b TT, b TT + TT) (all 4 streams). Pass A: res += out in
// place; pass B: re-read the tile (the down GEMM's xn); pass C: re-read it again and write mixed (fp32 + fp16, h
// per token). `passes` 1 runs A only, 3 all.
__global__ void __launch_bounds__(1024) StreamTileKernel(float4* res, const float4* out, float4* mixed, uint2* mixed16,
                                                         std::uint32_t tt, std::uint32_t h_f4, int passes, float* sink) {
  const std::size_t t0 = std::size_t{blockIdx.x} * tt, tile = std::size_t{tt} * 4 * h_f4;
  float4* r = res + t0 * 4 * h_f4;
  float acc = 0.0f;
  for (std::size_t i = threadIdx.x; i < tile; i += blockDim.x) {
    float4 v = r[i];
    const float4 o = out[(t0 + i / (4 * h_f4)) * h_f4 + i % h_f4];
    v.x += o.x, v.y += o.y, v.z += o.z, v.w += o.w;
    r[i] = v;
  }
  if (passes > 1) {
    __syncthreads();
    for (std::size_t i = threadIdx.x; i < tile; i += blockDim.x) acc += r[i].x;
    __syncthreads();
    for (std::size_t i = threadIdx.x; i < std::size_t{tt} * h_f4; i += blockDim.x) {
      const std::size_t t = i / h_f4, c = i % h_f4;
      float4 m = {acc, 0, 0, 0};
      for (std::uint32_t st = 0; st < 4; ++st) {
        const float4 v = r[(t * 4 + st) * h_f4 + c];
        m.x += v.x, m.y += v.y, m.z += v.z, m.w += v.w;
      }
      mixed[(t0 + t) * h_f4 + c] = m;
      const _Float16 hv[4] = {_Float16(m.x), _Float16(m.y), _Float16(m.z), _Float16(m.w)};
      mixed16[(t0 + t) * h_f4 + c] = *reinterpret_cast<const uint2*>(hv);
    }
  }
  if (acc == 12345.f) sink[0] = acc;
}

// ---- sweep: staged reads as a kernel's blocks issue them ----------------------
// Block b (coordinates b % n0, b / n0), stage k reads `seg` bytes from each of R rows at
// (b % n0) s0 + (b / n0) s1 + r rs + k ks (bytes); the next stage's loads are issued before the current one is
// consumed (a GEMM's one-stage prefetch), one barrier a stage. rmw: the read bytes are also written back.
// `lds` bytes of dynamic LDS per block give the kernel's residency. tools/perf_model.py prices access
// patterns from these measurements.
template <int P>
__global__ void __launch_bounds__(256) SweepKernel(char* g, std::uint32_t n0, std::uint64_t s0, std::uint64_t s1,
                                                   std::uint64_t rs, std::uint64_t ks, std::uint32_t R,
                                                   std::uint32_t seg, std::uint32_t S, int rmw, float* sink) {
  extern __shared__ float lds_dyn[];
  const std::uint32_t per_row = seg / 16, pieces = R * per_row;
  char* base = g + (blockIdx.x % n0) * s0 + (blockIdx.x / n0) * s1;
  float4 cur[P], nxt[P];
  const auto at = [&](std::uint32_t k, std::uint32_t q) {
    return reinterpret_cast<float4*>(base + (q / per_row) * rs + k * ks + (q % per_row) * 16);
  };
  const auto load = [&](std::uint32_t k) {
#pragma unroll
    for (int p = 0; p < P; ++p) {
      const std::uint32_t q = threadIdx.x + p * 256;
      nxt[p] = q < pieces ? *at(k, q) : make_float4(0, 0, 0, 0);
    }
  };
  float acc = 0.0f;
  load(0);
  for (std::uint32_t k = 0; k < S; ++k) {
#pragma unroll
    for (int p = 0; p < P; ++p) cur[p] = nxt[p];
    if (k + 1 < S) load(k + 1);
#pragma unroll
    for (int p = 0; p < P; ++p) {
      const std::uint32_t q = threadIdx.x + p * 256;
      acc += cur[p].x + cur[p].w;
      if (rmw && q < pieces) *at(k, q) = cur[p];
    }
    __syncthreads();
  }
  if (acc == 12345.f) sink[0] = acc + lds_dyn[0];
}

struct SweepCfg {
  const char* name;
  std::uint32_t blocks, n0;
  std::uint64_t s0, s1, rs, ks;
  std::uint32_t R, seg, S, lds;
  int rmw;
};

double RunSweep(const SweepCfg& c, char* g, std::size_t gbytes, float* sink) {
  const std::uint32_t pieces = c.R * (c.seg / 16), P = (pieces + 255) / 256;
  const std::uint64_t span = (c.n0 - 1) * c.s0 + (c.blocks / c.n0) * c.s1 + c.R * c.rs + c.S * c.ks + c.seg;
  if (span > gbytes) {
    std::printf("  %-40s span %zu MB exceeds the buffer\n", c.name, static_cast<std::size_t>(span >> 20));
    return 0;
  }
  const auto run = [&] {
    switch (P) {
#define HALO_SWEEP(n) case n: hipLaunchKernelGGL(SweepKernel<n>, c.blocks, 256, c.lds, 0, g, c.n0, c.s0, c.s1, c.rs, c.ks, c.R, c.seg, c.S, c.rmw, sink); break;
      HALO_SWEEP(1) HALO_SWEEP(2) HALO_SWEEP(3) HALO_SWEEP(4) HALO_SWEEP(8) HALO_SWEEP(16) HALO_SWEEP(32)
#undef HALO_SWEEP
      default: std::printf("  %-40s %u pieces a thread: no instance\n", c.name, P); std::exit(1);
    }
  };
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  for (int w = 0; w < 3; ++w) run();
  CHECK(hipDeviceSynchronize());
  CHECK(hipEventRecord(e0));
  for (int r = 0; r < 10; ++r) run();
  CHECK(hipEventRecord(e1));
  CHECK(hipEventSynchronize(e1));
  float ms;
  CHECK(hipEventElapsedTime(&ms, e0, e1));
  ms /= 10;
  const double bytes = double(c.blocks) * c.R * c.seg * c.S * (c.rmw ? 2 : 1);
  std::printf("  %-40s %7.3f ms  %6.1f GB/s  (%zu MB%s)\n", c.name, ms, bytes / (ms * 1e-3) / 1e9,
              static_cast<std::size_t>(bytes / 1e6), c.rmw ? " moved, read + write" : " read");
  return ms;
}

void Sweep(int argc, char** argv) {
  const std::size_t gbytes = 512ull << 20;
  char* g;
  float* sink;
  CHECK(hipMalloc(&g, gbytes));
  CHECK(hipMalloc(&sink, 64));
  CHECK(hipMemset(g, 0, gbytes));
  if (argc >= 13) {  // sweep blocks n0 s0 s1 rs ks R seg S lds rmw
    const SweepCfg c{"custom", static_cast<std::uint32_t>(std::atoi(argv[2])), static_cast<std::uint32_t>(std::atoi(argv[3])),
                     std::strtoull(argv[4], nullptr, 10), std::strtoull(argv[5], nullptr, 10),
                     std::strtoull(argv[6], nullptr, 10), std::strtoull(argv[7], nullptr, 10),
                     static_cast<std::uint32_t>(std::atoi(argv[8])), static_cast<std::uint32_t>(std::atoi(argv[9])),
                     static_cast<std::uint32_t>(std::atoi(argv[10])), static_cast<std::uint32_t>(std::atoi(argv[11])),
                     std::atoi(argv[12])};
    RunSweep(c, g, gbytes, sink);
  } else {
    // The prefill's activation patterns at T = 2048 (token rows: residual 40 KB fp32, fp16 copy 20 KB).
    const std::uint64_t KB = 1024;
    const SweepCfg cfgs[] = {
        {"down GEMM from res fp32 (_mr)", 32, 32, 64 * 40 * KB, 0, 40 * KB, 256, 64, 256, 160, 55552, 0},
        {"down GEMM from fp16 row-major (_m)", 32, 32, 64 * 20 * KB, 0, 20 * KB, 128, 64, 128, 160, 55296, 0},
        {"down GEMM from fp16 K-blocked (_mb)", 32, 32, 8 * KB, 0, 0, 2048 * 128, 1, 8192, 160, 55296, 0},
        {"down GEMM from res fp32 K-blocked", 32, 32, 16 * KB, 0, 0, 2048 * 256, 1, 16384, 160, 55552, 0},
        {"mix epilogue res, row tiles fastest", 640, 40, 256, 128 * 40 * KB, 10 * KB, 0, 512, 256, 1, 31744, 0},
        {"mix epilogue res, token tiles fastest", 640, 16, 128 * 40 * KB, 256, 10 * KB, 0, 512, 256, 1, 31744, 0},
        {"combine rows (one 10 KB row a block), rmw", 8192, 8192, 10 * KB, 0, 0, 0, 1, 10 * KB, 1, 384, 1},
        {"stream contiguous 84 MB, 1280 blocks", 1280, 1280, 64 * KB, 0, 0, 0, 1, 64 * KB, 1, 0, 0},
    };
    std::printf("sweep: staged reads (B blocks, R rows x seg bytes a stage, S stages; one-stage prefetch)\n");
    for (const auto& c : cfgs) RunSweep(c, g, gbytes, sink);
  }
  CHECK(hipFree(g));
}

// ---- rounds: dependent load rounds under concurrency ------------------------
// Each wave runs N rounds of L coalesced 128-byte loads (one float a lane); the next round's address depends
// on this round's values, so every round waits for its loads (a loop whose iterations each wait on their own
// loads). src 0: fresh addresses in a 512 MB buffer (DRAM); 1: a 1 MB buffer (L2 after the first touch).
// `offset` (floats) moves each launch to addresses no recent launch touched, so DRAM rounds miss the MALL.
template <int L>
__global__ void __launch_bounds__(1024) RoundsKernel(const float* g, std::uint64_t n_floats, int N, int src,
                                                     std::uint64_t offset, float* sink) {
  const std::uint64_t w = (blockIdx.x * std::uint64_t{blockDim.x} + threadIdx.x) / 32, lane = threadIdx.x % 32;
  const std::uint64_t mod = src ? (1ull << 18) : n_floats;
  float acc = 0.0f;
  std::uint64_t dep = 0;
  for (int i = 0; i < N; ++i) {
    const std::uint64_t b = ((w * N + i) * L * 32 + dep + (src ? 0 : offset)) % mod;
    float v[L];
#pragma unroll
    for (int l = 0; l < L; ++l) v[l] = g[(b + l * 32 + lane) % mod];
#pragma unroll
    for (int l = 0; l < L; ++l) acc += v[l];
    dep = acc == 12345.f ? 1 : 0;
  }
  if (acc == 12345.f) sink[0] = acc;
}

void Rounds() {
  const std::size_t n = (512ull << 20) / 4;
  float *g, *sink;
  CHECK(hipMalloc(&g, n * 4));
  CHECK(hipMalloc(&sink, 64));
  CHECK(hipMemset(g, 0, n * 4));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::printf("rounds: each wave N dependent rounds of L 128-byte loads; ns per round (per wave) and GB/s\n");
  std::printf("  %-5s %3s %6s %10s %10s\n", "src", "L", "waves", "ns/round", "GB/s");
  for (int src = 0; src < 2; ++src)
    for (const int L : {1, 8, 16})
      for (const std::uint32_t waves : {80u, 160u, 320u, 640u, 1280u, 2560u}) {
        const int N = 100;
        std::uint64_t offset = 0;
        const auto run = [&] {
          const std::uint32_t blocks = waves / 8;
          offset += std::uint64_t{waves} * N * L * 32 + (1ull << 20);  // past the last launch's footprint
          if (L == 1) hipLaunchKernelGGL(RoundsKernel<1>, blocks, 256, 0, 0, g, n, N, src, offset, sink);
          else if (L == 8) hipLaunchKernelGGL(RoundsKernel<8>, blocks, 256, 0, 0, g, n, N, src, offset, sink);
          else hipLaunchKernelGGL(RoundsKernel<16>, blocks, 256, 0, 0, g, n, N, src, offset, sink);
        };
        for (int w = 0; w < 2; ++w) run();
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(e0));
        for (int r = 0; r < 5; ++r) run();
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        ms /= 5;
        std::printf("  %-5s %3d %6u %10.0f %10.1f\n", src ? "L2" : "DRAM", L, waves, ms * 1e6 / N,
                    double(waves) * N * L * 128 / (ms * 1e-3) / 1e9);
      }
}

// ---- empty: the cost of blocks that exit at once (a grid sized for the most tiles) ----------
// Blocks with blockIdx.y >= *active return after one scalar load, as the grouped MoE kernels' blocks past the
// routed tile count do; the others spin `work` iterations.
__global__ void __launch_bounds__(256) EmptyTilesKernel(const std::uint32_t* active, std::uint32_t work, float* sink) {
  if (blockIdx.y >= *active) return;
  float a = threadIdx.x;
  for (std::uint32_t i = 0; i < work; ++i) a = a * 1.0001f + 0.5f;
  if (a == 12345.f) sink[0] = a;
}

void EmptyTiles() {
  std::uint32_t* active;
  float* sink;
  CHECK(hipMalloc(&active, 4));
  CHECK(hipMalloc(&sink, 64));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::printf("empty: grid (x, tiles) of 256-thread blocks; blocks with tile >= active exit at once\n");
  for (const std::uint32_t gx : {20u, 5u})
    for (const std::uint32_t n : {285u, 833u}) {
      CHECK(hipMemcpy(active, &n, 4, hipMemcpyHostToDevice));
      for (const std::uint32_t gy : {n, 833u}) {
        const auto run = [&] { hipLaunchKernelGGL(EmptyTilesKernel, dim3(gx, gy), 256, 0, 0, active, 2000u, sink); };
        for (int w = 0; w < 3; ++w) run();
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(e0));
        for (int r = 0; r < 10; ++r) run();
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        std::printf("  grid %2u x %3u, active tiles %3u: %7.3f ms\n", gx, gy, n, ms / 10);
      }
    }
  // Empty blocks alone.
  const std::uint32_t zero = 0;
  CHECK(hipMemcpy(active, &zero, 4, hipMemcpyHostToDevice));
  for (const std::uint32_t gy : {833u, 548u}) {
    const auto run = [&] { hipLaunchKernelGGL(EmptyTilesKernel, dim3(20, gy), 256, 0, 0, active, 0u, sink); };
    for (int w = 0; w < 3; ++w) run();
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    for (int r = 0; r < 10; ++r) run();
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    std::printf("  all-empty grid 20 x %u (%u blocks): %.3f ms (%.0f ns a block)\n", gy, 20 * gy, ms / 10, ms / 10 * 1e6 / (20.0 * gy));
  }
}

void Stream() {
  const std::uint32_t T = 2048, h = 2560;
  const std::size_t n = std::size_t{4} * T * h;  // floats
  float4 *res, *out;
  uint2* y16;
  float* sink;
  CHECK(hipMalloc(&res, n * 4));
  CHECK(hipMalloc(&out, n));
  CHECK(hipMalloc(&y16, n * 2));
  CHECK(hipMalloc(&sink, 64));
  CHECK(hipMemset(res, 0, n * 4));
  CHECK(hipMemset(out, 0, n));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::printf("stream: residual 4 x %u x %u fp32 (%zu MB)\n", T, h, (n * 4) >> 20);
  const char* names[] = {"read res", "res += 1 in place", "combine: res += out, fp16 copy"};
  const double moved[] = {n * 4.0, n * 8.0, n * 8.0 + n + n * 2.0};
  for (int mode = 0; mode < 3; ++mode) {
    for (const std::uint32_t blocks : {1280u, 8192u}) {
      const auto run = [&] {
        if (mode == 0) hipLaunchKernelGGL(StreamKernel<0>, blocks, 256, 0, 0, res, out, y16, n / 4, h / 4, sink);
        else if (mode == 1) hipLaunchKernelGGL(StreamKernel<1>, blocks, 256, 0, 0, res, out, y16, n / 4, h / 4, sink);
        else hipLaunchKernelGGL(StreamKernel<2>, blocks, 256, 0, 0, res, out, y16, n / 4, h / 4, sink);
      };
      for (int w = 0; w < 3; ++w) run();
      CHECK(hipDeviceSynchronize());
      CHECK(hipEventRecord(e0));
      for (int r = 0; r < 10; ++r) run();
      CHECK(hipEventRecord(e1));
      CHECK(hipEventSynchronize(e1));
      float ms;
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      std::printf("  %-32s %5u blocks  %7.3f ms  %6.1f GB/s (bytes moved %zu MB)\n", names[mode], blocks, ms / 10,
                  moved[mode] / (ms / 10 * 1e-3) / 1e9, static_cast<std::size_t>(moved[mode]) >> 20);
    }
  }
  // Token tiles (the fused mixer's memory side).
  float4* mixed;
  CHECK(hipMalloc(&mixed, n));
  for (const int passes : {1, 3})
    for (const std::uint32_t tt : {8u, 16u, 32u, 64u})
      for (const std::uint32_t threads : {256u, 1024u}) {
        const auto run = [&] {
          hipLaunchKernelGGL(StreamTileKernel, T / tt, threads, 0, 0, res, out, mixed, y16, tt, h / 4, passes, sink);
        };
        for (int w = 0; w < 3; ++w) run();
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(e0));
        for (int r = 0; r < 10; ++r) run();
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        const double dram = n * 8.0 + n + (passes > 1 ? n * 1.5 / 4 * 4 / 4 * 1.0 : 0.0);  // res rw, out, mixed
        std::printf("  tiles of %2u tokens x %4u threads (%4u blocks), %s  %7.3f ms  (%.0f GB/s of unique bytes)\n", tt,
                    threads, T / tt, passes > 1 ? "A+B+C" : "A    ", ms / 10, dram / (ms / 10 * 1e-3) / 1e9);
      }
  CHECK(hipFree(mixed));
  CHECK(hipFree(res));
  CHECK(hipFree(out));
  CHECK(hipFree(y16));
}

// ---- wmma --------------------------------------------------------------------
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));

// T: 0 f16, 1 bf16, 2 iu8, 3 iu4 (16x16x16, both operands 4-bit: 8 bytes per lane). ACC independent accumulators per wave; VALU
// independent FMAs per WMMA interleaved (0 = none). Each wave runs `iters`
// rounds of ACC WMMAs.
template <int T, int ACC, int VALU>
__global__ void WmmaKernel(int iters, float* out) {
  const int lane = threadIdx.x % 32;
  v16h ah, bh;
  v16s as, bs;
  v4i ai = {lane, 3, 5, 7}, bi = {1, lane, 2, 9};
  v2i a4 = {lane, 0x1234567}, b4 = {0x7654321, lane};
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
      else if constexpr (T == 2) ci[a] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ai, true, bi, ci[a], false);
      else ci[a] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a4, true, b4, ci[a], false);
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
  const char* names[4] = {"f16", "bf16", "iu8", "iu4"};
  std::printf("WMMA 16x16x16 (one wave-instruction = 8192 FLOP/OP); cycles per WMMA per SIMD at %.2f GHz\n", kClockGHz);
  const auto row = [&](const char* label, int t, int waves_per_simd, int acc, double ms, int iters) {
    const double wmmas = kSimds * waves_per_simd * double(iters) * acc;
    const double cyc = ms * 1e-3 * kClockGHz * 1e9 / (wmmas / kSimds);
    std::printf("  %-4s %-28s %6.2f cycles/WMMA/SIMD  %6.1f T%s/s\n", names[t], label, cyc,
                wmmas * 8192 / (ms * 1e-3) / 1e12, t >= 2 ? "OP" : "FLOP");
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
  WMMA_ROWS(3)
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

// ---- counters: known-answer kernels for validating hardware counters ---------
// Each kernel issues an exact number of one instruction kind per wave through
// inline asm (so the compiler cannot add, drop or fuse them); pairs of N cancel
// the fixed prologue. tools/counter_check.py runs them under rocprofv3 and
// checks SQ_INSTS_* and GL2C_* against these expectations, and checks the ISA.
#define HALO_REP4(x) x x x x
#define HALO_REP16(x) HALO_REP4(HALO_REP4(x))
#define HALO_REP64(x) HALO_REP16(HALO_REP4(x))
// kind 0: v_add_f32; 1: ds_load_b32; 2: global_load_b32; 3: WMMA f16
template <int KIND, int REPS64>
__global__ void __launch_bounds__(64) KnownKernel(float* g, float* out) {
  __shared__ float s[256];
  s[threadIdx.x] = 1.0f;
  float a = threadIdx.x, b = 1.0f;
  float* gp = g + threadIdx.x;
  // The LDS byte offset of s[threadIdx.x]: s is this kernel's only LDS allocation, at offset 0
  // (a generic pointer's value is a flat address, not an LDS offset).
  const unsigned lds_addr = threadIdx.x * 4u;
  __syncthreads();
  for (int r = 0; r < REPS64; ++r) {
    if constexpr (KIND == 0) {
      asm volatile(HALO_REP64("v_add_f32 %0, %0, %1\n") : "+v"(a) : "v"(b));
    } else if constexpr (KIND == 1) {
      // Early-clobber outputs: otherwise the result may share the address register, and
      // the first load overwrites the address (it faulted for global loads).
      asm volatile(HALO_REP64("ds_load_b32 %0, %1\n") "s_waitcnt lgkmcnt(0)\n" : "=&v"(b) : "v"(lds_addr));
    } else if constexpr (KIND == 2) {
      asm volatile(HALO_REP64("global_load_b32 %0, %1, off\n") "s_waitcnt vmcnt(0)\n" : "=&v"(b) : "v"(gp));
    } else {
      v16h x = {};
      v8f c = {};
      for (int i = 0; i < 64; ++i) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(x, x, c);
      a += c[0];
    }
  }
  if (a + b == 12345.f) out[0] = a;
}
// Exec mask all off for the asm block: are masked-off instructions counted?
__global__ void __launch_bounds__(64) KnownMaskedKernel(float* out) {
  float a = threadIdx.x, b = 1.0f;
  if (threadIdx.x > 1000) {  // false for every lane: the block runs with exec = 0 or is skipped
    for (int r = 0; r < 64; ++r) asm volatile(HALO_REP64("v_add_f32 %0, %0, %1\n") : "+v"(a) : "v"(b));
  }
  if (a == 12345.f) out[0] = a;
}
// Half the waves of each block take the branch.
__global__ void __launch_bounds__(64) KnownHalfKernel(float* out) {
  float a = threadIdx.x, b = 1.0f;
  if (threadIdx.x >= 32) {
    for (int r = 0; r < 64; ++r) asm volatile(HALO_REP64("v_add_f32 %0, %0, %1\n") : "+v"(a) : "v"(b));
  }
  if (a == 12345.f) out[0] = a;
}
// Exactly `bytes` read contiguously, once (cold if the buffer was evicted).
__global__ void __launch_bounds__(256) KnownReadKernel(const float4* g, std::size_t n_f4, float* out) {
  float acc = 0.0f;
  for (std::size_t i = blockIdx.x * 256ull + threadIdx.x; i < n_f4; i += gridDim.x * 256ull) acc += g[i].x;
  if (acc == 12345.f) out[0] = acc;
}
// Scattered 32-byte pieces: one float4 pair per 4 KB.
__global__ void __launch_bounds__(256) KnownScatterKernel(const float4* g, std::size_t pieces, float* out) {
  float acc = 0.0f;
  for (std::size_t i = blockIdx.x * 256ull + threadIdx.x; i < pieces * 2; i += gridDim.x * 256ull)
    acc += g[(i / 2) * 256 + i % 2].x;
  if (acc == 12345.f) out[0] = acc;
}

// Reads a 1 MB range twice in one launch; the second pass of block b reads
// what block (b + 80) % 160 read in the first (another CU), so a hit there is
// an L2 hit. Tells "L2 invalidated between launches" from "hit counter broken".
__global__ void __launch_bounds__(256) KnownReuseKernel(const float4* g, float* out) {
  constexpr std::size_t kChunk = (1ull << 20) / 16 / 160;  // float4s per block
  float acc = 0.0f;
  for (int pass = 0; pass < 2; ++pass) {
    const std::size_t b = pass == 0 ? blockIdx.x : (blockIdx.x + 80) % 160;
    for (std::size_t i = threadIdx.x; i < kChunk; i += 256) acc += g[b * kChunk + i].x;
    __syncthreads();
    if (pass == 0) __threadfence();
  }
  if (acc == 12345.f) out[0] = acc;
}

// Exactly n_f4 float4s written contiguously; with `half` only the first 64 bytes of every 128-byte line.
__global__ void __launch_bounds__(256) KnownWriteKernel(float4* g, std::size_t n_f4, int half) {
  for (std::size_t i = blockIdx.x * 256ull + threadIdx.x; i < n_f4; i += gridDim.x * 256ull)
    if (!half || i % 8 < 4) g[i] = make_float4(1.0f, 2.0f, 3.0f, float(i));
}

// 32 MB written as one 4-byte or 2-byte value per lane (a wave stores 128 or 64 contiguous bytes), and
// 32 MB read, modified and written back in place (the residual update's pattern).
template <typename E>
__global__ void __launch_bounds__(256) KnownNarrowWriteKernel(E* g, std::size_t n) {
  for (std::size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += gridDim.x * 256ull) g[i] = E(i & 7);
}
__global__ void __launch_bounds__(256) KnownRmwKernel(float* g, std::size_t n) {
  for (std::size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += gridDim.x * 256ull) g[i] = g[i] * 0.5f + 1.0f;
}

void Counters() {
  // Launch layout: 80 blocks x 64 threads = 160 waves for the instruction kernels.
  float *g, *out;
  const std::size_t gbytes = 512ull << 20;
  CHECK(hipMalloc(&g, gbytes));
  CHECK(hipMemset(g, 0, gbytes));
  CHECK(hipMalloc(&out, 64));
  int step = 0;
  const auto sync = [&] {
    const hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      std::fprintf(stderr, "counters: launch %d failed: %s\n", step, hipGetErrorString(e));
      std::exit(1);
    }
    ++step;
  };
  hipLaunchKernelGGL((KnownKernel<0, 10>), 80, 64, 0, 0, g, out); sync();
  hipLaunchKernelGGL((KnownKernel<0, 20>), 80, 64, 0, 0, g, out); sync();
  hipLaunchKernelGGL((KnownKernel<1, 10>), 80, 64, 0, 0, g, out); sync();
  hipLaunchKernelGGL((KnownKernel<1, 20>), 80, 64, 0, 0, g, out); sync();
  hipLaunchKernelGGL((KnownKernel<2, 10>), 80, 64, 0, 0, g, out); sync();
  hipLaunchKernelGGL((KnownKernel<2, 20>), 80, 64, 0, 0, g, out); sync();
  hipLaunchKernelGGL((KnownKernel<3, 10>), 80, 64, 0, 0, g, out); sync();
  hipLaunchKernelGGL((KnownKernel<3, 20>), 80, 64, 0, 0, g, out); sync();
  hipLaunchKernelGGL(KnownMaskedKernel, 80, 64, 0, 0, out); sync();
  hipLaunchKernelGGL(KnownHalfKernel, 80, 64, 0, 0, out); sync();
  // Bytes: evict with a 256 MB read of another range, then 64 MB cold, then the same 4 MB twice (warm).
  const std::size_t evict = 256ull << 20, cold = 64ull << 20, warm = 4ull << 20;
  hipLaunchKernelGGL(KnownReadKernel, 1280, 256, 0, 0, reinterpret_cast<const float4*>(g) + (cold / 16), evict / 16, out); sync();
  hipLaunchKernelGGL(KnownReadKernel, 1280, 256, 0, 0, reinterpret_cast<const float4*>(g), cold / 16, out); sync();
  hipLaunchKernelGGL(KnownReadKernel, 160, 256, 0, 0, reinterpret_cast<const float4*>(g), warm / 16, out); sync();
  hipLaunchKernelGGL(KnownReadKernel, 160, 256, 0, 0, reinterpret_cast<const float4*>(g), warm / 16, out); sync();
  hipLaunchKernelGGL(KnownScatterKernel, 1280, 256, 0, 0, reinterpret_cast<const float4*>(g) + (cold / 16), 65536, out); sync();
  // 1 MB twice: the second read should hit in L2 if L2 holds 1 MB.
  hipLaunchKernelGGL(KnownReadKernel, 160, 256, 0, 0, reinterpret_cast<const float4*>(g), (1ull << 20) / 16, out); sync();
  hipLaunchKernelGGL(KnownReadKernel, 160, 256, 0, 0, reinterpret_cast<const float4*>(g), (1ull << 20) / 16, out); sync();
  hipLaunchKernelGGL(KnownReuseKernel, 160, 256, 0, 0, reinterpret_cast<const float4*>(g) + (8ull << 20) / 16, out); sync();
  // Writes: 64 MB, 4 MB, and 64 MB of lines of which only half (64 B of each 128) is written.
  hipLaunchKernelGGL(KnownWriteKernel, 1280, 256, 0, 0, reinterpret_cast<float4*>(g) + (cold / 16), cold / 16, 0); sync();
  hipLaunchKernelGGL(KnownWriteKernel, 160, 256, 0, 0, reinterpret_cast<float4*>(g) + (cold / 16), warm / 16, 0); sync();
  hipLaunchKernelGGL(KnownWriteKernel, 1280, 256, 0, 0, reinterpret_cast<float4*>(g) + (cold / 16), cold / 16, 1); sync();
  const std::size_t mid = 32ull << 20;
  hipLaunchKernelGGL(KnownNarrowWriteKernel<float>, 1280, 256, 0, 0, reinterpret_cast<float*>(g) + (cold / 4), mid / 4); sync();
  hipLaunchKernelGGL(KnownNarrowWriteKernel<_Float16>, 1280, 256, 0, 0, reinterpret_cast<_Float16*>(g) + (cold / 2), mid / 2); sync();
  hipLaunchKernelGGL(KnownRmwKernel, 1280, 256, 0, 0, reinterpret_cast<float*>(g) + (cold / 4), mid / 4); sync();
  std::printf("counters: known-answer kernels launched (run under rocprofv3; see tools/counter_check.py)\n");
  CHECK(hipFree(g));
  CHECK(hipFree(out));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "clock") Clock();
  else if (cmd == "dram") Dram(argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 480);
  else if (cmd == "wmma") Wmma();
  else if (cmd == "lds") Lds();
  else if (cmd == "latency") Latency();
  else if (cmd == "sync") Sync();
  else if (cmd == "counters") Counters();
  else if (cmd == "stream") Stream();
  else if (cmd == "empty") EmptyTiles();
  else if (cmd == "sweep") Sweep(argc, argv);
  else if (cmd == "rounds") Rounds();
  else if (cmd == "keepalive") {  // keepalive [period_ms] [seconds]: see tools/pmc.sh
    const int period = argc > 2 ? std::atoi(argv[2]) : 10;
    const double secs = argc > 3 ? std::atof(argv[3]) : 3600.0;
    hipStream_t st;
    hipEvent_t ev;
    CHECK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    CHECK(hipEventCreate(&ev));
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
      CHECK(hipEventRecord(ev, st));  // a marker: no waves, so no counter is disturbed
      CHECK(hipStreamSynchronize(st));
      std::this_thread::sleep_for(std::chrono::milliseconds(period));
    }
  }
  else if (cmd == "counters-repeat") {  // the same known kernel N times: do counters hold up over many dispatches?
    const int n = argc > 2 ? std::atoi(argv[2]) : 2000;
    float *g, *out;
    CHECK(hipMalloc(&g, 1 << 20));
    CHECK(hipMalloc(&out, 64));
    // Optional third argument: a 1 MB host-to-device copy (and a sync) halfway, as the engine does.
    const bool copy = argc > 3 && std::string(argv[3]) == "copy";
    // "threads FILE": 16 host threads doing 4 KB O_DIRECT preads of FILE from halfway on, as the
    // engine's n-gram table reader does.
    const bool io = argc > 4 && std::string(argv[3]) == "threads";
    std::atomic<bool> stop{false};
    std::vector<std::thread> pool;
    std::vector<char> host(1 << 20, 1);
    for (int i = 0; i < n; ++i) {
      if (io && i == n / 2)
        for (int t = 0; t < 16; ++t)
          pool.emplace_back([&, t] {
            const int fd = ::open(argv[4], O_RDONLY | O_DIRECT);
            void* buf = nullptr;
            if (fd < 0 || posix_memalign(&buf, 4096, 4096) != 0) return;
            std::uint64_t x = 1234567 + t;
            while (!stop.load()) {
              x = x * 6364136223846793005ull + 1;
              (void)!::pread(fd, buf, 4096, static_cast<off_t>((x >> 20) % (1ull << 18)) * 4096);
            }
            std::free(buf);
            ::close(fd);
          });
      if (copy && i == n / 2) {
        CHECK(hipMemcpyAsync(g, host.data(), host.size(), hipMemcpyHostToDevice, 0));
        CHECK(hipDeviceSynchronize());
      }
      hipLaunchKernelGGL((KnownKernel<0, 10>), 80, 64, 0, 0, g, out);
    }
    CHECK(hipDeviceSynchronize());
    stop = true;
    for (auto& t : pool) t.join();
    std::printf("counters-repeat: %d launches of KnownKernel<0, 10> (102400 + overhead VALU each)\n", n);
  }
  else {
    std::fprintf(stderr, "usage: halo_hwprobe clock|dram|wmma|lds|latency|sync|counters\n");
    return 2;
  }
  return 0;
}
