// halo_hwprobe: measured facts about the GPU (gfx1151 Strix Halo) that the
// performance work relies on. Standalone HIP; each subcommand prints one table.
//
//   halo_hwprobe clock           shader clock (in-kernel counters) vs load
//   halo_hwprobe dram            bandwidth vs segment size, dispatch order, read/write
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

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "clock") Clock();
  else if (cmd == "dram") Dram();
  else {
    std::fprintf(stderr, "usage: halo_hwprobe clock|dram\n");
    return 2;
  }
  return 0;
}
