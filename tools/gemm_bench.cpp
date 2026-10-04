// Prefill GEMM throughput at the real model's dense shapes: the int8 WMMA
// kernel over repacked Q8_0 weights (with per-token activation
// quantization), for several prompt-chunk sizes. Reports TOPS (2 * rows *
// cols * tokens per second) and tokens per second for that one matrix.
//   halo_gemm_bench [reps]
//   halo_gemm_bench peak     register-only WMMA throughput (the ceiling)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/core_hsaco.hpp"
#include "rt/device.hpp"

using namespace halo;

int main(int argc, char** argv) {
  std::string e;
  if (argc > 1 && std::string(argv[1]) == "peak") {
    // Register-only WMMA throughput: the ceiling the GEMMs are measured against.
    auto d = rt::Device::Create(rt::Backend::kHip, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) return std::printf("%s\n", e.c_str()), 1;
    auto o = d->Alloc(64, &e);
    const auto k = d->Kernel("halo_wmma_peak", &e);
    for (std::uint32_t f16 : {0u, 1u})
      for (std::uint32_t blocks : {80u, 160u, 320u, 640u}) {
        const std::uint32_t iters = 20000;
        const rt::Arg a[] = {rt::Arg::Buf(o), rt::Arg::Val(iters), rt::Arg::Val(f16)};
        (void)d->Launch(k, {blocks}, {256}, a, &e);
        (void)d->Synchronize(&e);
        const auto t0 = std::chrono::steady_clock::now();
        (void)d->Launch(k, {blocks}, {256}, a, &e);
        (void)d->Synchronize(&e);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%s blocks %4u: %.1f TOPS\n", f16 ? "fp16" : "int8", blocks,
                    double(blocks) * 8 * iters * 4 * 16 * 16 * 16 * 2 / s / 1e12);
      }
    return 0;
  }
  const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
  for (auto be : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto d = rt::Device::Create(be, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) continue;
    struct Shape { const char* name; std::uint32_t rows, cols; };
    for (const Shape s : {Shape{"ssm qkv 10240x2560", 10240, 2560}, Shape{"ssm out  2560x6144", 2560, 6144},
                          Shape{"hc down   320x10240", 320, 10240}}) {
      std::vector<std::uint8_t> raw(std::size_t{s.rows} * (s.cols / 32) * 34);
      std::mt19937 g(1);
      for (auto& b : raw) b = static_cast<std::uint8_t>(g());
      for (std::size_t b = 0; b + 34 <= raw.size(); b += 34) {
        const std::uint16_t h = 0x2000 | (g() & 0x3FF);
        std::memcpy(raw.data() + b, &h, 2);
      }
      auto rb = d->Alloc(raw.size(), &e), qs = d->Alloc(std::size_t{s.rows} * s.cols, &e),
           dd = d->Alloc(std::size_t{s.rows} * s.cols / 16, &e);
      (void)d->Upload(rb, 0, raw.data(), raw.size(), &e);
      const rt::Arg ra[] = {rt::Arg::Buf(rb), rt::Arg::Buf(qs), rt::Arg::Buf(dd), rt::Arg::Val(s.rows),
                            rt::Arg::Val(s.cols), rt::Arg::Val(s.cols)};
      (void)d->Launch(d->Kernel("halo_repack_q8_0", &e), {4096}, {256}, ra, &e);
      for (const std::uint32_t T : {16u, 64u, 256u, 512u, 2048u}) {
        std::vector<float> x(std::size_t{T} * s.cols);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.01f * static_cast<float>(i));
        auto xb = d->Alloc(x.size() * 4, &e), xq = d->Alloc(x.size(), &e), xd = d->Alloc(x.size() / 8, &e),
             yb = d->Alloc(std::size_t{T} * s.rows * 4, &e);
        (void)d->Upload(xb, 0, x.data(), x.size() * 4, &e);
        const auto kq = d->Kernel("halo_quantize_q8_rows", &e), kg = d->Kernel("halo_gemm_q8r_wmma_t", &e);
        const rt::Arg qa[] = {rt::Arg::Buf(xb), rt::Arg::Buf(xq), rt::Arg::Buf(xd), rt::Arg::Val(s.cols)};
        const rt::Arg ga[] = {rt::Arg::Buf(qs), rt::Arg::Buf(dd), rt::Arg::Buf(xq), rt::Arg::Buf(xd), rt::Arg::Buf(yb),
                              rt::Arg::Val(s.rows), rt::Arg::Val(s.cols), rt::Arg::Val(s.cols), rt::Arg::Val(T)};
        const auto run = [&] {
          (void)d->Launch(kq, {T}, {256}, qa, &e);
          (void)d->Launch(kg, {(T + 127) / 128, (s.rows + 127) / 128}, {256}, ga, &e);
        };
        for (int i = 0; i < 3; ++i) run();
        (void)d->Synchronize(&e);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) run();
        (void)d->Synchronize(&e);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("%s %-20s T %5u  %9.1f us  %6.2f TOPS  %9.0f tok/s\n", rt::BackendName(be), s.name, T, sec * 1e6,
                    2.0 * s.rows * s.cols * T / sec / 1e12, T / sec);
        for (auto* b : {&xb, &xq, &xd, &yb}) d->Free(*b);
      }
      for (auto* b : {&rb, &qs, &dd}) d->Free(*b);
    }
  }
}
