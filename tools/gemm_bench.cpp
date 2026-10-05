// Prefill GEMM throughput at the real model's dense shapes: the int8 WMMA
// kernel over repacked Q8_0 weights (with per-token activation
// quantization), for several prompt-chunk sizes. Reports TOPS (2 * rows *
// cols * tokens per second) and tokens per second for that one matrix.
//   halo_gemm_bench [reps]
//   halo_gemm_bench peak      register-only WMMA throughput (the ceiling)
//   halo_gemm_bench deltanet  one DeltaNet layer's chunk: decode steps vs chunked kernels
//   halo_gemm_bench bf16      the Qwen3.5 BF16 GEMMs (fp16 WMMA), weights from DRAM
//   halo_gemm_bench memory    read bandwidth and dependent-load latency against
//                             working-set size (1 MB .. 512 MB): last-level cache
//                             size and bandwidth, DRAM bandwidth and latency
//   halo_gemm_bench moe       one layer's routed experts at the real shapes (512
//                             experts, top 10, Q4_K gate/up 640 x 2560, Q5_1 down
//                             2560 x 640), random routing, T = 2048: bucket,
//                             gate/up, down, combine (for profilers: small memory)
//   halo_gemm_bench model     every Flash-Next dense int8 GEMM shape at T = 2048,
//                             weighted by its count per 2048-token prefill
//   halo_gemm_bench study [rows cols tokens]
//                             FP16-route GEMM variants (Q8_0 weights) on one shape,
//                             default the mixer's down projection 320 x 10240 at
//                             T = 2048: 20 warm-up then 50 timed launches each

#include <algorithm>
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
  if (argc > 1 && std::string(argv[1]) == "gdnerr") {
    // Chunked DeltaNet (prep + scan) against the fp32 split kernel on the same
    // inputs at a layer's shape: L2-normalized q and k per head, gates drawn as
    // in the property test. Relative L2 error of the outputs and final state.
    auto d = rt::Device::Create(rt::Backend::kHip, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) return std::printf("%s\n", e.c_str()), 1;
    const std::uint32_t T = std::max(argc > 2 ? std::atoi(argv[2]) : 2048, 256), heads = 48, kheads = 16, stride = 10240,
                        nch = (T + 15) / 16;
    std::uint32_t seed = 12345;
    const auto rnd = [&] {
      seed = seed * 1664525u + 1013904223u;
      return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f;
    };
    std::vector<float> conv(std::size_t{T} * stride), al(std::size_t{T} * heads), be(al.size()), av(heads), dtv(heads),
        st(std::size_t{heads} * 128 * 128);
    // corr > 0: each q/k head's vectors share a fixed direction (weight corr
    // against unit noise), as real keys within a chunk do.
    const float corr = argc > 4 ? static_cast<float>(std::atof(argv[4])) : 0.0f;
    for (auto& x : conv) x = 4.0f * rnd();
    std::vector<float> base(2 * kheads * 128);
    for (auto& x : base) x = 4.0f * rnd();
    for (std::uint32_t t = 0; t < T; ++t)
      for (std::uint32_t v = 0; v < 2 * kheads; ++v) {  // q and k heads: unit L2
        float* p = &conv[std::size_t{t} * stride + v * 128];
        for (int i = 0; i < 128; ++i) p[i] += corr * base[v * 128 + i];
        double n = 0;
        for (int i = 0; i < 128; ++i) n += double(p[i]) * p[i];
        for (int i = 0; i < 128; ++i) p[i] = static_cast<float>(p[i] / std::sqrt(n));
      }
    for (auto& x : al) x = 4.0f * rnd();
    for (auto& x : be) x = 4.0f * rnd();
    for (auto& x : av) x = -std::exp(2.0f * rnd());
    for (auto& x : dtv) x = 2.0f * rnd();
    for (auto& x : st) x = 0.2f * rnd();
    const auto up = [&](const std::vector<float>& h) {
      auto b = d->Alloc(h.size() * 4, &e);
      (void)d->Upload(b, 0, h.data(), h.size() * 4, &e);
      return b;
    };
    auto cb = up(conv), alb = up(al), beb = up(be), ab = up(av), dtb = up(dtv), s1 = up(st), s2 = up(st);
    auto o1 = d->Alloc(std::size_t{T} * heads * 128 * 4, &e), o2 = d->Alloc(std::size_t{T} * heads * 128 * 4, &e),
         prep = d->Alloc(std::size_t{heads} * nch * 576 * 4, &e);
    const rt::Arg sa[] = {rt::Arg::Buf(cb), rt::Arg::Buf(cb, 2048ull * 4), rt::Arg::Buf(cb, 4096ull * 4),
                          rt::Arg::Buf(alb), rt::Arg::Buf(beb), rt::Arg::Buf(ab), rt::Arg::Buf(dtb), rt::Arg::Buf(s1),
                          rt::Arg::Buf(o1), rt::Arg::Val(kheads), rt::Arg::Val(T), rt::Arg::Val(stride),
                          rt::Arg::Val(stride), rt::Arg::Val(heads * 128u)};
    const rt::Arg pa[] = {rt::Arg::Buf(cb), rt::Arg::Buf(cb, 2048ull * 4), rt::Arg::Buf(alb), rt::Arg::Buf(beb),
                          rt::Arg::Buf(ab), rt::Arg::Buf(dtb), rt::Arg::Buf(prep), rt::Arg::Val(heads), rt::Arg::Val(T),
                          rt::Arg::Val(stride)};
    const char* scan = argc > 3 ? argv[3] : "halo_gdn_c16_scan";
    const rt::Arg ca[] = {rt::Arg::Buf(cb), rt::Arg::Buf(cb, 2048ull * 4), rt::Arg::Buf(cb, 4096ull * 4),
                          rt::Arg::Buf(prep), rt::Arg::Buf(s2), rt::Arg::Buf(o2), rt::Arg::Val(kheads), rt::Arg::Val(T),
                          rt::Arg::Val(stride), rt::Arg::Val(stride), rt::Arg::Val(heads * 128u)};
    (void)d->Launch(d->Kernel("halo_gdn_prefill_split", &e), {2, heads}, {256}, sa, &e);
    (void)d->Launch(d->Kernel("halo_gdn_c16_prep", &e), {nch, kheads}, {256}, pa, &e);
    (void)d->Launch(d->Kernel(scan, &e), {2, heads}, {128}, ca, &e);
    (void)d->Synchronize(&e);
    const auto get = [&](rt::Buffer b, std::size_t n) {
      std::vector<float> h(n);
      (void)d->Download(b, 0, h.data(), n * 4, &e);
      return h;
    };
    const auto rel = [](const std::vector<float>& x, const std::vector<float>& y) {
      double dd = 0, m = 0;
      for (std::size_t i = 0; i < x.size(); ++i) dd += double(x[i] - y[i]) * (x[i] - y[i]), m += double(y[i]) * y[i];
      return std::sqrt(dd / m);
    };
    const auto x1 = get(o1, std::size_t{T} * heads * 128), x2 = get(o2, x1.size());
    std::printf("%s T %u: output rel L2 %.3e (first 256 tokens %.3e, last 256 %.3e), state rel L2 %.3e\n", scan, T,
                rel(x2, x1),
                rel(std::vector<float>(x2.begin(), x2.begin() + 256 * heads * 128),
                    std::vector<float>(x1.begin(), x1.begin() + 256 * heads * 128)),
                rel(std::vector<float>(x2.end() - 256 * heads * 128, x2.end()),
                    std::vector<float>(x1.end() - 256 * heads * 128, x1.end())),
                rel(get(s2, st.size()), get(s1, st.size())));
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "gdnskel") {
    // Structural floor of chunked DeltaNet scans (no math) at one layer's real
    // shape: 2048 tokens, 48 v-heads, 16 k-heads, conv rows of 10240.
    auto d = rt::Device::Create(rt::Backend::kHip, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) return std::printf("%s\n", e.c_str()), 1;
    const std::uint32_t T = 2048, heads = 48, kheads = 16, stride = 10240;
    auto conv = d->Alloc(std::size_t{T} * stride * 4, &e), al = d->Alloc(std::size_t{T} * heads * 4, &e),
         st = d->Alloc(std::size_t{heads} * 128 * 128 * 4, &e), out = d->Alloc(64, &e);
    {
      std::vector<float> f(std::size_t{T} * stride, 0.01f);
      (void)d->Upload(conv, 0, f.data(), f.size() * 4, &e);
    }
    struct V {
      const char* k;
      std::uint32_t nc, nt;
    };
    const V vs[] = {{"halo_gdnskel_c16_n32", 32, 128},        {"halo_gdnskel_c16_n32_pf", 32, 128},
                    {"halo_gdnskel_c16_n64_pf", 64, 256},     {"halo_gdnskel_c32_n32", 32, 128},
                    {"halo_gdnskel_c32_n32_pf", 32, 128},     {"halo_gdnskel_c32_n64_pf_seq", 64, 256},
                    {"halo_gdnskel_c16_n64_pf_nost", 64, 256}, {"halo_gdnskel_c32_n64_pf_seq_nost", 64, 256},
                    {"halo_gdnskel_c64_n32_seq", 32, 256},    {"halo_gdnskel_c64_n32_pf_seq", 32, 256}};
    std::printf("structural floor per layer (2048 tokens, no math); the split kernel takes 8.2 ms in-engine\n");
    {  // the real kernels at the same shape: split, and chunked prep + scan
      auto ab = d->Alloc(heads * 4, &e), be = d->Alloc(std::size_t{T} * heads * 4, &e),
           prep = d->Alloc(std::size_t{heads} * (T / 16) * 576 * 4, &e), ob = d->Alloc(std::size_t{T} * heads * 128 * 4, &e);
      {
        std::vector<float> z(std::size_t{T} * heads, 0.0f), av(heads, -0.5f);
        (void)d->Upload(al, 0, z.data(), z.size() * 4, &e);
        (void)d->Upload(be, 0, z.data(), z.size() * 4, &e);
        (void)d->Upload(ab, 0, av.data(), av.size() * 4, &e);
      }
      const auto time = [&](const char* name, auto&& go) {
        for (int r = 0; r < 3; ++r) go();
        (void)d->Synchronize(&e);
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 10; ++r) go();
        (void)d->Synchronize(&e);
        std::printf("%-28s             %7.3f ms\n", name,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10);
      };
      const auto ks = d->Kernel("halo_gdn_prefill_split", &e), kp = d->Kernel("halo_gdn_c16_prep", &e),
                 kc = d->Kernel("halo_gdn_c16_scan", &e);
      const rt::Arg sa[] = {rt::Arg::Buf(conv), rt::Arg::Buf(conv, 2048ull * 4), rt::Arg::Buf(conv, 4096ull * 4),
                            rt::Arg::Buf(al), rt::Arg::Buf(be), rt::Arg::Buf(ab), rt::Arg::Buf(al), rt::Arg::Buf(st),
                            rt::Arg::Buf(ob), rt::Arg::Val(kheads), rt::Arg::Val(T), rt::Arg::Val(stride),
                            rt::Arg::Val(stride), rt::Arg::Val(heads * 128u)};
      const rt::Arg pa[] = {rt::Arg::Buf(conv), rt::Arg::Buf(conv, 2048ull * 4), rt::Arg::Buf(al), rt::Arg::Buf(be),
                            rt::Arg::Buf(ab), rt::Arg::Buf(al), rt::Arg::Buf(prep), rt::Arg::Val(heads), rt::Arg::Val(T),
                            rt::Arg::Val(stride)};
      const rt::Arg ca[] = {rt::Arg::Buf(conv), rt::Arg::Buf(conv, 2048ull * 4), rt::Arg::Buf(conv, 4096ull * 4),
                            rt::Arg::Buf(prep), rt::Arg::Buf(st), rt::Arg::Buf(ob), rt::Arg::Val(kheads), rt::Arg::Val(T),
                            rt::Arg::Val(stride), rt::Arg::Val(stride), rt::Arg::Val(heads * 128u)};
      time("halo_gdn_prefill_split", [&] { (void)d->Launch(ks, {2, heads}, {256}, sa, &e); });
      time("halo_gdn_c16_prep", [&] { (void)d->Launch(kp, {T / 16, kheads}, {256}, pa, &e); });
      const auto kp2 = d->Kernel("halo_gdn_c16_prep2", &e);
      for (const std::uint32_t cpb : {8u, 16u, 24u, 32u, 64u}) {
        const rt::Arg pa2[] = {rt::Arg::Buf(conv), rt::Arg::Buf(conv, 2048ull * 4), rt::Arg::Buf(al), rt::Arg::Buf(be),
                               rt::Arg::Buf(ab), rt::Arg::Buf(al), rt::Arg::Buf(prep), rt::Arg::Val(heads),
                               rt::Arg::Val(T), rt::Arg::Val(stride), rt::Arg::Val(cpb)};
        char name[64];
        std::snprintf(name, sizeof name, "halo_gdn_c16_prep2 cpb %u", cpb);
        time(name, [&] { (void)d->Launch(kp2, {(T / 16 + cpb - 1) / cpb, kheads}, {256}, pa2, &e); });
      }
      time("halo_gdn_c16_scan", [&] { (void)d->Launch(kc, {2, heads}, {128}, ca, &e); });
      const auto k3 = d->Kernel("halo_gdn_c16_scan_ks3", &e);
      time("halo_gdn_c16_scan_ks3", [&] { (void)d->Launch(k3, {2, heads}, {128}, ca, &e); });
      const auto k4 = d->Kernel("halo_gdn_c16_scan3", &e);
      time("halo_gdn_c16_scan3", [&] { (void)d->Launch(k4, {2, heads}, {128}, ca, &e); });
      const auto k5 = d->Kernel("halo_gdn_c16_scan3u", &e);
      time("halo_gdn_c16_scan3u", [&] { (void)d->Launch(k5, {2, heads}, {128}, ca, &e); });
      time("halo_gdn_c16 prep + scan", [&] {
        (void)d->Launch(kp, {T / 16, kheads}, {256}, pa, &e);
        (void)d->Launch(kc, {2, heads}, {128}, ca, &e);
      });
    }
    for (const auto& v : vs) {
      const auto k = d->Kernel(v.k, &e);
      const rt::Arg a[] = {rt::Arg::Buf(conv), rt::Arg::Buf(conv, 2048ull * 4), rt::Arg::Buf(conv, 4096ull * 4),
                           rt::Arg::Buf(al), rt::Arg::Buf(st), rt::Arg::Buf(out), rt::Arg::Val(kheads), rt::Arg::Val(T),
                           rt::Arg::Val(stride), rt::Arg::Val(stride)};
      const rt::Dim3 grid{128 / v.nc, heads};
      for (int r = 0; r < 3; ++r) (void)d->Launch(k, grid, {v.nt}, a, &e);
      (void)d->Synchronize(&e);
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < 10; ++r) (void)d->Launch(k, grid, {v.nt}, a, &e);
      (void)d->Synchronize(&e);
      std::printf("%-28s grid %ux%u  %7.3f ms\n", v.k, grid.x, grid.y,
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10);
    }
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "study") {
    auto d = rt::Device::Create(rt::Backend::kHip, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) return std::printf("%s\n", e.c_str()), 1;
    const std::uint32_t rows = argc > 2 ? std::atoi(argv[2]) : 320, cols = argc > 3 ? std::atoi(argv[3]) : 10240,
                        T = argc > 4 ? std::atoi(argv[4]) : 2048;
    const std::size_t row_bytes = cols / 32 * 34;
    std::vector<std::uint8_t> w(std::size_t{rows} * row_bytes);
    std::mt19937 g(5);
    for (auto& b : w) b = static_cast<std::uint8_t>(g());
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t b = 0; b < cols / 32; ++b) {
        const std::uint16_t hv = 0x2000 | (g() & 0x3FF);
        std::memcpy(w.data() + r * row_bytes + b * 34, &hv, 2);
      }
    std::vector<std::uint16_t> xh(std::size_t{T} * cols);
    for (auto& v : xh) v = static_cast<std::uint16_t>(0x3000 | (g() & 0x3FF));
    auto wb = d->Alloc(w.size(), &e), qs = d->Alloc(std::size_t{rows} * cols, &e),
         db = d->Alloc(std::size_t{rows} * (cols / 32) * 2, &e), xb = d->Alloc(xh.size() * 2, &e),
         yb = d->Alloc(std::size_t{32} * T * rows * 4, &e);  // room for split-K partials
    (void)d->Upload(wb, 0, w.data(), w.size(), &e);
    (void)d->Upload(xb, 0, xh.data(), xh.size() * 2, &e);
    const rt::Arg ra[] = {rt::Arg::Buf(wb), rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Val(rows), rt::Arg::Val(cols),
                          rt::Arg::Val(cols)};
    (void)d->Launch(d->Kernel("halo_repack_q8_0", &e), {4096}, {256}, ra, &e);
    struct V {
      const char* kernel;
      std::uint32_t bm, bn, splits;
    };
    const V variants[] = {
        {"halo_gemm_q8r_f16_m", 320, 64, 1},
        {"halo_gemm_q8r_f16_l", 256, 128, 1},
        {"halo_gemm_q8r_f16_lg8", 256, 128, 1},
        {"halo_gemm_q8r_f16_s", 64, 128, 1},
    };
    std::printf("shape %u x %u, T %u (%.1f GFLOP)\n", rows, cols, T, 2.0 * rows * cols * T / 1e9);
    const auto kf = d->Kernel("halo_bench_read", &e);
    auto fb = d->Alloc(96ull << 20, &e), fo = d->Alloc(64, &e);
    {
      std::vector<float> fill((96ull << 20) / 4, 1.0f);
      (void)d->Upload(fb, 0, fill.data(), fill.size() * 4, &e);
    }
    const rt::Arg fa[] = {rt::Arg::Buf(fb), rt::Arg::Val(std::uint64_t{(96ull << 20) / 16}), rt::Arg::Val(std::uint32_t{1}),
                          rt::Arg::Buf(fo)};
    for (const auto& v : variants) {
      const auto k = d->Kernel(v.kernel, &e);
      if (!k) {
        std::printf("%-28s missing\n", v.kernel);
        continue;
      }
      const rt::Arg a[] = {rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Buf(xb), rt::Arg::Buf(yb), rt::Arg::Val(rows),
                           rt::Arg::Val(cols), rt::Arg::Val(cols), rt::Arg::Val(T)};
      const rt::Dim3 grid{(T + v.bn - 1) / v.bn, (rows + v.bm - 1) / v.bm, v.splits};
      for (int r = 0; r < 20; ++r) (void)d->Launch(k, grid, {256}, a, &e);
      (void)d->Synchronize(&e);
      const int reps = 50;
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < reps; ++r) (void)d->Launch(k, grid, {256}, a, &e);
      (void)d->Synchronize(&e);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
      // Cold: a 96 MB read between launches evicts the last-level cache (as
      // the engine's preceding kernels do); each launch timed on its own.
      double cold = 0;
      for (int r = 0; r < 20; ++r) {
        (void)d->Launch(kf, {1280}, {256}, fa, &e);
        (void)d->Synchronize(&e);
        const auto c0 = std::chrono::steady_clock::now();
        (void)d->Launch(k, grid, {256}, a, &e);
        (void)d->Synchronize(&e);
        cold += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
      }
      cold /= 20;
      // Dirty: 96 MB written just before (as combine_norm writes res and xh):
      // the write-back of dirty lines overlaps the kernel.
      double dirty = 0;
      for (int r = 0; r < 20; ++r) {
        (void)d->Fill(fb, static_cast<std::uint8_t>(r), &e);
        (void)d->Synchronize(&e);
        const auto c0 = std::chrono::steady_clock::now();
        (void)d->Launch(k, grid, {256}, a, &e);
        (void)d->Synchronize(&e);
        dirty += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
      }
      dirty /= 20;
      std::printf("%-28s grid %4ux%ux%u  warm %7.3f ms (%5.1f TFLOP/s)  cold %7.3f ms  after-writes %7.3f ms\n",
                  v.kernel, grid.x, grid.y, grid.z, ms, 2.0 * rows * cols * T / ms / 1e9, cold, dirty);
    }
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "mixstudy") {
    // The mixer's fused up-projection + mix (rebuild form, as in prefill:
    // xn rebuilt from res, rscale and the norm gammas) at its real shape,
    // timed warm, cold (LLC evicted by a 96 MB read), after 96 MB of
    // unrelated writes, and after rewriting its own input res (84 MB), as
    // combine_norm does just before it in the engine.
    auto d = rt::Device::Create(rt::Backend::kHip, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) return std::printf("%s\n", e.c_str()), 1;
    const std::uint32_t h = 2560, rows = 4 * h, cols = 320, T = 2048;
    const std::size_t row_bytes = cols / 32 * 34;
    std::vector<std::uint8_t> w(std::size_t{rows} * row_bytes);
    std::mt19937 g(5);
    for (auto& b : w) b = static_cast<std::uint8_t>(g());
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t b = 0; b < cols / 32; ++b) {
        const std::uint16_t hv = 0x2000 | (g() & 0x3FF);
        std::memcpy(w.data() + r * row_bytes + b * 34, &hv, 2);
      }
    std::vector<std::uint16_t> lo(std::size_t{T} * cols);
    for (auto& v : lo) v = static_cast<std::uint16_t>(0x3000 | (g() & 0x3FF));
    std::vector<float> res(std::size_t{T} * rows, 0.5f), rs(std::size_t{T} * 4, 1.0f), nw(rows, 1.0f);
    auto wb = d->Alloc(w.size(), &e), qs = d->Alloc(std::size_t{rows} * cols, &e),
         db = d->Alloc(std::size_t{rows} * (cols / 32) * 2, &e), xb = d->Alloc(lo.size() * 2, &e),
         rb = d->Alloc(res.size() * 4, &e), rsb = d->Alloc(rs.size() * 4, &e), nwb = d->Alloc(nw.size() * 4, &e),
         yb = d->Alloc(std::size_t{T} * h * 4, &e), y16 = d->Alloc(std::size_t{T} * h * 2, &e);
    (void)d->Upload(wb, 0, w.data(), w.size(), &e);
    (void)d->Upload(xb, 0, lo.data(), lo.size() * 2, &e);
    (void)d->Upload(rb, 0, res.data(), res.size() * 4, &e);
    (void)d->Upload(rsb, 0, rs.data(), rs.size() * 4, &e);
    (void)d->Upload(nwb, 0, nw.data(), nw.size() * 4, &e);
    const rt::Arg ra[] = {rt::Arg::Buf(wb), rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Val(rows), rt::Arg::Val(cols),
                          rt::Arg::Val(cols)};
    (void)d->Launch(d->Kernel("halo_repack_q8_0", &e), {4096}, {256}, ra, &e);
    const auto k = d->Kernel(argc > 2 ? argv[2] : "halo_gemm_q8r_f16_mix", &e);
    const rt::Arg a[] = {rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Buf(xb), rt::Arg::Buf(yb), rt::Arg::Val(rows),
                         rt::Arg::Val(cols), rt::Arg::Val(cols), rt::Arg::Val(T), rt::Arg::Buf(rb), rt::Arg::Buf(y16),
                         rt::Arg::Buf(rsb), rt::Arg::Buf(nwb)};
    const bool rows_fast = argc > 2 && std::string(argv[2]) == "halo_gemm_q8r_f16_mixt";
    const rt::Dim3 grid = rows_fast ? rt::Dim3{(rows + 255) / 256, (T + 127) / 128} : rt::Dim3{(T + 127) / 128, (rows + 255) / 256};
    const auto kf = d->Kernel("halo_bench_read", &e);
    auto fb = d->Alloc(96ull << 20, &e), fo = d->Alloc(64, &e);
    {
      std::vector<float> fill((96ull << 20) / 4, 1.0f);
      (void)d->Upload(fb, 0, fill.data(), fill.size() * 4, &e);
    }
    const rt::Arg fa[] = {rt::Arg::Buf(fb), rt::Arg::Val(std::uint64_t{(96ull << 20) / 16}), rt::Arg::Val(std::uint32_t{1}),
                          rt::Arg::Buf(fo)};
    const auto timed = [&](auto&& before) {
      double sum = 0;
      for (int r = 0; r < 20; ++r) {
        before(r);
        (void)d->Synchronize(&e);
        const auto c0 = std::chrono::steady_clock::now();
        (void)d->Launch(k, grid, {256}, a, &e);
        (void)d->Synchronize(&e);
        sum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
      }
      return sum / 20;
    };
    for (int r = 0; r < 5; ++r) (void)d->Launch(k, grid, {256}, a, &e);
    const double warm = timed([](int) {});
    const double cold = timed([&](int) { (void)d->Launch(kf, {1280}, {256}, fa, &e); });
    const double other = timed([&](int r) { (void)d->Fill(fb, static_cast<std::uint8_t>(r), &e); });
    // Own input rewritten: res set to the same bit pattern again (0x3f000000 = 0.5f) through a fill kernel.
    const double own = timed([&](int) { (void)d->Fill(rb, 0x00, &e); });
    std::printf("mix 10240 x 320, T 2048: warm %.3f ms  cold %.3f  after 96 MB other writes %.3f  "
                "after rewriting its own res (84 MB) %.3f\n", warm, cold, other, own);
    return 0;
  }
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
  if (argc > 1 && std::string(argv[1]) == "memory") {
    auto d = rt::Device::Create(rt::Backend::kHip, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) return std::printf("%s\n", e.c_str()), 1;
    const auto kr = d->Kernel("halo_bench_read", &e), kc = d->Kernel("halo_bench_chase", &e);
    auto out = d->Alloc(64, &e);
    std::printf("%10s %12s %14s\n", "set", "read GB/s", "chase ns/load");
    for (std::size_t mb : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512}) {
      const std::size_t bytes = mb << 20;
      auto b = d->Alloc(bytes, &e);
      if (!b) break;
      // Write every page first: a fresh allocation may not be backed yet, and
      // reads of unbacked pages can all hit one zero page in cache.
      {
        std::vector<float> fill(bytes / 4, 1.0f);
        (void)d->Upload(b, 0, fill.data(), bytes, &e);
      }
      // Bandwidth: enough passes for ~1 GB of traffic, one pass per launch. A
      // repeat loop inside the kernel would let each resident block re-read
      // its own slice from cache (only ~160 of the 1280 blocks are resident).
      const std::uint64_t n = bytes / 16;
      const std::uint32_t reps = static_cast<std::uint32_t>(std::max<std::size_t>(1, (1ull << 30) / bytes));
      const rt::Arg ra[] = {rt::Arg::Buf(b), rt::Arg::Val(n), rt::Arg::Val(std::uint32_t{1}), rt::Arg::Buf(out)};
      (void)d->Launch(kr, {1280}, {256}, ra, &e);
      (void)d->Synchronize(&e);
      auto t0 = std::chrono::steady_clock::now();
      for (std::uint32_t r = 0; r < reps; ++r) (void)d->Launch(kr, {1280}, {256}, ra, &e);
      (void)d->Synchronize(&e);
      const double gbs = static_cast<double>(bytes) * reps /
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9;
      // Latency: a random cycle over 128-byte lines.
      const std::size_t lines = bytes / 128, stride = 32;  // uint32 per line
      std::vector<std::uint32_t> perm(lines), next(bytes / 4, 0);
      for (std::size_t i = 0; i < lines; ++i) perm[i] = static_cast<std::uint32_t>(i);
      std::mt19937 g(3);
      std::shuffle(perm.begin() + 1, perm.end(), g);
      for (std::size_t i = 0; i < lines; ++i)
        next[perm[i] * stride] = static_cast<std::uint32_t>(perm[(i + 1) % lines] * stride);
      (void)d->Upload(b, 0, next.data(), next.size() * 4, &e);
      const std::uint32_t steps = 200000;
      const rt::Arg ca[] = {rt::Arg::Buf(b), rt::Arg::Val(steps), rt::Arg::Buf(out)};
      (void)d->Launch(kc, {1}, {1}, ca, &e);
      (void)d->Synchronize(&e);
      t0 = std::chrono::steady_clock::now();
      (void)d->Launch(kc, {1}, {1}, ca, &e);
      (void)d->Synchronize(&e);
      const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / steps;
      std::printf("%7zu MB %12.1f %14.1f\n", mb, gbs, ns);
      d->Free(b);
    }
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "moe") {
    auto d = rt::Device::Create(rt::Backend::kHip, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) return std::printf("%s\n", e.c_str()), 1;
    const std::uint32_t E = 512, k = 10, ff = 640, H = 2560, T = 2048, P = T * k;
    const std::size_t gu_row = H / 256 * 144, dn_row = ff / 32 * 24;
    std::mt19937 g(7);
    const auto fill = [&](std::vector<std::uint8_t>& w, std::size_t rows, std::size_t row_bytes, std::size_t block,
                          std::initializer_list<std::size_t> halves) {
      for (auto& b : w) b = static_cast<std::uint8_t>(g());
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t b = 0; b < row_bytes / block; ++b)
          for (std::size_t off : halves) {
            const std::uint16_t h = 0x2000 | (g() & 0x3FF);
            std::memcpy(w.data() + r * row_bytes + b * block + off, &h, 2);
          }
    };
    std::vector<std::uint8_t> gate(std::size_t{E} * ff * gu_row), up(gate.size()), down(std::size_t{E} * H * dn_row);
    fill(gate, std::size_t{E} * ff, gu_row, 144, {0, 2});
    fill(up, std::size_t{E} * ff, gu_row, 144, {0, 2});
    fill(down, std::size_t{E} * H, dn_row, 24, {0, 2});
    std::vector<float> logits(std::size_t{T} * E), x(std::size_t{T} * H);
    std::normal_distribution<float> nd;
    for (auto& v : logits) v = nd(g);
    for (auto& v : x) v = 0.1f * nd(g);
    const auto up_buf = [&](const void* p, std::size_t n) {
      auto b = d->Alloc(n, &e);
      if (p) (void)d->Upload(b, 0, p, n, &e);
      return b;
    };
    auto gb = up_buf(gate.data(), gate.size()), ub = up_buf(up.data(), up.size()), db = up_buf(down.data(), down.size()),
         lb = up_buf(logits.data(), logits.size() * 4), xb = up_buf(x.data(), x.size() * 4),
         x16 = up_buf(nullptr, x.size() * 2), ib = up_buf(nullptr, P * 4), wb = up_buf(nullptr, P * 4),
         cb = up_buf(nullptr, std::size_t{(P + 1023) / 1024} * E * 4), bb = up_buf(nullptr, E * 4),
         sb = up_buf(nullptr, P * 4), pb = up_buf(nullptr, P * 4), tb = up_buf(nullptr, std::size_t{P / 64 + E + 1} * 12),
         nb = up_buf(nullptr, 4), hb = up_buf(nullptr, std::size_t{P} * ff * 2), yb = up_buf(nullptr, std::size_t{P} * H * 2),
         ob = up_buf(nullptr, std::size_t{T} * H * 4);
    const std::uint32_t chunks = (P + 1023) / 1024, max_tiles = P / 64 + E + 1;
    const auto L = [&](const char* name, rt::Dim3 grid, rt::Dim3 block, std::initializer_list<rt::Arg> a) {
      const std::vector<rt::Arg> args(a);
      (void)d->Launch(d->Kernel(name, &e), grid, block, args, &e);
    };
    const auto layer = [&] {
      L("halo_router_topk_rows", {T}, {256}, {rt::Arg::Buf(lb), rt::Arg::Buf(ib), rt::Arg::Buf(wb), rt::Arg::Val(E), rt::Arg::Val(k)});
      L("halo_moe_bucket_count", {chunks}, {1024}, {rt::Arg::Buf(ib), rt::Arg::Val(P), rt::Arg::Val(E), rt::Arg::Buf(cb)});
      L("halo_moe_bucket_scan", {1}, {1024},
        {rt::Arg::Buf(cb), rt::Arg::Val(chunks), rt::Arg::Val(E), rt::Arg::Buf(tb), rt::Arg::Buf(nb), rt::Arg::Buf(bb)});
      L("halo_moe_bucket_scatter", {chunks}, {1024},
        {rt::Arg::Buf(ib), rt::Arg::Val(P), rt::Arg::Val(E), rt::Arg::Buf(cb), rt::Arg::Buf(bb), rt::Arg::Buf(sb), rt::Arg::Buf(pb)});
      L("halo_f16_perm_rows", {(T * H + 255) / 256}, {256}, {rt::Arg::Buf(xb), rt::Arg::Buf(x16), rt::Arg::Val(T * H)});
      L("halo_moe_gate_up_grouped_q4_k", {(ff + 127) / 128, max_tiles}, {256},
        {rt::Arg::Buf(gb), rt::Arg::Buf(ub), rt::Arg::Val<std::uint64_t>(ff * gu_row), rt::Arg::Val<std::uint64_t>(gu_row),
         rt::Arg::Buf(x16), rt::Arg::Buf(sb), rt::Arg::Buf(tb), rt::Arg::Buf(nb), rt::Arg::Buf(hb), rt::Arg::Val(ff),
         rt::Arg::Val(H), rt::Arg::Val(k)});
      L("halo_moe_down_grouped_q5_1", {(H + 127) / 128, max_tiles}, {256},
        {rt::Arg::Buf(db), rt::Arg::Val<std::uint64_t>(H * dn_row), rt::Arg::Val<std::uint64_t>(dn_row), rt::Arg::Buf(hb),
         rt::Arg::Buf(tb), rt::Arg::Buf(nb), rt::Arg::Buf(yb), rt::Arg::Val(H), rt::Arg::Val(ff)});
      L("halo_moe_combine_grouped", {(H + 255) / 256, T}, {256},
        {rt::Arg::Buf(yb), rt::Arg::Buf(pb), rt::Arg::Buf(wb), rt::Arg::Buf(ob), rt::Arg::Val(k), rt::Arg::Val(H)});
    };
    layer();
    (void)d->Synchronize(&e);
    const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) layer();
    (void)d->Synchronize(&e);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    std::printf("moe layer (T %u, %u experts, top %u): %.2f ms; x48 layers = %.0f ms%s\n", T, E, k, ms, 48 * ms,
                e.empty() ? "" : (" error: " + e).c_str());
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "deltanet") {
    // One DeltaNet layer's conv + recurrence for a chunk of T tokens at the
    // real model's sizes (16 q/k heads, 48 v heads, d 128, kernel 4): T decode
    // steps against the chunked kernels.
    auto d = rt::Device::Create(rt::Backend::kHip, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) return std::printf("%s\n", e.c_str()), 1;
    const std::uint32_t nk = 16, nv = 48, hd = 128, kern = 4, kd = nk * hd, vd = nv * hd, ch = 2 * kd + vd;
    for (const std::uint32_t T : {64u, 512u, 2048u}) {
      std::vector<float> host(std::size_t{T} * ch, 0.01f);
      auto in = d->Alloc(host.size() * 4, &e), cs = d->Alloc(std::size_t{kern} * ch * 4, &e), cw = d->Alloc(std::size_t{ch} * kern * 4, &e),
           conv = d->Alloc(host.size() * 4, &e), ab = d->Alloc(std::size_t{T} * nv * 4, &e), st = d->Alloc(std::size_t{nv} * hd * hd * 4, &e),
           nw = d->Alloc(hd * 4, &e), z = d->Alloc(std::size_t{T} * vd * 4, &e), out = d->Alloc(std::size_t{T} * vd * 4, &e),
           hv = d->Alloc(nv * 4, &e);
      (void)d->Upload(in, 0, host.data(), host.size() * 4, &e);
      const auto time = [&](auto&& f) {
        f();
        (void)d->Synchronize(&e);
        const auto t0 = std::chrono::steady_clock::now();
        f();
        (void)d->Synchronize(&e);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      };
      const auto kc = d->Kernel("halo_conv_l2norm", &e), kg = d->Kernel("halo_gdn_step_d128_norm", &e),
                 kcp = d->Kernel("halo_conv_l2norm_prefill", &e), ka = d->Kernel("halo_conv_state_advance", &e),
                 kgp = d->Kernel("halo_gdn_prefill_d128_norm", &e);
      const double steps = time([&] {
        for (std::uint32_t t = 0; t < T; ++t) {
          const rt::Arg c[] = {rt::Arg::Buf(in, std::size_t{t} * ch * 4), rt::Arg::Buf(cs), rt::Arg::Buf(cw), rt::Arg::Buf(conv),
                               rt::Arg::Val(ch), rt::Arg::Val(kern), rt::Arg::Val(2 * kd), rt::Arg::Val(1e-6f)};
          (void)d->Launch(kc, {ch / hd}, {hd}, c, &e);
          const rt::Arg g[] = {rt::Arg::Buf(conv), rt::Arg::Buf(conv, kd * 4ull), rt::Arg::Buf(conv, 2ull * kd * 4),
                               rt::Arg::Buf(ab, std::size_t{t} * nv * 4), rt::Arg::Buf(ab, std::size_t{t} * nv * 4), rt::Arg::Buf(hv),
                               rt::Arg::Buf(hv), rt::Arg::Buf(st), rt::Arg::Buf(nw), rt::Arg::Buf(z, std::size_t{t} * vd * 4),
                               rt::Arg::Buf(out, std::size_t{t} * vd * 4), rt::Arg::Val(nk), rt::Arg::Val(1e-6f), rt::Arg::Val(0u)};
          (void)d->Launch(kg, {nv}, {hd}, g, &e);
        }
      });
      const double chunk = time([&] {
        const rt::Arg c[] = {rt::Arg::Buf(in), rt::Arg::Buf(cs), rt::Arg::Buf(cw), rt::Arg::Buf(conv), rt::Arg::Val(ch),
                             rt::Arg::Val(kern), rt::Arg::Val(2 * kd), rt::Arg::Val(1e-6f)};
        (void)d->Launch(kcp, {ch / hd, T}, {hd}, c, &e);
        const rt::Arg a[] = {rt::Arg::Buf(in), rt::Arg::Buf(cs), rt::Arg::Val(ch), rt::Arg::Val(kern), rt::Arg::Val(T)};
        (void)d->Launch(ka, {(ch + 255) / 256}, {256}, a, &e);
        const rt::Arg g[] = {rt::Arg::Buf(conv), rt::Arg::Buf(conv, kd * 4ull), rt::Arg::Buf(conv, 2ull * kd * 4), rt::Arg::Buf(ab),
                             rt::Arg::Buf(ab), rt::Arg::Buf(hv), rt::Arg::Buf(hv), rt::Arg::Buf(st), rt::Arg::Buf(nw), rt::Arg::Buf(z),
                             rt::Arg::Buf(out), rt::Arg::Val(nk), rt::Arg::Val(1e-6f), rt::Arg::Val(0u), rt::Arg::Val(T),
                             rt::Arg::Val(ch), rt::Arg::Val(ch), rt::Arg::Val(vd)};
        (void)d->Launch(kgp, {nv}, {hd}, g, &e);
      });
      // The recurrence alone, fused (bit-identical) and lane-split (+ gated norm).
      const auto ksp = d->Kernel("halo_gdn_prefill_split", &e), kn = d->Kernel("halo_gated_norm", &e);
      const double fused = time([&] {
        const rt::Arg g[] = {rt::Arg::Buf(conv), rt::Arg::Buf(conv, kd * 4ull), rt::Arg::Buf(conv, 2ull * kd * 4), rt::Arg::Buf(ab),
                             rt::Arg::Buf(ab), rt::Arg::Buf(hv), rt::Arg::Buf(hv), rt::Arg::Buf(st), rt::Arg::Buf(nw), rt::Arg::Buf(z),
                             rt::Arg::Buf(out), rt::Arg::Val(nk), rt::Arg::Val(1e-6f), rt::Arg::Val(0u), rt::Arg::Val(T),
                             rt::Arg::Val(ch), rt::Arg::Val(ch), rt::Arg::Val(vd)};
        (void)d->Launch(kgp, {nv}, {hd}, g, &e);
      });
      const double split = time([&] {
        const rt::Arg g[] = {rt::Arg::Buf(conv), rt::Arg::Buf(conv, kd * 4ull), rt::Arg::Buf(conv, 2ull * kd * 4), rt::Arg::Buf(ab),
                             rt::Arg::Buf(ab), rt::Arg::Buf(hv), rt::Arg::Buf(hv), rt::Arg::Buf(st), rt::Arg::Buf(out),
                             rt::Arg::Val(nk), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch), rt::Arg::Val(vd)};
        (void)d->Launch(ksp, {2, nv}, {256}, g, &e);
        const rt::Arg n[] = {rt::Arg::Buf(out), rt::Arg::Buf(nw), rt::Arg::Buf(z), rt::Arg::Val(hd), rt::Arg::Val(1e-6f),
                             rt::Arg::Val(0u)};
        (void)d->Launch(kn, {T * nv}, {hd}, n, &e);
      });
      std::printf("deltanet T %5u: %8.1f us as steps, %8.1f us chunked (%.1fx), %.1f us per token chunked; "
                  "recurrence fused %8.1f us, lane-split %8.1f us (%.2fx)\n",
                  T, steps * 1e6, chunk * 1e6, steps / chunk, chunk * 1e6 / T, fused * 1e6, split * 1e6, fused / split);
      for (auto* b : {&in, &cs, &cw, &conv, &ab, &st, &nw, &z, &out, &hv}) d->Free(*b);
    }
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "bf16") {
    // The Qwen3.5 torso's BF16 GEMM (fp16 WMMA), with split-K when the chunk
    // is short, at the 2B model's shapes.
    auto d = rt::Device::Create(rt::Backend::kHip, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) return std::printf("%s\n", e.c_str()), 1;
    struct Shape { const char* name; std::uint32_t rows, cols; };
    for (const Shape s : {Shape{"ffn up  6144x2048", 6144, 2048}, Shape{"ffn down 2048x6144", 2048, 6144},
                          Shape{"qkv     6144x2048", 6144, 2048}}) {
      // Eight copies cycled per launch (200 MB): weights come from DRAM, as
      // in a model whose layers each have their own.
      constexpr std::uint32_t kCopies = 8;
      std::vector<std::uint16_t> w(std::size_t{s.rows} * s.cols, 0x3c00);
      auto wb = d->Alloc(w.size() * 2 * kCopies, &e);
      for (std::uint32_t c = 0; c < kCopies; ++c) (void)d->Upload(wb, std::size_t{c} * w.size() * 2, w.data(), w.size() * 2, &e);
      std::uint32_t next = 0;
      for (const std::uint32_t T : {16u, 64u, 128u, 256u}) {
        std::vector<float> x(std::size_t{T} * s.cols, 0.01f);
        auto xb = d->Alloc(x.size() * 4, &e), yb = d->Alloc(std::size_t{T} * s.rows * 4 * 8, &e);
        (void)d->Upload(xb, 0, x.data(), x.size() * 4, &e);
        for (const char* name : {"halo_gemm_bf16_f16", "halo_gemm_bf16_f16_k128"}) {
          const auto k = d->Kernel(name, &e);
          const bool k128 = std::string(name).ends_with("k128");
          const auto run = [&] {
            const rt::Arg a[] = {rt::Arg::Buf(wb, std::size_t{next++ % kCopies} * w.size() * 2), rt::Arg::Buf(xb),
                                 rt::Arg::Buf(yb), rt::Arg::Val(s.rows), rt::Arg::Val(s.cols), rt::Arg::Val(T)};
            if (k128) (void)d->Launch(k, {(T + 63) / 64, (s.rows + 63) / 64}, {128}, a, &e);
            else (void)d->Launch(k, {(T + 63) / 64, (s.rows + 127) / 128}, {256}, a, &e);
          };
          for (int i = 0; i < 3; ++i) run();
          (void)d->Synchronize(&e);
          const auto t0 = std::chrono::steady_clock::now();
          for (int i = 0; i < 20; ++i) run();
          (void)d->Synchronize(&e);
          const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
          std::printf("%-24s %-20s T %4u  %8.1f us  %6.2f TFLOPS  %6.1f GB/s weights\n", name, s.name, T, sec * 1e6,
                      2.0 * s.rows * s.cols * T / sec / 1e12, 2.0 * s.rows * s.cols / sec / 1e9);
        }
        d->Free(xb);
        d->Free(yb);
      }
      d->Free(wb);
    }
    return 0;
  }
  const bool model = argc > 1 && std::string(argv[1]) == "model";
  const int reps = argc > 1 && !model ? std::atoi(argv[1]) : 20;
  for (auto be : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto d = rt::Device::Create(be, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) continue;
    struct Shape { const char* name; std::uint32_t rows, cols, count; };
    const std::vector<Shape> base = {Shape{"ssm qkv 10240x2560", 10240, 2560, 36},
                                     Shape{"ssm out  2560x6144", 2560, 6144, 48},
                                     Shape{"hc down   320x10240", 320, 10240, 96}};
    // Per 2048-token prefill: 36 DeltaNet and 12 attention layers, two mixers
    // and a shared expert per layer (attn out shares ssm out's shape).
    const std::vector<Shape> all = {
        Shape{"ssm qkv 10240x2560", 10240, 2560, 36}, Shape{"ssm z    6144x2560", 6144, 2560, 36},
        Shape{"out      2560x6144", 2560, 6144, 48},  Shape{"attn qg 12288x2560", 12288, 2560, 12},
        Shape{"attn kv   512x2560", 512, 2560, 24},   Shape{"hc down   320x10240", 320, 10240, 96},
        Shape{"hc up   10240x320", 10240, 320, 96},   Shape{"shexp gu  640x2560", 640, 2560, 96},
        Shape{"shexp dn 2560x640", 2560, 640, 48}};
    double total = 0, total_f16 = 0;
    if (model && be != rt::Backend::kHip) break;
    for (const Shape s : model ? all : base) {
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
      for (const std::uint32_t T : model ? std::vector<std::uint32_t>{2048u}
                                         : std::vector<std::uint32_t>{16u, 64u, 256u, 512u, 2048u}) {
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
        if (model) {
          // The FP16 route: fp16 activations (converted once), weights
          // dequantized while staging.
          auto xh = d->Alloc(x.size() * 2, &e);
          const auto kh = d->Kernel("halo_f16_rows", &e);
          const rt::Arg ha[] = {rt::Arg::Buf(xb), rt::Arg::Buf(xh), rt::Arg::Val(static_cast<std::uint32_t>(x.size()))};
          (void)d->Launch(kh, {static_cast<std::uint32_t>((x.size() + 255) / 256)}, {256}, ha, &e);
          const bool tall = s.rows <= 320 && s.cols >= 4096, small = !tall && s.rows <= 512;
          const std::uint32_t bm = tall ? 320 : small ? 64 : 256, bn = tall ? 64 : 128;
          const auto kf = d->Kernel(tall ? "halo_gemm_q8r_f16_m" : small ? "halo_gemm_q8r_f16_s" : "halo_gemm_q8r_f16_l", &e);
          const rt::Arg fa[] = {rt::Arg::Buf(qs), rt::Arg::Buf(dd), rt::Arg::Buf(xh), rt::Arg::Buf(yb),
                                rt::Arg::Val(s.rows), rt::Arg::Val(s.cols), rt::Arg::Val(s.cols), rt::Arg::Val(T)};
          const auto runf = [&] {
            (void)d->Launch(kf, {(T + bn - 1) / bn, (s.rows + bm - 1) / bm}, {256}, fa, &e);
          };
          for (int i = 0; i < 3; ++i) runf();
          (void)d->Synchronize(&e);
          const auto t1 = std::chrono::steady_clock::now();
          for (int i = 0; i < reps; ++i) runf();
          (void)d->Synchronize(&e);
          const double secf = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() / reps;
          d->Free(xh);
          total += sec * s.count;
          total_f16 += std::min(sec, secf) * s.count;
          std::printf("%-20s int8 %8.1f us %6.2f TOPS | fp16 %8.1f us %6.2f TFLOPS | x %3u = %6.1f / %6.1f ms\n", s.name,
                      sec * 1e6, 2.0 * s.rows * s.cols * T / sec / 1e12, secf * 1e6,
                      2.0 * s.rows * s.cols * T / secf / 1e12, s.count, 1e3 * sec * s.count, 1e3 * secf * s.count);
        } else {
          std::printf("%s %-20s T %5u  %9.1f us  %6.2f TOPS  %9.0f tok/s\n", rt::BackendName(be), s.name, T, sec * 1e6,
                      2.0 * s.rows * s.cols * T / sec / 1e12, T / sec);
        }
        for (auto* b : {&xb, &xq, &xd, &yb}) d->Free(*b);
      }
      for (auto* b : {&rb, &qs, &dd}) d->Free(*b);
    }
    if (model)
      std::printf("total per 2048-token prefill: int8 %.1f ms, best of int8/fp16 per shape %.1f ms\n", 1e3 * total,
                  1e3 * total_f16);
  }
}
