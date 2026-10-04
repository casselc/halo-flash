// Prefill GEMM throughput at the real model's dense shapes: the int8 WMMA
// kernel over repacked Q8_0 weights (with per-token activation
// quantization), for several prompt-chunk sizes. Reports TOPS (2 * rows *
// cols * tokens per second) and tokens per second for that one matrix.
//   halo_gemm_bench [reps]
//   halo_gemm_bench peak      register-only WMMA throughput (the ceiling)
//   halo_gemm_bench deltanet  one DeltaNet layer's chunk: decode steps vs chunked kernels
//   halo_gemm_bench bf16      the Qwen3.5 BF16 GEMMs (fp16 WMMA), weights from DRAM
//   halo_gemm_bench model     every Flash-Next dense int8 GEMM shape at T = 2048,
//                             weighted by its count per 2048-token prefill

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
