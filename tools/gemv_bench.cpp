// Achieved bandwidth of the decode GEMVs at the real model's shapes:
// back-to-back launches over one matrix (weights larger than the caches),
// reporting GB/s of weight bytes read. P0 measured a 241 GB/s read ceiling.
//   halo_gemv_bench [reps]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/core_hsaco.hpp"
#include "rt/device.hpp"

using namespace halo;

namespace {

void FillQuant(std::vector<std::uint8_t>& w, std::size_t block_bytes, std::initializer_list<std::size_t> halves) {
  std::mt19937 g(1);
  for (auto& b : w) b = static_cast<std::uint8_t>(g());
  for (std::size_t b = 0; b + block_bytes <= w.size(); b += block_bytes)
    for (std::size_t off : halves) {
      const std::uint16_t h = 0x2000 | (g() & 0x3FF);  // small finite fp16
      std::memcpy(w.data() + b + off, &h, 2);
    }
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
  std::string e;
  for (auto be : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto d = rt::Device::Create(be, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) continue;
    const auto time = [&](auto&& launch) {
      for (int i = 0; i < 5; ++i) launch();
      (void)d->Synchronize(&e);
      const auto t = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) launch();
      (void)d->Synchronize(&e);
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count() / reps;
    };
    std::vector<float> x(12288);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.01f * static_cast<float>(i));
    auto xb = d->Alloc(x.size() * 4, &e), yb = d->Alloc(16384 * 10 * 4, &e);
    (void)d->Upload(xb, 0, x.data(), x.size() * 4, &e);

    // Dense Q8_0, repacked: (rows, cols) of the model's largest dense matrices.
    struct Shape { const char* name; std::uint32_t rows, cols; };
    for (const Shape s : {Shape{"ssm qkv  10240x2560", 10240, 2560}, Shape{"ssm out   2560x6144", 2560, 6144},
                          Shape{"lm head 248320x2560", 248320, 2560}, Shape{"hc down    320x10240", 320, 10240}}) {
      std::vector<std::uint8_t> raw(std::size_t{s.rows} * (s.cols / 32) * 34);
      FillQuant(raw, 34, {0});
      auto rb = d->Alloc(raw.size(), &e), qs = d->Alloc(std::size_t{s.rows} * s.cols, &e),
           dd = d->Alloc(std::size_t{s.rows} * (s.cols / 32) * 2, &e);
      (void)d->Upload(rb, 0, raw.data(), raw.size(), &e);
      const rt::Arg ra[] = {rt::Arg::Buf(rb), rt::Arg::Buf(qs), rt::Arg::Buf(dd), rt::Arg::Val(s.rows), rt::Arg::Val(s.cols),
                            rt::Arg::Val(s.cols)};
      (void)d->Launch(d->Kernel("halo_repack_q8_0", &e), {4096}, {256}, ra, &e);
      const rt::Arg a[] = {rt::Arg::Buf(qs), rt::Arg::Buf(dd), rt::Arg::Buf(xb), rt::Arg::Buf(yb), rt::Arg::Val(s.rows),
                           rt::Arg::Val(s.cols), rt::Arg::Val(s.cols)};
      const double bytes = static_cast<double>(s.rows) * s.cols * (1 + 2.0 / 32);
      struct V { const char* kernel; std::uint32_t rows_per_block; };
      for (const V v : {V{s.cols > 8192 ? "halo_gemv_q8r_wide" : "halo_gemv_q8r", 8}, V{"halo_gemv_q8r_r2", 16},
                        V{"halo_gemv_q8r_r4", 32}, V{"halo_gemv_q8r_r4n", 32}}) {
        if (std::string(v.kernel) == "halo_gemv_q8r_r4n" && s.cols > 6144) continue;
        const auto k = d->Kernel(v.kernel, &e);
        const double sec = time([&] { (void)d->Launch(k, {(s.rows + v.rows_per_block - 1) / v.rows_per_block}, {256}, a, &e); });
        std::printf("%s %-20s %-22s %8.1f us  %6.1f GB/s\n", rt::BackendName(be), v.kernel, s.name, sec * 1e6,
                    bytes / sec / 1e9);
      }
      for (auto* b : {&rb, &qs, &dd}) d->Free(*b);
    }

    // Routed experts: 10 of 512, gate+up Q4_K [640 x 2560], down Q5_1 [2560 x 640].
    {
      const std::uint32_t E = 512, used = 10, ff = 640, H = 2560;
      const std::size_t gu_row = H / 256 * 144, dn_row = ff / 32 * 24;
      std::vector<std::uint8_t> gate(std::size_t{E} * ff * gu_row), down(std::size_t{E} * H * dn_row);
      FillQuant(gate, 144, {0, 2});
      FillQuant(down, 24, {0, 2});
      auto gb = d->Alloc(gate.size(), &e), ub = d->Alloc(gate.size(), &e), db = d->Alloc(down.size(), &e),
           ib = d->Alloc(used * 4, &e), hb = d->Alloc(used * ff * 4, &e), pb = d->Alloc(used * H * 4, &e);
      (void)d->Upload(gb, 0, gate.data(), gate.size(), &e);
      (void)d->Upload(ub, 0, gate.data(), gate.size(), &e);
      (void)d->Upload(db, 0, down.data(), down.size(), &e);
      std::vector<std::uint32_t> ids(used);
      for (std::uint32_t i = 0; i < used; ++i) ids[i] = (i * 53 + 7) % E;
      (void)d->Upload(ib, 0, ids.data(), used * 4, &e);
      const rt::Arg a1[] = {rt::Arg::Buf(gb), rt::Arg::Buf(ub), rt::Arg::Buf(ib), rt::Arg::Buf(xb), rt::Arg::Buf(hb),
                            rt::Arg::Val<std::uint64_t>(ff * gu_row), rt::Arg::Val(ff), rt::Arg::Val(H),
                            rt::Arg::Val<std::uint64_t>(gu_row)};
      const double b1 = 2.0 * used * ff * gu_row;
      for (auto [name, per_block] : {std::pair{"halo_moe_gate_up_q4_k", 8u}, std::pair{"halo_moe_gate_up_q4_k_r2", 16u},
                                     std::pair{"halo_moe_gate_up_q4_k_s2", 16u}, std::pair{"halo_moe_gate_up_q4_k_s4", 32u}}) {
        const auto k1 = d->Kernel(name, &e);
        const double s1 = time([&] { (void)d->Launch(k1, {(ff + per_block - 1) / per_block, used}, {256}, a1, &e); });
        std::printf("%s %-26s (10 experts, Q4_K) %8.1f us  %6.1f GB/s\n", rt::BackendName(be), name, s1 * 1e6, b1 / s1 / 1e9);
      }
      const rt::Arg a2[] = {rt::Arg::Buf(db), rt::Arg::Buf(ib), rt::Arg::Buf(hb), rt::Arg::Buf(pb),
                            rt::Arg::Val<std::uint64_t>(H * dn_row), rt::Arg::Val(H), rt::Arg::Val(ff),
                            rt::Arg::Val<std::uint64_t>(dn_row)};
      const double b2 = 1.0 * used * H * dn_row;
      for (auto [name, per_block] : {std::pair{"halo_moe_down_q5_1", 8u}, std::pair{"halo_moe_down_q5_1_r4", 32u}}) {
        const auto k2 = d->Kernel(name, &e);
        const double s2 = time([&] { (void)d->Launch(k2, {(H + per_block - 1) / per_block, used}, {256}, a2, &e); });
        std::printf("%s %-26s (10 experts, Q5_1) %8.1f us  %6.1f GB/s\n", rt::BackendName(be), name, s2 * 1e6, b2 / s2 / 1e9);
      }
      for (auto* b : {&gb, &ub, &db, &ib, &hb, &pb}) d->Free(*b);
    }
    d->Free(xb);
    d->Free(yb);
  }
}
