// Stream D2's decode kernels against the launches they replace: the same values in the same order, so every
// output must match bit for bit (Hegel properties over shapes, ids and value ranges, both backends).

#include <hegel/gtest.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include "kernels/core_hsaco.hpp"
#include "kernels/step_params.h"
#include "rt/device.hpp"

namespace gs = hegel::generators;
using namespace halo;

namespace {

struct Rng {
  std::uint32_t s;
  std::uint32_t U() { return s = s * 1664525u + 1013904223u; }
  float F(float amp) { return amp * (static_cast<float>(U() >> 8) / 16777216.0f - 0.5f); }
};

// A finite fp16 bit pattern of magnitude about 2^-14 .. 2^3, either sign.
std::uint16_t Half(Rng& r) {
  const std::uint16_t exp = static_cast<std::uint16_t>(1 + r.U() % 18), man = static_cast<std::uint16_t>(r.U() & 0x3FF);
  return static_cast<std::uint16_t>(((r.U() >> 31) << 15) | (exp << 10) | man);
}

void ExpectBitEqual(const std::vector<float>& want, const std::vector<float>& got, const std::string& what) {
  ASSERT_EQ(want.size(), got.size());
  for (std::size_t i = 0; i < want.size(); ++i)
    ASSERT_EQ(std::bit_cast<std::uint32_t>(got[i]), std::bit_cast<std::uint32_t>(want[i])) << what << " at " << i;
}

// Runs body(dev) for each available backend.
void ForEachDevice(const std::function<void(rt::Device&, rt::Backend)>& body) {
  std::string error;
  bool any = false;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    any = true;
    body(*dev, backend);
  }
  if (!any) GTEST_SKIP() << "no GPU backend";
}

struct Bufs {
  rt::Device& dev;
  std::vector<rt::Buffer> all;
  ~Bufs() {
    for (auto& b : all) dev.Free(b);
  }
  template <typename T>
  rt::Buffer Up(const std::vector<T>& v) {
    std::string error;
    rt::Buffer b = dev.Alloc(std::max<std::size_t>(v.size() * sizeof(T), 4), &error);
    EXPECT_TRUE(b) << error;
    if (!v.empty()) EXPECT_TRUE(dev.Upload(b, 0, v.data(), v.size() * sizeof(T), &error)) << error;
    all.push_back(b);
    return b;
  }
  rt::Buffer Zero(std::size_t floats) { return Up(std::vector<float>(floats, 0.0f)); }
  std::vector<float> Down(const rt::Buffer& b, std::size_t floats) {
    std::string error;
    std::vector<float> v(floats);
    EXPECT_TRUE(dev.Download(b, 0, v.data(), floats * 4, &error)) << error;
    return v;
  }
  void Run(const char* name, rt::Dim3 grid, std::initializer_list<rt::Arg> args) {
    std::string error;
    const std::vector<rt::Arg> a(args);
    ASSERT_TRUE(dev.Launch(dev.Kernel(name, &error), grid, {256}, a, &error)) << name << ": " << error;
  }
};

std::uint32_t Blocks(std::size_t n, std::uint32_t per) { return static_cast<std::uint32_t>((n + per - 1) / per); }

// Random repacked Q8 rows: int8 [rows][qstride], fp16 scales [rows][cols / 32].
void RandomQ8r(Rng& r, std::uint32_t rows, std::uint32_t cols, std::uint32_t qstride, std::vector<std::int8_t>& qs,
               std::vector<std::uint16_t>& d) {
  qs.assign(std::size_t{rows} * qstride, 0);
  d.resize(std::size_t{rows} * (cols / 32));
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t c = 0; c < cols; ++c) qs[std::size_t{i} * qstride + c] = static_cast<std::int8_t>(r.U() >> 24);
  for (auto& h : d) h = Half(r);
}

}  // namespace

// halo_moe_gate_up_q5_k_rows == per slot: halo_matvec_q5_k_expert (gate, up) + halo_silu_mul; and
// halo_moe_down_q8r_r4 + halo_moe_combine == per slot halo_gemv_q8r_expert_r4 + halo_axpy_slot onto zeros.
TEST(GpuDecode2, Q5kExpertsMatchPerExpertPath) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t experts = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
      const std::uint32_t slots = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 10}));
      const std::uint32_t ff = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 700}));
      const std::uint32_t H = 256 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.01f, 1.0f, 40.0f}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      const std::uint64_t row_bytes = H / 256 * 176, expert_bytes = row_bytes * ff;
      std::vector<std::uint8_t> gate(expert_bytes * experts), up(gate.size());
      for (auto* w : {&gate, &up}) {
        for (auto& b : *w) b = static_cast<std::uint8_t>(r.U() >> 24);
        for (std::size_t sb = 0; sb < w->size(); sb += 176) {  // finite d and dmin
          const std::uint16_t d = Half(r), m = Half(r);
          std::memcpy(w->data() + sb, &d, 2);
          std::memcpy(w->data() + sb + 2, &m, 2);
        }
      }
      std::vector<float> x(H), wts(slots);
      for (auto& v : x) v = r.F(amp);
      for (auto& v : wts) v = r.F(1.0f);
      std::vector<std::uint32_t> ids(slots);
      for (auto& e : ids) e = r.U() % experts;
      const std::uint32_t cols_d = (ff + 31) / 32 * 32, qstride = cols_d;
      std::vector<std::int8_t> dq;
      std::vector<std::uint16_t> dd;
      RandomQ8r(r, H * experts, cols_d, qstride, dq, dd);
      Bufs b{dev, {}};
      auto gb = b.Up(gate), ub = b.Up(up), xb = b.Up(x), ib = b.Up(ids), wb = b.Up(wts), dqb = b.Up(dq), ddb = b.Up(dd);
      // Reference: per expert.
      auto eg = b.Zero(ff), eu = b.Zero(ff), ed = b.Zero(H), out_ref = b.Zero(H);
      std::vector<float> h_ref;
      // The fused down reads h with a cols_d row stride: pad each slot's h with zeros past ff.
      std::vector<float> h_pad(std::size_t{slots} * cols_d, 0.0f);
      for (std::uint32_t s = 0; s < slots; ++s) {
        b.Run("halo_matvec_q5_k_expert", {Blocks(ff, 8)},
              {rt::Arg::Buf(gb), rt::Arg::Buf(ib), rt::Arg::Buf(xb), rt::Arg::Buf(eg), rt::Arg::Val(s),
               rt::Arg::Val<std::uint64_t>(expert_bytes), rt::Arg::Val(ff), rt::Arg::Val(H),
               rt::Arg::Val<std::uint64_t>(row_bytes)});
        b.Run("halo_matvec_q5_k_expert", {Blocks(ff, 8)},
              {rt::Arg::Buf(ub), rt::Arg::Buf(ib), rt::Arg::Buf(xb), rt::Arg::Buf(eu), rt::Arg::Val(s),
               rt::Arg::Val<std::uint64_t>(expert_bytes), rt::Arg::Val(ff), rt::Arg::Val(H),
               rt::Arg::Val<std::uint64_t>(row_bytes)});
        b.Run("halo_silu_mul", {Blocks(ff, 256)}, {rt::Arg::Buf(eg), rt::Arg::Buf(eu), rt::Arg::Val(ff)});
        const auto hs = b.Down(eg, ff);
        h_ref.insert(h_ref.end(), hs.begin(), hs.end());
        std::copy(hs.begin(), hs.end(), h_pad.begin() + std::size_t{s} * cols_d);
        auto egp = b.Up(std::vector<float>(h_pad.begin() + std::size_t{s} * cols_d, h_pad.begin() + std::size_t{s + 1} * cols_d));
        b.Run("halo_gemv_q8r_expert_r4", {Blocks(H, 32)},
              {rt::Arg::Buf(dqb), rt::Arg::Buf(ddb), rt::Arg::Buf(ib), rt::Arg::Buf(egp), rt::Arg::Buf(ed),
               rt::Arg::Val(s), rt::Arg::Val(H), rt::Arg::Val(cols_d), rt::Arg::Val(qstride)});
        b.Run("halo_axpy_slot", {Blocks(H, 256)},
              {rt::Arg::Buf(out_ref), rt::Arg::Buf(ed), rt::Arg::Buf(wb), rt::Arg::Val(s), rt::Arg::Val(H)});
      }
      // Fused.
      auto hb = b.Zero(std::size_t{slots} * ff);
      b.Run("halo_moe_gate_up_q5_k_rows", {Blocks(ff, 8), slots},
            {rt::Arg::Buf(gb), rt::Arg::Buf(ub), rt::Arg::Buf(ib), rt::Arg::Buf(xb), rt::Arg::Buf(hb),
             rt::Arg::Val<std::uint64_t>(expert_bytes), rt::Arg::Val(ff), rt::Arg::Val(H),
             rt::Arg::Val<std::uint64_t>(row_bytes)});
      ExpectBitEqual(h_ref, b.Down(hb, h_ref.size()), std::string(rt::BackendName(backend)) + " q5_k gate/up h");
      auto hpb = b.Up(h_pad), part = b.Zero(std::size_t{slots} * H), out = b.Zero(H);
      b.Run("halo_moe_down_q8r_r4", {Blocks(H, 32), slots},
            {rt::Arg::Buf(dqb), rt::Arg::Buf(ddb), rt::Arg::Buf(ib), rt::Arg::Buf(hpb), rt::Arg::Buf(part),
             rt::Arg::Val(H), rt::Arg::Val(cols_d), rt::Arg::Val(qstride)});
      b.Run("halo_moe_combine", {Blocks(H, 256)},
            {rt::Arg::Buf(part), rt::Arg::Buf(wb), rt::Arg::Buf(out), rt::Arg::Val(slots), rt::Arg::Val(H)});
      ExpectBitEqual(b.Down(out_ref, H), b.Down(out, H), std::string(rt::BackendName(backend)) + " down + combine");
    }, {.test_cases = 40});
  });
}

// halo_matvec_bf16_dual == halo_matvec_bf16 on each matrix.
TEST(GpuDecode2, Bf16DualMatchesMatvec) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t r1 = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 700}));
      const std::uint32_t r2 = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 300}));
      const std::uint32_t cols = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 200}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      std::vector<std::uint16_t> w1(std::size_t{r1} * cols), w2(std::size_t{r2} * cols);
      for (auto* w : {&w1, &w2})
        for (auto& v : *w) v = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(r.F(4.0f)) >> 16);
      std::vector<float> x(cols);
      for (auto& v : x) v = r.F(8.0f);
      Bufs b{dev, {}};
      auto w1b = b.Up(w1), w2b = b.Up(w2), xb = b.Up(x), y1 = b.Zero(r1), y2 = b.Zero(r2), z1 = b.Zero(r1),
           z2 = b.Zero(r2);
      for (auto [w, y, rows] : {std::tuple{w1b, y1, r1}, std::tuple{w2b, y2, r2}})
        b.Run("halo_matvec_bf16", {Blocks(rows, 8)},
              {rt::Arg::Buf(w), rt::Arg::Buf(xb), rt::Arg::Buf(y), rt::Arg::Val(rows), rt::Arg::Val(cols),
               rt::Arg::Val<std::uint64_t>(cols * 2ull)});
      const std::uint32_t b1 = Blocks(r1, 8);
      b.Run("halo_matvec_bf16_dual", {b1 + Blocks(r2, 8)},
            {rt::Arg::Buf(w1b), rt::Arg::Buf(z1), rt::Arg::Val(r1), rt::Arg::Buf(w2b), rt::Arg::Buf(z2), rt::Arg::Val(r2),
             rt::Arg::Buf(xb), rt::Arg::Val(cols), rt::Arg::Val(b1)});
      const std::string be = rt::BackendName(backend);
      ExpectBitEqual(b.Down(y1, r1), b.Down(z1, r1), be + " bf16 dual 1");
      ExpectBitEqual(b.Down(y2, r2), b.Down(z2, r2), be + " bf16 dual 2");
    }, {.test_cases = 40});
  });
}

// The smaller-LDS Q8 GEMVs (r2n, r4n, dual_n) == r2, r4, dual for cols within their LDS.
TEST(GpuDecode2, SmallLdsGemvsMatch) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t rows = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3000}));
      const std::uint32_t rows2 = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 1000}));
      const std::uint32_t cols = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 192}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      const std::uint32_t qstride = (cols + 15) / 16 * 16 + 16 * (r.U() % 2);
      std::vector<std::int8_t> q1, q2;
      std::vector<std::uint16_t> d1, d2;
      RandomQ8r(r, rows, cols, qstride, q1, d1);
      RandomQ8r(r, rows2, cols, qstride, q2, d2);
      std::vector<float> x(cols);
      for (auto& v : x) v = r.F(16.0f);
      Bufs b{dev, {}};
      auto q1b = b.Up(q1), d1b = b.Up(d1), q2b = b.Up(q2), d2b = b.Up(d2), xb = b.Up(x);
      const std::string be = rt::BackendName(backend);
      for (auto [a, n, pa, pn] : {std::tuple{"halo_gemv_q8r_r2", "halo_gemv_q8r_r2n", 16u, 16u},
                                   std::tuple{"halo_gemv_q8r_r4", "halo_gemv_q8r_r4n", 32u, 32u},
                                   std::tuple{"halo_gemv_q8r_r2", "halo_gemv_q8r_r2p", 16u, 16u},
                                   std::tuple{"halo_gemv_q8r_r4", "halo_gemv_q8r_r4p", 32u, 32u},
                                   std::tuple{"halo_gemv_q8r_r4", "halo_gemv_q8r_r1", 32u, 8u},
                                   std::tuple{"halo_gemv_q8r_r4", "halo_gemv_q8r_r2", 32u, 16u}}) {
        auto ya = b.Zero(rows), yn = b.Zero(rows);
        for (auto [k, y, per] : {std::tuple{a, ya, pa}, std::tuple{n, yn, pn}})
          b.Run(k, {Blocks(rows, per)},
                {rt::Arg::Buf(q1b), rt::Arg::Buf(d1b), rt::Arg::Buf(xb), rt::Arg::Buf(y), rt::Arg::Val(rows),
                 rt::Arg::Val(cols), rt::Arg::Val(qstride)});
        ExpectBitEqual(b.Down(ya, rows), b.Down(yn, rows), be + " " + n);
      }
      if (cols <= 4096) {
        auto a1 = b.Zero(rows), a2 = b.Zero(rows2), n1 = b.Zero(rows), n2 = b.Zero(rows2);
        const std::uint32_t b1 = Blocks(rows, 32);
        for (auto [k, y1, y2] : {std::tuple{"halo_gemv_q8r_dual", a1, a2}, std::tuple{"halo_gemv_q8r_dual_n", n1, n2}})
          b.Run(k, {b1 + Blocks(rows2, 32)},
                {rt::Arg::Buf(q1b), rt::Arg::Buf(d1b), rt::Arg::Buf(y1), rt::Arg::Val(rows), rt::Arg::Val(qstride),
                 rt::Arg::Buf(q2b), rt::Arg::Buf(d2b), rt::Arg::Buf(y2), rt::Arg::Val(rows2), rt::Arg::Val(qstride),
                 rt::Arg::Buf(xb), rt::Arg::Val(cols), rt::Arg::Val(b1)});
        ExpectBitEqual(b.Down(a1, rows), b.Down(n1, rows), be + " dual_n 1");
        ExpectBitEqual(b.Down(a2, rows2), b.Down(n2, rows2), be + " dual_n 2");
      }
    }, {.test_cases = 40});
  });
}

// halo_hc_mix_a_c == halo_hc_mix_a (xn, lo partials, inject partials) for hc 1..4 streams, chunks <= 2048.
TEST(GpuDecode2, HcMixAChunkLdsMatches) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t hc = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
      const std::uint32_t h = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 96}));
      const std::uint32_t low = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 20}));
      const std::uint32_t D = hc * h;
      const std::uint32_t parts_want = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
      const std::uint32_t chunk = std::min<std::uint32_t>(2048, (D / parts_want + 511) / 512 * 512);
      const std::uint32_t parts = (D + chunk - 1) / chunk;
      const bool inject = tc.draw(gs::booleans());
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      const std::uint32_t qstride = (D + 15) / 16 * 16;
      std::vector<std::int8_t> q;
      std::vector<std::uint16_t> d;
      RandomQ8r(r, low, D, qstride, q, d);
      std::vector<float> res(D), nw(D), scale(hc), injw(std::size_t{hc} * D);
      for (auto& v : res) v = r.F(20.0f);
      for (auto& v : nw) v = r.F(3.0f);
      for (auto& v : scale) v = 0.1f + r.F(1.0f);
      for (auto& v : injw) v = r.F(1.0f);
      Bufs b{dev, {}};
      auto rb = b.Up(res), nb = b.Up(nw), sb = b.Up(scale), qb = b.Up(q), db = b.Up(d), ib = b.Up(injw);
      const std::uint32_t down_blocks = Blocks(low, 16), inject_rows = inject ? hc : 0;
      std::vector<std::vector<float>> outs;
      for (const char* k : {"halo_hc_mix_a", "halo_hc_mix_a_c"}) {
        auto xn = b.Zero(D), lo = b.Zero(std::size_t{parts} * low), ip = b.Zero(std::size_t{parts} * hc);
        b.Run(k, {down_blocks + inject_rows, parts},
              {rt::Arg::Buf(rb), rt::Arg::Buf(nb), rt::Arg::Buf(sb), rt::Arg::Buf(qb), rt::Arg::Buf(db),
               rt::Arg::Buf(inject ? ib : rb), rt::Arg::Buf(xn), rt::Arg::Buf(lo), rt::Arg::Buf(ip), rt::Arg::Val(hc),
               rt::Arg::Val(h), rt::Arg::Val(low), rt::Arg::Val(down_blocks), rt::Arg::Val(qstride),
               rt::Arg::Val(chunk)});
        auto v = b.Down(xn, D);
        const auto l = b.Down(lo, std::size_t{parts} * low), p = b.Down(ip, std::size_t{parts} * hc);
        v.insert(v.end(), l.begin(), l.end());
        if (inject) v.insert(v.end(), p.begin(), p.end());
        outs.push_back(v);
      }
      ExpectBitEqual(outs[0], outs[1], std::string(rt::BackendName(backend)) + " hc_mix_a_c");
    }, {.test_cases = 40});
  });
}

// halo_moe_gate_up_q4_k_dot_p5 / _p8 (weights loaded before the activations are quantized) == _dot.
TEST(GpuDecode2, GateUpPrefetchMatchesDot) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t experts = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
      const std::uint32_t slots = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 10}));
      const std::uint32_t ff = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 700}));
      const std::uint32_t H = 256 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 16}));
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.01f, 1.0f, 40.0f}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      const std::uint64_t row_bytes = H / 256 * 144, expert_bytes = row_bytes * ff;
      std::vector<std::uint8_t> gate(expert_bytes * experts), up(gate.size());
      for (auto* w : {&gate, &up}) {
        for (auto& b : *w) b = static_cast<std::uint8_t>(r.U() >> 24);
        for (std::size_t sb = 0; sb < w->size(); sb += 144) {
          const std::uint16_t d = Half(r), m = Half(r);
          std::memcpy(w->data() + sb, &d, 2);
          std::memcpy(w->data() + sb + 2, &m, 2);
        }
      }
      std::vector<float> x(H);
      for (auto& v : x) v = r.F(amp);
      std::vector<std::uint32_t> ids(slots);
      for (auto& e : ids) e = r.U() % experts;
      Bufs b{dev, {}};
      auto gb = b.Up(gate), ub = b.Up(up), xb = b.Up(x), ib = b.Up(ids);
      std::vector<const char*> ks{"halo_moe_gate_up_q4_k_dot", "halo_moe_gate_up_q4_k_dot_p8"};
      if (4 * (H / 64) <= 160) ks.push_back("halo_moe_gate_up_q4_k_dot_p5");
      std::vector<std::vector<float>> outs;
      for (const char* k : ks) {
        auto hb = b.Zero(std::size_t{slots} * ff);
        b.Run(k, {Blocks(ff, 16), slots},
              {rt::Arg::Buf(gb), rt::Arg::Buf(ub), rt::Arg::Buf(ib), rt::Arg::Buf(xb), rt::Arg::Buf(hb),
               rt::Arg::Val<std::uint64_t>(expert_bytes), rt::Arg::Val(ff), rt::Arg::Val(H),
               rt::Arg::Val<std::uint64_t>(row_bytes)});
        outs.push_back(b.Down(hb, std::size_t{slots} * ff));
      }
      for (std::size_t i = 1; i < outs.size(); ++i)
        ExpectBitEqual(outs[0], outs[i], std::string(rt::BackendName(backend)) + " " + ks[i]);
    }, {.test_cases = 40});
  });
}

// halo_qsa_scores_b == halo_qsa_scores (1 .. 8 heads, dims 8 .. 256, up to 3000 complete blocks).
TEST(GpuDecode2, QsaScoresBatchedMatch) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t heads = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
      const std::uint32_t dim = 8 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 2048 / 8 / heads}));
      const std::uint32_t nb = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 3000}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      std::vector<float> q(std::size_t{heads} * dim), k(std::size_t{std::max(nb, 1u)} * dim);
      for (auto& v : q) v = r.F(6.0f);
      for (auto& v : k) v = r.F(6.0f);
      std::vector<std::uint32_t> params(16, 0);
      params[kHaloComplete] = nb;
      Bufs b{dev, {}};
      auto qb = b.Up(q), kb = b.Up(k), pb = b.Up(params);
      std::vector<std::vector<float>> outs;
      const std::uint32_t grid = std::min(Blocks(nb + 1, 256), 64u);
      for (const char* name : {"halo_qsa_scores", "halo_qsa_scores_b"}) {
        auto sb = b.Zero(std::max(nb, 1u));
        b.Run(name, {grid},
              {rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(sb), rt::Arg::Buf(pb), rt::Arg::Val(heads),
               rt::Arg::Val(dim)});
        outs.push_back(b.Down(sb, std::max(nb, 1u)));
      }
      ExpectBitEqual(outs[0], outs[1], std::string(rt::BackendName(backend)) + " qsa scores");
    }, {.test_cases = 40});
  });
}

// halo_hc_combine_scales == halo_hc_combine then halo_hc_norm_scales (residual and scales).
TEST(GpuDecode2, CombineScalesMatch) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t hc = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
      const std::uint32_t h = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4096}));
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.001f, 1.0f, 300.0f}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      std::vector<float> res(std::size_t{hc} * h), out(h), inj(hc);
      for (auto& v : res) v = r.F(amp);
      for (auto& v : out) v = r.F(amp);
      for (auto& v : inj) v = r.F(8.0f);
      const float eps = 1e-6f;
      Bufs b{dev, {}};
      auto r1 = b.Up(res), r2 = b.Up(res), ob = b.Up(out), ib = b.Up(inj), s1 = b.Zero(hc), s2 = b.Zero(hc);
      b.Run("halo_hc_combine", {Blocks(std::size_t{hc} * h, 256)},
            {rt::Arg::Buf(r1), rt::Arg::Buf(ob), rt::Arg::Buf(ib), rt::Arg::Val(hc), rt::Arg::Val(h)});
      b.Run("halo_hc_norm_scales", {hc}, {rt::Arg::Buf(r1), rt::Arg::Buf(s1), rt::Arg::Val(h), rt::Arg::Val(eps)});
      b.Run("halo_hc_combine_scales", {hc},
            {rt::Arg::Buf(r2), rt::Arg::Buf(ob), rt::Arg::Buf(ib), rt::Arg::Buf(s2), rt::Arg::Val(hc), rt::Arg::Val(h),
             rt::Arg::Val(eps)});
      const std::string be = rt::BackendName(backend);
      ExpectBitEqual(b.Down(r1, res.size()), b.Down(r2, res.size()), be + " combine res");
      ExpectBitEqual(b.Down(s1, hc), b.Down(s2, hc), be + " combine scales");
    }, {.test_cases = 40});
  });
}

// halo_shexp_down_combine == halo_moe_combine then halo_shexp_down_add.
TEST(GpuDecode2, ShexpDownCombineMatches) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t rows = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3000}));
      const std::uint32_t cols = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 64}));
      const std::uint32_t slots = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 10}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      const std::uint32_t qstride = (cols + 15) / 16 * 16;
      std::vector<std::int8_t> q;
      std::vector<std::uint16_t> d;
      RandomQ8r(r, rows, cols, qstride, q, d);
      std::vector<float> hin(cols), part(std::size_t{slots} * rows), w(slots), sg{r.F(10.0f)};
      for (auto& v : hin) v = r.F(4.0f);
      for (auto& v : part) v = r.F(30.0f);
      for (auto& v : w) v = r.F(1.0f);
      Bufs b{dev, {}};
      auto qb = b.Up(q), db = b.Up(d), hb = b.Up(hin), pb = b.Up(part), wb = b.Up(w), sgb = b.Up(sg),
           o1 = b.Zero(rows), o2 = b.Zero(rows);
      b.Run("halo_moe_combine", {Blocks(rows, 256)},
            {rt::Arg::Buf(pb), rt::Arg::Buf(wb), rt::Arg::Buf(o1), rt::Arg::Val(slots), rt::Arg::Val(rows)});
      b.Run("halo_shexp_down_add", {Blocks(rows, 32)},
            {rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(hb), rt::Arg::Buf(sgb), rt::Arg::Buf(o1), rt::Arg::Val(rows),
             rt::Arg::Val(cols), rt::Arg::Val(qstride)});
      auto o3 = b.Zero(rows);
      for (auto [k, o] : {std::pair{"halo_shexp_down_combine", o2}, std::pair{"halo_shexp_down_combine_p", o3}})
        b.Run(k, {Blocks(rows, 32)},
              {rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(hb), rt::Arg::Buf(sgb), rt::Arg::Buf(pb),
               rt::Arg::Buf(wb), rt::Arg::Val(slots), rt::Arg::Buf(o), rt::Arg::Val(rows), rt::Arg::Val(cols),
               rt::Arg::Val(qstride)});
      ExpectBitEqual(b.Down(o1, rows), b.Down(o2, rows), std::string(rt::BackendName(backend)) + " shexp down combine");
      ExpectBitEqual(b.Down(o1, rows), b.Down(o3, rows), std::string(rt::BackendName(backend)) + " shexp down combine p");
      // halo_shexp_gate_up_p == halo_shexp_gate_up (rows x cols gate and up, the same q8 data for both).
      std::vector<float> gin(cols);
      for (auto& v : gin) v = r.F(1.0f);
      auto gib = b.Up(gin), h1 = b.Zero(rows), h2 = b.Zero(rows), g1 = b.Zero(1), g2 = b.Zero(1);
      if (cols <= 8192)
        for (auto [k, hh, gg] : {std::tuple{"halo_shexp_gate_up", h1, g1}, std::tuple{"halo_shexp_gate_up_p", h2, g2}})
          b.Run(k, {Blocks(rows, 16)},
                {rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(gib),
                 rt::Arg::Buf(hb), rt::Arg::Buf(hh), rt::Arg::Buf(gg), rt::Arg::Val(rows), rt::Arg::Val(cols),
                 rt::Arg::Val(qstride)});
      ExpectBitEqual(b.Down(h1, rows), b.Down(h2, rows), std::string(rt::BackendName(backend)) + " shexp gate_up p");
      ExpectBitEqual(b.Down(g1, 1), b.Down(g2, 1), std::string(rt::BackendName(backend)) + " shexp sgate p");
    }, {.test_cases = 40});
  });
}

// halo_gemv_q8r_dual_ab == halo_gemv_q8r_dual + halo_matvec_f32_rowblock2.
TEST(GpuDecode2, DualAbMatches) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t r1 = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 2000}));
      const std::uint32_t r2 = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 1000}));
      const std::uint32_t ab = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 64}));
      const std::uint32_t cols = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 128}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      const std::uint32_t qstride = (cols + 15) / 16 * 16;
      std::vector<std::int8_t> q1, q2;
      std::vector<std::uint16_t> d1, d2;
      RandomQ8r(r, r1, cols, qstride, q1, d1);
      RandomQ8r(r, r2, cols, qstride, q2, d2);
      std::vector<float> x(cols), wa(std::size_t{ab} * cols), wbv(wa.size());
      for (auto& v : x) v = r.F(8.0f);
      for (auto& v : wa) v = r.F(1.0f);
      for (auto& v : wbv) v = r.F(1.0f);
      Bufs b{dev, {}};
      auto q1b = b.Up(q1), d1b = b.Up(d1), q2b = b.Up(q2), d2b = b.Up(d2), xb = b.Up(x), wab = b.Up(wa), wbb = b.Up(wbv);
      auto y1 = b.Zero(r1), y2 = b.Zero(r2), ya = b.Zero(ab), yb = b.Zero(ab);
      auto z1 = b.Zero(r1), z2 = b.Zero(r2), za = b.Zero(ab), zb = b.Zero(ab);
      const std::uint32_t b1 = Blocks(r1, 32), b12 = b1 + Blocks(r2, 32);
      b.Run("halo_gemv_q8r_dual", {b12},
            {rt::Arg::Buf(q1b), rt::Arg::Buf(d1b), rt::Arg::Buf(y1), rt::Arg::Val(r1), rt::Arg::Val(qstride),
             rt::Arg::Buf(q2b), rt::Arg::Buf(d2b), rt::Arg::Buf(y2), rt::Arg::Val(r2), rt::Arg::Val(qstride),
             rt::Arg::Buf(xb), rt::Arg::Val(cols), rt::Arg::Val(b1)});
      b.Run("halo_matvec_f32_rowblock2", {2 * ab},
            {rt::Arg::Buf(wab), rt::Arg::Buf(wbb), rt::Arg::Buf(xb), rt::Arg::Buf(ya), rt::Arg::Buf(yb), rt::Arg::Val(ab),
             rt::Arg::Val(cols)});
      const std::string be = rt::BackendName(backend);
      for (const char* k : {"halo_gemv_q8r_dual_ab", "halo_gemv_q8r_dual_abp"}) {
      b.Run(k, {b12 + 2 * ab},
            {rt::Arg::Buf(q1b), rt::Arg::Buf(d1b), rt::Arg::Buf(z1), rt::Arg::Val(r1), rt::Arg::Val(qstride),
             rt::Arg::Buf(q2b), rt::Arg::Buf(d2b), rt::Arg::Buf(z2), rt::Arg::Val(r2), rt::Arg::Val(qstride),
             rt::Arg::Buf(xb), rt::Arg::Val(cols), rt::Arg::Val(b1), rt::Arg::Val(b12), rt::Arg::Buf(wab),
             rt::Arg::Buf(wbb), rt::Arg::Buf(za), rt::Arg::Buf(zb), rt::Arg::Val(ab)});
      ExpectBitEqual(b.Down(y1, r1), b.Down(z1, r1), be + " dual_ab 1");
      ExpectBitEqual(b.Down(y2, r2), b.Down(z2, r2), be + " dual_ab 2");
      ExpectBitEqual(b.Down(ya, ab), b.Down(za, ab), be + " dual_ab alpha");
      ExpectBitEqual(b.Down(yb, ab), b.Down(zb, ab), be + " dual_ab beta");
      }
    }, {.test_cases = 40});
  });
}

// halo_moe_down_q5_1_r4p == halo_moe_down_q5_1_r4 (cols <= 768); halo_hc_mix_b_p == halo_hc_mix_b (low rank <= 512).
TEST(GpuDecode2, WeightsFirstKernelsMatch) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::string be = rt::BackendName(backend);
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      {
        const std::uint32_t experts = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
        const std::uint32_t slots = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 10}));
        const std::uint32_t rows = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3000}));
        const std::uint32_t cols = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 24}));
        const std::uint64_t row_bytes = cols / 32 * 24, expert_bytes = row_bytes * rows;
        std::vector<std::uint8_t> w(expert_bytes * experts);
        for (auto& b : w) b = static_cast<std::uint8_t>(r.U() >> 24);
        for (std::size_t o = 0; o < w.size(); o += 24) {
          const std::uint16_t d = Half(r), m = Half(r);
          std::memcpy(w.data() + o, &d, 2);
          std::memcpy(w.data() + o + 2, &m, 2);
        }
        std::vector<float> hv(std::size_t{slots} * cols);
        for (auto& v : hv) v = r.F(4.0f);
        std::vector<std::uint32_t> ids(slots);
        for (auto& e : ids) e = r.U() % experts;
        Bufs b{dev, {}};
        auto wb = b.Up(w), hb = b.Up(hv), ib = b.Up(ids);
        std::vector<std::vector<float>> outs;
        for (const char* k : {"halo_moe_down_q5_1_r4", "halo_moe_down_q5_1_r4p"}) {
          auto pb = b.Zero(std::size_t{slots} * rows);
          b.Run(k, {Blocks(rows, 32), slots},
                {rt::Arg::Buf(wb), rt::Arg::Buf(ib), rt::Arg::Buf(hb), rt::Arg::Buf(pb),
                 rt::Arg::Val<std::uint64_t>(expert_bytes), rt::Arg::Val(rows), rt::Arg::Val(cols),
                 rt::Arg::Val<std::uint64_t>(row_bytes)});
          outs.push_back(b.Down(pb, std::size_t{slots} * rows));
        }
        ExpectBitEqual(outs[0], outs[1], be + " q5_1 down r4p");
      }
      {
        const std::uint32_t hc = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
        const std::uint32_t h = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3000}));
        const std::uint32_t low = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 16}));
        const std::uint32_t parts = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
        const std::uint32_t inject_rows = tc.draw(gs::booleans()) ? hc : 0;
        const std::uint32_t qstride = low;
        std::vector<std::int8_t> q;
        std::vector<std::uint16_t> d;
        RandomQ8r(r, hc * h, low, qstride, q, d);
        std::vector<float> lo(std::size_t{parts} * low), ip(std::size_t{parts} * hc), xn(std::size_t{hc} * h);
        for (auto& v : lo) v = r.F(8.0f);
        for (auto& v : ip) v = r.F(8.0f);
        for (auto& v : xn) v = r.F(4.0f);
        Bufs b{dev, {}};
        auto lb = b.Up(lo), ipb = b.Up(ip), qb = b.Up(q), db = b.Up(d), xb = b.Up(xn);
        std::vector<std::vector<float>> outs;
        for (const char* k : {"halo_hc_mix_b", "halo_hc_mix_b_p"}) {
          auto inj = b.Zero(hc), mixed = b.Zero(h);
          b.Run(k, {Blocks(h, 8)},
                {rt::Arg::Buf(lb), rt::Arg::Val(parts), rt::Arg::Buf(ipb), rt::Arg::Buf(inj), rt::Arg::Val(inject_rows),
                 rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(xb), rt::Arg::Buf(mixed), rt::Arg::Val(hc),
                 rt::Arg::Val(h), rt::Arg::Val(low), rt::Arg::Val(qstride)});
          auto v = b.Down(mixed, h);
          const auto iv = b.Down(inj, hc);
          v.insert(v.end(), iv.begin(), iv.end());
          outs.push_back(v);
        }
        ExpectBitEqual(outs[0], outs[1], be + " hc_mix_b_p");
      }
    }, {.test_cases = 40});
  });
}

// halo_router_rowblock_q8act + halo_moe_gate_up_q4_k_dot_g == halo_matvec_f32_rowblock + halo_moe_gate_up_q4_k_dot.
TEST(GpuDecode2, Q8ActOnceMatches) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t experts = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
      const std::uint32_t slots = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 10}));
      const std::uint32_t ff = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 700}));
      const std::uint32_t H = 256 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 16}));
      const std::uint32_t nr = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 600}));
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.01f, 1.0f, 40.0f}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      const std::uint64_t row_bytes = H / 256 * 144, expert_bytes = row_bytes * ff;
      std::vector<std::uint8_t> gate(expert_bytes * experts), up(gate.size());
      for (auto* w : {&gate, &up}) {
        for (auto& b : *w) b = static_cast<std::uint8_t>(r.U() >> 24);
        for (std::size_t sb = 0; sb < w->size(); sb += 144) {
          const std::uint16_t d = Half(r), m = Half(r);
          std::memcpy(w->data() + sb, &d, 2);
          std::memcpy(w->data() + sb + 2, &m, 2);
        }
      }
      std::vector<float> x(H), wr(std::size_t{nr} * H);
      for (auto& v : x) v = r.F(amp);
      for (auto& v : wr) v = r.F(1.0f);
      std::vector<std::uint32_t> ids(slots);
      for (auto& e : ids) e = r.U() % experts;
      Bufs b{dev, {}};
      auto gb = b.Up(gate), ub = b.Up(up), xb = b.Up(x), ib = b.Up(ids), wrb = b.Up(wr);
      auto y1 = b.Zero(nr), y2 = b.Zero(nr), act = b.Zero(1280), h1 = b.Zero(std::size_t{slots} * ff),
           h2 = b.Zero(std::size_t{slots} * ff);
      b.Run("halo_matvec_f32_rowblock", {nr},
            {rt::Arg::Buf(wrb), rt::Arg::Buf(xb), rt::Arg::Buf(y1), rt::Arg::Val(nr), rt::Arg::Val(H)});
      b.Run("halo_router_rowblock_q8act", {std::max(nr, H / 32)},
            {rt::Arg::Buf(wrb), rt::Arg::Buf(xb), rt::Arg::Buf(y2), rt::Arg::Val(nr), rt::Arg::Val(H), rt::Arg::Buf(act)});
      for (auto [k, src, hb] : {std::tuple{"halo_moe_gate_up_q4_k_dot", xb, h1}, std::tuple{"halo_moe_gate_up_q4_k_dot_g", act, h2}})
        b.Run(k, {Blocks(ff, 16), slots},
              {rt::Arg::Buf(gb), rt::Arg::Buf(ub), rt::Arg::Buf(ib), rt::Arg::Buf(src), rt::Arg::Buf(hb),
               rt::Arg::Val<std::uint64_t>(expert_bytes), rt::Arg::Val(ff), rt::Arg::Val(H),
               rt::Arg::Val<std::uint64_t>(row_bytes)});
      const std::string be = rt::BackendName(backend);
      ExpectBitEqual(b.Down(y1, nr), b.Down(y2, nr), be + " router rows");
      ExpectBitEqual(b.Down(h1, std::size_t{slots} * ff), b.Down(h2, std::size_t{slots} * ff), be + " gate/up from act");
    }, {.test_cases = 40});
  });
}

// Hoisted-load F32 row dots == the looped originals: rowblock, rowblock2, shexp gate/up (sgate and h).
TEST(GpuDecode2, HoistedRowLoadsMatch) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t rows = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 600}));
      const std::uint32_t cols = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 128}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      std::vector<float> wa(std::size_t{rows} * cols), wb(wa.size()), x(cols);
      for (auto& v : wa) v = r.F(2.0f);
      for (auto& v : wb) v = r.F(2.0f);
      for (auto& v : x) v = r.F(8.0f);
      Bufs b{dev, {}};
      auto wab = b.Up(wa), wbb = b.Up(wb), xb = b.Up(x);
      const std::string be = rt::BackendName(backend);
      auto y1 = b.Zero(rows), y2 = b.Zero(rows);
      b.Run("halo_matvec_f32_rowblock", {rows}, {rt::Arg::Buf(wab), rt::Arg::Buf(xb), rt::Arg::Buf(y1), rt::Arg::Val(rows), rt::Arg::Val(cols)});
      b.Run("halo_matvec_f32_rowblock_u", {rows}, {rt::Arg::Buf(wab), rt::Arg::Buf(xb), rt::Arg::Buf(y2), rt::Arg::Val(rows), rt::Arg::Val(cols)});
      ExpectBitEqual(b.Down(y1, rows), b.Down(y2, rows), be + " rowblock_u");
      auto a1 = b.Zero(rows), b1 = b.Zero(rows), a2 = b.Zero(rows), b2 = b.Zero(rows);
      for (auto [k, ya, yb] : {std::tuple{"halo_matvec_f32_rowblock2", a1, b1}, std::tuple{"halo_matvec_f32_rowblock2_u", a2, b2}})
        b.Run(k, {2 * rows}, {rt::Arg::Buf(wab), rt::Arg::Buf(wbb), rt::Arg::Buf(xb), rt::Arg::Buf(ya), rt::Arg::Buf(yb),
                              rt::Arg::Val(rows), rt::Arg::Val(cols)});
      ExpectBitEqual(b.Down(a1, rows), b.Down(a2, rows), be + " rowblock2_u a");
      ExpectBitEqual(b.Down(b1, rows), b.Down(b2, rows), be + " rowblock2_u b");
      const std::uint32_t qstride = cols;
      std::vector<std::int8_t> q;
      std::vector<std::uint16_t> d;
      RandomQ8r(r, rows, cols, qstride, q, d);
      auto qb = b.Up(q), db = b.Up(d), h1 = b.Zero(rows), h2 = b.Zero(rows), g1 = b.Zero(1), g2 = b.Zero(1);
      for (auto [k, hh, gg] : {std::tuple{"halo_shexp_gate_up", h1, g1}, std::tuple{"halo_shexp_gate_up_u", h2, g2}})
        b.Run(k, {Blocks(rows, 16)},
              {rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(wab), rt::Arg::Buf(xb),
               rt::Arg::Buf(hh), rt::Arg::Buf(gg), rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val(qstride)});
      ExpectBitEqual(b.Down(h1, rows), b.Down(h2, rows), be + " shexp_gate_up_u h");
      ExpectBitEqual(b.Down(g1, 1), b.Down(g2, 1), be + " shexp_gate_up_u sgate");
    }, {.test_cases = 40});
  });
}

// The hoisted small kernels == their originals: norm scales, moe combine, shexp down + combine, mix_b, mix_a_c.
TEST(GpuDecode2, HoistedSmallKernelsMatch) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::string be = rt::BackendName(backend);
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      const std::uint32_t hc = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
      const std::uint32_t h = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 96}));
      const std::uint32_t slots = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 16}));
      Bufs b{dev, {}};
      std::vector<float> res(std::size_t{hc} * h), part(std::size_t{slots} * h), w(slots);
      for (auto& v : res) v = r.F(50.0f);
      for (auto& v : part) v = r.F(20.0f);
      for (auto& v : w) v = r.F(1.0f);
      auto rb = b.Up(res), pb = b.Up(part), wb = b.Up(w);
      {
        auto s1 = b.Zero(hc), s2 = b.Zero(hc);
        b.Run("halo_hc_norm_scales", {hc}, {rt::Arg::Buf(rb), rt::Arg::Buf(s1), rt::Arg::Val(h), rt::Arg::Val(1e-6f)});
        b.Run("halo_hc_norm_scales_u", {hc}, {rt::Arg::Buf(rb), rt::Arg::Buf(s2), rt::Arg::Val(h), rt::Arg::Val(1e-6f)});
        ExpectBitEqual(b.Down(s1, hc), b.Down(s2, hc), be + " norm_scales_u");
      }
      {
        auto o1 = b.Zero(h), o2 = b.Zero(h);
        for (auto [k, o] : {std::pair{"halo_moe_combine", o1}, std::pair{"halo_moe_combine_u", o2}})
          b.Run(k, {Blocks(h, 256)}, {rt::Arg::Buf(pb), rt::Arg::Buf(wb), rt::Arg::Buf(o), rt::Arg::Val(slots), rt::Arg::Val(h)});
        ExpectBitEqual(b.Down(o1, h), b.Down(o2, h), be + " moe_combine_u");
        const std::uint32_t cols = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 32}));
        std::vector<std::int8_t> q;
        std::vector<std::uint16_t> d;
        RandomQ8r(r, h, cols, cols, q, d);
        std::vector<float> hin(cols), sg{r.F(6.0f)};
        for (auto& v : hin) v = r.F(4.0f);
        auto qb = b.Up(q), db = b.Up(d), hb = b.Up(hin), sgb = b.Up(sg), c1 = b.Zero(h), c2 = b.Zero(h);
        for (auto [k, o] : {std::pair{"halo_shexp_down_combine", c1}, std::pair{"halo_shexp_down_combine_u", c2}})
          b.Run(k, {Blocks(h, 32)},
                {rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(hb), rt::Arg::Buf(sgb), rt::Arg::Buf(pb), rt::Arg::Buf(wb),
                 rt::Arg::Val(slots), rt::Arg::Buf(o), rt::Arg::Val(h), rt::Arg::Val(cols), rt::Arg::Val(cols)});
        ExpectBitEqual(b.Down(c1, h), b.Down(c2, h), be + " shexp_down_combine_u");
      }
      {
        const std::uint32_t low = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 16}));
        const std::uint32_t parts = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 16}));
        const std::uint32_t inject_rows = tc.draw(gs::booleans()) ? hc : 0;
        std::vector<std::int8_t> q;
        std::vector<std::uint16_t> d;
        RandomQ8r(r, hc * h, low, low, q, d);
        std::vector<float> lo(std::size_t{parts} * low), ip(std::size_t{parts} * hc);
        for (auto& v : lo) v = r.F(8.0f);
        for (auto& v : ip) v = r.F(8.0f);
        auto lb = b.Up(lo), ipb = b.Up(ip), qb = b.Up(q), db = b.Up(d);
        std::vector<std::vector<float>> outs;
        for (const char* k : {"halo_hc_mix_b", "halo_hc_mix_b_u"}) {
          auto inj = b.Zero(hc), mixed = b.Zero(h);
          b.Run(k, {Blocks(h, 8)},
                {rt::Arg::Buf(lb), rt::Arg::Val(parts), rt::Arg::Buf(ipb), rt::Arg::Buf(inj), rt::Arg::Val(inject_rows),
                 rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(rb), rt::Arg::Buf(mixed), rt::Arg::Val(hc),
                 rt::Arg::Val(h), rt::Arg::Val(low), rt::Arg::Val(low)});
          auto v = b.Down(mixed, h);
          const auto iv = b.Down(inj, hc);
          v.insert(v.end(), iv.begin(), iv.end());
          outs.push_back(v);
        }
        ExpectBitEqual(outs[0], outs[1], be + " hc_mix_b_u");
      }
      {
        const std::uint32_t D = hc * h, low = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
        const std::uint32_t pw = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
        const std::uint32_t chunk = std::min<std::uint32_t>(2048, (D / pw + 511) / 512 * 512), parts = (D + chunk - 1) / chunk;
        std::vector<std::int8_t> q;
        std::vector<std::uint16_t> d;
        RandomQ8r(r, low, D, D, q, d);
        std::vector<float> nw(D), sc(hc), injw(std::size_t{hc} * D);
        for (auto& v : nw) v = r.F(3.0f);
        for (auto& v : sc) v = 0.1f + r.F(1.0f);
        for (auto& v : injw) v = r.F(1.0f);
        auto nb = b.Up(nw), sb = b.Up(sc), qb = b.Up(q), db = b.Up(d), ib = b.Up(injw);
        const std::uint32_t down_blocks = Blocks(low, 16);
        std::vector<std::vector<float>> outs;
        for (const char* k : {"halo_hc_mix_a_c", "halo_hc_mix_a_cu"}) {
          auto xn = b.Zero(D), lo = b.Zero(std::size_t{parts} * low), ip = b.Zero(std::size_t{parts} * hc);
          b.Run(k, {down_blocks + hc, parts},
                {rt::Arg::Buf(rb), rt::Arg::Buf(nb), rt::Arg::Buf(sb), rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(ib),
                 rt::Arg::Buf(xn), rt::Arg::Buf(lo), rt::Arg::Buf(ip), rt::Arg::Val(hc), rt::Arg::Val(h), rt::Arg::Val(low),
                 rt::Arg::Val(down_blocks), rt::Arg::Val(D), rt::Arg::Val(chunk)});
          auto v = b.Down(xn, D);
          const auto l = b.Down(lo, std::size_t{parts} * low), p2 = b.Down(ip, std::size_t{parts} * hc);
          v.insert(v.end(), l.begin(), l.end());
          v.insert(v.end(), p2.begin(), p2.end());
          outs.push_back(v);
        }
        ExpectBitEqual(outs[0], outs[1], be + " hc_mix_a_cu");
      }
    }, {.test_cases = 40});
  });
}

// halo_gdn_step_d128_norm_r == halo_gdn_step_d128_norm (output and updated state, both gate modes).
TEST(GpuDecode2, GdnStepNoSpillMatches) {
  ForEachDevice([](rt::Device& dev, rt::Backend backend) {
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t kh = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 16}));
      const std::uint32_t vh = kh * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3}));
      const std::uint32_t mode = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 1}));
      Rng r{tc.draw(gs::integers<std::uint32_t>())};
      constexpr std::uint32_t d = 128;
      std::vector<float> q(std::size_t{kh} * d), k(q.size()), v(std::size_t{vh} * d), al(vh), be(vh), a(vh), dt(vh),
          st(std::size_t{vh} * d * d), nw(d), z(v.size());
      for (auto* vec : {&q, &k}) for (auto& x : *vec) x = r.F(0.3f);
      for (auto& x : v) x = r.F(2.0f);
      for (auto* vec : {&al, &be, &dt}) for (auto& x : *vec) x = r.F(4.0f);
      for (auto& x : a) x = -std::abs(r.F(2.0f));
      for (auto& x : st) x = r.F(1.0f);
      for (auto& x : nw) x = r.F(2.0f);
      for (auto& x : z) x = r.F(6.0f);
      Bufs b{dev, {}};
      auto qb = b.Up(q), kb = b.Up(k), vb = b.Up(v), alb = b.Up(al), beb = b.Up(be), ab = b.Up(a), dtb = b.Up(dt),
           nb = b.Up(nw), zb = b.Up(z);
      std::vector<std::vector<float>> outs;
      for (const char* name : {"halo_gdn_step_d128_norm", "halo_gdn_step_d128_norm_r"}) {
        auto sb = b.Up(st), ob = b.Zero(v.size());
        std::string error;
        const std::vector<rt::Arg> args{rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(vb), rt::Arg::Buf(alb),
                                        rt::Arg::Buf(beb), rt::Arg::Buf(ab), rt::Arg::Buf(dtb), rt::Arg::Buf(sb),
                                        rt::Arg::Buf(nb), rt::Arg::Buf(zb), rt::Arg::Buf(ob), rt::Arg::Val(kh),
                                        rt::Arg::Val(1e-6f), rt::Arg::Val(mode)};
        ASSERT_TRUE(dev.Launch(dev.Kernel(name, &error), {vh}, {128}, args, &error)) << error;
        auto o = b.Down(ob, v.size());
        const auto s2 = b.Down(sb, st.size());
        o.insert(o.end(), s2.begin(), s2.end());
        outs.push_back(o);
      }
      ExpectBitEqual(outs[0], outs[1], std::string(rt::BackendName(backend)) + " gdn step no-spill");
    }, {.test_cases = 30});
  });
}
