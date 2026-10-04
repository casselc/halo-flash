// GPU matvec for every storage format against the CPU oracle's dequantizer
// (model/cpu_ops.cpp), on random blocks: Hegel property, both backends.

#include <hegel/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "core/quant.hpp"
#include "kernels/core_hsaco.hpp"
#include "model/cpu_ops.hpp"
#include "rt/device.hpp"

namespace gs = hegel::generators;
using namespace halo;
using core::GgmlType;

namespace {

struct Format {
  GgmlType type;
  const char* kernel;
  std::size_t block;  // elements per block (1 for plain types)
  std::size_t bytes;  // bytes per block
  std::vector<std::size_t> f16_offsets;  // fp16 scale fields to keep finite
};

const std::vector<Format>& Formats() {
  static const std::vector<Format> f = {
      {GgmlType::kF32, "halo_matvec_f32", 1, 4, {}},
      {GgmlType::kBF16, "halo_matvec_bf16", 1, 2, {}},
      {GgmlType::kF16, "halo_matvec_f16", 1, 2, {}},
      {GgmlType::kQ8_0, "halo_matvec_q8_0", 32, 34, {0}},
      {GgmlType::kQ4_K, "halo_matvec_q4_k", 256, 144, {0, 2}},
      {GgmlType::kQ5_K, "halo_matvec_q5_k", 256, 176, {0, 2}},
      {GgmlType::kQ6_K, "halo_matvec_q6_k", 256, 210, {208}},
      {GgmlType::kQ5_1, "halo_matvec_q5_1", 32, 24, {0, 2}},
      {GgmlType::kIQ4_NL, "halo_matvec_iq4_nl", 32, 18, {0}},
  };
  return f;
}

std::uint16_t FiniteHalf(hegel::TestCase& tc) {
  const auto e = tc.draw(gs::integers<std::uint16_t>({.min_value = 8, .max_value = 16}));
  const auto m = tc.draw(gs::integers<std::uint16_t>({.min_value = 0, .max_value = 1023}));
  const auto s = tc.draw(gs::integers<std::uint16_t>({.min_value = 0, .max_value = 1}));
  return static_cast<std::uint16_t>((s << 15) | (e << 10) | m);
}

}  // namespace

TEST(GpuQuant, MatVecMatchesCpuDequantForEveryFormat) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const auto& f = Formats()[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = Formats().size() - 1}))];
      const std::size_t blocks = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 3}));
      const std::size_t cols = f.block == 1 ? tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 300}))
                                            : blocks * f.block;
      const std::size_t rows = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 20}));
      const std::size_t row_bytes = f.block == 1 ? cols * f.bytes : (cols / f.block) * f.bytes;
      std::vector<std::uint8_t> w(rows * row_bytes);
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      for (auto& b : w) { seed = seed * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(seed >> 24); }
      if (f.type == GgmlType::kF32) {  // keep values finite and modest
        for (std::size_t i = 0; i < rows * cols; ++i) {
          const float v = static_cast<float>(static_cast<std::int32_t>(i * 2654435761u) % 1000) / 250.0f;
          std::memcpy(w.data() + 4 * i, &v, 4);
        }
      } else if (f.type == GgmlType::kBF16 || f.type == GgmlType::kF16) {
        for (std::size_t i = 0; i < rows * cols; ++i) {
          const std::uint16_t h = f.type == GgmlType::kF16 ? FiniteHalf(tc) : static_cast<std::uint16_t>(0x3F00 + (i * 37) % 256);
          std::memcpy(w.data() + 2 * i, &h, 2);
        }
      } else {
        for (std::size_t r = 0; r < rows; ++r)
          for (std::size_t b = 0; b < cols / f.block; ++b)
            for (auto off : f.f16_offsets) {
              const auto h = FiniteHalf(tc);
              std::memcpy(w.data() + r * row_bytes + b * f.bytes + off, &h, 2);
            }
      }
      std::vector<float> x(cols);
      for (std::size_t i = 0; i < cols; ++i) x[i] = std::sin(static_cast<float>(i) * 0.37f);

      model::TensorRef t;
      t.data = w.data();
      t.type = f.type;
      t.cols = cols;
      t.rows = rows;
      std::vector<float> want(rows);
      model::cpu::MatVec(t, 0, x, want);

      auto wb = dev->Alloc(w.size(), &error), xb = dev->Alloc(cols * 4, &error), yb = dev->Alloc(rows * 4, &error);
      ASSERT_TRUE(wb && xb && yb) << error;
      ASSERT_TRUE(dev->Upload(wb, 0, w.data(), w.size(), &error)) << error;
      ASSERT_TRUE(dev->Upload(xb, 0, x.data(), cols * 4, &error)) << error;
      const auto k = dev->Kernel(f.kernel, &error);
      ASSERT_GE(k, 0) << error;
      const rt::Arg args[] = {rt::Arg::Buf(wb), rt::Arg::Buf(xb), rt::Arg::Buf(yb),
                              rt::Arg::Val(static_cast<std::uint32_t>(rows)), rt::Arg::Val(static_cast<std::uint32_t>(cols)),
                              rt::Arg::Val<std::uint64_t>(row_bytes)};
      ASSERT_TRUE(dev->Launch(k, {static_cast<std::uint32_t>((rows + 7) / 8)}, {256}, args, &error)) << error;
      std::vector<float> got(rows);
      ASSERT_TRUE(dev->Download(yb, 0, got.data(), rows * 4, &error)) << error;
      dev->Free(wb);
      dev->Free(xb);
      dev->Free(yb);
      for (std::size_t r = 0; r < rows; ++r) {
        // FP32 sums in a different order: bound by the magnitude sum.
        double mag = 0;
        std::vector<float> row(cols);
        model::cpu::DequantizeRow(t, 0, r, row.data());
        for (std::size_t i = 0; i < cols; ++i) mag += std::fabs(static_cast<double>(row[i]) * x[i]);
        ASSERT_LE(std::fabs(got[r] - want[r]), 1e-5 * mag + 1e-6)
            << core::ToString(f.type) << " row " << r << " got " << got[r] << " want " << want[r];
      }
    }, {.test_cases = 200});
  }
}

// P3 Q8_0 path: repack at load, then the bandwidth GEMV (dense and expert),
// against the CPU dequantizer on random matrices up to the LDS width.
TEST(GpuQuant, RepackedQ8GemvMatchesCpuDequant) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const std::size_t cols = 32 * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 384}));
      const std::size_t rows = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 40}));
      const std::size_t experts = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 3}));
      const std::size_t row_bytes = cols / 32 * 34;
      std::vector<std::uint8_t> w(experts * rows * row_bytes);
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      for (auto& b : w) { seed = seed * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(seed >> 24); }
      for (std::size_t r = 0; r < experts * rows; ++r)
        for (std::size_t b = 0; b < cols / 32; ++b) {
          const auto h = FiniteHalf(tc);
          std::memcpy(w.data() + r * row_bytes + b * 34, &h, 2);
        }
      std::vector<float> x(cols);
      for (std::size_t i = 0; i < cols; ++i) x[i] = std::cos(static_cast<float>(i) * 0.11f);
      const std::uint32_t pick = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = static_cast<std::uint32_t>(experts - 1)}));

      model::TensorRef t;
      t.data = w.data();
      t.type = GgmlType::kQ8_0;
      t.cols = cols;
      t.rows = rows;
      t.experts = experts;
      std::vector<float> want(rows);
      model::cpu::MatVec(t, pick, x, want);

      const auto qstride = static_cast<std::uint32_t>(cols);
      auto raw = dev->Alloc(w.size(), &error), qs = dev->Alloc(experts * rows * qstride, &error),
           d = dev->Alloc(experts * rows * (cols / 32) * 2, &error), xb = dev->Alloc(cols * 4, &error),
           yb = dev->Alloc(rows * 4, &error), ib = dev->Alloc(4, &error);
      ASSERT_TRUE(raw && qs && d && xb && yb && ib) << error;
      ASSERT_TRUE(dev->Upload(raw, 0, w.data(), w.size(), &error)) << error;
      ASSERT_TRUE(dev->Upload(xb, 0, x.data(), cols * 4, &error)) << error;
      ASSERT_TRUE(dev->Upload(ib, 0, &pick, 4, &error)) << error;
      const rt::Arg ra[] = {rt::Arg::Buf(raw), rt::Arg::Buf(qs), rt::Arg::Buf(d),
                            rt::Arg::Val(static_cast<std::uint32_t>(experts * rows)),
                            rt::Arg::Val(static_cast<std::uint32_t>(cols)), rt::Arg::Val(qstride)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_repack_q8_0", &error), {64}, {256}, ra, &error)) << error;
      const bool as_expert = cols <= 8192 && tc.draw(gs::booleans());  // expert kernels hold 8192 columns
      const auto R = static_cast<std::uint32_t>(rows), C = static_cast<std::uint32_t>(cols);
      if (as_expert) {
        const rt::Arg a[] = {rt::Arg::Buf(qs), rt::Arg::Buf(d), rt::Arg::Buf(ib), rt::Arg::Buf(xb), rt::Arg::Buf(yb),
                             rt::Arg::Val(0u), rt::Arg::Val(R), rt::Arg::Val(C), rt::Arg::Val(qstride)};
        const bool r4 = cols <= 6144 && tc.draw(gs::booleans());
        ASSERT_TRUE(dev->Launch(dev->Kernel(r4 ? "halo_gemv_q8r_expert_r4" : "halo_gemv_q8r_expert", &error),
                                {r4 ? (R + 31) / 32 : (R + 7) / 8}, {256}, a, &error))
            << error;
      } else {
        const rt::Arg a[] = {rt::Arg::Buf(qs, std::size_t{pick} * rows * qstride),
                             rt::Arg::Buf(d, std::size_t{pick} * rows * (cols / 32) * 2), rt::Arg::Buf(xb),
                             rt::Arg::Buf(yb), rt::Arg::Val(R), rt::Arg::Val(C), rt::Arg::Val(qstride)};
        struct V { const char* name; std::uint32_t rows_per_block, max_cols; };
        const V variants[] = {{"halo_gemv_q8r", 8, 8192}, {"halo_gemv_q8r_wide", 8, 12288}, {"halo_gemv_q8r_r2", 16, 12288},
                              {"halo_gemv_q8r_r4", 32, 12288}, {"halo_gemv_q8r_r4n", 32, 6144}};
        std::vector<V> ok;
        for (const auto& v : variants)
          if (cols <= v.max_cols) ok.push_back(v);
        const V v = ok[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = ok.size() - 1}))];
        ASSERT_TRUE(dev->Launch(dev->Kernel(v.name, &error), {(R + v.rows_per_block - 1) / v.rows_per_block}, {256}, a,
                                &error))
            << error;
      }
      std::vector<float> got(rows);
      ASSERT_TRUE(dev->Download(yb, 0, got.data(), rows * 4, &error)) << error;
      for (auto* b : {&raw, &qs, &d, &xb, &yb, &ib}) dev->Free(*b);
      for (std::size_t r = 0; r < rows; ++r) {
        double mag = 0;
        std::vector<float> row(cols);
        model::cpu::DequantizeRow(t, pick, r, row.data());
        for (std::size_t i = 0; i < cols; ++i) mag += std::fabs(static_cast<double>(row[i]) * x[i]);
        ASSERT_LE(std::fabs(got[r] - want[r]), 1e-5 * mag + 1e-6)
            << (as_expert ? "expert" : "dense") << " row " << r << " cols " << cols << " got " << got[r] << " want " << want[r];
      }
    }, {.test_cases = 150});
  }
}

// P3 expert formats: the block-wise bandwidth GEMVs (Q4_K, Q5_1), dense and
// expert, against the CPU dequantizer.
TEST(GpuQuant, BlockGemvMatchesCpuDequant) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const bool q4k = tc.draw(gs::booleans());
      const std::size_t block = q4k ? 256 : 32, bytes = q4k ? 144 : 24;
      const std::size_t cols = block * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = q4k ? 32ul : 256ul}));
      const std::size_t rows = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 40}));
      const std::size_t experts = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 3}));
      const std::size_t row_bytes = cols / block * bytes;
      std::vector<std::uint8_t> w(experts * rows * row_bytes);
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      for (auto& b : w) { seed = seed * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(seed >> 24); }
      for (std::size_t r = 0; r < experts * rows; ++r)
        for (std::size_t b = 0; b < cols / block; ++b)
          for (std::size_t off : {0, 2}) {
            const auto h = FiniteHalf(tc);
            std::memcpy(w.data() + r * row_bytes + b * bytes + off, &h, 2);
          }
      std::vector<float> x(cols);
      for (std::size_t i = 0; i < cols; ++i) x[i] = std::sin(static_cast<float>(i) * 0.23f) + 0.1f;
      const auto pick = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = static_cast<std::uint32_t>(experts - 1)}));
      model::TensorRef t;
      t.data = w.data();
      t.type = q4k ? GgmlType::kQ4_K : GgmlType::kQ5_1;
      t.cols = cols;
      t.rows = rows;
      t.experts = experts;
      std::vector<float> want(rows);
      model::cpu::MatVec(t, pick, x, want);
      auto wb = dev->Alloc(w.size(), &error), xb = dev->Alloc(cols * 4, &error), yb = dev->Alloc(rows * 4, &error),
           ib = dev->Alloc(4, &error);
      ASSERT_TRUE(wb && xb && yb && ib) << error;
      ASSERT_TRUE(dev->Upload(wb, 0, w.data(), w.size(), &error)) << error;
      ASSERT_TRUE(dev->Upload(xb, 0, x.data(), cols * 4, &error)) << error;
      ASSERT_TRUE(dev->Upload(ib, 0, &pick, 4, &error)) << error;
      const auto R = static_cast<std::uint32_t>(rows), C = static_cast<std::uint32_t>(cols);
      const std::string name = q4k ? "halo_gemv_q4_k" : "halo_gemv_q5_1";
      const bool as_expert = tc.draw(gs::booleans());
      if (as_expert) {
        const rt::Arg a[] = {rt::Arg::Buf(wb), rt::Arg::Buf(ib), rt::Arg::Buf(xb), rt::Arg::Buf(yb), rt::Arg::Val(0u),
                             rt::Arg::Val<std::uint64_t>(rows * row_bytes), rt::Arg::Val(R), rt::Arg::Val(C),
                             rt::Arg::Val<std::uint64_t>(row_bytes)};
        ASSERT_TRUE(dev->Launch(dev->Kernel(name + "_expert", &error), {(R + 7) / 8}, {256}, a, &error)) << error;
      } else {
        const rt::Arg a[] = {rt::Arg::Buf(wb, std::size_t{pick} * rows * row_bytes), rt::Arg::Buf(xb), rt::Arg::Buf(yb),
                             rt::Arg::Val(R), rt::Arg::Val(C), rt::Arg::Val<std::uint64_t>(row_bytes)};
        ASSERT_TRUE(dev->Launch(dev->Kernel(name, &error), {(R + 7) / 8}, {256}, a, &error)) << error;
      }
      std::vector<float> got(rows);
      ASSERT_TRUE(dev->Download(yb, 0, got.data(), rows * 4, &error)) << error;
      for (auto* b : {&wb, &xb, &yb, &ib}) dev->Free(*b);
      for (std::size_t r = 0; r < rows; ++r) {
        double mag = 0;
        std::vector<float> row(cols);
        model::cpu::DequantizeRow(t, pick, r, row.data());
        for (std::size_t i = 0; i < cols; ++i) mag += std::fabs(static_cast<double>(row[i]) * x[i]);
        ASSERT_LE(std::fabs(got[r] - want[r]), 1e-5 * mag + 1e-6)
            << name << (as_expert ? " expert" : "") << " row " << r << " cols " << cols << " got " << got[r] << " want " << want[r];
      }
    }, {.test_cases = 200});
  }
}

// P3 fused MoE (gate_up, down, combine) against the CPU composition of the
// per-expert matvecs: random stacked experts, ids and weights.
TEST(GpuQuant, FusedMoeMatchesCpuComposition) {
  std::string error;
  const auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const std::size_t H = 256 * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 4}));
      const std::size_t ff = 32 * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 24}));
      const std::size_t E = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 6}));
      const std::size_t used = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = E}));
      const bool q8_down = tc.draw(gs::booleans());
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto fill = [&](std::vector<std::uint8_t>& w, std::size_t rows, std::size_t row_bytes, std::size_t block_bytes,
                            std::initializer_list<std::size_t> halves) {
        for (auto& b : w) { seed = seed * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(seed >> 24); }
        for (std::size_t r = 0; r < rows; ++r)
          for (std::size_t b = 0; b < row_bytes / block_bytes; ++b)
            for (std::size_t off : halves) {
              const auto h = FiniteHalf(tc);
              std::memcpy(w.data() + r * row_bytes + b * block_bytes + off, &h, 2);
            }
      };
      const std::size_t gu_row = H / 256 * 144, dn_row = q8_down ? ff / 32 * 34 : ff / 32 * 24;
      std::vector<std::uint8_t> gate(E * ff * gu_row), up(E * ff * gu_row), down(E * H * dn_row);
      fill(gate, E * ff, gu_row, 144, {0, 2});
      fill(up, E * ff, gu_row, 144, {0, 2});
      if (q8_down) fill(down, E * H, dn_row, 34, {0});
      else fill(down, E * H, dn_row, 24, {0, 2});
      std::vector<std::uint32_t> ids(used);
      std::vector<float> wts(used);
      for (std::size_t s = 0; s < used; ++s) {
        ids[s] = static_cast<std::uint32_t>((s * 7 + seed) % E);
        wts[s] = 0.1f + 0.2f * static_cast<float>(s);
      }
      std::vector<float> x(H);
      for (std::size_t i = 0; i < H; ++i) x[i] = 0.05f * std::sin(static_cast<float>(i) * 0.37f);
      model::TensorRef tg, tu, td;
      tg.data = gate.data(); tg.type = GgmlType::kQ4_K; tg.cols = H; tg.rows = ff; tg.experts = E;
      tu = tg; tu.data = up.data();
      td.data = down.data(); td.type = q8_down ? GgmlType::kQ8_0 : GgmlType::kQ5_1; td.cols = ff; td.rows = H; td.experts = E;
      std::vector<double> want(H, 0.0), mag(H, 0.0);
      std::vector<float> g(ff), u(ff), h(ff), dn(H), row(ff);
      for (std::size_t s = 0; s < used; ++s) {
        model::cpu::MatVec(tg, ids[s], x, g);
        model::cpu::MatVec(tu, ids[s], x, u);
        for (std::size_t j = 0; j < ff; ++j) h[j] = silu(g[j]) * u[j];
        model::cpu::MatVec(td, ids[s], h, dn);
        for (std::size_t r = 0; r < H; ++r) {
          want[r] += wts[s] * dn[r];
          model::cpu::DequantizeRow(td, ids[s], r, row.data());
          for (std::size_t j = 0; j < ff; ++j) mag[r] += std::fabs(wts[s] * row[j] * h[j]);
        }
      }
      auto gb = dev->Alloc(gate.size(), &error), ub = dev->Alloc(up.size(), &error), db = dev->Alloc(down.size(), &error),
           ib = dev->Alloc(used * 4, &error), wb = dev->Alloc(used * 4, &error), xb = dev->Alloc(H * 4, &error),
           hb = dev->Alloc(used * ff * 4, &error), pb = dev->Alloc(used * H * 4, &error), ob = dev->Alloc(H * 4, &error),
           qs = dev->Alloc(E * H * ff, &error), dsc = dev->Alloc(E * H * (ff / 32) * 2, &error);
      ASSERT_TRUE(gb && ub && db && ib && wb && xb && hb && pb && ob && qs && dsc) << error;
      ASSERT_TRUE(dev->Upload(gb, 0, gate.data(), gate.size(), &error) && dev->Upload(ub, 0, up.data(), up.size(), &error) &&
                  dev->Upload(db, 0, down.data(), down.size(), &error) && dev->Upload(ib, 0, ids.data(), used * 4, &error) &&
                  dev->Upload(wb, 0, wts.data(), used * 4, &error) && dev->Upload(xb, 0, x.data(), H * 4, &error))
          << error;
      const auto uH = static_cast<std::uint32_t>(H), uF = static_cast<std::uint32_t>(ff), uU = static_cast<std::uint32_t>(used);
      const rt::Arg a1[] = {rt::Arg::Buf(gb), rt::Arg::Buf(ub), rt::Arg::Buf(ib), rt::Arg::Buf(xb), rt::Arg::Buf(hb),
                            rt::Arg::Val<std::uint64_t>(ff * gu_row), rt::Arg::Val(uF), rt::Arg::Val(uH),
                            rt::Arg::Val<std::uint64_t>(gu_row)};
      struct V { const char* name; std::uint32_t rows_per_block; };
      const V gate_up[] = {{"halo_moe_gate_up_q4_k", 8}, {"halo_moe_gate_up_q4_k_r2", 16},
                           {"halo_moe_gate_up_q4_k_s2", 16}, {"halo_moe_gate_up_q4_k_s4", 32}};
      const V gv = gate_up[tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}))];
      ASSERT_TRUE(dev->Launch(dev->Kernel(gv.name, &error), {(uF + gv.rows_per_block - 1) / gv.rows_per_block, uU}, {256}, a1,
                              &error))
          << error;
      if (q8_down) {
        const rt::Arg r[] = {rt::Arg::Buf(db), rt::Arg::Buf(qs), rt::Arg::Buf(dsc),
                             rt::Arg::Val(static_cast<std::uint32_t>(E * H)), rt::Arg::Val(uF), rt::Arg::Val(uF)};
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_repack_q8_0", &error), {64}, {256}, r, &error)) << error;
        const rt::Arg a2[] = {rt::Arg::Buf(qs), rt::Arg::Buf(dsc), rt::Arg::Buf(ib), rt::Arg::Buf(hb), rt::Arg::Buf(pb),
                              rt::Arg::Val(uH), rt::Arg::Val(uF), rt::Arg::Val(uF)};
        const bool r4 = tc.draw(gs::booleans());
        ASSERT_TRUE(dev->Launch(dev->Kernel(r4 ? "halo_moe_down_q8r_r4" : "halo_moe_down_q8r", &error),
                                {r4 ? (uH + 31) / 32 : (uH + 7) / 8, uU}, {256}, a2, &error))
            << error;
      } else {
        const rt::Arg a2[] = {rt::Arg::Buf(db), rt::Arg::Buf(ib), rt::Arg::Buf(hb), rt::Arg::Buf(pb),
                              rt::Arg::Val<std::uint64_t>(H * dn_row), rt::Arg::Val(uH), rt::Arg::Val(uF),
                              rt::Arg::Val<std::uint64_t>(dn_row)};
        const bool r4 = ff <= 2048 && tc.draw(gs::booleans());
        ASSERT_TRUE(dev->Launch(dev->Kernel(r4 ? "halo_moe_down_q5_1_r4" : "halo_moe_down_q5_1", &error),
                                {r4 ? (uH + 31) / 32 : (uH + 7) / 8, uU}, {256}, a2, &error))
            << error;
      }
      const rt::Arg a3[] = {rt::Arg::Buf(pb), rt::Arg::Buf(wb), rt::Arg::Buf(ob), rt::Arg::Val(uU), rt::Arg::Val(uH)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_moe_combine", &error), {(uH + 255) / 256}, {256}, a3, &error)) << error;
      std::vector<float> got(H);
      ASSERT_TRUE(dev->Download(ob, 0, got.data(), H * 4, &error)) << error;
      for (auto* b : {&gb, &ub, &db, &ib, &wb, &xb, &hb, &pb, &ob, &qs, &dsc}) dev->Free(*b);
      for (std::size_t r = 0; r < H; ++r)
        ASSERT_LE(std::fabs(got[r] - want[r]), 1e-4 * mag[r] + 1e-6)
            << (q8_down ? "q8" : "q5_1") << " down, row " << r << " got " << got[r] << " want " << want[r];
    }, {.test_cases = 60});
  }
}

// P3 single-pass DeltaNet step (d = 128) against the two-loop kernel, which
// the model tests validate end to end: several consecutive steps from a
// random state, outputs and states compared.
TEST(GpuKernels, SinglePassGdnMatchesTwoLoopKernel) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      constexpr std::uint32_t d = 128;
      const auto nk = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
      const std::uint32_t nv = nk * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&](float scale) {
        seed = seed * 1664525u + 1013904223u;
        return scale * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f);
      };
      std::vector<float> state(std::size_t{nv} * d * d), a(nv), dt(nv);
      for (auto& x : state) x = rnd(0.2f);
      for (std::uint32_t h = 0; h < nv; ++h) {
        a[h] = -std::exp(rnd(2.0f));
        dt[h] = rnd(2.0f);
      }
      const auto upload = [&](const std::vector<float>& v) {
        auto b = dev->Alloc(v.size() * 4, &error);
        EXPECT_TRUE(b && dev->Upload(b, 0, v.data(), v.size() * 4, &error)) << error;
        return b;
      };
      auto s1 = upload(state), s2 = upload(state), ab = upload(a), db = upload(dt);
      auto o1 = dev->Alloc(nv * d * 4, &error), o2 = dev->Alloc(nv * d * 4, &error);
      const int steps = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 4}));
      for (int step = 0; step < steps; ++step) {
        std::vector<float> qv(nk * d), kv(nk * d), vv(nv * d), al(nv), be(nv);
        for (auto& x : qv) x = rnd(0.3f);
        for (auto& x : kv) x = rnd(0.3f);
        for (auto& x : vv) x = rnd(2.0f);
        for (auto& x : al) x = rnd(4.0f);
        for (auto& x : be) x = rnd(4.0f);
        auto qb = upload(qv), kb = upload(kv), vb = upload(vv), alb = upload(al), beb = upload(be);
        const rt::Arg a1[] = {rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(vb), rt::Arg::Buf(alb), rt::Arg::Buf(beb),
                              rt::Arg::Buf(ab), rt::Arg::Buf(db), rt::Arg::Buf(s1), rt::Arg::Buf(o1), rt::Arg::Val(nk),
                              rt::Arg::Val(d)};
        const rt::Arg a2[] = {rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(vb), rt::Arg::Buf(alb), rt::Arg::Buf(beb),
                              rt::Arg::Buf(ab), rt::Arg::Buf(db), rt::Arg::Buf(s2), rt::Arg::Buf(o2), rt::Arg::Val(nk)};
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gdn_step", &error), {nv}, {d}, a1, &error)) << error;
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gdn_step_d128", &error), {nv}, {d}, a2, &error)) << error;
        std::vector<float> r1(nv * d), r2(nv * d);
        ASSERT_TRUE(dev->Download(o1, 0, r1.data(), r1.size() * 4, &error)) << error;
        ASSERT_TRUE(dev->Download(o2, 0, r2.data(), r2.size() * 4, &error)) << error;
        for (std::size_t i = 0; i < r1.size(); ++i)
          ASSERT_NEAR(r2[i], r1[i], 1e-5f * (1.0f + std::fabs(r1[i]))) << "output " << i << " step " << step;
        for (auto* b : {&qb, &kb, &vb, &alb, &beb}) dev->Free(*b);
      }
      std::vector<float> f1(state.size()), f2(state.size());
      ASSERT_TRUE(dev->Download(s1, 0, f1.data(), f1.size() * 4, &error)) << error;
      ASSERT_TRUE(dev->Download(s2, 0, f2.data(), f2.size() * 4, &error)) << error;
      for (std::size_t i = 0; i < f1.size(); ++i) ASSERT_NEAR(f2[i], f1[i], 1e-5f * (1.0f + std::fabs(f1[i]))) << "state " << i;
      for (auto* b : {&s1, &s2, &ab, &db, &o1, &o2}) dev->Free(*b);
    }, {.test_cases = 60});
  }
}
