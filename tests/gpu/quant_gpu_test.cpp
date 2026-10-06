// GPU matvec for every storage format against the CPU oracle's dequantizer
// (model/cpu_ops.cpp), on random blocks: Hegel property, both backends.

#include <hegel/gtest.h>

#include <cmath>
#include <cstring>
#include <optional>
#include <tuple>
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

// P3 Q4_K gate+up with int8 activations: within the bound that quantizing x
// (one scale per 32 values, round to nearest) allows, propagated through
// silu(g) * u; exact float CPU matvecs as the reference.
TEST(GpuQuant, Q4kDotGateUpWithinActivationQuantizationBound) {
  std::string error;
  const auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const std::size_t H = 256 * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 16}));
      const std::size_t ff = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 70}));
      const std::size_t E = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 4}));
      const std::size_t used = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = E}));
      const std::size_t row = H / 256 * 144;
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      std::vector<std::uint8_t> gate(E * ff * row), up(E * ff * row);
      // Mostly dmin = 0: then every weight is d * sc * q, the quantization
      // bound below is exact, and errors in the activation scales cannot
      // hide behind the conservative 63 * |dmin| allowance.
      const bool zero_min = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3})) != 0;
      for (auto* w : {&gate, &up}) {
        for (auto& b : *w) { seed = seed * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(seed >> 24); }
        for (std::size_t r = 0; r < E * ff; ++r)
          for (std::size_t b = 0; b < H / 256; ++b)
            for (std::size_t off : {0, 2}) {
              const std::uint16_t h = off == 2 && zero_min ? std::uint16_t{0} : FiniteHalf(tc);
              std::memcpy(w->data() + r * row + b * 144 + off, &h, 2);
            }
      }
      std::vector<float> x(H);
      const float xs = tc.draw(gs::sampled_from(std::vector<float>{1e-3f, 0.05f, 1.0f, 30.0f}));
      // Amplitude varies block to block, so per-32 scales differ.
      for (std::size_t i = 0; i < H; ++i)
        x[i] = xs * static_cast<float>(1 + (i / 32 * 7 + seed) % 9) * std::sin(static_cast<float>(i) * 0.71f + static_cast<float>(seed % 7));
      std::vector<std::uint32_t> ids(used);
      for (std::size_t s = 0; s < used; ++s) ids[s] = static_cast<std::uint32_t>((s + seed) % E);
      // Per-32 quantization error bound of x: scale / 2 per value.
      std::vector<double> qerr(H / 32);
      for (std::size_t b = 0; b < H / 32; ++b) {
        float amax = 0;
        for (std::size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        qerr[b] = amax / 127.0 / 2.0 * 1.0001 + 1e-30;
      }
      auto gb = dev->Alloc(gate.size(), &error), ub = dev->Alloc(up.size(), &error), ib = dev->Alloc(used * 4, &error),
           xb = dev->Alloc(H * 4, &error), hb = dev->Alloc(used * ff * 4, &error);
      ASSERT_TRUE(gb && ub && ib && xb && hb) << error;
      ASSERT_TRUE(dev->Upload(gb, 0, gate.data(), gate.size(), &error) && dev->Upload(ub, 0, up.data(), up.size(), &error) &&
                  dev->Upload(ib, 0, ids.data(), used * 4, &error) && dev->Upload(xb, 0, x.data(), H * 4, &error))
          << error;
      const auto uF = static_cast<std::uint32_t>(ff), uH = static_cast<std::uint32_t>(H);
      const rt::Arg a[] = {rt::Arg::Buf(gb), rt::Arg::Buf(ub), rt::Arg::Buf(ib), rt::Arg::Buf(xb), rt::Arg::Buf(hb),
                           rt::Arg::Val<std::uint64_t>(ff * row), rt::Arg::Val(uF), rt::Arg::Val(uH),
                           rt::Arg::Val<std::uint64_t>(row)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_moe_gate_up_q4_k_dot", &error),
                              {(uF + 15) / 16, static_cast<std::uint32_t>(used)}, {256}, a, &error))
          << error;
      std::vector<float> got(used * ff);
      ASSERT_TRUE(dev->Download(hb, 0, got.data(), got.size() * 4, &error)) << error;
      for (auto* b : {&gb, &ub, &ib, &xb, &hb}) dev->Free(*b);
      model::TensorRef tg;
      tg.data = gate.data(); tg.type = GgmlType::kQ4_K; tg.cols = H; tg.rows = ff; tg.experts = E;
      model::TensorRef tu = tg;
      tu.data = up.data();
      std::vector<float> wrow(H);
      for (std::size_t s = 0; s < used; ++s)
        for (std::size_t r = 0; r < ff; ++r) {
          double gv = 0, uv = 0, eg = 0, eu = 0;
          for (auto [t, val, err] : {std::tuple{&tg, &gv, &eg}, std::tuple{&tu, &uv, &eu}}) {
            model::cpu::DequantizeRow(*t, ids[s], r, wrow.data());
            const auto* blk = static_cast<const std::uint8_t*>(t->data) + (ids[s] * ff + r) * row;
            for (std::size_t i = 0; i < H; ++i) {
              *val += static_cast<double>(wrow[i]) * x[i];
              // |d*sc*q| <= |w| + |dmin| * 63 (the per-value scale part only).
              std::uint16_t dm;
              std::memcpy(&dm, blk + (i / 256) * 144 + 2, 2);
              const double dmin = std::fabs(static_cast<double>(static_cast<_Float16>(*reinterpret_cast<_Float16*>(&dm))));
              *err += (std::fabs(wrow[i]) + 63.0 * dmin) * qerr[i / 32];
            }
          }
          const double want = silu(gv) * uv;
          const double bound = 1.2 * (std::fabs(uv) * eg + (std::fabs(silu(gv)) + 1.2 * eg) * eu) +
                               1e-5 * (std::fabs(want) + 1e-6);
          ASSERT_LE(std::fabs(got[s * ff + r] - want), bound)
              << "slot " << s << " row " << r << " H " << H << " got " << got[s * ff + r] << " want " << want;
        }
    }, {.test_cases = 60});
  }
}

// P3 fused hyper-connection mixer (halo_hc_mix_a + _b) against the six-kernel
// sequence it replaces (grouped RMSNorm, Q8 down, scale_silu, Q8 up, hc_mix,
// F32 inject): mixed, xn and inject agree.
TEST(GpuKernels, FusedHcMixerMatchesKernelSequence) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t hc = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
      const std::uint32_t H = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 80}));
      const std::uint32_t L = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
      const bool with_inject = tc.draw(gs::booleans());
      const std::uint32_t D = hc * H;
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&](float scale) {
        seed = seed * 1664525u + 1013904223u;
        return scale * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f);
      };
      const auto q8 = [&](std::uint32_t rows, std::uint32_t cols) {
        std::vector<std::uint8_t> w(std::size_t{rows} * (cols / 32) * 34);
        for (auto& b : w) b = static_cast<std::uint8_t>(rnd(256.0f) + 128);
        for (std::size_t b = 0; b < w.size(); b += 34) {
          const std::uint16_t h = 0x1C00 | (static_cast<std::uint16_t>(seed >> 9) & 0x3FF);  // ~0.004-0.008
          std::memcpy(w.data() + b, &h, 2);
        }
        return w;
      };
      std::vector<float> res(D), norm(D), inj(std::size_t{hc} * D);
      for (auto& v : res) v = rnd(4.0f);
      for (auto& v : norm) v = 1.0f + rnd(0.5f);
      for (auto& v : inj) v = rnd(0.1f);
      const auto down = q8(L, D), up = q8(D, L);
      const auto upload = [&](const void* p, std::size_t bytes) {
        auto b = dev->Alloc(bytes, &error);
        EXPECT_TRUE(b && dev->Upload(b, 0, p, bytes, &error)) << error;
        return b;
      };
      auto rb = upload(res.data(), D * 4ull), nb = upload(norm.data(), D * 4ull),
           ib = upload(inj.data(), inj.size() * 4), draw = upload(down.data(), down.size()),
           uraw = upload(up.data(), up.size());
      auto dqs = dev->Alloc(std::size_t{L} * D, &error), dd = dev->Alloc(std::size_t{L} * (D / 32) * 2, &error),
           uqs = dev->Alloc(std::size_t{D} * L, &error), ud = dev->Alloc(std::size_t{D} * (L / 32) * 2, &error);
      const auto repack = [&](const rt::Buffer& raw, const rt::Buffer& qs, const rt::Buffer& d, std::uint32_t rows,
                              std::uint32_t cols) {
        const rt::Arg a[] = {rt::Arg::Buf(raw), rt::Arg::Buf(qs), rt::Arg::Buf(d), rt::Arg::Val(rows), rt::Arg::Val(cols),
                             rt::Arg::Val(cols)};
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_repack_q8_0", &error), {256}, {256}, a, &error)) << error;
      };
      repack(draw, dqs, dd, L, D);
      repack(uraw, uqs, ud, D, L);
      // Reference sequence.
      auto xn1 = dev->Alloc(D * 4ull, &error), lo1 = dev->Alloc(L * 4ull, &error), g1 = dev->Alloc(D * 4ull, &error),
           m1 = dev->Alloc(H * 4ull, &error), i1 = dev->Alloc(16 * 4, &error);
      const float eps = 1e-6f;
      const auto launch = [&](const char* k, rt::Dim3 grid, rt::Dim3 block, std::initializer_list<rt::Arg> args) {
        ASSERT_TRUE(dev->Launch(dev->Kernel(k, &error), grid, block, std::span<const rt::Arg>(args.begin(), args.size()),
                                &error))
            << k << ": " << error;
      };
      launch("halo_rmsnorm", {hc}, {256}, {rt::Arg::Buf(rb), rt::Arg::Buf(nb), rt::Arg::Buf(xn1), rt::Arg::Val(H),
                                            rt::Arg::Val(eps), rt::Arg::Val(H)});
      launch("halo_gemv_q8r_r2", {(L + 15) / 16}, {256}, {rt::Arg::Buf(dqs), rt::Arg::Buf(dd), rt::Arg::Buf(xn1),
                                                          rt::Arg::Buf(lo1), rt::Arg::Val(L), rt::Arg::Val(D), rt::Arg::Val(D)});
      launch("halo_scale_silu", {(L + 255) / 256}, {256},
             {rt::Arg::Buf(lo1), rt::Arg::Val(1.0f / static_cast<float>(hc)), rt::Arg::Val(L)});
      launch("halo_gemv_q8r_r2", {(D + 15) / 16}, {256}, {rt::Arg::Buf(uqs), rt::Arg::Buf(ud), rt::Arg::Buf(lo1),
                                                          rt::Arg::Buf(g1), rt::Arg::Val(D), rt::Arg::Val(L), rt::Arg::Val(L)});
      launch("halo_hc_mix", {(H + 255) / 256}, {256}, {rt::Arg::Buf(xn1), rt::Arg::Buf(g1), rt::Arg::Buf(m1),
                                                       rt::Arg::Val(hc), rt::Arg::Val(H)});
      if (with_inject)
        launch("halo_matvec_f32_rowblock", {hc}, {256}, {rt::Arg::Buf(ib), rt::Arg::Buf(xn1), rt::Arg::Buf(i1),
                                                         rt::Arg::Val(hc), rt::Arg::Val(D)});
      // Fused, split over a random number of column chunks.
      const std::uint32_t want_parts = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
      const std::uint32_t chunk = (D / want_parts + 15) / 16 * 16, parts = (D + chunk - 1) / chunk;
      auto xn2 = dev->Alloc(D * 4ull, &error), lop = dev->Alloc(std::size_t{parts} * L * 4, &error),
           m2 = dev->Alloc(H * 4ull, &error), ip = dev->Alloc(std::size_t{parts} * hc * 4, &error),
           i2 = dev->Alloc(16 * 4, &error);
      const std::uint32_t down_blocks = (L + 15) / 16;
      auto sc = dev->Alloc(16 * 4, &error);
      launch("halo_hc_norm_scales", {hc}, {256}, {rt::Arg::Buf(rb), rt::Arg::Buf(sc), rt::Arg::Val(H), rt::Arg::Val(eps)});
      launch("halo_hc_mix_a", {down_blocks + (with_inject ? hc : 0), parts}, {256},
             {rt::Arg::Buf(rb), rt::Arg::Buf(nb), rt::Arg::Buf(sc), rt::Arg::Buf(dqs), rt::Arg::Buf(dd), rt::Arg::Buf(ib),
              rt::Arg::Buf(xn2), rt::Arg::Buf(lop), rt::Arg::Buf(ip), rt::Arg::Val(hc), rt::Arg::Val(H), rt::Arg::Val(L),
              rt::Arg::Val(down_blocks), rt::Arg::Val(D), rt::Arg::Val(chunk)});
      launch("halo_hc_mix_b", {(H + 7) / 8}, {256}, {rt::Arg::Buf(lop), rt::Arg::Val(parts), rt::Arg::Buf(ip),
                                                     rt::Arg::Buf(i2), rt::Arg::Val(with_inject ? hc : 0u),
                                                     rt::Arg::Buf(uqs), rt::Arg::Buf(ud), rt::Arg::Buf(xn2),
                                                     rt::Arg::Buf(m2), rt::Arg::Val(hc), rt::Arg::Val(H), rt::Arg::Val(L),
                                                     rt::Arg::Val(L)});
      auto lo2 = lop;  // freed below
      const auto get = [&](const rt::Buffer& b, std::size_t n) {
        std::vector<float> v(n);
        EXPECT_TRUE(dev->Download(b, 0, v.data(), n * 4, &error)) << error;
        return v;
      };
      const auto expect_close = [&](const std::vector<float>& a, const std::vector<float>& b, const char* what) {
        double num = 0, den = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
          num += double(a[i] - b[i]) * (a[i] - b[i]);
          den += double(b[i]) * b[i];
        }
        ASSERT_LE(std::sqrt(num / std::max(den, 1e-30)), 1e-5) << what << " (hc " << hc << ", H " << H << ", L " << L << ")";
      };
      expect_close(get(xn2, D), get(xn1, D), "xn");
      expect_close(get(m2, H), get(m1, H), "mixed");
      if (with_inject) {
        // Inject logits are single dot products of length hc * H, which can
        // cancel to a small result (with hc = 1, a vector of one); bound the
        // two summation orders by the terms' magnitude instead:
        // |a - b| <= 1e-5 * sum_j |w_ij x_j| (found by Hegel: 1.02e-5 relative
        // at hc 1, H 1888).
        const auto got = get(i2, hc), want = get(i1, hc), xn = get(xn1, D);
        for (std::uint32_t r = 0; r < hc; ++r) {
          double mag = 0;
          for (std::uint32_t j = 0; j < D; ++j) mag += std::fabs(double(inj[std::size_t{r} * D + j]) * xn[j]);
          ASSERT_LE(std::fabs(double(got[r]) - want[r]), 1e-5 * mag + 1e-30)
              << "inject " << r << " (hc " << hc << ", H " << H << ", L " << L << ")";
        }
      }
      for (auto* b : {&rb, &nb, &ib, &draw, &uraw, &dqs, &dd, &uqs, &ud, &xn1, &lo1, &g1, &m1, &i1, &xn2, &lo2, &m2, &i2, &ip, &sc})
        dev->Free(*b);
    }, {.test_cases = 80});
  }
}

// P3 fused shared expert (halo_shexp_gate_up + halo_shexp_down_add) against
// the six-kernel sequence it replaces, accumulating onto an existing output.
TEST(GpuKernels, FusedSharedExpertMatchesKernelSequence) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t H = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 100}));
      const std::uint32_t F = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 30}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&](float scale) {
        seed = seed * 1664525u + 1013904223u;
        return scale * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f);
      };
      const auto q8 = [&](std::uint32_t rows, std::uint32_t cols) {
        std::vector<std::uint8_t> w(std::size_t{rows} * (cols / 32) * 34);
        for (auto& b : w) b = static_cast<std::uint8_t>(rnd(256.0f) + 128);
        for (std::size_t b = 0; b < w.size(); b += 34) {
          const std::uint16_t h = 0x1C00 | (static_cast<std::uint16_t>(seed >> 9) & 0x3FF);
          std::memcpy(w.data() + b, &h, 2);
        }
        return w;
      };
      std::vector<float> x(H), gi(H), out0(H);
      for (auto& v : x) v = rnd(4.0f);
      for (auto& v : gi) v = rnd(0.2f);
      for (auto& v : out0) v = rnd(1.0f);
      const auto gw = q8(F, H), uw = q8(F, H), dw = q8(H, F);
      const auto upload = [&](const void* p, std::size_t bytes) {
        auto b = dev->Alloc(bytes, &error);
        EXPECT_TRUE(b && dev->Upload(b, 0, p, bytes, &error)) << error;
        return b;
      };
      auto xb = upload(x.data(), H * 4ull), gib = upload(gi.data(), H * 4ull), o1 = upload(out0.data(), H * 4ull),
           o2 = upload(out0.data(), H * 4ull);
      rt::Buffer raw[3], qs[3], dd[3];
      const std::uint32_t rows[3] = {F, F, H}, cols[3] = {H, H, F};
      const std::vector<std::uint8_t>* src[3] = {&gw, &uw, &dw};
      for (int i = 0; i < 3; ++i) {
        raw[i] = upload(src[i]->data(), src[i]->size());
        qs[i] = dev->Alloc(std::size_t{rows[i]} * cols[i], &error);
        dd[i] = dev->Alloc(std::size_t{rows[i]} * (cols[i] / 32) * 2, &error);
        const rt::Arg a[] = {rt::Arg::Buf(raw[i]), rt::Arg::Buf(qs[i]), rt::Arg::Buf(dd[i]), rt::Arg::Val(rows[i]),
                             rt::Arg::Val(cols[i]), rt::Arg::Val(cols[i])};
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_repack_q8_0", &error), {256}, {256}, a, &error)) << error;
      }
      const auto launch = [&](const char* k, rt::Dim3 grid, std::initializer_list<rt::Arg> args) {
        ASSERT_TRUE(dev->Launch(dev->Kernel(k, &error), grid, {256}, std::span<const rt::Arg>(args.begin(), args.size()),
                                &error))
            << k << ": " << error;
      };
      auto g = dev->Alloc(F * 4ull, &error), u = dev->Alloc(F * 4ull, &error), y = dev->Alloc(H * 4ull, &error),
           s1 = dev->Alloc(4, &error), h2 = dev->Alloc(F * 4ull, &error), s2 = dev->Alloc(4, &error);
      launch("halo_gemv_q8r_r2", {(F + 15) / 16}, {rt::Arg::Buf(qs[0]), rt::Arg::Buf(dd[0]), rt::Arg::Buf(xb), rt::Arg::Buf(g),
                                                   rt::Arg::Val(F), rt::Arg::Val(H), rt::Arg::Val(H)});
      launch("halo_gemv_q8r_r2", {(F + 15) / 16}, {rt::Arg::Buf(qs[1]), rt::Arg::Buf(dd[1]), rt::Arg::Buf(xb), rt::Arg::Buf(u),
                                                   rt::Arg::Val(F), rt::Arg::Val(H), rt::Arg::Val(H)});
      launch("halo_silu_mul", {(F + 255) / 256}, {rt::Arg::Buf(g), rt::Arg::Buf(u), rt::Arg::Val(F)});
      launch("halo_gemv_q8r_r2", {(H + 15) / 16}, {rt::Arg::Buf(qs[2]), rt::Arg::Buf(dd[2]), rt::Arg::Buf(g), rt::Arg::Buf(y),
                                                   rt::Arg::Val(H), rt::Arg::Val(F), rt::Arg::Val(F)});
      launch("halo_matvec_f32_rowblock", {1}, {rt::Arg::Buf(gib), rt::Arg::Buf(xb), rt::Arg::Buf(s1), rt::Arg::Val(1u),
                                               rt::Arg::Val(H)});
      launch("halo_axpy_sigmoid", {(H + 255) / 256}, {rt::Arg::Buf(o1), rt::Arg::Buf(y), rt::Arg::Buf(s1), rt::Arg::Val(H)});
      launch("halo_shexp_gate_up", {(F + 15) / 16}, {rt::Arg::Buf(qs[0]), rt::Arg::Buf(dd[0]), rt::Arg::Buf(qs[1]),
                                                     rt::Arg::Buf(dd[1]), rt::Arg::Buf(gib), rt::Arg::Buf(xb), rt::Arg::Buf(h2),
                                                     rt::Arg::Buf(s2), rt::Arg::Val(F), rt::Arg::Val(H), rt::Arg::Val(H)});
      launch("halo_shexp_down_add", {(H + 31) / 32}, {rt::Arg::Buf(qs[2]), rt::Arg::Buf(dd[2]), rt::Arg::Buf(h2),
                                                      rt::Arg::Buf(s2), rt::Arg::Buf(o2), rt::Arg::Val(H), rt::Arg::Val(F),
                                                      rt::Arg::Val(F)});
      std::vector<float> a(H), b(H);
      ASSERT_TRUE(dev->Download(o1, 0, a.data(), H * 4ull, &error) && dev->Download(o2, 0, b.data(), H * 4ull, &error)) << error;
      double num = 0, den = 0;
      for (std::uint32_t i = 0; i < H; ++i) {
        num += double(a[i] - b[i]) * (a[i] - b[i]);
        den += double(a[i]) * a[i];
      }
      ASSERT_LE(std::sqrt(num / den), 1e-5) << "H " << H << " F " << F;
      for (auto* p : {&xb, &gib, &o1, &o2, &g, &u, &y, &s1, &h2, &s2}) dev->Free(*p);
      for (int i = 0; i < 3; ++i) {
        dev->Free(raw[i]);
        dev->Free(qs[i]);
        dev->Free(dd[i]);
      }
    }, {.test_cases = 80});
  }
}

// P3 fused DeltaNet block pieces against the kernels they replace: dual Q8
// GEMV vs two GEMVs, alpha+beta vs two row-block matvecs, conv+L2 vs
// conv_step then l2norm, DeltaNet+gated norm vs gdn_step_d128 then gated_norm.
TEST(GpuKernels, FusedDeltaNetPiecesMatchKernelSequences) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      constexpr std::uint32_t d = 128;
      const std::uint32_t nk = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
      const std::uint32_t nv = nk * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3}));
      const std::uint32_t H = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 80}));
      const std::uint32_t kd = nk * d, ch = 2 * kd + nv * d, kern = 4;
      const std::uint32_t mode = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 1}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&](float scale) {
        seed = seed * 1664525u + 1013904223u;
        return scale * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f);
      };
      const auto vec = [&](std::size_t n, float scale) {
        std::vector<float> v(n);
        for (auto& x : v) x = rnd(scale);
        return v;
      };
      const auto upload = [&](const void* p, std::size_t bytes) {
        auto b = dev->Alloc(std::max<std::size_t>(bytes, 4), &error);
        EXPECT_TRUE(b && dev->Upload(b, 0, p, bytes, &error)) << error;
        return b;
      };
      const auto get = [&](const rt::Buffer& b, std::size_t n) {
        std::vector<float> v(n);
        EXPECT_TRUE(dev->Download(b, 0, v.data(), n * 4, &error)) << error;
        return v;
      };
      const auto close = [&](const std::vector<float>& a, const std::vector<float>& b, const char* what) {
        double num = 0, den = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
          num += double(a[i] - b[i]) * (a[i] - b[i]);
          den += double(a[i]) * a[i];
        }
        ASSERT_LE(std::sqrt(num / std::max(den, 1e-30)), 1e-5) << what << " (nk " << nk << ", nv " << nv << ", H " << H << ")";
      };
      const auto launch = [&](const char* k, rt::Dim3 grid, rt::Dim3 block, std::initializer_list<rt::Arg> args) {
        ASSERT_TRUE(dev->Launch(dev->Kernel(k, &error), grid, block, std::span<const rt::Arg>(args.begin(), args.size()),
                                &error))
            << k << ": " << error;
      };
      std::vector<rt::Buffer> owned;
      const auto own = [&](rt::Buffer b) { owned.push_back(b); return b; };
      // Dual GEMV.
      const auto x = vec(H, 4.0f);
      auto xb = own(upload(x.data(), H * 4ull));
      const std::uint32_t r1 = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 20})) + 3,
                          r2 = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 20})) + 7;
      rt::Buffer mq[2], md[2], yref[2], yfused[2];
      for (int m = 0; m < 2; ++m) {
        const std::uint32_t rows = m == 0 ? r1 : r2;
        std::vector<std::uint8_t> raw(std::size_t{rows} * (H / 32) * 34);
        for (auto& b : raw) b = static_cast<std::uint8_t>(rnd(256.0f) + 128);
        for (std::size_t b = 0; b < raw.size(); b += 34) {
          const std::uint16_t h = 0x1C00 | (static_cast<std::uint16_t>(seed >> 9) & 0x3FF);
          std::memcpy(raw.data() + b, &h, 2);
        }
        auto rb = own(upload(raw.data(), raw.size()));
        mq[m] = own(dev->Alloc(std::size_t{rows} * H, &error));
        md[m] = own(dev->Alloc(std::size_t{rows} * (H / 32) * 2, &error));
        yref[m] = own(dev->Alloc(rows * 4ull, &error));
        yfused[m] = own(dev->Alloc(rows * 4ull, &error));
        launch("halo_repack_q8_0", {256}, {256}, {rt::Arg::Buf(rb), rt::Arg::Buf(mq[m]), rt::Arg::Buf(md[m]),
                                                  rt::Arg::Val(rows), rt::Arg::Val(H), rt::Arg::Val(H)});
        launch("halo_gemv_q8r_r4", {(rows + 31) / 32}, {256}, {rt::Arg::Buf(mq[m]), rt::Arg::Buf(md[m]), rt::Arg::Buf(xb),
                                                               rt::Arg::Buf(yref[m]), rt::Arg::Val(rows), rt::Arg::Val(H),
                                                               rt::Arg::Val(H)});
      }
      const std::uint32_t b1 = (r1 + 31) / 32;
      launch("halo_gemv_q8r_dual", {b1 + (r2 + 31) / 32}, {256},
             {rt::Arg::Buf(mq[0]), rt::Arg::Buf(md[0]), rt::Arg::Buf(yfused[0]), rt::Arg::Val(r1), rt::Arg::Val(H),
              rt::Arg::Buf(mq[1]), rt::Arg::Buf(md[1]), rt::Arg::Buf(yfused[1]), rt::Arg::Val(r2), rt::Arg::Val(H),
              rt::Arg::Buf(xb), rt::Arg::Val(H), rt::Arg::Val(b1)});
      close(get(yfused[0], r1), get(yref[0], r1), "dual gemv (first)");
      close(get(yfused[1], r2), get(yref[1], r2), "dual gemv (second)");
      // alpha + beta.
      const auto wa = vec(std::size_t{nv} * H, 0.2f), wb = vec(std::size_t{nv} * H, 0.2f);
      auto wab = own(upload(wa.data(), wa.size() * 4)), wbb = own(upload(wb.data(), wb.size() * 4));
      auto ya1 = own(dev->Alloc(nv * 4, &error)), yb1 = own(dev->Alloc(nv * 4, &error)),
           ya2 = own(dev->Alloc(nv * 4, &error)), yb2 = own(dev->Alloc(nv * 4, &error));
      launch("halo_matvec_f32_rowblock", {nv}, {256}, {rt::Arg::Buf(wab), rt::Arg::Buf(xb), rt::Arg::Buf(ya1), rt::Arg::Val(nv), rt::Arg::Val(H)});
      launch("halo_matvec_f32_rowblock", {nv}, {256}, {rt::Arg::Buf(wbb), rt::Arg::Buf(xb), rt::Arg::Buf(yb1), rt::Arg::Val(nv), rt::Arg::Val(H)});
      launch("halo_matvec_f32_rowblock2", {2 * nv}, {256}, {rt::Arg::Buf(wab), rt::Arg::Buf(wbb), rt::Arg::Buf(xb),
                                                            rt::Arg::Buf(ya2), rt::Arg::Buf(yb2), rt::Arg::Val(nv), rt::Arg::Val(H)});
      close(get(ya2, nv), get(ya1, nv), "alpha");
      close(get(yb2, nv), get(yb1, nv), "beta");
      // conv + L2.
      const auto in = vec(ch, 3.0f), st = vec(std::size_t{kern - 1} * ch, 1.0f), cw = vec(std::size_t{ch} * kern, 0.5f);
      auto inb = own(upload(in.data(), ch * 4ull)), s1 = own(upload(st.data(), st.size() * 4)),
           s2 = own(upload(st.data(), st.size() * 4)), cwb = own(upload(cw.data(), cw.size() * 4));
      auto c1 = own(dev->Alloc(ch * 4ull, &error)), c2 = own(dev->Alloc(ch * 4ull, &error));
      launch("halo_conv_step", {(ch + 255) / 256}, {256}, {rt::Arg::Buf(inb), rt::Arg::Buf(s1), rt::Arg::Buf(cwb),
                                                           rt::Arg::Buf(c1), rt::Arg::Val(ch), rt::Arg::Val(kern)});
      launch("halo_l2norm", {2 * nk}, {128}, {rt::Arg::Buf(c1), rt::Arg::Val(d), rt::Arg::Val(1e-6f)});
      launch("halo_conv_l2norm", {ch / 128}, {128}, {rt::Arg::Buf(inb), rt::Arg::Buf(s2), rt::Arg::Buf(cwb), rt::Arg::Buf(c2),
                                                     rt::Arg::Val(ch), rt::Arg::Val(kern), rt::Arg::Val(2 * kd),
                                                     rt::Arg::Val(1e-6f)});
      close(get(c2, ch), get(c1, ch), "conv+l2 output");
      close(get(s2, st.size()), get(s1, st.size()), "conv state");
      // DeltaNet + gated norm.
      auto a = vec(nv, 2.0f), dt = vec(nv, 2.0f);
      for (auto& v : a) v = -std::exp(v);
      const auto al = vec(nv, 4.0f), be = vec(nv, 4.0f), state = vec(std::size_t{nv} * d * d, 0.2f), nw = vec(d, 1.0f),
                 zv = vec(std::size_t{nv} * d, 4.0f);
      auto ab = own(upload(a.data(), nv * 4ull)), dtb = own(upload(dt.data(), nv * 4ull)),
           alb = own(upload(al.data(), nv * 4ull)), beb = own(upload(be.data(), nv * 4ull)),
           g1 = own(upload(state.data(), state.size() * 4)), g2 = own(upload(state.data(), state.size() * 4)),
           nwb = own(upload(nw.data(), d * 4ull)), zb = own(upload(zv.data(), zv.size() * 4));
      auto o1 = own(dev->Alloc(nv * d * 4ull, &error)), o2 = own(dev->Alloc(nv * d * 4ull, &error));
      const auto qkv = [&](std::size_t off) { return rt::Arg::Buf(c1, off * 4); };
      launch("halo_gdn_step_d128", {nv}, {d}, {qkv(0), qkv(kd), qkv(2 * kd), rt::Arg::Buf(alb), rt::Arg::Buf(beb),
                                               rt::Arg::Buf(ab), rt::Arg::Buf(dtb), rt::Arg::Buf(g1), rt::Arg::Buf(o1),
                                               rt::Arg::Val(nk)});
      launch("halo_gated_norm", {nv}, {128}, {rt::Arg::Buf(o1), rt::Arg::Buf(nwb), rt::Arg::Buf(zb), rt::Arg::Val(d),
                                              rt::Arg::Val(1e-6f), rt::Arg::Val(mode)});
      launch("halo_gdn_step_d128_norm", {nv}, {d}, {qkv(0), qkv(kd), qkv(2 * kd), rt::Arg::Buf(alb), rt::Arg::Buf(beb),
                                                    rt::Arg::Buf(ab), rt::Arg::Buf(dtb), rt::Arg::Buf(g2), rt::Arg::Buf(nwb),
                                                    rt::Arg::Buf(zb), rt::Arg::Buf(o2), rt::Arg::Val(nk),
                                                    rt::Arg::Val(1e-6f), rt::Arg::Val(mode)});
      close(get(o2, nv * d), get(o1, nv * d), "gdn + gated norm output");
      close(get(g2, state.size()), get(g1, state.size()), "gdn state");
      for (auto& b : owned) dev->Free(b);
    }, {.test_cases = 60});
  }
}

// The WMMA operand and result layout the P4 GEMMs assume (see core.hip):
// C = A * B^T for 16x16 int8 A (M x K) and B (N x K).
TEST(GpuKernels, WmmaInt8LayoutProbe) {
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << error;
  ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
  std::vector<std::int8_t> a(256), b(256);
  for (int i = 0; i < 256; ++i) {
    a[i] = static_cast<std::int8_t>((i * 37 + 11) % 255 - 127);
    b[i] = static_cast<std::int8_t>((i * 91 + 5) % 255 - 127);
  }
  auto ab = dev->Alloc(256, &error), bb = dev->Alloc(256, &error), cb = dev->Alloc(1024, &error);
  ASSERT_TRUE(dev->Upload(ab, 0, a.data(), 256, &error) && dev->Upload(bb, 0, b.data(), 256, &error));
  const rt::Arg args[] = {rt::Arg::Buf(ab), rt::Arg::Buf(bb), rt::Arg::Buf(cb)};
  ASSERT_TRUE(dev->Launch(dev->Kernel("halo_wmma_probe_i8", &error), {1}, {32}, args, &error)) << error;
  std::vector<int> c(256);
  ASSERT_TRUE(dev->Download(cb, 0, c.data(), 1024, &error)) << error;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      int want = 0;
      for (int k = 0; k < 16; ++k) want += a[m * 16 + k] * b[n * 16 + k];
      ASSERT_EQ(c[m * 16 + n], want) << "C[" << m << "][" << n << "]";
    }
  for (auto* x : {&ab, &bb, &cb}) dev->Free(*x);
}

// P4 prefill GEMM: int8 WMMA over repacked Q8_0 with activations quantized
// per token and 32 values. Reference: the same quantization on the CPU, the
// integer block sums exact, scales applied in float; ragged shapes (rows and
// tokens not multiples of the 16 x 64 wave tile).
TEST(GpuKernels, WmmaQ8GemmMatchesQuantizedReference) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t cols = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 96}));
      const std::uint32_t rows = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 300}));
      const std::uint32_t T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 300}));
      const std::size_t row_bytes = cols / 32 * 34;
      std::vector<std::uint8_t> w(rows * row_bytes);
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      for (auto& b : w) { seed = seed * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(seed >> 24); }
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t b = 0; b < cols / 32; ++b) {
          const auto h = FiniteHalf(tc);
          std::memcpy(w.data() + r * row_bytes + b * 34, &h, 2);
        }
      std::vector<float> x(static_cast<std::size_t>(T) * cols);
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{1e-3f, 1.0f, 50.0f}));
      for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = amp * std::sin(static_cast<float>(i) * 0.37f + static_cast<float>(seed % 13)) * static_cast<float>(1 + i / 32 % 5);
      if (tc.draw(gs::booleans())) std::fill(x.begin(), x.begin() + std::min<std::size_t>(32, x.size()), 0.0f);  // a zero block
      auto wb = dev->Alloc(w.size(), &error), qs = dev->Alloc(static_cast<std::size_t>(rows) * cols, &error),
           db = dev->Alloc(static_cast<std::size_t>(rows) * cols / 16, &error), xb = dev->Alloc(x.size() * 4, &error),
           xq = dev->Alloc(x.size(), &error), xd = dev->Alloc(x.size() / 8, &error),
           yb = dev->Alloc(static_cast<std::size_t>(T) * rows * 4, &error);
      ASSERT_TRUE(dev->Upload(wb, 0, w.data(), w.size(), &error) && dev->Upload(xb, 0, x.data(), x.size() * 4, &error))
          << error;
      const rt::Arg ra[] = {rt::Arg::Buf(wb), rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Val(rows), rt::Arg::Val(cols),
                            rt::Arg::Val(cols)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_repack_q8_0", &error), {4096}, {256}, ra, &error)) << error;
      const rt::Arg qa[] = {rt::Arg::Buf(xb), rt::Arg::Buf(xq), rt::Arg::Buf(xd), rt::Arg::Val(cols)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_quantize_q8_rows", &error), {T}, {256}, qa, &error)) << error;
      const rt::Arg ga[] = {rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Buf(xq), rt::Arg::Buf(xd), rt::Arg::Buf(yb),
                            rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val(cols), rt::Arg::Val(T)};
      // The simple kernel or the tiled production kernel.
      const bool v2 = tc.draw(gs::booleans());
      if (v2)
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gemm_q8r_wmma_t", &error), {(T + 127) / 128, (rows + 127) / 128},
                                {256}, ga, &error))
            << error;
      else
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gemm_q8r_wmma", &error), {(T + 63) / 64, (rows + 63) / 64}, {128}, ga,
                                &error))
            << error;
      std::vector<float> y(static_cast<std::size_t>(T) * rows);
      ASSERT_TRUE(dev->Download(yb, 0, y.data(), y.size() * 4, &error)) << error;
      for (auto* b : {&wb, &qs, &db, &xb, &xq, &xd, &yb}) dev->Free(*b);
      const std::size_t nblk = cols / 32;
      for (std::uint32_t t = 0; t < T; ++t) {
        std::vector<int> q(cols);
        std::vector<float> sx(nblk);
        for (std::size_t b = 0; b < nblk; ++b) {
          float amax = 0;
          for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[t * cols + b * 32 + i]));
          sx[b] = amax / 127.0f;
          for (int i = 0; i < 32; ++i)
            q[b * 32 + i] = sx[b] > 0 ? static_cast<int>(std::nearbyint(x[t * cols + b * 32 + i] / sx[b])) : 0;
        }
        for (std::uint32_t r = 0; r < rows; ++r) {
          double want = 0, mag = 0;
          for (std::size_t b = 0; b < nblk; ++b) {
            const std::uint8_t* blk = w.data() + r * row_bytes + b * 34;
            std::uint16_t hb;
            std::memcpy(&hb, blk, 2);
            const double dw = static_cast<double>(static_cast<float>(*reinterpret_cast<_Float16*>(&hb)));
            long dot = 0;
            for (int i = 0; i < 32; ++i) dot += static_cast<long>(static_cast<std::int8_t>(blk[2 + i])) * q[b * 32 + i];
            want += dw * sx[b] * static_cast<double>(dot);
            mag += std::fabs(dw * sx[b]) * 127.0 * 127.0 * 32.0;
          }
          ASSERT_NEAR(y[static_cast<std::size_t>(t) * rows + r], want, 1e-5 * mag + 1e-30)
              << rt::BackendName(backend) << (v2 ? " tiled" : " simple") << " t " << t << " r " << r << " rows " << rows
              << " cols " << cols << " T " << T;
        }
      }
    }, {.test_cases = 80});
  }
}

TEST(GpuKernels, WmmaF16LayoutProbe) {
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << error;
  ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
  std::vector<_Float16> a(256), b(256);
  for (int i = 0; i < 256; ++i) {
    a[i] = static_cast<_Float16>(static_cast<float>((i * 37 + 11) % 61 - 30) / 8.0f);
    b[i] = static_cast<_Float16>(static_cast<float>((i * 91 + 5) % 53 - 26) / 4.0f);
  }
  auto ab = dev->Alloc(512, &error), bb = dev->Alloc(512, &error), cb = dev->Alloc(1024, &error);
  ASSERT_TRUE(dev->Upload(ab, 0, a.data(), 512, &error) && dev->Upload(bb, 0, b.data(), 512, &error));
  const rt::Arg args[] = {rt::Arg::Buf(ab), rt::Arg::Buf(bb), rt::Arg::Buf(cb)};
  ASSERT_TRUE(dev->Launch(dev->Kernel("halo_wmma_probe_f16", &error), {1}, {32}, args, &error)) << error;
  std::vector<float> c(256);
  ASSERT_TRUE(dev->Download(cb, 0, c.data(), 1024, &error)) << error;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      float want = 0, mag = 0;  // WMMA rounds its internal sums: compare to 1e-6 of the magnitude
      for (int k = 0; k < 16; ++k) {
        want += static_cast<float>(a[m * 16 + k]) * static_cast<float>(b[n * 16 + k]);
        mag += std::fabs(static_cast<float>(a[m * 16 + k]) * static_cast<float>(b[n * 16 + k]));
      }
      ASSERT_NEAR(c[m * 16 + n], want, 1e-6f * mag) << "C[" << m << "][" << n << "]";
    }
  for (auto* x : {&ab, &bb, &cb}) dev->Free(*x);
}

// P4 DeltaNet over a chunk: conv + L2 for T tokens, the conv state advanced
// once, and the T-token recurrence with the state in registers must equal
// T decode steps (halo_conv_l2norm + halo_gdn_step_d128_norm) exactly:
// every output row and both final states, bit for bit. The lane-split
// recurrence (halo_gdn_prefill_split + halo_gated_norm) differs only in
// summation order: bound fixed in advance, relative L2 < 1e-5 on the
// outputs and the final recurrent state.
TEST(GpuKernels, DeltaNetPrefillEqualsDecodeSteps) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      constexpr std::uint32_t d = 128;
      const std::uint32_t nk = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 2}));
      const std::uint32_t nv = nk * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3}));
      const std::uint32_t kern = tc.draw(gs::sampled_from(std::vector<std::uint32_t>{2, 4}));
      const std::uint32_t T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 40}));
      const std::uint32_t mode = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 1}));
      const std::uint32_t kd = nk * d, vd = nv * d, ch = 2 * kd + vd;
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto vec = [&](std::size_t n, float amp) {
        std::vector<float> v(n);
        for (auto& x : v) {
          seed = seed * 1664525u + 1013904223u;
          x = amp * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f);
        }
        return v;
      };
      const auto in = vec(std::size_t{T} * ch, 3.0f), cst = vec(std::size_t{kern - 1} * ch, 1.0f),
                 cw = vec(std::size_t{ch} * kern, 0.5f), al = vec(std::size_t{T} * nv, 4.0f),
                 be = vec(std::size_t{T} * nv, 4.0f), gst = vec(std::size_t{nv} * d * d, 0.2f), nw = vec(d, 1.0f),
                 zv = vec(std::size_t{T} * vd, 4.0f), dtv = vec(nv, 2.0f);
      auto av = vec(nv, 2.0f);
      for (auto& x : av) x = -std::exp(x);
      std::vector<rt::Buffer> owned;
      const auto up = [&](const std::vector<float>& h) {
        auto b = dev->Alloc(h.size() * 4, &error);
        EXPECT_TRUE(dev->Upload(b, 0, h.data(), h.size() * 4, &error));
        owned.push_back(b);
        return b;
      };
      const auto zeros = [&](std::size_t n) { return up(std::vector<float>(n, 0.0f)); };
      const auto launch = [&](const char* k, rt::Dim3 g, rt::Dim3 b, std::initializer_list<rt::Arg> a) {
        ASSERT_TRUE(dev->Launch(dev->Kernel(k, &error), g, b, std::span<const rt::Arg>(a.begin(), a.size()), &error)) << k;
      };
      auto inb = up(in), cwb = up(cw), alb = up(al), beb = up(be), ab = up(av), dtb = up(dtv), nwb = up(nw), zb = up(zv);
      auto cs1 = up(cst), cs2 = up(cst), gs1 = up(gst), gs2 = up(gst);
      auto conv1 = zeros(ch), out1 = zeros(std::size_t{T} * vd), conv2 = zeros(std::size_t{T} * ch),
           out2 = zeros(std::size_t{T} * vd);
      // T decode steps.
      for (std::uint32_t t = 0; t < T; ++t) {
        launch("halo_conv_l2norm", {ch / d}, {d}, {rt::Arg::Buf(inb, std::size_t{t} * ch * 4), rt::Arg::Buf(cs1),
                                                  rt::Arg::Buf(cwb), rt::Arg::Buf(conv1), rt::Arg::Val(ch),
                                                  rt::Arg::Val(kern), rt::Arg::Val(2 * kd), rt::Arg::Val(1e-6f)});
        launch("halo_gdn_step_d128_norm", {nv}, {d},
               {rt::Arg::Buf(conv1), rt::Arg::Buf(conv1, kd * 4ull), rt::Arg::Buf(conv1, 2ull * kd * 4),
                rt::Arg::Buf(alb, std::size_t{t} * nv * 4), rt::Arg::Buf(beb, std::size_t{t} * nv * 4), rt::Arg::Buf(ab),
                rt::Arg::Buf(dtb), rt::Arg::Buf(gs1), rt::Arg::Buf(nwb), rt::Arg::Buf(zb, std::size_t{t} * vd * 4),
                rt::Arg::Buf(out1, std::size_t{t} * vd * 4), rt::Arg::Val(nk), rt::Arg::Val(1e-6f), rt::Arg::Val(mode)});
      }
      // One chunk.
      launch("halo_conv_l2norm_prefill", {ch / d, T}, {d}, {rt::Arg::Buf(inb), rt::Arg::Buf(cs2), rt::Arg::Buf(cwb),
                                                           rt::Arg::Buf(conv2), rt::Arg::Val(ch), rt::Arg::Val(kern),
                                                           rt::Arg::Val(2 * kd), rt::Arg::Val(1e-6f)});
      std::optional<rt::Buffer> conv8;
      if (kern == 4) {  // the 8-token form, before the state advances (it reads the old state)
        conv8 = zeros(std::size_t{T} * ch);
        launch("halo_conv_l2norm_prefill8", {ch / d, (T + 7) / 8}, {d},
               {rt::Arg::Buf(inb), rt::Arg::Buf(cs2), rt::Arg::Buf(cwb), rt::Arg::Buf(*conv8), rt::Arg::Val(ch),
                rt::Arg::Val(T), rt::Arg::Val(2 * kd), rt::Arg::Val(1e-6f)});
      }
      launch("halo_conv_state_advance", {(ch + 255) / 256}, {256},
             {rt::Arg::Buf(inb), rt::Arg::Buf(cs2), rt::Arg::Val(ch), rt::Arg::Val(kern), rt::Arg::Val(T)});
      launch("halo_gdn_prefill_d128_norm", {nv}, {d},
             {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(conv2, 2ull * kd * 4), rt::Arg::Buf(alb),
              rt::Arg::Buf(beb), rt::Arg::Buf(ab), rt::Arg::Buf(dtb), rt::Arg::Buf(gs2), rt::Arg::Buf(nwb), rt::Arg::Buf(zb),
              rt::Arg::Buf(out2), rt::Arg::Val(nk), rt::Arg::Val(1e-6f), rt::Arg::Val(mode), rt::Arg::Val(T),
              rt::Arg::Val(ch), rt::Arg::Val(ch), rt::Arg::Val(vd)});
      const auto get = [&](rt::Buffer b, std::size_t n) {
        std::vector<float> h(n);
        EXPECT_TRUE(dev->Download(b, 0, h.data(), n * 4, &error));
        return h;
      };
      const auto same = [&](const std::vector<float>& x, const std::vector<float>& y, const char* what) {
        ASSERT_EQ(x.size(), y.size());
        for (std::size_t i = 0; i < x.size(); ++i)
          ASSERT_EQ(std::bit_cast<std::uint32_t>(x[i]), std::bit_cast<std::uint32_t>(y[i]))
              << rt::BackendName(backend) << " " << what << "[" << i << "] " << x[i] << " vs " << y[i] << " T " << T
              << " kern " << kern;
      };
      auto gs3 = up(gst), out3 = zeros(std::size_t{T} * vd);
      launch("halo_gdn_prefill_split", {2, nv}, {256},
             {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(conv2, 2ull * kd * 4), rt::Arg::Buf(alb),
              rt::Arg::Buf(beb), rt::Arg::Buf(ab), rt::Arg::Buf(dtb), rt::Arg::Buf(gs3), rt::Arg::Buf(out3),
              rt::Arg::Val(nk), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch), rt::Arg::Val(vd)});
      launch("halo_gated_norm", {T * nv}, {d},
             {rt::Arg::Buf(out3), rt::Arg::Buf(nwb), rt::Arg::Buf(zb), rt::Arg::Val(d), rt::Arg::Val(1e-6f),
              rt::Arg::Val(mode)});
      const auto close = [&](const std::vector<float>& x, const std::vector<float>& y, const char* what) {
        double dd = 0, m = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
          ASSERT_TRUE(std::isfinite(x[i])) << what << "[" << i << "]";
          dd += double(x[i] - y[i]) * (x[i] - y[i]);
          m += double(y[i]) * y[i];
        }
        ASSERT_LT(std::sqrt(dd / std::max(m, 1e-30)), 1e-5) << rt::BackendName(backend) << " split " << what << " T " << T;
      };
      close(get(out3, std::size_t{T} * vd), get(out2, std::size_t{T} * vd), "output");
      close(get(gs3, gst.size()), get(gs2, gst.size()), "recurrent state");
      // The chunked form (halo_gdn_chunk_prep + halo_gdn_chunk_scan, 16-token
      // chunks) against the split kernel, before the gated norm: same bound.
      {
        constexpr std::uint32_t C = 16, kPrep = 2 * C + 2 * C * C;
        const std::uint32_t nch = (T + C - 1) / C;
        auto gs4 = up(gst), out4 = zeros(std::size_t{T} * vd), out5 = zeros(std::size_t{T} * vd), gs5 = up(gst),
             prep = zeros(std::size_t{nv} * nch * kPrep);
        launch("halo_gdn_prefill_split", {2, nv}, {256},
               {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(conv2, 2ull * kd * 4), rt::Arg::Buf(alb),
                rt::Arg::Buf(beb), rt::Arg::Buf(ab), rt::Arg::Buf(dtb), rt::Arg::Buf(gs5), rt::Arg::Buf(out5),
                rt::Arg::Val(nk), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch), rt::Arg::Val(vd)});
        launch("halo_gdn_chunk_prep", {nch, nv}, {256},
               {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(alb), rt::Arg::Buf(beb), rt::Arg::Buf(ab),
                rt::Arg::Buf(dtb), rt::Arg::Buf(prep), rt::Arg::Val(nk), rt::Arg::Val(T), rt::Arg::Val(ch)});
        launch("halo_gdn_chunk_scan", {2, nv}, {256},
               {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(conv2, 2ull * kd * 4), rt::Arg::Buf(prep),
                rt::Arg::Buf(gs4), rt::Arg::Buf(out4), rt::Arg::Val(nk), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch),
                rt::Arg::Val(vd)});
        close(get(out4, std::size_t{T} * vd), get(out5, std::size_t{T} * vd), "chunked output");
        close(get(gs4, gst.size()), get(gs5, gst.size()), "chunked recurrent state");
        // The WMMA chunked kernel: every product bf16 hi/lo split (~2^-15
        // relative per product), so its bound is 1e-4 (fixed in advance).
        auto gs6 = up(gst), out6 = zeros(std::size_t{T} * vd);
        launch("halo_gdn_chunk_wmma", {4, nv}, {128},
               {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(conv2, 2ull * kd * 4), rt::Arg::Buf(alb),
                rt::Arg::Buf(beb), rt::Arg::Buf(ab), rt::Arg::Buf(dtb), rt::Arg::Buf(gs6), rt::Arg::Buf(out6),
                rt::Arg::Val(nk), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch), rt::Arg::Val(vd)});
        const auto close4 = [&](const std::vector<float>& x, const std::vector<float>& y, const char* what) {
          double dd = 0, m = 0;
          for (std::size_t i = 0; i < x.size(); ++i) {
            ASSERT_TRUE(std::isfinite(x[i])) << what << "[" << i << "]";
            dd += double(x[i] - y[i]) * (x[i] - y[i]);
            m += double(y[i]) * y[i];
          }
          ASSERT_LT(std::sqrt(dd / std::max(m, 1e-30)), 1e-4) << rt::BackendName(backend) << " " << what << " T " << T;
        };
        close4(get(out6, std::size_t{T} * vd), get(out5, std::size_t{T} * vd), "wmma chunked output");
        close4(get(gs6, gst.size()), get(gs5, gst.size()), "wmma chunked recurrent state");
        // Prep + WMMA scan (halo_gdn_c16_prep, halo_gdn_c16_scan): same products, same bound.
        auto gs7 = up(gst), out7 = zeros(std::size_t{T} * vd), prep7 = zeros(std::size_t{nv} * nch * 576);
        launch("halo_gdn_c16_prep", {nch, nk}, {256},
               {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(alb), rt::Arg::Buf(beb), rt::Arg::Buf(ab),
                rt::Arg::Buf(dtb), rt::Arg::Buf(prep7), rt::Arg::Val(nv), rt::Arg::Val(T), rt::Arg::Val(ch)});
        launch("halo_gdn_c16_scan", {2, nv}, {128},
               {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(conv2, 2ull * kd * 4), rt::Arg::Buf(prep7),
                rt::Arg::Buf(gs7), rt::Arg::Buf(out7), rt::Arg::Val(nk), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch),
                rt::Arg::Val(vd)});
        close4(get(out7, std::size_t{T} * vd), get(out5, std::size_t{T} * vd), "c16 chunked output");
        close4(get(gs7, gst.size()), get(gs5, gst.size()), "c16 chunked recurrent state");
        // halo_gdn_c16_scan3 (three-way K S0, split at staging) on the same prep.
        auto gs8 = up(gst), out8 = zeros(std::size_t{T} * vd);
        launch("halo_gdn_c16_scan3", {2, nv}, {128},
               {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(conv2, 2ull * kd * 4), rt::Arg::Buf(prep7),
                rt::Arg::Buf(gs8), rt::Arg::Buf(out8), rt::Arg::Val(nk), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch),
                rt::Arg::Val(vd)});
        close4(get(out8, std::size_t{T} * vd), get(out5, std::size_t{T} * vd), "c16 scan3 output");
        close4(get(gs8, gst.size()), get(gs5, gst.size()), "c16 scan3 recurrent state");
        auto gs9 = up(gst), out9 = zeros(std::size_t{T} * vd);
        launch("halo_gdn_c16_scan3u", {2, nv}, {128},
               {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(conv2, 2ull * kd * 4), rt::Arg::Buf(prep7),
                rt::Arg::Buf(gs9), rt::Arg::Buf(out9), rt::Arg::Val(nk), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch),
                rt::Arg::Val(vd)});
        close4(get(out9, std::size_t{T} * vd), get(out5, std::size_t{T} * vd), "c16 scan3u output");
        close4(get(gs9, gst.size()), get(gs5, gst.size()), "c16 scan3u recurrent state");
        // prep2 (cpb chunks a block, prefetched) writes the same prep as prep.
        for (const std::uint32_t cpb : {1u, 2u, 3u}) {
          auto prep2 = zeros(std::size_t{nv} * nch * 576);
          launch("halo_gdn_c16_prep2", {(nch + cpb - 1) / cpb, nk}, {256},
                 {rt::Arg::Buf(conv2), rt::Arg::Buf(conv2, kd * 4ull), rt::Arg::Buf(alb), rt::Arg::Buf(beb),
                  rt::Arg::Buf(ab), rt::Arg::Buf(dtb), rt::Arg::Buf(prep2), rt::Arg::Val(nv), rt::Arg::Val(T),
                  rt::Arg::Val(ch), rt::Arg::Val(cpb)});
          same(get(prep2, std::size_t{nv} * nch * 576), get(prep7, std::size_t{nv} * nch * 576), "prep2");
        }
      }
      if (conv8) same(get(*conv8, std::size_t{T} * ch), get(conv2, std::size_t{T} * ch), "conv8");
      same(get(out2, std::size_t{T} * vd), get(out1, std::size_t{T} * vd), "output");
      same(get(cs2, cst.size()), get(cs1, cst.size()), "conv state");
      same(get(gs2, gst.size()), get(gs1, gst.size()), "recurrent state");
      for (auto& b : owned) dev->Free(b);
    }, {.test_cases = 40});
  }
}

// P4 stage B: routed experts of a chunk grouped by expert (bucket, gate/up
// with SwiGLU, down, weighted combine) against a CPU reference that applies
// the kernels' fp16 roundings (activations, dequantized weights, the SwiGLU
// output), so only float summation order differs. K-quant weights are
// dequantized as the kernels specify: fp16(q * fp16(d * sc) + fp16(-dmin * m)),
// within about one fp16 ulp of the sub-block offset of the exact value. Bound fixed in advance:
// |got - want| <= 1e-3 * sum |terms| per output. Also: bucketing is a
// permutation grouped by expert in pair order, tiles cover each expert's
// pairs exactly, and the row-wise router top-k equals the single-row kernel.
// The per-pair down rows are fp16 too: the reference rounds each expert's
// down output to fp16 before the weighted sum.
TEST(GpuQuant, GroupedExpertsMatchCpuReference) {
  std::string error;
  // One row of Q4_K / Q5_K the way the grouped kernels dequantize it.
  const auto kquant_row = [](const std::uint8_t* row, std::size_t cols, bool q5, float* out) {
    const auto h = [](const std::uint8_t* p) {
      std::uint16_t b;
      std::memcpy(&b, p, 2);
      _Float16 f;
      std::memcpy(&f, &b, 2);
      return static_cast<float>(f);
    };
    const auto r16 = [](float v) { return static_cast<float>(static_cast<_Float16>(v)); };
    const std::size_t bb = q5 ? 176 : 144;
    for (std::size_t c = 0; c < cols; ++c) {
      const std::uint8_t* b = row + (c / 256) * bb;
      const std::uint32_t e = c % 256, chunk = e / 64, hi = (e % 64) / 32, l = e % 32, j = 2 * chunk + hi;
      const std::uint8_t* q = b + 4;
      const std::uint32_t sc = j < 4 ? q[j] & 63 : (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
      const std::uint32_t m = j < 4 ? q[j + 4] & 63 : (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
      const std::uint8_t qb = b[(q5 ? 48 : 16) + chunk * 32 + l];
      std::uint32_t v = hi ? qb >> 4 : qb & 0xF;
      if (q5 && (b[16 + l] & (1u << j))) v += 16;
      out[c] = r16(static_cast<float>(v) * r16(h(b) * static_cast<float>(sc)) + r16(-h(b + 2) * static_cast<float>(m)));
    }
  };
  const auto f16 = [](float v) { return static_cast<float>(static_cast<_Float16>(v)); };
  const auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const std::size_t H = 256 * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 3}));
      const std::size_t ff = 32 * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 12}));
      const std::size_t E = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 40}));
      const std::size_t used = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = std::min<std::size_t>(E, 10)}));
      const std::size_t T = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 200}));
      const bool q5k = tc.draw(gs::booleans()), q8_down = tc.draw(gs::booleans());
      const bool repacked = q8_down && tc.draw(gs::booleans());  // the engine's repacked Q8_0 layout
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto fill = [&](std::vector<std::uint8_t>& w, std::size_t rows, std::size_t row_bytes, std::size_t block_bytes,
                            std::initializer_list<std::size_t> halves) {
        for (auto& b : w) { seed = seed * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(seed >> 24); }
        for (std::size_t r = 0; r < rows; ++r)
          for (std::size_t b = 0; b < row_bytes / block_bytes; ++b)
            for (std::size_t off : halves) {
              // Small finite scales so values stay well inside fp16.
              const auto h = static_cast<std::uint16_t>(FiniteHalf(tc) & 0xA3FF);
              std::memcpy(w.data() + r * row_bytes + b * block_bytes + off, &h, 2);
            }
      };
      const std::size_t gblk = q5k ? 176 : 144, gu_row = H / 256 * gblk, dn_row = q8_down ? ff / 32 * 34 : ff / 32 * 24;
      std::vector<std::uint8_t> gate(E * ff * gu_row), up(E * ff * gu_row), down(E * H * dn_row);
      fill(gate, E * ff, gu_row, gblk, {0, 2});
      fill(up, E * ff, gu_row, gblk, {0, 2});
      if (q8_down) fill(down, E * H, dn_row, 34, {0});
      else fill(down, E * H, dn_row, 24, {0, 2});
      // Routing: router logits per token; the CPU takes the GPU's top-k
      // (checked against the single-row kernel below).
      std::vector<float> logits(T * E), x(T * H);
      for (auto& v : logits) { seed = seed * 1664525u + 1013904223u; v = static_cast<float>(seed >> 8) / 16777216.0f * 4.0f; }
      for (auto& v : x) { seed = seed * 1664525u + 1013904223u; v = (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 0.2f; }
      const std::size_t P = T * used, max_tiles = P / 64 + E + 1;
      auto gb = dev->Alloc(gate.size(), &error), ub = dev->Alloc(up.size(), &error), db = dev->Alloc(down.size(), &error),
           lb = dev->Alloc(logits.size() * 4, &error), xb = dev->Alloc(x.size() * 4, &error),
           ib = dev->Alloc(P * 4, &error), wb = dev->Alloc(P * 4, &error), i1 = dev->Alloc(P * 4, &error),
           w1 = dev->Alloc(P * 4, &error), sb = dev->Alloc(P * 4, &error), pb = dev->Alloc(P * 4, &error),
           tb = dev->Alloc(max_tiles * 12, &error), nb = dev->Alloc(4, &error), hb = dev->Alloc(P * ff * 4, &error),
           yb = dev->Alloc(P * H * 4, &error), ob = dev->Alloc(T * H * 4, &error);
      ASSERT_TRUE(gb && ub && db && lb && xb && ib && wb && i1 && w1 && sb && pb && tb && nb && hb && yb && ob) << error;
      ASSERT_TRUE(dev->Upload(gb, 0, gate.data(), gate.size(), &error) && dev->Upload(ub, 0, up.data(), up.size(), &error) &&
                  dev->Upload(db, 0, down.data(), down.size(), &error) &&
                  dev->Upload(lb, 0, logits.data(), logits.size() * 4, &error) &&
                  dev->Upload(xb, 0, x.data(), x.size() * 4, &error))
          << error;
      const auto u32 = [](std::size_t v) { return static_cast<std::uint32_t>(v); };
      const auto launch = [&](const char* k, rt::Dim3 g, rt::Dim3 b, std::initializer_list<rt::Arg> a) {
        const std::vector<rt::Arg> args(a);
        ASSERT_TRUE(dev->Launch(dev->Kernel(k, &error), g, b, args, &error)) << k << ": " << error;
      };
      launch("halo_router_topk_rows", {u32(T)}, {256},
             {rt::Arg::Buf(lb), rt::Arg::Buf(ib), rt::Arg::Buf(wb), rt::Arg::Val(u32(E)), rt::Arg::Val(u32(used))});
      for (std::size_t t = 0; t < T; ++t)
        launch("halo_router_topk", {1}, {256},
               {rt::Arg::Buf(lb, t * E * 4), rt::Arg::Buf(i1, t * used * 4), rt::Arg::Buf(w1, t * used * 4),
                rt::Arg::Val(u32(E)), rt::Arg::Val(u32(used))});
      launch("halo_moe_bucket", {1}, {1024},
             {rt::Arg::Buf(ib), rt::Arg::Val(u32(P)), rt::Arg::Val(u32(E)), rt::Arg::Buf(sb), rt::Arg::Buf(pb),
              rt::Arg::Buf(tb), rt::Arg::Buf(nb)});
      const std::string gk = std::string("halo_moe_gate_up_grouped_") + (q5k ? "q5_k" : "q4_k");
      const std::string dk = std::string("halo_moe_down_grouped_") + (q8_down ? "q8_0" : "q5_1");
      auto x16 = dev->Alloc(x.size() * 2, &error);
      ASSERT_TRUE(x16) << error;
      launch("halo_f16_perm_rows", {u32((x.size() + 255) / 256)}, {256},
             {rt::Arg::Buf(xb), rt::Arg::Buf(x16), rt::Arg::Val(u32(x.size()))});
      launch(gk.c_str(), {u32((ff + 127) / 128), u32(max_tiles)}, {256},
             {rt::Arg::Buf(gb), rt::Arg::Buf(ub), rt::Arg::Val<std::uint64_t>(ff * gu_row),
              rt::Arg::Val<std::uint64_t>(gu_row), rt::Arg::Buf(x16), rt::Arg::Buf(sb), rt::Arg::Buf(tb), rt::Arg::Buf(nb),
              rt::Arg::Buf(hb), rt::Arg::Val(u32(ff)), rt::Arg::Val(u32(H)), rt::Arg::Val(u32(used))});
      if (repacked) {
        const std::uint32_t qstride = u32((ff + 15) / 16 * 16);
        auto qs = dev->Alloc(E * H * qstride, &error), dsc = dev->Alloc(E * H * (ff / 32) * 2, &error);
        ASSERT_TRUE(qs && dsc) << error;
        launch("halo_repack_q8_0", {4096}, {256},
               {rt::Arg::Buf(db), rt::Arg::Buf(qs), rt::Arg::Buf(dsc), rt::Arg::Val(u32(E * H)), rt::Arg::Val(u32(ff)),
                rt::Arg::Val(qstride)});
        launch("halo_moe_down_grouped_q8r", {u32((H + 127) / 128), u32(max_tiles)}, {256},
               {rt::Arg::Buf(qs), rt::Arg::Buf(dsc), rt::Arg::Val(qstride), rt::Arg::Buf(hb), rt::Arg::Buf(tb),
                rt::Arg::Buf(nb), rt::Arg::Buf(yb), rt::Arg::Val(u32(H)), rt::Arg::Val(u32(ff))});
        ASSERT_TRUE(dev->Synchronize(&error)) << error;
        dev->Free(qs);
        dev->Free(dsc);
      } else {
        launch(dk.c_str(), {u32((H + 127) / 128), u32(max_tiles)}, {256},
               {rt::Arg::Buf(db), rt::Arg::Val<std::uint64_t>(H * dn_row), rt::Arg::Val<std::uint64_t>(dn_row),
                rt::Arg::Buf(hb), rt::Arg::Buf(tb), rt::Arg::Buf(nb), rt::Arg::Buf(yb), rt::Arg::Val(u32(H)),
                rt::Arg::Val(u32(ff))});
      }
      launch("halo_moe_combine_grouped", {u32((H + 255) / 256), u32(T)}, {256},
             {rt::Arg::Buf(yb), rt::Arg::Buf(pb), rt::Arg::Buf(wb), rt::Arg::Buf(ob), rt::Arg::Val(u32(used)),
              rt::Arg::Val(u32(H))});
      std::vector<std::uint32_t> ids(P), ids1(P), sorted(P), pos(P), tiles(max_tiles * 3), ntiles(1);
      std::vector<float> wts(P), wts1(P), out(T * H);
      ASSERT_TRUE(dev->Download(ib, 0, ids.data(), P * 4, &error) && dev->Download(i1, 0, ids1.data(), P * 4, &error) &&
                  dev->Download(wb, 0, wts.data(), P * 4, &error) && dev->Download(w1, 0, wts1.data(), P * 4, &error) &&
                  dev->Download(sb, 0, sorted.data(), P * 4, &error) && dev->Download(pb, 0, pos.data(), P * 4, &error) &&
                  dev->Download(tb, 0, tiles.data(), tiles.size() * 4, &error) &&
                  dev->Download(nb, 0, ntiles.data(), 4, &error) && dev->Download(ob, 0, out.data(), out.size() * 4, &error))
          << error;
      // The three-pass parallel bucketing gives the same output exactly.
      {
        const std::uint32_t chunks = u32((P + 1023) / 1024);
        auto cb = dev->Alloc(std::size_t{chunks} * E * 4, &error), bb = dev->Alloc(E * 4, &error),
             s2 = dev->Alloc(P * 4, &error), p2 = dev->Alloc(P * 4, &error), t2 = dev->Alloc(max_tiles * 12, &error),
             n2 = dev->Alloc(4, &error);
        ASSERT_TRUE(cb && bb && s2 && p2 && t2 && n2) << error;
        launch("halo_moe_bucket_count", {chunks}, {1024},
               {rt::Arg::Buf(ib), rt::Arg::Val(u32(P)), rt::Arg::Val(u32(E)), rt::Arg::Buf(cb)});
        launch("halo_moe_bucket_scan", {1}, {1024},
               {rt::Arg::Buf(cb), rt::Arg::Val(chunks), rt::Arg::Val(u32(E)), rt::Arg::Buf(t2), rt::Arg::Buf(n2),
                rt::Arg::Buf(bb)});
        launch("halo_moe_bucket_scatter", {chunks}, {1024},
               {rt::Arg::Buf(ib), rt::Arg::Val(u32(P)), rt::Arg::Val(u32(E)), rt::Arg::Buf(cb), rt::Arg::Buf(bb),
                rt::Arg::Buf(s2), rt::Arg::Buf(p2)});
        std::vector<std::uint32_t> sorted2(P), pos2(P), tiles2(max_tiles * 3), ntiles2(1);
        ASSERT_TRUE(dev->Download(s2, 0, sorted2.data(), P * 4, &error) && dev->Download(p2, 0, pos2.data(), P * 4, &error) &&
                    dev->Download(t2, 0, tiles2.data(), tiles2.size() * 4, &error) &&
                    dev->Download(n2, 0, ntiles2.data(), 4, &error))
            << error;
        for (auto* b : {&cb, &bb, &s2, &p2, &t2, &n2}) dev->Free(*b);
        ASSERT_EQ(sorted2, sorted);
        ASSERT_EQ(pos2, pos);
        ASSERT_EQ(ntiles2[0], ntiles[0]);
        ASSERT_TRUE(std::equal(tiles2.begin(), tiles2.begin() + 3 * ntiles[0], tiles.begin()));
      }
      for (auto* b : {&gb, &ub, &db, &lb, &xb, &ib, &wb, &i1, &w1, &sb, &pb, &tb, &nb, &hb, &yb, &ob, &x16}) dev->Free(*b);
      ASSERT_EQ(ids, ids1);
      ASSERT_EQ(wts, wts1);
      // Bucketing: sorted is a permutation, grouped by expert, pair order
      // within an expert, pos its inverse; tiles cover each group.
      for (std::size_t i = 0; i < P; ++i) {
        ASSERT_LT(sorted[i], P);
        ASSERT_EQ(pos[sorted[i]], i);
        if (i > 0) {
          ASSERT_LE(ids[sorted[i - 1]], ids[sorted[i]]);
          if (ids[sorted[i - 1]] == ids[sorted[i]]) ASSERT_LT(sorted[i - 1], sorted[i]);
        }
      }
      std::vector<std::size_t> covered(P, 0);
      ASSERT_LE(ntiles[0], max_tiles);
      for (std::size_t i = 0; i < ntiles[0]; ++i) {
        ASSERT_GE(tiles[3 * i + 2], 1u);
        ASSERT_LE(tiles[3 * i + 2], 64u);
        for (std::uint32_t j = 0; j < tiles[3 * i + 2]; ++j) {
          ASSERT_EQ(ids[sorted[tiles[3 * i + 1] + j]], tiles[3 * i]);
          ++covered[tiles[3 * i + 1] + j];
        }
      }
      for (auto n : covered) ASSERT_EQ(n, 1u);
      // Reference with the kernels' fp16 roundings.
      model::TensorRef tg, tu, td;
      tg.data = gate.data(); tg.type = q5k ? GgmlType::kQ5_K : GgmlType::kQ4_K; tg.cols = H; tg.rows = ff; tg.experts = E;
      tu = tg; tu.data = up.data();
      td.data = down.data(); td.type = q8_down ? GgmlType::kQ8_0 : GgmlType::kQ5_1; td.cols = ff; td.rows = H; td.experts = E;
      std::vector<float> wg(H), wu(H), wd(ff), h(ff);
      for (std::size_t t = 0; t < T; ++t) {
        std::vector<double> want(H, 0.0), mag(H, 0.0);
        for (std::size_t s = 0; s < used; ++s) {
          const std::uint32_t e = ids[t * used + s];
          for (std::size_t j = 0; j < ff; ++j) {
            kquant_row(gate.data() + (e * ff + j) * gu_row, H, q5k, wg.data());
            kquant_row(up.data() + (e * ff + j) * gu_row, H, q5k, wu.data());
            double g = 0, u = 0;
            for (std::size_t c = 0; c < H; ++c) {
              const double xc = f16(x[t * H + c]);
              g += double(wg[c]) * xc;
              u += double(wu[c]) * xc;
            }
            h[j] = f16(silu(static_cast<float>(g)) * static_cast<float>(u));
          }
          for (std::size_t r = 0; r < H; ++r) {
            model::cpu::DequantizeRow(td, e, r, wd.data());
            double y = 0, m = 0;
            for (std::size_t j = 0; j < ff; ++j) {
              y += double(f16(wd[j])) * h[j];
              m += std::fabs(double(f16(wd[j])) * h[j]);
            }
            want[r] += wts[t * used + s] * f16(static_cast<float>(y));
            mag[r] += std::fabs(wts[t * used + s]) * m;
          }
        }
        for (std::size_t r = 0; r < H; ++r)
          ASSERT_NEAR(out[t * H + r], want[r], 1e-3 * mag[r] + 1e-30)
              << rt::BackendName(backend) << " t " << t << " r " << r << " H " << H << " ff " << ff << " E " << E
              << " used " << used << " T " << T << " q5k " << q5k << " q8_down " << q8_down << " repacked " << repacked;
      }
    }, {.test_cases = 40,
        // Each case runs a large CPU reference; on a GPU shared with another job
        // generation can exceed the 30 s budget without anything being wrong.
        .suppress_health_check = {hegel::HealthCheck::TooSlow}});
  }
}

// P4: the tiled FP32 GEMM for F32 projections against double sums, with
// the worst-case fp32 summation bound |err| <= cols * 2^-23 * sum |w x|,
// over ragged rows and tokens.
TEST(GpuQuant, TiledF32GemmMatchesDoubleSums) {  // and the skinny kernel for <= 16 rows
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const auto rows = tc.draw(gs::booleans()) ? tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 16}))
                                                : tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 600}));
      const auto cols = 16 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 80}));
      const auto T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 200}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      std::vector<float> w(std::size_t{rows} * cols), x(std::size_t{T} * cols), y(std::size_t{T} * rows);
      for (auto& v : w) v = rnd();
      for (auto& v : x) v = 3 * rnd();
      auto wb = dev->Alloc(w.size() * 4, &error), xb = dev->Alloc(x.size() * 4, &error), yb = dev->Alloc(y.size() * 4, &error);
      ASSERT_TRUE(wb && xb && yb && dev->Upload(wb, 0, w.data(), w.size() * 4, &error) &&
                  dev->Upload(xb, 0, x.data(), x.size() * 4, &error))
          << error;
      const rt::Arg a[] = {rt::Arg::Buf(wb), rt::Arg::Buf(xb), rt::Arg::Buf(yb), rt::Arg::Val(rows), rt::Arg::Val(cols),
                           rt::Arg::Val(T)};
      const bool skinny = rows <= 16 && tc.draw(gs::booleans());
      if (skinny)
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gemm_f32_skinny", &error), {(T + 7) / 8}, {256}, a, &error)) << error;
      else
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gemm_f32_tiled", &error), {(T + 63) / 64, (rows + 63) / 64}, {256}, a,
                                &error))
            << error;
      ASSERT_TRUE(dev->Download(yb, 0, y.data(), y.size() * 4, &error)) << error;
      for (auto* b : {&wb, &xb, &yb}) dev->Free(*b);
      for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t r = 0; r < rows; ++r) {
          double want = 0, mag = 0;
          for (std::uint32_t c = 0; c < cols; ++c) {
            want += double(w[std::size_t{r} * cols + c]) * x[std::size_t{t} * cols + c];
            mag += std::fabs(double(w[std::size_t{r} * cols + c]) * x[std::size_t{t} * cols + c]);
          }
          ASSERT_NEAR(y[std::size_t{t} * rows + r], want, cols * 0x1p-23 * mag + 1e-30)
              << rt::BackendName(backend) << " t " << t << " r " << r << " rows " << rows << " cols " << cols << " T " << T;
        }
    }, {.test_cases = 40});
  }
}

// P4: the FP32 GEMM on bf16 WMMA with hi/lo splitting (the router), against
// double sums. Bound fixed in advance from the split: each product within
// (2^-16 + 2 * 2^-17 + 2^-18) of |w x| (the dropped lo*lo, and lo rounded
// to bf16 on both operands), plus fp32 accumulation cols * 2^-23: so
// (2^-15 + cols * 2^-23) * sum |w x|. Ragged rows and tokens; cols % 32 == 0.
TEST(GpuQuant, Bf16x3GemmMatchesDoubleSums) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const auto rows = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 600}));
      const auto cols = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 100}));
      const auto T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 200}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      // Values over many binades, as router weights and activations are.
      const auto wide = [&] { return rnd() * std::ldexp(1.0f, static_cast<int>(rnd() * 16)); };
      std::vector<float> w(std::size_t{rows} * cols), x(std::size_t{T} * cols), y(std::size_t{T} * rows);
      for (auto& v : w) v = wide();
      for (auto& v : x) v = wide();
      auto wb = dev->Alloc(w.size() * 4, &error), xb = dev->Alloc(x.size() * 4, &error), yb = dev->Alloc(y.size() * 4, &error);
      ASSERT_TRUE(wb && xb && yb && dev->Upload(wb, 0, w.data(), w.size() * 4, &error) &&
                  dev->Upload(xb, 0, x.data(), x.size() * 4, &error))
          << error;
      const rt::Arg a[] = {rt::Arg::Buf(wb), rt::Arg::Buf(xb), rt::Arg::Buf(yb), rt::Arg::Val(rows), rt::Arg::Val(cols),
                           rt::Arg::Val(T)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gemm_f32_bf16x3", &error), {(T + 63) / 64, (rows + 63) / 64}, {256}, a,
                              &error))
          << error;
      ASSERT_TRUE(dev->Download(yb, 0, y.data(), y.size() * 4, &error)) << error;
      for (auto* b : {&wb, &xb, &yb}) dev->Free(*b);
      for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t r = 0; r < rows; ++r) {
          double want = 0, mag = 0;
          for (std::uint32_t c = 0; c < cols; ++c) {
            want += double(w[std::size_t{r} * cols + c]) * x[std::size_t{t} * cols + c];
            mag += std::fabs(double(w[std::size_t{r} * cols + c]) * x[std::size_t{t} * cols + c]);
          }
          ASSERT_NEAR(y[std::size_t{t} * rows + r], want, (0x1p-15 + cols * 0x1p-23) * mag + 1e-30)
              << rt::BackendName(backend) << " t " << t << " r " << r << " rows " << rows << " cols " << cols << " T " << T;
        }
    }, {.test_cases = 40});
  }
}

// Precision option (stream P): fp32 weights rounded to fp16 while staging,
// fp16 activations, FP16 WMMA with fp32 sums (halo_gemm_f32h_f16_k128),
// against double sums of the same fp16-rounded operands: fp16 products are
// exact in fp32, so only the fp32 summation differs. Bound fixed in advance:
// cols * 2^-23 * sum |w16 x16|. Ragged rows and tokens; cols % 128 == 0.
TEST(GpuQuant, F32WeightHalfActivationGemmMatchesFp16Sums) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const auto rows = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 600}));
      const auto cols = 128 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 24}));
      const auto T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 200}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      const auto wide = [&] { return rnd() * std::ldexp(1.0f, static_cast<int>(rnd() * 16)); };
      std::vector<float> w(std::size_t{rows} * cols), y(std::size_t{T} * rows);
      std::vector<_Float16> x(std::size_t{T} * cols);
      for (auto& v : w) v = wide();
      for (auto& v : x) v = static_cast<_Float16>(wide());
      auto wb = dev->Alloc(w.size() * 4, &error), xb = dev->Alloc(x.size() * 2, &error), yb = dev->Alloc(y.size() * 4, &error);
      ASSERT_TRUE(wb && xb && yb && dev->Upload(wb, 0, w.data(), w.size() * 4, &error) &&
                  dev->Upload(xb, 0, x.data(), x.size() * 2, &error))
          << error;
      const rt::Arg a[] = {rt::Arg::Buf(wb), rt::Arg::Buf(xb), rt::Arg::Buf(yb), rt::Arg::Val(rows), rt::Arg::Val(cols),
                           rt::Arg::Val(T)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gemm_f32h_f16_k128", &error), {(T + 63) / 64, (rows + 63) / 64}, {128},
                              a, &error))
          << error;
      ASSERT_TRUE(dev->Download(yb, 0, y.data(), y.size() * 4, &error)) << error;
      for (auto* b : {&wb, &xb, &yb}) dev->Free(*b);
      for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t r = 0; r < rows; ++r) {
          double want = 0, mag = 0;
          for (std::uint32_t c = 0; c < cols; ++c) {
            const double p = double(static_cast<float>(static_cast<_Float16>(w[std::size_t{r} * cols + c]))) *
                             double(static_cast<float>(x[std::size_t{t} * cols + c]));
            want += p;
            mag += std::fabs(p);
          }
          ASSERT_NEAR(y[std::size_t{t} * rows + r], want, cols * 0x1p-23 * mag + 1e-30)
              << rt::BackendName(backend) << " t " << t << " r " << r << " rows " << rows << " cols " << cols << " T " << T;
        }
    }, {.test_cases = 40});
  }
}

// The router variants with one operand split hi/lo: halo_gemm_f32h_f16_w2
// (fp32 W as fp16 hi + lo, fp16 x) and halo_gemm_f32_f16_x2 (fp16 W, fp32 x
// as hi + lo), against double sums of W x with the unsplit operand rounded
// to fp16. Bound fixed in advance: the split operand within 2^-22 relative
// (hi + lo, lo rounded; normal range), fp32 sums: (2^-21 + cols * 2^-23) *
// sum |w x|. cols % 64 == 0; values kept clear of fp16 subnormals.
TEST(GpuQuant, SplitOperandF16GemmsMatchSums) {
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << error;
  ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
  hegel::test([&](hegel::TestCase& tc) {
    const bool split_w = tc.draw(gs::booleans());
    const auto rows = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 300}));
    const auto cols = 64 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 40}));
    const auto T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 200}));
    std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
    const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f; };
    // magnitudes in [2^-6, 2^6), random sign: clear of fp16 subnormals for hi and lo
    const auto val = [&] { return (rnd() < 0.5f ? -1.0f : 1.0f) * std::ldexp(1.0f + rnd(), static_cast<int>(rnd() * 12) - 6); };
    std::vector<float> w(std::size_t{rows} * cols), xf(std::size_t{T} * cols), y(std::size_t{T} * rows);
    for (auto& v : w) v = val();
    for (auto& v : xf) v = val();
    std::vector<_Float16> xh(xf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) xh[i] = static_cast<_Float16>(xf[i]);
    auto wb = dev->Alloc(w.size() * 4, &error), xb = dev->Alloc(xf.size() * 4, &error), yb = dev->Alloc(y.size() * 4, &error);
    ASSERT_TRUE(wb && xb && yb && dev->Upload(wb, 0, w.data(), w.size() * 4, &error) &&
                (split_w ? dev->Upload(xb, 0, xh.data(), xh.size() * 2, &error)
                         : dev->Upload(xb, 0, xf.data(), xf.size() * 4, &error)))
        << error;
    const rt::Arg a[] = {rt::Arg::Buf(wb), rt::Arg::Buf(xb), rt::Arg::Buf(yb), rt::Arg::Val(rows), rt::Arg::Val(cols),
                         rt::Arg::Val(T)};
    ASSERT_TRUE(dev->Launch(dev->Kernel(split_w ? "halo_gemm_f32h_f16_w2" : "halo_gemm_f32_f16_x2", &error),
                            {(T + 63) / 64, (rows + 63) / 64}, {128}, a, &error))
        << error;
    ASSERT_TRUE(dev->Download(yb, 0, y.data(), y.size() * 4, &error)) << error;
    for (auto* b : {&wb, &xb, &yb}) dev->Free(*b);
    for (std::uint32_t t = 0; t < T; ++t)
      for (std::uint32_t r = 0; r < rows; ++r) {
        double want = 0, mag = 0;
        for (std::uint32_t c = 0; c < cols; ++c) {
          const float wv = w[std::size_t{r} * cols + c], xv = xf[std::size_t{t} * cols + c];
          const double p = split_w ? double(wv) * double(static_cast<float>(static_cast<_Float16>(xv)))
                                   : double(static_cast<float>(static_cast<_Float16>(wv))) * double(xv);
          want += p;
          mag += std::fabs(p);
        }
        ASSERT_NEAR(y[std::size_t{t} * rows + r], want, (0x1p-21 + cols * 0x1p-23) * mag)
            << (split_w ? "w2" : "x2") << " t " << t << " r " << r << " rows " << rows << " cols " << cols;
      }
  }, {.test_cases = 30});
}

// Precision option (stream P): activations quantized per token and 32 values
// into the K-blocked layout (halo_quantize_q8_kblocked: amax / 127, round to
// nearest, checked bit for bit against the host), then the W8A8 GEMM
// (halo_gemm_q8a8_128x128) against double sums of the same int8 operands.
// Bound fixed in advance: per 32-block term, (float)isum (exact) times the
// rounded dw * dx, accumulated in fp32: (nblk + 2) * 2^-23 * sum |term|.
// Ragged rows and tokens; cols % 64 == 0.
TEST(GpuQuant, W8A8GemmMatchesInt8Sums) {
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << error;
  ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
  hegel::test([&](hegel::TestCase& tc) {
    const auto rows = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 400}));
    const auto cols = 64 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 48}));
    const auto T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 300}));
    const std::uint32_t nblk = cols / 32;
    std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
    const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    std::vector<std::int8_t> q(std::size_t{rows} * cols);
    for (auto& v : q) v = static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
    std::vector<std::uint16_t> dw(std::size_t{rows} * nblk);
    for (auto& v : dw) {
      const _Float16 h = static_cast<_Float16>(std::ldexp(1.0f + (rnd() % 1000) / 1000.0f, -int(rnd() % 12)));
      std::memcpy(&v, &h, 2);
    }
    std::vector<float> x(std::size_t{T} * cols);
    for (auto& v : x) v = (static_cast<float>(rnd() % 20001) / 10000.0f - 1.0f) * std::ldexp(1.0f, int(rnd() % 8) - 4);
    if (tc.draw(gs::booleans())) x[0] = 0.0f, std::fill(x.begin(), x.begin() + std::min<std::size_t>(32, x.size()), 0.0f);
    auto qb = dev->Alloc(q.size(), &error), db = dev->Alloc(dw.size() * 2, &error), xb = dev->Alloc(x.size() * 4, &error),
         xqb = dev->Alloc(x.size(), &error), xdb = dev->Alloc(std::size_t{nblk} * T * 4, &error),
         yb = dev->Alloc(std::size_t{T} * rows * 4, &error);
    ASSERT_TRUE(qb && db && xb && xqb && xdb && yb) << error;
    ASSERT_TRUE(dev->Upload(qb, 0, q.data(), q.size(), &error) && dev->Upload(db, 0, dw.data(), dw.size() * 2, &error) &&
                dev->Upload(xb, 0, x.data(), x.size() * 4, &error))
        << error;
    const rt::Arg qa[] = {rt::Arg::Buf(xb), rt::Arg::Buf(xqb), rt::Arg::Buf(xdb), rt::Arg::Val(cols), rt::Arg::Val(T)};
    ASSERT_TRUE(dev->Launch(dev->Kernel("halo_quantize_q8_kblocked", &error), {T}, {256}, qa, &error)) << error;
    const rt::Arg ga[] = {rt::Arg::Buf(qb), rt::Arg::Buf(db), rt::Arg::Buf(xqb), rt::Arg::Buf(xdb), rt::Arg::Buf(yb),
                          rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val(cols), rt::Arg::Val(T)};
    ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gemm_q8a8_128x128", &error), {(T + 127) / 128, (rows + 127) / 128}, {256},
                            ga, &error))
        << error;
    std::vector<std::int8_t> xq(x.size());
    std::vector<float> xd(std::size_t{nblk} * T), y(std::size_t{T} * rows);
    ASSERT_TRUE(dev->Download(xqb, 0, xq.data(), xq.size(), &error) &&
                dev->Download(xdb, 0, xd.data(), xd.size() * 4, &error) &&
                dev->Download(yb, 0, y.data(), y.size() * 4, &error))
        << error;
    for (auto* b : {&qb, &db, &xb, &xqb, &xdb, &yb}) dev->Free(*b);
    for (std::uint32_t t = 0; t < T; ++t)
      for (std::uint32_t b = 0; b < nblk; ++b) {
        float amax = 0.0f;
        for (std::uint32_t c = b * 32; c < b * 32 + 32; ++c) amax = std::max(amax, std::fabs(x[std::size_t{t} * cols + c]));
        const float scale = amax / 127.0f;
        ASSERT_EQ(xd[std::size_t{b} * T + t], scale) << "t " << t << " b " << b;
        for (std::uint32_t c = b * 32; c < b * 32 + 32; ++c) {
          const int want = scale > 0.0f ? static_cast<int>(std::rint(x[std::size_t{t} * cols + c] / scale)) : 0;
          ASSERT_EQ(xq[(std::size_t{c / 64} * T + t) * 64 + c % 64], want) << "t " << t << " c " << c;
        }
      }
    for (std::uint32_t t = 0; t < T; ++t)
      for (std::uint32_t r = 0; r < rows; ++r) {
        double want = 0, mag = 0;
        for (std::uint32_t b = 0; b < nblk; ++b) {
          long isum = 0;
          for (std::uint32_t c = b * 32; c < b * 32 + 32; ++c)
            isum += long(q[std::size_t{r} * cols + c]) * xq[(std::size_t{c / 64} * T + t) * 64 + c % 64];
          _Float16 h;
          std::memcpy(&h, &dw[std::size_t{r} * nblk + b], 2);
          const double p = double(isum) * double(static_cast<float>(h)) * xd[std::size_t{b} * T + t];
          want += p;
          mag += std::fabs(p);
        }
        ASSERT_NEAR(y[std::size_t{t} * rows + r], want, (nblk + 2) * 0x1p-23 * mag + 1e-30)
            << "t " << t << " r " << r << " rows " << rows << " cols " << cols << " T " << T;
      }
  }, {.test_cases = 30});
}

// P4: the FP16-route dense GEMM (Q8_0 dequantized to fp16 while staging,
// fp16 activations, fp32 accumulation), both tile configurations, against
// double sums of the same fp16 operands: fp16(d * q) is exact in the
// kernel's arithmetic, so only fp32 summation order differs. Bound fixed
// in advance: cols * 2^-23 * sum |w x|. Ragged rows and tokens.
// The banded dispatch order (halo_gemm_q8r_f16_lg8) computes the same tiles
// with the same arithmetic as halo_gemm_q8r_f16_l: bit-identical, for 8 and
// 16 row tiles and ragged token counts.
// The mixer down GEMM reading K-blocked activations (halo_gemm_q8r_f16_mb,
// x stored [cols / 64][T][64]) equals halo_gemm_q8r_f16_m on the row-major
// copy bit for bit: same values, same order.
TEST(GpuQuant, BlockedActivationTallGemmIsBitIdentical) {
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << error;
  ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
  for (const std::uint32_t T : {1u, 70u, 200u}) {
    const std::uint32_t rows = 320, cols = 1024, row_bytes = cols / 32 * 34;
    std::vector<std::uint8_t> w(std::size_t{rows} * row_bytes);
    std::uint32_t seed = T * 31 + 7;
    for (auto& b : w) b = static_cast<std::uint8_t>((seed = seed * 1664525u + 1013904223u) >> 24);
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t b = 0; b < cols / 32; ++b) {
        const std::uint16_t h = 0x2000 | (r * 5 + b) % 1024;
        std::memcpy(w.data() + r * row_bytes + b * 34, &h, 2);
      }
    std::vector<_Float16> x(std::size_t{T} * cols), xb(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<_Float16>(std::sin(0.13 * i));
    for (std::uint32_t t = 0; t < T; ++t)
      for (std::uint32_t c = 0; c < cols; ++c) xb[(std::size_t{c / 64} * T + t) * 64 + c % 64] = x[std::size_t{t} * cols + c];
    auto wb = dev->Alloc(w.size(), &error), qs = dev->Alloc(std::size_t{rows} * cols, &error),
         db = dev->Alloc(std::size_t{rows} * (cols / 32) * 2, &error), x1 = dev->Alloc(x.size() * 2, &error),
         x2 = dev->Alloc(x.size() * 2, &error), y1 = dev->Alloc(std::size_t{T} * rows * 4, &error),
         y2 = dev->Alloc(std::size_t{T} * rows * 4, &error);
    ASSERT_TRUE(wb && qs && db && x1 && x2 && y1 && y2 && dev->Upload(wb, 0, w.data(), w.size(), &error) &&
                dev->Upload(x1, 0, x.data(), x.size() * 2, &error) && dev->Upload(x2, 0, xb.data(), xb.size() * 2, &error))
        << error;
    const rt::Arg ra[] = {rt::Arg::Buf(wb), rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Val(rows), rt::Arg::Val(cols),
                          rt::Arg::Val(cols)};
    ASSERT_TRUE(dev->Launch(dev->Kernel("halo_repack_q8_0", &error), {4096}, {256}, ra, &error)) << error;
    for (const auto& [k, xx, y] : {std::tuple{"halo_gemm_q8r_f16_m", &x1, &y1}, std::tuple{"halo_gemm_q8r_f16_mb", &x2, &y2}}) {
      const rt::Arg ga[] = {rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Buf(*xx), rt::Arg::Buf(*y), rt::Arg::Val(rows),
                            rt::Arg::Val(cols), rt::Arg::Val(cols), rt::Arg::Val(T)};
      ASSERT_TRUE(dev->Launch(dev->Kernel(k, &error), {(T + 63) / 64, 1}, {256}, ga, &error)) << error;
    }
    std::vector<float> a(std::size_t{T} * rows), b(a.size());
    ASSERT_TRUE(dev->Download(y1, 0, a.data(), a.size() * 4, &error) && dev->Download(y2, 0, b.data(), b.size() * 4, &error))
        << error;
    for (auto* buf : {&wb, &qs, &db, &x1, &x2, &y1, &y2}) dev->Free(*buf);
    for (std::size_t i = 0; i < a.size(); ++i)
      ASSERT_EQ(std::bit_cast<std::uint32_t>(a[i]), std::bit_cast<std::uint32_t>(b[i])) << "T " << T << " i " << i;
  }
}

TEST(GpuQuant, BandedQ8F16GemmIsBitIdentical) {
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << error;
  ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
  // Also the register-weight form (halo_gemm_q8r_f16_lr8): same values, same order.
  for (const char* kernel2 : {"halo_gemm_q8r_f16_lg8", "halo_gemm_q8r_f16_lr8"})
  for (const std::uint32_t rows : {2048u, 4096u})
    for (const std::uint32_t T : {1u, 130u, 300u}) {
      const std::uint32_t cols = 256, row_bytes = cols / 32 * 34;
      std::vector<std::uint8_t> w(std::size_t{rows} * row_bytes);
      std::uint32_t seed = rows * 7 + T;
      for (auto& b : w) b = static_cast<std::uint8_t>((seed = seed * 1664525u + 1013904223u) >> 24);
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t b = 0; b < cols / 32; ++b) {
          const std::uint16_t h = 0x2000 | (r * 7 + b) % 1024;
          std::memcpy(w.data() + r * row_bytes + b * 34, &h, 2);
        }
      std::vector<_Float16> x(std::size_t{T} * cols);
      for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<_Float16>(std::sin(0.37 * i));
      auto wb = dev->Alloc(w.size(), &error), qs = dev->Alloc(std::size_t{rows} * cols, &error),
           db = dev->Alloc(std::size_t{rows} * (cols / 32) * 2, &error), xb = dev->Alloc(x.size() * 2, &error),
           y1 = dev->Alloc(std::size_t{T} * rows * 4, &error), y2 = dev->Alloc(std::size_t{T} * rows * 4, &error);
      ASSERT_TRUE(wb && qs && db && xb && y1 && y2 && dev->Upload(wb, 0, w.data(), w.size(), &error) &&
                  dev->Upload(xb, 0, x.data(), x.size() * 2, &error))
          << error;
      const rt::Arg ra[] = {rt::Arg::Buf(wb), rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Val(rows), rt::Arg::Val(cols),
                            rt::Arg::Val(cols)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_repack_q8_0", &error), {4096}, {256}, ra, &error)) << error;
      for (const auto& [k, y] : {std::pair{"halo_gemm_q8r_f16_l", &y1}, std::pair{kernel2, &y2}}) {
        const rt::Arg ga[] = {rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Buf(xb), rt::Arg::Buf(*y), rt::Arg::Val(rows),
                              rt::Arg::Val(cols), rt::Arg::Val(cols), rt::Arg::Val(T)};
        ASSERT_TRUE(dev->Launch(dev->Kernel(k, &error), {(T + 127) / 128, rows / 256}, {256}, ga, &error)) << error;
      }
      std::vector<float> a(std::size_t{T} * rows), b(a.size());
      ASSERT_TRUE(dev->Download(y1, 0, a.data(), a.size() * 4, &error) && dev->Download(y2, 0, b.data(), b.size() * 4, &error))
          << error;
      for (auto* buf : {&wb, &qs, &db, &xb, &y1, &y2}) dev->Free(*buf);
      for (std::size_t i = 0; i < a.size(); ++i)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(a[i]), std::bit_cast<std::uint32_t>(b[i])) << kernel2 << " rows " << rows << " T " << T << " i " << i;
    }
}

TEST(GpuQuant, Q8F16GemmMatchesDoubleSums) {
  std::string error;
  const auto f16 = [](float v) { return static_cast<float>(static_cast<_Float16>(v)); };
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const auto rows = tc.draw(gs::booleans()) ? tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 16}))
                                                : tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 600}));
      const int config = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 2}));  // large, small, tall
      const bool large = config == 0, tall = config == 2;
      const auto cols = (config == 1 ? 128 : 64) * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
      const auto T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 300}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const std::size_t row_bytes = cols / 32 * 34;
      std::vector<std::uint8_t> w(rows * row_bytes);
      for (auto& b : w) { seed = seed * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(seed >> 24); }
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t b = 0; b < cols / 32; ++b) {
          const auto h = static_cast<std::uint16_t>(FiniteHalf(tc) & 0xA3FF);  // small scales: fp16 range
          std::memcpy(w.data() + r * row_bytes + b * 34, &h, 2);
        }
      std::vector<float> x(std::size_t{T} * cols);
      for (auto& v : x) { seed = seed * 1664525u + 1013904223u; v = 4.0f * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f); }
      std::vector<_Float16> xh(x.size());
      for (std::size_t i = 0; i < x.size(); ++i) xh[i] = static_cast<_Float16>(x[i]);
      const std::uint32_t qstride = cols;
      auto wb = dev->Alloc(w.size(), &error), qs = dev->Alloc(std::size_t{rows} * qstride, &error),
           db = dev->Alloc(std::size_t{rows} * (cols / 32) * 2, &error), xb = dev->Alloc(xh.size() * 2, &error),
           yb = dev->Alloc(std::size_t{T} * rows * 4, &error);
      ASSERT_TRUE(wb && qs && db && xb && yb) << error;
      ASSERT_TRUE(dev->Upload(wb, 0, w.data(), w.size(), &error) && dev->Upload(xb, 0, xh.data(), xh.size() * 2, &error))
          << error;
      const rt::Arg ra[] = {rt::Arg::Buf(wb), rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Val(rows), rt::Arg::Val(cols),
                            rt::Arg::Val(qstride)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_repack_q8_0", &error), {4096}, {256}, ra, &error)) << error;
      const rt::Arg ga[] = {rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Buf(xb), rt::Arg::Buf(yb), rt::Arg::Val(rows),
                            rt::Arg::Val(cols), rt::Arg::Val(qstride), rt::Arg::Val(T)};
      const std::uint32_t bm = large ? 256 : tall ? 320 : 64, bn = tall ? 64 : 128;
      ASSERT_TRUE(dev->Launch(dev->Kernel(large ? "halo_gemm_q8r_f16_l" : tall ? "halo_gemm_q8r_f16_m" : "halo_gemm_q8r_f16_s", &error),
                              {(T + bn - 1) / bn, (rows + bm - 1) / bm}, {256}, ga, &error))
          << error;
      std::vector<float> y(std::size_t{T} * rows);
      ASSERT_TRUE(dev->Download(yb, 0, y.data(), y.size() * 4, &error)) << error;
      for (auto* b : {&wb, &qs, &db, &xb, &yb}) dev->Free(*b);
      std::vector<float> wr(cols);
      for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
          const std::uint8_t* blk = w.data() + r * row_bytes + (c / 32) * 34;
          std::uint16_t hb;
          std::memcpy(&hb, blk, 2);
          _Float16 dh;
          std::memcpy(&dh, &hb, 2);
          wr[c] = f16(static_cast<float>(dh) * static_cast<float>(static_cast<std::int8_t>(blk[2 + c % 32])));
        }
        for (std::uint32_t t = 0; t < T; ++t) {
          double want = 0, mag = 0;
          for (std::uint32_t c = 0; c < cols; ++c) {
            const double p = double(wr[c]) * f16(x[std::size_t{t} * cols + c]);
            want += p;
            mag += std::fabs(p);
          }
          ASSERT_NEAR(y[std::size_t{t} * rows + r], want, cols * 0x1p-23 * mag + 1e-30)
              << rt::BackendName(backend) << " config " << config << " t " << t << " r " << r << " rows "
              << rows << " cols " << cols << " T " << T;
        }
      }
    }, {.test_cases = 40});
  }
}

// P4: the mixer's up projection fused with the stream mix: weights
// permuted index-major (halo_permute_hc_rows), the GEMM epilogue computes
// mixed[t][i] = sum_s xn[t][s h + i] * sigmoid(gate[t][s h + i]) / 4. Bound
// fixed in advance from the GEMM's: each gate within e = cols * 2^-23 *
// sum |w x|, so mixed within sum_s |xn_s| * 0.25 * e_s / 4 + 1e-6 *
// sum_s |xn_s sigmoid| / 4; the fp16 copy within that plus half an fp16 ulp.
TEST(GpuQuant, FusedMixerUpMatchesReference) {
  std::string error;
  const auto f16 = [](float v) { return static_cast<float>(static_cast<_Float16>(v)); };
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const auto h = 64 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 6}));
      const std::uint32_t rows = 4 * h;
      const auto cols = 64 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 6}));
      const auto T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 300}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      const std::size_t row_bytes = cols / 32 * 34;
      std::vector<std::uint8_t> w(rows * row_bytes);
      for (auto& b : w) { seed = seed * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(seed >> 24); }
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t b = 0; b < cols / 32; ++b) {
          const auto hv = static_cast<std::uint16_t>(FiniteHalf(tc) & 0xA3FF);
          std::memcpy(w.data() + r * row_bytes + b * 34, &hv, 2);
        }
      std::vector<float> lo(std::size_t{T} * cols), xn(std::size_t{T} * rows);
      for (auto& v : lo) v = 4 * rnd();
      for (auto& v : xn) v = 4 * rnd();
      // Rebuild mode (combine_norm skipped xn): the kernel reads a residual,
      // per-(token, stream) scales and gammas, and must form exactly
      // (res * scale) * gamma, the xn the reference below uses.
      const bool rebuild = tc.draw(gs::booleans());
      std::vector<float> rs(std::size_t{T} * 4), nw(rows), resv = xn;
      if (rebuild) {
        for (auto& v : rs) v = 0.25f + std::fabs(rnd());
        for (auto& v : nw) v = 2 * rnd();
        for (std::uint32_t t = 0; t < T; ++t)
          for (std::uint32_t r = 0; r < rows; ++r) {
            const std::size_t at = std::size_t{t} * rows + r;
            xn[at] = resv[at] * rs[t * 4 + r / h] * nw[r];
          }
      }
      std::vector<_Float16> loh(lo.size());
      for (std::size_t i = 0; i < lo.size(); ++i) loh[i] = static_cast<_Float16>(lo[i]);
      const std::uint32_t qstride = cols, nblk = cols / 32;
      auto wb = dev->Alloc(w.size(), &error), qs = dev->Alloc(std::size_t{rows} * qstride, &error),
           db = dev->Alloc(std::size_t{rows} * nblk * 2, &error), pqs = dev->Alloc(std::size_t{rows} * qstride, &error),
           pdb = dev->Alloc(std::size_t{rows} * nblk * 2, &error), xb = dev->Alloc(loh.size() * 2, &error),
           nb = dev->Alloc(xn.size() * 4, &error), yb = dev->Alloc(std::size_t{T} * h * 4, &error),
           y16 = dev->Alloc(std::size_t{T} * h * 2, &error), rsb = dev->Alloc(rs.size() * 4, &error),
           nwb = dev->Alloc(nw.size() * 4, &error);
      ASSERT_TRUE(wb && qs && db && pqs && pdb && xb && nb && yb && y16 && rsb && nwb) << error;
      ASSERT_TRUE(dev->Upload(wb, 0, w.data(), w.size(), &error) && dev->Upload(xb, 0, loh.data(), loh.size() * 2, &error) &&
                  dev->Upload(nb, 0, (rebuild ? resv : xn).data(), xn.size() * 4, &error) &&
                  dev->Upload(rsb, 0, rs.data(), rs.size() * 4, &error) &&
                  dev->Upload(nwb, 0, nw.data(), nw.size() * 4, &error))
          << error;
      const rt::Arg ra[] = {rt::Arg::Buf(wb), rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Val(rows), rt::Arg::Val(cols),
                            rt::Arg::Val(qstride)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_repack_q8_0", &error), {4096}, {256}, ra, &error)) << error;
      const rt::Arg pa[] = {rt::Arg::Buf(qs), rt::Arg::Buf(db), rt::Arg::Buf(pqs), rt::Arg::Buf(pdb), rt::Arg::Val(h),
                            rt::Arg::Val(qstride), rt::Arg::Val(nblk)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_permute_hc_rows", &error), {rows}, {64}, pa, &error)) << error;
      const rt::Arg ga[] = {rt::Arg::Buf(pqs), rt::Arg::Buf(pdb), rt::Arg::Buf(xb), rt::Arg::Buf(yb), rt::Arg::Val(rows),
                            rt::Arg::Val(cols), rt::Arg::Val(qstride), rt::Arg::Val(T), rt::Arg::Buf(nb), rt::Arg::Buf(y16),
                            rebuild ? rt::Arg::Buf(rsb) : rt::Arg::Val<std::uint64_t>(0),
                            rebuild ? rt::Arg::Buf(nwb) : rt::Arg::Val<std::uint64_t>(0)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gemm_q8r_f16_mix", &error), {(T + 127) / 128, (rows + 255) / 256}, {256},
                              ga, &error))
          << error;
      std::vector<float> y(std::size_t{T} * h);
      std::vector<_Float16> yh(y.size());
      ASSERT_TRUE(dev->Download(yb, 0, y.data(), y.size() * 4, &error) &&
                  dev->Download(y16, 0, yh.data(), yh.size() * 2, &error))
          << error;
      {  // halo_gemm_q8r_f16_mixt (row tiles fastest) is the same kernel in another dispatch order: bit-identical
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_gemm_q8r_f16_mixt", &error), {(rows + 255) / 256, (T + 127) / 128},
                                {256}, ga, &error))
            << error;
        std::vector<float> yt(y.size());
        std::vector<_Float16> yth(y.size());
        ASSERT_TRUE(dev->Download(yb, 0, yt.data(), yt.size() * 4, &error) &&
                    dev->Download(y16, 0, yth.data(), yth.size() * 2, &error))
            << error;
        for (std::size_t i = 0; i < y.size(); ++i) {
          ASSERT_EQ(std::bit_cast<std::uint32_t>(yt[i]), std::bit_cast<std::uint32_t>(y[i])) << "mixt " << i;
          ASSERT_EQ(std::bit_cast<std::uint16_t>(yth[i]), std::bit_cast<std::uint16_t>(yh[i])) << "mixt16 " << i;
        }
      }
      for (auto* b : {&wb, &qs, &db, &pqs, &pdb, &xb, &nb, &yb, &y16, &rsb, &nwb}) dev->Free(*b);
      // Weights in fp16 as the kernel stages them, row-major (stream s, index i at row s h + i).
      std::vector<float> wf(std::size_t{rows} * cols);
      for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t c = 0; c < cols; ++c) {
          const std::uint8_t* blk = w.data() + r * row_bytes + (c / 32) * 34;
          std::uint16_t hb;
          std::memcpy(&hb, blk, 2);
          _Float16 dh;
          std::memcpy(&dh, &hb, 2);
          wf[std::size_t{r} * cols + c] = f16(static_cast<float>(dh) * static_cast<float>(static_cast<std::int8_t>(blk[2 + c % 32])));
        }
      const auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
      for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t i = 0; i < h; ++i) {
          double want = 0, bound = 0, mag = 0;
          for (std::uint32_t st = 0; st < 4; ++st) {
            const std::uint32_t r = st * h + i;
            double g = 0, gm = 0;
            for (std::uint32_t c = 0; c < cols; ++c) {
              const double p = double(wf[std::size_t{r} * cols + c]) * f16(lo[std::size_t{t} * cols + c]);
              g += p;
              gm += std::fabs(p);
            }
            const double x = xn[std::size_t{t} * rows + r];
            want += x * sig(g) / 4;
            bound += std::fabs(x) * 0.25 * (cols * 0x1p-23 * gm) / 4;
            mag += std::fabs(x * sig(g)) / 4;
          }
          bound += 1e-6 * mag + 1e-30;
          const std::size_t at = std::size_t{t} * h + i;
          ASSERT_NEAR(y[at], want, bound) << rt::BackendName(backend) << " t " << t << " i " << i << " h " << h
                                          << " cols " << cols << " T " << T << " rebuild " << rebuild;
          ASSERT_NEAR(static_cast<float>(yh[at]), want, bound + std::fabs(want) * 0x1p-11)
              << "fp16 copy t " << t << " i " << i;
        }
    }, {.test_cases = 30});
  }
}

// P4: the fp16-emitting norms against the originals: halo_rmsnorm_rows_h's
// fp32 output bit for bit; both fp16 outputs within one fp16 ulp of the
// fp16 rounding of the original's fp32 result. (First asserted exact; the
// compiler can round the last product straight to fp16, a single rounding
// where the reference rounds twice, moving rare elements by one ulp.)
TEST(GpuKernels, Fp16EmittingNormsMatchOriginals) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const auto d = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 90}));
      const auto period = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
      const auto rows = period * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 40}));
      const auto mode = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 1}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto vec = [&](std::size_t n, float amp) {
        std::vector<float> v(n);
        for (auto& x : v) { seed = seed * 1664525u + 1013904223u; x = amp * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f); }
        return v;
      };
      const auto x = vec(std::size_t{rows} * d, 8.0f), w = vec(std::size_t{period} * d, 2.0f), z = vec(std::size_t{rows} * d, 6.0f);
      std::vector<rt::Buffer> owned;
      const auto up = [&](const void* p, std::size_t bytes) {
        auto b = dev->Alloc(bytes, &error);
        if (p) EXPECT_TRUE(dev->Upload(b, 0, p, bytes, &error));
        owned.push_back(b);
        return b;
      };
      const std::size_t n = std::size_t{rows} * d;
      auto xb = up(x.data(), n * 4), wb = up(w.data(), w.size() * 4), zb = up(z.data(), n * 4), y1 = up(nullptr, n * 4),
           y2 = up(nullptr, n * 4), h2 = up(nullptr, n * 2), g1 = up(x.data(), n * 4), h3 = up(nullptr, n * 2);
      const auto launch = [&](const char* k, std::uint32_t grid, std::initializer_list<rt::Arg> a) {
        ASSERT_TRUE(dev->Launch(dev->Kernel(k, &error), {grid}, {256}, std::span<const rt::Arg>(a.begin(), a.size()), &error)) << k;
      };
      launch("halo_rmsnorm_rows", rows, {rt::Arg::Buf(xb), rt::Arg::Buf(wb), rt::Arg::Buf(y1), rt::Arg::Val(d),
                                         rt::Arg::Val(1e-6f), rt::Arg::Val(d), rt::Arg::Val(period)});
      launch("halo_rmsnorm_rows_h", rows, {rt::Arg::Buf(xb), rt::Arg::Buf(wb), rt::Arg::Buf(y2), rt::Arg::Buf(h2),
                                           rt::Arg::Val(d), rt::Arg::Val(1e-6f), rt::Arg::Val(d), rt::Arg::Val(period),
                                           rt::Arg::Val<std::uint64_t>(0), rt::Arg::Val<std::uint64_t>(0),
                                           rt::Arg::Val(0u), rt::Arg::Val(0u)});
      // Gated norm: the original in place on g1 (a copy of x), the fp16 variant from x.
      launch("halo_gated_norm_h", rows, {rt::Arg::Buf(xb), rt::Arg::Buf(wb), rt::Arg::Buf(zb), rt::Arg::Buf(h3),
                                         rt::Arg::Val(d), rt::Arg::Val(1e-6f), rt::Arg::Val(mode)});
      launch("halo_gated_norm", rows, {rt::Arg::Buf(g1), rt::Arg::Buf(wb), rt::Arg::Buf(zb), rt::Arg::Val(d),
                                       rt::Arg::Val(1e-6f), rt::Arg::Val(mode)});
      std::vector<float> a(n), b(n), g(n);
      std::vector<_Float16> ah(n), gh(n);
      ASSERT_TRUE(dev->Download(y1, 0, a.data(), n * 4, &error) && dev->Download(y2, 0, b.data(), n * 4, &error) &&
                  dev->Download(h2, 0, ah.data(), n * 2, &error) && dev->Download(g1, 0, g.data(), n * 4, &error) &&
                  dev->Download(h3, 0, gh.data(), n * 2, &error))
          << error;
      for (auto& o : owned) dev->Free(o);
      for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(std::bit_cast<std::uint32_t>(a[i]), std::bit_cast<std::uint32_t>(b[i])) << "rmsnorm fp32 " << i;
        const auto ulp = [](_Float16 x, _Float16 y) {  // 0 for equal values (+0 / -0)
          return x == y ? 0
                        : std::abs(static_cast<int>(std::bit_cast<std::int16_t>(x)) -
                                   static_cast<int>(std::bit_cast<std::int16_t>(y)));
        };
        ASSERT_LE(ulp(ah[i], static_cast<_Float16>(a[i])), 1) << "rmsnorm fp16 " << i;
        ASSERT_LE(ulp(gh[i], static_cast<_Float16>(g[i])), 1) << "gated norm fp16 " << i << " mode " << mode;
      }
    }, {.test_cases = 30});
  }
}

// Precision option (stream P): halo_hc_combine_norm_rows_q8 leaves res, xn and
// the K-blocked fp16 copy as halo_hc_combine_norm_rows does (bit for bit), and
// its int8 copy equals the host's per-32 quantization of that xn (amax / 127,
// round to nearest), K-blocked [hc h / 64][T][64] with scales [hc h / 32][T].
TEST(GpuKernels, CombineNormQ8MatchesHostQuantization) {
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  if (!dev) GTEST_SKIP() << error;
  ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
  hegel::test([&](hegel::TestCase& tc) {
    const auto h = 64 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 64}));
    const std::uint32_t hc = 4;
    const auto T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 24}));
    std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
    const auto vec = [&](std::size_t n, float amp) {
      std::vector<float> v(n);
      for (auto& x : v) { seed = seed * 1664525u + 1013904223u; x = amp * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f); }
      return v;
    };
    const std::size_t n = std::size_t{T} * hc * h, nb = n / 32;
    const auto res = vec(n, 6.0f), out = vec(std::size_t{T} * h, 3.0f), inj = vec(std::size_t{T} * hc, 8.0f),
               w = vec(std::size_t{hc} * h, 2.0f);
    std::vector<rt::Buffer> owned;
    const auto up = [&](const void* p, std::size_t bytes) {
      auto b = dev->Alloc(bytes, &error);
      if (p) EXPECT_TRUE(dev->Upload(b, 0, p, bytes, &error));
      owned.push_back(b);
      return b;
    };
    auto r1 = up(res.data(), n * 4), r2 = up(res.data(), n * 4), ob = up(out.data(), out.size() * 4),
         ib = up(inj.data(), inj.size() * 4), wb = up(w.data(), w.size() * 4), y1 = up(nullptr, n * 4),
         y2 = up(nullptr, n * 4), h1 = up(nullptr, n * 2), h2 = up(nullptr, n * 2), qb = up(nullptr, n),
         db = up(nullptr, nb * 4), pp = up(nullptr, std::size_t{T} * hc * 4);
    const auto launch = [&](const char* k, std::initializer_list<rt::Arg> a) {
      ASSERT_TRUE(dev->Launch(dev->Kernel(k, &error), {T * hc}, {256}, std::span<const rt::Arg>(a.begin(), a.size()), &error)) << k;
    };
    launch("halo_hc_combine_norm_rows", {rt::Arg::Buf(r1), rt::Arg::Buf(ob), rt::Arg::Buf(ib), rt::Arg::Buf(wb),
                                         rt::Arg::Buf(y1), rt::Arg::Buf(h1), rt::Arg::Val(hc), rt::Arg::Val(h),
                                         rt::Arg::Val(1e-6f), rt::Arg::Val<std::uint64_t>(0), rt::Arg::Buf(pp),
                                         rt::Arg::Val(0u), rt::Arg::Val<std::uint64_t>(0), rt::Arg::Val(T)});
    launch("halo_hc_combine_norm_rows_q8", {rt::Arg::Buf(r2), rt::Arg::Buf(ob), rt::Arg::Buf(ib), rt::Arg::Buf(wb),
                                            rt::Arg::Buf(y2), rt::Arg::Buf(h2), rt::Arg::Val(hc), rt::Arg::Val(h),
                                            rt::Arg::Val(1e-6f), rt::Arg::Val<std::uint64_t>(0), rt::Arg::Buf(pp),
                                            rt::Arg::Val(0u), rt::Arg::Val<std::uint64_t>(0), rt::Arg::Val(T),
                                            rt::Arg::Buf(qb), rt::Arg::Buf(db)});
    std::vector<float> a1(n), a2(n), b1(n), b2(n), xd(nb);
    std::vector<std::uint16_t> c1(n), c2(n);
    std::vector<std::int8_t> xq(n);
    ASSERT_TRUE(dev->Download(r1, 0, a1.data(), n * 4, &error) && dev->Download(r2, 0, a2.data(), n * 4, &error) &&
                dev->Download(y1, 0, b1.data(), n * 4, &error) && dev->Download(y2, 0, b2.data(), n * 4, &error) &&
                dev->Download(h1, 0, c1.data(), n * 2, &error) && dev->Download(h2, 0, c2.data(), n * 2, &error) &&
                dev->Download(qb, 0, xq.data(), n, &error) && dev->Download(db, 0, xd.data(), nb * 4, &error))
        << error;
    for (auto& b : owned) dev->Free(b);
    ASSERT_EQ(std::memcmp(a1.data(), a2.data(), n * 4), 0) << "residual";
    ASSERT_EQ(std::memcmp(b1.data(), b2.data(), n * 4), 0) << "xn";
    ASSERT_EQ(std::memcmp(c1.data(), c2.data(), n * 2), 0) << "fp16 copy";
    const std::size_t D = std::size_t{hc} * h;
    for (std::uint32_t t = 0; t < T; ++t)
      for (std::size_t b = 0; b < D / 32; ++b) {
        float amax = 0.0f;
        for (std::size_t c = b * 32; c < b * 32 + 32; ++c) amax = std::max(amax, std::fabs(b1[t * D + c]));
        const float scale = amax / 127.0f;
        ASSERT_EQ(xd[b * T + t], scale) << "t " << t << " block " << b;
        for (std::size_t c = b * 32; c < b * 32 + 32; ++c) {
          const int want = scale > 0.0f ? static_cast<int>(std::rint(b1[t * D + c] / scale)) : 0;
          ASSERT_EQ(xq[((c / 64) * T + t) * 64 + c % 64], want) << "t " << t << " c " << c;
        }
      }
  }, {.test_cases = 30});
}

// P4: the combine fused with the next mixer's norm against the two kernels
// (and both norms' inject partial products, reduced, against double sums)
// (halo_hc_combine_rows, then halo_rmsnorm_rows_h): residual and normalized
// rows bit for bit in fp32; the fp16 copy within one ulp (first asserted
// exact: either kernel may round the final product straight to fp16).
TEST(GpuKernels, CombineNormEqualsSeparateKernels) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const auto h = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 128}));
      const auto hc = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4}));
      const auto T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 20}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto vec = [&](std::size_t n, float amp) {
        std::vector<float> v(n);
        for (auto& x : v) { seed = seed * 1664525u + 1013904223u; x = amp * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f); }
        return v;
      };
      const std::size_t n = std::size_t{T} * hc * h;
      const auto res = vec(n, 6.0f), out = vec(std::size_t{T} * h, 3.0f), inj = vec(std::size_t{T} * hc, 8.0f),
                 w = vec(std::size_t{hc} * h, 2.0f);
      std::vector<rt::Buffer> owned;
      const auto up = [&](const void* p, std::size_t bytes) {
        auto b = dev->Alloc(bytes, &error);
        if (p) EXPECT_TRUE(dev->Upload(b, 0, p, bytes, &error));
        owned.push_back(b);
        return b;
      };
      const auto irows = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
      const auto iw = vec(std::size_t{irows} * hc * h, 1.0f);
      auto iwb = up(iw.data(), iw.size() * 4), p1 = up(nullptr, std::size_t{T} * hc * irows * 4),
           p2 = up(nullptr, std::size_t{T} * hc * irows * 4), j1 = up(nullptr, std::size_t{T} * irows * 4),
           j2 = up(nullptr, std::size_t{T} * irows * 4);
      auto r1 = up(res.data(), n * 4), r2 = up(res.data(), n * 4), ob = up(out.data(), out.size() * 4),
           ib = up(inj.data(), inj.size() * 4), wb = up(w.data(), w.size() * 4), y1 = up(nullptr, n * 4),
           y2 = up(nullptr, n * 4), h1 = up(nullptr, n * 2), h2 = up(nullptr, n * 2);
      const auto launch = [&](const char* k, std::uint32_t grid, std::initializer_list<rt::Arg> a) {
        ASSERT_TRUE(dev->Launch(dev->Kernel(k, &error), {grid}, {256}, std::span<const rt::Arg>(a.begin(), a.size()), &error)) << k;
      };
      launch("halo_hc_combine_rows", static_cast<std::uint32_t>((n + 255) / 256),
             {rt::Arg::Buf(r1), rt::Arg::Buf(ob), rt::Arg::Buf(ib), rt::Arg::Val(hc), rt::Arg::Val(h), rt::Arg::Val(T)});
      launch("halo_rmsnorm_rows_h", T * hc, {rt::Arg::Buf(r1), rt::Arg::Buf(wb), rt::Arg::Buf(y1), rt::Arg::Buf(h1),
                                             rt::Arg::Val(h), rt::Arg::Val(1e-6f), rt::Arg::Val(h), rt::Arg::Val(hc),
                                             rt::Arg::Buf(iwb), rt::Arg::Buf(p1), rt::Arg::Val(irows), rt::Arg::Val(0u)});
      launch("halo_hc_combine_norm_rows", T * hc, {rt::Arg::Buf(r2), rt::Arg::Buf(ob), rt::Arg::Buf(ib), rt::Arg::Buf(wb),
                                                   rt::Arg::Buf(y2), rt::Arg::Buf(h2), rt::Arg::Val(hc), rt::Arg::Val(h),
                                                   rt::Arg::Val(1e-6f), rt::Arg::Buf(iwb), rt::Arg::Buf(p2),
                                                   rt::Arg::Val(irows), rt::Arg::Val<std::uint64_t>(0), rt::Arg::Val(0u)});
      // The same with xn skipped: per-row scales instead, everything else unchanged.
      auto r3 = up(res.data(), n * 4), h3 = up(nullptr, n * 2), p3 = up(nullptr, std::size_t{T} * hc * irows * 4),
           sb = up(nullptr, std::size_t{T} * hc * 4);
      launch("halo_hc_combine_norm_rows", T * hc, {rt::Arg::Buf(r3), rt::Arg::Buf(ob), rt::Arg::Buf(ib), rt::Arg::Buf(wb),
                                                   rt::Arg::Val<std::uint64_t>(0), rt::Arg::Buf(h3), rt::Arg::Val(hc),
                                                   rt::Arg::Val(h), rt::Arg::Val(1e-6f), rt::Arg::Buf(iwb),
                                                   rt::Arg::Buf(p3), rt::Arg::Val(irows), rt::Arg::Buf(sb), rt::Arg::Val(0u)});
      // The fp16 streams K-blocked ([cols / 64][T][64], when cols % 64 == 0): the same values, permuted.
      std::optional<std::vector<_Float16>> blocked;
      if ((hc * h) % 64 == 0) {
        auto r4 = up(res.data(), n * 4), h4 = up(nullptr, n * 2), p4 = up(nullptr, std::size_t{T} * hc * irows * 4),
             sb4 = up(nullptr, std::size_t{T} * hc * 4);
        launch("halo_hc_combine_norm_rows", T * hc, {rt::Arg::Buf(r4), rt::Arg::Buf(ob), rt::Arg::Buf(ib), rt::Arg::Buf(wb),
                                                     rt::Arg::Val<std::uint64_t>(0), rt::Arg::Buf(h4), rt::Arg::Val(hc),
                                                     rt::Arg::Val(h), rt::Arg::Val(1e-6f), rt::Arg::Buf(iwb),
                                                     rt::Arg::Buf(p4), rt::Arg::Val(irows), rt::Arg::Buf(sb4), rt::Arg::Val(T)});
        blocked.emplace(n);
        ASSERT_TRUE(dev->Download(h4, 0, blocked->data(), n * 2, &error)) << error;
      }
      launch("halo_inject_reduce", (T * irows + 255) / 256,
             {rt::Arg::Buf(p1), rt::Arg::Buf(j1), rt::Arg::Val(hc), rt::Arg::Val(irows), rt::Arg::Val(T)});
      launch("halo_inject_reduce", (T * irows + 255) / 256,
             {rt::Arg::Buf(p2), rt::Arg::Buf(j2), rt::Arg::Val(hc), rt::Arg::Val(irows), rt::Arg::Val(T)});
      std::vector<float> a1(n), a2(n), b1(n), b2(n);
      std::vector<std::uint16_t> c1(n), c2(n);
      ASSERT_TRUE(dev->Download(r1, 0, a1.data(), n * 4, &error) && dev->Download(r2, 0, a2.data(), n * 4, &error) &&
                  dev->Download(y1, 0, b1.data(), n * 4, &error) && dev->Download(y2, 0, b2.data(), n * 4, &error) &&
                  dev->Download(h1, 0, c1.data(), n * 2, &error) && dev->Download(h2, 0, c2.data(), n * 2, &error))
          << error;
      // The inject partial products: both kernels' (reduced) against double sums
      // over the normalized rows, with the fp32 summation bound.
      std::vector<float> in1(std::size_t{T} * irows), in2(in1.size());
      ASSERT_TRUE(dev->Download(j1, 0, in1.data(), in1.size() * 4, &error) &&
                  dev->Download(j2, 0, in2.data(), in2.size() * 4, &error))
          << error;
      for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t r = 0; r < irows; ++r) {
          double want = 0, mag = 0;
          for (std::size_t c = 0; c < std::size_t{hc} * h; ++c) {
            const double p = double(iw[r * hc * h + c]) * b1[std::size_t{t} * hc * h + c];
            want += p;
            mag += std::fabs(p);
          }
          const double bound = double(hc) * h * 0x1p-23 * mag + 1e-30;
          ASSERT_NEAR(in1[t * irows + r], want, bound) << "inject (norm kernel) t " << t << " r " << r;
          ASSERT_NEAR(in2[t * irows + r], want, bound) << "inject (combine+norm) t " << t << " r " << r;
        }
      std::vector<float> a3(n), q2(std::size_t{T} * hc * irows), q3(q2.size()), sc(std::size_t{T} * hc);
      std::vector<std::uint16_t> c3(n);
      ASSERT_TRUE(dev->Download(r3, 0, a3.data(), n * 4, &error) && dev->Download(h3, 0, c3.data(), n * 2, &error) &&
                  dev->Download(p2, 0, q2.data(), q2.size() * 4, &error) &&
                  dev->Download(p3, 0, q3.data(), q3.size() * 4, &error) && dev->Download(sb, 0, sc.data(), sc.size() * 4, &error))
          << error;
      for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(std::bit_cast<std::uint32_t>(a3[i]), std::bit_cast<std::uint32_t>(a2[i])) << "skipped-xn residual " << i;
        ASSERT_EQ(c3[i], c2[i]) << "skipped-xn fp16 " << i;
        // The rebuild the mix epilogue does: (res * scale) * gamma, exactly the xn not written.
        const std::size_t v = i / h;
        const float rebuilt = a3[i] * sc[v] * w[(v % hc) * h + i % h];
        ASSERT_EQ(std::bit_cast<std::uint32_t>(rebuilt), std::bit_cast<std::uint32_t>(b2[i])) << "rebuilt xn " << i;
      }
      for (std::size_t i = 0; i < q2.size(); ++i)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(q3[i]), std::bit_cast<std::uint32_t>(q2[i])) << "skipped-xn inject " << i;
      if (blocked)
        for (std::size_t i = 0; i < n; ++i) {  // element (t, c) of the row-major streams sits at XhBlocked(t, c)
          const std::size_t t = i / (hc * h), cc = i % (hc * h), at = ((cc / 64) * T + t) * 64 + cc % 64;
          ASSERT_EQ(std::bit_cast<std::uint16_t>((*blocked)[at]), c3[i]) << "blocked fp16 " << i;
        }
      for (auto& o : owned) dev->Free(o);
      for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(std::bit_cast<std::uint32_t>(a1[i]), std::bit_cast<std::uint32_t>(a2[i])) << "residual " << i;
        ASSERT_EQ(std::bit_cast<std::uint32_t>(b1[i]), std::bit_cast<std::uint32_t>(b2[i])) << "normalized " << i;
        // Equal values (including +0 / -0) or adjacent fp16 codes.
        const auto h1 = std::bit_cast<_Float16>(c1[i]), h2 = std::bit_cast<_Float16>(c2[i]);
        ASSERT_TRUE(h1 == h2 || std::abs(static_cast<int>(std::bit_cast<std::int16_t>(c1[i])) -
                                         static_cast<int>(std::bit_cast<std::int16_t>(c2[i]))) <= 1)
            << "fp16 " << i << ": " << c1[i] << " vs " << c2[i] << ", fp32 " << b1[i] << " / " << b2[i];
      }
    }, {.test_cases = 40});
  }
}
