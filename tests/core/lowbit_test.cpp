// Simulated low-bit codes for the CODESIGN first experiment (src/core/lowbit).
// Each container is decoded with the engine's own CPU dequantizers, so the
// properties hold for what the engine loads, not for the encoder's intent.
// Error bounds were fixed before the first run.

#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include "core/lowbit.hpp"
#include "core/quant.hpp"
#include "model/cpu_ops.hpp"

namespace gs = hegel::generators;
using halo::core::GgmlType;
namespace lowbit = halo::quant::lowbit;

namespace {

std::vector<float> Gaussian(std::uint32_t seed, std::size_t n, float sigma, bool outliers) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; i += 2) {
    seed = seed * 1664525u + 1013904223u;
    const double u1 = (static_cast<double>(seed >> 8) + 1.0) / 16777217.0;
    seed = seed * 1664525u + 1013904223u;
    const double u2 = static_cast<double>(seed >> 8) / 16777216.0;
    const double r = std::sqrt(-2.0 * std::log(u1));
    v[i] = static_cast<float>(sigma * r * std::cos(6.283185307179586 * u2));
    if (i + 1 < n) v[i + 1] = static_cast<float>(sigma * r * std::sin(6.283185307179586 * u2));
  }
  if (outliers)
    for (std::size_t i = 0; i < n; i += 97) v[i] *= 12.0f;
  return v;
}

double RelErr(const std::vector<float>& got, const std::vector<float>& want) {
  double e = 0, r = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    e += (static_cast<double>(got[i]) - want[i]) * (static_cast<double>(got[i]) - want[i]);
    r += static_cast<double>(want[i]) * want[i];
  }
  return std::sqrt(e / r);
}

std::vector<float> Decode(GgmlType type, const std::vector<std::uint8_t>& bytes, std::size_t cols) {
  std::vector<float> out(cols);
  halo::model::TensorRef t;
  t.data = bytes.data();
  t.type = type;
  t.cols = cols;
  halo::model::cpu::DequantizeRow(t, 0, 0, out.data());
  return out;
}

std::size_t RowBytes(GgmlType type, std::size_t cols) {
  return type == GgmlType::kQ5_1 ? cols / 32 * 24 : halo::quant::QuantizedRowBytes(type, cols);
}

}  // namespace

TEST(Lowbit, FwhtIsAnOrthogonalInvolution) {
  hegel::test([](hegel::TestCase& tc) {
    const std::size_t n = std::size_t{1} << tc.draw(gs::integers<int>({.min_value = 0, .max_value = 9}));
    auto x = Gaussian(tc.draw(gs::integers<std::uint32_t>()), n, 1.0f, false);
    const auto orig = x;
    lowbit::Fwht(x.data(), n);
    double a = 0, b = 0;
    for (std::size_t i = 0; i < n; ++i) {
      a += static_cast<double>(orig[i]) * orig[i];
      b += static_cast<double>(x[i]) * x[i];
    }
    ASSERT_NEAR(b, a, 1e-5 * std::max(1.0, a));
    lowbit::Fwht(x.data(), n);
    for (std::size_t i = 0; i < n; ++i) ASSERT_NEAR(x[i], orig[i], 1e-5f) << i;
  }, {.test_cases = 100});
}

TEST(Lowbit, HadamardBlockDividesRows) {
  EXPECT_EQ(lowbit::HadamardBlock(2560), 128u);
  EXPECT_EQ(lowbit::HadamardBlock(10240), 128u);
  EXPECT_EQ(lowbit::HadamardBlock(320), 64u);
  EXPECT_EQ(lowbit::HadamardBlock(640), 128u);
  EXPECT_EQ(lowbit::HadamardBlock(96), 32u);
}

// The stored row, rotated back into the code's basis, holds at most 16
// values per group (the IQ4_NL codebook times one fp16 scale), so the
// quantization happened in the rotated basis; and its error on Gaussian rows
// (with or without outliers) is at most 0.15 relative (4-bit codes of a
// Gaussian measure ~0.1).
TEST(Lowbit, RotatedQ4IsA4BitCodeInTheRotatedBasis) {
  hegel::test([](hegel::TestCase& tc) {
    const std::size_t cols = tc.draw(gs::sampled_from(std::vector<std::size_t>{128, 320, 640, 2560}));
    const bool outliers = tc.draw(gs::booleans());
    const auto w = Gaussian(tc.draw(gs::integers<std::uint32_t>()), cols, 0.02f, outliers);
    std::vector<float> signs(cols), out(cols);
    lowbit::RotationSigns(tc.draw(gs::integers<std::uint64_t>()), signs);
    lowbit::RotatedQ4Row(w.data(), cols, signs.data(), out.data());
    ASSERT_LT(RelErr(out, w), 0.15) << "cols " << cols << " outliers " << outliers;
    const std::size_t b = lowbit::HadamardBlock(cols);
    std::vector<float> r(cols);
    for (std::size_t i = 0; i < cols; ++i) r[i] = out[i] * signs[i];
    for (std::size_t g = 0; g < cols; g += b) {
      lowbit::Fwht(r.data() + g, b);
      float amax = 0;
      for (std::size_t i = 0; i < b; ++i) amax = std::max(amax, std::fabs(r[g + i]));
      std::set<long> levels;
      for (std::size_t i = 0; i < b; ++i) levels.insert(std::lround(r[g + i] / amax * 1e4));
      ASSERT_LE(levels.size(), 16u) << "group " << g / b;
    }
  }, {.test_cases = 60});
}

// Rotation earns its keep where it should: rows with outliers quantize
// better rotated than with the same 4-bit code applied directly.
TEST(Lowbit, RotationHelpsRowsWithOutliers) {
  const std::size_t cols = 2560;
  double rotated = 0, plain = 0;
  for (std::uint32_t seed = 1; seed <= 20; ++seed) {
    const auto w = Gaussian(seed, cols, 0.02f, true);
    std::vector<float> signs(cols), ones(cols, 1.0f), out(cols);
    lowbit::RotationSigns(seed, signs);
    lowbit::RotatedQ4Row(w.data(), cols, signs.data(), out.data());
    rotated += RelErr(out, w);
    // "No rotation": undo the transform first, so the code sees w itself.
    std::vector<float> pre(w);
    for (std::size_t g = 0; g < cols; g += 128) lowbit::Fwht(pre.data() + g, 128);
    lowbit::RotatedQ4Row(pre.data(), cols, ones.data(), out.data());
    for (std::size_t g = 0; g < cols; g += 128) lowbit::Fwht(out.data() + g, 128);
    plain += RelErr(out, w);
  }
  EXPECT_LT(rotated, 0.8 * plain) << rotated / 20 << " vs " << plain / 20;
}

TEST(Lowbit, Q8ContainerErrorIsHalfAStep) {
  hegel::test([](hegel::TestCase& tc) {
    const std::size_t cols = 32 * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 40}));
    const auto x = Gaussian(tc.draw(gs::integers<std::uint32_t>()), cols, 0.02f, tc.draw(gs::booleans()));
    std::vector<std::uint8_t> bytes(cols / 32 * 34);
    lowbit::EncodeQ8_0(x.data(), cols, bytes.data());
    const auto y = Decode(GgmlType::kQ8_0, bytes, cols);
    for (std::size_t j = 0; j < cols / 32; ++j) {
      float amax = 0;
      for (std::size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[32 * j + i]));
      for (std::size_t i = 0; i < 32; ++i)
        ASSERT_LE(std::fabs(y[32 * j + i] - x[32 * j + i]), amax / 127.0f * 0.5f * 1.01f + 1e-12f);
    }
  }, {.test_cases = 100});
}

// Arm D: decoding the container gives a 2-bit code exactly: in every 32
// values at most four distinct levels on one uniform grid. Error on Gaussian
// rows at most 0.45 relative (a uniform 2-bit code of a Gaussian: ~0.35).
TEST(Lowbit, TwoBitCodeIsExactInEveryContainer) {
  hegel::test([](hegel::TestCase& tc) {
    const auto type = tc.draw(gs::sampled_from(
        std::vector<GgmlType>{GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ5_1, GgmlType::kQ8_0}));
    const bool k = type == GgmlType::kQ4_K || type == GgmlType::kQ5_K;
    const std::size_t cols = k ? 256 * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 10}))
                               : 32 * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 80}));
    const bool outliers = tc.draw(gs::booleans());
    auto x = Gaussian(tc.draw(gs::integers<std::uint32_t>()), cols, 0.02f, outliers);
    if (tc.draw(gs::booleans()))  // all-positive groups (a = 0) and constant groups
      for (std::size_t i = 0; i < std::min<std::size_t>(64, cols); ++i) x[i] = i < 32 ? std::fabs(x[i]) : 0.01f;
    std::vector<std::uint8_t> bytes(RowBytes(type, cols));
    ASSERT_TRUE(lowbit::Encode2Bit(type, x.data(), cols, bytes.data()));
    const auto y = Decode(type, bytes, cols);
    for (std::size_t g = 0; g < cols / 32; ++g) {
      std::vector<float> lv(y.begin() + 32 * g, y.begin() + 32 * (g + 1));
      std::sort(lv.begin(), lv.end());
      lv.erase(std::unique(lv.begin(), lv.end()), lv.end());
      ASSERT_LE(lv.size(), 4u) << "group " << g;
      // Levels a + s l, l in 0..3, some possibly unused: every level sits an
      // integer number (<= 3) of the smallest gap above the lowest.
      float gap = INFINITY;
      for (std::size_t i = 1; i < lv.size(); ++i) gap = std::min(gap, lv[i] - lv[i - 1]);
      for (std::size_t i = 1; i < lv.size(); ++i) {
        const float r = (lv[i] - lv[0]) / gap;
        ASSERT_NEAR(r, std::nearbyint(r), 1e-4f) << "group " << g;
        ASSERT_LE(std::nearbyint(r), 3.0f) << "group " << g;
      }
    }
    if (!outliers) ASSERT_LT(RelErr(y, x), 0.45) << halo::core::ToString(type) << " cols " << cols;
  }, {.test_cases = 200});
}

TEST(Lowbit, TwoBitRejectsOtherContainers) {
  std::vector<float> x(256, 0.5f);
  std::vector<std::uint8_t> out(1024);
  EXPECT_FALSE(lowbit::Encode2Bit(GgmlType::kQ6_K, x.data(), 256, out.data()));
  EXPECT_FALSE(lowbit::Encode2Bit(GgmlType::kQ4_K, x.data(), 160, out.data()));
  EXPECT_DOUBLE_EQ(lowbit::Code2BitBpw(GgmlType::kQ4_K, 2560), 2.5);
  EXPECT_DOUBLE_EQ(lowbit::Code2BitBpw(GgmlType::kQ5_1, 640), 2.425);
}
