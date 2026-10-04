// Properties of the CPU quant oracle, which every GPU kernel is later checked
// against. Each format has two independent code paths in quant.cpp
// (dequantize, and the fused dot product); they must agree on arbitrary
// blocks, and storage sizes must match the block layouts.

#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "core/quant.hpp"

namespace gs = hegel::generators;
using halo::core::GgmlType;
namespace q = halo::quant;

namespace {

// A finite FP16 bit pattern with a modest exponent, so dot products stay far
// from overflow and the comparison measures the code paths, not float range.
std::uint16_t DrawScale(hegel::TestCase& tc) {
  const auto sign = tc.draw(gs::integers<std::uint16_t>({.min_value = 0, .max_value = 1}));
  const auto exp = tc.draw(gs::integers<std::uint16_t>({.min_value = 5, .max_value = 18}));
  const auto mant = tc.draw(gs::integers<std::uint16_t>({.min_value = 0, .max_value = 1023}));
  return static_cast<std::uint16_t>((sign << 15) | (exp << 10) | mant);
}

std::vector<std::uint8_t> DrawBlocks(hegel::TestCase& tc, GgmlType type, std::size_t blocks) {
  const std::size_t k = blocks * q::QuantizedBlockElements(type);
  std::vector<std::uint8_t> bytes(q::EncodedSizeBytes(type, k));
  auto seed = tc.draw(gs::binary({.min_size = 1, .max_size = 64}));
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = seed[i % seed.size()] ^ static_cast<std::uint8_t>(i * 131);
  // Overwrite the fp16 scale fields with sane values.
  const std::size_t bb = bytes.size() / blocks;
  for (std::size_t b = 0; b < blocks; ++b) {
    auto* blk = bytes.data() + b * bb;
    auto put = [&](std::size_t off) { const auto s = DrawScale(tc); std::memcpy(blk + off, &s, 2); };
    switch (type) {
      case GgmlType::kQ8_0: put(0); break;
      case GgmlType::kQ4_K: case GgmlType::kQ5_K: put(0); put(2); break;
      case GgmlType::kQ6_K: put(offsetof(q::block_q6_K, d)); break;
      case GgmlType::kIQ4_NL: case GgmlType::kIQ4_XS: put(0); break;
      default: break;
    }
  }
  return bytes;
}

}  // namespace

TEST(Quant, DotProductAgreesWithDequantize) {
  hegel::test([](hegel::TestCase& tc) {
    const auto type = tc.draw(gs::sampled_from({GgmlType::kQ8_0, GgmlType::kQ4_K, GgmlType::kQ5_K,
                                                GgmlType::kQ6_K, GgmlType::kIQ4_NL, GgmlType::kIQ4_XS}));
    const auto blocks = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 4}));
    const std::size_t k = blocks * q::QuantizedBlockElements(type);
    const auto row = DrawBlocks(tc, type, blocks);
    std::vector<float> vec(k);
    const auto vseed = tc.draw(gs::vectors(gs::floats<float>({.min_value = -2.0F, .max_value = 2.0F}),
                                           {.min_size = 1, .max_size = 16}));
    for (std::size_t i = 0; i < k; ++i) vec[i] = vseed[i % vseed.size()];

    std::vector<float> deq(k);
    ASSERT_TRUE(q::DequantizeRow(type, row.data(), deq.data(), k));
    double ref = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      ref += static_cast<double>(deq[i]) * vec[i];
      mag += std::fabs(static_cast<double>(deq[i]) * vec[i]);
      ASSERT_TRUE(std::isfinite(deq[i]));
    }
    float dot = 0.0F;
    switch (type) {
      case GgmlType::kQ8_0: dot = q::DotProductQ8_0(row.data(), vec, k); break;
      case GgmlType::kQ4_K: dot = q::DotProductQ4_K(row.data(), vec, k); break;
      case GgmlType::kQ5_K: dot = q::DotProductQ5_K(row.data(), vec, k); break;
      case GgmlType::kQ6_K: dot = q::DotProductQ6_K(row.data(), vec, k); break;
      case GgmlType::kIQ4_NL: dot = q::DotProductIQ4_NL(row.data(), vec, k); break;
      default: dot = q::DotProductIQ4_XS(row.data(), vec, k); break;
    }
    // fp32 accumulation over k <= 1024 terms: bound by the absolute sum.
    ASSERT_LE(std::fabs(dot - ref), 1e-5 * mag + 1e-6) << "type " << static_cast<int>(type);
  });
}

TEST(Quant, Q8_0IsScaleTimesInt) {
  hegel::test([](hegel::TestCase& tc) {
    q::block_q8_0 b{};
    b.d = DrawScale(tc);
    for (auto& v : b.qs) v = tc.draw(gs::integers<std::int8_t>());
    float out[32];
    q::DequantizeQ8_0(&b, out, 32);
    const float d = q::Fp16ToFloat(b.d);
    for (int i = 0; i < 32; ++i) ASSERT_EQ(out[i], d * static_cast<float>(b.qs[i]));
  });
}

TEST(Quant, Fp16AndBf16ConvertExactly) {
  hegel::test([](hegel::TestCase& tc) {
    const auto f = tc.draw(gs::floats<float>({.allow_nan = false, .allow_infinity = false}));
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const auto bf = static_cast<std::uint16_t>(bits >> 16);
    std::uint32_t back = static_cast<std::uint32_t>(bf) << 16;
    float expect;
    std::memcpy(&expect, &back, 4);
    ASSERT_EQ(q::Bf16ToFloat(bf), expect);
    // fp16: every finite pattern converts to the value its fields denote.
    const auto h = tc.draw(gs::integers<std::uint16_t>());
    const int exp = (h >> 10) & 31, mant = h & 1023, sign = h >> 15;
    if (exp == 31) return;
    const double v = exp == 0 ? std::ldexp(mant, -24) : std::ldexp(1024 + mant, exp - 25);
    ASSERT_EQ(static_cast<double>(q::Fp16ToFloat(h)), sign ? -v : v);
  });
}

TEST(Quant, EncodedSizesMatchBlockLayouts) {
  EXPECT_EQ(q::EncodedSizeBytes(GgmlType::kQ4_K, 256), sizeof(q::block_q4_K));
  EXPECT_EQ(q::EncodedSizeBytes(GgmlType::kQ8_0, 64), 2 * sizeof(q::block_q8_0));
  EXPECT_EQ(q::EncodedSizeBytes(GgmlType::kQ4_K, 255), 0U);
  EXPECT_EQ(q::EncodedSizeBytes(GgmlType::kBF16, 3), 6U);
  float out[32];
  EXPECT_FALSE(q::DequantizeRow(GgmlType::kQ2_K, nullptr, out, 256));
}

// Regression (found by UBSan through the n-gram table test): rows can start at
// any byte offset, and every format must decode identically there.
TEST(Quant, DequantizeIsOffsetIndependent) {
  hegel::test([](hegel::TestCase& tc) {
    const auto type = tc.draw(gs::sampled_from({GgmlType::kF16, GgmlType::kBF16, GgmlType::kF32,
                                                GgmlType::kQ8_0, GgmlType::kQ4_K, GgmlType::kIQ4_NL}));
    const std::size_t block = std::max<std::size_t>(1, q::QuantizedBlockElements(type));
    const std::size_t k = block * tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 3}));
    const auto row = type == GgmlType::kF32 || type == GgmlType::kF16 || type == GgmlType::kBF16
                         ? tc.draw(gs::binary({.min_size = q::EncodedSizeBytes(type, k),
                                               .max_size = q::EncodedSizeBytes(type, k)}))
                         : DrawBlocks(tc, type, k / block);
    const auto shift = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 7}));
    std::vector<std::uint8_t> shifted(row.size() + shift);
    std::memcpy(shifted.data() + shift, row.data(), row.size());
    std::vector<float> a(k), b(k);
    ASSERT_TRUE(q::DequantizeRow(type, row.data(), a.data(), k));
    ASSERT_TRUE(q::DequantizeRow(type, shifted.data() + shift, b.data(), k));
    ASSERT_EQ(std::memcmp(a.data(), b.data(), k * sizeof(float)), 0);
  });
}
