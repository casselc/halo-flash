// Properties of the CPU quant oracle, which every GPU kernel is later checked
// against. Each format has two independent code paths in quant.cpp
// (dequantize, and the fused dot product); they must agree on arbitrary
// blocks, and storage sizes must match the block layouts.

#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "core/quant.hpp"
#include "model/cpu_ops.hpp"

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

// ---------------------------------------------------------------------------
// Independent reference: tests/core/data/quant.tsv holds blocks dequantized by
// llama.cpp's gguf-py (tools/gen_quant_fixtures.py). The properties above
// compare two halo code paths that share their unpacking helpers, so a slip in
// a shared helper passes them (Mull showed it); this compares with numpy.

namespace {

std::vector<std::uint8_t> Unhex(std::string_view h) {
  std::vector<std::uint8_t> out(h.size() / 2);
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<std::uint8_t>(std::stoi(std::string(h.substr(2 * i, 2)), nullptr, 16));
  return out;
}

struct QuantFixture {
  std::string type;
  std::vector<std::uint8_t> block;
  std::vector<float> values;
};

std::vector<QuantFixture> LoadQuantFixtures() {
  std::ifstream f(std::string(HALO_CORE_TEST_DATA_DIR) + "/quant.tsv");
  EXPECT_TRUE(f.good());
  std::vector<QuantFixture> rows;
  std::string line;
  while (std::getline(f, line)) {
    const auto a = line.find('\t'), b = line.find('\t', a + 1);
    QuantFixture r{line.substr(0, a), Unhex(std::string_view(line).substr(a + 1, b - a - 1)), {}};
    const auto bits = Unhex(std::string_view(line).substr(b + 1));
    r.values.resize(bits.size() / 4);
    std::memcpy(r.values.data(), bits.data(), bits.size());
    rows.push_back(std::move(r));
  }
  return rows;
}

GgmlType TypeNamed(const std::string& name) {
  for (auto t : {GgmlType::kF16, GgmlType::kBF16, GgmlType::kQ8_0, GgmlType::kQ5_1, GgmlType::kQ3_K,
                 GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K, GgmlType::kIQ4_NL,
                 GgmlType::kIQ4_XS, GgmlType::kIQ3_S})
    if (halo::core::ToString(t) == name) return t;
  ADD_FAILURE() << "unknown type " << name;
  return GgmlType::kF32;
}

// The fused dot product of `row`, or NaN for a format without one.
float FusedDot(GgmlType type, const void* row, std::span<const float> v, std::size_t k) {
  switch (type) {
    case GgmlType::kQ8_0: return q::DotProductQ8_0(row, v, k);
    case GgmlType::kQ3_K: return q::DotProductQ3_K(row, v, k);
    case GgmlType::kQ4_K: return q::DotProductQ4_K(row, v, k);
    case GgmlType::kQ5_K: return q::DotProductQ5_K(row, v, k);
    case GgmlType::kQ6_K: return q::DotProductQ6_K(row, v, k);
    case GgmlType::kIQ4_NL: return q::DotProductIQ4_NL(row, v, k);
    case GgmlType::kIQ4_XS: return q::DotProductIQ4_XS(row, v, k);
    case GgmlType::kIQ3_S: return q::DotProductIQ3_S(row, v, k);
    default: return std::nanf("");
  }
}

}  // namespace

// Decodes `r` and its fused dot product against the reference values. Output
// and vector live in larger buffers whose tails are NaN, so touching anything
// past element k (a loop bound off by one) shows up in the results.
void CheckRow(const QuantFixture& r) {
  const auto type = TypeNamed(r.type);
  const std::size_t k = r.values.size();
  const float poison = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> got(k + 64, poison);
  if (type == GgmlType::kQ5_1) {
    // Q5_1 is decoded only by the model's CPU oracle.
    halo::model::TensorRef t{.data = r.block.data(), .type = type, .cols = k};
    halo::model::cpu::DequantizeRow(t, 0, 0, got.data());
  } else {
    ASSERT_TRUE(q::DequantizeRow(type, r.block.data(), got.data(), k)) << r.type;
  }
  for (std::size_t i = k; i < got.size(); ++i) ASSERT_TRUE(std::isnan(got[i])) << r.type << " wrote past k";
  double ref = 0.0, mag = 0.0;
  std::vector<float> v(k + 512, poison);
  for (std::size_t i = 0; i < k; ++i) {
    // numpy and C evaluate d * scale * q in different orders: allow a few ulps.
    ASSERT_NEAR(got[i], r.values[i], 4e-7 * std::fabs(r.values[i])) << r.type << " element " << i;
    v[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25F;
    ref += static_cast<double>(r.values[i]) * v[i];
    mag += std::fabs(static_cast<double>(r.values[i]) * v[i]);
  }
  const float dot = FusedDot(type, r.block.data(), std::span<const float>(v.data(), k), k);
  if (type != GgmlType::kF16 && type != GgmlType::kBF16 && type != GgmlType::kQ5_1)
    ASSERT_LE(std::fabs(dot - ref), 1e-5 * mag + 1e-6) << r.type << " (" << k << " elements)";
}

TEST(Quant, DequantizeMatchesGgufPy) {
  const auto rows = LoadQuantFixtures();
  ASSERT_EQ(rows.size(), 132U);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    CheckRow(rows[i]);
    if (i + 1 < rows.size() && rows[i + 1].type == rows[i].type) {
      // Two blocks in one row: block indexing and per-block loops.
      QuantFixture two = rows[i];
      two.block.insert(two.block.end(), rows[i + 1].block.begin(), rows[i + 1].block.end());
      two.values.insert(two.values.end(), rows[i + 1].values.begin(), rows[i + 1].values.end());
      CheckRow(two);
    }
  }
}

TEST(Quant, Fp16InfinityAndNan) {
  auto bits = [](float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
  };
  EXPECT_EQ(bits(q::Fp16ToFloat(0x7C00)), 0x7F800000U);  // +inf
  EXPECT_EQ(bits(q::Fp16ToFloat(0xFC00)), 0xFF800000U);  // -inf
  EXPECT_EQ(bits(q::Fp16ToFloat(0x7E00)), 0x7FC00000U);  // quiet NaN, payload kept
  EXPECT_EQ(bits(q::Fp16ToFloat(0xFD01)), 0xFFA02000U);  // signalling NaN payload
  EXPECT_EQ(bits(q::Fp16ToFloat(0x8000)), 0x80000000U);  // -0
}

// Q8_K (the activation format of the K-quant dot products) has an fp32 scale
// and no gguf-py decoder; its value is the scale times the int8.
TEST(Quant, Q8_KIsScaleTimesInt) {
  hegel::test([](hegel::TestCase& tc) {
    std::vector<q::block_q8_K> blocks(tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 3})));
    for (auto& b : blocks) {
      b.d = tc.draw(gs::floats<float>({.min_value = -4.0F, .max_value = 4.0F}));
      for (auto& v : b.qs) v = tc.draw(gs::integers<std::int8_t>());
    }
    const std::size_t k = 256 * blocks.size();
    std::vector<float> out(k + 16, std::numeric_limits<float>::quiet_NaN()), vec(k);
    ASSERT_TRUE(q::DequantizeRow(GgmlType::kQ8_K, blocks.data(), out.data(), k));
    double ref = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      const auto& b = blocks[i / 256];
      ASSERT_EQ(out[i], b.d * static_cast<float>(b.qs[i % 256]));
      vec[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
      ref += static_cast<double>(out[i]) * vec[i];
      mag += std::fabs(static_cast<double>(out[i]) * vec[i]);
    }
    for (std::size_t i = k; i < out.size(); ++i) ASSERT_TRUE(std::isnan(out[i])) << "wrote past k";
    ASSERT_LE(std::fabs(q::DotProductQ8_K(blocks.data(), vec, k) - ref), 1e-5 * mag + 1e-6);
  });
}

TEST(Quant, F32RowsAreCopied) {
  const std::vector<float> row = {1.5F, -2.0F, 3.25F, 0.0F, -0.0F, 7.0F};
  std::vector<float> out(row.size() + 4, -1.0F);
  ASSERT_TRUE(q::DequantizeRow(GgmlType::kF32, row.data(), out.data(), row.size()));
  EXPECT_EQ(std::memcmp(out.data(), row.data(), row.size() * sizeof(float)), 0);
  for (std::size_t i = row.size(); i < out.size(); ++i) EXPECT_EQ(out[i], -1.0F);
}

TEST(Quant, BlockElementsAndSizes) {
  struct Row {
    GgmlType type;
    std::size_t block, bytes;
  };
  for (const Row r : {Row{GgmlType::kF32, 1, 4}, Row{GgmlType::kF16, 1, 2}, Row{GgmlType::kBF16, 1, 2},
                      Row{GgmlType::kQ8_0, 32, 34}, Row{GgmlType::kQ3_K, 256, 110},
                      Row{GgmlType::kQ4_K, 256, 144}, Row{GgmlType::kQ5_K, 256, 176},
                      Row{GgmlType::kQ6_K, 256, 210}, Row{GgmlType::kQ8_K, 256, 292},
                      Row{GgmlType::kIQ4_NL, 32, 18}, Row{GgmlType::kIQ4_XS, 256, 136},
                      Row{GgmlType::kIQ3_S, 256, 110}, Row{GgmlType::kI32, 1, 4},
                      // storage-only layouts (ggml-common.h)
                      Row{GgmlType::kQ4_0, 32, 18}, Row{GgmlType::kQ4_1, 32, 20},
                      Row{GgmlType::kQ5_0, 32, 22}, Row{GgmlType::kQ5_1, 32, 24},
                      Row{GgmlType::kQ8_1, 32, 36}, Row{GgmlType::kQ2_K, 256, 84},
                      Row{GgmlType::kIQ2_XXS, 256, 66}}) {
    const auto name = halo::core::ToString(r.type);
    const std::size_t qb = q::QuantizedBlockElements(r.type);
    EXPECT_EQ(std::max<std::size_t>(qb, 1), r.block) << name;
    EXPECT_EQ(q::EncodedSizeBytes(r.type, 3 * r.block), 3 * r.bytes) << name;
    if (r.block > 1) EXPECT_EQ(q::EncodedSizeBytes(r.type, r.block + 1), 0U) << name;
    // The largest size that fits in size_t is accepted, one block more is
    // rejected rather than wrapped (reachable only when a block takes more
    // bytes than it has elements).
    if (r.bytes > r.block) {
      const std::size_t blocks = std::numeric_limits<std::size_t>::max() / r.bytes;
      EXPECT_EQ(q::EncodedSizeBytes(r.type, blocks * r.block), blocks * r.bytes) << name;
      EXPECT_EQ(q::EncodedSizeBytes(r.type, (blocks + 1) * r.block), 0U) << name;
      // ...and a size whose byte count wraps to a small number.
      if (r.block == 1) EXPECT_EQ(q::EncodedSizeBytes(r.type, blocks + 2), 0U) << name;
    }
  }
  EXPECT_EQ(q::EncodedSizeBytes(static_cast<GgmlType>(4), 32), 0U);  // a removed ggml type
}
