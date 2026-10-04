// LoRA through the CPU reference against HF transformers + PEFT.
//
// tools/make_tiny_lora_reference.py attaches a random rank-4 PEFT adapter to
// the tiny Flash-Next (every dense projection: attention q/k/v/o, Gated
// DeltaNet qkv/z/a/b/out, shared expert) and records its logits;
// tools/convert_lora_qwen4exp.py converts it to GGUF, applying the base
// converter's head reordering to A and B. Same tolerance as the base test,
// fixed in advance: relative L2 < 1e-4 and identical argmax at every position.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/gguf_writer.hpp"
#include "core/quant.hpp"
#include "model/lora.hpp"
#include "model/ngram.hpp"
#include "model/reference.hpp"
#include "model/weights.hpp"

namespace {

std::vector<float> ReadF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(f.good()) << path;
  std::vector<float> out(static_cast<std::size_t>(f.tellg()) / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
  return out;
}

double RelL2(const float* a, const float* b, std::size_t n) {
  double d = 0, m = 0;
  for (std::size_t i = 0; i < n; ++i) {
    d += double(a[i] - b[i]) * (a[i] - b[i]);
    m += double(b[i]) * b[i];
  }
  return std::sqrt(d / m);
}

std::size_t Argmax(const float* v, std::size_t n) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
  return best;
}

}  // namespace

TEST(LoraVsPeft, TinyFlashNextAdapterMatchesPeft) {
  const char* dir_env = std::getenv("HALO_TINY_HF_DIR");
  if (!dir_env) GTEST_SKIP() << "set HALO_TINY_HF_DIR (needs lora-f32.gguf and logits_lora.f32)";
  const std::string dir = dir_env;
  std::string error;
  auto base = halo::core::GgufReader::OpenFile(dir + "/tiny-f32.gguf", &error);
  ASSERT_TRUE(base) << error;
  auto adapter_file = halo::core::GgufReader::OpenFile(dir + "/lora-f32.gguf", &error);
  ASSERT_TRUE(adapter_file) << error;
  auto weights = halo::model::ModelWeights::Bind(*base, &error);
  ASSERT_TRUE(weights) << error;
  auto adapter = halo::model::LoraAdapter::Bind(*adapter_file, *base, &error);
  ASSERT_TRUE(adapter) << error;
  EXPECT_EQ(adapter->pairs.size(), 62U);

  const auto& c = weights->config;
  const auto& t = weights->ple_table;
  auto ngram = halo::model::NgramTable::Open(base->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
                                             t.rows, c.ple_head_dim, t.type, &error);
  ASSERT_TRUE(ngram) << error;

  std::ifstream tf(dir + "/tokens.i32", std::ios::binary);
  std::vector<std::int32_t> tokens(48);
  tf.read(reinterpret_cast<char*>(tokens.data()), 48 * 4);
  const auto want_lora = ReadF32(dir + "/logits_lora.f32");
  const auto want_base = ReadF32(dir + "/logits_prefill.f32");
  const std::size_t V = c.vocab_size;

  halo::model::ReferenceModel with(*weights, ngram.get(), 48), without(*weights, ngram.get(), 48);
  with.SetAdapter(&*adapter);
  std::vector<float> a(V), b(V);
  double worst = 0;
  for (std::size_t p = 0; p < tokens.size(); ++p) {
    ASSERT_TRUE(with.Step(tokens[p], a, {}, &error)) << error;
    ASSERT_TRUE(without.Step(tokens[p], b, {}, &error)) << error;
    const double rel = RelL2(a.data(), want_lora.data() + p * V, V);
    worst = std::max(worst, rel);
    EXPECT_LT(rel, 1e-4) << "position " << p;
    EXPECT_EQ(Argmax(a.data(), V), Argmax(want_lora.data() + p * V, V)) << "position " << p;
    // Detaching the adapter restores the base model exactly as recorded.
    EXPECT_LT(RelL2(b.data(), want_base.data() + p * V, V), 1e-4) << "position " << p;
  }
  std::printf("worst relative L2 with adapter: %.3g\n", worst);
}

TEST(LoraAdapter, RejectsNonAdapterFiles) {
  const char* dir_env = std::getenv("HALO_TINY_HF_DIR");
  if (!dir_env) GTEST_SKIP();
  std::string error;
  auto base = halo::core::GgufReader::OpenFile(std::string(dir_env) + "/tiny-f32.gguf", &error);
  ASSERT_TRUE(base) << error;
  EXPECT_FALSE(halo::model::LoraAdapter::Bind(*base, *base, &error));
  EXPECT_NE(error.find("adapter"), std::string::npos);
}

// Adapters come from users, so Bind checks every field; each check rejects
// exactly its defect (synthetic files, no fixture needed). Mull found all of
// them untested. (Rank 0 cannot be tested: the GGUF reader already rejects a
// zero dimension.)
namespace {

struct LoraSpec {
  std::string general_type = "adapter", adapter_type = "lora", arch = "qwen4exp";
  std::optional<float> alpha = 8.0F;
  std::vector<std::uint64_t> a_dims = {4, 2}, b_dims = {2, 3};
  halo::core::GgmlType a_type = halo::core::GgmlType::kF32;
  std::string target = "blk.0.w";
  bool with_b = true, extra = false;
};

std::vector<std::uint8_t> Image(const halo::core::GgufWriter& w) {
  std::string error;
  auto image = w.Serialize(&error);
  EXPECT_FALSE(image.empty()) << error;
  return image;
}

std::optional<halo::model::LoraAdapter> BindSpec(const LoraSpec& s, std::string* error) {
  using halo::core::GgmlType;
  halo::core::GgufWriter base;
  base.Set("general.architecture", std::string("qwen4exp"));
  base.AddTensor("blk.0.w", {4, 3}, GgmlType::kF32, std::vector<std::uint8_t>(48));
  base.AddTensor("blk.0.experts", {4, 3, 2}, GgmlType::kF32, std::vector<std::uint8_t>(96));
  halo::core::GgufWriter ad;
  ad.Set("general.type", s.general_type);
  ad.Set("adapter.type", s.adapter_type);
  ad.Set("general.architecture", s.arch);
  if (s.alpha) ad.Set("adapter.lora.alpha", *s.alpha);
  auto bytes = [](GgmlType t, const std::vector<std::uint64_t>& d) {
    std::uint64_t n = 1;
    for (auto x : d) n *= x;
    return std::vector<std::uint8_t>(halo::quant::EncodedSizeBytes(t, n));
  };
  ad.AddTensor(s.target + ".lora_a", s.a_dims, s.a_type, bytes(s.a_type, s.a_dims));
  if (s.with_b) ad.AddTensor(s.target + ".lora_b", s.b_dims, GgmlType::kF32, bytes(GgmlType::kF32, s.b_dims));
  if (s.extra) ad.AddTensor("blk.0.bias", {3}, GgmlType::kF32, std::vector<std::uint8_t>(12));
  // The bound adapter points into these images; they outlive each call's checks.
  static std::vector<std::uint8_t> base_image, ad_image;
  static std::unique_ptr<halo::core::GgufReader> a, b;
  base_image = Image(base);
  ad_image = Image(ad);
  b = halo::core::GgufReader::OpenMemory(base_image.data(), base_image.size(), error);
  a = halo::core::GgufReader::OpenMemory(ad_image.data(), ad_image.size(), error);
  EXPECT_TRUE(a && b) << *error;
  if (!a || !b) return std::nullopt;
  return halo::model::LoraAdapter::Bind(*a, *b, error);
}

}  // namespace

TEST(LoraAdapter, BindChecksEachField) {
  std::string error;
  const auto ok = BindSpec({}, &error);
  ASSERT_TRUE(ok) << error;
  EXPECT_EQ(ok->alpha, 8.0F);
  const auto* p = ok->Find("blk.0.w");
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(ok->Find("blk.0.v"), nullptr);
  EXPECT_EQ(p->rank, 2U);
  EXPECT_EQ(p->a.cols, 4U);
  EXPECT_EQ(p->a.rows, 2U);
  EXPECT_EQ(p->b.cols, 2U);
  EXPECT_EQ(p->b.rows, 3U);
  EXPECT_EQ(p->a.name, "blk.0.w.lora_a");

  struct Bad {
    const char* what;
    LoraSpec spec;
  };
  using halo::core::GgmlType;
  const Bad bad[] = {
      {"general.type", {.general_type = "model"}},
      {"adapter.type", {.adapter_type = "ia3"}},
      {"architecture", {.arch = "qwen35"}},
      {"no alpha", {.alpha = std::nullopt}},
      {"zero alpha", {.alpha = 0.0F}},
      {"negative alpha", {.alpha = -1.0F}},
      {"NaN alpha", {.alpha = std::nanf("")}},
      {"unpaired", {.with_b = false}},
      {"unknown base tensor", {.target = "blk.9.w"}},
      {"3-D base tensor", {.target = "blk.0.experts"}},
      {"F16 A", {.a_type = GgmlType::kF16}},
      {"3-D A", {.a_dims = {4, 2, 1}}},
      {"3-D B", {.b_dims = {2, 3, 1}}},
      {"A input width", {.a_dims = {5, 2}}},
      {"B rank", {.b_dims = {3, 3}}},
      {"B output width", {.b_dims = {2, 4}}},
      {"stray tensor", {.extra = true}},
  };
  for (const auto& b : bad) {
    error.clear();
    EXPECT_FALSE(BindSpec(b.spec, &error)) << b.what;
    EXPECT_FALSE(error.empty()) << b.what;
  }
}
