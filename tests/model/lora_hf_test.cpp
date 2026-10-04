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
#include <string>
#include <vector>

#include "core/gguf.hpp"
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
