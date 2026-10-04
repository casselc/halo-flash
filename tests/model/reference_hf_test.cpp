// The CPU reference against HF transformers' own Qwen4Exp modeling code.
//
// tools/make_tiny_hf_reference.py builds a tiny random Flash-Next with HF,
// records its logits, and llama.cpp's converter turns the checkpoint into
// tiny-f32.gguf. This test loads that GGUF through the production path
// (GgufReader, Config, ModelWeights, NgramTable) and steps the reference one
// token at a time. Set HALO_TINY_HF_DIR to the generator's output directory.
//
// Tolerances were fixed before the first run: both sides compute in FP32, so
// every position must agree to relative L2 < 1e-4 with identical argmax.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "model/config.hpp"
#include "model/ngram.hpp"
#include "model/reference.hpp"
#include "model/weights.hpp"

namespace {

constexpr double kMaxRelL2 = 1e-4;

template <typename T>
std::vector<T> ReadRaw(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(f.good()) << path;
  const auto bytes = static_cast<std::size_t>(f.tellg());
  std::vector<T> out(bytes / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
  return out;
}

std::size_t Argmax(const float* v, std::size_t n) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
  return best;
}

}  // namespace

TEST(ReferenceVsHf, TinyModelLogitsMatchTransformers) {
  const char* dir_env = std::getenv("HALO_TINY_HF_DIR");
  if (!dir_env) GTEST_SKIP() << "set HALO_TINY_HF_DIR (tools/make_tiny_hf_reference.py output)";
  const std::string dir = dir_env;

  std::string error;
  auto reader = halo::core::GgufReader::OpenFile(dir + "/tiny-f32.gguf", &error);
  ASSERT_TRUE(reader) << error;
  auto weights = halo::model::ModelWeights::Bind(*reader, &error);
  ASSERT_TRUE(weights) << error;
  const auto& c = weights->config;
  EXPECT_FALSE(c.IsFlashNextGeometry());
  ASSERT_GE(c.ple_layer, 0);

  const auto& t = weights->ple_table;
  const int fd = reader->GetMappedRegions()[t.shard].file_descriptor;
  auto ngram = halo::model::NgramTable::Open(fd, t.file_offset, t.rows, c.ple_head_dim, t.type, &error);
  ASSERT_TRUE(ngram) << error;

  const auto tokens = ReadRaw<std::int32_t>(dir + "/tokens.i32");
  const auto prefill = ReadRaw<float>(dir + "/logits_prefill.f32");
  const auto decode = ReadRaw<float>(dir + "/logits_decode.f32");
  const std::size_t V = c.vocab_size;
  ASSERT_EQ(prefill.size(), tokens.size() * V);
  ASSERT_EQ(decode.size(), tokens.size() * V);

  halo::model::ReferenceModel ref(*weights, ngram.get(), static_cast<std::uint32_t>(tokens.size()));
  std::vector<float> logits(V);
  double worst = 0.0;
  for (std::size_t p = 0; p < tokens.size(); ++p) {
    ASSERT_TRUE(ref.Step(tokens[p], logits, {}, &error)) << error;
    for (const auto* want_all : {&prefill, &decode}) {
      const float* want = want_all->data() + p * V;
      double diff = 0, norm = 0;
      for (std::size_t i = 0; i < V; ++i) {
        diff += double(logits[i] - want[i]) * (logits[i] - want[i]);
        norm += double(want[i]) * want[i];
      }
      const double rel = std::sqrt(diff / norm);
      worst = std::max(worst, rel);
      EXPECT_LT(rel, kMaxRelL2) << "position " << p << (want_all == &prefill ? " (prefill)" : " (decode)");
      EXPECT_EQ(Argmax(logits.data(), V), Argmax(want, V)) << "position " << p;
    }
  }
  std::printf("worst relative L2 over %zu positions: %.3g\n", tokens.size(), worst);
}
