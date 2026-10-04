// Qwen3.5 reference against HF transformers (modeling_qwen3_5.py) on a tiny
// random model converted with llama.cpp (tools/make_tiny_qwen35_reference.py).
// Tolerance fixed in advance, as for Flash-Next: relative L2 < 1e-4 on logits
// and the final hidden state, identical argmax, at every position.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "model/qwen35.hpp"

namespace {

template <typename T>
std::vector<T> ReadRaw(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(f.good()) << path;
  std::vector<T> out(static_cast<std::size_t>(f.tellg()) / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(T)));
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

}  // namespace

TEST(Qwen35VsHf, TinyModelMatchesTransformers) {
  const char* dir_env = std::getenv("HALO_TINY_QWEN35_DIR");
  if (!dir_env) GTEST_SKIP() << "set HALO_TINY_QWEN35_DIR (tools/make_tiny_qwen35_reference.py output)";
  const std::string dir = dir_env;
  std::string error;
  auto reader = halo::core::GgufReader::OpenFile(dir + "/tiny-qwen35-f32.gguf", &error);
  ASSERT_TRUE(reader) << error;
  auto w = halo::model::Qwen35Weights::Bind(*reader, &error);
  ASSERT_TRUE(w) << error;
  const auto tokens = ReadRaw<std::int32_t>(dir + "/tokens.i32");
  const auto logits = ReadRaw<float>(dir + "/logits.f32");
  const auto hidden = ReadRaw<float>(dir + "/hidden.f32");
  const std::size_t V = w->config.vocab_size, H = w->config.hidden_size;
  ASSERT_EQ(logits.size(), tokens.size() * V);
  halo::model::Qwen35Reference ref(*w, static_cast<std::uint32_t>(tokens.size()));
  std::vector<float> h(H), l(V);
  double worst = 0;
  for (std::size_t p = 0; p < tokens.size(); ++p) {
    ASSERT_TRUE(ref.Step(tokens[p], h, l, &error)) << error;
    const double rl = RelL2(l.data(), logits.data() + p * V, V), rh = RelL2(h.data(), hidden.data() + p * H, H);
    worst = std::max({worst, rl, rh});
    EXPECT_LT(rl, 1e-4) << "logits at " << p;
    EXPECT_LT(rh, 1e-4) << "hidden at " << p;
    EXPECT_EQ(std::max_element(l.begin(), l.end()) - l.begin(),
              std::max_element(logits.begin() + p * V, logits.begin() + (p + 1) * V) - (logits.begin() + p * V));
  }
  std::printf("worst relative L2: %.3g\n", worst);
}
