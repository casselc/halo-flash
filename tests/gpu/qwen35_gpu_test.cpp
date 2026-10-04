// GPU Qwen3.5 decode against the CPU reference, on every built backend
// (ROCm HIP and native HRX). Tolerances fixed in advance: relative L2 < 1e-4
// on logits and hidden state, identical argmax, at every position.
//   HALO_TINY_QWEN35_DIR   tiny random model (tools/make_tiny_qwen35_reference.py)
//   HALO_DECIDER_MODELS_DIR real Qwen3.5-2B + strands-decider LoRA

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "engine/qwen35_gpu.hpp"
#include "model/qwen35.hpp"
#include "rt/device.hpp"

using namespace halo;

namespace {

double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0, m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d += double(a[i] - b[i]) * (a[i] - b[i]);
    m += double(b[i]) * b[i];
  }
  return std::sqrt(d / m);
}

std::vector<rt::Backend> Backends() {
  std::vector<rt::Backend> out;
  for (auto b : {rt::Backend::kHip, rt::Backend::kHrx}) {
    std::string error;
    if (rt::Device::Create(b, &error)) out.push_back(b);
  }
  return out;
}

// Steps GPU and CPU through the same tokens; returns the worst relative L2.
double Compare(rt::Device& dev, const model::Qwen35Weights& w, const model::LoraAdapter* lora,
               const std::vector<std::int32_t>& tokens, bool check_fork) {
  std::string error;
  auto gpu = engine::Qwen35Gpu::Create(dev, w, lora, static_cast<std::uint32_t>(tokens.size() + 4), &error);
  EXPECT_TRUE(gpu) << error;
  if (!gpu) return 1e9;
  model::Qwen35Reference cpu(w, static_cast<std::uint32_t>(tokens.size() + 4));
  cpu.SetAdapter(lora);
  const auto H = w.config.hidden_size, V = w.config.vocab_size;
  std::vector<float> gh(H), gl(V), ch(H), cl(V);
  double worst = 0;
  const std::size_t fork_at = tokens.size() / 2;
  model::Qwen35Reference cpu_fork = cpu;
  for (std::size_t p = 0; p < tokens.size(); ++p) {
    if (check_fork && p == fork_at) {
      EXPECT_TRUE(gpu->SaveState(&error)) << error;
      cpu_fork = cpu;
    }
    EXPECT_TRUE(gpu->Step(tokens[p], gh, gl, &error)) << error;
    EXPECT_TRUE(cpu.Step(tokens[p], ch, cl, &error)) << error;
    const double rl = RelL2(gl, cl), rh = RelL2(gh, ch);
    worst = std::max({worst, rl, rh});
    EXPECT_LT(rl, 1e-4) << "logits at " << p;
    EXPECT_LT(rh, 1e-4) << "hidden at " << p;
    EXPECT_EQ(std::max_element(gl.begin(), gl.end()) - gl.begin(), std::max_element(cl.begin(), cl.end()) - cl.begin());
  }
  if (check_fork) {
    // Restoring the fork and replaying a different continuation must equal
    // the CPU reference forked at the same point.
    EXPECT_TRUE(gpu->RestoreState(&error)) << error;
    for (std::size_t p = fork_at; p < tokens.size(); ++p) {
      const auto t = tokens[tokens.size() - 1 - (p - fork_at)];
      EXPECT_TRUE(gpu->Step(t, gh, gl, &error)) << error;
      EXPECT_TRUE(cpu_fork.Step(t, ch, cl, &error)) << error;
      EXPECT_LT(RelL2(gl, cl), 1e-4) << "forked logits at " << p;
      worst = std::max(worst, RelL2(gl, cl));
    }
  }
  return worst;
}

}  // namespace

TEST(Qwen35Gpu, TinyModelMatchesCpuReference) {
  const char* dir = std::getenv("HALO_TINY_QWEN35_DIR");
  if (!dir) GTEST_SKIP() << "set HALO_TINY_QWEN35_DIR";
  std::string error;
  auto reader = core::GgufReader::OpenFile(std::string(dir) + "/tiny-qwen35-f32.gguf", &error);
  ASSERT_TRUE(reader) << error;
  auto w = model::Qwen35Weights::Bind(*reader, &error);
  ASSERT_TRUE(w) << error;
  std::vector<std::int32_t> tokens;
  for (int i = 0; i < 24; ++i) tokens.push_back((i * 7919 + 13) % 248000);
  const auto backends = Backends();
  ASSERT_FALSE(backends.empty()) << "no GPU backend available";
  for (auto b : backends) {
    auto dev = rt::Device::Create(b, &error);
    ASSERT_TRUE(dev) << error;
    const double worst = Compare(*dev, *w, nullptr, tokens, true);
    std::printf("%s: worst relative L2 %.3g\n", rt::BackendName(b), worst);
  }
}

TEST(Qwen35Gpu, DeciderTorsoWithLoraMatchesCpuReference) {
  const char* dir = std::getenv("HALO_DECIDER_MODELS_DIR");
  if (!dir) GTEST_SKIP() << "set HALO_DECIDER_MODELS_DIR";
  std::string error;
  const std::string d = dir;
  auto reader = core::GgufReader::OpenFile(d + "/Qwen3.5-2B-Base-BF16.gguf", &error);
  ASSERT_TRUE(reader) << error;
  auto lora_file = core::GgufReader::OpenFile(d + "/strands-decider-lora-f32.gguf", &error);
  ASSERT_TRUE(lora_file) << error;
  auto w = model::Qwen35Weights::Bind(*reader, &error);
  ASSERT_TRUE(w) << error;
  auto lora = model::LoraAdapter::Bind(*lora_file, *reader, &error);
  ASSERT_TRUE(lora) << error;
  const std::vector<std::int32_t> tokens = {27, 2056, 29, 198, 12620, 0, 3638, 220, 17, 1999, 13};
  for (auto b : Backends()) {
    auto dev = rt::Device::Create(b, &error);
    ASSERT_TRUE(dev) << error;
    const double worst = Compare(*dev, *w, &*lora, tokens, false);
    std::printf("%s: worst relative L2 %.3g (2B + LoRA)\n", rt::BackendName(b), worst);
  }
}
