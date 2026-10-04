// GPU Flash-Next decode against the CPU reference, on every built backend
// (ROCm HIP and native HRX). Tolerances fixed in advance, as for Qwen3.5:
// relative L2 < 1e-4 on logits and identical argmax at every position.
//   HALO_TINY_HF_DIR  tiny random Flash-Next (tools/make_tiny_hf_reference.py,
//                     tools/make_tiny_lora_reference.py); 48 positions with an
//                     8-token QSA budget, so block selection is exercised.

#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "engine/flash_next_gpu.hpp"
#include "model/lora.hpp"
#include "model/ngram.hpp"
#include "model/reference.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"

namespace gs = hegel::generators;
using namespace halo;

namespace {

double RelL2(const std::vector<float>& a, const float* b) {
  double d = 0, m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d += double(a[i] - b[i]) * (a[i] - b[i]);
    m += double(b[i]) * b[i];
  }
  return std::sqrt(d / m);
}

std::size_t Argmax(const float* v, std::size_t n) { return std::max_element(v, v + n) - v; }

std::vector<std::int32_t> ReadTokens(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<std::int32_t> out(static_cast<std::size_t>(f.tellg()) / 4);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * 4));
  return out;
}

struct Tiny {
  std::unique_ptr<core::GgufReader> reader, lora_reader;
  std::optional<model::ModelWeights> weights;
  std::optional<model::LoraAdapter> lora;
  std::unique_ptr<model::NgramTable> ngram;
  std::string dir;
};

Tiny* LoadTiny() {
  static Tiny* tiny = [] () -> Tiny* {
    const char* dir = std::getenv("HALO_TINY_HF_DIR");
    if (!dir) return nullptr;
    auto* t = new Tiny;
    t->dir = dir;
    std::string error;
    t->reader = core::GgufReader::OpenFile(t->dir + "/tiny-f32.gguf", &error);
    if (!t->reader) return nullptr;
    t->weights = model::ModelWeights::Bind(*t->reader, &error);
    if (!t->weights) return nullptr;
    const auto& p = t->weights->ple_table;
    t->ngram = model::NgramTable::Open(t->reader->GetMappedRegions()[p.shard].file_descriptor, p.file_offset, p.rows,
                                       t->weights->config.ple_head_dim, p.type, &error);
    t->lora_reader = core::GgufReader::OpenFile(t->dir + "/lora-f32.gguf", &error);
    if (t->lora_reader) t->lora = model::LoraAdapter::Bind(*t->lora_reader, *t->reader, &error);
    return t->ngram ? t : nullptr;
  }();
  return tiny;
}

std::vector<std::unique_ptr<rt::Device>> Devices() {
  std::vector<std::unique_ptr<rt::Device>> out;
  for (auto b : {rt::Backend::kHip, rt::Backend::kHrx}) {
    std::string error;
    if (auto d = rt::Device::Create(b, &error)) out.push_back(std::move(d));
  }
  return out;
}

// Steps GPU and CPU through `tokens`; every position must agree.
void Compare(rt::Device& dev, Tiny& t, const model::LoraAdapter* lora, const std::vector<std::int32_t>& tokens,
             const std::vector<float>* expected = nullptr, double* worst = nullptr) {
  std::string error;
  const auto n = static_cast<std::uint32_t>(tokens.size());
  auto gpu = engine::FlashNextGpu::Create(dev, *t.weights, t.ngram.get(), lora, n, &error);
  ASSERT_TRUE(gpu) << error;
  model::ReferenceModel cpu(*t.weights, t.ngram.get(), n);
  cpu.SetAdapter(lora);
  const std::size_t V = t.weights->config.vocab_size;
  std::vector<float> gl(V), cl(V);
  for (std::size_t p = 0; p < tokens.size(); ++p) {
    ASSERT_TRUE(gpu->Step(tokens[p], gl, &error)) << error;
    ASSERT_TRUE(cpu.Step(tokens[p], cl, {}, &error)) << error;
    const double rel = RelL2(gl, cl.data());
    if (worst) *worst = std::max(*worst, rel);
    ASSERT_LT(rel, 1e-4) << rt::BackendName(dev.backend()) << " position " << p;
    ASSERT_EQ(Argmax(gl.data(), V), Argmax(cl.data(), V)) << "position " << p;
    if (expected) {
      const float* want = expected->data() + p * V;
      ASSERT_LT(RelL2(gl, want), 1e-4) << "vs transformers at " << p;
      ASSERT_EQ(Argmax(gl.data(), V), Argmax(want, V));
    }
  }
}

}  // namespace

TEST(FlashNextGpu, TinyModelMatchesReferenceAndTransformers) {
  Tiny* t = LoadTiny();
  if (!t) GTEST_SKIP() << "set HALO_TINY_HF_DIR";
  const auto tokens = ReadTokens(t->dir + "/tokens.i32");
  const std::size_t V = t->weights->config.vocab_size;
  std::vector<float> decode(tokens.size() * V), lora_logits(tokens.size() * V);
  std::ifstream(t->dir + "/logits_decode.f32", std::ios::binary)
      .read(reinterpret_cast<char*>(decode.data()), static_cast<std::streamsize>(decode.size() * 4));
  std::ifstream(t->dir + "/logits_lora.f32", std::ios::binary)
      .read(reinterpret_cast<char*>(lora_logits.data()), static_cast<std::streamsize>(lora_logits.size() * 4));
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  for (auto& dev : devices) {
    double base = 0, adapted = 0;
    Compare(*dev, *t, nullptr, tokens, &decode, &base);
    ASSERT_TRUE(t->lora);
    Compare(*dev, *t, &*t->lora, tokens, &lora_logits, &adapted);
    std::printf("%s: worst relative L2 vs reference %.3g (base), %.3g (LoRA)\n", rt::BackendName(dev->backend()), base,
                adapted);
  }
}

// Random sequences (lengths across the QSA budget boundary, EOS inside the
// n-gram window, repeated tokens) with and without the adapter.
TEST(FlashNextGpu, RandomSequencesMatchReference) {
  Tiny* t = LoadTiny();
  if (!t) GTEST_SKIP() << "set HALO_TINY_HF_DIR";
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  const auto& c = t->weights->config;
  hegel::test([&](hegel::TestCase& tc) {
    auto& dev = *devices[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = devices.size() - 1}))];
    const bool use_lora = tc.draw(gs::booleans());
    const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 40}));
    std::vector<std::int32_t> tokens(n);
    for (auto& tok : tokens) {
      switch (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}))) {
        case 0: tok = static_cast<std::int32_t>(c.ple_eos_token); break;
        case 1: tok = tokens.front(); break;
        default:
          tok = tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = static_cast<std::int32_t>(c.vocab_size - 1)}));
      }
    }
    Compare(dev, *t, use_lora ? &*t->lora : nullptr, tokens);
  }, {.test_cases = 25});
}
