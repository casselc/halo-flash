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
#include <optional>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "engine/flash_next_gpu.hpp"
#include "core/sampling.hpp"
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

// Session state, swarm style (graph replay and eager launches mixed): each case enables a random subset of the
// operations, then runs a random program of them. The model is the CPU
// reference, copied into the same snapshot slots. Every logits output must
// match, so restore == replay and chunked prefill == token-at-a-time.
TEST(FlashNextGpu, SnapshotsResetAndPrefillMatchReference) {
  Tiny* t = LoadTiny();
  if (!t) GTEST_SKIP() << "set HALO_TINY_HF_DIR";
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  const auto& c = t->weights->config;
  const std::size_t V = c.vocab_size;
  constexpr std::uint32_t kMaxContext = 48;
  constexpr int kSlots = 3;
  hegel::test([&](hegel::TestCase& tc) {
    auto& dev = *devices[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = devices.size() - 1}))];
    const model::LoraAdapter* lora = tc.draw(gs::booleans()) ? &*t->lora : nullptr;
    std::string error;
    auto gpu = engine::FlashNextGpu::Create(dev, *t->weights, t->ngram.get(), lora, kMaxContext, &error);
    ASSERT_TRUE(gpu) << error;
    gpu->SetUseGraphs(tc.draw(gs::booleans()));
    std::optional<model::ReferenceModel> cpu;
    cpu.emplace(*t->weights, t->ngram.get(), kMaxContext);
    cpu->SetAdapter(lora);
    std::vector<std::unique_ptr<engine::FlashNextGpu::Snapshot>> gpu_slots(kSlots);
    std::vector<std::optional<model::ReferenceModel>> cpu_slots(kSlots);
    for (auto& s : gpu_slots) {
      s = gpu->NewSnapshot(&error);
      ASSERT_TRUE(s) << error;
    }
    enum Op { kStep, kPrefill, kSave, kRestore, kReset, kToggleGraphs, kOps };
    bool enabled[kOps];
    for (bool& e : enabled) e = tc.draw(gs::booleans());
    enabled[kStep] = true;  // the program must make progress
    std::vector<float> gl(V), cl(V);
    const auto token = [&] {
      return tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4})) == 0
                 ? static_cast<std::int32_t>(c.ple_eos_token)
                 : tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = static_cast<std::int32_t>(V - 1)}));
    };
    const auto check = [&](const char* what) {
      ASSERT_LT(RelL2(gl, cl.data()), 1e-4) << what << " at position " << gpu->Position();
      ASSERT_EQ(Argmax(gl.data(), V), Argmax(cl.data(), V)) << what;
    };
    const int ops = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 40}));
    for (int i = 0; i < ops; ++i) {
      // Weighted toward long histories and forks: a stale-state bug shows
      // only after save, divergence past the save point, then restore.
      const int pick = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 10}));
      const Op op = pick < 3 ? kStep : pick < 6 ? kPrefill : pick < 7 ? kSave : pick < 9 ? kRestore : pick < 10 ? kReset
                                                                                                           : kToggleGraphs;
      if (!enabled[op]) continue;
      const int slot = tc.draw(gs::integers<int>({.min_value = 0, .max_value = kSlots - 1}));
      switch (op) {
        case kStep: {
          if (gpu->Position() >= kMaxContext) break;
          const auto tok = token();
          ASSERT_TRUE(gpu->Step(tok, gl, &error)) << error;
          ASSERT_TRUE(cpu->Step(tok, cl, {}, &error)) << error;
          check("step");
          break;
        }
        case kPrefill: {
          const auto n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 12}));
          if (gpu->Position() + n > kMaxContext) break;
          std::vector<std::int32_t> chunk(n);
          for (auto& tok : chunk) tok = token();
          ASSERT_TRUE(gpu->Prefill(chunk, gl, &error)) << error;
          for (std::uint32_t j = 0; j < n; ++j)
            ASSERT_TRUE(cpu->Step(chunk[j], j + 1 == n ? std::span<float>(cl) : std::span<float>{}, {}, &error)) << error;
          check("prefill");
          break;
        }
        case kSave:
          ASSERT_TRUE(gpu->Save(*gpu_slots[slot], &error)) << error;
          cpu_slots[slot].emplace(*cpu);
          break;
        case kRestore:
          if (!cpu_slots[slot]) {
            ASSERT_FALSE(gpu->Restore(*gpu_slots[slot], &error));  // never-saved slot is refused
            break;
          }
          ASSERT_TRUE(gpu->Restore(*gpu_slots[slot], &error)) << error;
          cpu.emplace(*cpu_slots[slot]);
          break;
        case kReset:
          ASSERT_TRUE(gpu->Reset(&error)) << error;
          cpu->Reset();
          break;
        case kToggleGraphs:  // graph replay and eager launches share all state
          gpu->SetUseGraphs(tc.draw(gs::booleans()));
          break;
        case kOps: break;
      }
      ASSERT_EQ(gpu->Position(), cpu->Position());
    }
  }, {.test_cases = 100});
}

// Directed: restoring a snapshot after an unrelated history.
// - A longer one overwrites every row the snapshot covers. A plain
//   fork-and-continue never rewrites rows below the save point, so it
//   cannot see a snapshot that drops some of them.
// - A shorter one leaves a partially pooled QSA block in a slot the
//   snapshot holds complete, so a restore that copies too few pooled
//   blocks scores a corrupted block.
// The random programs above reach these shapes only rarely.
TEST(FlashNextGpu, RestoreAfterOverwritingHistory) {
  Tiny* t = LoadTiny();
  if (!t) GTEST_SKIP() << "set HALO_TINY_HF_DIR";
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  const std::size_t V = t->weights->config.vocab_size;
  const auto tokens = ReadTokens(t->dir + "/tokens.i32");  // 48 tokens
  const std::span<const std::int32_t> all(tokens);
  for (auto& dev : devices) {
    const std::pair<std::size_t, std::size_t> cases[] = {{1, 9}, {9, 17}, {13, 21}, {20, 28}, {20, 10}, {30, 6}, {28, 13}};
    for (auto [save_at, other] : cases) {
      std::string error;
      auto gpu = engine::FlashNextGpu::Create(*dev, *t->weights, t->ngram.get(), nullptr, 48, &error);
      ASSERT_TRUE(gpu) << error;
      auto snap = gpu->NewSnapshot(&error);
      ASSERT_TRUE(snap) << error;
      model::ReferenceModel cpu(*t->weights, t->ngram.get(), 48);
      std::vector<float> gl(V), cl(V);
      ASSERT_TRUE(gpu->Prefill(all.first(save_at), {}, &error)) << error;
      for (std::size_t i = 0; i < save_at; ++i) ASSERT_TRUE(cpu.Step(tokens[i], {}, {}, &error));
      ASSERT_TRUE(gpu->Save(*snap, &error)) << error;
      ASSERT_TRUE(gpu->Reset(&error)) << error;
      ASSERT_TRUE(gpu->Prefill(all.last(other), {}, &error)) << error;  // a different history
      ASSERT_TRUE(gpu->Restore(*snap, &error)) << error;
      for (std::size_t i = save_at; i < save_at + 12; ++i) {
        ASSERT_TRUE(gpu->Step(tokens[i], gl, &error)) << error;
        ASSERT_TRUE(cpu.Step(tokens[i], cl, {}, &error)) << error;
        ASSERT_LT(RelL2(gl, cl.data()), 1e-4) << rt::BackendName(dev->backend()) << " saved at " << save_at
                                              << " after " << other << " other tokens, position " << i;
      }
    }
  }
}

// Device sampling through the engine. Two engines run the same token
// stream in lockstep: one returns logits to the CPU sampler (the oracle),
// the other samples on the device and returns only the token. Greedy must
// agree exactly; exact mode must agree token for token and in RNG state,
// with penalties; fast mode must stay in the CPU support and be
// deterministic. Graph and eager modes both.
TEST(FlashNextGpu, StepSampleMatchesCpuSamplerInLockstep) {
  Tiny* t = LoadTiny();
  if (!t) GTEST_SKIP() << "set HALO_TINY_HF_DIR";
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  const std::size_t V = t->weights->config.vocab_size;
  hegel::test([&](hegel::TestCase& tc) {
    auto& dev = *devices[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = devices.size() - 1}))];
    sampling::SamplingConfig c;
    c.temperature = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.7f, 1.0f}));
    c.top_k = tc.draw(gs::sampled_from(std::vector<std::int32_t>{0, 1, 20}));
    c.top_p = tc.draw(gs::sampled_from(std::vector<float>{1.0f, 0.9f}));
    c.min_p = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.05f}));
    c.repeat_penalty = tc.draw(gs::sampled_from(std::vector<float>{1.0f, 1.2f}));
    c.presence_penalty = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.5f}));
    c.seed = 1;
    const bool exact = tc.draw(gs::booleans()), graphs = tc.draw(gs::booleans());
    std::string error;
    auto oracle = engine::FlashNextGpu::Create(dev, *t->weights, t->ngram.get(), nullptr, 32, &error);
    auto sampler = engine::FlashNextGpu::Create(dev, *t->weights, t->ngram.get(), nullptr, 32, &error);
    ASSERT_TRUE(oracle && sampler) << error;
    sampler->SetUseGraphs(graphs);
    sampler->SetSamplingMode(exact ? engine::FlashNextGpu::SamplingMode::kExact
                                   : engine::FlashNextGpu::SamplingMode::kFast);
    sampling::SamplerState cpu(c);
    cpu.SetRngState(tc.draw(gs::integers<std::uint64_t>()));
    std::vector<float> logits(V);
    std::int32_t token = tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = static_cast<std::int32_t>(V - 1)}));
    const int steps = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 12}));
    for (int i = 0; i < steps; ++i) {
      ASSERT_TRUE(oracle->Step(token, logits, &error)) << error;
      const auto penalties = cpu.penalties();
      const std::vector<sampling::TokenPenalty> pen(penalties.begin(), penalties.end());
      const std::uint64_t rng = cpu.rng_state();
      engine::FlashNextGpu::SampleResult got;
      ASSERT_TRUE(sampler->StepSample(token, {&c, pen, rng}, &got, &error)) << error;
      const auto d = cpu.Distribution(logits);
      const auto want = static_cast<std::int32_t>(cpu.Sample(logits));
      if (c.temperature == 0.0f || exact) {
        ASSERT_EQ(got.token, want) << (exact ? "exact" : "greedy") << " step " << i;
        ASSERT_EQ(got.rng, cpu.rng_state()) << "step " << i;
      } else {
        ASSERT_GT(d.probability(static_cast<sampling::TokenId>(got.token)), 0.0) << "outside the support at step " << i;
        cpu.SetRngState(got.rng);  // the fast sampler advances the RNG once per draw
      }
      token = got.token;  // both engines continue with the device's choice
      cpu.Accept(static_cast<sampling::TokenId>(token));
    }
  }, {.test_cases = 30});
}
