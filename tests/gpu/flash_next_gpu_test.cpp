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
#include <cstring>
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

// CODESIGN arm B: the engine routes each token to config.num_experts_used
// experts (ModelWeights::SetExpertsUsed), every path taking k from there.
// For each k below the trained count, GPU stepping equals the CPU reference
// at the same k (same bound as the trained k), and k changes the logits.
TEST(FlashNextGpu, ReducedTopKMatchesReference) {
  Tiny* t = LoadTiny();
  if (!t) GTEST_SKIP() << "set HALO_TINY_HF_DIR";
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  auto& w = *t->weights;
  const std::uint32_t trained = w.config.num_experts_used;
  if (trained < 2) GTEST_SKIP() << "the tiny model routes to one expert";
  struct Restore {  // the fixture is shared by every test
    model::ModelWeights& w;
    std::uint32_t k;
    ~Restore() { w.config.num_experts_used = k; }
  } restore{w, trained};
  const auto tokens = ReadTokens(t->dir + "/tokens.i32");
  const std::size_t V = w.config.vocab_size;
  std::vector<float> full(V), reduced(V);
  {
    model::ReferenceModel cpu(w, t->ngram.get(), static_cast<std::uint32_t>(tokens.size()));
    std::string error;
    for (auto tok : tokens) ASSERT_TRUE(cpu.Step(tok, full, {}, &error)) << error;
  }
  for (std::uint32_t k = 1; k < trained; ++k) {
    std::string error;
    ASSERT_TRUE(w.SetExpertsUsed(k, &error)) << error;
    for (auto& dev : devices) Compare(*dev, *t, nullptr, tokens);
    model::ReferenceModel cpu(w, t->ngram.get(), static_cast<std::uint32_t>(tokens.size()));
    for (auto tok : tokens) ASSERT_TRUE(cpu.Step(tok, reduced, {}, &error)) << error;
    EXPECT_GT(RelL2(reduced, full.data()), 1e-6) << "k = " << k << " left the logits unchanged";
  }
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

// Weights read from their files (parallel O_DIRECT loader) must be the same
// bytes as a direct upload: identical logits.
TEST(FlashNextGpu, FileLoadedWeightsMatchDirectUpload) {
  Tiny* t = LoadTiny();
  if (!t) GTEST_SKIP() << "set HALO_TINY_HF_DIR";
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  std::vector<rt::FileSource> sources;
  for (const auto& r : t->reader->GetMappedRegions()) sources.push_back({r.data, r.size, r.file_descriptor});
  const auto tokens = ReadTokens(t->dir + "/tokens.i32");
  const std::size_t V = t->weights->config.vocab_size;
  for (auto& dev : devices) {
    std::string error;
    auto direct = engine::FlashNextGpu::Create(*dev, *t->weights, t->ngram.get(), &*t->lora, 16, &error);
    auto loaded = engine::FlashNextGpu::Create(*dev, *t->weights, t->ngram.get(), &*t->lora, 16, &error, sources);
    ASSERT_TRUE(direct && loaded) << error;
    std::vector<float> a(V), b(V);
    for (std::size_t p = 0; p < 8; ++p) {
      ASSERT_TRUE(direct->Step(tokens[p], a, &error)) << error;
      ASSERT_TRUE(loaded->Step(tokens[p], b, &error)) << error;
      ASSERT_EQ(std::memcmp(a.data(), b.data(), V * 4), 0) << rt::BackendName(dev->backend()) << " position " << p;
    }
  }
}

// The small model (tools/make_tiny_hf_reference.py --small-quant, quantized by
// tools/make_small_quant.sh with UD-Q4_K_XL's per-tensor types) has widths
// the quantized fast paths accept, so this runs the kernels the real model
// runs: repacked Q8_0, Q4_K/Q5_1 experts with int8 activations, the Q5_K and
// Q8_0 expert fallbacks, the fused mixer, shared expert and DeltaNet block,
// d128 GDN and BF16 indexer.
//   Float activations, against the float reference: relative L2 < 1e-4 and
//   identical argmax at every position (fixed in advance, as above).
//   int8 activations, against the reference emulating the same quantization:
//   identical argmax, relative L2 < 1e-3 at every position, and over the
//   sequence at most a tenth of the emulation's own distance from float.
// Why int8 is not held to 1e-4: values land within 1e-6 of a rounding
// boundary (x / scale = 10.4999990 occurs here), so a 1-ulp difference in x,
// which float reordering produces, flips a quant and moves logits by up to
// 5e-4. An end-to-end bound of 1e-2 against the float reference, set in
// advance, also failed (2.7%): the random-weight model amplifies the ~0.5%
// per-projection quantization error; that comparison belongs to the real
// model's quality gate. A formula error (wrong scale, dropped min term,
// truncation) moves int8 as far as float is, which the tenth-bound catches.
//   HALO_SMALL_QUANT_DIR  holds small-q.gguf and tokens.i32
TEST(FlashNextGpu, SmallQuantizedModelRunsFastPathsAndMatchesReference) {
  const char* dir = std::getenv("HALO_SMALL_QUANT_DIR");
  if (!dir) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  std::string error;
  auto reader = core::GgufReader::OpenFile(std::string(dir) + "/small-q.gguf", &error);
  ASSERT_TRUE(reader) << error;
  auto weights = model::ModelWeights::Bind(*reader, &error);
  ASSERT_TRUE(weights) << error;
  const auto& p = weights->ple_table;
  auto ngram = model::NgramTable::Open(reader->GetMappedRegions()[p.shard].file_descriptor, p.file_offset, p.rows,
                                       weights->config.ple_head_dim, p.type, &error);
  ASSERT_TRUE(ngram) << error;
  const auto tokens = ReadTokens(std::string(dir) + "/tokens.i32");
  const auto n = static_cast<std::uint32_t>(tokens.size());
  const std::size_t V = weights->config.vocab_size;

  // Reference logits, float and int8 expert activations.
  std::vector<float> want[2];
  for (const bool int8 : {false, true}) {
    want[int8].resize(tokens.size() * V);
    model::ReferenceModel cpu(*weights, ngram.get(), n);
    cpu.SetInt8ExpertActivations(int8);
    for (std::size_t i = 0; i < tokens.size(); ++i)
      ASSERT_TRUE(cpu.Step(tokens[i], std::span<float>(want[int8].data() + i * V, V), {}, &error)) << error;
  }
  std::printf("int8 vs float reference: relative L2 %.3g over the sequence\n",
              RelL2(want[1], want[0].data()));
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  const bool saved = engine::FlashNextGpu::int8_act_moe_;
  for (auto& dev : devices)
    for (const bool int8 : {false, true}) {
      engine::FlashNextGpu::int8_act_moe_ = int8;
      auto gpu = engine::FlashNextGpu::Create(*dev, *weights, ngram.get(), nullptr, n, &error);
      ASSERT_TRUE(gpu) << error;
      std::vector<float> got(V);
      double worst = 0;
      std::vector<float> all(tokens.size() * V);
      for (std::size_t i = 0; i < tokens.size(); ++i) {
        ASSERT_TRUE(gpu->Step(tokens[i], got, &error)) << error;
        std::copy(got.begin(), got.end(), all.begin() + static_cast<std::ptrdiff_t>(i * V));
        const float* w = want[int8].data() + i * V;
        const double rel = RelL2(got, w);
        worst = std::max(worst, rel);
        ASSERT_LT(rel, int8 ? 1e-3 : 1e-4) << rt::BackendName(dev->backend()) << " int8 " << int8 << " position " << i;
        ASSERT_EQ(Argmax(got.data(), V), Argmax(w, V)) << "int8 " << int8 << " position " << i;
      }
      if (int8) EXPECT_LT(RelL2(all, want[1].data()), 0.1 * RelL2(want[1], want[0].data()));
      for (const char* k : {"halo_gemv_q8r_r2", "halo_gemv_q8r_r4", "halo_moe_down_q5_1_r4", "halo_moe_down_q8r_r4",
                             "halo_hc_mix_a", "halo_hc_mix_b", "halo_shexp_gate_up",
                             "halo_gemv_q8r_dual", "halo_gdn_step_d128_norm", "halo_matvec_f32_rowblock2"})
        EXPECT_TRUE(gpu->LaunchedKernel(k)) << k;
      EXPECT_EQ(gpu->LaunchedKernel("halo_moe_gate_up_q4_k_dot"), int8);
      EXPECT_EQ(gpu->LaunchedKernel("halo_moe_gate_up_q4_k_s2"), !int8);
      std::printf("%s int8 %d: worst relative L2 %.3g\n", rt::BackendName(dev->backend()), int8, worst);
    }
  engine::FlashNextGpu::int8_act_moe_ = saved;
}

// P4: chunked prefill (T-token GEMMs, row kernels, the DeltaNet chunk
// kernels) against stepping, over generated interleavings of prefills
// (crossing chunk boundaries: prefill_chunk_ is drawn small) and steps, in
// three modes:
// - exact projections (the decode GEMVs per token inside the chunk) with
//   float expert activations: equal to stepping up to float order. Bound
//   fixed in advance: relative L2 < 1e-4 and identical argmax everywhere.
//   With int8 expert activations (the default) a 1e-7 float-order
//   difference can flip one value's rounding, a discrete jump: on a
//   16-token alphabet, where routing and QSA scores sit near ties, that
//   measured up to 4e-4 against stepping while float activations stayed
//   below 1e-6 on the same sequence (and stepping matched the CPU
//   reference to 2e-7 either way). So the strict mode runs float.
// - fp16 GEMMs (Q8_0 dequantized to fp16 while staging, fp16 activations,
//   GpuQuant.Q8F16GemmMatchesDoubleSums; BF16 indexer projections on
//   FP16 WMMA, F32 ones on the row GEMM; routed experts grouped by expert on
//   FP16 WMMA, GpuQuant.GroupedExpertsMatchCpuReference; attention on FP16
//   WMMA with an online softmax) and int8 GEMMs (activations
//   quantized per token and 32 values): these change the arithmetic, so
//   stepping is not their oracle. Two checks instead:
//   * partition invariance, strict: every token's activations are
//     quantized on their own, so a second engine that splits the same
//     prompt at different points must agree to float order (relative L2
//     < 1e-4, identical argmax). This catches row-offset, tail and stride
//     errors in the GEMM plumbing, which noise cannot hide.
//   * drift from stepping, reported, with only a gross-error cap (relative
//     L2 < 0.5; a wrong row or a dropped scale moves logits by O(1)).
//     History: bounds of 3e-2, then 0.1, were set before measuring and
//     failed (0.033 on a one-token prefill, 0.112 after 92 tokens). A
//     diagnostic then ran every projection both ways inside the engine: the
//     int8 GEMM equalled the float GEMV on the same quantized activations
//     to < 2e-7 for every shape, while quantization moved activations by
//     ~0.55% per projection, and a rounding flip (a discrete jump) from
//     1e-7 noise is amplified through the random model's layers and state,
//     which is also why no strict int8-vs-float twin bound holds end to
//     end. Whether int8 dense prefill is accurate enough is decided on the
//     real model's NLL (halo_score), not here.
//     The worst drift is a tail, not a level: in these modes the stepped
//     oracle runs int8 expert activations too (logits 2-6e-3 from the CPU
//     reference), and a top-k choice near a tie (the router's last chosen
//     expert, or a QSA block at the budget) flips in one engine and not the
//     other, moving that position's logits by 0.025-0.134. Stream F
//     (2026-10-06) replayed seven outliers (0.026-0.134) against the CPU
//     reference; swapping exactly one choice in the reference brought the
//     outlying engine back to its noise floor (6e-4 to 3e-3) every time:
//     five flips in stepping (router logit gaps 3.6e-5 to 1.3e-3 of
//     ~0.25; QSA score gaps 0.024 of 34.9 and 0.44 of 18.4, the latter
//     with one block's score read back 0.70 low), one router flip in fp16
//     prefill (gap 1.4e-4 of 0.66, from --router-f16 3), one QSA flip in
//     int8 prefill (gap 0.055 of 20.7). With float expert activations,
//     stepping matched the reference to 2.7e-7 and the fp16 prefill stayed
//     within 6e-4 of it on the same cases (chunks of 1 / 17 / 64, up to
//     594 tokens); c95db3e, before the precision defaults, gave the same
//     stepped outliers. About 3% of fp16-mode checks exceed 0.02, hence the
//     count in the summary. A flip can also separate the chunked and split
//     engines (0.086 in int8 mode once stepping took float activations), as
//     large as a one-row error (0.07-0.11 for one changed token at a chunk
//     boundary), so the split bound below cannot tell the two apart there.
//   HALO_SMALL_QUANT_DIR  holds small-q.gguf
TEST(FlashNextGpu, ChunkedPrefillMatchesStepping) {
  const char* dir = std::getenv("HALO_SMALL_QUANT_DIR");
  if (!dir) GTEST_SKIP() << "set HALO_SMALL_QUANT_DIR";
  std::string error;
  auto reader = core::GgufReader::OpenFile(std::string(dir) + "/small-q.gguf", &error);
  ASSERT_TRUE(reader) << error;
  auto weights = model::ModelWeights::Bind(*reader, &error);
  ASSERT_TRUE(weights) << error;
  const auto& p = weights->ple_table;
  auto ngram = model::NgramTable::Open(reader->GetMappedRegions()[p.shard].file_descriptor, p.file_offset, p.rows,
                                       weights->config.ple_head_dim, p.type, &error);
  ASSERT_TRUE(ngram) << error;
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  const auto& c = weights->config;
  const std::size_t V = c.vocab_size;
  constexpr std::uint32_t kMaxContext = 640;
  using E = engine::FlashNextGpu;
  const bool saved_exact = E::prefill_exact_projections_, saved_int8 = E::prefill_int8_gemm_,
             saved_f16 = E::prefill_f16_gemm_,
             saved_moe = E::int8_act_moe_, saved_grouped = E::prefill_grouped_moe_,
             saved_wmma_attn = E::prefill_wmma_attention_;
  const std::uint32_t saved_chunk = E::prefill_chunk_;
  enum Mode { kExact, kFloatGemm, kInt8Gemm, kModes };
  const char* names[kModes] = {"exact", "fp16 GEMMs", "int8 GEMMs"};
  std::size_t compared[kModes] = {}, agreed[kModes] = {}, jumps[kModes] = {};
  double worst[kModes] = {}, worst_split[kModes] = {};
  hegel::test([&](hegel::TestCase& tc) {
    auto& dev = *devices[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = devices.size() - 1}))];
    const auto mode = static_cast<Mode>(tc.draw(gs::integers<int>({.min_value = 0, .max_value = kModes - 1})));
    E::prefill_exact_projections_ = mode == kExact;
    E::int8_act_moe_ = mode != kExact;
    E::prefill_grouped_moe_ = mode != kExact;  // fp16 expert GEMMs change the arithmetic
    E::prefill_wmma_attention_ = mode != kExact;  // and so does fp16 attention
    E::prefill_int8_gemm_ = mode == kInt8Gemm;
    E::prefill_f16_gemm_ = mode == kFloatGemm;
    E::prefill_chunk_ = tc.draw(gs::sampled_from(std::vector<std::uint32_t>{1, 17, 64, 256}));
    auto chunked = E::Create(dev, *weights, ngram.get(), nullptr, kMaxContext, &error);
    auto split = E::Create(dev, *weights, ngram.get(), nullptr, kMaxContext, &error);
    auto stepped = E::Create(dev, *weights, ngram.get(), nullptr, kMaxContext, &error);
    ASSERT_TRUE(chunked && split && stepped) << error;
    std::vector<float> a(V), b(V), s(V);
    const auto token = [&] {
      return tc.draw(gs::integers<int>({.min_value = 0, .max_value = 15})) == 0
                 ? static_cast<std::int32_t>(c.ple_eos_token)
                 : tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = static_cast<std::int32_t>(V - 1)}));
    };
    const auto check = [&](const char* what) {
      const double rel = RelL2(a, b.data()), rel_split = RelL2(a, s.data());
      worst[mode] = std::max(worst[mode], rel);
      worst_split[mode] = std::max(worst_split[mode], rel_split);
      ++compared[mode];
      jumps[mode] += rel > 0.01;  // mostly near-tie top-k flips in one engine (see above)
      const bool same = Argmax(a.data(), V) == Argmax(b.data(), V);
      agreed[mode] += same;
      const auto at = [&] { return std::string(what) + " at position " + std::to_string(chunked->Position()) + ", " + names[mode]; };
      // A split changes the chunk sizes, and the fp16 / int8 modes pick kernels and roundings by chunk size. Worst
      // split vs stepping over 30 cases (2026-10-06): before the precision defaults fp16 2.6e-4 / 4.2e-3, int8
      // 2.0e-3 / 3.0e-2; with them fp16 1.8e-3 / 3.3e-3, int8 1.3e-2 / 1.4e-2. Exact mode splits bit-identically.
      ASSERT_LT(rel_split, mode == kExact ? 1e-4 : 0.05) << "split " << at();
      ASSERT_EQ(Argmax(a.data(), V), Argmax(s.data(), V)) << "split " << at();
      ASSERT_LT(rel, mode == kExact ? 1e-4 : 0.5) << at();
      if (mode == kExact) ASSERT_TRUE(same) << at();
    };
    const int ops = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 6}));
    bool prefilled = false;
    for (int i = 0; i < ops; ++i) {
      if (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 2})) > 0) {
        const auto n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 300}));
        if (chunked->Position() + n > kMaxContext) break;
        std::vector<std::int32_t> chunk(n);
        for (auto& tok : chunk) tok = token();
        ASSERT_TRUE(chunked->Prefill(chunk, a, &error)) << error;
        prefilled = true;
        // The same tokens as two prefills, cut at a drawn point.
        const auto cut = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = n - 1}));
        const std::span<const std::int32_t> all(chunk);
        if (cut > 0) ASSERT_TRUE(split->Prefill(all.first(cut), {}, &error)) << error;
        ASSERT_TRUE(split->Prefill(all.subspan(cut), s, &error)) << error;
        for (std::uint32_t j = 0; j < n; ++j)
          ASSERT_TRUE(stepped->Step(chunk[j], j + 1 == n ? std::span<float>(b) : std::span<float>{}, &error)) << error;
        check("prefill");
      } else {
        if (chunked->Position() >= kMaxContext) break;
        const auto tok = token();
        ASSERT_TRUE(chunked->Step(tok, a, &error)) << error;
        ASSERT_TRUE(split->Step(tok, s, &error)) << error;
        ASSERT_TRUE(stepped->Step(tok, b, &error)) << error;
        check("step after prefill");
      }
      ASSERT_EQ(chunked->Position(), stepped->Position());
    }
    if (prefilled) {
      EXPECT_EQ(chunked->LaunchedKernel("halo_gdn_prefill_d128_norm"), !E::prefill_split_gdn_ || mode == kExact);
      EXPECT_EQ(chunked->LaunchedKernel("halo_gemm_q8r_wmma_t"), mode == kInt8Gemm);
      EXPECT_EQ(chunked->LaunchedKernel("halo_gemm_q8r_f16_l") || chunked->LaunchedKernel("halo_gemm_q8r_f16_s"),
                mode == kFloatGemm);
      EXPECT_EQ(chunked->LaunchedKernel("halo_gemm_bf16_f16_k128"), mode != kExact);
      EXPECT_EQ(chunked->LaunchedKernel(E::prefill_parallel_bucket_ ? "halo_moe_bucket_count" : "halo_moe_bucket"),
                mode != kExact);
      // Any of the WMMA attention paths (flash, multi-query, QSA-selected, per-row); the check predated the flash kernels.
      bool wmma_attention = false;
      for (const char* k : {"halo_attention_fa", "halo_attention_fa_h", "halo_attention_fa_hg", "halo_attention_wmma_mq",
                            "halo_attention_wmma_qsel", "halo_attention_wmma_qsel_h", "halo_attention_wmma_rows"})
        wmma_attention = wmma_attention || chunked->LaunchedKernel(k);
      EXPECT_TRUE(wmma_attention || mode == kExact);
    }
  }, {.test_cases = 30});
  E::prefill_exact_projections_ = saved_exact;
  E::prefill_int8_gemm_ = saved_int8;
  E::prefill_f16_gemm_ = saved_f16;
  E::int8_act_moe_ = saved_moe;
  E::prefill_grouped_moe_ = saved_grouped;
  E::prefill_chunk_ = saved_chunk;
  E::prefill_wmma_attention_ = saved_wmma_attn;
  for (int m = 0; m < kModes; ++m)
    std::printf("%s: %zu compared, argmax agreed %zu, worst relative L2 vs stepping %.3g (%zu above 0.01), vs a split "
                "prefill %.3g\n",
                names[m], compared[m], agreed[m], worst[m], jumps[m], worst_split[m]);
}
