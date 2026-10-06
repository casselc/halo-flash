// P10 variants on the Qwen3.5 GPU engine, on the tiny model:
//  - merged and unmerged LoRA agree (CPU reference and GPU), the merged
//    model being W + (alpha/r) B A materialized in F32;
//  - several adapters resident at once, selected per call, each equal to
//    the CPU reference with that adapter, and the base unchanged;
//  - a variant loaded through its manifest (base hash checked) drives the
//    engine like the bare adapter file;
//  - the residual capture at a layer boundary (the stage-1 trainer's trunk)
//    equals the CPU reference's.
// Tolerances fixed in advance: relative L2 < 1e-5 for merged vs unmerged
// on the CPU (same FP32 arithmetic up to association), < 1e-4 for any GPU
// vs CPU comparison (the existing Qwen35Gpu bound), identical argmax.
//   HALO_TINY_QWEN35_DIR  tiny model (make_tiny_qwen35_reference.py + tiny-qwen35-f32.gguf)
//   HALO_GRAD_REF_DIR     two random LoRAs on its last four layers (qwen35_grad_reference.py)

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/sha256.hpp"
#include "engine/qwen35_gpu.hpp"
#include "model/qwen35.hpp"
#include "model/variant.hpp"
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

std::size_t Argmax(const std::vector<float>& v) {
  return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

struct Tiny {
  std::string dir, ref;
  std::unique_ptr<core::GgufReader> base, lora_a_file, lora_b_file;
  std::optional<model::Qwen35Weights> w;
  std::optional<model::LoraAdapter> lora_a, lora_b;
  bool Load() {
    const char* d = std::getenv("HALO_TINY_QWEN35_DIR");
    const char* r = std::getenv("HALO_GRAD_REF_DIR");
    if (!d || !r) return false;
    dir = d;
    ref = r;
    std::string error;
    base = core::GgufReader::OpenFile(dir + "/tiny-qwen35-f32.gguf", &error);
    lora_a_file = core::GgufReader::OpenFile(ref + "/lm_lora_init.gguf", &error);
    lora_b_file = core::GgufReader::OpenFile(ref + "/cls_lora_init.gguf", &error);
    EXPECT_TRUE(base && lora_a_file && lora_b_file) << error;
    if (!base || !lora_a_file || !lora_b_file) return false;
    w = model::Qwen35Weights::Bind(*base, &error);
    lora_a = model::LoraAdapter::Bind(*lora_a_file, *base, &error);
    lora_b = model::LoraAdapter::Bind(*lora_b_file, *base, &error);
    EXPECT_TRUE(w && lora_a && lora_b) << error;
    return w && lora_a && lora_b;
  }
};

// Base weights with every adapted tensor replaced by its merged F32 copy.
struct Merged {
  model::Qwen35Weights w;
  std::vector<std::vector<float>> storage;
  Merged(const model::Qwen35Weights& base, const model::LoraAdapter& lora) : w(base) {
    storage.reserve(lora.pairs.size());
    auto merge = [&](model::TensorRef& t) {
      if (t.empty()) return;
      const auto* p = lora.Find(t.name);
      if (!p) return;
      storage.push_back(model::MergeLora(t, *p, lora.Scale(*p)));
      t.data = storage.back().data();
      t.type = core::GgmlType::kF32;
      t.name = {};  // a merged tensor carries no adapter
    };
    for (auto& L : w.layers)
      for (auto* t : {&L.ssm_qkv, &L.ssm_gate, &L.ssm_alpha, &L.ssm_beta, &L.ssm_out, &L.attn_q, &L.attn_k, &L.attn_v,
                      &L.attn_out, &L.ffn_gate, &L.ffn_up, &L.ffn_down})
        merge(*t);
  }
};

std::vector<std::int32_t> Tokens() {
  std::vector<std::int32_t> t;
  for (int i = 0; i < 20; ++i) t.push_back((i * 7919 + 101) % 248000);
  return t;
}

std::vector<std::vector<float>> RunCpu(const model::Qwen35Weights& w, const model::LoraAdapter* lora) {
  model::Qwen35Reference ref(w, 64);
  ref.SetAdapter(lora);
  std::vector<std::vector<float>> out;
  std::vector<float> l(w.config.vocab_size);
  std::string error;
  for (auto t : Tokens()) {
    EXPECT_TRUE(ref.Step(t, {}, l, &error)) << error;
    out.push_back(l);
  }
  return out;
}

std::vector<std::vector<float>> RunGpu(engine::Qwen35Gpu& gpu, std::size_t vocab) {
  std::vector<std::vector<float>> out;
  std::vector<float> l(vocab);
  std::string error;
  EXPECT_TRUE(gpu.Reset(&error)) << error;
  for (auto t : Tokens()) {
    EXPECT_TRUE(gpu.Step(t, {}, l, &error)) << error;
    out.push_back(l);
  }
  return out;
}

double Worst(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, bool argmax) {
  double worst = 0;
  for (std::size_t p = 0; p < a.size(); ++p) {
    worst = std::max(worst, RelL2(a[p], b[p]));
    if (argmax) EXPECT_EQ(Argmax(a[p]), Argmax(b[p])) << "position " << p;
  }
  return worst;
}

}  // namespace

TEST(VariantCpu, MergedAndUnmergedLoraAgree) {
  Tiny t;
  if (!t.Load()) GTEST_SKIP() << "set HALO_TINY_QWEN35_DIR and HALO_GRAD_REF_DIR";
  for (const auto* lora : {&*t.lora_a, &*t.lora_b}) {
    Merged m(*t.w, *lora);
    const double e = Worst(RunCpu(m.w, nullptr), RunCpu(*t.w, lora), true);
    std::printf("CPU merged vs unmerged: worst relative L2 %.3g\n", e);
    EXPECT_LT(e, 1e-5);
  }
}

TEST(VariantGpu, ResidentAdaptersMatchReferenceAndMergedModel) {
  Tiny t;
  if (!t.Load()) GTEST_SKIP() << "set HALO_TINY_QWEN35_DIR and HALO_GRAD_REF_DIR";
  std::string error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  ASSERT_TRUE(dev) << error;
  const auto V = t.w->config.vocab_size;
  auto gpu = engine::Qwen35Gpu::Create(*dev, *t.w, nullptr, 64, &error);
  ASSERT_TRUE(gpu) << error;
  ASSERT_TRUE(gpu->AddAdapter("a", *t.lora_a, &error)) << error;
  ASSERT_TRUE(gpu->AddAdapter("b", *t.lora_b, &error)) << error;
  EXPECT_FALSE(gpu->AddAdapter("a", *t.lora_b, &error));
  EXPECT_FALSE(gpu->SelectAdapter("missing", &error));
  const auto cpu_base = RunCpu(*t.w, nullptr);
  const auto cpu_a = RunCpu(*t.w, &*t.lora_a);
  const auto cpu_b = RunCpu(*t.w, &*t.lora_b);
  // Interleave selections: a, b, base, a again.
  ASSERT_TRUE(gpu->SelectAdapter("a", &error));
  const auto ga = RunGpu(*gpu, V);
  ASSERT_TRUE(gpu->SelectAdapter("b", &error));
  const auto gb = RunGpu(*gpu, V);
  ASSERT_TRUE(gpu->SelectAdapter("", &error));
  const auto g0 = RunGpu(*gpu, V);
  ASSERT_TRUE(gpu->SelectAdapter("a", &error));
  const auto ga2 = RunGpu(*gpu, V);
  const double ea = Worst(ga, cpu_a, true), eb = Worst(gb, cpu_b, true), e0 = Worst(g0, cpu_base, true);
  std::printf("GPU vs CPU: adapter a %.3g, adapter b %.3g, base %.3g\n", ea, eb, e0);
  EXPECT_LT(ea, 1e-4);
  EXPECT_LT(eb, 1e-4);
  EXPECT_LT(e0, 1e-4);
  EXPECT_EQ(Worst(ga, ga2, false), 0.0) << "reselecting an adapter must reproduce it exactly";
  // The adapters really differ from each other and from the base.
  EXPECT_GT(Worst(ga, g0, false), 1e-2);
  EXPECT_GT(Worst(ga, gb, false), 1e-2);
  // Merged models on the GPU.
  for (const auto& [lora, ref] : {std::pair{&*t.lora_a, &ga}, std::pair{&*t.lora_b, &gb}}) {
    Merged m(*t.w, *lora);
    auto merged = engine::Qwen35Gpu::Create(*dev, m.w, nullptr, 64, &error);
    ASSERT_TRUE(merged) << error;
    const double e = Worst(RunGpu(*merged, V), *ref, true);
    std::printf("GPU merged vs unmerged: %.3g\n", e);
    EXPECT_LT(e, 1e-4);
  }
  EXPECT_TRUE(gpu->RemoveAdapter("a"));
  EXPECT_EQ(gpu->selected_adapter(), "");
  EXPECT_FALSE(gpu->SelectAdapter("a", &error));
}

TEST(VariantGpu, ManifestLoadedVariantDrivesTheEngine) {
  Tiny t;
  if (!t.Load()) GTEST_SKIP() << "set HALO_TINY_QWEN35_DIR and HALO_GRAD_REF_DIR";
  namespace fs = std::filesystem;
  const fs::path dir = fs::path(testing::TempDir()) / "halo_variant_gpu";
  fs::remove_all(dir);
  fs::create_directories(dir);
  fs::copy_file(t.ref + "/lm_lora_init.gguf", dir / "lora.gguf");
  std::string error;
  const std::vector<std::string> files = {t.dir + "/tiny-qwen35-f32.gguf"};
  const auto hash = model::BaseContentHash(files, &error);
  ASSERT_TRUE(hash) << error;
  model::VariantManifest m;
  m.name = "tiny-a";
  m.base_architecture = "qwen35";
  m.base_sha256 = *hash;
  m.parts.push_back({model::VariantPartKind::kLora, "lora.gguf", *core::Sha256File((dir / "lora.gguf").string(), &error),
                     fs::file_size(dir / "lora.gguf")});
  ASSERT_TRUE(m.WriteFile((dir / "manifest.json").string(), &error)) << error;
  model::VariantRegistry reg(*t.base, *hash);
  const auto v = reg.Load((dir / "manifest.json").string(), &error);
  ASSERT_TRUE(v) << error;
  auto dev = rt::Device::Create(rt::Backend::kHip, &error);
  ASSERT_TRUE(dev) << error;
  auto gpu = engine::Qwen35Gpu::Create(*dev, *t.w, nullptr, 64, &error);
  ASSERT_TRUE(gpu) << error;
  ASSERT_TRUE(gpu->AddAdapter(v->manifest.name, *v->lora, &error)) << error;
  ASSERT_TRUE(gpu->SelectAdapter("tiny-a", &error)) << error;
  const double e = Worst(RunGpu(*gpu, t.w->config.vocab_size), RunCpu(*t.w, &*t.lora_a), true);
  EXPECT_LT(e, 1e-4);
  fs::remove_all(dir);
}

TEST(VariantCpu, StepToLayerCapturesTheResidualEnteringALayer) {
  Tiny t;
  if (!t.Load()) GTEST_SKIP() << "set HALO_TINY_QWEN35_DIR and HALO_GRAD_REF_DIR";
  // The grad reference's boundary is HF's hidden_states[L - 4] on tokens.i32.
  std::FILE* f = std::fopen((t.dir + "/tokens.i32").c_str(), "rb");
  ASSERT_TRUE(f);
  std::vector<std::int32_t> tokens(64);
  tokens.resize(std::fread(tokens.data(), 4, tokens.size(), f));
  std::fclose(f);
  f = std::fopen((t.ref + "/boundary.f32").c_str(), "rb");
  ASSERT_TRUE(f);
  const std::size_t H = t.w->config.hidden_size;
  std::vector<float> want(tokens.size() * H);
  ASSERT_EQ(std::fread(want.data(), 4, want.size(), f), want.size());
  std::fclose(f);
  model::Qwen35Reference ref(*t.w, 64);
  std::vector<float> got(tokens.size() * H);
  std::string error;
  const std::uint32_t layer = t.w->config.num_layers - 4;
  for (std::size_t p = 0; p < tokens.size(); ++p)
    ASSERT_TRUE(ref.StepToLayer(tokens[p], layer, std::span<float>(got).subspan(p * H, H), &error)) << error;
  const double e = RelL2(got, want);
  std::printf("CPU trunk capture vs HF hidden_states: %.3g\n", e);
  EXPECT_LT(e, 1e-4);
}
