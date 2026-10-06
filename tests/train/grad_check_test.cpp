// Stage-1 trainer gradients (src/train/last_k.cpp) against PyTorch autograd
// in float64 (tools/qwen35_grad_reference.py), layer by layer, on the tiny
// Qwen3.5 model with k = 4 (three Gated DeltaNet layers and one gated
// attention layer), LoRA on all twelve projection kinds, and both heads.
//
// Tolerances, fixed before the first run (FP32 trainer vs FP64 reference;
// the forward parity of the CPU reference with HF is 1e-4 relative L2):
//   loss                                   relative error   <= 1e-5
//   d(loss)/d(residual) entering each of
//     the last k layers and the final norm relative L2       <= 1e-4
//   every LoRA A and B gradient tensor     relative L2       <= 1e-4
//   every classifier-head gradient tensor  relative L2       <= 1e-4
//
//   HALO_TINY_QWEN35_DIR  make_tiny_qwen35_reference.py output (+ tiny-qwen35-f32.gguf)
//   HALO_GRAD_REF_DIR     qwen35_grad_reference.py output

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/json.hpp"
#include "model/lora.hpp"
#include "model/qwen35.hpp"
#include "train/last_k.hpp"

using namespace halo;

namespace {

constexpr double kLossTol = 1e-5, kGradTol = 1e-4;

std::vector<float> ReadF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(f.good()) << path;
  std::vector<float> out(static_cast<std::size_t>(f.tellg()) / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
  return out;
}

json::Value ReadJson(const std::string& path) {
  std::ifstream f(path);
  std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::string error;
  auto v = json::Parse(text, &error);
  EXPECT_TRUE(v) << path << ": " << error;
  return v ? *v : json::Value();
}

double RelL2(const float* a, const float* b, std::size_t n) {
  double d = 0, m = 0;
  for (std::size_t i = 0; i < n; ++i) {
    d += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
    m += double(b[i]) * b[i];
  }
  return m > 0 ? std::sqrt(d / m) : std::sqrt(d);
}

struct Fixture {
  std::string tiny, ref;
  std::unique_ptr<core::GgufReader> base;
  std::optional<model::Qwen35Weights> w;
  bool Load() {
    const char* t = std::getenv("HALO_TINY_QWEN35_DIR");
    const char* r = std::getenv("HALO_GRAD_REF_DIR");
    if (!t || !r) return false;
    tiny = t;
    ref = r;
    std::string error;
    base = core::GgufReader::OpenFile(tiny + "/tiny-qwen35-f32.gguf", &error);
    EXPECT_TRUE(base) << error;
    if (!base) return false;
    w = model::Qwen35Weights::Bind(*base, &error);
    EXPECT_TRUE(w) << error;
    return w.has_value();
  }
};

train::TailConfig Config(std::uint32_t k, bool classifier) {
  train::TailConfig c;
  c.last_k = k;
  c.lora_rank = 4;
  c.lora_alpha = 8;
  c.lora_targets = {"attn_q",   "attn_k",    "attn_v",    "attn_output", "attn_qkv", "attn_gate",
                    "ssm_alpha", "ssm_beta", "ssm_out", "ffn_gate",    "ffn_up",   "ffn_down"};
  if (classifier) {
    c.head = train::HeadKind::kClassifier;
    c.num_classes = 3;
  }
  return c;
}

void CheckCase(Fixture& fx, const std::string& name) {
  const auto meta = ReadJson(fx.ref + "/" + name + ".json");
  const auto k = static_cast<std::uint32_t>(meta.find("k")->number());
  const auto T = static_cast<std::uint32_t>(meta.find("T")->number());
  const bool cls = name == "cls";
  std::string error;
  auto tail = train::LastK::Create(*fx.w, Config(k, cls), &error);
  ASSERT_TRUE(tail) << error;
  auto init_file = core::GgufReader::OpenFile(fx.ref + "/" + name + "_lora_init.gguf", &error);
  ASSERT_TRUE(init_file) << error;
  auto init = model::LoraAdapter::Bind(*init_file, *fx.base, &error);
  ASSERT_TRUE(init) << error;
  ASSERT_TRUE(tail->LoadLora(*init, &error)) << error;
  if (cls) {
    for (const char* p : {"norm.weight", "norm.bias", "out.weight", "out.bias"}) {
      const auto v = ReadF32(fx.ref + "/cls_head_" + p + ".f32");
      const auto* s = tail->FindSlice(std::string("head.") + p);
      ASSERT_TRUE(s);
      ASSERT_EQ(v.size(), s->size());
      std::copy(v.begin(), v.end(), tail->params().begin() + static_cast<std::ptrdiff_t>(s->offset));
    }
  }
  const auto boundary = ReadF32(fx.ref + "/boundary.f32");
  train::Example ex;
  ex.boundary = boundary;
  ex.tokens = T;
  if (cls) {
    ex.label = static_cast<std::int32_t>(meta.find("label")->number());
    ex.label_position = static_cast<std::uint32_t>(meta.find("label_position")->number());
  } else {
    const auto& pos = meta.find("positions")->items();
    const auto& tgt = meta.find("targets")->items();
    for (std::size_t i = 0; i < pos.size(); ++i)
      ex.lm_targets.emplace_back(static_cast<std::uint32_t>(pos[i].number()), static_cast<std::int32_t>(tgt[i].number()));
  }
  const std::uint32_t count = cls ? 1 : static_cast<std::uint32_t>(ex.lm_targets.size());
  train::ThreadPool pool(4);
  std::vector<float> grad(tail->num_params(), 0.0F);
  train::Probe probe;
  const auto loss = tail->ForwardBackward(ex, 1.0F / static_cast<float>(count), grad, pool, &probe);
  ASSERT_EQ(loss.count, count);
  const double want_loss = meta.find("loss")->number();
  const double got_loss = loss.sum / count;
  const double loss_err = std::abs(got_loss - want_loss) / std::abs(want_loss);
  EXPECT_LE(loss_err, kLossTol) << "loss " << got_loss << " vs " << want_loss;
  std::printf("[%s] loss %.8f vs %.8f (rel %.2e)\n", name.c_str(), got_loss, want_loss, loss_err);

  const auto d_res = ReadF32(fx.ref + "/" + name + "_d_residual.f32");
  const std::size_t H = fx.w->config.hidden_size;
  ASSERT_EQ(d_res.size(), (k + 1) * T * H);
  ASSERT_EQ(probe.d_residual.size(), k + 1);
  double worst = 0;
  for (std::uint32_t i = 0; i <= k; ++i) {
    const double e = RelL2(probe.d_residual[i].data(), d_res.data() + i * T * H, T * H);
    worst = std::max(worst, e);
    EXPECT_LE(e, kGradTol) << "d residual entering " << (i < k ? "layer " + std::to_string(tail->first_layer() + i)
                                                                 : std::string("the final norm"));
    std::printf("[%s] d residual %-14s rel L2 %.2e\n", name.c_str(),
                (i < k ? "layer " + std::to_string(tail->first_layer() + i) : std::string("final norm")).c_str(), e);
  }
  auto grad_file = core::GgufReader::OpenFile(fx.ref + "/" + name + "_lora_grad.gguf", &error);
  ASSERT_TRUE(grad_file) << error;
  auto want = model::LoraAdapter::Bind(*grad_file, *fx.base, &error);
  ASSERT_TRUE(want) << error;
  std::size_t checked = 0;
  double worst_lora = 0;
  std::string worst_name;
  for (const auto& s : tail->slices()) {
    if (s.is_head) continue;
    const bool is_a = s.name.ends_with(".lora_a");
    const auto* pair = want->Find(s.name.substr(0, s.name.size() - 7));
    ASSERT_TRUE(pair) << s.name;
    const auto& t = is_a ? pair->a : pair->b;
    ASSERT_EQ(t.rows * t.cols, s.size());
    const double e = RelL2(grad.data() + s.offset, static_cast<const float*>(t.data), s.size());
    if (e > worst_lora) {
      worst_lora = e;
      worst_name = s.name;
    }
    EXPECT_LE(e, kGradTol) << s.name;
    ++checked;
  }
  EXPECT_EQ(checked, 2 * want->pairs.size());
  std::printf("[%s] %zu LoRA gradient tensors, worst rel L2 %.2e (%s)\n", name.c_str(), checked, worst_lora,
              worst_name.c_str());
  if (cls) {
    for (const char* p : {"norm.weight", "norm.bias", "out.weight", "out.bias"}) {
      const auto v = ReadF32(fx.ref + "/cls_head_" + p + ".grad.f32");
      const auto* s = tail->FindSlice(std::string("head.") + p);
      const double e = RelL2(grad.data() + s->offset, v.data(), v.size());
      EXPECT_LE(e, kGradTol) << p;
      std::printf("[cls] head.%-12s rel L2 %.2e\n", p, e);
    }
  }
}

}  // namespace

TEST(TrainGradCheck, LmHeadLossWithLoraOnLastFourLayers) {
  Fixture fx;
  if (!fx.Load()) GTEST_SKIP() << "set HALO_TINY_QWEN35_DIR and HALO_GRAD_REF_DIR";
  CheckCase(fx, "lm");
}

TEST(TrainGradCheck, ClassifierHeadWithLoraOnLastFourLayers) {
  Fixture fx;
  if (!fx.Load()) GTEST_SKIP() << "set HALO_TINY_QWEN35_DIR and HALO_GRAD_REF_DIR";
  CheckCase(fx, "cls");
}
