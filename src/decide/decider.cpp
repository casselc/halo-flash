#include "decide/decider.hpp"

#include <cmath>
#include <cstring>

namespace halo::decide {

namespace {

std::vector<float> ReadF32(const core::GgufReader& r, const std::string& name, std::size_t n, std::string* error) {
  const auto* t = r.FindTensor(name);
  if (!t || t->type != core::GgmlType::kF32 || t->ElementCount() != n) {
    if (error) *error = "head tensor " + name + " is missing or malformed";
    return {};
  }
  std::vector<float> out(n);
  std::memcpy(out.data(), t->data, n * sizeof(float));
  return out;
}

std::vector<float> LayerNorm(std::span<const float> x, const std::vector<float>& w, const std::vector<float>& b) {
  // torch.nn.LayerNorm(eps=1e-5), FP32.
  float mean = 0;
  for (float v : x) mean += v;
  mean /= static_cast<float>(x.size());
  float var = 0;
  for (float v : x) var += (v - mean) * (v - mean);
  var /= static_cast<float>(x.size());
  const float inv = 1.0F / std::sqrt(var + 1e-5F);
  std::vector<float> y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) y[i] = (x[i] - mean) * inv * w[i] + b[i];
  return y;
}

std::vector<float> Affine(const std::vector<float>& w, const std::vector<float>& b, std::span<const float> x) {
  const std::size_t out = b.size(), in = x.size();
  std::vector<float> y(out);
  for (std::size_t r = 0; r < out; ++r) {
    double acc = 0;
    for (std::size_t c = 0; c < in; ++c) acc += static_cast<double>(w[r * in + c]) * x[c];
    y[r] = static_cast<float>(acc) + b[r];
  }
  return y;
}

}  // namespace

std::vector<float> PointerHead::Logits(std::span<const float> decide, const std::vector<std::vector<float>>& options) const {
  const auto q = Affine(q_w, q_b, LayerNorm(decide, norm_w, norm_b));
  const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
  std::vector<float> logits;
  for (const auto& o : options) {
    const auto k = Affine(k_w, k_b, LayerNorm(o, norm_w, norm_b));
    double dot = 0;
    for (std::uint32_t i = 0; i < dim; ++i) dot += static_cast<double>(k[i]) * q[i];
    logits.push_back(static_cast<float>(dot) * scale);
  }
  return logits;
}

std::unique_ptr<StrandsDecider> StrandsDecider::Open(const DeciderFiles& f, std::string* error) {
  auto d = std::unique_ptr<StrandsDecider>(new StrandsDecider());
  d->torso_file_ = core::GgufReader::OpenFile(f.torso_gguf, error);
  if (!d->torso_file_) return nullptr;
  d->torso_ = model::Qwen35Weights::Bind(*d->torso_file_, error);
  if (!d->torso_) return nullptr;
  if (!f.lora_gguf.empty()) {
    d->lora_file_ = core::GgufReader::OpenFile(f.lora_gguf, error);
    if (!d->lora_file_) return nullptr;
    d->lora_ = model::LoraAdapter::Bind(*d->lora_file_, *d->torso_file_, error);
    if (!d->lora_) return nullptr;
  }
  d->tok_file_ = core::GgufReader::OpenFile(f.tokenizer_gguf, error);
  if (!d->tok_file_) return nullptr;
  d->tokenizer_ = text::Tokenizer::FromGguf(*d->tok_file_, error);
  if (!d->tokenizer_) return nullptr;
  d->head_file_ = core::GgufReader::OpenFile(f.head_gguf, error);
  if (!d->head_file_) return nullptr;
  const auto& h = *d->head_file_;
  if (h.GetMetadataString("decider.head_type") != std::optional<std::string_view>{"pointer"}) {
    if (error) *error = "head GGUF is not a pointer head";
    return nullptr;
  }
  auto& ph = d->head_;
  ph.hidden = d->torso_->config.hidden_size;
  ph.dim = h.GetMetadataUint32("decider.pointer_dim").value_or(0);
  ph.norm_w = ReadF32(h, "head.norm.weight", ph.hidden, error);
  ph.norm_b = ReadF32(h, "head.norm.bias", ph.hidden, error);
  ph.q_w = ReadF32(h, "head.q.weight", std::size_t{ph.dim} * ph.hidden, error);
  ph.q_b = ReadF32(h, "head.q.bias", ph.dim, error);
  ph.k_w = ReadF32(h, "head.k.weight", std::size_t{ph.dim} * ph.hidden, error);
  ph.k_b = ReadF32(h, "head.k.bias", ph.dim, error);
  if (ph.dim == 0 || ph.norm_w.empty() || ph.norm_b.empty() || ph.q_w.empty() || ph.q_b.empty() ||
      ph.k_w.empty() || ph.k_b.empty()) {
    if (error && error->empty()) *error = "malformed pointer head";
    return nullptr;
  }
  d->temperature_ = h.GetMetadataFloat32("decider.temperature").value_or(1.0F);
  for (Kind k : {Kind::kNoul, Kind::kChoice, Kind::kScore})
    if (auto t = h.GetMetadataFloat32(std::string("decider.temperature.") + KindName(k)))
      d->temperature_by_kind_.emplace_back(k, *t);
  d->ordinal_smoothing_ = h.GetMetadataFloat32("decider.ordinal_smoothing").value_or(0.0F);
  d->fit_.max_length = h.GetMetadataUint32("decider.max_length").value_or(4096);
  return d;
}

float StrandsDecider::Temperature(Kind kind) const {
  for (const auto& [k, t] : temperature_by_kind_)
    if (k == kind) return t;
  return temperature_;
}

std::optional<json::Value> StrandsDecider::Evaluate(const Request& r, std::string* error) {
  if (!r.images.empty()) {
    if (error) *error = "this engine has no vision tower";
    return std::nullopt;
  }
  std::vector<RenderedQuestion> rqs;
  for (const auto& q : r.questions) rqs.push_back(RenderQuestion(q));
  auto fit = FitTokens(*tokenizer_, RenderState(r.state), rqs, fit_, error);
  if (!fit) return std::nullopt;

  const auto& c = torso_->config;
  std::size_t total = fit->state_ids.size();
  for (const auto& q : fit->question_ids) total += q.size();
  model::Qwen35Reference prefix(*torso_, static_cast<std::uint32_t>(total));
  if (lora_) prefix.SetAdapter(&*lora_);
  for (auto id : fit->state_ids)
    if (!prefix.Step(static_cast<std::int32_t>(id), {}, {}, error)) return std::nullopt;

  json::Value answers = json::Value::MakeObject();
  std::vector<float> h(c.hidden_size);
  for (std::size_t q = 0; q < rqs.size(); ++q) {
    model::Qwen35Reference run = prefix;  // snapshot of the shared state prefix
    std::vector<std::vector<float>> hidden;
    for (auto id : fit->question_ids[q]) {
      if (!run.Step(static_cast<std::int32_t>(id), h, {}, error)) return std::nullopt;
      hidden.push_back(h);
    }
    std::vector<std::vector<float>> options;
    for (auto idx : fit->option_index[q]) options.push_back(hidden.at(idx));
    const auto logits = head_.Logits(hidden.back(), options);
    const auto probs = SlotProbabilities(logits, Temperature(rqs[q].kind));
    answers.set(r.questions[q].name, AnswerToJson(ReadAnswer(rqs[q], probs, ordinal_smoothing_)));
  }
  json::Value usage = json::Value::MakeObject();
  usage.set("input_tokens", json::Value::Int(static_cast<std::int64_t>(total)));
  usage.set("output_tokens", json::Value::Int(static_cast<std::int64_t>(rqs.size())));
  json::Value resp = json::Value::MakeObject();
  resp.set("model", json::Value::String(model_name));
  resp.set("answers", std::move(answers));
  resp.set("usage", std::move(usage));
  return resp;
}

}  // namespace halo::decide
