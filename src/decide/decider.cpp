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
  auto readout = Readout::FromGguf(h, d->torso_->config.hidden_size, error);
  if (!readout) return nullptr;
  d->readout_ = std::move(*readout);
  d->fit_.max_length = h.GetMetadataUint32("decider.max_length").value_or(4096);
  if (f.device != "cpu") {
#if HALO_WITH_HIP
    const auto backend = f.device == "hrx" ? rt::Backend::kHrx : rt::Backend::kHip;
    d->dev_ = rt::Device::Create(backend, error);
    if (!d->dev_) return nullptr;
    d->gpu_ = engine::Qwen35Gpu::Create(*d->dev_, *d->torso_, d->lora_ ? &*d->lora_ : nullptr,
                                        static_cast<std::uint32_t>(d->fit_.max_length + 8), error);
    if (!d->gpu_) return nullptr;
#else
    if (error) *error = "built without GPU support";
    return nullptr;
#endif
  }
  return d;
}

float Readout::Temperature(Kind kind) const {
  for (const auto& [k, t] : temperature_by_kind)
    if (k == kind) return t;
  return temperature;
}

std::optional<Readout> Readout::FromGguf(const core::GgufReader& h, std::uint32_t hidden, std::string* error) {
  if (h.GetMetadataString("decider.head_type") != std::optional<std::string_view>{"pointer"}) {
    if (error) *error = "head GGUF is not a pointer head";
    return std::nullopt;
  }
  Readout r;
  auto& ph = r.head;
  ph.hidden = hidden;
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
    return std::nullopt;
  }
  r.temperature = h.GetMetadataFloat32("decider.temperature").value_or(1.0F);
  for (Kind k : {Kind::kNoul, Kind::kChoice, Kind::kScore})
    if (auto t = h.GetMetadataFloat32(std::string("decider.temperature.") + KindName(k)))
      r.temperature_by_kind.emplace_back(k, *t);
  r.ordinal_smoothing = h.GetMetadataFloat32("decider.ordinal_smoothing").value_or(0.0F);
  return r;
}

bool StrandsDecider::AddVariant(std::shared_ptr<const model::LoadedVariant> v, std::string* error) {
  if (!v || !v->head() || !v->manifest.Find(model::VariantPartKind::kPointerHead)) {
    if (error) *error = "a decider variant needs a pointer_head part";
    return false;
  }
  const std::string& name = v->manifest.name;
  if (variants_.contains(name) || name == "default") {
    if (error) *error = "variant " + name + " is already added";
    return false;
  }
  auto readout = Readout::FromGguf(*v->head(), torso_->config.hidden_size, error);
  if (!readout) return false;
#if HALO_WITH_HIP
  if (gpu_ && v->lora && !gpu_->AddAdapter(name, *v->lora, error)) return false;
#endif
  variants_.emplace(name, Variant{std::move(v), std::move(*readout)});
  return true;
}

std::vector<std::string> StrandsDecider::Variants() const {
  std::vector<std::string> out;
  for (const auto& [n, _] : variants_) out.push_back(n);
  return out;
}

std::optional<json::Value> StrandsDecider::Evaluate(const Request& r, std::string* error) {
  if (!r.images.empty()) {
    if (error) *error = "this engine has no vision tower";
    return std::nullopt;
  }
  // The variant: its LoRA (or none) and readout, else the decider's own.
  const Readout* readout = &readout_;
  const model::LoraAdapter* lora = lora_ ? &*lora_ : nullptr;
  std::string adapter = lora_ ? "default" : "";
  if (!r.variant.empty()) {
    const auto it = variants_.find(r.variant);
    if (it == variants_.end()) {
      if (error) *error = "unknown variant " + r.variant;
      return std::nullopt;
    }
    readout = &it->second.readout;
    lora = it->second.v->lora ? &*it->second.v->lora : nullptr;
    adapter = lora ? r.variant : "";
  }
  std::vector<RenderedQuestion> rqs;
  for (const auto& q : r.questions) rqs.push_back(RenderQuestion(q));
  auto fit = FitTokens(*tokenizer_, RenderState(r.state), rqs, fit_, error);
  if (!fit) return std::nullopt;

  const auto& c = torso_->config;
  std::size_t total = fit->state_ids.size();
  for (const auto& q : fit->question_ids) total += q.size();
  std::vector<std::vector<std::vector<float>>> hidden_by_q(rqs.size());
  std::vector<float> h(c.hidden_size);
#if HALO_WITH_HIP
  if (gpu_) {
    // The GPU model is reused across requests: rewind by restoring an empty
    // state, run the state prefix once, save it, and restore it per question.
    // Each part is one chunked prefill (P4); the question's hidden states
    // come back for every position.
    if (!gpu_->SelectAdapter(adapter, error) || !gpu_->Reset(error)) return std::nullopt;
    const std::vector<std::int32_t> state(fit->state_ids.begin(), fit->state_ids.end());
    if (!gpu_->Prefill(state, {}, error)) return std::nullopt;
    if (!gpu_->SaveState(error)) return std::nullopt;
    std::vector<float> rows;
    for (std::size_t q = 0; q < rqs.size(); ++q) {
      if (q > 0 && !gpu_->RestoreState(error)) return std::nullopt;
      const std::vector<std::int32_t> ids(fit->question_ids[q].begin(), fit->question_ids[q].end());
      rows.resize(ids.size() * c.hidden_size);
      if (!gpu_->Prefill(ids, rows, error)) return std::nullopt;
      for (std::size_t i = 0; i < ids.size(); ++i)
        hidden_by_q[q].emplace_back(rows.begin() + static_cast<std::ptrdiff_t>(i * c.hidden_size),
                                    rows.begin() + static_cast<std::ptrdiff_t>((i + 1) * c.hidden_size));
    }
  }
#endif
  if (hidden_by_q.empty() || hidden_by_q[0].empty()) {
    model::Qwen35Reference prefix(*torso_, static_cast<std::uint32_t>(total));
    prefix.SetAdapter(lora);
    for (auto id : fit->state_ids)
      if (!prefix.Step(static_cast<std::int32_t>(id), {}, {}, error)) return std::nullopt;
    for (std::size_t q = 0; q < rqs.size(); ++q) {
      model::Qwen35Reference run = prefix;  // snapshot of the shared state prefix
      for (auto id : fit->question_ids[q]) {
        if (!run.Step(static_cast<std::int32_t>(id), h, {}, error)) return std::nullopt;
        hidden_by_q[q].push_back(h);
      }
    }
  }
  json::Value answers = json::Value::MakeObject();
  for (std::size_t q = 0; q < rqs.size(); ++q) {
    const auto& hidden = hidden_by_q[q];
    std::vector<std::vector<float>> options;
    for (auto idx : fit->option_index[q]) options.push_back(hidden.at(idx));
    const auto logits = readout->head.Logits(hidden.back(), options);
    const auto probs = SlotProbabilities(logits, readout->Temperature(rqs[q].kind));
    answers.set(r.questions[q].name, AnswerToJson(ReadAnswer(rqs[q], probs, readout->ordinal_smoothing)));
  }
  json::Value usage = json::Value::MakeObject();
  usage.set("input_tokens", json::Value::Int(static_cast<std::int64_t>(total)));
  usage.set("output_tokens", json::Value::Int(static_cast<std::int64_t>(rqs.size())));
  json::Value resp = json::Value::MakeObject();
  resp.set("model", json::Value::String(r.variant.empty() ? model_name : model_name + ":" + r.variant));
  resp.set("answers", std::move(answers));
  resp.set("usage", std::move(usage));
  return resp;
}

}  // namespace halo::decide
