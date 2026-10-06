#include "server/qwen35_generator.hpp"

#include <algorithm>

#include "core/sampling.hpp"
#include "server/flash_next_generator.hpp"  // CompleteUtf8Prefix

namespace halo::server {

Qwen35Generator::Qwen35Generator(engine::Qwen35Gpu& engine, const text::Tokenizer& tokenizer,
                                 std::uint32_t max_context)
    : engine_(engine), tok_(tokenizer), max_context_(max_context) {
  for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
    const auto ids = tok_.Encode(s, true);
    if (ids.size() == 1) stops_.push_back(ids[0]);
  }
  if (tok_.eos() != text::kNoToken) stops_.push_back(tok_.eos());
}

bool Qwen35Generator::AddVariant(const model::LoadedVariant& v, std::string* error) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!v.lora) {
    if (error) *error = "variant " + v.manifest.name + " has no LoRA part (heads are for /v1/systemone)";
    return false;
  }
  if (!engine_.AddAdapter(v.manifest.name, *v.lora, error)) return false;
  variants_.push_back(v.manifest.name);
  return true;
}

std::vector<std::string> Qwen35Generator::Variants() const {
  std::lock_guard<std::mutex> lk(mu_);
  return variants_;
}

TextGenerator::Result Qwen35Generator::Generate(const Params& params,
                                                const std::function<bool(std::string_view)>& on_text) {
  std::lock_guard<std::mutex> lk(mu_);
  Result r;
  std::string error;
  if (!engine_.SelectAdapter(params.variant, &error)) return r.error = error, r;
  const auto encoded = tok_.Encode(params.prompt, true);
  const std::vector<std::int32_t> ids(encoded.begin(), encoded.end());
  r.prompt_tokens = ids.size();
  if (ids.empty()) return r.error = "empty prompt", r;
  if (ids.size() >= max_context_)
    return r.error = "prompt of " + std::to_string(ids.size()) + " tokens exceeds the context of " +
                     std::to_string(max_context_), r;
  const std::uint32_t budget =
      std::min<std::uint32_t>(params.max_tokens, max_context_ - static_cast<std::uint32_t>(ids.size()));
  if (!engine_.Reset(&error)) return r.error = error, r;
  if (ids.size() > 1 && !engine_.Prefill(std::span<const std::int32_t>(ids).first(ids.size() - 1), {}, &error))
    return r.error = error, r;
  sampling::SamplerState state(params.sampling, encoded);
  std::vector<float> logits(engine_.vocab_size());
  std::int32_t token = ids.back();
  std::string pending;
  bool cancelled = false;
  while (r.completion_tokens < budget) {
    if (!engine_.Step(token, {}, logits, &error)) return r.error = error, r;
    const auto next = state.Sample(logits);
    if (std::find(stops_.begin(), stops_.end(), next) != stops_.end()) break;
    state.Accept(next);
    ++r.completion_tokens;
    token = static_cast<std::int32_t>(next);
    pending += tok_.TokenBytes(next);
    const std::size_t n = CompleteUtf8Prefix(pending);
    if (n > 0) {
      cancelled = !on_text(std::string_view(pending).substr(0, n));
      pending.erase(0, n);
    }
    if (cancelled) break;
  }
  r.length_limited = !cancelled && r.completion_tokens == budget && budget > 0;
  if (!cancelled && !pending.empty()) (void)on_text(pending);
  return r;
}

}  // namespace halo::server
