#include "server/flash_next_generator.hpp"

#include <algorithm>

#include "core/sampling.hpp"

namespace halo::server {

std::size_t CompleteUtf8Prefix(std::string_view bytes) {
  // Look back at most 3 bytes for the lead byte of a sequence that the
  // buffer cuts short.
  const std::size_t n = bytes.size();
  for (std::size_t back = 1; back <= std::min<std::size_t>(3, n); ++back) {
    const auto c = static_cast<unsigned char>(bytes[n - back]);
    if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking
    const std::size_t need = c >= 0xF0 && c <= 0xF4 ? 4 : c >= 0xE0 && c < 0xF0 ? 3 : c >= 0xC2 && c < 0xE0 ? 2 : 1;
    return need > back ? n - back : n;  // ASCII or invalid lead bytes pass through
  }
  return n;
}

FlashNextGenerator::FlashNextGenerator(engine::FlashNextGpu& engine, const text::Tokenizer& tokenizer,
                                       std::uint32_t max_context)
    : engine_(engine), tok_(tokenizer), max_context_(max_context) {
  for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
    const auto ids = tok_.Encode(s, true);
    if (ids.size() == 1) stops_.push_back(ids[0]);
  }
  if (tok_.eos() != text::kNoToken) stops_.push_back(tok_.eos());
}

bool FlashNextGenerator::IsStop(text::TokenId t) const {
  return std::find(stops_.begin(), stops_.end(), t) != stops_.end();
}

TextGenerator::Result FlashNextGenerator::Generate(const Params& params,
                                                   const std::function<bool(std::string_view)>& on_text) {
  std::lock_guard<std::mutex> lk(mu_);
  Result r;
  const auto encoded = tok_.Encode(params.prompt, true);
  const std::vector<std::int32_t> ids(encoded.begin(), encoded.end());  // the engine's token type
  r.prompt_tokens = ids.size();
  if (ids.empty()) return r.error = "empty prompt", r;
  if (ids.size() >= max_context_)
    return r.error = "prompt of " + std::to_string(ids.size()) + " tokens exceeds the context of " +
                     std::to_string(max_context_), r;
  const std::uint32_t budget =
      std::min<std::uint32_t>(params.max_tokens, max_context_ - static_cast<std::uint32_t>(ids.size()));
  std::string error;
  // Reuse the engine's state when everything it has run is a strict prefix
  // of the prompt (the last prompt token is always run with sampling).
  std::size_t reuse = 0;
  if (engine_.Position() == history_.size() && history_.size() < ids.size() &&
      std::equal(history_.begin(), history_.end(), ids.begin())) {
    reuse = history_.size();
  } else {
    history_.clear();
    if (!engine_.Reset(&error)) return r.error = error, r;
  }
  r.cached_tokens = reuse;
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  if (ids.size() - 1 > reuse) {
    if (!engine_.Prefill(prompt.subspan(reuse, ids.size() - 1 - reuse), {}, &error)) {
      history_.clear();
      (void)engine_.Reset(&error);
      return r.error = error, r;
    }
    history_.insert(history_.end(), ids.begin() + static_cast<std::ptrdiff_t>(reuse), ids.end() - 1);
  }

  sampling::SamplerState state(params.sampling, encoded);
  std::int32_t token = ids.back();
  std::string pending;
  bool cancelled = false;
  while (r.completion_tokens < budget) {
    engine::FlashNextGpu::SampleRequest req{&params.sampling, state.penalties(), state.rng_state()};
    engine::FlashNextGpu::SampleResult res;
    if (!engine_.StepSample(token, req, &res, &error)) {
      r.error = error;
      history_.clear();
      (void)engine_.Reset(&error);
      return r;
    }
    history_.push_back(token);
    state.SetRngState(res.rng);
    const auto next = static_cast<text::TokenId>(res.token);
    if (IsStop(next)) break;
    state.Accept(next);
    ++r.completion_tokens;
    token = res.token;
    pending += tok_.TokenBytes(next);
    const std::size_t n = CompleteUtf8Prefix(pending);
    if (n > 0) {
      cancelled = !on_text(std::string_view(pending).substr(0, n));
      pending.erase(0, n);
    }
    if (cancelled) break;
  }
  r.length_limited = !cancelled && r.completion_tokens == budget && budget > 0;
  // A sampled token that was neither a stop nor run is not in history_; the
  // engine's state is exactly history_.
  if (!cancelled && !pending.empty()) (void)on_text(pending);
  return r;
}

}  // namespace halo::server
