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
                                       std::uint32_t max_context, SessionCacheOptions cache)
    : engine_(engine), tok_(&tokenizer), max_context_(max_context), cache_(engine, cache) {
  for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
    const auto ids = tok_->Encode(s, true);
    if (ids.size() == 1) stops_.push_back(ids[0]);
  }
  if (tok_->eos() != text::kNoToken) stops_.push_back(tok_->eos());
  if (const auto ids = tok_->Encode("<|im_start|>", true); ids.size() == 1) im_start_ = ids[0];
}

FlashNextGenerator::FlashNextGenerator(engine::FlashNextGpu& engine, std::vector<text::TokenId> stops,
                                       std::uint32_t max_context, SessionCacheOptions cache)
    : engine_(engine), tok_(nullptr), max_context_(max_context), stops_(std::move(stops)), cache_(engine, cache) {}

bool FlashNextGenerator::IsStop(text::TokenId t) const {
  return std::find(stops_.begin(), stops_.end(), t) != stops_.end();
}

TextGenerator::Result FlashNextGenerator::Generate(const Params& params,
                                                   const std::function<bool(std::string_view)>& on_text) {
  Result r;
  if (!params.variant.empty()) return r.error = "Flash-Next variants are not supported yet (P11 stage 2)", r;
  if (!tok_) return r.error = "no tokenizer", r;
  const auto encoded = tok_->Encode(params.prompt, true);
  const std::vector<std::int32_t> ids(encoded.begin(), encoded.end());  // the engine's token type
  // The stable prefix: everything before the last <|im_start|> (the generation prompt), which the next turn of
  // the conversation renders again however it renders this turn's output.
  std::size_t stable = 0;
  for (std::size_t i = ids.size(); i-- > 0;)
    if (static_cast<text::TokenId>(ids[i]) == im_start_) {
      stable = i;
      break;
    }
  std::string pending;
  bool cancelled = false;
  r = GenerateIds(ids, stable, params, [&](text::TokenId next) {
    pending += tok_->TokenBytes(next);
    const std::size_t n = CompleteUtf8Prefix(pending);
    if (n > 0) {
      cancelled = !on_text(std::string_view(pending).substr(0, n));
      pending.erase(0, n);
    }
    return !cancelled;
  });
  if (r.error.empty() && !cancelled && !pending.empty()) (void)on_text(pending);
  return r;
}

TextGenerator::Result FlashNextGenerator::GenerateIds(std::span<const std::int32_t> ids, std::size_t stable,
                                                      const Params& params,
                                                      const std::function<bool(text::TokenId)>& on_token) {
  std::lock_guard<std::mutex> lk(mu_);
  Result r;
  r.prompt_tokens = ids.size();
  if (ids.empty()) return r.error = "empty prompt", r;
  if (ids.size() >= max_context_)
    return r.error = "prompt of " + std::to_string(ids.size()) + " tokens exceeds the context of " +
                     std::to_string(max_context_), r;
  const std::uint32_t budget =
      std::min<std::uint32_t>(params.max_tokens, max_context_ - static_cast<std::uint32_t>(ids.size()));
  std::string error;
  // A prompt the memory left cannot hold is refused before its prefill (minutes at long context), not midway.
  SessionCache::Prepared prep;
  if (!cache_.Prepare(ids, stable, static_cast<std::uint32_t>(ids.size()) + budget, &prep, &error))
    return r.error = error, r;
  r.cached_tokens = prep.reused;

  std::vector<text::TokenId> as_tok(ids.begin(), ids.end());
  sampling::SamplerState state(params.sampling, as_tok);
  std::int32_t token = ids.back();
  bool cancelled = false;
  while (r.completion_tokens < budget) {
    engine::FlashNextGpu::SampleRequest req{&params.sampling, state.penalties(), state.rng_state()};
    engine::FlashNextGpu::SampleResult res;
    if (!engine_.StepSample(token, req, &res, &error)) {
      cache_.Abandon();
      return r.error = error, r;
    }
    cache_.Ran(token);
    state.SetRngState(res.rng);
    const auto next = static_cast<text::TokenId>(res.token);
    if (IsStop(next)) break;
    state.Accept(next);
    ++r.completion_tokens;
    token = res.token;
    if (!on_token(next)) {
      cancelled = true;
      break;
    }
  }
  r.length_limited = !cancelled && r.completion_tokens == budget && budget > 0;
  // A sampled token that was neither a stop nor run is not in the session; its state is exactly what it ran.
  cache_.Finish();
  return r;
}

}  // namespace halo::server
