#ifndef HALO_SERVER_FLASH_NEXT_GENERATOR_HPP_
#define HALO_SERVER_FLASH_NEXT_GENERATOR_HPP_

// The OpenAI service's TextGenerator on the Flash-Next GPU engine: tokenize
// the rendered prompt, prefill, decode with device sampling (penalties and
// RNG carried by a core/sampling SamplerState), and detokenize into whole
// UTF-8 characters. Generation ends at <|im_end|>, <|endoftext|> or the
// tokenizer's EOS (none of which is emitted), or at max_tokens.
//
// One request at a time. Prompt prefixes are reused across requests through
// a SessionCache (server/session_cache.hpp): retained sessions, checkpoints
// parked where each prompt's generation prompt starts, LRU eviction. In exact
// mode (the default) a request's results are byte-identical to a cold
// prefill of its prompt; fast mode also extends finished sessions (fewer
// tokens prefilled, equal only up to the prefill-vs-decode rounding).

#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "engine/flash_next_gpu.hpp"
#include "server/openai_service.hpp"
#include "server/session_cache.hpp"
#include "text/tokenizer.hpp"

namespace halo::server {

class FlashNextGenerator : public TextGenerator {
 public:
  /// `engine` and `tokenizer` must outlive the generator.
  FlashNextGenerator(engine::FlashNextGpu& engine, const text::Tokenizer& tokenizer, std::uint32_t max_context,
                     SessionCacheOptions cache = {});

  Result Generate(const Params& params, const std::function<bool(std::string_view)>& on_text) override;

  /// Token-level only (GenerateIds; Generate fails): no tokenizer, generation stops at `stops`.
  FlashNextGenerator(engine::FlashNextGpu& engine, std::vector<text::TokenId> stops, std::uint32_t max_context,
                     SessionCacheOptions cache = {});

  /// Tokens the last generation's session ran (prompt and generated).
  [[nodiscard]] std::size_t cached_tokens() const { return cache_.live_tokens(); }
  [[nodiscard]] const SessionCache& cache() const { return cache_; }

  /// The token-level generation (tests drive it without a tokenizer): `prompt` ids, the stable prefix length
  /// (see SessionCache::Prepare), sampling and limits from `params` (its prompt text unused); `on_token` gets each
  /// generated token (not the stop) and returns false to cancel. `params.cancelled` (optional) is polled while
  /// queued behind another generation, between prefill chunks and before each decode step: a request whose
  /// client left gives up within one chunk or step (Result::cancelled) and leaves the cache as a cold replay of
  /// what was retained would (a partly prefilled live session is dropped; a decode stop keeps what ran).
  Result GenerateIds(std::span<const std::int32_t> prompt, std::size_t stable, const Params& params,
                     const std::function<bool(text::TokenId)>& on_token);

 private:
  bool IsStop(text::TokenId t) const;

  engine::FlashNextGpu& engine_;
  const text::Tokenizer* tok_;  // null: token-level only
  const std::uint32_t max_context_;
  std::vector<text::TokenId> stops_;
  std::timed_mutex mu_;  // one generation at a time (timed: a queued request polls Params::cancelled)
  SessionCache cache_;
  text::TokenId im_start_ = text::kNoToken;
};

/// The longest prefix of `bytes` made of whole UTF-8 sequences (a trailing
/// incomplete sequence is held back; invalid bytes pass through).
[[nodiscard]] std::size_t CompleteUtf8Prefix(std::string_view bytes);

}  // namespace halo::server

#endif  // HALO_SERVER_FLASH_NEXT_GENERATOR_HPP_
