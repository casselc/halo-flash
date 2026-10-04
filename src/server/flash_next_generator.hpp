#ifndef HALO_SERVER_FLASH_NEXT_GENERATOR_HPP_
#define HALO_SERVER_FLASH_NEXT_GENERATOR_HPP_

// The OpenAI service's TextGenerator on the Flash-Next GPU engine: tokenize
// the rendered prompt, prefill, decode with device sampling (penalties and
// RNG carried by a core/sampling SamplerState), and detokenize into whole
// UTF-8 characters. Generation ends at <|im_end|>, <|endoftext|> or the
// tokenizer's EOS (none of which is emitted), or at max_tokens.
//
// One sequence, one request at a time (P6 adds sessions and batching). The
// engine's state is reused across requests when the new prompt extends
// everything already run (a multi-turn conversation whose history renders
// as it was generated); otherwise it is reset. Reuse is exact: the engine
// state after running a token sequence does not depend on how it got there.

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "engine/flash_next_gpu.hpp"
#include "server/openai_service.hpp"
#include "text/tokenizer.hpp"

namespace halo::server {

class FlashNextGenerator : public TextGenerator {
 public:
  /// `engine` and `tokenizer` must outlive the generator.
  FlashNextGenerator(engine::FlashNextGpu& engine, const text::Tokenizer& tokenizer, std::uint32_t max_context);

  Result Generate(const Params& params, const std::function<bool(std::string_view)>& on_text) override;

  /// Tokens currently in the engine (prompt and generated, as run).
  [[nodiscard]] std::size_t cached_tokens() const { return history_.size(); }

 private:
  bool IsStop(text::TokenId t) const;

  engine::FlashNextGpu& engine_;
  const text::Tokenizer& tok_;
  const std::uint32_t max_context_;
  std::vector<text::TokenId> stops_;
  std::mutex mu_;                       // one generation at a time
  std::vector<std::int32_t> history_;  // tokens the engine has run, in order
};

/// The longest prefix of `bytes` made of whole UTF-8 sequences (a trailing
/// incomplete sequence is held back; invalid bytes pass through).
[[nodiscard]] std::size_t CompleteUtf8Prefix(std::string_view bytes);

}  // namespace halo::server

#endif  // HALO_SERVER_FLASH_NEXT_GENERATOR_HPP_
