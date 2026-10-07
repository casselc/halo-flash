#ifndef HALO_SERVER_OBSERVER_HPP_
#define HALO_SERVER_OBSERVER_HPP_

// What the OpenAI service reports about each generation, for observability
// (halo-scope's GENERATION observations, P12) without the server depending
// on the store. A harness joins its own steps to these through the trace and
// session ids it sends (X-Halo-Trace-Id, X-Halo-Session-Id) and the
// tool-call ids in the response.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/sampling.hpp"

namespace halo::server {

/// MTP speculative decoding of one generation (docs/MTP.md; all zero when it did not run).
struct SpecDecodeStats {
  static constexpr std::size_t kDepths = 8;  ///< drafts per cycle 0 .. 7 (engine::FlashNextGpu::kMaxDrafts)
  std::uint64_t cycles = 0;                  ///< decode cycles that drafted (verify passes)
  std::uint64_t drafted = 0, accepted = 0;   ///< draft tokens proposed / accepted
  std::array<std::uint64_t, kDepths> depth{};            ///< decode steps by drafts run (0: an ordinary step)
  std::array<std::uint64_t, kDepths - 1> accepted_at{};  ///< accepted drafts by position (index 0: the first)
  std::uint64_t catchup_tokens = 0;  ///< prompt positions the draft block caught up on (its history window)
  std::uint64_t capped = 0;          ///< cycles whose drafts the max_tokens cap lowered below the chooser's depth
  std::uint64_t ran_past_end = 0;    ///< 0 or 1: the generation ended inside a cycle's accepted run (a stop, a
                                     ///< cancel or a refusal), so the engine ran drafts that were not emitted
  [[nodiscard]] bool active() const {
    std::uint64_t steps = 0;
    for (const auto d : depth) steps += d;
    return steps > 0;
  }
};

struct GenerationRecord {
  std::string api;         // "chat", "responses" or "completions"
  std::string request_id;  // the response / completion id
  std::string trace_id;    // X-Halo-Trace-Id, else empty
  std::string session_id;  // X-Halo-Session-Id, else empty
  std::string prompt_name;  // X-Halo-Prompt (name@version), else empty
  std::uint32_t prompt_version = 0;
  std::string variant_id;   // the request's resolved P10 variant, else X-Halo-Variant-Id, else empty
  std::string user;
  std::string model;
  std::int64_t start_ns = 0, first_text_ns = 0, end_ns = 0;  // first_text_ns 0: no text
  std::uint64_t prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
  std::vector<std::uint32_t> gaps_us;  // between successive text pieces
  std::string finish_reason;           // stop, length, tool_calls, error
  sampling::SamplingConfig sampling;
  std::uint32_t max_tokens = 0;
  std::vector<std::string> tool_call_ids;
  std::vector<std::string> tool_names;
  std::string input;   // the rendered prompt
  std::string output;  // the generated text, as generated
  std::string error;
  SpecDecodeStats spec;
};

class GenerationObserver {
 public:
  virtual ~GenerationObserver() = default;
  /// Called once per generation, after the reply is complete, on the
  /// request's thread. Must not block for long or throw.
  virtual void OnGeneration(const GenerationRecord& record) noexcept = 0;
};

}  // namespace halo::server

#endif  // HALO_SERVER_OBSERVER_HPP_
