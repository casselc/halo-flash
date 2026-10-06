#ifndef HALO_SERVER_OBSERVER_HPP_
#define HALO_SERVER_OBSERVER_HPP_

// What the OpenAI service reports about each generation, for observability
// (halo-scope's GENERATION observations, P12) without the server depending
// on the store. A harness joins its own steps to these through the trace and
// session ids it sends (X-Halo-Trace-Id, X-Halo-Session-Id) and the
// tool-call ids in the response.

#include <cstdint>
#include <string>
#include <vector>

#include "core/sampling.hpp"

namespace halo::server {

struct GenerationRecord {
  std::string api;         // "chat", "responses" or "completions"
  std::string request_id;  // the response / completion id
  std::string trace_id;    // X-Halo-Trace-Id, else empty
  std::string session_id;  // X-Halo-Session-Id, else empty
  std::string prompt_name;  // X-Halo-Prompt (name@version), else empty
  std::uint32_t prompt_version = 0;
  std::string variant_id;   // X-Halo-Variant-Id, else empty
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
