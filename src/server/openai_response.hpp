#ifndef HALO_SERVER_OPENAI_RESPONSE_HPP_
#define HALO_SERVER_OPENAI_RESPONSE_HPP_

// One generation's output in an OpenAI API's shape, built from the streaming
// tool-call parser's events (text/tool_call_parser.hpp): the streamed events
// as they become available, and the final object.
//
//   Chat Completions: chat.completion.chunk deltas (role, reasoning_content,
//     content, tool_calls), then the finish chunk and, when asked, a usage
//     chunk; or one chat.completion.
//   Responses: typed events with sequence numbers (response.created, output
//     items added and done, output_text / reasoning_text / function-call
//     argument deltas and dones, response.completed or response.incomplete);
//     or one response object. A response whose client left before it was
//     delivered in full is marked cancelled (MarkCancelled): its object has
//     status "cancelled" (and Chat / Completions finish_reason "cancelled"). Items are reasoning (reasoning_text content),
//     message (output_text) and function_call.
//   Completions: text_completion chunks or object; the text is the raw
//     generation (no parsing).
//
// Invariants (tested): concatenating a stream's deltas gives the final
// object's fields; every Responses item added is done, in order; content
// bytes from a kError event (a malformed call) are kept as content.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "server/openai_request.hpp"
#include "text/tool_call_parser.hpp"

namespace halo::server {

enum class FinishReason : std::uint8_t { kStop, kLength };

struct Usage {
  std::uint64_t prompt_tokens = 0;
  std::uint64_t cached_tokens = 0;
  std::uint64_t completion_tokens = 0;
  std::uint64_t reasoning_tokens = 0;
};

class OutputBuilder {
 public:
  /// Returns a fresh random suffix for item ids (tests pass a counter).
  using IdSource = std::function<std::string()>;

  /// `request` must outlive the builder (its settings echo into Responses
  /// objects). `id` is the response or completion id.
  OutputBuilder(const GenerationRequest& request, std::string id, std::int64_t created_s, IdSource ids);

  /// Consumes parser events; when streaming, appends the API events they
  /// complete to `out` (each one SSE `data:` payload).
  void Add(const std::vector<text::StreamEvent>& events, std::vector<json::Value>* out);
  /// Raw text for the Completions API (no parser).
  void AddText(std::string_view text, std::vector<json::Value>* out);
  /// Ends the output: closes open items and appends the final events.
  void Finish(FinishReason reason, const Usage& usage, std::vector<json::Value>* out);

  /// The client left before the output was delivered in full: Final() reports it cancelled (Responses status
  /// "cancelled", incomplete_details null; Chat and Completions finish_reason "cancelled"), never completed.
  /// Events already produced are not taken back (they were not delivered).
  void MarkCancelled() { cancelled_ = true; }
  [[nodiscard]] bool cancelled() const { return cancelled_; }

  /// The non-streaming object (after Finish).
  [[nodiscard]] json::Value Final() const;

  [[nodiscard]] bool has_tool_calls() const { return !calls_.empty(); }
  /// (call id, function name) of each call, in order.
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> tool_calls() const {
    std::vector<std::pair<std::string, std::string>> out;
    for (const auto& c : calls_) out.emplace_back(c.call_id, c.name);
    return out;
  }

 private:
  enum class Open : std::uint8_t { kNone, kReasoning, kMessage };
  struct Call {
    std::string item_id, call_id, name, arguments;
    std::string ns;  // Responses: the namespace of a flattened namespace tool (name is then its member name)
  };
  struct Item {
    enum class Kind : std::uint8_t { kReasoning, kMessage, kFunctionCall } kind;
    std::string id, text;  // reasoning or message text
    std::size_t call = 0;  // index into calls_
  };

  json::Value Event(const char* type);  // Responses event with its sequence number
  json::Value ItemJson(const Item& item, bool done) const;
  json::Value ResponseJson(const char* status) const;
  json::Value ChatChunk(json::Value delta, const char* finish) const;
  void Text(Item::Kind kind, std::string_view text, std::vector<json::Value>* out);
  void CloseItem(std::vector<json::Value>* out);
  void ToolCall(const text::ParsedToolCall& call, std::vector<json::Value>* out);
  void StartStream(std::vector<json::Value>* out);

  const GenerationRequest& req_;
  const std::string id_;
  const std::int64_t created_;
  IdSource ids_;
  bool started_ = false;
  bool finished_ = false;
  std::uint64_t sequence_ = 0;
  std::vector<Item> items_;
  std::vector<Call> calls_;
  bool item_open_ = false;  // items_.back() is a reasoning or message item still open
  std::string reasoning_, content_;  // Chat and Completions accumulate
  FinishReason reason_ = FinishReason::kStop;
  bool cancelled_ = false;
  Usage usage_;
};

/// Compact JSON for tool-call arguments ({"a":1} style, as OpenAI sends it).
[[nodiscard]] std::string CompactJson(const json::Value& v);

}  // namespace halo::server

#endif  // HALO_SERVER_OPENAI_RESPONSE_HPP_
