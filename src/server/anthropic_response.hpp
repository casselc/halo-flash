#ifndef HALO_SERVER_ANTHROPIC_RESPONSE_HPP_
#define HALO_SERVER_ANTHROPIC_RESPONSE_HPP_

// One generation's output as an Anthropic Message, built from the streaming tool-call parser's events
// (text/tool_call_parser.hpp): the server-sent events as they become available, and the final object.
//
// Stream grammar (every stream this builder produces; tests/server/anthropic_test.cpp checks it as a property):
//   message_start  (block_start  block_delta*  block_stop)*  message_delta  message_stop
// with ping events anywhere after message_start. Block indices count up from 0 without gaps; a block's deltas
// match its type (text_delta for text, thinking_delta then one signature_delta for thinking, input_json_delta
// for tool_use). An error event, when the service sends one, ends the stream instead of message_delta.
//
// Blocks: reasoning becomes a thinking block (its signature carries the text, anthropic_request.hpp), content a
// text block, a tool call a tool_use block whose input is the parsed arguments. Concatenating a stream's deltas
// gives the final object's blocks (input_json_delta parses to the input).
//
// Usage: input_tokens counts the prompt tokens neither read from nor written to the session cache, as the API
// does (input + cache_read + cache_creation = the whole prompt). message_start carries the whole prompt as
// input_tokens provisionally (the cache is consulted only when the generation runs); the final message_delta
// carries every count, cumulatively, and the SDKs overwrite with it.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "text/tool_call_parser.hpp"

namespace halo::server {

enum class StopReason : std::uint8_t { kEndTurn, kMaxTokens, kStopSequence, kToolUse, kContextWindow };

[[nodiscard]] const char* StopReasonName(StopReason r);
/// The stop reason of a generation: the token budget (max_tokens, or the context window when the budget was cut
/// below max_tokens) wins over a stop sequence, which wins over tool use; end_turn otherwise.
[[nodiscard]] StopReason ChooseStopReason(bool length_limited, bool context_limited, bool stop_sequence,
                                          bool tool_use);

struct MessagesUsage {
  std::uint64_t prompt_tokens = 0;  // the whole prompt
  std::uint64_t cache_read = 0;     // prompt tokens whose state came from the session cache
  std::uint64_t cache_write = 0;    // prompt tokens prefilled now that a retained state holds afterwards
  std::uint64_t output_tokens = 0;
};

class MessageBuilder {
 public:
  using IdSource = std::function<std::string()>;
  struct Options {
    std::string id;     // msg_...
    std::string model;  // the served model
    bool omit_thinking = false;   // thinking blocks carry no text (thinking.display omitted)
    bool tools_allowed = true;    // false (tool_choice none): a tool call the model writes is returned as text
    IdSource ids;                 // tool_use id suffixes
  };
  explicit MessageBuilder(Options options);

  /// message_start (streams; once). `prompt_tokens`: the provisional input_tokens.
  void Start(std::uint64_t prompt_tokens, std::vector<json::Value>* out);
  /// Consumes parser events; when streaming, appends the events they complete to `out`.
  void Add(const std::vector<text::StreamEvent>& events, std::vector<json::Value>* out);
  /// A ping event (streams, after message_start).
  void Ping(std::vector<json::Value>* out);
  /// Closes the open block and appends message_delta and message_stop.
  void Finish(StopReason reason, const std::string& stop_sequence, const MessagesUsage& usage,
              std::vector<json::Value>* out);
  /// The client left before the message was delivered in full (the observer records it cancelled).
  void MarkCancelled() { cancelled_ = true; }
  [[nodiscard]] bool cancelled() const { return cancelled_; }

  /// The non-streaming Message (after Finish).
  [[nodiscard]] json::Value Final() const;
  [[nodiscard]] std::size_t tool_uses() const { return calls_.size(); }
  /// (tool_use id, name) of each call, in order.
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> tool_calls() const { return calls_; }

 private:
  enum class BlockKind : std::uint8_t { kText, kThinking, kToolUse };
  struct Block {
    BlockKind kind;
    std::string text;  // text, thinking, or the tool input's compact JSON
    std::string id, name;
    json::Value input;
  };
  void Text(BlockKind kind, std::string_view text, std::vector<json::Value>* out);
  void Close(std::vector<json::Value>* out);
  void ToolUse(const text::ParsedToolCall& call, std::vector<json::Value>* out);
  json::Value BlockJson(const Block& b, bool final) const;
  json::Value UsageJson() const;
  json::Value Delta(std::size_t index, json::Value delta) const;

  Options opt_;
  bool started_ = false, finished_ = false, open_ = false, cancelled_ = false;
  std::vector<Block> blocks_;
  std::vector<std::pair<std::string, std::string>> calls_;
  StopReason reason_ = StopReason::kEndTurn;
  std::string stop_sequence_;
  MessagesUsage usage_;
};

/// The SSE frame of one event: "event: <type>\ndata: <json>\n\n".
[[nodiscard]] std::string MessagesFrame(const json::Value& event);

}  // namespace halo::server

#endif  // HALO_SERVER_ANTHROPIC_RESPONSE_HPP_
