#ifndef HALO_TEXT_TOOL_CALL_PARSER_HPP_
#define HALO_TEXT_TOOL_CALL_PARSER_HPP_

// Streaming parser for Qwen assistant output: reasoning, content and tool
// calls, fed generated text in arbitrary chunks (a token at a time or more).
//
// Recognized markup (what the Qwen3.5/3.8 chat templates render):
//   <think>               opening of reasoning (only at the start, kDetect)
//   </think>              end of reasoning
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...
//   </function>\n</tool_call>            (the XML form these templates use)
//   <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>  (JSON form)
//
// Every input byte lands in exactly one event, in order: concatenating the
// `text` of all events reproduces the input. Content and reasoning events
// carry their bytes verbatim; markup and the whitespace the templates put
// around it (Jinja |trim removes it from content) are kFraming; a tool call's
// bytes are in its kToolCall event; anything that starts like a call but is
// not one is a kError event holding the raw bytes, so nothing is lost.
//
// Decisions are made only when the bytes seen so far determine them, so the
// events do not depend on how the text was chunked (adjacent events of the
// same text kind may split differently). A partial tag is held back, never
// emitted as content, until it completes or is ruled out; at Finish a
// dangling prefix such as "<tool_ca" is content. Content and reasoning deltas
// also hold back an incomplete trailing UTF-8 sequence.
//
// Adapted in part from gufo's Qwen output parsing in
// src/cli/serve/openai_chat.cpp (MIT, see NOTICE).

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"

namespace halo::text {

struct ParsedToolCall {
  std::string name;
  json::Value arguments = json::Value::MakeObject();  // always an object

  friend bool operator==(const ParsedToolCall&, const ParsedToolCall&) = default;
};

enum class StreamEventKind : std::uint8_t {
  kContent,    // user-visible answer text
  kReasoning,  // thinking text
  kToolCall,   // one complete call; `text` holds its raw bytes
  kFraming,    // template markup and separator whitespace, not shown
  kError,      // `text` started a tool call but is not one; see `error`
};

struct StreamEvent {
  StreamEventKind kind = StreamEventKind::kContent;
  std::string text;      // the exact input bytes this event covers
  ParsedToolCall call;   // kToolCall
  std::string error;     // kError

  friend bool operator==(const StreamEvent&, const StreamEvent&) = default;
};

/// Where the generated text begins. A Flash-Next generation prompt with
/// thinking on ends in "<think>\n" (kReasoning); with thinking off it ends
/// after "</think>\n\n" (kContent). kDetect accepts an optional leading
/// "<think>" (after whitespace) and is content otherwise.
enum class StreamStart : std::uint8_t { kReasoning, kContent, kDetect };

struct ToolCallParserOptions {
  StreamStart start = StreamStart::kReasoning;
  /// Qwen models sometimes open a call before closing </think>; when true a
  /// <tool_call> inside reasoning ends the reasoning and is parsed as a call.
  bool tool_calls_in_reasoning = true;
  /// Tool definitions (the request's `tools` array, OpenAI shape). Parameter
  /// values carry no type marker in the XML form: the template prints strings
  /// raw and everything else as JSON. With a schema, a property whose type is
  /// only "string" stays a string and other types are parsed as JSON; without
  /// one, a value that parses as a non-string JSON value is that value. With a
  /// boolean/null schema, Python's True/False/None (how the Qwen3.5 template
  /// prints them) are accepted.
  json::Value tools;
  /// A call (or malformed call) is buffered whole; past this many bytes it is
  /// cut into an error event so memory stays bounded.
  std::size_t max_tool_call_bytes = 1u << 20;
};

class ToolCallStreamParser {
 public:
  explicit ToolCallStreamParser(ToolCallParserOptions options = {});

  /// Feeds the next chunk of generated text; appends the events it completes.
  void Push(std::string_view chunk, std::vector<StreamEvent>* events);
  /// End of generation: flushes everything held back. The parser is then
  /// back in its initial state, ready for a new stream.
  void Finish(std::vector<StreamEvent>* events);

  /// Bytes received but not yet in any event (held partial tags, whitespace,
  /// UTF-8 tails or a call still being buffered).
  [[nodiscard]] std::size_t pending_bytes() const noexcept { return pending_.size(); }

 private:
  enum class State : std::uint8_t { kDetect, kReasoning, kContent, kCall, kBadCall };
  using Schemas = std::map<std::string, std::map<std::string, std::vector<std::string>, std::less<>>, std::less<>>;

  void Run(bool final, std::vector<StreamEvent>* events);
  bool StepDetect(bool final, std::vector<StreamEvent>* events);
  bool StepText(bool final, std::vector<StreamEvent>* events);
  bool StepCall(bool final, std::vector<StreamEvent>* events);
  bool StepBadCall(bool final, std::vector<StreamEvent>* events);
  void Emit(std::vector<StreamEvent>* events, StreamEventKind kind, std::size_t n);
  void EnterContent();

  ToolCallParserOptions options_;
  Schemas schemas_;
  State state_;
  bool segment_start_ = true;  // leading whitespace of a text segment is framing
  std::string pending_;
  std::string bad_reason_;
};

/// Parses complete assistant text in one go (Push + Finish).
[[nodiscard]] std::vector<StreamEvent> ParseAssistantText(std::string_view text,
                                                          const ToolCallParserOptions& options = {});

}  // namespace halo::text

#endif  // HALO_TEXT_TOOL_CALL_PARSER_HPP_
