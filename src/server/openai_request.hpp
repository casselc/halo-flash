#ifndef HALO_SERVER_OPENAI_REQUEST_HPP_
#define HALO_SERVER_OPENAI_REQUEST_HPP_

// OpenAI request bodies (/v1/chat/completions, /v1/responses,
// /v1/completions) normalized into one GenerationRequest: chat-template
// messages and tools (text/chat_template.hpp's JSON shapes), template
// options, sampling settings and limits. Every API renders into the same
// chat template; this is the only place that knows their wire formats.
//
// Responses input items map onto template messages:
//   instructions                 a leading system message
//   message (user/system/developer/assistant), or the {role, content}
//                                shorthand; input_text/output_text parts
//   reasoning                    reasoning_content of the assistant turn
//   function_call                a tool_calls entry of the assistant turn,
//                                arguments parsed from their JSON string
//   function_call_output         a tool message
// Consecutive assistant-side items (reasoning, assistant message, function
// calls) form one assistant turn, as the model produced them.
//
// Unsupported features fail with an OpenAI-style error naming the parameter
// (n > 1, images, hosted tools, item references), never silently ignored.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"
#include "core/sampling.hpp"
#include "text/chat_template.hpp"

namespace halo::server {

enum class Api : std::uint8_t { kCompletions, kChat, kResponses, kMessages };  // kMessages: Anthropic (anthropic_request.hpp)

struct ToolChoice {
  enum class Mode : std::uint8_t { kAuto, kNone, kRequired, kFunction } mode = Mode::kAuto;
  std::string function;  // kFunction
};

struct ResponseFormat {
  enum class Kind : std::uint8_t { kText, kJsonObject, kJsonSchema } kind = Kind::kText;
  std::string name;
  json::Value schema;  // kJsonSchema
  bool strict = false;
};

/// What the server applies when a request leaves a setting out (the model's
/// recommended sampling, not OpenAI's temperature 1).
struct RequestDefaults {
  sampling::SamplingConfig sampling{.temperature = 0.6F, .top_k = 20, .top_p = 0.95F};
  std::uint32_t max_tokens = 4096;
  std::uint32_t max_tokens_limit = 262144;
};

struct GenerationRequest {
  Api api = Api::kChat;
  std::string model;  // as sent (echoed in responses), including a ":variant" alias
  /// P10 variant: the "variant" field, or the suffix of a "base:variant"
  /// model alias (both given must agree); "" = the base model.
  std::string variant;
  // kChat and kResponses: template input. kCompletions: `prompt`.
  json::Value messages = json::Value::MakeArray();
  json::Value tools;  // null or an array of {"type": "function", "function": {...}}
  /// Responses namespace tools ({"type": "namespace", "name", "tools": [functions]}, Codex) are flattened into
  /// function tools named "<namespace>.<function>" (OpenAI's own rendering of a namespaced call); a call the
  /// model makes to one goes back as a function_call item with "name": <function>, "namespace": <namespace>, the
  /// form Codex routes by (codex-rs protocol ResponseItem::FunctionCall).
  struct NamespacedTool {
    std::string flat, ns, name;
  };
  std::vector<NamespacedTool> namespaced_tools;
  /// The namespaced tool a model-emitted function name refers to: its flat name, or a bare member name that no
  /// plain tool has and exactly one namespace defines. Null for a plain function.
  [[nodiscard]] const NamespacedTool* FindNamespaced(std::string_view emitted) const;
  text::ChatTemplateOptions template_options;
  std::string prompt;

  sampling::SamplingConfig sampling;
  std::uint32_t max_tokens = 0;
  std::vector<std::string> stop;
  bool stream = false;
  bool include_usage = false;  // chat stream_options.include_usage
  bool logprobs = false;
  std::uint32_t top_logprobs = 0;

  ToolChoice tool_choice;
  bool parallel_tool_calls = true;
  ResponseFormat format;

  // Responses state. `has_instructions`: messages[0] is the instructions'
  // system message (instructions do not carry over to continuations).
  bool has_instructions = false;
  bool store = true;
  std::string previous_response_id;
  bool truncation_auto = false;
  json::Value metadata;  // null or an object of strings
  std::string user;
  // Observability context (set by the service from X-Halo-* headers).
  std::string trace_id, session_id;
  std::string prompt_name;      // X-Halo-Prompt: name@version (a managed prompt)
  std::uint32_t prompt_version = 0;
  std::string variant_id;       // X-Halo-Variant-Id
};

struct RequestError {
  std::string message;
  std::string param;  // dotted path of the offending field, may be empty
};

[[nodiscard]] std::optional<GenerationRequest> ParseChatRequest(const json::Value& body, const RequestDefaults& d,
                                                                RequestError* error);
[[nodiscard]] std::optional<GenerationRequest> ParseResponsesRequest(const json::Value& body,
                                                                     const RequestDefaults& d, RequestError* error);
[[nodiscard]] std::optional<GenerationRequest> ParseCompletionsRequest(const json::Value& body,
                                                                       const RequestDefaults& d,
                                                                       RequestError* error);

/// The OpenAI error body: {"error": {"message", "type": "invalid_request_error", "param", "code": null}}.
[[nodiscard]] json::Value ErrorBody(const RequestError& e);

}  // namespace halo::server

#endif  // HALO_SERVER_OPENAI_REQUEST_HPP_
