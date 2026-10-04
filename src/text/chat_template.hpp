#ifndef HALO_TEXT_CHAT_TEMPLATE_HPP_
#define HALO_TEXT_CHAT_TEMPLATE_HPP_

// Qwen chat templates, compiled by hand from the models' Jinja sources and
// byte-exact against HF transformers' apply_chat_template (fixtures from
// tools/gen_chat_fixtures.py). Adapted in part from gufo's
// src/models/qwen/chat_template.cpp (MIT, see NOTICE).
//
// Messages and tools are JSON values shaped like the OpenAI/HF chat dicts:
//   {"role": "system"|"developer"|"user"|"assistant"|"tool",
//    "content": string | null | [{"type": "text", "text": ...} |
//                                {"type": "image"} | {"type": "video"}, ...],
//    "reasoning_content": string,              (assistant)
//    "tool_calls": [{"function": {"name": ..., "arguments": {...}}}, ...]}
// An absent key is Jinja-undefined, which differs from null in places.

#include <optional>
#include <string>
#include <string_view>

#include "core/json.hpp"

namespace halo::text {

enum class ChatTemplateKind {
  /// Qwen3.8-Flash-Next as shipped in the Unsloth GGUF (`tokenizer.chat_template`,
  /// sha256 12827f24...): merged leading system/developer messages, reasoning
  /// effort instructions, preserve_thinking, thinking on by default.
  kFlashNext,
  /// Qwen3.5 (Qwen3.5-2B-Base tokenizer_config.json, also the strands-decider
  /// checkpoint; sha256 273d8e0e...): one system message, thinking off by
  /// default, reasoning shown only after the last user query.
  kQwen35,
};

/// The template variables the Jinja sources read. nullopt is "undefined".
struct ChatTemplateOptions {
  bool add_generation_prompt = false;
  std::optional<bool> enable_thinking;
  std::optional<std::string> reasoning_effort;  // Flash-Next: low | medium | high | xhigh
  std::optional<bool> preserve_thinking;        // Flash-Next only
  bool add_vision_id = false;
};

/// The template whose exact source text this is (by SHA-256), if known.
[[nodiscard]] std::optional<ChatTemplateKind> IdentifyChatTemplate(std::string_view jinja_source);

/// Hex SHA-256 of the template source (pinning and diagnostics).
[[nodiscard]] std::string ChatTemplateSha256(std::string_view jinja_source);

/// Reads chat_template_kwargs-style options from a JSON object:
/// add_generation_prompt, enable_thinking, reasoning_effort, preserve_thinking,
/// add_vision_id. Unknown keys are ignored (the templates ignore them too);
/// a known key of the wrong type is an error rather than Jinja's loose truthiness.
[[nodiscard]] bool ChatOptionsFromJson(const json::Value& kwargs, ChatTemplateOptions* options,
                                       std::string* error);

/// Renders `messages` (a JSON array) and optional `tools` (null or an array
/// of objects) exactly as the template does. Returns nullopt with `error` set
/// wherever the template raises (raise_exception or a Jinja type error).
///
/// One deliberate deviation: a content part whose "text" is a list or object
/// is an error here, where Jinja would print its Python repr().
[[nodiscard]] std::optional<std::string> RenderChat(ChatTemplateKind kind, const json::Value& messages,
                                                    const json::Value* tools,
                                                    const ChatTemplateOptions& options,
                                                    std::string* error);

/// The text appended for add_generation_prompt (the assistant header and the
/// opening of the thinking block). A completion that follows it starts inside
/// <think> when this ends with "<think>\n", and in content otherwise.
[[nodiscard]] std::string_view GenerationPrompt(ChatTemplateKind kind, const ChatTemplateOptions& options);

/// Python str.strip() with no arguments (what Jinja's |trim does): removes
/// leading and trailing Unicode whitespace in Python's sense (White_Space plus
/// U+001C..U+001F). Ill-formed bytes are never whitespace.
[[nodiscard]] std::string_view PyStrip(std::string_view s);

}  // namespace halo::text

#endif  // HALO_TEXT_CHAT_TEMPLATE_HPP_
