#ifndef HALO_SERVER_ANTHROPIC_REQUEST_HPP_
#define HALO_SERVER_ANTHROPIC_REQUEST_HPP_

// Anthropic Messages API request bodies (POST /v1/messages, /v1/messages/count_tokens) normalized into the
// GenerationRequest every other API renders through the chat template (docs/ANTHROPIC.md has the coverage
// matrix). The wire format is the 2023-06-01 version as of the anthropic 1.12 SDK and Claude Code 2.1.293.
//
// Content blocks map onto template messages:
//   system (string or text blocks)    a leading system message; text blocks that are Claude Code's billing
//                                      attribution ("x-anthropic-billing-header: ...") are dropped: they are
//                                      metadata for Anthropic's billing, change every session, and would only
//                                      move the prompt's prefix
//   user text                          a user message
//   user tool_result                   a tool message (is_error prefixes the content with "[error] ")
//   assistant text / tool_use          content / a tool_calls entry of one assistant turn
//   assistant thinking                 reasoning_content: the exact text halo produced when the block's
//                                      signature is one of ours (EncodeThinkingSignature), else its text
//   assistant redacted_thinking        dropped (opaque to this model)
//   a mid-conversation "system" message  folded into the user turn before it (the template allows system
//                                      messages only at the start), or a user message of its own after tool
//                                      results
// Anything this server cannot honour is refused with a 400 naming the field, never silently ignored: images,
// documents and other non-text inputs, Anthropic-defined (server and client) tool types, structured output
// formats, assistant prefill, MCP servers and containers. Fields that only steer Anthropic's own service
// (metadata, service_tier, safeguards, context_management, inference_geo, diagnostics, cache_control) are
// accepted; the ones not acted on are listed in MessagesRequest::ignored (the service echoes them in the
// x-halo-ignored response header).

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"
#include "server/openai_request.hpp"

namespace halo::server {

struct MessagesRequest {
  GenerationRequest gen;  // api kMessages: template messages, tools, options, sampling, stop, stream
  /// thinking.display "omitted": thinking blocks carry no text (their signature still carries it, so the
  /// next turn renders the reasoning exactly as generated).
  bool omit_thinking = false;
  /// tool_choice any / tool: text appended to the rendered prompt that opens a tool call (thinking is off).
  std::string forced_prefix;
  bool disable_parallel_tool_use = false;
  /// cache_control breakpoints in the request (reported only; the session cache reuses every prefix).
  std::uint32_t cache_breakpoints = 0;
  /// Top-level fields accepted but not acted on (sorted).
  std::vector<std::string> ignored;
};

/// An Anthropic error: HTTP status and error type (invalid_request_error, authentication_error,
/// not_found_error, request_too_large, api_error, overloaded_error).
struct MessagesError {
  int status = 400;
  std::string type = "invalid_request_error";
  std::string message;
};

/// `count_tokens`: the count_tokens body (no max_tokens, stream or stop_sequences).
[[nodiscard]] std::optional<MessagesRequest> ParseMessagesRequest(const json::Value& body, const RequestDefaults& d,
                                                                  bool count_tokens, MessagesError* error);

/// {"type": "error", "error": {"type", "message"}}.
[[nodiscard]] json::Value MessagesErrorBody(const MessagesError& e);

/// The signature of a thinking block halo produced: the reasoning text itself, base64 behind a version tag, so
/// a client that sends the block back (as the API requires) gives the next turn the exact reasoning even when
/// the block's text was omitted. It is not a cryptographic signature: a client that edits it edits its own
/// prompt.
[[nodiscard]] std::string EncodeThinkingSignature(std::string_view reasoning);
/// The reasoning a signature of ours carries; nullopt for any other signature (Anthropic's).
[[nodiscard]] std::optional<std::string> DecodeThinkingSignature(std::string_view signature);

/// A model-name alias: `pattern` is a name, or a prefix ending in '*'; `target` the served model id (empty: the
/// server's model). The first match wins.
struct ModelAlias {
  std::string pattern;
  std::string target;
};
/// "PATTERN=TARGET" (TARGET may be empty); nullopt when malformed.
[[nodiscard]] std::optional<ModelAlias> ParseModelAliasSpec(std::string_view spec);
/// The served model for a requested name: the server's own id, or an alias's target; nullopt when nothing matches.
[[nodiscard]] std::optional<std::string> ResolveModel(std::string_view requested, const std::string& model_id,
                                                      const std::vector<ModelAlias>& aliases);

}  // namespace halo::server

#endif  // HALO_SERVER_ANTHROPIC_REQUEST_HPP_
