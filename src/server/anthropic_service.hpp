#ifndef HALO_SERVER_ANTHROPIC_SERVICE_HPP_
#define HALO_SERVER_ANTHROPIC_SERVICE_HPP_

// The Anthropic Messages API over a text generator (docs/ANTHROPIC.md):
//   POST /v1/messages               whole Message, or server-sent events with "stream": true
//   POST /v1/messages/count_tokens  {"input_tokens": n} for the rendered prompt
// A request is parsed (anthropic_request), its model name resolved through the alias table, rendered with the
// chat template, generated with stop sequences applied, parsed back into thinking, text and tool_use blocks
// (text/tool_call_parser) and returned in the API's shape (anthropic_response).
//
// Streams start with message_start as soon as the request is accepted; while the request is queued, prefilling
// or between tokens a ping event follows every sse_keepalive_ms without a write (so a client's idle timer does
// not fire, and a departed client fails the write). A client that leaves cancels its generation, as on the
// OpenAI routes: the observer records it "cancelled". Errors use the API's shape ({"type": "error", "error":
// {"type", "message"}}): before a stream starts as an HTTP status, after it as an error event that ends it.
//
// Usage reports the session cache honestly: cache_read_input_tokens are the prompt tokens whose state was
// reused, cache_creation_input_tokens the prompt tokens prefilled now that a retained state holds afterwards,
// input_tokens the rest. cache_control breakpoints are accepted but do not steer the cache, which reuses the
// longest retained prefix of every prompt by itself.

#include <memory>
#include <string>
#include <vector>

#include "server/anthropic_request.hpp"
#include "server/http.hpp"
#include "server/openai_service.hpp"

namespace halo::server {

class AnthropicService {
 public:
  /// `options`: the OpenAI service's (model id, defaults, template, observer, activity hook, keepalive, ids);
  /// `aliases`: model names accepted besides the model id (default: every "claude-*" name is the served model).
  AnthropicService(TextGenerator& generator, ServiceOptions options,
                   std::vector<ModelAlias> aliases = {{"claude-*", ""}});
  /// Registers the routes. The service must outlive the server's Stop() (streams run on its threads).
  void Register(HttpServer& server);

  HttpResponse Messages(const HttpRequest& req);
  HttpResponse CountTokens(const HttpRequest& req);

 private:
  struct Prepared;
  /// Parses, resolves the model and renders; on failure `error` holds the answer.
  std::shared_ptr<Prepared> Prepare(const HttpRequest& http, bool count_tokens, HttpResponse* error);

  TextGenerator& gen_;
  ServiceOptions opt_;
  std::vector<ModelAlias> aliases_;
};

/// An Anthropic error response.
[[nodiscard]] HttpResponse MessagesErrorResponse(const MessagesError& e);

}  // namespace halo::server

#endif  // HALO_SERVER_ANTHROPIC_SERVICE_HPP_
