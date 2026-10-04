#ifndef HALO_SERVER_OPENAI_SERVICE_HPP_
#define HALO_SERVER_OPENAI_SERVICE_HPP_

// The OpenAI endpoints over a text generator:
//   GET  /v1/models, /health
//   POST /v1/completions, /v1/chat/completions, /v1/responses
//   GET  /v1/responses/{id}   (stored responses)
// A request is parsed (openai_request), rendered with the chat template,
// generated with stop sequences applied, parsed back into reasoning,
// content and tool calls (text/tool_call_parser) and returned in the API's
// shape (openai_response), whole or as server-sent events.
//
// Responses with store (the default) are kept with their input, so
// previous_response_id continues a conversation by prefixing the stored
// turns. The store is in memory for now (P7 moves it to the halo-scope
// SQLite store and binds it to live engine sessions).

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"
#include "core/sampling.hpp"
#include "server/http.hpp"
#include "server/openai_request.hpp"
#include "text/chat_template.hpp"

namespace halo::server {

/// What the service needs from an engine.
class TextGenerator {
 public:
  struct Params {
    std::string prompt;  // fully rendered text (special tokens as text)
    sampling::SamplingConfig sampling;
    std::uint32_t max_tokens = 0;
  };
  struct Result {
    bool length_limited = false;  // stopped at max_tokens
    std::uint64_t prompt_tokens = 0;
    std::uint64_t cached_tokens = 0;
    std::uint64_t completion_tokens = 0;
    std::string error;  // non-empty: generation failed
  };
  virtual ~TextGenerator() = default;
  /// Generates, calling `on_text` with each new piece of text (whole UTF-8
  /// characters); stops early when it returns false.
  virtual Result Generate(const Params& params, const std::function<bool(std::string_view)>& on_text) = 0;
};

/// Holds back text that may be the start of a stop sequence; reports when
/// one is completed (the stop sequence itself is never emitted). Output ends
/// where the first stop sequence completes, as if text arrived a byte at a
/// time (the longest one completing at that byte wins), independent of
/// chunking.
class StopMatcher {
 public:
  explicit StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {}
  /// Appends `text`; returns what is now safe to emit. Once stopped() is
  /// true, the output ends before the stop sequence and further input is
  /// ignored.
  std::string Push(std::string_view text);
  /// End of input: whatever is held back.
  std::string Flush();
  [[nodiscard]] bool stopped() const { return stopped_; }

 private:
  std::vector<std::string> stops_;
  std::string pending_;
  bool stopped_ = false;
};

struct ServiceOptions {
  std::string model_id = "qwen3.8-flash-next";
  RequestDefaults defaults;
  text::ChatTemplateKind template_kind = text::ChatTemplateKind::kFlashNext;
  std::size_t max_stored_responses = 1000;  // oldest evicted first
  /// Ids and timestamps; tests pass deterministic ones.
  std::function<std::string()> new_id;
  std::function<std::int64_t()> now_s;
};

class OpenAiService {
 public:
  OpenAiService(TextGenerator& generator, ServiceOptions options);
  /// Registers the routes on `server`.
  void Register(HttpServer& server);

  HttpResponse Models(const HttpRequest& req);
  HttpResponse Completions(const HttpRequest& req);
  HttpResponse Chat(const HttpRequest& req);
  HttpResponse Responses(const HttpRequest& req);
  HttpResponse GetResponse(const std::string& id);

 private:
  struct Stored {
    json::Value messages;  // the conversation including this response's turn
    json::Value response;  // the response object
  };
  HttpResponse Generate(Api api, GenerationRequest req, const json::Value* prior);
  void Store(const std::string& id, Stored s);

  TextGenerator& gen_;
  ServiceOptions opt_;
  std::mutex mu_;
  std::map<std::string, Stored> stored_;  // guarded by mu_
  std::vector<std::string> stored_order_;  // guarded by mu_
};

}  // namespace halo::server

#endif  // HALO_SERVER_OPENAI_SERVICE_HPP_
