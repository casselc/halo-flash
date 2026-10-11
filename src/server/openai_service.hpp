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
// turns. A response whose client left before it was delivered in full is
// cancelled: the observer records finish_reason "cancelled", and the store
// keeps only the cancelled object (status "cancelled", for GET), never its
// turn, so previous_response_id cannot continue from it (400). The store is in memory for now (P7 moves it to the halo-scope
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
#include "server/observer.hpp"
#include "server/response_store.hpp"
#include "server/openai_request.hpp"
#include "text/chat_template.hpp"

namespace halo::server {

/// What the service needs from an engine.
class TextGenerator {
 public:
  struct Params {
    std::string prompt;  // fully rendered text (special tokens as text)
    std::string variant;  // P10 variant ("" = base); the service only passes ones `variants` lists
    sampling::SamplingConfig sampling;
    std::uint32_t max_tokens = 0;
    /// Optional: true once the requester is gone (its client left, or the server is stopping). The generator
    /// polls it while queued, between prefill chunks and between decode steps, and gives up (Result::cancelled)
    /// rather than compute for nobody; a request abandoned so leaves no state that differs from a cold replay.
    std::function<bool()> cancelled;
    /// G3 scheduling (docs/SCHEDULER.md; request headers, set by yard-relay / the broker, never by an untrusted
    /// client): the sandbox principal (X-Halo-Principal), the chain root it serves (X-Halo-Chain-Id; empty: the
    /// principal), and the idempotency key (Idempotency-Key, or the service's default digest; empty: none).
    std::string principal, chain_root, idempotency_key;
    /// What Admit() returned for this request (opaque to the service).
    std::shared_ptr<void> ticket;
  };
  /// G3: a request is admitted before any response is sent (a streamed 429 is impossible once the event stream
  /// started). Refused: answered 429 with Retry-After, and it never runs. Default: always admitted.
  struct Admission {
    bool refused = false;
    std::int64_t retry_after_s = 0;
    std::string why;
    std::shared_ptr<void> ticket;  ///< passed back in Params::ticket
  };
  virtual Admission Admit(const Params& params) {
    (void)params;
    return {};
  }
  struct Result {
    bool length_limited = false;  // stopped at max_tokens
    bool cancelled = false;       // stopped because Params::cancelled said so (not an error)
    std::uint64_t prompt_tokens = 0;
    std::uint64_t cached_tokens = 0;
    std::uint64_t completion_tokens = 0;
    std::string error;  // non-empty: generation failed
    SpecDecodeStats spec;  // MTP speculative decoding (zero when off)
    /// G3: refused before any work (429, never ran and never will): `retry_after_s` and why ("limit", "memory").
    bool refused = false;
    std::int64_t retry_after_s = 0;
    std::string refused_why;
    bool replayed = false;  // the stored answer of an earlier attempt with the same idempotency key
    /// G3: this reader left a request that runs on for its idempotency key (a retry adopted it, or it is within its
    /// keyed grace): not an end of the generation.
    bool detached = false;
    /// G3: another reader of the same generation counts its completion tokens (adoption and replay hand one
    /// generation to several readers; the first to receive the whole answer counts it).
    bool uncounted = false;
  };
  virtual ~TextGenerator() = default;
  /// Generates, calling `on_text` with each new piece of text (whole UTF-8
  /// characters); stops early when it returns false or Params::cancelled
  /// returns true.
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
  std::size_t max_stored_responses = 1000;  // for the default in-memory store
  /// Where stored responses live (optional; must outlive the service).
  /// Default: in memory, max_stored_responses of them.
  ResponseStore* store = nullptr;
  /// Told about every generation (optional; must outlive the service).
  GenerationObserver* observer = nullptr;
  /// Called when a generation starts and ends (optional), e.g. for the
  /// requests-running gauge.
  std::function<void(bool started)> on_activity;
  /// Called when a request is refused 429 before it runs (G3 admission), with why ("limit", "memory").
  std::function<void(const std::string& why)> on_refused;
  /// P10 variants the generator serves. Listed by /v1/models as
  /// "<model_id>:<name>"; a request naming any other variant (the
  /// "variant" field or a "base:variant" model alias) is answered 404.
  /// Unset: every variant is refused.
  std::function<std::vector<std::string>()> variants;
  /// Streamed requests: an SSE comment (": keepalive", which SSE clients
  /// ignore) after this many ms without a write, while the request is
  /// queued, prefilling or between tokens, so a client's idle-read timeout
  /// does not fire before the first token and a departed client fails the
  /// write. 0 (the default): none.
  int sse_keepalive_ms = 0;
  /// Ids and timestamps; tests pass deterministic ones.
  std::function<std::string()> new_id;
  std::function<std::int64_t()> now_s;
};

class OpenAiService {
 public:
  OpenAiService(TextGenerator& generator, ServiceOptions options);
  /// Registers the routes on `server`. The service must outlive the server's
  /// Stop(): streaming responses run on the server's worker threads.
  void Register(HttpServer& server);

  HttpResponse Models(const HttpRequest& req);
  HttpResponse Completions(const HttpRequest& req);
  HttpResponse Chat(const HttpRequest& req);
  HttpResponse Responses(const HttpRequest& req);
  HttpResponse GetResponse(const std::string& id);

 private:
  /// `connected` (optional): whether the client is still there (HttpRequest::connected).
  HttpResponse Generate(Api api, GenerationRequest req, const json::Value* prior,
                        const std::function<bool()>& connected);
  /// Stores a finished Responses turn (`final`, after `conversation`); empty on success, else why it was not
  /// stored (then nothing was kept).
  std::string Store(const std::string& id, const json::Value& conversation, const json::Value& final);

  TextGenerator& gen_;
  ServiceOptions opt_;
  std::unique_ptr<InMemoryResponseStore> own_store_;
  ResponseStore* store_ = nullptr;
};

}  // namespace halo::server

#endif  // HALO_SERVER_OPENAI_SERVICE_HPP_
