#ifndef HALO_SERVER_TURN_HPP_
#define HALO_SERVER_TURN_HPP_

// One generation turn, as every API route runs it (OpenAI completions, chat and responses; Anthropic messages):
// everything that does not depend on the API's request or response shape. A route parses its request and renders
// its prompt, then:
//   1. SetTurnContext: the halo request headers (trace, session, prompt, variant, G3 scheduling);
//   2. DefaultIdempotencyKey (G3 D5) and AdmitTurn: admitted before anything is sent (the 429 is the route's, in
//      its API's error shape);
//   3. RunTurn: the generation with its stop sequences, a departed client told from a stop (G3: only a stop
//      completes an answer the scheduler stores and replays), the activity hook and the observer's timing; the
//      route's sink turns text into its API's events;
//   4. ReportTurn: the observer record's common fields, once the route knows how its response ended.
// A streamed route runs inside an SsePump, which writes its events, keeps the stream alive while nothing else is
// written, and notices a client that left.

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"
#include "server/http.hpp"
#include "server/observer.hpp"
#include "server/openai_request.hpp"
#include "server/openai_service.hpp"

namespace halo::server {

/// X-Halo-Trace-Id, X-Halo-Session-Id, X-Halo-Prompt (name@version), X-Halo-Variant-Id, and the G3 scheduling
/// headers X-Halo-Principal, X-Halo-Chain-Id and Idempotency-Key.
void SetTurnContext(const HttpRequest& http, GenerationRequest& req);

/// G3 D5: a request a broker stamped with a principal and no Idempotency-Key gets a default key, a digest of
/// (principal, `material`, API), so a client's retry of the same turn replays or adopts instead of generating
/// again. Anonymous requests get none (callers repeating a prompt mean it).
void DefaultIdempotencyKey(GenerationRequest& req, std::string_view material, Api api);

/// G3 admission, before anything is sent.
[[nodiscard]] TextGenerator::Admission AdmitTurn(TextGenerator& gen, const GenerationRequest& req,
                                                 const std::string& prompt, const std::function<bool()>& gone);

/// The Halo-Answer header for an admitted request (none when the generator keeps no keys), and for a keyed one its
/// key's history: Halo-Discarded-Tokens (tokens its fenced attempts decoded) and Halo-Admitted-Sends (requests
/// admitted under the key's current state, this one included).
[[nodiscard]] std::vector<std::pair<std::string, std::string>> AnswerHeaders(const TextGenerator::Admission& a);
void AddAnswerHeader(HttpResponse& r, const TextGenerator::Admission& a);

struct TurnIo {
  /// Text after stop matching, in the API's events; false ends the generation here (a cut, not a departure).
  std::function<bool(std::string_view)> sink;
  /// Writes the events so far; false once the client is gone (always true when not streaming).
  std::function<bool()> flush;
  /// Whether the connection closed.
  std::function<bool()> gone;
  /// Polled while queued, prefilling and between tokens (gone, plus a stream's keepalive).
  std::function<bool()> poll;
};

struct TurnOutcome {
  TextGenerator::Result result;
  bool stopped = false;      ///< a stop sequence ended the output
  std::string matched;       ///< which one
  bool cut = false;          ///< the sink ended it
  bool client_gone = false;  ///< a write failed or the connection closed during the generation
};

/// Runs the generation. `rec` (with an observer) gets the start, first-text and gap timings and the output.
[[nodiscard]] TurnOutcome RunTurn(TextGenerator& gen, const ServiceOptions& opt, const GenerationRequest& req,
                                  const std::string& prompt, std::shared_ptr<void> ticket, const TurnIo& io,
                                  GenerationRecord* rec);

/// Fills `rec`'s common fields and tells the observer (if any). `finish`: how the response ended when it was
/// neither an error, detached nor cancelled ("stop", "length", "tool_calls").
void ReportTurn(const ServiceOptions& opt, GenerationRecord& rec, std::string api, std::string request_id,
                const GenerationRequest& req, const std::string& prompt, const TurnOutcome& out, bool cancelled,
                std::string finish, const std::vector<std::pair<std::string, std::string>>& tool_calls);

/// A streamed response's writer: events are written as `frame(event)`; after `keepalive_ms` without a write
/// (queued, prefilling, between tokens) `keepalive` writes something (an SSE comment, a ping event), so a
/// client's idle timer does not fire and a client that left fails the write.
class SsePump {
 public:
  SsePump(const BodyWriter& write, int keepalive_ms, std::function<std::string(const json::Value&)> frame,
          std::function<void(SsePump&)> keepalive);
  std::vector<json::Value> events;
  /// Writes `events` and clears them; false once a write failed.
  bool Flush();
  /// Writes a raw frame.
  bool Write(const std::string& raw);
  /// Params::cancelled for a streamed turn: gone, or a write failed (a due keepalive is written first).
  bool Poll(const std::function<bool()>& gone);
  [[nodiscard]] bool alive() const { return alive_; }

 private:
  using Clock = std::chrono::steady_clock;
  const BodyWriter& write_;
  int keepalive_ms_;
  std::function<std::string(const json::Value&)> frame_;
  std::function<void(SsePump&)> keepalive_;
  bool alive_ = true;
  Clock::time_point last_ = Clock::now();
};

}  // namespace halo::server

#endif  // HALO_SERVER_TURN_HPP_
