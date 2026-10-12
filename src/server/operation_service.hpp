#ifndef HALO_SERVER_OPERATION_SERVICE_HPP_
#define HALO_SERVER_OPERATION_SERVICE_HPP_

// The operation verbs (docs/OPERATIONS.md section 3; cordis RFC 0007), stage 3 of the operation API:
//   POST /v1/operations                                  submit
//   GET  /v1/operations/{op}/{attempt}                   status
//   GET  /v1/operations/{op}/{attempt}/result            fetch (while the body is held)
//   GET  /v1/operations/{op}/{attempt}/stream?from=N     observe (SSE from event N, while the body is held)
//   POST /v1/operations/{op}/{attempt}/cancel            cancel
//   POST /v1/operations/{op}/{attempt}/ack               ack (the platform committed the result)
//
// The records are Operations' (server/operations.hpp); this service binds them to HTTP and to execution. A submit's
// "api" names an adapter (server/operation_api.hpp) a service registered with AddApi: OpenAiService ("completions",
// "chat", "responses"), AnthropicService ("messages"), and the decider ("systemone"). An accepted attempt runs on a
// worker thread through its adapter: the scheduler's admission and fairness as any request, with no connection
// behind it (a disconnect, of the submitter or of an observer, never cancels; only the cancel verb does). A
// scheduler refusal (principal limit, memory) withdraws the record and answers the submit 429/503, as a refusal of
// that copy.
//
// Identity: the owner is the request's principal (X-Halo-Principal, set by the broker; empty: anonymous), the
// operation id and attempt are the platform's. Every verb carries Halo-Incarnation; status, fetch, observe,
// cancel and ack also carry Halo-Not-After and Halo-Digest. The digest is the SHA-256 (hex) of the submit's body
// bytes, which bind api, request, not_after and mode; a submit may carry it too, and must then match.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "core/json.hpp"
#include "server/http.hpp"
#include "server/incarnation.hpp"
#include "server/operation_api.hpp"
#include "server/operations.hpp"

namespace halo::server {

struct OperationServiceOptions {
  Operations::Options records;
  /// Retention reserved at acceptance: base + max_tokens x bytes_per_token (the final object and the observation
  /// stream's events, which carry each piece again).
  std::uint64_t base_bytes = 16 * 1024;
  std::uint64_t bytes_per_token = 256;
  /// The inference mode this server runs ("relaxed" or "exact"); a submit naming the other is refused (422).
  bool relaxed = true;
};

class OperationService {
 public:
  OperationService(const Incarnation& incarnation, OperationServiceOptions options);
  /// Cancels every running attempt (as the server stopping) and waits for their workers.
  ~OperationService();
  OperationService(const OperationService&) = delete;
  OperationService& operator=(const OperationService&) = delete;

  /// Serves `api` through `adapter`; before the server starts.
  void AddApi(std::string api, OperationApi adapter);
  void Register(HttpServer& server);
  /// The contract's "operations" section (GET /v1/halo/contract).
  [[nodiscard]] json::Value Contract() const;

  HttpResponse Submit(const HttpRequest& http);
  HttpResponse Verb(const HttpRequest& http);

  [[nodiscard]] Operations& records() { return ops_; }

 private:
  /// The observation stream of a held attempt: its events as SSE frames, and whether it has ended.
  struct Live {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::string> frames;
    bool ended = false;
  };

  HttpResponse Observe(const Operations::Ref& ref, std::size_t from);
  void DropReleased();

  std::map<std::string, OperationApi> apis_;
  const Incarnation& inc_;
  const OperationServiceOptions opt_;
  Operations ops_;
  std::mutex mu_;  // live_
  std::map<Operations::Id, std::shared_ptr<Live>> live_;
  std::atomic<bool> stopping_{false};
  std::mutex workers_mu_;
  std::condition_variable workers_cv_;
  int workers_ = 0;
};

}  // namespace halo::server

#endif  // HALO_SERVER_OPERATION_SERVICE_HPP_
