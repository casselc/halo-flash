#ifndef HALO_SERVER_OPERATION_API_HPP_
#define HALO_SERVER_OPERATION_API_HPP_

// How an API runs as an operation (docs/OPERATIONS.md section 3): the services adapt themselves to this, and the
// operation service (server/operation_service.hpp) dispatches a submit's "api" to the adapter registered under it.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "server/http.hpp"

namespace halo::server {

/// What a running operation reports, and how it learns it was cancelled. It has no connection behind it: a
/// disconnect never applies.
struct OperationHooks {
  /// The cancel verb (or the server stopping).
  std::function<bool()> cancelled;
  /// Right after admission. `refused`: the attempt never runs and `done` is not called; `retry_after_s` and `why`
  /// ("limit", "memory") say why.
  struct Admission {
    bool refused = false;
    std::int64_t retry_after_s = 0;
    std::string why;
  };
  std::function<void(const Admission&)> admitted;
  /// The answer's stream events as they are produced, each framed as that API's SSE frame: the observation stream.
  std::function<void(std::vector<std::string>&)> events;
  /// The end: the API's final object (cancelled answers too: their usage is real), or an error.
  struct End {
    json::Value final;
    std::string error;   ///< non-empty: it failed
    std::string finish;  ///< "stop", "length", "tool_calls", "cancelled" or "error"
    std::uint64_t prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
    double engine_ms = 0;
  };
  std::function<void(End)> done;
};

struct OperationApi {
  struct Prepared {
    std::shared_ptr<void> run;      ///< the adapter's own state
    std::uint64_t max_tokens = 0;   ///< for the retention reservation
  };
  /// Validates `request` for this API (the submit's headers give the principal and chain); nullopt with `error`
  /// when it is not a valid request.
  std::function<std::optional<Prepared>(const json::Value& request, const HttpRequest& submit, std::string* error)>
      prepare;
  /// Admits and runs it, blocking (a worker thread calls it). Returns an error response instead when it could not
  /// run, before admission; then no hook was called.
  std::function<std::optional<HttpResponse>(std::shared_ptr<void> run, const OperationHooks& hooks)> run;
};

}  // namespace halo::server

#endif  // HALO_SERVER_OPERATION_API_HPP_
