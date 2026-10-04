#ifndef HALO_SERVER_SYSTEMONE_SERVICE_HPP_
#define HALO_SERVER_SYSTEMONE_SERVICE_HPP_

// POST /v1/systemone (PLAN.md P9): typed decisions (noul, choice, score) over
// a state, in the Jev / strands-decider schema. Status codes and bodies
// follow strands-decider's FastAPI server: 200 with the answers plus
// "latency_ms"; 422 {"detail": message} for a request that fails validation
// or that the engine refuses (too many options, images on a text engine).

#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "core/json.hpp"
#include "decide/systemone.hpp"
#include "server/http.hpp"

namespace halo::server {

/// Answers one validated request; returns the response body or nullopt
/// with `error` set.
class DecisionEngine {
 public:
  virtual ~DecisionEngine() = default;
  virtual std::optional<json::Value> Evaluate(const decide::Request& request, std::string* error) = 0;
};

/// Registers POST /v1/systemone on `server` for `engine` (which must outlive
/// the server). Requests are evaluated one at a time.
void RegisterSystemOne(HttpServer& server, DecisionEngine& engine);

}  // namespace halo::server

#endif  // HALO_SERVER_SYSTEMONE_SERVICE_HPP_
