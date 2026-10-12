#ifndef HALO_SERVER_CONTRACT_HPP_
#define HALO_SERVER_CONTRACT_HPP_

// GET /v1/halo/contract: the incarnation, its clock, and the guarantees this server gives (docs/OPERATIONS.md
// section 5; cordis RFC 0007). Without the operation verbs (--operations 0), "operations" is null and the header
// contract (Idempotency-Key, Halo-Answer, Halo-Discarded-Tokens, Halo-Admitted-Sends) is the one in force.

#include <functional>
#include <string>

#include "core/json.hpp"

#include "server/http.hpp"
#include "server/incarnation.hpp"

namespace halo::server {

struct ContractInfo {
  /// Concurrent sessions decode through the relaxed batched kernels (`--batch-relaxed 1`): a greedy session's
  /// output may depend on which other sessions share its batch.
  bool relaxed = true;
  /// The operation verbs' section (OperationService::Contract); unset: "operations" is null (not served, and the
  /// header contract is the one in force).
  std::function<json::Value()> operations;
};

/// The contract document, as JSON (the incarnation's clock read now).
[[nodiscard]] std::string ContractJson(const Incarnation& inc, const ContractInfo& info);

/// Registers GET /v1/halo/contract. `inc` must outlive the server.
void RegisterContract(HttpServer& server, const Incarnation& inc, ContractInfo info);

}  // namespace halo::server

#endif  // HALO_SERVER_CONTRACT_HPP_
