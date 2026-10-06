#ifndef HALO_SCOPE_TRACE_READER_HPP_
#define HALO_SCOPE_TRACE_READER_HPP_

// The read side of halo-scope over HTTP (one process owns the chDB data, so
// readers go through the engine):
//   GET /v1/scope/traces?session_id=S&limit=N   recent traces, newest first:
//       trace_id, start/end (ns), observations, token sums, session_id
//       (the session filter applies per observation: a harness's own rows
//       carry no session, so they are not counted there)
//   GET /v1/scope/traces/{trace_id}             one trace: its context
//       (latest non-empty session/user/name/release, all tags, metadata by
//       key), observations by start time, and its live scores (asserted,
//       not superseded or retracted)
// Values reach SQL only as quoted literals; the statements are fixed.
// Records become visible once the writer has flushed them (about a second).

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "scope/chdb.hpp"
#include "server/http.hpp"

namespace halo::scope {

/// One observation's recorded content (capture policy "full" keeps it;
/// "digest" keeps only hashes, so has_input/has_output are false).
struct ObservationContent {
  std::string trace_id, span_id, type, name;
  std::string input, output;
  bool has_input = false, has_output = false;
};

class TraceReader {
 public:
  /// `conn` is a connection of its own (not the writer's); used under a mutex.
  TraceReader(std::unique_ptr<Conn> conn, std::string project) : conn_(std::move(conn)), project_(std::move(project)) {}
  [[nodiscard]] std::optional<json::Value> ListTraces(const std::string& session_id, std::uint32_t limit,
                                                      std::string* error);
  [[nodiscard]] std::optional<json::Value> Trace(const std::string& trace_id, std::string* error);
  /// The content of the observations with these span ids, or of every
  /// GENERATION observation in these traces (start order).
  [[nodiscard]] std::optional<std::vector<ObservationContent>> Contents(const std::vector<std::string>& trace_ids,
                                                                      const std::vector<std::string>& span_ids,
                                                                      std::string* error);
  /// Registers the two GET routes.
  void Register(server::HttpServer& server);

 private:
  std::optional<json::Value> Rows(const std::string& sql, std::string* error);
  std::unique_ptr<Conn> conn_;
  const std::string project_;
  std::mutex mu_;
};

}  // namespace halo::scope

#endif  // HALO_SCOPE_TRACE_READER_HPP_
