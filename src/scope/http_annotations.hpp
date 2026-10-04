#ifndef HALO_SCOPE_HTTP_ANNOTATIONS_HPP_
#define HALO_SCOPE_HTTP_ANNOTATIONS_HPP_

// The harness annotation API (include/halo_scope.h) over HTTP, for harnesses
// in other processes. JSON bodies; ids are strings.
//   POST /v1/scope/steps                    {trace_id, parent?, kind, name}
//                                           -> {"step_id"}
//   POST /v1/scope/steps/{step_id}/end      {status?, output?}
//   POST /v1/scope/tool_results             {trace_id, tool_call_id, tool_name?, status?,
//                                            output?, latency_ms?}
//   POST /v1/scope/scores                   {trace_id?, observation_id?, session_id?, name,
//                                            value (number | string | bool), source?,
//                                            comment?, supersedes?} -> {"score_id"}
//   POST /v1/scope/scores/{score_id}/retract {source?, comment?}
//   POST /v1/scope/traces/{trace_id}/meta   {key, value}
// kind: tool, retry, plan, memory, custom; status: ok (default) or error.
// 200 on success, 400 {"error": {"message"}} for a bad request, 429 when a
// ring is full (the record was dropped and counted; retry later).

#include "scope/annotate.hpp"
#include "server/http.hpp"

namespace halo::scope {

/// `annotator` must outlive the server's Stop().
void RegisterAnnotationRoutes(server::HttpServer& server, Annotator& annotator);

}  // namespace halo::scope

#endif  // HALO_SCOPE_HTTP_ANNOTATIONS_HPP_
