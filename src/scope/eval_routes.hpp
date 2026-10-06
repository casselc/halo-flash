#ifndef HALO_SCOPE_EVAL_ROUTES_HPP_
#define HALO_SCOPE_EVAL_ROUTES_HPP_

// Datasets, evaluators, experiment runs, scores and run comparison over HTTP
// (PLAN.md P12), on EvalStore. JSON bodies; errors are
// {"error": {"message"}} with 400 (bad request), 404 (unknown name/id) or
// 409 (conflict: a run name reused with a different binding, a frozen or
// archived dataset).
//
//   POST /v1/scope/datasets                    {name, description?, metadata?}
//   GET  /v1/scope/datasets[?archived=1]
//   GET  /v1/scope/datasets/{name}             versions (frozen: content hash, item count)
//   POST /v1/scope/datasets/{name}/items       {items: [{id?, input, expected_output?, metadata?,
//                                               source_trace_id?, source_observation_id?}]}
//                                              (or one item object) -> {version, ids}
//   GET  /v1/scope/datasets/{name}/items[?version=N]   (default: the draft, else the latest)
//   DELETE /v1/scope/datasets/{name}/items/{id}        (draft only)
//   POST /v1/scope/datasets/{name}/import      JSONL body, one item per line -> {version, count}
//   POST /v1/scope/datasets/{name}/from_traces {trace_ids?: [...], observation_ids?: [...],
//                                               expected?: "output" | "none"}
//                                              items from recorded observations (input = the
//                                              recorded input content; needs capture "full")
//   POST /v1/scope/datasets/{name}/freeze      -> {version, content_hash, item_count}
//   POST /v1/scope/datasets/{name}/archive
//   POST /v1/scope/evaluators                  {name, kind, config} -> {name, version, ...}
//   GET  /v1/scope/evaluators[/{name}[?version=N]]
//   POST /v1/scope/runs                        {name, dataset, dataset_version?, target,
//                                               evaluators: ["name" | "name@version" |
//                                               {name, version}], max_attempts?, concurrency?,
//                                               metadata?, wait?}
//                                              creates (or resumes: same name and binding) and
//                                              starts the run in the background; 202 with the
//                                              run (200 when wait is true and it finished)
//   GET  /v1/scope/runs[?dataset=D]
//   GET  /v1/scope/runs/{id}                   run, progress, per-score summary
//   GET  /v1/scope/runs/{id}/items             items with output, trace and scores (judge
//                                              model, prompt version, trace and raw output)
//   GET  /v1/scope/runs/{id}/scores            per-score summary (n, mean, se, counts, errors)
//   POST /v1/scope/runs/{id}/resume            restart a failed, cancelled or crashed run
//   POST /v1/scope/runs/{id}/cancel
//   GET  /v1/scope/compare?a=RUN&b=RUN&score=NAME[&items=1]
//                                              paired per-item deltas b - a: n, mean, se, wins,
//                                              losses, ties, unpaired counts

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "scope/evals.hpp"
#include "scope/runner.hpp"
#include "scope/trace_reader.hpp"
#include "server/http.hpp"

namespace halo::scope {

struct EvalRoutesOptions {
  /// OpenAI-compatible base URL of targets and judges without their own
  /// endpoint (the engine's own server, e.g. http://127.0.0.1:8080).
  std::string endpoint;
  /// Serves /v1/systemone (default: endpoint).
  std::string decider_endpoint;
  /// Headers for those calls (e.g. Authorization when the server has a key).
  HttpHeaders headers;
  Annotator* annotator = nullptr;  // score mirroring into chDB
  TraceReader* traces = nullptr;   // for from_traces
  const CodeHooks* hooks = nullptr;
  JsonPoster* poster = nullptr;    // default: an HttpPoster
};

/// Owns the background run executors; at most one per run.
class EvalService {
 public:
  EvalService(EvalStore& store, EvalRoutesOptions options);
  ~EvalService();  // cancels and joins every executor
  void Register(server::HttpServer& server);
  /// Starts executing `run_id` unless it is already running. False when it is.
  bool Start(const std::string& run_id, int concurrency = 1);
  /// Blocks until `run_id`'s executor (if any) finishes.
  void Wait(const std::string& run_id);
  bool Cancel(const std::string& run_id);
  bool Active(const std::string& run_id);

 private:
  struct Exec {
    std::thread thread;
    std::atomic<bool> cancel{false};
    std::atomic<bool> finished{false};
  };
  void Reap();
  EvalStore& store_;
  EvalRoutesOptions opt_;
  std::unique_ptr<JsonPoster> owned_poster_;
  std::mutex mu_;
  std::map<std::string, std::unique_ptr<Exec>> execs_;  // guarded by mu_
};

}  // namespace halo::scope

#endif  // HALO_SCOPE_EVAL_ROUTES_HPP_
