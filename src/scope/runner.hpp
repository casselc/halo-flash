#ifndef HALO_SCOPE_RUNNER_HPP_
#define HALO_SCOPE_RUNNER_HPP_

// The experiment runner (PLAN.md P12): runs a dataset run's pending items
// against its target, scores each with the run's evaluators, and commits
// each item's output and scores as one durable checkpoint (EvalStore).
//
// Resumable and idempotent: every item goes through BeginItem (a durable
// attempt with its own trace id), the target call, the evaluators, a mirror
// of the scores into chDB (llm_scores facts with ids derived from run, item
// and attempt, attached to the attempt's trace) and CompleteItem (output and
// scores in one transaction). A crash at any point leaves the item either
// done (counted once; never redone) or not done (retried by the next
// Execute under a new attempt and trace). Before any work the run's binding
// is recomputed from the stored dataset version, prompts and evaluators, so a
// resumed run provably runs on the content it started with.
//
// The target is an OpenAI-compatible /v1/chat/completions endpoint (the
// engine's own server, or target.endpoint) or an in-process callback. Each
// request carries X-Halo-Trace-Id (the attempt's trace), X-Halo-Session-Id
// (the run id), X-Halo-Prompt (name@version) and X-Halo-Variant-Id, so the
// engine's GENERATION observation lands in the item's trace.

#include <atomic>
#include <functional>
#include <string>
#include <string_view>

#include "core/json.hpp"
#include "scope/annotate.hpp"
#include "scope/evals.hpp"
#include "scope/evaluators.hpp"
#include "scope/http_client.hpp"

namespace halo::scope {

/// Thrown by a fault hook to simulate a crash: the runner stops at once,
/// writing nothing more (tests resume afterwards).
struct SimulatedCrash : std::exception {
  const char* what() const noexcept override { return "simulated crash"; }
};

struct TargetCall {
  std::string run_id, item_id, trace_id;
  int attempt = 0;
  json::Value body;  // the chat completion request
  HttpHeaders headers;
};

struct RunnerOptions {
  /// Concurrent items (1: run inline on the calling thread).
  int concurrency = 1;
  /// Base URL of the target when target.endpoint is absent.
  std::string target_endpoint;
  HttpHeaders target_headers;
  /// Judges, deciders and code hooks.
  EvalEnv env;
  /// Mirrors scores and run context into chDB when set.
  Annotator* annotator = nullptr;
  /// In-process target instead of HTTP (C ABI embedders).
  std::function<ChatReply(const TargetCall&)> target;
  /// Called at each checkpoint boundary: "begin", "started", "generated",
  /// "evaluated", "mirrored", "committed". May throw SimulatedCrash.
  std::function<void(std::string_view point, const std::string& item_id)> fault;
  /// After each item settles (done or failed).
  std::function<void(const RunProgress&)> on_progress;
  const std::atomic<bool>* cancel = nullptr;
};

struct RunOutcome {
  std::string status;  // completed, failed, cancelled, crashed, refused
  RunProgress progress;
  std::string error;
};

/// The chat completion body for one item under the run's target (prompt
/// rendered with the item's variables, decoding parameters, extra request
/// fields). nullopt with `*err` when the item cannot be rendered.
std::optional<json::Value> BuildTargetRequest(EvalStore& store, const json::Value& target, const DatasetItem& item,
                                              std::string* err);

/// The id of the llm_scores fact mirroring score `name` of an attempt.
std::string ScoreFactId(std::string_view run_id, std::string_view item_id, int attempt, std::string_view name);

/// Runs every pending item of `run_id`. Safe to call again after any crash
/// or cancellation: done items are never redone.
RunOutcome ExecuteRun(EvalStore& store, JsonPoster& poster, const std::string& run_id, const RunnerOptions& options);

}  // namespace halo::scope

#endif  // HALO_SCOPE_RUNNER_HPP_
