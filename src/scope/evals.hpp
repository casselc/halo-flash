#ifndef HALO_SCOPE_EVALS_HPP_
#define HALO_SCOPE_EVALS_HPP_

// halo-scope datasets, evaluators and experiment runs on SQLite (PLAN.md P12;
// the schema and its placement are in evals_schema.hpp). Langfuse's shape:
// a dataset holds items (input, expected output, metadata, optional source
// trace/observation), a dataset run ("experiment") runs every item against a
// target and links each item to the trace it produced, and evaluators score
// the results. Differences from Langfuse: dataset versions are explicit and
// immutable once frozen (a run always names a frozen version), evaluator
// versions are immutable, and a run is bound to the content it ran on.
//
// Thread-safe: one connection, each method under a mutex.

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"
#include "scope/metadata.hpp"

struct sqlite3;

namespace halo::scope {

/// JSON with object keys sorted (recursively), compact separators: the form
/// items, targets and configs are stored and hashed in.
std::string CanonicalJson(const json::Value& v);
/// Parses `text` and returns its canonical form, or nullopt with `*err`.
std::optional<std::string> Canonicalize(std::string_view text, std::string* err);
/// BLAKE3-128 of the length-framed fields, as 32 hex digits.
std::string FramedHash(const std::vector<std::string_view>& fields);

struct DatasetVersionInfo {
  std::int64_t version = 0;
  bool frozen = false;
  std::string content_hash;  // frozen only
  std::int64_t item_count = 0;  // frozen: as frozen; draft: current
  std::string created_at, frozen_at;
};

struct DatasetInfo {
  std::int64_t id = 0;
  std::string name, description, metadata_json, created_at;
  std::string archived_at;  // empty: active
  std::vector<DatasetVersionInfo> versions;  // ascending
};

struct DatasetItem {
  std::string id;  // empty on input: a fresh id is assigned
  std::string input_json = "null";
  std::string expected_json = "null";
  std::string metadata_json = "{}";
  std::string source_trace_id, source_observation_id;
  // filled by the store
  std::int64_t ordinal = 0;
  std::string content_hash;
};

/// The content hash of one item (its id and every content field).
std::string ItemHash(const DatasetItem& item);

struct EvaluatorDef {
  std::string name;
  std::int64_t version = 0;
  std::string kind;  // exact, regex, json_schema, numeric, code, llm_judge, decider
  std::string config_json = "{}";
};

struct RunSpec {
  std::string name;
  std::string dataset;
  std::int64_t dataset_version = 0;  // 0: the latest frozen version
  /// {"prompt": {"name", "version" | "label"}?, "model", "variant"?, "params": {...}?,
  ///  "request": {...}?, "endpoint"?}
  std::string target_json = "{}";
  /// (name, version); version 0 pins the latest.
  std::vector<std::pair<std::string, std::int64_t>> evaluators;
  int max_attempts = 3;
  std::string metadata_json = "{}";
};

struct RunRecord {
  std::string id, name;
  std::int64_t dataset_id = 0;
  std::string dataset;
  std::int64_t dataset_version = 0;
  std::string target_json;      // canonical, prompt version resolved
  std::string evaluators_json;  // canonical [{"name","version"}]
  std::string binding;
  std::int64_t item_count = 0;
  int max_attempts = 3;
  std::string status, metadata_json, created_at, updated_at, completed_at;
};

struct RunProgress {
  std::int64_t total = 0, done = 0, running = 0, error = 0, exhausted = 0, pending = 0;
};

/// One run item's committed result.
struct ItemResult {
  std::string observation_id, output, model, finish_reason;
  std::int64_t input_tokens = 0, output_tokens = 0;
  double latency_ms = 0;
};

/// One score, as committed with its item.
struct ScoreRow {
  std::string item_id;
  std::string name, evaluator;
  std::int64_t evaluator_version = 0;
  std::string data_type;  // NUMERIC, BOOLEAN, CATEGORICAL
  std::optional<double> value;  // nullopt: the evaluation failed (see error)
  std::string string_value, comment, error, score_id;
  std::string judge_model, judge_prompt;
  std::int64_t judge_prompt_version = 0;
  std::string judge_trace_id, raw_output;
};

struct RunItemRow {
  std::string item_id, item_hash, status, trace_id, observation_id, output, model, finish_reason, error;
  std::int64_t attempts = 0, input_tokens = 0, output_tokens = 0;
  double latency_ms = 0;
};

/// What BeginItem decided.
struct ItemStart {
  enum class State { kStarted, kAlreadyDone, kExhausted } state = State::kStarted;
  int attempt = 0;
  std::string trace_id;
};

/// The trace id of attempt `attempt` of item `item_id` in run `run_id`:
/// deterministic, so a crashed attempt's trace is recognisable and a new
/// attempt never reuses it.
std::string AttemptTraceId(std::string_view run_id, std::string_view item_id, int attempt);

class EvalStore {
 public:
  /// Opens (creating and migrating with MetadataStore::Migrations()) the
  /// metadata database at `path`.
  static std::unique_ptr<EvalStore> Open(const std::string& path, std::string* err);
  ~EvalStore();
  EvalStore(const EvalStore&) = delete;
  EvalStore& operator=(const EvalStore&) = delete;

  // --- datasets ---
  /// Creates a dataset. Fails if the name exists.
  bool CreateDataset(std::string_view name, std::string_view description, std::string_view metadata_json,
                     std::string* err);
  std::optional<DatasetInfo> GetDataset(std::string_view name);
  std::vector<DatasetInfo> ListDatasets(bool include_archived);
  /// Adds or replaces items (by id) in the draft, opening a draft (a copy of
  /// the latest frozen version) when there is none. All or nothing. Fills
  /// generated ids into `items`. Returns the draft version, or 0.
  std::int64_t UpsertItems(std::string_view dataset, std::vector<DatasetItem>& items, std::string* err);
  /// Parses JSONL ({"id"?, "input", "expected_output"?, "metadata"?,
  /// "source_trace_id"?, "source_observation_id"?} per line; blank lines
  /// skipped) and upserts every line or none. Returns the draft version.
  std::int64_t ImportJsonl(std::string_view dataset, std::string_view jsonl, std::size_t* count, std::string* err);
  /// Removes an item from the draft.
  bool DeleteItem(std::string_view dataset, std::string_view item_id, std::string* err);
  /// Freezes the draft: computes the content hash and item count and makes
  /// the version immutable. Returns its info.
  std::optional<DatasetVersionInfo> Freeze(std::string_view dataset, std::string* err);
  /// Archives the dataset (readable; no new drafts, items or runs).
  bool Archive(std::string_view dataset, std::string* err);
  /// Items of `version` (0: the draft if any, else the latest frozen), by ordinal.
  std::optional<std::vector<DatasetItem>> Items(std::string_view dataset, std::int64_t version, std::string* err);

  // --- evaluators ---
  /// Appends an immutable version. The config is validated by the caller
  /// (ValidateEvaluator in evaluators.hpp). Returns the version, or 0.
  std::int64_t CreateEvaluatorVersion(std::string_view name, std::string_view kind, std::string_view config_json,
                                      std::string* err);
  /// version 0: the latest.
  std::optional<EvaluatorDef> GetEvaluator(std::string_view name, std::int64_t version);
  std::vector<EvaluatorDef> ListEvaluators();

  // --- prompts (read from the same database) ---
  std::optional<PromptVersion> PromptByVersion(std::string_view name, std::int64_t version);
  std::optional<PromptVersion> PromptByLabel(std::string_view name, std::string_view label);

  // --- runs ---
  /// Creates the run, or returns the existing run of the same name on the
  /// dataset when its binding is identical (`*existed` true): resuming.
  /// A run of that name with a different binding is refused.
  std::optional<RunRecord> CreateRun(const RunSpec& spec, bool* existed, std::string* err);
  std::optional<RunRecord> GetRun(std::string_view run_id);
  std::vector<RunRecord> ListRuns(std::string_view dataset);
  /// Recomputes a run's binding from the stored dataset version, prompt and
  /// evaluator versions; equal to RunRecord::binding unless the stored
  /// content changed under it.
  std::optional<std::string> RecomputeBinding(const RunRecord& run, std::string* err);
  bool SetRunStatus(std::string_view run_id, std::string_view status, std::string* err);
  /// Items not yet done with attempts left, in dataset order.
  std::optional<std::vector<DatasetItem>> PendingItems(const RunRecord& run, std::string* err);
  /// Opens an attempt: a done item is skipped, one with max_attempts used
  /// is exhausted, otherwise the attempt count goes up and the item is
  /// 'running' under AttemptTraceId(run, item, attempt). Durable on return.
  std::optional<ItemStart> BeginItem(const RunRecord& run, const DatasetItem& item, std::string* err);
  /// Commits an attempt's output and all its scores in one transaction and
  /// marks the item done. Fails (changing nothing) unless the item is
  /// 'running' at `attempt`.
  bool CompleteItem(const RunRecord& run, std::string_view item_id, int attempt, const ItemResult& result,
                    const std::vector<ScoreRow>& scores, std::string* err);
  /// Records a failed attempt ('error'); retried while attempts remain.
  bool FailItem(const RunRecord& run, std::string_view item_id, int attempt, std::string_view error,
                std::string* err);
  RunProgress Progress(const RunRecord& run);
  std::vector<RunItemRow> RunItems(std::string_view run_id);
  std::vector<ScoreRow> RunScores(std::string_view run_id);

 private:
  EvalStore() = default;
  std::optional<std::int64_t> DatasetId(std::string_view name, bool* archived);
  std::int64_t EnsureDraft(std::int64_t dataset_id, std::string* err);
  bool UpsertLocked(std::int64_t dataset_id, std::int64_t version, std::vector<DatasetItem>& items, std::string* err);
  std::optional<std::string> BindingLocked(std::int64_t dataset_id, std::int64_t dataset_version,
                                           const std::string& target_json, const std::string& evaluators_json,
                                           std::string* err);
  std::optional<RunRecord> RunLocked(std::string_view run_id);
  std::optional<std::vector<DatasetItem>> ItemsLocked(std::int64_t dataset_id, std::int64_t version);
  std::string NewId();

  sqlite3* db_ = nullptr;
  std::mutex mu_;
};

}  // namespace halo::scope

#endif  // HALO_SCOPE_EVALS_HPP_
