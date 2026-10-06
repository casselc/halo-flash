/* halo-scope harness annotation C ABI (PLAN.md P12).
 *
 * An embedding harness attaches its own steps, tool results, scores and
 * trace context to the engine's traces, so tool executions appear in the
 * same waterfall as the generations that requested them. Every call writes
 * one typed row through halo-scope's non-blocking write path.
 *
 * Strings are NUL-terminated UTF-8; NULL means empty. Output content is
 * passed with an explicit length and may contain NULs. Every function is
 * thread-safe. Return codes:
 *   HALO_SCOPE_OK       recorded
 *   HALO_SCOPE_DROPPED  a ring was full; the drop is counted, never silent
 *   HALO_SCOPE_INVALID  bad argument (NULL handle, empty id, unknown step or
 *                       kind); nothing recorded
 */
#ifndef HALO_SCOPE_H_
#define HALO_SCOPE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct halo_scope halo_scope;

enum { HALO_SCOPE_OK = 0, HALO_SCOPE_DROPPED = 1, HALO_SCOPE_INVALID = -1 };

typedef enum {
  HALO_SCOPE_STEP_TOOL = 0,
  HALO_SCOPE_STEP_RETRY = 1,
  HALO_SCOPE_STEP_PLAN = 2,
  HALO_SCOPE_STEP_MEMORY = 3,
  HALO_SCOPE_STEP_CUSTOM = 4
} halo_scope_step_kind;

typedef enum { HALO_SCOPE_STATUS_OK = 0, HALO_SCOPE_STATUS_ERROR = 1 } halo_scope_status;

/* Output content kept: none, BLAKE3-128 digest and size (default), or full. */
typedef enum { HALO_SCOPE_CAPTURE_OFF = 0, HALO_SCOPE_CAPTURE_DIGEST = 1, HALO_SCOPE_CAPTURE_FULL = 2 } halo_scope_capture;

/* A score's subject: a trace, optionally one observation in it; or a
 * session (trace_id NULL). */
typedef struct {
  const char* trace_id;
  const char* observation_id;
  const char* session_id;
} halo_scope_target;

/* Ids written by the library: 16 hex digits and a NUL. */
#define HALO_SCOPE_ID_SIZE 17

/* Opens (creating and migrating) the halo-scope database at `data_path`,
 * with an optional write-ahead log (`wal_path`, NULL for none). libchdb is
 * loaded from $HALO_SCOPE_LIBCHDB or the build's default. chDB allows one
 * database per process: open one handle and share it. On failure returns
 * NULL and writes a message to `error` (if non-NULL, `error_size` bytes). */
halo_scope* halo_scope_open(const char* data_path, const char* project, halo_scope_capture capture,
                            const char* wal_path, char* error, size_t error_size);

/* Flushes everything recorded so far, stops the writer and frees the
 * handle. Calls must have returned on every thread first. */
void halo_scope_close(halo_scope* scope);

/* Returns once every record accepted before the call has been written or
 * counted as failed. */
void halo_scope_flush(halo_scope* scope);

/* Opens a step under `parent` (a span id, or NULL for the trace root);
 * writes its span id to `step_id`. */
int halo_scope_step_begin(halo_scope* scope, const char* trace_id, const char* parent, halo_scope_step_kind kind,
                          const char* name, char step_id[HALO_SCOPE_ID_SIZE]);
/* Closes a step opened by halo_scope_step_begin. */
int halo_scope_step_end(halo_scope* scope, const char* step_id, halo_scope_status status, const char* output,
                        size_t output_size);

/* A tool execution's result for the call the engine emitted as
 * `tool_call_id`; `latency_ns` is the execution time. */
int halo_scope_tool_result(halo_scope* scope, const char* trace_id, const char* tool_call_id, const char* tool_name,
                           halo_scope_status status, const char* output, size_t output_size, uint64_t latency_ns);

/* Score facts; `score_id` (may be NULL) receives the new fact's id.
 * `supersedes` (may be NULL) names a fact this one corrects. */
int halo_scope_score_numeric(halo_scope* scope, const halo_scope_target* target, const char* name, double value,
                             const char* source, const char* comment, const char* supersedes,
                             char score_id[HALO_SCOPE_ID_SIZE]);
int halo_scope_score_categorical(halo_scope* scope, const halo_scope_target* target, const char* name,
                                 const char* value, const char* source, const char* comment, const char* supersedes,
                                 char score_id[HALO_SCOPE_ID_SIZE]);
int halo_scope_score_boolean(halo_scope* scope, const halo_scope_target* target, const char* name, int value,
                             const char* source, const char* comment, const char* supersedes,
                             char score_id[HALO_SCOPE_ID_SIZE]);
/* Withdraws a score fact. */
int halo_scope_score_retract(halo_scope* scope, const char* score_id, const char* source, const char* comment);

/* Trace context: keys session_id, user_id, release, name and tag set those
 * fields (tag adds one tag); any other key is a metadata entry. */
int halo_scope_trace_meta(halo_scope* scope, const char* trace_id, const char* key, const char* value);

/* Write-path counters since open. */
typedef struct {
  uint64_t accepted;
  uint64_t dropped;
  uint64_t rows_written;
  uint64_t rows_failed;
} halo_scope_stats;
void halo_scope_get_stats(const halo_scope* scope, halo_scope_stats* stats);

/* ===================== Datasets, evaluators and experiments =====================
 *
 * Langfuse-style evaluation (PLAN.md P12) over halo-scope's SQLite metadata
 * database: versioned datasets that are immutable once frozen, versioned
 * evaluators, and experiment runs that are resumable and idempotent (a run
 * interrupted at any point continues where its durable checkpoints left
 * off and never counts an item twice). The same operations are served over
 * HTTP under /v1/scope/ by halo_serve --scope-dir.
 *
 * Return codes as above, plus HALO_SCOPE_INCOMPLETE (a run ended with items
 * still failed; run again to retry them). JSON arguments are UTF-8 JSON
 * texts; NULL means the default ("null" input/expected, "{}" metadata). */

typedef struct halo_scope_evals halo_scope_evals;

enum { HALO_SCOPE_INCOMPLETE = 2 };

/* Opens (creating and migrating) the metadata database at `metadata_path`.
 * `scope` (may be NULL; must outlive the handle) receives each experiment
 * score as an llm_scores fact on the item's trace, plus the run context. */
halo_scope_evals* halo_scope_evals_open(const char* metadata_path, halo_scope* scope, char* error, size_t error_size);
void halo_scope_evals_close(halo_scope_evals* evals);

int halo_scope_dataset_create(halo_scope_evals* evals, const char* name, const char* description,
                              const char* metadata_json, char* error, size_t error_size);
/* Adds or replaces (by id) an item in the dataset's draft, opening a draft
 * copied from the latest frozen version when there is none. With item_id
 * NULL a fresh id is generated and written to `new_id` (may be NULL). */
int halo_scope_dataset_add_item(halo_scope_evals* evals, const char* dataset, const char* item_id,
                                const char* input_json, const char* expected_json, const char* metadata_json,
                                char new_id[HALO_SCOPE_ID_SIZE], char* error, size_t error_size);
/* One item per line: {"id"?, "input", "expected_output"?, "metadata"?}; all
 * lines or none. `count` (may be NULL) receives the number of items. */
int halo_scope_dataset_import_jsonl(halo_scope_evals* evals, const char* dataset, const char* data, size_t size,
                                    size_t* count, char* error, size_t error_size);
/* Freezes the draft; writes its version and content hash (32 hex + NUL). */
int halo_scope_dataset_freeze(halo_scope_evals* evals, const char* dataset, int64_t* version,
                              char content_hash[33], char* error, size_t error_size);
int halo_scope_dataset_archive(halo_scope_evals* evals, const char* dataset, char* error, size_t error_size);

/* Appends an evaluator version. kind: exact, regex, json_schema, numeric,
 * code, llm_judge or decider (configs: src/scope/evaluators.hpp). */
int halo_scope_evaluator_create(halo_scope_evals* evals, const char* name, const char* kind, const char* config_json,
                                int64_t* version, char* error, size_t error_size);

/* Code evaluators: a `code` evaluator's {"hook": name} calls the function
 * registered under that name, which reports through one of the setters
 * (a result left unset reads as a failed evaluation). */
typedef struct halo_scope_eval_result halo_scope_eval_result;
typedef struct {
  const char* run_id;
  const char* item_id;
  int attempt;
  const char* input_json;
  const char* expected_json;
  const char* metadata_json;
  const char* output; /* the target's output */
  size_t output_size;
} halo_scope_eval_input;
typedef void (*halo_scope_eval_fn)(void* user, const halo_scope_eval_input* input, halo_scope_eval_result* result);
void halo_scope_eval_set_numeric(halo_scope_eval_result* result, double value, const char* comment);
void halo_scope_eval_set_boolean(halo_scope_eval_result* result, int value, const char* comment);
void halo_scope_eval_set_categorical(halo_scope_eval_result* result, const char* value, const char* comment);
/* A deterministic failure: recorded with the item, never averaged. */
void halo_scope_eval_set_failed(halo_scope_eval_result* result, const char* error);
/* A transient failure: the item is not committed and is retried later. */
void halo_scope_eval_set_retry(halo_scope_eval_result* result, const char* error);
/* `fn` may be called from several threads at once (run concurrency). */
int halo_scope_evals_register_hook(halo_scope_evals* evals, const char* name, halo_scope_eval_fn fn, void* user);

/* An in-process target (instead of an HTTP endpoint): receives the chat
 * completion request the runner built and reports the output. */
typedef struct halo_scope_target_reply halo_scope_target_reply;
typedef struct {
  const char* run_id;
  const char* item_id;
  const char* trace_id; /* record the generation under this trace */
  int attempt;
  const char* request_json; /* OpenAI chat completion body */
} halo_scope_target_call;
typedef void (*halo_scope_target_fn)(void* user, const halo_scope_target_call* call, halo_scope_target_reply* reply);
void halo_scope_target_set_output(halo_scope_target_reply* reply, const char* text, size_t text_size,
                                  const char* model, const char* finish_reason, uint64_t input_tokens,
                                  uint64_t output_tokens);
/* transient != 0: the item is retried on the next run; else it fails. */
void halo_scope_target_set_error(halo_scope_target_reply* reply, int transient, const char* error);

typedef struct {
  const char* name;            /* unique per dataset; the same name and binding resumes */
  const char* dataset;
  int64_t dataset_version;     /* 0: the latest frozen version */
  const char* target_json;     /* {"prompt"?: {"name", "version"|"label"}, "model", "variant"?, "params"?, "request"?, "endpoint"?} */
  const char* evaluators;      /* comma-separated name[@version] (latest when omitted) */
  int max_attempts;            /* 0: 3 */
  int concurrency;             /* 0: 1 */
  const char* endpoint;        /* OpenAI-compatible base URL (http://host:port) for the target and judges */
  const char* decider_endpoint; /* serves /v1/systemone; NULL: endpoint */
  halo_scope_target_fn target_fn; /* non-NULL: used instead of the endpoint for the target */
  void* target_user;
  void (*progress)(void* user, int64_t done, int64_t total, int64_t failed);
  void* progress_user;
} halo_scope_run_options;

/* Creates (or resumes) and runs an experiment to the end; writes the run id.
 * HALO_SCOPE_OK: every item done; HALO_SCOPE_INCOMPLETE: items failed (run
 * again to retry); HALO_SCOPE_INVALID: refused (see `error`). */
int halo_scope_run_experiment(halo_scope_evals* evals, const halo_scope_run_options* options,
                              char run_id[HALO_SCOPE_ID_SIZE], char* error, size_t error_size);

/* Paired comparison of score `score` between runs a and b, by item:
 * delta = b - a. Undefined statistics are NaN (se for n < 2). */
typedef struct {
  uint64_t n, only_a, only_b, wins, losses, ties;
  double mean_a, mean_b, mean_delta, se_delta;
} halo_scope_comparison;
int halo_scope_run_compare(halo_scope_evals* evals, const char* run_a, const char* run_b, const char* score,
                           halo_scope_comparison* out, char* error, size_t error_size);

/* The run, its progress and per-score summary as JSON. Writes at most
 * `size` bytes (NUL-terminated) and the full length to `needed`. */
int halo_scope_run_summary_json(halo_scope_evals* evals, const char* run_id, char* buffer, size_t size,
                                size_t* needed);

#ifdef __cplusplus
}
#endif

#endif /* HALO_SCOPE_H_ */
