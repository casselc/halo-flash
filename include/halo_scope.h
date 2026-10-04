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

#ifdef __cplusplus
}
#endif

#endif /* HALO_SCOPE_H_ */
