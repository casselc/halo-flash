#include "scope/evals_schema.hpp"

// HALO_SCOPE_MUTANT selects a deliberately broken variant for the positive
// controls in CMakeLists.txt; 0 is the real code.
#ifndef HALO_SCOPE_MUTANT
#define HALO_SCOPE_MUTANT 0
#endif

namespace halo::scope {

namespace {

#define HALO_NOW "(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"

// A dataset has numbered versions. At most one is a draft (frozen_at NULL)
// and takes edits; freezing records the content hash and item count, after
// which triggers refuse every change to the version and its items. A new
// draft starts as a copy of the latest frozen version. An archived dataset
// keeps its versions readable and refuses new drafts and items.
const Migration kDatasets{
    3,
    "datasets",
    {
        R"(CREATE TABLE datasets (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE CHECK (name <> ''),
  description TEXT NOT NULL DEFAULT '',
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT )" HALO_NOW R"(,
  archived_at TEXT
))",
        R"(CREATE TABLE dataset_versions (
  dataset_id INTEGER NOT NULL REFERENCES datasets(id),
  version INTEGER NOT NULL CHECK (version > 0),
  created_at TEXT NOT NULL DEFAULT )" HALO_NOW R"(,
  frozen_at TEXT,
  content_hash TEXT,
  item_count INTEGER,
  CHECK ((frozen_at IS NULL) = (content_hash IS NULL) AND (frozen_at IS NULL) = (item_count IS NULL)),
  PRIMARY KEY (dataset_id, version)
) WITHOUT ROWID)",
        R"(CREATE UNIQUE INDEX dataset_versions_one_draft ON dataset_versions (dataset_id) WHERE frozen_at IS NULL)",
        R"(CREATE TABLE dataset_items (
  dataset_id INTEGER NOT NULL,
  version INTEGER NOT NULL,
  item_id TEXT NOT NULL CHECK (item_id <> ''),
  ordinal INTEGER NOT NULL,
  input TEXT NOT NULL,
  expected_output TEXT NOT NULL DEFAULT 'null',
  metadata TEXT NOT NULL DEFAULT '{}',
  source_trace_id TEXT NOT NULL DEFAULT '',
  source_observation_id TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL,
  PRIMARY KEY (dataset_id, version, item_id),
  FOREIGN KEY (dataset_id, version) REFERENCES dataset_versions(dataset_id, version)
) WITHOUT ROWID)",
        R"(CREATE TRIGGER dataset_versions_frozen_update BEFORE UPDATE ON dataset_versions
WHEN OLD.frozen_at IS NOT NULL BEGIN SELECT RAISE(ABORT, 'dataset version is frozen'); END)",
        R"(CREATE TRIGGER dataset_versions_frozen_delete BEFORE DELETE ON dataset_versions
WHEN OLD.frozen_at IS NOT NULL BEGIN SELECT RAISE(ABORT, 'dataset version is frozen'); END)",
        R"(CREATE TRIGGER dataset_versions_archived_insert BEFORE INSERT ON dataset_versions
WHEN (SELECT archived_at FROM datasets WHERE id = NEW.dataset_id) IS NOT NULL
BEGIN SELECT RAISE(ABORT, 'dataset is archived'); END)",
        R"(CREATE TRIGGER dataset_items_frozen_insert BEFORE INSERT ON dataset_items
WHEN (SELECT frozen_at FROM dataset_versions WHERE dataset_id = NEW.dataset_id AND version = NEW.version) IS NOT NULL
  OR (SELECT archived_at FROM datasets WHERE id = NEW.dataset_id) IS NOT NULL
BEGIN SELECT RAISE(ABORT, 'dataset version is frozen or archived'); END)",
#if HALO_SCOPE_MUTANT == 12  // positive control: frozen items are editable
        "SELECT 1",
#else
        R"(CREATE TRIGGER dataset_items_frozen_update BEFORE UPDATE ON dataset_items
WHEN (SELECT frozen_at FROM dataset_versions WHERE dataset_id = OLD.dataset_id AND version = OLD.version) IS NOT NULL
  OR (SELECT frozen_at FROM dataset_versions WHERE dataset_id = NEW.dataset_id AND version = NEW.version) IS NOT NULL
  OR (SELECT archived_at FROM datasets WHERE id = NEW.dataset_id) IS NOT NULL
BEGIN SELECT RAISE(ABORT, 'dataset version is frozen or archived'); END)",
#endif
        R"(CREATE TRIGGER dataset_items_frozen_delete BEFORE DELETE ON dataset_items
WHEN (SELECT frozen_at FROM dataset_versions WHERE dataset_id = OLD.dataset_id AND version = OLD.version) IS NOT NULL
  OR (SELECT archived_at FROM datasets WHERE id = OLD.dataset_id) IS NOT NULL
BEGIN SELECT RAISE(ABORT, 'dataset version is frozen or archived'); END)",
    }};

// Evaluators are versioned like prompts: each version is an immutable
// (kind, config) fact. A run pins the versions it uses.
const Migration kEvaluators{
    4,
    "evaluators",
    {
        R"(CREATE TABLE evaluators (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE CHECK (name <> ''),
  created_at TEXT NOT NULL DEFAULT )" HALO_NOW R"(
))",
        R"(CREATE TABLE evaluator_versions (
  evaluator_id INTEGER NOT NULL REFERENCES evaluators(id),
  version INTEGER NOT NULL CHECK (version > 0),
  kind TEXT NOT NULL CHECK (kind IN ('exact', 'regex', 'json_schema', 'numeric', 'code', 'llm_judge', 'decider')),
  config TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT )" HALO_NOW R"(,
  PRIMARY KEY (evaluator_id, version)
) WITHOUT ROWID)",
        R"(CREATE TRIGGER evaluator_versions_immutable BEFORE UPDATE ON evaluator_versions
BEGIN SELECT RAISE(ABORT, 'evaluator versions are immutable'); END)",
        R"(CREATE TRIGGER evaluator_versions_undeletable BEFORE DELETE ON evaluator_versions
BEGIN SELECT RAISE(ABORT, 'evaluator versions are immutable'); END)",
    }};

// An experiment run: a frozen dataset version against a target (prompt
// version x model/variant x decoding parameters) with pinned evaluators.
// `binding` hashes everything a result depends on; a resumed run must
// recompute the same binding. run_items is the checkpoint ledger: one row
// per (run, item), 'running' while an attempt is in flight, 'done' once its
// output and every score committed in one transaction (then immutable), or
// 'error' after a failed attempt (retried on resume up to max_attempts).
const Migration kRuns{
    5,
    "runs",
    {
        R"(CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL CHECK (name <> ''),
  dataset_id INTEGER NOT NULL,
  dataset_version INTEGER NOT NULL,
  target TEXT NOT NULL,
  evaluators TEXT NOT NULL,
  binding TEXT NOT NULL,
  item_count INTEGER NOT NULL,
  max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT )" HALO_NOW R"(,
  updated_at TEXT NOT NULL DEFAULT )" HALO_NOW R"(,
  completed_at TEXT,
  UNIQUE (dataset_id, name),
  FOREIGN KEY (dataset_id, dataset_version) REFERENCES dataset_versions(dataset_id, version)
) WITHOUT ROWID)",
        R"(CREATE TRIGGER runs_need_frozen_version BEFORE INSERT ON runs
WHEN (SELECT frozen_at FROM dataset_versions WHERE dataset_id = NEW.dataset_id AND version = NEW.dataset_version) IS NULL
BEGIN SELECT RAISE(ABORT, 'runs need a frozen dataset version'); END)",
        R"(CREATE TRIGGER runs_binding_immutable BEFORE UPDATE OF id, dataset_id, dataset_version, target, evaluators,
binding, item_count ON runs BEGIN SELECT RAISE(ABORT, 'a run''s binding is immutable'); END)",
        R"(CREATE TABLE run_items (
  run_id TEXT NOT NULL REFERENCES runs(id),
  item_id TEXT NOT NULL,
  item_hash TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('running', 'done', 'error')),
  attempts INTEGER NOT NULL CHECK (attempts > 0),
  trace_id TEXT NOT NULL,
  observation_id TEXT NOT NULL DEFAULT '',
  output TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL DEFAULT '',
  finish_reason TEXT NOT NULL DEFAULT '',
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  latency_ms REAL NOT NULL DEFAULT 0,
  error TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL DEFAULT )" HALO_NOW R"(,
  PRIMARY KEY (run_id, item_id)
) WITHOUT ROWID)",
#if HALO_SCOPE_MUTANT == 11
        "SELECT 1",
#else
        R"(CREATE TRIGGER run_items_done_immutable BEFORE UPDATE ON run_items WHEN OLD.status = 'done'
BEGIN SELECT RAISE(ABORT, 'a done run item is immutable'); END)",
#endif
        R"(CREATE TRIGGER run_items_done_undeletable BEFORE DELETE ON run_items WHEN OLD.status = 'done'
BEGIN SELECT RAISE(ABORT, 'a done run item is immutable'); END)",
        R"(CREATE TABLE run_scores (
  run_id TEXT NOT NULL,
  item_id TEXT NOT NULL,
  name TEXT NOT NULL,
  evaluator TEXT NOT NULL,
  evaluator_version INTEGER NOT NULL,
  data_type TEXT NOT NULL CHECK (data_type IN ('NUMERIC', 'BOOLEAN', 'CATEGORICAL')),
  value REAL,
  string_value TEXT NOT NULL DEFAULT '',
  comment TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  score_id TEXT NOT NULL,
  judge_model TEXT NOT NULL DEFAULT '',
  judge_prompt TEXT NOT NULL DEFAULT '',
  judge_prompt_version INTEGER NOT NULL DEFAULT 0,
  judge_trace_id TEXT NOT NULL DEFAULT '',
  raw_output TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (run_id, item_id, name),
  FOREIGN KEY (run_id, item_id) REFERENCES run_items(run_id, item_id)
) WITHOUT ROWID)",
        R"(CREATE TRIGGER run_scores_only_while_running BEFORE INSERT ON run_scores
WHEN (SELECT status FROM run_items WHERE run_id = NEW.run_id AND item_id = NEW.item_id) IS NOT 'running'
BEGIN SELECT RAISE(ABORT, 'scores are written only with their item'); END)",
        R"(CREATE TRIGGER run_scores_immutable BEFORE UPDATE ON run_scores
BEGIN SELECT RAISE(ABORT, 'run scores are immutable'); END)",
        R"(CREATE TRIGGER run_scores_undeletable BEFORE DELETE ON run_scores
BEGIN SELECT RAISE(ABORT, 'run scores are immutable'); END)",
    }};

#undef HALO_NOW

}  // namespace

const std::vector<Migration>& EvalMigrations() {
  static const std::vector<Migration> m = {kDatasets, kEvaluators, kRuns};
  return m;
}

}  // namespace halo::scope
