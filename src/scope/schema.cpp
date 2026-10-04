// Table shapes adapted from casselc/observability otel-chdb/otap-rs/sql/
// llm_spans.sql, llm_payloads.sql and llm_scores.sql. See NOTICE.

#include "scope/schema.hpp"

#include <sstream>

namespace halo::scope {

namespace {

// Staging and typed tables share column names; the MV selects by name and
// ClickHouse casts each column to the target's type.

const Migration kTraces{
    1,
    "llm_traces",
    {
        R"(CREATE TABLE IF NOT EXISTS halo_scope.llm_traces
(
    `project` LowCardinality(String),
    `trace_id` String,
    `name` String,
    `user_id` String,
    `session_id` String,
    `tags` Array(String),
    `release` LowCardinality(String),
    `start_time` DateTime64(9, 'UTC'),
    `end_time` DateTime64(9, 'UTC'),
    `status_code` LowCardinality(String),
    `metadata` Map(LowCardinality(String), String),
    INDEX idx_trace_id trace_id TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_session_id session_id TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_user_id user_id TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(start_time)
ORDER BY (project, toStartOfMinute(start_time), xxHash32(trace_id))
SETTINGS non_replicated_deduplication_window = 1000)",
        R"(CREATE TABLE IF NOT EXISTS halo_scope.llm_traces_in
(
    `project` String,
    `trace_id` String,
    `name` String,
    `user_id` String,
    `session_id` String,
    `tags` Array(String),
    `release` String,
    `start_time` DateTime64(9, 'UTC'),
    `end_time` DateTime64(9, 'UTC'),
    `status_code` String,
    `metadata` Map(String, String)
)
ENGINE = Null)",
        R"(CREATE MATERIALIZED VIEW IF NOT EXISTS halo_scope.llm_traces_mv TO halo_scope.llm_traces AS
SELECT * FROM halo_scope.llm_traces_in)",
    }};

const Migration kObservations{
    2,
    "llm_observations",
    {
        R"(CREATE TABLE IF NOT EXISTS halo_scope.llm_observations
(
    `project` LowCardinality(String),
    `trace_id` String,
    `span_id` String,
    `parent_span_id` String,
    `start_time` DateTime64(9, 'UTC'),
    `end_time` DateTime64(9, 'UTC'),
    `completion_start_time` DateTime64(9, 'UTC'),
    `type` LowCardinality(String),
    `name` String,
    `user_id` String,
    `session_id` String,
    `tags` Array(String),
    `status_code` LowCardinality(String),
    `level` LowCardinality(String),
    `provider` LowCardinality(String),
    `model` LowCardinality(String),
    `variant_id` LowCardinality(String),
    `adapter_id` LowCardinality(String),
    `input_tokens` UInt64,
    `output_tokens` UInt64,
    `cached_tokens` UInt64,
    `usage_details` Map(LowCardinality(String), UInt64),
    `prompt_name` String,
    `prompt_version` UInt32,
    `tool_name` LowCardinality(String),
    `tool_call_ids` Array(String),
    `system_ref` Array(FixedString(16)),
    `input_refs` Array(FixedString(16)),
    `output_refs` Array(FixedString(16)),
    `input_bytes` UInt64,
    `output_bytes` UInt64,
    `truncated` UInt8,
    `tpot_p50_us` UInt32,
    `tpot_p90_us` UInt32,
    `tpot_p99_us` UInt32,
    `mtp_drafted` UInt32,
    `mtp_accepted` UInt32,
    `temperature` Float32,
    `top_p` Float32,
    `min_p` Float32,
    `top_k` UInt32,
    `seed` UInt64,
    `max_tokens` UInt32,
    `finish_reason` LowCardinality(String),
    `decision_answer` String,
    INDEX idx_trace_id trace_id TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_session_id session_id TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_user_id user_id TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(start_time)
ORDER BY (project, toStartOfMinute(start_time), xxHash32(trace_id), span_id)
SETTINGS non_replicated_deduplication_window = 1000)",
        R"(CREATE TABLE IF NOT EXISTS halo_scope.llm_observations_in
(
    `project` String,
    `trace_id` String,
    `span_id` String,
    `parent_span_id` String,
    `start_time` DateTime64(9, 'UTC'),
    `end_time` DateTime64(9, 'UTC'),
    `completion_start_time` DateTime64(9, 'UTC'),
    `type` String,
    `name` String,
    `user_id` String,
    `session_id` String,
    `tags` Array(String),
    `status_code` String,
    `level` String,
    `provider` String,
    `model` String,
    `variant_id` String,
    `adapter_id` String,
    `input_tokens` UInt64,
    `output_tokens` UInt64,
    `cached_tokens` UInt64,
    `usage_details` Map(String, UInt64),
    `prompt_name` String,
    `prompt_version` UInt32,
    `tool_name` String,
    `tool_call_ids` Array(String),
    `system_ref` Array(FixedString(16)),
    `input_refs` Array(FixedString(16)),
    `output_refs` Array(FixedString(16)),
    `input_bytes` UInt64,
    `output_bytes` UInt64,
    `truncated` UInt8,
    `tpot_p50_us` UInt32,
    `tpot_p90_us` UInt32,
    `tpot_p99_us` UInt32,
    `mtp_drafted` UInt32,
    `mtp_accepted` UInt32,
    `temperature` Float32,
    `top_p` Float32,
    `min_p` Float32,
    `top_k` UInt32,
    `seed` UInt64,
    `max_tokens` UInt32,
    `finish_reason` String,
    `decision_answer` String
)
ENGINE = Null)",
        R"(CREATE MATERIALIZED VIEW IF NOT EXISTS halo_scope.llm_observations_mv TO halo_scope.llm_observations AS
SELECT * FROM halo_scope.llm_observations_in)",
    }};

// Content-addressed: any copy of a hash is its content, so no read needs
// FINAL (LIMIT 1 BY hash); ReplacingMergeTree only reclaims space on merge.
const Migration kPayloads{
    3,
    "llm_payloads",
    {
        R"(CREATE TABLE IF NOT EXISTS halo_scope.llm_payloads
(
    `project` LowCardinality(String),
    `hash` FixedString(16),
    `bytes` UInt32,
    `content` String CODEC(ZSTD(3)),
    `ingested_at` DateTime64(3, 'UTC') DEFAULT now64(3) CODEC(Delta(8), ZSTD(1))
)
ENGINE = ReplacingMergeTree
ORDER BY (project, hash)
SETTINGS non_replicated_deduplication_window = 1000, index_granularity = 1024)",
        R"(CREATE TABLE IF NOT EXISTS halo_scope.llm_payloads_in
(
    `project` String,
    `hash` FixedString(16),
    `bytes` UInt32,
    `content` String
)
ENGINE = Null)",
        R"(CREATE MATERIALIZED VIEW IF NOT EXISTS halo_scope.llm_payloads_mv TO halo_scope.llm_payloads AS
SELECT project, hash, bytes, content FROM halo_scope.llm_payloads_in)",
    }};

// Append-only facts: a correction is a later fact naming the one it
// supersedes; a deletion is a `retract` fact. Resolved at query time.
const Migration kScores{
    4,
    "llm_scores",
    {
        R"(CREATE TABLE IF NOT EXISTS halo_scope.llm_scores
(
    `project` LowCardinality(String),
    `score_id` String,
    `supersedes` String,
    `kind` LowCardinality(String),
    `trace_id` String,
    `observation_id` String,
    `session_id` String,
    `name` LowCardinality(String),
    `source` LowCardinality(String),
    `data_type` LowCardinality(String),
    `value` Float64,
    `string_value` String,
    `comment` String,
    `event_time` DateTime64(9, 'UTC'),
    INDEX idx_trace_id trace_id TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (project, name, toStartOfHour(event_time), score_id)
SETTINGS non_replicated_deduplication_window = 1000)",
        R"(CREATE TABLE IF NOT EXISTS halo_scope.llm_scores_in
(
    `project` String,
    `score_id` String,
    `supersedes` String,
    `kind` String,
    `trace_id` String,
    `observation_id` String,
    `session_id` String,
    `name` String,
    `source` String,
    `data_type` String,
    `value` Float64,
    `string_value` String,
    `comment` String,
    `event_time` DateTime64(9, 'UTC')
)
ENGINE = Null)",
        R"(CREATE MATERIALIZED VIEW IF NOT EXISTS halo_scope.llm_scores_mv TO halo_scope.llm_scores AS
SELECT * FROM halo_scope.llm_scores_in)",
    }};

std::vector<AppliedMigration> ParseApplied(const std::string& tsv) {
  std::vector<AppliedMigration> out;
  std::istringstream in(tsv);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto t1 = line.find('\t');
    const auto t2 = line.find('\t', t1 + 1);
    out.push_back({static_cast<std::uint32_t>(std::stoul(line.substr(0, t1))), line.substr(t1 + 1, t2 - t1 - 1),
                   line.substr(t2 + 1)});
  }
  return out;
}

}  // namespace

const std::vector<Migration>& ChdbMigrations() {
  static const std::vector<Migration> m = {kTraces, kObservations, kPayloads, kScores};
  return m;
}

int MigrateChdb(Conn& conn, std::string* err) { return MigrateChdb(conn, ChdbMigrations(), err); }

int MigrateChdb(Conn& conn, const std::vector<Migration>& migrations, std::string* err) {
  for (const char* s : {"CREATE DATABASE IF NOT EXISTS halo_scope",
                        "CREATE TABLE IF NOT EXISTS halo_scope.schema_migrations (`version` UInt32, `name` String, "
                        "`checksum` String, `applied_at` DateTime64(3, 'UTC') DEFAULT now64(3)) "
                        "ENGINE = MergeTree ORDER BY version"}) {
    if (auto e = conn.Exec(s); !e.empty()) {
      *err = e;
      return -1;
    }
  }
  auto r = conn.Query("SELECT version, name, checksum FROM halo_scope.schema_migrations ORDER BY version",
                      "TabSeparated");
  if (!r.ok) {
    *err = r.error;
    return -1;
  }
  // Names and checksums are plain ASCII, so TabSeparated needs no unescaping.
  const auto applied = ParseApplied(r.data);
  const int first = PlanMigrations(migrations, applied, err);
  if (first < 0) return -1;
  int n = 0;
  for (std::size_t i = static_cast<std::size_t>(first); i < migrations.size(); ++i) {
    const auto& m = migrations[i];
    for (const char* s : m.statements) {
      if (auto e = conn.Exec(s); !e.empty()) {
        *err = "migration " + std::to_string(m.version) + " (" + m.name + "): " + e;
        return -1;
      }
    }
    const auto rec = "INSERT INTO halo_scope.schema_migrations (version, name, checksum) VALUES (" +
                     std::to_string(m.version) + ", " + SqlQuote(m.name) + ", " + SqlQuote(MigrationChecksum(m)) +
                     ")";
    if (auto e = conn.Exec(rec); !e.empty()) {
      *err = e;
      return -1;
    }
    ++n;
  }
  return n;
}

}  // namespace halo::scope
