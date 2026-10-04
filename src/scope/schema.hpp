#ifndef HALO_SCOPE_SCHEMA_HPP_
#define HALO_SCOPE_SCHEMA_HPP_

// The chDB schema of halo-scope as ordered, checksummed migrations into the
// `halo_scope` database. Tables follow casselc/observability's otap-rs
// llm_spans / llm_payloads / llm_scores DDL (see NOTICE), reshaped for one
// process owning its data: no cluster/namespace scope, no custody columns.
//
// Write pattern: producers insert RowBinary into `<table>_in` (ENGINE =
// Null, plain String/Map(String, ...) types); `<table>_mv` casts each block
// into the typed MergeTree `<table>`. Measured on libchdb 26.9.0, this is
// the cheapest way in: inserting RowBinary straight into the typed table
// costs ~2x per batch (LowCardinality parsing), and chDB's stream insert
// refuses INSERT ... SELECT FROM input(). Every MV target sets
// non_replicated_deduplication_window, so a WAL replay carrying the same
// insert_deduplication_token is written once.

#include <string>
#include <vector>

#include "scope/chdb.hpp"
#include "scope/migrate.hpp"

namespace halo::scope {

const std::vector<Migration>& ChdbMigrations();

/// Creates `halo_scope` and its migration table if missing, verifies the
/// applied migrations' checksums, and applies the missing ones in order.
/// Returns the number applied, or -1 with `*err` set.
int MigrateChdb(Conn& conn, std::string* err);

/// Same, against an explicit migration list (tests use edited lists).
int MigrateChdb(Conn& conn, const std::vector<Migration>& migrations, std::string* err);

}  // namespace halo::scope

#endif  // HALO_SCOPE_SCHEMA_HPP_
