#ifndef HALO_SCOPE_EVALS_SCHEMA_HPP_
#define HALO_SCOPE_EVALS_SCHEMA_HPP_

// SQLite migrations for halo-scope's datasets, evaluators and experiment
// runs (PLAN.md P12), appended after the prompt and response migrations of
// metadata.cpp in the same metadata database.
//
// Placement (SQLite, not chDB): these are config-like, versioned records
// that must be immutable once frozen, and the run ledger is the experiment's
// checkpoint. Both need what chDB does not give: transactions spanning
// tables (an item's output and all its scores commit together or not at
// all), unique keys (one result per run and item, so a resumed run cannot
// count an item twice) and triggers that refuse writes to frozen rows.
// Their volume is bounded by dataset size times evaluators. The high-volume
// side (every target and judge generation, its payloads, and a mirror of
// each score as an llm_scores fact on the item's trace) goes to chDB.

#include <vector>

#include "scope/migrate.hpp"

namespace halo::scope {

/// Versions 3 (datasets), 4 (evaluators) and 5 (runs).
const std::vector<Migration>& EvalMigrations();

}  // namespace halo::scope

#endif  // HALO_SCOPE_EVALS_SCHEMA_HPP_
