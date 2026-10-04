#ifndef HALO_SCOPE_MIGRATE_HPP_
#define HALO_SCOPE_MIGRATE_HPP_

// Ordered, checksummed migrations, shared by the chDB and SQLite stores.
// A store records (version, name, checksum) for each applied migration.
// On open: every recorded migration must match the compiled-in one byte for
// byte (same checksum), the database must not be newer than the binary, and
// the missing migrations are applied in order. A migration is recorded only
// after all its statements succeed; chDB has no DDL transactions, so every
// chDB statement is written to be idempotent (IF NOT EXISTS) and a migration
// interrupted part-way is simply re-run.

#include <cstdint>
#include <string>
#include <vector>

namespace halo::scope {

struct Migration {
  std::uint32_t version;
  const char* name;
  std::vector<const char*> statements;
};

/// FNV-1a 64 over name and statements (with separators), as 16 hex digits.
std::string MigrationChecksum(const Migration& m);

struct AppliedMigration {
  std::uint32_t version;
  std::string name;
  std::string checksum;
};

/// Checks `applied` against `known` (both ordered by version). Returns the
/// index into `known` of the first migration to apply, or -1 and sets
/// `*err` on a checksum mismatch, a gap, or an unknown applied version.
int PlanMigrations(const std::vector<Migration>& known, const std::vector<AppliedMigration>& applied,
                   std::string* err);

}  // namespace halo::scope

#endif  // HALO_SCOPE_MIGRATE_HPP_
