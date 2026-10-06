#include "scope/sqlite_util.hpp"

namespace halo::scope::sql {

bool Exec(sqlite3* db, const char* sql, std::string* err) {
  char* msg = nullptr;
  if (sqlite3_exec(db, sql, nullptr, nullptr, &msg) != SQLITE_OK) {
    *err = msg ? msg : sqlite3_errmsg(db);
    sqlite3_free(msg);
    return false;
  }
  return true;
}

sqlite3* OpenMigrated(const std::string& path, const std::vector<Migration>& migrations, std::string* err) {
  sqlite3* db = nullptr;
  if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
                      nullptr) != SQLITE_OK) {
    *err = db ? sqlite3_errmsg(db) : "sqlite3_open_v2 failed";
    if (db) sqlite3_close_v2(db);
    return nullptr;
  }
  auto fail = [&]() -> sqlite3* {
    sqlite3_close_v2(db);
    return nullptr;
  };
  sqlite3_busy_timeout(db, 5000);
  if (!Exec(db, "PRAGMA journal_mode = WAL", err) || !Exec(db, "PRAGMA foreign_keys = ON", err) ||
      !Exec(db, "PRAGMA synchronous = NORMAL", err) ||
      !Exec(db,
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
            "checksum TEXT NOT NULL, applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))",
            err))
    return fail();
  // Plan and apply under one write lock, so two connections opening the same
  // file at once cannot both apply a migration.
  if (!Exec(db, "BEGIN IMMEDIATE", err)) return fail();
  std::vector<AppliedMigration> applied;
  {
    Stmt q(db, "SELECT version, name, checksum FROM schema_migrations ORDER BY version");
    while (q.Step() == SQLITE_ROW) applied.push_back({static_cast<std::uint32_t>(q.Int(0)), q.Text(1), q.Text(2)});
  }
  const int first = PlanMigrations(migrations, applied, err);
  std::string ignored;
  if (first < 0) return Exec(db, "ROLLBACK", &ignored), fail();
  for (std::size_t i = static_cast<std::size_t>(first); i < migrations.size(); ++i) {
    const auto& m = migrations[i];
    bool ok = true;
    for (const char* stmt : m.statements) ok = ok && Exec(db, stmt, err);
    if (ok) {
      Stmt ins(db, "INSERT INTO schema_migrations (version, name, checksum) VALUES (?, ?, ?)");
      ins.Bind(1, static_cast<std::int64_t>(m.version)).Bind(2, m.name).Bind(3, MigrationChecksum(m));
      ok = ins.Step() == SQLITE_DONE;
      if (!ok) *err = sqlite3_errmsg(db);
    }
    if (!ok) {
      *err = "migration " + std::to_string(m.version) + " (" + m.name + "): " + *err;
      Exec(db, "ROLLBACK", &ignored);
      return fail();
    }
  }
  if (!Exec(db, "COMMIT", err)) return fail();
  return db;
}

}  // namespace halo::scope::sql
