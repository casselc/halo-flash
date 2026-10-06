#ifndef HALO_SCOPE_SQLITE_UTIL_HPP_
#define HALO_SCOPE_SQLITE_UTIL_HPP_

// Small SQLite helpers shared by the halo-scope metadata stores: a prepared
// statement wrapper, a connection opened in WAL mode and migrated with the
// ordered, checksummed migrations of migrate.hpp, and a transaction guard.

#include <sqlite3.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "scope/migrate.hpp"

namespace halo::scope::sql {

class Stmt {
 public:
  Stmt(sqlite3* db, const char* sql) : db_(db) { sqlite3_prepare_v2(db, sql, -1, &s_, nullptr); }
  ~Stmt() { sqlite3_finalize(s_); }
  Stmt(const Stmt&) = delete;
  Stmt& operator=(const Stmt&) = delete;
  bool ok() const { return s_ != nullptr; }
  Stmt& Bind(int i, std::string_view v) {
    sqlite3_bind_text(s_, i, v.data(), static_cast<int>(v.size()), SQLITE_TRANSIENT);
    return *this;
  }
  Stmt& Bind(int i, const char* v) { return Bind(i, std::string_view(v)); }
  Stmt& Bind(int i, const std::string& v) { return Bind(i, std::string_view(v)); }
  Stmt& Bind(int i, std::int64_t v) {
    sqlite3_bind_int64(s_, i, v);
    return *this;
  }
  Stmt& Bind(int i, int v) { return Bind(i, static_cast<std::int64_t>(v)); }
  Stmt& Bind(int i, double v) {
    sqlite3_bind_double(s_, i, v);
    return *this;
  }
  Stmt& BindNull(int i) {
    sqlite3_bind_null(s_, i);
    return *this;
  }
  /// SQLITE_ROW, SQLITE_DONE or an error code (prepare failures read as SQLITE_ERROR).
  int Step() { return s_ ? sqlite3_step(s_) : SQLITE_ERROR; }
  std::int64_t Int(int c) { return sqlite3_column_int64(s_, c); }
  double Double(int c) { return sqlite3_column_double(s_, c); }
  bool IsNull(int c) { return sqlite3_column_type(s_, c) == SQLITE_NULL; }
  std::string Text(int c) {
    const auto* p = reinterpret_cast<const char*>(sqlite3_column_text(s_, c));
    return p ? std::string(p, static_cast<std::size_t>(sqlite3_column_bytes(s_, c))) : std::string();
  }
  std::string Error() const { return sqlite3_errmsg(db_); }

 private:
  sqlite3* db_;
  sqlite3_stmt* s_ = nullptr;
};

/// Runs one or more statements; false with `*err` on failure.
bool Exec(sqlite3* db, const char* sql, std::string* err);

/// Opens (creating if missing) `path` with a full-mutex connection, WAL mode,
/// foreign keys on and a busy timeout, then applies `migrations` in order
/// (one transaction each) after checking the applied ones' checksums.
/// Returns nullptr with `*err` on failure.
sqlite3* OpenMigrated(const std::string& path, const std::vector<Migration>& migrations, std::string* err);

/// BEGIN IMMEDIATE ... COMMIT; rolls back unless Commit() succeeded.
class Tx {
 public:
  explicit Tx(sqlite3* db) : db_(db) {}
  bool Begin(std::string* err) { return began_ = Exec(db_, "BEGIN IMMEDIATE", err); }
  bool Commit(std::string* err) {
    if (!Exec(db_, "COMMIT", err)) return false;
    began_ = false;
    return true;
  }
  ~Tx() {
    if (began_) {
      std::string ignored;
      Exec(db_, "ROLLBACK", &ignored);
    }
  }

 private:
  sqlite3* db_;
  bool began_ = false;
};

}  // namespace halo::scope::sql

#endif  // HALO_SCOPE_SQLITE_UTIL_HPP_
