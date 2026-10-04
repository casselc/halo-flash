#ifndef HALO_SCOPE_CHDB_HPP_
#define HALO_SCOPE_CHDB_HPP_

// libchdb loaded at run time with dlopen. libchdb is ~566 MB, so halo-scope
// never links it: an engine built with HALO_SCOPE runs without it and only
// records when the library is found. The binding follows oscope-core's
// src/chdb.rs (oscope, spike/oscope-core): connection, binary-safe query,
// and streaming insert. See NOTICE.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace halo::scope {

/// The resolved libchdb entry points. Loaded once per process; never
/// unloaded (chDB cannot shut down cleanly with MergeTree tables, and the
/// engine is meant to start once per process).
class ChdbLib {
 public:
  /// Loads libchdb from `path`, or from $HALO_SCOPE_LIBCHDB when `path` is
  /// empty, then the compile-time default (HALO_SCOPE_LIBCHDB_DEFAULT) if
  /// one was configured. Disables chDB's signal handlers before any
  /// connection exists. Thread-safe; the first successful call wins and
  /// later calls return the same library. Returns nullptr and sets `*err`
  /// on failure.
  static const ChdbLib* Load(std::string_view path, std::string* err);

  /// chdb_version() of the loaded library.
  std::string version() const;
  const std::string& path() const { return path_; }

  struct Fns;
  const Fns& fns() const { return *fns_; }

 private:
  std::string path_;
  std::unique_ptr<Fns> fns_;
};

/// One chDB connection. A connection runs one statement at a time, so a
/// Conn must not be used from two threads at once; open one per thread
/// (connections to the same path share the engine and its data).
class Conn {
 public:
  /// `path` is the data directory (created if missing); empty means memory.
  static std::unique_ptr<Conn> Open(const ChdbLib& lib, std::string_view path, std::string* err);
  ~Conn();
  Conn(const Conn&) = delete;
  Conn& operator=(const Conn&) = delete;

  struct Result {
    bool ok = false;
    std::string data;   // output bytes in the requested format
    std::string error;  // set when !ok
    std::uint64_t rows_written = 0;
  };

  /// Binary-safe query (chdb_query_n).
  Result Query(std::string_view sql, std::string_view format = "TabSeparated");
  /// Runs a statement and returns its error, empty on success.
  std::string Exec(std::string_view sql);
  /// Streaming insert (chdb_stream_insert_n + one chdb_stream_append):
  /// `insert` is an INSERT without FORMAT, `data` the raw format bytes.
  /// Never put RowBinary inline after FORMAT in Query(): chDB's parser
  /// skips whitespace bytes (0x09 0x0A 0x0D 0x20) after FORMAT and
  /// misaligns the first row.
  Result StreamInsert(std::string_view insert, std::string_view format, std::string_view data);

 private:
  Conn() = default;
  const ChdbLib* lib_ = nullptr;
  void* conn_ = nullptr;  // chdb_connection*
};

/// Escapes a value as a single-quoted ClickHouse string literal.
std::string SqlQuote(std::string_view s);

}  // namespace halo::scope

#endif  // HALO_SCOPE_CHDB_HPP_
