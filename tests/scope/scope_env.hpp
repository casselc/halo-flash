// Shared chDB state for the halo-scope tests. chDB allows one data path per
// process and should boot once, so each test binary boots one engine on a
// fresh temporary directory, migrates it once, keeps a reader and a writer
// connection open, and isolates test cases by a unique `project` value
// instead of by database. scope_main.cpp closes both connections before
// exit: libchdb 26.7.0 segfaults in its exit handlers when a disk-backed
// connection is still open at exit, and exits cleanly once all are closed.

#ifndef HALO_TESTS_SCOPE_ENV_HPP_
#define HALO_TESTS_SCOPE_ENV_HPP_

#include <gtest/gtest.h>
#include <stdlib.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <memory>
#include <mutex>
#include <string>

#include "scope/chdb.hpp"
#include "scope/schema.hpp"

namespace halo::scope::testing {

struct ChdbEnv {
  const ChdbLib* lib = nullptr;
  std::string path;
  std::unique_ptr<Conn> reader;
  std::mutex reader_mu;  // one statement at a time per connection
  std::unique_ptr<Conn> writer;  // test thread only
  std::string error;
};

inline ChdbEnv* g_env = nullptr;

/// Temporary directories made by this process. Shutdown() removes them
/// after chDB has stopped (each holds a database), unless
/// HALO_SCOPE_KEEP_TMP is set for debugging. Without this every run leaks
/// thousands of files into $TMPDIR.
inline std::mutex g_tmp_mu;
inline std::vector<std::string> g_tmp_dirs;

inline std::string TempDir(const char* prefix) {
  std::string tmpl = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/" + prefix + "-XXXXXX";
  if (!mkdtemp(tmpl.data())) throw std::runtime_error("mkdtemp failed");
  std::lock_guard<std::mutex> lk(g_tmp_mu);
  g_tmp_dirs.push_back(tmpl);
  return tmpl;
}

inline ChdbEnv& Env() {
  static ChdbEnv* env = g_env = [] {
    auto* e = new ChdbEnv();  // leaked: chDB lives until process exit
    e->lib = ChdbLib::Load("", &e->error);
    if (!e->lib) return e;
    try {
      e->path = TempDir("halo-scope");
    } catch (const std::exception& ex) {
      e->error = ex.what();
      e->lib = nullptr;
      return e;
    }
    e->reader = Conn::Open(*e->lib, e->path, &e->error);
    if (!e->reader || MigrateChdb(*e->reader, &e->error) < 0) e->lib = nullptr;
    if (e->lib) e->writer = Conn::Open(*e->lib, e->path, &e->error);
    if (!e->writer) e->lib = nullptr;
    return e;
  }();
  return *env;
}

/// Closes the shared connections (the last ones open) before exit.
inline void Shutdown() {
  if (g_env) {
    g_env->writer.reset();
    g_env->reader.reset();
    if (g_env->lib) g_env->lib->Shutdown();  // no chDB thread outlives main()
  }
  if (std::getenv("HALO_SCOPE_KEEP_TMP")) return;
  std::lock_guard<std::mutex> lk(g_tmp_mu);
  for (const auto& d : g_tmp_dirs) {
    std::error_code ec;
    std::filesystem::remove_all(d, ec);
  }
  g_tmp_dirs.clear();
}

/// Skips the calling test when libchdb is unavailable.
#define HALO_REQUIRE_CHDB()                                                              \
  do {                                                                                   \
    if (!::halo::scope::testing::Env().lib)                                              \
      GTEST_SKIP() << "libchdb unavailable: " << ::halo::scope::testing::Env().error;    \
  } while (0)

/// A project name no other test case in this process uses.
inline std::string FreshProject(const char* prefix) {
  static std::atomic<std::uint64_t> n{0};
  return std::string(prefix) + "-" + std::to_string(n.fetch_add(1));
}

/// Runs a read query on the shared reader connection.
inline Conn::Result Read(std::string_view sql, std::string_view format) {
  auto& e = Env();
  std::lock_guard<std::mutex> lk(e.reader_mu);
  return e.reader->Query(sql, format);
}

}  // namespace halo::scope::testing

#endif  // HALO_TESTS_SCOPE_ENV_HPP_
