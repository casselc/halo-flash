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

inline ChdbEnv& Env() {
  static ChdbEnv* env = g_env = [] {
    auto* e = new ChdbEnv();  // leaked: chDB lives until process exit
    e->lib = ChdbLib::Load("", &e->error);
    if (!e->lib) return e;
    std::string tmpl = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/halo-scope-XXXXXX";
    if (!mkdtemp(tmpl.data())) {
      e->error = "mkdtemp failed";
      e->lib = nullptr;
      return e;
    }
    e->path = tmpl;
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
  if (!g_env) return;  // chDB never booted
  g_env->writer.reset();
  g_env->reader.reset();
  if (g_env->lib) g_env->lib->Shutdown();  // no chDB thread outlives main()
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
