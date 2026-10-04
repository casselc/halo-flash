// Adapted from oscope (github.com/chucklehead-dev/oscope, 3a14d2c,
// spike/oscope-core/src/chdb.rs), originally EPL-2.0; relicensed under MIT for
// halo-flash by its author, Chuck Cassel (2026-10-03). See NOTICE.
// Ported from a link-time binding to dlopen.

#include "scope/chdb.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <mutex>
#include <vector>

namespace halo::scope {

struct ChdbLib::Fns {
  using Conn = void*;  // chdb_connection (opaque)
  using Result = void;
  void (*set_signal_handlers_enabled)(int);
  const char* (*version)();
  Conn* (*connect)(int, char**);
  void (*close_conn)(Conn*);
  Result* (*query_n)(Conn, const char*, std::size_t, const char*, std::size_t);
  char* (*result_buffer)(Result*);
  std::size_t (*result_length)(Result*);
  const char* (*result_error)(Result*);
  std::uint64_t (*result_rows_written)(Result*);
  void (*destroy_query_result)(Result*);
  void* (*stream_insert_n)(Conn, const char*, std::size_t, const char*, std::size_t);
  int (*stream_append)(void*, const void*, std::size_t);
  Result* (*stream_done)(void*);
  const char* (*stream_insert_error)(void*);
  void (*destroy_insert_stream)(void*);
};

namespace {

std::mutex g_load_mu;
ChdbLib* g_lib = nullptr;  // leaked on purpose: chDB lives until process exit

template <typename F>
bool Sym(void* h, const char* name, F* out, std::string* err) {
  void* p = dlsym(h, name);
  if (!p) {
    *err = std::string("libchdb lacks symbol ") + name;
    return false;
  }
  *out = reinterpret_cast<F>(p);
  return true;
}

}  // namespace

const ChdbLib* ChdbLib::Load(std::string_view path, std::string* err) {
  std::lock_guard<std::mutex> lk(g_load_mu);
  if (g_lib) return g_lib;
  std::string p(path);
  if (p.empty()) {
    if (const char* e = std::getenv("HALO_SCOPE_LIBCHDB")) p = e;
  }
#ifdef HALO_SCOPE_LIBCHDB_DEFAULT
  if (p.empty()) p = HALO_SCOPE_LIBCHDB_DEFAULT;
#endif
  if (p.empty()) {
    *err = "no libchdb: set HALO_SCOPE_LIBCHDB or configure -DHALO_SCOPE_LIBCHDB=";
    return nullptr;
  }
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
  // Measured with libchdb 26.7.0: chDB's own threads crash inside TSan's
  // allocator on their first TLS access (__tls_get_addr -> malloc), before
  // any query runs. A TSan build records nothing rather than crash.
  if (!std::getenv("HALO_SCOPE_CHDB_UNDER_TSAN")) {
    *err = "libchdb is not usable under ThreadSanitizer (its threads crash in TSan's allocator); "
           "set HALO_SCOPE_CHDB_UNDER_TSAN=1 to try anyway";
    return nullptr;
  }
#endif
#endif
  // RTLD_LOCAL: chDB's bundled symbols must not interpose on the engine's.
  void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    *err = std::string("dlopen: ") + dlerror();
    return nullptr;
  }
  auto fns = std::make_unique<Fns>();
  bool ok = Sym(h, "chdb_set_signal_handlers_enabled", &fns->set_signal_handlers_enabled, err) &&
            Sym(h, "chdb_version", &fns->version, err) && Sym(h, "chdb_connect", &fns->connect, err) &&
            Sym(h, "chdb_close_conn", &fns->close_conn, err) && Sym(h, "chdb_query_n", &fns->query_n, err) &&
            Sym(h, "chdb_result_buffer", &fns->result_buffer, err) &&
            Sym(h, "chdb_result_length", &fns->result_length, err) &&
            Sym(h, "chdb_result_error", &fns->result_error, err) &&
            Sym(h, "chdb_result_rows_written", &fns->result_rows_written, err) &&
            Sym(h, "chdb_destroy_query_result", &fns->destroy_query_result, err) &&
            Sym(h, "chdb_stream_insert_n", &fns->stream_insert_n, err) &&
            Sym(h, "chdb_stream_append", &fns->stream_append, err) &&
            Sym(h, "chdb_stream_done", &fns->stream_done, err) &&
            Sym(h, "chdb_stream_insert_error", &fns->stream_insert_error, err) &&
            Sym(h, "chdb_destroy_insert_stream", &fns->destroy_insert_stream, err);
  if (!ok) {
    dlclose(h);
    return nullptr;
  }
  // Before the first connection: the engine owns its signals, not chDB.
  fns->set_signal_handlers_enabled(0);
  auto* lib = new ChdbLib();
  lib->path_ = std::move(p);
  lib->fns_ = std::move(fns);
  g_lib = lib;
  return g_lib;
}

std::string ChdbLib::version() const {
  const char* v = fns_->version();
  return v ? v : "";
}

namespace {

Conn::Result Finish(const ChdbLib::Fns& f, ChdbLib::Fns::Result* r) {
  Conn::Result out;
  if (!r) {
    out.error = "null result";
    return out;
  }
  const char* e = f.result_error(r);
  if (e && *e) {
    out.error = e;
  } else {
    out.ok = true;
    const std::size_t n = f.result_length(r);
    const char* b = f.result_buffer(r);
    if (n && b) out.data.assign(b, n);
    out.rows_written = f.result_rows_written(r);
  }
  f.destroy_query_result(r);
  return out;
}

}  // namespace

std::unique_ptr<Conn> Conn::Open(const ChdbLib& lib, std::string_view path, std::string* err) {
  std::vector<std::string> args = {"chdb"};
  if (!path.empty()) args.push_back("--path=" + std::string(path));
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  auto** h = lib.fns().connect(static_cast<int>(argv.size()), argv.data());
  if (!h) {
    *err = "chdb_connect failed";
    return nullptr;
  }
  std::unique_ptr<Conn> c(new Conn());
  c->lib_ = &lib;
  c->conn_ = h;
  return c;
}

Conn::~Conn() {
  if (conn_) lib_->fns().close_conn(static_cast<ChdbLib::Fns::Conn*>(conn_));
}

Conn::Result Conn::Query(std::string_view sql, std::string_view format) {
  const auto& f = lib_->fns();
  auto* r = f.query_n(*static_cast<ChdbLib::Fns::Conn*>(conn_), sql.data(), sql.size(), format.data(),
                      format.size());
  return Finish(f, r);
}

std::string Conn::Exec(std::string_view sql) {
  auto r = Query(sql);
  return r.ok ? std::string() : r.error;
}

Conn::Result Conn::StreamInsert(std::string_view insert, std::string_view format, std::string_view data) {
  const auto& f = lib_->fns();
  void* s = f.stream_insert_n(*static_cast<ChdbLib::Fns::Conn*>(conn_), insert.data(), insert.size(),
                              format.data(), format.size());
  Result out;
  if (const char* e = f.stream_insert_error(s); e && *e) {
    out.error = e;
    f.destroy_insert_stream(s);
    return out;
  }
  if (f.stream_append(s, data.data(), data.size()) != 0) {
    const char* e = f.stream_insert_error(s);
    out.error = (e && *e) ? e : "chdb_stream_append failed";
    f.destroy_insert_stream(s);
    return out;
  }
  out = Finish(f, f.stream_done(s));
  f.destroy_insert_stream(s);
  return out;
}

std::string SqlQuote(std::string_view s) {
  std::string o;
  o.reserve(s.size() + 2);
  o.push_back('\'');
  for (char c : s) {
    if (c == '\'' || c == '\\') o.push_back('\\');
    o.push_back(c);
  }
  o.push_back('\'');
  return o;
}

}  // namespace halo::scope
