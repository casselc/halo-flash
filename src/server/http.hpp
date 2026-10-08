#ifndef HALO_SERVER_HTTP_HPP_
#define HALO_SERVER_HTTP_HPP_

// A small HTTP/1.1 server for the OpenAI endpoints: an incremental request
// parser (bytes in any chunking give the same requests), keep-alive, bounded
// connections (one thread each; a few sessions, not thousands), fixed-length
// responses and streamed ones (chunked transfer encoding, for SSE), and
// cancellation when the client goes away: a handler can ask whether its
// client is still connected (HttpRequest::connected), a streamed body's
// writer fails once the client is gone, and a send that makes no progress
// for send_timeout_ms fails the same way (a client that stays connected but
// stops reading is treated as gone instead of holding its handler forever).
//
// Request bodies need Content-Length; chunked request bodies are refused
// (411), as are oversized heads (431) and bodies (413).

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace halo::server {

struct HttpRequest {
  std::string method;
  std::string target;  // as sent: path plus optional ?query
  std::string path;    // target without the query
  std::string query;
  std::vector<std::pair<std::string, std::string>> headers;  // names lowercased
  std::string body;
  bool keep_alive = true;
  /// Set by HttpServer: false once the client has closed or reset its end of
  /// the connection, or the server is stopping. A handler polls it to abandon
  /// work nobody will receive (a half-closed connection counts as gone).
  /// Empty for requests that do not come from a server connection.
  std::function<bool()> connected;

  /// The first header with this (lowercase) name, or null.
  [[nodiscard]] const std::string* Header(std::string_view lower_name) const;
};

struct HttpLimits {
  std::size_t max_head_bytes = 64 * 1024;
  std::size_t max_body_bytes = 32u << 20;
  std::size_t max_headers = 100;
};

/// A byte buffer for data that may hold secrets (request bodies carry spill keys): it never leaves a copy behind.
/// Growing copies into new storage and zeroes the old before freeing it; consuming a prefix moves the rest down and
/// zeroes the vacated tail; destruction zeroes everything.
class SecretBuffer {
 public:
  SecretBuffer() = default;
  ~SecretBuffer() { Wipe(); }
  SecretBuffer(const SecretBuffer&) = delete;
  SecretBuffer& operator=(const SecretBuffer&) = delete;
  void Append(std::string_view bytes);
  /// Drops the first `n` bytes.
  void Consume(std::size_t n);
  /// Zeroes and frees all storage.
  void Wipe() noexcept;
  [[nodiscard]] std::string_view view() const noexcept { return {p_.get(), n_}; }
  [[nodiscard]] std::size_t size() const noexcept { return n_; }
  /// Test seam: the whole storage, used and spare (everything beyond size() must be zero).
  [[nodiscard]] std::string_view storage_for_test() const noexcept { return {p_.get(), cap_}; }
  /// Test seam: called with each storage block just before it is freed (it must be all zero).
  static inline void (*on_free_for_test)(std::string_view) = nullptr;

 private:
  std::unique_ptr<char[]> p_;
  std::size_t n_ = 0, cap_ = 0;
};

/// Incremental HTTP/1.1 request parser. Feed bytes; complete requests come
/// out in order (pipelining works). After an error the connection must close.
class HttpParser {
 public:
  enum class Status : std::uint8_t { kNeedMore, kRequest, kError };
  explicit HttpParser(HttpLimits limits = {}) : limits_(limits) {}

  /// Appends bytes to the buffer.
  void Feed(std::string_view bytes) { buf_.Append(bytes); }
  /// Extracts the next complete request from the buffer, if any.
  Status Next(HttpRequest* out);
  /// For kError: the status code to answer with (400, 411, 413, 431, 505).
  [[nodiscard]] int error_status() const { return error_status_; }
  [[nodiscard]] const std::string& error() const { return error_; }
  [[nodiscard]] std::size_t buffered() const { return buf_.size(); }

 private:
  Status Fail(int status, std::string message);
  HttpLimits limits_;
  SecretBuffer buf_;  // zeroed when dropped: a partial request (a client that left mid-body) leaves nothing behind
  int error_status_ = 0;
  std::string error_;
};

/// Writes body bytes of a streamed response; returns false once the client
/// is gone (stop producing).
using BodyWriter = std::function<bool(std::string_view)>;

struct HttpResponse {
  int status = 200;
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;
  /// When set, the body is streamed (chunked) by calling this instead.
  std::function<void(const BodyWriter&)> stream;

  static HttpResponse Json(int status, std::string body);
  /// Server-sent events: `stream` writes `data: ...\n\n` frames.
  static HttpResponse EventStream(std::function<void(const BodyWriter&)> stream);
};

/// Serializes a fixed-length response head and body (also used by tests).
[[nodiscard]] std::string SerializeResponse(const HttpResponse& r, bool keep_alive);
[[nodiscard]] const char* ReasonPhrase(int status);

using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;

struct HttpServerOptions {
  std::string host = "127.0.0.1";
  int port = 0;  // 0: any free port (see port())
  std::size_t max_connections = 32;
  HttpLimits limits;
  int idle_timeout_ms = 60000;  // keep-alive connection with no request
  /// A write to a client that makes no progress for this long fails (the client is treated as gone): one client
  /// that stops reading cannot hold its handler, nor whatever the handler holds, for longer. For a streamed body
  /// "no progress" also means bytes already written that the kernel could not send (the client's receive window
  /// stayed closed), so a stalled client is noticed about this long after its window closes, not only once the
  /// server's send buffer is full too.
  int send_timeout_ms = 30000;
  std::string api_key;          // non-empty: require "Authorization: Bearer <key>"
  /// Non-empty: listen on this Unix socket instead of host:port. Its directory must exist, be ours and be 0700; the
  /// socket is 0600 and a peer whose uid (SO_PEERCRED) is not ours is closed at once (same-uid only). A live socket
  /// at the path (another server answers a connect) refuses the start; Stop() unlinks the path only while it is
  /// still this server's socket (same inode).
  std::string unix_path;
};

class HttpServer {
 public:
  explicit HttpServer(HttpServerOptions options);
  ~HttpServer();  // Stop()s
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  /// Exact-path routes; register before Start().
  void Route(std::string method, std::string path, HttpHandler handler);
  /// Paths starting with `prefix` that match no exact route (the longest
  /// matching prefix for the request's method wins).
  void RoutePrefix(std::string method, std::string prefix, HttpHandler handler);
  /// Binds, listens and starts the accept thread.
  [[nodiscard]] bool Start(std::string* error);
  /// Stops accepting, closes connections (streams see their writer fail),
  /// and joins every thread.
  void Stop();
  [[nodiscard]] int port() const { return port_; }
  [[nodiscard]] std::size_t active_connections() const { return active_.load(); }

 private:
  bool StartUnix(std::string* error);
  [[nodiscard]] bool PeerAllowed(int fd) const;
  void AcceptLoop();
  void Serve(int fd);
  HttpResponse Dispatch(const HttpRequest& req);

  HttpServerOptions opt_;
  std::map<std::pair<std::string, std::string>, HttpHandler> routes_;
  std::vector<std::pair<std::pair<std::string, std::string>, HttpHandler>> prefix_routes_;
  int listen_fd_ = -1;
  int port_ = 0;
  bool unix_ = false;
  std::uint64_t unix_ino_ = 0, unix_dev_ = 0;
  std::atomic<bool> stopping_{false};
  std::atomic<std::size_t> active_{0};
  std::thread acceptor_;
  std::mutex mu_;
  std::vector<int> fds_;  // open connections, guarded by mu_
};

}  // namespace halo::server

#endif  // HALO_SERVER_HTTP_HPP_
