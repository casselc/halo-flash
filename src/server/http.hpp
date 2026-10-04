#ifndef HALO_SERVER_HTTP_HPP_
#define HALO_SERVER_HTTP_HPP_

// A small HTTP/1.1 server for the OpenAI endpoints: an incremental request
// parser (bytes in any chunking give the same requests), keep-alive, bounded
// connections (one thread each; a few sessions, not thousands), fixed-length
// responses and streamed ones (chunked transfer encoding, for SSE), and
// cancellation when the client goes away.
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

  /// The first header with this (lowercase) name, or null.
  [[nodiscard]] const std::string* Header(std::string_view lower_name) const;
};

struct HttpLimits {
  std::size_t max_head_bytes = 64 * 1024;
  std::size_t max_body_bytes = 32u << 20;
  std::size_t max_headers = 100;
};

/// Incremental HTTP/1.1 request parser. Feed bytes; complete requests come
/// out in order (pipelining works). After an error the connection must close.
class HttpParser {
 public:
  enum class Status : std::uint8_t { kNeedMore, kRequest, kError };
  explicit HttpParser(HttpLimits limits = {}) : limits_(limits) {}

  /// Appends bytes to the buffer.
  void Feed(std::string_view bytes) { buf_.append(bytes); }
  /// Extracts the next complete request from the buffer, if any.
  Status Next(HttpRequest* out);
  /// For kError: the status code to answer with (400, 411, 413, 431, 505).
  [[nodiscard]] int error_status() const { return error_status_; }
  [[nodiscard]] const std::string& error() const { return error_; }
  [[nodiscard]] std::size_t buffered() const { return buf_.size(); }

 private:
  Status Fail(int status, std::string message);
  HttpLimits limits_;
  std::string buf_;
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
  std::string api_key;          // non-empty: require "Authorization: Bearer <key>"
};

class HttpServer {
 public:
  explicit HttpServer(HttpServerOptions options);
  ~HttpServer();  // Stop()s
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  /// Exact-path routes; register before Start().
  void Route(std::string method, std::string path, HttpHandler handler);
  /// Paths starting with `prefix` that match no exact route.
  void RoutePrefix(std::string method, std::string prefix, HttpHandler handler);
  /// Binds, listens and starts the accept thread.
  [[nodiscard]] bool Start(std::string* error);
  /// Stops accepting, closes connections (streams see their writer fail),
  /// and joins every thread.
  void Stop();
  [[nodiscard]] int port() const { return port_; }
  [[nodiscard]] std::size_t active_connections() const { return active_.load(); }

 private:
  void AcceptLoop();
  void Serve(int fd);
  HttpResponse Dispatch(const HttpRequest& req);

  HttpServerOptions opt_;
  std::map<std::pair<std::string, std::string>, HttpHandler> routes_;
  std::vector<std::pair<std::pair<std::string, std::string>, HttpHandler>> prefix_routes_;
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> stopping_{false};
  std::atomic<std::size_t> active_{0};
  std::thread acceptor_;
  std::mutex mu_;
  std::vector<int> fds_;  // open connections, guarded by mu_
};

}  // namespace halo::server

#endif  // HALO_SERVER_HTTP_HPP_
