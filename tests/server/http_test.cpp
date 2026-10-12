// HTTP/1.1 layer (server/http):
//   - parser: pipelined requests, fed in any chunking, parse back to what
//     was sent; malformed heads get the documented status; random bytes
//     never crash it;
//   - server over loopback: keep-alive, chunked SSE streams, routing errors,
//     API keys, concurrent clients, a client leaving mid-stream (the writer
//     reports it), a client that stops reading (cut off after the send
//     timeout, so others sharing its lock proceed), the connection limit,
//     and Stop() with idle connections;
//   - the operation contract's stamp: every answer (handled, refused, a parse error, the connection limit) names the
//     incarnation and its clock, and GET /v1/halo/contract agrees with them.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>

#include <hegel/gtest.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "core/json.hpp"
#include "server/contract.hpp"
#include "server/http.hpp"
#include "server/incarnation.hpp"

namespace gs = hegel::generators;
using namespace halo::server;

namespace {

struct Sent {
  std::string method, target, body;
  std::vector<std::pair<std::string, std::string>> headers;
  bool close = false;
};

std::string Serialize(const Sent& s) {
  std::string out = s.method + " " + s.target + " HTTP/1.1\r\n";
  for (const auto& [k, v] : s.headers) out += k + ":" + " " + v + "\r\n";
  if (s.close) out += "Connection: close\r\n";
  if (!s.body.empty() || s.method == "POST") out += "Content-Length: " + std::to_string(s.body.size()) + "\r\n";
  return out + "\r\n" + s.body;
}

}  // namespace

TEST(Http, PipelinedRequestsParseBackInAnyChunking) {
  hegel::test([](hegel::TestCase& tc) {
    std::vector<Sent> sent(tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 4})));
    std::string wire;
    for (auto& s : sent) {
      s.method = tc.draw(gs::sampled_from<std::string>({"GET", "POST", "DELETE"}));
      s.target = tc.draw(gs::sampled_from<std::string>({"/", "/v1/responses", "/v1/models?x=1&y=%20", "/a/b"}));
      if (s.method == "POST") {
        const auto b = tc.draw(gs::binary({.max_size = 300}));
        s.body.assign(b.begin(), b.end());
      }
      const auto nh = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 4}));
      for (int i = 0; i < nh; ++i)
        s.headers.emplace_back(tc.draw(gs::sampled_from<std::string>({"X-Trace", "Accept", "Authorization", "x-a"})),
                               tc.draw(gs::sampled_from<std::string>({"", "v", "a b", "text/event-stream", "Bearer k"})));
      s.close = &s == &sent.back() && tc.draw(gs::booleans());
      wire += Serialize(s);
    }
    HttpParser p;
    std::vector<HttpRequest> got;
    std::size_t i = 0;
    while (i < wire.size()) {
      const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 64}));
      p.Feed(std::string_view(wire).substr(i, n));
      i += n;
      HttpRequest r;
      for (HttpParser::Status st; (st = p.Next(&r)) == HttpParser::Status::kRequest;) got.push_back(std::move(r));
      ASSERT_NE(p.error_status(), 400) << p.error();
    }
    ASSERT_EQ(got.size(), sent.size());
    ASSERT_EQ(p.buffered(), 0u);
    for (std::size_t k = 0; k < sent.size(); ++k) {
      const auto& s = sent[k];
      const auto& r = got[k];
      ASSERT_EQ(r.method, s.method);
      ASSERT_EQ(r.target, s.target);
      ASSERT_EQ(r.path + (r.query.empty() && r.target.find('?') == std::string::npos ? "" : "?" + r.query), s.target);
      ASSERT_EQ(r.body, s.body);
      ASSERT_EQ(r.keep_alive, !s.close);
      for (std::size_t h = 0; h < s.headers.size(); ++h) {
        std::string lower = s.headers[h].first;
        for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        ASSERT_EQ(r.headers[h].first, lower);
        ASSERT_EQ(r.headers[h].second, s.headers[h].second);
      }
    }
  }, {.test_cases = 400});
}

TEST(Http, MalformedHeadsGetTheirStatus) {
  struct Case {
    std::string wire;
    int status;
  };
  const std::string big(70 * 1024, 'a');
  for (const auto& c : std::vector<Case>{
           {"GET /\r\n\r\n", 400},
           {"GET  / HTTP/1.1\r\n\r\n", 400},
           {"G@T / HTTP/1.1\r\n\r\n", 400},
           {"GET / HTTP/2.0\r\n\r\n", 505},
           {"GET / FTP/1.1\r\n\r\n", 400},
           {"GET / HTTP/1.1\r\nBad Name: x\r\n\r\n", 400},
           {"GET / HTTP/1.1\r\nNoColon\r\n\r\n", 400},
           {"GET / HTTP/1.1\r\nA: b\r\n folded\r\n\r\n", 400},
           {"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", 411},
           {"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 3\r\n\r\nabc", 400},
           {"POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nab", 400},
           {"POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n", 400},
           {"POST / HTTP/1.1\r\nContent-Length: 99999999999\r\n\r\n", 413},
           {"GET / HTTP/1.1\r\nX: " + big + "\r\n\r\n", 431},
           {"GET /" + big, 431},
       }) {
    HttpParser p;
    p.Feed(c.wire);
    HttpRequest r;
    EXPECT_EQ(p.Next(&r), HttpParser::Status::kError) << c.wire.substr(0, 60);
    EXPECT_EQ(p.error_status(), c.status) << c.wire.substr(0, 60) << " -> " << p.error();
  }
  // HTTP/1.0 closes by default; leading blank lines are ignored.
  HttpParser p;
  p.Feed("\r\n\r\nGET / HTTP/1.0\r\n\r\n");
  HttpRequest r;
  ASSERT_EQ(p.Next(&r), HttpParser::Status::kRequest);
  EXPECT_FALSE(r.keep_alive);
}

TEST(Http, ArbitraryBytesNeverCrashTheParser) {
  hegel::test([](hegel::TestCase& tc) {
    HttpParser p({.max_head_bytes = 512, .max_body_bytes = 256, .max_headers = 8});
    const auto n = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 8}));
    for (int i = 0; i < n; ++i) {
      const bool structured = tc.draw(gs::booleans());
      std::string chunk;
      if (structured) {
        chunk = tc.draw(gs::sampled_from<std::string>({"GET / HTTP/1.1\r\n", "Content-Length: 5\r\n", "\r\n", "hello",
                                                        ":\r\n", "POST /x HTTP/1.0\r\n", " \t"}));
      } else {
        const auto b = tc.draw(gs::binary({.max_size = 200}));
        chunk.assign(b.begin(), b.end());
      }
      p.Feed(chunk);
      HttpRequest r;
      while (p.Next(&r) == HttpParser::Status::kRequest) {
      }
    }
  }, {.test_cases = 1000});
}

namespace {

int Connect(int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

struct Reply {
  int status = 0;
  std::string head, body;
};

// Reads one response (Content-Length or chunked) from `fd`, with `extra`
// holding bytes already read past the previous response.
bool ReadReply(int fd, std::string& extra, Reply* out) {
  const auto fill = [&] {
    char b[4096];
    const ssize_t n = ::recv(fd, b, sizeof b, 0);
    if (n <= 0) return false;
    extra.append(b, static_cast<std::size_t>(n));
    return true;
  };
  std::size_t end;
  while ((end = extra.find("\r\n\r\n")) == std::string::npos)
    if (!fill()) return false;
  out->head = extra.substr(0, end);
  extra.erase(0, end + 4);
  out->status = std::stoi(out->head.substr(9, 3));
  if (out->head.find("Transfer-Encoding: chunked") != std::string::npos) {
    out->body.clear();
    for (;;) {
      std::size_t eol;
      while ((eol = extra.find("\r\n")) == std::string::npos)
        if (!fill()) return false;
      const std::size_t n = std::stoul(extra.substr(0, eol), nullptr, 16);
      while (extra.size() < eol + 2 + n + 2)
        if (!fill()) return false;
      out->body += extra.substr(eol + 2, n);
      extra.erase(0, eol + 2 + n + 2);
      if (n == 0) return true;
    }
  }
  const auto cl = out->head.find("Content-Length: ");
  const std::size_t n = std::stoul(out->head.substr(cl + 16));
  while (extra.size() < n)
    if (!fill()) return false;
  out->body = extra.substr(0, n);
  extra.erase(0, n);
  return true;
}

Reply Call(int port, const std::string& request) {
  const int fd = Connect(port);
  Reply r;
  std::string extra;
  if (fd >= 0 && ::send(fd, request.data(), request.size(), MSG_NOSIGNAL) == static_cast<ssize_t>(request.size()))
    ReadReply(fd, extra, &r);
  if (fd >= 0) ::close(fd);
  return r;
}

struct TestServer {
  explicit TestServer(HttpServerOptions o = {}) : server(std::move(o)) {
    server.Route("GET", "/ping", [](const HttpRequest&) { return HttpResponse::Json(200, R"({"ok": true})"); });
    server.Route("POST", "/echo", [](const HttpRequest& r) { return HttpResponse::Json(200, r.body); });
    server.Route("GET", "/throw", [](const HttpRequest&) -> HttpResponse { throw std::runtime_error("boom"); });
    server.RoutePrefix("POST", "/p/", [](const HttpRequest&) { return HttpResponse::Json(200, "post"); });
    server.RoutePrefix("GET", "/p/", [](const HttpRequest&) { return HttpResponse::Json(200, "shallow"); });
    server.RoutePrefix("GET", "/p/deep/", [](const HttpRequest&) { return HttpResponse::Json(200, "deep"); });
    server.Route("GET", "/sse", [this](const HttpRequest& r) {
      const int n = r.query.empty() ? 5 : std::stoi(r.query);
      return HttpResponse::EventStream([this, n](const BodyWriter& w) {
        for (int i = 0; i < n; ++i) {
          if (!w("data: " + std::to_string(i) + "\n\n")) {
            stream_stopped = true;
            return;
          }
          if (n > 100) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        (void)w("data: [DONE]\n\n");
      });
    });
    // One "generator" shared by every request (the engine's lock): /flood streams 64 KB frames while holding it,
    // /locked answers once it gets it.
    server.Route("GET", "/flood", [this](const HttpRequest&) {
      return HttpResponse::EventStream([this](const BodyWriter& w) {
        std::lock_guard<std::mutex> lk(engine);
        const std::string frame = "data: " + std::string(64 * 1024, 'x') + "\n\n";
        for (int i = 0; i < 100000; ++i)
          if (!w(frame)) {
            flood_stopped = true;
            return;
          }
      });
    });
    // Small frames at a token-like rate: the kernel would buffer them for a long time before a send blocks.
    server.Route("GET", "/trickle", [this](const HttpRequest&) {
      return HttpResponse::EventStream([this](const BodyWriter& w) {
        std::lock_guard<std::mutex> lk(engine);
        const std::string frame = "data: " + std::string(200, 'y') + "\n\n";
        for (int i = 0; i < 5000; ++i) {
          if (!w(frame)) {
            trickle_stopped = true;
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
      });
    });
    server.Route("GET", "/locked", [this](const HttpRequest&) {
      std::lock_guard<std::mutex> lk(engine);
      return HttpResponse::Json(200, R"({"ok": true})");
    });
    std::string err;
    if (!server.Start(&err)) throw std::runtime_error(err);
  }
  HttpServer server;
  std::atomic<bool> stream_stopped{false};
  std::mutex engine;
  std::atomic<bool> flood_stopped{false};
  std::atomic<bool> trickle_stopped{false};
};

}  // namespace

TEST(Http, ServesKeepAliveStreamsAndErrors) {
  TestServer t;
  const int port = t.server.port();
  // Two requests on one connection, then SSE on the same connection.
  const int fd = Connect(port);
  ASSERT_GE(fd, 0);
  const std::string reqs = "GET /ping HTTP/1.1\r\n\r\nPOST /echo HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello"
                           "GET /sse?3 HTTP/1.1\r\n\r\n";
  ASSERT_EQ(::send(fd, reqs.data(), reqs.size(), 0), static_cast<ssize_t>(reqs.size()));
  std::string extra;
  Reply r;
  ASSERT_TRUE(ReadReply(fd, extra, &r));
  EXPECT_EQ(r.status, 200);
  EXPECT_EQ(r.body, R"({"ok": true})");
  EXPECT_NE(r.head.find("Connection: keep-alive"), std::string::npos);
  ASSERT_TRUE(ReadReply(fd, extra, &r));
  EXPECT_EQ(r.body, "hello");
  ASSERT_TRUE(ReadReply(fd, extra, &r));
  EXPECT_NE(r.head.find("Content-Type: text/event-stream"), std::string::npos);
  EXPECT_EQ(r.body, "data: 0\n\ndata: 1\n\ndata: 2\n\ndata: [DONE]\n\n");
  ::close(fd);
  EXPECT_EQ(Call(port, "GET /nope HTTP/1.1\r\n\r\n").status, 404);
  // Prefix routes: the longest prefix for the method; other methods only: 405.
  EXPECT_EQ(Call(port, "GET /p/deep/x HTTP/1.1\r\n\r\n").body, "deep");
  EXPECT_EQ(Call(port, "GET /p/x HTTP/1.1\r\n\r\n").body, "shallow");
  EXPECT_EQ(Call(port, "POST /p/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n").body, "post");
  EXPECT_EQ(Call(port, "DELETE /p/x HTTP/1.1\r\n\r\n").status, 405);
  EXPECT_EQ(Call(port, "POST /ping HTTP/1.1\r\nContent-Length: 0\r\n\r\n").status, 405);
  EXPECT_EQ(Call(port, "GET /throw HTTP/1.1\r\n\r\n").status, 500);
  EXPECT_EQ(Call(port, "GET / HTTP/3.0\r\n\r\n").status, 505);
  const auto closing = Call(port, "GET /ping HTTP/1.1\r\nConnection: close\r\n\r\n");
  EXPECT_NE(closing.head.find("Connection: close"), std::string::npos);
}

TEST(Http, ApiKeyIsRequiredWhenSet) {
  TestServer t({.api_key = "sekret"});
  EXPECT_EQ(Call(t.server.port(), "GET /ping HTTP/1.1\r\n\r\n").status, 401);
  EXPECT_EQ(Call(t.server.port(), "GET /ping HTTP/1.1\r\nAuthorization: Bearer nope\r\n\r\n").status, 401);
  EXPECT_EQ(Call(t.server.port(), "GET /ping HTTP/1.1\r\nAuthorization: Bearer sekret\r\n\r\n").status, 200);
}

TEST(Http, ConcurrentClientsAndADepartingStreamReader) {
  TestServer t;
  const int port = t.server.port();
  std::atomic<int> ok{0};
  std::vector<std::thread> clients;
  for (int c = 0; c < 8; ++c)
    clients.emplace_back([&, c] {
      for (int i = 0; i < 20; ++i) {
        const std::string body = "client " + std::to_string(c) + " #" + std::to_string(i);
        const auto r = Call(port, "POST /echo HTTP/1.1\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body);
        ok += r.status == 200 && r.body == body;
      }
    });
  for (auto& th : clients) th.join();
  EXPECT_EQ(ok.load(), 160);
  // A reader that leaves mid-stream: the handler's writer must report it.
  const int fd = Connect(port);
  const std::string req = "GET /sse?100000 HTTP/1.1\r\n\r\n";
  ASSERT_EQ(::send(fd, req.data(), req.size(), 0), static_cast<ssize_t>(req.size()));
  char b[256];
  ASSERT_GT(::recv(fd, b, sizeof b, 0), 0);
  ::close(fd);
  for (int i = 0; i < 500 && !t.stream_stopped.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_TRUE(t.stream_stopped.load());
}

TEST(Http, ConnectionLimitAndStopWithIdleConnections) {
  TestServer t({.max_connections = 2});
  const int a = Connect(t.server.port()), b = Connect(t.server.port());
  ASSERT_GE(a, 0);
  ASSERT_GE(b, 0);
  for (int i = 0; i < 200 && t.server.active_connections() < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(Call(t.server.port(), "GET /ping HTTP/1.1\r\n\r\n").status, 503);
  const auto start = std::chrono::steady_clock::now();
  t.server.Stop();  // a and b are idle: Stop must not wait for their timeout
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
  EXPECT_EQ(t.server.active_connections(), 0u);
  ::close(a);
  ::close(b);
}

namespace {

/// A client that requests `path`, reads the response head and then never reads again; a second client then asks
/// for /locked (which needs the lock the stalled stream holds). Returns how long the second client waited.
std::chrono::steady_clock::duration StallThenAsk(int port, const std::string& path, bool* answered) {
  const int stalled = ::socket(AF_INET, SOCK_STREAM, 0);
  const int small = 4096;
  ::setsockopt(stalled, SOL_SOCKET, SO_RCVBUF, &small, sizeof small);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  EXPECT_EQ(::connect(stalled, reinterpret_cast<sockaddr*>(&a), sizeof a), 0);
  const std::string req = "GET " + path + " HTTP/1.1\r\n\r\n";
  EXPECT_EQ(::send(stalled, req.data(), req.size(), 0), static_cast<ssize_t>(req.size()));
  char b[256];
  EXPECT_GT(::recv(stalled, b, sizeof b, 0), 0);  // the head arrived; now it never reads again
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const auto start = std::chrono::steady_clock::now();
  const int fd = Connect(port);
  const timeval tv{5, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  const std::string q = "GET /locked HTTP/1.1\r\n\r\n";
  EXPECT_EQ(::send(fd, q.data(), q.size(), 0), static_cast<ssize_t>(q.size()));
  std::string extra;
  Reply r;
  *answered = ReadReply(fd, extra, &r) && r.status == 200;
  const auto waited = std::chrono::steady_clock::now() - start;
  ::close(fd);
  ::close(stalled);  // unblocks a server that never gave up on it, so the test ends either way
  return waited;
}

}  // namespace

// A client that stays connected but stops reading must not hold the server (audit F4): the stalled stream's writer
// fails once a send makes no progress for send_timeout_ms (the client-gone path), so a second client waiting on the
// same lock is answered within about that bound.
TEST(Http, AStalledReaderIsCutOffAfterTheSendTimeout) {
  TestServer t({.send_timeout_ms = 300});
  bool answered = false;
  const auto waited = StallThenAsk(t.server.port(), "/flood", &answered);
  EXPECT_TRUE(answered) << "the second client got no answer within 5 s";
  EXPECT_LT(waited, std::chrono::seconds(3));
  for (int i = 0; i < 500 && !t.flood_stopped.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_TRUE(t.flood_stopped.load());
}

// The same with small frames at a token-like rate: no send blocks for a long time (the kernel would buffer about a
// megabyte first), but the bytes it cannot send to the closed window make no progress, which is noticed.
TEST(Http, ASlowStalledReaderIsCutOffToo) {
  TestServer t({.send_timeout_ms = 300});
  bool answered = false;
  const auto waited = StallThenAsk(t.server.port(), "/trickle", &answered);
  EXPECT_TRUE(answered) << "the second client got no answer within 5 s";
  EXPECT_LT(waited, std::chrono::seconds(3));
  for (int i = 0; i < 500 && !t.trickle_stopped.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_TRUE(t.trickle_stopped.load());
}

// The admin Unix socket: served in a 0700 directory of ours as a 0600 socket; a directory others can enter is
// refused; the socket file goes with Stop().
TEST(Http, UnixSocketIsPrivate) {
  const std::string dir = "/tmp/halo-http-unix-" + std::to_string(::getpid());
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  std::filesystem::permissions(dir, std::filesystem::perms::owner_all);
  HttpServerOptions o;
  o.unix_path = dir + "/admin.sock";
  {
    HttpServer s(o);
    s.Route("GET", "/x", [](const HttpRequest&) { return HttpResponse::Json(200, "{\"ok\":true}"); });
    std::string err;
    ASSERT_TRUE(s.Start(&err)) << err;
    struct stat st {};
    ASSERT_EQ(::stat(o.unix_path.c_str(), &st), 0);
    EXPECT_TRUE(S_ISSOCK(st.st_mode));
    EXPECT_EQ(st.st_mode & 0777, 0600u);
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    std::memcpy(a.sun_path, o.unix_path.c_str(), o.unix_path.size() + 1);
    ASSERT_EQ(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a), 0);
    const std::string req = "GET /x HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n";
    ASSERT_EQ(::send(fd, req.data(), req.size(), 0), static_cast<ssize_t>(req.size()));
    std::string extra;
    Reply r;
    ASSERT_TRUE(ReadReply(fd, extra, &r));
    EXPECT_EQ(r.status, 200);
    ::close(fd);
    s.Stop();
    EXPECT_FALSE(std::filesystem::exists(o.unix_path)) << "the socket file stayed";
  }
  std::filesystem::permissions(dir, std::filesystem::perms::owner_all | std::filesystem::perms::group_read |
                                        std::filesystem::perms::group_exec);
  HttpServer loose(o);
  std::string err;
  EXPECT_FALSE(loose.Start(&err)) << "a socket in a group-readable directory";
  std::filesystem::remove_all(dir);
}

// Secret request bytes never stay behind: consuming a pipelined request zeroes the tail the move vacated, growing
// zeroes the old storage before freeing it, and dropping the parser (a client gone mid-body) zeroes everything.
TEST(Http, SecretBufferLeavesNoCopies) {
  static std::vector<std::string> freed;
  freed.clear();
  SecretBuffer::on_free_for_test = [](std::string_view block) { freed.emplace_back(block); };
  const auto clean_tail = [](const SecretBuffer& b) {
    const auto st = b.storage_for_test();
    return std::all_of(st.begin() + static_cast<std::ptrdiff_t>(b.size()), st.end(), [](char c) { return c == 0; });
  };
  {
    HttpParser p;
    const std::string a = "POST /a HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello";
    const std::string b = "POST /admin/keys/s HTTP/1.1\r\nContent-Length: 12\r\n\r\nSECRETKEY123";
    p.Feed(a + b + "POST /c HTTP/1.1\r\nContent-Length: 20\r\n\r\nPARTIALSECR");  // a third, cut off mid-body
    HttpRequest r;
    ASSERT_EQ(p.Next(&r), HttpParser::Status::kRequest);
    ASSERT_EQ(p.Next(&r), HttpParser::Status::kRequest);
    EXPECT_EQ(r.body, "SECRETKEY123");
    EXPECT_EQ(p.Next(&r), HttpParser::Status::kNeedMore);
    std::string big(20000, 'Z');  // grows the storage: the old block must be freed zeroed
    SecretBuffer s;
    s.Append("SECRETKEY123");
    s.Append(big);
    s.Consume(12);
    EXPECT_TRUE(clean_tail(s)) << "a consumed prefix left a copy in the spare storage";
  }
  SecretBuffer::on_free_for_test = nullptr;
  ASSERT_FALSE(freed.empty());
  for (const auto& block : freed)
    EXPECT_TRUE(std::all_of(block.begin(), block.end(), [](char c) { return c == 0; })) << "a block was freed unwiped";
}

// The admin socket is never taken from a live server, a stale one is replaced, and a server only unlinks its own.
TEST(Http, UnixSocketOwnership) {
  const std::string dir = (std::filesystem::temp_directory_path() / ("halo-http-own-" + std::to_string(::getpid()))).string();
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  std::filesystem::permissions(dir, std::filesystem::perms::owner_all);
  HttpServerOptions o;
  o.unix_path = dir + "/admin.sock";
  std::string err;
  auto a = std::make_unique<HttpServer>(o);
  ASSERT_TRUE(a->Start(&err)) << err;
  HttpServer b(o);
  EXPECT_FALSE(b.Start(&err)) << "a second server took a live socket";
  EXPECT_NE(err.find("live"), std::string::npos) << err;
  // A successor at the path (the first one's file removed): the first's Stop must not unlink it.
  std::filesystem::remove(o.unix_path);
  HttpServer c(o);
  ASSERT_TRUE(c.Start(&err)) << err;
  a->Stop();
  a.reset();
  EXPECT_TRUE(std::filesystem::exists(o.unix_path)) << "a server unlinked its successor's socket";
  c.Stop();
  EXPECT_FALSE(std::filesystem::exists(o.unix_path));
  // A stale socket (nobody listening) is replaced.
  {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un sa{};
    sa.sun_family = AF_UNIX;
    std::memcpy(sa.sun_path, o.unix_path.c_str(), o.unix_path.size() + 1);
    ASSERT_EQ(::bind(fd, reinterpret_cast<sockaddr*>(&sa), sizeof sa), 0);
    ::close(fd);
  }
  HttpServer d(o);
  EXPECT_TRUE(d.Start(&err)) << "a stale socket was not replaced: " << err;
  d.Stop();
  std::filesystem::remove_all(dir);
}

// The socket probe fails closed: an endpoint that cannot be classified (a full backlog: EAGAIN) is not "stale".
TEST(Http, UnixSocketProbeFailsClosed) {
  const std::string dir = (std::filesystem::temp_directory_path() / ("halo-http-probe-" + std::to_string(::getpid()))).string();
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  std::filesystem::permissions(dir, std::filesystem::perms::owner_all);
  const std::string path = dir + "/admin.sock";
  const int lfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  sockaddr_un sa{};
  sa.sun_family = AF_UNIX;
  std::memcpy(sa.sun_path, path.c_str(), path.size() + 1);
  ASSERT_EQ(::bind(lfd, reinterpret_cast<sockaddr*>(&sa), sizeof sa), 0);
  ASSERT_EQ(::listen(lfd, 0), 0);
  std::vector<int> fill;  // never accepted: the backlog fills and further connects get EAGAIN
  for (int i = 0; i < 64; ++i) {
    const int c = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
    const int rc = ::connect(c, reinterpret_cast<sockaddr*>(&sa), sizeof sa);
    fill.push_back(c);
    if (rc != 0 && errno == EAGAIN) break;
  }
  HttpServerOptions o;
  o.unix_path = path;
  HttpServer s(o);
  std::string err;
  EXPECT_FALSE(s.Start(&err)) << "an endpoint with a full backlog was taken for stale";
  EXPECT_TRUE(std::filesystem::exists(path));
  for (int c : fill) ::close(c);
  ::close(lfd);
  std::filesystem::remove_all(dir);
}

// An over-long socket path is refused before anything is copied into sockaddr_un.
TEST(Http, UnixSocketPathLengthIsBounded) {
  HttpServerOptions o;
  o.unix_path = (std::filesystem::temp_directory_path() / std::string(200, 'x')).string();
  HttpServer s(o);
  std::string err;
  EXPECT_FALSE(s.Start(&err));
  EXPECT_NE(err.find("too long"), std::string::npos) << err;
}

namespace {

/// The value of header `name` in a response head, or "" when absent.
std::string HeaderOf(const std::string& head, const std::string& name) {
  const std::string key = "\r\n" + name + ": ";
  const auto at = head.find(key);
  if (at == std::string::npos) return "";
  const auto from = at + key.size();
  return head.substr(from, head.find("\r\n", from) - from);
}

bool IsHex32(const std::string& s) {
  return s.size() == 32 && std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

}  // namespace

TEST(Http, IncarnationsAreRandomAndTheirClocksNeverRunBackward) {
  const Incarnation a, b;
  EXPECT_TRUE(IsHex32(a.id())) << a.id();
  EXPECT_NE(a.id(), b.id());
  std::int64_t last = a.NowMs();
  for (int i = 0; i < 1000; ++i) {
    const std::int64_t now = a.NowMs();
    EXPECT_GE(now, last);
    last = now;
  }
  const Incarnation old("00112233445566778899aabbccddeeff", Incarnation::Clock::now() - std::chrono::seconds(5));
  EXPECT_GE(old.NowMs(), 5000);
}

// Every answer the server writes names the incarnation and its clock, whichever path wrote it: a handler, the API
// key check, routing, a parse error, a stream's head, the connection limit (docs/OPERATIONS.md section 2).
TEST(Http, EveryAnswerNamesTheIncarnationAndItsClock) {
  const Incarnation inc("00112233445566778899aabbccddeeff", Incarnation::Clock::now() - std::chrono::seconds(5));
  TestServer t({.max_connections = 2,
                .api_key = "k",
                .stamp = [&inc](std::vector<std::pair<std::string, std::string>>& h) { inc.Stamp(h); }});
  const int port = t.server.port();
  std::int64_t last = 0;
  const auto check = [&](const Reply& r, int status, const char* what) {
    EXPECT_EQ(r.status, status) << what;
    EXPECT_EQ(HeaderOf(r.head, "Halo-Incarnation"), inc.id()) << what;
    const std::string time = HeaderOf(r.head, "Halo-Time");
    ASSERT_FALSE(time.empty()) << what;
    const std::int64_t ms = std::stoll(time);
    EXPECT_GE(ms, 5000) << what;
    EXPECT_GE(ms, last) << what;
    last = ms;
  };
  const std::string auth = "Authorization: Bearer k\r\n";
  check(Call(port, "GET /ping HTTP/1.1\r\n" + auth + "\r\n"), 200, "handled");
  check(Call(port, "GET /ping HTTP/1.1\r\n\r\n"), 401, "refused at the API key");
  check(Call(port, "GET /nope HTTP/1.1\r\n" + auth + "\r\n"), 404, "no route");
  check(Call(port, "GET /throw HTTP/1.1\r\n" + auth + "\r\n"), 500, "a handler that threw");
  check(Call(port, "GET / HTTP/3.0\r\n\r\n"), 505, "a parse error");
  check(Call(port, "GET /sse?2 HTTP/1.1\r\n" + auth + "\r\n"), 200, "a stream's head");
  const int a = Connect(port), b = Connect(port);
  ASSERT_GE(a, 0);
  ASSERT_GE(b, 0);
  for (int i = 0; i < 200 && t.server.active_connections() < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  check(Call(port, "GET /ping HTTP/1.1\r\n" + auth + "\r\n"), 503, "the connection limit");
  ::close(a);
  ::close(b);
}

// GET /v1/halo/contract: the incarnation and clock the answer's headers carry, the inference mode, and no operation
// verbs yet (stage 1).
TEST(Http, TheContractNamesTheIncarnationItsClockAndTheInferenceMode) {
  const Incarnation inc("00112233445566778899aabbccddeeff", Incarnation::Clock::now() - std::chrono::seconds(5));
  HttpServer server({.stamp = [&inc](std::vector<std::pair<std::string, std::string>>& h) { inc.Stamp(h); }});
  RegisterContract(server, inc, {.relaxed = true});
  std::string err;
  ASSERT_TRUE(server.Start(&err)) << err;
  const Reply r = Call(server.port(), "GET /v1/halo/contract HTTP/1.1\r\n\r\n");
  ASSERT_EQ(r.status, 200);
  const auto doc = halo::json::Parse(r.body, &err);
  ASSERT_TRUE(doc) << err;
  EXPECT_EQ(doc->find("incarnation")->str(), inc.id());
  EXPECT_EQ(HeaderOf(r.head, "Halo-Incarnation"), inc.id());
  const auto time_ms = static_cast<std::int64_t>(doc->find("time_ms")->number());
  EXPECT_GE(time_ms, 5000);
  EXPECT_LE(time_ms, std::stoll(HeaderOf(r.head, "Halo-Time")));  // the document is built before the head is sent
  EXPECT_EQ(doc->find("inference")->find("mode")->str(), "relaxed");
  EXPECT_TRUE(doc->find("operations")->is(halo::json::Value::Kind::kNull));
  const auto exact = halo::json::Parse(ContractJson(inc, {.relaxed = false}), &err);
  ASSERT_TRUE(exact) << err;
  EXPECT_EQ(exact->find("inference")->find("mode")->str(), "exact");
  server.Stop();
}
