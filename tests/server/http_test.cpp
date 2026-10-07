// HTTP/1.1 layer (server/http):
//   - parser: pipelined requests, fed in any chunking, parse back to what
//     was sent; malformed heads get the documented status; random bytes
//     never crash it;
//   - server over loopback: keep-alive, chunked SSE streams, routing errors,
//     API keys, concurrent clients, a client leaving mid-stream (the writer
//     reports it), a client that stops reading (cut off after the send
//     timeout, so others sharing its lock proceed), the connection limit,
//     and Stop() with idle connections.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <hegel/gtest.h>

#include <atomic>
#include <mutex>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "server/http.hpp"

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

// A client that stays connected but stops reading must not hold the server (audit F4): the stalled stream's writer
// fails once a send makes no progress for send_timeout_ms (the client-gone path), so a second client waiting on the
// same lock is answered within about that bound.
TEST(Http, AStalledReaderIsCutOffAfterTheSendTimeout) {
  TestServer t({.send_timeout_ms = 300});
  const int port = t.server.port();
  const int stalled = ::socket(AF_INET, SOCK_STREAM, 0);
  const int small = 4096;
  ::setsockopt(stalled, SOL_SOCKET, SO_RCVBUF, &small, sizeof small);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  ASSERT_EQ(::connect(stalled, reinterpret_cast<sockaddr*>(&a), sizeof a), 0);
  const std::string req = "GET /flood HTTP/1.1\r\n\r\n";
  ASSERT_EQ(::send(stalled, req.data(), req.size(), 0), static_cast<ssize_t>(req.size()));
  char b[256];
  ASSERT_GT(::recv(stalled, b, sizeof b, 0), 0);  // the head arrived; now it never reads again
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const auto start = std::chrono::steady_clock::now();
  const int fd = Connect(port);
  const timeval tv{5, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  const std::string q = "GET /locked HTTP/1.1\r\n\r\n";
  ASSERT_EQ(::send(fd, q.data(), q.size(), 0), static_cast<ssize_t>(q.size()));
  std::string extra;
  Reply r;
  const bool answered = ReadReply(fd, extra, &r);
  const auto waited = std::chrono::steady_clock::now() - start;
  ::close(fd);
  ::close(stalled);  // unblocks a server without a deadline, so the test ends either way
  EXPECT_TRUE(answered) << "the second client got no answer within 5 s";
  EXPECT_EQ(r.status, 200);
  EXPECT_LT(waited, std::chrono::seconds(3));
  for (int i = 0; i < 500 && !t.flood_stopped.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_TRUE(t.flood_stopped.load());
}
