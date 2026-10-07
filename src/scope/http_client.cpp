#include "scope/http_client.hpp"

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace halo::scope {

namespace {

struct Fd {
  int fd = -1;
  ~Fd() {
    if (fd >= 0) ::close(fd);
  }
};

bool ParseUrl(const std::string& url, std::string* host, std::string* port, std::string* base, std::string* err) {
  constexpr std::string_view kScheme = "http://";
  if (url.compare(0, kScheme.size(), kScheme) != 0) return *err = "only http:// URLs are supported: " + url, false;
  const std::string rest = url.substr(kScheme.size());
  const auto slash = rest.find('/');
  const std::string hostport = rest.substr(0, slash);
  *base = slash == std::string::npos ? "" : rest.substr(slash);
  while (!base->empty() && base->back() == '/') base->pop_back();
  const auto colon = hostport.rfind(':');
  if (colon == std::string::npos) {
    *host = hostport;
    *port = "80";
  } else {
    *host = hostport.substr(0, colon);
    *port = hostport.substr(colon + 1);
  }
  if (host->empty()) return *err = "no host in " + url, false;
  return true;
}

bool WaitFor(int fd, short events, int timeout_ms) {
  pollfd p{fd, events, 0};
  int rc;
  do rc = ::poll(&p, 1, timeout_ms);
  while (rc < 0 && errno == EINTR);
  return rc > 0;
}

// connect() bounded by `timeout_ms` (a blocking connect waits out the kernel's SYN retries, about 2 min): a
// non-blocking connect, a wait for writability, then SO_ERROR. The socket is left blocking. False with errno set.
bool Connect(int fd, const sockaddr* addr, socklen_t len, int timeout_ms) {
  const int flags = ::fcntl(fd, F_GETFL);
  if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return false;
  if (::connect(fd, addr, len) != 0) {
    if (errno != EINPROGRESS) return false;
    if (!WaitFor(fd, POLLOUT, timeout_ms)) return errno = ETIMEDOUT, false;
    int err = 0;
    socklen_t n = sizeof err;
    if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &n) < 0) return false;
    if (err != 0) return errno = err, false;
  }
  return ::fcntl(fd, F_SETFL, flags) == 0;
}

// Decodes a chunked body; false when incomplete or malformed.
bool Dechunk(std::string_view in, std::string* out) {
  out->clear();
  std::size_t i = 0;
  while (true) {
    const auto eol = in.find("\r\n", i);
    if (eol == std::string_view::npos) return false;
    const std::string size_line(in.substr(i, eol - i));
    char* end = nullptr;
    const unsigned long n = std::strtoul(size_line.c_str(), &end, 16);
    if (end == size_line.c_str()) return false;
    i = eol + 2;
    if (n == 0) return true;
    if (in.size() < i + n + 2) return false;
    out->append(in.substr(i, n));
    i += n + 2;
  }
}

}  // namespace

HttpResult HttpRequestJson(const std::string& method, const std::string& url, const std::string& path,
                           const std::string& body, const HttpHeaders& headers, int timeout_ms) {
  HttpResult r;
  std::string host, port, base;
  if (!ParseUrl(url, &host, &port, &base, &r.error)) return r;
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (const int rc = ::getaddrinfo(host.c_str(), port.c_str(), &hints, &res); rc != 0) {
    r.error = std::string("resolve ") + host + ": " + gai_strerror(rc);
    return r;
  }
  Fd s;
  for (addrinfo* a = res; a; a = a->ai_next) {
    s.fd = ::socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
    if (s.fd < 0) continue;
    if (Connect(s.fd, a->ai_addr, a->ai_addrlen, timeout_ms)) break;
    const int e = errno;
    ::close(s.fd);
    s.fd = -1;
    errno = e;
  }
  ::freeaddrinfo(res);
  if (s.fd < 0) {
    r.error = "connect " + host + ":" + port + ": " + std::strerror(errno);
    return r;
  }
  std::string req = method + " " + base + path + " HTTP/1.1\r\nHost: " + host + ":" + port +
                    "\r\nConnection: close\r\nAccept: application/json\r\n";
  if (method != "GET") req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
  for (const auto& [k, v] : headers) req += k + ": " + v + "\r\n";
  req += "\r\n";
  if (method != "GET") req += body;
  for (std::size_t off = 0; off < req.size();) {
    if (!WaitFor(s.fd, POLLOUT, timeout_ms)) return r.error = "send timed out", r;
    const ssize_t n = ::send(s.fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) continue;
      r.error = std::string("send: ") + std::strerror(errno);
      return r;
    }
    off += static_cast<std::size_t>(n);
  }
  std::string raw;
  char buf[65536];
  while (true) {
    if (!WaitFor(s.fd, POLLIN, timeout_ms)) return r.error = "read timed out", r;
    const ssize_t n = ::recv(s.fd, buf, sizeof buf, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      r.error = std::string("recv: ") + std::strerror(errno);
      return r;
    }
    if (n == 0) break;
    raw.append(buf, static_cast<std::size_t>(n));
  }
  const auto head_end = raw.find("\r\n\r\n");
  if (head_end == std::string::npos || raw.size() < 12 || raw.compare(0, 5, "HTTP/") != 0)
    return r.error = "malformed response", r;
  r.status = std::atoi(raw.c_str() + 9);
  std::string head = raw.substr(0, head_end);
  for (auto& c : head) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  const std::string_view rest(raw.data() + head_end + 4, raw.size() - head_end - 4);
  if (head.find("transfer-encoding: chunked") != std::string::npos) {
    if (!Dechunk(rest, &r.body)) return r.error = "truncated chunked body", r;
  } else if (const auto cl = head.find("content-length:"); cl != std::string::npos) {
    const std::size_t n = std::strtoull(head.c_str() + cl + 15, nullptr, 10);
    if (rest.size() < n) return r.error = "truncated body", r;
    r.body.assign(rest.substr(0, n));
  } else {
    r.body.assign(rest);
  }
  r.transport_ok = true;
  return r;
}

ChatReply CallChat(JsonPoster& poster, const std::string& url, const json::Value& body, const HttpHeaders& headers) {
  using Kind = json::Value::Kind;
  ChatReply r;
  const auto t0 = std::chrono::steady_clock::now();
  const HttpResult h = poster.Post(url, "/v1/chat/completions", json::DumpPython(body), headers);
  r.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  r.raw = h.body;
  if (!h.transport_ok) {
    r.status = ChatReply::Status::kTransient;
    r.error = h.error;
    return r;
  }
  if (h.status != 200) {
    r.status = h.status == 429 || h.status >= 500 ? ChatReply::Status::kTransient : ChatReply::Status::kFailed;
    r.error = "HTTP " + std::to_string(h.status) + ": " + h.body.substr(0, 500);
    return r;
  }
  std::string err;
  auto v = json::Parse(h.body, &err);
  const json::Value* choices = v ? v->find("choices") : nullptr;
  if (!choices || !choices->is(Kind::kArray) || choices->items().empty()) {
    r.error = "malformed chat completion: " + (v ? std::string("no choices") : err);
    return r;
  }
  const auto& c0 = choices->items()[0];
  if (const auto* m = c0.find("message"))
    if (const auto* t = m->find("content"); t && t->is(Kind::kString)) r.text = t->str();
  if (const auto* f = c0.find("finish_reason"); f && f->is(Kind::kString)) r.finish_reason = f->str();
  if (const auto* x = v->find("id"); x && x->is(Kind::kString)) r.id = x->str();
  if (const auto* x = v->find("model"); x && x->is(Kind::kString)) r.model = x->str();
  if (const auto* u = v->find("usage")) {
    if (const auto* x = u->find("prompt_tokens"); x && x->is(Kind::kInt)) r.prompt_tokens = static_cast<std::uint64_t>(x->number());
    if (const auto* x = u->find("completion_tokens"); x && x->is(Kind::kInt))
      r.completion_tokens = static_cast<std::uint64_t>(x->number());
  }
  r.status = ChatReply::Status::kOk;
  return r;
}

}  // namespace halo::scope
