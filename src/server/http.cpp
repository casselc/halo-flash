#include "server/http.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <linux/sockios.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string.h>
#include <list>

#include "core/json.hpp"

namespace halo::server {

namespace {

bool IsTchar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || std::strchr("!#$%&'*+-.^_`|~", c) != nullptr;
}

std::string Lower(std::string_view s) {
  std::string out(s);
  for (auto& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string_view Trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

/// Whether a comma-separated header value lists `token` (case-insensitive).
bool HasToken(std::string_view value, std::string_view token) {
  std::size_t i = 0;
  while (i <= value.size()) {
    const auto j = std::min(value.find(',', i), value.size());
    if (Lower(Trim(value.substr(i, j - i))) == token) return true;
    i = j + 1;
  }
  return false;
}

std::string ErrorJson(std::string_view message) {
  return R"({"error": {"message": )" + json::DumpPython(json::Value::String(std::string(message))) +
         R"(, "type": "invalid_request_error", "param": null, "code": null}})";
}

/// Writes all of `data`, or fails: the peer is gone, or the write made no progress for `timeout_ms` (a full send
/// buffer: the peer stopped reading). Non-blocking sends, so the wait is bounded by the deadline.
bool SendAll(int fd, std::string_view data, int timeout_ms) {
  using Clock = std::chrono::steady_clock;
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  while (!data.empty()) {
    const ssize_t n = ::send(fd, data.data(), data.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
    if (n >= 0) {
      data.remove_prefix(static_cast<std::size_t>(n));
      continue;
    }
    if (errno == EINTR) continue;
    if (errno != EAGAIN && errno != EWOULDBLOCK) return false;
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    if (left <= 0) return false;
    pollfd p{fd, POLLOUT, 0};
    if (::poll(&p, 1, static_cast<int>(left)) < 0 && errno != EINTR) return false;
  }
  return true;
}

/// Whether the peer still has its end of the connection open (no FIN, reset or error seen yet).
bool PeerOpen(int fd) {
  pollfd p{fd, POLLRDHUP, 0};
  const int r = ::poll(&p, 1, 0);
  if (r < 0) return errno == EINTR;
  return (p.revents & (POLLRDHUP | POLLHUP | POLLERR | POLLNVAL)) == 0;
}

}  // namespace

const std::string* HttpRequest::Header(std::string_view lower_name) const {
  for (const auto& [k, v] : headers)
    if (k == lower_name) return &v;
  return nullptr;
}

void SecretBuffer::Append(std::string_view bytes) {
  if (bytes.empty()) return;
  if (n_ + bytes.size() > cap_) {
    const std::size_t cap = std::max({n_ + bytes.size(), cap_ * 2, std::size_t{4096}});
    std::unique_ptr<char[]> q(new char[cap]());
    const std::size_t n = n_;
    if (n) std::memcpy(q.get(), p_.get(), n);
    Wipe();  // the old storage is zeroed before it is freed
    p_ = std::move(q);
    cap_ = cap;
    n_ = n;
  }
  std::memcpy(p_.get() + n_, bytes.data(), bytes.size());
  n_ += bytes.size();
}

void SecretBuffer::Consume(std::size_t n) {
  n = std::min(n, n_);
  if (n == 0) return;
  std::memmove(p_.get(), p_.get() + n, n_ - n);
  explicit_bzero(p_.get() + (n_ - n), n);  // the tail the move vacated still held the moved bytes
  n_ -= n;
}

void SecretBuffer::Wipe() noexcept {
  if (p_) {
    explicit_bzero(p_.get(), cap_);
    if (on_free_for_test) on_free_for_test({p_.get(), cap_});
  }
  p_.reset();
  cap_ = n_ = 0;
}

HttpParser::Status HttpParser::Fail(int status, std::string message) {
  error_status_ = status;
  error_ = std::move(message);
  return Status::kError;
}

HttpParser::Status HttpParser::Next(HttpRequest* out) {
  if (error_status_) return Status::kError;
  // RFC 9112 2.2: ignore empty lines before the request line.
  std::size_t lead = 0;
  const std::string_view b0 = buf_.view();
  while (lead + 1 < b0.size() && b0[lead] == '\r' && b0[lead + 1] == '\n') lead += 2;
  if (lead) buf_.Consume(lead);
  const std::string_view buf = buf_.view();
  const auto end = buf.find("\r\n\r\n");
  if (end == std::string::npos) {
    if (buf_.size() > limits_.max_head_bytes) return Fail(431, "request head too large");
    return Status::kNeedMore;
  }
  if (end + 4 > limits_.max_head_bytes) return Fail(431, "request head too large");
  const std::string_view head(buf.data(), end);
  HttpRequest r;
  // Request line: method SP target SP version.
  const auto line_end = head.find("\r\n");
  const std::string_view line = head.substr(0, line_end);
  const auto sp1 = line.find(' ');
  const auto sp2 = sp1 == std::string_view::npos ? sp1 : line.find(' ', sp1 + 1);
  if (sp1 == std::string_view::npos || sp2 == std::string_view::npos || sp1 == 0 || sp2 == sp1 + 1 ||
      line.find(' ', sp2 + 1) != std::string_view::npos)
    return Fail(400, "malformed request line");
  r.method = std::string(line.substr(0, sp1));
  if (!std::all_of(r.method.begin(), r.method.end(), IsTchar)) return Fail(400, "malformed method");
  r.target = std::string(line.substr(sp1 + 1, sp2 - sp1 - 1));
  for (char c : r.target)
    if (static_cast<unsigned char>(c) <= 0x20 || c == 0x7f) return Fail(400, "malformed request target");
  const std::string_view version = line.substr(sp2 + 1);
  bool http10 = false;
  if (version == "HTTP/1.0") http10 = true;
  else if (version != "HTTP/1.1") return Fail(version.starts_with("HTTP/") ? 505 : 400, "unsupported HTTP version");
  const auto q = r.target.find('?');
  r.path = r.target.substr(0, q);
  if (q != std::string::npos) r.query = r.target.substr(q + 1);
  // Header fields.
  std::size_t pos = line_end == std::string_view::npos ? head.size() : line_end + 2;
  while (pos < head.size()) {
    auto eol = head.find("\r\n", pos);
    if (eol == std::string_view::npos) eol = head.size();
    const std::string_view field = head.substr(pos, eol - pos);
    pos = eol + 2;
    if (field.empty() || field.front() == ' ' || field.front() == '\t') return Fail(400, "obsolete line folding");
    const auto colon = field.find(':');
    if (colon == std::string_view::npos || colon == 0) return Fail(400, "malformed header field");
    const std::string_view name = field.substr(0, colon);
    if (!std::all_of(name.begin(), name.end(), IsTchar)) return Fail(400, "malformed header name");
    const std::string_view value = Trim(field.substr(colon + 1));
    for (char c : value)
      if ((static_cast<unsigned char>(c) < 0x20 && c != '\t') || c == 0x7f) return Fail(400, "control character in header");
    if (r.headers.size() == limits_.max_headers) return Fail(431, "too many header fields");
    r.headers.emplace_back(Lower(name), std::string(value));
  }
  // Body framing: Content-Length only.
  if (r.Header("transfer-encoding"))
    return Fail(r.Header("content-length") ? 400 : 411, "chunked request bodies are not supported; send Content-Length");
  std::size_t length = 0;
  bool have_length = false;
  for (const auto& [k, v] : r.headers) {
    if (k != "content-length") continue;
    std::size_t n = 0;
    const auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
    if (v.empty() || ec != std::errc() || p != v.data() + v.size()) return Fail(400, "malformed Content-Length");
    if (have_length && n != length) return Fail(400, "conflicting Content-Length");
    length = n;
    have_length = true;
  }
  if (length > limits_.max_body_bytes) return Fail(413, "request body too large");
  if (buf.size() - (end + 4) < length) return Status::kNeedMore;
  r.body = std::string(buf.substr(end + 4, length));
  const std::string* conn = r.Header("connection");
  r.keep_alive = http10 ? (conn && HasToken(*conn, "keep-alive")) : !(conn && HasToken(*conn, "close"));
  buf_.Consume(end + 4 + length);  // the vacated bytes are zeroed (a request may carry a key)
  *out = std::move(r);
  return Status::kRequest;
}

const char* ReasonPhrase(int status) {
  switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 411: return "Length Required";
    case 413: return "Content Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 429: return "Too Many Requests";
    case 503: return "Service Unavailable";
    case 505: return "HTTP Version Not Supported";
    default: return "Unknown";
  }
}

HttpResponse HttpResponse::Json(int status, std::string body) {
  HttpResponse r;
  r.status = status;
  r.headers.emplace_back("Content-Type", "application/json");
  r.body = std::move(body);
  return r;
}

HttpResponse HttpResponse::EventStream(std::function<void(const BodyWriter&)> stream) {
  HttpResponse r;
  r.headers.emplace_back("Content-Type", "text/event-stream");
  r.headers.emplace_back("Cache-Control", "no-cache");
  r.stream = std::move(stream);
  return r;
}

namespace {

std::string Head(const HttpResponse& r, bool keep_alive, bool chunked) {
  std::string h = "HTTP/1.1 " + std::to_string(r.status) + " " + ReasonPhrase(r.status) + "\r\n";
  for (const auto& [k, v] : r.headers) h += k + ": " + v + "\r\n";
  if (chunked) h += "Transfer-Encoding: chunked\r\n";
  else h += "Content-Length: " + std::to_string(r.body.size()) + "\r\n";
  h += keep_alive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n";
  return h;
}

}  // namespace

std::string SerializeResponse(const HttpResponse& r, bool keep_alive) { return Head(r, keep_alive, false) + r.body; }

HttpServer::HttpServer(HttpServerOptions options) : opt_(std::move(options)) {}

HttpServer::~HttpServer() { Stop(); }

void HttpServer::Route(std::string method, std::string path, HttpHandler handler) {
  routes_[{std::move(method), std::move(path)}] = std::move(handler);
}

void HttpServer::RoutePrefix(std::string method, std::string prefix, HttpHandler handler) {
  prefix_routes_.push_back({{std::move(method), std::move(prefix)}, std::move(handler)});
}

bool HttpServer::StartUnix(std::string* error) {
  // A Unix socket in a directory only this user can enter (0700, ours), the socket itself 0600; every connection's
  // peer credentials are checked too (accept loop), so neither a loose umask nor a moved socket opens it up.
  const std::string& path = opt_.unix_path;
  sockaddr_un addr{};
  if (path.size() >= sizeof addr.sun_path) return *error = "unix socket path too long: " + path, false;
  const auto slash = path.rfind('/');
  const std::string dir = slash == std::string::npos ? "." : (slash == 0 ? "/" : path.substr(0, slash));
  struct stat st {};
  if (::lstat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != ::geteuid() || (st.st_mode & 077) != 0)
    return *error = "unix socket directory " + dir + " must exist, be ours and mode 0700", false;
  if (::lstat(path.c_str(), &st) == 0) {
    if (!S_ISSOCK(st.st_mode)) return *error = path + " exists and is not a socket", false;
    // Live, stale, or unknown: a bounded non-blocking connect. Only a refusal (nobody listens: ECONNREFUSED) or the
    // path vanishing (ENOENT) proves it stale; a connect, a full backlog (EAGAIN), a timeout or any other error
    // fails closed (the start is refused, nothing is unlinked).
    sockaddr_un pa{};
    pa.sun_family = AF_UNIX;
    if (path.size() >= sizeof pa.sun_path) return *error = "unix socket path too long: " + path, false;  // also above
    const int probe = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (probe < 0) return *error = "cannot probe " + path + ": " + std::strerror(errno), false;
    std::memcpy(pa.sun_path, path.c_str(), path.size() + 1);
    int rc = ::connect(probe, reinterpret_cast<sockaddr*>(&pa), sizeof pa);
    int err = rc == 0 ? 0 : errno;
    if (rc != 0 && (err == EINPROGRESS || err == EINTR)) {
      pollfd pf{probe, POLLOUT, 0};
      if (::poll(&pf, 1, 500) == 1) {
        socklen_t len = sizeof err;
        if (::getsockopt(probe, SOL_SOCKET, SO_ERROR, &err, &len) != 0) err = EIO;
      } else {
        err = ETIMEDOUT;
      }
    }
    ::close(probe);
    if (err != ECONNREFUSED && err != ENOENT)
      return *error = path + (err == 0 ? " is served by another live process" : " could not be probed (" +
                                                                                    std::string(std::strerror(err)) + ")") +
                      ": refusing to take it over",
             false;
    ::unlink(path.c_str());
  }
  listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listen_fd_ < 0) return *error = std::string("socket: ") + std::strerror(errno), false;
  addr.sun_family = AF_UNIX;
  std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0 || ::chmod(path.c_str(), 0600) != 0 ||
      ::listen(listen_fd_, 128) < 0) {
    *error = "bind/listen " + path + ": " + std::strerror(errno);
    ::close(listen_fd_);
    listen_fd_ = -1;
    ::unlink(path.c_str());
    return false;
  }
  unix_ = true;
  if (::lstat(path.c_str(), &st) == 0) unix_ino_ = st.st_ino, unix_dev_ = st.st_dev;
  acceptor_ = std::thread([this] { AcceptLoop(); });
  return true;
}

bool HttpServer::PeerAllowed(int fd) const {
  if (!unix_) return true;
  ucred cr{};
  socklen_t len = sizeof cr;
  if (::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cr, &len) != 0) return false;
  return cr.uid == ::geteuid();
}

bool HttpServer::Start(std::string* error) {
  if (!opt_.unix_path.empty()) return StartUnix(error);
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listen_fd_ < 0) return *error = std::string("socket: ") + std::strerror(errno), false;
  const int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(opt_.port));
  if (::inet_pton(AF_INET, opt_.host.c_str(), &addr.sin_addr) != 1)
    return *error = "invalid IPv4 host: " + opt_.host, false;
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0 || ::listen(listen_fd_, 128) < 0) {
    *error = std::string("bind/listen: ") + std::strerror(errno);
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  socklen_t len = sizeof addr;
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
  port_ = ntohs(addr.sin_port);
  acceptor_ = std::thread([this] { AcceptLoop(); });
  return true;
}

void HttpServer::AcceptLoop() {
  struct Worker {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::list<Worker> workers;
  const auto reap = [&](bool all) {
    for (auto it = workers.begin(); it != workers.end();) {
      if (all || it->done->load()) {
        it->thread.join();
        it = workers.erase(it);
      } else {
        ++it;
      }
    }
  };
  while (!stopping_.load()) {
    pollfd p{listen_fd_, POLLIN, 0};
    if (::poll(&p, 1, 200) <= 0) {
      reap(false);
      continue;
    }
    const int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd < 0) continue;
    reap(false);
    if (!PeerAllowed(fd)) {  // a Unix socket peer of another uid: closed without a word
      ::close(fd);
      continue;
    }
    if (active_.load() >= opt_.max_connections) {
      // On the accept thread: a short deadline, so a client that does not read cannot hold up accepting.
      (void)SendAll(fd, SerializeResponse(Stamped(HttpResponse::Json(503, ErrorJson("too many connections"))), false),
                    std::min(opt_.send_timeout_ms, 1000));
      ::close(fd);
      continue;
    }
    const int nodelay = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof nodelay);
    ++active_;
    {
      std::lock_guard<std::mutex> lk(mu_);
      fds_.push_back(fd);
    }
    auto done = std::make_shared<std::atomic<bool>>(false);
    workers.push_back({std::thread([this, fd, done] {
                         Serve(fd);
                         {
                           std::lock_guard<std::mutex> lk(mu_);
                           fds_.erase(std::find(fds_.begin(), fds_.end(), fd));
                         }
                         ::close(fd);
                         --active_;
                         done->store(true);
                       }),
                       done});
  }
  // Wake every connection blocked in poll/recv/send, then join them.
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (int fd : fds_) ::shutdown(fd, SHUT_RDWR);
  }
  reap(true);
}

HttpResponse HttpServer::Stamped(HttpResponse r) const {
  if (opt_.stamp) opt_.stamp(r.headers);
  return r;
}

HttpResponse HttpServer::Dispatch(const HttpRequest& req) {
  if (!opt_.api_key.empty()) {
    // The key as OpenAI clients send it ("Authorization: Bearer"), or as Anthropic's do ("x-api-key").
    const std::string* auth = req.Header("authorization");
    const std::string* key = req.Header("x-api-key");
    if (!(auth && *auth == "Bearer " + opt_.api_key) && !(key && *key == opt_.api_key)) {
      for (const auto& [prefix, body] : auth_errors_)
        if (req.path.starts_with(prefix)) return HttpResponse::Json(401, body);
      auto r = HttpResponse::Json(401, ErrorJson("invalid or missing API key"));
      r.headers.emplace_back("WWW-Authenticate", "Bearer");
      return r;
    }
  }
  const HttpHandler* handler = nullptr;
  if (const auto it = routes_.find({req.method, req.path}); it != routes_.end()) handler = &it->second;
  bool other_method = false;
  if (!handler) {
    for (const auto& [key, h] : routes_) other_method |= key.second == req.path;
    // The longest matching prefix registered for this method; a path that
    // only other methods' prefixes match is a 405.
    std::size_t best = 0;
    for (const auto& [key, h] : prefix_routes_) {
      if (!req.path.starts_with(key.second)) continue;
      if (key.first != req.method) {
        other_method = true;
      } else if (!handler || key.second.size() > best) {
        handler = &h;
        best = key.second.size();
      }
    }
  }
  if (!handler)
    return other_method ? HttpResponse::Json(405, ErrorJson("method not allowed"))
                        : HttpResponse::Json(404, ErrorJson("no route for " + req.path));
  try {
    return (*handler)(req);
  } catch (const std::exception& e) {
    return HttpResponse::Json(500, ErrorJson(std::string("internal error: ") + e.what()));
  }
}

void HttpServer::Serve(int fd) {
  HttpParser parser(opt_.limits);
  char buf[16384];
  int idle_ms = 0;
  while (!stopping_.load()) {
    HttpRequest req;
    struct BodyWipe {  // every path out of this request (handled, refused at the API key, streamed, failed): the body
      HttpRequest& r;  // and every header value (Authorization carries the API key) are zeroed
      ~BodyWipe() {
        if (!r.body.empty()) explicit_bzero(r.body.data(), r.body.size());
        for (auto& [k, v] : r.headers)
          if (!v.empty()) explicit_bzero(v.data(), v.size());
      }
    } body_wipe{req};
    const auto st = parser.Next(&req);
    if (st == HttpParser::Status::kError) {
      (void)SendAll(fd, SerializeResponse(Stamped(HttpResponse::Json(parser.error_status(), ErrorJson(parser.error()))), false),
                    opt_.send_timeout_ms);
      return;
    }
    if (st == HttpParser::Status::kRequest) {
      idle_ms = 0;
      req.connected = [this, fd] { return !stopping_.load() && PeerOpen(fd); };
      HttpResponse resp = Stamped(Dispatch(req));
      const bool keep = req.keep_alive && !stopping_.load();
      if (!resp.stream) {
        if (!SendAll(fd, SerializeResponse(resp, keep), opt_.send_timeout_ms) || !keep) return;
        continue;
      }
      const int t = opt_.send_timeout_ms;
      const std::string head = Head(resp, keep, true);
      bool alive = SendAll(fd, head, t);
      // A client that stops reading: the kernel keeps taking frames until its buffers fill, which at token rates
      // takes minutes (about 1 MB on loopback), so a send that blocks notices late. Bytes written that the kernel
      // has not sent yet (the peer's window is closed) and that make no progress for send_timeout_ms mean the
      // client is not reading: it is treated as gone.
      std::size_t written = head.size(), accepted = written;
      auto progress = std::chrono::steady_clock::now();
      const auto reading = [&] {
        int unsent = 0;
        if (::ioctl(fd, SIOCOUTQNSD, &unsent) < 0) return true;  // cannot tell
        const std::size_t now_accepted = written - std::min<std::size_t>(written, static_cast<std::size_t>(std::max(unsent, 0)));
        const auto now = std::chrono::steady_clock::now();
        if (unsent <= 0 || now_accepted > accepted) {
          accepted = now_accepted;
          progress = now;
          return true;
        }
        return now - progress < std::chrono::milliseconds(t);
      };
      const BodyWriter writer = [&](std::string_view data) {
        if (!alive || stopping_.load()) return alive = false;
        if (data.empty()) return true;  // an empty chunk would end the body
        char size[32];
        const int n = std::snprintf(size, sizeof size, "%zx\r\n", data.size());
        alive = SendAll(fd, std::string_view(size, static_cast<std::size_t>(n)), t) && SendAll(fd, data, t) &&
                SendAll(fd, "\r\n", t);
        written += static_cast<std::size_t>(n) + data.size() + 2;
        return alive = alive && reading();
      };
      resp.stream(writer);
      if (!alive || !SendAll(fd, "0\r\n\r\n", t) || !keep) return;
      continue;
    }
    pollfd p{fd, POLLIN, 0};
    const int ready = ::poll(&p, 1, 200);
    if (ready == 0) {
      idle_ms += 200;
      if (idle_ms >= opt_.idle_timeout_ms) return;
      continue;
    }
    if (ready < 0 && errno == EINTR) continue;
    const ssize_t n = ready < 0 ? -1 : ::recv(fd, buf, sizeof buf, 0);
    if (n <= 0) return;  // closed, reset or shut down by Stop()
    parser.Feed(std::string_view(buf, static_cast<std::size_t>(n)));
    explicit_bzero(buf, static_cast<std::size_t>(n));
  }
}

void HttpServer::Stop() {
  if (stopping_.exchange(true)) return;
  if (acceptor_.joinable()) acceptor_.join();
  if (listen_fd_ >= 0) ::close(listen_fd_);
  listen_fd_ = -1;
  if (unix_) {  // only our own socket: a successor may have replaced a path we no longer serve
    struct stat st {};
    if (::lstat(opt_.unix_path.c_str(), &st) == 0 && st.st_ino == unix_ino_ && st.st_dev == unix_dev_)
      ::unlink(opt_.unix_path.c_str());
  }
}

}  // namespace halo::server
