// The operation verbs over HTTP (server/operation_service.hpp; docs/OPERATIONS.md section 3): submit, status,
// fetch, observe, cancel and ack against a scripted generator, with the contract's answers for each fault: a
// duplicate, another incarnation, another digest, a late submit, a mode the server does not run, a scheduler
// refusal (not accepted: no record), the record ceiling, a cancel before acceptance and while running, an observer
// that leaves (the attempt runs on), and the contract's operations section.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#include "core/json.hpp"
#include "core/sha256.hpp"
#include "server/contract.hpp"
#include "server/http.hpp"
#include "server/incarnation.hpp"
#include "server/openai_service.hpp"
#include "server/operation_service.hpp"

using namespace halo::server;
namespace json = halo::json;

namespace {

/// Emits `script` 3 bytes a token, `delay_ms` apart; refuses admission while `refuse` is set.
class Gen : public TextGenerator {
 public:
  Admission Admit(const Params&) override {
    Admission a;
    if (refuse) a.refused = true, a.retry_after_s = 7, a.why = "limit";
    return a;
  }
  Result Generate(const Params& p, const std::function<bool(std::string_view)>& on_text) override {
    ++runs;
    Result r;
    r.prompt_tokens = 4;
    for (std::size_t i = 0; i < script.size(); i += 3) {
      if (p.cancelled && p.cancelled()) {
        r.cancelled = true;
        break;
      }
      ++r.completion_tokens;
      if (!on_text(std::string_view(script).substr(i, 3))) break;
      if (delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    return r;
  }
  std::string script = "hello world, all well";
  std::atomic<int> delay_ms{0}, runs{0};
  std::atomic<bool> refuse{false};
};

struct Reply {
  int status = 0;
  std::string head, body;
  std::string Header(const std::string& name) const {
    const std::string key = "\r\n" + name + ": ";
    const auto at = head.find(key);
    if (at == std::string::npos) return "";
    const auto from = at + key.size();
    return head.substr(from, head.find("\r\n", from) - from);
  }
  json::Value Json() const {
    std::string err;
    auto v = json::Parse(body, &err);
    if (!v) throw std::runtime_error("bad JSON: " + err + ": " + body);
    return *v;
  }
};

std::string Dechunk(const std::string& s) {
  std::string out;
  std::size_t at = 0;
  while (at < s.size()) {
    const std::size_t eol = s.find("\r\n", at);
    if (eol == std::string::npos) break;
    const std::size_t n = std::stoul(s.substr(at, eol - at), nullptr, 16);
    if (n == 0) break;
    out += s.substr(eol + 2, n);
    at = eol + 2 + n + 2;
  }
  return out;
}

/// One request on a fresh connection, read to the end (`max_bytes` > 0: close after that many body bytes).
Reply Call(int port, const std::string& method, const std::string& path, const std::string& body,
           const std::string& headers, std::size_t max_bytes = 0) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
  const std::string req = method + " " + path + " HTTP/1.1\r\nConnection: close\r\n" + headers +
                          "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
  (void)::send(fd, req.data(), req.size(), 0);
  std::string all;
  char buf[4096];
  for (;;) {
    const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
    if (n <= 0) break;
    all.append(buf, static_cast<std::size_t>(n));
    if (max_bytes && all.find("\r\n\r\n") != std::string::npos && all.size() - all.find("\r\n\r\n") > max_bytes) break;
  }
  ::close(fd);
  Reply r;
  const auto end = all.find("\r\n\r\n");
  r.head = all.substr(0, end + 2);
  r.status = std::stoi(all.substr(9, 3));
  r.body = all.substr(end + 4);
  if (r.head.find("Transfer-Encoding: chunked") != std::string::npos) r.body = Dechunk(r.body);
  return r;
}

struct Rig {
  explicit Rig(Operations::Options records = {}, bool relaxed = true)
      : server({.stamp = [this](std::vector<std::pair<std::string, std::string>>& h) { inc.Stamp(h); }}) {
    ServiceOptions so;
    int n = 0;
    so.new_id = [n]() mutable { return std::to_string(++n); };
    so.now_s = [] { return std::int64_t{1700000000}; };
    service = std::make_unique<OpenAiService>(gen, so);
    ops = std::make_unique<OperationService>(*service, inc, OperationServiceOptions{.records = records, .relaxed = relaxed});
    ops->Register(server);
    RegisterContract(server, inc, {.relaxed = relaxed, .operations = [this] { return ops->Contract(); }});
    std::string err;
    if (!server.Start(&err)) throw std::runtime_error(err);
  }
  ~Rig() { server.Stop(); }

  /// A submit body for attempt `attempt` of operation `op`, not-after `ms_ahead` from now.
  std::string Body(const std::string& op, int attempt, std::int64_t ms_ahead = 60000, const std::string& mode = "relaxed",
                   const std::string& prompt = "hi") const {
    return R"({"operation_id": ")" + op + R"(", "attempt": )" + std::to_string(attempt) +
           R"(, "not_after": )" + std::to_string(inc.NowMs() + ms_ahead) + R"(, "api": "completions", "mode": ")" +
           mode + R"(", "request": {"prompt": ")" + prompt + R"(", "max_tokens": 64}})";
  }
  Reply Submit(const std::string& body, const std::string& incarnation = "") {
    return Call(server.port(), "POST", "/v1/operations", body,
                "Halo-Incarnation: " + (incarnation.empty() ? inc.id() : incarnation) + "\r\nX-Halo-Principal: p1\r\n");
  }
  /// A verb on the attempt `body` submitted (its not-after and digest), for `incarnation` (default: ours).
  Reply Verb(const std::string& method, const std::string& body, const std::string& what = "",
             const std::string& incarnation = "", const std::string& digest = "", std::size_t max_bytes = 0) {
    std::string err;
    const auto v = *json::Parse(body, &err);
    const std::string path = "/v1/operations/" + v.find("operation_id")->str() + "/" +
                             v.find("attempt")->int_digits() + (what.empty() ? "" : "/" + what);
    return Call(server.port(), method, path, "",
                "Halo-Incarnation: " + (incarnation.empty() ? inc.id() : incarnation) + "\r\nHalo-Not-After: " +
                    v.find("not_after")->int_digits() + "\r\nHalo-Digest: " +
                    (digest.empty() ? halo::core::Sha256Hex(body) : digest) + "\r\nX-Halo-Principal: p1\r\n",
                max_bytes);
  }
  /// A request to `path`; with `body`, carrying that attempt's identity headers.
  Reply Call0(const std::string& method, const std::string& path, const std::string& body = "") {
    std::string h;
    if (!body.empty()) {
      std::string err;
      const auto v = *json::Parse(body, &err);
      h = "Halo-Incarnation: " + inc.id() + "\r\nHalo-Not-After: " + v.find("not_after")->int_digits() +
          "\r\nHalo-Digest: " + halo::core::Sha256Hex(body) + "\r\nX-Halo-Principal: p1\r\n";
    }
    return Call(server.port(), method, path, "", h);
  }
  std::string StatusOf(const std::string& body) { return Verb("GET", body).Json().find("status")->str(); }
  /// Polls until the attempt leaves `running`.
  std::string AwaitEnd(const std::string& body) {
    for (int i = 0; i < 1000; ++i) {
      const std::string s = StatusOf(body);
      if (s != "running") return s;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return "running";
  }

  Gen gen;
  Incarnation inc;
  HttpServer server;
  std::unique_ptr<OpenAiService> service;
  std::unique_ptr<OperationService> ops;
};

}  // namespace

TEST(OperationService, AnAttemptRunsIsFetchedAndAcknowledged) {
  Rig r;
  const std::string body = r.Body("op1", 0);
  auto s = r.Submit(body);
  ASSERT_EQ(s.status, 202) << s.body;
  EXPECT_EQ(s.Header("Halo-Incarnation"), r.inc.id());
  EXPECT_EQ(s.Json().find("digest")->str(), halo::core::Sha256Hex(body));
  ASSERT_EQ(r.AwaitEnd(body), "done");
  const auto st = r.Verb("GET", body).Json();
  EXPECT_EQ(st.find("finish")->str(), "stop");
  EXPECT_EQ(st.find("usage")->find("completion_tokens")->int_digits(), "7");
  EXPECT_TRUE(st.find("expires_at"));
  const auto res = r.Verb("GET", body, "result");
  ASSERT_EQ(res.status, 200);
  EXPECT_EQ(res.Json().find("choices")->items()[0].find("text")->str(), "hello world, all well");
  EXPECT_EQ(r.Submit(body).status, 200);  // a duplicate: its status, nothing run again
  EXPECT_EQ(r.gen.runs.load(), 1);
  const auto ack = r.Verb("POST", body, "ack");
  ASSERT_EQ(ack.status, 200);
  EXPECT_EQ(ack.Json().find("status")->str(), "released");
  EXPECT_EQ(r.Verb("GET", body, "result").status, 409);
  EXPECT_EQ(r.Verb("GET", body, "stream").status, 410);
}

TEST(OperationService, VerbsBindIncarnationDigestDeadlineAndMode) {
  Rig r;
  const std::string body = r.Body("op1", 0);
  auto s = r.Submit(body, "00000000000000000000000000000000");
  EXPECT_EQ(s.status, 409);
  EXPECT_EQ(s.Json().find("incarnation")->str(), r.inc.id());
  EXPECT_EQ(r.Verb("GET", body, "", "00000000000000000000000000000000").Json().find("status")->str(), "unknown");
  EXPECT_EQ(r.Verb("POST", body, "cancel", "00000000000000000000000000000000").status, 409);
  EXPECT_EQ(r.Call0("GET", "/v1/operations/op1/0").status, 400);  // no identity headers
  EXPECT_EQ(Call(r.server.port(), "POST", "/v1/operations", body,
                 "Halo-Incarnation: " + r.inc.id() + "\r\nHalo-Digest: abc\r\n").status, 400);
  EXPECT_EQ(r.Submit(r.Body("op2", 0, -1)).status, 410);           // past its not-after
  EXPECT_EQ(r.Submit(r.Body("op3", 0, 10'000'000)).status, 400);   // too far ahead
  EXPECT_EQ(r.Submit(r.Body("op4", 0, 60000, "exact")).status, 422);
  ASSERT_EQ(r.Submit(body).status, 202);
  ASSERT_EQ(r.AwaitEnd(body), "done");
  EXPECT_EQ(r.Submit(r.Body("op1", 0, 60000, "relaxed", "other")).status, 409);  // same attempt, another request
  EXPECT_EQ(r.Verb("GET", body, "", "", "beef").status, 409);
  EXPECT_EQ(r.Verb("GET", r.Body("op5", 0)).Json().find("status")->str(), "never_accepted");
  EXPECT_EQ(r.Verb("GET", r.Body("op6", 0, -1)).Json().find("status")->str(), "closed");
}

TEST(OperationService, ASchedulerRefusalIsNotAnAcceptance) {
  Rig r;
  r.gen.refuse = true;
  const std::string body = r.Body("op1", 0);
  const auto s = r.Submit(body);
  EXPECT_EQ(s.status, 429);
  EXPECT_EQ(s.Header("Retry-After"), "7");
  EXPECT_EQ(r.StatusOf(body), "never_accepted");  // no record: a later copy may still be accepted
  r.gen.refuse = false;
  EXPECT_EQ(r.Submit(body).status, 202);
  EXPECT_EQ(r.AwaitEnd(body), "done");
}

TEST(OperationService, AtTheRecordCeilingSubmitsAreRefused) {
  Operations::Options o;
  o.max_records = 1;
  Rig r(o);
  const std::string a = r.Body("op1", 0), b = r.Body("op2", 0);
  ASSERT_EQ(r.Submit(a).status, 202);
  ASSERT_EQ(r.AwaitEnd(a), "done");
  const auto s = r.Submit(b);
  EXPECT_EQ(s.status, 503);
  EXPECT_FALSE(s.Header("Retry-After").empty());
  EXPECT_EQ(s.Json().find("error")->find("type")->str(), "records");
  const auto c = r.Call0("GET", "/v1/halo/contract").Json();
  EXPECT_EQ(c.find("operations")->find("records")->find("unacked_outcomes")->int_digits(), "1");
}

TEST(OperationService, CancelBeforeAcceptanceAndWhileRunning) {
  Rig r;
  const std::string early = r.Body("op1", 0);
  auto c = r.Verb("POST", early, "cancel");
  ASSERT_EQ(c.status, 200);
  EXPECT_EQ(c.Json().find("status")->str(), "cancelled_before_accept");
  const auto late = r.Submit(early);  // the delayed copy
  EXPECT_EQ(late.status, 200);
  EXPECT_EQ(late.Json().find("status")->str(), "cancelled_before_accept");
  EXPECT_EQ(r.gen.runs.load(), 0);

  r.gen.delay_ms = 20;
  const std::string running = r.Body("op2", 0);
  ASSERT_EQ(r.Submit(running).status, 202);
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  c = r.Verb("POST", running, "cancel");
  ASSERT_EQ(c.status, 200);
  EXPECT_EQ(c.Json().find("status")->str(), "running");
  EXPECT_TRUE(c.Json().find("cancel_requested")->boolean());
  ASSERT_EQ(r.AwaitEnd(running), "done");
  const auto st = r.Verb("GET", running).Json();
  EXPECT_EQ(st.find("finish")->str(), "cancelled");
  const auto used = std::stoull(st.find("usage")->find("completion_tokens")->int_digits());
  EXPECT_GT(used, 0u);  // a cancelled attempt that ran reports what it spent
  EXPECT_LT(used, 7u);
}

TEST(OperationService, AnObserverSeesTheEventsAndItsDepartureCancelsNothing) {
  Rig r;
  r.gen.delay_ms = 10;
  const std::string body = r.Body("op1", 0);
  ASSERT_EQ(r.Submit(body).status, 202);
  // An observer that leaves after the first bytes.
  (void)r.Verb("GET", body, "stream", "", "", 10);
  ASSERT_EQ(r.AwaitEnd(body), "done");
  EXPECT_EQ(r.Verb("GET", body).Json().find("finish")->str(), "stop");
  // A late observer replays every event from 0, then the status.
  const auto all = r.Verb("GET", body, "stream");
  ASSERT_EQ(all.status, 200);
  EXPECT_NE(all.body.find("\"text\":\"hel\""), std::string::npos) << all.head << all.body;
  EXPECT_NE(all.body.find("event: status\ndata: "), std::string::npos);
  EXPECT_NE(all.body.find("\"status\":\"done\""), std::string::npos);
  const auto tail = r.Call0("GET", "/v1/operations/op1/0/stream?from=3", body);
  EXPECT_EQ(tail.body.find("\"text\":\"hel\""), std::string::npos);
  EXPECT_NE(tail.body.find("event: status"), std::string::npos);
}

TEST(OperationService, TheContractDescribesTheVerbs) {
  Rig r;
  const auto c = r.Call0("GET", "/v1/halo/contract").Json();
  const auto* ops = c.find("operations");
  ASSERT_TRUE(ops && ops->is(json::Value::Kind::kObject));
  EXPECT_EQ(ops->find("disconnect")->str(), "never cancels");
  EXPECT_EQ(ops->find("apis")->items().size(), 2u);
  EXPECT_TRUE(ops->find("retention")->find("capture_s"));
}
