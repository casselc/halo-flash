// POST /v1/systemone over loopback with a fake decision engine: answers come
// back with latency_ms; invalid requests and engine refusals are 422 with a
// "detail" message, as strands-decider's server answers them.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <atomic>
#include <functional>
#include <string>

#include "core/json.hpp"
#include "server/http.hpp"
#include "server/systemone_service.hpp"

using namespace halo;
using namespace halo::server;

namespace {

class FakeEngine : public DecisionEngine {
 public:
  std::optional<json::Value> Evaluate(const decide::Request& r, std::string* error) override {
    ++calls;
    if (r.questions.size() > 2) return *error = "at most 2 questions in this test", std::nullopt;
    json::Value body = json::Value::MakeObject();
    body.set("answers", json::Value::Int(static_cast<std::int64_t>(r.questions.size())));
    return body;
  }
  int calls = 0;
};

std::pair<int, std::string> Post(int port, const std::string& body) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
  const std::string req = "POST /v1/systemone HTTP/1.1\r\nConnection: close\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\n\r\n" + body;
  ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
  std::string raw;
  char b[4096];
  for (ssize_t n; (n = ::recv(fd, b, sizeof b, 0)) > 0;) raw.append(b, static_cast<std::size_t>(n));
  ::close(fd);
  return {std::stoi(raw.substr(9, 3)), raw.substr(raw.find("\r\n\r\n") + 4)};
}

json::Value J(const std::string& s) {
  std::string e;
  auto v = json::Parse(s, &e);
  if (!v) throw std::runtime_error(e);
  return *v;
}

}  // namespace

TEST(SystemOneService, AnswersAndRefusalsInTheDeciderShape) {
  FakeEngine engine;
  HttpServer server({});
  RegisterSystemOne(server, engine);
  std::string err;
  ASSERT_TRUE(server.Start(&err)) << err;
  const std::string ok = R"({"state": "The build is red.", "questions": {
      "deploy": {"type": "noul", "instructions": "Should we deploy?"},
      "next": {"type": "choice", "instructions": "Next step?", "criteria": {"fix": "Fix the test", "revert": "Revert"}}}})";
  auto [s1, b1] = Post(server.port(), ok);
  ASSERT_EQ(s1, 200) << b1;
  const auto v = J(b1);
  EXPECT_EQ(v.find("answers")->number(), 2);
  ASSERT_TRUE(v.find("latency_ms"));
  EXPECT_GE(v.find("latency_ms")->number(), 0.0);
  // Validation: a choice needs at least two options.
  auto [s2, b2] = Post(server.port(), R"({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x",
                                          "criteria": {"only": "one"}}}})");
  EXPECT_EQ(s2, 422);
  EXPECT_TRUE(J(b2).find("detail")->is(json::Value::Kind::kString));
  EXPECT_EQ(Post(server.port(), "{oops").first, 422);
  // The engine's refusal.
  auto [s3, b3] = Post(server.port(), R"({"state": "s", "questions": {"a": {"type": "noul", "instructions": "a"},
      "b": {"type": "noul", "instructions": "b"}, "c": {"type": "noul", "instructions": "c"}}})");
  EXPECT_EQ(s3, 422);
  EXPECT_EQ(J(b3).find("detail")->str(), "at most 2 questions in this test");
  EXPECT_EQ(engine.calls, 2);
  server.Stop();
}

// G3 M4: the route evaluates through a DeciderGate: concurrent (its own mutex) or scheduled (the runner, i.e.
// ScheduledGenerator::RunDecider); the gate counts and times each evaluation for /metrics, and a runner that
// could not run the evaluation is a 503, never an answer.
TEST(SystemOneService, EvaluationsGoThroughTheDeciderGate) {
  const std::string ok = R"({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}})";
  {
    FakeEngine engine;
    DeciderGate gate;
    HttpServer server({});
    RegisterSystemOne(server, engine, &gate);
    std::string err;
    ASSERT_TRUE(server.Start(&err)) << err;
    EXPECT_EQ(Post(server.port(), ok).first, 200);
    EXPECT_EQ(Post(server.port(), ok).first, 200);
    server.Stop();
    EXPECT_EQ(gate.mode(), DeciderGate::Mode::kConcurrent);
    EXPECT_EQ(gate.stats().runs, 2u);
    EXPECT_EQ(gate.stats().eval.count, 2u);
    std::string m;
    gate.Render(m, "model_name=\"t\"");
    EXPECT_NE(m.find("halo:decider_requests_total{model_name=\"t\",mode=\"concurrent\",outcome=\"ran\"} 2\n"),
              std::string::npos) << m;
    EXPECT_NE(m.find("halo:decider_wait_seconds_count{model_name=\"t\",mode=\"concurrent\"} 2\n"), std::string::npos);
  }
  {
    FakeEngine engine;
    std::atomic<int> runs{0};
    std::atomic<bool> fail{false};
    DeciderGate gate(DeciderGate::Mode::kScheduled, [&](const std::function<void()>& work, std::string* e) {
      ++runs;
      if (fail) return *e = "stopped", false;
      work();
      return true;
    });
    HttpServer server({});
    RegisterSystemOne(server, engine, &gate);
    std::string err;
    ASSERT_TRUE(server.Start(&err)) << err;
    EXPECT_EQ(Post(server.port(), ok).first, 200);
    fail = true;
    auto [s, b] = Post(server.port(), ok);
    EXPECT_EQ(s, 503) << b;
    server.Stop();
    EXPECT_EQ(gate.mode(), DeciderGate::Mode::kScheduled);
    EXPECT_EQ(runs.load(), 2);
    EXPECT_EQ(engine.calls, 1);
    EXPECT_EQ(gate.stats().runs, 1u);
    EXPECT_EQ(gate.stats().failed, 1u);
  }
}
