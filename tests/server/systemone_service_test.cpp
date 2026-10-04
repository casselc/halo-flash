// POST /v1/systemone over loopback with a fake decision engine: answers come
// back with latency_ms; invalid requests and engine refusals are 422 with a
// "detail" message, as strands-decider's server answers them.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

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
