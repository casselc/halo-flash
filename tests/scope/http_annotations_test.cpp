// The harness annotation API over HTTP: each route writes the typed row the C
// ABI would, ids come back in the body, bad requests are 400 with a message,
// and unknown steps are refused.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <mutex>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "scope/annotate.hpp"
#include "scope/http_annotations.hpp"
#include "scope/recorder.hpp"
#include "server/http.hpp"

using namespace halo;
using namespace halo::scope;

namespace {

class Rows : public Sink {
 public:
  bool Write(Table table, std::string_view rows, std::size_t, std::string_view, std::string*) override {
    std::lock_guard<std::mutex> lk(mu);
    rb::Reader r(rows);
    while (!r.done()) {
      switch (table) {
        case Table::kObservations: observations.push_back(DecodeObservation(r)); break;
        case Table::kPayloads: (void)DecodePayload(r); break;
        case Table::kScores: scores.push_back(DecodeScore(r)); break;
        case Table::kTraces: traces.push_back(DecodeTraceFact(r)); break;
        default: throw std::runtime_error("table");
      }
    }
    return true;
  }
  std::mutex mu;
  std::vector<Observation> observations;
  std::vector<Score> scores;
  std::vector<TraceFact> traces;
};

std::pair<int, json::Value> Post(int port, const std::string& path, const std::string& body) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
  const std::string req = "POST " + path + " HTTP/1.1\r\nConnection: close\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\n\r\n" + body;
  ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
  std::string raw;
  char b[4096];
  for (ssize_t n; (n = ::recv(fd, b, sizeof b, 0)) > 0;) raw.append(b, static_cast<std::size_t>(n));
  ::close(fd);
  std::string err;
  auto v = json::Parse(raw.substr(raw.find("\r\n\r\n") + 4), &err);
  return {std::stoi(raw.substr(9, 3)), v ? *v : json::Value()};
}

}  // namespace

TEST(ScopeHttpAnnotations, RoutesWriteTypedRows) {
  Rows sink;
  RecorderOptions opt;
  opt.writer_thread = false;
  Recorder rec(sink, opt);
  Annotator ann(rec, "proj", Capture::kDigest);
  server::HttpServer srv({});
  RegisterAnnotationRoutes(srv, ann);
  std::string err;
  ASSERT_TRUE(srv.Start(&err)) << err;
  const int port = srv.port();

  auto [s1, b1] = Post(port, "/v1/scope/steps", R"({"trace_id": "t1", "kind": "tool", "name": "run tests"})");
  ASSERT_EQ(s1, 200);
  const std::string step = b1.find("step_id")->str();
  EXPECT_EQ(step.size(), 16u);
  EXPECT_EQ(Post(port, "/v1/scope/steps/" + step + "/end", R"({"status": "error", "output": "1 failed"})").first, 200);
  EXPECT_EQ(Post(port, "/v1/scope/steps/" + step + "/end", "{}").first, 400);  // already ended
  EXPECT_EQ(Post(port, "/v1/scope/tool_results",
                 R"({"trace_id": "t1", "tool_call_id": "call_9", "tool_name": "bash", "output": "ok", "latency_ms": 12.5})")
                .first,
            200);
  auto [s2, b2] = Post(port, "/v1/scope/scores", R"({"trace_id": "t1", "name": "passed", "value": true, "source": "harness"})");
  ASSERT_EQ(s2, 200);
  const std::string score = b2.find("score_id")->str();
  EXPECT_EQ(Post(port, "/v1/scope/scores", R"({"session_id": "s", "name": "grade", "value": "B"})").first, 200);
  EXPECT_EQ(Post(port, "/v1/scope/scores", R"({"trace_id": "t1", "name": "acc", "value": 0.75})").first, 200);
  EXPECT_EQ(Post(port, "/v1/scope/scores/" + score + "/retract", R"({"comment": "flaky"})").first, 200);
  EXPECT_EQ(Post(port, "/v1/scope/traces/t1/meta", R"({"key": "session_id", "value": "s"})").first, 200);
  // Refusals.
  EXPECT_EQ(Post(port, "/v1/scope/steps", R"({"trace_id": "t1", "kind": "nap", "name": "x"})").first, 400);
  EXPECT_EQ(Post(port, "/v1/scope/steps", "[1]").first, 400);
  EXPECT_EQ(Post(port, "/v1/scope/scores", R"({"trace_id": "t1", "name": "x", "value": [1]})").first, 400);
  EXPECT_EQ(Post(port, "/v1/scope/tool_results", R"({"trace_id": "t1", "tool_call_id": "c", "status": "meh"})").first, 400);
  EXPECT_EQ(Post(port, "/v1/scope/nowhere", "{}").first, 404);
  srv.Stop();
  rec.Flush();
  ASSERT_EQ(sink.observations.size(), 2u);
  EXPECT_EQ(sink.observations[0].type, "SPAN");
  EXPECT_EQ(sink.observations[0].status_code, "ERROR");
  EXPECT_EQ(sink.observations[1].type, "TOOL");
  EXPECT_EQ(sink.observations[1].tool_call_ids, std::vector<std::string>{"call_9"});
  EXPECT_EQ(sink.observations[1].end_ns - sink.observations[1].start_ns, 12'500'000);
  ASSERT_EQ(sink.scores.size(), 4u);
  EXPECT_EQ(sink.scores[0].data_type, "BOOLEAN");
  EXPECT_EQ(sink.scores[1].data_type, "CATEGORICAL");
  EXPECT_EQ(sink.scores[2].data_type, "NUMERIC");
  EXPECT_EQ(sink.scores[3].kind, "retract");
  EXPECT_EQ(sink.scores[3].supersedes, score);
  ASSERT_EQ(sink.traces.size(), 1u);
  EXPECT_EQ(sink.traces[0].session_id, "s");
}
