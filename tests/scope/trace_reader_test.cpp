// The read side over HTTP against chDB: a trace written through the
// recorder (a generation, a harness tool step, scores with a correction and
// a retraction, context facts) comes back from GET /v1/scope/traces/{id}
// with resolved scores and merged context; the list is filtered by session;
// hostile ids are literals, not SQL.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <string>

#include "core/json.hpp"
#include "scope/annotate.hpp"
#include "scope/recorder.hpp"
#include "scope/trace_reader.hpp"
#include "scope_env.hpp"
#include "server/http.hpp"

using namespace halo;
using namespace halo::scope;
using halo::scope::testing::Env;
using halo::scope::testing::FreshProject;

namespace {

std::pair<int, json::Value> Get(int port, const std::string& target) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) throw std::runtime_error("connect");
  const std::string req = "GET " + target + " HTTP/1.1\r\nConnection: close\r\n\r\n";
  ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
  std::string raw;
  char b[8192];
  for (ssize_t n; (n = ::recv(fd, b, sizeof b, 0)) > 0;) raw.append(b, static_cast<std::size_t>(n));
  ::close(fd);
  std::string err;
  auto v = json::Parse(raw.substr(raw.find("\r\n\r\n") + 4), &err);
  return {std::stoi(raw.substr(9, 3)), v ? *v : json::Value()};
}

}  // namespace

TEST(ScopeTraceReader, ATraceComesBackResolved) {
  HALO_REQUIRE_CHDB();
  const std::string project = FreshProject("tr");
  std::string err;
  ChdbSink sink(Conn::Open(*Env().lib, Env().path, &err));
  RecorderOptions opt;
  opt.writer_thread = false;
  Recorder rec(sink, opt);
  std::int64_t now = 1'790'000'000'000'000'000;
  Annotator a(rec, project, Capture::kDigest, [&] { return now += 1000; });
  // A generation (as the server records it) and a harness tool step.
  Observation gen;
  gen.trace_id = "trace-1";
  gen.span_id = "chatcmpl-1";
  gen.start_ns = now;
  gen.end_ns = now + 5000;
  gen.type = "GENERATION";
  gen.name = "chat";
  gen.session_id = "sess-1";
  gen.input_tokens = 30;
  gen.output_tokens = 7;
  gen.tool_call_ids = {"call_1"};
  ASSERT_EQ(a.RecordObservation(gen, "prompt", "output"), Outcome::kOk);
  ASSERT_EQ(a.ToolResult("trace-1", "call_1", "bash", Status::kOk, "done", 500), Outcome::kOk);  // starts after the generation
  // Scores: one corrected, one retracted, one left alone.
  std::string s1, s2, s3;
  ASSERT_EQ(a.ScoreNumeric({"trace-1", "", ""}, "acc", 0.5, "judge", "", "", &s1), Outcome::kOk);
  ASSERT_EQ(a.ScoreNumeric({"trace-1", "", ""}, "acc", 0.9, "human", "", s1, &s2), Outcome::kOk);
  ASSERT_EQ(a.ScoreBoolean({"trace-1", "chatcmpl-1", ""}, "valid", true, "rule", "", "", &s3), Outcome::kOk);
  ASSERT_EQ(a.Retract(s3, "human", "wrong"), Outcome::kOk);
  for (const auto& [k, v] : std::vector<std::pair<const char*, const char*>>{
           {"session_id", "sess-old"}, {"session_id", "sess-1"}, {"tag", "eval"}, {"repo", "a"}, {"repo", "b"}})
    ASSERT_EQ(a.TraceMeta("trace-1", k, v), Outcome::kOk);
  // Another trace in another session.
  Observation other = gen;
  other.trace_id = "trace-2";
  other.span_id = "chatcmpl-2";
  other.session_id = "sess-2";
  other.start_ns = gen.start_ns + 1'000'000;  // a later trace
  other.end_ns = other.start_ns + 5000;
  ASSERT_EQ(a.RecordObservation(other, "", ""), Outcome::kOk);
  rec.Flush();
  ASSERT_EQ(rec.stats().rows_failed, 0u) << rec.stats().last_error;

  TraceReader reader(Conn::Open(*Env().lib, Env().path, &err), project);
  server::HttpServer srv({});
  reader.Register(srv);
  ASSERT_TRUE(srv.Start(&err)) << err;
  auto [s, t] = Get(srv.port(), "/v1/scope/traces/trace-1");
  ASSERT_EQ(s, 200) << json::DumpPython(t);
  const auto& obs = t.find("observations")->items();
  ASSERT_EQ(obs.size(), 2u);
  EXPECT_EQ(obs[0].find("type")->str(), "GENERATION");
  EXPECT_EQ(obs[1].find("type")->str(), "TOOL");
  EXPECT_EQ(obs[1].find("tool_call_ids")->items()[0].str(), "call_1");
  EXPECT_EQ(obs[0].find("output_refs")->items().size(), 1u);  // digest capture, as hex
  EXPECT_EQ(obs[0].find("output_refs")->items()[0].str().size(), 32u);
  const auto& scores = t.find("scores")->items();
  ASSERT_EQ(scores.size(), 1u);  // s1 superseded, s3 retracted
  EXPECT_EQ(scores[0].find("score_id")->str(), s2);
  EXPECT_EQ(scores[0].find("value")->number(), 0.9);
  EXPECT_EQ(t.find("context")->find("session_id")->str(), "sess-1");
  EXPECT_EQ(t.find("context")->find("tags")->items()[0].str(), "eval");
  EXPECT_EQ(t.find("metadata")->find("repo")->str(), "b");
  // Lists, newest first; by session.
  auto [ls, list] = Get(srv.port(), "/v1/scope/traces?limit=10");
  ASSERT_EQ(ls, 200);
  ASSERT_EQ(list.find("data")->items().size(), 2u);
  EXPECT_EQ(list.find("data")->items()[0].find("trace_id")->str(), "trace-2");
  auto [ss, by] = Get(srv.port(), "/v1/scope/traces?session_id=sess%2D1");
  ASSERT_EQ(ss, 200);
  ASSERT_EQ(by.find("data")->items().size(), 1u);
  EXPECT_EQ(by.find("data")->items()[0].find("trace_id")->str(), "trace-1");
  EXPECT_EQ(by.find("data")->items()[0].find("output_tokens")->number(), 7);
  // Ids are literals: quotes and SQL in them find nothing and break nothing.
  EXPECT_EQ(Get(srv.port(), "/v1/scope/traces/x'%20OR%20'1'='1").first, 404);
  EXPECT_EQ(Get(srv.port(), "/v1/scope/traces?session_id=%27%3B%20DROP%20TABLE%20x%3B--").second.find("data")->items().size(), 0u);
  EXPECT_EQ(Get(srv.port(), "/v1/scope/traces?limit=0").first, 400);
  EXPECT_EQ(Get(srv.port(), "/v1/scope/traces/nope").first, 404);
  srv.Stop();
}
