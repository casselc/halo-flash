// Datasets, evaluators, runs and comparisons over HTTP, end to end with the
// real HTTP client: a fake OpenAI-compatible server is the target and the
// judge, halo-scope's routes run the experiment in the background, and the
// client reads progress, items, scores and the paired comparison back.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>

#include "core/json.hpp"
#include "eval_fakes.hpp"
#include "scope/eval_routes.hpp"
#include "scope/metadata.hpp"
#include "scope/prompt_routes.hpp"
#include "scope_env.hpp"
#include "server/http.hpp"

using namespace halo;
using namespace halo::scope;
using json::Value;

namespace {

std::pair<int, Value> Call(int port, const std::string& method, const std::string& target, const std::string& body = "") {
  const auto r = HttpRequestJson(method, "http://127.0.0.1:" + std::to_string(port), target, body, {}, 30000);
  if (!r.transport_ok) throw std::runtime_error(r.error);
  std::string err;
  auto v = json::Parse(r.body, &err);
  return {r.status, v ? *v : Value()};
}

Value WaitDone(int port, const std::string& id) {
  for (int i = 0; i < 500; ++i) {
    auto [s, v] = Call(port, "GET", "/v1/scope/runs/" + id);
    if (s != 200) throw std::runtime_error("GET run " + std::to_string(s));
    if (!v.find("active")->boolean() && v.find("status")->str() != "running" && v.find("status")->str() != "pending")
      return v;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  throw std::runtime_error("run did not finish");
}

}  // namespace

TEST(ScopeEvalRoutes, DatasetToComparisonOverHttp) {
  // The fake model: "careful" decoding answers with the expected text,
  // "sloppy" adds a word; the judge scores 5 for an exact echo, else 2.
  server::HttpServer model({});
  model.Route("POST", "/v1/chat/completions", [](const server::HttpRequest& r) {
    std::string err;
    auto b = json::Parse(r.body, &err);
    const std::string text = halo::scope::testing::UserText(*b);
    std::string out;
    if (b->find("model")->str() == "judge") {
      const bool exact = text.find("A: Say: ") != std::string::npos && text.find(" please") == std::string::npos;
      out = exact ? "{\"score\": 5}" : "{\"score\": 2, \"reasoning\": \"extra words\"}";
    } else {
      const double temp = b->find("temperature") ? b->find("temperature")->number() : 0;
      out = text + (temp > 0.5 && text.back() != '1' ? " please" : "");
    }
    return server::HttpResponse::Json(200, halo::scope::testing::ChatBody(out, "gen-" + std::to_string(text.size()), b->find("model")->str()));
  });
  std::string err;
  ASSERT_TRUE(model.Start(&err)) << err;

  const auto path = halo::scope::testing::TempDir("halo-scope-routes") + "/meta.db";
  auto meta = MetadataStore::Open(path, &err);
  auto store = EvalStore::Open(path, &err);
  ASSERT_TRUE(meta && store) << err;
  server::HttpServer srv({});
  RegisterPromptRoutes(srv, *meta);
  EvalRoutesOptions eo;
  eo.endpoint = "http://127.0.0.1:" + std::to_string(model.port());
  EvalService service(*store, eo);
  service.Register(srv);
  ASSERT_TRUE(srv.Start(&err)) << err;
  const int port = srv.port();

  ASSERT_EQ(Call(port, "POST", "/v1/scope/prompts", R"({"name": "task", "template": "Say: {{input}}"})").first, 200);
  ASSERT_EQ(Call(port, "POST", "/v1/scope/prompts", R"({"name": "grade", "template": "Q: {{input}}\nA: {{output}}"})").first, 200);
  auto [s1, d1] = Call(port, "POST", "/v1/scope/datasets", R"({"name": "greet", "description": "d", "metadata": {"k": "v"}})");
  ASSERT_EQ(s1, 200);
  EXPECT_EQ(Call(port, "POST", "/v1/scope/datasets", R"({"name": "greet"})").first, 409);
  auto [s2, v2] = Call(port, "POST", "/v1/scope/datasets/greet/items",
                       R"({"items": [{"id": "a1", "input": "hi", "expected_output": "Say: hi"}, {"input": "yo1", "expected_output": "Say: yo1"}]})");
  ASSERT_EQ(s2, 200);
  EXPECT_EQ(v2.find("version")->number(), 1);
  auto [s3, v3] = Call(port, "POST", "/v1/scope/datasets/greet/import",
                       "{\"id\": \"a3\", \"input\": \"hey\", \"expected_output\": \"Say: hey\"}\n"
                       "{\"id\": \"gone\", \"input\": \"x\"}\n");
  ASSERT_EQ(s3, 200);
  EXPECT_EQ(Call(port, "DELETE", "/v1/scope/datasets/greet/items/gone").first, 200);
  EXPECT_EQ(Call(port, "POST", "/v1/scope/runs", R"({"name": "early", "dataset": "greet", "target": {"model": "m"}})").first, 409);
  auto [s4, v4] = Call(port, "POST", "/v1/scope/datasets/greet/freeze", "{}");
  ASSERT_EQ(s4, 200);
  EXPECT_EQ(v4.find("item_count")->number(), 3);
  EXPECT_EQ(Call(port, "DELETE", "/v1/scope/datasets/greet/items/a1").first, 404);  // no draft
  auto [s5, items] = Call(port, "GET", "/v1/scope/datasets/greet/items?version=1");
  ASSERT_EQ(s5, 200);
  ASSERT_EQ(items.find("data")->items().size(), 3u);

  EXPECT_EQ(Call(port, "POST", "/v1/scope/evaluators", R"({"name": "em", "kind": "exact", "config": {}})").first, 200);
  EXPECT_EQ(Call(port, "POST", "/v1/scope/evaluators", R"({"name": "bad", "kind": "json_schema", "config": {"schema": {"$ref": "#"}}})").first, 400);
  auto [s6, j6] = Call(port, "POST", "/v1/scope/evaluators",
                       R"({"name": "judge", "kind": "llm_judge", "config": {"prompt": {"name": "grade"}, "model": "judge", "score": {"type": "numeric", "min": 1, "max": 5}}})");
  ASSERT_EQ(s6, 200) << json::DumpPython(j6);
  EXPECT_EQ(j6.find("config")->find("prompt")->find("version")->number(), 1);

  const std::string base = R"("dataset": "greet", "evaluators": ["em", "judge@1"], "target": {"prompt": {"name": "task", "label": "latest"}, "model": "flash", "params": {"temperature": )";
  auto [s7, r7] = Call(port, "POST", "/v1/scope/runs", "{\"name\": \"careful\", " + base + "0}}, \"wait\": true}");
  ASSERT_EQ(s7, 200) << json::DumpPython(r7);
  EXPECT_EQ(r7.find("status")->str(), "completed");
  auto [s8, r8] = Call(port, "POST", "/v1/scope/runs", "{\"name\": \"sloppy\", " + base + "1.0}}, \"concurrency\": 2}");
  ASSERT_EQ(s8, 202);
  const std::string a = r7.find("id")->str(), b = r8.find("id")->str();
  auto done = WaitDone(port, b);
  EXPECT_EQ(done.find("status")->str(), "completed");
  EXPECT_EQ(done.find("progress")->find("done")->number(), 3);
  // Same name, same binding: the same run, resumed (nothing left to do).
  auto [s9, r9] = Call(port, "POST", "/v1/scope/runs", "{\"name\": \"careful\", " + base + "0}}}");
  EXPECT_EQ(s9, 202);
  EXPECT_EQ(r9.find("id")->str(), a);
  EXPECT_TRUE(r9.find("resumed")->boolean());
  EXPECT_EQ(Call(port, "POST", "/v1/scope/runs", "{\"name\": \"careful\", " + base + "0.3}}}").first, 409);

  auto [s10, scores] = Call(port, "GET", "/v1/scope/runs/" + a + "/scores");
  ASSERT_EQ(s10, 200);
  for (const auto& s : scores.find("data")->items()) {
    EXPECT_EQ(s.find("n")->number(), 3);
    EXPECT_EQ(s.find("mean")->number(), s.find("name")->str() == "em" ? 1.0 : 5.0);
  }
  auto [s11, its] = Call(port, "GET", "/v1/scope/runs/" + b + "/items");
  ASSERT_EQ(s11, 200);
  ASSERT_EQ(its.find("data")->items().size(), 3u);
  const auto& first = its.find("data")->items()[0];
  EXPECT_EQ(first.find("item_id")->str(), "a1");
  EXPECT_EQ(first.find("output")->str(), "Say: hi please");
  EXPECT_EQ(first.find("trace_id")->str().size(), 32u);
  const auto& judge = first.find("scores")->items()[1].find("judge");
  ASSERT_TRUE(judge);
  EXPECT_EQ(judge->find("prompt")->str(), "grade");
  EXPECT_EQ(judge->find("raw_output")->str(), "{\"score\": 2, \"reasoning\": \"extra words\"}");

  auto [s12, cmp] = Call(port, "GET", "/v1/scope/compare?a=" + a + "&b=" + b + "&score=judge&items=1");
  ASSERT_EQ(s12, 200) << json::DumpPython(cmp);
  EXPECT_EQ(cmp.find("n")->number(), 3);
  EXPECT_EQ(cmp.find("losses")->number(), 2);
  EXPECT_EQ(cmp.find("ties")->number(), 1);
  EXPECT_DOUBLE_EQ(cmp.find("mean_delta")->number(), -2.0);  // (-3 - 3 + 0) / 3
  EXPECT_DOUBLE_EQ(cmp.find("se_delta")->number(), 1.0);     // sd 1.7320508 / sqrt 3
  EXPECT_EQ(cmp.find("items")->items().size(), 3u);
  EXPECT_EQ(Call(port, "GET", "/v1/scope/compare?a=" + a + "&b=nope&score=em").first, 404);
  auto [s13, list] = Call(port, "GET", "/v1/scope/runs?dataset=greet");
  EXPECT_EQ(list.find("data")->items().size(), 2u);
  EXPECT_EQ(Call(port, "POST", "/v1/scope/runs/" + a + "/resume", "{}").first, 409);  // complete

  EXPECT_EQ(Call(port, "POST", "/v1/scope/datasets/greet/archive", "{}").first, 200);
  EXPECT_EQ(Call(port, "POST", "/v1/scope/datasets/greet/items", R"({"input": "late"})").first, 409);
  auto [s14, all] = Call(port, "GET", "/v1/scope/datasets");
  EXPECT_TRUE(all.find("data")->items().empty());
  auto [s15, archived] = Call(port, "GET", "/v1/scope/datasets?archived=1");
  EXPECT_EQ(archived.find("data")->items().size(), 1u);
  srv.Stop();
  model.Stop();
}
