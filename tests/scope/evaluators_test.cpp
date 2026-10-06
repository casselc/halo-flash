// halo-scope evaluators: the deterministic checks (exact, regex,
// json_schema, numeric) on directed cases and Hegel properties, the judge's
// verdict extraction, and the llm_judge and decider evaluators against a
// fake endpoint (request shape, auditable judge fields, transient versus
// deterministic failures).

#include <hegel/gtest.h>

#include <cmath>
#include <string>

#include "eval_fakes.hpp"
#include "scope/evaluators.hpp"
#include "scope/json_schema.hpp"
#include "scope_env.hpp"

namespace gs = hegel::generators;
using namespace halo;
using namespace halo::scope;
using halo::scope::testing::ChatBody;
using halo::scope::testing::FakePoster;
using halo::scope::testing::UserText;
using json::Value;

namespace {

struct Case {
  DatasetItem item;
  EvalContext ctx;
  Case(const std::string& input_json, const std::string& expected_json, const std::string& output) {
    item.id = "item-1";
    item.input_json = input_json;
    item.expected_json = expected_json;
    std::string err;
    ctx.item = &item;
    ctx.input = *json::Parse(input_json, &err);
    ctx.expected = *json::Parse(expected_json, &err);
    ctx.metadata = Value::MakeObject();
    ctx.output = output;
    ctx.run_id = "run-1";
    ctx.attempt = 1;
  }
};

EvalOutcome Eval(const std::string& kind, const std::string& config, Case& c, const EvalEnv& env = {}) {
  return RunEvaluator(EvaluatorDef{"ev", 1, kind, config}, c.ctx, env);
}

bool Passed(const EvalOutcome& o) { return o.status == EvalOutcome::Status::kScored && o.row.value && *o.row.value == 1; }

std::unique_ptr<EvalStore> Store() {
  std::string err;
  auto s = EvalStore::Open(halo::scope::testing::TempDir("halo-scope-evaluators") + "/meta.db", &err);
  if (!s) throw std::runtime_error(err);
  return s;
}

}  // namespace

TEST(ScopeEvaluators, DeterministicChecks) {
  {
    Case c("\"q\"", "\"Paris\"", "  Paris\n");
    EXPECT_TRUE(Passed(Eval("exact", "{}", c)));
    EXPECT_FALSE(Passed(Eval("exact", R"({"trim": false})", c)));
    c.ctx.output = "PARIS is the capital.\nMore text";
    EXPECT_FALSE(Passed(Eval("exact", R"({"case_insensitive": true})", c)));
    c.ctx.expected = Value::String("paris is the capital.");
    EXPECT_TRUE(Passed(Eval("exact", R"({"case_insensitive": true, "first_line": true})", c)));
    Case j("\"q\"", R"({"a": [1, 2], "b": null})", "```json\n{\"b\": null, \"a\": [1, 2]}\n```");
    EXPECT_TRUE(Passed(Eval("exact", R"({"json": true})", j)));
    Case none("\"q\"", "null", "x");
    EXPECT_EQ(Eval("exact", "{}", none).status, EvalOutcome::Status::kFailed);  // nothing to compare against
  }
  {
    Case c("\"q\"", "\"^ab+c$\"", "abbbc");
    EXPECT_TRUE(Passed(Eval("regex", R"({"pattern": "b{3}"})", c)));
    EXPECT_FALSE(Passed(Eval("regex", R"({"pattern": "b{3}", "negate": true})", c)));
    EXPECT_TRUE(Passed(Eval("regex", R"({"from_expected": true})", c)));
    EXPECT_FALSE(Passed(Eval("regex", R"({"pattern": "bbb", "full_match": true})", c)));
  }
  {
    Case c("\"q\"", "42", "The answer is 41.99 (approximately) or -3");
    EXPECT_FALSE(Passed(Eval("numeric", "{}", c)));  // the last number, -3
    EXPECT_TRUE(Passed(Eval("numeric", R"({"pick": "first", "abs_tol": 0.05})", c)));
    c.ctx.expected = Value::String("1,000");
    c.ctx.output = "about 1,000.4 units";
    EXPECT_TRUE(Passed(Eval("numeric", R"({"rel_tol": 0.001})", c)));
    c.ctx.output = "no numbers";
    EXPECT_FALSE(Passed(Eval("numeric", "{}", c)));
  }
  {
    const char* schema = R"({"schema": {"type": "object", "required": ["name", "tags"], "additionalProperties": false,
      "properties": {"name": {"type": "string", "minLength": 1}, "tags": {"type": "array", "items": {"enum": ["a", "b"]},
      "maxItems": 2}, "n": {"type": "integer", "minimum": 0}}}})";
    Case c("\"q\"", "null", R"({"name": "x", "tags": ["a"], "n": 3})");
    EXPECT_TRUE(Passed(Eval("json_schema", schema, c)));
    for (const char* bad : {R"({"name": "", "tags": []})", R"({"name": "x"})", R"({"name": "x", "tags": ["c"]})",
                            R"({"name": "x", "tags": [], "n": -1})", R"({"name": "x", "tags": [], "n": 1.5})",
                            R"({"name": "x", "tags": [], "extra": 1})", "not json"}) {
      c.ctx.output = bad;
      const auto o = Eval("json_schema", schema, c);
      EXPECT_FALSE(Passed(o)) << bad;
      EXPECT_FALSE(o.row.comment.empty()) << bad;  // names the violation
    }
  }
}

TEST(ScopeEvaluators, ConfigsAreValidatedAndPinned) {
  auto store = Store();
  std::string err;
  EXPECT_FALSE(PrepareEvaluatorConfig(*store, "regex", R"({"pattern": "("})", &err));
  EXPECT_FALSE(PrepareEvaluatorConfig(*store, "json_schema", R"({"schema": {"type": "object", "if": {}}})", &err));
  EXPECT_NE(err.find("unsupported keyword"), std::string::npos) << err;
  EXPECT_FALSE(PrepareEvaluatorConfig(*store, "telepathy", "{}", &err));
  EXPECT_FALSE(PrepareEvaluatorConfig(*store, "llm_judge", R"({"prompt": {"name": "j"}, "model": "m", "score": {}})", &err));
  // A judge prompt label resolves to the version it names now.
  auto path = halo::scope::testing::TempDir("halo-scope-pin2") + "/m.db";
  auto meta = MetadataStore::Open(path, &err);
  ASSERT_EQ(meta->CreatePromptVersion("j", "v1 {{output}}", "", &err), 1);
  ASSERT_EQ(meta->CreatePromptVersion("j", "v2 {{output}}", "", &err), 2);
  ASSERT_TRUE(meta->SetLabel("j", "production", 1, &err));
  auto s2 = EvalStore::Open(path, &err);
  auto c = PrepareEvaluatorConfig(*s2, "llm_judge",
                                  R"({"prompt": {"name": "j", "label": "production"}, "model": "m", "score": {"type": "numeric"}})", &err);
  ASSERT_TRUE(c) << err;
  EXPECT_NE(c->find(R"("prompt":{"name":"j","version":1})"), std::string::npos) << *c;
}

TEST(ScopeEvaluators, ExactAndRegexProperties) {
  hegel::test(
      [](hegel::TestCase& tc) {
        const auto s = tc.draw(gs::text({.max_size = 40}));
        Case c("\"q\"", json::DumpPython(Value::String(s)), s);
        if (!Passed(Eval("exact", R"({"trim": false})", c))) throw std::runtime_error("exact(x, x) failed");
        if (!Passed(Eval("exact", "{}", c))) throw std::runtime_error("trimmed exact(x, x) failed");
        c.ctx.output = s + "!";
        if (Passed(Eval("exact", R"({"trim": false})", c))) throw std::runtime_error("exact(x, x!) passed");
        // A regex of the escaped literal finds the text in any surrounding.
        std::string esc;
        for (char ch : s) {
          if (std::string("\\^$.|?*+()[]{}").find(ch) != std::string::npos) esc += '\\';
          esc += ch;
        }
        const auto pre = tc.draw(gs::text({.max_size = 5}));
        c.ctx.output = pre + s + "tail";
        Value cfg = Value::MakeObject();
        cfg.set("pattern", Value::String(esc));
        if (!Passed(Eval("regex", json::DumpPython(cfg), c))) throw std::runtime_error("literal regex missed: " + esc);
      },
      {.test_cases = 200});
}

TEST(ScopeEvaluators, NumericToleranceProperty) {
  hegel::test(
      [](hegel::TestCase& tc) {
        const auto e = tc.draw(gs::integers<int>({.min_value = -100000, .max_value = 100000}));
        const auto d = tc.draw(gs::integers<int>({.min_value = -50, .max_value = 50}));
        const auto tol = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 30}));
        Case c("\"q\"", std::to_string(e), "so the result is " + std::to_string(e + d));
        Value cfg = Value::MakeObject();
        cfg.set("abs_tol", Value::Int(tol));
        const bool got = Passed(Eval("numeric", json::DumpPython(cfg), c));
        if (got != (std::abs(d) <= tol)) throw std::runtime_error("tolerance decided wrongly");
      },
      {.test_cases = 300});
}

TEST(ScopeEvaluators, JsonSchemaTypeProperty) {
  // A document satisfies the schema of its own types and fails when the
  // types are swapped for another.
  hegel::test(
      [](hegel::TestCase& tc) {
        const std::vector<std::pair<std::string, std::string>> samples = {
            {"null", "null"}, {"true", "boolean"}, {"3", "integer"}, {"2.5", "number"},
            {"\"s\"", "string"}, {"[1]", "array"}, {"{\"a\": 1}", "object"}};
        const auto i = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = samples.size() - 1}));
        const auto j = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = samples.size() - 1}));
        std::string err;
        auto doc = json::Parse(samples[i].first, &err);
        auto schema = json::Parse("{\"type\": \"" + samples[j].second + "\"}", &err);
        std::string why;
        const bool ok = ValidateJsonSchema(*schema, *doc, &why);
        const bool expect = i == j || (samples[j].second == "number" && samples[i].second == "integer");
        if (ok != expect) throw std::runtime_error(samples[i].first + " vs " + samples[j].second);
      },
      {.test_cases = 100});
}

TEST(ScopeEvaluators, VerdictExtraction) {
  auto v = ExtractVerdict("<think>maybe {\"score\": 1}</think>\nThe answer {is} fine.\n```json\n{\"score\": 4, "
                          "\"reasoning\": \"uses {braces} and \\\"quotes\\\"\"}\n```");
  ASSERT_TRUE(v);
  EXPECT_EQ(v->find("score")->number(), 4);
  EXPECT_FALSE(ExtractVerdict("no json here"));
  EXPECT_FALSE(ExtractVerdict("{\"verdict\": 3}"));
  v = ExtractVerdict("{\"score\": 1} then {\"score\": 2}");
  EXPECT_EQ(v->find("score")->number(), 2);
  std::string err;
  EXPECT_EQ(*RenderTemplate("a {{ x }} b {{y}}", {{"x", "1"}, {"y", "{{x}}"}}, &err), "a 1 b {{x}}");
  EXPECT_FALSE(RenderTemplate("{{missing}}", {}, &err));
  EXPECT_FALSE(RenderTemplate("{{open", {}, &err));
}

TEST(ScopeEvaluators, JudgeRecordsEverythingNeededToAuditAScore) {
  const auto path = halo::scope::testing::TempDir("halo-scope-judge") + "/m.db";
  std::string err;
  auto meta = MetadataStore::Open(path, &err);
  ASSERT_EQ(meta->CreatePromptVersion("judge", "Q: {{input}}\nA: {{output}}\nRef: {{expected}}\nTopic: {{topic}}",
                                      R"({"system": "You grade answers."})", &err),
            1);
  auto store = EvalStore::Open(path, &err);
  std::string reply = "Thinking... {\"score\": 4, \"reasoning\": \"close\"}";
  int status = 200;
  FakePoster poster([&](const std::string& p, const Value&, const halo::scope::testing::Recorded&) {
    if (p != "/v1/chat/completions") return HttpResult{true, 404, "{}", ""};
    if (status == 0) return HttpResult{false, 0, "", "connection refused"};
    return HttpResult{true, status, status == 200 ? ChatBody(reply, "gen-judge", "judge-model-x") : "{}", ""};
  });
  EvalEnv env;
  env.poster = &poster;
  env.default_endpoint = "http://judge:1";
  env.store = store.get();
  auto cfg = PrepareEvaluatorConfig(*store, "llm_judge",
                                    R"({"prompt": {"name": "judge"}, "model": "flash", "score": {"type": "numeric", "min": 1, "max": 5}, "max_tokens": 64})", &err);
  ASSERT_TRUE(cfg) << err;
  Case c(R"({"question": "2+2?", "topic": "math"})", "\"4\"", "four");
  const EvaluatorDef def{"quality", 3, "llm_judge", *cfg};
  auto o = RunEvaluator(def, c.ctx, env);
  ASSERT_EQ(o.status, EvalOutcome::Status::kScored) << o.row.error;
  EXPECT_EQ(*o.row.value, 4);
  EXPECT_EQ(o.row.comment, "close");
  EXPECT_EQ(o.row.judge_model, "judge-model-x");
  EXPECT_EQ(o.row.judge_prompt, "judge");
  EXPECT_EQ(o.row.judge_prompt_version, 1);
  EXPECT_EQ(o.row.raw_output, reply);
  EXPECT_EQ(o.row.judge_trace_id, EvaluatorTraceId("run-1", "item-1", 1, "quality"));
  EXPECT_EQ(o.row.evaluator_version, 3);
  const auto req = poster.requests().back();
  EXPECT_EQ(req.url, "http://judge:1");
  EXPECT_EQ(req.Header("X-Halo-Trace-Id"), o.row.judge_trace_id);
  EXPECT_EQ(req.Header("X-Halo-Prompt"), "judge@1");
  auto body = *json::Parse(req.body, &err);
  EXPECT_EQ(body.find("model")->str(), "flash");
  EXPECT_EQ(body.find("max_tokens")->number(), 64);
  EXPECT_EQ(body.find("messages")->items()[0].find("content")->str(), "You grade answers.");
  EXPECT_EQ(UserText(body), "Q: {\"question\":\"2+2?\",\"topic\":\"math\"}\nA: four\nRef: 4\nTopic: math");

  reply = "{\"score\": 9}";  // out of range: a recorded failure, not a value
  o = RunEvaluator(def, c.ctx, env);
  EXPECT_EQ(o.status, EvalOutcome::Status::kFailed);
  EXPECT_FALSE(o.row.value);
  EXPECT_EQ(o.row.raw_output, reply);
  reply = "I refuse to answer in JSON.";
  o = RunEvaluator(def, c.ctx, env);
  EXPECT_EQ(o.status, EvalOutcome::Status::kFailed);
  EXPECT_EQ(o.row.raw_output, reply);
  status = 503;
  EXPECT_EQ(RunEvaluator(def, c.ctx, env).status, EvalOutcome::Status::kTransient);
  status = 0;
  EXPECT_EQ(RunEvaluator(def, c.ctx, env).status, EvalOutcome::Status::kTransient);
  status = 400;
  EXPECT_EQ(RunEvaluator(def, c.ctx, env).status, EvalOutcome::Status::kFailed);
}

TEST(ScopeEvaluators, DeciderReadsTypedAnswers) {
  std::string answer;
  FakePoster poster([&](const std::string& p, const Value& body, const halo::scope::testing::Recorded&) {
    if (p != "/v1/systemone" || !body.find("questions") || !body.find("state")) return HttpResult{true, 422, "{}", ""};
    return HttpResult{true, 200, answer, ""};
  });
  auto store = Store();
  EvalEnv env;
  env.poster = &poster;
  env.decider_endpoint = "http://decider:2";
  env.store = store.get();
  Case c("\"Is the sky blue?\"", "\"yes\"", "Yes, it is.");
  answer = R"({"answers": {"correct": {"type": "noul", "noul": 0.83, "confidence": 0.66}}, "latency_ms": 12.5})";
  auto o = RunEvaluator({"dec", 1, "decider", R"({"question": {"name": "correct", "type": "noul", "instructions": "Is the output correct?"}})"},
                        c.ctx, env);
  ASSERT_EQ(o.status, EvalOutcome::Status::kScored);
  EXPECT_EQ(o.row.data_type, "NUMERIC");
  EXPECT_DOUBLE_EQ(*o.row.value, 0.83);
  EXPECT_EQ(o.row.raw_output, answer);
  o = RunEvaluator({"dec", 1, "decider", R"({"question": {"name": "correct", "type": "noul"}, "threshold": 0.9})"}, c.ctx, env);
  EXPECT_EQ(o.row.data_type, "BOOLEAN");
  EXPECT_EQ(*o.row.value, 0);
  answer = R"({"answers": {"tone": {"type": "choice", "choice": "polite", "confidence": 0.9}}})";
  o = RunEvaluator({"dec", 1, "decider", R"({"question": {"name": "tone", "type": "choice"}})"}, c.ctx, env);
  EXPECT_EQ(o.row.data_type, "CATEGORICAL");
  EXPECT_EQ(o.row.string_value, "polite");
  answer = R"({"answers": {"q": {"type": "score", "score": 3.5, "confidence": 0.4}}})";
  o = RunEvaluator({"dec", 1, "decider", R"({"question": {"name": "q", "type": "score"}, "state": "Answer: {{output}}"})"}, c.ctx, env);
  EXPECT_EQ(*o.row.value, 3.5);
  std::string err;
  auto sent = *json::Parse(poster.requests().back().body, &err);
  EXPECT_EQ(sent.find("state")->str(), "Answer: Yes, it is.");
  answer = R"({"answers": {}})";
  EXPECT_EQ(RunEvaluator({"dec", 1, "decider", R"({"question": {"name": "q", "type": "score"}})"}, c.ctx, env).status,
            EvalOutcome::Status::kFailed);
}
