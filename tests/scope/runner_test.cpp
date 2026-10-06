// The experiment runner: resumable and idempotent under injected crashes.
//
// - CrashesAnywhereResumeToTheUninterruptedResult (Hegel): a run is crashed
//   at generated checkpoint boundaries (and its target fails transiently at
//   random) any number of times, then resumed until complete. The final
//   ledger must equal an uninterrupted run's (outputs and every score), every
//   item counted exactly once, every attempt under its own trace, and every
//   committed score mirrored as a chDB fact with its derived id.
// - RealCrashesResume: the same with real crashes: a forked child _exit()s
//   at a random checkpoint (no unwinding, no destructors), the parent
//   resumes.
// - ConcurrentRunMatchesSequential: concurrency 4 gives the sequential
//   result (run under TSan).
// - Attempts are bounded, the run's status says so, and a resume after the
//   cause is fixed completes it; a run whose content changed is refused.
// - The C ABI drives a run with an in-process target and a code hook.

#include <hegel/gtest.h>
#include <sqlite3.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>
#include <set>
#include <string>
#include <vector>

#include "eval_fakes.hpp"
#include "halo_scope.h"
#include "scope/c_api.hpp"
#include "scope/eval_stats.hpp"
#include "scope/recorder.hpp"
#include "scope/runner.hpp"
#include "scope_env.hpp"

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define HALO_TSAN 1
#endif
#endif
#if defined(__SANITIZE_THREAD__)
#define HALO_TSAN 1
#endif

namespace gs = hegel::generators;
using namespace halo;
using namespace halo::scope;
using halo::scope::testing::ChatBody;
using halo::scope::testing::FakePoster;
using halo::scope::testing::UserText;
using json::Value;

namespace {

class ScoreSink : public Sink {
 public:
  bool Write(Table table, std::string_view rows, std::size_t, std::string_view, std::string*) override {
    std::lock_guard<std::mutex> lk(mu);
    rb::Reader r(rows);
    while (!r.done()) {
      if (table == Table::kScores) scores.push_back(DecodeScore(r));
      else if (table == Table::kTraces) traces.push_back(DecodeTraceFact(r));
      else if (table == Table::kObservations) (void)DecodeObservation(r);
      else (void)DecodePayload(r);
    }
    return true;
  }
  std::mutex mu;
  std::vector<Score> scores;
  std::vector<TraceFact> traces;
};

// The world a run executes in: a store with a frozen dataset, three
// evaluators (exact, a code hook, an LLM judge), a fake endpoint that is the
// target and the judge, and a recorder mirroring scores.
struct World {
  std::string path;
  std::unique_ptr<EvalStore> store;
  CodeHooks hooks;
  std::atomic<int> transient_every{0};  // >0: every k-th target call fails with 503
  std::atomic<int> calls{0};
  FakePoster poster;
  ScoreSink sink;
  std::unique_ptr<Recorder> recorder;
  std::unique_ptr<Annotator> annotator;
  std::string run_id;

  explicit World(int items, int max_attempts = 1000)
      : poster([this](const std::string& p, const Value& body, const halo::scope::testing::Recorded&) {
          if (p != "/v1/chat/completions") return HttpResult{true, 404, "{}", ""};
          const std::string model = body.find("model")->str();
          const std::string text = UserText(body);
          if (model == "judge") {  // score = output length mod 5, plus 1
            const auto a = text.find("A: ");
            const auto out = text.substr(a + 3);
            return HttpResult{true, 200, ChatBody("{\"score\": " + std::to_string(out.size() % 5 + 1) + "}", "j", "judge-m"), ""};
          }
          const int k = transient_every.load();
          if (k > 0 && ++calls % k == 0) return HttpResult{true, 503, "{}", ""};
          // The target: upper-cases even-numbered words, reverses the rest.
          std::string out = text;
          if (!text.empty() && (text.back() - '0') % 2 == 0)
            for (auto& ch : out) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
          else
            std::reverse(out.begin(), out.end());
          return HttpResult{true, 200, ChatBody(out, "gen-" + text, "target-m"), ""};
        }) {
    path = halo::scope::testing::TempDir("halo-scope-runner") + "/meta.db";
    std::string err;
    {
      auto meta = MetadataStore::Open(path, &err);
      if (!meta || meta->CreatePromptVersion("judge-p", "Q: {{input}}\nA: {{output}}", "", &err) != 1 ||
          meta->CreatePromptVersion("task", "Say: {{input}}", "", &err) != 1)
        throw std::runtime_error(err);
    }
    store = EvalStore::Open(path, &err);
    if (!store || !store->CreateDataset("ds", "", "", &err)) throw std::runtime_error(err);
    std::string jsonl;
    for (int i = 0; i < items; ++i) {
      const std::string w = "word" + std::to_string(i);
      std::string up = w;
      for (auto& ch : up) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
      jsonl += "{\"id\": \"i" + std::to_string(i) + "\", \"input\": \"" + w + "\", \"expected_output\": \"Say: " +
               (i % 3 ? up : w) + "\"}\n";
    }
    std::size_t n;
    if (!store->ImportJsonl("ds", jsonl, &n, &err) || !store->Freeze("ds", &err)) throw std::runtime_error(err);
    auto mk = [&](const char* name, const char* kind, const char* cfg) {
      auto c = PrepareEvaluatorConfig(*store, kind, cfg, &err);
      if (!c || store->CreateEvaluatorVersion(name, kind, *c, &err) == 0) throw std::runtime_error(err);
    };
    mk("em", "exact", R"({"case_insensitive": false})");
    mk("len", "code", R"({"hook": "len"})");
    mk("judge", "llm_judge", R"({"prompt": {"name": "judge-p"}, "model": "judge", "score": {"type": "numeric", "min": 1, "max": 5}})");
    hooks.Register("len", [](const EvalContext& c) { return EvalOutcome::Numeric(static_cast<double>(c.output.size())); });
    RecorderOptions ro;
    ro.writer_thread = false;
    recorder = std::make_unique<Recorder>(sink, ro);
    annotator = std::make_unique<Annotator>(*recorder, "p", Capture::kDigest);
    RunSpec spec;
    spec.name = "exp";
    spec.dataset = "ds";
    spec.target_json = R"({"prompt": {"name": "task"}, "model": "tgt", "variant": "v0", "params": {"temperature": 0, "max_tokens": 16, "seed": 7}})";
    spec.evaluators = {{"em", 0}, {"len", 0}, {"judge", 0}};
    spec.max_attempts = max_attempts;
    auto run = store->CreateRun(spec, nullptr, &err);
    if (!run) throw std::runtime_error(err);
    run_id = run->id;
  }

  RunnerOptions Options() {
    RunnerOptions o;
    o.target_endpoint = "http://target:1";
    o.env.poster = &poster;
    o.env.default_endpoint = "http://target:1";
    o.env.store = store.get();
    o.env.hooks = &hooks;
    o.annotator = annotator.get();
    return o;
  }

  // (item -> output), (item/name -> value)
  std::pair<std::map<std::string, std::string>, std::map<std::string, double>> Ledger() {
    std::map<std::string, std::string> out;
    std::map<std::string, double> scores;
    for (const auto& r : store->RunItems(run_id))
      if (r.status == "done") out[r.item_id] = r.output;
    for (const auto& s : store->RunScores(run_id)) scores[s.item_id + "/" + s.name] = s.value.value_or(-999);
    return {out, scores};
  }
};

void CheckInvariants(World& w, int items) {
  const auto run = *w.store->GetRun(w.run_id);
  const auto p = w.store->Progress(run);
  if (p.done != items || run.status != "completed")
    throw std::runtime_error("run not complete: done " + std::to_string(p.done) + " status " + run.status);
  const auto scores = w.store->RunScores(w.run_id);
  std::set<std::string> keys;
  for (const auto& s : scores)
    if (!keys.insert(s.item_id + "/" + s.name).second) throw std::runtime_error("score counted twice");
  if (scores.size() != static_cast<std::size_t>(items) * 3) throw std::runtime_error("score count " + std::to_string(scores.size()));
  // Every target request ran under its own attempt's trace.
  std::set<std::string> traces;
  std::map<std::string, std::int64_t> attempts;
  for (const auto& r : w.store->RunItems(w.run_id)) attempts[r.item_id] = r.attempts;
  for (const auto& req : w.poster.requests()) {
    std::string err;
    auto body = json::Parse(req.body, &err);
    if (body->find("model")->str() != "tgt") continue;
    const std::string t = req.Header("X-Halo-Trace-Id");
    if (!traces.insert(t).second) throw std::runtime_error("two target calls under one trace");
    if (req.Header("X-Halo-Session-Id") != w.run_id || req.Header("X-Halo-Prompt") != "task@1" ||
        req.Header("X-Halo-Variant-Id") != "v0")
      throw std::runtime_error("target call lacks its context headers");
    bool known = false;
    for (const auto& [item, n] : attempts)
      for (int a = 1; a <= n; ++a) known = known || AttemptTraceId(w.run_id, item, a) == t;
    if (!known) throw std::runtime_error("target call under an unknown trace");
  }
  // Every committed score was mirrored under its id, on its item's final trace.
  w.recorder->Stop();
  std::map<std::string, const Score*> facts;
  for (const auto& f : w.sink.scores) facts[f.score_id] = &f;
  std::map<std::string, std::string> trace_of;
  for (const auto& r : w.store->RunItems(w.run_id)) trace_of[r.item_id] = r.trace_id;
  for (const auto& s : scores) {
    auto it = facts.find(s.score_id);
    if (it == facts.end()) throw std::runtime_error("committed score " + s.score_id + " never mirrored");
    if (it->second->trace_id != trace_of[s.item_id] || it->second->value != *s.value || it->second->session_id != w.run_id)
      throw std::runtime_error("mirrored fact differs from the ledger");
  }
}

}  // namespace

TEST(ScopeRunner, CrashesAnywhereResumeToTheUninterruptedResult) {
  hegel::test(
      [](hegel::TestCase& tc) {
        const int items = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 6}));
        World ref(items);
        auto r0 = ExecuteRun(*ref.store, ref.poster, ref.run_id, ref.Options());
        if (r0.status != "completed") throw std::runtime_error("reference run: " + r0.status + " " + r0.error);
        const auto expected = ref.Ledger();

        World w(items);
        if (tc.draw(gs::booleans())) w.transient_every = tc.draw(gs::integers<int>({.min_value = 2, .max_value = 5}));
        const int crashes = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 6}));
        int executions = 0;
        for (int c = 0; c < crashes; ++c) {
          const int at = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 40}));
          int seen = 0;
          auto opt = w.Options();
          opt.fault = [&](std::string_view, const std::string&) {
            if (++seen == at) throw SimulatedCrash();
          };
          auto out = ExecuteRun(*w.store, w.poster, w.run_id, opt);
          ++executions;
          if (out.status != "crashed" && out.status != "completed" && out.status != "failed")
            throw std::runtime_error("unexpected status " + out.status + ": " + out.error);
          // Durable progress never goes backwards and never overcounts.
          const auto p = w.store->Progress(*w.store->GetRun(w.run_id));
          if (p.done > items || p.done + p.running + p.error + p.exhausted + p.pending != items)
            throw std::runtime_error("progress does not add up");
        }
        w.transient_every = 0;
        auto out = ExecuteRun(*w.store, w.poster, w.run_id, w.Options());
        if (out.status != "completed") throw std::runtime_error("final resume: " + out.status + " " + out.error);
        if (w.Ledger() != expected) throw std::runtime_error("resumed ledger differs from the uninterrupted run");
        CheckInvariants(w, items);
        // Resuming a complete run does nothing.
        const auto before = w.poster.requests().size();
        auto again = ExecuteRun(*w.store, w.poster, w.run_id, w.Options());
        if (again.status != "completed" || w.poster.requests().size() != before)
          throw std::runtime_error("a complete run did work again");
        (void)executions;
      },
      {.test_cases = std::getenv("HALO_SCOPE_TEST_CASES") ? std::strtoull(std::getenv("HALO_SCOPE_TEST_CASES"), nullptr, 10) : 60});
}

#ifndef HALO_TSAN
TEST(ScopeRunner, RealCrashesResume) {
  // fork() then _exit() at a checkpoint: SQLite sees a process die mid-run.
  hegel::test(
      [](hegel::TestCase& tc) {
        const int items = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 5}));
        World ref(items);
        if (ExecuteRun(*ref.store, ref.poster, ref.run_id, ref.Options()).status != "completed")
          throw std::runtime_error("reference run failed");
        World w(items);
        const int crashes = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 4}));
        for (int c = 0; c < crashes; ++c) {
          const int at = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 30}));
          w.store.reset();  // the child opens its own connection, as a new process would
          const pid_t pid = fork();
          if (pid == 0) {
            std::string err;
            auto store = EvalStore::Open(w.path, &err);
            int seen = 0;
            RunnerOptions o;
            o.target_endpoint = "http://target:1";
            o.env.poster = &w.poster;
            o.env.default_endpoint = "http://target:1";
            o.env.store = store.get();
            o.env.hooks = &w.hooks;
            o.fault = [&](std::string_view, const std::string&) {
              if (++seen == at) _exit(17);
            };
            (void)ExecuteRun(*store, w.poster, w.run_id, o);
            _exit(0);
          }
          int status = 0;
          waitpid(pid, &status, 0);
          if (!WIFEXITED(status) || (WEXITSTATUS(status) != 17 && WEXITSTATUS(status) != 0))
            throw std::runtime_error("child died unexpectedly");
          std::string err;
          w.store = EvalStore::Open(w.path, &err);
          if (!w.store) throw std::runtime_error(err);
        }
        // The parent's fake endpoint never saw the children's calls: only the
        // ledger is compared.
        auto opt = w.Options();
        opt.annotator = nullptr;
        auto out = ExecuteRun(*w.store, w.poster, w.run_id, opt);
        if (out.status != "completed") throw std::runtime_error("resume: " + out.status + " " + out.error);
        if (w.Ledger() != ref.Ledger()) throw std::runtime_error("ledger differs after real crashes");
        const auto scores = w.store->RunScores(w.run_id);
        if (scores.size() != static_cast<std::size_t>(items) * 3) throw std::runtime_error("double count");
      },
      {.test_cases = 25});
}
#endif

TEST(ScopeRunner, ConcurrentRunMatchesSequential) {
  World seq(24), par(24);
  ASSERT_EQ(ExecuteRun(*seq.store, seq.poster, seq.run_id, seq.Options()).status, "completed");
  auto opt = par.Options();
  opt.concurrency = 4;
  std::atomic<int> progress_calls{0};
  opt.on_progress = [&](const RunProgress& p) {
    ++progress_calls;
    EXPECT_LE(p.done, 24);
  };
  auto out = ExecuteRun(*par.store, par.poster, par.run_id, opt);
  ASSERT_EQ(out.status, "completed") << out.error;
  EXPECT_EQ(progress_calls.load(), 24);
  EXPECT_EQ(par.Ledger(), seq.Ledger());
  CheckInvariants(par, 24);
}

TEST(ScopeRunner, TwoExecutorsOnOneRunNeverDoubleCount) {
  // Two executors (two connections, as two processes would have) race over
  // the same run: the attempt check in CompleteItem lets exactly one commit
  // per item.
  World w(16);
  std::string err;
  auto other = EvalStore::Open(w.path, &err);
  ASSERT_TRUE(other) << err;
  auto o1 = w.Options(), o2 = w.Options();
  o1.annotator = o2.annotator = nullptr;
  o1.concurrency = o2.concurrency = 2;
  RunOutcome r1, r2;
  std::thread t1([&] { r1 = ExecuteRun(*w.store, w.poster, w.run_id, o1); });
  std::thread t2([&] { r2 = ExecuteRun(*other, w.poster, w.run_id, o2); });
  t1.join();
  t2.join();
  auto again = ExecuteRun(*w.store, w.poster, w.run_id, o1);  // retries any item the race left running
  if (again.status != "completed") {
    again = ExecuteRun(*w.store, w.poster, w.run_id, o1);
  }
  EXPECT_EQ(again.status, "completed") << again.error << " / " << r1.error << " / " << r2.error;
  std::set<std::string> keys;
  const auto scores = w.store->RunScores(w.run_id);
  for (const auto& s : scores) EXPECT_TRUE(keys.insert(s.item_id + "/" + s.name).second) << "counted twice";
  EXPECT_EQ(scores.size(), 16u * 3);
}

TEST(ScopeRunner, AttemptsAreBoundedAndResumeCompletes) {
  World w(4, /*max_attempts=*/2);
  w.transient_every = 1;  // every target call fails
  auto out = ExecuteRun(*w.store, w.poster, w.run_id, w.Options());
  EXPECT_EQ(out.status, "failed");
  EXPECT_EQ(out.progress.error, 4);  // one attempt used, one left
  out = ExecuteRun(*w.store, w.poster, w.run_id, w.Options());
  EXPECT_EQ(out.status, "failed");
  EXPECT_EQ(out.progress.exhausted, 4);
  EXPECT_EQ(w.store->GetRun(w.run_id)->status, "failed");
  // Exhausted items are not retried.
  const auto calls = w.poster.count("/v1/chat/completions");
  w.transient_every = 0;
  out = ExecuteRun(*w.store, w.poster, w.run_id, w.Options());
  EXPECT_EQ(w.poster.count("/v1/chat/completions"), calls);
  EXPECT_EQ(out.progress.done, 0);

  World ok(3, 3);
  ok.transient_every = 1;
  EXPECT_EQ(ExecuteRun(*ok.store, ok.poster, ok.run_id, ok.Options()).status, "failed");
  ok.transient_every = 0;
  out = ExecuteRun(*ok.store, ok.poster, ok.run_id, ok.Options());
  EXPECT_EQ(out.status, "completed") << out.error;
  for (const auto& r : ok.store->RunItems(ok.run_id)) EXPECT_EQ(r.attempts, 2);
}

TEST(ScopeRunner, ExactMatchAndJudgeScoresAreRight) {
  World w(6);
  ASSERT_EQ(ExecuteRun(*w.store, w.poster, w.run_id, w.Options()).status, "completed");
  // Items 0, 3 (i % 3 == 0) expect the lowercase word; even items are upper-cased.
  std::map<std::string, ScoreRow> by;
  for (const auto& s : w.store->RunScores(w.run_id)) by[s.item_id + "/" + s.name] = s;
  for (int i = 0; i < 6; ++i) {
    const std::string id = "i" + std::to_string(i);
    std::string out = "Say: word" + std::to_string(i);
    if (i % 2 == 0)
      for (auto& ch : out) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    else
      std::reverse(out.begin(), out.end());
    std::string word = "word" + std::to_string(i);
    if (i % 3)
      for (auto& ch : word) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    const std::string expected = "Say: " + word;
    EXPECT_EQ(*by[id + "/em"].value, out == expected ? 1.0 : 0.0) << id;
    EXPECT_EQ(*by[id + "/len"].value, static_cast<double>(out.size()));
    EXPECT_EQ(*by[id + "/judge"].value, static_cast<double>(out.size() % 5 + 1));
    EXPECT_EQ(by[id + "/judge"].judge_model, "judge-m");
    EXPECT_EQ(by[id + "/judge"].judge_prompt, "judge-p");
    EXPECT_EQ(by[id + "/judge"].judge_prompt_version, 1);
    EXPECT_FALSE(by[id + "/judge"].raw_output.empty());
  }
  // The per-item seed is fixed and differs across items.
  std::set<std::string> seeds;
  for (const auto& r : w.poster.requests()) {
    std::string err;
    auto b = json::Parse(r.body, &err);
    if (b->find("model")->str() == "tgt") seeds.insert(b->find("seed")->int_digits());
  }
  EXPECT_EQ(seeds.size(), 6u);
}

TEST(ScopeRunner, RefusesARunWhoseContentChanged) {
  World w(2);
  auto opt = w.Options();
  {
    sqlite3* db = nullptr;
    sqlite3_open(w.path.c_str(), &db);
    ASSERT_EQ(sqlite3_exec(db, "DROP TRIGGER prompt_versions_immutable", nullptr, nullptr, nullptr), SQLITE_OK);
    ASSERT_EQ(sqlite3_exec(db, "UPDATE prompt_versions SET template = 'Shout: {{input}}'", nullptr, nullptr, nullptr),
              SQLITE_OK);
    sqlite3_close(db);
  }
  auto out = ExecuteRun(*w.store, w.poster, w.run_id, opt);
  EXPECT_EQ(out.status, "refused");
  EXPECT_NE(out.error.find("binding"), std::string::npos) << out.error;
  EXPECT_EQ(w.poster.requests().size(), 0u);
}

namespace {

void LenHook(void*, const halo_scope_eval_input* in, halo_scope_eval_result* r) {
  halo_scope_eval_set_numeric(r, static_cast<double>(in->output_size), "length");
}

void Target(void* user, const halo_scope_target_call* call, halo_scope_target_reply* reply) {
  ++*static_cast<int*>(user);
  std::string err;
  auto body = json::Parse(call->request_json, &err);
  const std::string text = UserText(*body);
  if (std::string(call->trace_id).size() != 32) return halo_scope_target_set_error(reply, 0, "bad trace id");
  halo_scope_target_set_output(reply, text.data(), text.size(), "cb-model", "stop", 5, 2);
}

}  // namespace

TEST(ScopeRunner, CAbiRunsAnExperiment) {
  char err[512] = {};
  const std::string path = halo::scope::testing::TempDir("halo-scope-cabi") + "/meta.db";
  halo_scope_evals* ev = halo_scope_evals_open(path.c_str(), nullptr, err, sizeof err);
  ASSERT_TRUE(ev) << err;
  ASSERT_EQ(halo_scope_dataset_create(ev, "d", "desc", nullptr, err, sizeof err), HALO_SCOPE_OK) << err;
  char id[HALO_SCOPE_ID_SIZE] = {};
  ASSERT_EQ(halo_scope_dataset_add_item(ev, "d", nullptr, "\"hello\"", "\"hello\"", nullptr, id, err, sizeof err), HALO_SCOPE_OK);
  EXPECT_EQ(std::strlen(id), 16u);
  const std::string jsonl = "{\"id\": \"x\", \"input\": \"abc\", \"expected_output\": \"abd\"}\n";
  std::size_t count = 0;
  ASSERT_EQ(halo_scope_dataset_import_jsonl(ev, "d", jsonl.data(), jsonl.size(), &count, err, sizeof err), HALO_SCOPE_OK);
  EXPECT_EQ(count, 1u);
  std::int64_t version = 0;
  char hash[33] = {};
  ASSERT_EQ(halo_scope_dataset_freeze(ev, "d", &version, hash, err, sizeof err), HALO_SCOPE_OK) << err;
  EXPECT_EQ(version, 1);
  EXPECT_EQ(std::strlen(hash), 32u);
  EXPECT_EQ(halo_scope_dataset_add_item(ev, "d", "y", "1", nullptr, nullptr, nullptr, err, sizeof err), HALO_SCOPE_OK);  // a new draft
  EXPECT_EQ(halo_scope_evaluator_create(ev, "em", "exact", "{}", &version, err, sizeof err), HALO_SCOPE_OK);
  EXPECT_EQ(halo_scope_evaluator_create(ev, "len", "code", R"({"hook": "len"})", &version, err, sizeof err), HALO_SCOPE_OK);
  EXPECT_EQ(halo_scope_evaluator_create(ev, "bad", "regex", R"({"pattern": "("})", &version, err, sizeof err), HALO_SCOPE_INVALID);
  int calls = 0;
  halo_scope_run_options o{};
  o.name = "r1";
  o.dataset = "d";
  o.target_json = R"({"model": "cb"})";
  o.evaluators = "em, len@1";
  o.target_fn = Target;
  o.target_user = &calls;
  char run[HALO_SCOPE_ID_SIZE] = {};
  // The hook is not registered yet: refused before any work.
  EXPECT_EQ(halo_scope_run_experiment(ev, &o, run, err, sizeof err), HALO_SCOPE_INVALID);
  EXPECT_NE(std::string(err).find("not registered"), std::string::npos) << err;
  EXPECT_EQ(calls, 0);
  ASSERT_EQ(halo_scope_evals_register_hook(ev, "len", LenHook, nullptr), HALO_SCOPE_OK);
  ASSERT_EQ(halo_scope_run_experiment(ev, &o, run, err, sizeof err), HALO_SCOPE_OK) << err;
  EXPECT_EQ(calls, 2);
  ASSERT_EQ(halo_scope_run_experiment(ev, &o, run, err, sizeof err), HALO_SCOPE_OK) << err;  // resume: no work
  EXPECT_EQ(calls, 2);
  o.name = "r2";
  o.target_json = R"({"model": "cb2"})";
  char run2[HALO_SCOPE_ID_SIZE] = {};
  ASSERT_EQ(halo_scope_run_experiment(ev, &o, run2, err, sizeof err), HALO_SCOPE_OK) << err;
  halo_scope_comparison c{};
  ASSERT_EQ(halo_scope_run_compare(ev, run, run2, "em", &c, err, sizeof err), HALO_SCOPE_OK) << err;
  EXPECT_EQ(c.n, 2u);
  EXPECT_EQ(c.ties, 2u);
  EXPECT_EQ(c.mean_a, 0.5);
  EXPECT_EQ(c.mean_delta, 0.0);
  size_t needed = 0;
  char small[8];
  ASSERT_EQ(halo_scope_run_summary_json(ev, run, small, sizeof small, &needed), HALO_SCOPE_OK);
  std::string full(needed + 1, '\0');
  ASSERT_EQ(halo_scope_run_summary_json(ev, run, full.data(), full.size(), &needed), HALO_SCOPE_OK);
  EXPECT_NE(full.find("\"status\": \"completed\""), std::string::npos) << full;
  halo_scope_evals_close(ev);
}
