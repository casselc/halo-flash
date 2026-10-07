// The harness annotation C ABI (include/halo_scope.h) over a memory sink and
// over chDB:
//   - directed: each call writes the typed row it documents (fields, times
//     from a fake clock, capture policy, score facts, trace context);
//   - property: any sequence of calls, NULLs and bad ids included, returns
//     OK exactly as often as rows reach the sink, and never crashes;
//   - concurrent: threads annotating at once lose nothing (run under TSan);
//   - chDB: score and trace facts round-trip through the staging tables, and
//     trace facts resolve to the latest value per field at query time.

#include <hegel/gtest.h>

#include <atomic>
#include <cstring>
#include <deque>
#include <set>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "halo_scope.h"
#include "scope/annotate.hpp"
#include "scope/c_api.hpp"
#include "scope/recorder.hpp"
#include "scope_env.hpp"

namespace gs = hegel::generators;
using namespace halo::scope;
using halo::scope::testing::Env;
using halo::scope::testing::FreshProject;
using halo::scope::testing::Read;

namespace {

/// Decodes and keeps every row delivered, per table.
class RowSink : public Sink {
 public:
  bool Write(Table table, std::string_view rows, std::size_t nrows, std::string_view, std::string*) override {
    std::lock_guard<std::mutex> lk(mu);
    rb::Reader r(rows);
    std::size_t n = 0;
    while (!r.done()) {
      switch (table) {
        case Table::kObservations: observations.push_back(DecodeObservation(r)); break;
        case Table::kPayloads: payloads.push_back(DecodePayload(r)); break;
        case Table::kScores: scores.push_back(DecodeScore(r)); break;
        case Table::kTraces: traces.push_back(DecodeTraceFact(r)); break;
        default: throw std::runtime_error("unknown table");
      }
      ++n;
    }
    if (n != nrows) throw std::runtime_error("row count mismatch");
    return true;
  }
  std::size_t Rows() {
    std::lock_guard<std::mutex> lk(mu);
    return observations.size() + payloads.size() + scores.size() + traces.size();
  }
  std::mutex mu;
  std::vector<Observation> observations;
  std::vector<Payload> payloads;
  std::vector<Score> scores;
  std::vector<TraceFact> traces;
};

struct Fixture {
  explicit Fixture(Capture capture, bool writer_thread = false) {
    RecorderOptions opt;
    opt.writer_thread = writer_thread;
    opt.batch_rows = 64;
    rec = std::make_unique<Recorder>(sink, opt);
    scope = WrapRecorder(*rec, "proj", capture, [this] { return now.load(); });
  }
  ~Fixture() { halo_scope_close(scope); }
  void Flush() { rec->Flush(); }
  RowSink sink;
  std::unique_ptr<Recorder> rec;
  halo_scope* scope = nullptr;
  std::atomic<std::int64_t> now{1'000'000'000};
};

}  // namespace

TEST(ScopeAnnotate, StepBecomesOneSpanFromBeginToEnd) {
  Fixture f(Capture::kDigest);
  char step[HALO_SCOPE_ID_SIZE];
  ASSERT_EQ(halo_scope_step_begin(f.scope, "t1", "gen-span", HALO_SCOPE_STEP_TOOL, "run tests", step), HALO_SCOPE_OK);
  EXPECT_EQ(std::strlen(step), 16u);
  f.now += 2500;
  ASSERT_EQ(f.sink.Rows(), 0u);  // nothing until the step ends
  const std::string out = "3 passed";
  ASSERT_EQ(halo_scope_step_end(f.scope, step, HALO_SCOPE_STATUS_ERROR, out.data(), out.size()), HALO_SCOPE_OK);
  EXPECT_EQ(halo_scope_step_end(f.scope, step, HALO_SCOPE_STATUS_OK, nullptr, 0), HALO_SCOPE_INVALID);  // ended
  f.Flush();
  ASSERT_EQ(f.sink.observations.size(), 1u);
  const auto& o = f.sink.observations[0];
  EXPECT_EQ(o.project, "proj");
  EXPECT_EQ(o.trace_id, "t1");
  EXPECT_EQ(o.span_id, step);
  EXPECT_EQ(o.parent_span_id, "gen-span");
  EXPECT_EQ(o.type, "SPAN");
  EXPECT_EQ(o.name, "run tests");
  EXPECT_EQ(o.tags, std::vector<std::string>{"step:tool"});
  EXPECT_EQ(o.status_code, "ERROR");
  EXPECT_EQ(o.end_ns - o.start_ns, 2500);
  // Digest capture: the hash and size, no content.
  EXPECT_EQ(o.output_refs, std::vector<rb::Hash16>{PayloadHash(out)});
  EXPECT_EQ(o.output_bytes, out.size());
  EXPECT_TRUE(f.sink.payloads.empty());
}

TEST(ScopeAnnotate, ToolResultLinksToItsCallAndCapturePolicyHolds) {
  for (const Capture c : {Capture::kOff, Capture::kDigest, Capture::kFull}) {
    Fixture f(c);
    const std::string out("line\0with nul", 13);
    ASSERT_EQ(halo_scope_tool_result(f.scope, "t", "call_7", "bash", HALO_SCOPE_STATUS_OK, out.data(), out.size(), 400),
              HALO_SCOPE_OK);
    f.Flush();
    ASSERT_EQ(f.sink.observations.size(), 1u);
    const auto& o = f.sink.observations[0];
    EXPECT_EQ(o.type, "TOOL");
    EXPECT_EQ(o.tool_name, "bash");
    EXPECT_EQ(o.tool_call_ids, std::vector<std::string>{"call_7"});
    EXPECT_EQ(o.end_ns, f.now.load());
    EXPECT_EQ(o.end_ns - o.start_ns, 400);
    EXPECT_EQ(o.output_refs.empty(), c == Capture::kOff);
    EXPECT_EQ(o.output_bytes, c == Capture::kOff ? 0u : out.size());
    ASSERT_EQ(f.sink.payloads.size(), c == Capture::kFull ? 1u : 0u);
    if (c == Capture::kFull) {
      EXPECT_EQ(f.sink.payloads[0].content, out);
      EXPECT_EQ(f.sink.payloads[0].hash, o.output_refs[0]);
    }
  }
}

TEST(ScopeAnnotate, ScoresAreTypedAppendOnlyFacts) {
  Fixture f(Capture::kDigest);
  halo_scope_target trace{"t", nullptr, nullptr}, obs{"t", "span9", nullptr}, session{nullptr, nullptr, "s1"};
  char a[HALO_SCOPE_ID_SIZE], b[HALO_SCOPE_ID_SIZE], c[HALO_SCOPE_ID_SIZE], d[HALO_SCOPE_ID_SIZE];
  ASSERT_EQ(halo_scope_score_numeric(f.scope, &trace, "accuracy", 0.5, "judge", "first", nullptr, a), HALO_SCOPE_OK);
  ASSERT_EQ(halo_scope_score_numeric(f.scope, &trace, "accuracy", 0.9, "human", "fixed", a, b), HALO_SCOPE_OK);
  ASSERT_EQ(halo_scope_score_categorical(f.scope, &obs, "verdict", "pass", "rule", nullptr, nullptr, c), HALO_SCOPE_OK);
  ASSERT_EQ(halo_scope_score_boolean(f.scope, &session, "task_success", 1, "harness", nullptr, nullptr, d), HALO_SCOPE_OK);
  ASSERT_EQ(halo_scope_score_retract(f.scope, c, "human", "wrong span"), HALO_SCOPE_OK);
  // Invalid: no subject, an observation without its trace, no name.
  halo_scope_target none{nullptr, nullptr, nullptr}, orphan{nullptr, "span9", "s1"};
  EXPECT_EQ(halo_scope_score_numeric(f.scope, &none, "x", 1, nullptr, nullptr, nullptr, nullptr), HALO_SCOPE_INVALID);
  EXPECT_EQ(halo_scope_score_numeric(f.scope, &orphan, "x", 1, nullptr, nullptr, nullptr, nullptr), HALO_SCOPE_INVALID);
  EXPECT_EQ(halo_scope_score_numeric(f.scope, &trace, "", 1, nullptr, nullptr, nullptr, nullptr), HALO_SCOPE_INVALID);
  EXPECT_EQ(halo_scope_score_retract(f.scope, nullptr, nullptr, nullptr), HALO_SCOPE_INVALID);
  f.Flush();
  ASSERT_EQ(f.sink.scores.size(), 5u);
  const auto& s = f.sink.scores;
  EXPECT_EQ(s[0].data_type, "NUMERIC");
  EXPECT_EQ(s[0].value, 0.5);
  EXPECT_EQ(s[1].supersedes, std::string(a));
  EXPECT_EQ(s[1].score_id, std::string(b));
  EXPECT_EQ(s[2].data_type, "CATEGORICAL");
  EXPECT_EQ(s[2].string_value, "pass");
  EXPECT_EQ(s[2].observation_id, "span9");
  EXPECT_EQ(s[3].data_type, "BOOLEAN");
  EXPECT_EQ(s[3].value, 1.0);
  EXPECT_EQ(s[3].session_id, "s1");
  EXPECT_TRUE(s[3].trace_id.empty());
  EXPECT_EQ(s[4].kind, "retract");
  EXPECT_EQ(s[4].supersedes, std::string(c));
  for (const auto& x : s) EXPECT_EQ(x.kind == "assert", &x != &s[4]);
}

TEST(ScopeAnnotate, TraceMetaSetsFieldsOrMetadata) {
  Fixture f(Capture::kDigest);
  for (const auto& [k, v] : std::vector<std::pair<const char*, const char*>>{
           {"session_id", "s"}, {"user_id", "u"}, {"release", "r1"}, {"name", "fix bug"}, {"tag", "eval"}, {"repo", "halo"}})
    ASSERT_EQ(halo_scope_trace_meta(f.scope, "t", k, v), HALO_SCOPE_OK) << k;
  EXPECT_EQ(halo_scope_trace_meta(f.scope, "t", "", "v"), HALO_SCOPE_INVALID);
  EXPECT_EQ(halo_scope_trace_meta(f.scope, nullptr, "k", "v"), HALO_SCOPE_INVALID);
  f.Flush();
  const auto& t = f.sink.traces;
  ASSERT_EQ(t.size(), 6u);
  EXPECT_EQ(t[0].session_id, "s");
  EXPECT_EQ(t[1].user_id, "u");
  EXPECT_EQ(t[2].release, "r1");
  EXPECT_EQ(t[3].name, "fix bug");
  EXPECT_EQ(t[4].tags, std::vector<std::string>{"eval"});
  EXPECT_EQ(t[5].metadata, (std::vector<std::pair<std::string, std::string>>{{"repo", "halo"}}));
}

// Any call sequence: OK returns == rows delivered (payload rows counted with
// their observation), INVALID writes nothing, and open steps are exactly
// those begun and not ended.
TEST(ScopeAnnotate, EveryOkIsExactlyTheRowsWritten) {
  hegel::test([](hegel::TestCase& tc) {
    const auto capture = static_cast<Capture>(tc.draw(gs::integers<int>({.min_value = 0, .max_value = 2})));
    Fixture f(capture);
    std::vector<std::string> steps, ended, scores;
    std::set<std::string> payloads;  // full capture writes each distinct output once
    std::size_t expect_rows = 0, open = 0;
    std::deque<std::string> held;  // keeps every drawn C string alive for the case
    const auto maybe = [&](const std::vector<std::string>& pool) -> const char* {
      const int pick = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3}));
      if (pick == 0) return nullptr;
      if (pick == 1) return "";
      if (pool.empty() || pick == 2) {
        held.push_back(tc.draw(gs::text({.max_size = 8})));
      } else {
        held.push_back(pool[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = pool.size() - 1}))]);
      }
      return held.back().c_str();
    };
    const auto n = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 40}));
    for (int i = 0; i < n; ++i) {
      f.now += tc.draw(gs::integers<std::int64_t>({.min_value = 0, .max_value = 1000}));
      const std::string out = tc.draw(gs::text({.max_size = 16}));
      // The observation, plus a payload row the first time full capture sees this output.
      const auto out_rows = [&] {
        return std::size_t{1} + (capture == Capture::kFull && !out.empty() && payloads.insert(out).second);
      };
      switch (tc.draw(gs::integers<int>({.min_value = 0, .max_value = 5}))) {
        case 0: {
          char id[HALO_SCOPE_ID_SIZE];
          const std::string trace = maybe({"t"}) ? "t" : "";
          const int kind = tc.draw(gs::integers<int>({.min_value = -1, .max_value = 5}));
          const int rc = halo_scope_step_begin(f.scope, trace.empty() ? nullptr : trace.c_str(), maybe({}),
                                               static_cast<halo_scope_step_kind>(kind), maybe({}), id);
          const bool valid = !trace.empty() && kind >= 0 && kind <= 4;
          ASSERT_EQ(rc, valid ? HALO_SCOPE_OK : HALO_SCOPE_INVALID);
          if (valid) steps.push_back(id), ++open;
          break;
        }
        case 1: {
          std::vector<std::string> pool = steps;  // open steps, and ended ones (must be refused)
          pool.insert(pool.end(), ended.begin(), ended.end());
          const char* id = maybe(pool);
          const std::string sid = id ? id : "";
          const bool known = std::find(steps.begin(), steps.end(), sid) != steps.end();
          const int rc = halo_scope_step_end(f.scope, id, HALO_SCOPE_STATUS_OK, out.data(), out.size());
          ASSERT_EQ(rc, known ? HALO_SCOPE_OK : HALO_SCOPE_INVALID);
          if (known) {
            steps.erase(std::find(steps.begin(), steps.end(), sid));
            ended.push_back(sid);
            --open;
            expect_rows += out_rows();
          }
          break;
        }
        case 2: {
          const char* call = maybe({"call_1"});
          const bool valid = call && *call;
          ASSERT_EQ(halo_scope_tool_result(f.scope, "t", call, maybe({}), HALO_SCOPE_STATUS_ERROR, out.data(), out.size(), 5),
                    valid ? HALO_SCOPE_OK : HALO_SCOPE_INVALID);
          if (valid) expect_rows += out_rows();
          break;
        }
        case 3: {
          const std::string name = tc.draw(gs::text({.max_size = 6}));
          halo_scope_target t{"t", nullptr, nullptr};
          char id[HALO_SCOPE_ID_SIZE];
          const int rc = halo_scope_score_numeric(f.scope, &t, name.c_str(), 1.5, maybe({}), maybe({}), maybe(scores), id);
          ASSERT_EQ(rc, name.c_str()[0] == 0 ? HALO_SCOPE_INVALID : HALO_SCOPE_OK);  // C string: "\0..." is empty
          if (rc == HALO_SCOPE_OK) scores.push_back(id), ++expect_rows;
          break;
        }
        case 4: {
          const char* id = maybe(scores);
          const int rc = halo_scope_score_retract(f.scope, id, nullptr, nullptr);
          ASSERT_EQ(rc, id && *id ? HALO_SCOPE_OK : HALO_SCOPE_INVALID);
          expect_rows += rc == HALO_SCOPE_OK;
          break;
        }
        case 5: {
          const char* key = maybe({"session_id", "tag", "k"});
          const int rc = halo_scope_trace_meta(f.scope, "t", key, maybe({}));
          ASSERT_EQ(rc, key && *key ? HALO_SCOPE_OK : HALO_SCOPE_INVALID);
          expect_rows += rc == HALO_SCOPE_OK;
          break;
        }
      }
    }
    f.Flush();
    halo_scope_stats st{};
    halo_scope_get_stats(f.scope, &st);
    EXPECT_EQ(st.dropped, 0u);
    EXPECT_EQ(f.sink.Rows(), expect_rows);
    EXPECT_EQ(st.rows_written, expect_rows);
    EXPECT_EQ(open, steps.size());
  });
}

// Threads annotate concurrently, ending steps begun on other threads; with
// the writer thread on, nothing is lost or duplicated.
TEST(ScopeAnnotate, ConcurrentHarnessThreadsLoseNothing) {
  Fixture f(Capture::kDigest, /*writer_thread=*/true);
  constexpr int kThreads = 4, kSteps = 200;
  std::mutex mu;
  std::vector<std::string> begun;
  std::atomic<int> ended{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t)
    ts.emplace_back([&, t] {
      for (int i = 0; i < kSteps; ++i) {
        char id[HALO_SCOPE_ID_SIZE];
        ASSERT_EQ(halo_scope_step_begin(f.scope, "t", nullptr, HALO_SCOPE_STEP_PLAN, "p", id), HALO_SCOPE_OK);
        std::string other;
        {
          std::lock_guard<std::mutex> lk(mu);
          begun.push_back(id);
          if (begun.size() > 1 && (i + t) % 2) {
            other = begun.front();
            begun.erase(begun.begin());
          }
        }
        if (!other.empty() && halo_scope_step_end(f.scope, other.c_str(), HALO_SCOPE_STATUS_OK, "x", 1) == HALO_SCOPE_OK)
          ++ended;
        halo_scope_target tg{"t", nullptr, nullptr};
        ASSERT_EQ(halo_scope_score_boolean(f.scope, &tg, "ok", i % 2, nullptr, nullptr, nullptr, nullptr), HALO_SCOPE_OK);
      }
    });
  for (auto& th : ts) th.join();
  for (const auto& id : begun)
    if (halo_scope_step_end(f.scope, id.c_str(), HALO_SCOPE_STATUS_OK, nullptr, 0) == HALO_SCOPE_OK) ++ended;
  f.Flush();
  EXPECT_EQ(ended.load(), kThreads * kSteps);
  std::lock_guard<std::mutex> lk(f.sink.mu);
  EXPECT_EQ(f.sink.observations.size(), static_cast<std::size_t>(kThreads * kSteps));
  EXPECT_EQ(f.sink.scores.size(), static_cast<std::size_t>(kThreads * kSteps));
  std::set<std::string> ids;
  for (const auto& o : f.sink.observations) ids.insert(o.span_id);
  EXPECT_EQ(ids.size(), f.sink.observations.size());
}

TEST(ScopeAnnotate, ScoreAndTraceFactsRoundTripThroughChdb) {
  HALO_REQUIRE_CHDB();
  ChdbSink sink(Conn::Open(*Env().lib, Env().path, &Env().error));
  const std::string project = FreshProject("an");
  RecorderOptions opt;
  opt.writer_thread = false;
  Recorder rec(sink, opt);
  std::int64_t now = 1'790'000'000'000'000'000;
  halo_scope* h = WrapRecorder(rec, project, Capture::kDigest, [&] { return now; });
  halo_scope_target t{"tr", "obs", "sess"};
  char a[HALO_SCOPE_ID_SIZE];
  ASSERT_EQ(halo_scope_score_numeric(h, &t, "acc", 0.25, "judge", "c\n'q'", nullptr, a), HALO_SCOPE_OK);
  ASSERT_EQ(halo_scope_score_categorical(h, &t, "label", "x\ty", "rule", nullptr, a, nullptr), HALO_SCOPE_OK);
  for (const auto& [k, v] : std::vector<std::pair<const char*, const char*>>{
           {"session_id", "s-old"}, {"repo", "a"}, {"tag", "t1"}, {"session_id", "s-new"}, {"repo", "b"}, {"tag", "t2"}}) {
    now += 1000;
    ASSERT_EQ(halo_scope_trace_meta(h, "tr", k, v), HALO_SCOPE_OK);
  }
  halo_scope_close(h);
  rec.Flush();
  const auto st = rec.stats();
  ASSERT_EQ(st.rows_failed, 0u) << st.last_error;

  auto r = Read(std::string("SELECT ") + kScoreColumns + " FROM halo_scope.llm_scores WHERE project = " +
                    SqlQuote(project) + " ORDER BY event_time, data_type DESC",
                "RowBinary");
  ASSERT_TRUE(r.ok) << r.error;
  rb::Reader rr(r.data);
  const Score s0 = DecodeScore(rr), s1 = DecodeScore(rr);
  ASSERT_TRUE(rr.done());
  EXPECT_EQ(s0.score_id, std::string(a));
  EXPECT_EQ(s0.comment, "c\n'q'");
  EXPECT_EQ(s0.value, 0.25);
  EXPECT_EQ(s0.event_ns, 1'790'000'000'000'000'000);
  EXPECT_EQ(s1.string_value, "x\ty");
  EXPECT_EQ(s1.supersedes, std::string(a));

  // Trace facts resolve per field (latest non-empty value) and per metadata
  // key; tags accumulate.
  auto m = Read("SELECT argMaxIf(session_id, start_time, session_id != ''), "
                "arraySort(groupUniqArrayArray(tags)), "
                "argMaxIf(metadata['repo'], start_time, mapContains(metadata, 'repo')) "
                "FROM halo_scope.llm_traces WHERE project = " + SqlQuote(project) + " AND trace_id = 'tr'",
                "TabSeparated");
  ASSERT_TRUE(m.ok) << m.error;
  EXPECT_EQ(m.data, "s-new\t['t1','t2']\tb\n");
}

// One thread per HTTP connection, each recording a little and exiting (audit F6): every exited thread's producer
// is drained and freed, so the rings held stay bounded by the threads alive (none here, after each flush), and no
// record is lost. Before the fix every thread left its 1 MiB pre-faulted ring behind for the life of the process.
TEST(ScopeAnnotate, ShortLivedThreadsLeaveNoRingsBehindAndLoseNothing) {
  Fixture f(Capture::kDigest, /*writer_thread=*/true);
  constexpr int kWaves = 25, kThreads = 8, kScores = 20;
  for (int w = 0; w < kWaves; ++w) {
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t)
      ts.emplace_back([&] {
        for (int i = 0; i < kScores; ++i) {
          halo_scope_target tg{"t", nullptr, nullptr};
          ASSERT_EQ(halo_scope_score_boolean(f.scope, &tg, "ok", i % 2, nullptr, nullptr, nullptr, nullptr),
                    HALO_SCOPE_OK);
        }
      });
    for (auto& th : ts) th.join();
    f.Flush();
    const auto st = f.rec->stats();
    ASSERT_EQ(st.producers, 0u) << "wave " << w << ": exited threads still hold rings";
    ASSERT_EQ(st.producers_freed, static_cast<std::uint64_t>((w + 1) * kThreads));
  }
  const auto st = f.rec->stats();
  constexpr auto kAll = static_cast<std::uint64_t>(kWaves * kThreads * kScores);
  EXPECT_EQ(st.accepted, kAll) << "the counts of freed producers carry over";
  EXPECT_EQ(st.dropped, 0u);
  std::lock_guard<std::mutex> lk(f.sink.mu);
  EXPECT_EQ(f.sink.scores.size(), kAll) << "records of exited threads were lost";
}
