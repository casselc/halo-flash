// halo-scope write path as a Hegel state machine: producers (on the test
// thread and in concurrent bursts) record through bounded rings, the writer
// (its own thread, or driven by a `pump` rule in the deterministic mode)
// batches into a sink, and rules flush, fill a ring to overflow, fail the
// sink, and stop. Swarm: each case enables a random subset of the optional
// rules. Accounting invariant, checked after every flush and at the end:
//   rows read back + rows the sink refused + counted drops == rows produced,
// with no row read back twice and none that was not accepted.
//
// Two sinks: an in-memory sink (no chDB; the configuration run under TSan)
// and chDB itself, read back with SELECT.

#include <hegel/gtest.h>

#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "scope/recorder.hpp"
#include "scope_env.hpp"

namespace gs = hegel::generators;
using namespace halo::scope;
using halo::scope::testing::Env;
using halo::scope::testing::FreshProject;
using halo::scope::testing::Read;
using hegel::stateful::Invariant;
using hegel::stateful::Rule;

namespace {

/// Keeps every delivered span id, and an ordered log of payload hashes
/// delivered or refused; can be told to refuse batches.
class MemorySink : public Sink {
 public:
  struct PayloadEvent {
    std::uint8_t key;
    bool delivered;
  };
  bool Write(Table table, std::string_view rows, std::size_t nrows, std::string_view, std::string* err) override {
    std::lock_guard<std::mutex> lk(mu_);
    rb::Reader r(rows);
    std::size_t n = 0;
    std::vector<std::string> ids;
    std::vector<std::uint8_t> keys;
    while (!r.done()) {
      if (table == Table::kObservations)
        ids.push_back(DecodeObservation(r).span_id);
      else
        keys.push_back(DecodePayload(r).hash[0]);
      ++n;
    }
    if (n != nrows) throw std::runtime_error("row count mismatch in batch");
    for (auto k : keys) events_.push_back({k, !fail_});
    if (fail_) {
      *err = "injected sink failure";
      return false;
    }
    ids_.insert(ids_.end(), ids.begin(), ids.end());
    payload_rows_ += keys.size();
    return true;
  }
  std::size_t EventCount() {
    std::lock_guard<std::mutex> lk(mu_);
    return events_.size();
  }
  std::vector<PayloadEvent> Events() {
    std::lock_guard<std::mutex> lk(mu_);
    return events_;
  }
  std::size_t PayloadRows() {
    std::lock_guard<std::mutex> lk(mu_);
    return payload_rows_;
  }
  void SetFail(bool f) {
    std::lock_guard<std::mutex> lk(mu_);
    fail_ = f;
  }
  std::vector<std::string> Delivered() {
    std::lock_guard<std::mutex> lk(mu_);
    return ids_;
  }

 private:
  std::mutex mu_;
  bool fail_ = false;
  std::vector<std::string> ids_;
  std::vector<PayloadEvent> events_;
  std::size_t payload_rows_ = 0;
};

/// ChdbSink that can refuse batches, for the failure rule.
class FailableChdbSink : public Sink {
 public:
  explicit FailableChdbSink(std::unique_ptr<Conn> c) : inner_(std::move(c)) {}
  bool Write(Table t, std::string_view rows, std::size_t n, std::string_view token, std::string* err) override {
    if (fail_.load()) {
      *err = "injected sink failure";
      return false;
    }
    return inner_.Write(t, rows, n, token, err);
  }
  std::atomic<bool> fail_{false};

 private:
  ChdbSink inner_;
};

struct Model {
  std::uint64_t produced = 0;
  std::uint64_t dropped = 0;           // Record() returned false
  std::set<std::string> accepted;      // Record() returned true
  // accepted payloads: (hash key, sink payload-event count when offered)
  std::vector<std::pair<std::uint8_t, std::size_t>> payloads;
  bool stopped = false;
};

constexpr std::size_t kProducers = 4;

struct WritePath : hegel::stateful::StateMachine<WritePath, Model> {
  bool use_chdb;
  bool threaded = false;
  std::string project;
  std::unique_ptr<MemorySink> mem;
  std::unique_ptr<FailableChdbSink> ch;
  Sink* sink;
  std::unique_ptr<Recorder> rec;
  std::vector<Producer*> producers;
  std::vector<std::uint64_t> offered = std::vector<std::uint64_t>(kProducers, 0);
  std::uint64_t next_id = 0;
  // swarm: which optional rules this case uses
  bool en_burst, en_overflow, en_flush, en_fail, en_stop, en_payload;

  WritePath(hegel::TestCase& tc, bool chdb) : StateMachine({.initial_state = {}}), use_chdb(chdb) {
    project = FreshProject(chdb ? "sm-ch" : "sm-mem");
    if (use_chdb) {
      std::string err;
      auto c = Conn::Open(*Env().lib, Env().path, &err);
      if (!c) throw std::runtime_error(err);
      ch = std::make_unique<FailableChdbSink>(std::move(c));
      sink = ch.get();
    } else {
      mem = std::make_unique<MemorySink>();
      sink = mem.get();
    }
    RecorderOptions opt;
    opt.ring_bytes = tc.draw(gs::sampled_from<std::size_t>({256, 1024, 4096, 65536}));
    opt.batch_rows = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 64}));
    opt.flush_interval = std::chrono::milliseconds(tc.draw(gs::sampled_from<int>({1, 20, 3600000})));
    // false is the deterministic writer the test drives with `pump`; true
    // runs the real writer thread, whose failures cannot replay exactly
    // (Hegel then reports them as flaky). HALO_SCOPE_TEST_WRITER=manual or
    // =thread pins one mode, as the positive controls do.
    const char* mode = std::getenv("HALO_SCOPE_TEST_WRITER");
    const std::string pin = mode ? mode : "";
    opt.writer_thread = pin == "manual" ? false : pin == "thread" ? true : tc.draw(gs::booleans());
    threaded = opt.writer_thread;
    en_burst = tc.draw(gs::booleans());
    en_overflow = tc.draw(gs::booleans());
    en_flush = tc.draw(gs::booleans());
    en_fail = tc.draw(gs::booleans());
    en_stop = tc.draw(gs::booleans());
    en_payload = !use_chdb && tc.draw(gs::booleans());
    opt.payload_cache_entries = tc.draw(gs::sampled_from<std::size_t>({0, 2, 1 << 20}));
    rec = std::make_unique<Recorder>(*sink, opt);
    for (std::size_t i = 0; i < kProducers; ++i) producers.push_back(rec->NewProducer());
  }

  Observation Make(hegel::TestCase& tc) {
    Observation o;
    o.project = project;
    o.span_id = "r" + std::to_string(next_id++);
    o.start_ns = 1790812800LL * 1000000000LL;
    o.name = std::string(tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 300})), 'n');
    return o;
  }

  // Records through producer p; the model learns the outcome.
  void Offer(std::size_t p, const Observation& o, Model& m) {
    ++offered[p];
    ++m.produced;
    if (producers[p]->Record(o))
      m.accepted.insert(o.span_id);
    else
      ++m.dropped;
  }

  std::vector<std::string> Delivered() const {
    if (!use_chdb) return mem->Delivered();
    auto r = Read("SELECT span_id FROM halo_scope.llm_observations WHERE project = " + SqlQuote(project), "TSV");
    if (!r.ok) throw std::runtime_error(r.error);
    std::vector<std::string> ids;
    std::size_t s = 0;
    for (std::size_t i = 0; i < r.data.size(); ++i)
      if (r.data[i] == '\n') {
        ids.push_back(r.data.substr(s, i - s));
        s = i + 1;
      }
    return ids;
  }
  void SetFail(bool f) {
    if (use_chdb)
      ch->fail_.store(f);
    else
      mem->SetFail(f);
  }

  // The payload cache never suppresses a hash that is not durably written:
  // each accepted payload was either delivered before it was offered, or
  // reached the sink (delivered or refused, so counted) afterwards.
  void CheckPayloads() const {
    if (!mem) return;
    const auto ev = mem->Events();
    for (const auto& [key, e0] : state.payloads) {
      bool ok = false;
      for (std::size_t i = 0; i < ev.size() && !ok; ++i)
        ok = ev[i].key == key && (i >= e0 || ev[i].delivered);
      if (!ok)
        throw std::runtime_error("payload " + std::to_string(key) + " suppressed but never written");
    }
  }

  // After a flush or stop: everything accepted was either delivered,
  // refused by the sink or deduplicated as already durable; nothing silent.
  void CheckSettled() const {
    CheckPayloads();
    const auto d = Delivered();
    const auto st = rec->stats();
    const std::size_t pd = mem ? mem->PayloadRows() : 0;
    const std::set<std::string> uniq(d.begin(), d.end());
    if (uniq.size() != d.size()) throw std::runtime_error("duplicate rows delivered");
    if (st.dropped != state.dropped)
      throw std::runtime_error("recorder counted " + std::to_string(st.dropped) + " drops, producers saw " +
                               std::to_string(state.dropped));
    if (d.size() + pd + st.rows_failed + st.payloads_deduped + state.dropped != state.produced)
      throw std::runtime_error("read back " + std::to_string(d.size() + pd) + " + refused " +
                               std::to_string(st.rows_failed) + " + deduplicated " +
                               std::to_string(st.payloads_deduped) + " + dropped " + std::to_string(state.dropped) +
                               " != produced " + std::to_string(state.produced));
  }

  std::vector<Rule<WritePath>> rules() {
    return {
        Rule<WritePath>("produce",
                        [](hegel::TestCase& tc, WritePath& m) {
                          const auto p = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = kProducers - 1}));
                          const auto n = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 20}));
                          for (int i = 0; i < n; ++i) m.Offer(p, m.Make(tc), m.state);
                        }),
        Rule<WritePath>("burst",
                        [](hegel::TestCase& tc, WritePath& m) {
                          if (!m.en_burst) return;
                          // Concurrent producers, one thread per producer handle.
                          const auto threads =
                              tc.draw(gs::integers<std::size_t>({.min_value = 2, .max_value = kProducers}));
                          const auto n = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 200}));
                          std::vector<std::vector<Observation>> work(threads);
                          for (auto& w : work)
                            for (int i = 0; i < n; ++i) w.push_back(m.Make(tc));
                          std::vector<Model> local(threads);
                          std::vector<std::thread> ts;
                          for (std::size_t t = 0; t < threads; ++t)
                            ts.emplace_back([&, t] {
                              for (const auto& o : work[t]) m.Offer(t, o, local[t]);
                            });
                          for (auto& t : ts) t.join();
                          for (auto& l : local) {
                            m.state.produced += l.produced;
                            m.state.dropped += l.dropped;
                            m.state.accepted.insert(l.accepted.begin(), l.accepted.end());
                          }
                        }),
        Rule<WritePath>("fill_to_overflow",
                        [](hegel::TestCase& tc, WritePath& m) {
                          if (!m.en_overflow || m.state.stopped) return;
                          const auto p = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = kProducers - 1}));
                          // Offer until the ring refuses one. With no writer thread
                          // that must happen within capacity/record-size offers; a
                          // writer thread may legitimately keep up, so the bound
                          // only limits the rule's length there.
                          const std::uint64_t before = m.state.dropped;
                          Observation o = m.Make(tc);
                          o.name.assign(200, 'f');
                          const int bound = m.threaded ? 20000 : 1 << 20;
                          for (int i = 0; i < bound && m.state.dropped == before; ++i) {
                            o.span_id = "r" + std::to_string(m.next_id++);
                            m.Offer(p, o, m.state);
                          }
                          if (!m.threaded && m.state.dropped == before)
                            throw std::runtime_error("ring never overflowed");
                        }),
        Rule<WritePath>("produce_payloads",
                        [](hegel::TestCase& tc, WritePath& m) {
                          if (!m.en_payload) return;
                          // A small hash pool, so repeats (resent history) are common.
                          const auto p = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = kProducers - 1}));
                          const auto n = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 10}));
                          for (int i = 0; i < n; ++i) {
                            Payload pl;
                            pl.project = m.project;
                            pl.hash[0] = tc.draw(gs::integers<std::uint8_t>({.min_value = 0, .max_value = 5}));
                            pl.content = "content-" + std::to_string(pl.hash[0]);
                            pl.bytes = static_cast<std::uint32_t>(pl.content.size());
                            const std::size_t e0 = m.mem->EventCount();
                            ++m.offered[p];
                            ++m.state.produced;
                            if (m.producers[p]->Record(pl))
                              m.state.payloads.emplace_back(pl.hash[0], e0);
                            else
                              ++m.state.dropped;
                          }
                        }),
        Rule<WritePath>("pump", [](hegel::TestCase&, WritePath& m) { m.rec->Pump(); }),
        Rule<WritePath>("flush",
                        [](hegel::TestCase&, WritePath& m) {
                          if (!m.en_flush) return;
                          m.rec->Flush();
                          m.CheckSettled();
                        }),
        Rule<WritePath>("toggle_sink_failure",
                        [](hegel::TestCase& tc, WritePath& m) {
                          if (!m.en_fail) return;
                          m.SetFail(tc.draw(gs::booleans()));
                        }),
        Rule<WritePath>("stop",
                        [](hegel::TestCase&, WritePath& m) {
                          if (!m.en_stop) return;
                          m.rec->Stop();
                          m.state.stopped = true;
                          m.CheckSettled();
                        }),
    };
  }

  std::vector<Invariant<WritePath>> invariants() {
    return {
        Invariant<WritePath>("producer_counts_balance",
                             [](const WritePath& m) {
                               for (std::size_t p = 0; p < kProducers; ++p)
                                 if (m.producers[p]->accepted() + m.producers[p]->dropped() != m.offered[p])
                                   throw std::runtime_error("producer " + std::to_string(p) + " lost count");
                             }),
        Invariant<WritePath>("delivered_were_accepted_once",
                             [](const WritePath& m) {
                               const auto d = m.Delivered();
                               std::set<std::string> seen;
                               for (const auto& id : d) {
                                 if (!seen.insert(id).second) throw std::runtime_error("duplicate row " + id);
                                 if (!m.state.accepted.count(id)) throw std::runtime_error("row never accepted " + id);
                               }
                             }),
        Invariant<WritePath>("stopped_refuses_and_counts",
                             [](const WritePath& m) {
                               if (m.state.stopped && m.rec->stats().accepted != m.state.accepted.size() + m.state.payloads.size())
                                 throw std::runtime_error("accepted after stop");
                             }),
    };
  }
};

void RunMachine(hegel::TestCase& tc, bool chdb) {
  WritePath m(tc, chdb);
  hegel::stateful::run(m, tc);
  m.rec->Stop();
  m.SetFail(false);
  m.CheckSettled();
}

}  // namespace

TEST(ScopeWritePath, AccountingHoldsWithMemorySink) {
  // HALO_SCOPE_TEST_CASES raises the budget (the payload-cache positive
  // control needs a failing sink, a resend and a recovery in one case).
  const char* n = std::getenv("HALO_SCOPE_TEST_CASES");
  const std::uint64_t cases = n ? std::strtoull(n, nullptr, 10) : 200;
  hegel::test([](hegel::TestCase& tc) { RunMachine(tc, false); }, {.test_cases = cases});
}

TEST(ScopeWritePath, AccountingHoldsWithChdbSink) {
  HALO_REQUIRE_CHDB();
  hegel::test([](hegel::TestCase& tc) { RunMachine(tc, true); }, {.test_cases = 40});
}

// Recording keeps working when the sink is chDB and the ring is large: a
// plain end-to-end smoke check of row counts.
TEST(ScopeWritePath, ChdbEndToEndCounts) {
  HALO_REQUIRE_CHDB();
  const std::string project = FreshProject("e2e");
  std::string err;
  auto conn = Conn::Open(*Env().lib, Env().path, &err);
  ASSERT_TRUE(conn) << err;
  ChdbSink sink(std::move(conn));
  Recorder rec(sink, {.ring_bytes = 1 << 22, .batch_rows = 1000});
  constexpr int kThreads = 4, kPer = 5000;
  std::vector<std::thread> ts;
  std::atomic<std::uint64_t> ok{0};
  for (int t = 0; t < kThreads; ++t)
    ts.emplace_back([&, t, p = rec.NewProducer()] {
      for (int i = 0; i < kPer; ++i) {
        Observation o;
        o.project = project;
        o.span_id = std::to_string(t) + "-" + std::to_string(i);
        o.start_ns = 1790812800LL * 1000000000LL + i;
        ok += p->Record(o);
      }
    });
  for (auto& t : ts) t.join();
  rec.Flush();
  const auto st = rec.stats();
  EXPECT_EQ(st.accepted + st.dropped, std::uint64_t{kThreads * kPer});
  EXPECT_EQ(st.rows_failed, 0u) << st.last_error;
  auto r = Read("SELECT count(), uniqExact(span_id) FROM halo_scope.llm_observations WHERE project = " +
                    SqlQuote(project),
                "CSV");
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(r.data, std::to_string(ok.load()) + "," + std::to_string(ok.load()) + "\n");
}
