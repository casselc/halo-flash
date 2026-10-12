// The executor's operation records (server/operations.hpp) against their model (spec/operation/operation.qnt).
//
// OperationsModel.PropertiesHoldUnderTheModelsActions is a Hegel state machine with the model's actors: a platform
// (the outbox) that submits attempts of two operations under a reservation, resends, cancels, asks status, fetches,
// commits durably, acknowledges and crashes (losing what it fetched); a network that loses, delays and duplicates
// every verb; and the executor, an Operations instance that finishes attempts it accepted (a cancelled one with its
// actual usage) and restarts (a new instance under a new incarnation). The platform keeps the contract's rules
// (commit before ack, a refusal closes only its copy, settle never-accepted only through a terminal answer, resend
// exactly as first sent). After every action the model's properties must hold: runsOnce, accounted,
// neverAcceptedTrue, noReceiptLost, promiseKept, records within the ceiling and reservations within the budget.
// Each executor rule has a planted mutant in tests/mutants.toml that this machine kills.
//
// The directed tests below check one rule each, with the defaults' shape.

#include <hegel/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "server/operations.hpp"

namespace gs = hegel::generators;
using halo::server::Operations;
using St = Operations::Status;
using V = Operations::Verdict;

namespace {

Operations::Ref RefOf(std::uint64_t attempt, std::int64_t not_after, std::string inc = "inc0", std::string digest = "d") {
  return {{"owner", "op", attempt}, std::move(inc), not_after, std::move(digest)};
}

Operations::Options Small() {
  Operations::Options o;
  o.capture_ms = 100;
  o.max_submit_age_ms = 1000;
  o.budget_bytes = 1000;
  o.max_records = 4;
  return o;
}

}  // namespace

TEST(Operations, AnAttemptRunsIsFetchedAcknowledgedAndThenForgotten) {
  Operations ops("inc0", Small());
  const auto ref = RefOf(1, 50);
  EXPECT_EQ(ops.StatusOf(ref, 0).view.status, St::kNeverAccepted);
  auto a = ops.Submit(ref, 100, 0);
  ASSERT_EQ(a.verdict, V::kOk);
  EXPECT_EQ(a.view.status, St::kRunning);
  EXPECT_FALSE(ops.Fetch(ref, 1));
  ASSERT_TRUE(ops.Finish(ref.id, "body", {.prompt = 5, .completion = 7}, "stop", 10));
  a = ops.StatusOf(ref, 20);
  EXPECT_EQ(a.view.status, St::kDone);
  EXPECT_EQ(a.view.expires_at, 110);
  EXPECT_EQ(a.view.usage.completion, 7u);
  EXPECT_EQ(ops.Fetch(ref, 20).value_or(""), "body");
  EXPECT_EQ(ops.Submit(ref, 100, 21).verdict, V::kDuplicate);  // a duplicate copy is a no-op
  EXPECT_EQ(ops.stats(21).reserved_bytes, 100u);
  a = ops.Ack(ref, 30);
  EXPECT_EQ(a.view.status, St::kReleased);
  EXPECT_EQ(a.view.usage.completion, 7u);
  EXPECT_FALSE(ops.Fetch(ref, 31));
  EXPECT_EQ(ops.stats(31).reserved_bytes, 0u);
  EXPECT_EQ(ops.Submit(ref, 100, 40).verdict, V::kDuplicate);  // remembered until its not-after
  EXPECT_EQ(ops.stats(40).records, 1u);
  EXPECT_EQ(ops.stats(60).records, 1u);  // (pruned at the next verb, not by stats)
  EXPECT_EQ(ops.StatusOf(ref, 60).view.status, St::kClosed);
  EXPECT_EQ(ops.stats(60).records, 0u);
}

TEST(Operations, VerbsForAnotherIncarnationOrDigestActOnNothing) {
  Operations ops("inc1", Small());
  const auto old = RefOf(1, 50, "inc0");
  auto a = ops.Submit(old, 100, 0);
  EXPECT_EQ(a.verdict, V::kStale);
  EXPECT_EQ(a.view.status, St::kUnknown);
  EXPECT_EQ(ops.StatusOf(old, 0).view.status, St::kUnknown);
  EXPECT_EQ(ops.Cancel(old, 0).verdict, V::kStale);
  EXPECT_EQ(ops.Ack(old, 0).verdict, V::kStale);
  EXPECT_EQ(ops.stats(0).records, 0u);
  const auto ref = RefOf(1, 50, "inc1", "d1");
  ASSERT_EQ(ops.Submit(ref, 100, 0).verdict, V::kOk);
  const auto other = RefOf(1, 50, "inc1", "d2");
  EXPECT_EQ(ops.Submit(other, 100, 1).verdict, V::kConflict);
  EXPECT_EQ(ops.Cancel(other, 1).verdict, V::kConflict);
  EXPECT_FALSE(ops.CancelRequested(ref.id));
  ASSERT_TRUE(ops.Finish(ref.id, "b", {}, "stop", 2));
  EXPECT_EQ(ops.Ack(other, 3).verdict, V::kConflict);
  EXPECT_FALSE(ops.Fetch(other, 3));
  EXPECT_EQ(ops.StatusOf(ref, 3).view.status, St::kDone);
}

TEST(Operations, DeadlinesAreTheCarriedNotAfterInTheIncarnationsClock) {
  Operations ops("inc0", Small());
  EXPECT_EQ(ops.Submit(RefOf(1, 10), 100, 11).verdict, V::kLate);
  EXPECT_EQ(ops.Submit(RefOf(2, 2000), 100, 0).verdict, V::kTooFarAhead);
  // An attempt whose submit never arrived: answered against the deadline the query carries.
  EXPECT_EQ(ops.StatusOf(RefOf(3, 10), 10).view.status, St::kNeverAccepted);
  EXPECT_EQ(ops.StatusOf(RefOf(3, 10), 11).view.status, St::kClosed);
  EXPECT_EQ(ops.stats(11).records, 0u);
}

TEST(Operations, ACancelBeforeAcceptanceFencesLateCopies) {
  Operations ops("inc0", Small());
  const auto ref = RefOf(1, 50, "inc0", "d1");
  auto a = ops.Cancel(ref, 0);
  EXPECT_EQ(a.verdict, V::kOk);
  EXPECT_EQ(a.view.status, St::kCancelledBeforeAccept);
  a = ops.Submit(ref, 100, 5);  // the delayed copy arrives
  EXPECT_EQ(a.verdict, V::kDuplicate);
  EXPECT_EQ(a.view.status, St::kCancelledBeforeAccept);
  EXPECT_EQ(ops.Submit(RefOf(1, 50, "inc0", "d2"), 100, 5).verdict, V::kConflict);
  EXPECT_EQ(ops.stats(5).reserved_bytes, 0u);
  EXPECT_EQ(ops.StatusOf(ref, 51).view.status, St::kClosed);  // pruned past its not-after: closed, still not run
  // A cancel past the deadline needs no tombstone.
  EXPECT_EQ(ops.Cancel(RefOf(2, 10), 60).view.status, St::kClosed);
  EXPECT_EQ(ops.stats(60).records, 0u);
}

TEST(Operations, ACancelledRunningAttemptFinishesWithItsUsage) {
  Operations ops("inc0", Small());
  const auto ref = RefOf(1, 50);
  ASSERT_EQ(ops.Submit(ref, 100, 0).verdict, V::kOk);
  auto a = ops.Cancel(ref, 1);
  EXPECT_EQ(a.view.status, St::kRunning);
  EXPECT_TRUE(a.view.cancel_requested);
  EXPECT_TRUE(ops.CancelRequested(ref.id));
  ASSERT_TRUE(ops.Finish(ref.id, "partial", {.completion = 3}, "cancelled", 2));
  a = ops.StatusOf(ref, 3);
  EXPECT_EQ(a.view.status, St::kDone);
  EXPECT_EQ(a.view.finish, "cancelled");
  EXPECT_EQ(a.view.usage.completion, 3u);
  EXPECT_EQ(ops.Cancel(ref, 4).view.status, St::kDone);  // finished: no change
  EXPECT_FALSE(ops.Finish(ref.id, "again", {}, "stop", 5));
}

TEST(Operations, AnExpiredOutcomeIsKeptUntilAcknowledged) {
  Operations ops("inc0", Small());
  const auto ref = RefOf(1, 50);
  ASSERT_EQ(ops.Submit(ref, 100, 0).verdict, V::kOk);
  ASSERT_TRUE(ops.Finish(ref.id, "body", {.completion = 9}, "stop", 10));
  EXPECT_EQ(ops.StatusOf(ref, 109).view.status, St::kDone);  // held for the whole window
  auto a = ops.StatusOf(ref, 110);
  EXPECT_EQ(a.view.status, St::kExpired);
  EXPECT_EQ(a.view.usage.completion, 9u);
  EXPECT_FALSE(ops.Fetch(ref, 110));
  EXPECT_EQ(ops.stats(110).reserved_bytes, 0u);
  EXPECT_EQ(ops.stats(110).unacked_outcomes, 1u);
  EXPECT_EQ(ops.StatusOf(ref, 500).view.status, St::kExpired);  // long past its not-after: still known
  EXPECT_EQ(ops.Ack(ref, 501).view.status, St::kExpired);
  EXPECT_EQ(ops.StatusOf(ref, 502).view.status, St::kClosed);  // acknowledged, so pruned
}

TEST(Operations, RetentionIsReservedAtAcceptance) {
  Operations ops("inc0", Small());  // budget 1000
  ASSERT_EQ(ops.Submit(RefOf(1, 50), 600, 0).verdict, V::kOk);
  const auto a = ops.Submit(RefOf(2, 50), 600, 0);  // the first is still running: its result is not here yet
  EXPECT_EQ(a.verdict, V::kRefusedRetention);
  EXPECT_GT(a.retry_after_s, 0);
  EXPECT_EQ(a.view.status, St::kNeverAccepted);  // a refusal closes the copy, not the attempt
  ASSERT_EQ(ops.Submit(RefOf(3, 50), 400, 0).verdict, V::kOk);
  ASSERT_TRUE(ops.Finish({"owner", "op", 1}, "b", {}, "stop", 1));
  ASSERT_EQ(ops.Ack(RefOf(1, 50), 2).view.status, St::kReleased);
  EXPECT_EQ(ops.Submit(RefOf(2, 50), 600, 3).verdict, V::kOk);
}

TEST(Operations, AtTheRecordCeilingItRefusesAndErasesNothing) {
  Operations ops("inc0", Small());  // 4 records
  for (std::uint64_t i = 1; i <= 2; ++i) ASSERT_EQ(ops.Cancel(RefOf(i, 50), 0).view.status, St::kCancelledBeforeAccept);
  for (std::uint64_t i = 3; i <= 4; ++i) ASSERT_EQ(ops.Submit(RefOf(i, 50), 1, 0).verdict, V::kOk);
  auto a = ops.Submit(RefOf(5, 50), 1, 1);
  EXPECT_EQ(a.verdict, V::kRefusedRecords);
  EXPECT_GT(a.retry_after_s, 0);
  EXPECT_EQ(ops.Cancel(RefOf(6, 50), 1).verdict, V::kRefusedRecords);
  for (std::uint64_t i = 1; i <= 2; ++i) EXPECT_EQ(ops.StatusOf(RefOf(i, 50), 2).view.status, St::kCancelledBeforeAccept);
  EXPECT_EQ(ops.stats(2).records, 4u);
  EXPECT_EQ(ops.Submit(RefOf(5, 100), 1, 51).verdict, V::kOk);  // the tombstones' deadline passed: room again
}

// ---- the model's actions ----------------------------------------------------------------------------------------

namespace {

constexpr int kOps = 2, kMaxAttempts = 3;
constexpr std::uint64_t kR = 2;          // worst-case cost of an attempt (its reservation, in tokens)
constexpr std::uint64_t kBytes = 100;    // retention bytes an attempt reserves
constexpr std::int64_t kW = 3;           // a submit's not-after: kW past the platform's latest clock reading
constexpr std::int64_t kMaxT = 40;

Operations::Options MachineOptions() {
  Operations::Options o;
  o.capture_ms = 4;
  o.max_submit_age_ms = 10;
  o.budget_bytes = kBytes;  // one held attempt
  o.max_records = 3;
  o.retry_after_s = 1;
  return o;
}

enum class Kind : std::uint8_t { kSubmit, kCancel, kAck };
struct Msg {
  Kind kind;
  Operations::Ref ref;
};

enum class PSt : std::uint8_t { kPending, kCommitted, kLost, kClosed };
struct Attempt {
  int op = 0;
  Operations::Ref ref;
  PSt st = PSt::kPending;
  bool cancel_seen = false;
};

class Machine {
 public:
  explicit Machine(hegel::TestCase& tc) : tc_(tc) { Restart(); }

  [[nodiscard]] const std::string& failure() const { return failure_; }
  [[nodiscard]] const std::set<std::string>& events() const { return events_; }
  [[nodiscard]] std::int64_t now() const { return now_; }

  // platform
  void SubmitNew() {
    const int op = Draw(0, kOps - 1);
    if (applied_[op] || tries_[op] >= kMaxAttempts) return;
    for (const auto& [n, a] : atts_)
      if (a.op == op && (a.st == PSt::kPending || a.st == PSt::kCommitted)) return;
    const std::uint64_t n = static_cast<std::uint64_t>(op * kMaxAttempts + tries_[op]++);
    Attempt a{op, {{"owner", "op" + std::to_string(op), n}, pinc_, now_ + kW, "d" + std::to_string(n)}};
    atts_[n] = a;
    reserved_ += kR;
    net_.push_back({Kind::kSubmit, a.ref});
  }
  void Resend() {
    if (auto* a = PickPending()) net_.push_back({Kind::kSubmit, a->ref});
  }
  void CancelOne() {
    if (auto* a = PickPending()) net_.push_back({Kind::kCancel, a->ref});
  }
  void Learn() { pinc_ = ops_->incarnation(); }
  void StatusOne() {
    Attempt* a = PickPending();
    if (!a) return;
    const auto ans = ops_->StatusOf(a->ref, now_);
    const std::uint64_t n = a->ref.id.attempt;
    const bool ran = runs_[n] > 0;
    switch (ans.view.status) {
      case St::kRunning: break;
      case St::kUnknown: Lose(*a, PSt::kLost, kR), pinc_ = ops_->incarnation(), Event("unknown"); break;
      case St::kDone: {
        const auto body = ops_->Fetch(a->ref, now_);
        if (!body) return Fail("done without a body");
        fetched_[n] = ans.view.usage.completion;
        break;
      }
      case St::kReleased: Fail("released before the platform committed it"); break;
      case St::kExpired:
        if (now_ < done_at_[n] + MachineOptions().capture_ms) return Fail("expired inside its capture window");
        Lose(*a, PSt::kLost, ans.view.usage.completion);
        net_.push_back({Kind::kAck, a->ref});
        Event("expired settled");
        break;
      case St::kCancelledBeforeAccept:
      case St::kClosed:
        if (ran) return Fail(std::string(halo::server::StatusName(ans.view.status)) + " for an attempt that ran");
        Lose(*a, PSt::kClosed, 0);
        Event(ans.view.status == St::kClosed ? "closed" : "cancelled before accept");
        break;
      case St::kNeverAccepted:
        if (ran) return Fail("never_accepted for an attempt that ran");
        Event("never accepted");
        break;
    }
  }
  void TakeCancel() {
    for (auto& [n, a] : atts_)
      if (a.st == PSt::kPending && a.cancel_seen) {
        if (runs_[n] > 0) return Fail("cancelled_before_accept for an attempt that ran");
        Lose(a, PSt::kClosed, 0);
        return;
      }
  }
  void Commit() {
    if (fetched_.empty()) return;
    auto it = std::next(fetched_.begin(), Draw(0, static_cast<int>(fetched_.size()) - 1));
    Attempt& a = atts_[it->first];
    if (a.st == PSt::kPending) {
      a.st = PSt::kCommitted;
      reserved_ -= kR;
      settled_ += it->second;
      if (applied_[a.op]) Fail("a second receipt for one operation");
      applied_[a.op] = true;
      if (crashes_ > 0) Event("committed after a platform crash");
      if (restarts_ > 1) Event("committed after an executor restart");
    }
    fetched_.erase(it);
  }
  void AckOne() {
    std::vector<Attempt*> c;
    for (auto& [n, a] : atts_)
      if (a.st == PSt::kCommitted) c.push_back(&a);
    if (!c.empty()) net_.push_back({Kind::kAck, c[Draw(0, static_cast<int>(c.size()) - 1)]->ref});
  }
  void Crash() {
    fetched_.clear();
    ++crashes_;
  }

  // network
  void Deliver() {
    if (net_.empty()) return;
    const std::size_t i = static_cast<std::size_t>(Draw(0, static_cast<int>(net_.size()) - 1));
    const Msg m = net_[i];
    if (!tc_.draw(gs::booleans())) net_.erase(net_.begin() + static_cast<std::ptrdiff_t>(i));  // else a duplicate stays
    const std::uint64_t n = m.ref.id.attempt;
    switch (m.kind) {
      case Kind::kSubmit: {
        const auto ans = ops_->Submit(m.ref, kBytes, now_);
        if (ans.verdict == V::kOk) {
          if (++runs_[n] > 1) return Fail("attempt " + std::to_string(n) + " accepted twice");
          running_.insert(n);
        } else if (ans.verdict == V::kStale) {
          Event("stale rejected");
        } else if (ans.verdict == V::kLate) {
          Event("late rejected");
        } else if (ans.verdict == V::kDuplicate) {
          Event("duplicate");
        } else if (ans.verdict == V::kRefusedRetention || ans.verdict == V::kRefusedRecords) {
          Event("refused");
        }
        break;
      }
      case Kind::kCancel: {
        const auto ans = ops_->Cancel(m.ref, now_);
        if (ans.view.status == St::kCancelledBeforeAccept && atts_.contains(n)) atts_[n].cancel_seen = true;
        if (ans.view.cancel_requested) Event("cancel while running");
        break;
      }
      case Kind::kAck: (void)ops_->Ack(m.ref, now_); break;
    }
  }
  void Drop() {
    if (net_.empty()) return;
    const std::size_t i = static_cast<std::size_t>(Draw(0, static_cast<int>(net_.size()) - 1));
    if (net_[i].kind == Kind::kAck) Event("ack lost");
    net_.erase(net_.begin() + static_cast<std::ptrdiff_t>(i));
  }

  // executor
  void Run() {
    if (running_.empty()) return;
    const std::uint64_t n = *std::next(running_.begin(), Draw(0, static_cast<int>(running_.size()) - 1));
    const bool cancelled = ops_->CancelRequested(atts_[n].ref.id);
    const std::uint64_t cost = static_cast<std::uint64_t>(Draw(cancelled ? 0 : 1, static_cast<int>(kR)));
    if (!ops_->Finish(atts_[n].ref.id, "result " + std::to_string(n), {.completion = cost},
                      cancelled ? "cancelled" : "stop", now_))
      return Fail("an accepted attempt could not finish");
    real_ += cost;
    done_at_[n] = now_;
    running_.erase(n);
    if (cancelled) Event("cancelled while running");
  }
  void Restart() {
    ops_ = std::make_unique<Operations>("inc" + std::to_string(restarts_++), MachineOptions());
    running_.clear();
  }
  void Tick() { now_ += Draw(1, 3); }

  /// The model's properties, checked after every action.
  void Check() {
    if (!failure_.empty()) return;
    if (real_ > settled_ + reserved_) Fail("accounted: real " + std::to_string(real_) + " > settled + reserved");
    const auto s = ops_->stats(now_);
    if (s.records > MachineOptions().max_records) Fail("records above the ceiling");
    if (s.reserved_bytes > MachineOptions().budget_bytes) Fail("reservations above the budget");
    if (applied_[0] && applied_[1]) Event("done");
  }

 private:
  int Draw(int lo, int hi) { return tc_.draw(gs::integers<int>({.min_value = lo, .max_value = hi})); }
  void Fail(std::string why) {
    if (failure_.empty()) failure_ = std::move(why);
  }
  void Event(const std::string& e) { events_.insert(e); }
  Attempt* PickPending() {
    std::vector<Attempt*> p;
    for (auto& [n, a] : atts_)
      if (a.st == PSt::kPending) p.push_back(&a);
    return p.empty() ? nullptr : p[static_cast<std::size_t>(Draw(0, static_cast<int>(p.size()) - 1))];
  }
  void Lose(Attempt& a, PSt end, std::uint64_t cost) {
    a.st = end;
    reserved_ -= kR;
    settled_ += cost;
  }

  hegel::TestCase& tc_;
  std::unique_ptr<Operations> ops_;
  int restarts_ = 0, crashes_ = 0;
  std::int64_t now_ = 0;
  std::string pinc_ = "inc0";
  std::map<std::uint64_t, Attempt> atts_;
  int tries_[kOps] = {0, 0};
  bool applied_[kOps] = {false, false};
  std::map<std::uint64_t, std::uint64_t> fetched_;  // attempt -> usage (volatile: lost in a crash)
  std::vector<Msg> net_;
  std::set<std::uint64_t> running_;  // accepted by the current incarnation, not finished
  // ghosts
  std::map<std::uint64_t, int> runs_;
  std::map<std::uint64_t, std::int64_t> done_at_;
  std::uint64_t real_ = 0, settled_ = 0, reserved_ = 0;
  std::string failure_;
  std::set<std::string> events_;
};

}  // namespace

TEST(OperationsModel, PropertiesHoldUnderTheModelsActions) {
  std::map<std::string, int> seen;
  hegel::test([&](hegel::TestCase& tc) {
    Machine m(tc);
    enum Rule { kSubmit, kResend, kCancel, kLearn, kStatus, kTakeCancel, kCommit, kAck, kCrash, kDeliver, kDrop, kRun,
                kRestart, kTick, kRules };
    bool on[kRules];
    for (bool& e : on) e = tc.draw(gs::booleans());  // swarm
    on[kSubmit] = on[kStatus] = on[kCommit] = on[kDeliver] = on[kRun] = on[kTick] = true;
    const int steps = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 300}));
    for (int i = 0; i < steps && m.failure().empty() && m.now() < kMaxT; ++i) {
      switch (static_cast<Rule>(tc.draw(gs::integers<int>({.min_value = 0, .max_value = kRules + 4})) % kRules)) {
        case kSubmit: if (on[kSubmit]) m.SubmitNew(); break;
        case kResend: if (on[kResend]) m.Resend(); break;
        case kCancel: if (on[kCancel]) m.CancelOne(); break;
        case kLearn: if (on[kLearn]) m.Learn(); break;
        case kStatus: if (on[kStatus]) m.StatusOne(); break;
        case kTakeCancel: if (on[kTakeCancel]) m.TakeCancel(); break;
        case kCommit: if (on[kCommit]) m.Commit(); break;
        case kAck: if (on[kAck]) m.AckOne(); break;
        case kCrash: if (on[kCrash]) m.Crash(); break;
        case kDeliver: if (on[kDeliver]) m.Deliver(); break;
        case kDrop: if (on[kDrop]) m.Drop(); break;
        case kRun: if (on[kRun]) m.Run(); break;
        case kRestart: if (on[kRestart]) m.Restart(); break;
        case kTick: if (on[kTick]) m.Tick(); break;
        case kRules: break;
      }
      m.Check();
    }
    for (const auto& e : m.events()) ++seen[e];
    if (!m.failure().empty()) tc.note(m.failure());
    ASSERT_TRUE(m.failure().empty()) << m.failure();
  }, {.test_cases = 2000});
  // Positive controls: the machine reaches every situation the properties are about.
  for (const char* w : {"done", "committed after an executor restart", "unknown", "expired settled", "closed", "cancelled before accept",
                        "never accepted", "stale rejected", "late rejected", "duplicate", "refused", "ack lost",
                        "cancelled while running", "committed after a platform crash"})
    EXPECT_GT(seen[w], 0) << "never reached: " << w;
}
