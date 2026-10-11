// G3 M2: the Scheduler class (server/scheduler.hpp) against its model (spec/scheduler/scheduler.qnt).
//
// SchedulerModel.PropertiesHoldUnderTheModelsActions is a Hegel state machine with the model's actions and numbers:
// three sandboxes A, B, C (conversations 1, 3, 5, two turns each, a turn's prompt 3 then 5 slices; each sandbox
// serves its own chain or the chain X), deciders, and clients with every outcome of an HTTP call (lost request,
// late arrival, abort with FIN, a peer that vanishes without FIN and is known gone ACK_TO ticks later, idle
// timeout, lost answer, retries with the same idempotency key after a 429 or a loss). A fake engine runs the
// units in ticks (a slice 2, a decode unit 1, a decider 1), maps a page when a unit starts, keeps one retained
// prefix per conversation (bound at a request's first unit, parked when a started request leaves, extended when a
// turn finishes) and carries out the scheduler's evictions. After every action the model's properties must hold,
// with the model's bounds: rejectedNeverRuns, boundedWait (WAIT_BOUND / WAIT_CHAIN), deciderLatency,
// workAfterGoneBounded, noOvercommit, withinReservation, refuseOnlyIfShort, noRecompute, noSelfInflicted429,
// generatedOnce and chainFair. (noSpuriousTimeout is the service's: heartbeats, OpenAiService tests.)
// tests/mutants.toml plants one defect per policy rule in scheduler.cpp; this machine kills each.

#include <hegel/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "server/scheduler.hpp"

namespace gs = hegel::generators;
using halo::server::Scheduler;

namespace {

// ---- the model's numbers (scheduler.qnt) ----
constexpr std::int64_t kSlice = 2, kDecode = 1, kDecCost = 1, kMaxUnit = 2;
constexpr std::int64_t kCT = 5, kAckTo = 2, kRA = 3, kCap = 14, kMaxAtt = 4, kMaxFaults = 2, kNDec = 12, kMaxT = 48;
constexpr std::int64_t kWaitBound = kMaxUnit + 3 + 3 + 3 + 2;
constexpr std::int64_t kWaitChain = 2 * kWaitBound - kMaxUnit;
constexpr std::int64_t kDecBound = kMaxUnit + 2 * kDecCost;
constexpr std::int64_t kGoneBound = kAckTo + kMaxUnit;
constexpr std::int64_t kLagBound = 2 * kMaxUnit + kMaxUnit + kDecode;

Scheduler::Options ModelOptions() {
  Scheduler::Options o;
  o.per_second = 8;  // the decider credit refills one per 8 ticks
  o.slice_contended = o.slice_alone = 1;  // one model slice
  o.plimit = 1;
  o.max_batch = 8;
  o.decode_steps = 1;
  o.slice_cost_per_token = static_cast<double>(kSlice);
  o.step_cost = static_cast<double>(kDecode);
  o.batch_step_extra = 0;
  o.decider_cost = static_cast<double>(kDecCost);
  o.decider_burst = 1;
  o.decider_refill_per_s = 1;
  o.capacity = kCap;
  o.replay_ttl_s = 1000;
  return o;
}

int PromptOf(int turn) { return turn == 1 ? 3 : 5; }
std::string PrincipalOf(int c) { return c == 1 ? "A" : c == 3 ? "B" : "C"; }
const std::vector<int> kConvs = {1, 3, 5};

enum class Ans { kNone, kPending, kGot, kLost, k429 };

struct Attempt {
  int conv = 0, turn = 0;  // conv 0: a decider
  bool dec = false;
  std::string key;
  int max_tok = 1;
  enum { kWaiting, kLostReq, kArrived, kRefused, kDropped } st = kWaiting;
  Scheduler::Id id = 0;  // the request it is attached to (its own, or the adopted / replayed one)
  bool conn = true, fin = true, told = false;  // told: the scheduler was told it left
  std::int64_t left_at = 0, heard = 0;
  Ans ans = Ans::kNone;
  std::int64_t gone_work = 0;
};

struct Client {
  int turn = 1;
  bool busy = false;
  int attempts = 0;
  std::int64_t retry_at = 0;
  Ans last = Ans::kNone;
  std::string root;
};

// The fake engine's per-request state.
struct Run {
  int conv = 0, turn = 0, prompt = 0, max_tok = 0;
  bool dec = false, started = false, finished = false, served = false, completed = false;
  int reused = 0, done = 0, decoded = 0, inherited = 0;
  Scheduler::Key bound = 0;
  bool in_unit = false;
};

struct Held {
  int len = 0;
  Scheduler::Key key = 0;
};

class Machine {
 public:
  explicit Machine(hegel::TestCase& tc) : tc_(tc) {
    Scheduler::Hooks hooks;
    hooks.park = [this](Scheduler::Id id) { return Park(id); };
    hooks.evict = [this](Scheduler::Key k) { return Evict(k); };
    sched_ = std::make_unique<Scheduler>(ModelOptions(), hooks);
    for (int c : kConvs) {
      clients_[c].root = tc.draw(gs::booleans()) ? PrincipalOf(c) : "X";
      held_[c] = {};
    }
  }

  // ---------------------------------------------------------------- actions
  void ClientSend(int c) {
    Client& cl = clients_[c];
    if (cl.busy || cl.turn > 2 || cl.attempts >= kMaxAtt || now_ < cl.retry_at) return;
    Attempt a;
    a.conv = c;
    a.turn = cl.turn;
    a.key = std::to_string(c) + ":" + std::to_string(cl.turn);
    a.max_tok = tc_.draw(gs::integers<int>({.min_value = 1, .max_value = 2}));
    a.heard = now_;
    cl.busy = true;
    ++cl.attempts;
    atts_.push_back(a);
  }
  void DeciderSend() {
    if (dec_sent_ >= kNDec) return;
    Attempt a;
    a.dec = true;
    a.heard = now_;
    ++dec_sent_;
    atts_.push_back(a);
  }
  void LoseRequest(Attempt& a) {
    if (a.st != Attempt::kWaiting || a.dec || faults_ >= kMaxFaults) return;
    a.st = Attempt::kLostReq;
    ++faults_;
  }
  void Arrive(Attempt& a) {
    if (a.st != Attempt::kWaiting) return;
    Scheduler::Turn t;
    t.decider = a.dec;
    if (!a.dec) {
      t.principal = PrincipalOf(a.conv);
      t.root = clients_[a.conv].root;
      t.key = a.key;
      t.prompt_tokens = static_cast<std::uint64_t>(PromptOf(a.turn));
      t.max_tokens = static_cast<std::uint64_t>(a.max_tok);
      t.prompt = Tokens(a.conv, PromptOf(a.turn) + 1);
      t.bytes_per_token = 1;
    }
    // refuseOnlyIfShort's truth, before the call: what fits if every evictable prefix went.
    const std::int64_t need = a.dec ? 0 : PromptOf(a.turn) - Reuse(a.conv, PromptOf(a.turn)) + a.max_tok;
    const std::int64_t true_free = a.dec ? 0 : TrueFree(a.conv, PromptOf(a.turn));
    const auto adm = sched_->Arrive(t, now_, a.conn);
    using O = Scheduler::Outcome;
    switch (adm.outcome) {
      case O::kQueued:
        a.st = Attempt::kArrived;
        a.id = adm.id;
        runs_[adm.id] = {a.conv, a.turn, a.dec ? 0 : PromptOf(a.turn), a.max_tok, a.dec};
        if (!a.dec) max_prompt_[a.conv] = std::max(max_prompt_[a.conv], PromptOf(a.turn));
        if (!a.conn) told_[adm.id] = true;
        break;
      case O::kReplayed:
        ev_.insert("replayed");
        a.st = Attempt::kArrived;
        a.id = adm.target;
        a.ans = a.conn ? Ans::kPending : Ans::kNone;  // the stored answer, no engine work
        break;
      case O::kAdopted:
        ev_.insert("adopted");
        a.st = Attempt::kArrived;
        a.id = adm.target;
        told_[adm.target] = false;
        break;
      case O::kDuplicate:
        ev_.insert("lateDuplicate");
        a.st = Attempt::kDropped;
        break;
      case O::kLimit:
      case O::kMemory: {
        a.st = Attempt::kRefused;
        ev_.insert(adm.outcome == O::kMemory ? "mem429" : "limit429");
        if (adm.id) refused_ids_.insert(adm.id);  // (a 429 that still queued: rejectedNeverRuns sees it run)
        if (adm.id) runs_[adm.id] = {a.conv, a.turn, PromptOf(a.turn), a.max_tok, false};
        if (adm.outcome == O::kMemory && need <= true_free)
          Fail("refuseOnlyIfShort: a memory 429 for " + std::to_string(need) + " units with " +
               std::to_string(true_free) + " free once every evictable prefix went");
        if (adm.outcome == O::kLimit && a.conn) {
          // noSelfInflicted429: refused only by the client's own earlier attempt of this turn.
          bool all_mine = true, any = false;
          for (const auto& v : sched_->Requests())
            if (!v.decider && v.principal == PrincipalOf(a.conv) && !v.departed) {
              any = true;
              all_mine = all_mine && v.key == a.key;
            }
          if (any && all_mine) Fail("noSelfInflicted429: a retry refused by its own attempt (key " + a.key + ")");
        }
        if (a.conn) {
          a.ans = Ans::k429;
          Client& cl = clients_[a.conv];
          cl.busy = false;
          cl.retry_at = now_ + kRA;
          cl.last = Ans::k429;
        }
        break;
      }
    }
  }
  void Leave(Attempt& a, bool fin) {
    a.conn = false;
    a.fin = fin;
    a.left_at = now_;
    if (!a.dec) {
      Client& cl = clients_[a.conv];
      cl.busy = false;
      cl.retry_at = now_;
      cl.last = a.ans;
    }
    TellGone();
  }
  bool WaitingOn(const Attempt& a) const {
    return !a.dec && a.conn && (a.st == Attempt::kWaiting || a.st == Attempt::kLostReq || a.st == Attempt::kArrived) &&
           (a.ans == Ans::kNone || a.ans == Ans::kLost);
  }
  void Timeout(Attempt& a) {
    // Heartbeats while the server holds it: only a request the server does not hold (lost, in flight) or a lost
    // answer times out.
    const bool held = a.st == Attempt::kArrived && Holds(a.id);
    const std::int64_t last = held ? now_ : a.heard;
    if (!WaitingOn(a) || now_ - last < kCT) return;
    Leave(a, true);
  }
  void Abort(Attempt& a) {
    if (!WaitingOn(a) || faults_ >= kMaxFaults) return;
    ++faults_;
    Leave(a, true);
  }
  void Vanish(Attempt& a) {
    if (!WaitingOn(a) || faults_ >= kMaxFaults) return;
    ++faults_;
    Leave(a, false);
  }
  void Deliver(Attempt& a) {
    if (a.ans != Ans::kPending || !a.conn) return;
    a.ans = Ans::kGot;
    a.heard = now_;
    if (!a.dec) {
      if (a.turn == 2) ev_.insert("turn2Done");
      Client& cl = clients_[a.conv];
      cl.busy = false;
      ++cl.turn;
      cl.last = Ans::kGot;
    }
  }
  void LoseAnswer(Attempt& a) {
    if (a.ans != Ans::kPending || a.dec || faults_ >= kMaxFaults) return;
    ++faults_;
    a.ans = Ans::kLost;
  }
  // The engine thread: a unit boundary (fence, evict, pick) or a unit's end.
  void Engine() {
    if (unit_) {
      if (now_ < until_) return;
      EndUnit();
      return;
    }
    auto u = sched_->Next(now_);
    for (auto id : u.fenced) runs_[id].finished = true;
    if (!u.fenced.empty()) ev_.insert("fenced");
    if (u.kind == Scheduler::Kind::kNone) return;
    StartUnit(std::move(u));
  }
  bool Tick() {
    if (now_ >= kMaxT) return false;
    if (unit_) {
      if (now_ >= until_) return false;
    } else {
      // Work-conserving: time passes idle only when nothing is runnable or waiting to be fenced.
      if (!sched_->Requests().empty()) return false;
    }
    ++now_;
    TellGone();
    return true;
  }

  // ---------------------------------------------------------------- properties
  void Check() {
    const auto views = sched_->Requests();
    for (const auto& v : views) {
      const Attempt* a = AttemptOf(v.id);
      const bool conn = a && a->conn;
      const Run& r = runs_[v.id];
      if (!v.decider && conn && !r.finished) {
        const std::int64_t bound = v.chained ? kWaitChain : kWaitBound;
        if (now_ - v.last_served > bound)
          Fail("boundedWait: request " + std::to_string(v.id) + " (" + v.principal + "/" + v.root +
               (v.chained ? ", chained" : "") + ") unserved for " + std::to_string(now_ - v.last_served) + " > " +
               std::to_string(bound) + " ticks");
      }
      if (v.decider && v.prio && v.phase == Scheduler::Phase::kQueued && now_ - v.admitted_at > kDecBound)
        Fail("deciderLatency: a credited decider waits " + std::to_string(now_ - v.admitted_at));
      if (!v.decider) {
        const int own = r.done - r.reused + r.decoded + (r.in_unit ? 1 : 0);
        if (own > static_cast<int>(v.need))
          Fail("withinReservation: request " + std::to_string(v.id) + " maps " + std::to_string(own) +
               " > its reservation " + std::to_string(v.need));
      }
    }
    for (const auto& a : atts_)
      if (a.gone_work > kGoneBound) Fail("workAfterGoneBounded: " + std::to_string(a.gone_work) + " ticks for a gone client");
    int physical = 0;
    for (const auto& [c, h] : held_) physical += h.len;
    for (const auto& [id, r] : runs_)
      if (!r.dec && r.started && !r.finished) physical += r.done - r.reused + r.decoded + r.inherited + (r.in_unit ? 1 : 0);
    if (physical > kCap) Fail("noOvercommit: " + std::to_string(physical) + " > " + std::to_string(kCap));
    for (int c : kConvs)
      if (computed_[c] > max_prompt_[c] + evicted_[c])
        Fail("noRecompute: conversation " + std::to_string(c) + " computed " + std::to_string(computed_[c]) +
             " slices, longest prompt " + std::to_string(max_prompt_[c]) + " + evicted " + std::to_string(evicted_[c]));
    std::map<std::pair<int, int>, int> generated;
    for (const auto& [id, r] : runs_)
      if (!r.dec && r.completed && r.served && ++generated[{r.conv, r.turn}] > 1)
        Fail("generatedOnce: turn " + std::to_string(r.turn) + " of conversation " + std::to_string(r.conv) + " twice");
    for (const auto& [pair, base] : co_) {
      const std::int64_t d = svc_[pair.first] - svc_[pair.second] - base;
      if (d > kLagBound || -d > kLagBound)
        Fail("chainFair: roots " + pair.first + " and " + pair.second + " backlogged together, service apart by " +
             std::to_string(d) + " ticks");
    }
  }

  // Whether a rule is enabled for an attempt (the test draws among those).
  bool Can(int rule, const Attempt& a) const {
    switch (rule) {
      case 0: return a.st == Attempt::kWaiting && !a.dec && faults_ < kMaxFaults;  // lose the request
      case 1: return a.st == Attempt::kWaiting;                                    // it arrives
      case 2: {                                                                    // idle timeout
        const bool held = a.st == Attempt::kArrived && Holds(a.id);
        return WaitingOn(a) && now_ - (held ? now_ : a.heard) >= kCT;
      }
      case 3: case 4: return WaitingOn(a) && faults_ < kMaxFaults;                 // abort, vanish
      case 5: return a.ans == Ans::kPending && a.conn;                             // the answer arrives
      case 6: return a.ans == Ans::kPending && !a.dec && faults_ < kMaxFaults;     // ... or is lost
      default: return false;
    }
  }
  std::int64_t now() const { return now_; }
  std::vector<Attempt>& atts() { return atts_; }
  const std::string& failure() const { return failure_; }
  const std::set<std::string>& events() const { return ev_; }
  std::string Trace() const { return trace_.str(); }
  std::ostringstream& log() { return trace_; }

 private:
  static std::shared_ptr<const std::vector<std::int32_t>> Tokens(int c, int n) {
    auto v = std::make_shared<std::vector<std::int32_t>>();
    for (int i = 0; i < n; ++i) v->push_back(c * 100 + i);
    return v;
  }
  void Fail(const std::string& why) {
    if (failure_.empty()) failure_ = "t=" + std::to_string(now_) + " " + why;
  }
  bool Holds(Scheduler::Id id) const {
    const auto v = sched_->Request(id);
    return v && (v->phase == Scheduler::Phase::kQueued || v->phase == Scheduler::Phase::kRunning);
  }
  const Attempt* AttemptOf(Scheduler::Id id) const {  // the attempt attached last (an adopter replaces the original)
    const Attempt* found = nullptr;
    for (const auto& a : atts_)
      if (a.id == id && a.st == Attempt::kArrived) found = &a;
    return found;
  }
  int Reuse(int c, int prompt) const { return held_.at(c).len <= prompt ? held_.at(c).len : 0; }
  std::int64_t TrueFree(int c, int prompt) const {
    // Pinned prefixes (planned by an admitted request that has not started) stay; others' reservations are real;
    // an evictable prefix a started request forked from stays as that request's pages.
    std::set<Scheduler::Key> pinned, forked;
    std::int64_t reserved = 0;
    for (const auto& v : sched_->Requests())
      if (!v.decider) {
        reserved += static_cast<std::int64_t>(v.need + v.inherited);
        if (!v.started && v.planned) pinned.insert(v.planned);
        if (v.started && v.bound && v.inherited == 0) forked.insert(v.bound);
      }
    const Scheduler::Key mine = Reuse(c, prompt) > 0 ? held_.at(c).key : 0;
    std::int64_t stays = 0;
    for (const auto& [o, h] : held_)
      if (h.len > 0 && (pinned.count(h.key) || forked.count(h.key) || h.key == mine)) stays += h.len;
    return kCap - stays - reserved;
  }
  void TellGone() {
    for (auto& a : atts_) {
      if (a.conn || a.st != Attempt::kArrived || a.told) continue;
      if (a.fin || now_ >= a.left_at + kAckTo) {
        a.told = true;
        // Only the attempt the request is attached to speaks for it (an adopter took it over).
        if (AttemptOf(a.id) == &a) {
          sched_->Depart(a.id);
          told_[a.id] = true;
        }
      }
    }
  }
  std::optional<Scheduler::Retained> Park(Scheduler::Id id) {
    Run& r = runs_[id];
    r.finished = true;
    if (r.dec || !r.started || r.done <= held_[r.conv].len) return std::nullopt;
    held_[r.conv] = {r.done, next_key_++};
    ev_.insert("parkedOnCancel");
    return Scheduler::Retained{held_[r.conv].key, static_cast<std::uint64_t>(r.done), Tokens(r.conv, r.done), true};
  }
  std::uint64_t Evict(Scheduler::Key k) {
    for (auto& [c, h] : held_)
      if (h.key == k && h.len > 0) {
        bool fork = false;
        for (auto& [id, r] : runs_)
          if (!r.dec && r.started && !r.finished && r.bound == k && r.inherited == 0) r.inherited = r.reused, fork = true;
        evicted_[c] += h.len;
        ev_.insert(fork ? "evictedForked" : "evicted");
        const auto freed = fork ? 0 : static_cast<std::uint64_t>(h.len);
        h = {};
        return freed;
      }
    return 0;
  }
  void Sync() {
    std::vector<Scheduler::Retained> rs;
    for (const auto& [c, h] : held_)
      if (h.len > 0) rs.push_back({h.key, static_cast<std::uint64_t>(h.len), Tokens(c, h.len), true});
    sched_->SyncRetained(std::move(rs));
  }
  void StartUnit(Scheduler::Unit u) {
    // chainFair's ghost: the agent chain roots (each conversation's true root) backlogged now.
    std::set<std::string> bl;
    for (const auto& v : sched_->Requests())
      if (!v.decider && !v.departed) bl.insert(clients_[runs_[v.id].conv].root);
    std::map<std::pair<std::string, std::string>, std::int64_t> co;
    for (const auto& p : bl)
      for (const auto& q : bl)
        if (p != q) co[{p, q}] = co_.count({p, q}) ? co_[{p, q}] : svc_[p] - svc_[q];
    std::int64_t cost = 0;
    std::set<std::string> roots;
    for (auto id : u.ids) {
      Run& r = runs_[id];
      if (refused_ids_.count(id)) Fail("rejectedNeverRuns: request " + std::to_string(id) + " got a 429 and runs");
      if (u.kind == Scheduler::Kind::kSlice && !r.started && Reuse(r.conv, r.prompt) > 0) ev_.insert("reused");
      if (u.kind == Scheduler::Kind::kSlice && !r.started) {
        // Bind at start: the conversation's retained prefix as it is now.
        r.started = true;
        r.reused = Reuse(r.conv, r.prompt);
        r.bound = r.reused > 0 ? held_[r.conv].key : 0;
        r.done = r.reused;
        sched_->Bind(id, r.bound, static_cast<std::uint64_t>(r.reused));
      }
      r.served = true;
      if (u.kind == Scheduler::Kind::kSlice) cost = r.done < r.prompt ? kSlice : 0;
      else if (u.kind == Scheduler::Kind::kBatch) cost = kDecode;
      else cost = kDecCost;
      r.in_unit = u.kind != Scheduler::Kind::kDecide && cost > 0;
      if (!r.dec) roots.insert(clients_[r.conv].root);
    }
    for (auto id : u.ids)
      for (auto& a : atts_)
        if (a.id == id && a.st == Attempt::kArrived && !a.conn && a.ans != Ans::k429 && AttemptOf(id) == &a)
          a.gone_work += cost;
    for (const auto& q : roots) svc_[q] += cost;
    co_ = std::move(co);
    if (u.ids.size() >= 2) ev_.insert("batched");
    if (u.kind == Scheduler::Kind::kDecide) ev_.insert(runs_[u.ids[0]].dec && sched_->counters().decides ? "decider" : "");
    {
      std::map<std::string, std::set<std::string>> by_root;
      for (const auto& v : sched_->Requests())
        if (!v.decider && !v.departed) by_root[clients_[runs_[v.id].conv].root].insert(v.principal);
      bool fan = false;
      for (const auto& [q, ps] : by_root) fan = fan || ps.size() >= 2;
      if (fan && by_root.size() >= 2) ev_.insert("chainFanout");
    }
    for (const auto& v : sched_->Requests())
      if (!v.decider && v.started && !v.prefill_done && !v.departed && std::find(u.ids.begin(), u.ids.end(), v.id) == u.ids.end())
        ev_.insert(u.kind == Scheduler::Kind::kDecide ? "deciderBetweenSlices" : "preempted");
    trace_ << "t" << now_ << " unit " << static_cast<int>(u.kind) << " [";
    for (auto id : u.ids) trace_ << id << " ";
    trace_ << "] cost " << cost << "\n";
    started_at_ = now_;
    until_ = now_ + cost;
    unit_ = std::move(u);
  }
  void EndUnit() {
    const Scheduler::Unit u = std::move(*unit_);
    unit_.reset();
    std::vector<Scheduler::Progress> ps;
    bool sync = false;
    for (auto id : u.ids) {
      Run& r = runs_[id];
      r.in_unit = false;
      Scheduler::Progress p;
      p.id = id;
      if (u.kind == Scheduler::Kind::kSlice) {
        if (r.done < r.prompt) {
          ++r.done;
          p.prefilled = 1;
          ++computed_[r.conv];
        }
        p.prefill_done = r.done >= r.prompt;
      } else if (u.kind == Scheduler::Kind::kBatch) {
        ++r.decoded;
        p.decoded = 1;
        if (r.decoded >= r.max_tok) {
          p.finished = p.completed = r.finished = r.completed = true;
          // A finished turn keeps its prompt as a retained prefix (the exact-mode checkpoint).
          if (r.prompt > held_[r.conv].len) held_[r.conv] = {r.prompt, next_key_++};
          sync = true;
        }
      } else {
        p.finished = p.completed = r.finished = r.completed = true;
      }
      if (p.finished)
        for (auto& a : atts_)
          if (a.id == id && a.st == Attempt::kArrived && a.conn && AttemptOf(id) == &a) a.ans = Ans::kPending;
      ps.push_back(p);
    }
    sched_->Done(u, ps, now_, until_ - started_at_);
    if (sync) Sync();
  }

  hegel::TestCase& tc_;
  std::unique_ptr<Scheduler> sched_;
  std::int64_t now_ = 0, until_ = 0, started_at_ = 0;
  std::optional<Scheduler::Unit> unit_;
  std::vector<Attempt> atts_;
  std::map<int, Client> clients_;
  std::map<Scheduler::Id, Run> runs_;
  std::map<Scheduler::Id, bool> told_;
  std::map<int, Held> held_;
  std::map<int, int> computed_, max_prompt_, evicted_;
  std::set<Scheduler::Id> refused_ids_;
  std::map<std::string, std::int64_t> svc_;
  std::map<std::pair<std::string, std::string>, std::int64_t> co_;
  Scheduler::Key next_key_ = 1;
  int dec_sent_ = 0, faults_ = 0;
  std::string failure_;
  std::ostringstream trace_;
  std::set<std::string> ev_;  // witnesses reached
};

}  // namespace

TEST(SchedulerModel, PropertiesHoldUnderTheModelsActions) {
  std::map<std::string, int> seen;  // runs that reached each witness (positive control: the machine gets there)
  int runs = 0;
  hegel::test([&](hegel::TestCase& tc) {
    Machine m(tc);
    struct Count {
      Machine& m;
      std::map<std::string, int>& seen;
      int& runs;
      ~Count() {
        ++runs;
        for (const auto& e : m.events()) ++seen[e];
      }
    } count{m, seen, runs};
    // (kSend, kArrive, kDeliver and kEngine are drawn twice as often: the draw is taken modulo kRules)
    enum Rule { kSend, kArrive, kDeliver, kEngine, kTick, kDecider, kLose, kTimeout, kAbort, kVanish, kLoseAns, kRules };
    bool on[kRules];
    for (bool& e : on) e = tc.draw(gs::booleans());  // swarm: each run a random subset of the environment's rules
    on[kSend] = on[kArrive] = on[kEngine] = on[kTick] = on[kDeliver] = true;
    const int steps = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 600}));
    for (int i = 0; i < steps && m.failure().empty() && m.now() < kMaxT; ++i) {
      const auto rule = static_cast<Rule>(tc.draw(gs::integers<int>({.min_value = 0, .max_value = kRules + 3})) % kRules);
      if (!on[rule]) continue;
      // The attempt a rule acts on: one of those it is enabled for.
      const auto pick = [&](int which) -> Attempt* {
        std::vector<Attempt*> ok;
        for (auto& a : m.atts())
          if (m.Can(which, a)) ok.push_back(&a);
        if (ok.empty()) return nullptr;
        return ok[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = ok.size() - 1}))];
      };
      switch (rule) {
        case kSend: m.ClientSend(kConvs[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 2}))]); break;
        case kDecider: m.DeciderSend(); break;
        case kLose: if (auto* a = pick(0)) m.LoseRequest(*a); break;
        case kArrive: if (auto* a = pick(1)) m.Arrive(*a); break;
        case kTimeout: if (auto* a = pick(2)) m.Timeout(*a); break;
        case kAbort: if (auto* a = pick(3)) m.Abort(*a); break;
        case kVanish: if (auto* a = pick(4)) m.Vanish(*a); break;
        case kDeliver: if (auto* a = pick(5)) m.Deliver(*a); break;
        case kLoseAns: if (auto* a = pick(6)) m.LoseAnswer(*a); break;
        case kEngine: m.Engine(); break;
        case kTick: m.Tick(); break;
        default: break;
      }
      m.Check();
    }
    ASSERT_TRUE(m.failure().empty()) << m.failure() << "\n" << m.Trace();
  }, {.test_cases = 10000});
  std::printf("witnesses over %d runs:", runs);
  for (const auto& [e, n] : seen) std::printf(" %s %d", e.c_str(), n);
  std::printf("\n");
  for (const char* w : {"preempted", "batched", "deciderBetweenSlices", "chainFanout", "parkedOnCancel", "evicted",
                        "mem429", "fenced", "adopted", "replayed", "reused", "turn2Done", "lateDuplicate"})
    EXPECT_GT(seen[w], 0) << "witness " << w << " never reached";
}

// ---- directed checks of single rules (the ms-unit defaults halo_serve uses) ----

namespace {
Scheduler::Turn Agent(const std::string& principal, const std::string& root = "", const std::string& key = "",
                      std::uint64_t prompt = 4096, std::uint64_t max = 100) {
  Scheduler::Turn t;
  t.principal = principal;
  t.root = root;
  t.key = key;
  t.prompt_tokens = prompt;
  t.max_tokens = max;
  t.bytes_per_token = 1;
  return t;
}
// Runs one unit to its end at its estimate; every slice prefills its tokens, every batch decodes its steps.
Scheduler::Unit RunUnit(Scheduler& s, std::int64_t* now, std::map<Scheduler::Id, std::uint64_t>* left) {
  auto u = s.Next(*now);
  if (u.kind == Scheduler::Kind::kNone) return u;
  std::vector<Scheduler::Progress> ps;
  for (auto id : u.ids) {
    Scheduler::Progress p{id};
    if (u.kind == Scheduler::Kind::kSlice) {
      s.Bind(id, 0, 0);
      auto& l = (*left)[id];
      p.prefilled = std::min<std::uint64_t>(l, u.slice_tokens);
      l -= p.prefilled;
      p.prefill_done = l == 0;
    } else if (u.kind == Scheduler::Kind::kBatch) {
      p.decoded = u.steps;
    } else {
      p.finished = p.completed = true;
    }
    ps.push_back(p);
  }
  *now += u.charged;
  s.Done(u, ps, *now, u.charged);
  return u;
}
}  // namespace

// D1: 2048-token slices while anyone else is runnable, 8192 when the prefill is alone.
TEST(Scheduler, SliceIsShortOnlyWhenContended) {
  Scheduler s({});
  std::int64_t now = 0;
  std::map<Scheduler::Id, std::uint64_t> left;
  const auto a = s.Arrive(Agent("A", "", "", 20000), now);
  left[a.id] = 20000;
  auto u = RunUnit(s, &now, &left);
  ASSERT_EQ(u.kind, Scheduler::Kind::kSlice);
  EXPECT_EQ(u.slice_tokens, 8192u);
  const auto b = s.Arrive(Agent("B", "", "", 20000), now);
  left[b.id] = 20000;
  u = RunUnit(s, &now, &left);
  ASSERT_EQ(u.kind, Scheduler::Kind::kSlice);
  EXPECT_EQ(u.slice_tokens, 2048u);
}

// D3: one in-flight turn per principal; the 429's Retry-After is the in-flight request's expected remaining time,
// capped at 30 s. Another principal is not limited by it.
TEST(Scheduler, PrincipalLimitAnswers429WithRetryAfter) {
  Scheduler s({});
  const auto a = s.Arrive(Agent("A", "", "k1", 4096, 100), 0);
  ASSERT_EQ(a.outcome, Scheduler::Outcome::kQueued);
  const auto a2 = s.Arrive(Agent("A", "", "k2", 4096, 100), 0);
  ASSERT_EQ(a2.outcome, Scheduler::Outcome::kLimit);
  // 4096 x 0.61 ms + 100 x 37 ms = 6.2 s: Retry-After 7.
  EXPECT_EQ(a2.retry_after_s, 7);
  const auto huge = s.Arrive(Agent("C", "", "k3", 200000, 4000), 0);
  ASSERT_EQ(huge.outcome, Scheduler::Outcome::kQueued);
  EXPECT_EQ(s.Arrive(Agent("C", "", "k4"), 0).retry_after_s, 30);
  EXPECT_EQ(s.Arrive(Agent("B", "", "k5"), 0).outcome, Scheduler::Outcome::kQueued);
  // A departed request no longer holds the principal's slot (it is fenced at the next boundary).
  s.Depart(a.id);
  EXPECT_EQ(s.Arrive(Agent("A", "", "k6"), 0).outcome, Scheduler::Outcome::kQueued);
}

// D4: decider credit, burst 4, refill 2/s: the fifth decider in a burst waits for a fair turn, and the credit
// comes back at two a second.
TEST(Scheduler, DeciderCreditBurstAndRefill) {
  Scheduler s({});
  Scheduler::Turn d;
  d.decider = true;
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(s.Request(s.Arrive(d, 0).id)->prio) << i;
  EXPECT_FALSE(s.Request(s.Arrive(d, 0).id)->prio);
  EXPECT_FALSE(s.Request(s.Arrive(d, 499).id)->prio);
  EXPECT_TRUE(s.Request(s.Arrive(d, 500).id)->prio);   // one credit back after 0.5 s
  EXPECT_FALSE(s.Request(s.Arrive(d, 999).id)->prio);
  EXPECT_TRUE(s.Request(s.Arrive(d, 1000).id)->prio);
  EXPECT_EQ(s.counters().decider_out_of_credit, 3u);
}

// A credited decider runs at the next unit boundary, ahead of fair queuing; an uncredited one takes a fair turn.
TEST(Scheduler, CreditedDeciderRunsAtTheNextBoundary) {
  Scheduler::Options o;
  o.decider_burst = 1;
  Scheduler s(o);
  std::int64_t now = 0;
  std::map<Scheduler::Id, std::uint64_t> left;
  const auto a = s.Arrive(Agent("A", "", "", 100000), now);
  left[a.id] = 100000;
  const auto b = s.Arrive(Agent("B", "", "", 100000), now);
  left[b.id] = 100000;
  RunUnit(s, &now, &left);
  Scheduler::Turn d;
  d.decider = true;
  const auto d1 = s.Arrive(d, now);
  auto u = RunUnit(s, &now, &left);
  ASSERT_EQ(u.kind, Scheduler::Kind::kDecide);
  EXPECT_EQ(u.ids[0], d1.id);
}

// Two-level fair queuing (A2A-MCP D9): a chain X fanned out to sandboxes A and B, and a lone sandbox C, all
// prefilling: C gets half the GPU time, A and B a quarter each. One level (per sandbox) would give C a third.
TEST(Scheduler, FanOutChainHoldsOneShare) {
  Scheduler s({});
  std::int64_t now = 0;
  std::map<Scheduler::Id, std::uint64_t> left;
  std::map<Scheduler::Id, std::string> who;
  for (const auto& [p, root] : std::vector<std::pair<std::string, std::string>>{{"A", "X"}, {"B", "X"}, {"C", ""}}) {
    const auto r = s.Arrive(Agent(p, root, "", 1u << 30), now);
    ASSERT_EQ(r.outcome, Scheduler::Outcome::kQueued);
    left[r.id] = 1u << 30;
    who[r.id] = p;
  }
  std::map<std::string, std::int64_t> gpu;
  for (int i = 0; i < 400; ++i) {
    auto u = RunUnit(s, &now, &left);
    ASSERT_EQ(u.kind, Scheduler::Kind::kSlice);
    gpu[who[u.ids[0]]] += u.charged;
  }
  const double total = static_cast<double>(gpu["A"] + gpu["B"] + gpu["C"]);
  EXPECT_NEAR(gpu["C"] / total, 0.5, 0.01);
  EXPECT_NEAR(gpu["A"] / total, 0.25, 0.01);
  EXPECT_NEAR(gpu["B"] / total, 0.25, 0.01);
}

// Every decoding request steps in one unit; a slice and a decode unit alternate when one request decodes and
// another prefills.
TEST(Scheduler, DecodersBatchAndAlternateWithSlices) {
  Scheduler s({});
  std::int64_t now = 0;
  std::map<Scheduler::Id, std::uint64_t> left;
  const auto a = s.Arrive(Agent("A", "", "", 100, 1000), now);
  const auto b = s.Arrive(Agent("B", "", "", 100, 1000), now);
  left[a.id] = left[b.id] = 100;
  RunUnit(s, &now, &left);
  RunUnit(s, &now, &left);
  auto u = RunUnit(s, &now, &left);
  ASSERT_EQ(u.kind, Scheduler::Kind::kBatch);
  EXPECT_EQ(u.ids.size(), 2u);
  EXPECT_EQ(u.steps, 6u);  // ~0.3 s at B=2 (52 ms a step)
  const auto c = s.Arrive(Agent("C", "", "", 100000, 10), now);
  left[c.id] = 100000;
  std::vector<Scheduler::Kind> kinds;
  for (int i = 0; i < 6; ++i) kinds.push_back(RunUnit(s, &now, &left).kind);
  int slices = 0;
  for (auto k : kinds) slices += k == Scheduler::Kind::kSlice;
  EXPECT_GE(slices, 2);
  EXPECT_LE(slices, 4);
}

// D5: a finished key is replayed while within its TTL; afterwards the same key is a new request.
TEST(Scheduler, FinishedKeyReplaysWithinItsTtl) {
  Scheduler::Options o;
  o.replay_ttl_s = 10;
  Scheduler s(o);
  std::int64_t now = 0;
  std::map<Scheduler::Id, std::uint64_t> left;
  const auto a = s.Arrive(Agent("A", "", "turn-1", 100, 1), now);
  left[a.id] = 100;
  RunUnit(s, &now, &left);
  auto u = s.Next(now);
  ASSERT_EQ(u.kind, Scheduler::Kind::kBatch);
  s.Done(u, {{a.id, 0, 1, false, true, true}}, now, u.charged);
  const auto r = s.Arrive(Agent("A", "", "turn-1", 100, 1), now + 1000);
  EXPECT_EQ(r.outcome, Scheduler::Outcome::kReplayed);
  EXPECT_EQ(r.target, a.id);
  EXPECT_EQ(s.Arrive(Agent("A", "", "turn-1", 100, 1), now + 11000).outcome, Scheduler::Outcome::kQueued);
}

// The ledger: an admission evicts unpinned retained prefixes least recently used first, and refuses (429) only
// when even that cannot make room; the evictions are carried out at the next boundary.
TEST(Scheduler, AdmissionEvictsLruAndRefusesOnlyWhenShort) {
  Scheduler::Options o;
  o.capacity = 1000;
  std::vector<Scheduler::Key> evicted;
  Scheduler::Hooks h;
  h.evict = [&](Scheduler::Key k) {
    evicted.push_back(k);
    return 300;
  };
  Scheduler s(o, h);
  s.SyncRetained({{1, 300, nullptr, true}, {2, 300, nullptr, true}, {3, 300, nullptr, true}});
  const auto a = s.Arrive(Agent("A", "", "", 300, 100), 0);  // 400: one eviction (key 1, the oldest)
  ASSERT_EQ(a.outcome, Scheduler::Outcome::kQueued);
  EXPECT_EQ(s.ledger().retained, 600u);
  const auto b = s.Arrive(Agent("B", "", "", 1000, 100), 0);  // 1100 > 1000 even with everything gone
  EXPECT_EQ(b.outcome, Scheduler::Outcome::kMemory);
  EXPECT_EQ(b.retry_after_s, 5);
  (void)s.Next(0);
  EXPECT_EQ(evicted, (std::vector<Scheduler::Key>{1, 2, 3}));
}
