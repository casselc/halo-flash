// G3 M5: the model's traces replayed against the Scheduler class (docs/SCHEDULER.md section 9; as yard's
// internal/lifecycle/replay_test.go).
//
// spec/scheduler/replay.sh runs `quint run --mbt --out-itf` on the design instance and turns each trace into the
// server-side events spec/scheduler/itf_events.py documents (tests/server/data/sched_traces/*.jsonl): admissions,
// departures the server learns, unit picks (with the fences before them), unit ends. This test drives a Scheduler
// with the model's numbers (ModelOptions, as the Hegel machine in scheduler_test.cpp) through each trace, the
// engine side (binds, parks, the retained prefixes) taken from the model, and compares after every step:
//   admission outcome (queued, replayed, adopted, duplicate, limit or memory 429), its target, a decider's credit;
//   the requests fenced at a boundary; the unit picked (kind and requests); every root's and principal's virtual
//   time after the pick's charge.
// Known refinement of the code over the model, accepted: a request whose whole prompt is reused starts with an
// empty slice (the engine finds nothing to prefill: zero length, then it decodes); the model starts it in a batch.
// The replay runs that empty slice and compares the next pick.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "server/scheduler.hpp"

using halo::json::Value;
using halo::server::Scheduler;

namespace {

Scheduler::Options ReplayOptions() {  // the model's numbers (scheduler.qnt), as scheduler_test.cpp's ModelOptions
  Scheduler::Options o;
  o.per_second = 8;  // DEC_REFILL: one credit every 8 ticks
  o.slice_contended = o.slice_alone = 1;  // one model slice is one token here
  o.plimit = 1;
  o.max_batch = 8;
  o.decode_steps = 1;
  o.slice_cost_per_token = 2;  // SLICE
  o.step_cost = 1;             // DECODE
  o.batch_step_extra = 0;
  o.decider_cost = 1;          // DEC_COST
  o.decider_burst = 1;         // DEC_BURST
  o.decider_refill_per_s = 1;
  o.capacity = 14;             // CAP
  o.replay_ttl_s = 1000;
  return o;
}

std::shared_ptr<const std::vector<std::int32_t>> Tokens(int c, int n) {
  auto v = std::make_shared<std::vector<std::int32_t>>();
  for (int i = 0; i < n; ++i) v->push_back(c * 100 + i);
  return v;
}

std::int64_t I(const Value& v, const char* k) {
  const Value* x = v.find(k);
  return x ? static_cast<std::int64_t>(x->number()) : 0;
}
bool B(const Value& v, const char* k) {
  const Value* x = v.find(k);
  return x && x->boolean();
}
std::string S(const Value& v, const char* k) {
  const Value* x = v.find(k);
  return x ? x->str() : "";
}
std::string Cls(const std::string& q) { return q == "D" ? Scheduler::kDeciderClass : q; }

const char* KindName(Scheduler::Kind k) {
  switch (k) {
    case Scheduler::Kind::kSlice: return "slice";
    case Scheduler::Kind::kBatch: return "batch";
    case Scheduler::Kind::kDecide: return "decide";
    default: return "idle";
  }
}
const char* OutName(Scheduler::Outcome o) {
  switch (o) {
    case Scheduler::Outcome::kQueued: return "queued";
    case Scheduler::Outcome::kReplayed: return "replayed";
    case Scheduler::Outcome::kAdopted: return "adopted";
    case Scheduler::Outcome::kDuplicate: return "duplicate";
    case Scheduler::Outcome::kLimit: return "limit";
    case Scheduler::Outcome::kMemory: return "memory";
  }
  return "?";
}

class Replay {
 public:
  Replay() {
    Scheduler::Hooks h;
    h.park = [this](Scheduler::Id id) -> std::optional<Scheduler::Retained> {
      const auto mid = mid_.at(id);
      if (const Value* p = parks_ ? parks_->find(std::to_string(mid)) : nullptr) {
        const int len = static_cast<int>(p->number());
        return Scheduler::Retained{static_cast<Scheduler::Key>(conv_.at(mid)), static_cast<std::uint64_t>(len),
                                   Tokens(conv_.at(mid), len), true};
      }
      return std::nullopt;
    };
    h.evict = [](Scheduler::Key) -> std::uint64_t { return 0; };
    s_ = std::make_unique<Scheduler>(ReplayOptions(), std::move(h));
  }

  /// Applies one event; returns a mismatch ("" when the code did what the model did).
  std::string Step(const Value& e) {
    const std::string kind = S(e, "e");
    const std::int64_t t = I(e, "t");
    if (kind == "arrive") return Arrive(e, t);
    if (kind == "depart") {
      if (auto it = cid_.find(I(e, "id")); it != cid_.end()) s_->Depart(it->second);
      return "";
    }
    if (kind == "fence") return Fence(e, t);
    if (kind == "pick") return Pick(e, t);
    if (kind == "end") return End(e, t);
    return "unknown event " + kind;
  }
  std::uint64_t compared = 0, empty_slices = 0;

 private:
  std::string Arrive(const Value& e, std::int64_t t) {
    const std::int64_t mid = I(e, "id");
    const bool dec = B(e, "dec");
    const int conv = static_cast<int>(I(e, "conv")), prompt = static_cast<int>(I(e, "prompt"));
    Scheduler::Turn turn;
    turn.decider = dec;
    if (!dec) {
      turn.principal = S(e, "p");
      turn.root = S(e, "root");
      turn.key = std::to_string(conv) + "/" + std::to_string(I(e, "turn"));
      turn.prompt_tokens = static_cast<std::uint64_t>(prompt);
      turn.max_tokens = static_cast<std::uint64_t>(I(e, "max"));
      turn.prompt = Tokens(conv, prompt + 1);  // + the token the first decode step reads
      turn.bytes_per_token = 1;
    }
    const auto a = s_->Arrive(turn, t, B(e, "conn"));
    conv_[mid] = conv;
    prompt_[mid] = prompt;
    std::ostringstream why;
    if (OutName(a.outcome) != S(e, "out")) why << "admission: code " << OutName(a.outcome) << ", model " << S(e, "out");
    if (a.outcome == Scheduler::Outcome::kQueued) {
      cid_[mid] = a.id;
      mid_[a.id] = mid;
      if (dec) {
        const auto v = s_->Request(a.id);
        if (!v || v->prio != B(e, "prio")) why << "decider credit: code prio " << (v && v->prio) << ", model " << B(e, "prio");
      }
    } else if (a.outcome == Scheduler::Outcome::kReplayed || a.outcome == Scheduler::Outcome::kAdopted) {
      const auto it = cid_.find(I(e, "target"));
      if (it == cid_.end() || it->second != a.target) why << "target: code " << a.target << ", model " << I(e, "target");
    }
    Sync(*e.find("held"));
    ++compared;
    return why.str();
  }

  // The model fences one request a step between units; the code fences every request known gone at a boundary.
  // Both sets since the last pick must agree when the next unit is picked (and at the end of the trace).
  std::string Fence(const Value& e, std::int64_t t) {
    parks_ = e.find("parks");
    for (auto id : s_->Boundary(t)) fenced_code_.insert(mid_.at(id));
    for (const auto& x : e.find("ids")->items()) fenced_model_.insert(static_cast<std::int64_t>(x.number()));
    return "";
  }

 public:
  std::string Finish() { return CompareFences(); }

 private:
  std::string CompareFences() {
    std::string why;
    if (fenced_code_ != fenced_model_)
      why = "fenced: code " + Set(fenced_code_) + ", model " + Set(fenced_model_) + "; ";
    fenced_code_.clear();
    fenced_model_.clear();
    return why;
  }

  std::string Pick(const Value& e, std::int64_t t) {
    auto u = s_->Next(t);
    parks_ = nullptr;
    std::ostringstream why;
    for (auto id : u.fenced) fenced_code_.insert(mid_.at(id));
    why << CompareFences();
    const Value* binds = e.find("binds");
    // The code's refinement: a request whose whole prompt is reused starts with an empty slice.
    while (u.kind == Scheduler::Kind::kSlice && binds && S(e, "kind") != "slice") {
      const auto mid = mid_.at(u.ids[0]);
      const Value* b = binds->find(std::to_string(mid));
      if (!b || static_cast<int>(b->number()) < prompt_.at(mid)) break;
      s_->Bind(u.ids[0], static_cast<Scheduler::Key>(conv_.at(mid)), static_cast<std::uint64_t>(b->number()));
      Scheduler::Progress p{u.ids[0]};
      p.prefill_done = true;
      s_->Done(u, {p}, t, 0);
      u = s_->Next(t);
      ++empty_slices;
    }
    std::set<std::int64_t> ids, want_ids;
    for (auto id : u.ids) ids.insert(mid_.at(id));
    for (const auto& x : e.find("ids") ? e.find("ids")->items() : Value::Array{})
      want_ids.insert(static_cast<std::int64_t>(x.number()));
    const std::string want_kind = S(e, "kind");
    if (KindName(u.kind) != want_kind || ids != want_ids)
      why << "unit: code " << KindName(u.kind) << " " << Set(ids) << ", model " << want_kind << " " << Set(want_ids)
          << "; ";
    if (binds)
      for (const auto& [k, v] : binds->members()) {
        const auto mid = std::stoll(k);
        if (!cid_.count(mid)) continue;
        const auto reused = static_cast<std::uint64_t>(v.number());
        s_->Bind(cid_[mid], reused ? static_cast<Scheduler::Key>(conv_.at(mid)) : 0, reused);
      }
    if (const Value* vr = e.find("vroot"))
      for (const auto& [q, v] : vr->members())
        if (s_->vtime_root(Cls(q)) != static_cast<std::int64_t>(v.number()))
          why << "vtime root " << q << ": code " << s_->vtime_root(Cls(q)) << ", model " << v.number() << "; ";
    if (const Value* vp = e.find("vprin"))
      for (const auto& [q, v] : vp->members())
        if (s_->vtime_principal(Cls(q)) != static_cast<std::int64_t>(v.number()))
          why << "vtime principal " << q << ": code " << s_->vtime_principal(Cls(q)) << ", model " << v.number()
              << "; ";
    unit_ = u;
    ++compared;
    return why.str();
  }

  std::string End(const Value& e, std::int64_t t) {
    std::vector<Scheduler::Progress> ps;
    for (const auto& p : e.find("prog")->items()) {
      Scheduler::Progress x;
      x.id = cid_.at(I(p, "id"));
      x.prefilled = static_cast<std::uint64_t>(I(p, "pre"));
      x.decoded = static_cast<std::uint64_t>(I(p, "dec"));
      x.prefill_done = B(p, "pdone");
      x.finished = B(p, "fin");
      x.completed = B(p, "comp");
      ps.push_back(x);
    }
    s_->Done(unit_, ps, t, I(e, "elapsed"));
    Sync(*e.find("held"));
    return "";
  }

  void Sync(const Value& held) {
    std::vector<Scheduler::Retained> rs;
    for (const auto& [c, v] : held.members()) {
      const int len = static_cast<int>(v.number());
      if (len > 0)
        rs.push_back({static_cast<Scheduler::Key>(std::stoi(c)), static_cast<std::uint64_t>(len), Tokens(std::stoi(c), len),
                      true});
    }
    s_->SyncRetained(std::move(rs));
  }
  static std::string Set(const std::set<std::int64_t>& s) {
    std::string out = "{";
    for (auto x : s) out += (out.size() > 1 ? "," : "") + std::to_string(x);
    return out + "}";
  }

  std::unique_ptr<Scheduler> s_;
  std::map<std::int64_t, Scheduler::Id> cid_;  // model id -> code id
  std::map<Scheduler::Id, std::int64_t> mid_;
  std::map<std::int64_t, int> conv_, prompt_;
  const Value* parks_ = nullptr;
  std::set<std::int64_t> fenced_code_, fenced_model_;
  Scheduler::Unit unit_;
};

std::vector<std::filesystem::path> Traces() {
  std::vector<std::filesystem::path> out;
  const std::filesystem::path dir = std::getenv("HALO_SCHED_TRACE_DIR") ? std::getenv("HALO_SCHED_TRACE_DIR")
                                                                         : HALO_SERVER_TEST_DATA_DIR "/sched_traces";
  if (std::filesystem::exists(dir))
    for (const auto& f : std::filesystem::directory_iterator(dir))
      if (f.path().extension() == ".jsonl") out.push_back(f.path());
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

TEST(SchedulerReplay, ModelTracesMatchTheCodeStepByStep) {
  const auto traces = Traces();
  ASSERT_FALSE(traces.empty()) << "no traces: spec/scheduler/replay.sh writes them";
  std::uint64_t events = 0, compared = 0, empty = 0;
  for (const auto& path : traces) {
    std::ifstream in(path);
    Replay r;
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
      ++n;
      std::string err;
      const auto e = halo::json::Parse(line, &err);
      ASSERT_TRUE(e) << path << ":" << n << ": " << err;
      const std::string why = r.Step(*e);
      ASSERT_TRUE(why.empty()) << path.filename().string() << ":" << n << " " << line << "\n  " << why;
    }
    const std::string why = r.Finish();
    ASSERT_TRUE(why.empty()) << path.filename().string() << " at the end\n  " << why;
    events += static_cast<std::uint64_t>(n);
    compared += r.compared;
    empty += r.empty_slices;
  }
  std::printf("replayed %zu traces, %llu events, %llu compared steps (%llu empty slices)\n", traces.size(),
              static_cast<unsigned long long>(events), static_cast<unsigned long long>(compared),
              static_cast<unsigned long long>(empty));
}
