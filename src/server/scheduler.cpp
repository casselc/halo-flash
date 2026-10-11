#include "server/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace halo::server {

namespace {
constexpr std::int64_t kNoTag = std::numeric_limits<std::int64_t>::max();
}

Scheduler::Scheduler(Options options, Hooks hooks) : opt_(options), hooks_(std::move(hooks)) {
  credit_ = static_cast<std::int64_t>(opt_.decider_burst);
}

// ---------------------------------------------------------------- helpers (under mu_)

bool Scheduler::Runnable(const Req& r) const { return Active(r) && !r.departed; }

std::uint64_t Scheduler::Reserved() const {
  std::uint64_t s = 0;
  for (const auto& [id, r] : reqs_)
    if (Active(r) && !r.decider) s += r.need + r.inherited;
  return s;
}

std::uint64_t Scheduler::RetainedBytes() const {
  std::uint64_t s = 0;
  for (const auto& [k, e] : retained_) s += e.r.bytes;
  return s;
}

std::uint64_t Scheduler::Free() const {
  const std::uint64_t used = RetainedBytes() + Reserved();
  return used >= opt_.capacity ? 0 : opt_.capacity - used;
}

Scheduler::Key Scheduler::Plan(const Turn& turn, std::uint64_t* tokens) const {
  *tokens = 0;
  if (!turn.prompt || turn.prompt->empty()) return 0;
  const auto& p = *turn.prompt;
  Key best = 0;
  for (const auto& [k, e] : retained_) {
    if (!e.r.usable || !e.r.tokens) continue;
    const auto& t = *e.r.tokens;
    if (t.empty() || t.size() > p.size() - 1 || t.size() <= *tokens) continue;
    if (!std::equal(t.begin(), t.end(), p.begin())) continue;
    best = k;
    *tokens = t.size();
  }
  return best;
}

// Start-time fair queuing, two levels: a root that becomes backlogged joins at the system virtual time `vnow_` (the
// least tag among the backlogged roots at the last pick: it banks no credit while idle), and a principal joins at
// its root's own system virtual time.
void Scheduler::Join(const Req& r) {
  bool root_busy = false, prin_busy = false;
  for (const auto& [id, x] : reqs_) {
    if (x.id == r.id || !Runnable(x)) continue;
    root_busy = root_busy || x.root == r.root;
    prin_busy = prin_busy || x.principal == r.principal;
  }
  if (!root_busy) vroot_[r.root] = std::max(vroot_[r.root], vnow_);
  if (!prin_busy) vprin_[r.principal] = std::max(vprin_[r.principal], pvnow_[r.root]);
}

void Scheduler::Refill(std::int64_t now) {
  if (opt_.decider_refill_per_s <= 0) return;
  const auto period = std::max<std::int64_t>(1, std::llround(static_cast<double>(opt_.per_second) / opt_.decider_refill_per_s));
  const std::int64_t add = now / period - credit_at_ / period;
  if (add > 0) credit_ = std::min<std::int64_t>(static_cast<std::int64_t>(opt_.decider_burst), credit_ + add);
  credit_at_ = std::max(credit_at_, now);
}

std::int64_t Scheduler::SliceCost(std::uint32_t tokens) const {
  return std::max<std::int64_t>(1, std::llround(opt_.slice_cost_per_token * tokens));
}

std::int64_t Scheduler::BatchStepCost(std::size_t b) const {
  return std::max<std::int64_t>(1, std::llround(opt_.step_cost + opt_.batch_step_extra * (b > 0 ? b - 1 : 0)));
}

std::int64_t Scheduler::RemainingEstimate(const Req& r) const {
  const std::uint64_t pre = r.prompt_tokens > r.reused + r.prefilled ? r.prompt_tokens - r.reused - r.prefilled : 0;
  const std::uint64_t dec = r.max_tokens > r.decoded ? r.max_tokens - r.decoded : 0;
  return std::llround(opt_.slice_cost_per_token * static_cast<double>(pre) + opt_.step_cost * static_cast<double>(dec));
}

void Scheduler::ForgetKeys(std::int64_t now) {
  const std::int64_t ttl = opt_.replay_ttl_s * opt_.per_second;
  for (auto it = keys_.begin(); it != keys_.end();) {
    if (it->second.finished && now - it->second.at > ttl) it = keys_.erase(it);
    else ++it;
  }
  // Bound the finished keys kept: the oldest go first.
  std::size_t finished = 0;
  for (const auto& [k, s] : keys_) finished += s.finished;
  while (finished > opt_.max_keys) {
    auto oldest = keys_.end();
    for (auto it = keys_.begin(); it != keys_.end(); ++it)
      if (it->second.finished && (oldest == keys_.end() || it->second.at < oldest->second.at)) oldest = it;
    keys_.erase(oldest);
    --finished;
  }
  // Finished and cancelled requests no key refers to are gone.
  for (auto it = reqs_.begin(); it != reqs_.end();) {
    const Req& r = it->second;
    const bool keyed = !r.key.empty() && keys_.count(r.key) && keys_.at(r.key).id == r.id;
    if (!Active(r) && !keyed) it = reqs_.erase(it);
    else ++it;
  }
}

void Scheduler::Finish(Req& r, bool completed, std::int64_t now) {
  r.phase = Phase::kFinished;
  r.completed = completed;
  if (r.key.empty()) return;
  auto it = keys_.find(r.key);
  if (it == keys_.end() || it->second.id != r.id) return;
  if (completed) it->second = {r.id, true, now};
  else keys_.erase(it);  // nothing to replay: a retry is a new request
}

// ---------------------------------------------------------------- admission (HTTP threads)

Scheduler::Admission Scheduler::Arrive(const Turn& turn, std::int64_t now, bool connected) {
  std::lock_guard<std::mutex> lk(mu_);
  Refill(now);
  ForgetKeys(now);
  Req r;
  r.decider = turn.decider;
  r.principal = turn.decider ? kDeciderClass : turn.principal.empty() ? kAnonymous : turn.principal;
  r.root = turn.decider ? kDeciderClass : turn.root.empty() ? r.principal : turn.root;
  r.key = turn.key;
  r.prompt_tokens = turn.prompt_tokens;
  r.max_tokens = turn.max_tokens;
  r.bytes_per_token = turn.bytes_per_token;
  r.fixed_bytes = turn.fixed_bytes;
  r.admitted_at = r.last_served = now;
  r.departed = !connected;
  Admission a;
  if (r.decider) {
    // A decider with credit runs at the next unit boundary; out of credit it is the decider principal's turn in
    // the fair queue (D4: burst 4, refill 2/s).
    r.prio = credit_ > 0;
    if (r.prio) --credit_;
    else ++counters_.decider_out_of_credit;
    r.id = next_id_++;
    Join(r);
    reqs_[r.id] = r;
    ++counters_.admitted;
    return a.id = r.id, a;
  }
  // 1. idempotency: a finished key replays; an in-flight one is adopted (its client left, or this is that
  // client's retry over a connection the server has not seen fail yet); a late copy of an in-flight key whose
  // sender is gone is dropped.
  if (!r.key.empty()) {
    if (auto it = keys_.find(r.key); it != keys_.end()) {
      if (it->second.finished) {
        ++counters_.replayed;
        a.outcome = Outcome::kReplayed;
        a.target = it->second.id;
        return a;
      }
      auto rit = reqs_.find(it->second.id);
      if (rit != reqs_.end() && Active(rit->second)) {
        if (!connected) {
          ++counters_.duplicates;
          a.outcome = Outcome::kDuplicate;
          a.target = rit->second.id;
          return a;
        }
        rit->second.departed = false;
        ++counters_.adopted;
        a.outcome = Outcome::kAdopted;
        a.target = rit->second.id;
        return a;
      }
    }
  }
  // 2. the principal's in-flight limit (D3): its active requests whose clients are not known gone.
  std::int64_t longest = 0;
  std::uint32_t mine = 0;
  for (const auto& [id, x] : reqs_)
    if (Active(x) && !x.decider && x.principal == r.principal && !x.departed) {
      ++mine;
      longest = std::max(longest, RemainingEstimate(x));
    }
  if (mine >= (r.principal == kAnonymous ? opt_.anonymous_plimit : opt_.plimit)) {
    ++counters_.limit429;
    a.outcome = Outcome::kLimit;
    a.retry_after_s = std::clamp<std::int64_t>((longest + opt_.per_second - 1) / opt_.per_second, 1, opt_.retry_after_cap_s);
    return a;
  }
  // 3. memory: the reservation against the ledger, evicting unpinned retained prefixes least recently used first.
  std::uint64_t reuse = 0;
  r.planned = Plan(turn, &reuse);
  const auto positions = static_cast<double>(r.prompt_tokens - std::min(reuse, r.prompt_tokens) + r.max_tokens);
  r.need = static_cast<std::uint64_t>(std::ceil(positions * r.bytes_per_token)) + r.fixed_bytes;
  if (opt_.capacity > 0) {
    std::set<Key> pinned;
    if (r.planned) pinned.insert(r.planned);
    for (const auto& [id, x] : reqs_)
      if (Active(x) && !x.started && x.planned) pinned.insert(x.planned);
    while (r.need > Free()) {
      Key lru = 0;
      std::uint64_t lru_used = std::numeric_limits<std::uint64_t>::max();
      for (const auto& [k, e] : retained_)
        if (!pinned.count(k) && e.used < lru_used) lru = k, lru_used = e.used;
      if (lru == 0) break;
      // The prefix goes; a started request forked from it keeps the pages it shares (they become its own).
      const std::uint64_t bytes = retained_[lru].r.bytes;
      bool forked = false;
      for (auto& [id, x] : reqs_)
        if (Active(x) && !x.decider && x.started && x.bound == lru && x.inherited == 0) {
          x.inherited = bytes;
          forked = true;
        }
      retained_.erase(lru);
      evicting_.push_back(lru);
      evicting_bytes_ += forked ? 0 : bytes;
      ++counters_.evictions;
    }
    if (r.need > Free()) {
      // Refused: the evictions already made stay made (as SessionCache's own loop does).
      ++counters_.memory429;
      a.outcome = Outcome::kMemory;
      a.retry_after_s = opt_.retry_after_memory_s;
      return a;
    }
  }
  r.id = next_id_++;
  // A request that shares its chain root with another sandbox's request is chained (both ways).
  for (auto& [id, x] : reqs_)
    if (Active(x) && !x.decider && x.root == r.root && x.principal != r.principal) x.chained = r.chained = true;
  Join(r);
  reqs_[r.id] = r;
  if (!r.key.empty()) keys_[r.key] = {r.id, false, now};
  ++counters_.admitted;
  return a.id = r.id, a;
}

void Scheduler::Depart(Id id) {
  std::lock_guard<std::mutex> lk(mu_);
  if (auto it = reqs_.find(id); it != reqs_.end() && Active(it->second)) it->second.departed = true;
}

// ---------------------------------------------------------------- the engine thread

void Scheduler::Fence(std::int64_t now, Unit* unit) {
  for (auto& [id, r] : reqs_) {
    if (!Active(r) || !r.departed) continue;
    if (r.started && !r.decider) {
      std::optional<Retained> parked;
      if (hooks_.park) parked = hooks_.park(id);
      if (parked && parked->key) {
        retained_[parked->key] = {*parked, ++clock_};
        ++counters_.parked;
      }
    }
    r.phase = Phase::kCancelled;
    if (!r.key.empty())
      if (auto it = keys_.find(r.key); it != keys_.end() && it->second.id == id) keys_.erase(it);
    unit->fenced.push_back(id);
    ++counters_.fenced;
  }
  (void)now;
}

void Scheduler::FenceAndEvict(std::int64_t now, Unit* u) {
  Refill(now);
  Fence(now, u);
  for (Key k : evicting_)
    if (hooks_.evict) (void)hooks_.evict(k);
  evicting_.clear();
  evicting_bytes_ = 0;
}

std::vector<Scheduler::Id> Scheduler::Boundary(std::int64_t now) {
  std::lock_guard<std::mutex> lk(mu_);
  Unit u;
  FenceAndEvict(now, &u);
  return u.fenced;
}

Scheduler::Unit Scheduler::Next(std::int64_t now) {
  std::lock_guard<std::mutex> lk(mu_);
  Unit u;
  FenceAndEvict(now, &u);

  std::vector<Req*> run;
  for (auto& [id, r] : reqs_)
    if (Runnable(r)) run.push_back(&r);
  if (run.empty()) return u;

  // The system virtual times, before this pick's charge: the least tag among the backlogged roots, and within
  // each backlogged root among its backlogged principals.
  std::map<std::string, std::int64_t> root_min;
  std::int64_t vmin = kNoTag;
  for (Req* r : run) {
    vmin = std::min(vmin, vroot_[r->root]);
    auto& m = root_min.try_emplace(r->root, kNoTag).first->second;
    m = std::min(m, vprin_[r->principal]);
  }
  vnow_ = std::max(vnow_, vmin);
  for (const auto& [root, m] : root_min) pvnow_[root] = std::max(pvnow_[root], m);

  Req* pick = nullptr;
  for (Req* r : run)  // a decider with credit first (FIFO among them)
    if (r->decider && r->prio && (!pick || r->id < pick->id)) pick = r;
  if (!pick) {
    // The root with the least tag, then the principal with the least tag within it, then FIFO within the
    // principal; ties go to the one holding the oldest request.
    std::map<std::string, Id> root_first, prin_first;
    for (Req* r : run) {
      auto& rf = root_first.try_emplace(r->root, r->id).first->second;
      rf = std::min(rf, r->id);
      auto& pf = prin_first.try_emplace(r->principal, r->id).first->second;
      pf = std::min(pf, r->id);
    }
    std::string root;
    for (const auto& [q, first] : root_first)
      if (root.empty() || std::make_pair(vroot_[q], first) < std::make_pair(vroot_[root], root_first[root])) root = q;
    std::string prin;
    for (Req* r : run)
      if (r->root == root && (prin.empty() || std::make_pair(vprin_[r->principal], prin_first[r->principal]) <
                                                  std::make_pair(vprin_[prin], prin_first[prin])))
        prin = r->principal;
    for (Req* r : run)
      if (r->root == root && r->principal == prin && (!pick || r->id < pick->id)) pick = r;
  }

  std::vector<Req*> in;
  if (pick->decider) {
    u.kind = Kind::kDecide;
    u.charged = std::llround(opt_.decider_cost);
    in.push_back(pick);
    ++counters_.decides;
  } else if (!pick->prefill_done) {
    u.kind = Kind::kSlice;
    const bool alone = run.size() == 1;
    u.slice_tokens = alone ? opt_.slice_alone : opt_.slice_contended;
    const std::uint64_t left = pick->prompt_tokens > pick->reused + pick->prefilled
                                   ? pick->prompt_tokens - pick->reused - pick->prefilled : 1;
    u.charged = SliceCost(static_cast<std::uint32_t>(std::min<std::uint64_t>(u.slice_tokens, left)));
    in.push_back(pick);
    ++counters_.slices;
    for (Req* r : run)
      if (r != pick && r->started && !r->prefill_done && !r->decider && last_slice_ == r->id) ++counters_.preemptions;
    last_slice_ = pick->id;
  } else {
    // Every decoding request steps in the same unit (StepBatch); beyond max_batch, those first in fair order.
    u.kind = Kind::kBatch;
    std::vector<Req*> dec;
    for (Req* r : run)
      if (!r->decider && r->prefill_done) dec.push_back(r);
    std::sort(dec.begin(), dec.end(), [&](Req* a, Req* b) {
      return std::make_tuple(a != pick, vroot_[a->root], vprin_[a->principal], a->id) <
             std::make_tuple(b != pick, vroot_[b->root], vprin_[b->principal], b->id);
    });
    if (dec.size() > opt_.max_batch) dec.resize(opt_.max_batch);
    in = dec;
    const std::int64_t step = BatchStepCost(in.size());
    const double target = opt_.decode_unit_target > 0 ? static_cast<double>(opt_.decode_unit_target)
                                                       : opt_.decode_steps * opt_.step_cost;
    u.steps = std::max<std::uint32_t>(in.size() > 1 ? 2u : 1u, static_cast<std::uint32_t>(std::lround(target / static_cast<double>(step))));
    u.steps = std::min(u.steps, opt_.decode_steps);
    u.charged = step * u.steps;
    ++counters_.batches;
    if (in.size() >= 2) ++counters_.batched;
  }
  // Charge: each participant's root and principal pay the unit's estimate once.
  std::set<std::string> roots, prins;
  for (Req* r : in) {
    u.ids.push_back(r->id);
    roots.insert(r->root);
    prins.insert(r->principal);
    r->phase = Phase::kRunning;
    r->last_served = now;
  }
  for (const auto& q : roots) vroot_[q] += u.charged;
  for (const auto& q : prins) vprin_[q] += u.charged;
  ++counters_.units;
  return u;
}

void Scheduler::Done(const Unit& unit, const std::vector<Progress>& progress, std::int64_t now, std::int64_t elapsed) {
  std::lock_guard<std::mutex> lk(mu_);
  // GPU time actually used: the charge is corrected to the measured length.
  if (unit.kind != Kind::kNone && elapsed != unit.charged) {
    std::set<std::string> roots, prins;
    for (Id id : unit.ids)
      if (auto it = reqs_.find(id); it != reqs_.end()) {
        roots.insert(it->second.root);
        prins.insert(it->second.principal);
      }
    for (const auto& q : roots) vroot_[q] += elapsed - unit.charged;
    for (const auto& q : prins) vprin_[q] += elapsed - unit.charged;
  }
  for (const auto& p : progress) {
    auto it = reqs_.find(p.id);
    if (it == reqs_.end()) continue;
    Req& r = it->second;
    if (r.decider) r.started = true;
    r.prefilled += p.prefilled;
    r.decoded += p.decoded;
    r.prefill_done = r.prefill_done || p.prefill_done;
    r.last_served = now;
    if (p.finished && Active(r)) Finish(r, p.completed, now);
  }
}

void Scheduler::Bind(Id id, Key bound, std::uint64_t reused) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = reqs_.find(id);
  if (it == reqs_.end() || it->second.started) return;
  Req& r = it->second;
  r.started = true;
  r.bound = bound;
  r.reused = reused;
  r.planned = 0;
  const auto positions = static_cast<double>(r.prompt_tokens - std::min(r.reused, r.prompt_tokens) + r.max_tokens);
  r.need = std::min(r.need, static_cast<std::uint64_t>(std::ceil(positions * r.bytes_per_token)) + r.fixed_bytes);
}

void Scheduler::SyncRetained(std::vector<Retained> retained) {
  std::lock_guard<std::mutex> lk(mu_);
  std::map<Key, Entry> next;
  for (auto& x : retained) {
    if (std::find(evicting_.begin(), evicting_.end(), x.key) != evicting_.end()) continue;
    auto it = retained_.find(x.key);
    const std::uint64_t used = it != retained_.end() ? it->second.used : ++clock_;
    next[x.key] = {std::move(x), used};
  }
  // A prefix the cache dropped on its own: a started fork of it keeps the shared pages.
  for (const auto& [k, e] : retained_)
    if (!next.count(k))
      for (auto& [id, r] : reqs_)
        if (Active(r) && !r.decider && r.started && r.bound == k && r.inherited == 0) r.inherited = e.r.bytes;
  retained_ = std::move(next);
}

// ---------------------------------------------------------------- views

Scheduler::View Scheduler::ToView(const Req& r) const {
  View v;
  v.id = r.id;
  v.decider = r.decider;
  v.prio = r.prio;
  v.departed = r.departed;
  v.started = r.started;
  v.prefill_done = r.prefill_done;
  v.chained = r.chained;
  v.principal = r.principal;
  v.root = r.root;
  v.key = r.key;
  v.phase = r.phase;
  v.need = r.need;
  v.inherited = r.inherited;
  v.reused = r.reused;
  v.prefilled = r.prefilled;
  v.decoded = r.decoded;
  v.planned = r.planned;
  v.bound = r.bound;
  v.admitted_at = r.admitted_at;
  v.last_served = r.last_served;
  return v;
}

std::vector<Scheduler::View> Scheduler::Requests() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<View> out;
  for (const auto& [id, r] : reqs_)
    if (Active(r)) out.push_back(ToView(r));
  return out;
}

std::optional<Scheduler::View> Scheduler::Request(Id id) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (auto it = reqs_.find(id); it != reqs_.end()) return ToView(it->second);
  return std::nullopt;
}

Scheduler::Ledger Scheduler::ledger() const {
  std::lock_guard<std::mutex> lk(mu_);
  return {opt_.capacity, RetainedBytes(), Reserved(), evicting_bytes_};
}

std::vector<Scheduler::Retained> Scheduler::retained() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<Retained> out;
  for (const auto& [k, e] : retained_) out.push_back(e.r);
  return out;
}

double Scheduler::credit(std::int64_t now) const {
  std::lock_guard<std::mutex> lk(mu_);
  (void)now;
  return static_cast<double>(credit_);
}

std::int64_t Scheduler::vtime_root(const std::string& root) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = vroot_.find(root);
  return it == vroot_.end() ? 0 : it->second;
}

std::int64_t Scheduler::vtime_principal(const std::string& principal) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = vprin_.find(principal);
  return it == vprin_.end() ? 0 : it->second;
}

Scheduler::Counters Scheduler::counters() const {
  std::lock_guard<std::mutex> lk(mu_);
  return counters_;
}

}  // namespace halo::server
