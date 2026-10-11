#ifndef HALO_SERVER_SCHEDULER_HPP_
#define HALO_SERVER_SCHEDULER_HPP_

// G3: the fair GPU scheduler of halo_serve (docs/SCHEDULER.md; model spec/scheduler/scheduler.qnt).
//
// Host-only policy and bookkeeping; the engine work is the caller's. One engine thread runs one unit of GPU work
// at a time: a prefill slice of one request, a batched decode unit over every decoding request, or a decider
// evaluation. Between units it calls Next(), which
//   1. fences: every admitted request whose client is known gone leaves (a started prefill parks its
//      slice-aligned progress as a retained prefix: Hooks::park);
//   2. carries out the evictions admissions decided (Hooks::evict);
//   3. picks the next unit: a decider with credit first, then two-level start-time fair queuing by GPU time:
//      the chain root with the least virtual time, then the principal (sandbox) with the least virtual time
//      within that root, then FIFO within the principal (A2A-MCP D9: a chain that fans out to several sandboxes
//      holds one share, not one per sandbox).
// HTTP threads call Arrive() under the same lock: idempotency (replay a finished turn, adopt an in-flight one,
// drop a late duplicate), the per-principal in-flight limit, then the reservation ledger (retained prefixes plus
// every admitted request's reservation, never the memory reading alone), evicting unpinned retained prefixes
// least recently used first. A refusal is a 429 with Retry-After and the request never enters the queue (the
// answer and the removal are one critical section).
//
// Time is the caller's unit (`Options::per_second` of them a second): ms in halo_serve, ticks in the tests that
// replay the model. Every method is thread-safe (one mutex); the hooks run under it, on the thread that called
// Next().

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace halo::server {

class Scheduler {
 public:
  using Id = std::uint64_t;
  using Key = std::uint64_t;  // a retained prefix (the session cache's entry key)

  struct Options {
    std::int64_t per_second = 1000;      ///< time units per second
    std::uint32_t slice_contended = 2048;  ///< prefill slice while anyone else is runnable (D1)
    std::uint32_t slice_alone = 8192;      ///< ... and when the prefill is alone
    std::uint32_t plimit = 1;              ///< active agent requests per principal (D3)
    /// ... for callers with no principal (not through the broker: one shared principal). Above 1 so that an
    /// un-brokered harness's concurrent requests (a title request beside the turn) queue as they did before G3.
    std::uint32_t anonymous_plimit = 8;
    std::uint32_t max_batch = 8;           ///< decode unit: at most this many requests (StepBatch)
    std::uint32_t decode_steps = 8;        ///< decode unit: steps per unit at B = 1 (fewer at larger B, >= 2)
    std::int64_t decode_unit_target = 0;   ///< decode unit length to aim for (0: decode_steps x step cost at B=1)
    double decider_burst = 4;              ///< decider credit: evaluations banked (D4)
    double decider_refill_per_s = 2;       ///< ... refilled per second
    std::uint64_t capacity = 0;            ///< ledger capacity (bytes); 0: no memory admission
    std::int64_t retry_after_memory_s = 5;  ///< Retry-After of a memory 429
    std::int64_t retry_after_cap_s = 30;     ///< cap of a limit 429's Retry-After
    std::int64_t replay_ttl_s = 600;        ///< a finished key is replayed for this long
    /// A departed request with an idempotency key keeps running this long after the server learns its client left
    /// before it is fenced: a retry with the same key adopts it, or replays it if it finished meanwhile (a
    /// non-streaming caller whose answer outlasts its timeout). 0: fenced at the next boundary like any other.
    double keyed_grace_s = 30;
    std::size_t max_keys = 4096;            ///< finished keys remembered (oldest forgotten first)
    // Cost estimates in time units (the charge at a unit's start; Done() corrects it to the measured length).
    double slice_cost_per_token = 0.61;    ///< ms per prefill token (BENCHMARKS)
    double step_cost = 37;                 ///< ms per decode step at B = 1
    double batch_step_extra = 15;          ///< ms each further batch entry adds to a step (B=2 52.9, B=4 80.3)
    double decider_cost = 150;             ///< ms per decider evaluation
  };

  /// What admission needs to know about a request.
  struct Turn {
    bool decider = false;
    std::string principal;  ///< sandbox (X-Halo-Principal); empty: the shared anonymous principal
    std::string root;       ///< chain root (X-Halo-Chain-Id); empty: the principal itself
    std::string key;        ///< idempotency key; empty: none
    std::uint64_t prompt_tokens = 0, max_tokens = 0;
    /// Prompt tokens, for planning which retained prefix the request will reuse (may be null: no reuse planned).
    std::shared_ptr<const std::vector<std::int32_t>> prompt;
    double bytes_per_token = 0;  ///< ledger bytes a position costs
    std::uint64_t fixed_bytes = 0;  ///< ledger bytes a request costs besides its positions (recurrent state, slack)
  };

  enum class Outcome : std::uint8_t {
    kQueued,     ///< admitted: it will run
    kReplayed,   ///< its key finished: answer with the stored result of `target`; no engine work
    kAdopted,    ///< its key is in flight: this connection takes over `target`
    kDuplicate,  ///< a late duplicate of a key in flight whose client is gone: dropped, no answer owed
    kLimit,      ///< 429: the principal has `plimit` active requests
    kMemory,     ///< 429: no room even after every evictable prefix went
  };
  struct Admission {
    Outcome outcome = Outcome::kQueued;
    Id id = 0;          ///< this request (kQueued)
    Id target = 0;      ///< the earlier attempt (kReplayed, kAdopted)
    std::int64_t retry_after_s = 0;  ///< kLimit, kMemory
  };

  /// A retained prefix the ledger counts (the session cache's resident entries).
  struct Retained {
    Key key = 0;
    std::uint64_t bytes = 0;
    std::shared_ptr<const std::vector<std::int32_t>> tokens;  ///< its sequence (planning); may be null
    bool usable = true;  ///< may be reused (exact mode: canonical)
  };

  /// The engine side, called under the scheduler's lock from Next().
  struct Hooks {
    /// A started request leaves: its live session goes; what it computed that is worth keeping (a prefill's
    /// slice-aligned progress, exact) is parked as a retained prefix and returned (nullopt: nothing parked).
    std::function<std::optional<Retained>(Id)> park;
    /// Evict a retained prefix (admission decided it); returns the device bytes it freed.
    std::function<std::uint64_t(Key)> evict;
  };

  enum class Kind : std::uint8_t { kNone, kSlice, kBatch, kDecide };
  struct Unit {
    Kind kind = Kind::kNone;
    std::vector<Id> ids;             ///< kSlice / kDecide: one; kBatch: every decoding request (<= max_batch)
    std::uint32_t slice_tokens = 0;  ///< kSlice: prompt tokens to prefill at most
    std::uint32_t steps = 0;         ///< kBatch: decode steps
    std::int64_t charged = 0;        ///< the estimate charged at the start (time units, per participant)
    std::vector<Id> fenced;          ///< requests that left at this boundary (their HTTP threads are told)
  };
  /// What a unit did, per request (Done()).
  struct Progress {
    Id id = 0;
    std::uint64_t prefilled = 0;   ///< prompt tokens prefilled by this unit
    std::uint64_t decoded = 0;     ///< tokens generated by this unit
    bool prefill_done = false;     ///< no prompt token left to prefill (the next unit decodes)
    bool finished = false;         ///< the request ended (stop, max_tokens, error): it leaves
    bool completed = false;        ///< finished with its whole answer (its key replays it)
  };

  explicit Scheduler(Options options, Hooks hooks = {});
  /// Replaces the hooks (before any Next()).
  void SetHooks(Hooks hooks) {
    std::lock_guard<std::mutex> lk(mu_);
    hooks_ = std::move(hooks);
  }

  Admission Arrive(const Turn& turn, std::int64_t now, bool connected = true);
  /// The client of `id` is known gone at `now` (FIN, a failed write, the send deadline): fenced at the next
  /// boundary, or, for a keyed request, at the first boundary after its grace (Options::keyed_grace_s).
  void Depart(Id id, std::int64_t now);
  /// Engine thread: fence, evict, pick. kNone: nothing runnable (wait for an arrival).
  Unit Next(std::int64_t now);
  /// A boundary with no pick: fences and evictions as Next() does them; returns the requests fenced. (The trace
  /// replay calls it where the model fences between units; the engine thread needs only Next().)
  std::vector<Id> Boundary(std::int64_t now);
  /// Engine thread, at the start of a request's first unit: its live state was bound from retained prefix
  /// `bound` (0: none) holding `reused` prompt tokens (bind at start: a departed predecessor may have parked a
  /// longer prefix since admission). The reservation shrinks to what is left; the planned prefix's pin goes.
  void Bind(Id id, Key bound, std::uint64_t reused);
  /// Engine thread: the unit ended at `now` after `elapsed` time units.
  void Done(const Unit& unit, const std::vector<Progress>& progress, std::int64_t now, std::int64_t elapsed);
  /// Engine thread: the retained prefixes as they are now (after a finish, a park or the cache's own eviction).
  void SyncRetained(std::vector<Retained> retained);

  // ---- views (tests, metrics) ----
  enum class Phase : std::uint8_t { kQueued, kRunning, kFinished, kCancelled };
  struct View {
    Id id = 0;
    bool decider = false, prio = false, departed = false, started = false, prefill_done = false, chained = false;
    std::string principal, root, key;
    Phase phase = Phase::kQueued;
    std::uint64_t need = 0, inherited = 0, reused = 0, prefilled = 0, decoded = 0;
    Key planned = 0, bound = 0;
    std::int64_t admitted_at = 0, last_served = 0;
  };
  [[nodiscard]] std::vector<View> Requests() const;  ///< active (queued or running) requests
  [[nodiscard]] std::optional<View> Request(Id id) const;
  struct Ledger {
    std::uint64_t capacity = 0, retained = 0, reserved = 0, evicting = 0;
  };
  [[nodiscard]] Ledger ledger() const;
  [[nodiscard]] std::vector<Retained> retained() const;
  [[nodiscard]] double credit(std::int64_t now) const;
  [[nodiscard]] std::int64_t vtime_root(const std::string& root) const;
  [[nodiscard]] std::int64_t vtime_principal(const std::string& principal) const;
  struct Counters {
    std::uint64_t admitted = 0, replayed = 0, adopted = 0, duplicates = 0, limit429 = 0, memory429 = 0,
                  evictions = 0, fenced = 0, parked = 0, grace_finished = 0, units = 0, slices = 0, batches = 0, batched = 0, decides = 0,
                  preemptions = 0, decider_out_of_credit = 0;
  };
  [[nodiscard]] Counters counters() const;
  [[nodiscard]] const Options& options() const { return opt_; }
  static constexpr const char* kDeciderClass = "systemone";  ///< the deciders' principal and root
  static constexpr const char* kAnonymous = "anonymous";     ///< callers with no principal

 private:
  struct Req {
    Id id = 0;
    bool decider = false, prio = false, departed = false, started = false, prefill_done = false, chained = false;
    std::string principal, root, key;
    Phase phase = Phase::kQueued;
    std::uint64_t prompt_tokens = 0, max_tokens = 0, need = 0, inherited = 0, reused = 0, prefilled = 0,
                  decoded = 0;
    double bytes_per_token = 0;
    std::uint64_t fixed_bytes = 0;
    Key planned = 0, bound = 0;
    std::int64_t admitted_at = 0, last_served = 0;
    std::int64_t departed_at = 0;  ///< when the server learned its client left (departed)
    /// Virtual time within its principal (the third fairness level): the principal's requests share its turns, so a
    /// newcomer's prefill is not queued behind an older request's whole decode (the anonymous pool shares one
    /// principal: under FIFO its requests ran one at a time and never batched).
    std::int64_t vreq = 0;
    bool completed = false;
  };
  struct KeyState {
    Id id = 0;
    bool finished = false;
    std::int64_t at = 0;
  };
  struct Entry {
    Retained r;
    std::uint64_t used = 0;  // LRU clock
  };

  bool Active(const Req& r) const { return r.phase == Phase::kQueued || r.phase == Phase::kRunning; }
  bool Runnable(const Req& r, std::int64_t now) const;
  bool InGrace(const Req& r, std::int64_t now) const;
  std::uint64_t Reserved() const;
  std::uint64_t RetainedBytes() const;
  std::uint64_t Free() const;
  Key Plan(const Turn& turn, std::uint64_t* tokens) const;
  void Join(const Req& r, std::int64_t now);
  void Refill(std::int64_t now);
  void Finish(Req& r, bool completed, std::int64_t now);
  void Fence(std::int64_t now, Unit* unit);
  void FenceAndEvict(std::int64_t now, Unit* unit);
  std::int64_t RemainingEstimate(const Req& r) const;
  std::int64_t SliceCost(std::uint32_t tokens) const;
  std::int64_t BatchStepCost(std::size_t b) const;
  void ForgetKeys(std::int64_t now);
  View ToView(const Req& r) const;

  const Options opt_;
  Hooks hooks_;
  mutable std::mutex mu_;
  std::map<Id, Req> reqs_;            // active ones, and finished / cancelled ones until their key is forgotten
  std::map<std::string, KeyState> keys_;
  std::map<Key, Entry> retained_;
  std::vector<Key> evicting_;         // decided by an admission, carried out at the next boundary
  std::uint64_t evicting_bytes_ = 0;  // ... their bytes (no longer counted as retained)
  std::map<std::string, std::int64_t> vroot_, vprin_, pvnow_;  // virtual times (per root, per principal, per root's own)
  std::int64_t vnow_ = 0;
  std::int64_t credit_ = 0;  // decider evaluations banked
  std::int64_t credit_at_ = 0;
  Id next_id_ = 1;
  std::uint64_t clock_ = 0;
  Id last_slice_ = 0;  // the request whose slice ran last (preemption count)
  Counters counters_;
};

}  // namespace halo::server

#endif  // HALO_SERVER_SCHEDULER_HPP_
