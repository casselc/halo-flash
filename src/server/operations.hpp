#ifndef HALO_SERVER_OPERATIONS_HPP_
#define HALO_SERVER_OPERATIONS_HPP_

// The executor's operation records (docs/OPERATIONS.md; cordis RFC 0007; model spec/operation/operation.qnt).
//
// Host-only bookkeeping for the operation verbs: what this incarnation accepted, what it is running, the results it
// holds, and what it must remember after releasing them. The engine work is the caller's: an accepted attempt runs
// as an ordinary request through the scheduler, and the caller reports its end with Finish().
//
// Every verb carries the attempt's identity (owner, operation, attempt), the incarnation it is meant for, the
// attempt's not-after and its binding digest (a `Ref`). The rules, each one a rule of the model:
//  - fence: a verb for another incarnation acts on nothing (status answers kUnknown, the others kStale);
//  - binding: a verb naming another digest than the attempt's record is a conflict and acts on nothing;
//  - deadline: a submit past its not-after is late; an attempt with no record past its not-after is kClosed
//    (never accepted, and no copy can be any more); before it, kNeverAccepted (not yet);
//  - tomb: an attempt's record stays until its not-after has passed and, if it ran, its outcome was acknowledged:
//    a duplicate or late submit finds it;
//  - outcome kept: a finished result's body is held until acknowledged or until its capture window ends; after the
//    window it is kExpired, and its usage is kept until acknowledged;
//  - reserve: a submit is accepted only if its reservation fits the retention budget beside every held attempt's
//    (from acceptance until the body is released), so results finishing together cannot overflow it;
//  - ceiling: records are capped; a submit or a cancel that would need one more is refused, never served by
//    erasing a record.
//
// Time is the incarnation's clock in ms (Incarnation::NowMs), passed in by the caller. Thread-safe (one mutex).

#include <compare>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>

namespace halo::server {

class Operations {
 public:
  struct Options {
    std::int64_t capture_ms = 3600 * 1000;         ///< a finished body is held this long without an ack
    std::int64_t max_submit_age_ms = 600 * 1000;   ///< a not-after at most this far ahead
    std::uint64_t budget_bytes = 256ull << 20;     ///< retention budget: reservations of held attempts
    std::size_t max_records = 100000;              ///< records: held attempts, tombstones, unacknowledged outcomes
    std::int64_t retry_after_s = 5;                ///< Retry-After of a refusal
  };

  struct Id {
    std::string owner, op;
    std::uint64_t attempt = 0;
    auto operator<=>(const Id&) const = default;
  };

  /// What every verb carries.
  struct Ref {
    Id id;
    std::string incarnation;
    std::int64_t not_after = 0;  ///< in the incarnation's clock (ms)
    std::string digest;          ///< the binding: api, request, not_after and mode
  };

  struct Usage {
    std::uint64_t prompt = 0, cached = 0, completion = 0;
    double engine_ms = 0;
  };

  enum class Status : std::uint8_t {
    kRunning,                ///< accepted, executing
    kDone,                   ///< finished; the body is held until acked or expires_at
    kReleased,               ///< acknowledged and released
    kExpired,                ///< finished, the capture window passed: body gone, usage kept until acked
    kCancelledBeforeAccept,  ///< cancelled before any copy was accepted (terminal, settle at 0)
    kClosed,                 ///< never accepted and past not-after (terminal, settle at 0)
    kNeverAccepted,          ///< not accepted yet: a delayed copy may still arrive
    kUnknown,                ///< the verb was meant for another incarnation
  };

  struct View {
    Status status = Status::kNeverAccepted;
    std::string finish;            ///< kDone / kReleased / kExpired: the finish reason ("stop", "length", "cancelled", ...)
    Usage usage;                   ///< kDone / kReleased / kExpired
    std::int64_t expires_at = -1;  ///< kDone: when the body is released without an ack
    bool cancel_requested = false; ///< kRunning: a cancel arrived; it finishes at the next unit boundary
  };

  enum class Verdict : std::uint8_t {
    kOk,                ///< the verb took effect (accepted, cancelled, released) or, for status, answered
    kDuplicate,         ///< submit: the attempt already has a record; `view` is its status
    kStale,             ///< meant for another incarnation: acts on nothing (409)
    kConflict,          ///< another digest than the attempt's record (409)
    kLate,              ///< submit: past its not-after (410)
    kTooFarAhead,       ///< not-after more than max_submit_age_ms ahead (400)
    kRefusedRetention,  ///< submit: the reservation does not fit the retention budget (503, Retry-After)
    kRefusedRecords,    ///< submit or cancel: at the record ceiling (503, Retry-After)
  };

  struct Answer {
    Verdict verdict = Verdict::kOk;
    View view;                     ///< the attempt's status after the verb
    std::int64_t retry_after_s = 0;
  };

  struct Stats {
    std::size_t records = 0, held = 0, unacked_outcomes = 0;
    std::uint64_t reserved_bytes = 0;
    std::int64_t oldest_unacked_ms = 0;  ///< age of the oldest finished, unacknowledged outcome (0: none)
  };

  Operations(std::string incarnation, Options options);

  /// Accepts the attempt with a retention reservation of `reserve_bytes`, or says why not.
  Answer Submit(const Ref& ref, std::uint64_t reserve_bytes, std::int64_t now);
  Answer StatusOf(const Ref& ref, std::int64_t now);
  /// The result body of a kDone attempt (nullopt otherwise: see Status).
  std::optional<std::string> Fetch(const Ref& ref, std::int64_t now);
  /// Before acceptance: tombstones the attempt (kCancelledBeforeAccept). Running: asks it to stop (kRunning with
  /// cancel_requested). Finished: no change.
  Answer Cancel(const Ref& ref, std::int64_t now);
  /// The platform has durably committed the result: release the body (kReleased), or acknowledge an expired outcome.
  Answer Ack(const Ref& ref, std::int64_t now);

  // ---- the executor's side ----
  /// The running attempt finished (a cancelled one too, finish "cancelled"), with its actual usage. False if the
  /// attempt is not running here.
  bool Finish(const Id& id, std::string body, Usage usage, std::string finish, std::int64_t now);
  /// Whether a cancel asked the running attempt to stop.
  [[nodiscard]] bool CancelRequested(const Id& id) const;

  [[nodiscard]] Stats stats(std::int64_t now) const;
  [[nodiscard]] const std::string& incarnation() const { return incarnation_; }
  [[nodiscard]] const Options& options() const { return opt_; }

 private:
  enum class St : std::uint8_t { kRunning, kDone, kReleased, kExpired, kCancelled };
  struct Rec {
    St st = St::kRunning;
    std::int64_t not_after = 0;
    std::string digest;
    std::uint64_t reserve = 0;  ///< held while kRunning or kDone
    bool cancel = false;
    bool acked = false;
    std::int64_t done_at = 0;
    std::string body, finish;
    Usage usage;
  };

  /// Expires bodies past their window and prunes records whose not-after has passed and whose outcome (if any) was
  /// acknowledged. Under mu_.
  void Sweep(std::int64_t now);
  /// The answer for an id with no record, against the deadline the verb carries.
  [[nodiscard]] static View Absent(const Ref& ref, std::int64_t now);
  [[nodiscard]] View ViewOf(const Rec& r) const;
  void Release(Rec& r);

  const std::string incarnation_;
  const Options opt_;
  mutable std::mutex mu_;
  std::map<Id, Rec> recs_;
  std::uint64_t reserved_ = 0;
};

[[nodiscard]] const char* StatusName(Operations::Status s);

}  // namespace halo::server

#endif  // HALO_SERVER_OPERATIONS_HPP_
