#include "server/operations.hpp"

#include <algorithm>
#include <utility>

namespace halo::server {

const char* StatusName(Operations::Status s) {
  switch (s) {
    case Operations::Status::kRunning: return "running";
    case Operations::Status::kDone: return "done";
    case Operations::Status::kReleased: return "released";
    case Operations::Status::kExpired: return "expired";
    case Operations::Status::kCancelledBeforeAccept: return "cancelled_before_accept";
    case Operations::Status::kClosed: return "closed";
    case Operations::Status::kNeverAccepted: return "never_accepted";
    case Operations::Status::kUnknown: return "unknown";
  }
  return "unknown";
}

Operations::Operations(std::string incarnation, Options options)
    : incarnation_(std::move(incarnation)), opt_(options) {}

Operations::View Operations::Absent(const Ref& ref, std::int64_t now) {
  View v;
  v.status = now > ref.not_after ? Status::kClosed : Status::kNeverAccepted;
  return v;
}

Operations::View Operations::ViewOf(const Rec& r) const {
  View v;
  switch (r.st) {
    case St::kRunning:
      v.status = Status::kRunning;
      v.cancel_requested = r.cancel;
      return v;
    case St::kDone: v.status = Status::kDone; v.expires_at = r.done_at + opt_.capture_ms; break;
    case St::kReleased: v.status = Status::kReleased; break;
    case St::kExpired: v.status = Status::kExpired; break;
    case St::kCancelled: v.status = Status::kCancelledBeforeAccept; return v;
  }
  v.finish = r.finish;
  v.usage = r.usage;
  return v;
}

void Operations::Release(Rec& r) {
  reserved_ -= r.reserve;
  r.reserve = 0;
  r.body.clear();
  r.body.shrink_to_fit();
}

void Operations::Sweep(std::int64_t now) {
  for (auto it = recs_.begin(); it != recs_.end();) {
    Rec& r = it->second;
    // The capture window ends: the body goes, the outcome stays until acknowledged.
    if (r.st == St::kDone && now >= r.done_at + opt_.capture_ms) {
      Release(r);
      r.st = St::kExpired;
      r.acked = false;
    }
    // A record goes once its not-after has passed (no copy of its submit can be accepted any more) and, if it ran,
    // its outcome was acknowledged.
    const bool settled = r.st == St::kReleased || r.st == St::kCancelled || (r.st == St::kExpired && r.acked);
    if (settled && now > r.not_after) it = recs_.erase(it);
    else ++it;
  }
}

Operations::Answer Operations::Submit(const Ref& ref, std::uint64_t reserve_bytes, std::int64_t now) {
  std::lock_guard<std::mutex> lk(mu_);
  Sweep(now);
  Answer a;
  if (ref.incarnation != incarnation_) {
    a.verdict = Verdict::kStale;
    a.view.status = Status::kUnknown;
    return a;
  }
  if (const auto it = recs_.find(ref.id); it != recs_.end()) {
    a.verdict = it->second.digest == ref.digest ? Verdict::kDuplicate : Verdict::kConflict;
    a.view = ViewOf(it->second);
    return a;
  }
  a.view = Absent(ref, now);
  if (now > ref.not_after) return a.verdict = Verdict::kLate, a;
  if (ref.not_after > now + opt_.max_submit_age_ms) return a.verdict = Verdict::kTooFarAhead, a;
  if (recs_.size() >= opt_.max_records) {
    a.verdict = Verdict::kRefusedRecords;
    a.retry_after_s = opt_.retry_after_s;
    return a;
  }
  if (reserved_ + reserve_bytes > opt_.budget_bytes) {
    a.verdict = Verdict::kRefusedRetention;
    a.retry_after_s = opt_.retry_after_s;
    return a;
  }
  Rec r;
  r.not_after = ref.not_after;
  r.digest = ref.digest;
  r.reserve = reserve_bytes;
  reserved_ += reserve_bytes;
  a.view = ViewOf(recs_.emplace(ref.id, std::move(r)).first->second);
  return a;
}

Operations::Answer Operations::StatusOf(const Ref& ref, std::int64_t now) {
  std::lock_guard<std::mutex> lk(mu_);
  Sweep(now);
  Answer a;
  if (ref.incarnation != incarnation_) {
    a.verdict = Verdict::kStale;
    a.view.status = Status::kUnknown;
    return a;
  }
  const auto it = recs_.find(ref.id);
  if (it == recs_.end()) return a.view = Absent(ref, now), a;
  if (it->second.digest != ref.digest) a.verdict = Verdict::kConflict;
  a.view = ViewOf(it->second);
  return a;
}

std::optional<std::string> Operations::Fetch(const Ref& ref, std::int64_t now) {
  std::lock_guard<std::mutex> lk(mu_);
  Sweep(now);
  if (ref.incarnation != incarnation_) return std::nullopt;
  const auto it = recs_.find(ref.id);
  if (it == recs_.end() || it->second.digest != ref.digest || it->second.st != St::kDone) return std::nullopt;
  return it->second.body;
}

Operations::Answer Operations::Cancel(const Ref& ref, std::int64_t now) {
  std::lock_guard<std::mutex> lk(mu_);
  Sweep(now);
  Answer a;
  if (ref.incarnation != incarnation_) {
    a.verdict = Verdict::kStale;
    a.view.status = Status::kUnknown;
    return a;
  }
  if (const auto it = recs_.find(ref.id); it != recs_.end()) {
    Rec& r = it->second;
    if (r.digest != ref.digest) {
      a.verdict = Verdict::kConflict;
    } else if (r.st == St::kRunning) {
      r.cancel = true;  // finishes at the next unit boundary, with its actual usage
    }
    a.view = ViewOf(r);
    return a;
  }
  a.view = Absent(ref, now);
  if (now > ref.not_after) return a;  // already closed: nothing to fence
  if (ref.not_after > now + opt_.max_submit_age_ms) return a.verdict = Verdict::kTooFarAhead, a;
  if (recs_.size() >= opt_.max_records) {
    a.verdict = Verdict::kRefusedRecords;
    a.retry_after_s = opt_.retry_after_s;
    return a;
  }
  // Tombstone it: a later submit (a delayed copy) finds the record and is a no-op; one naming another digest is a
  // conflict.
  Rec r;
  r.st = St::kCancelled;
  r.not_after = ref.not_after;
  r.digest = ref.digest;
  a.view = ViewOf(recs_.emplace(ref.id, std::move(r)).first->second);
  return a;
}

Operations::Answer Operations::Ack(const Ref& ref, std::int64_t now) {
  std::lock_guard<std::mutex> lk(mu_);
  Sweep(now);
  Answer a;
  if (ref.incarnation != incarnation_) {
    a.verdict = Verdict::kStale;
    a.view.status = Status::kUnknown;
    return a;
  }
  const auto it = recs_.find(ref.id);
  if (it == recs_.end()) return a.view = Absent(ref, now), a;
  Rec& r = it->second;
  if (r.digest != ref.digest) {
    a.verdict = Verdict::kConflict;
  } else if (r.st == St::kDone) {
    Release(r);
    r.st = St::kReleased;
    r.acked = true;
  } else if (r.st == St::kExpired) {
    r.acked = true;
  }
  a.view = ViewOf(r);
  return a;
}

bool Operations::Finish(const Id& id, std::string body, Usage usage, std::string finish, std::int64_t now) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto it = recs_.find(id);
  if (it == recs_.end() || it->second.st != St::kRunning) return false;
  Rec& r = it->second;
  r.st = St::kDone;
  r.done_at = now;
  r.body = std::move(body);
  r.usage = usage;
  r.finish = std::move(finish);
  return true;
}

bool Operations::CancelRequested(const Id& id) const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto it = recs_.find(id);
  return it != recs_.end() && it->second.st == St::kRunning && it->second.cancel;
}

Operations::Stats Operations::stats(std::int64_t now) const {
  std::lock_guard<std::mutex> lk(mu_);
  Stats s;
  s.records = recs_.size();
  s.reserved_bytes = reserved_;
  for (const auto& [id, r] : recs_) {
    if (r.st == St::kRunning || r.st == St::kDone) ++s.held;
    if (r.st == St::kDone || (r.st == St::kExpired && !r.acked)) {
      ++s.unacked_outcomes;
      s.oldest_unacked_ms = std::max(s.oldest_unacked_ms, now - r.done_at);
    }
  }
  return s;
}

}  // namespace halo::server
