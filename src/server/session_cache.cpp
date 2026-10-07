#include "server/session_cache.hpp"

#include <algorithm>

namespace halo::server {

SessionCache::SessionCache(Engine& engine, SessionCacheOptions options) : e_(engine), opt_(options) {
  if (opt_.align == 0) opt_.align = 1;
}

SessionCache::~SessionCache() {
  std::string error;
  if (e_.BoundSession() != 0) (void)e_.BindSession(0, &error);
  if (has_live_ && live_.id != 0) (void)e_.DropSession(live_.id, &error);
  for (const auto& en : entries_) (void)e_.DropSession(en.id, &error);
}

bool SessionCache::Drop(Engine::SessionId id, std::string* error) {
  if (id == 0) return true;
  if (e_.BoundSession() == id && !e_.BindSession(0, error)) return false;
  return e_.DropSession(id, error);
}

std::size_t SessionCache::Bytes(const Entry& en) const {
  const auto b = e_.SessionMemory(en.id);
  return b.recurrent + b.kv_exclusive;
}

std::size_t SessionCache::retained_bytes() const {
  std::size_t total = 0;
  for (const auto& en : entries_) total += Bytes(en);
  return total;
}

bool SessionCache::EvictOne() {
  if (entries_.empty()) return false;
  auto lru = std::min_element(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
  std::string error;
  (void)Drop(lru->id, &error);
  entries_.erase(lru);
  return true;
}

void SessionCache::Evict() {
  while (entries_.size() > opt_.max_retained) EvictOne();
  for (;;) {
    std::size_t total = 0;
    for (const auto& en : entries_) total += Bytes(en);
    if (total <= opt_.budget_bytes || !EvictOne()) break;
  }
}

bool SessionCache::Prepare(std::span<const std::int32_t> prompt, std::size_t stable, std::uint32_t positions,
                           Prepared* out, std::string* error, const std::function<bool()>& cancelled,
                           const std::function<bool(std::size_t, std::string*)>& before_prefill) {
  *out = {};
  if (has_live_) Abandon();  // a Prepare without Finish: the previous state is not trusted
  const auto stop = [&] { return out->cancelled = out->cancelled || (cancelled && cancelled()); };
  if (stop()) return *error = "cancelled", false;
  const std::size_t n = prompt.empty() ? 0 : prompt.size() - 1;
  const bool exact = opt_.mode == SessionCacheOptions::Mode::kExact;

  // The retained state with the longest usable prefix of the prompt.
  Entry* best = nullptr;
  if (opt_.mode != SessionCacheOptions::Mode::kOff)
    for (auto& en : entries_) {
      if (en.tokens.size() > n || en.tokens.empty()) continue;
      if (exact && !en.canonical) continue;
      if (!std::equal(en.tokens.begin(), en.tokens.end(), prompt.begin())) continue;
      if (!best || en.tokens.size() > best->tokens.size()) best = &en;
    }

  // The live session: a fork of a checkpoint, a finished live session taken over, or a fresh one.
  live_ = {};
  const bool extends_live = best && !best->checkpoint;
  if (best && best->checkpoint) {
    if (!e_.ForkSession(best->id, &live_.id, error)) return false;
    best->used = ++clock_;
    live_.tokens = best->tokens;
    live_.canonical = best->canonical;
  } else if (best) {
    live_ = std::move(*best);
    entries_.erase(entries_.begin() + (best - entries_.data()));
  } else {
    if (!e_.NewSession(&live_.id, error)) return false;
    live_.canonical = true;  // an empty sequence
  }
  has_live_ = true;
  if (!e_.BindSession(live_.id, error)) return Abandon(), false;
  out->reused = live_.tokens.size();

  // Room for the prompt and its generation, evicting retained states (least recently used first) until the
  // device can hold it.
  for (;;) {
    std::string why;
    if (e_.KvRoomFor(positions, static_cast<std::uint32_t>(n - out->reused), &why)) break;
    if (!EvictOne()) {
      Abandon();
      return *error = why, false;
    }
  }

  if (before_prefill && !before_prefill(out->reused, error)) return Abandon(), false;

  // Checkpoint position: where the generation prompt starts (exact: rounded down to the alignment).
  std::size_t ck = std::min(stable, n);
  if (exact) ck = ck / opt_.align * opt_.align;
  // A checkpoint costs a second prefill call (~140 ms fixed per call on the real model, stream H c1) and a fork.
  // Fast mode skips it when this request extends a finished session: that client echoes its history, so the
  // next request will most likely extend this one again (opencode, pi); a client that re-renders (Hermes) never
  // extends, so it always gets one.
  const bool want_ck = opt_.mode != SessionCacheOptions::Mode::kOff && ck > out->reused && !extends_live &&
                       !(exact && !live_.canonical) &&
                       std::none_of(entries_.begin(), entries_.end(), [&](const Entry& en) {
                         return en.tokens.size() == ck && std::equal(en.tokens.begin(), en.tokens.end(), prompt.begin());
                       });
  std::size_t at = out->reused;
  const auto run = [&](std::size_t to) {
    if (to <= at) return true;
    if (!e_.Prefill(prompt.subspan(at, to - at), {}, error, stop)) return false;
    live_.tokens.insert(live_.tokens.end(), prompt.begin() + static_cast<std::ptrdiff_t>(at),
                        prompt.begin() + static_cast<std::ptrdiff_t>(to));
    out->prefilled += to - at;
    at = to;
    return true;
  };
  if (want_ck) {
    if (!run(ck)) return Abandon(), false;
    Entry park;
    if (!e_.ForkSession(live_.id, &park.id, error)) return Abandon(), false;
    park.tokens = live_.tokens;
    park.checkpoint = true;
    // Byte-identical to a cold prefill when every split of its prefill sat on the alignment.
    park.canonical = live_.canonical && out->reused % opt_.align == 0 && ck % opt_.align == 0;
    park.used = ++clock_;
    entries_.push_back(std::move(park));
    out->checkpointed = true;
    if (!e_.BindSession(live_.id, error)) return Abandon(), false;
  }
  if (!run(n)) return Abandon(), false;
  // Its splits sat on the alignment (the end of a prefill need not).
  live_.canonical = live_.canonical && out->reused % opt_.align == 0 && (!want_ck || ck % opt_.align == 0);
  last_live_tokens_ = live_.tokens.size();
  return true;
}

void SessionCache::Ran(std::int32_t token) {
  if (!has_live_) return;
  live_.tokens.push_back(token);
  live_.canonical = false;  // a decode step's arithmetic differs from a prefill's
  last_live_tokens_ = live_.tokens.size();
}

void SessionCache::Finish() {
  if (!has_live_) return;
  has_live_ = false;
  std::string error;
  if (opt_.mode == SessionCacheOptions::Mode::kFast && !live_.tokens.empty()) {
    live_.checkpoint = false;
    live_.used = ++clock_;
    (void)e_.BindSession(0, &error);
    entries_.push_back(std::move(live_));
  } else {
    (void)Drop(live_.id, &error);
  }
  live_ = {};
  Evict();
}

void SessionCache::Abandon() {
  if (!has_live_) return;
  has_live_ = false;
  std::string error;
  (void)Drop(live_.id, &error);
  live_ = {};
  last_live_tokens_ = 0;
}

}  // namespace halo::server
