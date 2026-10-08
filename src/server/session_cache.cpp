#include "server/session_cache.hpp"

#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>

namespace halo::server {

namespace {
// Spill files are named s<pid>-<key>.spill: the index of what they hold lives in this process only, so files of a
// process that no longer runs are unreachable and deleted at startup (with every crash-left temp file).
void CleanSpillDir(const std::string& dir) {
  engine::FlashNextGpu::SpillCleanTemps(dir);
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
    const std::string n = e.path().filename().string();
    if (n.size() < 8 || n[0] != 's' || n.substr(n.size() - 6) != ".spill") continue;
    const long pid = std::strtol(n.c_str() + 1, nullptr, 10);
    if (pid > 0 && pid != ::getpid() && ::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH)
      std::filesystem::remove(e.path(), ec);
  }
}
}  // namespace

SessionCache::SessionCache(Engine& engine, SessionCacheOptions options) : e_(engine), opt_(options) {
  if (opt_.align == 0) opt_.align = 1;
  if (!opt_.spill_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(opt_.spill_dir, ec);
    ::chmod(opt_.spill_dir.c_str(), 0700);
    CleanSpillDir(opt_.spill_dir);
  }
}

SessionCache::~SessionCache() {
  std::string error;
  if (e_.BoundSession() != 0) (void)e_.BindSession(0, &error);
  if (has_live_ && live_.id != 0) (void)e_.DropSession(live_.id, &error);
  for (const auto& en : entries_) {
    std::error_code ec;
    if (en.spilled) std::filesystem::remove(en.path, ec);
    else (void)e_.DropSession(en.id, &error);
  }
}

void SessionCache::EraseEntry(std::size_t i) {
  std::string error;
  std::error_code ec;
  if (entries_[i].spilled) {
    std::filesystem::remove(entries_[i].path, ec);
    ++counters_.files_deleted;
  } else {
    (void)Drop(entries_[i].id, &error);
  }
  entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(i));
}

std::string SessionCache::Label(const Entry& en) const {
  return en.session.empty() ? "s" + std::to_string(::getpid()) + "-" + std::to_string(en.key) : en.session;
}

bool SessionCache::SpillEntry(Entry& en, std::string* error) {
  if (opt_.spill_dir.empty()) return *error = "no spill directory configured", false;
  if (en.spilled) return true;
  // The key first: a provider without a key for this session refuses the spill (never a plaintext fallback);
  // with no provider, plaintext only when the operator chose it.
  Engine::SpillProtection prot;
  core::crypt::Kek kek;
  const std::string label = Label(en);
  if (opt_.spill_keys) {
    prot.kek = &kek;
    if (!opt_.spill_keys->Key(label, SpillKeys::Use::kSeal, &kek, error)) {  // no key, no spill: never plaintext
      ++counters_.spill_failed;
      ++counters_.no_key;
      return false;
    }
  } else if (!opt_.spill_plaintext_ok) {
    ++counters_.spill_failed;
    return *error = "no spill key configured (--spill-key) and plaintext spills not allowed (--spill-plaintext-ok)", false;
  }
  const std::string path = opt_.spill_dir + "/s" + std::to_string(::getpid()) + "-" + std::to_string(en.key) + ".spill";
  const auto prev = e_.BoundSession();
  if (!e_.BindSession(en.id, error)) return false;
  std::vector<std::uint8_t> blob(en.tokens.size() * 4);
  if (!blob.empty()) std::memcpy(blob.data(), en.tokens.data(), blob.size());
  Engine::SpillInfo info;
  const bool ok = e_.SpillSession(path, prot, blob, &info, error);
  std::string e2;
  (void)e_.BindSession(prev == en.id ? 0 : prev, &e2);
  if (!ok) {  // not done: the entry stays resident and unchanged
    ++counters_.spill_failed;
    return false;
  }
  if (opt_.spill_keys) opt_.spill_keys->Used(label, SpillKeys::Use::kSeal);
  // Done: the file holds the state; the device copy goes (the memory guard sees it at once: DropSession frees the
  // session's ranges, docs/SESSIONS.md section 7).
  (void)Drop(en.id, &e2);
  en.id = 0;
  en.spilled = true;
  en.path = path;
  en.file_bytes = info.file_bytes;
  en.key_id = info.key_id;
  ++counters_.spilled;
  return true;
}

SessionCache::Restored SessionCache::RestoreEntry(Entry& en, std::string* error) {
  if (!en.spilled) return Restored::kDone;
  Engine::SpillProtection prot;
  prot.accept_v1 = opt_.spill_accept_v1;
  core::crypt::Kek kek;
  const std::string label = Label(en);
  if (opt_.spill_keys && !en.key_id.empty()) {
    if (!opt_.spill_keys->KeyForFile(label, en.key_id, &kek, error)) {
      ++counters_.no_key;
      return Restored::kNoKey;  // nothing touched: the file waits for its key
    }
    prot.kek = &kek;
  }
  const auto prev = e_.BoundSession();
  Engine::SessionId id;
  if (!e_.NewSession(&id, error)) return ++counters_.restore_failed, Restored::kRefused;
  std::string e2;
  const auto give_up = [&] {
    (void)e_.BindSession(prev, &e2);
    (void)e_.DropSession(id, &e2);
    ++counters_.restore_failed;
    return Restored::kRefused;
  };
  if (!e_.BindSession(id, error)) return give_up();
  // Room for its pages: other resident entries are evicted (spilled or dropped) first, least recently used first.
  for (;;) {
    std::string why;
    if (e_.KvRoomFor(static_cast<std::uint32_t>(en.tokens.size()), 0, &why)) break;
    if (!EvictOne()) return *error = why, give_up();
  }
  std::vector<std::uint8_t> blob;
  if (!e_.RestoreSession(en.path, prot, &blob, nullptr, error)) return give_up();
  // The file must hold exactly the entry's tokens (it is the entry's own file; a mismatch means it was replaced).
  if (blob.size() != en.tokens.size() * 4 || (!blob.empty() && std::memcmp(blob.data(), en.tokens.data(), blob.size()) != 0)) {
    *error = "spill file holds other tokens than its entry";
    return give_up();
  }
  if (opt_.spill_keys && prot.kek) opt_.spill_keys->Used(label, SpillKeys::Use::kOpen);
  (void)e_.BindSession(prev, &e2);
  std::error_code ec;
  std::filesystem::remove(en.path, ec);
  en.id = id;
  en.spilled = false;
  en.path.clear();
  en.file_bytes = 0;
  en.key_id.clear();
  ++counters_.restored;
  return Restored::kDone;
}

void SessionCache::TrimSpilled() {
  for (;;) {
    std::size_t n = 0;
    std::uint64_t bytes = 0;
    std::ptrdiff_t oldest = -1;
    for (std::size_t i = 0; i < entries_.size(); ++i)
      if (entries_[i].spilled) {
        ++n;
        bytes += entries_[i].file_bytes;
        if (oldest < 0 || entries_[i].used < entries_[static_cast<std::size_t>(oldest)].used) oldest = static_cast<std::ptrdiff_t>(i);
      }
    if (oldest < 0 || (n <= opt_.max_spilled && bytes <= opt_.spill_budget_bytes)) return;
    EraseEntry(static_cast<std::size_t>(oldest));
  }
}

std::vector<SessionCache::EntryInfo> SessionCache::List() const {
  std::vector<EntryInfo> out;
  for (const auto& en : entries_)
    out.push_back({en.key, en.tokens.size(), en.checkpoint, en.canonical, en.spilled, en.spilled ? 0 : Bytes(en),
                   en.file_bytes, en.used, Label(en), en.key_id});
  return out;
}

bool SessionCache::Spill(std::uint64_t key, std::string* error, std::string_view session) {
  for (auto& en : entries_)
    if (en.key == key) {
      if (!session.empty() && !en.spilled) en.session = std::string(session);
      if (!SpillEntry(en, error)) return false;
      TrimSpilled();
      return true;
    }
  return *error = "no such entry", false;
}

bool SessionCache::Restore(std::uint64_t key, std::string* error, bool* kept) {
  if (kept) *kept = false;
  for (std::size_t i = 0; i < entries_.size(); ++i)
    if (entries_[i].key == key) {
      const Restored r = RestoreEntry(entries_[i], error);
      if (r == Restored::kDone) return true;
      if (r == Restored::kNoKey) {
        if (kept) *kept = true;
        return false;
      }
      // Gone for good: its next request re-prefills (a cold prefill).
      for (std::size_t j = 0; j < entries_.size(); ++j)
        if (entries_[j].key == key) EraseEntry(j);
      return false;
    }
  return *error = "no such entry", false;
}

std::size_t SessionCache::DropSpilledOf(std::string_view session) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < entries_.size();)
    if (entries_[i].spilled && entries_[i].session == session) {
      EraseEntry(i);
      ++n;
    } else {
      ++i;
    }
  return n;
}

std::size_t SessionCache::DropUnreadable() {
  std::size_t n = 0;
  for (std::size_t i = 0; i < entries_.size();) {
    const Entry& en = entries_[i];
    bool unreadable = false;
    if (en.spilled && opt_.spill_keys && !en.key_id.empty()) {
      core::crypt::Kek kek;
      std::string why;
      // A provider that has no key now (yard: not registered) may get one later: only a different current id
      // (rotated master, a later epoch) makes the file unreadable for good.
      unreadable = opt_.spill_keys->KeyForFile(Label(en), en.key_id, &kek, &why) && kek.id != en.key_id;
    }
    if (unreadable) {
      EraseEntry(i);
      ++n;
    } else {
      ++i;
    }
  }
  return n;
}

bool SessionCache::Drop(Engine::SessionId id, std::string* error) {
  if (id == 0) return true;
  if (e_.BoundSession() == id && !e_.BindSession(0, error)) return false;
  return e_.DropSession(id, error);
}

std::size_t SessionCache::Bytes(const Entry& en) const {
  if (en.spilled) return 0;
  const auto b = e_.SessionMemory(en.id);
  return b.recurrent + b.kv_exclusive;
}

std::size_t SessionCache::retained_bytes() const {
  std::size_t total = 0;
  for (const auto& en : entries_) total += Bytes(en);
  return total;
}

bool SessionCache::EvictOne() {
  // The least recently used resident entry.
  std::ptrdiff_t lru = -1;
  for (std::size_t i = 0; i < entries_.size(); ++i)
    if (!entries_[i].spilled && (lru < 0 || entries_[i].used < entries_[static_cast<std::size_t>(lru)].used))
      lru = static_cast<std::ptrdiff_t>(i);
  if (lru < 0) return false;
  std::string error;
  if (opt_.spill_on_evict && SpillEntry(entries_[static_cast<std::size_t>(lru)], &error)) {
    TrimSpilled();
    return true;
  }
  EraseEntry(static_cast<std::size_t>(lru));  // dropped (no disk tier, or the spill was not done)
  return true;
}

void SessionCache::Evict() {
  const auto resident = [&] {
    return static_cast<std::size_t>(std::count_if(entries_.begin(), entries_.end(), [](const Entry& en) { return !en.spilled; }));
  };
  while (resident() > opt_.max_retained && EvictOne()) {
  }
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

  // The retained state with the longest usable prefix of the prompt. A spilled one is restored first (SR4); one
  // whose restore fails is erased and the search runs again (its tokens are then re-prefilled: a cold prefill).
  Entry* best = nullptr;
  std::vector<std::uint64_t> waiting;  // spilled entries whose key is not available now (left as they are)
  for (bool again = true; again;) {
    again = false;
    best = nullptr;
    if (opt_.mode != SessionCacheOptions::Mode::kOff)
      for (auto& en : entries_) {
        if (en.tokens.size() > n || en.tokens.empty()) continue;
        if (std::find(waiting.begin(), waiting.end(), en.key) != waiting.end()) continue;
        if (exact && !en.canonical) continue;
        if (!std::equal(en.tokens.begin(), en.tokens.end(), prompt.begin())) continue;
        if (!best || en.tokens.size() > best->tokens.size()) best = &en;
      }
    if (best && best->spilled) {
      const auto key = best->key;
      std::string why;
      bool kept = false;
      if (!Restore(key, &why, &kept)) {
        again = true;
        if (kept) waiting.push_back(key);
      } else
        for (auto& en : entries_)
          if (en.key == key) best = &en;  // Restore may have erased other entries (eviction): find it again
    }
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
    park.key = next_key_++;
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
    if (live_.key == 0) live_.key = next_key_++;
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
