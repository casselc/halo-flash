#ifndef HALO_SERVER_SESSION_CACHE_HPP_
#define HALO_SERVER_SESSION_CACHE_HPP_

// Prompt-prefix reuse for the server over the engine's sessions (P6, docs/SESSIONS.md): instead of one engine
// state that a request either extends or resets, the cache retains several sessions and starts each request
// from the retained state with the longest prefix of its prompt, prefilling only the rest.
//
// Retained states are of two kinds:
//   - checkpoints: parked forks taken during a prefill at the request's "stable" position (where the
//     generation prompt starts: everything before it is history the next turn renders again). They make reuse
//     work when a client re-renders its history differently after that point: Hermes drops reasoning_content
//     from earlier assistant turns, so what the engine generated is never a prefix of the next prompt, but the
//     prompt up to the previous generation prompt is.
//   - live sessions (fast mode only): the state after a generation, prompt plus generated tokens, reused when
//     the next prompt extends it exactly (opencode and pi echo reasoning, so their next prompt does).
// Several conversations keep their own states, so a request with an unrelated prompt (opencode's title request)
// no longer resets the conversation's state; LRU eviction under a count and a byte budget, and whenever the
// device cannot hold a prompt (KvRoomFor).
//
// Exactness. The engine's state after a token sequence depends on the sequence and on where its prefill was
// split: decode steps, and prefill pieces that start at arbitrary positions, change the arithmetic (fp16/int8
// GEMMs, the DeltaNet chunk scan; FlashNextGpu.ChunkedPrefillMatchesStepping). Exact mode (the default) reuses
// only checkpoints at multiples of `align` positions taken from such states, never a live session, so every
// request's state, logits and greedy tokens are byte-identical to one cold prefill of its prompt
// (SessionCache.ReuseIsByteIdenticalToColdPrefill). Fast mode also reuses live sessions and unaligned
// checkpoints (main's behaviour before the cache, for the extension case): fewer tokens prefilled, results equal
// to a cold run only up to the prefill-vs-decode rounding.

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "engine/flash_next_gpu.hpp"

namespace halo::server {

struct SessionCacheOptions {
  enum class Mode : std::uint8_t { kOff, kExact, kFast };
  Mode mode = Mode::kExact;
  /// Exact mode: checkpoint positions are multiples of this. On the real model a prefill split at any multiple
  /// of 16 positions (the DeltaNet scan's piece) left the state byte-identical to an unsplit prefill (24 split
  /// points from 16 to 9008, across the 8192-token long-chunk boundary), and every split at a multiple of 8 or 4
  /// only moved the logits by 1e-2..6e-2 relative (SessionCache.RealModelSplitProbe, stream H p1/p2). 64 keeps a
  /// margin (SessionCache.SmallModelAlignedSplitsAreByteIdentical gates it); 16 would save ~24 tokens a turn.
  std::uint32_t align = 64;
  std::size_t max_retained = 8;                 ///< checkpoints and live sessions kept between requests
  std::size_t budget_bytes = 8ull << 30;        ///< device bytes they may hold (recurrent + exclusive KV pages)
  /// SR4 disk tier (docs/SESSIONS.md section 10). Directory for spill files (created 0700; files 0600). Empty: no
  /// disk tier (Spill refuses; eviction drops). Set: Spill(key) works, and Prepare restores a spilled entry that holds
  /// the longest usable prefix instead of re-prefilling it.
  std::string spill_dir;
  bool spill_on_evict = false;              ///< eviction (count, budget, KvRoomFor) spills instead of dropping
  std::size_t max_spilled = 64;             ///< spilled entries kept (oldest file deleted beyond)
  std::size_t spill_budget_bytes = 64ull << 30;  ///< disk bytes they may hold
};

class SessionCache {
 public:
  using Engine = engine::FlashNextGpu;
  /// `engine` must outlive the cache; the cache owns every session but session 0, which it keeps bound and
  /// empty between requests.
  SessionCache(Engine& engine, SessionCacheOptions options);
  ~SessionCache();
  SessionCache(const SessionCache&) = delete;
  SessionCache& operator=(const SessionCache&) = delete;

  struct Prepared {
    std::size_t reused = 0;     ///< prompt tokens whose state came from a retained session
    std::size_t prefilled = 0;  ///< prompt tokens prefilled now
    bool checkpointed = false;  ///< a checkpoint was parked during the prefill
    bool cancelled = false;     ///< `cancelled` stopped it (Prepare returned false; not an error)
  };
  /// Binds a session whose state is `prompt` minus its last token (the caller runs that token with sampling).
  /// `stable`: prompt positions later requests are expected to share (a checkpoint is parked there, rounded down
  /// to `align` in exact mode). `positions`: the prompt plus the generation budget (the memory check).
  /// `cancelled` (optional) is polled before the prefill and between its chunks; when it stops the prefill the
  /// live session is dropped (its state is partly advanced) and only states equal to a cold prefill of their
  /// tokens stay retained (a checkpoint parked before the stop is one).
  /// `before_prefill` (optional) runs once the live session is bound with `reused` positions, before any prefill
  /// (the generator sets the draft block's history window there); false fails the Prepare with its error.
  [[nodiscard]] bool Prepare(std::span<const std::int32_t> prompt, std::size_t stable, std::uint32_t positions,
                             Prepared* out, std::string* error, const std::function<bool()>& cancelled = {},
                             const std::function<bool(std::size_t reused, std::string* error)>& before_prefill = {});
  /// The bound session ran `token` (a decode step).
  void Ran(std::int32_t token);
  /// The generation ended: fast mode retains the live session, exact mode drops it.
  void Finish();
  /// The generation failed: the live session's state is unknown; drop it.
  void Abandon();

  [[nodiscard]] std::size_t retained() const { return entries_.size(); }
  /// Device bytes the retained states hold (recurrent buffers and KV pages no other session shares).
  [[nodiscard]] std::size_t retained_bytes() const;
  /// Tokens the live session has run (or the last finished one's, until the next Prepare).
  [[nodiscard]] std::size_t live_tokens() const { return last_live_tokens_; }
  [[nodiscard]] const SessionCacheOptions& options() const { return opt_; }

  // ---- SR4: suspend / resume of retained states by key (an internal API for an admin route and yard-ctl) ----
  struct EntryInfo {
    std::uint64_t key = 0;       ///< stable while the entry exists (resident or spilled)
    std::size_t tokens = 0;
    bool checkpoint = false, canonical = false, spilled = false;
    std::size_t device_bytes = 0;  ///< resident: recurrent + exclusive KV pages
    std::uint64_t file_bytes = 0;  ///< spilled: the file's size
    std::uint64_t used = 0;        ///< LRU clock
  };
  [[nodiscard]] std::vector<EntryInfo> List() const;
  /// Writes a resident entry's state to the spill directory and frees its device memory. Outcomes: true = done
  /// (file complete and durable, device state dropped); false with the entry still resident = not done (nothing
  /// changed but a removed temp file); the entry is never half-spilled.
  [[nodiscard]] bool Spill(std::uint64_t key, std::string* error);
  /// Brings a spilled entry back onto the device (a resume ahead of its request). On false the entry is gone (its
  /// file refused or damaged, or no memory): its next request re-prefills, which is exactly a cold prefill.
  [[nodiscard]] bool Restore(std::uint64_t key, std::string* error);
  /// Counters since construction (tests, metrics).
  struct SpillCounters {
    std::size_t spilled = 0, spill_failed = 0, restored = 0, restore_failed = 0, files_deleted = 0;
  };
  [[nodiscard]] const SpillCounters& spill_counters() const { return counters_; }

 private:
  struct Entry {
    Engine::SessionId id = 0;
    std::vector<std::int32_t> tokens;  // the sequence the state holds
    bool checkpoint = false;           // parked (forked from); else a finished live session (fast mode)
    bool canonical = false;            // byte-identical to one cold prefill of `tokens`
    std::uint64_t used = 0;
    std::uint64_t key = 0;             // SR4: stable handle (List / Spill / Restore)
    bool spilled = false;              // SR4: state in `path`, no engine session (id meaningless)
    std::string path;
    std::uint64_t file_bytes = 0;
  };
  bool Drop(Engine::SessionId id, std::string* error);
  void Evict();  // down to max_retained and budget_bytes
  bool EvictOne();  // the least recently used resident entry: spilled (spill_on_evict) or dropped
  bool SpillEntry(Entry& en, std::string* error);
  bool RestoreEntry(Entry& en, std::string* error);  // on false the caller erases the entry
  void TrimSpilled();  // down to max_spilled and spill_budget_bytes (oldest files deleted)
  void EraseEntry(std::size_t index);  // drops its session or deletes its file
  std::size_t Bytes(const Entry& e) const;

  Engine& e_;
  SessionCacheOptions opt_;
  std::vector<Entry> entries_;
  Entry live_;
  bool has_live_ = false;
  std::uint64_t clock_ = 0;
  std::size_t last_live_tokens_ = 0;
  std::uint64_t next_key_ = 1;
  SpillCounters counters_;
};

}  // namespace halo::server

#endif  // HALO_SERVER_SESSION_CACHE_HPP_
