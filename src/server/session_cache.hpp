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

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/flash_next_gpu.hpp"
#include "server/spill_keys.hpp"

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
  /// Spill encryption (format v2). `spill_keys` (not owned; must outlive the cache) seals every file; when it has
  /// no key for an entry the spill is refused, never written in plaintext. Null: spills are refused unless
  /// `spill_plaintext_ok` (then plaintext v1 files). `spill_accept_v1`: restore the plaintext v1 files this process
  /// wrote (an entry recorded as v2 never accepts a v1 file).
  SpillKeys* spill_keys = nullptr;
  bool spill_plaintext_ok = false;
  bool spill_accept_v1 = false;
  /// This server incarnation's id: entry ids are "<incarnation>.<key>" (yard's seal claim names one), so an id
  /// never names another conversation after a restart. Empty: 16 random hex digits (halo_serve passes yard's).
  std::string incarnation;
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
  void Ran(std::int32_t token) { Ran(0, token); }
  /// The generation ended: fast mode retains the live session, exact mode drops it.
  void Finish() { Finish(0); }
  /// The generation failed: the live session's state is unknown; drop it.
  void Abandon() { Abandon(0); }

  // ---- G3: several live sessions at once (docs/SCHEDULER.md M3), one per lease (a scheduled request) ----
  // Prepare / Ran / Finish / Abandon above are lease 0. A scheduler begins a lease (bind: the longest usable retained
  // prefix, forked; nothing prefilled), prefills it in slices (one engine call each; the checkpoint at `stable` is
  // parked when a slice reaches it), steps it, and finishes, abandons or parks it. Leases alternate freely: each owns
  // its engine session. Exact mode stays exact: every split sits on the alignment (slices of a multiple of `align`
  // tokens from an aligned start), so the parked checkpoints are byte-identical to a cold prefill.
  using LeaseId = std::uint64_t;
  /// Binds lease `lease` (the engine's bound session afterwards) for `prompt`: reuse, memory check (evicting
  /// unpinned entries until KvRoomFor passes), `before_prefill`. Nothing is prefilled yet (out->prefilled 0).
  [[nodiscard]] bool Begin(LeaseId lease, std::span<const std::int32_t> prompt, std::size_t stable, std::uint32_t positions,
                           Prepared* out, std::string* error,
                           const std::function<bool(std::size_t reused, std::string* error)>& before_prefill = {});
  /// One slice: binds the lease's session and prefills at most `max_tokens` prompt tokens in one engine call
  /// (`chunk_tokens` per chunk, 0: the engine's default), stopping early at the checkpoint position (which is then
  /// parked). `*ran` gets the tokens prefilled, `*done` whether the prompt (minus its last token) is all in.
  [[nodiscard]] bool PrefillSlice(LeaseId lease, std::uint32_t max_tokens, std::uint32_t chunk_tokens, std::size_t* ran,
                                  bool* done, std::string* error);
  void Ran(LeaseId lease, std::int32_t token);
  void Finish(LeaseId lease);
  void Abandon(LeaseId lease);
  /// The lease's client left: its live session goes, but a prefill's progress (slice-aligned, exact: every split
  /// on the alignment) is kept as a checkpoint first, so a retry reuses it. Returns the parked entry's key (0: none).
  std::uint64_t Park(LeaseId lease);
  [[nodiscard]] bool HasLease(LeaseId lease) const { return lives_.count(lease) > 0; }
  [[nodiscard]] Engine::SessionId LeaseSession(LeaseId lease) const;
  [[nodiscard]] std::size_t LeasePosition(LeaseId lease) const;  ///< tokens its live session holds
  /// The retained entry a Begin bound from (its key; 0: a fresh session or a taken-over live one).
  [[nodiscard]] std::uint64_t LeaseBoundKey(LeaseId lease) const;
  /// Entries a scheduler planned to reuse: never evicted by the cache's own memory loop while pinned.
  void Pin(std::uint64_t key) { ++pins_[key]; }
  void SetPins(const std::set<std::uint64_t>& keys) {
    pins_.clear();
    for (auto k : keys) pins_[k] = 1;
  }
  void Unpin(std::uint64_t key);
  /// Evicts one entry (a scheduler's decision): spilled (spill_on_evict) or dropped. False: no such resident entry.
  bool EvictKey(std::uint64_t key);
  /// What the scheduler's ledger counts: resident entries, their tokens and device bytes (recurrent + every KV page
  /// they map, shared ones included: an evicted parent's pages stay with its forks).
  struct Resident {
    std::uint64_t key = 0;
    std::shared_ptr<const std::vector<std::int32_t>> tokens;
    std::size_t bytes = 0;
    bool usable = true;  ///< reusable in this mode (exact: canonical)
  };
  [[nodiscard]] std::vector<Resident> Residents() const;
  /// Bumped whenever the retained set changes (a new checkpoint, an eviction, a finish): resync only then.
  [[nodiscard]] std::uint64_t generation() const { return generation_; }

  [[nodiscard]] std::size_t retained() const { return entries_.size(); }
  /// The longest prefix of `prompt` a retained state (resident or spilled) holds whole: a checkpoint whose tokens
  /// are a prefix of `prompt`, or a finished session (fast mode) that extends `prompt`. After Finish, minus the
  /// request's reused tokens, it is what the request added to the cache (Anthropic's cache_creation_input_tokens).
  [[nodiscard]] std::size_t RetainedPrefix(std::span<const std::int32_t> prompt) const;
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
    std::string session;           ///< the spill key's session label (a yard session, or "s<pid>-<key>")
    std::string key_id;            ///< spilled: the id of the key that seals the file ("" plaintext)
    bool yard_bound = false;       ///< sealed under a yard key once: never under a master key again
    std::string entry_id;          ///< "<incarnation>.<key>": what a yard seal registration names
  };
  [[nodiscard]] std::vector<EntryInfo> List() const;
  /// Writes a resident entry's state to the spill directory and frees its device memory. Outcomes: true = done
  /// (file complete and durable, device state dropped); false with the entry still resident = not done (nothing
  /// changed but a removed temp file); the entry is never half-spilled.
  /// The key: the provider's seal lease for this entry (yard: a seal registration naming this entry, which also
  /// binds the entry to that yard session for good).
  [[nodiscard]] bool Spill(std::uint64_t key, std::string* error);
  /// Brings a spilled entry back onto the device (a resume ahead of its request). On false the entry is gone (its
  /// file refused or damaged, or no memory): its next request re-prefills, which is exactly a cold prefill; except
  /// when the key provider has no key for it yet (`*kept` true): then the entry and its file stay, untouched.
  [[nodiscard]] bool Restore(std::uint64_t key, std::string* error, bool* kept = nullptr);
  /// Erases spilled entries whose file the provider can never open again (an older master epoch after
  /// FenceEpoch, a revoked yard epoch). Returns how many. Their files are deleted.
  std::size_t DropUnreadable();
  /// A yard revocation of `session` through file epoch `through`: erases that session's entries (spilled or
  /// resident) whose key epoch is <= `through`, and deletes their files; newer ones stay. Returns how many.
  std::size_t DropRevoked(std::string_view session, std::uint64_t through);
  /// Counters since construction (tests, metrics).
  struct SpillCounters {
    std::size_t spilled = 0, spill_failed = 0, restored = 0, restore_failed = 0, files_deleted = 0, no_key = 0;
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
    std::string session;               // spill key label (see EntryInfo::session); "" = the default
    std::string key_id;                // spilled: the file's key id ("" plaintext v1)
    bool yard_bound = false;           // sealed under a yard key once (sticky)
    std::uint64_t key_epoch = 0;       // the file epoch of its last yard seal (revocation fences by it)
    std::uint32_t version = 0;         // spilled: the format written (1 / 2): a v2 entry never takes a v1 file
    std::array<std::uint8_t, 16> checkpoint_id{};  // spilled (v2): the file's random checkpoint id
  };
  struct Live;
  enum class Restored : std::uint8_t { kDone, kRefused, kNoKey };
  std::string Label(const Entry& en) const;
  std::string EntryId(const Entry& en) const;
  bool Drop(Engine::SessionId id, std::string* error);
  bool RunTo(Live& lv, std::size_t to, std::uint32_t chunk, const std::function<bool()>& cancelled, Prepared* out,
             std::string* error);  // prefill the live session up to prompt position `to` (one engine call)
  bool ParkCheckpoint(Live& lv, std::string* error);  // fork the live session at its checkpoint position
  void Evict();  // down to max_retained and budget_bytes
  bool EvictOne();  // the least recently used resident entry: spilled (spill_on_evict) or dropped
  bool SpillEntry(Entry& en, std::string* error);
  Restored RestoreEntry(Entry& en, std::string* error);  // kRefused: the caller erases the entry; kNoKey: keeps it
  void TrimSpilled();  // down to max_spilled and spill_budget_bytes (oldest files deleted)
  void EraseEntry(std::size_t index);  // drops its session or deletes its file
  std::size_t Bytes(const Entry& e) const;

  // A lease's live session and its prefill plan.
  struct Live {
    Entry e;
    std::vector<std::int32_t> prompt;
    std::size_t n = 0, at = 0, ck = 0, reused = 0;
    bool want_ck = false, extends = false, splits_aligned = true;
    std::uint64_t bound_key = 0;
  };

  Engine& e_;
  SessionCacheOptions opt_;
  std::vector<Entry> entries_;
  std::map<LeaseId, Live> lives_;
  std::map<std::uint64_t, int> pins_;
  std::uint64_t generation_ = 0;
  std::uint64_t clock_ = 0;
  std::size_t last_live_tokens_ = 0;
  std::uint64_t next_key_ = 1;
  SpillCounters counters_;
};

}  // namespace halo::server

#endif  // HALO_SERVER_SESSION_CACHE_HPP_
