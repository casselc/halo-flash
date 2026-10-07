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
  };
  /// Binds a session whose state is `prompt` minus its last token (the caller runs that token with sampling).
  /// `stable`: prompt positions later requests are expected to share (a checkpoint is parked there, rounded down
  /// to `align` in exact mode). `positions`: the prompt plus the generation budget (the memory check).
  [[nodiscard]] bool Prepare(std::span<const std::int32_t> prompt, std::size_t stable, std::uint32_t positions,
                             Prepared* out, std::string* error);
  /// The bound session ran `token` (a decode step).
  void Ran(std::int32_t token);
  /// The generation ended: fast mode retains the live session, exact mode drops it.
  void Finish();
  /// The generation failed: the live session's state is unknown; drop it.
  void Abandon();

  [[nodiscard]] std::size_t retained() const { return entries_.size(); }
  /// Tokens the live session has run (or the last finished one's, until the next Prepare).
  [[nodiscard]] std::size_t live_tokens() const { return last_live_tokens_; }
  [[nodiscard]] const SessionCacheOptions& options() const { return opt_; }

 private:
  struct Entry {
    Engine::SessionId id = 0;
    std::vector<std::int32_t> tokens;  // the sequence the state holds
    bool checkpoint = false;           // parked (forked from); else a finished live session (fast mode)
    bool canonical = false;            // byte-identical to one cold prefill of `tokens`
    std::uint64_t used = 0;
  };
  bool Drop(Engine::SessionId id, std::string* error);
  void Evict();  // down to max_retained and budget_bytes
  bool EvictOne();
  std::size_t Bytes(const Entry& e) const;

  Engine& e_;
  SessionCacheOptions opt_;
  std::vector<Entry> entries_;
  Entry live_;
  bool has_live_ = false;
  std::uint64_t clock_ = 0;
  std::size_t last_live_tokens_ = 0;
};

}  // namespace halo::server

#endif  // HALO_SERVER_SESSION_CACHE_HPP_
