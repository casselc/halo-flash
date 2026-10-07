#ifndef HALO_SERVER_DECODE_LOOP_HPP_
#define HALO_SERVER_DECODE_LOOP_HPP_

// The decode loop of one request, with or without MTP speculative decoding (docs/MTP.md, server items 3-4),
// apart from the engine so its bookkeeping is testable on the host (tests/server/decode_loop_test.cpp).
//
// A step runs `token` (and, when it drafts, the accepted drafts after it) and returns what follows: one token for
// an ordinary step, 1 + accepted for a speculative cycle. The loop
//   - asks the chooser for at most (tokens left - 1) drafts, so no cycle emits past the budget (max_tokens);
//   - records in `ran` exactly what the engine ran: `token` and the accepted drafts, also past a stop or a
//     cancel inside the cycle (the session holds them; the prefix cache decides reuse from that record);
//   - emits the cycle's tokens in order until a stop token (never emitted), the budget, the consumer's refusal or
//     the requester leaving (checked before every step and between a cycle's tokens).

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "engine/mtp_depth.hpp"
#include "server/observer.hpp"
#include "text/tokenizer.hpp"

namespace halo::server {

struct DecodeCycle {
  std::vector<std::int32_t> tokens;  ///< emitted after the step's token, in order (1 + accepted)
  std::uint32_t drafted = 0, accepted = 0;
};

struct DecodeIo {
  /// Runs `token` with up to `drafts` drafts (0: an ordinary step); false on an engine error.
  std::function<bool(std::int32_t token, std::uint32_t drafts, DecodeCycle* cycle, std::string* error)> step;
  std::function<void(std::int32_t)> ran;              ///< the engine ran this token (history)
  std::function<bool()> gone;                         ///< optional: the requester left
  std::function<bool(text::TokenId)> is_stop;         ///< generation ends here; the token is not emitted
  std::function<bool(text::TokenId)> emit;            ///< a generated token; false: stop (cancel or stop string)
};

struct DecodeOutcome {
  std::uint32_t completion = 0;  ///< tokens emitted
  bool stopped = false;          ///< a stop token ended it
  bool refused = false;          ///< emit returned false
  bool cancelled = false;        ///< the requester left (DecodeIo::gone)
  bool length_limited = false;   ///< ended at the budget
  std::string error;
};

/// Drafts for the next cycle: the chooser's depth, capped so the cycle emits at most `left` tokens.
[[nodiscard]] inline std::uint32_t DraftCap(const engine::MtpDepthChooser& chooser, std::uint32_t left) {
  return left == 0 ? 0 : chooser.Choose(left - 1);
}

/// Decodes from `first` (the prompt's last token, not yet run) for at most `budget` tokens. `chooser` null: no
/// drafting (every step ordinary). `stats` (optional) accumulates drafted / accepted / depth counts.
DecodeOutcome RunDecode(std::int32_t first, std::uint32_t budget, engine::MtpDepthChooser* chooser,
                        const DecodeIo& io, SpecDecodeStats* stats);

}  // namespace halo::server

#endif  // HALO_SERVER_DECODE_LOOP_HPP_
