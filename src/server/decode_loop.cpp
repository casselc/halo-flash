#include "server/decode_loop.hpp"

#include <algorithm>

namespace halo::server {

DecodeOutcome RunDecode(std::int32_t first, std::uint32_t budget, engine::MtpDepthChooser* chooser,
                        const DecodeIo& io, SpecDecodeStats* stats) {
  DecodeOutcome o;
  std::int32_t token = first;
  const auto gone = [&] { return io.gone && io.gone(); };
  bool done = false;
  while (!done && o.completion < budget) {
    if (gone()) {  // the session holds exactly what ran: retained as a request with max_tokens = completion
      o.cancelled = true;
      break;
    }
    const std::uint32_t left = budget - o.completion;
    const std::uint32_t k = chooser ? DraftCap(*chooser, left) : 0;
    const bool capped = chooser && k < chooser->Choose(SpecDecodeStats::kDepths);
    DecodeCycle c;
    if (!io.step(token, k, &c, &o.error)) {
      if (o.error.empty()) o.error = "decode step failed";
      return o;
    }
    if (c.tokens.empty() || c.drafted > k || c.accepted > c.drafted || c.tokens.size() != c.accepted + 1u ||
        c.tokens.size() > left) {
      o.error = "decode step returned an inconsistent cycle";
      return o;
    }
    if (chooser) chooser->Observe(c.accepted, c.drafted);
    if (stats) {
      ++stats->depth[std::min<std::size_t>(c.drafted, SpecDecodeStats::kDepths - 1)];
      if (c.drafted > 0) {
        ++stats->cycles;
        stats->drafted += c.drafted;
        stats->accepted += c.accepted;
        for (std::uint32_t i = 0; i < c.accepted && i < stats->accepted_at.size(); ++i) ++stats->accepted_at[i];
      }
      stats->capped += capped;
    }
    // What the engine ran: the step's token and the accepted drafts (the last emitted token is not run yet).
    io.ran(token);
    for (std::uint32_t i = 0; i < c.accepted; ++i) io.ran(c.tokens[i]);
    std::size_t unemitted = 0;  // the cycle's first token not emitted
    for (std::size_t i = 0; i < c.tokens.size(); ++i, unemitted = i) {
      if (i > 0 && gone()) {
        o.cancelled = done = true;
        break;
      }
      const auto next = static_cast<text::TokenId>(c.tokens[i]);
      if (io.is_stop(next)) {
        o.stopped = done = true;
        break;
      }
      ++o.completion;
      token = c.tokens[i];
      if (!io.emit(next)) {
        o.refused = done = true;
        unemitted = i + 1;
        break;
      }
    }
    // Ended inside the accepted run: the engine ran drafts that were not emitted (tokens[unemitted .. accepted)).
    if (done && stats && unemitted < c.accepted) stats->ran_past_end = 1;
  }
  o.length_limited = !o.cancelled && !o.refused && !o.stopped && o.completion == budget && budget > 0;
  return o;
}

}  // namespace halo::server
