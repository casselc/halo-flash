#include "server/decode_loop.hpp"

#include <algorithm>

namespace halo::server {

std::size_t CompleteUtf8Prefix(std::string_view bytes) {
  // Look back at most 3 bytes for the lead byte of a sequence that the
  // buffer cuts short.
  const std::size_t n = bytes.size();
  for (std::size_t back = 1; back <= std::min<std::size_t>(3, n); ++back) {
    const auto c = static_cast<unsigned char>(bytes[n - back]);
    if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking
    const std::size_t need = c >= 0xF0 && c <= 0xF4 ? 4 : c >= 0xE0 && c < 0xF0 ? 3 : c >= 0xC2 && c < 0xE0 ? 2 : 1;
    return need > back ? n - back : n;  // ASCII or invalid lead bytes pass through
  }
  return n;
}


void DecodeRun::Cycle(const DecodeIo& io, bool draft) {
  if (done()) return;
  DecodeOutcome& o = o_;
  if (io.gone && io.gone()) {  // the session holds exactly what ran: retained as a request with max_tokens = completion
    o.cancelled = done_ = true;
    return;
  }
  const std::uint32_t left = budget_ - o.completion;
  engine::MtpDepthChooser* chooser = draft ? chooser_ : nullptr;
  const std::uint32_t k = chooser ? DraftCap(*chooser, left) : 0;
  const bool capped = chooser && k < chooser->Choose(SpecDecodeStats::kDepths);
  DecodeCycle c;
  if (!io.step(token_, k, &c, &o.error)) {
    if (o.error.empty()) o.error = "decode step failed";
    done_ = true;
    return;
  }
  if (c.tokens.empty() || c.drafted > k || c.accepted > c.drafted || c.tokens.size() != c.accepted + 1u ||
      c.tokens.size() > left) {
    o.error = "decode step returned an inconsistent cycle";
    done_ = true;
    return;
  }
  if (chooser) chooser->Observe(c.accepted, c.drafted);
  if (stats_) {
    ++stats_->depth[std::min<std::size_t>(c.drafted, SpecDecodeStats::kDepths - 1)];
    if (c.drafted > 0) {
      ++stats_->cycles;
      stats_->drafted += c.drafted;
      stats_->accepted += c.accepted;
      for (std::uint32_t i = 0; i < c.accepted && i < stats_->accepted_at.size(); ++i) ++stats_->accepted_at[i];
    }
    stats_->capped += capped;
  }
  // What the engine ran: the step's token and the accepted drafts (the last emitted token is not run yet).
  io.ran(token_);
  for (std::uint32_t i = 0; i < c.accepted; ++i) io.ran(c.tokens[i]);
  std::size_t unemitted = 0;  // the cycle's first token not emitted
  for (std::size_t i = 0; i < c.tokens.size(); ++i, unemitted = i) {
    if (i > 0 && io.gone && io.gone()) {
      o.cancelled = done_ = true;
      break;
    }
    const auto next = static_cast<text::TokenId>(c.tokens[i]);
    if (io.is_stop(next)) {
      o.stopped = done_ = true;
      break;
    }
    ++o.completion;
    token_ = c.tokens[i];
    if (!io.emit(next)) {
      o.refused = done_ = true;
      unemitted = i + 1;
      break;
    }
  }
  // Ended inside the accepted run: the engine ran drafts that were not emitted (tokens[unemitted .. accepted)).
  if (done_ && stats_ && unemitted < c.accepted) stats_->ran_past_end = 1;
}

DecodeOutcome DecodeRun::Finish() {
  DecodeOutcome o = o_;
  if (o.error.empty())
    o.length_limited = !o.cancelled && !o.refused && !o.stopped && o.completion == budget_ && budget_ > 0;
  return o;
}

DecodeOutcome RunDecode(std::int32_t first, std::uint32_t budget, engine::MtpDepthChooser* chooser,
                        const DecodeIo& io, SpecDecodeStats* stats) {
  DecodeRun run(first, budget, chooser, stats);
  while (!run.done()) run.Cycle(io);
  return run.Finish();
}

}  // namespace halo::server
