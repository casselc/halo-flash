#include "server/flash_next_generator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/sampling.hpp"
#include "server/decode_loop.hpp"

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

FlashNextGenerator::FlashNextGenerator(engine::FlashNextGpu& engine, const text::Tokenizer& tokenizer,
                                       std::uint32_t max_context, SessionCacheOptions cache, SpecOptions spec)
    : engine_(engine), tok_(&tokenizer), max_context_(max_context), cache_(engine, cache), spec_(std::move(spec)),
      chooser_(engine::MtpMeasuredCosts(), spec_.depth) {
  for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
    const auto ids = tok_->Encode(s, true);
    if (ids.size() == 1) stops_.push_back(ids[0]);
  }
  if (tok_->eos() != text::kNoToken) stops_.push_back(tok_->eos());
  if (const auto ids = tok_->Encode("<|im_start|>", true); ids.size() == 1) im_start_ = ids[0];
}

FlashNextGenerator::FlashNextGenerator(engine::FlashNextGpu& engine, std::vector<text::TokenId> stops,
                                       std::uint32_t max_context, SessionCacheOptions cache, SpecOptions spec)
    : engine_(engine), tok_(nullptr), max_context_(max_context), stops_(std::move(stops)), cache_(engine, cache),
      spec_(std::move(spec)), chooser_(engine::MtpMeasuredCosts(), spec_.depth) {}

bool FlashNextGenerator::IsStop(text::TokenId t) const {
  return std::find(stops_.begin(), stops_.end(), t) != stops_.end();
}

TextGenerator::Result FlashNextGenerator::Generate(const Params& params,
                                                   const std::function<bool(std::string_view)>& on_text) {
  Result r;
  if (!params.variant.empty()) return r.error = "Flash-Next variants are not supported yet (P11 stage 2)", r;
  if (!tok_) return r.error = "no tokenizer", r;
  const auto encoded = tok_->Encode(params.prompt, true);
  const std::vector<std::int32_t> ids(encoded.begin(), encoded.end());  // the engine's token type
  // The stable prefix: everything before the last <|im_start|> (the generation prompt), which the next turn of
  // the conversation renders again however it renders this turn's output.
  std::size_t stable = 0;
  for (std::size_t i = ids.size(); i-- > 0;)
    if (static_cast<text::TokenId>(ids[i]) == im_start_) {
      stable = i;
      break;
    }
  std::string pending;
  bool cancelled = false;
  r = GenerateIds(ids, stable, params, [&](text::TokenId next) {
    pending += tok_->TokenBytes(next);
    const std::size_t n = CompleteUtf8Prefix(pending);
    if (n > 0) {
      cancelled = !on_text(std::string_view(pending).substr(0, n));
      pending.erase(0, n);
    }
    return !cancelled;
  });
  if (r.error.empty() && !cancelled && !r.cancelled && !pending.empty()) (void)on_text(pending);
  return r;
}

TextGenerator::Result FlashNextGenerator::GenerateIds(std::span<const std::int32_t> ids, std::size_t stable,
                                                      const Params& params,
                                                      const std::function<bool(text::TokenId)>& on_token) {
  Result r;
  r.prompt_tokens = ids.size();
  const auto gone = [&] { return params.cancelled && params.cancelled(); };
  // Queued behind another generation: give up as soon as the requester is gone (the work would be for nobody).
  std::unique_lock<std::timed_mutex> lk(mu_, std::defer_lock);
  while (!lk.try_lock_for(std::chrono::milliseconds(20)))
    if (gone()) return r.cancelled = true, r;
  if (ids.empty()) return r.error = "empty prompt", r;
  if (ids.size() >= max_context_)
    return r.error = "prompt of " + std::to_string(ids.size()) + " tokens exceeds the context of " +
                     std::to_string(max_context_), r;
  const std::uint32_t budget =
      std::min<std::uint32_t>(params.max_tokens, max_context_ - static_cast<std::uint32_t>(ids.size()));
  // MTP: greedy at any sampler setting but penalties, sampled without penalties (the speculative acceptance does
  // not model them; such requests step as without MTP).
  const bool greedy = params.sampling.temperature == 0.0F;
  const bool spec = spec_.enabled && !params.sampling.penalties_enabled();
  std::string error;
  // A prompt the memory left cannot hold is refused before its prefill (minutes at long context), not midway.
  // A speculative cycle writes up to kMaxDrafts rows past the last emitted position (item 5).
  const std::uint32_t positions = std::min<std::uint32_t>(
      max_context_, static_cast<std::uint32_t>(ids.size()) + budget + (spec ? engine::FlashNextGpu::kMaxDrafts : 0));
  using Clock = std::chrono::steady_clock;
  static const bool trace = std::getenv("HALO_CACHE_TRACE") != nullptr;  // per-request timing on stderr
  const auto t0 = Clock::now();
  SessionCache::Prepared prep;
  const std::size_t n = ids.size() - 1;  // prefilled; the last prompt token is the first decode step's
  const auto before_prefill = [&](std::size_t reused, std::string* err) {
    if (spec_.cycle) return true;  // test hook: no draft block
    if (!spec) return !engine_.MtpTracking() || engine_.MtpTrack(false, err);
    // The draft block follows the last `window` prompt positions this request prefills (reused ones are gone).
    const std::size_t start = std::max(reused, n > spec_.window ? n - spec_.window : std::size_t{0});
    r.spec.catchup_tokens = n - std::min(start, n);
    return engine_.MtpTrackFrom(static_cast<std::uint32_t>(start), err);
  };
  if (!cache_.Prepare(ids, stable, positions, &prep, &error, gone, before_prefill)) {
    if (prep.cancelled) return r.cancelled = true, r;
    return r.error = error, r;
  }
  r.cached_tokens = prep.reused;
  const auto t1 = Clock::now();
  auto t2 = t1;
  bool first = true;

  std::vector<text::TokenId> as_tok(ids.begin(), ids.end());
  sampling::SamplerState state(params.sampling, as_tok);
  DecodeIo io;
  io.step = [&](std::int32_t token, std::uint32_t drafts, DecodeCycle* c, std::string* err) {
    if (drafts == 0) {  // an ordinary step, device sampling (a tracking draft block follows it)
      engine::FlashNextGpu::SampleRequest req{&params.sampling, state.penalties(), state.rng_state()};
      engine::FlashNextGpu::SampleResult res;
      if (!engine_.StepSample(token, req, &res, err)) return false;
      state.SetRngState(res.rng);
      c->tokens.assign(1, res.token);
    } else {  // greedy: device argmax per verified row (StepSample's tokens); sampled: p / q acceptance on the host
      engine::FlashNextGpu::SpecResult sr;
      sampling::SamplerState* sampler = greedy ? nullptr : &state;
      if (!(spec_.cycle ? spec_.cycle(token, drafts, sampler, &sr, err) : engine_.SpecStep(token, drafts, sampler, &sr, err)))
        return false;
      c->tokens = std::move(sr.tokens);
      c->drafted = sr.drafted;
      c->accepted = sr.accepted;
    }
    if (first) t2 = Clock::now(), first = false;
    return true;
  };
  io.ran = [&](std::int32_t t) { cache_.Ran(t); };
  io.gone = gone;
  io.is_stop = [&](text::TokenId t) { return IsStop(t); };
  io.emit = [&](text::TokenId t) {
    state.Accept(t);
    return on_token(t);
  };
  if (spec) chooser_.Reset();
  const DecodeOutcome out = RunDecode(ids.back(), budget, spec ? &chooser_ : nullptr, io, &r.spec);
  if (!out.error.empty()) {
    cache_.Abandon();
    return r.error = out.error, r;
  }
  r.completion_tokens = out.completion;
  r.cancelled = out.cancelled;
  r.length_limited = out.length_limited;
  // A sampled token that was neither a stop nor run is not in the session; its state is exactly what it ran.
  const auto t3 = Clock::now();
  cache_.Finish();
  if (trace) {
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::fprintf(stderr, "cache: prompt %zu reused %zu prefilled %zu ckpt %d | prepare %.1f ms, first step %.1f ms, "
                 "decode %.1f ms, finish %.1f ms | %llu tokens, retained %zu | spec catch-up %llu, cycles %llu, "
                 "drafted %llu, accepted %llu\n", ids.size(), prep.reused, prep.prefilled,
                 prep.checkpointed ? 1 : 0, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, Clock::now()),
                 static_cast<unsigned long long>(r.completion_tokens), cache_.retained(),
                 static_cast<unsigned long long>(r.spec.catchup_tokens), static_cast<unsigned long long>(r.spec.cycles),
                 static_cast<unsigned long long>(r.spec.drafted), static_cast<unsigned long long>(r.spec.accepted));
  }
  return r;
}

}  // namespace halo::server
