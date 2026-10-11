#include "server/flash_next_backend.hpp"

#include <algorithm>

namespace halo::server {

FlashNextBackend::FlashNextBackend(engine::FlashNextGpu& engine, const text::Tokenizer* tokenizer,
                                   std::vector<text::TokenId> stops, SessionCacheOptions cache, SpecOptions spec)
    : engine_(engine), tok_(tokenizer), stops_(std::move(stops)), cache_(engine, cache), spec_(std::move(spec)) {
  if (tok_) {
    for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
      const auto ids = tok_->Encode(s, true);
      if (ids.size() == 1) stops_.push_back(ids[0]);
    }
    if (tok_->eos() != text::kNoToken) stops_.push_back(tok_->eos());
    if (const auto ids = tok_->Encode("<|im_start|>", true); ids.size() == 1) im_start_ = ids[0];
  }
  bytes_per_token_ = engine_.KvBytesPerToken();
}

bool FlashNextBackend::Encode(const std::string& prompt, std::vector<std::int32_t>* ids, std::size_t* stable) const {
  if (!tok_) return false;
  const auto encoded = tok_->Encode(prompt, true);
  ids->assign(encoded.begin(), encoded.end());
  // The stable prefix: everything before the last <|im_start|> (the generation prompt), which the next turn of the
  // conversation renders again however it renders this turn's output (as FlashNextGenerator).
  *stable = 0;
  for (std::size_t i = ids->size(); i-- > 0;)
    if (static_cast<text::TokenId>((*ids)[i]) == im_start_) {
      *stable = i;
      break;
    }
  return true;
}

bool FlashNextBackend::IsStop(text::TokenId t) const {
  return std::find(stops_.begin(), stops_.end(), t) != stops_.end();
}

bool FlashNextBackend::Begin(Id id, std::span<const std::int32_t> prompt, std::size_t stable, std::uint32_t positions,
                             bool spec, bool keep_tracking, SpecDecodeStats* spec_stats, std::size_t* reused,
                             std::uint64_t* bound_key, std::string* error) {
  const std::size_t n = prompt.size() - 1;
  const auto before_prefill = [&](std::size_t r, std::string* err) {
    if (spec_.cycle) return true;  // test hook: no draft block
    if (!spec) return keep_tracking || !engine_.MtpTracking() || engine_.MtpTrack(false, err);
    // The draft block follows the last `window` prompt positions this request prefills (reused ones are gone).
    const std::size_t start = std::max(r, n > spec_.window ? n - spec_.window : std::size_t{0});
    if (spec_stats) spec_stats->catchup_tokens = n - std::min(start, n);
    return engine_.MtpTrackFrom(static_cast<std::uint32_t>(start), err);
  };
  SessionCache::Prepared prep;
  if (!cache_.Begin(id, prompt, stable, positions, &prep, error, before_prefill)) return false;
  *reused = prep.reused;
  *bound_key = cache_.LeaseBoundKey(id);
  return true;
}

bool FlashNextBackend::PrefillSlice(Id id, std::uint32_t max_tokens, std::size_t* ran, bool* done, std::string* error) {
  // One engine call; its chunks are the slice (2048 contended, 8192 alone: D1).
  return cache_.PrefillSlice(id, max_tokens, max_tokens, ran, done, error);
}

bool FlashNextBackend::Step(Id id, std::int32_t token, std::uint32_t drafts, const sampling::SamplingConfig& sampling,
                            sampling::SamplerState* state, DecodeCycle* c, std::string* error) {
  if (engine_.BoundSession() != cache_.LeaseSession(id) && !engine_.BindSession(cache_.LeaseSession(id), error))
    return false;
  if (drafts == 0) {  // an ordinary step, device sampling (a tracking draft block follows it)
    engine::FlashNextGpu::SampleRequest req{&sampling, state->penalties(), state->rng_state()};
    engine::FlashNextGpu::SampleResult res;
    if (!engine_.StepSample(token, req, &res, error)) return false;
    state->SetRngState(res.rng);
    c->tokens.assign(1, res.token);
    return true;
  }
  // greedy: device argmax per verified row (StepSample's tokens); sampled: p / q acceptance on the host
  engine::FlashNextGpu::SpecResult sr;
  sampling::SamplerState* sampler = sampling.temperature == 0.0F ? nullptr : state;
  if (!(spec_.cycle ? spec_.cycle(token, drafts, sampler, &sr, error) : engine_.SpecStep(token, drafts, sampler, &sr, error)))
    return false;
  c->tokens = std::move(sr.tokens);
  c->drafted = sr.drafted;
  c->accepted = sr.accepted;
  return true;
}

bool FlashNextBackend::StepBatch(std::vector<BatchItem>& items, std::string* error) {
  // Each entry byte-identical to stepping its session alone (SESSIONS 3); ordinary steps only.
  std::vector<engine::FlashNextGpu::SampleRequest> reqs(items.size());
  std::vector<engine::FlashNextGpu::SampleResult> res(items.size());
  std::vector<engine::FlashNextGpu::BatchEntry> entries(items.size());
  for (std::size_t i = 0; i < items.size(); ++i) {
    reqs[i] = {items[i].sampling, items[i].state->penalties(), items[i].state->rng_state()};
    entries[i].session = cache_.LeaseSession(items[i].id);
    entries[i].token = items[i].token;
    entries[i].sample = &reqs[i];
    entries[i].result = &res[i];
  }
  if (!engine_.StepBatch(entries, error)) return false;
  for (std::size_t i = 0; i < items.size(); ++i) {
    items[i].state->SetRngState(res[i].rng);
    items[i].out = res[i].token;
  }
  return true;
}

// Called inside Scheduler::Next (its lock held): takes no other lock.
std::optional<Scheduler::Retained> FlashNextBackend::Park(Id id) {
  if (!cache_.HasLease(id)) return std::nullopt;
  const std::uint64_t key = cache_.Park(id);
  if (key == 0) return std::nullopt;
  for (const auto& r : cache_.Residents())
    if (r.key == key) return Scheduler::Retained{r.key, r.bytes, r.tokens, r.usable};
  return std::nullopt;
}

std::vector<Scheduler::Retained> FlashNextBackend::Retained() const {
  std::vector<Scheduler::Retained> rs;
  for (const auto& r : cache_.Residents()) rs.push_back({r.key, r.bytes, r.tokens, r.usable});
  return rs;
}

std::unique_ptr<ScheduledGenerator> MakeScheduledFlashNext(engine::FlashNextGpu& engine, const text::Tokenizer* tokenizer,
                                                           std::vector<text::TokenId> stops, std::uint32_t max_context,
                                                           SessionCacheOptions cache, SpecOptions spec,
                                                           ScheduledOptions options, FlashNextBackend** backend_out) {
  const auto depth = spec.depth;
  auto backend = std::make_unique<FlashNextBackend>(engine, tokenizer, std::move(stops), cache, std::move(spec));
  if (backend_out) *backend_out = backend.get();
  return std::make_unique<ScheduledGenerator>(std::move(backend), max_context, depth, std::move(options));
}

}  // namespace halo::server
