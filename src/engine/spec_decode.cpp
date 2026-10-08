// P5 speculative decoding with the MTP draft block (docs/MTP.md).
//
// One cycle (SpecStep): the draft block proposes k tokens; the trunk verifies [token, d1 .. dk] as k + 1 rows of
// one batched pass (RecordBatch with every entry the bound session at consecutive positions: each row's
// arithmetic is the single step's, so its logits are byte-identical to stepping); the longest agreeing prefix
// is kept. Rows past it are undone without replay: the DeltaNet state, conv windows and PLE history were copied
// after every row but the last (verify_snap_), the KV / indexer rows past the position are simply overwritten
// later. Greedy acceptance compares the drafts with the device greedy sample of each row (StepSample's kernels),
// so the emitted tokens equal AR's exactly. Sampled acceptance is p / q with residual correction on the host.

#include <algorithm>
#include <cstring>

#include "engine/flash_next_gpu.hpp"
#include "engine/mtp_sampling.hpp"
#include "kernels/sample_params.h"
#include "kernels/step_params.h"

namespace halo::engine {

namespace {
rt::Buffer View(const rt::Buffer& b, std::size_t offset_bytes, std::size_t bytes) {
  return rt::Buffer{static_cast<char*>(b.device) + offset_bytes, bytes, nullptr};
}
}  // namespace

bool FlashNextGpu::MtpTrack(bool on, std::string* error) {
  if (on && !mtp_) {
    if (error) *error = "MtpTrack needs LoadMtp first";
    return false;
  }
  if (on && !mtp_hpend_) {
    mtp_hpend_ = Scratch(w_.config.HcDim(), error);
    if (!mtp_hpend_) return false;
  }
  mtp_track_ = on;
  mtp_owner_ = bound_;
  // From position 0 the draft block follows every step; mid-sequence it has no history, so no drafting until the
  // next Reset (MtpTrackFrom starts a history mid-sequence).
  mtp_pos_ = 0;
  mtp_base_ = position_ == 0 ? 0 : kMtpNever;
  return true;
}

bool FlashNextGpu::MtpTrackFrom(std::uint32_t start, std::string* error) {
  if (!MtpTrack(true, error)) return false;
  mtp_base_ = std::max(start, position_);
  return true;
}

bool FlashNextGpu::MtpSave(MtpCheckpoint* cp, std::string* error) {
  if (!mtp_hpend_) {
    if (error) *error = "MtpSave: the draft block is not tracking";
    return false;
  }
  cp->pos = mtp_pos_;
  cp->base = mtp_base_;
  cp->hpend.resize(w_.config.HcDim());
  return dev_.Download(mtp_hpend_, 0, cp->hpend.data(), cp->hpend.size() * 4, error);
}

bool FlashNextGpu::MtpRestore(const MtpCheckpoint& cp, std::string* error) {
  if (!mtp_hpend_ || cp.hpend.size() != w_.config.HcDim()) {
    if (error) *error = "MtpRestore: not tracking, or not a checkpoint of this model";
    return false;
  }
  mtp_pos_ = cp.pos;  // the draft block's cache rows past it are overwritten by later steps
  mtp_base_ = cp.base;
  return dev_.Upload(mtp_hpend_, 0, cp.hpend.data(), cp.hpend.size() * 4, error);
}

bool FlashNextGpu::AllocVerifySnaps(std::uint32_t rows, std::string* error) {
  const auto& c = w_.config;
  while (verify_snap_.size() < rows) {
    std::vector<rt::Buffer> v(2 * c.num_layers + 1);
    for (std::uint32_t l = 0; l < c.num_layers; ++l)
      if (c.IsLinearLayer(l)) {
        v[2 * l] = Scratch(linear_[l].state.bytes / 4, error);
        v[2 * l + 1] = Scratch(linear_[l].conv.bytes / 4, error);
        if (!v[2 * l] || !v[2 * l + 1]) return false;
      }
    v[2 * c.num_layers] = Scratch(ple_history_.bytes / 4, error);
    if (!v[2 * c.num_layers]) return false;
    verify_snap_.push_back(std::move(v));
  }
  return true;
}

bool FlashNextGpu::SpecStep(std::int32_t token, std::uint32_t drafts, sampling::SamplerState* sampler,
                            SpecResult* result, std::string* error) {
  const auto& c = w_.config;
  const auto fail = [&](const char* what) {
    if (error) *error = what;
    return false;
  };
  if (!result) return fail("SpecStep needs a result");
  result->tokens.clear();
  result->drafts.clear();
  result->drafted = result->accepted = 0;
  const std::uint32_t p = position_;
  std::uint32_t k = std::min({drafts, kMaxDrafts, kMaxBatch - 1});
  if (p + k + 1 > max_context_) k = max_context_ > p + 1 ? max_context_ - p - 1 : 0;
  static const sampling::SamplingConfig greedy;
  const bool can_draft = k > 0 && mtp_ && MtpPending() && BatchSupported();
  if (!can_draft) {  // an ordinary step; the Run hook keeps the draft block following
    if (sampler) {
      std::vector<float> l(c.vocab_size);
      if (!Step(token, l, error)) return false;
      result->tokens.push_back(static_cast<std::int32_t>(sampler->Sample(l)));
      return true;
    }
    SampleResult r;
    if (!StepSample(token, {&greedy, {}, 0}, &r, error)) return false;
    result->tokens.push_back(r.token);
    return true;
  }
  if (!AllocBatch(error) || !AllocVerifySnaps(k, error) || !EnsureKv(p + k + 1, p, error)) return false;
  // 1. Drafts: the pending base step (hidden of p - 1, token) proposes d1, then the chain.
  const std::size_t QV = draft_rows_ ? draft_rows_ : c.vocab_size;
  std::vector<float> q(QV);
  std::vector<std::int32_t> rows_tok(k + 1);
  rows_tok[0] = token;
  std::vector<MtpProposal> props(sampler ? k : 0);
  const auto draft_token = [&](std::uint32_t j) -> std::int32_t {
    if (!sampler) {
      const auto i = static_cast<std::int32_t>(std::max_element(q.begin(), q.end()) - q.begin());
      return draft_rows_ ? draft_ids_[static_cast<std::size_t>(i)] : i;
    }
    // Proposal over the draft's top candidates (gufo's bounded-support q), drawn with the request's RNG.
    MtpCandidateLogits cand;
    std::vector<std::uint32_t> idx(QV);
    for (std::uint32_t i = 0; i < QV; ++i) idx[i] = i;
    const std::size_t n = std::min<std::size_t>(kMtpCandidates, QV);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(n), idx.end(),
                      [&](std::uint32_t a, std::uint32_t b) { return q[a] > q[b] || (q[a] == q[b] && a < b); });
    cand.size = n;
    for (std::size_t i = 0; i < n; ++i) {
      cand.ids[i] = static_cast<sampling::TokenId>(draft_rows_ ? draft_ids_[idx[i]] : static_cast<std::int32_t>(idx[i]));
      cand.logits[i] = q[idx[i]];
    }
    props[j] = SampleMtpProposal(cand, *sampler, sampler->mutable_rng_state());
    return static_cast<std::int32_t>(props[j].token);
  };
  if (!MtpRun(token, &mtp_hpend_, q, error)) return false;  // mtp_pos_: p - 1 -> p
  rows_tok[1] = draft_token(0);
  for (std::uint32_t j = 1; j < k; ++j) {
    if (!MtpRun(rows_tok[j], nullptr, q, error)) return false;
    rows_tok[j + 1] = draft_token(j);
  }
  return SpecVerify(rows_tok, props, sampler, result, error);
}

bool FlashNextGpu::SpecStepDrafts(std::int32_t token, std::span<const std::int32_t> drafts, SpecResult* result,
                                  std::string* error) {
  const auto& c = w_.config;
  if (!result || drafts.empty() || drafts.size() > std::min(kMaxDrafts, kMaxBatch - 1) ||
      position_ + drafts.size() + 1 > max_context_ || !BatchSupported() || (mtp_track_ && bound_ == mtp_owner_)) {
    if (error) *error = "SpecStepDrafts: bad drafts, full context, no batch path, or the draft block is tracking";
    return false;
  }
  for (const auto d : drafts)
    if (d < 0 || static_cast<std::uint32_t>(d) >= c.vocab_size) {
      if (error) *error = "SpecStepDrafts: draft token out of range";
      return false;
    }
  result->tokens.clear();
  result->drafted = result->accepted = 0;
  if (!AllocBatch(error) || !AllocVerifySnaps(static_cast<std::uint32_t>(drafts.size()), error) ||
      !EnsureKv(position_ + static_cast<std::uint32_t>(drafts.size()) + 1, position_, error))
    return false;
  std::vector<std::int32_t> rows_tok{token};
  rows_tok.insert(rows_tok.end(), drafts.begin(), drafts.end());
  return SpecVerify(rows_tok, {}, nullptr, result, error);
}

bool FlashNextGpu::SpecVerify(std::span<const std::int32_t> rows_tok, std::span<const MtpProposal> props,
                              sampling::SamplerState* sampler, SpecResult* result, std::string* error) {
  const auto& c = w_.config;
  const auto fail = [&](const char* what) {
    if (error) *error = what;
    return false;
  };
  static const sampling::SamplingConfig greedy;
  const std::uint32_t p = position_, k = static_cast<std::uint32_t>(rows_tok.size()) - 1;
  // 2. Verify the k + 1 rows in one batched pass.
  const std::uint32_t B = k + 1, ratio = c.compress_ratio, budget = c.indexer_top_k / ratio;
  std::vector<model::NgramHistory> hist(B + 1);
  hist[0] = ngram_history_;
  std::vector<std::uint32_t> ngram_rows(c.ple_layer >= 0 ? c.ple_heads : 0);
  std::size_t upload = 0;
  const Head head = sampler ? Head::kLogits : Head::kSample;
  for (std::uint32_t i = 0; i < B; ++i) {
    std::uint8_t* host = bb_.host.data() + i * bb_.entry_stride;
    hist[i + 1] = hist[i];
    if (c.ple_layer >= 0) {
      model::HashNgramRows(c, hist[i + 1], std::span<const std::int32_t>(&rows_tok[i], 1), ngram_rows);
      const std::span<float> emb(reinterpret_cast<float*>(host), c.PleEmbeddingDim());
      if (!ngram_ || !ngram_->Read(ngram_rows, emb)) return fail("n-gram table read failed");
    }
    const std::uint32_t pos = p + i, n_kv = pos + 1, complete = n_kv / ratio;
    std::uint32_t params[kHaloStepParamCount] = {};
    params[kHaloToken] = static_cast<std::uint32_t>(rows_tok[i]);
    params[kHaloPosition] = pos;
    params[kHaloNkv] = n_kv;
    params[kHaloComplete] = complete;
    params[kHaloBlock] = pos / ratio;
    params[kHaloNVisible] = complete <= budget ? n_kv : budget * ratio + (n_kv - complete * ratio);
    std::memcpy(host + params_offset_, params, sizeof(params));
    std::size_t used = sample_offset_;
    if (head == Head::kSample) {
      HaloSampleParams sp{greedy.temperature, greedy.top_p, greedy.min_p, greedy.repeat_penalty,
                          greedy.frequency_penalty, greedy.presence_penalty,
                          static_cast<std::uint32_t>(std::max(greedy.top_k, 0)),
                          static_cast<std::uint32_t>(std::min<std::size_t>(greedy.min_keep, UINT32_MAX)), 0u,
                          c.vocab_size, 0, sampling_mode_ == SamplingMode::kFast ? 1u : 0u, 0};
      std::memcpy(host + sample_offset_, &sp, sizeof(sp));
      used = penalty_offset_;
    }
    upload = i * bb_.entry_stride + used;
  }
  if (!dev_.Upload(bb_.params, 0, bb_.host.data(), upload, error)) return false;
  const std::vector<SessionId> ids(B, bound_);
  BatchGraph* g = nullptr;
  if (use_graphs_ && !timing_) {
    for (auto& bg : batch_graphs_)
      if (bg.verify && bg.head == static_cast<int>(head) && bg.sessions == ids) g = &bg;
    if (!g) {
      BatchGraph n;
      n.sessions = ids;
      n.head = static_cast<int>(head);
      n.verify = true;
      verify_ = true;
      if (!dev_.BeginCapture(error)) return verify_ = false, false;
      const bool recorded = RecordBatch(ids, head, error);
      std::string end_error;
      const bool ended = dev_.EndCapture(&n.graph, &end_error);
      verify_ = false;
      if (!recorded || !ended) {
        if (recorded && error) *error = end_error;
        return false;
      }
      batch_graphs_.push_back(std::move(n));
      g = &batch_graphs_.back();
    }
    g->used = ++batch_clock_;
    if (!dev_.LaunchGraph(g->graph, error)) return false;
  } else {
    verify_ = true;
    const bool ok = RecordBatch(ids, head, error);
    verify_ = false;
    if (!ok) return false;
  }
  // 3. Acceptance.
  std::uint32_t a = 0;  // accepted drafts
  if (!sampler) {
    const std::size_t per = (sizeof(HaloSampleOut) + 3) / 4 * 4;
    std::vector<std::uint8_t> outs(per * B);
    if (!dev_.Download(bb_.sample_out, 0, outs.data(), outs.size(), error)) return false;
    std::vector<std::int32_t> y(B);
    for (std::uint32_t i = 0; i < B; ++i) {
      HaloSampleOut o;
      std::memcpy(&o, outs.data() + i * per, sizeof(o));
      if (o.token >= c.vocab_size) return fail("device sampling did not finish");
      y[i] = static_cast<std::int32_t>(o.token);
    }
    while (a < k && rows_tok[a + 1] == y[a]) ++a;
    result->tokens.assign(y.begin(), y.begin() + a + 1);
  } else {
    std::vector<float> l(std::size_t{B} * c.vocab_size);
    if (!dev_.Download(bb_.logits, 0, l.data(), l.size() * 4, error)) return false;
    bool stopped = false;
    for (std::uint32_t j = 0; j < k && !stopped; ++j) {
      const std::span<const float> lj(l.data() + std::size_t{j} * c.vocab_size, c.vocab_size);
      const auto v = VerifyMtpProposal(lj, props[j], *sampler);
      result->tokens.push_back(static_cast<std::int32_t>(v.token));
      if (v.accepted) ++a;
      else stopped = true;
    }
    if (!stopped)  // every draft accepted: the bonus token from the last row
      result->tokens.push_back(static_cast<std::int32_t>(
          sampler->Sample(std::span<const float>(l.data() + std::size_t{k} * c.vocab_size, c.vocab_size))));
  }
  result->drafted = k;
  result->accepted = a;
  result->drafts.assign(rows_tok.begin() + 1, rows_tok.end());
  // 4. Commit rows 0 .. a: roll back the recurrent state when a row was rejected.
  if (a < k) {
    const auto& v = verify_snap_[a];
    for (std::uint32_t l = 0; l < c.num_layers; ++l)
      if (c.IsLinearLayer(l) && (!CopyFloats(linear_[l].state, v[2 * l], linear_[l].state.bytes / 4, error) ||
                                 !CopyFloats(linear_[l].conv, v[2 * l + 1], linear_[l].conv.bytes / 4, error)))
        return false;
    if (c.ple_layer >= 0 && !CopyFloats(ple_history_, v[2 * c.num_layers], ple_history_.bytes / 4, error)) return false;
  }
  position_ = p + a + 1;
  ngram_history_ = hist[a + 1];
  for (auto& s : attn_) {
    s.blocks = position_ / ratio;
    s.valid16 = std::min(s.valid16, position_);
  }
  // 5. The draft block catches up on the committed rows (their true hiddens), the last one stays pending.
  if (!(mtp_track_ && bound_ == mtp_owner_)) return dev_.Synchronize(error);  // SpecStepDrafts: no draft block
  const std::size_t D = c.HcDim();
  mtp_pos_ = p - mtp_base_;  // drafting needed MtpPending: mtp_base_ + mtp_pos_ + 1 == p before the drafts
  for (std::uint32_t e = 0; e < a; ++e) {
    const rt::Buffer h = View(bb_.res, std::size_t{e} * D * 4, D * 4);
    if (!MtpRun(rows_tok[e + 1], &h, {}, error, mtp_catchup_ != kMtpCatchUpFull)) return false;
  }
  return CopyFloats(mtp_hpend_, View(bb_.res, std::size_t{a} * D * 4, D * 4), D, error) && dev_.Synchronize(error);
}

}  // namespace halo::engine
