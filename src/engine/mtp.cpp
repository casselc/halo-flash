// P5 measurement stage: the MTP draft block on the GPU (ReferenceModel::MtpStep's graph), eager launches, the
// trunk's helpers (HcMix, Attention, Moe) on the draft block's weights. No verify path: the trunk is unchanged.

#include <cstdlib>
#include <cstring>

#include "engine/flash_next_gpu.hpp"
#include "kernels/step_params.h"

namespace halo::engine {

namespace {
constexpr std::uint32_t kBlock = 256;
std::uint32_t Blocks(std::size_t n, std::uint32_t per = kBlock) { return static_cast<std::uint32_t>((n + per - 1) / per); }
rt::Buffer View(const rt::Buffer& b, std::size_t offset_bytes, std::size_t bytes) {
  return rt::Buffer{static_cast<char*>(b.device) + offset_bytes, bytes, nullptr};
}
}  // namespace

bool FlashNextGpu::LoadMtp(const model::MtpWeights& m, std::string* error) {
  const auto& c = w_.config;
  const auto& l = m.block;
  const auto up_mixer = [&](const model::HcMixer& x) {
    return Upload(x.norm, error) && Upload(x.down, error) && Upload(x.up, error) && Upload(x.inject, error);
  };
  if (!up_mixer(l.hc_attn) || !up_mixer(l.hc_ffn) || !up_mixer(l.nextn_head)) return false;
  for (const auto* t : {&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_out, &l.attn_q_norm, &l.attn_k_norm, &l.indexer_q,
                        &l.indexer_k, &l.indexer_q_norm, &l.indexer_k_norm, &l.router, &l.ffn_gate_exps, &l.ffn_up_exps,
                        &l.ffn_down_exps, &l.shexp_gate_inp, &l.shexp_gate, &l.shexp_up, &l.shexp_down, &l.nextn_enorm,
                        &l.nextn_hnorm, &l.nextn_eh_proj})
    if (!Upload(*t, error)) return false;
  if (repack_q8_)
    for (const auto* t : {&l.hc_attn.down, &l.hc_attn.up, &l.hc_ffn.down, &l.hc_ffn.up, &l.nextn_head.down,
                          &l.nextn_head.up, &l.attn_q, &l.attn_k, &l.attn_v, &l.attn_out, &l.ffn_gate_exps,
                          &l.ffn_up_exps, &l.ffn_down_exps, &l.shexp_gate, &l.shexp_up, &l.shexp_down,
                          &l.nextn_eh_proj})
      if (!t->empty() && !RepackQ8(*t, tensors_.at(t->data), error)) return false;
  const std::size_t D = c.HcDim(), mc = max_context_;
  mtp_res_ = Scratch(D, error);
  mtp_in_ = Scratch(D, error);
  mtp_hn_ = Scratch(D, error);
  mtp_cat_ = Scratch(2ull * c.hidden_size, error);
  mtp_attn_.k = Scratch(mc * c.AttentionKvDim(), error);
  mtp_attn_.v = Scratch(mc * c.AttentionKvDim(), error);
  mtp_attn_.index_k = Scratch(mc * c.indexer_head_dim, error);
  mtp_attn_.block_k = Scratch((mc / c.compress_ratio + 1) * std::size_t{c.indexer_head_dim}, error);
  if (!mtp_res_ || !mtp_in_ || !mtp_hn_ || !mtp_cat_ || !mtp_attn_.k || !mtp_attn_.v || !mtp_attn_.index_k ||
      !mtp_attn_.block_k)
    return false;
  mtp_ = &m;
  mtp_pos_ = 0;
  return dev_.Synchronize(error);
}

bool FlashNextGpu::SetDraftVocab(std::span<const std::int32_t> ids, std::string* error) {
  const auto& out = w_.output;
  const auto& d = tensors_.at(out.data);
  if (draft_qs_) dev_.Free(draft_qs_);
  if (draft_d_) dev_.Free(draft_d_);
  draft_rows_ = 0;
  if (ids.empty()) return true;
  if (!d.qs) {
    if (error) *error = "trimmed draft head needs the repacked Q8 LM head";
    return false;
  }
  const std::size_t cols = out.cols, rowq = d.qstride, rowd = cols / 32 * 2, V = out.rows;
  std::vector<std::uint8_t> qs(V * rowq), dd(V * rowd), tq(ids.size() * rowq), td(ids.size() * rowd);
  if (!dev_.Download(d.qs, 0, qs.data(), qs.size(), error) || !dev_.Download(d.d, 0, dd.data(), dd.size(), error))
    return false;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    std::memcpy(tq.data() + i * rowq, qs.data() + static_cast<std::size_t>(ids[i]) * rowq, rowq);
    std::memcpy(td.data() + i * rowd, dd.data() + static_cast<std::size_t>(ids[i]) * rowd, rowd);
  }
  draft_qs_ = dev_.Alloc(tq.size(), error);
  draft_d_ = dev_.Alloc(td.size(), error);
  if (!draft_qs_ || !draft_d_ || !dev_.Upload(draft_qs_, 0, tq.data(), tq.size(), error) ||
      !dev_.Upload(draft_d_, 0, td.data(), td.size(), error))
    return false;
  owned_.push_back(draft_qs_);
  owned_.push_back(draft_d_);
  draft_rows_ = static_cast<std::uint32_t>(ids.size());
  draft_ids_.assign(ids.begin(), ids.end());
  return true;
}

bool FlashNextGpu::TrunkHidden(std::vector<float>* out, std::string* error) {
  out->resize(w_.config.HcDim());
  return dev_.Download(res_, 0, out->data(), out->size() * 4, error);
}

bool FlashNextGpu::MtpStep(std::int32_t token, bool from_trunk, std::span<float> logits, std::string* error) {
  return MtpRun(token, from_trunk ? &res_ : nullptr, logits, error);
}

bool FlashNextGpu::MtpRun(std::int32_t token, const rt::Buffer* hidden, std::span<float> logits, std::string* error) {
  const auto& c = w_.config;
  if (!mtp_) {
    if (error) *error = "MTP block not loaded";
    return false;
  }
  if (mtp_pos_ >= max_context_ || token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
    if (error) *error = mtp_pos_ >= max_context_ ? "MTP context is full" : "token out of range";
    return false;
  }
  const auto& l = mtp_->block;
  const std::uint32_t H = c.hidden_size, D = c.HcDim(), ratio = c.compress_ratio, budget = c.indexer_top_k / ratio;
  const std::uint32_t n_kv = mtp_pos_ + 1, complete = n_kv / ratio;
  std::uint32_t params[kHaloStepParamCount] = {};
  params[kHaloToken] = static_cast<std::uint32_t>(token);
  params[kHaloPosition] = mtp_pos_;
  params[kHaloNkv] = n_kv;
  params[kHaloComplete] = complete;
  params[kHaloBlock] = mtp_pos_ / ratio;
  params[kHaloNVisible] = complete <= budget ? n_kv : budget * ratio + (n_kv - complete * ratio);
  if (!dev_.Upload(ple_emb_, params_offset_, params, sizeof(params), error)) return false;
  // Hidden input: the trunk residual (res_) or the block's previous output.
  if (!CopyFloats(mtp_in_, hidden ? *hidden : mtp_res_, D, error)) return false;
  std::swap(res_, mtp_res_);  // the helpers below work on res_
  struct Back {
    rt::Buffer& a;
    rt::Buffer& b;
    ~Back() { std::swap(a, b); }
  } back{res_, mtp_res_};
  const auto& emb = tensors_.at(w_.token_embd.data);
  const rt::Arg eargs[] = {rt::Arg::Buf(emb.buf), rt::Arg::Buf(embd_), rt::Arg::Buf(ple_emb_, params_offset_),
                           rt::Arg::Val(H), rt::Arg::Val<std::uint64_t>(w_.token_embd.RowBytes())};
  const auto norm = [&](const rt::Buffer& x, const model::TensorRef& w, const rt::Buffer& y, std::uint32_t d) {
    return Launch("halo_rmsnorm", {1}, {kBlock},
                  {rt::Arg::Buf(x), rt::Arg::Buf(tensors_.at(w.data).buf), rt::Arg::Buf(y), rt::Arg::Val(d),
                   rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(0)},
                  error);
  };
  const auto combine = [&]() {
    return Launch("halo_hc_combine", {Blocks(D)}, {kBlock},
                  {rt::Arg::Buf(res_), rt::Arg::Buf(block_out_), rt::Arg::Buf(inject_), rt::Arg::Val(c.hc_count),
                   rt::Arg::Val(H)},
                  error);
  };
  if (!dev_.Launch(emb.embed, {Blocks(H)}, {kBlock}, eargs, error) || !norm(embd_, l.nextn_enorm, mtp_cat_, H) ||
      !(std::getenv("HALO_MTP_HNORM_GROUPED")  // wiring ablation: one RMS per stream instead of one over all
            ? Launch("halo_rmsnorm", {c.hc_count}, {kBlock},
                     {rt::Arg::Buf(mtp_in_), rt::Arg::Buf(tensors_.at(l.nextn_hnorm.data).buf), rt::Arg::Buf(mtp_hn_),
                      rt::Arg::Val(H), rt::Arg::Val(c.rms_eps), rt::Arg::Val(H)},
                     error)
            : norm(mtp_in_, l.nextn_hnorm, mtp_hn_, D)))
    return false;
  // fused stream s = eh_proj [fc_embedding | fc_hidden] . [enorm(e) ; hnorm(h)_s] (one dot over 2H; the reference
  // sums the two halves separately in double, so draft logits differ from it at float rounding).
  for (std::uint32_t s = 0; s < c.hc_count; ++s)
    if (!CopyFloats(View(mtp_cat_, std::size_t{H} * 4, std::size_t{H} * 4), View(mtp_hn_, std::size_t{s} * H * 4, H * 4ull),
                    H, error) ||
        !MatVec(l.nextn_eh_proj, 0, mtp_cat_, res_, error, 0, std::size_t{s} * H * 4))
      return false;
  if (!HcMix(l.hc_attn, true, error) || !Attention(l, mtp_attn_, error) || !combine() ||
      !HcMix(l.hc_ffn, true, error) || !Moe(l, error) || !combine())
    return false;
  const bool trimmed = draft_rows_ > 0 && logits.size() == draft_rows_;
  const auto head = [&]() {
    if (!trimmed) return MatVec(w_.output, 0, mixed_, logits_, error);
    const auto& d = tensors_.at(w_.output.data);
    const auto cols = static_cast<std::uint32_t>(w_.output.cols);
    return Launch(draft_rows_ < 2048 ? "halo_gemv_q8r_r2" : "halo_gemv_q8r_r4", {Blocks(draft_rows_, draft_rows_ < 2048 ? 16 : 32)},
                  {256},
                  {rt::Arg::Buf(draft_qs_), rt::Arg::Buf(draft_d_), rt::Arg::Buf(mixed_), rt::Arg::Buf(logits_),
                   rt::Arg::Val(draft_rows_), rt::Arg::Val(cols), rt::Arg::Val(d.qstride)},
                  error);
  };
  if (!logits.empty() &&
      (!HcMix(l.nextn_head, false, error) || !head() ||
       !dev_.Download(logits_, 0, logits.data(), (trimmed ? draft_rows_ : c.vocab_size) * 4ull, error)))
    return false;
  if (logits.empty() && !dev_.Synchronize(error)) return false;
  mtp_attn_.blocks = complete;
  ++mtp_pos_;
  return true;
}

}  // namespace halo::engine
