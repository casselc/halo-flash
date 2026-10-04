#include "engine/flash_next_gpu.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "kernels/core_hsaco.hpp"

namespace halo::engine {

namespace {

const char* MatVecKernel(core::GgmlType t) {
  switch (t) {
    case core::GgmlType::kF32: return "halo_matvec_f32";
    case core::GgmlType::kBF16: return "halo_matvec_bf16";
    case core::GgmlType::kF16: return "halo_matvec_f16";
    case core::GgmlType::kQ8_0: return "halo_matvec_q8_0";
    case core::GgmlType::kQ4_K: return "halo_matvec_q4_k";
    case core::GgmlType::kQ5_K: return "halo_matvec_q5_k";
    case core::GgmlType::kQ6_K: return "halo_matvec_q6_k";
    case core::GgmlType::kQ5_1: return "halo_matvec_q5_1";
    case core::GgmlType::kIQ4_NL: return "halo_matvec_iq4_nl";
    default: return nullptr;
  }
}

constexpr std::uint32_t kBlock = 256;
std::uint32_t Blocks(std::size_t n, std::uint32_t per = kBlock) { return static_cast<std::uint32_t>((n + per - 1) / per); }

}  // namespace

std::unique_ptr<FlashNextGpu> FlashNextGpu::Create(rt::Device& device, const model::ModelWeights& weights,
                                                   model::NgramTable* ngram, const model::LoraAdapter* lora,
                                                   std::uint32_t max_context, std::string* error) {
  auto m = std::unique_ptr<FlashNextGpu>(new FlashNextGpu(device, weights, ngram));
  if (!m->Init(lora, max_context, error)) return nullptr;
  return m;
}

FlashNextGpu::~FlashNextGpu() {
  for (auto& [_, t] : tensors_) dev_.Free(t.buf);
  for (auto& b : owned_) dev_.Free(b);
}

rt::Buffer FlashNextGpu::Scratch(std::size_t floats, std::string* error) {
  rt::Buffer b = dev_.Alloc(std::max<std::size_t>(floats, 1) * sizeof(float), error);
  if (b) owned_.push_back(b);
  return b;
}

bool FlashNextGpu::Upload(const model::TensorRef& t, std::string* error) {
  if (t.empty() || tensors_.contains(t.data)) return true;
  DevTensor d;
  const std::size_t bytes = t.SizeBytes();
  d.buf = dev_.Alloc(bytes, error);
  if (!d.buf || !dev_.Upload(d.buf, 0, t.data, bytes, error)) return false;
  if (const char* k = MatVecKernel(t.type)) {
    d.matvec = dev_.Kernel(k, error);
    d.embed = dev_.Kernel(std::string(k) + "_embed", error);
  }
  tensors_.emplace(t.data, d);
  return true;
}

bool FlashNextGpu::Launch(const char* name, rt::Dim3 grid, rt::Dim3 block, std::initializer_list<rt::Arg> args,
                          std::string* error) {
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    const auto id = dev_.Kernel(name, error);
    if (id < 0) return false;
    it = kernels_.emplace(name, id).first;
  }
  return dev_.Launch(it->second, grid, block, std::span<const rt::Arg>(args.begin(), args.size()), error);
}

bool FlashNextGpu::MatVec(const model::TensorRef& t, std::uint64_t expert, const rt::Buffer& x, const rt::Buffer& y,
                          std::string* error) {
  const auto& d = tensors_.at(t.data);
  if (d.matvec < 0) {
    if (error) *error = "no GPU matvec for " + std::string(t.name);
    return false;
  }
  const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
  const rt::Arg args[] = {rt::Arg::Buf(d.buf, expert * t.RowBytes() * t.rows), rt::Arg::Buf(x), rt::Arg::Buf(y),
                          rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val<std::uint64_t>(t.RowBytes())};
  if (!dev_.Launch(d.matvec, {Blocks(rows, kBlock / 32)}, {kBlock}, args, error)) return false;
  // LoRA: y += (alpha / r) * B (A x), on dense projections, as the reference.
  if (auto it = lora_.find(t.data); it != lora_.end()) {
    const auto& [pair, scale] = it->second;
    if (!MatVec(pair->a, 0, x, lora_tmp_, error) || !MatVec(pair->b, 0, lora_tmp_, lora_out_, error) ||
        !Launch("halo_axpy", {Blocks(rows)}, {kBlock},
                {rt::Arg::Buf(y), rt::Arg::Buf(lora_out_), rt::Arg::Val(scale), rt::Arg::Val(rows)}, error))
      return false;
  }
  return true;
}

bool FlashNextGpu::Init(const model::LoraAdapter* lora, std::uint32_t max_context, std::string* error) {
  const auto& c = w_.config;
  max_context_ = max_context;
  if (!dev_.LoadModule(kernels::CoreHsaco(), error)) return false;
  std::uint32_t max_rank = 1;
  // Every tensor the step reads goes up once; LoRA pairs attach to dense
  // projections by base name, exactly where the reference applies them.
  const auto up = [&](const model::TensorRef& t) {
    if (!Upload(t, error)) return false;
    if (lora == nullptr || t.empty() || t.experts != 1) return true;
    const model::LoraPair* p = lora->Find(t.name);
    if (p == nullptr) return true;
    max_rank = std::max(max_rank, static_cast<std::uint32_t>(p->rank));
    lora_[t.data] = {p, lora->Scale(*p)};
    return Upload(p->a, error) && Upload(p->b, error);
  };
  const auto up_mixer = [&](const model::HcMixer& m) {
    return up(m.norm) && up(m.down) && up(m.up) && up(m.inject);
  };
  if (!up(w_.token_embd) || !up(w_.output) || !up_mixer(w_.hc_head)) return false;
  for (const auto& l : w_.layers) {
    if (!up_mixer(l.hc_attn) || !up_mixer(l.hc_ffn)) return false;
    for (const auto* t : {&l.ssm_qkv, &l.ssm_gate, &l.ssm_conv1d, &l.ssm_alpha, &l.ssm_beta, &l.ssm_dt, &l.ssm_a,
                          &l.ssm_norm, &l.ssm_out, &l.attn_q, &l.attn_k, &l.attn_v, &l.attn_out, &l.attn_q_norm,
                          &l.attn_k_norm, &l.indexer_q, &l.indexer_k, &l.indexer_q_norm, &l.indexer_k_norm,
                          &l.ple_key, &l.ple_value, &l.ple_norm_key, &l.ple_norm_query, &l.ple_norm_conv,
                          &l.ple_conv1d, &l.router, &l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps,
                          &l.shexp_gate_inp, &l.shexp_gate, &l.shexp_up, &l.shexp_down})
      if (!up(*t)) return false;
  }
  const std::size_t H = c.hidden_size, D = c.HcDim();
  res_ = Scratch(D, error);
  xn_ = Scratch(D, error);
  lo_ = Scratch(c.hc_low_rank, error);
  gate_hc_ = Scratch(D, error);
  mixed_ = Scratch(H, error);
  inject_ = Scratch(c.hc_count, error);
  block_out_ = Scratch(H, error);
  embd_ = Scratch(H, error);
  ple_emb_ = Scratch(c.PleEmbeddingDim(), error);
  ple_key_ = Scratch(D, error);
  ple_value_ = Scratch(H, error);
  ple_query_ = Scratch(D, error);
  ple_gated_ = Scratch(D, error);
  ple_norm_ = Scratch(D, error);
  ple_history_ = Scratch(std::size_t{c.PleConvHistory()} * D, error);
  if (ple_history_ && !dev_.Fill(ple_history_, 0, error)) return false;
  qkv_ = Scratch(c.SsmConvChannels(), error);
  conv_out_ = Scratch(c.SsmConvChannels(), error);
  z_ = Scratch(c.SsmValueDim(), error);
  alpha_ = Scratch(c.ssm_num_v_heads, error);
  beta_ = Scratch(c.ssm_num_v_heads, error);
  gdn_out_ = Scratch(c.SsmValueDim(), error);
  qg_ = Scratch(2ull * c.AttentionQDim(), error);
  q_ = Scratch(c.AttentionQDim(), error);
  gate_ = Scratch(c.AttentionQDim(), error);
  k_ = Scratch(c.AttentionKvDim(), error);
  v_ = Scratch(c.AttentionKvDim(), error);
  ctx_ = Scratch(c.AttentionQDim(), error);
  scores_ = Scratch(std::size_t{c.num_heads} * max_context, error);
  iq_ = Scratch(std::size_t{c.indexer_heads} * c.indexer_head_dim, error);
  ik_ = Scratch(c.indexer_head_dim, error);
  pooled_ = Scratch(c.indexer_head_dim, error);
  block_scores_ = Scratch(max_context / c.compress_ratio + 1, error);
  visible_ = Scratch(max_context, error);
  router_ = Scratch(c.num_experts, error);
  eg_ = Scratch(std::max(c.expert_ff, c.shared_expert_ff), error);
  eu_ = Scratch(std::max(c.expert_ff, c.shared_expert_ff), error);
  ed_ = Scratch(H, error);
  sgate_ = Scratch(1, error);
  logits_ = Scratch(c.vocab_size, error);
  lora_tmp_ = Scratch(max_rank, error);
  lora_out_ = Scratch(std::max<std::size_t>({D, c.SsmConvChannels(), 2ull * c.AttentionQDim(), c.vocab_size}), error);
  linear_.resize(c.num_layers);
  attn_.resize(c.num_layers);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (c.IsLinearLayer(l)) {
      auto& s = linear_[l];
      s.conv = Scratch(std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels(), error);
      s.state = Scratch(std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim, error);
      if (!s.conv || !s.state || !dev_.Fill(s.conv, 0, error) || !dev_.Fill(s.state, 0, error)) return false;
    } else {
      auto& s = attn_[l];
      s.k = Scratch(std::size_t{max_context} * c.AttentionKvDim(), error);
      s.v = Scratch(std::size_t{max_context} * c.AttentionKvDim(), error);
      s.index_k = Scratch(std::size_t{max_context} * c.indexer_head_dim, error);
      s.block_k = Scratch((max_context / c.compress_ratio + 1) * std::size_t{c.indexer_head_dim}, error);
      if (!s.k || !s.v || !s.index_k || !s.block_k) return false;
    }
  }
  return dev_.Synchronize(error);
}

bool FlashNextGpu::GroupedNorm(const rt::Buffer& x, const model::TensorRef& w, const rt::Buffer& y, std::string* error) {
  const auto& c = w_.config;
  return Launch("halo_rmsnorm", {c.hc_count}, {kBlock},
                {rt::Arg::Buf(x), rt::Arg::Buf(tensors_.at(w.data).buf), rt::Arg::Buf(y), rt::Arg::Val(c.hidden_size),
                 rt::Arg::Val(c.rms_eps), rt::Arg::Val(c.hidden_size)},
                error);
}

bool FlashNextGpu::HcMix(const model::HcMixer& m, bool inject, std::string* error) {
  const auto& c = w_.config;
  return GroupedNorm(res_, m.norm, xn_, error) && MatVec(m.down, 0, xn_, lo_, error) &&
         Launch("halo_scale_silu", {Blocks(c.hc_low_rank)}, {kBlock},
                {rt::Arg::Buf(lo_), rt::Arg::Val(1.0F / static_cast<float>(c.hc_count)), rt::Arg::Val(c.hc_low_rank)},
                error) &&
         MatVec(m.up, 0, lo_, gate_hc_, error) &&
         Launch("halo_hc_mix", {Blocks(c.hidden_size)}, {kBlock},
                {rt::Arg::Buf(xn_), rt::Arg::Buf(gate_hc_), rt::Arg::Buf(mixed_), rt::Arg::Val(c.hc_count),
                 rt::Arg::Val(c.hidden_size)},
                error) &&
         (!inject || MatVec(m.inject, 0, xn_, inject_, error));
}

bool FlashNextGpu::Ple(const model::LayerWeights& l, std::int32_t token, std::string* error) {
  const auto& c = w_.config;
  std::vector<std::uint32_t> rows(c.ple_heads);
  model::HashNgramRows(c, ngram_history_, std::span<const std::int32_t>(&token, 1), rows);
  std::vector<float> emb(c.PleEmbeddingDim());
  if (!ngram_ || !ngram_->Read(rows, emb)) {
    if (error) *error = "n-gram table read failed";
    return false;
  }
  const std::uint32_t D = c.HcDim();
  return dev_.Upload(ple_emb_, 0, emb.data(), emb.size() * 4, error) && MatVec(l.ple_key, 0, ple_emb_, ple_key_, error) &&
         MatVec(l.ple_value, 0, ple_emb_, ple_value_, error) && GroupedNorm(ple_key_, l.ple_norm_key, ple_key_, error) &&
         GroupedNorm(res_, l.ple_norm_query, ple_query_, error) &&
         Launch("halo_ple_gate", {c.hc_count}, {kBlock},
                {rt::Arg::Buf(ple_key_), rt::Arg::Buf(ple_query_), rt::Arg::Buf(ple_value_), rt::Arg::Buf(ple_gated_),
                 rt::Arg::Val(c.hidden_size)},
                error) &&
         GroupedNorm(ple_gated_, l.ple_norm_conv, ple_norm_, error) &&
         Launch("halo_ple_conv", {Blocks(D)}, {kBlock},
                {rt::Arg::Buf(ple_norm_), rt::Arg::Buf(ple_gated_), rt::Arg::Buf(ple_history_),
                 rt::Arg::Buf(tensors_.at(l.ple_conv1d.data).buf), rt::Arg::Buf(res_), rt::Arg::Val(D),
                 rt::Arg::Val(c.ple_conv_kernel), rt::Arg::Val(c.ple_ngram_size)},
                error);
}

bool FlashNextGpu::Linear(const model::LayerWeights& l, LinearState& s, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t ch = c.SsmConvChannels(), kd = c.SsmKeyDim(), d = c.ssm_head_dim;
  return MatVec(l.ssm_qkv, 0, mixed_, qkv_, error) && MatVec(l.ssm_gate, 0, mixed_, z_, error) &&
         MatVec(l.ssm_alpha, 0, mixed_, alpha_, error) && MatVec(l.ssm_beta, 0, mixed_, beta_, error) &&
         Launch("halo_conv_step", {Blocks(ch)}, {kBlock},
                {rt::Arg::Buf(qkv_), rt::Arg::Buf(s.conv), rt::Arg::Buf(tensors_.at(l.ssm_conv1d.data).buf),
                 rt::Arg::Buf(conv_out_), rt::Arg::Val(ch), rt::Arg::Val(c.ssm_conv_kernel)},
                error) &&
         Launch("halo_l2norm", {2 * c.ssm_num_k_heads}, {128},
                {rt::Arg::Buf(conv_out_), rt::Arg::Val(d), rt::Arg::Val(c.rms_eps)}, error) &&
         Launch("halo_gdn_step", {c.ssm_num_v_heads}, {d},
                {rt::Arg::Buf(conv_out_), rt::Arg::Buf(conv_out_, kd * 4ull), rt::Arg::Buf(conv_out_, 2ull * kd * 4),
                 rt::Arg::Buf(alpha_), rt::Arg::Buf(beta_), rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf),
                 rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf), rt::Arg::Buf(s.state), rt::Arg::Buf(gdn_out_),
                 rt::Arg::Val(c.ssm_num_k_heads), rt::Arg::Val(d)},
                error) &&
         Launch("halo_gated_norm", {c.ssm_num_v_heads}, {128},
                {rt::Arg::Buf(gdn_out_), rt::Arg::Buf(tensors_.at(l.ssm_norm.data).buf), rt::Arg::Buf(z_),
                 rt::Arg::Val(d), rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(1)},
                error) &&
         MatVec(l.ssm_out, 0, gdn_out_, block_out_, error);
}

bool FlashNextGpu::Attention(const model::LayerWeights& l, AttnState& s, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t nh = c.num_heads, nkv = c.num_kv_heads, hd = c.head_dim, idim = c.indexer_head_dim;
  const auto norm = [&](const rt::Buffer& x, const model::TensorRef& w, std::uint32_t nvec, std::uint32_t dim) {
    return Launch("halo_rmsnorm", {nvec}, {kBlock},
                  {rt::Arg::Buf(x), rt::Arg::Buf(tensors_.at(w.data).buf), rt::Arg::Buf(x), rt::Arg::Val(dim),
                   rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(0)},
                  error);
  };
  const auto rope = [&](const rt::Buffer& x, std::uint32_t heads, std::uint32_t dim, std::uint32_t pos) {
    return Launch("halo_rope_neox", {heads}, {64},
                  {rt::Arg::Buf(x), rt::Arg::Val(dim), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(pos),
                   rt::Arg::Val(c.rope_theta)},
                  error);
  };
  const std::size_t kv_off = std::size_t{position_} * nkv * hd * 4;
  if (!MatVec(l.attn_q, 0, mixed_, qg_, error) || !MatVec(l.attn_k, 0, mixed_, k_, error) ||
      !MatVec(l.attn_v, 0, mixed_, v_, error) ||
      !Launch("halo_split_qgate", {nh}, {kBlock},
              {rt::Arg::Buf(qg_), rt::Arg::Buf(q_), rt::Arg::Buf(gate_), rt::Arg::Val(hd)}, error) ||
      !norm(q_, l.attn_q_norm, nh, hd) || !norm(k_, l.attn_k_norm, nkv, hd) || !rope(q_, nh, hd, position_) ||
      !rope(k_, nkv, hd, position_) ||
      !Launch("halo_copy", {Blocks(nkv * hd)}, {kBlock}, {rt::Arg::Buf(s.k, kv_off), rt::Arg::Buf(k_), rt::Arg::Val(nkv * hd)},
              error) ||
      !Launch("halo_copy", {Blocks(nkv * hd)}, {kBlock}, {rt::Arg::Buf(s.v, kv_off), rt::Arg::Buf(v_), rt::Arg::Val(nkv * hd)},
              error))
    return false;
  // QSA indexer: raw keys cached; complete blocks pooled, normalized, rotated.
  if (!MatVec(l.indexer_q, 0, mixed_, iq_, error) || !MatVec(l.indexer_k, 0, mixed_, ik_, error) ||
      !Launch("halo_copy", {Blocks(idim)}, {kBlock},
              {rt::Arg::Buf(s.index_k, std::size_t{position_} * idim * 4), rt::Arg::Buf(ik_), rt::Arg::Val(idim)},
              error) ||
      !norm(iq_, l.indexer_q_norm, c.indexer_heads, idim) || !rope(iq_, c.indexer_heads, idim, position_))
    return false;
  const std::uint32_t n_kv = position_ + 1, ratio = c.compress_ratio, complete = n_kv / ratio;
  while (s.blocks < complete) {
    const rt::Buffer dst = s.block_k;
    if (!Launch("halo_pool_block", {1}, {128},
                {rt::Arg::Buf(s.index_k), rt::Arg::Buf(pooled_), rt::Arg::Val(s.blocks), rt::Arg::Val(ratio),
                 rt::Arg::Val(idim)},
                error) ||
        !norm(pooled_, l.indexer_k_norm, 1, idim) || !rope(pooled_, 1, idim, s.blocks * ratio) ||
        !Launch("halo_copy", {Blocks(idim)}, {kBlock},
                {rt::Arg::Buf(dst, std::size_t{s.blocks} * idim * 4), rt::Arg::Buf(pooled_), rt::Arg::Val(idim)},
                error))
      return false;
    ++s.blocks;
  }
  std::vector<std::uint32_t> visible;
  const std::uint32_t budget = c.indexer_top_k / ratio;
  if (complete <= budget) {
    visible.resize(n_kv);
    std::iota(visible.begin(), visible.end(), 0U);
  } else {
    std::vector<float> score(complete);
    if (!Launch("halo_qsa_scores", {Blocks(complete)}, {kBlock},
                {rt::Arg::Buf(iq_), rt::Arg::Buf(s.block_k), rt::Arg::Buf(block_scores_), rt::Arg::Val(c.indexer_heads),
                 rt::Arg::Val(idim), rt::Arg::Val(complete)},
                error) ||
        !dev_.Download(block_scores_, 0, score.data(), complete * 4, error))
      return false;
    // The reference's selection: stable sort by score, keep `budget` blocks
    // in position order, then the incomplete tail.
    std::vector<std::uint32_t> order(complete);
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return score[a] > score[b]; });
    order.resize(budget);
    std::sort(order.begin(), order.end());
    for (std::uint32_t b : order)
      for (std::uint32_t t = 0; t < ratio; ++t) visible.push_back(b * ratio + t);
    for (std::uint32_t p = complete * ratio; p < n_kv; ++p) visible.push_back(p);
  }
  return dev_.Upload(visible_, 0, visible.data(), visible.size() * 4, error) &&
         Launch("halo_attention_select", {nh}, {kBlock},
                {rt::Arg::Buf(q_), rt::Arg::Buf(s.k), rt::Arg::Buf(s.v), rt::Arg::Buf(gate_), rt::Arg::Buf(visible_),
                 rt::Arg::Buf(ctx_), rt::Arg::Buf(scores_), rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(hd),
                 rt::Arg::Val(static_cast<std::uint32_t>(visible.size())), rt::Arg::Val(max_context_)},
                error) &&
         MatVec(l.attn_out, 0, ctx_, block_out_, error);
}

bool FlashNextGpu::Moe(const model::LayerWeights& l, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t H = c.hidden_size;
  std::vector<float> logits(c.num_experts);
  if (!MatVec(l.router, 0, mixed_, router_, error) || !dev_.Download(router_, 0, logits.data(), logits.size() * 4, error))
    return false;
  // The reference's routing: softmax over all experts, top-k, renormalize.
  const float mx = *std::max_element(logits.begin(), logits.end());
  double den = 0;
  for (float& v : logits) {
    v = std::exp(v - mx);
    den += v;
  }
  for (float& v : logits) v = static_cast<float>(v / den);
  std::vector<std::uint32_t> order(c.num_experts);
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(order.begin(), order.begin() + c.num_experts_used, order.end(),
                    [&](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b]; });
  float sum = 0;
  for (std::uint32_t i = 0; i < c.num_experts_used; ++i) sum += logits[order[i]];
  sum = std::max(sum, 6.103515625e-5F);
  if (!dev_.Fill(block_out_, 0, error)) return false;
  for (std::uint32_t i = 0; i < c.num_experts_used; ++i) {
    const std::uint32_t e = order[i];
    if (!MatVec(l.ffn_gate_exps, e, mixed_, eg_, error) || !MatVec(l.ffn_up_exps, e, mixed_, eu_, error) ||
        !Launch("halo_silu_mul", {Blocks(c.expert_ff)}, {kBlock},
                {rt::Arg::Buf(eg_), rt::Arg::Buf(eu_), rt::Arg::Val(c.expert_ff)}, error) ||
        !MatVec(l.ffn_down_exps, e, eg_, ed_, error) ||
        !Launch("halo_axpy", {Blocks(H)}, {kBlock},
                {rt::Arg::Buf(block_out_), rt::Arg::Buf(ed_), rt::Arg::Val(logits[e] / sum), rt::Arg::Val(H)}, error))
      return false;
  }
  float sg = 0;
  if (!MatVec(l.shexp_gate, 0, mixed_, eg_, error) || !MatVec(l.shexp_up, 0, mixed_, eu_, error) ||
      !Launch("halo_silu_mul", {Blocks(c.shared_expert_ff)}, {kBlock},
              {rt::Arg::Buf(eg_), rt::Arg::Buf(eu_), rt::Arg::Val(c.shared_expert_ff)}, error) ||
      !MatVec(l.shexp_down, 0, eg_, ed_, error) || !MatVec(l.shexp_gate_inp, 0, mixed_, sgate_, error) ||
      !dev_.Download(sgate_, 0, &sg, 4, error))
    return false;
  return Launch("halo_axpy", {Blocks(H)}, {kBlock},
                {rt::Arg::Buf(block_out_), rt::Arg::Buf(ed_), rt::Arg::Val(1.0F / (1.0F + std::exp(-sg))), rt::Arg::Val(H)},
                error);
}

bool FlashNextGpu::Step(std::int32_t token, std::span<float> logits, std::string* error) {
  const auto& c = w_.config;
  if (position_ >= max_context_ || token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
    if (error) *error = position_ >= max_context_ ? "context is full" : "token out of range";
    return false;
  }
  const auto& emb = tensors_.at(w_.token_embd.data);
  const rt::Arg eargs[] = {rt::Arg::Buf(emb.buf), rt::Arg::Buf(embd_), rt::Arg::Val<std::uint32_t>(token),
                           rt::Arg::Val(c.hidden_size), rt::Arg::Val<std::uint64_t>(w_.token_embd.RowBytes())};
  if (!dev_.Launch(emb.embed, {Blocks(c.hidden_size)}, {kBlock}, eargs, error) ||
      !Launch("halo_broadcast", {Blocks(c.HcDim())}, {kBlock},
              {rt::Arg::Buf(res_), rt::Arg::Buf(embd_), rt::Arg::Val(c.hc_count), rt::Arg::Val(c.hidden_size)}, error))
    return false;
  const auto combine = [&]() {
    return Launch("halo_hc_combine", {Blocks(c.HcDim())}, {kBlock},
                  {rt::Arg::Buf(res_), rt::Arg::Buf(block_out_), rt::Arg::Buf(inject_), rt::Arg::Val(c.hc_count),
                   rt::Arg::Val(c.hidden_size)},
                  error);
  };
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const auto& l = w_.layers[il];
    if (c.IsPleLayer(il) && !Ple(l, token, error)) return false;
    if (!HcMix(l.hc_attn, true, error)) return false;
    if (l.linear ? !Linear(l, linear_[il], error) : !Attention(l, attn_[il], error)) return false;
    if (!combine() || !HcMix(l.hc_ffn, true, error) || !Moe(l, error) || !combine()) return false;
  }
  if (!logits.empty()) {
    if (!HcMix(w_.hc_head, false, error) || !MatVec(w_.output, 0, mixed_, logits_, error) ||
        !dev_.Download(logits_, 0, logits.data(), c.vocab_size * 4, error))
      return false;
  }
  ++position_;
  return dev_.Synchronize(error);
}

template <typename F>
bool FlashNextGpu::ForEachState(F&& f) {
  const auto& c = w_.config;
  const std::size_t pos = position_;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (c.IsLinearLayer(l)) {
      auto& s = linear_[l];
      if (!f(s.conv, s.conv.bytes / 4) || !f(s.state, s.state.bytes / 4)) return false;
    } else {
      auto& s = attn_[l];
      if (!f(s.k, pos * c.AttentionKvDim()) || !f(s.v, pos * c.AttentionKvDim()) ||
          !f(s.index_k, pos * c.indexer_head_dim) || !f(s.block_k, std::size_t{s.blocks} * c.indexer_head_dim))
        return false;
    }
  }
  return f(ple_history_, ple_history_.bytes / 4);
}

bool FlashNextGpu::CopyFloats(const rt::Buffer& dst, const rt::Buffer& src, std::size_t n, std::string* error) {
  return n == 0 || Launch("halo_copy", {Blocks(n)}, {kBlock},
                          {rt::Arg::Buf(dst), rt::Arg::Buf(src), rt::Arg::Val(static_cast<std::uint32_t>(n))}, error);
}

std::unique_ptr<FlashNextGpu::Snapshot> FlashNextGpu::NewSnapshot(std::string* error) {
  auto snap = std::unique_ptr<Snapshot>(new Snapshot(dev_));
  const bool ok = ForEachState([&](const rt::Buffer& live, std::size_t) {
    rt::Buffer b = dev_.Alloc(live.bytes, error);
    if (!b) return false;
    snap->buffers_.push_back(b);
    return true;
  });
  return ok ? std::move(snap) : nullptr;
}

bool FlashNextGpu::Save(Snapshot& snap, std::string* error) {
  std::size_t i = 0;
  if (!ForEachState([&](const rt::Buffer& live, std::size_t n) { return CopyFloats(snap.buffers_[i++], live, n, error); }))
    return false;
  snap.blocks_.clear();
  for (const auto& s : attn_) snap.blocks_.push_back(s.blocks);
  snap.ngram_history_ = ngram_history_;
  snap.position_ = position_;
  snap.valid_ = true;
  return dev_.Synchronize(error);
}

bool FlashNextGpu::Restore(const Snapshot& snap, std::string* error) {
  if (!snap.valid_ || &snap.dev_ != &dev_) {
    if (error) *error = "snapshot is empty or from another device";
    return false;
  }
  // Counts first: ForEachState sizes the KV copies from the restored position.
  position_ = snap.position_;
  for (std::size_t l = 0; l < attn_.size(); ++l) attn_[l].blocks = snap.blocks_[l];
  ngram_history_ = snap.ngram_history_;
  std::size_t i = 0;
  return ForEachState([&](const rt::Buffer& live, std::size_t n) { return CopyFloats(live, snap.buffers_[i++], n, error); }) &&
         dev_.Synchronize(error);
}

bool FlashNextGpu::Reset(std::string* error) {
  const auto& c = w_.config;
  for (std::uint32_t l = 0; l < c.num_layers; ++l)
    if (c.IsLinearLayer(l) && (!dev_.Fill(linear_[l].conv, 0, error) || !dev_.Fill(linear_[l].state, 0, error)))
      return false;
  for (auto& s : attn_) s.blocks = 0;  // KV beyond position is never read
  ngram_history_.Reset();
  position_ = 0;
  return dev_.Fill(ple_history_, 0, error) && dev_.Synchronize(error);
}

bool FlashNextGpu::Prefill(std::span<const std::int32_t> tokens, std::span<float> logits, std::string* error) {
  for (std::size_t i = 0; i < tokens.size(); ++i)
    if (!Step(tokens[i], i + 1 == tokens.size() ? logits : std::span<float>{}, error)) return false;
  return true;
}

}  // namespace halo::engine
