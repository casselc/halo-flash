// P6 batched decode (docs/SESSIONS.md): one step for up to kMaxBatch sessions, each weight read once.
//
// The batched kernels (src/kernels/batch_decode.hip) compute every column with the single-token kernel's
// arithmetic, so each session's result is byte-identical to stepping it alone. Work that is per session
// (the DeltaNet recurrence, attention over its own KV, norms and combines, routing, sampling) runs the
// single-token kernels once per entry, on slices of the batch buffers: BatchEntryScope points the engine's
// per-token scratch members at entry b's slice and binds b's session, so the single path's helpers
// (Attention, Ple, the sampling launches) record exactly the launches they record for one session.
// A captured batch graph bakes the sessions' state addresses, so graphs are cached per session list.

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "engine/flash_next_gpu.hpp"
#include "kernels/sample_params.h"
#include "kernels/step_params.h"

namespace halo::engine {

namespace {
constexpr std::uint32_t kBlock = 256;
std::uint32_t Blocks(std::size_t n, std::uint32_t per = kBlock) { return static_cast<std::uint32_t>((n + per - 1) / per); }
rt::Buffer View(const rt::Buffer& b, std::size_t offset_bytes, std::size_t bytes) {
  return rt::Buffer{static_cast<char*>(b.device) + offset_bytes, bytes, nullptr};
}
constexpr unsigned kBatchDecodeOpts =
    FlashNextGpu::kDecodeQ5kExperts | FlashNextGpu::kDecodeIndexerDual | FlashNextGpu::kDecodeHcMixLds |
    FlashNextGpu::kDecodeQsaScores | FlashNextGpu::kDecodeShexpCombine | FlashNextGpu::kDecodeQ8ActOnce |
    FlashNextGpu::kDecodeHoistRowLoads | FlashNextGpu::kDecodeHoistSmall | FlashNextGpu::kDecodeGdnNoSpill |
    FlashNextGpu::kDecodeTopkWarp;
constexpr std::size_t kQ8ActFloats = 1280;  // struct Q8Act in core.hip
}  // namespace

bool FlashNextGpu::BatchSupported() const { return BatchUnsupported().empty(); }

// The layers whose routed experts take the grouped batched kernels: Moe()'s fast path with the activations
// quantized by the router launch (Q4_K gate / up, Q5_1 or repacked Q8 down).
bool FlashNextGpu::BatchedMoe(const model::LayerWeights& l) const {
  const auto& g = l.ffn_gate_exps;
  const auto& dn = l.ffn_down_exps;
  const auto it = tensors_.find(dn.data);
  const bool down_q8r = it != tensors_.end() && static_cast<bool>(it->second.qs) && dn.cols <= 1024;
  const bool down_q5_1 = dn.type == core::GgmlType::kQ5_1 && dn.cols % 32 == 0 && dn.cols <= 1024;
  return g.type == core::GgmlType::kQ4_K && l.ffn_up_exps.type == g.type && l.ffn_up_exps.rows == g.rows &&
         l.ffn_up_exps.cols == g.cols && g.cols % 256 == 0 && g.cols <= 4096 && (down_q5_1 || down_q8r) &&
         dn.cols == g.rows && l.router.type == core::GgmlType::kF32 && l.router.rows <= 1024 &&
         !lora_.contains(l.router.data) && w_.config.hidden_size % 32 == 0;
}

std::string FlashNextGpu::BatchUnsupported() const {
  const auto& c = w_.config;
  if (!batch_decode_) return "batch_decode_ off";
  if (dev_.backend() != rt::Backend::kHip) return "backend";
  if (!lora_.empty()) return "LoRA";
  if (!(fused_hc_ && fused_linear_ && fused_moe_ && fused_shexp_ && int8_act_moe_ && small_row_kernels_ &&
        decode_attn_batched_ && repack_q8_ && block_gemv_))
    return "decode options";
  if ((decode_opt_ & kBatchDecodeOpts) != kBatchDecodeOpts) return "decode_opt bits";
  const std::uint32_t H = c.hidden_size, D = c.HcDim();
  const std::uint32_t chunk = (D / kHcParts + 511) / 512 * 512, parts = (D + chunk - 1) / chunk;
  if (H > 4096 || H % 32 || chunk > 1536 || c.hc_count > 4 || c.hc_low_rank > 512 || c.hc_low_rank % 32 ||
      parts > 16 || c.expert_ff > 1024 || c.num_experts_used > 16 || kMaxBatch * c.num_experts_used > 128 ||
      c.ssm_head_dim != 128 || c.num_experts > 1024)
    return "shapes";
  const auto q8 = [&](const model::TensorRef& t) {
    auto it = tensors_.find(t.data);
    return it != tensors_.end() && static_cast<bool>(it->second.qs);
  };
  const auto mixer = [&](const model::HcMixer& m, bool inject) {
    return q8(m.down) && q8(m.up) && (!inject || (m.inject.type == core::GgmlType::kF32 && m.inject.rows <= 16));
  };
  if (!q8(w_.output)) return "output not repacked Q8";
  if (!mixer(w_.hc_head, false)) return "head mixer";
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const auto& l = w_.layers[il];
    const std::string at = " (layer " + std::to_string(il) + ")";
    if (!mixer(l.hc_attn, true) || !mixer(l.hc_ffn, true)) return "mixer" + at;
    if (l.linear) {
      if (!q8(l.ssm_qkv) || !q8(l.ssm_gate) || !q8(l.ssm_out) || l.ssm_alpha.type != core::GgmlType::kF32 ||
          l.ssm_beta.type != core::GgmlType::kF32)
        return "DeltaNet projections" + at;
    } else if (!q8(l.attn_q) || !q8(l.attn_k) || !q8(l.attn_v) || !q8(l.attn_out)) {
      return "attention projections" + at;
    }
    if (l.router.type != core::GgmlType::kF32 || l.router.rows > 1024) return "router" + at;
    if (!q8(l.shexp_gate) || !q8(l.shexp_up) || !q8(l.shexp_down) || l.shexp_gate_inp.type != core::GgmlType::kF32)
      return "shared expert" + at;
    (void)0;  // routed experts: batched when BatchedMoe(l), else the single path's Moe() per entry
  }
  return {};
}

bool FlashNextGpu::AllocBatch(std::string* error) {
  if (bb_ready_) return true;
  const auto& c = w_.config;
  const std::size_t B = kMaxBatch, H = c.hidden_size, D = c.HcDim(), hc = c.hc_count, k = c.num_experts_used;
  const std::size_t sff = std::max(c.expert_ff, c.shared_expert_ff);
  auto& b = bb_;
  const auto a = [&](rt::Buffer& buf, std::size_t floats) {
    buf = Scratch(B * std::max<std::size_t>(floats, 1), error);
    return static_cast<bool>(buf);
  };
  bb_.entry_stride = (step_host_.size() + 255) / 256 * 256;
  bb_.host.assign(B * bb_.entry_stride, 0);
  bb_.params = dev_.Alloc(bb_.host.size(), error);
  if (!bb_.params) return false;
  owned_.push_back(bb_.params);
  const bool ok =
      a(b.res, D) && a(b.xn, D) && a(b.mixed, H) && a(b.block_out, H) && a(b.embd, H) && a(b.inject, hc) &&
      a(b.scale, hc) && a(b.lo_part, std::size_t{kHcParts + 1} * c.hc_low_rank) &&
      a(b.inject_part, std::size_t{kHcParts + 1} * hc) && a(b.qkv, c.SsmConvChannels()) &&
      a(b.conv_out, c.SsmConvChannels()) && a(b.z, c.SsmValueDim()) && a(b.alpha, c.ssm_num_v_heads) &&
      a(b.beta, c.ssm_num_v_heads) && a(b.gdn_out, c.SsmValueDim()) && a(b.qg, 2ull * c.AttentionQDim()) &&
      a(b.q, c.AttentionQDim()) && a(b.gate, c.AttentionQDim()) && a(b.k, c.AttentionKvDim()) &&
      a(b.v, c.AttentionKvDim()) && a(b.ctx, c.AttentionQDim()) &&
      a(b.iq, std::size_t{c.indexer_heads} * c.indexer_head_dim) && a(b.ik, c.indexer_head_dim) &&
      a(b.pooled, c.indexer_head_dim) && a(b.router, c.num_experts) && a(b.ids, k) && a(b.wts, k) &&
      a(b.q8act, kQ8ActFloats) && a(b.ucount, 1) && a(b.uexp, k) && a(b.unum, k) && a(b.upair, k * 8) &&
      a(b.moe_h, k * c.expert_ff) && a(b.moe_part, k * H) && a(b.eg, sff) && a(b.sgate, 1) &&
      a(b.logits, c.vocab_size) && a(b.sample_out, (sizeof(HaloSampleOut) + 3) / 4) &&
      a(b.xq, (std::max<std::size_t>({H, c.SsmValueDim(), c.AttentionQDim()}) + 3) / 4) &&
      a(b.xs, std::max<std::size_t>({H, c.SsmValueDim(), c.AttentionQDim()}) / 32) &&
      a(b.xq2, (std::max<std::size_t>({H, c.SsmValueDim(), c.AttentionQDim()}) + 3) / 4) &&
      a(b.xs2, std::max<std::size_t>({H, c.SsmValueDim(), c.AttentionQDim()}) / 32);
  if (!ok) return false;
  bb_ready_ = true;
  return true;
}

void FlashNextGpu::BatchEntryScope(std::uint32_t e, std::span<const SessionId> sessions, bool enter) {
  const auto& c = w_.config;
  const std::size_t H = c.hidden_size, D = c.HcDim(), hc = c.hc_count, k = c.num_experts_used;
  const std::size_t sff = std::max(c.expert_ff, c.shared_expert_ff);
  struct Slot {
    rt::Buffer* member;
    const rt::Buffer* batch;
    std::size_t floats;
  };
  const Slot slots[] = {
      {&res_, &bb_.res, D}, {&xn_, &bb_.xn, D}, {&mixed_, &bb_.mixed, H}, {&block_out_, &bb_.block_out, H},
      {&embd_, &bb_.embd, H}, {&inject_, &bb_.inject, hc}, {&hc_scale_, &bb_.scale, hc},
      {&qkv_, &bb_.qkv, c.SsmConvChannels()}, {&conv_out_, &bb_.conv_out, c.SsmConvChannels()},
      {&z_, &bb_.z, c.SsmValueDim()}, {&alpha_, &bb_.alpha, c.ssm_num_v_heads}, {&beta_, &bb_.beta, c.ssm_num_v_heads},
      {&gdn_out_, &bb_.gdn_out, c.SsmValueDim()}, {&qg_, &bb_.qg, 2ull * c.AttentionQDim()},
      {&q_, &bb_.q, c.AttentionQDim()}, {&gate_, &bb_.gate, c.AttentionQDim()}, {&k_, &bb_.k, c.AttentionKvDim()},
      {&v_, &bb_.v, c.AttentionKvDim()}, {&ctx_, &bb_.ctx, c.AttentionQDim()},
      {&iq_, &bb_.iq, std::size_t{c.indexer_heads} * c.indexer_head_dim}, {&ik_, &bb_.ik, c.indexer_head_dim},
      {&pooled_, &bb_.pooled, c.indexer_head_dim}, {&router_, &bb_.router, c.num_experts},
      {&expert_ids_, &bb_.ids, k}, {&expert_weights_, &bb_.wts, k}, {&q8act_, &bb_.q8act, kQ8ActFloats},
      {&moe_h_, &bb_.moe_h, k * c.expert_ff}, {&moe_part_, &bb_.moe_part, k * H}, {&eg_, &bb_.eg, sff},
      {&sgate_, &bb_.sgate, 1}, {&logits_, &bb_.logits, c.vocab_size},
      {&sample_out_, &bb_.sample_out, (sizeof(HaloSampleOut) + 3) / 4}};
  static std::vector<rt::Buffer> saved;
  if (enter) {
    saved.clear();
    for (const auto& s : slots) {
      saved.push_back(*s.member);
      *s.member = View(*s.batch, e * s.floats * 4, s.floats * 4);
    }
    saved.push_back(ple_emb_);
    ple_emb_ = View(bb_.params, e * bb_.entry_stride, bb_.entry_stride);
    (void)BindSession(sessions[e], nullptr);
  } else {
    std::size_t i = 0;
    for (const auto& s : slots) *s.member = saved[i++];
    ple_emb_ = saved[i];
  }
}

namespace {
// Columns per block for B entries: the largest divisor of B not above `target` (env HALO_BATCH_G_<kind>).
std::uint32_t GroupCols(std::uint32_t B, std::uint32_t target) {
  for (std::uint32_t g = std::min(B, std::max(target, 1u)); g > 1; --g)
    if (B % g == 0) return g;
  return 1;
}
std::uint32_t EnvOr(const char* name, std::uint32_t fallback) {
  const char* v = std::getenv(name);
  return v ? static_cast<std::uint32_t>(std::atoi(v)) : fallback;
}
}  // namespace

bool FlashNextGpu::RecordBatch(std::span<const SessionId> sessions, Head head, std::string* error) {
  const auto& c = w_.config;
  const auto B = static_cast<std::uint32_t>(sessions.size());
  const std::string nb = "_b" + std::to_string(B);  // routed experts: up to B pairs an expert
  static const std::uint32_t g_gemv = EnvOr("HALO_BATCH_G_GEMV", 4), g_mix = EnvOr("HALO_BATCH_G_MIX", 1),
                             g_small = EnvOr("HALO_BATCH_G_SMALL", 4);
  const std::uint32_t G = GroupCols(B, g_gemv), NG = B / G, GM = GroupCols(B, g_mix), NGM = B / GM;
  const std::uint32_t GS = GroupCols(B, g_small), NGS = B / GS;
  static const std::string var = [] {
    const char* v = std::getenv("HALO_BATCH_GEMV_VAR");
    return std::string(v ? v : "");
  }();
  const std::string ng = "_b" + std::to_string(G), ngm = "_b" + std::to_string(GM), ngs = "_b" + std::to_string(GS);
  const std::uint32_t H = c.hidden_size, D = c.HcDim(), hc = c.hc_count, low = c.hc_low_rank, k = c.num_experts_used;
  const std::uint32_t chunk = (D / kHcParts + 511) / 512 * 512, parts = (D + chunk - 1) / chunk;
  const SessionId bound = bound_;
  // Per entry: runs f() with the entry's scratch slices and session bound.
  const auto each = [&](auto&& f) {
    for (std::uint32_t e = 0; e < B; ++e) {
      BatchEntryScope(e, sessions, true);
      const bool ok = f(e);
      BatchEntryScope(e, sessions, false);
      if (!ok) return false;
    }
    return true;
  };
  // Relaxed (B >= 2, not a verify): the dense projections' inputs quantized once per call, then the int8-dot kernels.
  const bool relaxed = batch_relaxed_ && !verify_ && B >= 2;
  // A wave's rows and the column groups: one row a wave, 4 columns a group (the B = 8 sweep, streams/BD/sweep.txt:
  // best for every dense shape); HALO_RELAXED_ROWS / HALO_RELAXED_G override for sweeps.
  static const std::uint32_t env_rows = EnvOr("HALO_RELAXED_ROWS", 1), env_g = EnvOr("HALO_RELAXED_G", 4);
  const std::uint32_t relaxed_rows = env_rows, relaxed_g = GroupCols(B, env_g);
  // Activations as two int8 levels (x ~ q s + q2 s2), a second v_dot4 chain per column: the one-level arithmetic
  // cost 0.27 pp of API agreement, this none, for 2 ms at B = 8 (streams/BD). HALO_RELAXED_RESID=0: one level.
  static const bool resid = EnvOr("HALO_RELAXED_RESID", 1) != 0;
  const std::string i8 = resid ? "halo_gemv_q8r_i8x" : "halo_gemv_q8r_i8";
  const rt::Buffer none{};
  const auto& xq2 = resid ? bb_.xq2 : none;
  const auto& xs2 = resid ? bb_.xs2 : none;
  const auto quantize = [&](const rt::Buffer& x, std::uint32_t xstride, std::uint32_t cols) {
    return Launch("halo_q8x_cols", {(cols / 32 + 7) / 8, B}, {256},
                  {rt::Arg::Buf(x), rt::Arg::Val(xstride), rt::Arg::Val(cols), rt::Arg::Buf(bb_.xq), rt::Arg::Buf(bb_.xs),
                   rt::Arg::Buf(xq2), rt::Arg::Buf(xs2)},
                  error);
  };
  const auto gemv = [&](const model::TensorRef& t, const rt::Buffer& x, std::uint32_t xstride, const rt::Buffer& y,
                        std::uint32_t ystride) {
    const auto& d = tensors_.at(t.data);
    const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
    const bool r2 = rows < 2048;
    // HALO_RELAXED_HEAD=0 (ablation): the LM head on the exact kernels.
    static const bool relaxed_head = EnvOr("HALO_RELAXED_HEAD", 1) != 0;
    if (relaxed && (relaxed_head || &t != &w_.output)) {
      const std::uint32_t wr = relaxed_rows, gr = relaxed_g, ngr = B / gr;
      return quantize(x, xstride, cols) &&
             Launch((i8 + "_r" + std::to_string(wr) + "_b" + std::to_string(gr)).c_str(),
                    {Blocks(rows, 8 * wr) * ngr}, {256},
                    {rt::Arg::Buf(d.qs), rt::Arg::Buf(d.d), rt::Arg::Buf(bb_.xq), rt::Arg::Buf(bb_.xs), rt::Arg::Buf(xq2),
                     rt::Arg::Buf(xs2), rt::Arg::Buf(y),
                     rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val(d.qstride), rt::Arg::Val(ystride),
                     rt::Arg::Val(ngr)},
                    error);
    }
    return Launch(((r2 ? "halo_gemv_q8r_r2" : "halo_gemv_q8r_r4") + ng + var).c_str(), {Blocks(rows, r2 ? 16 : 32) * NG},
                  {256},
                  {rt::Arg::Buf(d.qs), rt::Arg::Buf(d.d), rt::Arg::Buf(x), rt::Arg::Buf(y), rt::Arg::Val(rows),
                   rt::Arg::Val(cols), rt::Arg::Val(d.qstride), rt::Arg::Val(xstride), rt::Arg::Val(ystride),
                   rt::Arg::Val(NG)},
                  error);
  };
  const auto mix = [&](const model::HcMixer& m, bool inject) {
    const auto& dn = tensors_.at(m.down.data);
    const auto& upt = tensors_.at(m.up.data);
    const std::uint32_t down_blocks = Blocks(low, 16), inject_rows = inject ? hc : 0;
    return Launch("halo_hc_norm_scales_u_y", {hc, B}, {256},
                  {rt::Arg::Buf(bb_.res), rt::Arg::Buf(bb_.scale), rt::Arg::Val(c.hidden_size), rt::Arg::Val(c.rms_eps),
                   rt::Arg::Val(hc)},
                  error) &&
           Launch(("halo_hc_mix_a_cu" + ngm).c_str(), {(down_blocks + inject_rows) * NGM, parts}, {256},
                  {rt::Arg::Buf(bb_.res), rt::Arg::Buf(tensors_.at(m.norm.data).buf), rt::Arg::Buf(bb_.scale),
                   rt::Arg::Buf(dn.qs), rt::Arg::Buf(dn.d),
                   rt::Arg::Buf(inject ? tensors_.at(m.inject.data).buf : bb_.res), rt::Arg::Buf(bb_.xn),
                   rt::Arg::Buf(bb_.lo_part), rt::Arg::Buf(bb_.inject_part), rt::Arg::Val(hc), rt::Arg::Val(H),
                   rt::Arg::Val(low), rt::Arg::Val(down_blocks), rt::Arg::Val(dn.qstride), rt::Arg::Val(chunk),
                   rt::Arg::Val(parts), rt::Arg::Val(NGM)},
                  error) &&
           Launch(("halo_hc_mix_b_u" + ngm).c_str(), {Blocks(H, 8) * NGM}, {256},
                  {rt::Arg::Buf(bb_.lo_part), rt::Arg::Val(parts), rt::Arg::Buf(bb_.inject_part),
                   rt::Arg::Buf(bb_.inject), rt::Arg::Val(inject_rows), rt::Arg::Buf(upt.qs), rt::Arg::Buf(upt.d),
                   rt::Arg::Buf(bb_.xn), rt::Arg::Buf(bb_.mixed), rt::Arg::Val(hc), rt::Arg::Val(H), rt::Arg::Val(low),
                   rt::Arg::Val(upt.qstride), rt::Arg::Val(NGM)},
                  error);
  };
  const auto combine = [&]() {
    return Launch("halo_hc_combine_y", {Blocks(D), B}, {kBlock},
                  {rt::Arg::Buf(bb_.res), rt::Arg::Buf(bb_.block_out), rt::Arg::Buf(bb_.inject), rt::Arg::Val(hc),
                   rt::Arg::Val(H)},
                  error);
  };
  const auto& emb = tensors_.at(w_.token_embd.data);
  bool ok = each([&](std::uint32_t) {
    const rt::Arg eargs[] = {rt::Arg::Buf(emb.buf), rt::Arg::Buf(embd_), rt::Arg::Buf(ple_emb_, params_offset_),
                             rt::Arg::Val(c.hidden_size), rt::Arg::Val<std::uint64_t>(w_.token_embd.RowBytes())};
    return dev_.Launch(emb.embed, {Blocks(c.hidden_size)}, {kBlock}, eargs, error) &&
           Launch("halo_broadcast", {Blocks(D)}, {kBlock},
                  {rt::Arg::Buf(res_), rt::Arg::Buf(embd_), rt::Arg::Val(hc), rt::Arg::Val(H)}, error);
  });
  for (std::uint32_t il = 0; ok && il < c.num_layers; ++il) {
    const auto& l = w_.layers[il];
    cur_layer_ = il;
    if (c.IsPleLayer(il) && !each([&](std::uint32_t e) {
          return Ple(l, error) && (!verify_ || e + 1 >= B ||
                                   CopyFloats(verify_snap_[e][2 * c.num_layers], ple_history_, ple_history_.bytes / 4, error));
        }))
      return false;
    if (!mix(l.hc_attn, true)) return false;
    if (l.linear) {
      const auto& tq = tensors_.at(l.ssm_qkv.data);
      const auto& tz = tensors_.at(l.ssm_gate.data);
      const auto rq = static_cast<std::uint32_t>(l.ssm_qkv.rows), rz = static_cast<std::uint32_t>(l.ssm_gate.rows);
      const std::uint32_t b1 = Blocks(rq, 32), ch = c.SsmConvChannels(), kd = c.SsmKeyDim();
      ok = (relaxed
               ? quantize(bb_.mixed, H, H) &&
                     Launch((i8 + "_dual_r" + std::to_string(relaxed_rows) + "_b" + std::to_string(relaxed_g))
                                .c_str(),
                            {(Blocks(rq, 8 * relaxed_rows) + Blocks(rz, 8 * relaxed_rows)) * (B / relaxed_g)}, {256},
                            {rt::Arg::Buf(tq.qs), rt::Arg::Buf(tq.d), rt::Arg::Buf(bb_.qkv), rt::Arg::Val(rq),
                             rt::Arg::Val(tq.qstride), rt::Arg::Buf(tz.qs), rt::Arg::Buf(tz.d), rt::Arg::Buf(bb_.z),
                             rt::Arg::Val(rz), rt::Arg::Val(tz.qstride), rt::Arg::Buf(bb_.xq), rt::Arg::Buf(bb_.xs),
                             rt::Arg::Buf(xq2), rt::Arg::Buf(xs2),
                             rt::Arg::Val(H), rt::Arg::Val(Blocks(rq, 8 * relaxed_rows)), rt::Arg::Val(ch),
                             rt::Arg::Val(c.SsmValueDim()), rt::Arg::Val(B / relaxed_g)},
                            error)
               : Launch(("halo_gemv_q8r_dual" + ng + var).c_str(), {(b1 + Blocks(rz, 32)) * NG}, {256},
                  {rt::Arg::Buf(tq.qs), rt::Arg::Buf(tq.d), rt::Arg::Buf(bb_.qkv), rt::Arg::Val(rq),
                   rt::Arg::Val(tq.qstride), rt::Arg::Buf(tz.qs), rt::Arg::Buf(tz.d), rt::Arg::Buf(bb_.z),
                   rt::Arg::Val(rz), rt::Arg::Val(tz.qstride), rt::Arg::Buf(bb_.mixed), rt::Arg::Val(H),
                   rt::Arg::Val(b1), rt::Arg::Val(H), rt::Arg::Val(ch), rt::Arg::Val(c.SsmValueDim()), rt::Arg::Val(NG)},
                  error)) &&
           Launch(("halo_matvec_f32_rowblock2_u" + ngs).c_str(), {2 * c.ssm_num_v_heads * NGS}, {256},
                  {rt::Arg::Buf(tensors_.at(l.ssm_alpha.data).buf), rt::Arg::Buf(tensors_.at(l.ssm_beta.data).buf),
                   rt::Arg::Buf(bb_.mixed), rt::Arg::Buf(bb_.alpha), rt::Arg::Buf(bb_.beta),
                   rt::Arg::Val(c.ssm_num_v_heads), rt::Arg::Val(H), rt::Arg::Val(H), rt::Arg::Val(c.ssm_num_v_heads),
                   rt::Arg::Val(NGS)},
                  error) &&
           each([&](std::uint32_t e) {
             auto& s = linear_[il];
             // Verify rows: the state after row e (all rows but the last) for rollback.
             const auto snap = [&]() {
               if (!verify_ || e + 1 >= B) return true;
               const auto& v = verify_snap_[e];
               return CopyFloats(v[2 * il], s.state, s.state.bytes / 4, error) &&
                      CopyFloats(v[2 * il + 1], s.conv, s.conv.bytes / 4, error);
             };
             return Launch("halo_conv_l2norm", {ch / 128}, {128},
                           {rt::Arg::Buf(qkv_), rt::Arg::Buf(s.conv), rt::Arg::Buf(tensors_.at(l.ssm_conv1d.data).buf),
                            rt::Arg::Buf(conv_out_), rt::Arg::Val(ch), rt::Arg::Val(c.ssm_conv_kernel),
                            rt::Arg::Val(2 * kd), rt::Arg::Val(c.rms_eps)},
                           error) &&
                    Launch("halo_gdn_step_d128_norm_r", {c.ssm_num_v_heads}, {128},
                           {rt::Arg::Buf(conv_out_), rt::Arg::Buf(conv_out_, kd * 4ull),
                            rt::Arg::Buf(conv_out_, 2ull * kd * 4), rt::Arg::Buf(alpha_), rt::Arg::Buf(beta_),
                            rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf), rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf),
                            rt::Arg::Buf(s.state), rt::Arg::Buf(tensors_.at(l.ssm_norm.data).buf), rt::Arg::Buf(z_),
                            rt::Arg::Buf(gdn_out_), rt::Arg::Val(c.ssm_num_k_heads), rt::Arg::Val(c.rms_eps),
                            rt::Arg::Val<std::uint32_t>(1)},
                           error) &&
                    snap();
           }) &&
           gemv(l.ssm_out, bb_.gdn_out, c.SsmValueDim(), bb_.block_out, H);
    } else {
      const std::uint32_t qd = c.AttentionQDim(), kvd = c.AttentionKvDim();
      ok = gemv(l.attn_q, bb_.mixed, H, bb_.qg, 2 * qd) && gemv(l.attn_k, bb_.mixed, H, bb_.k, kvd) &&
           gemv(l.attn_v, bb_.mixed, H, bb_.v, kvd) && each([&](std::uint32_t) {
             skip_attn_proj_ = true;
             const bool r = Attention(l, attn_[il], error);
             skip_attn_proj_ = false;
             return r;
           }) &&
           gemv(l.attn_out, bb_.ctx, qd, bb_.block_out, H);
    }
    if (!ok || !combine() || !mix(l.hc_ffn, true)) return false;
    // Routed + shared experts.
    if (!BatchedMoe(l)) {  // the single path, entry by entry
      if (!each([&](std::uint32_t) { return Moe(l, error); }) || !combine()) return false;
      continue;
    }
    const auto& gate = l.ffn_gate_exps;
    const auto& up = l.ffn_up_exps;
    const auto& down = l.ffn_down_exps;
    const auto& dd = tensors_.at(down.data);
    const auto ff = static_cast<std::uint32_t>(gate.rows), E = c.num_experts;
    const auto rrows = static_cast<std::uint32_t>(l.router.rows);
    const bool down_q5_1 = down.type == core::GgmlType::kQ5_1;
    ok = Launch(("halo_router_rowblock_q8act" + ngs).c_str(), {std::max(rrows, H / 32) * NGS}, {256},
                {rt::Arg::Buf(tensors_.at(l.router.data).buf), rt::Arg::Buf(bb_.mixed), rt::Arg::Buf(bb_.router),
                 rt::Arg::Val(rrows), rt::Arg::Val(H), rt::Arg::Buf(bb_.q8act), rt::Arg::Val(H), rt::Arg::Val(rrows),
                 rt::Arg::Val(NGS)},
                error) &&
         (E <= 1024 && k <= 16
              ? Launch("halo_router_topk_w_y", {1, B}, {kBlock},
                       {rt::Arg::Buf(bb_.router), rt::Arg::Buf(bb_.ids), rt::Arg::Buf(bb_.wts), rt::Arg::Val(E),
                        rt::Arg::Val(k), rt::Arg::Val(rrows)},
                       error)
              : each([&](std::uint32_t) {
                  return Launch("halo_router_topk", {1}, {kBlock},
                                {rt::Arg::Buf(router_), rt::Arg::Buf(expert_ids_), rt::Arg::Buf(expert_weights_),
                                 rt::Arg::Val(E), rt::Arg::Val(k)},
                                error);
                })) &&
         Launch("halo_moe_bucket_b", {1}, {128},
                {rt::Arg::Buf(bb_.ids), rt::Arg::Val(B * k), rt::Arg::Buf(bb_.ucount), rt::Arg::Buf(bb_.uexp),
                 rt::Arg::Buf(bb_.unum), rt::Arg::Buf(bb_.upair)},
                error) &&
         Launch(("halo_moe_gate_up_q4_k_dot_g" + nb).c_str(), {Blocks(ff, 16), B * k}, {256},
                {rt::Arg::Buf(tensors_.at(gate.data).buf), rt::Arg::Buf(tensors_.at(up.data).buf),
                 rt::Arg::Buf(bb_.ucount), rt::Arg::Buf(bb_.uexp), rt::Arg::Buf(bb_.unum), rt::Arg::Buf(bb_.upair),
                 rt::Arg::Buf(bb_.q8act), rt::Arg::Buf(bb_.moe_h),
                 rt::Arg::Val<std::uint64_t>(gate.RowBytes() * gate.rows), rt::Arg::Val(ff), rt::Arg::Val(H),
                 rt::Arg::Val<std::uint64_t>(gate.RowBytes()), rt::Arg::Val(k)},
                error) &&
         (down_q5_1 ? Launch(("halo_moe_down_q5_1_r4" + nb).c_str(), {Blocks(H, 32), B * k}, {256},
                             {rt::Arg::Buf(dd.buf), rt::Arg::Buf(bb_.ucount), rt::Arg::Buf(bb_.uexp),
                              rt::Arg::Buf(bb_.unum), rt::Arg::Buf(bb_.upair), rt::Arg::Buf(bb_.moe_h),
                              rt::Arg::Buf(bb_.moe_part), rt::Arg::Val<std::uint64_t>(down.RowBytes() * down.rows),
                              rt::Arg::Val(H), rt::Arg::Val(ff), rt::Arg::Val<std::uint64_t>(down.RowBytes())},
                             error)
                    : Launch(("halo_moe_down_q8r_r4" + nb).c_str(), {Blocks(H, 32), B * k}, {256},
                             {rt::Arg::Buf(dd.qs), rt::Arg::Buf(dd.d), rt::Arg::Buf(bb_.ucount),
                              rt::Arg::Buf(bb_.uexp), rt::Arg::Buf(bb_.unum), rt::Arg::Buf(bb_.upair),
                              rt::Arg::Buf(bb_.moe_h), rt::Arg::Buf(bb_.moe_part), rt::Arg::Val(H), rt::Arg::Val(ff),
                              rt::Arg::Val(dd.qstride)},
                             error));
    const auto& sg = tensors_.at(l.shexp_gate.data);
    const auto& su = tensors_.at(l.shexp_up.data);
    const auto& sd = tensors_.at(l.shexp_down.data);
    const std::uint32_t sff = c.shared_expert_ff, sffs = std::max(c.expert_ff, c.shared_expert_ff);
    ok = ok &&
         Launch(("halo_shexp_gate_up_u" + ng).c_str(), {Blocks(sff, 16) * NG}, {256},
                {rt::Arg::Buf(sg.qs), rt::Arg::Buf(sg.d), rt::Arg::Buf(su.qs), rt::Arg::Buf(su.d),
                 rt::Arg::Buf(tensors_.at(l.shexp_gate_inp.data).buf), rt::Arg::Buf(bb_.mixed), rt::Arg::Buf(bb_.eg),
                 rt::Arg::Buf(bb_.sgate), rt::Arg::Val(sff), rt::Arg::Val(H), rt::Arg::Val(sg.qstride), rt::Arg::Val(H),
                 rt::Arg::Val(sffs), rt::Arg::Val(NG)},
                error) &&
         Launch(("halo_shexp_down_combine_u" + ng).c_str(), {Blocks(H, 32) * NG}, {256},
                {rt::Arg::Buf(sd.qs), rt::Arg::Buf(sd.d), rt::Arg::Buf(bb_.eg), rt::Arg::Buf(bb_.sgate),
                 rt::Arg::Buf(bb_.moe_part), rt::Arg::Buf(bb_.wts), rt::Arg::Val(k), rt::Arg::Buf(bb_.block_out),
                 rt::Arg::Val(H), rt::Arg::Val(sff), rt::Arg::Val(sd.qstride), rt::Arg::Val(sffs), rt::Arg::Val(H),
                 rt::Arg::Val(NG)},
                error) &&
         combine();
  }
  if (ok && head != Head::kNone) {
    ok = mix(w_.hc_head, false) && gemv(w_.output, bb_.mixed, H, bb_.logits, c.vocab_size);
    if (ok && head == Head::kSample)
      ok = each([&](std::uint32_t) {
        const rt::Arg sp = rt::Arg::Buf(ple_emb_, sample_offset_), pen = rt::Arg::Buf(ple_emb_, penalty_offset_);
        return Launch("halo_sample_blocks", {sample_blocks_}, {1024},
                      {rt::Arg::Buf(logits_), sp, pen, rt::Arg::Buf(sample_block_out_), rt::Arg::Buf(sample_candidates_)},
                      error) &&
               Launch("halo_sample_final", {1}, {1024},
                      {sp, rt::Arg::Buf(sample_block_out_), rt::Arg::Buf(sample_candidates_), rt::Arg::Val(sample_blocks_),
                       rt::Arg::Buf(sample_key_), rt::Arg::Buf(sample_rid_), rt::Arg::Buf(sample_w_),
                       rt::Arg::Buf(sample_out_)},
                      error) &&
               Launch("halo_sample", {1}, {1024},
                      {rt::Arg::Buf(logits_), sp, pen, rt::Arg::Buf(sample_key_), rt::Arg::Buf(sample_rid_),
                       rt::Arg::Buf(sample_w_), rt::Arg::Buf(sample_out_)},
                      error);
      });
  }
  (void)BindSession(bound, nullptr);
  return ok;
}

bool FlashNextGpu::StepBatch(std::span<const BatchEntry> entries, std::string* error) {
  if (poisoned_) return PoisonedFail(error);
  const auto& c = w_.config;
  const auto fail = [&](const char* what) {
    if (error) *error = what;
    return false;
  };
  const auto B = static_cast<std::uint32_t>(entries.size());
  if (B == 0 || B > kMaxBatch) return fail("StepBatch takes 1..kMaxBatch entries");
  const bool sample = entries[0].sample != nullptr, want_logits = !entries[0].logits.empty();
  for (std::uint32_t i = 0; i < B; ++i) {
    const auto& e = entries[i];
    if (!HasSession(e.session)) return fail("no such session");
    for (std::uint32_t j = 0; j < i; ++j)
      if (entries[j].session == e.session) return fail("StepBatch: a session appears twice");
    if ((e.sample != nullptr) != sample || (!e.logits.empty()) != want_logits || (sample && !e.result))
      return fail("StepBatch: entries must all sample, all want logits, or neither");
    if (sample && want_logits) return fail("StepBatch: sample or logits, not both");
    if (!e.logits.empty() && e.logits.size() < c.vocab_size) return fail("logits span too small");
  }
  const SessionId bound = bound_;
  for (const auto& e : entries)
    if (mtp_track_ && e.session == mtp_owner_) mtp_pos_ = 0, mtp_base_ = kMtpNever;  // no longer follows: no drafting
  // One session, or no batched kernels for this model / these options: the entries in turn.
  if (B == 1 || !BatchSupported()) {
    for (const auto& e : entries) {
      if (!BindSession(e.session, error)) return false;
      const bool ok = sample ? StepSample(e.token, *e.sample, e.result, error) : Step(e.token, e.logits, error);
      if (!ok) {
        (void)BindSession(bound, nullptr);
        return false;
      }
    }
    return BindSession(bound, error);
  }
  if (!AllocBatch(error)) return false;
  if (sample && c.vocab_size > HALO_SAMPLE_MAX_VOCAB) return fail("vocabulary too large for device sampling");
  // Host side per entry: the single step's upload layout in its own region of bb_.params.
  const std::uint32_t ratio = c.compress_ratio, budget = c.indexer_top_k / ratio;
  std::vector<model::NgramHistory> hist(B);
  std::vector<std::uint32_t> pos(B), rows(c.ple_layer >= 0 ? c.ple_heads : 0);
  std::size_t upload = 0;
  for (std::uint32_t i = 0; i < B; ++i) {
    const auto& e = entries[i];
    if (!BindSession(e.session, error) || !EnsureKv(position_ + 1, position_, error)) {
      (void)BindSession(bound, nullptr);
      return false;
    }
    pos[i] = position_;
    hist[i] = ngram_history_;
    if (position_ >= max_context_ || e.token < 0 || static_cast<std::uint32_t>(e.token) >= c.vocab_size) {
      (void)BindSession(bound, nullptr);
      return fail(position_ >= max_context_ ? "context is full" : "token out of range");
    }
    std::uint8_t* host = bb_.host.data() + i * bb_.entry_stride;
    if (c.ple_layer >= 0) {
      model::HashNgramRows(c, hist[i], std::span<const std::int32_t>(&e.token, 1), rows);
      const std::span<float> emb(reinterpret_cast<float*>(host), c.PleEmbeddingDim());
      if (!ngram_ || !ngram_->Read(rows, emb)) {
        (void)BindSession(bound, nullptr);
        return fail("n-gram table read failed");
      }
    }
    const std::uint32_t n_kv = position_ + 1, complete = n_kv / ratio;
    std::uint32_t params[kHaloStepParamCount] = {};
    params[kHaloToken] = static_cast<std::uint32_t>(e.token);
    params[kHaloPosition] = position_;
    params[kHaloNkv] = n_kv;
    params[kHaloComplete] = complete;
    params[kHaloBlock] = position_ / ratio;
    params[kHaloNVisible] = complete <= budget ? n_kv : budget * ratio + (n_kv - complete * ratio);
    std::memcpy(host + params_offset_, params, sizeof(params));
    std::size_t used = sample_offset_;
    if (sample) {
      const auto& kc = *e.sample->config;
      if (e.sample->penalties.size() > HALO_SAMPLE_MAX_PENALTIES) {
        (void)BindSession(bound, nullptr);
        return fail("too many penalized tokens for the device");
      }
      try {
        kc.Validate();
      } catch (const std::exception& ex) {
        (void)BindSession(bound, nullptr);
        if (error) *error = ex.what();
        return false;
      }
      HaloSampleParams sp{kc.temperature,
                          kc.top_p,
                          kc.min_p,
                          kc.repeat_penalty,
                          kc.frequency_penalty,
                          kc.presence_penalty,
                          static_cast<std::uint32_t>(std::max(kc.top_k, 0)),
                          static_cast<std::uint32_t>(std::min<std::size_t>(kc.min_keep, UINT32_MAX)),
                          static_cast<std::uint32_t>(e.sample->penalties.size()),
                          c.vocab_size,
                          e.sample->rng,
                          sampling_mode_ == SamplingMode::kFast ? 1u : 0u,
                          0};
      std::memcpy(host + sample_offset_, &sp, sizeof(sp));
      auto* pen = reinterpret_cast<HaloPenalty*>(host + penalty_offset_);
      for (std::size_t j = 0; j < e.sample->penalties.size(); ++j) {
        const auto& t = e.sample->penalties[j];
        pen[j] = {t.token, t.generated_count, t.repeated, 0};
      }
      used = penalty_offset_ + e.sample->penalties.size() * sizeof(HaloPenalty);
    }
    upload = i * bb_.entry_stride + used;
  }
  if (!BindSession(bound, error) || !dev_.Upload(bb_.params, 0, bb_.host.data(), upload, error)) return false;
  const Head head = sample ? Head::kSample : want_logits ? Head::kLogits : Head::kNone;
  std::vector<SessionId> ids(B);
  for (std::uint32_t i = 0; i < B; ++i) ids[i] = entries[i].session;
  if (use_graphs_ && !timing_) {
    BatchGraph* g = nullptr;
    for (auto& bg : batch_graphs_)
      if (bg.head == static_cast<int>(head) && !bg.verify && bg.relaxed == batch_relaxed_ && bg.sessions == ids) g = &bg;
    if (!g) {
      constexpr std::size_t kMaxGraphs = 16;
      if (batch_graphs_.size() >= kMaxGraphs) {  // least recently used out
        auto lru = std::min_element(batch_graphs_.begin(), batch_graphs_.end(),
                                    [](const BatchGraph& a, const BatchGraph& b) { return a.used < b.used; });
        dev_.FreeGraph(lru->graph);
        batch_graphs_.erase(lru);
      }
      BatchGraph n;
      n.sessions = ids;
      n.head = static_cast<int>(head);
      n.relaxed = batch_relaxed_;
      if (!dev_.BeginCapture(error)) return false;
      const bool recorded = RecordBatch(ids, head, error);
      std::string end_error;
      const bool ended = dev_.EndCapture(&n.graph, &end_error);
      if (!recorded || !ended) {
        if (recorded && error) *error = end_error;
        return false;
      }
      batch_graphs_.push_back(std::move(n));
      g = &batch_graphs_.back();
    }
    g->used = ++batch_clock_;
    if (!dev_.LaunchGraph(g->graph, error)) return false;
  } else if (!RecordBatch(ids, head, error)) {
    return false;
  }
  if (want_logits)
    for (std::uint32_t i = 0; i < B; ++i)
      if (!dev_.Download(bb_.logits, std::size_t{i} * c.vocab_size * 4, entries[i].logits.data(), c.vocab_size * 4,
                         error))
        return false;
  if (sample) {
    const std::size_t per = (sizeof(HaloSampleOut) + 3) / 4 * 4;
    std::vector<std::uint8_t> outs(per * B);
    if (!dev_.Download(bb_.sample_out, 0, outs.data(), outs.size(), error)) return false;
    for (std::uint32_t i = 0; i < B; ++i) {
      HaloSampleOut out;
      std::memcpy(&out, outs.data() + i * per, sizeof(out));
      if (out.token >= c.vocab_size)
        return fail(out.token == 0xFFFFFFFFu ? "no finite logit to sample" : "device sampling did not finish");
      entries[i].result->token = static_cast<std::int32_t>(out.token);
      entries[i].result->rng = out.rng;
    }
  }
  // Every session advanced by one token.
  for (std::uint32_t i = 0; i < B; ++i) {
    if (!BindSession(entries[i].session, error)) return false;
    ngram_history_ = hist[i];  // advanced by the row hashing above
    const std::uint32_t complete = (pos[i] + 1) / ratio;
    for (auto& s : attn_) {
      s.blocks = complete;
      if (kv16_only_) s.valid16 = pos[i] + 1;  // the step appended its fp16 rows
    }
    position_ = pos[i] + 1;
  }
  return BindSession(bound, error);
}

}  // namespace halo::engine
