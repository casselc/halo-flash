// P4: Flash-Next prefill a chunk of tokens at a time.
//
// The step's arithmetic, reorganized around T token rows:
// - every dense projection is one T-token GEMM: int8 WMMA over repacked Q8_0
//   (activations quantized per token and 32 values), the BF16 / F32 GEMMs
//   otherwise, or the decode GEMV per token (prefill_exact_projections_,
//   LoRA, other formats);
// - norms and elementwise ops run over all T rows (the *_rows kernels);
// - DeltaNet runs the chunk kernels (conv window, state advance, recurrence
//   with the state in registers), bit-identical to stepping;
// - what is sequential or cheap stays per token, in order, each reading its
//   own step parameters: the PLE conv, attention with QSA block selection,
//   expert routing and (for now) the routed experts.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "engine/flash_next_gpu.hpp"
#include "kernels/step_params.h"

namespace halo::engine {

namespace {
constexpr std::uint32_t kBlock = 256;
constexpr std::uint32_t kMoeBlockRows = 128;  // = core.hip's kMoeBlockRows (grouped expert block rows)
std::uint32_t Blocks(std::size_t n, std::uint32_t per = kBlock) {
  return static_cast<std::uint32_t>(std::min<std::size_t>((n + per - 1) / per, 1u << 20));
}
}  // namespace

bool FlashNextGpu::ChunkSupported() const {
  const auto& c = w_.config;
  if (c.ssm_head_dim != 128 || c.SsmConvChannels() % 128 != 0 || (2 * c.SsmKeyDim()) % 128 != 0) return false;
  return true;
}

namespace {
// Kernel name for the residual's element type (precision option prefill_res16_: 1 fp16, 2 bf16).
std::string Rk(const char* base, int res16) {
  return std::string(base) + (res16 == 1 ? "_r16" : res16 == 2 ? "_rb" : "");
}
}  // namespace

bool FlashNextGpu::AllocChunk(std::string* error) {
  if (ck_ready_) return true;
  const auto& c = w_.config;
  const std::size_t T = std::max<std::uint32_t>(1, std::min(prefill_chunk_, max_context_)), H = c.hidden_size,
                    D = c.HcDim();
  const std::size_t qdim = c.AttentionQDim(), kvdim = c.AttentionKvDim(), vd = c.SsmValueDim();
  const std::size_t max_cols = std::max({D, H, qdim, vd, std::size_t{c.shared_expert_ff}, std::size_t{c.hc_low_rank}});
  auto& k = ck_;
  const auto f = [&](std::size_t n) { return Scratch(n, error); };
  k.res = f(T * D);
  k.xn = f(T * D);
  k.lo = f(T * c.hc_low_rank);
  k.gate = f(T * D);
  k.mixed = f(T * H);
  k.inject = f(T * c.hc_count);
  k.block_out = f(T * H);
  k.embd = f(T * H);
  k.xq = f(T * max_cols / 4 + 1);   // int8
  k.xd = f(T * max_cols / 32 + 1);
  k.qkv = f(T * c.SsmConvChannels());
  k.conv = f(T * c.SsmConvChannels());
  k.z = f(T * vd);
  k.alpha = f(T * c.ssm_num_v_heads);
  k.beta = f(T * c.ssm_num_v_heads);
  k.gdn = f(T * vd);
  k.qg = f(T * 2 * qdim);
  k.q = f(T * qdim);
  k.gate_a = f(T * qdim);
  k.k = f(T * kvdim);
  k.v = f(T * kvdim);
  k.ctx = f(T * qdim);
  k.iq = f(T * std::size_t{c.indexer_heads} * c.indexer_head_dim);
  k.ik = f(T * c.indexer_head_dim);
  k.router = f(T * c.num_experts);
  k.ids = f(T * c.num_experts_used);
  k.wts = f(T * c.num_experts_used);
  k.eg = f(T * c.shared_expert_ff);
  k.eu = f(T * c.shared_expert_ff);
  k.ed = f(T * H);
  k.sgate = f(T);
  const std::size_t pairs = T * c.num_experts_used;
  k.sorted = f(pairs);
  k.pos = f(pairs);
  k.tiles = f(3 * (pairs / 64 + c.num_experts + 1));
  k.ntiles = f(1);
  k.hp = f(pairs * c.expert_ff);
  k.yp = f(pairs * H);
  k.x16 = f(T * H / 2 + 1);
  k.bcounts = f((pairs + 1023) / 1024 * c.num_experts);
  k.bbase = f(c.num_experts);
  k.pa = f(T * D);
  k.ipart = f(T * c.hc_count * 8);
  k.gdnprep = f(std::size_t{c.ssm_num_v_heads} * ((T + 15) / 16) * 576);  // kGdnP16
  k.rscale = f(T * c.hc_count);
  k.xh = f(T * max_cols / 2 + 1);
  k.mh = f(T * H / 2 + 1);
  k.lh = f(T * c.hc_low_rank / 2 + 1);
  k.ml = f(T * H / 2 + 1);  // the mixer's lo in fp16 when xh still holds xn (prefill_mix_h16_)
  k.pb = f(T * D);
  const std::size_t sub = std::min<std::size_t>(T, kAttnSub);
  k.bscores = f(sub * (max_context_ / c.compress_ratio + 1));
  k.vis = f(sub * AttnVisibleStride());
  k.ascores = f(sub * c.num_heads * AttnVisibleStride());
  {
    const std::size_t groups = (sub + 3) / 4, max_u = 4ull * (c.indexer_top_k / c.compress_ratio) + 4;
    k.ublk = f(groups * max_u);
    k.umask = f((groups * max_u + 3) / 4);  // bytes
    k.ucount = f(groups);
  }
  k.params_offset = (T * c.PleEmbeddingDim() * 4 + 63) / 64 * 64;
  ck_host_.assign(k.params_offset + T * 64, 0);
  k.host = dev_.Alloc(ck_host_.size(), error);
  if (k.host) owned_.push_back(k.host);
  for (const auto* b : {&k.res, &k.xn, &k.lo, &k.gate, &k.mixed, &k.inject, &k.block_out, &k.embd, &k.xq, &k.xd,
                        &k.qkv, &k.conv, &k.z, &k.alpha, &k.beta, &k.gdn, &k.qg, &k.q, &k.gate_a, &k.k, &k.v, &k.ctx,
                        &k.iq, &k.ik, &k.router, &k.ids, &k.wts, &k.eg, &k.eu, &k.ed, &k.sgate, &k.host,
                        &k.sorted, &k.pos, &k.tiles, &k.ntiles, &k.hp, &k.yp, &k.x16, &k.bcounts, &k.bbase, &k.pa, &k.pb, &k.ipart, &k.rscale, &k.gdnprep, &k.xh, &k.mh, &k.lh, &k.ml,
                        &k.bscores, &k.vis, &k.ascores, &k.ublk, &k.umask, &k.ucount})
    if (!*b) return false;
  // Index-major copies of the mixers' up projections (fused GEMM + mix).
  if (c.hc_count == 4)
    for (const auto& l : w_.layers)
      for (const auto* m : {&l.hc_attn, &l.hc_ffn}) {
        const auto& t = m->up;
        const auto& dt = tensors_.at(t.data);
        if (!dt.qs || t.rows != std::uint64_t{4} * H || t.cols % 64 != 0 || hc_up_perm_.contains(t.data)) continue;
        const auto rows = static_cast<std::uint32_t>(t.rows), nblk = static_cast<std::uint32_t>(t.cols / 32);
        PermutedUp p{dev_.Alloc(std::size_t{rows} * dt.qstride, error), dev_.Alloc(std::size_t{rows} * nblk * 2, error)};
        if (!p.qs || !p.d) return false;
        owned_.push_back(p.qs);
        owned_.push_back(p.d);
        if (!Launch("halo_permute_hc_rows", {rows}, {64},
                    {rt::Arg::Buf(dt.qs), rt::Arg::Buf(dt.d), rt::Arg::Buf(p.qs), rt::Arg::Buf(p.d),
                     rt::Arg::Val(static_cast<std::uint32_t>(H)), rt::Arg::Val(dt.qstride), rt::Arg::Val(nblk)},
                    error))
          return false;
        hc_up_perm_.emplace(t.data, p);
      }
  if (!dev_.Synchronize(error)) return false;
  k.cap = static_cast<std::uint32_t>(T);
  ck_ready_ = true;
  return true;
}

// Router mode 4 (stream R): the MoE-half mix epilogue writes `mixed`'s fp16 lo half next to mh.
bool FlashNextGpu::RouterHL(const model::LayerWeights& l) {
  router_hl_ok_ = prefill_router_f16_ == 4 && prefill_f16_gemm_ && !prefill_exact_projections_ && prefill_mix_h16_ &&
                  MixFused(l.hc_ffn) && MixH(l.hc_ffn) && !lora_.contains(l.router.data) &&
                  l.router.type == core::GgmlType::kF32 && l.router.cols % 64 == 0 && l.router.rows % 128 == 0;
  return router_hl_ok_;
}

// `mixed` in fp16 for the GEMMs that share it (only when the FP16 route runs).
bool FlashNextGpu::MixedHalf(std::uint32_t T, std::string* error) {
  return !prefill_f16_gemm_ || prefill_exact_projections_ || mh_valid_ ||
         Half(ck_.mixed, ck_.mh, T * w_.config.hidden_size, error);
}
const rt::Buffer* FlashNextGpu::MixedH() const {
  return prefill_f16_gemm_ && !prefill_exact_projections_ ? &ck_.mh : nullptr;
}

bool FlashNextGpu::Half(const rt::Buffer& x, const rt::Buffer& out, std::uint32_t n, std::string* error) {
  return Launch("halo_f16_rows", {Blocks(n)}, {kBlock}, {rt::Arg::Buf(x), rt::Arg::Buf(out), rt::Arg::Val(n)}, error);
}

// Whether a mixer's F32 inject projection is computed inside the norm kernel
// that produces xn (partial products per stream, then halo_inject_reduce)
// instead of a GEMM over xn.
bool FlashNextGpu::FuseInject(const model::HcMixer& m) const {
  const auto& c = w_.config;
  return prefill_fused_inject_ && !prefill_exact_projections_ && !lora_.contains(m.inject.data) &&
         m.inject.type == core::GgmlType::kF32 && m.inject.rows <= 8 && m.inject.cols == c.HcDim();
}

// Whether the mixer's up projection and stream mix run as one FP16 GEMM
// (halo_gemm_q8r_f16_mix over the index-major permuted weights).
bool FlashNextGpu::MixFused(const model::HcMixer& m) const {
  return prefill_f16_gemm_ && !prefill_exact_projections_ && hc_up_perm_.contains(m.up.data) &&
         !lora_.contains(m.up.data);
}

// Whether Gemm takes the FP16 route for t (and so reads a provided x16).
bool FlashNextGpu::F16Route(const model::TensorRef& t) const {
  const auto& d = tensors_.at(t.data);
  const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
  const bool small = !(rows <= 320 && cols >= 4096) && rows <= SmallRows();
  return !prefill_exact_projections_ && !lora_.contains(t.data) && prefill_f16_gemm_ && d.qs &&
         cols % (small ? 128 : 64) == 0 && d.qstride % 16 == 0;
}

// Whether combine_norm before mixer m can run per token (halo_hc_combine_norm_tok): four streams, xn
// skipped (the mix epilogue rebuilds it), fp16 copy and fused inject partials.
bool FlashNextGpu::TokCombineFits(const model::HcMixer& m) const {
  const auto& c = w_.config;
  return c.hc_count == 4 && c.hidden_size <= 2560 && m.inject.rows <= 8 && prefill_mix_rebuild_ && F16Route(m.down) &&
         FuseInject(m) && MixFused(m);
}
bool FlashNextGpu::TokCombine(const model::HcMixer& m) const { return prefill_tok_combine_ && TokCombineFits(m); }

bool FlashNextGpu::BlockedXh(const model::HcMixer& m) const {
  const auto rows = static_cast<std::uint32_t>(m.down.rows), cols = static_cast<std::uint32_t>(m.down.cols);
  return prefill_blocked_xh_ && F16Route(m.down) && rows <= 320 && cols >= 4096 && cols % 64 == 0;
}

// Whether the mixer's mix epilogue reads xn as the norm's fp16 K-blocked copy (prefill_mix_h16_).
bool FlashNextGpu::MixH(const model::HcMixer& m) const {
  return prefill_mix_h16_ && MixFused(m) && F16Route(m.down) && BlockedXh(m) && w_.config.hc_count == 4 &&
         m.up.rows % 256 == 0;
}

bool FlashNextGpu::Gemm(const model::TensorRef& t, const rt::Buffer& x, const rt::Buffer& y, std::uint32_t T,
                        std::string* error, const rt::Buffer* x16, bool x16_blocked) {
  const auto& d = tensors_.at(t.data);
  const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
  const bool per_token = prefill_exact_projections_ || lora_.contains(t.data);
  // Tile by shape: all rows in one block for <= 320 rows with long K (the
  // activations dominate), 64-row tiles up to 512, 256-row tiles beyond.
  const bool tall = rows <= 320 && cols >= 4096, small = !tall && rows <= SmallRows();
  const std::uint32_t bm = tall ? 320 : small ? 64 : 256, bn = tall ? 64 : 128;
  if (F16Route(t)) {
    if (!x16 && !Half(x, ck_.xh, T * cols, error)) return false;
    // Large tiles walk in bands of 8 row tiles when they divide evenly.
    const bool banded = !tall && !small && prefill_banded_gemm_ && (rows / 256) % 8 == 0 && rows % 256 == 0;
    return Launch(tall    ? (x16 && x16_blocked ? "halo_gemm_q8r_f16_mb" : "halo_gemm_q8r_f16_m")
                  : small ? "halo_gemm_q8r_f16_s"
                  : banded ? (prefill_reg_gemm_ ? "halo_gemm_q8r_f16_lr8" : "halo_gemm_q8r_f16_lg8")
                           : "halo_gemm_q8r_f16_l",
                  {(T + bn - 1) / bn, (rows + bm - 1) / bm}, {256},
                  {rt::Arg::Buf(d.qs), rt::Arg::Buf(d.d), rt::Arg::Buf(x16 ? *x16 : ck_.xh), rt::Arg::Buf(y),
                   rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val(d.qstride), rt::Arg::Val(T)},
                  error);
  }
  if (!per_token && prefill_int8_gemm_ && d.qs && cols % 32 == 0 && d.qstride % 16 == 0) {
    return Launch("halo_quantize_q8_rows", {T}, {256},
                  {rt::Arg::Buf(x), rt::Arg::Buf(ck_.xq), rt::Arg::Buf(ck_.xd), rt::Arg::Val(cols)}, error) &&
           Launch("halo_gemm_q8r_wmma_t", {(T + 127) / 128, (rows + 127) / 128}, {256},
                  {rt::Arg::Buf(d.qs), rt::Arg::Buf(d.d), rt::Arg::Buf(ck_.xq), rt::Arg::Buf(ck_.xd), rt::Arg::Buf(y),
                   rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val(d.qstride), rt::Arg::Val(T)},
                  error);
  }
  if (!per_token && t.type == core::GgmlType::kBF16 && cols % 128 == 0)
    return Launch("halo_gemm_bf16_f16_k128", {(T + 63) / 64, (rows + 63) / 64}, {128},
                  {rt::Arg::Buf(d.buf), rt::Arg::Buf(x), rt::Arg::Buf(y), rt::Arg::Val(rows), rt::Arg::Val(cols),
                   rt::Arg::Val(T)},
                  error);
  if (!per_token && t.type == core::GgmlType::kF32 && rows <= 16 && prefill_skinny_f32_)
    return Launch("halo_gemm_f32_skinny", {(T + 7) / 8}, {256},
                  {rt::Arg::Buf(d.buf), rt::Arg::Buf(x), rt::Arg::Buf(y), rt::Arg::Val(rows), rt::Arg::Val(cols),
                   rt::Arg::Val(T)},
                  error);
  if (!per_token && t.type == core::GgmlType::kF32 && cols % 16 == 0)
    return Launch("halo_gemm_f32_tiled", {(T + 63) / 64, (rows + 63) / 64}, {256},
                  {rt::Arg::Buf(d.buf), rt::Arg::Buf(x), rt::Arg::Buf(y), rt::Arg::Val(rows), rt::Arg::Val(cols),
                   rt::Arg::Val(T)},
                  error);
  if (!per_token && t.type == core::GgmlType::kF32)
    return Launch("halo_gemm_f32_rows", {Blocks(rows, 8), T}, {256},
                  {rt::Arg::Buf(d.buf), rt::Arg::Buf(x), rt::Arg::Buf(y), rt::Arg::Val(rows), rt::Arg::Val(cols)},
                  error);
  for (std::uint32_t i = 0; i < T; ++i)
    if (!MatVec(t, 0, x, y, error, std::size_t{i} * cols * 4, std::size_t{i} * rows * 4)) return false;
  return true;
}

bool FlashNextGpu::MixedGemmF16(const model::TensorRef& t, const rt::Buffer& y, std::uint32_t T,
                                std::string* error) {
  const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
  if (!(prefill_small_f16_ && prefill_f16_gemm_ && !prefill_exact_projections_ && !lora_.contains(t.data) &&
        t.type == core::GgmlType::kF32 && cols % 128 == 0))
    return Gemm(t, ck_.mixed, y, T, error);
  return MixedHalf(T, error) &&
         Launch("halo_gemm_f32h_f16_k128", {(T + 63) / 64, (rows + 63) / 64}, {128},
                {rt::Arg::Buf(tensors_.at(t.data).buf), rt::Arg::Buf(ck_.mh), rt::Arg::Buf(y), rt::Arg::Val(rows),
                 rt::Arg::Val(cols), rt::Arg::Val(T)},
                error);
}

bool FlashNextGpu::HcMixChunk(const model::HcMixer& m, bool inject, std::uint32_t T, std::string* error,
                              bool normed) {
  const auto& c = w_.config;
  const std::uint32_t H = c.hidden_size, low = c.hc_low_rank;
  // On the FP16 route the norm also leaves xn in fp16 for the down projection.
  const bool h16 = F16Route(m.down);
  // The inject's partial products come from the norm kernel when it is
  // fused (here on the FP16 route, or in the combine that normalized already).
  const bool fused_inject = inject && FuseInject(m) && (normed || h16);
  const auto irows = static_cast<std::uint32_t>(m.inject.rows);
  const bool norm_ok =
      normed ||
      (h16 ? Launch(Rk("halo_rmsnorm_rows_h", prefill_res16_).c_str(), {T * c.hc_count}, {kBlock},
                   {rt::Arg::Buf(ck_.res), rt::Arg::Buf(tensors_.at(m.norm.data).buf), rt::Arg::Buf(ck_.xn),
                    rt::Arg::Buf(ck_.xh), rt::Arg::Val(H), rt::Arg::Val(c.rms_eps), rt::Arg::Val(H),
                    rt::Arg::Val(c.hc_count),
                    fused_inject ? rt::Arg::Buf(tensors_.at(m.inject.data).buf) : rt::Arg::Val<std::uint64_t>(0),
                    rt::Arg::Buf(ck_.ipart), rt::Arg::Val(irows), rt::Arg::Val(BlockedXh(m) ? T : 0u)},
                   error)
          : Launch(Rk("halo_rmsnorm_rows", prefill_res16_).c_str(), {T * c.hc_count}, {kBlock},
                   {rt::Arg::Buf(ck_.res), rt::Arg::Buf(tensors_.at(m.norm.data).buf), rt::Arg::Buf(ck_.xn),
                    rt::Arg::Val(H), rt::Arg::Val(c.rms_eps), rt::Arg::Val(H), rt::Arg::Val(c.hc_count)},
                   error));
  // Precision option: W8A8 down projection over the int8 copy the combine wrote (normed only).
  const bool down_a8 = normed && xq_valid_;
  xq_valid_ = false;
  const auto& dd = tensors_.at(m.down.data);
  const auto drows = static_cast<std::uint32_t>(m.down.rows), dcols = static_cast<std::uint32_t>(m.down.cols);
  const bool down_ok =
      down_a8 ? Launch("halo_gemm_q8a8_160x64", {(T + 63) / 64, (drows + 159) / 160}, {256},
                       {rt::Arg::Buf(dd.qs), rt::Arg::Buf(dd.d), rt::Arg::Buf(ck_.xq), rt::Arg::Buf(ck_.xd),
                        rt::Arg::Buf(ck_.lo), rt::Arg::Val(drows), rt::Arg::Val(dcols), rt::Arg::Val(dd.qstride),
                        rt::Arg::Val(T)},
                       error)
              : Gemm(m.down, ck_.xn, ck_.lo, T, error, h16 ? &ck_.xh : nullptr, h16 && BlockedXh(m));
  // With the fused mix, lo is read only as fp16: the SiLU pass writes just that copy (into lh when the fp16
  // mix still needs xn's fp16 copy in xh).
  lo_half_ = prefill_f16_out_ && MixFused(m);
  return norm_ok && down_ok &&
         (lo_half_ ? Launch("halo_scale_silu_h", {Blocks(std::size_t{T} * low)}, {kBlock},
                            {rt::Arg::Buf(ck_.lo), rt::Arg::Buf(MixH(m) ? ck_.lh : ck_.xh),
                             rt::Arg::Val(1.0F / static_cast<float>(c.hc_count)), rt::Arg::Val(T * low)},
                            error)
                   : Launch("halo_scale_silu", {Blocks(std::size_t{T} * low)}, {kBlock},
                            {rt::Arg::Buf(ck_.lo), rt::Arg::Val(1.0F / static_cast<float>(c.hc_count)),
                             rt::Arg::Val(T * low)},
                            error)) &&
         MixUp(m, T, error) &&
         (!inject ||
          (fused_inject ? Launch("halo_inject_reduce", {Blocks(std::size_t{T} * irows)}, {kBlock},
                                 {rt::Arg::Buf(ck_.ipart), rt::Arg::Buf(ck_.inject), rt::Arg::Val(c.hc_count),
                                  rt::Arg::Val(irows), rt::Arg::Val(T)},
                                 error)
                        : Gemm(m.inject, ck_.xn, ck_.inject, T, error)));
}

// gate = up(lo), mixed = sum_s xn * sigmoid(gate) / hc: one fused GEMM on the
// FP16 route (also leaving `mixed` in fp16 for the GEMMs that read it),
// otherwise the GEMM and the row kernel.
bool FlashNextGpu::MixUp(const model::HcMixer& m, std::uint32_t T, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t H = c.hidden_size, low = c.hc_low_rank;
  const auto it = hc_up_perm_.find(m.up.data);
  mh_valid_ = MixFused(m);
  if (mh_valid_) {
    const auto& dt = tensors_.at(m.up.data);
    const auto rows = static_cast<std::uint32_t>(m.up.rows);
    const bool rebuild = xn_skipped_;  // xn was not written: rebuild it from res
    xn_skipped_ = false;
    const bool lo_half = lo_half_;
    lo_half_ = false;
    const bool mix_h = MixH(m);
    if (prefill_res16_ && rebuild && !mix_h) {
      if (error) *error = "internal: the 16-bit residual needs the fp16 mix (--mix-h16)";
      return false;
    }
    // Precision option: the epilogue reads the norm's fp16 K-blocked copy of xn (still in xh).
    const bool ml = want_ml_;
    want_ml_ = false;
    if (mix_h)
      return (lo_half || Half(ck_.lo, ck_.lh, T * low, error)) &&
             (ml ? Launch("halo_gemm_q8r_f16_mixthl", {(rows + 255) / 256, (T + 127) / 128}, {256},
                          {rt::Arg::Buf(it->second.qs), rt::Arg::Buf(it->second.d), rt::Arg::Buf(ck_.lh),
                           rt::Arg::Buf(ck_.mixed), rt::Arg::Val(rows), rt::Arg::Val(low), rt::Arg::Val(dt.qstride),
                           rt::Arg::Val(T), rt::Arg::Buf(ck_.xh), rt::Arg::Buf(ck_.mh), rt::Arg::Buf(ck_.ml)},
                          error)
                 : Launch("halo_gemm_q8r_f16_mixth", {(rows + 255) / 256, (T + 127) / 128}, {256},
                          {rt::Arg::Buf(it->second.qs), rt::Arg::Buf(it->second.d), rt::Arg::Buf(ck_.lh),
                           rt::Arg::Buf(ck_.mixed), rt::Arg::Val(rows), rt::Arg::Val(low), rt::Arg::Val(dt.qstride),
                           rt::Arg::Val(T), rt::Arg::Buf(ck_.xh), rt::Arg::Buf(ck_.mh)},
                          error));
    return (lo_half || Half(ck_.lo, ck_.xh, T * low, error)) &&
           // Row tiles fastest: resident blocks read whole residual rows of the same
           // tokens (token tiles fastest: 1.41 vs 1.15 ms warm, 1.62 vs 1.19 after
           // its input was just written).
           Launch("halo_gemm_q8r_f16_mixt", {(rows + 255) / 256, (T + 127) / 128}, {256},
                  {rt::Arg::Buf(it->second.qs), rt::Arg::Buf(it->second.d), rt::Arg::Buf(ck_.xh), rt::Arg::Buf(ck_.mixed),
                   rt::Arg::Val(rows), rt::Arg::Val(low), rt::Arg::Val(dt.qstride), rt::Arg::Val(T),
                   rt::Arg::Buf(rebuild ? ck_.res : ck_.xn), rt::Arg::Buf(ck_.mh),
                   rebuild ? rt::Arg::Buf(ck_.rscale) : rt::Arg::Val<std::uint64_t>(0),
                   rebuild ? rt::Arg::Buf(tensors_.at(m.norm.data).buf) : rt::Arg::Val<std::uint64_t>(0)},
                  error);
  }
  if (xn_skipped_) {
    if (error) *error = "internal: xn was skipped but the mix is not the fused epilogue";
    return false;
  }
  return Gemm(m.up, ck_.lo, ck_.gate, T, error) &&
         Launch("halo_hc_mix_rows", {Blocks(std::size_t{T} * H)}, {kBlock},
                {rt::Arg::Buf(ck_.xn), rt::Arg::Buf(ck_.gate), rt::Arg::Buf(ck_.mixed), rt::Arg::Val(c.hc_count),
                 rt::Arg::Val(H), rt::Arg::Val(T)},
                error);
}

bool FlashNextGpu::PleChunk(const model::LayerWeights& l, std::uint32_t T, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t D = c.HcDim(), H = c.hidden_size, hc = c.hc_count;
  // Rows: key pa, value mixed, query xn, gated gate, normalized pb (the
  // mixer buffers are free until this layer's mixer runs).
  const auto norm = [&](const rt::Buffer& x, const model::TensorRef& w, const rt::Buffer& y) {
    return Launch(Rk("halo_rmsnorm_rows", &x == &ck_.res ? prefill_res16_ : 0).c_str(), {T * hc}, {kBlock},
                  {rt::Arg::Buf(x), rt::Arg::Buf(tensors_.at(w.data).buf), rt::Arg::Buf(y), rt::Arg::Val(H),
                   rt::Arg::Val(c.rms_eps), rt::Arg::Val(H), rt::Arg::Val(hc)},
                  error);
  };
  // Both projections read the same rows: on the FP16 route the value GEMM reuses the key GEMM's fp16 copy.
  const bool share = prefill_f16_out_ && F16Route(l.ple_key) && F16Route(l.ple_value) && l.ple_key.cols == l.ple_value.cols;
  return Gemm(l.ple_key, ck_.host, ck_.pa, T, error) &&
         Gemm(l.ple_value, ck_.host, ck_.mixed, T, error, share ? &ck_.xh : nullptr) && norm(ck_.pa, l.ple_norm_key, ck_.pa) &&
         norm(ck_.res, l.ple_norm_query, ck_.xn) &&
         Launch("halo_ple_gate_rows", {T * hc}, {kBlock},
                {rt::Arg::Buf(ck_.pa), rt::Arg::Buf(ck_.xn), rt::Arg::Buf(ck_.mixed), rt::Arg::Buf(ck_.gate),
                 rt::Arg::Val(H), rt::Arg::Val(hc)},
                error) &&
         norm(ck_.gate, l.ple_norm_conv, ck_.pb) &&
         Launch(Rk("halo_ple_conv_prefill", prefill_res16_).c_str(), {Blocks(D), T}, {kBlock},
                {rt::Arg::Buf(ck_.pb), rt::Arg::Buf(ck_.gate), rt::Arg::Buf(ple_history_),
                 rt::Arg::Buf(tensors_.at(l.ple_conv1d.data).buf), rt::Arg::Buf(ck_.res), rt::Arg::Val(D),
                 rt::Arg::Val(c.ple_conv_kernel), rt::Arg::Val(c.ple_ngram_size)},
                error) &&
         Launch("halo_ple_history_advance", {Blocks(D)}, {kBlock},
                {rt::Arg::Buf(ck_.pb), rt::Arg::Buf(ple_history_), rt::Arg::Val(D), rt::Arg::Val(c.PleConvHistory()),
                 rt::Arg::Val(T)},
                error);
}

namespace {
// Chunks a halo_gdn_c16_prep2 block walks: ~64 blocks a layer (cpb 32 at
// 2048 tokens and 16 k-heads: 0.35 ms vs 0.38 at cpb 16, 1.22 one a block).
std::uint32_t GdnPrepCpb(std::uint32_t nch, std::uint32_t nk) { return std::max(1u, (nch * nk + 63) / 64); }
}  // namespace

bool FlashNextGpu::LinearChunk(const model::LayerWeights& l, LinearState& s, std::uint32_t T, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t nch = (T + 15) / 16;
  const std::uint32_t ch = c.SsmConvChannels(), kd = c.SsmKeyDim(), vd = c.SsmValueDim();
  // Precision option: qkv and z written in fp16 (their readers: the conv, the state advance, the gated norm).
  const auto banded_h = [&](const model::TensorRef& t) {
    const auto rows = static_cast<std::uint32_t>(t.rows);
    return F16Route(t) && prefill_banded_gemm_ && prefill_reg_gemm_ && rows > 512 && rows % 256 == 0 &&
           (rows / 256) % 8 == 0 && !(rows <= 320 && t.cols >= 4096);
  };
  const bool io16_ok = c.ssm_conv_kernel == 4 && prefill_split_gdn_ && !prefill_exact_projections_ &&
                       F16Route(l.ssm_out) && banded_h(l.ssm_qkv) && banded_h(l.ssm_gate);
  // Precision option (stream R): everything from the qkv / z GEMMs to the gated norm in fp16.
  const bool f16 = prefill_gdn_f16_ > 0 && io16_ok && prefill_gdn_chunked_ && !prefill_gdn_chunk_fp32_ &&
                   !prefill_gdn_c16_bf16_ && prefill_gated_wave_ && c.ssm_head_dim == 128 &&
                   c.ssm_num_v_heads % c.ssm_num_k_heads == 0 && c.ssm_num_v_heads / c.ssm_num_k_heads <= 8;
  const bool io16 = (prefill_gdn_io16_ || f16) && io16_ok;
  const auto gemm_h = [&](const model::TensorRef& t, const rt::Buffer& y) {
    const auto& d = tensors_.at(t.data);
    const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
    return Launch("halo_gemm_q8r_f16_lr8h", {(T + 127) / 128, rows / 256}, {256},
                  {rt::Arg::Buf(d.qs), rt::Arg::Buf(d.d), rt::Arg::Buf(ck_.mh), rt::Arg::Buf(y), rt::Arg::Val(rows),
                   rt::Arg::Val(cols), rt::Arg::Val(d.qstride), rt::Arg::Val(T)},
                  error);
  };
  if (f16) {
    const std::uint32_t cpb = GdnPrepCpb(nch, c.ssm_num_k_heads);
    return MixedHalf(T, error) && gemm_h(l.ssm_qkv, ck_.qkv) && gemm_h(l.ssm_gate, ck_.z) &&
           MixedGemmF16(l.ssm_alpha, ck_.alpha, T, error) && MixedGemmF16(l.ssm_beta, ck_.beta, T, error) &&
           Launch("halo_conv_l2norm_prefill8_hh", {ch / 128, (T + 7) / 8}, {128},
                  {rt::Arg::Buf(ck_.qkv), rt::Arg::Buf(s.conv), rt::Arg::Buf(tensors_.at(l.ssm_conv1d.data).buf),
                   rt::Arg::Buf(ck_.conv), rt::Arg::Val(ch), rt::Arg::Val(T), rt::Arg::Val(2 * kd),
                   rt::Arg::Val(c.rms_eps)},
                  error) &&
           Launch("halo_conv_state_advance_h", {Blocks(ch)}, {kBlock},
                  {rt::Arg::Buf(ck_.qkv), rt::Arg::Buf(s.conv), rt::Arg::Val(ch), rt::Arg::Val(c.ssm_conv_kernel),
                   rt::Arg::Val(T)},
                  error) &&
           Launch("halo_gdn_c16_prep2_h", {(nch + cpb - 1) / cpb, c.ssm_num_k_heads}, {256},
                  {rt::Arg::Buf(ck_.conv), rt::Arg::Buf(ck_.conv, kd * 2ull), rt::Arg::Buf(ck_.alpha),
                   rt::Arg::Buf(ck_.beta), rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf),
                   rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf), rt::Arg::Buf(ck_.gdnprep),
                   rt::Arg::Val(c.ssm_num_v_heads), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(cpb)},
                  error) &&
           Launch(prefill_gdn_f16_ == 2 ? "halo_gdn_c16_scanft" : "halo_gdn_c16_scanf", {2, c.ssm_num_v_heads}, {128},
                  {rt::Arg::Buf(ck_.conv), rt::Arg::Buf(ck_.conv, kd * 2ull), rt::Arg::Buf(ck_.conv, 2ull * kd * 2),
                   rt::Arg::Buf(ck_.gdnprep), rt::Arg::Buf(s.state), rt::Arg::Buf(ck_.gdn),
                   rt::Arg::Val(c.ssm_num_k_heads), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch),
                   rt::Arg::Val(vd)},
                  error) &&
           Launch("halo_gated_norm_h128w_hh", {(T * c.ssm_num_v_heads + 7) / 8}, {256},
                  {rt::Arg::Buf(ck_.gdn), rt::Arg::Buf(tensors_.at(l.ssm_norm.data).buf), rt::Arg::Buf(ck_.z),
                   rt::Arg::Buf(ck_.xh), rt::Arg::Val(T * c.ssm_num_v_heads), rt::Arg::Val(c.rms_eps),
                   rt::Arg::Val<std::uint32_t>(1)},
                  error) &&
           Gemm(l.ssm_out, ck_.gdn, ck_.block_out, T, error, &ck_.xh);
  }
  return MixedHalf(T, error) &&
         (io16 ? gemm_h(l.ssm_qkv, ck_.qkv) && gemm_h(l.ssm_gate, ck_.z)
               : Gemm(l.ssm_qkv, ck_.mixed, ck_.qkv, T, error, MixedH()) &&
                     Gemm(l.ssm_gate, ck_.mixed, ck_.z, T, error, MixedH())) &&
         MixedGemmF16(l.ssm_alpha, ck_.alpha, T, error) && MixedGemmF16(l.ssm_beta, ck_.beta, T, error) &&
         (c.ssm_conv_kernel == 4  // 8 tokens a block, the same bits
              ? Launch(io16 ? "halo_conv_l2norm_prefill8_h" : "halo_conv_l2norm_prefill8", {ch / 128, (T + 7) / 8}, {128},
                       {rt::Arg::Buf(ck_.qkv), rt::Arg::Buf(s.conv), rt::Arg::Buf(tensors_.at(l.ssm_conv1d.data).buf),
                        rt::Arg::Buf(ck_.conv), rt::Arg::Val(ch), rt::Arg::Val(T), rt::Arg::Val(2 * kd),
                        rt::Arg::Val(c.rms_eps)},
                       error)
              : Launch("halo_conv_l2norm_prefill", {ch / 128, T}, {128},
                       {rt::Arg::Buf(ck_.qkv), rt::Arg::Buf(s.conv), rt::Arg::Buf(tensors_.at(l.ssm_conv1d.data).buf),
                        rt::Arg::Buf(ck_.conv), rt::Arg::Val(ch), rt::Arg::Val(c.ssm_conv_kernel), rt::Arg::Val(2 * kd),
                        rt::Arg::Val(c.rms_eps)},
                       error)) &&
         Launch(io16 ? "halo_conv_state_advance_h" : "halo_conv_state_advance", {Blocks(ch)}, {kBlock},
                {rt::Arg::Buf(ck_.qkv), rt::Arg::Buf(s.conv), rt::Arg::Val(ch), rt::Arg::Val(c.ssm_conv_kernel),
                 rt::Arg::Val(T)},
                error) &&
         (prefill_split_gdn_ && !prefill_exact_projections_
              ? (prefill_gdn_chunked_ && c.ssm_head_dim == 128 && c.ssm_num_v_heads % c.ssm_num_k_heads == 0 &&
                         c.ssm_num_v_heads / c.ssm_num_k_heads <= 8
                     // 16-token chunks: everything but the state-dependent
                     // products over all chunks at once, then a WMMA scan
                     // over chunks instead of over tokens.
                     ? (prefill_gdn_chunk_fp32_
                            ? Launch("halo_gdn_chunk_prep", {(T + 15) / 16, c.ssm_num_v_heads}, {256},
                                     {rt::Arg::Buf(ck_.conv), rt::Arg::Buf(ck_.conv, kd * 4ull), rt::Arg::Buf(ck_.alpha),
                                      rt::Arg::Buf(ck_.beta), rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf),
                                      rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf), rt::Arg::Buf(ck_.gdnprep),
                                      rt::Arg::Val(c.ssm_num_k_heads), rt::Arg::Val(T), rt::Arg::Val(ch)},
                                     error) &&
                                  Launch("halo_gdn_chunk_scan", {2, c.ssm_num_v_heads}, {256},
                                         {rt::Arg::Buf(ck_.conv), rt::Arg::Buf(ck_.conv, kd * 4ull),
                                          rt::Arg::Buf(ck_.conv, 2ull * kd * 4), rt::Arg::Buf(ck_.gdnprep),
                                          rt::Arg::Buf(s.state), rt::Arg::Buf(ck_.gdn), rt::Arg::Val(c.ssm_num_k_heads),
                                          rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch), rt::Arg::Val(vd)},
                                         error)
                            // prep: ~64 blocks a layer, each walking cpb chunks
                            // with the next one prefetched (halo_gdn_c16_prep2).
                            : Launch("halo_gdn_c16_prep2",
                                     {(nch + GdnPrepCpb(nch, c.ssm_num_k_heads) - 1) / GdnPrepCpb(nch, c.ssm_num_k_heads),
                                      c.ssm_num_k_heads},
                                     {256},
                                     {rt::Arg::Buf(ck_.conv), rt::Arg::Buf(ck_.conv, kd * 4ull), rt::Arg::Buf(ck_.alpha),
                                      rt::Arg::Buf(ck_.beta), rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf),
                                      rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf), rt::Arg::Buf(ck_.gdnprep),
                                      rt::Arg::Val(c.ssm_num_v_heads), rt::Arg::Val(T), rt::Arg::Val(ch),
                                      rt::Arg::Val(GdnPrepCpb(nch, c.ssm_num_k_heads))},
                                     error) &&
                           Launch(prefill_gdn_c16_bf16_ ? "halo_gdn_c16_scan" : prefill_gdn_scan_h_ ? "halo_gdn_c16_scanh" : prefill_gdn_c16_u3_ ? "halo_gdn_c16_scan3u" : "halo_gdn_c16_scan3", {2, c.ssm_num_v_heads}, {128},
                                  {rt::Arg::Buf(ck_.conv), rt::Arg::Buf(ck_.conv, kd * 4ull),
                                   rt::Arg::Buf(ck_.conv, 2ull * kd * 4), rt::Arg::Buf(ck_.gdnprep),
                                   rt::Arg::Buf(s.state), rt::Arg::Buf(ck_.gdn), rt::Arg::Val(c.ssm_num_k_heads),
                                   rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch), rt::Arg::Val(vd)},
                                  error))
                     : Launch("halo_gdn_prefill_split", {2, c.ssm_num_v_heads}, {256},
                       {rt::Arg::Buf(ck_.conv), rt::Arg::Buf(ck_.conv, kd * 4ull), rt::Arg::Buf(ck_.conv, 2ull * kd * 4),
                        rt::Arg::Buf(ck_.alpha), rt::Arg::Buf(ck_.beta), rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf),
                        rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf), rt::Arg::Buf(s.state), rt::Arg::Buf(ck_.gdn),
                        rt::Arg::Val(c.ssm_num_k_heads), rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch),
                        rt::Arg::Val(vd)},
                       error)) &&
                    (F16Route(l.ssm_out)
                         ? (prefill_gated_wave_ && !io16 && c.ssm_head_dim == 128
                                ? Launch("halo_gated_norm_h128w", {(T * c.ssm_num_v_heads + 7) / 8}, {256},
                                         {rt::Arg::Buf(ck_.gdn), rt::Arg::Buf(tensors_.at(l.ssm_norm.data).buf),
                                          rt::Arg::Buf(ck_.z), rt::Arg::Buf(ck_.xh), rt::Arg::Val(T * c.ssm_num_v_heads),
                                          rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(1)},
                                         error)
                                : Launch(io16 ? "halo_gated_norm_hz" : "halo_gated_norm_h", {T * c.ssm_num_v_heads}, {128},
                                  {rt::Arg::Buf(ck_.gdn), rt::Arg::Buf(tensors_.at(l.ssm_norm.data).buf),
                                   rt::Arg::Buf(ck_.z), rt::Arg::Buf(ck_.xh), rt::Arg::Val(c.ssm_head_dim),
                                   rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(1)},
                                  error))
                         : Launch("halo_gated_norm", {T * c.ssm_num_v_heads}, {128},
                                  {rt::Arg::Buf(ck_.gdn), rt::Arg::Buf(tensors_.at(l.ssm_norm.data).buf),
                                   rt::Arg::Buf(ck_.z), rt::Arg::Val(c.ssm_head_dim), rt::Arg::Val(c.rms_eps),
                                   rt::Arg::Val<std::uint32_t>(1)},
                                  error))
              : Launch("halo_gdn_prefill_d128_norm", {c.ssm_num_v_heads}, {128},
                {rt::Arg::Buf(ck_.conv), rt::Arg::Buf(ck_.conv, kd * 4ull), rt::Arg::Buf(ck_.conv, 2ull * kd * 4),
                 rt::Arg::Buf(ck_.alpha), rt::Arg::Buf(ck_.beta), rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf),
                 rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf), rt::Arg::Buf(s.state),
                 rt::Arg::Buf(tensors_.at(l.ssm_norm.data).buf), rt::Arg::Buf(ck_.z), rt::Arg::Buf(ck_.gdn),
                 rt::Arg::Val(c.ssm_num_k_heads), rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(1),
                 rt::Arg::Val(T), rt::Arg::Val(ch), rt::Arg::Val(ch), rt::Arg::Val(vd)},
                error)) &&
         // The split path's gated norm left its output in fp16 only (in xh).
         Gemm(l.ssm_out, ck_.gdn, ck_.block_out, T, error,
              prefill_split_gdn_ && !prefill_exact_projections_ && F16Route(l.ssm_out) ? &ck_.xh : nullptr);
}

bool FlashNextGpu::AttentionChunk(const model::LayerWeights& l, AttnState& s, std::uint32_t T, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t nh = c.num_heads, nkv = c.num_kv_heads, hd = c.head_dim, idim = c.indexer_head_dim;
  const std::uint32_t qdim = c.AttentionQDim(), kvdim = c.AttentionKvDim(), ih = c.indexer_heads;
  const std::uint32_t ratio = c.compress_ratio, budget = c.indexer_top_k / ratio;
  const std::uint32_t max_blocks = max_context_ / ratio + 1;
  const auto norm = [&](const rt::Buffer& x, const model::TensorRef& w, std::uint32_t nvec, std::uint32_t dim) {
    return Launch("halo_rmsnorm", {nvec}, {kBlock},
                  {rt::Arg::Buf(x), rt::Arg::Buf(tensors_.at(w.data).buf), rt::Arg::Buf(x), rt::Arg::Val(dim),
                   rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(0)},
                  error);
  };
  // The q rows split, normalized and rotated in one pass (the batched path's rope then skips q).
  q_roped_ = prefill_q_fused_ && prefill_batched_attention_ && hd == 256 && c.rotary_dim <= hd &&
             c.rotary_dim % 2 == 0;
  // The flash kernel (AttentionBatched's dense branch, same condition) reads the gate in place from qg.
  {
    const std::uint32_t budget_ = c.indexer_top_k / c.compress_ratio;
    const bool wmma_ = prefill_wmma_attention_ && hd == 256 && nh % c.num_kv_heads == 0 && nh / c.num_kv_heads <= 16;
    gate_in_qg_ = q_roped_ && prefill_gate_in_place_ && prefill_f16_out_ && F16Route(l.attn_out) && wmma_ &&
                  prefill_mq_attention_ && prefill_fa_attention_ && prefill_kv16_ && s.k16 &&
                  (position_ + T) / c.compress_ratio <= budget_;
  }
  // k (with its cache append) and the indexer q: norm and rope in one pass each.
  kq_roped_ = q_roped_ && prefill_kq_fused_ && hd <= 256 && idim <= 256 && c.rotary_dim <= idim;
  const auto NormRope = [&](const rt::Buffer& x, const model::TensorRef& w, std::uint32_t heads, std::uint32_t dim,
                            std::uint32_t nt, const rt::Buffer* cache) {
    return Launch("halo_norm_rope_rows", {nt * heads}, {kBlock},
                  {rt::Arg::Buf(x), rt::Arg::Buf(tensors_.at(w.data).buf), rt::Arg::Val(c.rms_eps), rt::Arg::Val(dim),
                   rt::Arg::Val(c.rotary_dim), rt::Arg::Val(position_), rt::Arg::Val(heads), rt::Arg::Val(c.rope_theta),
                   cache ? rt::Arg::Buf(*cache, std::size_t{position_} * heads * dim * 4) : rt::Arg::Val<std::uint64_t>(0)},
                  error);
  };
  if (!MixedHalf(T, error) || !Gemm(l.attn_q, ck_.mixed, ck_.qg, T, error, MixedH()) ||
      !Gemm(l.attn_k, ck_.mixed, ck_.k, T, error, MixedH()) || !Gemm(l.attn_v, ck_.mixed, ck_.v, T, error, MixedH()) || !Gemm(l.indexer_q, ck_.mixed, ck_.iq, T, error) ||
      !Gemm(l.indexer_k, ck_.mixed, ck_.ik, T, error) ||
      !(q_roped_ ? Launch("halo_split_qgate_norm_rope", {T * nh}, {hd},
                          {rt::Arg::Buf(ck_.qg), rt::Arg::Buf(ck_.q),
                           gate_in_qg_ ? rt::Arg::Val<std::uint64_t>(0) : rt::Arg::Buf(ck_.gate_a),
                           rt::Arg::Buf(tensors_.at(l.attn_q_norm.data).buf), rt::Arg::Val(c.rms_eps),
                           rt::Arg::Val(c.rotary_dim), rt::Arg::Val(position_), rt::Arg::Val(nh),
                           rt::Arg::Val(c.rope_theta)},
                          error)
                 : Launch("halo_split_qgate", {T * nh}, {kBlock},
                          {rt::Arg::Buf(ck_.qg), rt::Arg::Buf(ck_.q), rt::Arg::Buf(ck_.gate_a), rt::Arg::Val(hd)},
                          error) &&
                       norm(ck_.q, l.attn_q_norm, T * nh, hd)) ||
      !(kq_roped_ ? NormRope(ck_.k, l.attn_k_norm, nkv, hd, T, &s.k) : norm(ck_.k, l.attn_k_norm, T * nkv, hd)) ||
      !(kq_roped_ ? NormRope(ck_.iq, l.indexer_q_norm, ih, idim, T, nullptr) : norm(ck_.iq, l.indexer_q_norm, T * ih, idim)))
    return false;
  if (prefill_batched_attention_) return AttentionBatched(l, s, T, error);
  for (std::uint32_t t = 0; t < T; ++t) {
    const rt::Arg params = rt::Arg::Buf(ck_.host, ck_.params_offset + std::size_t{t} * 64);
    const std::size_t qo = std::size_t{t} * qdim * 4, ko = std::size_t{t} * kvdim * 4;
    const std::size_t io = std::size_t{t} * ih * idim * 4, iko = std::size_t{t} * idim * 4;
    const auto rope = [&](const rt::Buffer& x, std::size_t off, std::uint32_t heads, std::uint32_t dim,
                          std::uint32_t slot, std::uint32_t scale) {
      return Launch("halo_rope_neox_p", {heads}, {64},
                    {rt::Arg::Buf(x, off), params, rt::Arg::Val(dim), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(slot),
                     rt::Arg::Val(scale), rt::Arg::Val(c.rope_theta)},
                    error);
    };
    const auto append = [&](const rt::Buffer& cache, const rt::Buffer& row, std::size_t off, std::uint32_t slot,
                            std::uint32_t n) {
      return Launch("halo_copy_at", {Blocks(n)}, {kBlock},
                    {rt::Arg::Buf(cache), rt::Arg::Buf(row, off), params, rt::Arg::Val(slot), rt::Arg::Val(n),
                     rt::Arg::Val(n)},
                    error);
    };
    if (!rope(ck_.q, qo, nh, hd, kHaloPosition, 1) || !rope(ck_.k, ko, nkv, hd, kHaloPosition, 1) ||
        !append(s.k, ck_.k, ko, kHaloPosition, kvdim) || !append(s.v, ck_.v, ko, kHaloPosition, kvdim) ||
        !append(s.index_k, ck_.ik, iko, kHaloPosition, idim) || !rope(ck_.iq, io, ih, idim, kHaloPosition, 1) ||
        !Launch("halo_pool_block", {1}, {128},
                {rt::Arg::Buf(s.index_k), rt::Arg::Buf(pooled_), params, rt::Arg::Val(ratio), rt::Arg::Val(idim)},
                error) ||
        !norm(pooled_, l.indexer_k_norm, 1, idim) || !rope(pooled_, 0, 1, idim, kHaloBlock, ratio) ||
        !append(s.block_k, pooled_, 0, kHaloBlock, idim) ||
        !Launch("halo_qsa_scores", {std::min(Blocks(max_blocks), 64u)}, {kBlock},
                {rt::Arg::Buf(ck_.iq, io), rt::Arg::Buf(s.block_k), rt::Arg::Buf(block_scores_), params,
                 rt::Arg::Val(ih), rt::Arg::Val(idim)},
                error) ||
        !Launch("halo_qsa_select", {1}, {1024},
                {rt::Arg::Buf(block_scores_), rt::Arg::Buf(visible_), params, rt::Arg::Val(budget), rt::Arg::Val(ratio)},
                error) ||
        !Launch("halo_attention_select", {nh}, {kBlock},
                {rt::Arg::Buf(ck_.q, qo), rt::Arg::Buf(s.k), rt::Arg::Buf(s.v), rt::Arg::Buf(ck_.gate_a, qo),
                 rt::Arg::Buf(visible_), rt::Arg::Buf(ck_.ctx, qo), rt::Arg::Buf(scores_), params, rt::Arg::Val(nh),
                 rt::Arg::Val(nkv), rt::Arg::Val(hd), rt::Arg::Val(max_context_)},
                error))
      return false;
  }
  return Gemm(l.attn_out, ck_.ctx, ck_.block_out, T, error);
}

// A token's visible list holds at most every position (under budget) or
// budget blocks of `ratio` plus the incomplete tail.
std::uint32_t FlashNextGpu::AttnVisibleStride() const {
  const auto& c = w_.config;
  return std::min(max_context_, (c.indexer_top_k / c.compress_ratio + 1) * c.compress_ratio);
}

// The chunk's attention in batched launches, each the decode kernel's body
// per token with that token's step parameters, so the arithmetic equals
// stepping: rope on all rows; K, V and indexer keys appended as contiguous
// rows; every QSA block the chunk touches pooled, normalized and rotated
// (only complete blocks are ever scored, and a block's final value is the
// one stepping leaves); then scores, selection and attention, kAttnSub
// tokens at a time (bounded scratch).
bool FlashNextGpu::AttentionBatched(const model::LayerWeights& l, AttnState& s, std::uint32_t T, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t nh = c.num_heads, nkv = c.num_kv_heads, hd = c.head_dim, idim = c.indexer_head_dim;
  const std::uint32_t qdim = c.AttentionQDim(), kvdim = c.AttentionKvDim(), ih = c.indexer_heads;
  const std::uint32_t ratio = c.compress_ratio, budget = c.indexer_top_k / ratio;
  const std::uint32_t max_blocks = max_context_ / ratio + 1, vstride = AttnVisibleStride();
  const std::uint32_t pos0 = position_;
  const bool wmma = prefill_wmma_attention_ && hd == 256 && nh % nkv == 0 && nh / nkv <= 16;
  const auto rope = [&](const rt::Buffer& x, std::uint32_t heads, std::uint32_t dim) {
    return Launch("halo_rope_neox_rows", {T * heads}, {64},
                  {rt::Arg::Buf(x), rt::Arg::Val(dim), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(pos0),
                   rt::Arg::Val(heads), rt::Arg::Val(c.rope_theta)},
                  error);
  };
  const auto append = [&](const rt::Buffer& cache, const rt::Buffer& rows, std::uint32_t n) {
    return Launch("halo_copy", {Blocks(std::size_t{T} * n)}, {kBlock},
                  {rt::Arg::Buf(cache, std::size_t{pos0} * n * 4), rt::Arg::Buf(rows), rt::Arg::Val(T * n)}, error);
  };
  const std::uint32_t b0 = pos0 / ratio, nb = (pos0 + T - 1) / ratio - b0 + 1;
  if ((!q_roped_ && !rope(ck_.q, nh, hd)) || (!kq_roped_ && (!rope(ck_.k, nkv, hd) || !rope(ck_.iq, ih, idim) ||
                                                              !append(s.k, ck_.k, kvdim))) ||
      !append(s.v, ck_.v, kvdim) || !append(s.index_k, ck_.ik, idim) ||
      !Launch("halo_pool_blocks", {nb}, {128},
              {rt::Arg::Buf(s.index_k), rt::Arg::Buf(s.block_k), rt::Arg::Val(b0), rt::Arg::Val(ratio),
               rt::Arg::Val(idim)},
              error) ||
      !Launch("halo_rmsnorm", {nb}, {kBlock},
              {rt::Arg::Buf(s.block_k, std::size_t{b0} * idim * 4), rt::Arg::Buf(tensors_.at(l.indexer_k_norm.data).buf),
               rt::Arg::Buf(s.block_k, std::size_t{b0} * idim * 4), rt::Arg::Val(idim), rt::Arg::Val(c.rms_eps),
               rt::Arg::Val<std::uint32_t>(0)},
              error) ||
      !Launch("halo_rope_neox_blocks", {nb}, {64},
              {rt::Arg::Buf(s.block_k, std::size_t{b0} * idim * 4), rt::Arg::Val(idim), rt::Arg::Val(c.rotary_dim),
               rt::Arg::Val(b0), rt::Arg::Val(ratio), rt::Arg::Val(c.rope_theta)},
              error))
    return false;
  // Dense causal (every token still sees every earlier position: QSA has
  // selected nothing away yet): 4 tokens per block share K/V, and the QSA
  // scores and selection are not needed.
  const std::uint32_t G = nh / nkv;
  // The fp16 K/V copy: convert what is not current yet (this chunk, and
  // anything decode wrote since); the flash and QSA kernels read it.
  const auto kv16_current = [&] {
    if (s.valid16 >= pos0 + T) return true;
    const std::size_t lo = std::size_t{s.valid16} * kvdim, nv = std::size_t{pos0 + T - s.valid16} * kvdim;
    for (const auto& [f, h] : {std::pair{&s.k, &s.k16}, std::pair{&s.v, &s.v16}})
      if (!Launch("halo_f16_rows", {static_cast<std::uint32_t>((nv + 255) / 256)}, {kBlock},
                  {rt::Arg::Buf(*f, lo * 4), rt::Arg::Buf(*h, lo * 2), rt::Arg::Val(static_cast<std::uint32_t>(nv))},
                  error))
        return false;
    s.valid16 = pos0 + T;
    return true;
  };
  // Dense causal, flash style: 16 tokens x 4 heads a block, each wave its
  // own rows and softmax (halo_attention_fa).
  if (wmma && prefill_mq_attention_ && prefill_fa_attention_ && prefill_kv16_ && s.k16 &&
      (pos0 + T) / ratio <= budget)
  {
    // The context in fp16 only when the out projection reads it that way.
    const bool ctx_half = prefill_f16_out_ && F16Route(l.attn_out);
    if (gate_in_qg_ && !ctx_half) {
      if (error) *error = "internal: gate left in qg but the attention kernel reads gate_a";
      return false;
    }
    return kv16_current() &&
           Launch(gate_in_qg_ ? "halo_attention_fa_hg" : ctx_half ? "halo_attention_fa_h" : "halo_attention_fa",
                  {(T + 15) / 16, nkv * ((G + 3) / 4)}, {256},
                  {rt::Arg::Buf(ck_.q), rt::Arg::Buf(s.k16), rt::Arg::Buf(s.v16),
                   rt::Arg::Buf(gate_in_qg_ ? ck_.qg : ck_.gate_a),
                   rt::Arg::Buf(ctx_half ? ck_.xh : ck_.ctx), rt::Arg::Buf(ck_.host, ck_.params_offset),
                   rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(T)},
                  error) &&
           Gemm(l.attn_out, ck_.ctx, ck_.block_out, T, error, ctx_half ? &ck_.xh : nullptr);
  }
  if (wmma && prefill_mq_attention_ && (pos0 + T) / ratio <= budget && 4 * G <= 48)
    return Launch("halo_attention_wmma_mq", {nkv, (T + 3) / 4}, {256},
                  {rt::Arg::Buf(ck_.q), rt::Arg::Buf(s.k), rt::Arg::Buf(s.v), rt::Arg::Buf(ck_.gate_a),
                   rt::Arg::Buf(ck_.ctx), rt::Arg::Buf(ck_.host, ck_.params_offset), rt::Arg::Val(nh),
                   rt::Arg::Val(nkv), rt::Arg::Val(T)},
                  error) &&
           Gemm(l.attn_out, ck_.ctx, ck_.block_out, T, error);
  // 4 tokens at a time over the union of their selections (shared K/V).
  const std::uint32_t max_u = 4 * budget + 4;
  const bool qsel = wmma && prefill_qsel_attention_ && 4 * G <= 48 && max_blocks <= 65537;
  const bool scores_wmma = prefill_scores_wmma_ && !prefill_exact_projections_ && ih == 4 && idim % 16 == 0;
  // The fp16 K/V copy: convert what is not current yet (this chunk, and
  // anything decode wrote since), then the union attention reads it.
  const bool kv16 = qsel && prefill_kv16_ && s.k16;
  if (kv16 && !kv16_current()) return false;
  for (std::uint32_t t0 = 0; t0 < T; t0 += kAttnSub) {
    const std::uint32_t n = std::min(kAttnSub, T - t0);
    const rt::Arg params = rt::Arg::Buf(ck_.host, ck_.params_offset + std::size_t{t0} * 64);
    // Block scores as one bf16x3 WMMA GEMM per sub-chunk (rows: token x head,
    // columns: the complete blocks of the sub-chunk's last token).
    const std::uint32_t cols = (pos0 + t0 + n) / ratio;
    if (!(scores_wmma && cols > 0
              ? Launch("halo_qsa_scores_wmma", {(cols + 63) / 64, (n + 15) / 16}, {128},
                       {rt::Arg::Buf(ck_.iq, std::size_t{t0} * ih * idim * 4), rt::Arg::Buf(s.block_k),
                        rt::Arg::Buf(ck_.bscores), rt::Arg::Val(n), rt::Arg::Val(cols), rt::Arg::Val(idim),
                        rt::Arg::Val(max_blocks)},
                       error)
              : Launch("halo_qsa_scores_rows", {std::min(Blocks(max_blocks), 64u), n}, {kBlock},
                       {rt::Arg::Buf(ck_.iq, std::size_t{t0} * ih * idim * 4), rt::Arg::Buf(s.block_k),
                        rt::Arg::Buf(ck_.bscores), params, rt::Arg::Val(ih), rt::Arg::Val(idim),
                        rt::Arg::Val(max_blocks)},
                       error)) ||
        !Launch("halo_qsa_select_rows", {n}, {1024},
                {rt::Arg::Buf(ck_.bscores), rt::Arg::Buf(ck_.vis), params, rt::Arg::Val(budget), rt::Arg::Val(ratio),
                 rt::Arg::Val(max_blocks), rt::Arg::Val(vstride)},
                error) ||
        !(qsel ? Launch("halo_qsa_union4", {(n + 3) / 4}, {1024},
                        {rt::Arg::Buf(ck_.vis), params, rt::Arg::Val(budget), rt::Arg::Val(ratio), rt::Arg::Val(vstride),
                         rt::Arg::Val(max_blocks), rt::Arg::Val(n), rt::Arg::Buf(ck_.ublk), rt::Arg::Buf(ck_.umask),
                         rt::Arg::Buf(ck_.ucount), rt::Arg::Val(max_u)},
                        error) &&
                     Launch(kv16 ? "halo_attention_wmma_qsel_h" : "halo_attention_wmma_qsel", {nkv, (n + 3) / 4}, {256},
                            {rt::Arg::Buf(ck_.q, std::size_t{t0} * qdim * 4), rt::Arg::Buf(kv16 ? s.k16 : s.k),
                             rt::Arg::Buf(kv16 ? s.v16 : s.v),
                             rt::Arg::Buf(ck_.gate_a, std::size_t{t0} * qdim * 4),
                             rt::Arg::Buf(ck_.ctx, std::size_t{t0} * qdim * 4), params, rt::Arg::Buf(ck_.ublk),
                             rt::Arg::Buf(ck_.umask), rt::Arg::Buf(ck_.ucount), rt::Arg::Val(max_u),
                             rt::Arg::Val(ratio), rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(n)},
                            error)
         : wmma ? Launch("halo_attention_wmma_rows", {nkv, n}, {128},
                        {rt::Arg::Buf(ck_.q, std::size_t{t0} * qdim * 4), rt::Arg::Buf(s.k), rt::Arg::Buf(s.v),
                         rt::Arg::Buf(ck_.gate_a, std::size_t{t0} * qdim * 4), rt::Arg::Buf(ck_.vis),
                         rt::Arg::Buf(ck_.ctx, std::size_t{t0} * qdim * 4), params, rt::Arg::Val(nh),
                         rt::Arg::Val(nkv), rt::Arg::Val(vstride)},
                        error)
               : Launch("halo_attention_select_rows", {nh, n}, {kBlock},
                {rt::Arg::Buf(ck_.q, std::size_t{t0} * qdim * 4), rt::Arg::Buf(s.k), rt::Arg::Buf(s.v),
                 rt::Arg::Buf(ck_.gate_a, std::size_t{t0} * qdim * 4), rt::Arg::Buf(ck_.vis),
                 rt::Arg::Buf(ck_.ctx, std::size_t{t0} * qdim * 4), rt::Arg::Buf(ck_.ascores), params,
                 rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(hd), rt::Arg::Val(vstride)},
                error)))
      return false;
  }
  return Gemm(l.attn_out, ck_.ctx, ck_.block_out, T, error);
}

bool FlashNextGpu::MoeChunk(const model::LayerWeights& l, std::uint32_t T, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t H = c.hidden_size, used = c.num_experts_used, E = c.num_experts;
  const auto& gate = l.ffn_gate_exps;
  const auto& up = l.ffn_up_exps;
  const auto& down = l.ffn_down_exps;
  const auto& dd = tensors_.at(down.data);
  const auto ff = static_cast<std::uint32_t>(gate.rows);
  // The router on bf16 WMMA with hi/lo splitting (~2^-15 per product)
  // instead of the fp32 VALU GEMM.
  const auto& rt_ = tensors_.at(l.router.data);
  const auto rrows = static_cast<std::uint32_t>(l.router.rows), rcols = static_cast<std::uint32_t>(l.router.cols);
  const bool rf16 = prefill_router_f16_ != 0 && prefill_f16_gemm_ && !prefill_exact_projections_ &&
                    !lora_.contains(l.router.data) && l.router.type == core::GgmlType::kF32 && rcols % 128 == 0;
  const bool bx3 = !rf16 && prefill_router_bf16x3_ && !prefill_exact_projections_ && !lora_.contains(l.router.data) &&
                   l.router.type == core::GgmlType::kF32 && rcols % 32 == 0;
  const int rmode = prefill_router_f16_;
  if (rf16 && rmode == 4) {
    if (!router_hl_ok_) {
      if (error) *error = "internal: router mode 4 without the mix's fp16 hi / lo";
      return false;
    }
    router_hl_ok_ = false;
    auto it16 = router16_.find(l.router.data);
    if (it16 == router16_.end()) {
      auto b = dev_.Alloc(std::size_t{rrows} * rcols * 2, error);
      if (!b) return false;
      owned_.push_back(b);
      if (!Half(rt_.buf, b, rrows * rcols, error)) return false;
      it16 = router16_.emplace(l.router.data, b).first;
    }
    const int tile = prefill_router_tile_;
    const std::uint32_t br = tile == 2 ? 64 : 128;
    if (!Launch(tile == 0 ? "halo_gemm_h_hl_128x64" : tile == 1 ? "halo_gemm_h_hl_128x64k32" : "halo_gemm_h_hl_64x64",
                {(T + 63) / 64, (rrows + br - 1) / br}, {tile == 2 ? 128u : 256u},
                {rt::Arg::Buf(it16->second), rt::Arg::Buf(ck_.mh), rt::Arg::Buf(ck_.ml), rt::Arg::Buf(ck_.router),
                 rt::Arg::Val(rrows), rt::Arg::Val(rcols), rt::Arg::Val(T)},
                error))
      return false;
  } else if (rf16 && !((rmode == 3 || MixedHalf(T, error)) &&
                Launch(rmode == 2 ? "halo_gemm_f32h_f16_w2" : rmode == 3 ? "halo_gemm_f32_f16_x2" : "halo_gemm_f32h_f16_k128",
                       {(T + 63) / 64, (rrows + 63) / 64}, {128},
                       {rt::Arg::Buf(rt_.buf), rt::Arg::Buf(rmode == 3 ? ck_.mixed : ck_.mh), rt::Arg::Buf(ck_.router),
                        rt::Arg::Val(rrows), rt::Arg::Val(rcols), rt::Arg::Val(T)},
                       error)))
    return false;
  if (!rf16 && !(bx3 ? Launch("halo_gemm_f32_bf16x3", {(T + 63) / 64, (rrows + 63) / 64}, {256},
                     {rt::Arg::Buf(rt_.buf), rt::Arg::Buf(ck_.mixed), rt::Arg::Buf(ck_.router), rt::Arg::Val(rrows),
                      rt::Arg::Val(rcols), rt::Arg::Val(T)},
                     error)
            : Gemm(l.router, ck_.mixed, ck_.router, T, error)))
    return false;
  const bool gate_up_fast = fused_moe_ && gate.type == core::GgmlType::kQ4_K && up.type == gate.type &&
                            up.rows == gate.rows && up.cols == gate.cols && gate.cols % 256 == 0 && H <= 4096;
  const bool down_q5_1 = down.type == core::GgmlType::kQ5_1 && down.cols % 32 == 0 && ff <= 2048;
  const bool fast = gate_up_fast && (down_q5_1 || static_cast<bool>(dd.qs));
  if (prefill_grouped_moe_ && GroupedMoeFits(l))
    return GroupedMoe(l, T, error, !prefill_fused_moe_combine_) &&
           SharedExpertChunk(l, T, error, prefill_fused_moe_combine_);
  const char* gu = int8_act_moe_ ? "halo_moe_gate_up_q4_k_dot" : "halo_moe_gate_up_q4_k_s2";
  for (std::uint32_t t = 0; t < T; ++t) {
    const std::size_t ido = std::size_t{t} * used * 4, xo = std::size_t{t} * H * 4;
    if (!fast) {
      // Other expert formats: the step's unfused loop on the decode buffers.
      if (!Launch("halo_router_topk", {1}, {kBlock},
                  {rt::Arg::Buf(ck_.router, std::size_t{t} * E * 4), rt::Arg::Buf(expert_ids_),
                   rt::Arg::Buf(expert_weights_), rt::Arg::Val(E), rt::Arg::Val(used)},
                  error) ||
          !Launch("halo_copy", {Blocks(H)}, {kBlock},
                  {rt::Arg::Buf(mixed_), rt::Arg::Buf(ck_.mixed, xo), rt::Arg::Val(H)}, error) ||
          !dev_.Fill(block_out_, 0, error))
        return false;
      for (std::uint32_t i = 0; i < used; ++i)
        if (!ExpertMatVec(gate, i, mixed_, eg_, error) || !ExpertMatVec(up, i, mixed_, eu_, error) ||
            !Launch("halo_silu_mul", {Blocks(c.expert_ff)}, {kBlock},
                    {rt::Arg::Buf(eg_), rt::Arg::Buf(eu_), rt::Arg::Val(c.expert_ff)}, error) ||
            !ExpertMatVec(down, i, eg_, ed_, error) ||
            !Launch("halo_axpy_slot", {Blocks(H)}, {kBlock},
                    {rt::Arg::Buf(block_out_), rt::Arg::Buf(ed_), rt::Arg::Buf(expert_weights_), rt::Arg::Val(i),
                     rt::Arg::Val(H)},
                    error))
          return false;
      if (!Launch("halo_copy", {Blocks(H)}, {kBlock},
                  {rt::Arg::Buf(ck_.block_out, xo), rt::Arg::Buf(block_out_), rt::Arg::Val(H)}, error))
        return false;
      continue;
    }
    if (!Launch("halo_router_topk", {1}, {kBlock},
                {rt::Arg::Buf(ck_.router, std::size_t{t} * E * 4), rt::Arg::Buf(ck_.ids, ido),
                 rt::Arg::Buf(ck_.wts, ido), rt::Arg::Val(E), rt::Arg::Val(used)},
                error) ||
        !Launch(gu, {Blocks(ff, 16), used}, {256},
                {rt::Arg::Buf(tensors_.at(gate.data).buf), rt::Arg::Buf(tensors_.at(up.data).buf),
                 rt::Arg::Buf(ck_.ids, ido), rt::Arg::Buf(ck_.mixed, xo), rt::Arg::Buf(moe_h_),
                 rt::Arg::Val<std::uint64_t>(gate.RowBytes() * gate.rows), rt::Arg::Val(ff), rt::Arg::Val(H),
                 rt::Arg::Val<std::uint64_t>(gate.RowBytes())},
                error))
      return false;
    const bool down_ok =
        down_q5_1 ? Launch("halo_moe_down_q5_1_r4", {Blocks(H, 32), used}, {256},
                           {rt::Arg::Buf(dd.buf), rt::Arg::Buf(ck_.ids, ido), rt::Arg::Buf(moe_h_),
                            rt::Arg::Buf(moe_part_), rt::Arg::Val<std::uint64_t>(down.RowBytes() * down.rows),
                            rt::Arg::Val(H), rt::Arg::Val(ff), rt::Arg::Val<std::uint64_t>(down.RowBytes())},
                           error)
                  : Launch("halo_moe_down_q8r_r4", {Blocks(H, 32), used}, {256},
                           {rt::Arg::Buf(dd.qs), rt::Arg::Buf(dd.d), rt::Arg::Buf(ck_.ids, ido), rt::Arg::Buf(moe_h_),
                            rt::Arg::Buf(moe_part_), rt::Arg::Val(H), rt::Arg::Val(ff), rt::Arg::Val(dd.qstride)},
                           error);
    if (!down_ok || !Launch("halo_moe_combine", {Blocks(H)}, {kBlock},
                            {rt::Arg::Buf(moe_part_), rt::Arg::Buf(ck_.wts, ido), rt::Arg::Buf(ck_.block_out, xo),
                             rt::Arg::Val(used), rt::Arg::Val(H)},
                            error))
      return false;
  }
  return SharedExpertChunk(l, T, error);
}

namespace {
// Kernel-name suffix of a grouped expert GEMM for a weight type, or null.
const char* GroupedSuffix(core::GgmlType t) {
  switch (t) {
    case core::GgmlType::kQ4_K: return "q4_k";
    case core::GgmlType::kQ5_K: return "q5_k";
    case core::GgmlType::kQ6_K: return "q6_k";
    case core::GgmlType::kQ5_1: return "q5_1";
    case core::GgmlType::kQ8_0: return "q8_0";
    case core::GgmlType::kF32: return "f32";
    default: return nullptr;
  }
}
}  // namespace

// Pairs sorted by expert (pair order within one), positions and tiles: one
// block, or (prefill_parallel_bucket_) three parallel passes with the same
// output.
bool FlashNextGpu::Bucket(std::uint32_t pairs, std::uint32_t E, std::string* error) {
  if (!prefill_parallel_bucket_)
    return Launch("halo_moe_bucket", {1}, {1024},
                  {rt::Arg::Buf(ck_.ids), rt::Arg::Val(pairs), rt::Arg::Val(E), rt::Arg::Buf(ck_.sorted),
                   rt::Arg::Buf(ck_.pos), rt::Arg::Buf(ck_.tiles), rt::Arg::Buf(ck_.ntiles)},
                  error);
  const std::uint32_t chunks = (pairs + 1023) / 1024;
  return Launch("halo_moe_bucket_count", {chunks}, {1024},
                {rt::Arg::Buf(ck_.ids), rt::Arg::Val(pairs), rt::Arg::Val(E), rt::Arg::Buf(ck_.bcounts)}, error) &&
         Launch("halo_moe_bucket_scan", {1}, {1024},
                {rt::Arg::Buf(ck_.bcounts), rt::Arg::Val(chunks), rt::Arg::Val(E), rt::Arg::Buf(ck_.tiles),
                 rt::Arg::Buf(ck_.ntiles), rt::Arg::Buf(ck_.bbase)},
                error) &&
         Launch("halo_moe_bucket_scatter", {chunks}, {1024},
                {rt::Arg::Buf(ck_.ids), rt::Arg::Val(pairs), rt::Arg::Val(E), rt::Arg::Buf(ck_.bcounts),
                 rt::Arg::Buf(ck_.bbase), rt::Arg::Buf(ck_.sorted), rt::Arg::Buf(ck_.pos)},
                error);
}

bool FlashNextGpu::GroupedMoeFits(const model::LayerWeights& l) const {
  const auto& c = w_.config;
  const auto& gate = l.ffn_gate_exps;
  const auto& up = l.ffn_up_exps;
  const auto& down = l.ffn_down_exps;
  return GroupedSuffix(gate.type) && GroupedSuffix(down.type) && up.type == gate.type && up.rows == gate.rows &&
         up.cols == gate.cols && !tensors_.at(gate.data).qs && !tensors_.at(up.data).qs &&
         (!tensors_.at(down.data).qs || (down.type == core::GgmlType::kQ8_0 && down.cols % 32 == 0)) &&
         c.hidden_size % 256 == 0 && gate.cols == c.hidden_size && gate.rows % 16 == 0 &&
         c.num_experts <= 1024 && down.cols == gate.rows && down.rows == c.hidden_size;
}

// Routed experts grouped by expert (stage B): top-k for every token,
// bucketing, gate/up with SwiGLU, down, weighted combine; five launches.
bool FlashNextGpu::GroupedMoe(const model::LayerWeights& l, std::uint32_t T, std::string* error, bool combine) {
  const auto& c = w_.config;
  const std::uint32_t H = c.hidden_size, used = c.num_experts_used, E = c.num_experts;
  const auto& gate = l.ffn_gate_exps;
  const auto& down = l.ffn_down_exps;
  const auto ff = static_cast<std::uint32_t>(gate.rows);
  const auto& tg = tensors_.at(gate.data);
  const auto& tu = tensors_.at(l.ffn_up_exps.data);
  const auto& td = tensors_.at(down.data);
  // Prototype (prefill_moe_blocked_): Q4_K gate/up repacked K-blocked on first use, via a temporary buffer.
  const bool blocked = prefill_moe_blocked_ && gate.type == core::GgmlType::kQ4_K && ff % kMoeBlockRows == 0 &&
                       H % 256 == 0;
  if (blocked) {
    for (const auto* t : {&gate, &l.ffn_up_exps}) {
      if (moe_blocked_.contains(t->data)) continue;
      const std::size_t bytes = t->SizeBytes();
      if (!moe_blocked_tmp_ || moe_blocked_tmp_.bytes < bytes) {
        if (moe_blocked_tmp_) dev_.Free(moe_blocked_tmp_);
        moe_blocked_tmp_ = dev_.Alloc(bytes, error);
        if (!moe_blocked_tmp_) return false;
      }
      const auto& dt = tensors_.at(t->data);
      const auto rows = static_cast<std::uint32_t>(t->rows * t->experts);
      const auto rb = static_cast<std::uint32_t>(t->RowBytes());
      if (!Launch("halo_repack_rows_blocked", {static_cast<std::uint32_t>((bytes / 16 + 255) / 256)}, {256},
                  {rt::Arg::Buf(dt.buf), rt::Arg::Buf(moe_blocked_tmp_), rt::Arg::Val(rows), rt::Arg::Val(rb),
                   rt::Arg::Val(144u)},
                  error) ||
          !Launch("halo_copy", {4096}, {256},
                  {rt::Arg::Buf(dt.buf), rt::Arg::Buf(moe_blocked_tmp_), rt::Arg::Val(static_cast<std::uint32_t>(bytes / 4))},
                  error))
        return false;
      moe_blocked_.insert(t->data);
    }
  }
  const std::string gu = std::string("halo_moe_gate_up_grouped_") + (blocked ? "q4_kb" : GroupedSuffix(gate.type));
  const std::string dn = std::string("halo_moe_down_grouped_") + GroupedSuffix(down.type);
  const std::uint32_t pairs = T * used, max_tiles = pairs / 64 + E + 1;
  return Launch("halo_router_topk_rows", {T}, {kBlock},
                {rt::Arg::Buf(ck_.router), rt::Arg::Buf(ck_.ids), rt::Arg::Buf(ck_.wts), rt::Arg::Val(E),
                 rt::Arg::Val(used)},
                error) &&
         Bucket(pairs, E, error) &&
         Launch("halo_f16_perm_rows", {Blocks(std::size_t{T} * H)}, {kBlock},
                {rt::Arg::Buf(ck_.mixed), rt::Arg::Buf(ck_.x16), rt::Arg::Val(T * H)}, error) &&
         Launch(gu.c_str(), {(ff + kMoeBlockRows - 1) / kMoeBlockRows, max_tiles}, {256},
                {rt::Arg::Buf(tg.buf), rt::Arg::Buf(tu.buf), rt::Arg::Val<std::uint64_t>(gate.RowBytes() * gate.rows),
                 rt::Arg::Val<std::uint64_t>(gate.RowBytes()), rt::Arg::Buf(ck_.x16), rt::Arg::Buf(ck_.sorted),
                 rt::Arg::Buf(ck_.tiles), rt::Arg::Buf(ck_.ntiles), rt::Arg::Buf(ck_.hp), rt::Arg::Val(ff),
                 rt::Arg::Val(H), rt::Arg::Val(used)},
                error) &&
         (td.qs ? Launch("halo_moe_down_grouped_q8r", {(H + kMoeBlockRows - 1) / kMoeBlockRows, max_tiles}, {256},
                         {rt::Arg::Buf(td.qs), rt::Arg::Buf(td.d), rt::Arg::Val(td.qstride), rt::Arg::Buf(ck_.hp),
                          rt::Arg::Buf(ck_.tiles), rt::Arg::Buf(ck_.ntiles), rt::Arg::Buf(ck_.yp), rt::Arg::Val(H),
                          rt::Arg::Val(ff)},
                         error)
                : Launch(dn.c_str(), {(H + kMoeBlockRows - 1) / kMoeBlockRows, max_tiles}, {256},
                         {rt::Arg::Buf(td.buf), rt::Arg::Val<std::uint64_t>(down.RowBytes() * down.rows),
                          rt::Arg::Val<std::uint64_t>(down.RowBytes()), rt::Arg::Buf(ck_.hp), rt::Arg::Buf(ck_.tiles),
                          rt::Arg::Buf(ck_.ntiles), rt::Arg::Buf(ck_.yp), rt::Arg::Val(H), rt::Arg::Val(ff)},
                         error)) &&
         (!combine ||  // else SharedExpertChunk combines (fused_combine)
          Launch("halo_moe_combine_grouped", {Blocks(H), T}, {kBlock},
                 {rt::Arg::Buf(ck_.yp), rt::Arg::Buf(ck_.pos), rt::Arg::Buf(ck_.wts), rt::Arg::Buf(ck_.block_out),
                  rt::Arg::Val(used), rt::Arg::Val(H)},
                 error));
}

bool FlashNextGpu::SharedExpertChunk(const model::LayerWeights& l, std::uint32_t T, std::string* error,
                                     bool fused_combine) {
  const auto& c = w_.config;
  const std::uint32_t H = c.hidden_size;
  const std::uint32_t sff = c.shared_expert_ff;
  // Precision option: gate and up as W8A8 (activations quantized per token and 32 values).
  const auto w8a8 = [&](const model::TensorRef& t) {
    const auto& d = tensors_.at(t.data);
    return prefill_shexp_w8a8_ && !prefill_exact_projections_ && !lora_.contains(t.data) && d.qs &&
           t.cols % 64 == 0 && d.qstride % 16 == 0;
  };
  const auto q8a8 = [&](const model::TensorRef& t, const rt::Buffer& y) {
    const auto& d = tensors_.at(t.data);
    const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
    return Launch("halo_gemm_q8a8_128x128", {(T + 127) / 128, (rows + 127) / 128}, {256},
                  {rt::Arg::Buf(d.qs), rt::Arg::Buf(d.d), rt::Arg::Buf(ck_.xq), rt::Arg::Buf(ck_.xd), rt::Arg::Buf(y),
                   rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val(d.qstride), rt::Arg::Val(T)},
                  error);
  };
  const bool a8 = w8a8(l.shexp_gate) && w8a8(l.shexp_up) && l.shexp_gate.cols == H && l.shexp_up.cols == H;
  const bool gate_up_ok =
      a8 ? Launch("halo_quantize_q8_kblocked", {T}, {256},
                  {rt::Arg::Buf(ck_.mixed), rt::Arg::Buf(ck_.xq), rt::Arg::Buf(ck_.xd), rt::Arg::Val(H), rt::Arg::Val(T)},
                  error) &&
               q8a8(l.shexp_gate, ck_.eg) && q8a8(l.shexp_up, ck_.eu)
         : MixedHalf(T, error) && Gemm(l.shexp_gate, ck_.mixed, ck_.eg, T, error, MixedH()) &&
               Gemm(l.shexp_up, ck_.mixed, ck_.eu, T, error, MixedH());
  const bool eg_half = prefill_f16_out_ && F16Route(l.shexp_down);  // the SiLU product written in fp16 only
  return gate_up_ok &&
         (eg_half ? Launch("halo_silu_mul_h", {Blocks(std::size_t{T} * sff)}, {kBlock},
                           {rt::Arg::Buf(ck_.eg), rt::Arg::Buf(ck_.eu), rt::Arg::Buf(ck_.xh), rt::Arg::Val(T * sff)},
                           error)
                  : Launch("halo_silu_mul", {Blocks(std::size_t{T} * sff)}, {kBlock},
                           {rt::Arg::Buf(ck_.eg), rt::Arg::Buf(ck_.eu), rt::Arg::Val(T * sff)}, error)) &&
         Gemm(l.shexp_down, ck_.eg, ck_.ed, T, error, eg_half ? &ck_.xh : nullptr) &&
         MixedGemmF16(l.shexp_gate_inp, ck_.sgate, T, error) &&
         (fused_combine && moe_defer_ ? (moe_pending_ = true)  // summed by the combine_norm that follows
          : fused_combine  // the routed experts' weighted sum and this expert in one pass
              ? Launch("halo_moe_combine_shared", {Blocks(H), T}, {kBlock},
                       {rt::Arg::Buf(ck_.yp), rt::Arg::Buf(ck_.pos), rt::Arg::Buf(ck_.wts), rt::Arg::Buf(ck_.ed),
                        rt::Arg::Buf(ck_.sgate), rt::Arg::Buf(ck_.block_out),
                        rt::Arg::Val(w_.config.num_experts_used), rt::Arg::Val(H)},
                       error)
              : Launch("halo_axpy_sigmoid_rows", {Blocks(std::size_t{T} * H)}, {kBlock},
                       {rt::Arg::Buf(ck_.block_out), rt::Arg::Buf(ck_.ed), rt::Arg::Buf(ck_.sgate), rt::Arg::Val(H),
                        rt::Arg::Val(T)},
                       error));
}

bool FlashNextGpu::PrefillChunk(std::span<const std::int32_t> tokens, std::span<float> logits, std::string* error,
                                const std::vector<float>* pre_emb,
                                const std::function<void(const model::NgramHistory&)>& after_prep) {
  const auto& c = w_.config;
  const auto T = static_cast<std::uint32_t>(tokens.size());
  if (T == 0) return true;
  if (position_ + T > max_context_) {
    if (error) *error = "context is full";
    return false;
  }
  for (auto t : tokens)
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.vocab_size) {
      if (error) *error = "token out of range";
      return false;
    }
  if (!AllocChunk(error)) return false;
  // HALO_PREFILL_TIMING: host preparation, upload and device time per chunk.
  static const bool timing = std::getenv("HALO_PREFILL_TIMING") != nullptr;
  const auto t_start = std::chrono::steady_clock::now();
  auto t_hash = t_start, t_read = t_start;
  // Host side: every token's n-gram rows and step parameters, one upload.
  model::NgramHistory history = ngram_history_;
  const std::uint32_t ratio = c.compress_ratio, budget = c.indexer_top_k / ratio;
  // With the PLE past layer 0 and no prefetched rows, the n-gram read (tens
  // of ms from disk) waits until the layers before the PLE are queued, so it
  // runs while the GPU works on them.
  const std::span<float> ple_emb(reinterpret_cast<float*>(ck_host_.data()),
                             c.ple_layer >= 0 ? std::size_t{T} * c.PleEmbeddingDim() : 0);
  const bool defer_read = c.ple_layer > 0 && prefill_ple_overlap_ && !(pre_emb && pre_emb->size() == ple_emb.size());
  std::vector<std::uint32_t> rows;
  double deferred_read_ms = 0;
  if (c.ple_layer >= 0) {
    // One sorted, batched read for the chunk's rows.
    rows.resize(std::size_t{T} * c.ple_heads);
    model::HashNgramRows(c, history, tokens, rows);
    t_hash = std::chrono::steady_clock::now();
    if (pre_emb && pre_emb->size() == ple_emb.size()) {
      std::memcpy(ple_emb.data(), pre_emb->data(), ple_emb.size_bytes());
    } else if (!defer_read && (!ngram_ || !ngram_->Read(rows, ple_emb))) {
      if (error) *error = "n-gram table read failed";
      return false;
    }
    t_read = std::chrono::steady_clock::now();
  }
  const auto read_ple = [&] {
    const auto t0 = std::chrono::steady_clock::now();
    if (!ngram_ || !ngram_->Read(rows, ple_emb)) {
      if (error) *error = "n-gram table read failed";
      return false;
    }
    deferred_read_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!dev_.Upload(ck_.host, 0, ple_emb.data(), ple_emb.size_bytes(), error)) return false;
    if (after_prep) after_prep(history);
    return true;
  };
  for (std::uint32_t t = 0; t < T; ++t) {
    const std::uint32_t pos = position_ + t, n_kv = pos + 1, complete = n_kv / ratio;
    std::uint32_t params[kHaloStepParamCount] = {};
    params[kHaloToken] = static_cast<std::uint32_t>(tokens[t]);
    params[kHaloPosition] = pos;
    params[kHaloNkv] = n_kv;
    params[kHaloComplete] = complete;
    params[kHaloBlock] = pos / ratio;
    params[kHaloNVisible] = complete <= budget ? n_kv : budget * ratio + (n_kv - complete * ratio);
    std::memcpy(ck_host_.data() + ck_.params_offset + std::size_t{t} * 64, params, sizeof(params));
  }
  const auto t_prep = std::chrono::steady_clock::now();
  if (!(defer_read ? dev_.Upload(ck_.host, ck_.params_offset, ck_host_.data() + ck_.params_offset, std::size_t{T} * 64,
                                 error)
                   : dev_.Upload(ck_.host, 0, ck_host_.data(), ck_.params_offset + std::size_t{T} * 64, error)))
    return false;
  const auto t_upload = std::chrono::steady_clock::now();
  // after_prep may start the next chunk's n-gram prefetch; the table serves
  // one read at a time, so with the read deferred it starts after that read.
  if (after_prep && !defer_read) after_prep(history);

  const std::uint32_t H = c.hidden_size, D = c.HcDim();
  const auto& emb = tensors_.at(w_.token_embd.data);
  {  // all T rows in one launch (a launch per token cost ~45 ms per 2048 tokens)
    const rt::Arg eargs[] = {rt::Arg::Buf(emb.buf), rt::Arg::Buf(ck_.embd),
                             rt::Arg::Buf(ck_.host, ck_.params_offset), rt::Arg::Val(H),
                             rt::Arg::Val<std::uint64_t>(w_.token_embd.RowBytes())};
    if (!dev_.Launch(emb.embed_rows, {Blocks(H), T}, {kBlock}, eargs, error)) return false;
  }
  if (timing_) Stamp("token embedding (per token)");
  if (!Launch(Rk("halo_broadcast_rows", prefill_res16_).c_str(), {Blocks(std::size_t{T} * D)}, {kBlock},
              {rt::Arg::Buf(ck_.res), rt::Arg::Buf(ck_.embd), rt::Arg::Val(c.hc_count), rt::Arg::Val(H),
               rt::Arg::Val(T)},
              error))
    return false;
  const auto combine = [&] {
    return Launch(Rk("halo_hc_combine_rows", prefill_res16_).c_str(), {Blocks(std::size_t{T} * D)}, {kBlock},
                  {rt::Arg::Buf(ck_.res), rt::Arg::Buf(ck_.block_out), rt::Arg::Buf(ck_.inject),
                   rt::Arg::Val(c.hc_count), rt::Arg::Val(H), rt::Arg::Val(T)},
                  error);
  };
  // The combine after a block and the next mixer's norm run as one pass
  // (prefill_fused_combine_), except where PLE changes the residual between
  // them; `normed` carries "the next mixer's norm is done".
  const auto combine_norm = [&](const model::HcMixer& next) {
    const bool h16 = F16Route(next.down);
    // xn's readers when it is skipped: the down GEMM takes xh (h16), the
    // inject comes from the partials (fused), and the mix epilogue rebuilds it.
    xn_skipped_ = prefill_mix_rebuild_ && h16 && FuseInject(next) && MixFused(next) && c.hc_count == 4;
    const bool moe = moe_pending_;
    moe_pending_ = false;
    if (moe || TokCombine(next)) {
      static const bool probe_noinject = std::getenv("HALO_PROBE_NOINJECT") != nullptr;  // PROBE (wrong results)
      const auto irows = probe_noinject ? 0u : static_cast<std::uint32_t>(next.inject.rows);
      const auto norm = rt::Arg::Buf(tensors_.at(next.norm.data).buf), iw = rt::Arg::Buf(tensors_.at(next.inject.data).buf);
      const auto blocked = rt::Arg::Val(BlockedXh(next) ? T : 0u);
      if (moe)
        return Launch(Rk(prefill_moe_vec_ && H % 8 == 0 ? "halo_hc_combine_norm_tok_moev" : "halo_hc_combine_norm_tok_moe",
                         prefill_res16_).c_str(),
                      {T}, {kBlock},
                      {rt::Arg::Buf(ck_.res), rt::Arg::Buf(ck_.yp), rt::Arg::Buf(ck_.pos), rt::Arg::Buf(ck_.wts),
                       rt::Arg::Buf(ck_.ed), rt::Arg::Buf(ck_.sgate), rt::Arg::Val(c.num_experts_used),
                       rt::Arg::Buf(ck_.inject), norm, rt::Arg::Buf(ck_.xh), rt::Arg::Val(H), rt::Arg::Val(c.rms_eps),
                       iw, rt::Arg::Buf(ck_.ipart), rt::Arg::Val(irows), rt::Arg::Buf(ck_.rscale), blocked},
                      error);
      return Launch(Rk("halo_hc_combine_norm_tok", prefill_res16_).c_str(), {T}, {kBlock},
                    {rt::Arg::Buf(ck_.res), rt::Arg::Buf(ck_.block_out), rt::Arg::Buf(ck_.inject), norm,
                     rt::Arg::Buf(ck_.xh), rt::Arg::Val(H), rt::Arg::Val(c.rms_eps), iw, rt::Arg::Buf(ck_.ipart),
                     rt::Arg::Val(irows), rt::Arg::Buf(ck_.rscale), blocked},
                    error);
    }
    // Precision option: the down projection as W8A8 reads an int8 copy of xn written here.
    const auto& dd = tensors_.at(next.down.data);
    xq_valid_ = prefill_down_w8a8_ && prefill_res16_ == 0 && h16 && BlockedXh(next) && H % 32 == 0 && dd.qs &&
                !lora_.contains(next.down.data);
    const bool want16 = h16 && (!xq_valid_ || prefill_mix_h16_);  // xh's readers: the fp16 down GEMM, the h16 mix
    const rt::Arg a0 = rt::Arg::Buf(ck_.res), a1 = rt::Arg::Buf(ck_.block_out), a2 = rt::Arg::Buf(ck_.inject),
                  a3 = rt::Arg::Buf(tensors_.at(next.norm.data).buf),
                  a4 = xn_skipped_ ? rt::Arg::Val<std::uint64_t>(0) : rt::Arg::Buf(ck_.xn),
                  a5 = want16 ? rt::Arg::Buf(ck_.xh) : rt::Arg::Val<std::uint64_t>(0), a6 = rt::Arg::Val(c.hc_count),
                  a7 = rt::Arg::Val(H), a8 = rt::Arg::Val(c.rms_eps),
                  a9 = FuseInject(next) ? rt::Arg::Buf(tensors_.at(next.inject.data).buf)
                                        : rt::Arg::Val<std::uint64_t>(0),
                  a10 = rt::Arg::Buf(ck_.ipart), a11 = rt::Arg::Val(static_cast<std::uint32_t>(next.inject.rows)),
                  a12 = xn_skipped_ ? rt::Arg::Buf(ck_.rscale) : rt::Arg::Val<std::uint64_t>(0),
                  a13 = rt::Arg::Val(h16 && BlockedXh(next) ? T : 0u);
    return xq_valid_ ? Launch("halo_hc_combine_norm_rows_q8", {T * c.hc_count}, {kBlock},
                              {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, rt::Arg::Buf(ck_.xq),
                               rt::Arg::Buf(ck_.xd)},
                              error)
                     : Launch(Rk("halo_hc_combine_norm_rows", prefill_res16_).c_str(), {T * c.hc_count}, {kBlock},
                              {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13}, error);
  };
  const bool fuse = prefill_fused_combine_ && H <= 16 * kBlock;
  bool normed = false;
  moe_pending_ = false;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const auto& l = w_.layers[il];
    if (c.IsPleLayer(il) && ((defer_read && !read_ple()) || !PleChunk(l, T, error))) return false;
    if (!HcMixChunk(l.hc_attn, true, T, error, normed)) return false;
    if (l.linear ? !LinearChunk(l, linear_[il], T, error) : !AttentionChunk(l, attn_[il], T, error)) return false;
    want_ml_ = RouterHL(l);  // the router reads the mix's fp16 hi / lo (mh, ml)
    if (!(fuse ? combine_norm(l.hc_ffn) : combine()) || !HcMixChunk(l.hc_ffn, true, T, error, fuse))
      return false;
    want_ml_ = false;
    normed = fuse && il + 1 < c.num_layers && !c.IsPleLayer(il + 1);
    // The routed + shared experts' sum is left to the per-token combine_norm when that one follows.
    moe_defer_ = normed && prefill_moe_in_combine_ && prefill_fused_moe_combine_ && TokCombineFits(w_.layers[il + 1].hc_attn);
    const bool moe_ok = MoeChunk(l, T, error);
    moe_defer_ = false;
    if (!moe_ok) return false;
    if (!(normed ? combine_norm(w_.layers[il + 1].hc_attn) : combine())) return false;
  }
  // The last token's logits through the decode head.
  if (!logits.empty()) {
    if (!(prefill_res16_ ? Launch(Rk("halo_res_to_f32", prefill_res16_).c_str(), {Blocks(D)}, {kBlock},
                                  {rt::Arg::Buf(res_), rt::Arg::Buf(ck_.res, std::size_t{T - 1} * D * 2), rt::Arg::Val(D)},
                                  error)
                         : Launch("halo_copy", {Blocks(D)}, {kBlock},
                                  {rt::Arg::Buf(res_), rt::Arg::Buf(ck_.res, std::size_t{T - 1} * D * 4), rt::Arg::Val(D)},
                                  error)) ||
        !HcMix(w_.hc_head, false, error) || !MatVec(w_.output, 0, mixed_, logits_, error) ||
        !dev_.Download(logits_, 0, logits.data(), c.vocab_size * 4, error))
      return false;
  } else if (!dev_.Synchronize(error)) {
    return false;
  }
  if (timing) {
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::fprintf(stderr,
                 "prefill chunk %u tokens: host prep %.1f ms (hash %.1f, n-gram read %.1f, deferred %.1f), "
                 "upload %.1f ms, device %.1f ms\n",
                 T, ms(t_start, t_prep), ms(t_start, t_hash), ms(t_hash, t_read), deferred_read_ms,
                 ms(t_prep, t_upload), ms(t_upload, std::chrono::steady_clock::now()));
  }
  ngram_history_ = history;
  const std::uint32_t complete = (position_ + T) / ratio;
  for (auto& s : attn_) s.blocks = complete;
  position_ += T;
  return true;
}

}  // namespace halo::engine
