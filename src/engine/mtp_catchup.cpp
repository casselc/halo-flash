// Batched draft-block catch-up over prefilled prompt rows (docs/MTP.md "Batched draft prefill").
//
// A catch-up step of the draft block (a base step whose output is not drafted from) leaves only its cache rows:
// the attention's K / V rows, the indexer key row and the pooled block. Those depend on the step's token, the
// trunk hidden of the position before it and the draft block's position, never on earlier steps, so the rows of a
// whole prefilled chunk can be computed at once. Each launch here computes many rows (columns) with the arithmetic
// of the single-step kernel it replaces: the batched-decode GEMVs and mixers (each column the single kernel's FMA
// chain, weights read once for the group), the *_rows variants of norms, rope, mix and copies (the same per-vector
// code), and the pooling of every block the rows touch (as stepping leaves it). The cache rows are therefore
// bit-identical to stepping (tests: MtpCatchUp.* in tests/gpu).

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "engine/flash_next_gpu.hpp"
#include "kernels/step_params.h"

namespace halo::engine {

namespace {
constexpr std::uint32_t kBlock = 256;
std::uint32_t Blocks(std::size_t n, std::uint32_t per = kBlock) {
  return static_cast<std::uint32_t>((n + per - 1) / per);
}
rt::Buffer View(const rt::Buffer& b, std::size_t offset_bytes, std::size_t bytes) {
  return rt::Buffer{static_cast<char*>(b.device) + offset_bytes, bytes, nullptr};
}
// The prefill residual's element type (as prefill_chunk.cpp's Res16 / Rk).
int Res16() { return FlashNextGpu::prefill_exact_projections_ ? 0 : FlashNextGpu::prefill_res16_; }
std::string Rk(const char* base, int res16) {
  return std::string(base) + (res16 == 1 ? "_r16" : res16 == 2 ? "_rb" : "");
}
std::uint32_t EnvU(const char* name, std::uint32_t fallback) {
  const char* v = std::getenv(name);
  return v && *v ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10)) : fallback;
}
}  // namespace

std::string FlashNextGpu::MtpBatchUnsupported() const {
  const auto& c = w_.config;
  if (!mtp_) return "no draft block";
  if (dev_.backend() != rt::Backend::kHip) return "backend";
  if (!lora_.empty()) return "LoRA";
  if (kv16_only_) return "fp16-only KV";
  if (std::getenv("HALO_MTP_HNORM_GROUPED")) return "hnorm ablation";
  const auto& l = mtp_->block;
  const auto q8 = [&](const model::TensorRef& t) {
    const auto it = tensors_.find(t.data);
    return !t.empty() && it != tensors_.end() && static_cast<bool>(it->second.qs);
  };
  if (!q8(l.nextn_eh_proj) || !q8(l.attn_k) || !q8(l.attn_v)) return "draft projections not repacked Q8";
  const auto& emb = tensors_.at(w_.token_embd.data);
  if (emb.embed < 0 || emb.embed_rows < 0) return "embedding";
  // The mixer: the batched kernels of whichever path HcMix takes.
  const auto& m = l.hc_attn;
  if (!q8(m.down) || !q8(m.up)) return "mixer not repacked Q8";
  const std::uint32_t H = c.hidden_size, D = c.HcDim(), low = c.hc_low_rank;
  const std::uint32_t chunk = (D / kHcParts + 511) / 512 * 512, parts = (D + chunk - 1) / chunk;
  if (c.hc_count > 4 || H % 32 || low % 32) return "shapes";
  if (MtpMixFused() &&
      !((decode_opt_ & kDecodeHoistSmall) && (decode_opt_ & kDecodeHcMixLds) && H <= 4096 && chunk <= 1536 &&
        low <= 512 && parts <= 16))
    return "fused mixer variant";
  // The indexer key: HcMix's dual BF16 launch (the batched kernel is its column form).
  const auto& iq = l.indexer_q;
  const auto& ik = l.indexer_k;
  if (!((decode_opt_ & kDecodeIndexerDual) && iq.type == core::GgmlType::kBF16 && ik.type == iq.type &&
        iq.cols == ik.cols && iq.cols % 32 == 0))
    return "indexer projections";
  if (c.rotary_dim > c.head_dim || c.rotary_dim > c.indexer_head_dim) return "rope";
  return {};
}

// HcMix's choice for the draft block's attention mixer (inject on).
bool FlashNextGpu::MtpMixFused() const {
  const auto& c = w_.config;
  const auto& m = mtp_->block.hc_attn;
  const auto& dn = tensors_.at(m.down.data);
  const auto& upt = tensors_.at(m.up.data);
  const bool inject_f32 = m.inject.type == core::GgmlType::kF32 && m.inject.rows <= 16;
  return fused_hc_ && dn.qs && upt.qs && c.HcDim() <= 12288 && c.hc_low_rank <= 1024 && c.hc_count <= 4 &&
         c.hc_low_rank % 32 == 0 && inject_f32;
}

bool FlashNextGpu::AllocMtpBatch(std::uint32_t rows, std::string* error) {
  rows = (rows + 7) / 8 * 8;
  if (mb_.cap >= rows) return true;
  if (!dev_.Synchronize(error)) return false;
  auto& b = mb_;
  for (rt::Buffer* x : {&b.embd, &b.en, &b.hn, &b.cat, &b.res, &b.xn, &b.lo, &b.gate, &b.mixed,
                        &b.scale, &b.lo_part, &b.inject_part, &b.inject, &b.k, &b.v, &b.iq, &b.ik}) {
    if (!*x) continue;
    std::erase_if(owned_, [&](const rt::Buffer& o) { return o.device == x->device; });
    dev_.Free(*x);
    *x = rt::Buffer{};
  }
  b.cap = 0;
  const auto& c = w_.config;
  const std::size_t S = rows, H = c.hidden_size, D = c.HcDim(), hc = c.hc_count, low = c.hc_low_rank;
  const auto a = [&](rt::Buffer& x, std::size_t floats) { return static_cast<bool>(x = Scratch(S * floats, error)); };
  if (!(a(b.embd, H) && a(b.en, H) && a(b.hn, D) && a(b.cat, hc * 2 * H) &&
        a(b.res, D) && a(b.xn, D) && a(b.lo, low) && a(b.gate, D) && a(b.mixed, H) && a(b.scale, hc) &&
        a(b.lo_part, (kHcParts + 1) * low) && a(b.inject_part, (kHcParts + 1) * hc) && a(b.inject, hc) &&
        a(b.k, c.AttentionKvDim()) && a(b.v, c.AttentionKvDim()) &&
        a(b.iq, std::size_t{c.indexer_heads} * c.indexer_head_dim) && a(b.ik, c.indexer_head_dim)))
    return false;
  b.cap = rows;
  return true;
}

bool FlashNextGpu::MtpCatchUpBatched(std::span<const std::int32_t> tokens, std::uint32_t a, std::string* error) {
  const auto& c = w_.config;
  const auto& l = mtp_->block;
  const auto T = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t steps = T - a;
  if (mtp_pos_ + steps > max_context_) {
    if (error) *error = "MTP context is full";
    return false;
  }
  for (std::uint32_t t = a; t < T; ++t)
    if (tokens[t] < 0 || static_cast<std::uint32_t>(tokens[t]) >= c.vocab_size) {
      if (error) *error = "token out of range";
      return false;
    }
  static const std::uint32_t env_rows = EnvU("HALO_MTP_BATCH_ROWS", 0), g_gemv = EnvU("HALO_MTP_BATCH_G", 8),
                             g_mix = EnvU("HALO_MTP_BATCH_G_MIX", 8);
  // Batched GEMV variant (scheduling only, the same arithmetic): _s (16 KB of LDS, more resident blocks) measured
  // fastest here (doc 8K, 7,962 rows: 512 ms vs 785 plain, 813 _p, 658 _ps; G 4 _s 540).
  static const std::string var = [] {
    const char* v = std::getenv("HALO_MTP_BATCH_VAR");
    return std::string(v ? v : "_s");
  }();
  const std::uint32_t S = std::max(8u, env_rows ? env_rows : mtp_batch_rows_) / 8 * 8;
  if (!AllocMtpBatch(std::min(S, (steps + 7) / 8 * 8), error)) return false;
  const std::uint32_t cap = mb_.cap;
  const std::uint32_t H = c.hidden_size, D = c.HcDim(), hc = c.hc_count, low = c.hc_low_rank;
  const std::uint32_t nh_kv = c.num_kv_heads, hd = c.head_dim, kvdim = c.AttentionKvDim(), idim = c.indexer_head_dim;
  const std::uint32_t ratio = c.compress_ratio;
  const int r16 = Res16();
  const std::size_t res_row = std::size_t{D} * (r16 ? 2 : 4);
  const auto f = [&](const model::TensorRef& t) { return rt::Arg::Buf(tensors_.at(t.data).buf); };
  // Columns [0, m) in groups: G-column launches for the multiple of G, one launch for the rest.
  const auto groups = [&](std::uint32_t m, std::uint32_t gmax, auto&& run) {
    const std::uint32_t G = std::clamp(gmax, 1u, 8u), full = m / G * G;
    return (full == 0 || run(G, 0u, full / G)) && (full == m || run(m - full, full, 1u));
  };
  // halo_gemv_q8r_r2 / r4 (MatVec's choice by rows) over m columns.
  const auto gemv = [&](const model::TensorRef& t, const rt::Buffer& x, std::uint32_t xstride, const rt::Buffer& y,
                        std::uint32_t ystride, std::uint32_t m) {
    const auto& d = tensors_.at(t.data);
    const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
    const bool r2 = rows < 2048;
    return groups(m, g_gemv, [&](std::uint32_t G, std::uint32_t c0, std::uint32_t ng) {
      return Launch(((r2 ? "halo_gemv_q8r_r2_b" : "halo_gemv_q8r_r4_b") + std::to_string(G) + var).c_str(),
                    {Blocks(rows, r2 ? 16 : 32) * ng}, {256},
                    {rt::Arg::Buf(d.qs), rt::Arg::Buf(d.d), rt::Arg::Buf(x, std::size_t{c0} * xstride * 4),
                     rt::Arg::Buf(y, std::size_t{c0} * ystride * 4), rt::Arg::Val(rows), rt::Arg::Val(cols),
                     rt::Arg::Val(d.qstride), rt::Arg::Val(xstride), rt::Arg::Val(ystride), rt::Arg::Val(ng)},
                    error);
    });
  };
  const auto norm = [&](const rt::Buffer& x, const model::TensorRef& w, const rt::Buffer& y, std::uint32_t d,
                        std::uint32_t nvec) {
    return Launch("halo_rmsnorm", {nvec}, {kBlock},
                  {rt::Arg::Buf(x), f(w), rt::Arg::Buf(y), rt::Arg::Val(d), rt::Arg::Val(c.rms_eps),
                   rt::Arg::Val<std::uint32_t>(0)},
                  error);
  };
  const bool fused = MtpMixFused();
  const std::uint32_t chunk = (D / kHcParts + 511) / 512 * 512, parts = (D + chunk - 1) / chunk;
  // HcMix(hc_attn) over m columns of mb_.res -> mb_.mixed (the inject is not needed for the cache rows).
  const auto mix = [&](std::uint32_t m) {
    const auto& mm = l.hc_attn;
    const auto& dn = tensors_.at(mm.down.data);
    const auto& upt = tensors_.at(mm.up.data);
    if (!fused)
      return Launch("halo_rmsnorm_rows", {m * hc}, {kBlock},
                    {rt::Arg::Buf(mb_.res), f(mm.norm), rt::Arg::Buf(mb_.xn), rt::Arg::Val(H), rt::Arg::Val(c.rms_eps),
                     rt::Arg::Val(H), rt::Arg::Val(hc)},
                    error) &&
             gemv(mm.down, mb_.xn, D, mb_.lo, low, m) &&
             Launch("halo_scale_silu", {Blocks(std::size_t{m} * low)}, {kBlock},
                    {rt::Arg::Buf(mb_.lo), rt::Arg::Val(1.0F / static_cast<float>(hc)), rt::Arg::Val(m * low)}, error) &&
             gemv(mm.up, mb_.lo, low, mb_.gate, D, m) &&
             Launch("halo_hc_mix_rows", {Blocks(std::size_t{m} * H)}, {kBlock},
                    {rt::Arg::Buf(mb_.xn), rt::Arg::Buf(mb_.gate), rt::Arg::Buf(mb_.mixed), rt::Arg::Val(hc),
                     rt::Arg::Val(H), rt::Arg::Val(m)},
                    error);
    const std::uint32_t down_blocks = Blocks(low, 16);
    const std::size_t lps = std::size_t{parts} * low, ips = std::size_t{parts} * hc;
    return Launch("halo_hc_norm_scales_u_y", {hc, m}, {256},
                  {rt::Arg::Buf(mb_.res), rt::Arg::Buf(mb_.scale), rt::Arg::Val(H), rt::Arg::Val(c.rms_eps),
                   rt::Arg::Val(hc)},
                  error) &&
           groups(m, g_mix, [&](std::uint32_t G, std::uint32_t c0, std::uint32_t ng) {
             return Launch(("halo_hc_mix_a_cu_b" + std::to_string(G)).c_str(), {down_blocks * ng, parts}, {256},
                           {rt::Arg::Buf(mb_.res, std::size_t{c0} * D * 4), f(mm.norm),
                            rt::Arg::Buf(mb_.scale, std::size_t{c0} * hc * 4), rt::Arg::Buf(dn.qs), rt::Arg::Buf(dn.d),
                            rt::Arg::Buf(mb_.res), rt::Arg::Buf(mb_.xn, std::size_t{c0} * D * 4),
                            rt::Arg::Buf(mb_.lo_part, c0 * lps * 4), rt::Arg::Buf(mb_.inject_part, c0 * ips * 4),
                            rt::Arg::Val(hc), rt::Arg::Val(H), rt::Arg::Val(low), rt::Arg::Val(down_blocks),
                            rt::Arg::Val(dn.qstride), rt::Arg::Val(chunk), rt::Arg::Val(parts), rt::Arg::Val(ng)},
                           error) &&
                    Launch(("halo_hc_mix_b_u_b" + std::to_string(G)).c_str(), {Blocks(H, 8) * ng}, {256},
                           {rt::Arg::Buf(mb_.lo_part, c0 * lps * 4), rt::Arg::Val(parts),
                            rt::Arg::Buf(mb_.inject_part, c0 * ips * 4), rt::Arg::Buf(mb_.inject, std::size_t{c0} * hc * 4),
                            rt::Arg::Val<std::uint32_t>(0), rt::Arg::Buf(upt.qs), rt::Arg::Buf(upt.d),
                            rt::Arg::Buf(mb_.xn, std::size_t{c0} * D * 4), rt::Arg::Buf(mb_.mixed, std::size_t{c0} * H * 4),
                            rt::Arg::Val(hc), rt::Arg::Val(H), rt::Arg::Val(low), rt::Arg::Val(upt.qstride),
                            rt::Arg::Val(ng)},
                           error);
           });
  };
  // Step parameters (only the token is read: embed_rows), all rows in one upload.
  if (!AllocChunkParams(steps, error)) return false;
  mb_.host.assign(std::size_t{steps} * 16, 0);
  for (std::uint32_t i = 0; i < steps; ++i) mb_.host[std::size_t{i} * 16 + kHaloToken] = static_cast<std::uint32_t>(tokens[a + i]);
  if (!dev_.Upload(mb_.params, 0, mb_.host.data(), mb_.host.size() * 4, error)) return false;
  const auto& emb = tensors_.at(w_.token_embd.data);
  const auto& iqt = l.indexer_q;
  const auto& ikt = l.indexer_k;
  const auto rq = static_cast<std::uint32_t>(iqt.rows), rk = static_cast<std::uint32_t>(ikt.rows);
  for (std::uint32_t t0 = a; t0 < T; t0 += cap) {
    const std::uint32_t n = std::min(cap, T - t0), pos0 = mtp_pos_;
    // Hidden inputs: row t takes the trunk hidden of row t - 1 (the pending one for row 0), normalized straight
    // from the chunk's residual rows (fp16 / bf16 rows widen exactly, as halo_res_to_f32 does for a step).
    std::uint32_t lead = 0;
    if (t0 == 0) {
      if (!norm(mtp_hpend_, l.nextn_hnorm, mb_.hn, D, 1)) return false;
      lead = 1;
    }
    if (n > lead &&
        !Launch(Rk("halo_rmsnorm_rows", r16).c_str(), {n - lead}, {kBlock},
                {rt::Arg::Buf(ck_.res, std::size_t{t0 + lead - 1} * res_row),
                 f(l.nextn_hnorm), rt::Arg::Buf(mb_.hn, std::size_t{lead} * D * 4), rt::Arg::Val(D),
                 rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(0), rt::Arg::Val<std::uint32_t>(1)},
                error))
      return false;
    const rt::Arg eargs[] = {rt::Arg::Buf(emb.buf), rt::Arg::Buf(mb_.embd),
                             rt::Arg::Buf(mb_.params, std::size_t{t0 - a} * 64), rt::Arg::Val(H),
                             rt::Arg::Val<std::uint64_t>(w_.token_embd.RowBytes())};
    const std::uint32_t b0 = pos0 / ratio, nb = (pos0 + n - 1) / ratio - b0 + 1;
    if (!dev_.Launch(emb.embed_rows, {Blocks(H), n}, {kBlock}, eargs, error) ||
        !norm(mb_.embd, l.nextn_enorm, mb_.en, H, n) ||
        !Launch("halo_mtp_cat_rows", {Blocks(2 * H), n * hc}, {kBlock},
                {rt::Arg::Buf(mb_.en), rt::Arg::Buf(mb_.hn), rt::Arg::Buf(mb_.cat), rt::Arg::Val(H), rt::Arg::Val(hc)},
                error) ||
        // eh_proj per stream: column (t, s) of the cat rows -> stream s of residual row t.
        !gemv(l.nextn_eh_proj, mb_.cat, 2 * H, mb_.res, H, n * hc) || !mix(n) ||
        !gemv(l.attn_k, mb_.mixed, H, mb_.k, kvdim, n) || !gemv(l.attn_v, mb_.mixed, H, mb_.v, kvdim, n) ||
        !Launch("halo_matvec_bf16_dual_y", {Blocks(rk, 8), n}, {256},
                {f(iqt), rt::Arg::Buf(mb_.iq), rt::Arg::Val<std::uint32_t>(0), f(ikt), rt::Arg::Buf(mb_.ik),
                 rt::Arg::Val(rk), rt::Arg::Buf(mb_.mixed), rt::Arg::Val(static_cast<std::uint32_t>(ikt.cols)),
                 rt::Arg::Val<std::uint32_t>(0), rt::Arg::Val(H), rt::Arg::Val(rq), rt::Arg::Val(rk)},
                error) ||
        !norm(mb_.k, l.attn_k_norm, mb_.k, hd, n * nh_kv) ||
        !Launch("halo_rope_neox_rows", {n * nh_kv}, {64},
                {rt::Arg::Buf(mb_.k), rt::Arg::Val(hd), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(pos0),
                 rt::Arg::Val(nh_kv), rt::Arg::Val(c.rope_theta)},
                error) ||
        !CopyFloats(View(mtp_attn_.k, std::size_t{pos0} * kvdim * 4, std::size_t{n} * kvdim * 4), mb_.k,
                    std::size_t{n} * kvdim, error) ||
        !CopyFloats(View(mtp_attn_.v, std::size_t{pos0} * kvdim * 4, std::size_t{n} * kvdim * 4), mb_.v,
                    std::size_t{n} * kvdim, error) ||
        !CopyFloats(View(mtp_attn_.index_k, std::size_t{pos0} * idim * 4, std::size_t{n} * idim * 4), mb_.ik,
                    std::size_t{n} * idim, error) ||
        // Every block these rows touch, pooled as stepping leaves it (the last one maybe incomplete).
        !Launch("halo_pool_blocks", {nb}, {128},
                {rt::Arg::Buf(mtp_attn_.index_k), rt::Arg::Buf(mtp_attn_.block_k), rt::Arg::Val(b0),
                 rt::Arg::Val(ratio), rt::Arg::Val(idim)},
                error) ||
        !norm(View(mtp_attn_.block_k, std::size_t{b0} * idim * 4, std::size_t{nb} * idim * 4), l.indexer_k_norm,
              View(mtp_attn_.block_k, std::size_t{b0} * idim * 4, std::size_t{nb} * idim * 4), idim, nb) ||
        !Launch("halo_rope_neox_blocks", {nb}, {64},
                {rt::Arg::Buf(mtp_attn_.block_k, std::size_t{b0} * idim * 4), rt::Arg::Val(idim),
                 rt::Arg::Val(c.rotary_dim), rt::Arg::Val(b0), rt::Arg::Val(ratio), rt::Arg::Val(c.rope_theta)},
                error))
      return false;
    mtp_pos_ += n;
    mtp_attn_.blocks = mtp_pos_ / ratio;
  }
  // The last row's hidden waits for the next token.
  const bool ok = r16 ? Launch(Rk("halo_res_to_f32", r16).c_str(), {Blocks(D)}, {kBlock},
                               {rt::Arg::Buf(mtp_hpend_), rt::Arg::Buf(ck_.res, std::size_t{T - 1} * res_row), rt::Arg::Val(D)},
                               error)
                      : CopyFloats(mtp_hpend_, View(ck_.res, std::size_t{T - 1} * res_row, res_row), D, error);
  return ok && dev_.Synchronize(error);
}

bool FlashNextGpu::AllocChunkParams(std::uint32_t rows, std::string* error) {
  if (mb_.params && mb_.params.bytes >= std::size_t{rows} * 64) return true;
  if (!dev_.Synchronize(error)) return false;
  if (mb_.params) {
    std::erase_if(owned_, [&](const rt::Buffer& o) { return o.device == mb_.params.device; });
    dev_.Free(mb_.params);
  }
  mb_.params = Scratch(std::size_t{std::max(rows, mb_.cap)} * 16, error);
  return static_cast<bool>(mb_.params);
}

bool FlashNextGpu::MtpStateBytes(std::vector<float>* out, std::string* error) {
  const auto& c = w_.config;
  if (!mtp_) {
    if (error) *error = "no draft block";
    return false;
  }
  const std::size_t n = mtp_pos_, kv = std::size_t{n} * c.AttentionKvDim(), ik = std::size_t{n} * c.indexer_head_dim;
  // Complete blocks only: an incomplete block's slot also holds the stale rows past the position (pooled as
  // stepping pools them) and is pooled again before it is ever read.
  const std::size_t bk = std::size_t{n / c.compress_ratio} * c.indexer_head_dim;
  out->resize(2 * kv + ik + bk + c.HcDim());
  float* p = out->data();
  return dev_.Synchronize(error) && dev_.Download(mtp_attn_.k, 0, p, kv * 4, error) &&
         dev_.Download(mtp_attn_.v, 0, p + kv, kv * 4, error) &&
         dev_.Download(mtp_attn_.index_k, 0, p + 2 * kv, ik * 4, error) &&
         dev_.Download(mtp_attn_.block_k, 0, p + 2 * kv + ik, bk * 4, error) &&
         (!mtp_hpend_ || dev_.Download(mtp_hpend_, 0, p + 2 * kv + ik + bk, std::size_t{c.HcDim()} * 4, error));
}

}  // namespace halo::engine
