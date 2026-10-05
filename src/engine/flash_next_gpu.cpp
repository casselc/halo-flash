#include <future>
#include "engine/flash_next_gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>

#include "kernels/core_hsaco.hpp"
#include "kernels/sample_params.h"
#include "kernels/step_params.h"

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
                                                   std::uint32_t max_context, std::string* error,
                                                   std::span<const rt::FileSource> sources) {
  auto m = std::unique_ptr<FlashNextGpu>(new FlashNextGpu(device, weights, ngram));
  if (!m->Init(lora, max_context, error, sources)) return nullptr;
  return m;
}

FlashNextGpu::~FlashNextGpu() {
  for (auto& g : graphs_) dev_.FreeGraph(g);
  for (auto& [_, t] : tensors_) {
    dev_.Free(t.buf);
    dev_.Free(t.qs);
    dev_.Free(t.d);
  }
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
  if (!d.buf) return false;
  if (!(loader_ && loader_->Add(d.buf, 0, t.data, bytes)) && !dev_.Upload(d.buf, 0, t.data, bytes, error)) return false;
  if (const char* k = MatVecKernel(t.type)) {
    d.matvec = dev_.Kernel(k, error);
    d.embed = dev_.Kernel(std::string(k) + "_embed_p", error);
    d.embed_rows = dev_.Kernel(std::string(k) + "_embed_rows", error);
    d.expert = dev_.Kernel(std::string(k) + "_expert", error);
    // P3 bandwidth GEMVs (same argument layout) where the row fits in LDS.
    const char* fast = t.type == core::GgmlType::kQ4_K && t.cols % 256 == 0 && t.cols <= 8192   ? "halo_gemv_q4_k"
                       : t.type == core::GgmlType::kQ5_1 && t.cols % 32 == 0 && t.cols <= 8192 ? "halo_gemv_q5_1"
                                                                                               : nullptr;
    if (fast && block_gemv_) {
      d.matvec = dev_.Kernel(fast, error);
      d.expert = dev_.Kernel(std::string(fast) + "_expert", error);
    }
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
  // HALO_PREFILL_SKIP (profiling by ablation): comma-separated kernel names
  // that are not launched. Results are wrong; wall time shows what the
  // kernels cost at the clocks and overlap of the real workload, which a
  // synchronize after every launch (halo_run --profile) does not.
  static const std::string skip = [] {
    const char* v = std::getenv("HALO_PREFILL_SKIP");
    return v ? "," + std::string(v) + "," : std::string();
  }();
  if (!skip.empty() && skip.find("," + std::string(name) + ",") != std::string::npos) return true;
  const bool traced = span_seen_ >= 0 && span_kernel_ == name && span_seen_++ == span_occurrence_;
  const auto span_set = [&](std::uint32_t on) {
    const rt::Arg a[] = {rt::Arg::Val(on)};
    const auto id = dev_.Kernel("halo_span_set", error);
    return id >= 0 && dev_.Launch(id, {1}, {1}, a, error);
  };
  if (traced && !span_set(1)) return false;
  if (!dev_.Launch(it->second, grid, block, std::span<const rt::Arg>(args.begin(), args.size()), error)) return false;
  if (traced && !span_set(0)) return false;
  if (timing_)
    Stamp(std::string(name) + " [" + std::to_string(grid.x) + "x" + std::to_string(grid.y) + "x" +
          std::to_string(grid.z) + "]");
  return true;
}

void FlashNextGpu::Stamp(const std::string& key) {
  if (!timing_ || stamp_keys_.size() >= stamp_cap_) return;
  std::string e;
  auto it = kernels_.find("halo_timestamp");
  if (it == kernels_.end()) {
    const auto id = dev_.Kernel("halo_timestamp", &e);
    if (id < 0) return;
    it = kernels_.emplace("halo_timestamp", id).first;
  }
  const rt::Arg a[] = {rt::Arg::Buf(stamps_), rt::Arg::Val(static_cast<std::uint32_t>(stamp_keys_.size()))};
  if (dev_.Launch(it->second, {1}, {32}, a, &e)) stamp_keys_.push_back(key);
}

bool FlashNextGpu::StartKernelTiming(std::size_t capacity, std::string* error) {
  if (!stamps_ || stamp_cap_ < capacity) {
    if (stamps_) dev_.Free(stamps_);
    stamps_ = dev_.Alloc(capacity * 8, error);
    if (!stamps_) return false;
    stamp_cap_ = capacity;
  }
  if (!dev_.Synchronize(error)) return false;
  stamp_keys_.clear();
  timing_ = true;
  timing_start_ = std::chrono::steady_clock::now();
  Stamp("(start)");
  return true;
}

std::vector<FlashNextGpu::KernelTime> FlashNextGpu::StopKernelTiming(double* span_ms, double* ticks_per_ms,
                                                                  std::string* error) {
  std::vector<KernelTime> out;
  timing_ = false;
  if (!dev_.Synchronize(error)) return out;
  const double host_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - timing_start_).count();
  std::vector<std::uint64_t> t(stamp_keys_.size());
  if (t.size() < 2 || !dev_.Download(stamps_, 0, t.data(), t.size() * 8, error)) return out;
  double rate = ticks_per_ms && *ticks_per_ms > 0 ? *ticks_per_ms
                                                  : static_cast<double>(t.back() - t.front()) / host_ms;
  if (ticks_per_ms) *ticks_per_ms = rate;
  if (span_ms) *span_ms = static_cast<double>(t.back() - t.front()) / rate;
  std::unordered_map<std::string, std::size_t> at;
  for (std::size_t i = 1; i < t.size(); ++i) {
    auto [it, fresh] = at.emplace(stamp_keys_[i], out.size());
    if (fresh) out.push_back({stamp_keys_[i], 0, 0});
    out[it->second].ms += static_cast<double>(t[i] - t[i - 1]) / rate;
    ++out[it->second].calls;
  }
  std::sort(out.begin(), out.end(), [](const KernelTime& a, const KernelTime& b) { return a.ms > b.ms; });
  return out;
}

bool FlashNextGpu::MatVec(const model::TensorRef& t, std::uint64_t expert, const rt::Buffer& x, const rt::Buffer& y,
                          std::string* error, std::size_t x_off, std::size_t y_off) {
  const auto& d = tensors_.at(t.data);
  if (d.matvec < 0) {
    if (error) *error = "no GPU matvec for " + std::string(t.name);
    return false;
  }
  const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
  if (d.qs) {
    const std::size_t e = expert;
    // Multi-row variants (4 rows per wave; 2 for wide inputs with few rows,
    // where 32-row blocks would leave the GPU idle); see halo_gemv_bench.
    // From halo_gemv_bench at DRAM working sets: 4 rows per wave is best
    // (LM head 242 GB/s); 2 for few rows (hc down, 320 rows).
    const char* k = rows < 2048 ? "halo_gemv_q8r_r2" : "halo_gemv_q8r_r4";
    const std::uint32_t per_block = rows < 2048 ? 16 : 32;
    if (!Launch(k, {Blocks(rows, per_block)}, {256},
                {rt::Arg::Buf(d.qs, e * rows * d.qstride), rt::Arg::Buf(d.d, e * rows * (cols / 32) * 2),
                 rt::Arg::Buf(x, x_off), rt::Arg::Buf(y, y_off), rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val(d.qstride)},
                error))
      return false;
  } else if (t.type == core::GgmlType::kF32 && rows <= 1024 && expert == 0 && small_row_kernels_) {
    if (!Launch("halo_matvec_f32_rowblock", {rows}, {256},
                {rt::Arg::Buf(d.buf), rt::Arg::Buf(x, x_off), rt::Arg::Buf(y, y_off), rt::Arg::Val(rows), rt::Arg::Val(cols)}, error))
      return false;
  } else {
    const rt::Arg args[] = {rt::Arg::Buf(d.buf, expert * t.RowBytes() * t.rows), rt::Arg::Buf(x, x_off), rt::Arg::Buf(y, y_off),
                            rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val<std::uint64_t>(t.RowBytes())};
    if (!dev_.Launch(d.matvec, {Blocks(rows, kBlock / 32)}, {kBlock}, args, error)) return false;
    if (timing_) Stamp("matvec " + std::string(t.name));
  }
  // LoRA: y += (alpha / r) * B (A x), on dense projections, as the reference.
  if (auto it = lora_.find(t.data); it != lora_.end()) {
    const auto& [pair, scale] = it->second;
    if (!MatVec(pair->a, 0, x, lora_tmp_, error, x_off) || !MatVec(pair->b, 0, lora_tmp_, lora_out_, error) ||
        !Launch("halo_axpy", {Blocks(rows)}, {kBlock},
                {rt::Arg::Buf(y, y_off), rt::Arg::Buf(lora_out_), rt::Arg::Val(scale), rt::Arg::Val(rows)}, error))
      return false;
  }
  return true;
}

bool FlashNextGpu::Init(const model::LoraAdapter* lora, std::uint32_t max_context, std::string* error,
                        std::span<const rt::FileSource> sources) {
  rt::FileUploader loader(dev_, sources);
  loader_ = &loader;
  struct Clear {
    rt::FileUploader*& p;
    ~Clear() { p = nullptr; }
  } clear{loader_};
  const auto& c = w_.config;
  max_context_ = max_context;
  if (c.num_experts > 1024 || c.num_experts_used == 0 || c.num_experts_used > c.num_experts) {
    if (error) *error = "halo_router_topk supports 1..1024 experts with 0 < used <= experts";
    return false;
  }
  if (!dev_.LoadModule(span_hsaco_.empty() ? kernels::CoreHsaco() : std::span<const std::uint8_t>(span_hsaco_), error))
    return false;
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
  // Weights queued by Upload() are read from their files in parallel now.
  if (!loader.Run(std::max(4u, std::min(16u, std::thread::hardware_concurrency() / 2)), error)) return false;
  loader_ = nullptr;
  // P3: repack Q8_0 matrices for the bandwidth GEMV (the embedding keeps its
  // raw rows for the gather).
  if (repack_q8_) {
    std::vector<const model::TensorRef*> all;
    const auto collect = [&](const model::TensorRef& t) {
      if (!t.empty()) all.push_back(&t);
    };
    const auto mixer = [&](const model::HcMixer& m) {
      collect(m.down);
      collect(m.up);
    };
    collect(w_.output);
    mixer(w_.hc_head);
    for (const auto& l : w_.layers) {
      mixer(l.hc_attn);
      mixer(l.hc_ffn);
      for (const auto* t : {&l.ssm_qkv, &l.ssm_gate, &l.ssm_out, &l.attn_q, &l.attn_k, &l.attn_v, &l.attn_out,
                            &l.ple_key, &l.ple_value, &l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps,
                            &l.shexp_gate, &l.shexp_up, &l.shexp_down})
        collect(*t);
    }
    for (const auto* t : all)
      if (t->data != w_.token_embd.data && !RepackQ8(*t, tensors_.at(t->data), error)) return false;
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
  // One upload per step: n-gram rows first (read in place by the PLE
  // matvecs), then the step parameters at a 64-byte aligned offset.
  params_offset_ = (std::size_t{c.PleEmbeddingDim()} * 4 + 63) / 64 * 64;
  sample_offset_ = params_offset_ + 64;
  penalty_offset_ = sample_offset_ + (sizeof(HaloSampleParams) + 63) / 64 * 64;
  step_host_.assign(penalty_offset_ + std::size_t{HALO_SAMPLE_MAX_PENALTIES} * sizeof(HaloPenalty), 0);
  ple_emb_ = dev_.Alloc(step_host_.size(), error);
  if (!ple_emb_) return false;
  owned_.push_back(ple_emb_);
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
  expert_ids_ = Scratch(c.num_experts_used, error);
  expert_weights_ = Scratch(c.num_experts_used, error);
  moe_h_ = Scratch(std::size_t{c.num_experts_used} * c.expert_ff, error);
  hc_lo_part_ = Scratch(std::size_t{kHcParts + 1} * c.hc_low_rank, error);
  hc_inject_part_ = Scratch(std::size_t{kHcParts + 1} * c.hc_count, error);
  hc_scale_ = Scratch(c.hc_count, error);
  moe_part_ = Scratch(std::size_t{c.num_experts_used} * c.hidden_size, error);
  eg_ = Scratch(std::max(c.expert_ff, c.shared_expert_ff), error);
  eu_ = Scratch(std::max(c.expert_ff, c.shared_expert_ff), error);
  ed_ = Scratch(H, error);
  sgate_ = Scratch(1, error);
  logits_ = Scratch(c.vocab_size, error);
  sample_blocks_ = (c.vocab_size + HALO_SAMPLE_BLOCK - 1) / HALO_SAMPLE_BLOCK;
  {
    const std::size_t cap = std::max<std::size_t>(c.vocab_size, std::size_t{sample_blocks_} * HALO_SAMPLE_TOPK) +
                            HALO_SAMPLE_MAX_PENALTIES;
    for (auto [buf, bytes] : {std::pair{&sample_block_out_, sample_blocks_ * sizeof(HaloSampleBlock)},
                              std::pair{&sample_candidates_, std::size_t{sample_blocks_} * HALO_SAMPLE_TOPK * 8},
                              std::pair{&sample_key_, cap * 8}, std::pair{&sample_rid_, cap * 4},
                              std::pair{&sample_w_, cap * 8}, std::pair{&sample_out_, sizeof(HaloSampleOut)}}) {
      *buf = dev_.Alloc(bytes, error);
      if (!*buf) return false;
      owned_.push_back(*buf);
    }
  }
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
      if (prefill_kv16_) {
        const std::size_t halves = std::size_t{max_context} * c.AttentionKvDim();
        s.k16 = Scratch((halves + 1) / 2, error);
        s.v16 = Scratch((halves + 1) / 2, error);
        if (!s.k16 || !s.v16) return false;
      }
    }
  }
  return dev_.Synchronize(error);
}

bool FlashNextGpu::RepackQ8(const model::TensorRef& t, DevTensor& d, std::string* error) {
  // Expert kernels hold x for up to 8192 columns; dense ones up to 12288.
  if (t.type != core::GgmlType::kQ8_0 || d.qs || t.cols % 32 != 0 || t.cols > (t.experts > 1 ? 8192u : 12288u))
    return true;
  const std::uint64_t rows = t.rows * t.experts;
  const auto cols = static_cast<std::uint32_t>(t.cols);
  d.qstride = (cols + 15) / 16 * 16;
  d.qs = dev_.Alloc(rows * d.qstride, error);
  d.d = dev_.Alloc(rows * (cols / 32) * 2, error);
  if (!d.qs || !d.d) return false;
  if (!Launch("halo_repack_q8_0", {std::min<std::uint32_t>(Blocks(rows * (cols / 32)), 4096)}, {kBlock},
              {rt::Arg::Buf(d.buf), rt::Arg::Buf(d.qs), rt::Arg::Buf(d.d), rt::Arg::Val(static_cast<std::uint32_t>(rows)),
               rt::Arg::Val(cols), rt::Arg::Val(d.qstride)},
              error) ||
      !dev_.Synchronize(error))
    return false;
  dev_.Free(d.buf);  // the raw copy is no longer read
  return true;
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
  // P3 fused path: two launches when both projections are repacked Q8_0.
  const auto& dn = tensors_.at(m.down.data);
  const auto& upt = tensors_.at(m.up.data);
  const bool inject_f32 = !inject || (m.inject.type == core::GgmlType::kF32 && m.inject.rows <= 16);
  if (fused_hc_ && dn.qs && upt.qs && c.HcDim() <= 12288 && c.hc_low_rank <= 1024 && c.hc_count <= 4 &&
      c.hc_low_rank % 32 == 0 && inject_f32) {
    const std::uint32_t down_blocks = Blocks(c.hc_low_rank, 16), inject_rows = inject ? c.hc_count : 0;
    // Columns split into kHcParts chunks (multiples of 512) so the 320-row
    // down projection fills the GPU.
    const std::uint32_t chunk = (c.HcDim() / kHcParts + 511) / 512 * 512;
    const std::uint32_t parts = (c.HcDim() + chunk - 1) / chunk;
    return Launch("halo_hc_norm_scales", {c.hc_count}, {256},
                  {rt::Arg::Buf(res_), rt::Arg::Buf(hc_scale_), rt::Arg::Val(c.hidden_size), rt::Arg::Val(c.rms_eps)},
                  error) &&
           Launch("halo_hc_mix_a", {down_blocks + inject_rows, parts}, {256},
                  {rt::Arg::Buf(res_), rt::Arg::Buf(tensors_.at(m.norm.data).buf), rt::Arg::Buf(hc_scale_),
                   rt::Arg::Buf(dn.qs), rt::Arg::Buf(dn.d), rt::Arg::Buf(inject ? tensors_.at(m.inject.data).buf : res_),
                   rt::Arg::Buf(xn_), rt::Arg::Buf(hc_lo_part_), rt::Arg::Buf(hc_inject_part_), rt::Arg::Val(c.hc_count),
                   rt::Arg::Val(c.hidden_size), rt::Arg::Val(c.hc_low_rank), rt::Arg::Val(down_blocks),
                   rt::Arg::Val(dn.qstride), rt::Arg::Val(chunk)},
                  error) &&
           Launch("halo_hc_mix_b", {Blocks(c.hidden_size, 8)}, {256},
                  {rt::Arg::Buf(hc_lo_part_), rt::Arg::Val(parts), rt::Arg::Buf(hc_inject_part_), rt::Arg::Buf(inject_),
                   rt::Arg::Val(inject_rows), rt::Arg::Buf(upt.qs), rt::Arg::Buf(upt.d), rt::Arg::Buf(xn_),
                   rt::Arg::Buf(mixed_), rt::Arg::Val(c.hc_count), rt::Arg::Val(c.hidden_size),
                   rt::Arg::Val(c.hc_low_rank), rt::Arg::Val(upt.qstride)},
                  error);
  }
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

bool FlashNextGpu::Ple(const model::LayerWeights& l, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t D = c.HcDim();
  return MatVec(l.ple_key, 0, ple_emb_, ple_key_, error) &&
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
  // P3 fused path (head dim 128, repacked Q8 projections, F32 alpha/beta):
  // qkv+z, alpha+beta, conv+L2, DeltaNet+gated norm, out: 5 launches.
  const auto& tq = tensors_.at(l.ssm_qkv.data);
  const auto& tz = tensors_.at(l.ssm_gate.data);
  const std::uint32_t H = c.hidden_size;
  if (fused_linear_ && d == 128 && tq.qs && tz.qs && H <= 12288 && l.ssm_alpha.type == core::GgmlType::kF32 &&
      l.ssm_beta.type == core::GgmlType::kF32 && ch % 128 == 0 && (2 * kd) % 128 == 0) {
    const auto rq = static_cast<std::uint32_t>(l.ssm_qkv.rows), rz = static_cast<std::uint32_t>(l.ssm_gate.rows);
    const std::uint32_t b1 = Blocks(rq, 32);
    return Launch("halo_gemv_q8r_dual", {b1 + Blocks(rz, 32)}, {256},
                  {rt::Arg::Buf(tq.qs), rt::Arg::Buf(tq.d), rt::Arg::Buf(qkv_), rt::Arg::Val(rq), rt::Arg::Val(tq.qstride),
                   rt::Arg::Buf(tz.qs), rt::Arg::Buf(tz.d), rt::Arg::Buf(z_), rt::Arg::Val(rz), rt::Arg::Val(tz.qstride),
                   rt::Arg::Buf(mixed_), rt::Arg::Val(H), rt::Arg::Val(b1)},
                  error) &&
           Launch("halo_matvec_f32_rowblock2", {2 * c.ssm_num_v_heads}, {256},
                  {rt::Arg::Buf(tensors_.at(l.ssm_alpha.data).buf), rt::Arg::Buf(tensors_.at(l.ssm_beta.data).buf),
                   rt::Arg::Buf(mixed_), rt::Arg::Buf(alpha_), rt::Arg::Buf(beta_), rt::Arg::Val(c.ssm_num_v_heads),
                   rt::Arg::Val(H)},
                  error) &&
           Launch("halo_conv_l2norm", {ch / 128}, {128},
                  {rt::Arg::Buf(qkv_), rt::Arg::Buf(s.conv), rt::Arg::Buf(tensors_.at(l.ssm_conv1d.data).buf),
                   rt::Arg::Buf(conv_out_), rt::Arg::Val(ch), rt::Arg::Val(c.ssm_conv_kernel), rt::Arg::Val(2 * kd),
                   rt::Arg::Val(c.rms_eps)},
                  error) &&
           Launch("halo_gdn_step_d128_norm", {c.ssm_num_v_heads}, {128},
                  {rt::Arg::Buf(conv_out_), rt::Arg::Buf(conv_out_, kd * 4ull), rt::Arg::Buf(conv_out_, 2ull * kd * 4),
                   rt::Arg::Buf(alpha_), rt::Arg::Buf(beta_), rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf),
                   rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf), rt::Arg::Buf(s.state),
                   rt::Arg::Buf(tensors_.at(l.ssm_norm.data).buf), rt::Arg::Buf(z_), rt::Arg::Buf(gdn_out_),
                   rt::Arg::Val(c.ssm_num_k_heads), rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(1)},
                  error) &&
           MatVec(l.ssm_out, 0, gdn_out_, block_out_, error);
  }
  return MatVec(l.ssm_qkv, 0, mixed_, qkv_, error) && MatVec(l.ssm_gate, 0, mixed_, z_, error) &&
         MatVec(l.ssm_alpha, 0, mixed_, alpha_, error) && MatVec(l.ssm_beta, 0, mixed_, beta_, error) &&
         Launch("halo_conv_step", {Blocks(ch)}, {kBlock},
                {rt::Arg::Buf(qkv_), rt::Arg::Buf(s.conv), rt::Arg::Buf(tensors_.at(l.ssm_conv1d.data).buf),
                 rt::Arg::Buf(conv_out_), rt::Arg::Val(ch), rt::Arg::Val(c.ssm_conv_kernel)},
                error) &&
         Launch("halo_l2norm", {2 * c.ssm_num_k_heads}, {128},
                {rt::Arg::Buf(conv_out_), rt::Arg::Val(d), rt::Arg::Val(c.rms_eps)}, error) &&
         (d == 128 ? Launch("halo_gdn_step_d128", {c.ssm_num_v_heads}, {128},
                            {rt::Arg::Buf(conv_out_), rt::Arg::Buf(conv_out_, kd * 4ull),
                             rt::Arg::Buf(conv_out_, 2ull * kd * 4), rt::Arg::Buf(alpha_), rt::Arg::Buf(beta_),
                             rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf), rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf),
                             rt::Arg::Buf(s.state), rt::Arg::Buf(gdn_out_), rt::Arg::Val(c.ssm_num_k_heads)},
                            error)
                   : Launch("halo_gdn_step", {c.ssm_num_v_heads}, {d},
                            {rt::Arg::Buf(conv_out_), rt::Arg::Buf(conv_out_, kd * 4ull),
                             rt::Arg::Buf(conv_out_, 2ull * kd * 4), rt::Arg::Buf(alpha_), rt::Arg::Buf(beta_),
                             rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf), rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf),
                             rt::Arg::Buf(s.state), rt::Arg::Buf(gdn_out_), rt::Arg::Val(c.ssm_num_k_heads),
                             rt::Arg::Val(d)},
                            error)) &&
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
  // Positions come from the step parameters (captured graphs replay at any
  // position): rope at params[slot] * scale, cache rows at params[slot].
  const rt::Arg params = rt::Arg::Buf(ple_emb_, params_offset_);
  const auto rope = [&](const rt::Buffer& x, std::uint32_t heads, std::uint32_t dim, std::uint32_t slot,
                        std::uint32_t scale) {
    return Launch("halo_rope_neox_p", {heads}, {64},
                  {rt::Arg::Buf(x), params, rt::Arg::Val(dim), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(slot),
                   rt::Arg::Val(scale), rt::Arg::Val(c.rope_theta)},
                  error);
  };
  const auto append = [&](const rt::Buffer& cache, const rt::Buffer& row, std::uint32_t slot, std::uint32_t n) {
    return Launch("halo_copy_at", {Blocks(n)}, {kBlock},
                  {rt::Arg::Buf(cache), rt::Arg::Buf(row), params, rt::Arg::Val(slot), rt::Arg::Val(n), rt::Arg::Val(n)},
                  error);
  };
  const std::uint32_t ratio = c.compress_ratio, budget = c.indexer_top_k / ratio;
  const std::uint32_t max_blocks = max_context_ / ratio + 1;
  return MatVec(l.attn_q, 0, mixed_, qg_, error) && MatVec(l.attn_k, 0, mixed_, k_, error) &&
         MatVec(l.attn_v, 0, mixed_, v_, error) &&
         Launch("halo_split_qgate", {nh}, {kBlock},
                {rt::Arg::Buf(qg_), rt::Arg::Buf(q_), rt::Arg::Buf(gate_), rt::Arg::Val(hd)}, error) &&
         norm(q_, l.attn_q_norm, nh, hd) && norm(k_, l.attn_k_norm, nkv, hd) && rope(q_, nh, hd, kHaloPosition, 1) &&
         rope(k_, nkv, hd, kHaloPosition, 1) && append(s.k, k_, kHaloPosition, nkv * hd) &&
         append(s.v, v_, kHaloPosition, nkv * hd) &&
         // QSA indexer: raw keys cached; the block holding this token is
         // pooled, normalized and rotated at its start every step, final
         // once complete (incomplete slots are never scored).
         MatVec(l.indexer_q, 0, mixed_, iq_, error) && MatVec(l.indexer_k, 0, mixed_, ik_, error) &&
         append(s.index_k, ik_, kHaloPosition, idim) && norm(iq_, l.indexer_q_norm, c.indexer_heads, idim) &&
         rope(iq_, c.indexer_heads, idim, kHaloPosition, 1) &&
         Launch("halo_pool_block", {1}, {128},
                {rt::Arg::Buf(s.index_k), rt::Arg::Buf(pooled_), params, rt::Arg::Val(ratio), rt::Arg::Val(idim)},
                error) &&
         norm(pooled_, l.indexer_k_norm, 1, idim) && rope(pooled_, 1, idim, kHaloBlock, ratio) &&
         append(s.block_k, pooled_, kHaloBlock, idim) &&
         // Selection on device with the reference's rule.
         Launch("halo_qsa_scores", {std::min(Blocks(max_blocks), 64u)}, {kBlock},
                {rt::Arg::Buf(iq_), rt::Arg::Buf(s.block_k), rt::Arg::Buf(block_scores_), params,
                 rt::Arg::Val(c.indexer_heads), rt::Arg::Val(idim)},
                error) &&
         Launch("halo_qsa_select", {1}, {1024},
                {rt::Arg::Buf(block_scores_), rt::Arg::Buf(visible_), params, rt::Arg::Val(budget), rt::Arg::Val(ratio)},
                error) &&
         Launch("halo_attention_select", {nh}, {kBlock},
                {rt::Arg::Buf(q_), rt::Arg::Buf(s.k), rt::Arg::Buf(s.v), rt::Arg::Buf(gate_), rt::Arg::Buf(visible_),
                 rt::Arg::Buf(ctx_), rt::Arg::Buf(scores_), params, rt::Arg::Val(nh), rt::Arg::Val(nkv),
                 rt::Arg::Val(hd), rt::Arg::Val(max_context_)},
                error) &&
         MatVec(l.attn_out, 0, ctx_, block_out_, error);
}

bool FlashNextGpu::ExpertMatVec(const model::TensorRef& t, std::uint32_t slot, const rt::Buffer& x,
                                const rt::Buffer& y, std::string* error) {
  const auto& d = tensors_.at(t.data);
  const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
  if (d.qs)
    return Launch("halo_gemv_q8r_expert_r4", {Blocks(rows, 32)}, {256},
                  {rt::Arg::Buf(d.qs), rt::Arg::Buf(d.d), rt::Arg::Buf(expert_ids_), rt::Arg::Buf(x), rt::Arg::Buf(y),
                   rt::Arg::Val(slot), rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val(d.qstride)},
                  error);
  const rt::Arg args[] = {rt::Arg::Buf(d.buf), rt::Arg::Buf(expert_ids_), rt::Arg::Buf(x), rt::Arg::Buf(y),
                          rt::Arg::Val(slot), rt::Arg::Val<std::uint64_t>(t.RowBytes() * t.rows),
                          rt::Arg::Val(rows), rt::Arg::Val(cols), rt::Arg::Val<std::uint64_t>(t.RowBytes())};
  return dev_.Launch(d.expert, {Blocks(rows, kBlock / 32)}, {kBlock}, args, error);
}

bool FlashNextGpu::Moe(const model::LayerWeights& l, std::string* error) {
  const auto& c = w_.config;
  const std::uint32_t H = c.hidden_size;
  // Routing on device: softmax, top-k (ties keep the lower expert), renormalize.
  if (!MatVec(l.router, 0, mixed_, router_, error) ||
      !Launch("halo_router_topk", {1}, {kBlock},
              {rt::Arg::Buf(router_), rt::Arg::Buf(expert_ids_), rt::Arg::Buf(expert_weights_),
               rt::Arg::Val(c.num_experts), rt::Arg::Val(c.num_experts_used)},
              error))
    return false;
  // P3 fused path: every routed expert in three launches.
  const auto& gate = l.ffn_gate_exps;
  const auto& up = l.ffn_up_exps;
  const auto& down = l.ffn_down_exps;
  const auto& dd = tensors_.at(down.data);
  const bool gate_up_fast = fused_moe_ && gate.type == core::GgmlType::kQ4_K && up.type == gate.type &&
                            up.rows == gate.rows && up.cols == gate.cols && gate.cols % 256 == 0 && gate.cols <= 8192;
  const bool down_q5_1 = down.type == core::GgmlType::kQ5_1 && down.cols % 32 == 0 && down.cols <= 8192;
  const bool down_q8r = static_cast<bool>(dd.qs);
  if (gate_up_fast && (down_q5_1 || down_q8r)) {
    const auto ff = static_cast<std::uint32_t>(gate.rows), used = c.num_experts_used;
    // 4 rows per wave with precomputed x sub-block sums (inputs <= 4096).
    // int8 activations + integer dot (default; ~1.3x the float s2 kernel) or
    // float s2 with precomputed x sums; both for inputs <= 4096.
    const bool s2 = H <= 4096;
    const char* gu = !s2 ? "halo_moe_gate_up_q4_k" : int8_act_moe_ ? "halo_moe_gate_up_q4_k_dot" : "halo_moe_gate_up_q4_k_s2";
    if (!Launch(gu, {Blocks(ff, s2 ? 16 : 8), used}, {256},
                {rt::Arg::Buf(tensors_.at(gate.data).buf), rt::Arg::Buf(tensors_.at(up.data).buf),
                 rt::Arg::Buf(expert_ids_), rt::Arg::Buf(mixed_), rt::Arg::Buf(moe_h_),
                 rt::Arg::Val<std::uint64_t>(gate.RowBytes() * gate.rows), rt::Arg::Val(ff), rt::Arg::Val(H),
                 rt::Arg::Val<std::uint64_t>(gate.RowBytes())},
                error))
      return false;
    const bool down_ok =
        down_q5_1 ? Launch(ff <= 2048 ? "halo_moe_down_q5_1_r4" : "halo_moe_down_q5_1",
                           {Blocks(H, ff <= 2048 ? 32 : 8), used}, {256},
                           {rt::Arg::Buf(dd.buf), rt::Arg::Buf(expert_ids_), rt::Arg::Buf(moe_h_), rt::Arg::Buf(moe_part_),
                            rt::Arg::Val<std::uint64_t>(down.RowBytes() * down.rows), rt::Arg::Val(H), rt::Arg::Val(ff),
                            rt::Arg::Val<std::uint64_t>(down.RowBytes())},
                           error)
                  : Launch("halo_moe_down_q8r_r4", {Blocks(H, 32), used}, {256},
                           {rt::Arg::Buf(dd.qs), rt::Arg::Buf(dd.d), rt::Arg::Buf(expert_ids_), rt::Arg::Buf(moe_h_),
                            rt::Arg::Buf(moe_part_), rt::Arg::Val(H), rt::Arg::Val(ff), rt::Arg::Val(dd.qstride)},
                           error);
    if (!down_ok || !Launch("halo_moe_combine", {Blocks(H)}, {kBlock},
                            {rt::Arg::Buf(moe_part_), rt::Arg::Buf(expert_weights_), rt::Arg::Buf(block_out_),
                             rt::Arg::Val(used), rt::Arg::Val(H)},
                            error))
      return false;
  } else if (!dev_.Fill(block_out_, 0, error)) {
    return false;
  }
  for (std::uint32_t i = 0; !(gate_up_fast && (down_q5_1 || down_q8r)) && i < c.num_experts_used; ++i) {
    if (!ExpertMatVec(l.ffn_gate_exps, i, mixed_, eg_, error) || !ExpertMatVec(l.ffn_up_exps, i, mixed_, eu_, error) ||
        !Launch("halo_silu_mul", {Blocks(c.expert_ff)}, {kBlock},
                {rt::Arg::Buf(eg_), rt::Arg::Buf(eu_), rt::Arg::Val(c.expert_ff)}, error) ||
        !ExpertMatVec(l.ffn_down_exps, i, eg_, ed_, error) ||
        !Launch("halo_axpy_slot", {Blocks(H)}, {kBlock},
                {rt::Arg::Buf(block_out_), rt::Arg::Buf(ed_), rt::Arg::Buf(expert_weights_), rt::Arg::Val(i),
                 rt::Arg::Val(H)},
                error))
      return false;
  }
  // P3 fused shared expert: two launches when its projections are repacked Q8.
  const auto& sg = tensors_.at(l.shexp_gate.data);
  const auto& su = tensors_.at(l.shexp_up.data);
  const auto& sd = tensors_.at(l.shexp_down.data);
  if (fused_shexp_ && sg.qs && su.qs && sd.qs && l.shexp_gate_inp.type == core::GgmlType::kF32 && H <= 8192 &&
      c.shared_expert_ff <= 8192 && l.shexp_gate.cols == H) {
    const auto sff = c.shared_expert_ff;
    return Launch("halo_shexp_gate_up", {Blocks(sff, 16)}, {256},
                  {rt::Arg::Buf(sg.qs), rt::Arg::Buf(sg.d), rt::Arg::Buf(su.qs), rt::Arg::Buf(su.d),
                   rt::Arg::Buf(tensors_.at(l.shexp_gate_inp.data).buf), rt::Arg::Buf(mixed_), rt::Arg::Buf(eg_),
                   rt::Arg::Buf(sgate_), rt::Arg::Val(sff), rt::Arg::Val(H), rt::Arg::Val(sg.qstride)},
                  error) &&
           Launch("halo_shexp_down_add", {Blocks(H, 32)}, {256},
                  {rt::Arg::Buf(sd.qs), rt::Arg::Buf(sd.d), rt::Arg::Buf(eg_), rt::Arg::Buf(sgate_),
                   rt::Arg::Buf(block_out_), rt::Arg::Val(H), rt::Arg::Val(sff), rt::Arg::Val(sd.qstride)},
                  error);
  }
  return MatVec(l.shexp_gate, 0, mixed_, eg_, error) && MatVec(l.shexp_up, 0, mixed_, eu_, error) &&
         Launch("halo_silu_mul", {Blocks(c.shared_expert_ff)}, {kBlock},
                {rt::Arg::Buf(eg_), rt::Arg::Buf(eu_), rt::Arg::Val(c.shared_expert_ff)}, error) &&
         MatVec(l.shexp_down, 0, eg_, ed_, error) && MatVec(l.shexp_gate_inp, 0, mixed_, sgate_, error) &&
         Launch("halo_axpy_sigmoid", {Blocks(H)}, {kBlock},
                {rt::Arg::Buf(block_out_), rt::Arg::Buf(ed_), rt::Arg::Buf(sgate_), rt::Arg::Val(H)}, error);
}

bool FlashNextGpu::Record(Head head, std::string* error) {
  const auto& c = w_.config;
  const auto& emb = tensors_.at(w_.token_embd.data);
  const rt::Arg eargs[] = {rt::Arg::Buf(emb.buf), rt::Arg::Buf(embd_), rt::Arg::Buf(ple_emb_, params_offset_),
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
    if (c.IsPleLayer(il) && !Ple(l, error)) return false;
    if (!HcMix(l.hc_attn, true, error)) return false;
    if (l.linear ? !Linear(l, linear_[il], error) : !Attention(l, attn_[il], error)) return false;
    if (!combine() || !HcMix(l.hc_ffn, true, error) || !Moe(l, error) || !combine()) return false;
  }
  if (head == Head::kNone) return true;
  if (!HcMix(w_.hc_head, false, error) || !MatVec(w_.output, 0, mixed_, logits_, error)) return false;
  if (head == Head::kLogits) return true;
  // Sampling: fast blocks + final, then the exact kernel, which runs only
  // when the final kernel asked for it (or always, in exact mode).
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
}

bool FlashNextGpu::Step(std::int32_t token, std::span<float> logits, std::string* error) {
  return Run(token, logits.empty() ? Head::kNone : Head::kLogits, nullptr, logits, nullptr, error);
}

bool FlashNextGpu::StepSample(std::int32_t token, const SampleRequest& request, SampleResult* result,
                              std::string* error) {
  const auto fail = [&](const char* what) {
    if (error) *error = what;
    return false;
  };
  if (request.config == nullptr || result == nullptr) return fail("StepSample needs a config and a result");
  if (request.penalties.size() > HALO_SAMPLE_MAX_PENALTIES) return fail("too many penalized tokens for the device");
  if (w_.config.vocab_size > HALO_SAMPLE_MAX_VOCAB) return fail("vocabulary too large for device sampling");
  try {
    request.config->Validate();
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
  return Run(token, Head::kSample, &request, {}, result, error);
}

bool FlashNextGpu::Run(std::int32_t token, Head head, const SampleRequest* request, std::span<float> logits,
                       SampleResult* result, std::string* error) {
  const auto& c = w_.config;
  if (position_ >= max_context_ || token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
    if (error) *error = position_ >= max_context_ ? "context is full" : "token out of range";
    return false;
  }
  // Host side of the step: n-gram rows and the step parameters, uploaded
  // together; everything else is device work (captured once, replayed).
  model::NgramHistory history = ngram_history_;
  if (c.ple_layer >= 0) {
    std::vector<std::uint32_t> rows(c.ple_heads);
    model::HashNgramRows(c, history, std::span<const std::int32_t>(&token, 1), rows);
    const std::span<float> emb(reinterpret_cast<float*>(step_host_.data()), c.PleEmbeddingDim());
    if (!ngram_ || !ngram_->Read(rows, emb)) {
      if (error) *error = "n-gram table read failed";
      return false;
    }
  }
  const std::uint32_t ratio = c.compress_ratio, budget = c.indexer_top_k / ratio;
  const std::uint32_t n_kv = position_ + 1, complete = n_kv / ratio;
  std::uint32_t params[kHaloStepParamCount] = {};
  params[kHaloToken] = static_cast<std::uint32_t>(token);
  params[kHaloPosition] = position_;
  params[kHaloNkv] = n_kv;
  params[kHaloComplete] = complete;
  params[kHaloBlock] = position_ / ratio;
  params[kHaloNVisible] = complete <= budget ? n_kv : budget * ratio + (n_kv - complete * ratio);
  std::memcpy(step_host_.data() + params_offset_, params, sizeof(params));
  std::size_t upload = sample_offset_;
  if (head == Head::kSample) {
    const auto& k = *request->config;
    HaloSampleParams sp{k.temperature,
                        k.top_p,
                        k.min_p,
                        k.repeat_penalty,
                        k.frequency_penalty,
                        k.presence_penalty,
                        static_cast<std::uint32_t>(std::max(k.top_k, 0)),
                        static_cast<std::uint32_t>(std::min<std::size_t>(k.min_keep, UINT32_MAX)),
                        static_cast<std::uint32_t>(request->penalties.size()),
                        c.vocab_size,
                        request->rng,
                        sampling_mode_ == SamplingMode::kFast ? 1u : 0u,
                        0};
    std::memcpy(step_host_.data() + sample_offset_, &sp, sizeof(sp));
    auto* pen = reinterpret_cast<HaloPenalty*>(step_host_.data() + penalty_offset_);
    for (std::size_t i = 0; i < request->penalties.size(); ++i) {
      const auto& t = request->penalties[i];
      pen[i] = {t.token, t.generated_count, t.repeated, 0};
    }
    upload = penalty_offset_ + request->penalties.size() * sizeof(HaloPenalty);
  }
  if (!dev_.Upload(ple_emb_, 0, step_host_.data(), upload, error)) return false;

  if (use_graphs_ && !timing_) {  // kernel timing stamps eager launches only
    rt::Graph& g = graphs_[static_cast<int>(head)];
    if (!g) {
      if (!dev_.BeginCapture(error)) return false;
      const bool recorded = Record(head, error);
      std::string end_error;
      const bool ended = dev_.EndCapture(&g, &end_error);
      if (!recorded || !ended) {
        if (recorded && error) *error = end_error;
        return false;
      }
    }
    if (!dev_.LaunchGraph(g, error)) return false;
  } else if (!Record(head, error)) {
    return false;
  }
  if (head == Head::kLogits && !dev_.Download(logits_, 0, logits.data(), c.vocab_size * 4, error)) return false;
  if (head == Head::kSample) {
    HaloSampleOut out{};
    if (!dev_.Download(sample_out_, 0, &out, sizeof(out), error)) return false;
    if (out.token >= c.vocab_size) {
      if (error) *error = out.token == 0xFFFFFFFFu ? "no finite logit to sample" : "device sampling did not finish";
      return false;
    }
    result->token = static_cast<std::int32_t>(out.token);
    result->rng = out.rng;
  }
  ngram_history_ = history;
  for (auto& s : attn_) s.blocks = complete;
  ++position_;
  return true;
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
  for (std::size_t l = 0; l < attn_.size(); ++l) {
    attn_[l].blocks = snap.blocks_[l];
    attn_[l].valid16 = 0;  // the restored K/V may differ anywhere
  }
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
  for (auto& s : attn_) {
    s.blocks = 0;  // KV beyond position is never read
    s.valid16 = 0;
  }
  ngram_history_.Reset();
  position_ = 0;
  return dev_.Fill(ple_history_, 0, error) && dev_.Synchronize(error);
}

bool FlashNextGpu::Prefill(std::span<const std::int32_t> tokens, std::span<float> logits, std::string* error) {
  if (!batched_prefill_ || !ChunkSupported()) {
    for (std::size_t i = 0; i < tokens.size(); ++i)
      if (!Step(tokens[i], i + 1 == tokens.size() ? logits : std::span<float>{}, error)) return false;
    return true;
  }
  if (!AllocChunk(error)) return false;
  const std::size_t chunk = std::max<std::uint32_t>(1, std::min(prefill_chunk_, ck_.cap));
  // The next chunk's n-gram rows are read on a host thread while this chunk
  // runs on the device. One read is in flight at a time: the prefetch starts
  // after this chunk's own read, and the next chunk then skips its read.
  const auto& c = w_.config;
  std::future<std::vector<float>> next;
  for (std::size_t i = 0; i < tokens.size(); i += chunk) {
    const std::size_t n = std::min(chunk, tokens.size() - i);
    std::vector<float> pre;
    if (next.valid()) pre = next.get();  // empty when the prefetch failed: the chunk reads itself
    const std::size_t i2 = i + n, n2 = std::min(chunk, tokens.size() - std::min(i2, tokens.size()));
    const auto prefetch = [&](const model::NgramHistory& after) {
      if (!prefill_ngram_prefetch_ || n2 == 0 || c.ple_layer < 0 || !ngram_) return;
      next = std::async(std::launch::async, [this, &c, hist = model::NgramHistory(after), toks = tokens.subspan(i2, n2)]() mutable {
        std::vector<std::uint32_t> rows(toks.size() * c.ple_heads);
        model::HashNgramRows(c, hist, toks, rows);
        std::vector<float> emb(toks.size() * c.PleEmbeddingDim());
        if (!ngram_->Read(rows, emb)) emb.clear();
        return emb;
      });
    };
    const bool ok = PrefillChunk(tokens.subspan(i, n), i + n == tokens.size() ? logits : std::span<float>{}, error,
                                 pre.empty() ? nullptr : &pre, prefetch);
    if (!ok) {
      if (next.valid()) next.wait();  // never leave a read running past the call
      return false;
    }
  }
  return true;
}

}  // namespace halo::engine
