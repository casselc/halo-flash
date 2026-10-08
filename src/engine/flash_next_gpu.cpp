#include <future>
#include "engine/flash_next_gpu.hpp"

#include <algorithm>
#include <cstdio>
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
  // Poisoned: device work may still be queued against this engine's memory (a failed restore whose synchronize
  // failed). Nothing goes back to the allocator for reuse; the process exit releases it with the device context.
  if (poisoned_) return;
  (void)dev_.Synchronize(nullptr);
  for (auto& g : batch_graphs_) dev_.FreeGraph(g.graph);
  for (auto& sp : sessions_)
    if (sp) {
      if (sp.get() == sessions_[bound_].get()) SwapBound(*sp);  // the bound state back into its shell
      FreeSeq(*sp);
    }
  dev_.FreeHost(ple_rows_host_);
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
  {  // SR4 model fingerprint (FNV-1a over the tensor's identity and its first bytes)
    const auto mix = [&](const void* p, std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) upload_fp_ = (upload_fp_ ^ static_cast<const std::uint8_t*>(p)[i]) * 0x100000001b3ull;
    };
    if (upload_fp_ == 0) upload_fp_ = 0xcbf29ce484222325ull;
    mix(t.name.data(), t.name.size());
    const std::uint64_t dims[6] = {static_cast<std::uint64_t>(t.type), t.cols, t.rows, t.experts, t.file_offset, t.shard};
    mix(dims, sizeof(dims));
    mix(t.data, std::min<std::size_t>(64, t.SizeBytes()));
  }
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
  if (!dev_.Launch(it->second, grid, block, std::span<const rt::Arg>(args.begin(), args.size()), error)) {
    if (error) *error += std::string(" (launching ") + name + ")";  // a fault surfaces here under AMD_SERIALIZE_KERNEL
    return false;
  }
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
    if (!Launch((decode_opt_ & kDecodeHoistRowLoads) && cols <= 4096 ? "halo_matvec_f32_rowblock_u"
                                                                     : "halo_matvec_f32_rowblock",
                {rows}, {256},
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
  kv16_only_ = kv_f16_only_;
  index_window_ = kv_index_window_;
  if (const char* v = std::getenv("HALO_KV_INDEX_WINDOW")) index_window_ = v[0] != '0';  // A/B and tests
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
  if (c.ple_layer > 0) {
    ple_rows_host_ = static_cast<float*>(dev_.AllocHost(std::size_t{c.PleEmbeddingDim()} * 4, error));
    if (!ple_rows_host_) return false;
  }
  owned_.push_back(ple_emb_);
  ple_key_ = Scratch(D, error);
  ple_value_ = Scratch(H, error);
  ple_query_ = Scratch(D, error);
  ple_gated_ = Scratch(D, error);
  ple_norm_ = Scratch(D, error);
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
  // Decode attention scores: a head's row holds the visible positions, at most AttnVisibleStride() (the QSA
  // budget's positions plus a block), not max_context.
  scores_ = Scratch(std::size_t{c.num_heads} * AttnVisibleStride(), error);
  iq_ = Scratch(std::size_t{c.indexer_heads} * c.indexer_head_dim, error);
  ik_ = Scratch(c.indexer_head_dim, error);
  pooled_ = Scratch(c.indexer_head_dim, error);
  block_scores_ = Scratch(max_context / c.compress_ratio + 1, error);
  visible_ = Scratch(max_context, error);
  router_ = Scratch(c.num_experts, error);
  q8act_ = Scratch(1280, error);  // struct Q8Act in core.hip: 1024 words + 128 + 128 floats
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
  if (route_trace_) route_trace_buf_ = Scratch(std::size_t{c.num_layers} * c.num_experts_used, error);
  // Session 0: allocated like every later session, then bound.
  sessions_.push_back(std::make_unique<SeqState>());
  if (!AllocSeq(*sessions_[0], error)) return false;
  SwapBound(*sessions_[0]);
  bound_ = 0;
  model_fp_ = upload_fp_ ^ (lora_.size() * 0x9e3779b97f4a7c15ull);
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
    return (Launch((decode_opt_ & kDecodeHoistSmall) && c.hidden_size <= 4096 ? "halo_hc_norm_scales_u"
                                                                               : "halo_hc_norm_scales",
                   {c.hc_count}, {256},
                   {rt::Arg::Buf(res_), rt::Arg::Buf(hc_scale_), rt::Arg::Val(c.hidden_size), rt::Arg::Val(c.rms_eps)},
                   error)) &&
           Launch((decode_opt_ & kDecodeHcMixLds) && chunk <= 2048
                      ? ((decode_opt_ & kDecodeHoistSmall) ? "halo_hc_mix_a_cu" : "halo_hc_mix_a_c")
                      : "halo_hc_mix_a",
                  {down_blocks + inject_rows, parts}, {256},
                  {rt::Arg::Buf(res_), rt::Arg::Buf(tensors_.at(m.norm.data).buf), rt::Arg::Buf(hc_scale_),
                   rt::Arg::Buf(dn.qs), rt::Arg::Buf(dn.d), rt::Arg::Buf(inject ? tensors_.at(m.inject.data).buf : res_),
                   rt::Arg::Buf(xn_), rt::Arg::Buf(hc_lo_part_), rt::Arg::Buf(hc_inject_part_), rt::Arg::Val(c.hc_count),
                   rt::Arg::Val(c.hidden_size), rt::Arg::Val(c.hc_low_rank), rt::Arg::Val(down_blocks),
                   rt::Arg::Val(dn.qstride), rt::Arg::Val(chunk)},
                  error) &&
           Launch((decode_opt_ & kDecodeHoistSmall) && c.hc_low_rank <= 512 && parts <= 16 ? "halo_hc_mix_b_u"
                                                                                               : "halo_hc_mix_b",
                  {Blocks(c.hidden_size, 8)}, {256},
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
    const auto tail = [&]() {  // conv + L2, DeltaNet + gated norm, out
      return Launch("halo_conv_l2norm", {ch / 128}, {128},
                  {rt::Arg::Buf(qkv_), rt::Arg::Buf(s.conv), rt::Arg::Buf(tensors_.at(l.ssm_conv1d.data).buf),
                   rt::Arg::Buf(conv_out_), rt::Arg::Val(ch), rt::Arg::Val(c.ssm_conv_kernel), rt::Arg::Val(2 * kd),
                   rt::Arg::Val(c.rms_eps)},
                  error) &&
           Launch(decode_opt_ & kDecodeGdnNoSpill ? "halo_gdn_step_d128_norm_r" : "halo_gdn_step_d128_norm",
                  {c.ssm_num_v_heads}, {128},
                  {rt::Arg::Buf(conv_out_), rt::Arg::Buf(conv_out_, kd * 4ull), rt::Arg::Buf(conv_out_, 2ull * kd * 4),
                   rt::Arg::Buf(alpha_), rt::Arg::Buf(beta_), rt::Arg::Buf(tensors_.at(l.ssm_a.data).buf),
                   rt::Arg::Buf(tensors_.at(l.ssm_dt.data).buf), rt::Arg::Buf(s.state),
                   rt::Arg::Buf(tensors_.at(l.ssm_norm.data).buf), rt::Arg::Buf(z_), rt::Arg::Buf(gdn_out_),
                   rt::Arg::Val(c.ssm_num_k_heads), rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(1)},
                  error) &&
           MatVec(l.ssm_out, 0, gdn_out_, block_out_, error);
    };
    return Launch("halo_gemv_q8r_dual",
                  {b1 + Blocks(rz, 32)}, {256},
                  {rt::Arg::Buf(tq.qs), rt::Arg::Buf(tq.d), rt::Arg::Buf(qkv_), rt::Arg::Val(rq), rt::Arg::Val(tq.qstride),
                   rt::Arg::Buf(tz.qs), rt::Arg::Buf(tz.d), rt::Arg::Buf(z_), rt::Arg::Val(rz), rt::Arg::Val(tz.qstride),
                   rt::Arg::Buf(mixed_), rt::Arg::Val(H), rt::Arg::Val(b1)},
                  error) &&
           Launch((decode_opt_ & kDecodeHoistRowLoads) && H <= 4096 ? "halo_matvec_f32_rowblock2_u"
                                                                    : "halo_matvec_f32_rowblock2",
                  {2 * c.ssm_num_v_heads}, {256},
                  {rt::Arg::Buf(tensors_.at(l.ssm_alpha.data).buf), rt::Arg::Buf(tensors_.at(l.ssm_beta.data).buf),
                   rt::Arg::Buf(mixed_), rt::Arg::Buf(alpha_), rt::Arg::Buf(beta_), rt::Arg::Val(c.ssm_num_v_heads),
                   rt::Arg::Val(H)},
                  error) &&
           tail();
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
  // K / V rows: fp32 (with the fp16 copy filled lazily by prefill), or fp16 only (kv16_only_).
  const auto append_kv = [&](const rt::Buffer& f32, const rt::Buffer& f16, const rt::Buffer& row, std::uint32_t n) {
    if (!kv16_only_) return append(f32, row, kHaloPosition, n);
    return Launch("halo_copy_at_h", {Blocks(n)}, {kBlock},
                  {rt::Arg::Buf(f16), rt::Arg::Buf(row), params, rt::Arg::Val(kHaloPosition), rt::Arg::Val(n),
                   rt::Arg::Val(n)},
                  error);
  };
  // D2: the indexer's two BF16 projections in one launch (halo_matvec_bf16's sums, batched loads).
  const auto indexer_qk = [&]() {
    const auto& iq = l.indexer_q;
    const auto& ik = l.indexer_k;
    if ((decode_opt_ & kDecodeIndexerDual) && iq.type == core::GgmlType::kBF16 && ik.type == iq.type &&
        iq.cols == ik.cols && iq.cols % 32 == 0 && !lora_.contains(iq.data) && !lora_.contains(ik.data)) {
      const auto rq = static_cast<std::uint32_t>(iq.rows), rk = static_cast<std::uint32_t>(ik.rows);
      const std::uint32_t b1 = Blocks(rq, 8);
      return Launch("halo_matvec_bf16_dual", {b1 + Blocks(rk, 8)}, {256},
                    {rt::Arg::Buf(tensors_.at(iq.data).buf), rt::Arg::Buf(iq_), rt::Arg::Val(rq),
                     rt::Arg::Buf(tensors_.at(ik.data).buf), rt::Arg::Buf(ik_), rt::Arg::Val(rk), rt::Arg::Buf(mixed_),
                     rt::Arg::Val(static_cast<std::uint32_t>(iq.cols)), rt::Arg::Val(b1)},
                    error);
    }
    return MatVec(iq, 0, mixed_, iq_, error) && MatVec(ik, 0, mixed_, ik_, error);
  };
  const std::uint32_t ratio = c.compress_ratio, budget = c.indexer_top_k / ratio;
  const std::uint32_t max_blocks = max_context_ / ratio + 1;
  return (skip_attn_proj_ || (MatVec(l.attn_q, 0, mixed_, qg_, error) && MatVec(l.attn_k, 0, mixed_, k_, error) &&
                              MatVec(l.attn_v, 0, mixed_, v_, error))) &&
         Launch("halo_split_qgate", {nh}, {kBlock},
                {rt::Arg::Buf(qg_), rt::Arg::Buf(q_), rt::Arg::Buf(gate_), rt::Arg::Val(hd)}, error) &&
         norm(q_, l.attn_q_norm, nh, hd) && norm(k_, l.attn_k_norm, nkv, hd) && rope(q_, nh, hd, kHaloPosition, 1) &&
         rope(k_, nkv, hd, kHaloPosition, 1) && append_kv(s.k, s.k16, k_, nkv * hd) &&
         append_kv(s.v, s.v16, v_, nkv * hd) &&
         // QSA indexer: raw keys cached; the block holding this token is
         // pooled, normalized and rotated at its start every step, final
         // once complete (incomplete slots are never scored).
         indexer_qk() &&
         append(s.index_k, ik_, kHaloPosition, idim) && norm(iq_, l.indexer_q_norm, c.indexer_heads, idim) &&
         rope(iq_, c.indexer_heads, idim, kHaloPosition, 1) &&
         Launch("halo_pool_block", {1}, {128},
                {rt::Arg::Buf(s.index_k), rt::Arg::Buf(pooled_), params, rt::Arg::Val(ratio), rt::Arg::Val(idim)},
                error) &&
         norm(pooled_, l.indexer_k_norm, 1, idim) && rope(pooled_, 1, idim, kHaloBlock, ratio) &&
         append(s.block_k, pooled_, kHaloBlock, idim) &&
         // Selection on device with the reference's rule.
         Launch((decode_opt_ & kDecodeQsaScores) && c.indexer_heads <= 8 && c.indexer_heads * idim <= 2048 &&
                        idim % 8 == 0
                    ? "halo_qsa_scores_b"
                    : "halo_qsa_scores",
                {(decode_opt_ & kDecodeScoresWide) ? Blocks(max_blocks) : std::min(Blocks(max_blocks), 64u)}, {kBlock},
                {rt::Arg::Buf(iq_), rt::Arg::Buf(s.block_k), rt::Arg::Buf(block_scores_), params,
                 rt::Arg::Val(c.indexer_heads), rt::Arg::Val(idim)},
                error) &&
         Launch((decode_opt_ & kDecodeSelect256) ? "halo_qsa_select256" : "halo_qsa_select", {1},
                {(decode_opt_ & kDecodeSelect256) ? 256u : 1024u},
                {rt::Arg::Buf(block_scores_), rt::Arg::Buf(visible_), params, rt::Arg::Val(budget), rt::Arg::Val(ratio)},
                error) &&
         Launch(decode_attn_batched_ && hd % 32 == 0 && hd <= kBlock && kBlock <= 256
                    ? (kv16_only_ ? "halo_attention_select_bh" : "halo_attention_select_b")
                    : (kv16_only_ ? "halo_attention_select_h" : "halo_attention_select"),
                {nh}, {kBlock},
                {rt::Arg::Buf(q_), rt::Arg::Buf(kv16_only_ ? s.k16 : s.k), rt::Arg::Buf(kv16_only_ ? s.v16 : s.v),
                 rt::Arg::Buf(gate_), rt::Arg::Buf(visible_),
                 rt::Arg::Buf(ctx_), rt::Arg::Buf(scores_), params, rt::Arg::Val(nh), rt::Arg::Val(nkv),
                 rt::Arg::Val(hd), rt::Arg::Val(AttnVisibleStride())},
                error) &&
         (skip_attn_proj_ || MatVec(l.attn_out, 0, ctx_, block_out_, error));
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
  // D2: the router launch also quantizes x for the routed gate/up (kDecodeQ8ActOnce).
  const bool act_once = (decode_opt_ & kDecodeQ8ActOnce) && fused_moe_ && int8_act_moe_ && H <= 4096 && H % 32 == 0 &&
                        l.router.type == core::GgmlType::kF32 && l.router.rows <= 1024 && small_row_kernels_ &&
                        !lora_.contains(l.router.data) && l.ffn_gate_exps.type == core::GgmlType::kQ4_K &&
                        l.ffn_up_exps.type == core::GgmlType::kQ4_K && l.ffn_gate_exps.cols % 256 == 0;
  const auto router = [&]() {
    if (!act_once) return MatVec(l.router, 0, mixed_, router_, error);
    const auto rows = static_cast<std::uint32_t>(l.router.rows);
    return Launch("halo_router_rowblock_q8act", {std::max(rows, H / 32)}, {256},
                  {rt::Arg::Buf(tensors_.at(l.router.data).buf), rt::Arg::Buf(mixed_), rt::Arg::Buf(router_),
                   rt::Arg::Val(rows), rt::Arg::Val(H), rt::Arg::Buf(q8act_)},
                  error);
  };
  // Routing on device: softmax, top-k (ties keep the lower expert), renormalize.
  if (!router() ||
      !Launch((decode_opt_ & kDecodeTopkWarp) && c.num_experts <= 1024 && c.num_experts_used <= 16 ? "halo_router_topk_w"
                                                                                                 : "halo_router_topk",
              {1}, {kBlock},
              {rt::Arg::Buf(router_), rt::Arg::Buf(expert_ids_), rt::Arg::Buf(expert_weights_),
               rt::Arg::Val(c.num_experts), rt::Arg::Val(c.num_experts_used)},
              error))
    return false;
  if (route_trace_ && route_trace_buf_ &&
      !CopyFloats(rt::Buffer{static_cast<char*>(route_trace_buf_.device) + std::size_t{cur_layer_} * c.num_experts_used * 4,
                             c.num_experts_used * 4, route_trace_buf_.handle},
                  expert_ids_, c.num_experts_used, error))
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
  const bool q5k_fast = (decode_opt_ & kDecodeQ5kExperts) && gate.type == core::GgmlType::kQ5_K &&
                        up.type == gate.type && up.rows == gate.rows && up.cols == gate.cols && gate.cols % 256 == 0 &&
                        down_q8r && down.cols == gate.rows && down.cols <= 6144 &&
                        !(gate_up_fast && (down_q5_1 || down_q8r));
  const auto& sg = tensors_.at(l.shexp_gate.data);
  const auto& su = tensors_.at(l.shexp_up.data);
  const auto& sd = tensors_.at(l.shexp_down.data);
  const bool shexp_fast = fused_shexp_ && sg.qs && su.qs && sd.qs && l.shexp_gate_inp.type == core::GgmlType::kF32 &&
                          H <= 8192 && c.shared_expert_ff <= 8192 && l.shexp_gate.cols == H;
  // D2: the routed combine folded into the shared expert's down launch.
  const bool defer_combine =
      (decode_opt_ & kDecodeShexpCombine) && shexp_fast && ((gate_up_fast && (down_q5_1 || down_q8r)) || q5k_fast);
  if (gate_up_fast && (down_q5_1 || down_q8r)) {
    const auto ff = static_cast<std::uint32_t>(gate.rows), used = c.num_experts_used;
    // 4 rows per wave with precomputed x sub-block sums (inputs <= 4096).
    // int8 activations + integer dot (default; ~1.3x the float s2 kernel) or
    // float s2 with precomputed x sums; both for inputs <= 4096.
    const bool s2 = H <= 4096;
    const char* gu = !s2 ? "halo_moe_gate_up_q4_k" : int8_act_moe_ ? "halo_moe_gate_up_q4_k_dot" : "halo_moe_gate_up_q4_k_s2";
    if (act_once) gu = "halo_moe_gate_up_q4_k_dot_g";
    if (!Launch(gu, {Blocks(ff, s2 ? 16 : 8), used}, {256},
                {rt::Arg::Buf(tensors_.at(gate.data).buf), rt::Arg::Buf(tensors_.at(up.data).buf),
                 rt::Arg::Buf(expert_ids_), rt::Arg::Buf(act_once ? q8act_ : mixed_), rt::Arg::Buf(moe_h_),
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
    if (!down_ok || (!defer_combine && !Launch((decode_opt_ & kDecodeHoistSmall) && c.num_experts_used <= 16 ? "halo_moe_combine_u" : "halo_moe_combine", {Blocks(H)}, {kBlock},
                            {rt::Arg::Buf(moe_part_), rt::Arg::Buf(expert_weights_), rt::Arg::Buf(block_out_),
                             rt::Arg::Val(used), rt::Arg::Val(H)},
                            error)))
      return false;
  } else if (q5k_fast) {
    // D2: the Q5_K layer's experts in three launches instead of five per expert (same sums, same order).
    const auto ff = static_cast<std::uint32_t>(gate.rows), used = c.num_experts_used;
    if (!Launch("halo_moe_gate_up_q5_k_rows", {Blocks(ff, 8), used}, {256},
                {rt::Arg::Buf(tensors_.at(gate.data).buf), rt::Arg::Buf(tensors_.at(up.data).buf),
                 rt::Arg::Buf(expert_ids_), rt::Arg::Buf(mixed_), rt::Arg::Buf(moe_h_),
                 rt::Arg::Val<std::uint64_t>(gate.RowBytes() * gate.rows), rt::Arg::Val(ff), rt::Arg::Val(H),
                 rt::Arg::Val<std::uint64_t>(gate.RowBytes())},
                error) ||
        !Launch("halo_moe_down_q8r_r4", {Blocks(H, 32), used}, {256},
                {rt::Arg::Buf(dd.qs), rt::Arg::Buf(dd.d), rt::Arg::Buf(expert_ids_), rt::Arg::Buf(moe_h_),
                 rt::Arg::Buf(moe_part_), rt::Arg::Val(H), rt::Arg::Val(ff), rt::Arg::Val(dd.qstride)},
                error) ||
        (!defer_combine && !Launch((decode_opt_ & kDecodeHoistSmall) && c.num_experts_used <= 16 ? "halo_moe_combine_u" : "halo_moe_combine", {Blocks(H)}, {kBlock},
                                   {rt::Arg::Buf(moe_part_), rt::Arg::Buf(expert_weights_), rt::Arg::Buf(block_out_),
                                    rt::Arg::Val(used), rt::Arg::Val(H)},
                                   error)))
      return false;
  } else if (!dev_.Fill(block_out_, 0, error)) {
    return false;
  }
  for (std::uint32_t i = 0; !(gate_up_fast && (down_q5_1 || down_q8r)) && !q5k_fast && i < c.num_experts_used; ++i) {
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
  if (shexp_fast) {
    const auto sff = c.shared_expert_ff;
    const char* sgu = (decode_opt_ & kDecodeHoistRowLoads) && H <= 4096 ? "halo_shexp_gate_up_u"
                                                                           : "halo_shexp_gate_up";
    if (defer_combine)
      return Launch(sgu, {Blocks(sff, 16)}, {256},
                    {rt::Arg::Buf(sg.qs), rt::Arg::Buf(sg.d), rt::Arg::Buf(su.qs), rt::Arg::Buf(su.d),
                     rt::Arg::Buf(tensors_.at(l.shexp_gate_inp.data).buf), rt::Arg::Buf(mixed_), rt::Arg::Buf(eg_),
                     rt::Arg::Buf(sgate_), rt::Arg::Val(sff), rt::Arg::Val(H), rt::Arg::Val(sg.qstride)},
                    error) &&
             Launch((decode_opt_ & kDecodeHoistSmall) && c.num_experts_used <= 32 ? "halo_shexp_down_combine_u"
                                                                                    : "halo_shexp_down_combine",
                    {Blocks(H, 32)}, {256},
                    {rt::Arg::Buf(sd.qs), rt::Arg::Buf(sd.d), rt::Arg::Buf(eg_), rt::Arg::Buf(sgate_),
                     rt::Arg::Buf(moe_part_), rt::Arg::Buf(expert_weights_), rt::Arg::Val(c.num_experts_used),
                     rt::Arg::Buf(block_out_), rt::Arg::Val(H), rt::Arg::Val(sff), rt::Arg::Val(sd.qstride)},
                    error);
    return Launch(sgu, {Blocks(sff, 16)}, {256},
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

bool FlashNextGpu::Record(Head head, std::string* error, int part) {
  const auto& c = w_.config;
  const auto& emb = tensors_.at(w_.token_embd.data);
  const rt::Arg eargs[] = {rt::Arg::Buf(emb.buf), rt::Arg::Buf(embd_), rt::Arg::Buf(ple_emb_, params_offset_),
                           rt::Arg::Val(c.hidden_size), rt::Arg::Val<std::uint64_t>(w_.token_embd.RowBytes())};
  if (part != 2 && (!dev_.Launch(emb.embed, {Blocks(c.hidden_size)}, {kBlock}, eargs, error) ||
                    !Launch("halo_broadcast", {Blocks(c.HcDim())}, {kBlock},
                            {rt::Arg::Buf(res_), rt::Arg::Buf(embd_), rt::Arg::Val(c.hc_count),
                             rt::Arg::Val(c.hidden_size)},
                            error)))
    return false;
  const std::uint32_t split = c.ple_layer > 0 ? static_cast<std::uint32_t>(c.ple_layer) : 0;
  const std::uint32_t l_begin = part == 2 ? split : 0, l_end = part == 1 ? split : c.num_layers;
  const auto combine = [&]() {
    return Launch("halo_hc_combine", {Blocks(c.HcDim())}, {kBlock},
                  {rt::Arg::Buf(res_), rt::Arg::Buf(block_out_), rt::Arg::Buf(inject_), rt::Arg::Val(c.hc_count),
                   rt::Arg::Val(c.hidden_size)},
                  error);
  };
  const auto mix = [&](const model::HcMixer& m, bool inject) { return HcMix(m, inject, error); };
  for (std::uint32_t il = l_begin; il < l_end; ++il) {
    const auto& l = w_.layers[il];
    cur_layer_ = il;
    if (c.IsPleLayer(il) && !Ple(l, error)) return false;
    if (!mix(l.hc_attn, true)) return false;
    if (l.linear ? !Linear(l, linear_[il], error) : !Attention(l, attn_[il], error)) return false;
    if (!combine() || !mix(l.hc_ffn, true) || !Moe(l, error) || !combine()) return false;
  }
  if (head == Head::kNone || part == 1) return true;
  if (!mix(w_.hc_head, false) || !MatVec(w_.output, 0, mixed_, logits_, error)) return false;
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
  if (poisoned_) return PoisonedFail(error);
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
  if (poisoned_) return PoisonedFail(error);
  const auto& c = w_.config;
  if (position_ >= max_context_ || token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
    if (error) *error = position_ >= max_context_ ? "context is full" : "token out of range";
    return false;
  }
  if (!EnsureKv(position_ + 1, position_, error)) return false;
  // Host side of the step: n-gram rows and the step parameters, uploaded
  // together; everything else is device work (captured once, replayed).
  model::NgramHistory history = ngram_history_;
  // D2: with the split graph, the rows are read while the layers before the PLE layer run.
  const bool split = (decode_opt_ & kDecodeSplitGraph) && use_graphs_ && !timing_ && c.ple_layer > 0 &&
                     ple_rows_host_ != nullptr;
  std::vector<std::uint32_t> rows(c.ple_layer >= 0 ? c.ple_heads : 0);
  if (c.ple_layer >= 0) {
    model::HashNgramRows(c, history, std::span<const std::int32_t>(&token, 1), rows);
    const std::span<float> emb(reinterpret_cast<float*>(step_host_.data()), c.PleEmbeddingDim());
    if (!split && (!ngram_ || !ngram_->Read(rows, emb))) {
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
  const auto capture = [&](rt::Graph& g, int part) {
    if (g) return true;
    if (!dev_.BeginCapture(error)) return false;
    const bool recorded = Record(head, error, part);
    std::string end_error;
    const bool ended = dev_.EndCapture(&g, &end_error);
    if (!recorded || !ended) {
      if (recorded && error) *error = end_error;
      return false;
    }
    return true;
  };
  if (split) {
    if (!dev_.Upload(ple_emb_, params_offset_, step_host_.data() + params_offset_, upload - params_offset_, error) ||
        !capture(pre_graph_, 1) || !capture(post_graphs_[static_cast<int>(head)], 2) ||
        !dev_.LaunchGraph(pre_graph_, error))
      return false;
    const std::span<float> emb(ple_rows_host_, c.PleEmbeddingDim());
    if (!ngram_ || !ngram_->Read(rows, emb)) {
      if (error) *error = "n-gram table read failed";
      return false;
    }
    if (!dev_.UploadAsync(ple_emb_, 0, ple_rows_host_, emb.size_bytes(), error) ||
        !dev_.LaunchGraph(post_graphs_[static_cast<int>(head)], error))
      return false;
  } else if (!dev_.Upload(ple_emb_, 0, step_host_.data(), upload, error)) {
    return false;
  }

  if (split) {
    // launched above
  } else if (use_graphs_ && !timing_) {  // kernel timing stamps eager launches only
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
  if (mtp_track_ && bound_ == mtp_owner_) {
    // The draft block follows: its pending base step (last position's hidden, this token), then this step's
    // hidden becomes the pending one.
    if (MtpPending() && !MtpRun(token, &mtp_hpend_, {}, error)) return false;
    if (MtpKeepsHidden() && !CopyFloats(mtp_hpend_, res_, c.HcDim(), error)) return false;
  }
  ngram_history_ = history;
  for (auto& s : attn_) {
    s.blocks = complete;
    if (kv16_only_) s.valid16 = position_ + 1;  // the step appended its fp16 rows
  }
  ++position_;
  return true;
}

// ---- P6 sessions -------------------------------------------------------------------------------------------

struct FlashNextGpu::KvPage {
  rt::Device& dev;
  void* handle = nullptr;
  KvPage(rt::Device& d, void* h) : dev(d), handle(h) {}
  ~KvPage() { dev.ReleasePage(handle); }
  KvPage(const KvPage&) = delete;
  KvPage& operator=(const KvPage&) = delete;
};

void FlashNextGpu::SwapBound(SeqState& s) noexcept {
  std::swap(linear_, s.linear);
  std::swap(attn_, s.attn);
  std::swap(ple_history_, s.ple_history);
  std::swap(ngram_history_, s.ngram_history);
  std::swap(position_, s.position);
  for (int i = 0; i < 3; ++i) {
    std::swap(graphs_[i], s.graphs[i]);
    std::swap(post_graphs_[i], s.post_graphs[i]);
  }
  std::swap(pre_graph_, s.pre_graph);
  std::swap(kv_, s.kv);
}

namespace {
// Positions of slack mapped past the ones a step or chunk writes: tiled kernels may load (and mask) rows a
// tile beyond the last valid one.
constexpr std::uint32_t kKvSlackTokens = 512;
}  // namespace

bool FlashNextGpu::AllocSeq(SeqState& s, std::string* error) {
  const auto& c = w_.config;
  const std::size_t D = c.HcDim();
  const auto alloc = [&](std::size_t floats, bool zero) {
    rt::Buffer b = dev_.Alloc(std::max<std::size_t>(floats, 1) * 4, error);
    if (b && zero && !dev_.Fill(b, 0, error)) dev_.Free(b);
    return b;
  };
  s.ple_history = alloc(std::size_t{c.PleConvHistory()} * D, true);
  if (!s.ple_history) return false;
  s.linear.assign(c.num_layers, {});
  s.attn.assign(c.num_layers, {});
  const bool paged = dev_.PageBytes() > 0;
  std::uint32_t layer = 0;
  const auto region = [&](KvKind kind, std::size_t row_bytes, std::uint32_t row_tokens,
                          std::size_t capacity_bytes) -> rt::Buffer {
    KvRegion r;
    r.kind = kind;
    r.layer = layer;
    r.row_bytes = row_bytes;
    r.row_tokens = row_tokens;
    r.paged = paged;
    r.va = paged ? dev_.ReserveRange(capacity_bytes, error) : dev_.Alloc(capacity_bytes, error);
    if (!r.va) return {};
    s.kv.push_back(std::move(r));
    return s.kv.back().va;
  };
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    layer = l;
    if (c.IsLinearLayer(l)) {
      auto& ls = s.linear[l];
      ls.conv = alloc(std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels(), true);
      ls.state = alloc(std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim, true);
      if (!ls.conv || !ls.state) return false;
    } else {
      auto& as = s.attn[l];
      const std::size_t kv = c.AttentionKvDim(), idim = c.indexer_head_dim, mc = max_context_;
      // The kind table: fp32 K / V (unless kv16_only_), indexer keys, pooled blocks, fp16 K / V.
      if (!kv16_only_) {
        as.k = region(KvKind::kK, kv * 4, 1, mc * kv * 4);
        as.v = region(KvKind::kV, kv * 4, 1, mc * kv * 4);
        if (!as.k || !as.v) return false;
      }
      as.index_k = region(KvKind::kIndexK, idim * 4, 1, mc * idim * 4);
      as.block_k = region(KvKind::kBlockK, idim * 4, c.compress_ratio, (mc / c.compress_ratio + 1) * idim * 4);
      if (!as.index_k || !as.block_k) return false;
      if (prefill_kv16_ || kv16_only_) {
        const std::size_t halves = mc * kv;
        as.k16 = region(KvKind::kK16, kv * 2, 1, (halves + 1) / 2 * 4);
        as.v16 = region(KvKind::kV16, kv * 2, 1, (halves + 1) / 2 * 4);
        if (!as.k16 || !as.v16) return false;
      }
    }
  }
  return true;
}

void FlashNextGpu::FreeSeq(SeqState& s) noexcept {
  if (poisoned_) return;  // quarantine: no allocation of a poisoned engine is released for reuse
  (void)dev_.Synchronize(nullptr);
  for (auto& g : s.graphs) dev_.FreeGraph(g);
  for (auto& g : s.post_graphs) dev_.FreeGraph(g);
  dev_.FreeGraph(s.pre_graph);
  for (auto& l : s.linear) {
    dev_.Free(l.conv);
    dev_.Free(l.state);
  }
  dev_.Free(s.ple_history);
  for (auto& r : s.kv) {
    if (r.paged) {
      const std::size_t pb = dev_.PageBytes();
      for (std::size_t j = 0; j < r.pages.size(); ++j)
        if (r.pages[j]) (void)dev_.UnmapPage(r.va, j * pb, nullptr);
      r.pages.clear();  // the last holder releases each page
      dev_.ReleaseRange(r.va);
    } else {
      dev_.Free(r.va);
    }
  }
  s.kv.clear();
  s.linear.clear();
  s.attn.clear();
}

bool FlashNextGpu::EnsureKv(std::uint32_t tokens, std::uint32_t live, std::string* error) {
  const std::size_t pb = dev_.PageBytes();
  const std::uint64_t want = std::min<std::uint64_t>(std::uint64_t{tokens} + kKvSlackTokens, max_context_);
  bool synced = false;
  for (auto& r : kv_) {
    if (!r.paged) continue;
    const std::size_t rows = (want + r.row_tokens - 1) / r.row_tokens + 1;  // +1: block_k's incomplete row
    const std::size_t pages = std::min((rows * r.row_bytes + pb - 1) / pb, r.va.bytes / pb);
    // Indexer keys: pages wholly below the live row are released, the others (re)mapped.
    const std::size_t first = r.kind == KvKind::kIndexK ? std::size_t{IndexLiveRow(live)} * r.row_bytes / pb : 0;
    for (std::size_t j = 0; j < std::min(first, r.pages.size()); ++j)
      if (r.pages[j]) {
        if (!synced && !dev_.Synchronize(error)) return false;  // earlier work may still read the page
        synced = true;
        if (!dev_.UnmapPage(r.va, j * pb, error)) return false;
        r.pages[j].reset();  // the last holder (forks share pages) releases it
      }
    for (std::size_t j = first; j < pages; ++j) {
      if (j < r.pages.size() && r.pages[j]) continue;
      void* h = dev_.CreatePage(error);
      if (!h) return false;
      auto page = std::make_shared<KvPage>(dev_, h);
      if (!dev_.MapPage(r.va, j * pb, h, error)) return false;
      if (j >= r.pages.size()) r.pages.resize(j + 1);
      r.pages[j] = std::move(page);
    }
  }
  return true;
}

bool FlashNextGpu::HasSession(SessionId id) const noexcept { return id < sessions_.size() && sessions_[id]; }

std::uint32_t FlashNextGpu::SessionPositionOf(SessionId id) const {
  return id == bound_ ? position_ : sessions_.at(id)->position;
}

bool FlashNextGpu::NewSession(SessionId* id, std::string* error) {
  if (poisoned_) return PoisonedFail(error);
  auto s = std::make_unique<SeqState>();
  if (!AllocSeq(*s, error)) {
    FreeSeq(*s);
    return false;
  }
  *id = static_cast<SessionId>(sessions_.size());
  sessions_.push_back(std::move(s));
  return true;
}

bool FlashNextGpu::BindSession(SessionId id, std::string* error) {
  if (poisoned_) return PoisonedFail(error);
  if (!HasSession(id)) {
    if (error) *error = "no such session";
    return false;
  }
  if (id == bound_) return true;
  SwapBound(*sessions_[bound_]);  // park the bound one
  SwapBound(*sessions_[id]);
  bound_ = id;
  return true;
}

bool FlashNextGpu::DropSession(SessionId id, std::string* error) {
  if (poisoned_) return PoisonedFail(error);  // quarantine: the session and its memory stay allocated
  if (!HasSession(id) || id == bound_) {
    if (error) *error = id == bound_ ? "cannot drop the bound session" : "no such session";
    return false;
  }
  FreeSeq(*sessions_[id]);
  sessions_[id].reset();
  // Batch graphs that recorded its state.
  for (auto it = batch_graphs_.begin(); it != batch_graphs_.end();)
    if (std::find(it->sessions.begin(), it->sessions.end(), id) != it->sessions.end()) {
      dev_.FreeGraph(it->graph);
      it = batch_graphs_.erase(it);
    } else {
      ++it;
    }
  return true;
}

bool FlashNextGpu::ForkSession(SessionId src, SessionId* id, std::string* error) {
  if (poisoned_) return PoisonedFail(error);
  if (!HasSession(src)) {
    if (error) *error = "no such session";
    return false;
  }
  const auto& c = w_.config;
  // The source's state as seen from outside the bound slot.
  const bool src_bound = src == bound_;
  SeqState& from = *sessions_[src];
  if (src_bound) SwapBound(from);  // from now holds the bound state; swapped back below
  auto to = std::make_unique<SeqState>();
  bool ok = true;
  const std::size_t D = c.HcDim();
  const auto alloc_copy = [&](const rt::Buffer& b) {
    rt::Buffer n = dev_.Alloc(b.bytes, error);
    if (n && !CopyFloats(n, b, b.bytes / 4, error)) dev_.Free(n);
    return n;
  };
  to->ple_history = alloc_copy(from.ple_history);
  ok = static_cast<bool>(to->ple_history);
  to->linear.assign(c.num_layers, {});
  to->attn.assign(c.num_layers, {});
  for (std::uint32_t l = 0; ok && l < c.num_layers; ++l)
    if (c.IsLinearLayer(l)) {
      to->linear[l].conv = alloc_copy(from.linear[l].conv);
      to->linear[l].state = alloc_copy(from.linear[l].state);
      ok = to->linear[l].conv && to->linear[l].state;
    }
  (void)D;
  // KV: region by region, as the source's kind table lists them (AllocSeq's order). Every page is mapped before any
  // copy into one is launched: the device holds launches back until the GPU has settled the mappings
  // (rt::HipDevice, docs/SESSIONS.md section 11), so one wait per fork instead of one per copied page.
  const std::size_t pb = dev_.PageBytes();
  std::vector<std::pair<void*, void*>> copies;  // (fresh page in the fork, the source's page)
  for (std::size_t ri = 0; ok && ri < from.kv.size(); ++ri) {
    const KvRegion& fr = from.kv[ri];
    const auto& fa = from.attn[fr.layer];
    auto& ta = to->attn[fr.layer];
    ta.blocks = fa.blocks;
    ta.valid16 = fa.valid16;
    const auto k = static_cast<std::size_t>(fr.kind);
    rt::Buffer* bufs[6] = {&ta.k, &ta.v, &ta.index_k, &ta.block_k, &ta.k16, &ta.v16};
    // Rows no later step rewrites: positions below the source's position; complete QSA blocks; fp16 rows
    // already converted.
    const std::uint64_t immut[6] = {from.position, from.position, from.position, fa.blocks,
                                    std::min(fa.valid16, from.position), std::min(fa.valid16, from.position)};
    // Rows that hold data (copied when not shared): written positions, the incomplete block row.
    const std::uint64_t written[6] = {from.position, from.position, from.position, std::uint64_t{fa.blocks} + 1,
                                      fa.valid16, fa.valid16};
    {
      KvRegion tr;
      tr.kind = fr.kind;
      tr.layer = fr.layer;
      tr.row_bytes = fr.row_bytes;
      tr.row_tokens = fr.row_tokens;
      tr.paged = fr.paged;
      if (fr.paged) {
        tr.va = dev_.ReserveRange(fr.va.bytes, error);
        ok = static_cast<bool>(tr.va);
        const std::size_t shared = immut[k] * fr.row_bytes / pb;  // whole pages below the immutable bound
        for (std::size_t j = 0; ok && j < fr.pages.size(); ++j) {
          if (j < shared) {
            if (fr.pages[j]) ok = dev_.MapPage(tr.va, j * pb, fr.pages[j]->handle, error);  // null: released
            if (ok) tr.pages.push_back(fr.pages[j]);
          } else {
            void* h = dev_.CreatePage(error);
            ok = h != nullptr;
            if (!ok) break;
            auto page = std::make_shared<KvPage>(dev_, h);
            ok = dev_.MapPage(tr.va, j * pb, h, error);
            if (ok) {
              copies.push_back({static_cast<char*>(tr.va.device) + j * pb, static_cast<char*>(fr.va.device) + j * pb});
              tr.pages.push_back(std::move(page));
            }
          }
        }
      } else {
        tr.va = dev_.Alloc(fr.va.bytes, error);
        const std::size_t bytes = std::min<std::size_t>(written[k] * fr.row_bytes, fr.va.bytes);
        ok = tr.va && CopyFloats(tr.va, fr.va, bytes / 4, error);
      }
      if (tr.va) {
        *bufs[k] = tr.va;
        to->kv.push_back(std::move(tr));
      }
    }
  }
  for (std::size_t i = 0; ok && i < copies.size(); ++i)
    ok = CopyFloats(rt::Buffer{copies[i].first, pb, nullptr}, rt::Buffer{copies[i].second, pb, nullptr}, pb / 4, error);
  to->ngram_history = from.ngram_history;
  to->position = from.position;
  if (src_bound) SwapBound(from);
  ok = ok && dev_.Synchronize(error);
  if (!ok) {
    FreeSeq(*to);
    return false;
  }
  *id = static_cast<SessionId>(sessions_.size());
  sessions_.push_back(std::move(to));
  return true;
}

FlashNextGpu::SessionBytes FlashNextGpu::SessionMemory(SessionId id) const {
  SessionBytes b;
  const SeqState* s = id == bound_ ? nullptr : sessions_.at(id).get();
  const auto& lin = s ? s->linear : linear_;
  const auto& kv = s ? s->kv : kv_;
  b.recurrent = (s ? s->ple_history : ple_history_).bytes;
  for (const auto& l : lin) b.recurrent += l.conv.bytes + l.state.bytes;
  const std::size_t pb = dev_.PageBytes();
  for (const auto& r : kv) {
    if (!r.paged) {
      b.kv_mapped += r.va.bytes;
      b.kv_exclusive += r.va.bytes;
      continue;
    }
    for (const auto& p : r.pages) {
      if (!p) continue;  // released
      b.kv_mapped += pb;
      if (p.use_count() == 1) b.kv_exclusive += pb;
    }
  }
  return b;
}

double FlashNextGpu::KvBytesPerToken() const {
  double b = 0;
  for (const auto& r : kv_)
    if (!(index_window_ && r.kind == KvKind::kIndexK)) b += double(r.row_bytes) / r.row_tokens;
  return b;
}

bool FlashNextGpu::KvRoomFor(std::uint32_t positions, std::uint32_t prefill_tokens, std::string* why) const {
  const std::size_t avail = dev_.AvailableBytes();
  if (avail == SIZE_MAX) return true;  // the backend cannot tell
  // Positions the bound session's pages already hold (its K or fp16 K region: one row a position).
  std::uint64_t held = 0;
  for (const auto& r : kv_)
    if (r.kind == (kv16_only_ ? KvKind::kK16 : KvKind::kK)) {
      std::size_t mapped = 0;
      for (const auto& p : r.pages) mapped += p ? 1 : 0;
      held = r.paged ? mapped * dev_.PageBytes() / r.row_bytes : max_context_;
      break;
    }
  const std::uint64_t want = std::min<std::uint64_t>(std::uint64_t{positions} + kKvSlackTokens, max_context_);
  double need = want > held ? (want - held) * KvBytesPerToken() : 0.0;
  // The long-prompt chunk buffers, allocated on the first prompt longer than prefill_chunk_ (~0.7 MB a token).
  const std::uint32_t chunk = prefill_long_chunk_ > prefill_chunk_ && prefill_tokens > prefill_chunk_
                                  ? std::min(prefill_long_chunk_, max_context_)
                                  : std::min(prefill_chunk_, max_context_);
  if (prefill_tokens > 0 && ck_.cap < chunk) {
    const double per = ck_.cap ? double(ck_.bytes) / ck_.cap : 0.72e6;
    need += (chunk - ck_.cap) * per;
  }
  constexpr double kReserve = 2e9;  // driver, host buffers, the n-gram cache's growth
  if (need + kReserve <= double(avail)) return true;
  if (why) {
    char b[256];
    std::snprintf(b, sizeof(b),
                  "not enough memory for %u positions: needs %.1f GB more (KV %.1f KB a token), %.1f GB available",
                  positions, need / 1e9, KvBytesPerToken() / 1e3, avail / 1e9);
    *why = b;
  }
  return false;
}

bool FlashNextGpu::RouteTrace(std::vector<std::uint32_t>* ids, std::string* error) {
  const auto& c = w_.config;
  if (!route_trace_buf_) {
    if (error) *error = "routing trace was not enabled before Create";
    return false;
  }
  ids->resize(std::size_t{c.num_layers} * c.num_experts_used);
  return dev_.Download(route_trace_buf_, 0, ids->data(), ids->size() * 4, error);
}

template <typename F>
bool FlashNextGpu::ForEachState(F&& f) {
  const auto& c = w_.config;
  const std::size_t pos = position_;
  // Each state buffer with the float range [offset, offset + n) that holds live data.
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (c.IsLinearLayer(l)) {
      auto& s = linear_[l];
      if (!f(s.conv, 0, s.conv.bytes / 4) || !f(s.state, 0, s.state.bytes / 4)) return false;
    } else {
      auto& s = attn_[l];
      // fp16-only K / V: pos rows of fp16 pairs (kv dim even), counted in floats. Indexer keys: the live window.
      const std::size_t live = IndexLiveRow(position_);
      if ((kv16_only_ ? !f(s.k16, 0, (pos * c.AttentionKvDim() + 1) / 2) || !f(s.v16, 0, (pos * c.AttentionKvDim() + 1) / 2)
                      : !f(s.k, 0, pos * c.AttentionKvDim()) || !f(s.v, 0, pos * c.AttentionKvDim())) ||
          !f(s.index_k, live * c.indexer_head_dim, (pos - live) * c.indexer_head_dim) ||
          !f(s.block_k, 0, std::size_t{s.blocks} * c.indexer_head_dim))
        return false;
    }
  }
  return f(ple_history_, 0, ple_history_.bytes / 4);
}

bool FlashNextGpu::CopyFloats(const rt::Buffer& dst, const rt::Buffer& src, std::size_t n, std::string* error) {
  return n == 0 || Launch("halo_copy", {Blocks(n)}, {kBlock},
                          {rt::Arg::Buf(dst), rt::Arg::Buf(src), rt::Arg::Val(static_cast<std::uint32_t>(n))}, error);
}

std::unique_ptr<FlashNextGpu::Snapshot> FlashNextGpu::NewSnapshot(std::string*) {
  // Buffers are sized by Save to what the state holds then (not the KV capacity at max_context).
  return std::unique_ptr<Snapshot>(new Snapshot(dev_));
}

bool FlashNextGpu::Save(Snapshot& snap, std::string* error) {
  std::size_t i = 0;
  snap.valid_ = false;
  if (!ForEachState([&](const rt::Buffer& live, std::size_t off, std::size_t n) {
        if (snap.buffers_.size() <= i) snap.buffers_.emplace_back();
        rt::Buffer& b = snap.buffers_[i++];
        if (b.bytes < std::max<std::size_t>(n, 1) * 4) {  // grown on demand
          if (!dev_.Synchronize(error)) return false;
          dev_.Free(b);
          b = dev_.Alloc(std::max<std::size_t>(n, 1) * 4, error);
          if (!b) return false;
        }
        return CopyFloats(b, rt::Buffer{static_cast<char*>(live.device) + off * 4, n * 4, nullptr}, n, error);
      }))
    return false;
  snap.blocks_.clear();
  for (const auto& s : attn_) snap.blocks_.push_back(s.blocks);
  snap.ngram_history_ = ngram_history_;
  snap.position_ = position_;
  snap.valid_ = true;
  return dev_.Synchronize(error);
}

bool FlashNextGpu::Restore(const Snapshot& snap, std::string* error) {
  if (poisoned_) return PoisonedFail(error);
  if (!snap.valid_ || &snap.dev_ != &dev_) {
    if (error) *error = "snapshot is empty or from another device";
    return false;
  }
  // Fresh pages for the copy: the bound session's own may be shared with a fork (see Reset).
  if (!ReleaseKvPages(error) || !EnsureKv(snap.position_, snap.position_, error)) return false;
  // Counts first: ForEachState sizes the KV copies from the restored position.
  position_ = snap.position_;
  for (std::size_t l = 0; l < attn_.size(); ++l) {
    attn_[l].blocks = snap.blocks_[l];
    attn_[l].valid16 = kv16_only_ ? position_ : 0;  // fp32 K/V: the restored rows may differ anywhere
  }
  ngram_history_ = snap.ngram_history_;
  std::size_t i = 0;
  return ForEachState([&](const rt::Buffer& live, std::size_t off, std::size_t n) {
           return CopyFloats(rt::Buffer{static_cast<char*>(live.device) + off * 4, n * 4, nullptr}, snap.buffers_[i++],
                             n, error);
         }) &&
         dev_.Synchronize(error);
}

bool FlashNextGpu::ReleaseKvPages(std::string* error) {
  if (poisoned_) return PoisonedFail(error);  // quarantine: pages stay mapped
  if (!dev_.Synchronize(error)) return false;  // queued work may still read them
  const std::size_t pb = dev_.PageBytes();
  for (auto& r : kv_) {
    if (!r.paged) continue;
    // Only pages a fork (or a fork's fork) also holds go; pages this session alone holds stay mapped where they
    // are and are rewritten in place. Releasing those too returned them to the HIP runtime, which keeps them
    // charged to GTT (so KvRoomFor refused a prompt the process could hold), and pooling them for a remap at
    // another address faulted (GPU aperture violation in the swarm at 2 MB pages, stream C t13/t14).
    for (std::size_t j = 0; j < r.pages.size(); ++j)
      if (r.pages[j] && r.pages[j].use_count() > 1) {
        if (!dev_.UnmapPage(r.va, j * pb, error)) return false;
        r.pages[j].reset();  // the fork keeps it
      }
  }
  return true;
}

bool FlashNextGpu::Reset(std::string* error) {
  if (poisoned_) return PoisonedFail(error);
  const auto& c = w_.config;
  // The bound session's shared KV pages go: a fork holds them, and the rewritten positions must not reach it
  // (sessions only append, so shared pages are otherwise never written). EnsureKv maps fresh ones as needed;
  // pages only this session holds are kept and rewritten in place.
  if (!ReleaseKvPages(error)) return false;
  for (std::uint32_t l = 0; l < c.num_layers; ++l)
    if (c.IsLinearLayer(l) && (!dev_.Fill(linear_[l].conv, 0, error) || !dev_.Fill(linear_[l].state, 0, error)))
      return false;
  for (auto& s : attn_) {
    s.blocks = 0;  // KV beyond position is never read
    s.valid16 = 0;
  }
  ngram_history_.Reset();
  position_ = 0;
  if (bound_ == mtp_owner_) mtp_pos_ = 0, mtp_base_ = 0;
  return dev_.Fill(ple_history_, 0, error) && dev_.Synchronize(error);
}

bool FlashNextGpu::Prefill(std::span<const std::int32_t> tokens, std::span<float> logits, std::string* error,
                           const std::function<bool()>& cancelled) {
  const auto stop = [&] {
    if (!cancelled || !cancelled()) return false;
    if (error) *error = "cancelled";
    return true;
  };
  if (!batched_prefill_ || !ChunkSupported()) {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (i > 0 && stop()) return false;
      if (!Step(tokens[i], i + 1 == tokens.size() ? logits : std::span<float>{}, error)) return false;
    }
    return true;
  }
  const std::uint32_t want =
      prefill_long_chunk_ > prefill_chunk_ && tokens.size() > prefill_chunk_ ? prefill_long_chunk_ : prefill_chunk_;
  if (!AllocChunk(want, error)) return false;
  const std::size_t chunk = std::max<std::uint32_t>(1, std::min(want, ck_.cap));
  // The next chunk's n-gram rows are read on a host thread while this chunk
  // runs on the device. One read is in flight at a time: the prefetch starts
  // after this chunk's own read, and the next chunk then skips its read.
  const auto& c = w_.config;
  std::future<std::vector<float>> next;
  for (std::size_t i = 0; i < tokens.size(); i += chunk) {
    if (i > 0 && stop()) {
      if (next.valid()) next.wait();  // never leave a read running past the call
      return false;
    }
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
