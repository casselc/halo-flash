#include "engine/qwen35_gpu.hpp"

#include <algorithm>

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
std::uint32_t Blocks(std::size_t n, std::uint32_t per = kBlock) {
  return static_cast<std::uint32_t>((n + per - 1) / per);
}

}  // namespace

std::unique_ptr<Qwen35Gpu> Qwen35Gpu::Create(rt::Device& device, const model::Qwen35Weights& weights,
                                             const model::LoraAdapter* lora, std::uint32_t max_context,
                                             std::string* error) {
  auto m = std::unique_ptr<Qwen35Gpu>(new Qwen35Gpu(device, weights));
  if (!m->Init(lora, max_context, error)) return nullptr;
  return m;
}

Qwen35Gpu::~Qwen35Gpu() {
  for (auto& [_, t] : tensors_) dev_.Free(t.buf);
  for (auto& [_, l] : lora_) {
    dev_.Free(l.a);
    dev_.Free(l.b);
  }
  for (auto& b : owned_) dev_.Free(b);
}

rt::Buffer Qwen35Gpu::Scratch(std::size_t floats, std::string* error) {
  rt::Buffer b = dev_.Alloc(floats * sizeof(float), error);
  if (b) owned_.push_back(b);
  return b;
}

bool Qwen35Gpu::Upload(const model::TensorRef& t, std::string* error) {
  if (t.empty() || tensors_.contains(t.data)) return true;
  DevTensor d;
  d.ref = &t;
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

bool Qwen35Gpu::Launch(const char* name, rt::Dim3 grid, rt::Dim3 block, std::initializer_list<rt::Arg> args,
                       std::string* error) {
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    const auto id = dev_.Kernel(name, error);
    if (id < 0) return false;
    it = kernels_.emplace(name, id).first;
  }
  return dev_.Launch(it->second, grid, block, std::span<const rt::Arg>(args.begin(), args.size()), error);
}

bool Qwen35Gpu::MatVec(const model::TensorRef& t, const rt::Buffer& x, const rt::Buffer& y, std::string* error) {
  const auto& d = tensors_.at(t.data);
  if (d.matvec < 0) {
    if (error) *error = "no GPU matvec for " + std::string(t.name);
    return false;
  }
  const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
  const rt::Arg args[] = {rt::Arg::Buf(d.buf), rt::Arg::Buf(x), rt::Arg::Buf(y), rt::Arg::Val(rows),
                          rt::Arg::Val(cols), rt::Arg::Val<std::uint64_t>(t.RowBytes())};
  if (!dev_.Launch(d.matvec, {Blocks(rows, kBlock / 32)}, {kBlock}, args, error)) return false;
  if (auto it = lora_.find(t.data); it != lora_.end()) {
    const auto& l = it->second;
    const rt::Arg a_args[] = {rt::Arg::Buf(l.a), rt::Arg::Buf(x), rt::Arg::Buf(lora_tmp_), rt::Arg::Val(l.rank),
                              rt::Arg::Val(cols), rt::Arg::Val<std::uint64_t>(cols * 4ull)};
    const rt::Arg b_args[] = {rt::Arg::Buf(l.b), rt::Arg::Buf(lora_tmp_), rt::Arg::Buf(lora_out_), rt::Arg::Val(rows),
                              rt::Arg::Val(l.rank), rt::Arg::Val<std::uint64_t>(l.rank * 4ull)};
    const auto mv = kernels_.at("halo_matvec_f32");
    if (!dev_.Launch(mv, {Blocks(l.rank, kBlock / 32)}, {kBlock}, a_args, error) ||
        !dev_.Launch(mv, {Blocks(rows, kBlock / 32)}, {kBlock}, b_args, error) ||
        !Launch("halo_axpy", {Blocks(rows)}, {kBlock},
                {rt::Arg::Buf(y), rt::Arg::Buf(lora_out_), rt::Arg::Val(l.scale), rt::Arg::Val(rows)}, error))
      return false;
  }
  return true;
}

bool Qwen35Gpu::Init(const model::LoraAdapter* lora, std::uint32_t max_context, std::string* error) {
  const auto& c = w_.config;
  max_context_ = max_context;
  if (!dev_.LoadModule(kernels::CoreHsaco(), error)) return false;
  kernels_["halo_matvec_f32"] = dev_.Kernel("halo_matvec_f32", error);
  for (const auto* t : {&w_.token_embd, &w_.output_norm, &w_.output}) if (!Upload(*t, error)) return false;
  for (const auto& L : w_.layers)
    for (const auto* t : {&L.attn_norm, &L.post_attention_norm, &L.ssm_qkv, &L.ssm_gate, &L.ssm_conv1d, &L.ssm_alpha,
                          &L.ssm_beta, &L.ssm_dt, &L.ssm_a, &L.ssm_norm, &L.ssm_out, &L.attn_q, &L.attn_k, &L.attn_v,
                          &L.attn_out, &L.attn_q_norm, &L.attn_k_norm, &L.ffn_gate, &L.ffn_up, &L.ffn_down})
      if (!Upload(*t, error)) return false;
  std::uint32_t max_rank = 1;
  if (lora) {
    for (const auto& [name, pair] : lora->pairs) {
      // Find the base tensor with this name among the uploaded ones.
      const model::TensorRef* base = nullptr;
      for (auto& [_, t] : tensors_)
        if (t.ref->name == name) base = t.ref;
      if (!base) continue;  // adapter tensors outside this model's layers
      DevLora l;
      l.rank = static_cast<std::uint32_t>(pair.rank);
      l.scale = lora->Scale(pair);
      l.a = dev_.Alloc(pair.a.SizeBytes(), error);
      l.b = dev_.Alloc(pair.b.SizeBytes(), error);
      if (!l.a || !l.b || !dev_.Upload(l.a, 0, pair.a.data, pair.a.SizeBytes(), error) ||
          !dev_.Upload(l.b, 0, pair.b.data, pair.b.SizeBytes(), error))
        return false;
      max_rank = std::max(max_rank, l.rank);
      lora_.emplace(base->data, l);
    }
  }
  const std::size_t H = c.hidden_size, F = c.ffn, nh = c.num_heads, nkv = c.num_kv_heads, hd = c.head_dim;
  const std::size_t widest =
      std::max({H, F, static_cast<std::size_t>(c.ConvChannels()), 2 * nh * hd, static_cast<std::size_t>(c.vocab_size)});
  for (auto* b : {&res_, &xn_, &out_}) *b = Scratch(H, error);
  g_ = Scratch(F, error);
  up_ = Scratch(F, error);
  qkv_ = Scratch(c.ConvChannels(), error);
  conv_out_ = Scratch(c.ConvChannels(), error);
  z_ = Scratch(c.ValueDim(), error);
  alpha_ = Scratch(c.ssm_num_v_heads, error);
  beta_ = Scratch(c.ssm_num_v_heads, error);
  gdn_out_ = Scratch(c.ValueDim(), error);
  qg_ = Scratch(2 * nh * hd, error);
  q_ = Scratch(nh * hd, error);
  gate_ = Scratch(nh * hd, error);
  k_ = Scratch(nkv * hd, error);
  v_ = Scratch(nkv * hd, error);
  ctx_ = Scratch(nh * hd, error);
  scores_ = Scratch(nh * max_context, error);
  logits_ = Scratch(c.vocab_size, error);
  lora_tmp_ = Scratch(max_rank, error);
  lora_out_ = Scratch(widest, error);
  layers_.resize(c.num_layers);
  saved_.resize(c.num_layers);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    auto make = [&](std::vector<LayerState>& v) -> bool {
      auto& s = v[l];
      if (c.IsLinearLayer(l)) {
        s.conv = Scratch(std::size_t{c.ssm_conv_kernel - 1} * c.ConvChannels(), error);
        s.state = Scratch(std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim, error);
        return s.conv && s.state && dev_.Fill(s.conv, 0, error) && dev_.Fill(s.state, 0, error);
      }
      s.k = Scratch(max_context * nkv * hd, error);
      s.v = Scratch(max_context * nkv * hd, error);
      return s.k && s.v;
    };
    if (!make(layers_) || !make(saved_)) return false;
  }
  return dev_.Synchronize(error);
}

bool Qwen35Gpu::Step(std::int32_t token, std::span<float> hidden, std::span<float> logits, std::string* error) {
  const auto& c = w_.config;
  if (position_ >= max_context_ || token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
    if (error) *error = position_ >= max_context_ ? "context is full" : "token out of range";
    return false;
  }
  const std::uint32_t H = c.hidden_size, nh = c.num_heads, nkv = c.num_kv_heads, hd = c.head_dim;
  const auto& emb = tensors_.at(w_.token_embd.data);
  const rt::Arg eargs[] = {rt::Arg::Buf(emb.buf), rt::Arg::Buf(res_), rt::Arg::Val<std::uint32_t>(token),
                           rt::Arg::Val(H), rt::Arg::Val<std::uint64_t>(w_.token_embd.RowBytes())};
  if (!dev_.Launch(emb.embed, {Blocks(H)}, {kBlock}, eargs, error)) return false;
  const auto norm = [&](const rt::Buffer& x, const model::TensorRef& w, const rt::Buffer& y, std::uint32_t nvec,
                        std::uint32_t d) {
    return Launch("halo_rmsnorm", {nvec}, {kBlock},
                  {rt::Arg::Buf(x), rt::Arg::Buf(tensors_.at(w.data).buf), rt::Arg::Buf(y), rt::Arg::Val(d),
                   rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(0)},
                  error);
  };
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const auto& L = w_.layers[l];
    auto& S = layers_[l];
    if (!norm(res_, L.attn_norm, xn_, 1, H)) return false;
    if (L.linear) {
      const std::uint32_t ch = c.ConvChannels(), kd = c.KeyDim(), d = c.ssm_head_dim;
      if (!MatVec(L.ssm_qkv, xn_, qkv_, error) || !MatVec(L.ssm_gate, xn_, z_, error) ||
          !MatVec(L.ssm_alpha, xn_, alpha_, error) || !MatVec(L.ssm_beta, xn_, beta_, error) ||
          !Launch("halo_conv_step", {Blocks(ch)}, {kBlock},
                  {rt::Arg::Buf(qkv_), rt::Arg::Buf(S.conv), rt::Arg::Buf(tensors_.at(L.ssm_conv1d.data).buf),
                   rt::Arg::Buf(conv_out_), rt::Arg::Val(ch), rt::Arg::Val(c.ssm_conv_kernel)},
                  error) ||
          !Launch("halo_l2norm", {2 * c.ssm_num_k_heads}, {128},
                  {rt::Arg::Buf(conv_out_), rt::Arg::Val(d), rt::Arg::Val(1e-6F)}, error) ||
          !Launch("halo_gdn_step", {c.ssm_num_v_heads}, {d},
                  {rt::Arg::Buf(conv_out_), rt::Arg::Buf(conv_out_, kd * 4ull), rt::Arg::Buf(conv_out_, 2ull * kd * 4),
                   rt::Arg::Buf(alpha_), rt::Arg::Buf(beta_), rt::Arg::Buf(tensors_.at(L.ssm_a.data).buf),
                   rt::Arg::Buf(tensors_.at(L.ssm_dt.data).buf), rt::Arg::Buf(S.state), rt::Arg::Buf(gdn_out_),
                   rt::Arg::Val(c.ssm_num_k_heads), rt::Arg::Val(d)},
                  error) ||
          !Launch("halo_gated_norm", {c.ssm_num_v_heads}, {128},
                  {rt::Arg::Buf(gdn_out_), rt::Arg::Buf(tensors_.at(L.ssm_norm.data).buf), rt::Arg::Buf(z_),
                   rt::Arg::Val(d), rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(0)},
                  error) ||
          !MatVec(L.ssm_out, gdn_out_, out_, error))
        return false;
    } else {
      const std::size_t kv_off = std::size_t{position_} * nkv * hd * 4;
      if (!MatVec(L.attn_q, xn_, qg_, error) ||
          !Launch("halo_split_qgate", {nh}, {kBlock},
                  {rt::Arg::Buf(qg_), rt::Arg::Buf(q_), rt::Arg::Buf(gate_), rt::Arg::Val(hd)}, error) ||
          !norm(q_, L.attn_q_norm, q_, nh, hd) ||
          // K and V are written straight into the cache at this position.
          !MatVec(L.attn_k, xn_, k_, error) || !norm(k_, L.attn_k_norm, k_, nkv, hd) ||
          !Launch("halo_rope_neox", {nh}, {64},
                  {rt::Arg::Buf(q_), rt::Arg::Val(hd), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(position_),
                   rt::Arg::Val(c.rope_theta)},
                  error) ||
          !Launch("halo_rope_neox", {nkv}, {64},
                  {rt::Arg::Buf(k_), rt::Arg::Val(hd), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(position_),
                   rt::Arg::Val(c.rope_theta)},
                  error) ||
          !Launch("halo_copy", {Blocks(nkv * hd)}, {kBlock},
                  {rt::Arg::Buf(S.k, kv_off), rt::Arg::Buf(k_), rt::Arg::Val(nkv * hd)}, error) ||
          !MatVec(L.attn_v, xn_, v_, error) ||
          !Launch("halo_copy", {Blocks(nkv * hd)}, {kBlock},
                  {rt::Arg::Buf(S.v, kv_off), rt::Arg::Buf(v_), rt::Arg::Val(nkv * hd)}, error) ||
          !Launch("halo_attention_decode", {nh}, {kBlock},
                  {rt::Arg::Buf(q_), rt::Arg::Buf(S.k), rt::Arg::Buf(S.v), rt::Arg::Buf(gate_), rt::Arg::Buf(ctx_),
                   rt::Arg::Buf(scores_), rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(hd),
                   rt::Arg::Val(position_ + 1), rt::Arg::Val(max_context_)},
                  error) ||
          !MatVec(L.attn_out, ctx_, out_, error))
        return false;
    }
    if (!Launch("halo_add", {Blocks(H)}, {kBlock}, {rt::Arg::Buf(res_), rt::Arg::Buf(out_), rt::Arg::Val(H)}, error) ||
        !norm(res_, L.post_attention_norm, xn_, 1, H) || !MatVec(L.ffn_gate, xn_, g_, error) ||
        !MatVec(L.ffn_up, xn_, up_, error) ||
        !Launch("halo_silu_mul", {Blocks(c.ffn)}, {kBlock}, {rt::Arg::Buf(g_), rt::Arg::Buf(up_), rt::Arg::Val(c.ffn)},
                error) ||
        !MatVec(L.ffn_down, g_, out_, error) ||
        !Launch("halo_add", {Blocks(H)}, {kBlock}, {rt::Arg::Buf(res_), rt::Arg::Buf(out_), rt::Arg::Val(H)}, error))
      return false;
  }
  if (!norm(res_, w_.output_norm, xn_, 1, H)) return false;
  if (!logits.empty() && !MatVec(w_.output, xn_, logits_, error)) return false;
  ++position_;
  if (!hidden.empty() && !dev_.Download(xn_, 0, hidden.data(), H * sizeof(float), error)) return false;
  if (!logits.empty() && !dev_.Download(logits_, 0, logits.data(), c.vocab_size * sizeof(float), error)) return false;
  return hidden.empty() && logits.empty() ? dev_.Synchronize(error) : true;
}

bool Qwen35Gpu::AllocChunk(std::string* error) {
  if (chunk_ready_) return true;
  const auto& c = w_.config;
  const std::size_t T = kMaxChunk, H = c.hidden_size, F = c.ffn, nh = c.num_heads, nkv = c.num_kv_heads,
                    hd = c.head_dim;
  std::uint32_t max_rank = 1;
  for (const auto& [_, l] : lora_) max_rank = std::max(max_rank, l.rank);
  const std::size_t widest =
      std::max({H, F, static_cast<std::size_t>(c.ConvChannels()), 2 * nh * hd, std::size_t{c.ssm_num_v_heads}});
  auto& k = chunk_;
  for (auto* b : {&k.res, &k.xn, &k.out}) *b = Scratch(T * H, error);
  k.g = Scratch(T * F, error);
  k.up = Scratch(T * F, error);
  k.qkv = Scratch(T * c.ConvChannels(), error);
  k.conv = Scratch(T * c.ConvChannels(), error);
  k.z = Scratch(T * c.ValueDim(), error);
  k.alpha = Scratch(T * c.ssm_num_v_heads, error);
  k.beta = Scratch(T * c.ssm_num_v_heads, error);
  k.gdn = Scratch(T * c.ValueDim(), error);
  k.qg = Scratch(T * 2 * nh * hd, error);
  k.q = Scratch(T * nh * hd, error);
  k.gate = Scratch(T * nh * hd, error);
  k.k = Scratch(T * nkv * hd, error);
  k.v = Scratch(T * nkv * hd, error);
  k.ctx = Scratch(T * nh * hd, error);
  k.lora_tmp = Scratch(T * max_rank, error);
  k.lora_out = Scratch(T * widest, error);
  if (max_context_ <= kMaxBatchedAttention) {
    k.scores = Scratch(T * nh * max_context_, error);
    if (!k.scores) return false;
  }
  for (const auto* b : {&k.res, &k.xn, &k.out, &k.g, &k.up, &k.qkv, &k.conv, &k.z, &k.alpha, &k.beta, &k.gdn, &k.qg,
                        &k.q, &k.gate, &k.k, &k.v, &k.ctx, &k.lora_tmp, &k.lora_out})
    if (!*b) return false;
  chunk_ready_ = true;
  return true;
}

bool Qwen35Gpu::Gemm(const model::TensorRef& t, const rt::Buffer& x, const rt::Buffer& y, std::uint32_t tokens,
                     std::string* error) {
  const auto& d = tensors_.at(t.data);
  const auto rows = static_cast<std::uint32_t>(t.rows), cols = static_cast<std::uint32_t>(t.cols);
  if (t.type == core::GgmlType::kBF16 && cols % 128 == 0) {
    // Weights streamed 256 bytes per row at a time (docs/BENCHMARKS.md).
    if (!Launch("halo_gemm_bf16_f16_k128", {(tokens + 63) / 64, (rows + 63) / 64}, {128},
                {rt::Arg::Buf(d.buf), rt::Arg::Buf(x), rt::Arg::Buf(y), rt::Arg::Val(rows), rt::Arg::Val(cols),
                 rt::Arg::Val(tokens)},
                error))
      return false;
  } else if (t.type == core::GgmlType::kBF16 && cols % 32 == 0) {
    if (!Launch("halo_gemm_bf16_f16", {(tokens + 63) / 64, (rows + 127) / 128}, {256},
                {rt::Arg::Buf(d.buf), rt::Arg::Buf(x), rt::Arg::Buf(y), rt::Arg::Val(rows), rt::Arg::Val(cols),
                 rt::Arg::Val(tokens)},
                error))
      return false;
  } else if (t.type == core::GgmlType::kF32 && cols <= 8192) {
    if (!Launch("halo_gemm_f32_rows", {Blocks(rows, 8), tokens}, {kBlock},
                {rt::Arg::Buf(d.buf), rt::Arg::Buf(x), rt::Arg::Buf(y), rt::Arg::Val(rows), rt::Arg::Val(cols)}, error))
      return false;
  } else {
    // Other formats: a token at a time through the decode matvec.
    if (d.matvec < 0) {
      if (error) *error = "no GPU matvec for " + std::string(t.name);
      return false;
    }
    for (std::uint32_t i = 0; i < tokens; ++i) {
      const rt::Arg args[] = {rt::Arg::Buf(d.buf), rt::Arg::Buf(x, std::size_t{i} * cols * 4),
                              rt::Arg::Buf(y, std::size_t{i} * rows * 4), rt::Arg::Val(rows), rt::Arg::Val(cols),
                              rt::Arg::Val<std::uint64_t>(t.RowBytes())};
      if (!dev_.Launch(d.matvec, {Blocks(rows, kBlock / 32)}, {kBlock}, args, error)) return false;
    }
  }
  if (auto it = lora_.find(t.data); it != lora_.end()) {
    const auto& l = it->second;
    if (!Launch("halo_gemm_f32_rows", {Blocks(l.rank, 8), tokens}, {kBlock},
                {rt::Arg::Buf(l.a), rt::Arg::Buf(x), rt::Arg::Buf(chunk_.lora_tmp), rt::Arg::Val(l.rank),
                 rt::Arg::Val(cols)},
                error) ||
        !Launch("halo_gemm_f32_narrow", {Blocks(rows), tokens}, {kBlock},
                {rt::Arg::Buf(l.b), rt::Arg::Buf(chunk_.lora_tmp), rt::Arg::Buf(chunk_.lora_out), rt::Arg::Val(rows),
                 rt::Arg::Val(l.rank)},
                error) ||
        !Launch("halo_axpy", {Blocks(std::size_t{rows} * tokens)}, {kBlock},
                {rt::Arg::Buf(y), rt::Arg::Buf(chunk_.lora_out), rt::Arg::Val(l.scale), rt::Arg::Val(rows * tokens)},
                error))
      return false;
  }
  return true;
}

bool Qwen35Gpu::Prefill(std::span<const std::int32_t> tokens, std::span<float> hidden, std::string* error) {
  const std::size_t H = w_.config.hidden_size;
  if (!hidden.empty() && hidden.size() != tokens.size() * H) {
    if (error) *error = "hidden must hold tokens x hidden_size floats";
    return false;
  }
  for (std::size_t i = 0; i < tokens.size(); i += kMaxChunk) {
    const std::size_t n = std::min<std::size_t>(kMaxChunk, tokens.size() - i);
    if (!PrefillChunk(tokens.subspan(i, n), hidden.empty() ? hidden : hidden.subspan(i * H, n * H), error))
      return false;
  }
  return true;
}

bool Qwen35Gpu::PrefillChunk(std::span<const std::int32_t> tokens, std::span<float> hidden, std::string* error) {
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
  if (c.ssm_head_dim != 128 || c.ConvChannels() % 128 != 0) {
    if (error) *error = "Prefill needs DeltaNet head dim 128";
    return false;
  }
  if (!AllocChunk(error)) return false;
  auto& K = chunk_;
  const std::uint32_t H = c.hidden_size, nh = c.num_heads, nkv = c.num_kv_heads, hd = c.head_dim;
  const auto& emb = tensors_.at(w_.token_embd.data);
  for (std::uint32_t t = 0; t < T; ++t) {
    const rt::Arg eargs[] = {rt::Arg::Buf(emb.buf), rt::Arg::Buf(K.res, std::size_t{t} * H * 4),
                             rt::Arg::Val<std::uint32_t>(static_cast<std::uint32_t>(tokens[t])), rt::Arg::Val(H),
                             rt::Arg::Val<std::uint64_t>(w_.token_embd.RowBytes())};
    if (!dev_.Launch(emb.embed, {Blocks(H)}, {kBlock}, eargs, error)) return false;
  }
  const auto norm = [&](const rt::Buffer& x, const model::TensorRef& w, const rt::Buffer& y, std::uint32_t nvec,
                        std::uint32_t d) {
    return Launch("halo_rmsnorm", {nvec}, {kBlock},
                  {rt::Arg::Buf(x), rt::Arg::Buf(tensors_.at(w.data).buf), rt::Arg::Buf(y), rt::Arg::Val(d),
                   rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(0)},
                  error);
  };
  const auto buf = [&](const model::TensorRef& t) { return rt::Arg::Buf(tensors_.at(t.data).buf); };
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const auto& L = w_.layers[l];
    auto& S = layers_[l];
    if (!norm(K.res, L.attn_norm, K.xn, T, H)) return false;
    if (L.linear) {
      const std::uint32_t ch = c.ConvChannels(), kd = c.KeyDim(), vd = c.ValueDim(), nv = c.ssm_num_v_heads;
      if (!Gemm(L.ssm_qkv, K.xn, K.qkv, T, error) || !Gemm(L.ssm_gate, K.xn, K.z, T, error) ||
          !Gemm(L.ssm_alpha, K.xn, K.alpha, T, error) || !Gemm(L.ssm_beta, K.xn, K.beta, T, error) ||
          !Launch("halo_conv_l2norm_prefill", {ch / 128, T}, {128},
                  {rt::Arg::Buf(K.qkv), rt::Arg::Buf(S.conv), buf(L.ssm_conv1d), rt::Arg::Buf(K.conv), rt::Arg::Val(ch),
                   rt::Arg::Val(c.ssm_conv_kernel), rt::Arg::Val(2 * kd), rt::Arg::Val(1e-6F)},
                  error) ||
          !Launch("halo_conv_state_advance", {Blocks(ch)}, {kBlock},
                  {rt::Arg::Buf(K.qkv), rt::Arg::Buf(S.conv), rt::Arg::Val(ch), rt::Arg::Val(c.ssm_conv_kernel),
                   rt::Arg::Val(T)},
                  error) ||
          !Launch("halo_gdn_prefill_d128_norm", {nv}, {128},
                  {rt::Arg::Buf(K.conv), rt::Arg::Buf(K.conv, kd * 4ull), rt::Arg::Buf(K.conv, 2ull * kd * 4),
                   rt::Arg::Buf(K.alpha), rt::Arg::Buf(K.beta), buf(L.ssm_a), buf(L.ssm_dt), rt::Arg::Buf(S.state),
                   buf(L.ssm_norm), rt::Arg::Buf(K.z), rt::Arg::Buf(K.gdn), rt::Arg::Val(c.ssm_num_k_heads),
                   rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(0), rt::Arg::Val(T), rt::Arg::Val(ch),
                   rt::Arg::Val(ch), rt::Arg::Val(vd)},
                  error) ||
          !Gemm(L.ssm_out, K.gdn, K.out, T, error))
        return false;
    } else {
      if (!Gemm(L.attn_q, K.xn, K.qg, T, error) || !Gemm(L.attn_k, K.xn, K.k, T, error) ||
          !Gemm(L.attn_v, K.xn, K.v, T, error) || !norm(K.k, L.attn_k_norm, K.k, T * nkv, hd))
        return false;
      const std::size_t kv_base = std::size_t{position_} * nkv * hd * 4;
      if (K.scores) {
        // The whole chunk at once: split, norm and rotate every token's heads,
        // write the chunk's K and V into the cache, then each query attends
        // to the positions up to its own.
        if (!Launch("halo_split_qgate", {T * nh}, {kBlock},
                    {rt::Arg::Buf(K.qg), rt::Arg::Buf(K.q), rt::Arg::Buf(K.gate), rt::Arg::Val(hd)}, error) ||
            !norm(K.q, L.attn_q_norm, K.q, T * nh, hd) ||
            !Launch("halo_rope_neox_rows", {T * nh}, {64},
                    {rt::Arg::Buf(K.q), rt::Arg::Val(hd), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(position_),
                     rt::Arg::Val(nh), rt::Arg::Val(c.rope_theta)},
                    error) ||
            !Launch("halo_rope_neox_rows", {T * nkv}, {64},
                    {rt::Arg::Buf(K.k), rt::Arg::Val(hd), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(position_),
                     rt::Arg::Val(nkv), rt::Arg::Val(c.rope_theta)},
                    error) ||
            !Launch("halo_copy", {Blocks(std::size_t{T} * nkv * hd)}, {kBlock},
                    {rt::Arg::Buf(S.k, kv_base), rt::Arg::Buf(K.k), rt::Arg::Val(T * nkv * hd)}, error) ||
            !Launch("halo_copy", {Blocks(std::size_t{T} * nkv * hd)}, {kBlock},
                    {rt::Arg::Buf(S.v, kv_base), rt::Arg::Buf(K.v), rt::Arg::Val(T * nkv * hd)}, error) ||
            !Launch("halo_attention_prefill", {nh, T}, {kBlock},
                    {rt::Arg::Buf(K.q), rt::Arg::Buf(S.k), rt::Arg::Buf(S.v), rt::Arg::Buf(K.gate), rt::Arg::Buf(K.ctx),
                     rt::Arg::Buf(K.scores), rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(hd),
                     rt::Arg::Val(position_), rt::Arg::Val(max_context_)},
                    error) ||
            !Gemm(L.attn_out, K.ctx, K.out, T, error))
          return false;
      } else {
        for (std::uint32_t t = 0; t < T; ++t) {
          const std::uint32_t pos = position_ + t;
          const std::size_t qo = std::size_t{t} * nh * hd * 4, ko = std::size_t{t} * nkv * hd * 4,
                            kv_off = std::size_t{pos} * nkv * hd * 4;
          if (!Launch("halo_split_qgate", {nh}, {kBlock},
                      {rt::Arg::Buf(K.qg, std::size_t{t} * 2 * nh * hd * 4), rt::Arg::Buf(K.q, qo), rt::Arg::Buf(K.gate, qo),
                       rt::Arg::Val(hd)},
                      error) ||
              !Launch("halo_rmsnorm", {nh}, {kBlock},
                      {rt::Arg::Buf(K.q, qo), buf(L.attn_q_norm), rt::Arg::Buf(K.q, qo), rt::Arg::Val(hd),
                       rt::Arg::Val(c.rms_eps), rt::Arg::Val<std::uint32_t>(0)},
                      error) ||
              !Launch("halo_rope_neox", {nh}, {64},
                      {rt::Arg::Buf(K.q, qo), rt::Arg::Val(hd), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(pos),
                       rt::Arg::Val(c.rope_theta)},
                      error) ||
              !Launch("halo_rope_neox", {nkv}, {64},
                      {rt::Arg::Buf(K.k, ko), rt::Arg::Val(hd), rt::Arg::Val(c.rotary_dim), rt::Arg::Val(pos),
                       rt::Arg::Val(c.rope_theta)},
                      error) ||
              !Launch("halo_copy", {Blocks(nkv * hd)}, {kBlock},
                      {rt::Arg::Buf(S.k, kv_off), rt::Arg::Buf(K.k, ko), rt::Arg::Val(nkv * hd)}, error) ||
              !Launch("halo_copy", {Blocks(nkv * hd)}, {kBlock},
                      {rt::Arg::Buf(S.v, kv_off), rt::Arg::Buf(K.v, ko), rt::Arg::Val(nkv * hd)}, error) ||
              !Launch("halo_attention_decode", {nh}, {kBlock},
                      {rt::Arg::Buf(K.q, qo), rt::Arg::Buf(S.k), rt::Arg::Buf(S.v), rt::Arg::Buf(K.gate, qo),
                       rt::Arg::Buf(K.ctx, qo), rt::Arg::Buf(scores_), rt::Arg::Val(nh), rt::Arg::Val(nkv),
                       rt::Arg::Val(hd), rt::Arg::Val(pos + 1), rt::Arg::Val(max_context_)},
                      error))
            return false;
        }
        if (!Gemm(L.attn_out, K.ctx, K.out, T, error)) return false;
      }
    }
    if (!Launch("halo_add", {Blocks(std::size_t{T} * H)}, {kBlock},
                {rt::Arg::Buf(K.res), rt::Arg::Buf(K.out), rt::Arg::Val(T * H)}, error) ||
        !norm(K.res, L.post_attention_norm, K.xn, T, H) || !Gemm(L.ffn_gate, K.xn, K.g, T, error) ||
        !Gemm(L.ffn_up, K.xn, K.up, T, error) ||
        !Launch("halo_silu_mul", {Blocks(std::size_t{T} * c.ffn)}, {kBlock},
                {rt::Arg::Buf(K.g), rt::Arg::Buf(K.up), rt::Arg::Val(T * c.ffn)}, error) ||
        !Gemm(L.ffn_down, K.g, K.out, T, error) ||
        !Launch("halo_add", {Blocks(std::size_t{T} * H)}, {kBlock},
                {rt::Arg::Buf(K.res), rt::Arg::Buf(K.out), rt::Arg::Val(T * H)}, error))
      return false;
  }
  if (!norm(K.res, w_.output_norm, K.xn, T, H)) return false;
  position_ += T;
  if (!hidden.empty()) return dev_.Download(K.xn, 0, hidden.data(), hidden.size() * sizeof(float), error);
  return dev_.Synchronize(error);
}

bool Qwen35Gpu::Reset(std::string* error) {
  for (auto& s : layers_) {
    if (s.conv && (!dev_.Fill(s.conv, 0, error) || !dev_.Fill(s.state, 0, error))) return false;
  }
  position_ = 0;  // KV beyond position is never read
  return true;
}

bool Qwen35Gpu::SaveState(std::string* error) {
  const auto& c = w_.config;
  const std::size_t kv = std::size_t{position_} * c.num_kv_heads * c.head_dim;
  for (std::size_t l = 0; l < layers_.size(); ++l) {
    auto copy = [&](const rt::Buffer& dst, const rt::Buffer& src, std::size_t n) {
      return n == 0 || Launch("halo_copy", {Blocks(n)}, {kBlock},
                              {rt::Arg::Buf(dst), rt::Arg::Buf(src), rt::Arg::Val(static_cast<std::uint32_t>(n))}, error);
    };
    const auto& s = layers_[l];
    const auto& d = saved_[l];
    if (s.conv) {
      if (!copy(d.conv, s.conv, s.conv.bytes / 4) || !copy(d.state, s.state, s.state.bytes / 4)) return false;
    } else if (!copy(d.k, s.k, kv) || !copy(d.v, s.v, kv)) {
      return false;
    }
  }
  saved_position_ = position_;
  return dev_.Synchronize(error);
}

bool Qwen35Gpu::RestoreState(std::string* error) {
  std::swap(layers_, saved_);
  position_ = saved_position_;
  // Keep the saved copy intact for the next fork.
  const bool ok = SaveState(error);
  return ok;
}

}  // namespace halo::engine
