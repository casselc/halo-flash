#include "model/qwen35.hpp"

#include <algorithm>
#include <cmath>

#include "model/cpu_ops.hpp"

namespace halo::model {

namespace {

using core::GgmlType;

struct Binder {
  const core::GgufReader& r;
  std::string* error;
  bool ok = true;
  void Fail(const std::string& m) {
    if (ok && error) *error = m;
    ok = false;
  }
  std::uint32_t U32(const std::string& key) {
    const auto v = r.GetMetadataUint32(key);
    if (!v) Fail("missing GGUF integer " + key);
    return v.value_or(0);
  }
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows, bool required = true) {
    TensorRef t;
    const auto* info = r.FindTensor(name);
    if (!info) {
      if (required) Fail("missing tensor " + name);
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t c = d.size() > 0 ? d[0] : 1, rr = d.size() > 1 ? d[1] : 1;
    if (c != cols || rr != rows || d.size() > 2) {
      Fail("tensor " + name + " has an unexpected shape");
      return t;
    }
    const auto type = info->type;
    if (type != GgmlType::kF32 && type != GgmlType::kF16 && type != GgmlType::kBF16 && type != GgmlType::kQ8_0 &&
        type != GgmlType::kQ4_K && type != GgmlType::kQ5_K && type != GgmlType::kQ6_K) {
      Fail("tensor " + name + " has unsupported format " + std::string(core::ToString(type)));
      return t;
    }
    t.data = info->data;
    t.type = type;
    t.cols = cols;
    t.rows = rows;
    t.name = info->name;
    return t;
  }
};

void RmsNormOnePlus(std::span<float> x, const TensorRef& w, float eps) {
  // GGUF stores HF's zero-centred gamma with the +1 folded in.
  cpu::RmsNorm(x, static_cast<const float*>(w.data), eps);
}

}  // namespace

std::optional<Qwen35Weights> Qwen35Weights::Bind(const core::GgufReader& r, std::string* error) {
  Binder b{r, error};
  if (r.GetMetadataString("general.architecture") != std::optional<std::string_view>{"qwen35"}) {
    if (error) *error = "GGUF architecture is not qwen35";
    return std::nullopt;
  }
  Qwen35Weights w;
  auto& c = w.config;
  const std::string p = "qwen35.";
  const auto total = b.U32(p + "block_count");
  const auto nextn = r.GetMetadataUint32(p + "nextn_predict_layers").value_or(0);
  c.num_layers = total - std::min(total, nextn);
  c.hidden_size = b.U32(p + "embedding_length");
  c.ffn = b.U32(p + "feed_forward_length");
  c.num_heads = b.U32(p + "attention.head_count");
  c.num_kv_heads = b.U32(p + "attention.head_count_kv");
  c.head_dim = b.U32(p + "attention.key_length");
  c.rotary_dim = b.U32(p + "rope.dimension_count");
  c.rope_theta = r.GetMetadataFloat32(p + "rope.freq_base").value_or(0);
  c.rms_eps = r.GetMetadataFloat32(p + "attention.layer_norm_rms_epsilon").value_or(1e-6F);
  c.full_attention_interval = b.U32(p + "full_attention_interval");
  c.ssm_conv_kernel = b.U32(p + "ssm.conv_kernel");
  c.ssm_head_dim = b.U32(p + "ssm.state_size");
  c.ssm_num_k_heads = b.U32(p + "ssm.group_count");
  c.ssm_num_v_heads = b.U32(p + "ssm.time_step_rank");
  if (!b.ok) return std::nullopt;
  if (c.num_layers == 0 || c.full_attention_interval == 0 || c.num_kv_heads == 0 ||
      c.num_heads % c.num_kv_heads != 0 || c.rotary_dim > c.head_dim || c.rotary_dim % 2 != 0 ||
      c.ssm_num_k_heads == 0 || c.ssm_num_v_heads % c.ssm_num_k_heads != 0 || !(c.rope_theta > 0)) {
    if (error) *error = "qwen35 metadata describes an unsupported shape";
    return std::nullopt;
  }
  const auto* embd = r.FindTensor("token_embd.weight");
  if (!embd || embd->dimensions.size() != 2) {
    if (error) *error = "missing token_embd.weight";
    return std::nullopt;
  }
  c.vocab_size = static_cast<std::uint32_t>(embd->dimensions[1]);
  const std::uint64_t H = c.hidden_size;
  w.token_embd = b.Get("token_embd.weight", H, c.vocab_size);
  w.output_norm = b.Get("output_norm.weight", H, 1);
  w.output = b.Get("output.weight", H, c.vocab_size, false);
  if (w.output.empty()) w.output = w.token_embd;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::string q = "blk." + std::to_string(l) + ".";
    Qwen35Layer L;
    L.linear = c.IsLinearLayer(l);
    L.attn_norm = b.Get(q + "attn_norm.weight", H, 1);
    L.post_attention_norm = b.Get(q + "post_attention_norm.weight", H, 1);
    if (L.linear) {
      L.ssm_qkv = b.Get(q + "attn_qkv.weight", H, c.ConvChannels());
      L.ssm_gate = b.Get(q + "attn_gate.weight", H, c.ValueDim());
      L.ssm_conv1d = b.Get(q + "ssm_conv1d.weight", c.ssm_conv_kernel, c.ConvChannels());
      L.ssm_alpha = b.Get(q + "ssm_alpha.weight", H, c.ssm_num_v_heads);
      L.ssm_beta = b.Get(q + "ssm_beta.weight", H, c.ssm_num_v_heads);
      L.ssm_dt = b.Get(q + "ssm_dt.bias", c.ssm_num_v_heads, 1);
      L.ssm_a = b.Get(q + "ssm_a", c.ssm_num_v_heads, 1);
      L.ssm_norm = b.Get(q + "ssm_norm.weight", c.ssm_head_dim, 1);
      L.ssm_out = b.Get(q + "ssm_out.weight", c.ValueDim(), H);
    } else {
      L.attn_q = b.Get(q + "attn_q.weight", H, 2ull * c.num_heads * c.head_dim);
      L.attn_k = b.Get(q + "attn_k.weight", H, 1ull * c.num_kv_heads * c.head_dim);
      L.attn_v = b.Get(q + "attn_v.weight", H, 1ull * c.num_kv_heads * c.head_dim);
      L.attn_out = b.Get(q + "attn_output.weight", 1ull * c.num_heads * c.head_dim, H);
      L.attn_q_norm = b.Get(q + "attn_q_norm.weight", c.head_dim, 1);
      L.attn_k_norm = b.Get(q + "attn_k_norm.weight", c.head_dim, 1);
    }
    L.ffn_gate = b.Get(q + "ffn_gate.weight", H, c.ffn);
    L.ffn_up = b.Get(q + "ffn_up.weight", H, c.ffn);
    L.ffn_down = b.Get(q + "ffn_down.weight", c.ffn, H);
    w.layers.push_back(L);
  }
  // Norm vectors are read directly as floats.
  for (const auto& L : w.layers)
    for (const TensorRef* t : {&L.attn_norm, &L.post_attention_norm, &L.ssm_norm, &L.attn_q_norm, &L.attn_k_norm,
                               &L.ssm_dt, &L.ssm_a, &L.ssm_conv1d})
      if (!t->empty() && t->type != GgmlType::kF32) b.Fail(std::string(t->name) + " must be F32");
  if (w.output_norm.type != GgmlType::kF32) b.Fail("output_norm must be F32");
  if (!b.ok) return std::nullopt;
  return w;
}

Qwen35Reference::Qwen35Reference(const Qwen35Weights& weights, std::uint32_t max_context)
    : w_(&weights), max_context_(max_context) {
  const auto& c = w_->config;
  linear_.resize(c.num_layers);
  attn_.resize(c.num_layers);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (c.IsLinearLayer(l)) {
      linear_[l].conv.assign(static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.ConvChannels(), 0.0F);
      linear_[l].state.assign(static_cast<std::size_t>(c.ssm_num_v_heads) * c.ssm_head_dim * c.ssm_head_dim, 0.0F);
    }
  }
}

void Qwen35Reference::MatVec(const TensorRef& w, std::span<const float> x, std::span<float> y) const {
  cpu::MatVec(w, 0, x, y);
  if (lora_) {
    if (const LoraPair* p = lora_->Find(w.name)) {
      std::vector<float> ax(p->rank), bax(y.size());
      cpu::MatVec(p->a, 0, x, ax);
      cpu::MatVec(p->b, 0, ax, bax);
      const float s = lora_->Scale(*p);
      for (std::size_t i = 0; i < y.size(); ++i) y[i] += s * bax[i];
    }
  }
}

void Qwen35Reference::Linear(const Qwen35Layer& l, std::size_t index, std::span<const float> x, std::span<float> out) {
  const auto& c = w_->config;
  auto& s = linear_[index];
  const std::uint32_t channels = c.ConvChannels(), kd = c.KeyDim(), vd = c.ValueDim(), d = c.ssm_head_dim;
  const std::uint32_t kern = c.ssm_conv_kernel;
  std::vector<float> qkv(channels), z(vd), alpha(c.ssm_num_v_heads), beta(c.ssm_num_v_heads);
  MatVec(l.ssm_qkv, x, qkv);
  MatVec(l.ssm_gate, x, z);
  MatVec(l.ssm_alpha, x, alpha);
  MatVec(l.ssm_beta, x, beta);
  const auto* cw = static_cast<const float*>(l.ssm_conv1d.data);
  std::vector<float> conv(channels);
  for (std::uint32_t ch = 0; ch < channels; ++ch) {
    float acc = cw[static_cast<std::size_t>(ch) * kern + kern - 1] * qkv[ch];
    for (std::uint32_t k = 0; k + 1 < kern; ++k)
      acc += cw[static_cast<std::size_t>(ch) * kern + k] * s.conv[static_cast<std::size_t>(k) * channels + ch];
    conv[ch] = cpu::Silu(acc);
  }
  if (kern > 1) {
    std::copy(s.conv.begin() + channels, s.conv.end(), s.conv.begin());
    std::copy(qkv.begin(), qkv.end(), s.conv.end() - channels);
  }
  float* q = conv.data();
  float* k = conv.data() + kd;
  float* v = conv.data() + 2 * kd;
  for (std::uint32_t h = 0; h < c.ssm_num_k_heads; ++h) {
    cpu::L2Norm(std::span<float>(q + static_cast<std::size_t>(h) * d, d), 1e-6F);
    cpu::L2Norm(std::span<float>(k + static_cast<std::size_t>(h) * d, d), 1e-6F);
  }
  const auto* a = static_cast<const float*>(l.ssm_a.data);
  const auto* dt = static_cast<const float*>(l.ssm_dt.data);
  const auto* nw = static_cast<const float*>(l.ssm_norm.data);
  const float qs = 1.0F / std::sqrt(static_cast<float>(d));
  std::vector<float> attn(vd), u(d);
  for (std::uint32_t h = 0; h < c.ssm_num_v_heads; ++h) {
    // The GGUF converter tiles value heads: value head h reads key head h % n_k.
    const float* qh = q + static_cast<std::size_t>(h % c.ssm_num_k_heads) * d;
    const float* kh = k + static_cast<std::size_t>(h % c.ssm_num_k_heads) * d;
    const float* vh = v + static_cast<std::size_t>(h) * d;
    float* S = s.state.data() + static_cast<std::size_t>(h) * d * d;
    const float decay = std::exp(a[h] * cpu::Softplus(alpha[h] + dt[h]));
    const float b = cpu::Sigmoid(beta[h]);
    for (std::uint32_t j = 0; j < d; ++j) {
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] *= decay;
        acc += static_cast<double>(row[i]) * kh[i];
      }
      u[j] = static_cast<float>(acc);
    }
    float* o = attn.data() + static_cast<std::size_t>(h) * d;
    for (std::uint32_t j = 0; j < d; ++j) {
      const float delta = (vh[j] - u[j]) * b;
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] += delta * kh[i];
        acc += static_cast<double>(row[i]) * qh[i];
      }
      o[j] = static_cast<float>(acc) * qs;
    }
    // RMSNormGated: plain weight (not 1+w), SiLU output gate.
    cpu::RmsNorm(std::span<float>(o, d), nw, c.rms_eps);
    for (std::uint32_t j = 0; j < d; ++j) o[j] *= cpu::Silu(z[static_cast<std::size_t>(h) * d + j]);
  }
  MatVec(l.ssm_out, attn, out);
}

void Qwen35Reference::Attention(const Qwen35Layer& l, std::size_t index, std::span<const float> x, std::span<float> out) {
  const auto& c = w_->config;
  auto& s = attn_[index];
  const std::uint32_t hd = c.head_dim, nh = c.num_heads, nkv = c.num_kv_heads, group = nh / nkv;
  std::vector<float> qg(2ull * nh * hd), k(1ull * nkv * hd), v(1ull * nkv * hd);
  MatVec(l.attn_q, x, qg);
  MatVec(l.attn_k, x, k);
  MatVec(l.attn_v, x, v);
  std::vector<float> q(1ull * nh * hd), gate(1ull * nh * hd);
  for (std::uint32_t h = 0; h < nh; ++h) {
    std::copy_n(qg.begin() + static_cast<std::ptrdiff_t>(2ull * h * hd), hd, q.begin() + static_cast<std::ptrdiff_t>(1ull * h * hd));
    std::copy_n(qg.begin() + static_cast<std::ptrdiff_t>(2ull * h * hd + hd), hd, gate.begin() + static_cast<std::ptrdiff_t>(1ull * h * hd));
    RmsNormOnePlus(std::span<float>(q.data() + 1ull * h * hd, hd), l.attn_q_norm, c.rms_eps);
  }
  for (std::uint32_t h = 0; h < nkv; ++h) RmsNormOnePlus(std::span<float>(k.data() + 1ull * h * hd, hd), l.attn_k_norm, c.rms_eps);
  cpu::Rope(q.data(), nh, hd, c.rotary_dim, position_, c.rope_theta);
  cpu::Rope(k.data(), nkv, hd, c.rotary_dim, position_, c.rope_theta);
  s.k.insert(s.k.end(), k.begin(), k.end());
  s.v.insert(s.v.end(), v.begin(), v.end());
  const std::uint32_t n = position_ + 1;
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  std::vector<float> ctx(1ull * nh * hd, 0.0F), sc(n);
  for (std::uint32_t h = 0; h < nh; ++h) {
    const std::uint32_t kvh = h / group;
    const float* qh = q.data() + 1ull * h * hd;
    float mx = -INFINITY;
    for (std::uint32_t j = 0; j < n; ++j) {
      const float* kj = s.k.data() + (1ull * j * nkv + kvh) * hd;
      double dot = 0;
      for (std::uint32_t i = 0; i < hd; ++i) dot += static_cast<double>(qh[i]) * kj[i];
      sc[j] = static_cast<float>(dot) * scale;
      mx = std::max(mx, sc[j]);
    }
    double den = 0;
    for (auto& t : sc) {
      t = std::exp(t - mx);
      den += t;
    }
    float* oh = ctx.data() + 1ull * h * hd;
    for (std::uint32_t j = 0; j < n; ++j) {
      const float p = static_cast<float>(sc[j] / den);
      const float* vj = s.v.data() + (1ull * j * nkv + kvh) * hd;
      for (std::uint32_t i = 0; i < hd; ++i) oh[i] += p * vj[i];
    }
    for (std::uint32_t i = 0; i < hd; ++i) oh[i] *= cpu::Sigmoid(gate[1ull * h * hd + i]);
  }
  MatVec(l.attn_out, ctx, out);
}

bool Qwen35Reference::Step(std::int32_t token, std::span<float> hidden, std::span<float> logits, std::string* error) {
  return Run(token, w_->config.num_layers, {}, hidden, logits, error);
}

bool Qwen35Reference::StepToLayer(std::int32_t token, std::uint32_t layer, std::span<float> residual,
                                  std::string* error) {
  if (layer > w_->config.num_layers || residual.size() != w_->config.hidden_size) {
    if (error) *error = "StepToLayer: bad layer or residual size";
    return false;
  }
  return Run(token, layer, residual, {}, {}, error);
}

bool Qwen35Reference::Run(std::int32_t token, std::uint32_t stop_layer, std::span<float> residual,
                          std::span<float> hidden, std::span<float> logits, std::string* error) {
  const auto& c = w_->config;
  if (position_ >= max_context_ || token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
    if (error) *error = position_ >= max_context_ ? "context is full" : "token out of range";
    return false;
  }
  const std::uint32_t H = c.hidden_size;
  std::vector<float> res(H), xn(H), out(H), g(c.ffn), up(c.ffn);
  cpu::DequantizeRow(w_->token_embd, 0, static_cast<std::uint64_t>(token), res.data());
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (l == stop_layer) {
      std::copy(res.begin(), res.end(), residual.begin());
      ++position_;
      return true;
    }
    const auto& L = w_->layers[l];
    xn = res;
    RmsNormOnePlus(xn, L.attn_norm, c.rms_eps);
    if (L.linear) Linear(L, l, xn, out);
    else Attention(L, l, xn, out);
    for (std::uint32_t i = 0; i < H; ++i) res[i] += out[i];
    xn = res;
    RmsNormOnePlus(xn, L.post_attention_norm, c.rms_eps);
    MatVec(L.ffn_gate, xn, g);
    MatVec(L.ffn_up, xn, up);
    for (std::uint32_t i = 0; i < c.ffn; ++i) g[i] = cpu::Silu(g[i]) * up[i];
    MatVec(L.ffn_down, g, out);
    for (std::uint32_t i = 0; i < H; ++i) res[i] += out[i];
  }
  if (stop_layer == c.num_layers && !residual.empty()) std::copy(res.begin(), res.end(), residual.begin());
  RmsNormOnePlus(res, w_->output_norm, c.rms_eps);
  if (!hidden.empty()) std::copy(res.begin(), res.end(), hidden.begin());
  if (!logits.empty()) MatVec(w_->output, res, logits);
  ++position_;
  return true;
}

}  // namespace halo::model
