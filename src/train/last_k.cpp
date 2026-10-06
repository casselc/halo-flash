#include "train/last_k.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "model/cpu_ops.hpp"
#include "train/gemm.hpp"

namespace halo::train {

namespace {

using Buf = std::vector<float>;

float Sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }
float Silu(float x) { return x * Sigmoid(x); }
float SiluGrad(float x) {
  const float s = Sigmoid(x);
  return s * (1.0F + x * (1.0F - s));
}
float Softplus(float x) { return x > 20.0F ? x : std::log1p(std::exp(x)); }
float SoftplusGrad(float x) { return x > 20.0F ? 1.0F : Sigmoid(x); }

// splitmix64 -> uniform in [-bound, bound); deterministic on every platform.
struct Rng {
  std::uint64_t s;
  std::uint64_t Next() {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  float Uniform(float bound) {
    const double u = static_cast<double>(Next() >> 11) * 0x1.0p-53;  // [0, 1)
    return static_cast<float>((2.0 * u - 1.0) * bound);
  }
};

// RMSNorm of each row of x [rows][n] with weight w (y = x * r * w), storing r.
void RmsNormRows(ThreadPool& pool, const float* x, const float* w, float eps, std::size_t rows, std::size_t n, float* y,
                 float* rinv) {
  pool.ParallelFor(rows, [&](std::size_t t) {
    const float* xr = x + t * n;
    double ss = 0;
    for (std::size_t i = 0; i < n; ++i) ss += static_cast<double>(xr[i]) * xr[i];
    const float r = 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(n)) + eps);
    rinv[t] = r;
    float* yr = y + t * n;
    for (std::size_t i = 0; i < n; ++i) yr[i] = xr[i] * r * w[i];
  });
}

// dx (+)= d/dx of y = x * r(x) * w, for one row.
void RmsNormBackRow(const float* x, const float* w, float r, const float* dy, std::size_t n, float* dx, bool accumulate) {
  double dot = 0;
  for (std::size_t i = 0; i < n; ++i) dot += static_cast<double>(dy[i]) * w[i] * x[i];
  const float c = static_cast<float>(dot) * r * r * r / static_cast<float>(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float g = r * w[i] * dy[i] - c * x[i];
    dx[i] = accumulate ? dx[i] + g : g;
  }
}

void RmsNormBackRows(ThreadPool& pool, const float* x, const float* w, const float* rinv, const float* dy,
                     std::size_t rows, std::size_t n, float* dx, bool accumulate) {
  pool.ParallelFor(rows, [&](std::size_t t) {
    RmsNormBackRow(x + t * n, w, rinv[t], dy + t * n, n, dx + t * n, accumulate);
  });
}

// The rotation cpu::Rope applies at `pos` (same float math), forward or inverse.
void RopeHead(float* v, std::uint32_t rotary_dim, std::uint32_t pos, float theta, bool inverse) {
  const std::uint32_t half = rotary_dim / 2;
  for (std::uint32_t i = 0; i < half; ++i) {
    const float freq = std::pow(theta, -2.0F * static_cast<float>(i) / static_cast<float>(rotary_dim));
    const float angle = static_cast<float>(pos) * freq;
    const float c = std::cos(angle), s = inverse ? -std::sin(angle) : std::sin(angle);
    const float a = v[i], b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = a * s + b * c;
  }
}

}  // namespace

DenseWeight DenseWeight::From(const model::TensorRef& t) {
  DenseWeight d;
  d.rows = t.rows;
  d.cols = t.cols;
  if (t.type == core::GgmlType::kF32) {
    d.data = static_cast<const float*>(t.data);
    return d;
  }
  d.owned.resize(d.rows * d.cols);
  for (std::size_t r = 0; r < d.rows; ++r) model::cpu::DequantizeRow(t, 0, r, d.owned.data() + r * d.cols);
  d.data = d.owned.data();
  return d;
}

// ---------------------------------------------------------------------------
// Creation and parameters

std::unique_ptr<LastK> LastK::Create(const model::Qwen35Weights& w, const TailConfig& cfg, std::string* error) {
  const auto fail = [&](std::string m) -> std::unique_ptr<LastK> {
    if (error) *error = std::move(m);
    return nullptr;
  };
  const auto& c = w.config;
  if (cfg.last_k == 0 && cfg.lora_rank > 0) return fail("LoRA needs last_k >= 1");
  if (cfg.last_k > c.num_layers) return fail("last_k exceeds the model's layer count");
  if (cfg.head == HeadKind::kClassifier && cfg.num_classes < 2) return fail("a classifier head needs >= 2 classes");
  if (cfg.lora_rank > 0 && !(cfg.lora_alpha > 0)) return fail("lora_alpha must be positive");
  // Fixed-size row scratch in the backward kernels.
  if (c.ssm_head_dim > 1024 || 2 * c.KeyDim() > 8192 || c.head_dim > 1024)
    return fail("head dimensions exceed the trainer's limits");
  static const std::vector<std::string> kKnown = {"attn_q",   "attn_k",    "attn_v",    "attn_output",
                                                  "attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta",
                                                  "ssm_out",  "ffn_gate",  "ffn_up",    "ffn_down"};
  for (const auto& t : cfg.lora_targets)
    if (std::find(kKnown.begin(), kKnown.end(), t) == kKnown.end()) return fail("unknown LoRA target " + t);

  auto m = std::unique_ptr<LastK>(new LastK());
  m->w_ = &w;
  m->cfg_ = cfg;
  m->first_layer_ = c.num_layers - cfg.last_k;
  std::size_t offset = 0;
  const auto add_slice = [&](std::string name, std::size_t rows, std::size_t cols, bool head, bool decay) {
    m->slices_.push_back({std::move(name), offset, rows, cols, head, decay});
    offset += rows * cols;
    return m->slices_.back().offset;
  };
  const auto make_proj = [&](const model::TensorRef& t, std::string_view target) {
    Proj p;
    p.w = DenseWeight::From(t);
    p.name = std::string(t.name);
    if (cfg.lora_rank > 0 && std::find(cfg.lora_targets.begin(), cfg.lora_targets.end(), target) != cfg.lora_targets.end()) {
      Lora l;
      l.rank = cfg.lora_rank;
      l.scale = cfg.lora_alpha / static_cast<float>(cfg.lora_rank);
      l.a = add_slice(p.name + ".lora_a", cfg.lora_rank, t.cols, false, true);
      l.b = add_slice(p.name + ".lora_b", t.rows, cfg.lora_rank, false, true);
      p.lora = static_cast<int>(m->loras_.size());
      m->loras_.push_back(l);
    }
    return p;
  };
  for (std::uint32_t l = m->first_layer_; l < c.num_layers; ++l) {
    const auto& L = w.layers[l];
    Layer y;
    y.linear = L.linear;
    y.attn_norm = static_cast<const float*>(L.attn_norm.data);
    y.post_norm = static_cast<const float*>(L.post_attention_norm.data);
    if (L.linear) {
      y.qkv = make_proj(L.ssm_qkv, "attn_qkv");
      y.gate = make_proj(L.ssm_gate, "attn_gate");
      y.alpha = make_proj(L.ssm_alpha, "ssm_alpha");
      y.beta = make_proj(L.ssm_beta, "ssm_beta");
      y.out = make_proj(L.ssm_out, "ssm_out");
      y.conv = static_cast<const float*>(L.ssm_conv1d.data);
      y.dt = static_cast<const float*>(L.ssm_dt.data);
      y.a = static_cast<const float*>(L.ssm_a.data);
      y.ssm_norm = static_cast<const float*>(L.ssm_norm.data);
    } else {
      y.q = make_proj(L.attn_q, "attn_q");
      y.k = make_proj(L.attn_k, "attn_k");
      y.v = make_proj(L.attn_v, "attn_v");
      y.o = make_proj(L.attn_out, "attn_output");
      y.q_norm = static_cast<const float*>(L.attn_q_norm.data);
      y.k_norm = static_cast<const float*>(L.attn_k_norm.data);
    }
    y.ffn_gate = make_proj(L.ffn_gate, "ffn_gate");
    y.ffn_up = make_proj(L.ffn_up, "ffn_up");
    y.ffn_down = make_proj(L.ffn_down, "ffn_down");
    m->layers_.push_back(std::move(y));
  }
  if (cfg.head == HeadKind::kClassifier) {
    const std::size_t H = c.hidden_size, C = cfg.num_classes;
    m->head_norm_w_ = add_slice("head.norm.weight", 1, H, true, false);
    m->head_norm_b_ = add_slice("head.norm.bias", 1, H, true, false);
    m->head_w_ = add_slice("head.out.weight", C, H, true, true);
    m->head_b_ = add_slice("head.out.bias", 1, C, true, false);
  }
  m->params_.assign(offset, 0.0F);
  m->output_norm_ = static_cast<const float*>(w.output_norm.data);
  m->output_ = w.output;
  // The LM head as FP32 rows once (2 GB on Qwen3.5-2B): dequantizing it per
  // example costs more than the whole tail. Heads beyond 4 GB (none of the
  // dense models) are dequantized per chunk instead.
  if (cfg.head == HeadKind::kLm &&
      (w.output.type == core::GgmlType::kF32 || w.output.rows * w.output.cols * sizeof(float) <= (4ull << 30)))
    m->output_dense_ = DenseWeight::From(w.output);
  m->InitParams(0);
  return m;
}

const ParamSlice* LastK::FindSlice(std::string_view name) const {
  for (const auto& s : slices_)
    if (s.name == name) return &s;
  return nullptr;
}

void LastK::InitParams(std::uint64_t seed) {
  Rng rng{seed * 0x2545F4914F6CDD1DULL + 1};
  for (const auto& s : slices_) {
    float* p = params_.data() + s.offset;
    if (s.name.ends_with(".lora_a")) {
      const float bound = 1.0F / std::sqrt(static_cast<float>(s.cols));
      for (std::size_t i = 0; i < s.size(); ++i) p[i] = rng.Uniform(bound);
    } else if (s.name.ends_with(".lora_b")) {
      std::fill(p, p + s.size(), 0.0F);
    } else if (s.name == "head.norm.weight") {
      std::fill(p, p + s.size(), 1.0F);
    } else if (s.name == "head.norm.bias") {
      std::fill(p, p + s.size(), 0.0F);
    } else {
      const float bound = 1.0F / std::sqrt(static_cast<float>(w_->config.hidden_size));
      for (std::size_t i = 0; i < s.size(); ++i) p[i] = rng.Uniform(bound);
    }
  }
}

bool LastK::LoadLora(const model::LoraAdapter& adapter, std::string* error) {
  for (const auto& s : slices_) {
    const bool is_a = s.name.ends_with(".lora_a");
    if (!is_a && !s.name.ends_with(".lora_b")) continue;
    const std::string base = s.name.substr(0, s.name.size() - 7);
    const auto* pair = adapter.Find(base);
    if (!pair) {
      if (error) *error = "adapter has no pair for " + base;
      return false;
    }
    if (pair->rank != cfg_.lora_rank || std::abs(adapter.alpha - cfg_.lora_alpha) > 1e-6F) {
      if (error) *error = "adapter rank/alpha differ from the config for " + base;
      return false;
    }
    const auto& t = is_a ? pair->a : pair->b;
    if (t.rows * t.cols != s.size()) {
      if (error) *error = "adapter shape differs for " + s.name;
      return false;
    }
    std::copy_n(static_cast<const float*>(t.data), s.size(), params_.data() + s.offset);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Projections with LoRA

namespace {

struct Ctx {
  ThreadPool& pool;
  const std::vector<float>& params;
  const std::vector<LastK::Lora>& loras;
  float* grad;  // null in forward-only runs
};

// y[T][out] = x W^T + s (x A^T) B^T; ax[T][r] kept for the backward.
void ProjFwd(Ctx& c, const LastK::Proj& p, const float* x, std::size_t T, float* y, Buf* ax) {
  const std::size_t in = p.w.cols, out = p.w.rows;
  Gemm(c.pool, Op::kN, Op::kT, T, out, in, 1.0F, x, in, p.w.data, in, 0.0F, y, out);
  if (p.lora < 0) return;
  const auto& l = c.loras[static_cast<std::size_t>(p.lora)];
  ax->assign(T * l.rank, 0.0F);
  Gemm(c.pool, Op::kN, Op::kT, T, l.rank, in, 1.0F, x, in, c.params.data() + l.a, in, 0.0F, ax->data(), l.rank);
  Gemm(c.pool, Op::kN, Op::kT, T, out, l.rank, l.scale, ax->data(), l.rank, c.params.data() + l.b, l.rank, 1.0F, y,
       out);
}

// dx (= or +=) dy W + s (dy B) A; grads of A and B accumulate into c.grad.
void ProjBwd(Ctx& c, const LastK::Proj& p, const float* x, const Buf& ax, const float* dy, std::size_t T, float* dx,
             bool accumulate) {
  const std::size_t in = p.w.cols, out = p.w.rows;
  Gemm(c.pool, Op::kN, Op::kN, T, in, out, 1.0F, dy, out, p.w.data, in, accumulate ? 1.0F : 0.0F, dx, in);
  if (p.lora < 0) return;
  const auto& l = c.loras[static_cast<std::size_t>(p.lora)];
  Buf dyb(T * l.rank);
  Gemm(c.pool, Op::kN, Op::kN, T, l.rank, out, 1.0F, dy, out, c.params.data() + l.b, l.rank, 0.0F, dyb.data(), l.rank);
  Gemm(c.pool, Op::kN, Op::kN, T, in, l.rank, l.scale, dyb.data(), l.rank, c.params.data() + l.a, in, 1.0F, dx, in);
  Gemm(c.pool, Op::kT, Op::kN, out, l.rank, T, l.scale, dy, out, ax.data(), l.rank, 1.0F, c.grad + l.b, l.rank);
  Gemm(c.pool, Op::kT, Op::kN, l.rank, in, T, l.scale, dyb.data(), l.rank, x, in, 1.0F, c.grad + l.a, in);
}

}  // namespace

// ---------------------------------------------------------------------------
// One layer: forward with a tape, backward from it

struct LastK::Tape {
  // Common
  Buf xn, rinv1, mid, xn2, rinv2, g, up, act, mlp_out;
  Buf ax_gate, ax_up, ax_down;
  // DeltaNet
  Buf qkv, z, al, be, pre, conv, qinv, kinv, decay, bsig, states, u, o, orinv, y, mix;
  Buf ax_qkv, ax_z, ax_al, ax_be, ax_out;
  // Attention
  Buf qg, kraw, v, qn_r, kn_r, qr, kr, probs, ctx, og;
  Buf ax_q, ax_k, ax_v, ax_o;
};

namespace {

struct Dims {
  std::size_t T, H, F;
  // DeltaNet
  std::size_t C, kd, vd, d, nk, nv, kern;
  // Attention
  std::size_t nh, nkv, hd, rot;
  float eps, theta;
};

Dims MakeDims(const model::Qwen35Config& c, std::size_t T) {
  return {T,
          c.hidden_size,
          c.ffn,
          c.ConvChannels(),
          c.KeyDim(),
          c.ValueDim(),
          c.ssm_head_dim,
          c.ssm_num_k_heads,
          c.ssm_num_v_heads,
          c.ssm_conv_kernel,
          c.num_heads,
          c.num_kv_heads,
          c.head_dim,
          c.rotary_dim,
          c.rms_eps,
          c.rope_theta};
}

// DeltaNet mixer forward: xn [T][H] -> tape.mix [T][H].
void DeltaNetFwd(Ctx& c, const LastK::Layer& L, const Dims& D, LastK::Tape& tp, bool keep_states) {
  const std::size_t T = D.T, C = D.C, d = D.d;
  tp.qkv.resize(T * C);
  tp.z.resize(T * D.vd);
  tp.al.resize(T * D.nv);
  tp.be.resize(T * D.nv);
  ProjFwd(c, L.qkv, tp.xn.data(), T, tp.qkv.data(), &tp.ax_qkv);
  ProjFwd(c, L.gate, tp.xn.data(), T, tp.z.data(), &tp.ax_z);
  ProjFwd(c, L.alpha, tp.xn.data(), T, tp.al.data(), &tp.ax_al);
  ProjFwd(c, L.beta, tp.xn.data(), T, tp.be.data(), &tp.ax_be);
  // Causal depthwise conv over the sequence (zero history) and SiLU.
  tp.pre.resize(T * C);
  tp.conv.resize(T * C);
  c.pool.ParallelFor(T, [&](std::size_t t) {
    for (std::size_t ch = 0; ch < C; ++ch) {
      float acc = L.conv[ch * D.kern + D.kern - 1] * tp.qkv[t * C + ch];
      for (std::size_t j = 0; j + 1 < D.kern; ++j) {
        const std::ptrdiff_t src = static_cast<std::ptrdiff_t>(t) - static_cast<std::ptrdiff_t>(D.kern - 1) +
                                   static_cast<std::ptrdiff_t>(j);
        if (src >= 0) acc += L.conv[ch * D.kern + j] * tp.qkv[static_cast<std::size_t>(src) * C + ch];
      }
      tp.pre[t * C + ch] = acc;
      tp.conv[t * C + ch] = Silu(acc);
    }
    // L2-normalize q and k per key head, in place; keep the inverse norms.
    for (std::size_t h = 0; h < 2 * D.nk; ++h) {
      float* x = tp.conv.data() + t * C + h * d;
      double ss = 0;
      for (std::size_t i = 0; i < d; ++i) ss += static_cast<double>(x[i]) * x[i];
      const float s = 1.0F / std::sqrt(static_cast<float>(ss) + 1e-6F);
      for (std::size_t i = 0; i < d; ++i) x[i] *= s;
      if (h < D.nk) tp.qinv[t * D.nk + h] = s;
      else tp.kinv[t * D.nk + (h - D.nk)] = s;
    }
  });
  tp.decay.resize(T * D.nv);
  tp.bsig.resize(T * D.nv);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t h = 0; h < D.nv; ++h) {
      tp.decay[t * D.nv + h] = std::exp(L.a[h] * Softplus(tp.al[t * D.nv + h] + L.dt[h]));
      tp.bsig[t * D.nv + h] = Sigmoid(tp.be[t * D.nv + h]);
    }
  tp.o.assign(T * D.vd, 0.0F);
  tp.u.assign(T * D.vd, 0.0F);
  if (keep_states) tp.states.resize(T * D.nv * d * d);
  const float qs = 1.0F / std::sqrt(static_cast<float>(d));
  c.pool.ParallelFor(D.nv, [&](std::size_t h) {
    Buf S(d * d, 0.0F);
    const std::size_t kh = h % D.nk;
    for (std::size_t t = 0; t < T; ++t) {
      const float* q = tp.conv.data() + t * C + kh * d;
      const float* k = tp.conv.data() + t * C + D.kd + kh * d;
      const float* v = tp.conv.data() + t * C + 2 * D.kd + h * d;
      const float g = tp.decay[t * D.nv + h], b = tp.bsig[t * D.nv + h];
      float* u = tp.u.data() + t * D.vd + h * d;
      for (std::size_t j = 0; j < d; ++j) {
        float* row = S.data() + j * d;
        double acc = 0;
        for (std::size_t i = 0; i < d; ++i) {
          row[i] *= g;
          acc += static_cast<double>(row[i]) * k[i];
        }
        u[j] = static_cast<float>(acc);
      }
      float* o = tp.o.data() + t * D.vd + h * d;
      for (std::size_t j = 0; j < d; ++j) {
        const float delta = (v[j] - u[j]) * b;
        float* row = S.data() + j * d;
        double acc = 0;
        for (std::size_t i = 0; i < d; ++i) {
          row[i] += delta * k[i];
          acc += static_cast<double>(row[i]) * q[i];
        }
        o[j] = static_cast<float>(acc) * qs;
      }
      if (keep_states) std::copy(S.begin(), S.end(), tp.states.begin() + static_cast<std::ptrdiff_t>((h * T + t) * d * d));
    }
  });
  // Gated RMSNorm per value head, SiLU(z) gate.
  tp.orinv.resize(T * D.nv);
  tp.y.resize(T * D.vd);
  c.pool.ParallelFor(T, [&](std::size_t t) {
    for (std::size_t h = 0; h < D.nv; ++h) {
      const float* o = tp.o.data() + t * D.vd + h * d;
      double ss = 0;
      for (std::size_t j = 0; j < d; ++j) ss += static_cast<double>(o[j]) * o[j];
      const float r = 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(d)) + D.eps);
      tp.orinv[t * D.nv + h] = r;
      for (std::size_t j = 0; j < d; ++j)
        tp.y[t * D.vd + h * d + j] = o[j] * r * L.ssm_norm[j] * Silu(tp.z[t * D.vd + h * d + j]);
    }
  });
  tp.mix.resize(T * D.H);
  ProjFwd(c, L.out, tp.y.data(), T, tp.mix.data(), &tp.ax_out);
}

// d_mix [T][H] -> d_xn [T][H] (overwritten).
void DeltaNetBwd(Ctx& c, const LastK::Layer& L, const Dims& D, LastK::Tape& tp, const float* d_mix, float* d_xn) {
  const std::size_t T = D.T, C = D.C, d = D.d;
  Buf dy(T * D.vd);
  ProjBwd(c, L.out, tp.y.data(), tp.ax_out, d_mix, T, dy.data(), false);
  Buf dz(T * D.vd), d_o(T * D.vd);
  c.pool.ParallelFor(T, [&](std::size_t t) {
    for (std::size_t h = 0; h < D.nv; ++h) {
      const std::size_t base = t * D.vd + h * d;
      const float r = tp.orinv[t * D.nv + h];
      float dn[1024];
      for (std::size_t j = 0; j < d; ++j) {
        const float zz = tp.z[base + j];
        const float normed = tp.o[base + j] * r * L.ssm_norm[j];
        dz[base + j] = dy[base + j] * normed * SiluGrad(zz);
        dn[j] = dy[base + j] * Silu(zz);
      }
      RmsNormBackRow(tp.o.data() + base, L.ssm_norm, r, dn, d, d_o.data() + base, false);
    }
  });
  // Recurrence backward, one value head per task; q/k grads per value head,
  // summed into their key head afterwards in a fixed order.
  Buf dqh(D.nv * T * d, 0.0F), dkh(D.nv * T * d, 0.0F), dv(T * D.vd, 0.0F), dg(T * D.nv), db(T * D.nv);
  const float qs = 1.0F / std::sqrt(static_cast<float>(d));
  c.pool.ParallelFor(D.nv, [&](std::size_t h) {
    const std::size_t kh = h % D.nk;
    Buf dS(d * d, 0.0F), dA(d * d), ddelta(d), du(d);
    for (std::size_t tt = T; tt-- > 0;) {
      const float* q = tp.conv.data() + tt * C + kh * d;
      const float* k = tp.conv.data() + tt * C + D.kd + kh * d;
      const float* v = tp.conv.data() + tt * C + 2 * D.kd + h * d;
      const float* St = tp.states.data() + (h * T + tt) * d * d;
      const float* Sp = tt > 0 ? tp.states.data() + (h * T + tt - 1) * d * d : nullptr;
      const float* u = tp.u.data() + tt * D.vd + h * d;
      const float* dO = d_o.data() + tt * D.vd + h * d;
      const float g = tp.decay[tt * D.nv + h], b = tp.bsig[tt * D.nv + h];
      float* dq = dqh.data() + (h * T + tt) * d;
      float* dk = dkh.data() + (h * T + tt) * d;
      // o = qs * S_t q
      for (std::size_t i = 0; i < d; ++i) {
        double acc = 0;
        for (std::size_t j = 0; j < d; ++j) acc += static_cast<double>(St[j * d + i]) * dO[j];
        dq[i] = static_cast<float>(acc) * qs;
      }
      for (std::size_t j = 0; j < d; ++j)
        for (std::size_t i = 0; i < d; ++i) dS[j * d + i] += qs * dO[j] * q[i];
      // S_t = A + delta k^T
      double dbeta = 0;
      for (std::size_t j = 0; j < d; ++j) {
        double acc = 0;
        for (std::size_t i = 0; i < d; ++i) acc += static_cast<double>(dS[j * d + i]) * k[i];
        ddelta[j] = static_cast<float>(acc);
      }
      for (std::size_t i = 0; i < d; ++i) {
        double acc = 0;
        for (std::size_t j = 0; j < d; ++j) acc += static_cast<double>(dS[j * d + i]) * ((v[j] - u[j]) * b);
        dk[i] = static_cast<float>(acc);
      }
      // delta = b (v - u)
      for (std::size_t j = 0; j < d; ++j) {
        dv[tt * D.vd + h * d + j] = b * ddelta[j];
        du[j] = -b * ddelta[j];
        dbeta += static_cast<double>(v[j] - u[j]) * ddelta[j];
      }
      // u = A k, A = g S_{t-1}
      double dgs = 0;
      for (std::size_t j = 0; j < d; ++j)
        for (std::size_t i = 0; i < d; ++i) dA[j * d + i] = dS[j * d + i] + du[j] * k[i];
      if (Sp) {
        for (std::size_t i = 0; i < d; ++i) {
          double acc = 0;
          for (std::size_t j = 0; j < d; ++j) acc += static_cast<double>(Sp[j * d + i]) * du[j];
          dk[i] += g * static_cast<float>(acc);
        }
        for (std::size_t x = 0; x < d * d; ++x) dgs += static_cast<double>(dA[x]) * Sp[x];
      }
      dg[tt * D.nv + h] = static_cast<float>(dgs);
      db[tt * D.nv + h] = static_cast<float>(dbeta);
      for (std::size_t x = 0; x < d * d; ++x) dS[x] = g * dA[x];
    }
  });
  // Gate gradients.
  Buf dal(T * D.nv), dbe(T * D.nv);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t h = 0; h < D.nv; ++h) {
      const std::size_t i = t * D.nv + h;
      const float x = tp.al[i] + L.dt[h];
      dal[i] = dg[i] * tp.decay[i] * L.a[h] * SoftplusGrad(x);
      dbe[i] = db[i] * tp.bsig[i] * (1.0F - tp.bsig[i]);
    }
  // conv output grads: [dq, dk] through the L2 norm, dv directly; then SiLU.
  Buf dpre(T * C);
  c.pool.ParallelFor(T, [&](std::size_t t) {
    float dact[8192];
    float* dqk = dact;
    for (std::size_t kh = 0; kh < D.nk; ++kh) {
      for (int which = 0; which < 2; ++which) {
        float* out = dqk + (which * D.nk + kh) * d;
        std::fill(out, out + d, 0.0F);
        for (std::size_t h = kh; h < D.nv; h += D.nk) {
          const float* src = (which == 0 ? dqh : dkh).data() + (h * T + t) * d;
          for (std::size_t i = 0; i < d; ++i) out[i] += src[i];
        }
        // y = x s, s = 1/sqrt(sum x^2 + eps): dx = s dy - s^3 x (x . dy); y = x s so x = y / s.
        const float s = which == 0 ? tp.qinv[t * D.nk + kh] : tp.kinv[t * D.nk + kh];
        const float* yv = tp.conv.data() + t * C + which * D.kd + kh * d;
        double dot = 0;
        for (std::size_t i = 0; i < d; ++i) dot += static_cast<double>(yv[i]) * out[i];
        // x . dy = (y . dy) / s; s^3 x (x.dy) = s y (y.dy)
        for (std::size_t i = 0; i < d; ++i) out[i] = s * out[i] - s * yv[i] * static_cast<float>(dot);
      }
    }
    for (std::size_t ch = 0; ch < C; ++ch) {
      const float g = ch < 2 * D.kd ? dqk[ch] : dv[t * D.vd + (ch - 2 * D.kd)];
      dpre[t * C + ch] = g * SiluGrad(tp.pre[t * C + ch]);
    }
  });
  Buf dqkv(T * C, 0.0F);
  c.pool.ParallelFor(T, [&](std::size_t s) {
    // qkv[s] feeds pre[t] for t = s .. s + kern - 1 with weight index j = s - t + kern - 1.
    for (std::size_t ch = 0; ch < C; ++ch) {
      float acc = 0;
      for (std::size_t j = 0; j < D.kern; ++j) {
        const std::size_t t = s + (D.kern - 1) - j;
        if (t < T) acc += L.conv[ch * D.kern + j] * dpre[t * C + ch];
      }
      dqkv[s * C + ch] = acc;
    }
  });
  ProjBwd(c, L.qkv, tp.xn.data(), tp.ax_qkv, dqkv.data(), T, d_xn, false);
  ProjBwd(c, L.gate, tp.xn.data(), tp.ax_z, dz.data(), T, d_xn, true);
  ProjBwd(c, L.alpha, tp.xn.data(), tp.ax_al, dal.data(), T, d_xn, true);
  ProjBwd(c, L.beta, tp.xn.data(), tp.ax_be, dbe.data(), T, d_xn, true);
}

void AttentionFwd(Ctx& c, const LastK::Layer& L, const Dims& D, LastK::Tape& tp) {
  const std::size_t T = D.T, nh = D.nh, nkv = D.nkv, hd = D.hd, group = nh / nkv;
  tp.qg.resize(T * 2 * nh * hd);
  tp.kraw.resize(T * nkv * hd);
  tp.v.resize(T * nkv * hd);
  ProjFwd(c, L.q, tp.xn.data(), T, tp.qg.data(), &tp.ax_q);
  ProjFwd(c, L.k, tp.xn.data(), T, tp.kraw.data(), &tp.ax_k);
  ProjFwd(c, L.v, tp.xn.data(), T, tp.v.data(), &tp.ax_v);
  tp.qn_r.resize(T * nh);
  tp.kn_r.resize(T * nkv);
  tp.qr.resize(T * nh * hd);
  tp.kr.resize(T * nkv * hd);
  c.pool.ParallelFor(T, [&](std::size_t t) {
    for (std::size_t h = 0; h < nh; ++h) {
      const float* x = tp.qg.data() + t * 2 * nh * hd + 2 * h * hd;
      float* y = tp.qr.data() + (t * nh + h) * hd;
      double ss = 0;
      for (std::size_t i = 0; i < hd; ++i) ss += static_cast<double>(x[i]) * x[i];
      const float r = 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(hd)) + D.eps);
      tp.qn_r[t * nh + h] = r;
      for (std::size_t i = 0; i < hd; ++i) y[i] = x[i] * r * L.q_norm[i];
      RopeHead(y, static_cast<std::uint32_t>(D.rot), static_cast<std::uint32_t>(t), D.theta, false);
    }
    for (std::size_t h = 0; h < nkv; ++h) {
      const float* x = tp.kraw.data() + (t * nkv + h) * hd;
      float* y = tp.kr.data() + (t * nkv + h) * hd;
      double ss = 0;
      for (std::size_t i = 0; i < hd; ++i) ss += static_cast<double>(x[i]) * x[i];
      const float r = 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(hd)) + D.eps);
      tp.kn_r[t * nkv + h] = r;
      for (std::size_t i = 0; i < hd; ++i) y[i] = x[i] * r * L.k_norm[i];
      RopeHead(y, static_cast<std::uint32_t>(D.rot), static_cast<std::uint32_t>(t), D.theta, false);
    }
  });
  tp.probs.assign(nh * T * T, 0.0F);
  tp.ctx.assign(T * nh * hd, 0.0F);
  tp.og.resize(T * nh * hd);
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  c.pool.ParallelFor(nh * T, [&](std::size_t task) {
    const std::size_t h = task / T, t = task % T, kvh = h / group;
    const float* q = tp.qr.data() + (t * nh + h) * hd;
    float* p = tp.probs.data() + (h * T + t) * T;
    float mx = -std::numeric_limits<float>::infinity();
    for (std::size_t j = 0; j <= t; ++j) {
      const float* k = tp.kr.data() + (j * nkv + kvh) * hd;
      double dot = 0;
      for (std::size_t i = 0; i < hd; ++i) dot += static_cast<double>(q[i]) * k[i];
      p[j] = static_cast<float>(dot) * scale;
      mx = std::max(mx, p[j]);
    }
    double den = 0;
    for (std::size_t j = 0; j <= t; ++j) {
      p[j] = std::exp(p[j] - mx);
      den += p[j];
    }
    float* o = tp.ctx.data() + (t * nh + h) * hd;
    for (std::size_t j = 0; j <= t; ++j) {
      p[j] = static_cast<float>(p[j] / den);
      const float* v = tp.v.data() + (j * nkv + kvh) * hd;
      for (std::size_t i = 0; i < hd; ++i) o[i] += p[j] * v[i];
    }
    const float* gate = tp.qg.data() + t * 2 * nh * hd + 2 * h * hd + hd;
    float* og = tp.og.data() + (t * nh + h) * hd;
    for (std::size_t i = 0; i < hd; ++i) og[i] = o[i] * Sigmoid(gate[i]);
  });
  tp.mix.resize(T * D.H);
  ProjFwd(c, L.o, tp.og.data(), T, tp.mix.data(), &tp.ax_o);
}

void AttentionBwd(Ctx& c, const LastK::Layer& L, const Dims& D, LastK::Tape& tp, const float* d_mix, float* d_xn) {
  const std::size_t T = D.T, nh = D.nh, nkv = D.nkv, hd = D.hd, group = nh / nkv;
  Buf dog(T * nh * hd);
  ProjBwd(c, L.o, tp.og.data(), tp.ax_o, d_mix, T, dog.data(), false);
  Buf dqg(T * 2 * nh * hd), dctx(T * nh * hd);
  c.pool.ParallelFor(T * nh, [&](std::size_t task) {
    const std::size_t t = task / nh, h = task % nh;
    const float* gate = tp.qg.data() + t * 2 * nh * hd + 2 * h * hd + hd;
    const float* o = tp.ctx.data() + (t * nh + h) * hd;
    const float* dg = dog.data() + (t * nh + h) * hd;
    float* dgate = dqg.data() + t * 2 * nh * hd + 2 * h * hd + hd;
    float* dc = dctx.data() + (t * nh + h) * hd;
    for (std::size_t i = 0; i < hd; ++i) {
      const float s = Sigmoid(gate[i]);
      dc[i] = dg[i] * s;
      dgate[i] = dg[i] * o[i] * s * (1.0F - s);
    }
  });
  // Per query head: dq (rotated), and its contributions to dk / dv of its kv head.
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  Buf dqr(T * nh * hd, 0.0F), dkr_h(nh * T * hd, 0.0F), dv_h(nh * T * hd, 0.0F);
  c.pool.ParallelFor(nh, [&](std::size_t h) {
    const std::size_t kvh = h / group;
    Buf ds(T);
    for (std::size_t t = 0; t < T; ++t) {
      const float* p = tp.probs.data() + (h * T + t) * T;
      const float* dc = dctx.data() + (t * nh + h) * hd;
      double sum = 0;
      for (std::size_t j = 0; j <= t; ++j) {
        const float* v = tp.v.data() + (j * nkv + kvh) * hd;
        double dp = 0;
        for (std::size_t i = 0; i < hd; ++i) dp += static_cast<double>(dc[i]) * v[i];
        ds[j] = static_cast<float>(dp);
        sum += static_cast<double>(p[j]) * dp;
        float* dv = dv_h.data() + (h * T + j) * hd;
        for (std::size_t i = 0; i < hd; ++i) dv[i] += p[j] * dc[i];
      }
      const float* q = tp.qr.data() + (t * nh + h) * hd;
      float* dq = dqr.data() + (t * nh + h) * hd;
      for (std::size_t j = 0; j <= t; ++j) {
        const float dsj = p[j] * (ds[j] - static_cast<float>(sum)) * scale;
        const float* k = tp.kr.data() + (j * nkv + kvh) * hd;
        float* dk = dkr_h.data() + (h * T + j) * hd;
        for (std::size_t i = 0; i < hd; ++i) {
          dq[i] += dsj * k[i];
          dk[i] += dsj * q[i];
        }
      }
    }
  });
  Buf dk(T * nkv * hd), dv(T * nkv * hd);
  c.pool.ParallelFor(T, [&](std::size_t t) {
    for (std::size_t kvh = 0; kvh < nkv; ++kvh) {
      float* dkt = dk.data() + (t * nkv + kvh) * hd;
      float* dvt = dv.data() + (t * nkv + kvh) * hd;
      std::fill(dkt, dkt + hd, 0.0F);
      std::fill(dvt, dvt + hd, 0.0F);
      for (std::size_t h = kvh * group; h < (kvh + 1) * group; ++h)
        for (std::size_t i = 0; i < hd; ++i) {
          dkt[i] += dkr_h[(h * T + t) * hd + i];
          dvt[i] += dv_h[(h * T + t) * hd + i];
        }
      // Undo RoPE, then the RMSNorm.
      RopeHead(dkt, static_cast<std::uint32_t>(D.rot), static_cast<std::uint32_t>(t), D.theta, true);
      float tmp[1024];
      std::copy(dkt, dkt + hd, tmp);
      RmsNormBackRow(tp.kraw.data() + (t * nkv + kvh) * hd, L.k_norm, tp.kn_r[t * nkv + kvh], tmp, hd, dkt, false);
    }
    for (std::size_t h = 0; h < nh; ++h) {
      float tmp[1024];
      float* dq = dqr.data() + (t * nh + h) * hd;
      RopeHead(dq, static_cast<std::uint32_t>(D.rot), static_cast<std::uint32_t>(t), D.theta, true);
      std::copy(dq, dq + hd, tmp);
      RmsNormBackRow(tp.qg.data() + t * 2 * nh * hd + 2 * h * hd, L.q_norm, tp.qn_r[t * nh + h], tmp, hd,
                     dqg.data() + t * 2 * nh * hd + 2 * h * hd, false);
    }
  });
  ProjBwd(c, L.q, tp.xn.data(), tp.ax_q, dqg.data(), T, d_xn, false);
  ProjBwd(c, L.k, tp.xn.data(), tp.ax_k, dk.data(), T, d_xn, true);
  ProjBwd(c, L.v, tp.xn.data(), tp.ax_v, dv.data(), T, d_xn, true);
}

// Whole layer forward: res_in -> res_out, filling the tape.
void LayerFwd(Ctx& c, const LastK::Layer& L, const Dims& D, const float* res_in, float* res_out, LastK::Tape& tp,
              bool for_backward) {
  const std::size_t T = D.T, H = D.H, F = D.F;
  tp.xn.resize(T * H);
  tp.rinv1.resize(T);
  RmsNormRows(c.pool, res_in, L.attn_norm, D.eps, T, H, tp.xn.data(), tp.rinv1.data());
  if (L.linear) {
    tp.qinv.resize(T * D.nk);
    tp.kinv.resize(T * D.nk);
    DeltaNetFwd(c, L, D, tp, for_backward);
  } else {
    AttentionFwd(c, L, D, tp);
  }
  tp.mid.resize(T * H);
  for (std::size_t i = 0; i < T * H; ++i) tp.mid[i] = res_in[i] + tp.mix[i];
  tp.xn2.resize(T * H);
  tp.rinv2.resize(T);
  RmsNormRows(c.pool, tp.mid.data(), L.post_norm, D.eps, T, H, tp.xn2.data(), tp.rinv2.data());
  tp.g.resize(T * F);
  tp.up.resize(T * F);
  tp.act.resize(T * F);
  ProjFwd(c, L.ffn_gate, tp.xn2.data(), T, tp.g.data(), &tp.ax_gate);
  ProjFwd(c, L.ffn_up, tp.xn2.data(), T, tp.up.data(), &tp.ax_up);
  for (std::size_t i = 0; i < T * F; ++i) tp.act[i] = Silu(tp.g[i]) * tp.up[i];
  tp.mlp_out.resize(T * H);
  ProjFwd(c, L.ffn_down, tp.act.data(), T, tp.mlp_out.data(), &tp.ax_down);
  for (std::size_t i = 0; i < T * H; ++i) res_out[i] = tp.mid[i] + tp.mlp_out[i];
}

// d_res (in: d res_out, out: d res_in).
void LayerBwd(Ctx& c, const LastK::Layer& L, const Dims& D, const float* res_in, LastK::Tape& tp, Buf& d_res) {
  const std::size_t T = D.T, H = D.H, F = D.F;
  Buf dact(T * F);
  ProjBwd(c, L.ffn_down, tp.act.data(), tp.ax_down, d_res.data(), T, dact.data(), false);
  Buf dg(T * F), du(T * F);
  for (std::size_t i = 0; i < T * F; ++i) {
    dg[i] = dact[i] * tp.up[i] * SiluGrad(tp.g[i]);
    du[i] = dact[i] * Silu(tp.g[i]);
  }
  Buf dxn2(T * H);
  ProjBwd(c, L.ffn_gate, tp.xn2.data(), tp.ax_gate, dg.data(), T, dxn2.data(), false);
  ProjBwd(c, L.ffn_up, tp.xn2.data(), tp.ax_up, du.data(), T, dxn2.data(), true);
  // d_mid = d_res + norm backward
  RmsNormBackRows(c.pool, tp.mid.data(), L.post_norm, tp.rinv2.data(), dxn2.data(), T, H, d_res.data(), true);
  Buf dxn(T * H);
  if (L.linear) DeltaNetBwd(c, L, D, tp, d_res.data(), dxn.data());
  else AttentionBwd(c, L, D, tp, d_res.data(), dxn.data());
  RmsNormBackRows(c.pool, res_in, L.attn_norm, tp.rinv1.data(), dxn.data(), T, H, d_res.data(), true);
}

}  // namespace

// ---------------------------------------------------------------------------

LastK::Loss LastK::ForwardBackward(const Example& ex, float scale, std::span<float> grad, ThreadPool& pool,
                                   Probe* probe) const {
  return Run(ex, true, scale, grad, pool, probe);
}

LastK::Loss LastK::Forward(const Example& ex, ThreadPool& pool, Probe* probe) const {
  return Run(ex, false, 0.0F, {}, pool, probe);
}

LastK::Loss LastK::Run(const Example& ex, bool backward, float scale, std::span<float> grad, ThreadPool& pool,
                       Probe* probe) const {
  const auto& cfg = w_->config;
  const std::size_t T = ex.tokens, H = cfg.hidden_size, K = layers_.size();
  Loss loss;
  if (T == 0 || T > kMaxTokens || ex.boundary.size() != T * H) return loss;
  if (backward && grad.size() != params_.size()) return loss;
  const Dims D = MakeDims(cfg, T);
  Ctx c{pool, params_, loras_, backward ? grad.data() : nullptr};

  // Forward through the tail, keeping each layer's input residual.
  std::vector<Buf> res(K + 1);
  res[0].assign(ex.boundary.begin(), ex.boundary.end());
  {
    Tape tp;
    for (std::size_t l = 0; l < K; ++l) {
      res[l + 1].resize(T * H);
      LayerFwd(c, layers_[l], D, res[l].data(), res[l + 1].data(), tp, false);
    }
  }
  Buf hn(T * H), rfin(T);
  RmsNormRows(pool, res[K].data(), output_norm_, cfg.rms_eps, T, H, hn.data(), rfin.data());
  Buf dhn(backward ? T * H : 0, 0.0F);

  if (cfg_.head == HeadKind::kClassifier) {
    if (ex.label < 0 || static_cast<std::uint32_t>(ex.label) >= cfg_.num_classes || ex.label_position >= T) return loss;
    const std::size_t C = cfg_.num_classes;
    const float* x = hn.data() + ex.label_position * H;
    const float* nw = params_.data() + head_norm_w_;
    const float* nb = params_.data() + head_norm_b_;
    const float* W = params_.data() + head_w_;
    const float* bias = params_.data() + head_b_;
    double mean = 0;
    for (std::size_t i = 0; i < H; ++i) mean += x[i];
    mean /= static_cast<double>(H);
    double var = 0;
    for (std::size_t i = 0; i < H; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= static_cast<double>(H);
    const float inv = 1.0F / std::sqrt(static_cast<float>(var) + 1e-5F);
    Buf xhat(H), ln(H), logits(C);
    for (std::size_t i = 0; i < H; ++i) {
      xhat[i] = (x[i] - static_cast<float>(mean)) * inv;
      ln[i] = xhat[i] * nw[i] + nb[i];
    }
    for (std::size_t k = 0; k < C; ++k) {
      double acc = 0;
      for (std::size_t i = 0; i < H; ++i) acc += static_cast<double>(W[k * H + i]) * ln[i];
      logits[k] = static_cast<float>(acc) + bias[k];
    }
    if (probe) probe->logits = logits;
    const float mx = *std::max_element(logits.begin(), logits.end());
    double den = 0;
    for (float l : logits) den += std::exp(static_cast<double>(l - mx));
    loss.sum = std::log(den) + mx - logits[static_cast<std::size_t>(ex.label)];
    loss.count = 1;
    loss.correct = static_cast<std::uint32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin() == ex.label);
    if (backward) {
      Buf dl(C), dln(H, 0.0F);
      for (std::size_t k = 0; k < C; ++k)
        dl[k] = scale * (static_cast<float>(std::exp(static_cast<double>(logits[k] - mx)) / den) -
                         (static_cast<std::int32_t>(k) == ex.label ? 1.0F : 0.0F));
      float* gW = grad.data() + head_w_;
      float* gb = grad.data() + head_b_;
      for (std::size_t k = 0; k < C; ++k) {
        gb[k] += dl[k];
        for (std::size_t i = 0; i < H; ++i) {
          gW[k * H + i] += dl[k] * ln[i];
          dln[i] += dl[k] * W[k * H + i];
        }
      }
      float* gnw = grad.data() + head_norm_w_;
      float* gnb = grad.data() + head_norm_b_;
      Buf dxhat(H);
      double m1 = 0, m2 = 0;
      for (std::size_t i = 0; i < H; ++i) {
        gnw[i] += dln[i] * xhat[i];
        gnb[i] += dln[i];
        dxhat[i] = dln[i] * nw[i];
        m1 += dxhat[i];
        m2 += static_cast<double>(dxhat[i]) * xhat[i];
      }
      m1 /= static_cast<double>(H);
      m2 /= static_cast<double>(H);
      float* dx = dhn.data() + ex.label_position * H;
      for (std::size_t i = 0; i < H; ++i)
        dx[i] = inv * (dxhat[i] - static_cast<float>(m1) - xhat[i] * static_cast<float>(m2));
    }
  } else {
    const std::size_t n = ex.lm_targets.size(), V = output_.rows;
    if (n == 0) return loss;
    Buf hs(n * H), logits(n * V);
    for (std::size_t r = 0; r < n; ++r) {
      if (ex.lm_targets[r].first >= T || ex.lm_targets[r].second < 0 ||
          static_cast<std::size_t>(ex.lm_targets[r].second) >= V)
        return Loss{};
      std::copy_n(hn.data() + ex.lm_targets[r].first * H, H, hs.data() + r * H);
    }
    constexpr std::size_t kChunk = 8192;
    Buf chunk;
    const auto rows_of = [&](std::size_t r0, std::size_t cr) -> const float* {
      if (output_dense_.data) return output_dense_.data + r0 * H;
      chunk.resize(cr * H);
      pool.ParallelFor((cr + 63) / 64, [&](std::size_t b) {
        for (std::size_t r = b * 64; r < std::min(cr, (b + 1) * 64); ++r)
          model::cpu::DequantizeRow(output_, 0, r0 + r, chunk.data() + r * H);
      });
      return chunk.data();
    };
    for (std::size_t r0 = 0; r0 < V; r0 += kChunk) {
      const std::size_t cr = std::min(kChunk, V - r0);
      Gemm(pool, Op::kN, Op::kT, n, cr, H, 1.0F, hs.data(), H, rows_of(r0, cr), H, 0.0F, logits.data() + r0, V);
    }
    if (probe) probe->logits = logits;
    Buf dlog(backward ? n * V : 0);
    for (std::size_t r = 0; r < n; ++r) {
      const float* l = logits.data() + r * V;
      const float mx = *std::max_element(l, l + V);
      double den = 0;
      for (std::size_t j = 0; j < V; ++j) den += std::exp(static_cast<double>(l[j] - mx));
      const auto y = static_cast<std::size_t>(ex.lm_targets[r].second);
      loss.sum += std::log(den) + mx - l[y];
      loss.correct += static_cast<std::uint32_t>(static_cast<std::size_t>(std::max_element(l, l + V) - l) == y);
      if (backward)
        for (std::size_t j = 0; j < V; ++j)
          dlog[r * V + j] =
              scale * (static_cast<float>(std::exp(static_cast<double>(l[j] - mx)) / den) - (j == y ? 1.0F : 0.0F));
    }
    loss.count = static_cast<std::uint32_t>(n);
    if (backward) {
      Buf dhs(n * H);
      for (std::size_t r0 = 0; r0 < V; r0 += kChunk) {
        const std::size_t cr = std::min(kChunk, V - r0);
        Gemm(pool, Op::kN, Op::kN, n, H, cr, 1.0F, dlog.data() + r0, V, rows_of(r0, cr), H, r0 == 0 ? 0.0F : 1.0F,
             dhs.data(), H);
      }
      for (std::size_t r = 0; r < n; ++r) {
        float* dst = dhn.data() + ex.lm_targets[r].first * H;
        for (std::size_t i = 0; i < H; ++i) dst[i] += dhs[r * H + i];
      }
    }
  }
  if (!backward) return loss;

  // Backward: final norm, then each layer (recomputing its tape).
  Buf d_res(T * H);
  RmsNormBackRows(pool, res[K].data(), output_norm_, rfin.data(), dhn.data(), T, H, d_res.data(), false);
  if (probe) probe->d_residual.assign(K + 1, {});
  if (probe) probe->d_residual[K] = d_res;
  for (std::size_t l = K; l-- > 0;) {
    Tape tp;
    Buf scratch(T * H);
    LayerFwd(c, layers_[l], D, res[l].data(), scratch.data(), tp, true);
    LayerBwd(c, layers_[l], D, res[l].data(), tp, d_res);
    if (probe) probe->d_residual[l] = d_res;
  }
  return loss;
}

}  // namespace halo::train
