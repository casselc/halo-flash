#include "train/adamw.hpp"

#include <algorithm>
#include <cmath>

namespace halo::train {

double LrScale(const AdamWConfig& c, std::uint64_t t) {
  if (c.warmup_steps > 0 && t <= c.warmup_steps) return static_cast<double>(t) / static_cast<double>(c.warmup_steps);
  if (c.total_steps == 0 || c.total_steps <= c.warmup_steps) return 1.0;
  if (t >= c.total_steps) return 0.0;
  return static_cast<double>(c.total_steps - t) / static_cast<double>(c.total_steps - c.warmup_steps);
}

AdamWStepInfo AdamWStep(const AdamWConfig& c, std::span<const ParamSlice> slices, std::span<float> params,
                        std::span<const float> grad, AdamWState& st) {
  if (st.m.size() != params.size()) st.m.assign(params.size(), 0.0F);
  if (st.v.size() != params.size()) st.v.assign(params.size(), 0.0F);
  AdamWStepInfo info;
  double ss = 0;
  for (float g : grad) ss += static_cast<double>(g) * g;
  info.grad_norm = std::sqrt(ss);
  const float clip = c.max_grad_norm > 0 && info.grad_norm > c.max_grad_norm
                         ? static_cast<float>(c.max_grad_norm / (info.grad_norm + 1e-6))
                         : 1.0F;
  const std::uint64_t t = ++st.step;
  const double scale = LrScale(c, t);
  info.lr = c.lr * scale;
  info.head_lr = c.head_lr * scale;
  const double bc1 = 1.0 - std::pow(static_cast<double>(c.beta1), static_cast<double>(t));
  const double bc2 = 1.0 - std::pow(static_cast<double>(c.beta2), static_cast<double>(t));
  for (const auto& s : slices) {
    const float lr = static_cast<float>(s.is_head ? info.head_lr : info.lr);
    const float decay = s.decay ? 1.0F - lr * c.weight_decay : 1.0F;
    const float step_size = static_cast<float>(lr / bc1);
    const float bc2_sqrt = static_cast<float>(std::sqrt(bc2));
    for (std::size_t i = s.offset; i < s.offset + s.size(); ++i) {
      const float g = grad[i] * clip;
      st.m[i] = c.beta1 * st.m[i] + (1.0F - c.beta1) * g;
      st.v[i] = c.beta2 * st.v[i] + (1.0F - c.beta2) * g * g;
      params[i] *= decay;
      params[i] -= step_size * st.m[i] / (std::sqrt(st.v[i]) / bc2_sqrt + c.eps);
    }
  }
  return info;
}

}  // namespace halo::train
