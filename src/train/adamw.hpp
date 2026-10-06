#ifndef HALO_TRAIN_ADAMW_HPP_
#define HALO_TRAIN_ADAMW_HPP_

// AdamW over the trainer's FP32 master weights (torch.optim.AdamW semantics:
// decoupled weight decay p *= 1 - lr * wd, bias-corrected moments), with
// global gradient-norm clipping and a linear warmup / linear decay schedule
// per parameter group (LoRA, head). Deterministic: the norm is a fixed-order
// double sum.

#include <cstdint>
#include <span>
#include <vector>

#include "train/last_k.hpp"

namespace halo::train {

struct AdamWConfig {
  float lr = 1e-4F;       ///< LoRA group peak learning rate
  float head_lr = 1e-3F;  ///< head group peak learning rate
  float beta1 = 0.9F, beta2 = 0.999F, eps = 1e-8F;
  float weight_decay = 0.01F;
  float max_grad_norm = 1.0F;  ///< 0: no clipping
  std::uint64_t warmup_steps = 0;
  std::uint64_t total_steps = 0;  ///< linear decay to 0 at this step (0: constant after warmup)
};

/// Learning-rate multiplier for 1-based step `t`.
[[nodiscard]] double LrScale(const AdamWConfig& c, std::uint64_t t);

struct AdamWState {
  std::vector<float> m, v;
  std::uint64_t step = 0;  ///< updates applied so far
};

struct AdamWStepInfo {
  double grad_norm = 0;  ///< before clipping
  double lr = 0, head_lr = 0;
};

/// Applies one update to `params` from `grad` (num_params each).
AdamWStepInfo AdamWStep(const AdamWConfig& c, std::span<const ParamSlice> slices, std::span<float> params,
                        std::span<const float> grad, AdamWState& state);

}  // namespace halo::train

#endif  // HALO_TRAIN_ADAMW_HPP_
