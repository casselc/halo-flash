#ifndef HALO_MODEL_CPU_OPS_HPP_
#define HALO_MODEL_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "model/weights.hpp"

/// Scalar CPU operators for the reference model: one row at a time, every
/// format decoded on the fly. Correctness oracle only; nothing here is tuned.
namespace halo::model::cpu {

/// Decodes one row of `t` (expert `e`) into `out` (t.cols floats).
void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out);

/// y[r] = sum_c W[e][r][c] * x[c] for every row of the matrix.
void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y);

/// Dot product of one weight row with x.
float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x);

/// RMSNorm over `x` in place with gamma `w` (nullptr keeps the raw scale).
void RmsNorm(std::span<float> x, const float* w, float eps);
/// x / sqrt(sum x^2 + eps), the DeltaNet query/key normalization.
void L2Norm(std::span<float> x, float eps);

[[nodiscard]] float Sigmoid(float x) noexcept;
[[nodiscard]] float Silu(float x) noexcept;
[[nodiscard]] float Softplus(float x) noexcept;

/// NEOX-style partial rotary embedding on the first `rotary_dim` elements of
/// each `head_dim` head: pair (i, i + rotary_dim/2) rotates by pos * theta_i.
/// Text-only IMRoPE has equal positions on every section, so it is exactly
/// this rotation.
void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta);

/// MoE routing as the GPU's halo_router_topk computes it: softmax over every
/// expert's logit, the k largest probabilities (ties to the lower index) in
/// descending order, renormalized to sum to 1 (sum floored at 2^-14). The
/// trained count k comes from the file; a smaller k (ModelWeights::
/// SetExpertsUsed) is the untrained renormalized top-k of CODESIGN arm B.
/// `ids` and `weights` hold k entries, 1 <= k <= logits.size().
void RouteTopK(std::span<const float> logits, std::span<std::uint32_t> ids,
               std::span<float> weights);

}  // namespace halo::model::cpu

#endif  // HALO_MODEL_CPU_OPS_HPP_
