// Adapted from gufo src/models/qwen38_flash_next/mtp_policy.hpp (MIT, Copyright (c) 2026 gufo contributors; see NOTICE).
// Changes: namespace; the cycle costs come from the caller (halo's measured verify costs), not gufo's table;
// MtpBatchController dropped (batches of MTP sessions are a follow-up).
#ifndef HALO_ENGINE_MTP_POLICY_HPP_
#define HALO_ENGINE_MTP_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>


namespace halo::engine {

inline constexpr std::uint32_t kMaxMtpDraftTokens = 7;

// Milliseconds of one speculative cycle with `width` drafts (index 0: an ordinary step), measured on halo-flash
// (docs/MTP.md, pinned, short context): verify of width + 1 rows with the exact batched kernels, plus the
// drafts on the trimmed head and the draft block's catch-up. Context does not change the ratios much below
// 8K; the policy only needs ratios.
inline std::array<float, kMaxMtpDraftTokens + 1> MtpCycleCosts(std::uint32_t /*context*/,
                                                               std::uint32_t /*concurrency*/) noexcept {
  return {38.3F, 55.4F, 69.3F, 82.8F, 99.9F, 114.0F, 128.0F, 142.0F};
}

struct MtpLengthState {
  std::array<float, kMaxMtpDraftTokens> successes;
  std::array<float, kMaxMtpDraftTokens> failures;
  std::uint32_t retry_tokens{0};
  std::uint32_t probe_depth{0};
  std::uint32_t explored_depth{0};
  std::uint32_t probe_delay{0};
  std::uint32_t failed_depths{0};
};

// Length depends on committed acceptance history, never request timings.
// Only the first rejected draft is an observed failure; its unverified tail
// and a fully accepted, budget-limited chain are censored observations.
class MtpLengthController {
public:
  explicit MtpLengthController(std::uint32_t limit,
                               std::uint32_t concurrency = 1)
      : limit_(std::clamp(limit, 1U, kMaxMtpDraftTokens)),
        concurrency_(std::clamp(concurrency, 1U, 8U)) {
    Reset();
  }

  [[nodiscard]] std::uint32_t Choose(std::uint32_t budget,
                                     std::uint32_t context = 0) const noexcept {
    const auto cap = std::min(budget, limit_);
    if (cap == 0 || state_.retry_tokens != 0) {
      return 0;
    }
    // After a bounded AR interval, try one proposal to detect recovery.
    return std::min(
        cap, std::max({1U, BestLength(cap, context), state_.probe_depth}));
  }

  [[nodiscard]] float ExpectedTokens(std::uint32_t drafts) const noexcept {
    float expected = 1.0F, survival = 1.0F;
    for (unsigned depth = 0; depth < std::min(drafts, limit_); ++depth) {
      survival *= Probability(depth);
      expected += survival;
    }
    return expected;
  }

  void Observe(std::size_t accepted, std::size_t drafted,
               std::uint32_t context = 0) noexcept {
    if (drafted == 0)
      return;
    drafted = std::min<std::size_t>(drafted, limit_);
    accepted = std::min(accepted, drafted);
    const auto observed = accepted + (accepted < drafted ? 1 : 0);
    for (std::size_t depth = 0; depth < observed; ++depth) {
      state_.successes[depth] =
          0.75F * state_.successes[depth] + (depth < accepted ? 1.0F : 0.0F);
      state_.failures[depth] =
          0.75F * state_.failures[depth] + (depth == accepted ? 1.0F : 0.0F);
    }
    // A fully accepted chain supplies no observations for its tail. Probe
    // farther rather than permanently treating those depths as the prior.
    state_.probe_depth = 0;
    if (accepted == drafted && drafted < limit_) {
      if (drafted >= state_.explored_depth || state_.probe_delay == 0) {
        state_.probe_depth = std::min<std::uint32_t>(2 * drafted, limit_);
        state_.probe_delay = 16;
      } else {
        --state_.probe_delay;
      }
    }
    state_.explored_depth =
        std::max<std::uint32_t>(state_.explored_depth, observed);
    if (accepted < drafted)
      state_.failed_depths |= 1U << accepted;
    if (BestLength(limit_, context) == 0)
      state_.retry_tokens = 16;
  }

  void ObserveArToken() noexcept {
    if (state_.retry_tokens != 0)
      --state_.retry_tokens;
  }

  void Reset() noexcept {
    state_.successes.fill(1.5F);
    state_.failures.fill(0.5F);
    state_.retry_tokens = 0;
    state_.probe_depth = 0;
    state_.explored_depth = 0;
    state_.probe_delay = 0;
    state_.failed_depths = 0;
  }

  [[nodiscard]] MtpLengthState State() const noexcept { return state_; }

  [[nodiscard]] bool Restore(const MtpLengthState& state) noexcept {
    if (state.retry_tokens > 16 || state.probe_depth > limit_ ||
        state.explored_depth > limit_ || state.probe_delay > 16 ||
        state.failed_depths >= (1U << limit_))
      return false;
    for (std::size_t depth = 0; depth < kMaxMtpDraftTokens; ++depth) {
      const float success = state.successes[depth];
      const float failure = state.failures[depth];
      if (!std::isfinite(success) || !std::isfinite(failure) ||
          success < 0.0F || failure < 0.0F ||
          !std::isfinite(success + failure) || success + failure <= 0.0F)
        return false;
    }
    state_ = state;
    return true;
  }

private:
  [[nodiscard]] float Probability(unsigned depth) const noexcept {
    return depth < state_.explored_depth &&
                   (state_.failed_depths & (1U << depth)) == 0
               ? 1.0F
               : state_.successes[depth] /
                     (state_.successes[depth] + state_.failures[depth]);
  }
  [[nodiscard]] std::uint32_t BestLength(std::uint32_t cap,
                                         std::uint32_t context) const noexcept {
    const auto costs = MtpCycleCosts(context, concurrency_);
    float survival = 1.0F;
    float expected_tokens = 1.0F;  // The sampled target anchor.
    float best_score = 1.0F;       // Ordinary target decoding.
    std::uint32_t best = 0;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      const auto depth = length - 1;
      // An explored depth with no actual rejection remains optimistic.
      // Otherwise the prior alone shortens perfect chains and prevents
      // their less frequently sampled tails from collecting evidence.
      const float probability = Probability(depth);
      survival *= probability;
      expected_tokens += survival;
      const float cost = costs[length] / costs[0];
      const float score = expected_tokens / cost;
      if (score > best_score) {
        best_score = score;
        best = length;
      }
    }
    return best;
  }

  std::uint32_t limit_;
  std::uint32_t concurrency_;
  MtpLengthState state_{};
};

}  // namespace halo::engine

#endif  // HALO_ENGINE_MTP_POLICY_HPP_
