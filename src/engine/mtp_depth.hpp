// Per-session draft depth for MTP speculative decoding (P5 round 3, docs/MTP.md).
//
// Chooses k = argmax E[tokens | k] / E[cost | k] over k = 0 .. cap, where
//   E[tokens | k] = 1 + sum_{i=1..k} prod_{j<i} a_j   (a_j: conditional acceptance of draft j + 1),
//   E[cost | k]   = cycle_ms[k] + catchup_ms * (E[tokens | k] - 1) + rollback_ms * P(some draft rejected)
// and cycle_ms[0] is an ordinary step (with the draft block's one following step, so k = 0 is AR + catch-up).
// cycle_ms / catchup_ms are measured halo-flash costs (MtpMeasuredCosts); decisions depend only on committed
// acceptance history, never on request timings, so a run is reproducible.
//
// a_j is a Beta estimate with exponential forgetting (every depth's counts decay by lambda each drafted cycle)
// around a persistent prior (weight prior_weight) that never decays away: one unlucky cycle cannot zero a depth,
// a long run of rejections still drags it down within ~1 / (1 - lambda) cycles, and a depth that stops being
// drafted (censored) drifts back to its prior instead of keeping a stale low estimate that would never be retried. The prior mean of depth 0 is prior_mean;
// depth j's is depth j - 1's estimate, so depths past the deepest one drafted (never observed: censored) are
// expected to accept like the last observed one rather than like a fixed prior, which would cap the depth one
// short of the best at high acceptance. Only the first rejected depth is observed as a failure; drafts past it
// and depths past a fully accepted chain are censored.
// When the best k is 0 (drafting does not pay), a depth-1 probe runs after probe_every ordinary tokens so that
// acceptance recovering is noticed; the interval doubles after each rejected probe (up to probe_max), so a
// session where drafting never pays loses little to probes, and returns to probe_every after an accepted one.
#ifndef HALO_ENGINE_MTP_DEPTH_HPP_
#define HALO_ENGINE_MTP_DEPTH_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace halo::engine {

inline constexpr std::uint32_t kMtpDepthMax = 7;

struct MtpCosts {
  std::array<float, kMtpDepthMax + 1> cycle_ms{};  ///< [k]: verify of k + 1 rows + k drafts (+ the AR step at 0)
  float catchup_ms = 0;                            ///< one headless draft-block step per accepted draft
  float rollback_ms = 0;                           ///< recurrent-state copy when a draft is rejected
};

/// halo-flash measured costs (main e102375 + round 3, pinned, 9 prompts x 256 at fixed depths 0..5, trimmed 32K
/// draft head; docs/MTP.md round 3): joint least squares over 4284 cycles of ms = C[k] + c * accepted
/// + r * [accepted < k]; C[0] is the AR step (38.65 ms measured with the draft block's follow step, minus c);
/// k = 6, 7 extrapolated at +14.5 ms a draft.
inline MtpCosts MtpMeasuredCosts() noexcept {
  return {{37.3F, 54.15F, 69.73F, 84.75F, 102.25F, 113.14F, 127.6F, 142.1F}, 1.37F, 1.46F};
}

struct MtpDepthOptions {
  std::uint32_t limit = 4;  ///< deepest draft (verify rows <= limit + 1)
  float lambda = 0.9F;      ///< forgetting per drafted cycle
  float prior_mean = 0.7F;
  float prior_weight = 4.0F;
  std::uint32_t probe_every = 8;  ///< ordinary tokens before the first depth-1 probe while k = 0 wins
  std::uint32_t probe_max = 64;   ///< longest interval between probes
};

class MtpDepthChooser {
 public:
  explicit MtpDepthChooser(MtpCosts costs = MtpMeasuredCosts(), MtpDepthOptions o = {})
      : costs_(costs), o_(o) {
    o_.limit = std::clamp<std::uint32_t>(o_.limit, 1, kMtpDepthMax);
    Reset();
  }

  void Reset() noexcept {
    s_.fill(0.0F);
    f_.fill(0.0F);
    since_obs_ = 0;
    probe_interval_ = o_.probe_every;
  }

  /// Conditional acceptance estimate of draft depth j (0-based: draft j + 1 given drafts 1..j accepted).
  [[nodiscard]] float Acceptance(std::uint32_t j) const noexcept {
    float a = o_.prior_mean;
    for (std::uint32_t i = 0; i <= j; ++i) a = (s_[i] + o_.prior_weight * a) / (s_[i] + f_[i] + o_.prior_weight);
    return a;
  }

  [[nodiscard]] float ExpectedTokens(std::uint32_t k) const noexcept {
    float e = 1.0F, survive = 1.0F, a = o_.prior_mean;
    for (std::uint32_t j = 0; j < std::min(k, o_.limit); ++j) {
      a = (s_[j] + o_.prior_weight * a) / (s_[j] + f_[j] + o_.prior_weight);
      e += survive *= a;
    }
    return e;
  }

  /// Expected tokens per millisecond at depth k.
  [[nodiscard]] float Rate(std::uint32_t k) const noexcept {
    if (k == 0) return 1.0F / (costs_.cycle_ms[0] + costs_.catchup_ms);
    float e = 1.0F, survive = 1.0F, a = o_.prior_mean;
    for (std::uint32_t j = 0; j < std::min(k, o_.limit); ++j) {
      a = (s_[j] + o_.prior_weight * a) / (s_[j] + f_[j] + o_.prior_weight);
      e += survive *= a;
    }
    return e / (costs_.cycle_ms[k] + costs_.catchup_ms * (e - 1.0F) + costs_.rollback_ms * (1.0F - survive));
  }

  /// The best depth by expected rate (0: an ordinary step).
  [[nodiscard]] std::uint32_t Best(std::uint32_t cap) const noexcept {
    cap = std::min(cap, o_.limit);
    std::uint32_t best = 0;
    float best_rate = Rate(0);
    for (std::uint32_t k = 1; k <= cap; ++k)
      if (const float r = Rate(k); r > best_rate) best = k, best_rate = r;
    return best;
  }

  /// Depth for the next cycle; `budget`: drafts that can still be used (tokens left - 1).
  [[nodiscard]] std::uint32_t Choose(std::uint32_t budget) const noexcept {
    if (budget == 0) return 0;
    const std::uint32_t k = Best(budget);
    if (k == 0 && since_obs_ >= probe_interval_) return 1;
    return k;
  }

  /// A cycle with `drafted` drafts, `accepted` of them accepted (0: an ordinary step).
  void Observe(std::uint32_t accepted, std::uint32_t drafted) noexcept {
    if (drafted == 0) {
      ++since_obs_;
      return;
    }
    const bool probe = Best(o_.limit) == 0;  // drafted although AR was the choice
    since_obs_ = 0;
    drafted = std::min(drafted, o_.limit);
    accepted = std::min(accepted, drafted);
    for (std::uint32_t j = 0; j < kMtpDepthMax; ++j) s_[j] *= o_.lambda, f_[j] *= o_.lambda;
    for (std::uint32_t j = 0; j < accepted; ++j) s_[j] += 1.0F;
    if (accepted < drafted) f_[accepted] += 1.0F;
    if (Best(o_.limit) > 0) probe_interval_ = o_.probe_every;
    else if (probe) probe_interval_ = accepted > 0 ? o_.probe_every : std::min(2 * probe_interval_, o_.probe_max);
  }

  [[nodiscard]] const MtpDepthOptions& options() const noexcept { return o_; }

 private:
  MtpCosts costs_;
  MtpDepthOptions o_;
  std::array<float, kMtpDepthMax> s_{}, f_{};
  std::uint32_t since_obs_ = 0, probe_interval_ = 0;
};

}  // namespace halo::engine

#endif  // HALO_ENGINE_MTP_DEPTH_HPP_
