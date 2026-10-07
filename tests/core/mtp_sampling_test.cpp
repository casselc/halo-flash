// MTP speculative sampling (src/engine/mtp_sampling.hpp, gufo's algorithm; this test follows gufo's
// tests/models/qwen38_flash_next/mtp_sampling_test.cpp): proposals are exact discrete masses, acceptance plus
// residual reproduces the target distribution exactly (algebraically and by frequency), verification uses the
// canonical AR distribution and RNG, and the length controller grows on full acceptance and backs off on
// rejection.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>

#include "core/sampling.hpp"
#include "engine/mtp_depth.hpp"
#include "engine/mtp_policy.hpp"
#include "engine/mtp_sampling.hpp"

using namespace halo;

namespace {
engine::MtpCandidateLogits Candidates() {
  engine::MtpCandidateLogits c;
  c.size = 3;
  c.ids = {3, 1, 4};
  c.logits = {1.2F, 0.1F, -0.7F};
  return c;
}
}  // namespace

TEST(MtpSampling, ProposalMassesAndResidualConserveTheTarget) {
  const std::array<float, 5> target_logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  for (const float t : {0.5F, 0.8F, 1.0F, 1.3F}) {
    sampling::SamplingConfig cfg;
    cfg.temperature = t;
    sampling::SamplerState sampler(cfg);
    const auto p = sampler.Distribution(target_logits);
    for (std::uint64_t seed = 0; seed < 16; ++seed) {
      auto rng = seed, replay_rng = seed;
      const auto q = engine::SampleMtpProposal(Candidates(), sampler, &rng);
      const auto replay = engine::SampleMtpProposal(Candidates(), sampler, &replay_rng);
      ASSERT_EQ(q.token, replay.token);
      ASSERT_EQ(rng, replay_rng);
      double sum = 0, accepted = 0, residual = 0;
      std::array<double, 5> draft{};
      for (std::size_t i = 0; i < q.size; ++i) {
        const double mass = q.probabilities[i] * double(1U << 24);
        ASSERT_EQ(mass, std::floor(mass)) << "not an exact discrete mass";
        draft[q.ids[i]] += q.probabilities[i];
        sum += q.probabilities[i];
      }
      ASSERT_EQ(sum, 1.0);
      ASSERT_EQ(draft[q.token], q.probability);
      for (std::size_t i = 0; i < draft.size(); ++i) {
        accepted += std::min(p.probability(i), draft[i]);
        residual += std::max(p.probability(i) - draft[i], 0.0);
      }
      ASSERT_NEAR(accepted + residual, 1.0, 1e-12);
      for (std::size_t i = 0; i < draft.size(); ++i) {
        const double corr = residual > 0 ? (1 - accepted) * std::max(p.probability(i) - draft[i], 0.0) / residual : 0;
        ASSERT_NEAR(std::min(p.probability(i), draft[i]) + corr, p.probability(i), 1e-12);
      }
    }
  }
}

TEST(MtpSampling, SampledOutputFrequenciesMatchTheTarget) {
  const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  constexpr unsigned kTrials = 100000;
  for (const float t : {0.7F, 0.8F, 1.0F}) {
    sampling::SamplingConfig cfg;
    cfg.temperature = t;
    cfg.seed = 73;
    sampling::SamplerState sampler(cfg);
    const auto p = sampler.Distribution(logits);
    std::array<unsigned, 5> counts{};
    for (unsigned trial = 0; trial < kTrials; ++trial) {
      auto draft_rng = sampling::NextRandom(sampler.mutable_rng_state());
      const auto q = engine::SampleMtpProposal(Candidates(), sampler, &draft_rng);
      ++counts[engine::VerifyMtpProposal(logits, q, sampler).token];
    }
    for (unsigned tok = 0; tok < counts.size(); ++tok)
      EXPECT_NEAR(double(counts[tok]) / kTrials, p.probability(tok), 0.008) << "temperature " << t << " token " << tok;
  }
}

TEST(MtpSampling, VerificationUsesTheCanonicalDistributionAndRng) {
  const std::array<float, 3> logits{0, -1, -2};
  for (const float top_p : {0.8F, 0.8668133F, 0.8668134F, 1.0F}) {
    sampling::SamplingConfig cfg;
    cfg.temperature = 0.5F;
    cfg.top_p = top_p;
    cfg.seed = 6;
    sampling::SamplerState sampler(cfg);
    const auto target = sampler.Distribution(logits);
    for (unsigned draft = 0; draft < logits.size(); ++draft) {
      engine::MtpProposal prop;
      prop.size = 1;
      prop.ids[0] = prop.token = draft;
      prop.probabilities[0] = prop.probability = 1;
      for (unsigned draw = 0; draw < 16; ++draw) {
        auto reference = sampler;
        const bool accepted = reference.Uniform() < target.probability(draft);
        const auto expected = accepted ? draft
                                       : target.SampleResidual(std::span(prop.ids).first(1),
                                                               std::span(prop.probabilities).first(1),
                                                               reference.mutable_rng_state());
        const auto actual = engine::VerifyMtpProposal(logits, prop, sampler);
        ASSERT_EQ(actual.accepted, accepted);
        ASSERT_EQ(actual.token, expected);
        ASSERT_EQ(sampler.rng_state(), reference.rng_state());
      }
    }
  }
}

TEST(MtpPolicy, GrowsOnFullAcceptanceAndBacksOffOnRejection) {
  engine::MtpLengthController controller(7);
  ASSERT_EQ(controller.Choose(0), 0u);
  ASSERT_EQ(controller.Choose(1), 1u);
  for (unsigned i = 0; i < 12; ++i) {
    const auto n = controller.Choose(7);
    controller.Observe(n, n);
  }
  EXPECT_EQ(controller.Choose(7), 7u);
  EXPECT_EQ(controller.Choose(2), 2u);
  for (unsigned i = 0; i < 12; ++i) controller.Observe(0, controller.Choose(7));
  EXPECT_EQ(controller.Choose(7), 0u);
}

namespace {
// Simulated sessions for the depth chooser: draft j + 1 is accepted with probability alpha[j] given the earlier
// ones were (a fresh Bernoulli chain per cycle); cost from the chooser's own cost model.
struct SimResult {
  double tokens = 0, ms = 0;
};
template <typename Choose, typename Observe>
SimResult Simulate(const std::array<float, engine::kMtpDepthMax>& alpha, std::size_t tokens, std::uint32_t seed,
                   Choose choose, Observe observe) {
  const auto costs = engine::MtpMeasuredCosts();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.0F, 1.0F);
  SimResult r;
  while (r.tokens < tokens) {
    const std::uint32_t k = choose();
    std::uint32_t a = 0;
    while (a < k && u(rng) < alpha[a]) ++a;
    observe(a, k);
    r.tokens += a + 1;
    r.ms += costs.cycle_ms[k] + costs.catchup_ms * (k ? a : 1) + (a < k ? costs.rollback_ms : 0.0F);
  }
  return r;
}
double BestFixedRate(const std::array<float, engine::kMtpDepthMax>& alpha, std::uint32_t limit) {
  const auto costs = engine::MtpMeasuredCosts();
  double best = 0;
  for (std::uint32_t k = 0; k <= limit; ++k) {
    double e = 1, s = 1;
    for (std::uint32_t j = 0; j < k; ++j) e += s *= alpha[j];
    best = std::max(best, e / (costs.cycle_ms[k] + costs.catchup_ms * (k ? e - 1 : 1) + (k ? costs.rollback_ms * (1 - s) : 0)));
  }
  return best;
}
}  // namespace

// The chooser's long-run rate reaches the best fixed depth's expected rate (to within 4%) whatever the
// session's acceptance, including sessions where drafting never pays (it must fall back to AR, probes included).
TEST(MtpDepth, LongRunRateMatchesTheBestFixedDepth) {
  for (const float a : {0.0F, 0.2F, 0.35F, 0.45F, 0.55F, 0.7F, 0.8F, 0.9F, 0.97F}) {
    for (const float decay : {1.0F, 0.93F}) {  // flat, and acceptance falling with depth
      std::array<float, engine::kMtpDepthMax> alpha{};
      for (std::uint32_t j = 0; j < alpha.size(); ++j) alpha[j] = a * std::pow(decay, static_cast<float>(j));
      engine::MtpDepthChooser chooser;
      const auto r = Simulate(alpha, 20000, 7, [&] { return chooser.Choose(100); },
                              [&](std::uint32_t acc, std::uint32_t k) { chooser.Observe(acc, k); });
      const double best = BestFixedRate(alpha, chooser.options().limit);
      EXPECT_GE(r.tokens / r.ms, 0.96 * best) << "acceptance " << a << " decay " << decay;
    }
  }
}

// Acceptance moving within a session (prose into code into a list): the chooser follows within a few cycles.
TEST(MtpDepth, FollowsAcceptanceChanges) {
  engine::MtpDepthChooser chooser;
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(0.0F, 1.0F);
  const auto run = [&](float a, int cycles) {
    std::uint32_t k = 0;
    for (int i = 0; i < cycles; ++i) {
      k = chooser.Choose(100);
      std::uint32_t acc = 0;
      while (acc < k && u(rng) < a) ++acc;
      chooser.Observe(acc, k);
    }
    return k;
  };
  EXPECT_GE(run(0.9F, 60), 3u);
  EXPECT_EQ(run(0.1F, 80), 0u);  // drafting stops paying: AR (between probes)
  EXPECT_GE(run(0.9F, 120), 3u);  // probes notice the recovery
}

TEST(MtpDepth, ProbesWhileArWinsAndRespectsTheBudget) {
  engine::MtpDepthChooser chooser;
  int guard = 0;
  while (chooser.Choose(100) != 0 && ++guard < 100) chooser.Observe(0, chooser.Choose(100));  // all rejected
  ASSERT_EQ(chooser.Choose(100), 0u) << "rejections never made AR the choice";
  // Failed probes back off: 8, 16, 32, 64, 64 ordinary tokens between depth-1 probes.
  for (const std::uint32_t want : {8u, 16u, 32u, 64u, 64u}) {
    std::uint32_t ar = 0;
    while (chooser.Choose(100) == 0 && ar < 1000) chooser.Observe(0, 0), ++ar;
    EXPECT_EQ(ar, want);
    ASSERT_EQ(chooser.Choose(100), 1u);  // the probe
    chooser.Observe(0, 1);
  }
  EXPECT_EQ(chooser.Choose(0), 0u);  // no budget: no drafts
  engine::MtpDepthChooser fresh;
  for (int i = 0; i < 50; ++i) fresh.Observe(4, 4);
  EXPECT_EQ(fresh.Choose(2), 2u);  // capped by the budget
  EXPECT_LE(fresh.Choose(100), fresh.options().limit);
}
