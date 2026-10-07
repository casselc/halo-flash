// MTP speculative sampling (src/engine/mtp_sampling.hpp, gufo's algorithm; this test follows gufo's
// tests/models/qwen38_flash_next/mtp_sampling_test.cpp): proposals are exact discrete masses, acceptance plus
// residual reproduces the target distribution exactly (algebraically and by frequency), verification uses the
// canonical AR distribution and RNG, and the length controller grows on full acceptance and backs off on
// rejection.

#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "core/sampling.hpp"
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
