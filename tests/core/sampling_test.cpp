// Sampling (core/sampling, adapted from gufo) as Hegel properties:
//  - an independent reference of the filter chain (penalties, top-k,
//    temperature, top-p, min-p) agrees with both of the module's paths,
//    the request-state path and BuildDistribution;
//  - every distribution keeps the (penalized) argmax and sums to one;
//  - seeded streams are deterministic, and a copied state continues exactly
//    as the original (speculative rollback);
//  - drawn tokens follow the distribution, and speculative acceptance with
//    the residual draw reproduces the target distribution (chi-square).

#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

#include "core/sampling.hpp"

namespace gs = hegel::generators;
using namespace halo::sampling;

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct Case {
  std::vector<float> logits;
  SamplingConfig config;
  std::vector<TokenId> prompt, generated;
};

// Logits on a coarse grid (ties are frequent and exact), a few masked.
Case DrawCase(hegel::TestCase& tc, std::size_t max_vocab) {
  Case c;
  const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = max_vocab}));
  for (std::size_t i = 0; i < n; ++i) {
    const auto v = tc.draw(gs::integers<int>({.min_value = -12, .max_value = 12}));
    c.logits.push_back(v == -12 && i > 0 ? kNegInf : 0.5f * static_cast<float>(v));
  }
  auto& k = c.config;
  k.temperature = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.3f, 0.7f, 1.0f, 1.5f}));
  k.top_k = tc.draw(gs::integers<std::int32_t>({.min_value = 0, .max_value = static_cast<std::int32_t>(n) + 2}));
  k.top_p = tc.draw(gs::sampled_from(std::vector<float>{1.0f, 0.95f, 0.8f, 0.5f, 0.1f, 0.0f}));
  k.min_p = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.05f, 0.2f, 0.9f, 1.0f}));
  k.min_keep = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 3}));
  k.repeat_penalty = tc.draw(gs::sampled_from(std::vector<float>{1.0f, 1.1f, 0.8f, 2.0f}));
  k.repeat_last_n = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 6}));
  k.frequency_penalty = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.5f, -0.25f}));
  k.presence_penalty = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.75f, -0.5f}));
  k.seed = tc.draw(gs::integers<std::int64_t>({.min_value = 0, .max_value = 1 << 30}));
  const auto token = [&] { return tc.draw(gs::integers<TokenId>({.min_value = 0, .max_value = static_cast<TokenId>(n - 1)})); };
  const auto np = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 6}));
  const auto ng = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 6}));
  for (std::size_t i = 0; i < np; ++i) c.prompt.push_back(token());
  for (std::size_t i = 0; i < ng; ++i) c.generated.push_back(token());
  return c;
}

// Independent statement of the documented chain. Ties keep the lower id.
std::map<TokenId, double> Reference(const Case& c) {
  const auto& k = c.config;
  std::map<TokenId, std::pair<std::uint32_t, bool>> pen;  // generated count, in repeat window
  if (k.frequency_penalty != 0 || k.presence_penalty != 0)
    for (auto t : c.generated) ++pen[t].first;
  if (k.repeat_penalty != 1 && k.repeat_last_n != 0) {
    const std::size_t g = std::min(c.generated.size(), k.repeat_last_n);
    const std::size_t p = std::min(c.prompt.size(), k.repeat_last_n - g);
    for (std::size_t i = c.generated.size() - g; i < c.generated.size(); ++i) pen[c.generated[i]].second = true;
    for (std::size_t i = c.prompt.size() - p; i < c.prompt.size(); ++i) pen[c.prompt[i]].second = true;
  }
  std::vector<std::pair<double, TokenId>> cand;
  for (TokenId t = 0; t < c.logits.size(); ++t) {
    if (!std::isfinite(c.logits[t])) continue;
    double l = c.logits[t];
    if (auto it = pen.find(t); it != pen.end()) {
      if (it->second.second && k.repeat_penalty != 1) l = l <= 0 ? l * k.repeat_penalty : l / k.repeat_penalty;
      l -= double(k.frequency_penalty) * it->second.first;
      if (it->second.first) l -= k.presence_penalty;
    }
    cand.push_back({l, t});
  }
  std::sort(cand.begin(), cand.end(), [](auto a, auto b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
  const auto floor = [&] { return std::min(cand.size(), std::max<std::size_t>(k.min_keep, 1)); };
  if (k.top_k > 0) cand.resize(std::min(cand.size(), std::max<std::size_t>(k.top_k, floor())));
  if (k.temperature == 0) return {{cand[0].second, 1.0}};
  const auto softmax = [&] {
    std::vector<double> w;
    double s = 0;
    for (auto& [l, t] : cand) s += w.emplace_back(std::exp((l - cand[0].first) / k.temperature));
    for (auto& x : w) x /= s;
    return w;
  };
  if (k.top_p < 1 && cand.size() > 1) {
    const auto w = softmax();
    double cum = 0;
    std::size_t keep = 0;
    while (keep < cand.size()) {
      cum += w[keep++];
      if (keep >= floor() && cum >= k.top_p) break;
    }
    cand.resize(keep);
  }
  if (k.min_p > 0 && cand.size() > 1) {
    const double thr = double(k.temperature) * std::log(double(k.min_p));
    std::size_t keep = 0;
    while (keep < cand.size() && cand[keep].first - cand[0].first >= thr) ++keep;
    cand.resize(std::min(std::max(keep, floor()), cand.size()));
  }
  const auto w = softmax();
  std::map<TokenId, double> out;
  for (std::size_t i = 0; i < cand.size(); ++i)
    if (w[i] > 0) out[cand[i].second] = w[i];
  return out;
}

std::map<TokenId, double> AsMap(const SamplingDistribution& d) {
  std::map<TokenId, double> m;
  for (const auto& e : d.entries()) m[e.token] = e.value;
  return m;
}

void ExpectSame(const std::map<TokenId, double>& got, const std::map<TokenId, double>& want, const char* what) {
  ASSERT_EQ(got.size(), want.size()) << what << ": support differs";
  for (const auto& [t, p] : want) {
    ASSERT_TRUE(got.contains(t)) << what << ": token " << t << " missing";
    ASSERT_NEAR(got.at(t), p, 1e-9) << what << ": token " << t;
  }
}

// Chi-square goodness of fit with a bound loose enough to be deterministic
// in practice for the sample sizes used, tight enough to catch a biased draw.
// Tokens expected fewer than 5 times are pooled into one cell (the usual
// validity rule): one draw of a token with expected count 0.02 is a 2% event,
// not a bias, but alone it would add ~40 to the statistic.
void ExpectFollows(const std::map<TokenId, int>& counts, const std::map<TokenId, double>& p, int n) {
  double chi2 = 0, rare_expected = 0, rare_observed = 0;
  std::size_t cells = 0;
  for (const auto& [t, pt] : p) {
    const double expected = pt * n;
    const double observed = counts.contains(t) ? counts.at(t) : 0;
    if (expected < 5) {
      rare_expected += expected;
      rare_observed += observed;
    } else {
      chi2 += (observed - expected) * (observed - expected) / expected;
      ++cells;
    }
  }
  if (rare_expected >= 5) {
    chi2 += (rare_observed - rare_expected) * (rare_observed - rare_expected) / rare_expected;
    ++cells;
  } else {  // too rare for chi-square: bound the pooled count (Poisson mean rare_expected; P(> mean + 12) is tiny)
    ASSERT_LE(rare_observed, rare_expected + 12) << "rare tokens drawn " << rare_observed << " times, expected "
                                                 << rare_expected;
  }
  for (const auto& [t, c] : counts) ASSERT_TRUE(p.contains(t) && p.at(t) > 0) << "drew token " << t << " outside the support";
  const double df = std::max<double>(1, static_cast<double>(cells) - 1);
  ASSERT_LT(chi2, df + 8 * std::sqrt(2 * df) + 25) << "chi-square " << chi2 << " with " << df << " dof";
}

}  // namespace

TEST(Sampling, FilterChainMatchesIndependentReferenceOnBothPaths) {
  hegel::test([](hegel::TestCase& tc) {
    const Case c = DrawCase(tc, 24);
    const auto want = Reference(c);
    ExpectSame(AsMap(BuildDistribution(c.logits, c.config, c.prompt, c.generated)), want, "BuildDistribution");
    SamplerState s(c.config, c.prompt);
    s.Accept(c.generated);
    ExpectSame(AsMap(s.Distribution(c.logits)), want, "SamplerState");
    // The penalized argmax always survives filtering, and greedy picks it.
    const auto best = std::max_element(want.begin(), want.end(), [](auto a, auto b) {
      return a.second != b.second ? a.second < b.second : a.first > b.first;
    })->first;
    double total = 0;
    for (auto& [t, p] : want) total += p;
    ASSERT_NEAR(total, 1.0, 1e-9);
    if (c.config.temperature == 0) ASSERT_EQ(s.Sample(c.logits), best);
  }, {.test_cases = 600});
}

// Above 1024 tokens, top-p without top-k first keeps the best 256 and falls
// back to every token only when they hold less than top_p of the mass; the
// cut-off then uses the full softmax sum. Drawn so both branches occur: a
// flat tail makes the best 256 too light for high top_p, a steep head makes
// them enough for low top_p.
TEST(Sampling, LargeVocabularyTopPMatchesReferenceOnBothBranches) {
  hegel::test([](hegel::TestCase& tc) {
    Case c = DrawCase(tc, 24);
    const auto n = tc.draw(gs::integers<std::size_t>({.min_value = 1025, .max_value = 3000}));
    const auto head = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 400}));
    const float lift = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 2.0f, 6.0f}));
    c.logits.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      const auto v = tc.draw(gs::integers<int>({.min_value = -4, .max_value = 4}));
      c.logits[i] = 0.25f * static_cast<float>(v) + (i < head ? lift : 0.0f);
    }
    if (tc.draw(gs::booleans())) c.logits[tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = n - 1}))] = kNegInf;
    // Anywhere in the vocabulary, not only near the front.
    std::rotate(c.logits.begin(),
                c.logits.begin() + static_cast<std::ptrdiff_t>(tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = n - 1}))),
                c.logits.end());
    // Mostly top-p alone (the fast path); sometimes a large top-k first.
    c.config.top_k = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 3})) == 0
                         ? tc.draw(gs::integers<std::int32_t>({.min_value = 1, .max_value = 400}))
                         : 0;
    c.config.top_p = tc.draw(gs::sampled_from(std::vector<float>{0.95f, 0.8f, 0.5f, 0.1f}));
    c.config.temperature = tc.draw(gs::sampled_from(std::vector<float>{0.3f, 0.7f, 1.0f, 1.5f}));
    const auto want = Reference(c);
    SamplerState st(c.config, c.prompt);
    st.Accept(c.generated);
    ExpectSame(AsMap(st.Distribution(c.logits)), want, "SamplerState");
    ExpectSame(AsMap(BuildDistribution(c.logits, c.config, c.prompt, c.generated)), want, "BuildDistribution");
  }, {.test_cases = 150});
}

// The fallback decision itself: 200 tokens at 0 and 900 at -1, temperature
// 0.3. The best 256 hold 86.9% of the mass, below top_p = 0.95, so every
// token must be considered; misjudging their mass (by 17% here is enough)
// would cut the distribution to 256 tokens.
TEST(Sampling, LargeVocabularyTopPFallsBackWhenTheBest256AreTooLight) {
  Case c;
  c.logits.assign(1100, -1.0f);
  std::fill(c.logits.begin(), c.logits.begin() + 200, 0.0f);
  c.config.top_k = 0;
  c.config.top_p = 0.95f;
  c.config.temperature = 0.3f;
  const auto want = Reference(c);
  ASSERT_GT(want.size(), 256u);
  SamplerState st(c.config, c.prompt);
  ExpectSame(AsMap(st.Distribution(c.logits)), want, "SamplerState");
  ExpectSame(AsMap(BuildDistribution(c.logits, c.config, c.prompt, c.generated)), want, "BuildDistribution");
}

// Seed 0 is a seed like any other (only negative seeds draw from the OS).
// top_p keeps the shortest prefix whose mass reaches top_p, inclusive: two
// tied tokens at top_p = 0.5 keep one.
TEST(Sampling, TopPBoundaryIsInclusive) {
  Case c;
  c.logits = {1.0f, 1.0f, kNegInf};
  c.config.temperature = 1.0f;
  c.config.top_p = 0.5f;
  const auto want = Reference(c);
  ASSERT_EQ(want.size(), 1u);
  SamplerState st(c.config, c.prompt);
  ExpectSame(AsMap(st.Distribution(c.logits)), want, "SamplerState");
  ExpectSame(AsMap(BuildDistribution(c.logits, c.config, c.prompt, c.generated)), want, "BuildDistribution");
}

// No finite logit is an error on every selection path.
TEST(Sampling, AllMaskedLogitsAreAnError) {
  for (const std::size_t n : {std::size_t{5}, std::size_t{2000}})
    for (const auto& [top_k, top_p] : std::vector<std::pair<std::int32_t, float>>{{0, 1.0f}, {3, 1.0f}, {0, 0.9f}}) {
      SamplingConfig k;
      k.temperature = 1.0f;
      k.top_k = top_k;
      k.top_p = top_p;
      const std::vector<float> masked(n, kNegInf);
      SamplerState st(k, {});
      EXPECT_THROW((void)st.Distribution(masked), std::runtime_error) << n << " " << top_k << " " << top_p;
    }
}

TEST(Sampling, SeedZeroIsDeterministic) {
  SamplingConfig k;
  k.temperature = 1.0f;
  k.seed = 0;
  const std::vector<float> flat(64, 0.0f);
  SamplerState a(k, {}), b(k, {});
  for (int i = 0; i < 32; ++i) ASSERT_EQ(a.Sample(flat), b.Sample(flat)) << "draw " << i;
}

TEST(Sampling, SeededStreamsAreDeterministicAndCopiesContinueIdentically) {
  hegel::test([](hegel::TestCase& tc) {
    Case c = DrawCase(tc, 16);
    c.config.temperature = std::max(c.config.temperature, 0.5f);
    SamplerState a(c.config, c.prompt), b(c.config, c.prompt);
    const auto steps = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 20}));
    const auto fork_at = tc.draw(gs::integers<int>({.min_value = 0, .max_value = steps}));
    std::optional<SamplerState> fork;
    std::vector<TokenId> after_fork;
    for (int i = 0; i < steps; ++i) {
      if (i == fork_at) fork.emplace(a);
      const auto ta = a.Sample(c.logits), tb = b.Sample(c.logits);
      ASSERT_EQ(ta, tb) << "step " << i;
      a.Accept(ta);
      b.Accept(tb);
      if (i >= fork_at) after_fork.push_back(ta);
    }
    if (fork) {  // a rolled-back copy replays the same continuation
      for (auto t : after_fork) {
        ASSERT_EQ(fork->Sample(c.logits), t);
        fork->Accept(t);
      }
    }
  }, {.test_cases = 300});
}

// The goodness-of-fit check itself: the Hegel case that broke the unpooled
// statistic (two tokens, p = 5.9e-7 drawn once in 40000: chi-square 40.3 > 37.3)
// passes, and biased counts still fail (positive controls).
TEST(Sampling, GoodnessOfFitPoolsRareTokensAndStillCatchesBias) {
  const int n = 40000;
  EXPECT_NO_FATAL_FAILURE(ExpectFollows({{0, 1}, {1, n - 1}}, {{0, 5.9e-7}, {1, 1 - 5.9e-7}}, n));
  // A 2% token drawn 2.5x too often (expected 800, drew 2000).
  EXPECT_FATAL_FAILURE(ExpectFollows({{0, 2000}, {1, n - 2000}}, {{0, 0.02}, {1, 0.98}}, n), "chi-square");
  // A rare token (expected 0.024) drawn 40 times.
  EXPECT_FATAL_FAILURE(ExpectFollows({{0, 40}, {1, n - 40}}, {{0, 5.9e-7}, {1, 1 - 5.9e-7}}, n), "rare tokens");
}

TEST(Sampling, DrawsFollowTheDistribution) {
  hegel::test([](hegel::TestCase& tc) {
    Case c = DrawCase(tc, 6);
    c.config.temperature = tc.draw(gs::sampled_from(std::vector<float>{0.5f, 1.0f, 2.0f}));
    const auto d = BuildDistribution(c.logits, c.config, c.prompt, c.generated);
    std::uint64_t rng = static_cast<std::uint64_t>(c.config.seed) + 1;
    constexpr int n = 40000;
    std::map<TokenId, int> counts;
    for (int i = 0; i < n; ++i) ++counts[d.Sample(&rng)];
    ExpectFollows(counts, AsMap(d), n);
  }, {.test_cases = 40});
}

// Speculative decoding: draw x from a draft q, accept with min(1, p/q), else
// draw from the residual max(p - q, 0). The result must be distributed as p.
TEST(Sampling, SpeculativeAcceptancePreservesTheTarget) {
  hegel::test([](hegel::TestCase& tc) {
    Case c = DrawCase(tc, 6);
    c.config.temperature = 1.0f;
    const auto p = BuildDistribution(c.logits, c.config);
    const std::size_t V = c.logits.size();
    std::vector<TokenId> ids(V);
    std::vector<float> q(V);
    double qs = 0;
    for (TokenId t = 0; t < V; ++t) {
      ids[t] = t;
      qs += q[t] = static_cast<float>(tc.draw(gs::integers<int>({.min_value = 0, .max_value = 8})));
    }
    if (qs == 0) q[0] = qs = 1;
    for (auto& x : q) x = static_cast<float>(x / qs);
    std::uint64_t rng = static_cast<std::uint64_t>(c.config.seed) + 7;
    constexpr int n = 40000;
    std::map<TokenId, int> counts;
    for (int i = 0; i < n; ++i) {
      double u = Uniform(&rng);
      TokenId x = 0;
      for (; x + 1 < V && (u -= q[x]) >= 0; ++x) {}
      while (q[x] == 0) --x;  // rounding at the tail: last token with mass
      const double accept = std::min(1.0, p.probability(x) / q[x]);
      counts[Uniform(&rng) < accept ? x : p.SampleResidual(ids, q, &rng)]++;
    }
    ExpectFollows(counts, AsMap(p), n);
  }, {.test_cases = 40});
}
