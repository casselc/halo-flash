// MoE routing with a runtime expert count (CODESIGN section 5, arm B):
// cpu::RouteTopK, the reference's routing rule, which the GPU's
// halo_router_topk must equal (GpuSelect.RouterTopkMatchesReferenceRule),
// and ModelWeights::SetExpertsUsed, which every engine path reads k from.

#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "model/cpu_ops.hpp"
#include "model/weights.hpp"

namespace gs = hegel::generators;
using halo::model::ModelWeights;
namespace cpu = halo::model::cpu;

namespace {

// Logits on a 0.25 grid around an offset: ties are common and exact.
std::vector<float> GridLogits(hegel::TestCase& tc, std::uint32_t n) {
  const auto levels = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 64}));
  const float offset = tc.draw(gs::sampled_from(std::vector<float>{-40.0f, 0.0f, 30.0f}));
  std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
  std::vector<float> v(n);
  for (auto& x : v) {
    seed = seed * 1664525u + 1013904223u;
    x = offset + 0.25f * static_cast<float>((seed >> 8) % levels);
  }
  return v;
}

}  // namespace

// Oracle: a full stable sort of the probabilities (lower index first on
// ties), the first k, renormalized.
TEST(Routing, TopKEqualsStableSortOracle) {
  hegel::test([](hegel::TestCase& tc) {
    const auto n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 600}));
    const auto k = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = std::min(n, 16u)}));
    const auto logits = GridLogits(tc, n);
    const float mx = *std::max_element(logits.begin(), logits.end());
    double den = 0;
    std::vector<float> p(n);
    for (std::uint32_t i = 0; i < n; ++i) den += p[i] = std::exp(logits[i] - mx);
    for (auto& v : p) v = static_cast<float>(v / den);
    std::vector<std::uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return p[a] > p[b]; });
    double sum = 0;
    for (std::uint32_t i = 0; i < k; ++i) sum += p[order[i]];

    std::vector<std::uint32_t> ids(k);
    std::vector<float> w(k);
    cpu::RouteTopK(logits, ids, w);
    double total = 0;
    for (std::uint32_t i = 0; i < k; ++i) {
      ASSERT_EQ(ids[i], order[i]) << "slot " << i << " n=" << n << " k=" << k;
      ASSERT_NEAR(w[i], p[order[i]] / sum, 1e-6 * std::max(1.0, p[order[i]] / sum)) << "slot " << i;
      if (i > 0) ASSERT_LE(w[i], w[i - 1]);
      total += w[i];
    }
    ASSERT_NEAR(total, 1.0, 1e-5);
  }, {.test_cases = 300});
}

// Reducing k keeps the same experts' order (a prefix) and their relative
// weights; only the normalizer changes. So k = 8 or 6 is exactly "drop the
// smallest routes and renormalize", the untrained arm-B variant.
TEST(Routing, SmallerKIsRenormalizedPrefix) {
  hegel::test([](hegel::TestCase& tc) {
    const auto n = tc.draw(gs::integers<std::uint32_t>({.min_value = 2, .max_value = 600}));
    const auto k = tc.draw(gs::integers<std::uint32_t>({.min_value = 2, .max_value = std::min(n, 16u)}));
    const auto k2 = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = k - 1}));
    const auto logits = GridLogits(tc, n);
    std::vector<std::uint32_t> ids(k), ids2(k2);
    std::vector<float> w(k), w2(k2);
    cpu::RouteTopK(logits, ids, w);
    cpu::RouteTopK(logits, ids2, w2);
    double head = 0;
    for (std::uint32_t i = 0; i < k2; ++i) head += w[i];
    for (std::uint32_t i = 0; i < k2; ++i) {
      ASSERT_EQ(ids2[i], ids[i]) << "slot " << i;
      ASSERT_NEAR(w2[i], w[i] / head, 1e-5 * std::max(1.0, w[i] / head)) << "slot " << i;
    }
  }, {.test_cases = 300});
}

TEST(Routing, SetExpertsUsedValidatesAndSets) {
  hegel::test([](hegel::TestCase& tc) {
    ModelWeights w;
    w.config.num_experts = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 1024}));
    const auto before = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = w.config.num_experts}));
    w.config.num_experts_used = before;
    const auto k = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 1100}));
    std::string error;
    const bool ok = w.SetExpertsUsed(k, &error);
    ASSERT_EQ(ok, k >= 1 && k <= w.config.num_experts) << k;
    ASSERT_EQ(w.config.num_experts_used, ok ? k : before);
    ASSERT_EQ(error.empty(), ok);
  }, {.test_cases = 200});
}
