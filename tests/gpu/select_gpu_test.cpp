// Device-side MoE routing and QSA block selection against the reference's
// rules, at production sizes and with heavy ties (Hegel properties, both
// backends). Values come from coarse grids, so two scores are either equal
// or far apart and CPU/GPU rounding cannot reorder them; the tie rule is
// then checked exactly.

#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "kernels/core_hsaco.hpp"
#include "kernels/step_params.h"
#include "rt/device.hpp"

namespace gs = hegel::generators;
using namespace halo;

namespace {

std::vector<std::unique_ptr<rt::Device>> Devices() {
  std::vector<std::unique_ptr<rt::Device>> out;
  for (auto b : {rt::Backend::kHip, rt::Backend::kHrx}) {
    std::string error;
    if (auto d = rt::Device::Create(b, &error); d && d->LoadModule(kernels::CoreHsaco(), &error)) out.push_back(std::move(d));
  }
  return out;
}

// Seeded values on a grid of `levels` steps (levels == 1: all equal).
std::vector<float> GridValues(std::size_t n, std::uint32_t seed, std::uint32_t levels, float step, float zero_frac) {
  std::vector<float> v(n);
  for (auto& x : v) {
    seed = seed * 1664525u + 1013904223u;
    const float u = static_cast<float>(seed >> 8) / 16777216.0f;
    seed = seed * 1664525u + 1013904223u;
    x = u < zero_frac ? ((seed >> 31) ? -0.0f : 0.0f) : step * static_cast<float>((seed >> 8) % levels + 1);
  }
  return v;
}

// The reference (src/model/reference.cpp): stable sort by score, keep k,
// position order, expand, then the incomplete tail.
std::vector<std::uint32_t> ReferenceVisible(const std::vector<float>& score, std::uint32_t k, std::uint32_t ratio,
                                            std::uint32_t n_kv) {
  const auto n = static_cast<std::uint32_t>(score.size());
  std::vector<std::uint32_t> visible;
  if (n <= k) {
    visible.resize(n_kv);
    std::iota(visible.begin(), visible.end(), 0u);
    return visible;
  }
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return score[a] > score[b]; });
  order.resize(k);
  std::sort(order.begin(), order.end());
  for (auto b : order)
    for (std::uint32_t t = 0; t < ratio; ++t) visible.push_back(b * ratio + t);
  for (std::uint32_t p = n * ratio; p < n_kv; ++p) visible.push_back(p);
  return visible;
}

}  // namespace

TEST(GpuSelect, QsaSelectionMatchesReferenceRule) {
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  hegel::test([&](hegel::TestCase& tc) {
    auto& dev = *devices[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = devices.size() - 1}))];
    // Up to 70k blocks: past the 262K-token context at ratio 4.
    const auto n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 70000}));
    const auto k = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 2 * n}));
    const auto ratio = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
    const auto tail = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = ratio - 1}));
    const auto levels = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 1u << 20}));
    const auto score = GridValues(n, tc.draw(gs::integers<std::uint32_t>()), levels,
                                  tc.draw(gs::sampled_from(std::vector<float>{1e-30f, 0.25f, 3.0f, 1e30f})),
                                  tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.5f, 1.0f})));
    const std::uint32_t n_kv = n * ratio + tail;
    const auto want = ReferenceVisible(score, k, ratio, n_kv);
    std::string error;
    auto sb = dev.Alloc(n * 4, &error), vb = dev.Alloc((n_kv + 1) * 4ull, &error);
    ASSERT_TRUE(sb && vb) << error;
    ASSERT_TRUE(dev.Upload(sb, 0, score.data(), n * 4ull, &error)) << error;
    ASSERT_TRUE(dev.Fill(vb, 0xEE, &error)) << error;
    std::uint32_t params[kHaloStepParamCount] = {};
    params[kHaloComplete] = n;
    params[kHaloNkv] = n_kv;
    auto pb = dev.Alloc(sizeof(params), &error);
    ASSERT_TRUE(pb && dev.Upload(pb, 0, params, sizeof(params), &error)) << error;
    const auto kid = dev.Kernel("halo_qsa_select", &error);
    const rt::Arg args[] = {rt::Arg::Buf(sb), rt::Arg::Buf(vb), rt::Arg::Buf(pb), rt::Arg::Val(k), rt::Arg::Val(ratio)};
    ASSERT_TRUE(dev.Launch(kid, {1}, {1024}, args, &error)) << error;
    dev.Free(pb);
    std::vector<std::uint32_t> got(want.size() + 1);
    ASSERT_TRUE(dev.Download(vb, 0, got.data(), got.size() * 4, &error)) << error;
    dev.Free(sb);
    dev.Free(vb);
    ASSERT_EQ(got[want.size()], 0xEEEEEEEEu) << "wrote past the visible list";
    got.pop_back();
    ASSERT_EQ(got, want) << "n=" << n << " k=" << k << " ratio=" << ratio << " levels=" << levels;
  }, {.test_cases = 150});
}

TEST(GpuSelect, RouterTopkMatchesReferenceRule) {
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  hegel::test([&](hegel::TestCase& tc) {
    auto& dev = *devices[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = devices.size() - 1}))];
    const auto n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 1024}));
    const auto k = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = std::min(n, 16u)}));
    const auto levels = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 64}));
    // Logits on a 0.25 grid around an offset: equal or clearly apart.
    auto logits = GridValues(n, tc.draw(gs::integers<std::uint32_t>()), levels, 0.25f, 0.0f);
    const float offset = tc.draw(gs::sampled_from(std::vector<float>{-40.0f, 0.0f, 30.0f}));
    for (auto& x : logits) x += offset;
    // Reference: softmax, top-k by probability with lower index first, renorm.
    const float mx = *std::max_element(logits.begin(), logits.end());
    double den = 0;
    std::vector<float> p(n);
    for (std::uint32_t i = 0; i < n; ++i) den += p[i] = std::exp(logits[i] - mx);
    for (auto& v : p) v = static_cast<float>(v / den);
    std::vector<std::uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return p[a] > p[b]; });
    float sum = 0;
    for (std::uint32_t i = 0; i < k; ++i) sum += p[order[i]];
    sum = std::max(sum, 6.103515625e-5f);

    std::string error;
    auto lb = dev.Alloc(n * 4ull, &error), ib = dev.Alloc(k * 4ull, &error), wb = dev.Alloc(k * 4ull, &error);
    ASSERT_TRUE(lb && ib && wb) << error;
    ASSERT_TRUE(dev.Upload(lb, 0, logits.data(), n * 4ull, &error)) << error;
    const auto kid = dev.Kernel("halo_router_topk", &error);
    const rt::Arg args[] = {rt::Arg::Buf(lb), rt::Arg::Buf(ib), rt::Arg::Buf(wb), rt::Arg::Val(n), rt::Arg::Val(k)};
    ASSERT_TRUE(dev.Launch(kid, {1}, {256}, args, &error)) << error;
    std::vector<std::uint32_t> ids(k);
    std::vector<float> w(k);
    ASSERT_TRUE(dev.Download(ib, 0, ids.data(), k * 4ull, &error)) << error;
    ASSERT_TRUE(dev.Download(wb, 0, w.data(), k * 4ull, &error)) << error;
    dev.Free(lb);
    dev.Free(ib);
    dev.Free(wb);
    for (std::uint32_t i = 0; i < k; ++i) {
      ASSERT_EQ(ids[i], order[i]) << "slot " << i << " n=" << n << " k=" << k << " levels=" << levels;
      ASSERT_NEAR(w[i], p[order[i]] / sum, 1e-5f * std::max(1.0f, p[order[i]] / sum)) << "slot " << i;
    }
  }, {.test_cases = 200});
}
