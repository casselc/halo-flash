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

// P4: the chunk attention on FP16 WMMA (online softmax) against the decode
// kernel's arithmetic over the same visible lists (halo_attention_select_rows),
// head dim 256, G = 12 or 16 query heads per KV head, 1 .. 2100 visible keys
// (several 64-key chunks, ragged tails), random positions. Bound fixed in
// advance: relative L2 < 5e-3 per token (fp16 inputs: ~5e-4 per value).
TEST(GpuSelect, WmmaChunkAttentionMatchesSelectKernel) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      constexpr std::uint32_t hd = 256;
      const std::uint32_t nkv = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 2}));
      const std::uint32_t G = tc.draw(gs::sampled_from(std::vector<std::uint32_t>{1, 12, 16}));
      const std::uint32_t nh = nkv * G;
      const std::uint32_t T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 6}));
      const std::uint32_t max_n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 2100}));
      const std::uint32_t ctx = max_n + tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 500}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.5f, 2.0f, 4.0f, 8.0f}));
      std::vector<float> q(std::size_t{T} * nh * hd), gate(q.size()), k(std::size_t{ctx} * nkv * hd), v(k.size());
      for (auto& x : q) x = amp * rnd();
      for (auto& x : gate) x = 4 * rnd();
      for (auto& x : k) x = amp * rnd();
      for (auto& x : v) x = rnd();
      std::vector<std::uint32_t> visible(std::size_t{T} * max_n), params(std::size_t{T} * 16, 0);
      for (std::uint32_t t = 0; t < T; ++t) {
        const std::uint32_t n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = max_n}));
        params[t * 16 + kHaloNVisible] = n;
        for (std::uint32_t j = 0; j < n; ++j) {
          seed = seed * 1664525u + 1013904223u;
          visible[std::size_t{t} * max_n + j] = (seed >> 8) % ctx;
        }
      }
      auto qb = dev->Alloc(q.size() * 4, &error), gb = dev->Alloc(gate.size() * 4, &error),
           kb = dev->Alloc(k.size() * 4, &error), vb = dev->Alloc(v.size() * 4, &error),
           sb = dev->Alloc(visible.size() * 4, &error), pb = dev->Alloc(params.size() * 4, &error),
           o1 = dev->Alloc(q.size() * 4, &error), o2 = dev->Alloc(q.size() * 4, &error),
           sc = dev->Alloc(std::size_t{T} * nh * max_n * 4, &error);
      ASSERT_TRUE(qb && gb && kb && vb && sb && pb && o1 && o2 && sc) << error;
      ASSERT_TRUE(dev->Upload(qb, 0, q.data(), q.size() * 4, &error) &&
                  dev->Upload(gb, 0, gate.data(), gate.size() * 4, &error) &&
                  dev->Upload(kb, 0, k.data(), k.size() * 4, &error) && dev->Upload(vb, 0, v.data(), v.size() * 4, &error) &&
                  dev->Upload(sb, 0, visible.data(), visible.size() * 4, &error) &&
                  dev->Upload(pb, 0, params.data(), params.size() * 4, &error))
          << error;
      const rt::Arg ra[] = {rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(vb), rt::Arg::Buf(gb), rt::Arg::Buf(sb),
                            rt::Arg::Buf(o1), rt::Arg::Buf(sc), rt::Arg::Buf(pb), rt::Arg::Val(nh), rt::Arg::Val(nkv),
                            rt::Arg::Val(hd), rt::Arg::Val(max_n)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_attention_select_rows", &error), {nh, T}, {256}, ra, &error)) << error;
      const rt::Arg wa[] = {rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(vb), rt::Arg::Buf(gb), rt::Arg::Buf(sb),
                            rt::Arg::Buf(o2), rt::Arg::Buf(pb), rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(max_n)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_attention_wmma_rows", &error), {nkv, T}, {128}, wa, &error)) << error;
      std::vector<float> want(q.size()), got(q.size());
      ASSERT_TRUE(dev->Download(o1, 0, want.data(), want.size() * 4, &error) &&
                  dev->Download(o2, 0, got.data(), got.size() * 4, &error))
          << error;
      for (auto* b : {&qb, &gb, &kb, &vb, &sb, &pb, &o1, &o2, &sc}) dev->Free(*b);
      for (std::uint32_t t = 0; t < T; ++t) {
        double d = 0, m = 0;
        for (std::size_t i = std::size_t{t} * nh * hd; i < std::size_t{t + 1} * nh * hd; ++i) {
          ASSERT_TRUE(std::isfinite(got[i])) << i;
          d += double(got[i] - want[i]) * (got[i] - want[i]);
          m += double(want[i]) * want[i];
        }
        ASSERT_LT(std::sqrt(d / m), 5e-3) << rt::BackendName(backend) << " t " << t << " n "
                                          << params[t * 16 + kHaloNVisible] << " G " << G << " nkv " << nkv;
      }
    }, {.test_cases = 40});
  }
}
