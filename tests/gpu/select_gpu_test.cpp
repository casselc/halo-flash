// Device-side MoE routing and QSA block selection against the reference's
// rules, at production sizes and with heavy ties (Hegel properties, both
// backends). Values come from coarse grids, so two scores are either equal
// or far apart and CPU/GPU rounding cannot reorder them; the tie rule is
// then checked exactly.

#include <bit>
#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <tuple>
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

namespace {
void QsaSelectionCheck(const char* kernel, std::uint32_t block) {
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
    const auto kid = dev.Kernel(kernel, &error);
    const rt::Arg args[] = {rt::Arg::Buf(sb), rt::Arg::Buf(vb), rt::Arg::Buf(pb), rt::Arg::Val(k), rt::Arg::Val(ratio)};
    ASSERT_TRUE(dev.Launch(kid, {1}, {block}, args, &error)) << error;
    dev.Free(pb);
    std::vector<std::uint32_t> got(want.size() + 1);
    ASSERT_TRUE(dev.Download(vb, 0, got.data(), got.size() * 4, &error)) << error;
    dev.Free(sb);
    dev.Free(vb);
    ASSERT_EQ(got[want.size()], 0xEEEEEEEEu) << "wrote past the visible list";
    got.pop_back();
    ASSERT_EQ(got, want) << kernel << " n=" << n << " k=" << k << " ratio=" << ratio << " levels=" << levels;
  }, {.test_cases = 150});
}
}  // namespace

TEST(GpuSelect, QsaSelectionMatchesReferenceRule) { QsaSelectionCheck("halo_qsa_select", 1024); }

// The 256-thread restructure (wave digit search, ballot compaction) must give the same lists.
TEST(GpuSelect, QsaSelection256MatchesReferenceRule) { QsaSelectionCheck("halo_qsa_select256", 256); }

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

// P4: QSA block scores of a chunk as one bf16x3 WMMA GEMM
// (halo_qsa_scores_wmma) against the fp32 per-token kernel
// (halo_qsa_scores_rows). Bound fixed in advance: relu is 1-Lipschitz, so
// |d score| <= sum over heads of (2^-15 + dim 2^-23) * sum_i |q_i k_i| (the
// hi/lo split as in the router GEMM, plus fp32 summation either side).
// Ragged token counts and block counts.
TEST(GpuSelect, WmmaBlockScoresMatchFp32Kernel) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      constexpr std::uint32_t heads = 4, dim = 128;
      const std::uint32_t T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 70}));
      const std::uint32_t cols = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 700}));
      const std::uint32_t stride = cols + tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 9}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.25f, 2.0f, 16.0f}));
      std::vector<float> q(std::size_t{T} * heads * dim), kb(std::size_t{cols} * dim);
      for (auto& x : q) x = amp * rnd();
      for (auto& x : kb) x = rnd();
      std::vector<std::uint32_t> params(std::size_t{T} * 16, 0);
      for (std::uint32_t t = 0; t < T; ++t) params[t * 16 + kHaloComplete] = cols;
      auto qb = dev->Alloc(q.size() * 4, &error), bb = dev->Alloc(kb.size() * 4, &error), pb = dev->Alloc(params.size() * 4, &error),
           s1 = dev->Alloc(std::size_t{T} * stride * 4, &error), s2 = dev->Alloc(std::size_t{T} * stride * 4, &error);
      ASSERT_TRUE(qb && bb && pb && s1 && s2) << error;
      ASSERT_TRUE(dev->Upload(qb, 0, q.data(), q.size() * 4, &error) && dev->Upload(bb, 0, kb.data(), kb.size() * 4, &error) &&
                  dev->Upload(pb, 0, params.data(), params.size() * 4, &error))
          << error;
      const rt::Arg ra[] = {rt::Arg::Buf(qb), rt::Arg::Buf(bb), rt::Arg::Buf(s1), rt::Arg::Buf(pb), rt::Arg::Val(heads),
                            rt::Arg::Val(dim), rt::Arg::Val(stride)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_qsa_scores_rows", &error), {(cols + 255) / 256, T}, {256}, ra, &error)) << error;
      const rt::Arg wa[] = {rt::Arg::Buf(qb), rt::Arg::Buf(bb), rt::Arg::Buf(s2), rt::Arg::Val(T), rt::Arg::Val(cols),
                            rt::Arg::Val(dim), rt::Arg::Val(stride)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_qsa_scores_wmma", &error), {(cols + 63) / 64, (T + 15) / 16}, {128}, wa,
                              &error))
          << error;
      std::vector<float> want(std::size_t{T} * stride), got(want.size());
      ASSERT_TRUE(dev->Download(s1, 0, want.data(), want.size() * 4, &error) &&
                  dev->Download(s2, 0, got.data(), got.size() * 4, &error))
          << error;
      for (auto* b : {&qb, &bb, &pb, &s1, &s2}) dev->Free(*b);
      for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t b = 0; b < cols; ++b) {
          double bound = 0;
          for (std::uint32_t h = 0; h < heads; ++h) {
            double mag = 0;
            for (std::uint32_t i = 0; i < dim; ++i)
              mag += std::fabs(double(q[(std::size_t{t} * heads + h) * dim + i]) * kb[std::size_t{b} * dim + i]);
            bound += (0x1p-15 + dim * 0x1p-23) * mag;
          }
          ASSERT_NEAR(got[std::size_t{t} * stride + b], want[std::size_t{t} * stride + b], bound + 1e-30)
              << rt::BackendName(backend) << " t " << t << " b " << b << " T " << T << " cols " << cols;
        }
    }, {.test_cases = 40});
  }
}

// P4: QSA-selected chunk attention, 4 tokens per block over the union of
// their selections (halo_qsa_union4 + halo_attention_wmma_qsel), against the
// decode kernel's arithmetic over each token's own list
// (halo_attention_select_rows). Lists in halo_qsa_select_rows' format:
// k selected complete blocks in block order, expanded to `ratio`
// positions, then the incomplete tail; every complete block when under
// budget. Consecutive tokens at pos0 + t, budgets from 1 block, tokens on
// both sides of the budget, ragged 4-token groups. Bound fixed in advance
// as for the other WMMA attention kernels: relative L2 < 5e-3 per token.
TEST(GpuSelect, QsaUnionAttentionMatchesSelectKernel) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      constexpr std::uint32_t hd = 256, ratio = 4;
      const std::uint32_t nkv = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 2}));
      const std::uint32_t G = tc.draw(gs::sampled_from(std::vector<std::uint32_t>{1, 2, 12}));
      const std::uint32_t nh = nkv * G;
      const std::uint32_t T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 13}));
      const std::uint32_t k = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 40}));
      const std::uint32_t pos0 = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 700}));
      const std::uint32_t ctx = pos0 + T, max_blocks = ctx / ratio + 1, vstride = ctx, max_u = 4 * k + 4;
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.5f, 2.0f, 8.0f}));
      std::vector<float> q(std::size_t{T} * nh * hd), gate(q.size()), kc(std::size_t{ctx} * nkv * hd), vc(kc.size());
      for (auto& x : q) x = amp * rnd();
      for (auto& x : gate) x = 4 * rnd();
      for (auto& x : kc) x = amp * rnd();
      for (auto& x : vc) x = rnd();
      std::vector<std::uint32_t> visible(std::size_t{T} * vstride, 0), params(std::size_t{T} * 16, 0);
      for (std::uint32_t t = 0; t < T; ++t) {
        const std::uint32_t n_kv = pos0 + t + 1, n = n_kv / ratio;
        std::uint32_t* vis = visible.data() + std::size_t{t} * vstride;
        std::uint32_t w = 0;
        if (k >= n) {
          for (std::uint32_t p = 0; p < n_kv; ++p) vis[w++] = p;
        } else {
          std::vector<std::uint32_t> blocks(n);
          for (std::uint32_t b = 0; b < n; ++b) blocks[b] = b;
          for (std::uint32_t i = 0; i < k; ++i) {  // partial Fisher-Yates
            seed = seed * 1664525u + 1013904223u;
            std::swap(blocks[i], blocks[i + (seed >> 8) % (n - i)]);
          }
          std::sort(blocks.begin(), blocks.begin() + k);
          for (std::uint32_t i = 0; i < k; ++i)
            for (std::uint32_t r = 0; r < ratio; ++r) vis[w++] = blocks[i] * ratio + r;
          for (std::uint32_t p = n * ratio; p < n_kv; ++p) vis[w++] = p;
        }
        params[t * 16 + kHaloNkv] = n_kv;
        params[t * 16 + kHaloComplete] = n;
        params[t * 16 + kHaloNVisible] = w;
      }
      const std::uint32_t groups = (T + 3) / 4;
      auto qb = dev->Alloc(q.size() * 4, &error), gb = dev->Alloc(gate.size() * 4, &error),
           kb = dev->Alloc(kc.size() * 4, &error), vb = dev->Alloc(vc.size() * 4, &error),
           sb = dev->Alloc(visible.size() * 4, &error), pb = dev->Alloc(params.size() * 4, &error),
           o1 = dev->Alloc(q.size() * 4, &error), o2 = dev->Alloc(q.size() * 4, &error),
           sc = dev->Alloc(std::size_t{T} * nh * vstride * 4, &error), ub = dev->Alloc(std::size_t{groups} * max_u * 4, &error),
           mb = dev->Alloc(std::size_t{groups} * max_u, &error), cb = dev->Alloc(groups * 4, &error);
      ASSERT_TRUE(qb && gb && kb && vb && sb && pb && o1 && o2 && sc && ub && mb && cb) << error;
      ASSERT_TRUE(dev->Upload(qb, 0, q.data(), q.size() * 4, &error) &&
                  dev->Upload(gb, 0, gate.data(), gate.size() * 4, &error) &&
                  dev->Upload(kb, 0, kc.data(), kc.size() * 4, &error) && dev->Upload(vb, 0, vc.data(), vc.size() * 4, &error) &&
                  dev->Upload(sb, 0, visible.data(), visible.size() * 4, &error) &&
                  dev->Upload(pb, 0, params.data(), params.size() * 4, &error))
          << error;
      const rt::Arg ra[] = {rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(vb), rt::Arg::Buf(gb), rt::Arg::Buf(sb),
                            rt::Arg::Buf(o1), rt::Arg::Buf(sc), rt::Arg::Buf(pb), rt::Arg::Val(nh), rt::Arg::Val(nkv),
                            rt::Arg::Val(hd), rt::Arg::Val(vstride)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_attention_select_rows", &error), {nh, T}, {256}, ra, &error)) << error;
      const rt::Arg ua[] = {rt::Arg::Buf(sb), rt::Arg::Buf(pb), rt::Arg::Val(k), rt::Arg::Val(ratio), rt::Arg::Val(vstride),
                            rt::Arg::Val(max_blocks), rt::Arg::Val(T), rt::Arg::Buf(ub), rt::Arg::Buf(mb), rt::Arg::Buf(cb),
                            rt::Arg::Val(max_u)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_qsa_union4", &error), {groups}, {1024}, ua, &error)) << error;
      const rt::Arg wa[] = {rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(vb), rt::Arg::Buf(gb), rt::Arg::Buf(o2),
                            rt::Arg::Buf(pb), rt::Arg::Buf(ub), rt::Arg::Buf(mb), rt::Arg::Buf(cb), rt::Arg::Val(max_u),
                            rt::Arg::Val(ratio), rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(T)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_attention_wmma_qsel", &error), {nkv, groups}, {256}, wa, &error)) << error;
      // The fp16-cache variant on the same values rounded to fp16 (as the
      // engine's copy is): bit-identical to the fp32-cache kernel, which
      // rounds them the same way while staging.
      std::vector<_Float16> k16(kc.size()), v16(vc.size());
      for (std::size_t i = 0; i < kc.size(); ++i) {
        k16[i] = static_cast<_Float16>(kc[i]);
        v16[i] = static_cast<_Float16>(vc[i]);
      }
      auto k16b = dev->Alloc(k16.size() * 2, &error), v16b = dev->Alloc(v16.size() * 2, &error),
           o3 = dev->Alloc(q.size() * 4, &error);
      ASSERT_TRUE(k16b && v16b && o3 && dev->Upload(k16b, 0, k16.data(), k16.size() * 2, &error) &&
                  dev->Upload(v16b, 0, v16.data(), v16.size() * 2, &error))
          << error;
      const rt::Arg ha[] = {rt::Arg::Buf(qb), rt::Arg::Buf(k16b), rt::Arg::Buf(v16b), rt::Arg::Buf(gb), rt::Arg::Buf(o3),
                            rt::Arg::Buf(pb), rt::Arg::Buf(ub), rt::Arg::Buf(mb), rt::Arg::Buf(cb), rt::Arg::Val(max_u),
                            rt::Arg::Val(ratio), rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(T)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_attention_wmma_qsel_h", &error), {nkv, groups}, {256}, ha, &error)) << error;
      // qsel_h2 (V packed in registers, 16-byte softmax LDS accesses): bit-identical to qsel_h.
      auto o4 = dev->Alloc(q.size() * 4, &error);
      ASSERT_TRUE(o4) << error;
      const rt::Arg h2a[] = {rt::Arg::Buf(qb), rt::Arg::Buf(k16b), rt::Arg::Buf(v16b), rt::Arg::Buf(gb), rt::Arg::Buf(o4),
                             rt::Arg::Buf(pb), rt::Arg::Buf(ub), rt::Arg::Buf(mb), rt::Arg::Buf(cb), rt::Arg::Val(max_u),
                             rt::Arg::Val(ratio), rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(T)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_attention_wmma_qsel_h2", &error), {nkv, groups}, {256}, h2a, &error))
          << error;
      std::vector<float> want(q.size()), got(q.size()), got16(q.size()), got2(q.size());
      ASSERT_TRUE(dev->Download(o1, 0, want.data(), want.size() * 4, &error) &&
                  dev->Download(o2, 0, got.data(), got.size() * 4, &error) &&
                  dev->Download(o3, 0, got16.data(), got16.size() * 4, &error) &&
                  dev->Download(o4, 0, got2.data(), got2.size() * 4, &error))
          << error;
      for (auto* b : {&qb, &gb, &kb, &vb, &sb, &pb, &o1, &o2, &sc, &ub, &mb, &cb, &k16b, &v16b, &o3, &o4}) dev->Free(*b);
      for (std::size_t i = 0; i < got.size(); ++i) {
        ASSERT_EQ(std::bit_cast<std::uint32_t>(got16[i]), std::bit_cast<std::uint32_t>(got[i]))
            << "fp16-cache variant " << i << ": " << got16[i] << " vs " << got[i];
        ASSERT_EQ(std::bit_cast<std::uint32_t>(got2[i]), std::bit_cast<std::uint32_t>(got16[i]))
            << "qsel_h2 " << i << ": " << got2[i] << " vs " << got16[i];
      }
      for (std::uint32_t t = 0; t < T; ++t) {
        double d = 0, m = 0;
        for (std::size_t i = std::size_t{t} * nh * hd; i < std::size_t{t + 1} * nh * hd; ++i) {
          ASSERT_TRUE(std::isfinite(got[i])) << i;
          d += double(got[i] - want[i]) * (got[i] - want[i]);
          m += double(want[i]) * want[i];
        }
        ASSERT_LT(std::sqrt(d / m), 5e-3) << rt::BackendName(backend) << " t " << t << " pos0 " << pos0 << " k " << k
                                          << " G " << G << " nkv " << nkv << " T " << T;
      }
    }, {.test_cases = 40});
  }
}

// P4: dense chunk attention, 4 query tokens per block sharing K/V
// (halo_attention_wmma_mq), against the decode kernel's arithmetic
// (halo_attention_select_rows) on dense causal inputs: token t of a chunk at
// position pos0 + t sees positions 0 .. pos0 + t. G in {1, 2, 12}, ragged
// token counts (partial 4-token groups). Bound fixed in advance, as for the
// single-token WMMA kernel: relative L2 < 5e-3 per token.
TEST(GpuSelect, MultiTokenDenseAttentionMatchesSelectKernel) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      constexpr std::uint32_t hd = 256;
      const std::uint32_t nkv = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 2}));
      const std::uint32_t G = tc.draw(gs::sampled_from(std::vector<std::uint32_t>{1, 2, 12}));
      const std::uint32_t nh = nkv * G;
      const std::uint32_t T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 41}));
      const std::uint32_t pos0 = tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 300}));
      const std::uint32_t ctx = pos0 + T;
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.5f, 2.0f, 8.0f}));
      std::vector<float> q(std::size_t{T} * nh * hd), gate(q.size()), k(std::size_t{ctx} * nkv * hd), v(k.size());
      for (auto& x : q) x = amp * rnd();
      for (auto& x : gate) x = 4 * rnd();
      for (auto& x : k) x = amp * rnd();
      for (auto& x : v) x = rnd();
      std::vector<std::uint32_t> visible(std::size_t{T} * ctx), params(std::size_t{T} * 16, 0);
      for (std::uint32_t t = 0; t < T; ++t) {
        params[t * 16 + kHaloNVisible] = pos0 + t + 1;
        for (std::uint32_t j = 0; j <= pos0 + t; ++j) visible[std::size_t{t} * ctx + j] = j;
      }
      auto qb = dev->Alloc(q.size() * 4, &error), gb = dev->Alloc(gate.size() * 4, &error),
           kb = dev->Alloc(k.size() * 4, &error), vb = dev->Alloc(v.size() * 4, &error),
           sb = dev->Alloc(visible.size() * 4, &error), pb = dev->Alloc(params.size() * 4, &error),
           o1 = dev->Alloc(q.size() * 4, &error), o2 = dev->Alloc(q.size() * 4, &error),
           sc = dev->Alloc(std::size_t{T} * nh * ctx * 4, &error);
      ASSERT_TRUE(qb && gb && kb && vb && sb && pb && o1 && o2 && sc) << error;
      ASSERT_TRUE(dev->Upload(qb, 0, q.data(), q.size() * 4, &error) &&
                  dev->Upload(gb, 0, gate.data(), gate.size() * 4, &error) &&
                  dev->Upload(kb, 0, k.data(), k.size() * 4, &error) && dev->Upload(vb, 0, v.data(), v.size() * 4, &error) &&
                  dev->Upload(sb, 0, visible.data(), visible.size() * 4, &error) &&
                  dev->Upload(pb, 0, params.data(), params.size() * 4, &error))
          << error;
      const rt::Arg ra[] = {rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(vb), rt::Arg::Buf(gb), rt::Arg::Buf(sb),
                            rt::Arg::Buf(o1), rt::Arg::Buf(sc), rt::Arg::Buf(pb), rt::Arg::Val(nh), rt::Arg::Val(nkv),
                            rt::Arg::Val(hd), rt::Arg::Val(ctx)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_attention_select_rows", &error), {nh, T}, {256}, ra, &error)) << error;
      const rt::Arg ma[] = {rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(vb), rt::Arg::Buf(gb), rt::Arg::Buf(o2),
                            rt::Arg::Buf(pb), rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(T)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_attention_wmma_mq", &error), {nkv, (T + 3) / 4}, {256}, ma, &error)) << error;
      // halo_attention_fa reads the fp16 K/V copy (halo_f16_rows), as prefill keeps it.
      auto k16 = dev->Alloc(k.size() * 2, &error), v16 = dev->Alloc(v.size() * 2, &error), o3 = dev->Alloc(q.size() * 4, &error);
      ASSERT_TRUE(k16 && v16 && o3) << error;
      for (const auto& [src, dst] : {std::pair{&kb, &k16}, std::pair{&vb, &v16}}) {
        const rt::Arg ha[] = {rt::Arg::Buf(*src), rt::Arg::Buf(*dst), rt::Arg::Val(static_cast<std::uint32_t>(k.size()))};
        ASSERT_TRUE(dev->Launch(dev->Kernel("halo_f16_rows", &error), {static_cast<std::uint32_t>((k.size() + 255) / 256)},
                                {256}, ha, &error))
            << error;
      }
      const rt::Arg fa[] = {rt::Arg::Buf(qb), rt::Arg::Buf(k16), rt::Arg::Buf(v16), rt::Arg::Buf(gb), rt::Arg::Buf(o3),
                            rt::Arg::Buf(pb), rt::Arg::Val(nh), rt::Arg::Val(nkv), rt::Arg::Val(T)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_attention_fa", &error), {(T + 15) / 16, nkv * ((G + 3) / 4)}, {256}, fa,
                              &error))
          << error;
      std::vector<float> want(q.size()), got(q.size()), got_fa(q.size());
      ASSERT_TRUE(dev->Download(o1, 0, want.data(), want.size() * 4, &error) &&
                  dev->Download(o2, 0, got.data(), got.size() * 4, &error) &&
                  dev->Download(o3, 0, got_fa.data(), got_fa.size() * 4, &error))
          << error;
      for (auto* b : {&qb, &gb, &kb, &vb, &sb, &pb, &o1, &o2, &sc, &k16, &v16, &o3}) dev->Free(*b);
      for (std::uint32_t t = 0; t < T; ++t) {  // the flash kernel, same bound
        double d = 0, m = 0;
        for (std::size_t i = std::size_t{t} * nh * hd; i < std::size_t{t + 1} * nh * hd; ++i) {
          ASSERT_TRUE(std::isfinite(got_fa[i])) << "fa " << i;
          d += double(got_fa[i] - want[i]) * (got_fa[i] - want[i]);
          m += double(want[i]) * want[i];
        }
        ASSERT_LT(std::sqrt(d / m), 5e-3) << rt::BackendName(backend) << " fa t " << t << " T " << T << " pos0 " << pos0
                                          << " G " << G << " nkv " << nkv;
      }
      for (std::uint32_t t = 0; t < T; ++t) {
        double d = 0, m = 0;
        for (std::size_t i = std::size_t{t} * nh * hd; i < std::size_t{t + 1} * nh * hd; ++i) {
          ASSERT_TRUE(std::isfinite(got[i])) << i;
          d += double(got[i] - want[i]) * (got[i] - want[i]);
          m += double(want[i]) * want[i];
        }
        ASSERT_LT(std::sqrt(d / m), 5e-3) << rt::BackendName(backend) << " t " << t << " T " << T << " pos0 " << pos0
                                          << " G " << G << " nkv " << nkv;
      }
    }, {.test_cases = 40});
  }
}

// Decode: the batched-load attention (halo_attention_select_b) against the serial kernel
// (halo_attention_select): the same arithmetic in the same order, so outputs and scores must match bit for bit.
// Head dim 32 .. 256 (multiples of 32), G 1 .. 16 query heads per KV head, 1 .. 3000 visible positions (one to
// three 1024-position tiles, ragged 16-position batches), repeated and random positions.
TEST(GpuSelect, BatchedDecodeAttentionIsBitIdentical) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t hd = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
      const std::uint32_t nkv = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3}));
      const std::uint32_t G = tc.draw(gs::sampled_from(std::vector<std::uint32_t>{1, 2, 12, 16}));
      const std::uint32_t nh = nkv * G;
      const std::uint32_t max_n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3000}));
      const std::uint32_t n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = max_n}));
      const std::uint32_t ctx = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4000}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.5f, 2.0f, 8.0f, 30.0f}));
      std::vector<float> q(std::size_t{nh} * hd), gate(q.size()), k(std::size_t{ctx} * nkv * hd), v(k.size());
      for (auto& x : q) x = amp * rnd();
      for (auto& x : gate) x = 4 * rnd();
      for (auto& x : k) x = amp * rnd();
      for (auto& x : v) x = rnd();
      std::vector<std::uint32_t> visible(max_n), params(16, 0);
      params[kHaloNVisible] = n;
      for (std::uint32_t j = 0; j < n; ++j) {
        seed = seed * 1664525u + 1013904223u;
        visible[j] = (seed >> 8) % ctx;
      }
      auto qb = dev->Alloc(q.size() * 4, &error), gb = dev->Alloc(gate.size() * 4, &error),
           kb = dev->Alloc(k.size() * 4, &error), vb = dev->Alloc(v.size() * 4, &error),
           sb = dev->Alloc(visible.size() * 4, &error), pb = dev->Alloc(params.size() * 4, &error),
           o1 = dev->Alloc(q.size() * 4, &error), o2 = dev->Alloc(q.size() * 4, &error),
           s1 = dev->Alloc(std::size_t{nh} * max_n * 4, &error), s2 = dev->Alloc(std::size_t{nh} * max_n * 4, &error);
      ASSERT_TRUE(qb && gb && kb && vb && sb && pb && o1 && o2 && s1 && s2) << error;
      ASSERT_TRUE(dev->Upload(qb, 0, q.data(), q.size() * 4, &error) &&
                  dev->Upload(gb, 0, gate.data(), gate.size() * 4, &error) &&
                  dev->Upload(kb, 0, k.data(), k.size() * 4, &error) && dev->Upload(vb, 0, v.data(), v.size() * 4, &error) &&
                  dev->Upload(sb, 0, visible.data(), visible.size() * 4, &error) &&
                  dev->Upload(pb, 0, params.data(), params.size() * 4, &error))
          << error;
      for (auto [name, o, s] : {std::tuple{"halo_attention_select", &o1, &s1}, std::tuple{"halo_attention_select_b", &o2, &s2}}) {
        const rt::Arg ra[] = {rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(vb), rt::Arg::Buf(gb), rt::Arg::Buf(sb),
                              rt::Arg::Buf(*o), rt::Arg::Buf(*s), rt::Arg::Buf(pb), rt::Arg::Val(nh), rt::Arg::Val(nkv),
                              rt::Arg::Val(hd), rt::Arg::Val(max_n)};
        ASSERT_TRUE(dev->Launch(dev->Kernel(name, &error), {nh}, {256}, ra, &error)) << error;
      }
      std::vector<float> want(q.size()), got(q.size()), sw(std::size_t{nh} * max_n), sg(sw.size());
      ASSERT_TRUE(dev->Download(o1, 0, want.data(), want.size() * 4, &error) &&
                  dev->Download(o2, 0, got.data(), got.size() * 4, &error) &&
                  dev->Download(s1, 0, sw.data(), sw.size() * 4, &error) &&
                  dev->Download(s2, 0, sg.data(), sg.size() * 4, &error))
          << error;
      for (auto* b : {&qb, &gb, &kb, &vb, &sb, &pb, &o1, &o2, &s1, &s2}) dev->Free(*b);
      for (std::size_t i = 0; i < want.size(); ++i)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(got[i]), std::bit_cast<std::uint32_t>(want[i]))
            << rt::BackendName(backend) << " out " << i << " n " << n << " hd " << hd << " G " << G;
      for (std::uint32_t h = 0; h < nh; ++h)
        for (std::uint32_t j = 0; j < n; ++j)
          ASSERT_EQ(std::bit_cast<std::uint32_t>(sg[std::size_t{h} * max_n + j]),
                    std::bit_cast<std::uint32_t>(sw[std::size_t{h} * max_n + j]))
              << rt::BackendName(backend) << " score h " << h << " j " << j;
    }, {.test_cases = 60});
  }
}

// Stream C, fp16-only KV: the fp16-cache kernels (halo_attention_select_h, _bh, _rows_h) against the fp32 serial
// kernel run on the same cache widened to fp32 (the widening is exact), bit for bit; and halo_copy_at_h rounds
// as the host's fp16 conversion (round to nearest even), the rounding halo_f16_rows applies.
TEST(GpuSelect, Fp16KvAttentionMatchesFp32OnRoundedCache) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      const std::uint32_t hd = 32 * tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 8}));
      const std::uint32_t nkv = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3}));
      const std::uint32_t G = tc.draw(gs::sampled_from(std::vector<std::uint32_t>{1, 2, 12, 16}));
      const std::uint32_t nh = nkv * G;
      const std::uint32_t max_n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 3000}));
      const std::uint32_t n = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = max_n}));
      const std::uint32_t ctx = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 4000}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.5f, 2.0f, 8.0f, 30.0f, 3e4f}));
      std::vector<float> q(std::size_t{nh} * hd), gate(q.size()), k(std::size_t{ctx} * nkv * hd), v(k.size());
      for (auto& x : q) x = rnd();
      for (auto& x : gate) x = 4 * rnd();
      for (auto& x : k) x = amp * rnd();
      for (auto& x : v) x = amp * rnd();
      std::vector<_Float16> k16(k.size()), v16(v.size());
      std::vector<float> kf(k.size()), vf(v.size());
      for (std::size_t i = 0; i < k.size(); ++i) {
        k16[i] = static_cast<_Float16>(k[i]);
        v16[i] = static_cast<_Float16>(v[i]);
        kf[i] = static_cast<float>(k16[i]);
        vf[i] = static_cast<float>(v16[i]);
      }
      std::vector<std::uint32_t> visible(max_n), params(16, 0);
      params[kHaloNVisible] = n;
      for (std::uint32_t j = 0; j < n; ++j) {
        seed = seed * 1664525u + 1013904223u;
        visible[j] = (seed >> 8) % ctx;
      }
      const std::size_t nout = q.size(), nsc = std::size_t{nh} * max_n;
      auto qb = dev->Alloc(q.size() * 4, &error), gb = dev->Alloc(gate.size() * 4, &error),
           kb = dev->Alloc(k.size() * 4, &error), vb = dev->Alloc(v.size() * 4, &error),
           kh = dev->Alloc(k.size() * 2 + 4, &error), vh = dev->Alloc(v.size() * 2 + 4, &error),
           sb = dev->Alloc(visible.size() * 4, &error), pb = dev->Alloc(params.size() * 4, &error),
           o = dev->Alloc(nout * 4 * 4, &error), s = dev->Alloc(nsc * 4 * 4, &error);
      ASSERT_TRUE(qb && gb && kb && vb && kh && vh && sb && pb && o && s) << error;
      ASSERT_TRUE(dev->Upload(qb, 0, q.data(), q.size() * 4, &error) &&
                  dev->Upload(gb, 0, gate.data(), gate.size() * 4, &error) &&
                  dev->Upload(kb, 0, kf.data(), kf.size() * 4, &error) &&
                  dev->Upload(vb, 0, vf.data(), vf.size() * 4, &error) &&
                  dev->Upload(sb, 0, visible.data(), visible.size() * 4, &error) &&
                  dev->Upload(pb, 0, params.data(), params.size() * 4, &error))
          << error;
      // The fp16 cache written by halo_copy_at_h, one row (slot = position) at a time from the fp32 values.
      {
        auto rowb = dev->Alloc(std::size_t{nkv} * hd * 4, &error), pp = dev->Alloc(64, &error);
        ASSERT_TRUE(rowb && pp) << error;
        const std::uint32_t rowlen = nkv * hd;
        for (std::uint32_t t = 0; t < ctx; t += std::max<std::uint32_t>(1, ctx / 7)) {
          std::uint32_t p[16] = {};
          p[kHaloPosition] = t;
          ASSERT_TRUE(dev->Upload(pp, 0, p, sizeof(p), &error)) << error;
          for (auto [src, dst] : {std::pair{&k, &kh}, std::pair{&v, &vh}}) {
            ASSERT_TRUE(dev->Upload(rowb, 0, src->data() + std::size_t{t} * rowlen, rowlen * 4ull, &error)) << error;
            const rt::Arg ca[] = {rt::Arg::Buf(*dst), rt::Arg::Buf(rowb), rt::Arg::Buf(pp), rt::Arg::Val(kHaloPosition),
                                  rt::Arg::Val(rowlen), rt::Arg::Val(rowlen)};
            ASSERT_TRUE(dev->Launch(dev->Kernel("halo_copy_at_h", &error), {(rowlen + 255) / 256}, {256}, ca, &error))
                << error;
          }
          std::vector<_Float16> back(rowlen);
          ASSERT_TRUE(dev->Download(kh, std::size_t{t} * rowlen * 2, back.data(), rowlen * 2ull, &error)) << error;
          ASSERT_EQ(std::memcmp(back.data(), k16.data() + std::size_t{t} * rowlen, rowlen * 2ull), 0) << "copy_at_h row " << t;
        }
        dev->Free(rowb);
        dev->Free(pp);
      }
      // The whole fp16 cache, uploaded (the rows not appended above too).
      ASSERT_TRUE(dev->Upload(kh, 0, k16.data(), k16.size() * 2, &error) &&
                  dev->Upload(vh, 0, v16.data(), v16.size() * 2, &error))
          << error;
      const char* names[4] = {"halo_attention_select", "halo_attention_select_h", "halo_attention_select_bh",
                              "halo_attention_select_rows_h"};
      for (int a = 0; a < 4; ++a) {
        const bool half = a > 0;
        const rt::Arg ra[] = {rt::Arg::Buf(qb), rt::Arg::Buf(half ? kh : kb), rt::Arg::Buf(half ? vh : vb),
                              rt::Arg::Buf(gb), rt::Arg::Buf(sb), rt::Arg::Buf(o, a * nout * 4),
                              rt::Arg::Buf(s, a * nsc * 4), rt::Arg::Buf(pb), rt::Arg::Val(nh), rt::Arg::Val(nkv),
                              rt::Arg::Val(hd), rt::Arg::Val(max_n)};
        ASSERT_TRUE(dev->Launch(dev->Kernel(names[a], &error), {nh, 1}, {256}, ra, &error)) << names[a] << error;
      }
      std::vector<float> out(nout * 4), sc(nsc * 4);
      ASSERT_TRUE(dev->Download(o, 0, out.data(), out.size() * 4, &error) &&
                  dev->Download(s, 0, sc.data(), sc.size() * 4, &error))
          << error;
      for (auto* b : {&qb, &gb, &kb, &vb, &kh, &vh, &sb, &pb, &o, &s}) dev->Free(*b);
      for (int a = 1; a < 4; ++a) {
        for (std::size_t i = 0; i < nout; ++i)
          ASSERT_EQ(std::bit_cast<std::uint32_t>(out[a * nout + i]), std::bit_cast<std::uint32_t>(out[i]))
              << rt::BackendName(backend) << " " << names[a] << " out " << i << " n " << n << " hd " << hd << " G " << G;
        for (std::uint32_t h = 0; h < nh; ++h)
          for (std::uint32_t j = 0; j < n; ++j)
            ASSERT_EQ(std::bit_cast<std::uint32_t>(sc[a * nsc + std::size_t{h} * max_n + j]),
                      std::bit_cast<std::uint32_t>(sc[std::size_t{h} * max_n + j]))
                << rt::BackendName(backend) << " " << names[a] << " score h " << h << " j " << j;
      }
    }, {.test_cases = 60});
  }
}

// Stream C: halo_qsa_scores_wmma_b (block-key tile split once per block in LDS, q pre-split by
// halo_split_bf16_rows) against halo_qsa_scores_wmma, bit for bit: the same WMMAs per score in the same order.
// Up to 512 tokens (a sub-chunk) and 5000 blocks, ragged both ways, rows past the 512-row block boundary,
// unused stride columns untouched.
TEST(GpuSelect, WmmaBlockScoresBIsBitIdentical) {
  std::string error;
  for (auto backend : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto dev = rt::Device::Create(backend, &error);
    if (!dev) continue;
    ASSERT_TRUE(dev->LoadModule(kernels::CoreHsaco(), &error)) << error;
    hegel::test([&](hegel::TestCase& tc) {
      constexpr std::uint32_t heads = 4, dim = 128;
      const std::uint32_t T = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 512}));
      const std::uint32_t cols = tc.draw(gs::integers<std::uint32_t>({.min_value = 1, .max_value = 5000}));
      const std::uint32_t stride = cols + tc.draw(gs::integers<std::uint32_t>({.min_value = 0, .max_value = 9}));
      std::uint32_t seed = tc.draw(gs::integers<std::uint32_t>());
      const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
      const float amp = tc.draw(gs::sampled_from(std::vector<float>{0.25f, 2.0f, 16.0f, 3e4f}));
      std::vector<float> q(std::size_t{T} * heads * dim), kb(std::size_t{cols} * dim);
      for (auto& x : q) x = amp * rnd();
      for (auto& x : kb) x = rnd();
      const std::size_t nq = q.size();
      auto qb = dev->Alloc(nq * 4, &error), bb = dev->Alloc(kb.size() * 4, &error), qh = dev->Alloc(nq * 2, &error),
           ql = dev->Alloc(nq * 2, &error), s1 = dev->Alloc(std::size_t{T} * stride * 4, &error),
           s2 = dev->Alloc(std::size_t{T} * stride * 4, &error);
      ASSERT_TRUE(qb && bb && qh && ql && s1 && s2) << error;
      ASSERT_TRUE(dev->Upload(qb, 0, q.data(), nq * 4, &error) && dev->Upload(bb, 0, kb.data(), kb.size() * 4, &error) &&
                  dev->Fill(s1, 0x7F, &error) && dev->Fill(s2, 0x7F, &error))
          << error;
      const rt::Arg wa[] = {rt::Arg::Buf(qb), rt::Arg::Buf(bb), rt::Arg::Buf(s1), rt::Arg::Val(T), rt::Arg::Val(cols),
                            rt::Arg::Val(dim), rt::Arg::Val(stride)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_qsa_scores_wmma", &error), {(cols + 63) / 64, (T + 15) / 16}, {128}, wa,
                              &error))
          << error;
      const rt::Arg sa[] = {rt::Arg::Buf(qb), rt::Arg::Buf(qh), rt::Arg::Buf(ql), rt::Arg::Val(static_cast<std::uint32_t>(nq))};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_split_bf16_rows", &error), {static_cast<std::uint32_t>((nq + 255) / 256)},
                              {256}, sa, &error))
          << error;
      const rt::Arg ba[] = {rt::Arg::Buf(qh), rt::Arg::Buf(ql), rt::Arg::Buf(bb), rt::Arg::Buf(s2), rt::Arg::Val(T),
                            rt::Arg::Val(cols), rt::Arg::Val(stride)};
      ASSERT_TRUE(dev->Launch(dev->Kernel("halo_qsa_scores_wmma_b", &error), {(cols + 63) / 64, (T * heads + 511) / 512},
                              {256}, ba, &error))
          << error;
      std::vector<std::uint32_t> want(std::size_t{T} * stride), got(want.size());
      ASSERT_TRUE(dev->Download(s1, 0, want.data(), want.size() * 4, &error) &&
                  dev->Download(s2, 0, got.data(), got.size() * 4, &error))
          << error;
      for (auto* b : {&qb, &bb, &qh, &ql, &s1, &s2}) dev->Free(*b);
      for (std::size_t i = 0; i < want.size(); ++i)
        ASSERT_EQ(got[i], want[i]) << rt::BackendName(backend) << " t " << i / stride << " b " << i % stride << " T " << T
                                   << " cols " << cols;
    }, {.test_cases = 40});
  }
}
