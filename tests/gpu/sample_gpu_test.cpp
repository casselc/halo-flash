// Device-side sampling (halo_sample) against core/sampling's SamplerState,
// the CPU oracle with gufo's semantics. Hegel property, both backends:
//  - the drawn token and the advanced RNG state equal the CPU's;
//  - a different token is tolerated only when a perturbation of 1e-9 in u
//    makes the CPU agree (rounding at a CDF boundary), and such cases must
//    be rare;
//  - two device runs are bit-identical (integer mass sums).

#include <hegel/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <vector>

#include "core/sampling.hpp"
#include "kernels/core_hsaco.hpp"
#include "kernels/sample_params.h"
#include "rt/device.hpp"

namespace gs = hegel::generators;
using namespace halo;
using sampling::SamplerState;
using sampling::SamplingConfig;
using sampling::TokenId;

namespace {

std::vector<std::unique_ptr<rt::Device>> Devices() {
  std::vector<std::unique_ptr<rt::Device>> out;
  for (auto b : {rt::Backend::kHip, rt::Backend::kHrx}) {
    std::string error;
    if (auto d = rt::Device::Create(b, &error); d && d->LoadModule(kernels::CoreHsaco(), &error)) out.push_back(std::move(d));
  }
  return out;
}

HaloSampleOut RunDevice(rt::Device& dev, const std::vector<float>& logits, const SamplingConfig& c,
                        std::span<const sampling::TokenPenalty> penalties, std::uint64_t rng) {
  std::string error;
  const auto V = static_cast<std::uint32_t>(logits.size());
  HaloSampleParams p{c.temperature,       c.top_p,           c.min_p, c.repeat_penalty,
                     c.frequency_penalty, c.presence_penalty, static_cast<std::uint32_t>(std::max(c.top_k, 0)),
                     static_cast<std::uint32_t>(c.min_keep), static_cast<std::uint32_t>(penalties.size()), V, rng};
  std::vector<HaloPenalty> pen;
  for (const auto& t : penalties) pen.push_back({t.token, t.generated_count, t.repeated, 0});
  auto lb = dev.Alloc(V * 4ull, &error), pb = dev.Alloc(sizeof(p), &error),
       qb = dev.Alloc(std::max<std::size_t>(pen.size(), 1) * sizeof(HaloPenalty), &error),
       kb = dev.Alloc((V + pen.size()) * 8ull, &error), rb = dev.Alloc((V + pen.size()) * 4ull, &error),
       wb = dev.Alloc((V + pen.size()) * 8ull, &error), ob = dev.Alloc(sizeof(HaloSampleOut), &error);
  EXPECT_TRUE(lb && pb && qb && kb && rb && wb && ob) << error;
  EXPECT_TRUE(dev.Upload(lb, 0, logits.data(), V * 4ull, &error)) << error;
  EXPECT_TRUE(dev.Upload(pb, 0, &p, sizeof(p), &error)) << error;
  if (!pen.empty()) EXPECT_TRUE(dev.Upload(qb, 0, pen.data(), pen.size() * sizeof(HaloPenalty), &error)) << error;
  const auto k = dev.Kernel("halo_sample", &error);
  const rt::Arg args[] = {rt::Arg::Buf(lb), rt::Arg::Buf(pb), rt::Arg::Buf(qb), rt::Arg::Buf(kb),
                          rt::Arg::Buf(rb), rt::Arg::Buf(wb), rt::Arg::Buf(ob)};
  EXPECT_TRUE(dev.Launch(k, {1}, {1024}, args, &error)) << error;
  HaloSampleOut out{};
  EXPECT_TRUE(dev.Download(ob, 0, &out, sizeof(out), &error)) << error;
  for (auto* b : {&lb, &pb, &qb, &kb, &rb, &wb, &ob}) dev.Free(*b);
  return out;
}

// The CPU's token for this state with the draw replaced by `u` (the draw's
// inverse CDF over the distribution's sorted entries, as Sample does).
TokenId CpuTokenAt(const SamplerState& s, const std::vector<float>& logits, double u) {
  const auto d = s.Distribution(logits);
  const auto e = d.entries();
  if (e.size() == 1) return e.front().token;
  double x = u;
  for (const auto& p : e)
    if ((x -= p.value) < 0.0) return p.token;
  return e.back().token;
}

}  // namespace

TEST(GpuSample, DrawsMatchTheCpuSampler) {
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  int tolerated = 0, cases = 0;
  hegel::test([&](hegel::TestCase& tc) {
    auto& dev = *devices[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = devices.size() - 1}))];
    const auto V = tc.draw(gs::sampled_from(std::vector<std::size_t>{1, 2, 3, 7, 64, 1000, 5000, 248320}));
    const auto seed = tc.draw(gs::integers<std::uint32_t>());
    const auto levels = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 40}));
    std::vector<float> logits(V);
    std::uint32_t x = seed;
    for (auto& l : logits) {
      x = x * 1664525u + 1013904223u;
      l = (x >> 28) == 0 && V > 1 ? -std::numeric_limits<float>::infinity()
                                  : 0.25f * static_cast<float>(static_cast<int>((x >> 8) % levels) - levels / 2);
    }
    if (tc.draw(gs::booleans())) logits[x % V] = 9.0f;  // a clear winner sometimes
    SamplingConfig c;
    c.temperature = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.2f, 0.7f, 1.0f, 1.8f}));
    c.top_k = tc.draw(gs::sampled_from(std::vector<std::int32_t>{0, 1, 2, 5, 40, 1000}));
    c.top_p = tc.draw(gs::sampled_from(std::vector<float>{1.0f, 0.95f, 0.8f, 0.5f, 0.1f, 0.0f}));
    c.min_p = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.05f, 0.3f, 1.0f}));
    c.min_keep = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 4}));
    if (tc.draw(gs::booleans())) {  // SamplerState's linear path: token-order draws
      c.top_k = 0;
      c.top_p = 1.0f;
      c.min_keep = std::min<std::size_t>(c.min_keep, 1);
    }
    c.repeat_penalty = tc.draw(gs::sampled_from(std::vector<float>{1.0f, 1.3f, 0.7f}));
    c.repeat_last_n = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 32}));
    c.frequency_penalty = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.4f}));
    c.presence_penalty = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.6f, -0.3f}));
    c.seed = 1;
    SamplerState cpu(c);
    const auto history = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 40}));
    for (std::size_t i = 0; i < history; ++i) {
      x = x * 1664525u + 1013904223u;
      cpu.Accept(static_cast<TokenId>((x >> 8) % V));
    }
    const std::uint64_t rng = tc.draw(gs::integers<std::uint64_t>());
    cpu.SetRngState(rng);
    const SamplerState before = cpu;

    const auto got = RunDevice(dev, logits, c, cpu.penalties(), rng);
    const auto again = RunDevice(dev, logits, c, cpu.penalties(), rng);
    ASSERT_EQ(std::memcmp(&got, &again, sizeof(got)), 0) << "device sampling is not deterministic";
    if (std::none_of(logits.begin(), logits.end(), [](float l) { return std::isfinite(l); })) {
      ASSERT_EQ(got.token, 0xFFFFFFFFu) << "no finite logit must be reported, not sampled";
      ASSERT_THROW((void)cpu.Sample(logits), std::runtime_error);
      return;
    }
    const TokenId want = cpu.Sample(logits);
    ASSERT_EQ(got.rng, cpu.rng_state()) << "RNG draws diverge (V=" << V << ", T=" << c.temperature << ")";
    ++cases;
    if (got.token == want) return;
    // Rounding at a CDF boundary: a perturbation of u must make the CPU
    // choose the device's token.
    auto probe = before;
    const double u = sampling::Uniform(probe.mutable_rng_state());
    bool explained = false;
    for (double du : {-1e-9, 1e-9})
      explained = explained || CpuTokenAt(before, logits, std::clamp(u + du, 0.0, 1.0 - 1e-12)) == got.token;
    ASSERT_TRUE(explained) << "token " << got.token << " vs CPU " << want << " (V=" << V << ", T=" << c.temperature
                           << ", top_k=" << c.top_k << ", top_p=" << c.top_p << ", min_p=" << c.min_p << ")";
    ++tolerated;
  }, {.test_cases = 400});
  std::printf("%d of %d draws needed the u boundary tolerance\n", tolerated, cases);
  EXPECT_LE(tolerated * 100, cases) << tolerated << " of " << cases << " draws needed boundary tolerance";
}

// Exact boundaries the random grid rarely produces: cumulative mass exactly
// at top-p, and tokens exactly at the min-p cut (ties at the maximum).
TEST(GpuSample, ExactBoundariesMatchTheCpuSampler) {
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  struct Case {
    std::vector<float> logits;
    float temperature, top_p, min_p;
    std::int32_t top_k;
  };
  const Case cases[] = {
      {{1, 1}, 1.0f, 0.5f, 0.0f, 0},          // first token's mass is exactly p
      {{1, 1, 1, 1}, 1.0f, 0.5f, 0.0f, 0},    // two of four reach p exactly
      {{2, 2, 0}, 1.0f, 0.0f, 0.0f, 3},       // top-p 0 keeps one token
      {{1, 1, 0}, 1.0f, 1.0f, 1.0f, 0},       // min-p 1 keeps the tied maxima (linear path)
      {{1, 1, 0, 1}, 0.7f, 1.0f, 1.0f, 2},    // and on the sorted path
      {{3, 3, 3}, 1.0f, 1.0f / 3, 0.0f, 0},   // mass 1/3 per token
  };
  for (auto& dev : devices)
    for (const auto& k : cases)
      for (std::uint64_t rng : {1ull, 2ull, 0x9e3779b97f4a7c15ull, 12345ull}) {
        SamplingConfig c;
        c.temperature = k.temperature;
        c.top_p = k.top_p;
        c.min_p = k.min_p;
        c.top_k = k.top_k;
        c.seed = 1;
        SamplerState cpu(c);
        cpu.SetRngState(rng);
        const auto got = RunDevice(*dev, k.logits, c, cpu.penalties(), rng);
        const TokenId want = cpu.Sample(k.logits);
        EXPECT_EQ(got.token, want) << "top_p " << k.top_p << " min_p " << k.min_p << " rng " << rng;
        EXPECT_EQ(got.rng, cpu.rng_state()) << "top_p " << k.top_p << " min_p " << k.min_p << " rng " << rng;
      }
}

// ---- Fast sampler (halo_sample_blocks + halo_sample_final + exact fallback)

namespace {

struct FastSampler {
  rt::Device& dev;
  std::uint32_t V;
  std::string error;
  rt::Buffer lb, pb, qb, bb, cb, kb, rb, wb, ob;
  std::uint32_t nb;
  explicit FastSampler(rt::Device& d, std::uint32_t vocab) : dev(d), V(vocab) {
    nb = (V + HALO_SAMPLE_BLOCK - 1) / HALO_SAMPLE_BLOCK;
    const std::size_t cap = std::max<std::size_t>(V, nb * HALO_SAMPLE_TOPK) + HALO_SAMPLE_MAX_PENALTIES;
    lb = dev.Alloc(V * 4ull, &error);
    pb = dev.Alloc(sizeof(HaloSampleParams), &error);
    qb = dev.Alloc(HALO_SAMPLE_MAX_PENALTIES * sizeof(HaloPenalty), &error);
    bb = dev.Alloc(nb * sizeof(HaloSampleBlock), &error);
    cb = dev.Alloc(nb * HALO_SAMPLE_TOPK * 8ull, &error);
    kb = dev.Alloc(cap * 8, &error);
    rb = dev.Alloc(cap * 4, &error);
    wb = dev.Alloc(cap * 8, &error);
    ob = dev.Alloc(sizeof(HaloSampleOut), &error);
  }
  ~FastSampler() {
    for (auto* b : {&lb, &pb, &qb, &bb, &cb, &kb, &rb, &wb, &ob}) dev.Free(*b);
  }
  void SetLogits(const std::vector<float>& logits) { EXPECT_TRUE(dev.Upload(lb, 0, logits.data(), V * 4ull, &error)); }
  // Returns the output and whether the exact fallback ran.
  std::pair<HaloSampleOut, bool> Draw(const SamplingConfig& c, std::span<const sampling::TokenPenalty> penalties,
                                      std::uint64_t rng) {
    HaloSampleParams p{c.temperature, c.top_p, c.min_p, c.repeat_penalty, c.frequency_penalty, c.presence_penalty,
                       static_cast<std::uint32_t>(std::max(c.top_k, 0)), static_cast<std::uint32_t>(c.min_keep),
                       static_cast<std::uint32_t>(penalties.size()), V, rng, 1, 0};
    std::vector<HaloPenalty> pen;
    for (const auto& t : penalties) pen.push_back({t.token, t.generated_count, t.repeated, 0});
    EXPECT_TRUE(dev.Upload(pb, 0, &p, sizeof(p), &error)) << error;
    if (!pen.empty()) EXPECT_TRUE(dev.Upload(qb, 0, pen.data(), pen.size() * sizeof(HaloPenalty), &error)) << error;
    const rt::Arg a1[] = {rt::Arg::Buf(lb), rt::Arg::Buf(pb), rt::Arg::Buf(qb), rt::Arg::Buf(bb), rt::Arg::Buf(cb)};
    const rt::Arg a2[] = {rt::Arg::Buf(pb), rt::Arg::Buf(bb), rt::Arg::Buf(cb), rt::Arg::Val(nb),
                          rt::Arg::Buf(kb), rt::Arg::Buf(rb), rt::Arg::Buf(wb), rt::Arg::Buf(ob)};
    const rt::Arg a3[] = {rt::Arg::Buf(lb), rt::Arg::Buf(pb), rt::Arg::Buf(qb), rt::Arg::Buf(kb),
                          rt::Arg::Buf(rb), rt::Arg::Buf(wb), rt::Arg::Buf(ob)};
    EXPECT_TRUE(dev.Launch(dev.Kernel("halo_sample_blocks", &error), {nb}, {1024}, a1, &error)) << error;
    EXPECT_TRUE(dev.Launch(dev.Kernel("halo_sample_final", &error), {1}, {1024}, a2, &error)) << error;
    HaloSampleOut mid{};
    EXPECT_TRUE(dev.Download(ob, 0, &mid, sizeof(mid), &error)) << error;
    EXPECT_TRUE(dev.Launch(dev.Kernel("halo_sample", &error), {1}, {1024}, a3, &error)) << error;
    HaloSampleOut out{};
    EXPECT_TRUE(dev.Download(ob, 0, &out, sizeof(out), &error)) << error;
    return {out, mid.token == HALO_SAMPLE_FALLBACK};
  }
};

std::vector<float> RandomLogits(std::size_t V, std::uint32_t seed, int levels, float step) {
  std::vector<float> logits(V);
  std::uint32_t x = seed;
  for (auto& l : logits) {
    x = x * 1664525u + 1013904223u;
    l = (x >> 29) == 0 && V > 1 ? -std::numeric_limits<float>::infinity()
                                : step * static_cast<float>(static_cast<int>((x >> 8) % levels) - levels / 2);
  }
  return logits;
}

SamplingConfig DrawConfig(hegel::TestCase& tc) {
  SamplingConfig c;
  c.temperature = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.5f, 1.0f, 1.5f}));
  c.top_k = tc.draw(gs::sampled_from(std::vector<std::int32_t>{0, 1, 3, 20, 200, 300}));
  c.top_p = tc.draw(gs::sampled_from(std::vector<float>{1.0f, 0.95f, 0.6f, 0.0f}));
  c.min_p = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.02f, 0.3f}));
  c.min_keep = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 3}));
  c.repeat_penalty = tc.draw(gs::sampled_from(std::vector<float>{1.0f, 1.3f}));
  c.repeat_last_n = 16;
  c.presence_penalty = tc.draw(gs::sampled_from(std::vector<float>{0.0f, 0.6f}));
  c.seed = 1;
  return c;
}

}  // namespace

// Greedy is exact; a draw is always in the CPU distribution's support and
// never the fallback marker; repeated runs agree bit for bit.
TEST(GpuFastSample, GreedyExactSupportAndDeterminism) {
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  int fallbacks = 0, draws = 0;
  hegel::test([&](hegel::TestCase& tc) {
    auto& dev = *devices[tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = devices.size() - 1}))];
    const auto V = tc.draw(gs::sampled_from(std::vector<std::uint32_t>{1, 5, 1000, 1500, 5000, 248320}));
    const auto logits = RandomLogits(V, tc.draw(gs::integers<std::uint32_t>()),
                                     tc.draw(gs::integers<int>({.min_value = 1, .max_value = 60})),
                                     tc.draw(gs::sampled_from(std::vector<float>{0.05f, 0.25f, 1.0f})));
    if (std::none_of(logits.begin(), logits.end(), [](float l) { return std::isfinite(l); })) return;
    const SamplingConfig c = DrawConfig(tc);
    SamplerState cpu(c);
    for (int i = 0; i < 8; ++i) cpu.Accept(static_cast<TokenId>((i * 7919) % V));
    FastSampler fs(dev, V);
    fs.SetLogits(logits);
    const std::uint64_t rng = tc.draw(gs::integers<std::uint64_t>());
    const auto [got, fell_back] = fs.Draw(c, cpu.penalties(), rng);
    const auto again = fs.Draw(c, cpu.penalties(), rng).first;
    ASSERT_EQ(std::memcmp(&got, &again, sizeof(got)), 0) << "not deterministic";
    ASSERT_NE(got.token, HALO_SAMPLE_FALLBACK);
    ++draws;
    fallbacks += fell_back;
    const auto d = cpu.Distribution(logits);
    if (c.temperature == 0) {
      ASSERT_EQ(got.token, d.best_token()) << "greedy differs (V=" << V << ")";
      return;
    }
    ASSERT_GT(d.probability(got.token), 0.0) << "token " << got.token << " outside the support (V=" << V
                                             << ", top_k=" << c.top_k << ", top_p=" << c.top_p << ", min_p="
                                             << c.min_p << ")";
  }, {.test_cases = 300});
  std::printf("%d of %d fast draws fell back to the exact kernel\n", fallbacks, draws);
}

// Draws follow the CPU distribution (chi-square over many RNG keys), for
// configurations on the fast path and ones that force the fallback, with
// the vocabulary spread over several blocks.
TEST(GpuFastSample, DrawsFollowTheCpuDistribution) {
  auto devices = Devices();
  if (devices.empty()) GTEST_SKIP() << "no GPU backend";
  struct Case {
    const char* name;
    float temperature, top_p, min_p;
    std::int32_t top_k;
    bool zipf = false;  // log-rank decay, like language-model logits
    bool steep = false;
  };
  const Case cases[] = {{"plain T=1", 1.0f, 1.0f, 0.0f, 0}, {"plain T=0.6", 0.6f, 1.0f, 0.0f, 0},
                        {"top-k 20", 0.8f, 1.0f, 0.0f, 20},
                        {"top-p .9 zipf (fast)", 0.8f, 0.9f, 0.0f, 0, true}, {"min-p .05 zipf (fast)", 1.0f, 1.0f, 0.05f, 0, true},
                        // Steep decay within blocks whose maxima sit below the global one:
                        // per-block masses must be rescaled to one scale before top-p.
                        {"top-p .3 steep blocks", 1.0f, 0.3f, 0.0f, 0, false, true},
                        {"top-k 40 top-p .8", 1.0f, 0.8f, 0.0f, 40}, {"top-p .9 (no top-k)", 1.0f, 0.9f, 0.0f, 0},
                        {"min-p .1", 1.0f, 1.0f, 0.1f, 0}, {"top-p .99 flat (fallback)", 3.0f, 0.99f, 0.0f, 0}};
  for (auto& dev : devices)
    for (const auto& k : cases) {
      const std::uint32_t V = 3000;
      auto logits = RandomLogits(V, 7, 24, 0.25f);
      if (k.zipf)
        for (std::uint32_t i = 0; i < V; ++i) logits[i] = -1.6f * std::log(static_cast<float>((i * 2654435761u) % V) + 1.0f);
      if (k.steep)
        for (std::uint32_t i = 0; i < V; ++i)
          logits[i] = i < 1024 ? -0.5f * static_cast<float>(i) : -2.0f - static_cast<float>(i % 1024);
      SamplingConfig c;
      c.temperature = k.temperature;
      c.top_p = k.top_p;
      c.min_p = k.min_p;
      c.top_k = k.top_k;
      c.seed = 1;
      SamplerState cpu(c);
      const auto d = cpu.Distribution(logits);
      FastSampler fs(*dev, V);
      fs.SetLogits(logits);
      constexpr int n = 4000;
      std::map<TokenId, int> counts;
      int fell_back = 0;
      std::uint64_t rng = 0x1234567;
      for (int i = 0; i < n; ++i) {
        const auto [out, fb] = fs.Draw(c, {}, rng);
        rng = out.rng;
        fell_back += fb;
        ++counts[out.token];
      }
      // Bins: tokens expected at least 5 times; the rest pooled.
      double chi2 = 0, pooled_e = 0, pooled_o = 0;
      int bins = 0;
      for (const auto& e : d.entries()) {
        const double expected = e.value * n, observed = counts.contains(e.token) ? counts[e.token] : 0;
        if (expected >= 5) {
          chi2 += (observed - expected) * (observed - expected) / expected;
          ++bins;
        } else {
          pooled_e += expected;
          pooled_o += observed;
        }
      }
      if (pooled_e >= 5) {
        chi2 += (pooled_o - pooled_e) * (pooled_o - pooled_e) / pooled_e;
        ++bins;
      }
      for (const auto& [t, cnt] : counts) ASSERT_GT(d.probability(t), 0.0) << k.name << ": token " << t;
      const double df = std::max(1, bins - 1);
      EXPECT_LT(chi2, df + 8 * std::sqrt(2 * df) + 25) << k.name << ": chi-square " << chi2 << " over " << bins << " bins";
      std::printf("%s %-26s chi2 %7.1f / %3d bins, %d of %d fell back\n", rt::BackendName(dev->backend()), k.name, chi2,
                  bins, fell_back, n);
    }
}
