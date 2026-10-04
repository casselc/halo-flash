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
