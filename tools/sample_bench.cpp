#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <random>
#include <vector>
#include "kernels/core_hsaco.hpp"
#include "kernels/sample_params.h"
#include "rt/device.hpp"
using namespace halo;
int main() {
  std::string e;
  const std::uint32_t V = 248320;
  std::mt19937 g(1);
  std::normal_distribution<float> nd(0, 3);
  // Two logit profiles: i.i.d. N(0, 3) (very flat: a worst case for top-p),
  // and Zipf-like (log-rank decay, as language-model logits roughly follow).
  std::vector<float> flat(V), zipf(V);
  for (auto& l : flat) l = nd(g);
  std::vector<std::uint32_t> rank(V);
  for (std::uint32_t i = 0; i < V; ++i) rank[i] = i;
  std::shuffle(rank.begin(), rank.end(), g);
  for (std::uint32_t i = 0; i < V; ++i) zipf[i] = -1.6f * std::log(static_cast<float>(rank[i]) + 1.0f) + 0.3f * nd(g) / 3;
  struct Cfg { const char* name; float T, top_p, min_p; std::uint32_t top_k, npen; };
  const Cfg cfgs[] = {{"greedy", 0, 1, 0, 0, 0}, {"T=1 plain", 1, 1, 0, 0, 0}, {"T=0.7 top_p .95", 0.7f, 0.95f, 0, 0, 0},
                      {"T=0.7 top_k 20 top_p .8", 0.7f, 0.8f, 0, 20, 0}, {"T=0.6 top_k 20 top_p .95 min_p .05 +64 pen", 0.6f, 0.95f, 0.05f, 20, 64}};
  for (auto be : {rt::Backend::kHip, rt::Backend::kHrx}) {
    auto d = rt::Device::Create(be, &e);
    if (!d || !d->LoadModule(kernels::CoreHsaco(), &e)) continue;
    auto lb = d->Alloc(V * 4, &e), pb = d->Alloc(sizeof(HaloSampleParams), &e), qb = d->Alloc(64 * sizeof(HaloPenalty), &e),
         kb = d->Alloc((V + 64) * 8ull, &e), rb = d->Alloc((V + 64) * 4ull, &e), wb = d->Alloc((V + 64) * 8ull, &e),
         ob = d->Alloc(sizeof(HaloSampleOut), &e);
    for (const auto* logits : {&zipf, &flat}) {
    std::printf("-- %s logits\n", logits == &zipf ? "zipf-like" : "flat N(0,3)");
    (void)d->Upload(lb, 0, logits->data(), V * 4, &e);
    std::vector<HaloPenalty> pen(64);
    for (int i = 0; i < 64; ++i) pen[i] = {static_cast<std::uint32_t>(i * 997 % V), 1, 1, 0};
    (void)d->Upload(qb, 0, pen.data(), pen.size() * sizeof(HaloPenalty), &e);
    const auto k = d->Kernel("halo_sample", &e);
    for (const auto& c : cfgs) {
      HaloSampleParams p{c.T, c.top_p, c.min_p, 1.1f, 0.0f, 0.0f, c.top_k, 0, c.npen, V, 42, 0, 0};
      (void)d->Upload(pb, 0, &p, sizeof(p), &e);
      const rt::Arg args[] = {rt::Arg::Buf(lb), rt::Arg::Buf(pb), rt::Arg::Buf(qb), rt::Arg::Buf(kb), rt::Arg::Buf(rb), rt::Arg::Buf(wb), rt::Arg::Buf(ob)};
      for (int i = 0; i < 5; ++i) (void)d->Launch(k, {1}, {1024}, args, &e);
      (void)d->Synchronize(&e);
      const int N = 50;
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < N; ++i) (void)d->Launch(k, {1}, {1024}, args, &e);
      (void)d->Synchronize(&e);
      const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
      // Fast pipeline: blocks + final (+ exact kernel in fallback-only mode).
      const std::uint32_t nb = (V + HALO_SAMPLE_BLOCK - 1) / HALO_SAMPLE_BLOCK;
      auto bb = d->Alloc(nb * sizeof(HaloSampleBlock), &e), cb = d->Alloc(nb * HALO_SAMPLE_TOPK * 8ull, &e);
      HaloSampleParams pf = p;
      pf.fallback_only = 1;
      (void)d->Upload(pb, 0, &pf, sizeof(pf), &e);
      const auto kb1 = d->Kernel("halo_sample_blocks", &e), kb2 = d->Kernel("halo_sample_final", &e);
      const rt::Arg a1[] = {rt::Arg::Buf(lb), rt::Arg::Buf(pb), rt::Arg::Buf(qb), rt::Arg::Buf(bb), rt::Arg::Buf(cb)};
      const rt::Arg a2[] = {rt::Arg::Buf(pb), rt::Arg::Buf(bb), rt::Arg::Buf(cb), rt::Arg::Val(nb),
                            rt::Arg::Buf(kb), rt::Arg::Buf(rb), rt::Arg::Buf(wb), rt::Arg::Buf(ob)};
      const auto fast = [&] {
        (void)d->Launch(kb1, {nb}, {1024}, a1, &e);
        (void)d->Launch(kb2, {1}, {1024}, a2, &e);
        (void)d->Launch(k, {1}, {1024}, args, &e);
      };
      for (int i = 0; i < 5; ++i) fast();
      (void)d->Synchronize(&e);
      auto t1 = std::chrono::steady_clock::now();
      for (int i = 0; i < N; ++i) fast();
      (void)d->Synchronize(&e);
      const double us_fast = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / N;
      const auto time = [&](auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();
        (void)d->Synchronize(&e);
        auto t = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) fn();
        (void)d->Synchronize(&e);
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count() / N;
      };
      const double t_blocks = time([&] { (void)d->Launch(kb1, {nb}, {1024}, a1, &e); });
      const double t_final = time([&] { (void)d->Launch(kb2, {1}, {1024}, a2, &e); });
      const double t_gate = time([&] { (void)d->Launch(k, {1}, {1024}, args, &e); });
      HaloSampleOut o{};
      (void)d->Download(ob, 0, &o, sizeof(o), &e);
      d->Free(bb);
      d->Free(cb);
      (void)d->Download(ob, 0, &o, sizeof(o), &e);
      std::printf("%s %-44s exact %8.1f us   fast %8.1f us  [blocks %6.1f  final %6.1f  exact-gate %6.1f]%s\n",
                  rt::BackendName(be), c.name, us, us_fast, t_blocks, t_final, t_gate,
                  o.token == HALO_SAMPLE_FALLBACK ? "  FALLBACK" : "");
    }
    }
  }
}
