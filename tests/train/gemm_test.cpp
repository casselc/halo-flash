// The trainer's FP32 GEMM against the double-accumulating triple loop, on
// generated shapes, transposes, strides and alpha/beta (Hegel), and its
// determinism contract: bit-identical results at any thread count.

#include <gtest/gtest.h>
#include <hegel/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "train/gemm.hpp"
#include "train/parallel.hpp"

using namespace halo::train;
namespace gs = hegel::generators;

TEST(TrainGemm, MatchesReferenceOnGeneratedShapes) {
  ThreadPool pool(4);
  hegel::test([&](hegel::TestCase& tc) {
    const auto M = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 140}));
    const auto N = tc.draw(gs::integers<std::size_t>({.min_value = 1, .max_value = 300}));
    const auto K = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 600}));
    const Op ta = tc.draw(gs::booleans()) ? Op::kT : Op::kN, tb = tc.draw(gs::booleans()) ? Op::kT : Op::kN;
    const auto pad = tc.draw(gs::integers<std::size_t>({.min_value = 0, .max_value = 3}));
    const float alpha = tc.draw(gs::sampled_from<float>({1.0F, -0.5F, 2.0F}));
    const float beta = tc.draw(gs::sampled_from<float>({0.0F, 1.0F, 0.25F}));
    const std::size_t lda = (ta == Op::kN ? K : M) + pad, ldb = (tb == Op::kN ? N : K) + pad, ldc = N + pad;
    const std::size_t arows = ta == Op::kN ? M : K, brows = tb == Op::kN ? K : N;
    std::vector<float> A(arows * lda + 1), B(brows * ldb + 1), C(M * ldc), R;
    std::uint32_t s = static_cast<std::uint32_t>(M * 131 + N * 7 + K);
    auto rnd = [&] {
      s = s * 1664525u + 1013904223u;
      return static_cast<float>(static_cast<std::int32_t>(s >> 8) % 2001) / 1000.0F;
    };
    for (auto& x : A) x = rnd();
    for (auto& x : B) x = rnd();
    for (auto& x : C) x = rnd();
    R = C;
    Gemm(pool, ta, tb, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    GemmReference(ta, tb, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, R.data(), ldc);
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j) {
        const float got = C[i * ldc + j], want = R[i * ldc + j];
        // FP32 accumulation over K terms of magnitude <= 2 * 2.
        ASSERT_NEAR(got, want, 1e-5F * static_cast<float>(K + 1) * 4.0F + 1e-5F) << i << "," << j;
      }
    // Padding columns of C are never written.
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = N; j < ldc; ++j) ASSERT_EQ(C[i * ldc + j], R[i * ldc + j]);
  }, {.test_cases = 300});
}

TEST(TrainGemm, BitIdenticalAcrossThreadCounts) {
  const std::size_t M = 97, N = 333, K = 700;
  std::vector<float> A(M * K), B(N * K);
  for (std::size_t i = 0; i < A.size(); ++i) A[i] = std::sin(static_cast<float>(i));
  for (std::size_t i = 0; i < B.size(); ++i) B[i] = std::cos(static_cast<float>(i) * 0.37F);
  std::vector<float> first;
  for (unsigned threads : {1u, 2u, 3u, 8u, 16u}) {
    ThreadPool pool(threads);
    std::vector<float> C(M * N, 1.0F);
    Gemm(pool, Op::kN, Op::kT, M, N, K, 0.7F, A.data(), K, B.data(), K, 0.5F, C.data(), N);
    if (first.empty()) first = C;
    ASSERT_EQ(std::memcmp(first.data(), C.data(), C.size() * sizeof(float)), 0) << threads << " threads";
  }
}

TEST(TrainThreadPool, RunsEveryIndexOnceAcrossManyCalls) {
  ThreadPool pool(6);
  for (int round = 0; round < 200; ++round) {
    const std::size_t n = static_cast<std::size_t>(round % 37);
    std::vector<int> hits(n, 0);
    pool.ParallelFor(n, [&](std::size_t i) { ++hits[i]; });
    for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(hits[i], 1);
  }
}
