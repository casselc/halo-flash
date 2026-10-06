#ifndef HALO_TRAIN_GEMM_HPP_
#define HALO_TRAIN_GEMM_HPP_

// FP32 GEMM for the stage-1 trainer (CPU; docs/TRAINING.md has the
// CPU-vs-GPU measurements). Row-major, BLAS argument conventions:
//   C[M][N] = alpha * op(A)[M][K] * op(B)[K][N] + beta * C
// op(X) = X (kN) or X^T (kT); an A with kT is stored [K][M], a B with kT is
// stored [N][K] (a weight matrix W[out][in] gives y = x W^T with tb = kT).
// Packed panels and a 6x32 register-tiled micro-kernel; work is split into
// (row block, column block) tasks whose K loop runs in a fixed order, so
// the result is bit-identical for any thread count.

#include <cstddef>

#include "train/parallel.hpp"

namespace halo::train {

enum class Op { kN, kT };

void Gemm(ThreadPool& pool, Op ta, Op tb, std::size_t M, std::size_t N, std::size_t K, float alpha, const float* A,
          std::size_t lda, const float* B, std::size_t ldb, float beta, float* C, std::size_t ldc);

/// The plain triple loop (double accumulation), for tests.
void GemmReference(Op ta, Op tb, std::size_t M, std::size_t N, std::size_t K, float alpha, const float* A,
                   std::size_t lda, const float* B, std::size_t ldb, float beta, float* C, std::size_t ldc);

}  // namespace halo::train

#endif  // HALO_TRAIN_GEMM_HPP_
