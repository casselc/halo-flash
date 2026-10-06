#include "train/gemm.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace halo::train {

namespace {

constexpr std::size_t kMr = 6, kNr = 32, kKc = 256, kMc = 96, kNc = 128;
using V16 = float __attribute__((ext_vector_type(16)));

inline float At(const float* X, std::size_t ld, Op op, std::size_t r, std::size_t c) {
  return op == Op::kN ? X[r * ld + c] : X[c * ld + r];
}

// Packs rows [m0, m0+mb) x k [k0, k0+kb) of op(A) into kMr-row panels:
// panel p, step k holds rows p*kMr .. p*kMr+5 (zero padded).
void PackA(Op ta, const float* A, std::size_t lda, std::size_t m0, std::size_t mb, std::size_t k0, std::size_t kb,
           float* out) {
  for (std::size_t p = 0; p < mb; p += kMr) {
    float* dst = out + p * kb;
    const std::size_t rows = std::min(kMr, mb - p);
    for (std::size_t k = 0; k < kb; ++k)
      for (std::size_t i = 0; i < kMr; ++i) dst[k * kMr + i] = i < rows ? At(A, lda, ta, m0 + p + i, k0 + k) : 0.0F;
  }
}

// Packs k [k0, k0+kb) x columns [n0, n0+nb) of op(B) into kNr-column panels.
void PackB(Op tb, const float* B, std::size_t ldb, std::size_t k0, std::size_t kb, std::size_t n0, std::size_t nb,
           float* out) {
  for (std::size_t p = 0; p < nb; p += kNr) {
    float* dst = out + p * kb;
    const std::size_t cols = std::min(kNr, nb - p);
    if (tb == Op::kN && cols == kNr) {
      for (std::size_t k = 0; k < kb; ++k) std::memcpy(dst + k * kNr, B + (k0 + k) * ldb + n0 + p, kNr * sizeof(float));
      continue;
    }
    for (std::size_t k = 0; k < kb; ++k)
      for (std::size_t j = 0; j < kNr; ++j) dst[k * kNr + j] = j < cols ? At(B, ldb, tb, k0 + k, n0 + p + j) : 0.0F;
  }
}

// acc[6][32] = sum_k a[k][6] (x) b[k][32]; then C = alpha*acc + (first ? beta*C : C).
void MicroKernel(std::size_t kb, const float* a, const float* b, float* C, std::size_t ldc, std::size_t rows,
                 std::size_t cols, float alpha, float beta, bool first) {
  V16 acc[kMr][2];
  for (auto& r : acc) r[0] = r[1] = V16(0.0F);
  for (std::size_t k = 0; k < kb; ++k) {
    V16 b0, b1;
    std::memcpy(&b0, b + k * kNr, sizeof(V16));
    std::memcpy(&b1, b + k * kNr + 16, sizeof(V16));
    const float* ak = a + k * kMr;
    for (std::size_t i = 0; i < kMr; ++i) {
      const V16 ai(ak[i]);
      acc[i][0] += ai * b0;
      acc[i][1] += ai * b1;
    }
  }
  for (std::size_t i = 0; i < rows; ++i) {
    float tmp[kNr];
    std::memcpy(tmp, &acc[i][0], sizeof(V16));
    std::memcpy(tmp + 16, &acc[i][1], sizeof(V16));
    float* c = C + i * ldc;
    if (first && beta == 0.0F) {
      for (std::size_t j = 0; j < cols; ++j) c[j] = alpha * tmp[j];
    } else {
      const float bscale = first ? beta : 1.0F;
      for (std::size_t j = 0; j < cols; ++j) c[j] = alpha * tmp[j] + bscale * c[j];
    }
  }
}

// Few rows (M <= kSkinny): a packed GEMM would copy all of B for a handful
// of rows, so B is streamed once instead (the LM head over a few loss
// positions is 2 GB of weights per call on Qwen3.5-2B).
constexpr std::size_t kSkinny = 4;

void SkinnyNT(ThreadPool& pool, std::size_t M, std::size_t N, std::size_t K, float alpha, const float* A,
              std::size_t lda, const float* B, std::size_t ldb, float beta, float* C, std::size_t ldc) {
  constexpr std::size_t kCols = 64;
  pool.ParallelFor((N + kCols - 1) / kCols, [&](std::size_t blk) {
    for (std::size_t j = blk * kCols; j < std::min(N, (blk + 1) * kCols); ++j) {
      const float* b = B + j * ldb;
      for (std::size_t i = 0; i < M; ++i) {
        const float* a = A + i * lda;
        V16 acc(0.0F);
        std::size_t k = 0;
        for (; k + 16 <= K; k += 16) {
          V16 av, bv;
          std::memcpy(&av, a + k, sizeof(V16));
          std::memcpy(&bv, b + k, sizeof(V16));
          acc += av * bv;
        }
        float sum = 0;
        for (int l = 0; l < 16; ++l) sum += acc[l];
        for (; k < K; ++k) sum += a[k] * b[k];
        float& c = C[i * ldc + j];
        c = alpha * sum + (beta == 0.0F ? 0.0F : beta * c);
      }
    }
  });
}

void SkinnyNN(ThreadPool& pool, std::size_t M, std::size_t N, std::size_t K, float alpha, const float* A,
              std::size_t lda, const float* B, std::size_t ldb, float beta, float* C, std::size_t ldc) {
  constexpr std::size_t kCols = 64;
  pool.ParallelFor((N + kCols - 1) / kCols, [&](std::size_t blk) {
    const std::size_t j0 = blk * kCols, cols = std::min(kCols, N - j0);
    float acc[kSkinny][kCols] = {};
    for (std::size_t k = 0; k < K; ++k) {
      const float* b = B + k * ldb + j0;
      for (std::size_t i = 0; i < M; ++i) {
        const float a = A[i * lda + k];
        for (std::size_t j = 0; j < cols; ++j) acc[i][j] += a * b[j];
      }
    }
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < cols; ++j) {
        float& c = C[i * ldc + j0 + j];
        c = alpha * acc[i][j] + (beta == 0.0F ? 0.0F : beta * c);
      }
  });
}

}  // namespace

void Gemm(ThreadPool& pool, Op ta, Op tb, std::size_t M, std::size_t N, std::size_t K, float alpha, const float* A,
          std::size_t lda, const float* B, std::size_t ldb, float beta, float* C, std::size_t ldc) {
  if (M == 0 || N == 0) return;
  if (K == 0) {
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j) C[i * ldc + j] = beta == 0.0F ? 0.0F : beta * C[i * ldc + j];
    return;
  }
  if (M <= kSkinny && ta == Op::kN && K >= 256) {
    if (tb == Op::kT) return SkinnyNT(pool, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return SkinnyNN(pool, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  const std::size_t mblocks = (M + kMc - 1) / kMc, nblocks = (N + kNc - 1) / kNc;
  const std::size_t mpad = mblocks * kMc;
  std::vector<float> apack(mpad * kKc);
  for (std::size_t k0 = 0; k0 < K; k0 += kKc) {
    const std::size_t kb = std::min(kKc, K - k0);
    const bool first = k0 == 0;
    pool.ParallelFor(mblocks, [&](std::size_t mb) {
      const std::size_t m0 = mb * kMc, rows = std::min(kMc, M - m0);
      PackA(ta, A, lda, m0, rows, k0, kb, apack.data() + m0 * kb);
    });
    pool.ParallelFor(mblocks * nblocks, [&](std::size_t task) {
      const std::size_t mb = task % mblocks, nb = task / mblocks;
      const std::size_t m0 = mb * kMc, rows = std::min(kMc, M - m0);
      const std::size_t n0 = nb * kNc, cols = std::min(kNc, N - n0);
      thread_local std::vector<float> bpack;
      bpack.resize(kNc * kKc);
      PackB(tb, B, ldb, k0, kb, n0, cols, bpack.data());
      for (std::size_t j = 0; j < cols; j += kNr)
        for (std::size_t i = 0; i < rows; i += kMr)
          MicroKernel(kb, apack.data() + m0 * kb + i * kb, bpack.data() + j * kb, C + (m0 + i) * ldc + n0 + j, ldc,
                      std::min(kMr, rows - i), std::min(kNr, cols - j), alpha, beta, first);
    });
  }
}

void GemmReference(Op ta, Op tb, std::size_t M, std::size_t N, std::size_t K, float alpha, const float* A,
                   std::size_t lda, const float* B, std::size_t ldb, float beta, float* C, std::size_t ldc) {
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      double acc = 0;
      for (std::size_t k = 0; k < K; ++k) acc += static_cast<double>(At(A, lda, ta, i, k)) * At(B, ldb, tb, k, j);
      const float prev = beta == 0.0F ? 0.0F : beta * C[i * ldc + j];
      C[i * ldc + j] = static_cast<float>(alpha * acc) + prev;
    }
}

}  // namespace halo::train
