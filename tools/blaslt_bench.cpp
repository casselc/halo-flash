// hipBLASLt timing for the prefill GEMM shapes (a reference for our kernels).
//
//   halo_blaslt_bench [tokens]
//
// y[t][r] = sum_k W[r][k] x[t][k] with W rows x K and x tokens x K, both
// row-major (column-major: A = W^T stored K x rows, op T; B = x stored K x
// tokens, op N). For every shape and input/output type pair it asks the
// heuristic for up to 32 candidates, times each (5 warm-up, 20 timed launches,
// GPU events) and prints the best, with TFLOP/s. Types with no solution are
// reported as such. Random inputs; nothing is checked, this times only.
//
// Not in the CMake build (hipBLASLt is not an engine dependency). Inside the
// dev shell, with hipblaslt and hipblas-common from the pinned nixpkgs:
//   hipcc -O2 -std=c++20 -I<hipblaslt>/include -I<hipblas-common>/include \
//     tools/blaslt_bench.cpp -L<hipblaslt>/lib -lhipblaslt -Wl,-rpath,<hipblaslt>/lib
//
// 2026-10-04, pinned clock, 2048 tokens, hipBLASLt 7.2.3, best of fp16/bf16
// in and fp16/bf16/fp32 out: 320x10240 0.638 ms, 10240x320 0.672,
// 512x2560 0.193, 10240x2560 3.530 (ours: 1.048, 1.460 incl. the mix
// epilogue, 1.276 in F32, 3.457 incl. Q8_0 dequant).

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CHECK(x)                                                                    \
  do {                                                                              \
    const auto s_ = (x);                                                            \
    if (s_ != 0) {                                                                  \
      std::fprintf(stderr, "%s:%d %s failed (%d)\n", __FILE__, __LINE__, #x, int(s_)); \
      std::exit(1);                                                                 \
    }                                                                               \
  } while (0)

namespace {

struct Shape {
  const char* name;
  std::uint32_t rows, k;
};

std::size_t Bytes(hipDataType t) { return t == HIP_R_32F ? 4 : 2; }
const char* Name(hipDataType t) { return t == HIP_R_32F ? "f32" : t == HIP_R_16BF ? "bf16" : "f16"; }

}  // namespace

int main(int argc, char** argv) {
  const std::uint32_t T = argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 2048;
  const Shape shapes[] = {
      {"hc down 320x10240", 320, 10240},  {"hc down+inj 336x10240", 336, 10240}, {"hc up 10240x320", 10240, 320},
      {"router 512x2560", 512, 2560},     {"dense 10240x2560", 10240, 2560},     {"dense 2560x6144", 2560, 6144},
  };
  hipblasLtHandle_t h;
  CHECK(hipblasLtCreate(&h));
  const std::size_t ws_bytes = 64ull << 20;
  void* ws = nullptr;
  CHECK(hipMalloc(&ws, ws_bytes));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::printf("tokens %u; best of up to 32 heuristic candidates, 20 timed launches each\n", T);
  std::printf("%-24s %-6s %-5s %10s %9s %6s\n", "shape", "in", "out", "ms", "TFLOP/s", "algos");
  for (const auto& s : shapes) {
    for (const hipDataType in : {HIP_R_16BF, HIP_R_16F}) {
      for (const hipDataType out : {HIP_R_32F, in}) {
        void *a = nullptr, *b = nullptr, *c = nullptr;
        CHECK(hipMalloc(&a, std::size_t{s.rows} * s.k * Bytes(in)));
        CHECK(hipMalloc(&b, std::size_t{T} * s.k * Bytes(in)));
        CHECK(hipMalloc(&c, std::size_t{T} * s.rows * Bytes(out)));
        {
          std::vector<std::uint16_t> fill(std::max(std::size_t{s.rows}, std::size_t{T}) * s.k);
          std::mt19937 g(1);
          for (auto& v : fill) v = static_cast<std::uint16_t>(0x3000 | (g() & 0x3FF));  // small positive values
          CHECK(hipMemcpy(a, fill.data(), std::size_t{s.rows} * s.k * 2, hipMemcpyHostToDevice));
          CHECK(hipMemcpy(b, fill.data(), std::size_t{T} * s.k * 2, hipMemcpyHostToDevice));
        }
        hipblasLtMatmulDesc_t desc;
        CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        hipblasLtMatrixLayout_t la, lb, lc;
        CHECK(hipblasLtMatrixLayoutCreate(&la, in, s.k, s.rows, s.k));
        CHECK(hipblasLtMatrixLayoutCreate(&lb, in, s.k, T, s.k));
        CHECK(hipblasLtMatrixLayoutCreate(&lc, out, s.rows, T, s.rows));
        hipblasLtMatmulPreference_t pref;
        CHECK(hipblasLtMatmulPreferenceCreate(&pref));
        CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
                                                    sizeof(ws_bytes)));
        hipblasLtMatmulHeuristicResult_t res[32];
        int got = 0;
        const auto st = hipblasLtMatmulAlgoGetHeuristic(h, desc, la, lb, lc, lc, pref, 32, res, &got);
        double best = 1e30;
        const float alpha = 1.0f, beta = 0.0f;
        for (int i = 0; st == HIPBLAS_STATUS_SUCCESS && i < got; ++i) {
          bool ok = true;
          for (int w = 0; w < 5 && ok; ++w)
            ok = hipblasLtMatmul(h, desc, &alpha, a, la, b, lb, &beta, c, lc, c, lc, &res[i].algo, ws, ws_bytes,
                                 nullptr) == HIPBLAS_STATUS_SUCCESS;
          if (!ok) continue;
          CHECK(hipEventRecord(e0, nullptr));
          for (int r = 0; r < 20; ++r)
            (void)hipblasLtMatmul(h, desc, &alpha, a, la, b, lb, &beta, c, lc, c, lc, &res[i].algo, ws, ws_bytes,
                                  nullptr);
          CHECK(hipEventRecord(e1, nullptr));
          CHECK(hipEventSynchronize(e1));
          float ms = 0;
          CHECK(hipEventElapsedTime(&ms, e0, e1));
          best = std::min(best, static_cast<double>(ms) / 20);
        }
        if (best < 1e29)
          std::printf("%-24s %-6s %-5s %10.3f %9.1f %6d\n", s.name, Name(in), Name(out), best,
                      2.0 * s.rows * s.k * T / best / 1e9, got);
        else
          std::printf("%-24s %-6s %-5s %10s %9s %6d\n", s.name, Name(in), Name(out), "none", "-", got);
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatrixLayoutDestroy(la);
        hipblasLtMatrixLayoutDestroy(lb);
        hipblasLtMatrixLayoutDestroy(lc);
        hipblasLtMatmulDescDestroy(desc);
        CHECK(hipFree(a));
        CHECK(hipFree(b));
        CHECK(hipFree(c));
      }
    }
  }
  return 0;
}
