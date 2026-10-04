// Adapted from gufo (MIT), src/models/qwen38_flash_next/cpu_ops.cpp.
#include "model/cpu_ops.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <thread>
#include <vector>

#include "core/quant.hpp"

namespace halo::model::cpu {
namespace {

using core::GgmlType;

struct BlockQ5_1 {
  std::uint16_t d;
  std::uint16_t m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(BlockQ5_1) == 24);

void DequantizeQ5_1(const std::uint8_t* src, float* dst, std::size_t k) {
  for (std::size_t b = 0; b < k / 32; ++b) {
    BlockQ5_1 blk;
    std::memcpy(&blk, src + b * sizeof(BlockQ5_1), sizeof(BlockQ5_1));
    const float d = halo::quant::Fp16ToFloat(blk.d);
    const float m = halo::quant::Fp16ToFloat(blk.m);
    float* y = dst + b * 32;
    for (int j = 0; j < 16; ++j) {
      const int xh0 = ((blk.qh >> j) & 1U) << 4;
      const int xh1 = ((blk.qh >> (j + 16)) & 1U) << 4;
      y[j] = static_cast<float>((blk.qs[j] & 0x0F) | xh0) * d + m;
      y[j + 16] = static_cast<float>((blk.qs[j] >> 4) | xh1) * d + m;
    }
  }
}

}  // namespace

void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out) {
  const std::uint8_t* src = t.Expert(e) + row * t.RowBytes();
  const std::size_t k = t.cols;
  switch (t.type) {
    case GgmlType::kF32:
      std::memcpy(out, src, k * sizeof(float));
      break;
    case GgmlType::kF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + 2 * i, 2);
        out[i] = halo::quant::Fp16ToFloat(h);
      }
      break;
    case GgmlType::kBF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + 2 * i, 2);
        const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
        std::memcpy(out + i, &bits, 4);
      }
      break;
    case GgmlType::kQ8_0:
      halo::quant::DequantizeQ8_0(src, out, k);
      break;
    case GgmlType::kQ5_1:
      DequantizeQ5_1(src, out, k);
      break;
    case GgmlType::kIQ4_NL:
      halo::quant::DequantizeIQ4_NL(src, out, k);
      break;
    case GgmlType::kQ4_K:
      halo::quant::DequantizeQ4_K(src, out, k);
      break;
    case GgmlType::kQ5_K:
      halo::quant::DequantizeQ5_K(src, out, k);
      break;
    case GgmlType::kQ6_K:
      halo::quant::DequantizeQ6_K(src, out, k);
      break;
    default:
      // Unreachable: ModelWeights::Bind rejects every other format. Never
      // substitute zeros for weights the oracle cannot decode.
      std::fprintf(stderr, "cpu::DequantizeRow: unsupported format %s\n",
                   std::string(core::ToString(t.type)).c_str());
      std::abort();
  }
}

float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x) {
  std::vector<float> w(t.cols);
  DequantizeRow(t, e, row, w.data());
  double acc = 0.0;
  for (std::size_t i = 0; i < t.cols; ++i) {
    acc += static_cast<double>(w[i]) * x[i];
  }
  return static_cast<float>(acc);
}

void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y) {
  const auto rows = static_cast<std::size_t>(t.rows);
  const auto work = [&](std::size_t begin, std::size_t end) {
    std::vector<float> w(t.cols);
    for (std::size_t r = begin; r < end; ++r) {
      DequantizeRow(t, e, r, w.data());
      double acc = 0.0;
      for (std::size_t i = 0; i < t.cols; ++i) acc += static_cast<double>(w[i]) * x[i];
      y[r] = static_cast<float>(acc);
    }
  };
  // Large matrices are split across threads (std::thread, so the reference
  // stays TSan-checkable); each row is computed identically either way.
  const std::size_t threads =
      t.rows * t.cols >= (1u << 20) ? std::max(1u, std::thread::hardware_concurrency()) : 1;
  if (threads == 1 || rows < 2 * threads) {
    work(0, rows);
    return;
  }
  std::vector<std::thread> pool;
  const std::size_t chunk = (rows + threads - 1) / threads;
  for (std::size_t b = 0; b < rows; b += chunk) pool.emplace_back(work, b, std::min(rows, b + chunk));
  for (auto& th : pool) th.join();
}

void RmsNorm(std::span<float> x, const float* w, float eps) {
  double ss = 0.0;
  for (float v : x) {
    ss += static_cast<double>(v) * v;
  }
  const float scale =
      1.0F /
      std::sqrt(static_cast<float>(ss / static_cast<double>(x.size())) + eps);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = x[i] * scale * (w != nullptr ? w[i] : 1.0F);
  }
}

void L2Norm(std::span<float> x, float eps) {
  double ss = 0.0;
  for (float v : x) {
    ss += static_cast<double>(v) * v;
  }
  const float scale = 1.0F / std::sqrt(static_cast<float>(ss) + eps);
  for (float& v : x) {
    v *= scale;
  }
}

float Sigmoid(float x) noexcept {
  return 1.0F / (1.0F + std::exp(-x));
}
float Silu(float x) noexcept {
  return x * Sigmoid(x);
}
float Softplus(float x) noexcept {
  return x > 20.0F ? x : std::log1p(std::exp(x));
}

void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta) {
  const std::uint32_t half = rotary_dim / 2;
  for (std::uint32_t h = 0; h < heads; ++h) {
    float* v = x + static_cast<std::size_t>(h) * head_dim;
    for (std::uint32_t i = 0; i < half; ++i) {
      const float freq = std::pow(theta, -2.0F * static_cast<float>(i) /
                                             static_cast<float>(rotary_dim));
      const float angle = static_cast<float>(pos) * freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const float a = v[i];
      const float b = v[i + half];
      v[i] = a * c - b * s;
      v[i + half] = a * s + b * c;
    }
  }
}

}  // namespace halo::model::cpu
