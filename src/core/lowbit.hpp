#ifndef HALO_CORE_LOWBIT_HPP_
#define HALO_CORE_LOWBIT_HPP_

// Simulated low-bit quantization for CODESIGN section 5 (the first
// experiment): rows are quantized to a target code and written back in a
// container format the engine already loads, so the unchanged engine scores
// the quantized model.
//
// Arm A (dense projections and LM head): a 4.125-bit code in a randomized
// Hadamard basis. y = W x = (W R)(R^T x) with R = diag(sign) H / sqrt(b)
// block-diagonal (b = 128, or the largest power of two <= 128 dividing the
// row). The code quantizes W R, and the row stored is dequant(Q(W R)) R^T.
// Rotating and rotating back without quantizing in between returns W; the
// quantization must happen in the rotated basis. Container: Q8_0, which adds
// its own rounding (measured by the tool, reported as a fraction of the
// 4-bit error).
//
// Arm D (routed experts): a 2-bit code, stored exactly. Every 32 values hold
// the four levels a + s * l (l = 0..3, a <= 0), s and -a quantized to 6 bits
// against the maximum over a super-block, as Q4_K quantizes its scales. The
// encoders write that code in the tensor's own container (Q4_K q = 5 l, Q5_K
// q = 10 l, Q5_1 q = 10 l, Q8_0 q = o + j l), so decoding the container
// yields the code itself and the tensor keeps its size and kernels.

#include <cstddef>
#include <cstdint>
#include <span>

#include "core/gguf.hpp"

namespace halo::quant::lowbit {

/// x <- H x / sqrt(n) in place (Sylvester order), n a power of two. The
/// normalized transform is symmetric and orthogonal: applying it twice is
/// the identity.
void Fwht(float* x, std::size_t n);

/// Hadamard block for rows of `cols` values: the largest power of two <= 128
/// that divides cols (0 if cols is odd).
[[nodiscard]] std::size_t HadamardBlock(std::size_t cols) noexcept;

/// Arm A on one row: out = dequant(Q4(w R)) R^T with the block-diagonal
/// R = diag(signs) H_b / sqrt(b), b = HadamardBlock(cols), groups of b
/// values sharing one fp16 scale over the 16-entry IQ4_NL codebook; the
/// scale minimizes the group's squared error (grid search plus a
/// least-squares refit, no calibration data). `signs` holds cols entries of
/// +-1. `out` may alias nothing.
void RotatedQ4Row(const float* w, std::size_t cols, const float* signs,
                  float* out);

/// Deterministic +-1 vector for a tensor (seeded by its name).
void RotationSigns(std::uint64_t seed, std::span<float> signs);

/// Encodes `cols` values (cols % 32 == 0) as Q8_0 with the fp16 block scale
/// rounded first and q = round(x / d) clamped to [-127, 127].
void EncodeQ8_0(const float* x, std::size_t cols, std::uint8_t* out);

/// Arm D: the 2-bit code of `x` written in container `type` (Q4_K, Q5_K:
/// cols % 256 == 0; Q5_1, Q8_0: cols % 32 == 0, one super-block per row).
/// Returns false for any other type.
[[nodiscard]] bool Encode2Bit(core::GgmlType type, const float* x,
                              std::size_t cols, std::uint8_t* out);

/// Effective bits per weight of the 2-bit code in `type` for rows of
/// `cols` (2 bits, 6-bit scale and min per 32, two fp16 per super-block).
[[nodiscard]] double Code2BitBpw(core::GgmlType type, std::size_t cols) noexcept;

}  // namespace halo::quant::lowbit

#endif  // HALO_CORE_LOWBIT_HPP_
