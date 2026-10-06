#ifndef HALO_TRAIN_LAST_K_HPP_
#define HALO_TRAIN_LAST_K_HPP_

// P11 stage 1: the trainable tail of a Qwen3.5 dense model. The frozen trunk
// (layers 0 .. L-k-1) runs on the inference engine once per example and
// hands over the residual stream entering layer L-k ("boundary"); this class
// runs layers L-k .. L-1, the final norm and a head in FP32 on the CPU,
// forward and backward, with gradients for:
//   - LoRA pairs on the last k layers' projections (base weights frozen),
//   - a classifier head (LayerNorm + Linear on one position's final hidden
//     state), when configured; the LM head itself stays frozen.
// Semantics follow model::Qwen35Reference exactly (GGUF layout: tiled value
// heads, +1 folded into norm weights); tools/qwen35_grad_reference.py checks
// every gradient against PyTorch autograd on the tiny model.
//
// Memory: activations are recomputed per layer during backward (gradient
// checkpointing at layer boundaries: only each layer's input residual is
// kept between the passes). A DeltaNet layer's backward holds every
// position's recurrent state (T x v_heads x d x d floats; 268 MB at T = 256
// on Qwen3.5-2B), so examples are limited to kMaxTokens.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "model/lora.hpp"
#include "model/qwen35.hpp"
#include "train/parallel.hpp"

namespace halo::train {

/// A frozen weight as FP32 rows: borrowed when the file stores F32,
/// otherwise dequantized once (or per chunk, for the LM head).
struct DenseWeight {
  const float* data = nullptr;
  std::size_t rows = 0, cols = 0;
  std::vector<float> owned;
  [[nodiscard]] static DenseWeight From(const model::TensorRef& t);
};

enum class HeadKind { kLm, kClassifier };

struct TailConfig {
  std::uint32_t last_k = 2;
  /// LoRA on the last k layers. Targets are GGUF projection names: attn_q,
  /// attn_k, attn_v, attn_output (attention); attn_qkv, attn_gate,
  /// ssm_alpha, ssm_beta, ssm_out (DeltaNet); ffn_gate, ffn_up, ffn_down.
  std::uint32_t lora_rank = 0;  ///< 0: no LoRA
  float lora_alpha = 0;
  std::vector<std::string> lora_targets;
  HeadKind head = HeadKind::kLm;
  std::uint32_t num_classes = 0;  ///< classifier head
};

/// One trainable tensor inside the flat parameter vector.
struct ParamSlice {
  std::string name;  ///< "<base tensor>.lora_a" / ".lora_b", or "head.norm.weight", ...
  std::size_t offset = 0, rows = 0, cols = 0;
  bool is_head = false;  ///< optimizer group (separate learning rate)
  bool decay = true;     ///< AdamW weight decay applies
  [[nodiscard]] std::size_t size() const { return rows * cols; }
};

struct Example {
  std::span<const float> boundary;  ///< [T][hidden]: residual entering layer L-k
  std::uint32_t tokens = 0;          ///< T
  /// LM loss: (position, next token) pairs; position p's logits predict it.
  std::vector<std::pair<std::uint32_t, std::int32_t>> lm_targets;
  /// Classifier loss: label of the final hidden state at `label_position`.
  std::int32_t label = -1;
  std::uint32_t label_position = 0;
};

/// Optional instrumentation for gradient checks.
struct Probe {
  /// d(loss)/d(residual entering layer L-k+i), i = 0..k-1, and entering the
  /// final norm (index k); each [T][hidden].
  std::vector<std::vector<float>> d_residual;
  /// Classifier / LM logits of the loss positions ([n][classes or vocab]).
  std::vector<float> logits;
};

class LastK {
 public:
  static constexpr std::uint32_t kMaxTokens = 2048;
  [[nodiscard]] static std::unique_ptr<LastK> Create(const model::Qwen35Weights& w, const TailConfig& cfg,
                                                     std::string* error);

  [[nodiscard]] const TailConfig& config() const { return cfg_; }
  [[nodiscard]] std::uint32_t first_layer() const { return first_layer_; }
  [[nodiscard]] std::span<const ParamSlice> slices() const { return slices_; }
  [[nodiscard]] std::size_t num_params() const { return params_.size(); }
  [[nodiscard]] std::vector<float>& params() { return params_; }
  [[nodiscard]] const std::vector<float>& params() const { return params_; }
  [[nodiscard]] const ParamSlice* FindSlice(std::string_view name) const;

  /// PEFT's default initialization: A ~ U(-1/sqrt(in), 1/sqrt(in)), B = 0;
  /// classifier LayerNorm (1, 0), Linear weight and bias ~ U(-1/sqrt(H), 1/sqrt(H)).
  void InitParams(std::uint64_t seed);
  /// Copies A and B of every configured pair from `adapter` (its rank and
  /// alpha must match the config).
  [[nodiscard]] bool LoadLora(const model::LoraAdapter& adapter, std::string* error);

  struct Loss {
    double sum = 0;           ///< summed cross-entropy over the loss positions
    std::uint32_t count = 0;  ///< loss positions
    std::uint32_t correct = 0;  ///< argmax == target
  };
  /// Forward and backward on one example; adds scale * d(sum loss)/d(param)
  /// to `grad` (num_params floats).
  [[nodiscard]] Loss ForwardBackward(const Example& ex, float scale, std::span<float> grad, ThreadPool& pool,
                                     Probe* probe = nullptr) const;
  /// Forward only.
  [[nodiscard]] Loss Forward(const Example& ex, ThreadPool& pool, Probe* probe = nullptr) const;

  // Implementation types (public for the .cpp's helpers).
  struct Lora {
    std::size_t a = 0, b = 0;  ///< offsets of A [rank][in] and B [out][rank] in params
    std::uint32_t rank = 0;
    float scale = 0;
  };
  struct Proj {
    DenseWeight w;
    std::string name;
    int lora = -1;  ///< index into loras_
  };
  struct Layer {
    bool linear = false;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    Proj qkv, gate, alpha, beta, out;  // DeltaNet
    const float *conv = nullptr, *dt = nullptr, *a = nullptr, *ssm_norm = nullptr;
    Proj q, k, v, o;  // attention
    const float *q_norm = nullptr, *k_norm = nullptr;
    Proj ffn_gate, ffn_up, ffn_down;
  };
  struct Tape;

 private:
  LastK() = default;
  Loss Run(const Example& ex, bool backward, float scale, std::span<float> grad, ThreadPool& pool, Probe* probe) const;

  const model::Qwen35Weights* w_ = nullptr;
  TailConfig cfg_;
  std::uint32_t first_layer_ = 0;
  std::vector<Layer> layers_;
  std::vector<Lora> loras_;
  std::vector<ParamSlice> slices_;
  std::vector<float> params_;
  const float* output_norm_ = nullptr;
  model::TensorRef output_;  ///< LM head (may be the token embedding)
  DenseWeight output_dense_;  ///< FP32 LM head rows (unset: dequantized per chunk)
  std::size_t head_norm_w_ = 0, head_norm_b_ = 0, head_w_ = 0, head_b_ = 0;
};

}  // namespace halo::train

#endif  // HALO_TRAIN_LAST_K_HPP_
