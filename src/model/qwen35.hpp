#ifndef HALO_MODEL_QWEN35_HPP_
#define HALO_MODEL_QWEN35_HPP_

// Qwen3.5 dense text models (GGUF architecture `qwen35`), the torso of the
// strands-decider System One model. Same operator families as Flash-Next:
// Gated DeltaNet on three of every four layers, gated GQA with partial RoPE on
// the fourth, a dense SwiGLU MLP, and pre-norm residuals. Semantics follow HF
// transformers' modeling_qwen3_5.py; tests/model checks this reference
// against it on a tiny model.

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "model/lora.hpp"
#include "model/weights.hpp"

namespace halo::model {

struct Qwen35Config {
  std::uint32_t num_layers = 0;  ///< trunk layers (MTP block excluded)
  std::uint32_t hidden_size = 0;
  std::uint32_t ffn = 0;
  std::uint32_t vocab_size = 0;
  std::uint32_t num_heads = 0, num_kv_heads = 0, head_dim = 0, rotary_dim = 0;
  float rope_theta = 0, rms_eps = 1e-6F;
  std::uint32_t full_attention_interval = 0;
  std::uint32_t ssm_conv_kernel = 0, ssm_head_dim = 0, ssm_num_k_heads = 0, ssm_num_v_heads = 0;

  [[nodiscard]] bool IsLinearLayer(std::uint32_t l) const { return (l + 1) % full_attention_interval != 0; }
  [[nodiscard]] std::uint32_t KeyDim() const { return ssm_num_k_heads * ssm_head_dim; }
  [[nodiscard]] std::uint32_t ValueDim() const { return ssm_num_v_heads * ssm_head_dim; }
  [[nodiscard]] std::uint32_t ConvChannels() const { return 2 * KeyDim() + ValueDim(); }
};

struct Qwen35Layer {
  bool linear = false;
  TensorRef attn_norm, post_attention_norm;
  // Gated DeltaNet
  TensorRef ssm_qkv, ssm_gate, ssm_conv1d, ssm_alpha, ssm_beta, ssm_dt, ssm_a, ssm_norm, ssm_out;
  // Gated attention
  TensorRef attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm;
  // MLP
  TensorRef ffn_gate, ffn_up, ffn_down;
};

struct Qwen35Weights {
  Qwen35Config config;
  TensorRef token_embd, output_norm, output;  ///< output may be token_embd (tied)
  std::vector<Qwen35Layer> layers;
  [[nodiscard]] static std::optional<Qwen35Weights> Bind(const core::GgufReader& reader, std::string* error);
};

/// Single-token FP32 reference. Copying the object snapshots the whole
/// sequence state (KV, DeltaNet state and conv history), which the decider
/// uses to share one state prefix across several questions.
class Qwen35Reference {
 public:
  Qwen35Reference(const Qwen35Weights& weights, std::uint32_t max_context);

  /// Runs `token` at the next position. `hidden` (hidden_size) receives the
  /// final normalized hidden state; `logits` (vocab) the LM head output.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> hidden, std::span<float> logits,
                          std::string* error = nullptr);
  void SetAdapter(const LoraAdapter* adapter) noexcept { lora_ = adapter; }
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }

 private:
  void MatVec(const TensorRef& w, std::span<const float> x, std::span<float> y) const;
  void Linear(const Qwen35Layer& l, std::size_t index, std::span<const float> x, std::span<float> out);
  void Attention(const Qwen35Layer& l, std::size_t index, std::span<const float> x, std::span<float> out);

  const Qwen35Weights* w_;
  const LoraAdapter* lora_ = nullptr;
  std::uint32_t max_context_;
  std::uint32_t position_ = 0;
  struct LinearState { std::vector<float> conv, state; };
  struct AttnState { std::vector<float> k, v; };
  std::vector<LinearState> linear_;
  std::vector<AttnState> attn_;
};

}  // namespace halo::model

#endif  // HALO_MODEL_QWEN35_HPP_
