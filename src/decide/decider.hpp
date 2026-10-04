#ifndef HALO_DECIDE_DECIDER_HPP_
#define HALO_DECIDE_DECIDER_HPP_

// The strands-decider System One model on halo-flash: a Qwen3.5 torso with a
// LoRA adapter, a pointer readout head, and the decision layer of
// systemone.hpp. This is the CPU reference path (correctness first); the GPU
// path will reuse the same request/answer plumbing.

#include <memory>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/json.hpp"
#include "decide/systemone.hpp"
#include "model/lora.hpp"
#include "model/qwen35.hpp"
#include "text/tokenizer.hpp"

namespace halo::decide {

struct DeciderFiles {
  std::string torso_gguf;      ///< Qwen3.5 base model (e.g. Qwen3.5-2B-Base-BF16.gguf)
  std::string lora_gguf;       ///< tools/convert_lora_gguf.py output ("" = none)
  std::string head_gguf;       ///< tools/convert_decider_head.py output
  std::string tokenizer_gguf;  ///< tools/tokenizer_json_to_gguf.py output of the checkpoint's tokenizer.json
};

/// Pointer readout: logit_k = <k(LN(h_option_k)), q(LN(h_answer))> / sqrt(dim).
struct PointerHead {
  std::uint32_t hidden = 0, dim = 0;
  std::vector<float> norm_w, norm_b, q_w, q_b, k_w, k_b;
  [[nodiscard]] std::vector<float> Logits(std::span<const float> decide,
                                          const std::vector<std::vector<float>>& options) const;
};

class StrandsDecider {
 public:
  [[nodiscard]] static std::unique_ptr<StrandsDecider> Open(const DeciderFiles& files, std::string* error);

  /// Answers a /v1/systemone request. Returns the response body, or nullopt
  /// with `error` set (e.g. images on this text-only engine, or an option list
  /// truncated away).
  [[nodiscard]] std::optional<json::Value> Evaluate(const Request& request, std::string* error);

  std::string model_name = "strands-decider-2B-hobson-v19";

 private:
  StrandsDecider() = default;
  [[nodiscard]] float Temperature(Kind kind) const;

  std::unique_ptr<core::GgufReader> torso_file_, lora_file_, head_file_, tok_file_;
  std::optional<model::Qwen35Weights> torso_;
  std::optional<model::LoraAdapter> lora_;
  std::unique_ptr<text::Tokenizer> tokenizer_;
  PointerHead head_;
  float temperature_ = 1.0F;
  std::vector<std::pair<Kind, float>> temperature_by_kind_;
  double ordinal_smoothing_ = 0.0;
  FitOptions fit_;
};

}  // namespace halo::decide

#endif  // HALO_DECIDE_DECIDER_HPP_
