#ifndef HALO_DECIDE_DECIDER_HPP_
#define HALO_DECIDE_DECIDER_HPP_

// The strands-decider System One model on halo-flash: a Qwen3.5 torso with a
// LoRA adapter, a pointer readout head, and the decision layer of
// systemone.hpp. This is the CPU reference path (correctness first); the GPU
// path will reuse the same request/answer plumbing.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/json.hpp"
#include "decide/systemone.hpp"
#include "model/lora.hpp"
#include "model/qwen35.hpp"
#include "model/variant.hpp"
#include "text/tokenizer.hpp"
#if HALO_WITH_HIP
#include "engine/qwen35_gpu.hpp"
#include "rt/device.hpp"
#endif

namespace halo::decide {

struct DeciderFiles {
  std::string torso_gguf;      ///< Qwen3.5 base model (e.g. Qwen3.5-2B-Base-BF16.gguf)
  std::string lora_gguf;       ///< tools/convert_lora_gguf.py output ("" = none)
  std::string head_gguf;       ///< tools/convert_decider_head.py output
  std::string tokenizer_gguf;  ///< tools/tokenizer_json_to_gguf.py output of the checkpoint's tokenizer.json
  /// "cpu" (reference), "hip" or "hrx".
  std::string device = "cpu";
};

/// Pointer readout: logit_k = <k(LN(h_option_k)), q(LN(h_answer))> / sqrt(dim).
struct PointerHead {
  std::uint32_t hidden = 0, dim = 0;
  std::vector<float> norm_w, norm_b, q_w, q_b, k_w, k_b;
  [[nodiscard]] std::vector<float> Logits(std::span<const float> decide,
                                          const std::vector<std::vector<float>>& options) const;
};

/// A pointer head and its inference settings, as a head GGUF carries them
/// (tools/convert_decider_head.py).
struct Readout {
  PointerHead head;
  float temperature = 1.0F;
  std::vector<std::pair<Kind, float>> temperature_by_kind;
  double ordinal_smoothing = 0.0;
  [[nodiscard]] float Temperature(Kind kind) const;
  [[nodiscard]] static std::optional<Readout> FromGguf(const core::GgufReader& head, std::uint32_t hidden,
                                                       std::string* error);
};

class StrandsDecider {
 public:
  [[nodiscard]] static std::unique_ptr<StrandsDecider> Open(const DeciderFiles& files, std::string* error);

  /// P10: adds a variant of the torso's base model, selected per request by
  /// Request::variant. Its LoRA part (if any) replaces the decider's own
  /// LoRA; its pointer_head part (required) replaces the readout. Fails for
  /// a variant without a pointer head or a name already added.
  [[nodiscard]] bool AddVariant(std::shared_ptr<const model::LoadedVariant> variant, std::string* error);
  [[nodiscard]] std::vector<std::string> Variants() const;
  [[nodiscard]] const core::GgufReader& torso_file() const { return *torso_file_; }

  /// Answers a /v1/systemone request. Returns the response body, or nullopt
  /// with `error` set (e.g. images on this text-only engine, or an option list
  /// truncated away).
  [[nodiscard]] std::optional<json::Value> Evaluate(const Request& request, std::string* error);

  std::string model_name = "strands-decider-2B-hobson-v19";

 private:
  StrandsDecider() = default;
  std::unique_ptr<core::GgufReader> torso_file_, lora_file_, head_file_, tok_file_;
  std::optional<model::Qwen35Weights> torso_;
  std::optional<model::LoraAdapter> lora_;
  std::unique_ptr<text::Tokenizer> tokenizer_;
  Readout readout_;
  FitOptions fit_;
  struct Variant {
    std::shared_ptr<const model::LoadedVariant> v;
    Readout readout;
  };
  std::map<std::string, Variant, std::less<>> variants_;
#if HALO_WITH_HIP
  std::unique_ptr<rt::Device> dev_;
  std::unique_ptr<engine::Qwen35Gpu> gpu_;
#endif
};

}  // namespace halo::decide

#endif  // HALO_DECIDE_DECIDER_HPP_
