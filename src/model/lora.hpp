#ifndef HALO_MODEL_LORA_HPP_
#define HALO_MODEL_LORA_HPP_

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "core/gguf.hpp"
#include "model/weights.hpp"

namespace halo::model {

/// One low-rank update W' = W + scale * B A for a base tensor.
struct LoraPair {
  TensorRef a;  ///< [rank rows x in cols]  (GGUF `<name>.lora_a`, dims (in, r))
  TensorRef b;  ///< [out rows x rank cols] (GGUF `<name>.lora_b`, dims (r, out))
  std::uint64_t rank = 0;
};

/// A GGUF LoRA adapter in llama.cpp's layout (general.type = adapter,
/// adapter.type = lora, adapter.lora.alpha), bound against a base model.
/// Every pair is validated against its base tensor's shape at load time.
/// Expert (3-D) tensors are not adaptable yet.
struct LoraAdapter {
  float alpha = 0;
  std::unordered_map<std::string, LoraPair> pairs;  ///< base tensor name -> pair

  [[nodiscard]] const LoraPair* Find(std::string_view base_name) const;
  [[nodiscard]] float Scale(const LoraPair& p) const { return alpha / static_cast<float>(p.rank); }

  [[nodiscard]] static std::optional<LoraAdapter> Bind(const core::GgufReader& adapter,
                                                       const core::GgufReader& base,
                                                       std::string* error);
};

}  // namespace halo::model

#endif  // HALO_MODEL_LORA_HPP_
