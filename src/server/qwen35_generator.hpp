#ifndef HALO_SERVER_QWEN35_GENERATOR_HPP_
#define HALO_SERVER_QWEN35_GENERATOR_HPP_

// The OpenAI service's TextGenerator on a Qwen3.5 dense model (the GPU
// engine of the decider torso), with P10 variants: every LoRA variant is
// resident on the engine, and each request selects one (params.variant,
// "" = the base). Prompts are prefilled (all but the last token) and
// decoded a token at a time with CPU sampling over the full logits. One
// request at a time; the engine is reset for every request, so requests of
// different variants never share state.

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "engine/qwen35_gpu.hpp"
#include "model/variant.hpp"
#include "server/openai_service.hpp"
#include "text/tokenizer.hpp"

namespace halo::server {

class Qwen35Generator : public TextGenerator {
 public:
  /// `engine` and `tokenizer` must outlive the generator.
  Qwen35Generator(engine::Qwen35Gpu& engine, const text::Tokenizer& tokenizer, std::uint32_t max_context);

  /// Uploads a loaded variant's LoRA to the engine under its name.
  [[nodiscard]] bool AddVariant(const model::LoadedVariant& v, std::string* error);
  [[nodiscard]] std::vector<std::string> Variants() const;

  Result Generate(const Params& params, const std::function<bool(std::string_view)>& on_text) override;

 private:
  engine::Qwen35Gpu& engine_;
  const text::Tokenizer& tok_;
  const std::uint32_t max_context_;
  std::vector<text::TokenId> stops_;
  mutable std::mutex mu_;
  std::vector<std::string> variants_;
};

}  // namespace halo::server

#endif  // HALO_SERVER_QWEN35_GENERATOR_HPP_
