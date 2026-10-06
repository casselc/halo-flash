#ifndef HALO_TRAIN_EXPORT_HPP_
#define HALO_TRAIN_EXPORT_HPP_

// Writes a trained tail as a P10 variant: lora.gguf (llama.cpp adapter
// layout, F32), head.gguf (classifier heads) and manifest.json with the
// parts' sizes and SHA-256, the base hash, provenance and eval summary.

#include <optional>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "model/variant.hpp"
#include "train/last_k.hpp"

namespace halo::train {

struct ExportInfo {
  std::string name;
  std::int64_t version = 1;
  std::string out_dir;
  std::string base_architecture = "qwen35", base_name, base_sha256;
  json::Value provenance = json::Value::MakeObject();
  std::vector<model::VariantEval> eval;
};

/// Returns the manifest path.
[[nodiscard]] std::optional<std::string> ExportVariant(const LastK& model, const ExportInfo& info, std::string* error);

/// Loads a classifier head GGUF (as ExportVariant writes it) into `model`'s
/// head parameters.
[[nodiscard]] bool LoadClassifierHead(const core::GgufReader& head, LastK& model, std::string* error);

}  // namespace halo::train

#endif  // HALO_TRAIN_EXPORT_HPP_
