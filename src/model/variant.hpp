#ifndef HALO_MODEL_VARIANT_HPP_
#define HALO_MODEL_VARIANT_HPP_

// Variants (PLAN.md P10): a named, versioned bundle of parts applied on top
// of an unchanged base model, described by a JSON manifest:
//
//   {
//     "format": "halo-variant", "format_version": 1,
//     "name": "sentiment-ab", "version": 3,
//     "base": {"architecture": "qwen35", "name": "Qwen3.5-2B-Base",
//              "content_sha256": "<BaseContentHash of the base GGUF>"},
//     "parts": [{"kind": "lora", "file": "lora.gguf", "sha256": "...", "bytes": 123},
//               {"kind": "classifier_head" | "pointer_head", "file": ..., ...}],
//     "provenance": {"trainer": ..., "code_commit": ..., "data_sha256": ...,
//                    "config": {...}, "config_sha256": ..., ...},
//     "eval": [{"name": ..., "metric": ..., "value": 0.93, "baseline": 0.61,
//               "n": 400, "dataset_sha256": ..., "step": 300}]
//   }
//
// Part files live next to the manifest (relative names only). Loading
// verifies the format version, that the base hash equals the loaded base's
// (otherwise the variant is refused), every part's size and SHA-256, and
// binds LoRA parts against the base tensors' shapes.

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf.hpp"
#include "core/json.hpp"
#include "model/lora.hpp"

namespace halo::model {

enum class VariantPartKind { kLora, kClassifierHead, kPointerHead };
[[nodiscard]] const char* VariantPartKindName(VariantPartKind k) noexcept;

struct VariantPart {
  VariantPartKind kind = VariantPartKind::kLora;
  std::string file;  ///< relative to the manifest's directory
  std::string sha256;
  std::uint64_t bytes = 0;
};

/// One evaluation result in the manifest's eval summary (the trainer's
/// EvalSink writes these; train/eval_report.hpp).
struct VariantEval {
  std::string name, metric, dataset_sha256;
  double value = 0;
  std::optional<double> baseline;  ///< the base model's value on the same set
  std::uint64_t n = 0;
  std::int64_t step = -1;
};

struct VariantManifest {
  static constexpr std::int64_t kFormatVersion = 1;
  std::string name;
  std::int64_t version = 1;
  std::string base_architecture, base_name, base_sha256;
  std::vector<VariantPart> parts;
  json::Value provenance = json::Value::MakeObject();
  std::vector<VariantEval> eval;

  [[nodiscard]] static std::optional<VariantManifest> FromJson(const json::Value& v, std::string* error);
  [[nodiscard]] json::Value ToJson() const;
  [[nodiscard]] static std::optional<VariantManifest> ReadFile(const std::string& path, std::string* error);
  /// Atomic (temp file, fsync, rename).
  [[nodiscard]] bool WriteFile(const std::string& path, std::string* error) const;
  [[nodiscard]] const VariantPart* Find(VariantPartKind k) const;
};

/// Variant names: 1-64 of [A-Za-z0-9._-], not starting with '.'; no ':'
/// (the OpenAI model alias is "base:variant").
[[nodiscard]] bool ValidVariantName(std::string_view name) noexcept;

/// Splits an OpenAI `model` value "base:variant" into its parts; a value
/// without ':' is all base. An empty variant after ':' is an error.
struct ModelAlias {
  std::string base, variant;
};
[[nodiscard]] std::optional<ModelAlias> ParseModelAlias(std::string_view model, std::string* error);

/// A base model's content hash: the SHA-256 of the file's bytes for a
/// single GGUF file; for a split model, the SHA-256 of the lines
/// "<sha256 of shard>  <shard basename>\n" in shard order.
[[nodiscard]] std::optional<std::string> BaseContentHash(std::span<const std::string> files, std::string* error);
/// BaseContentHash with a per-file sidecar cache "<file>.halo-sha256"
/// holding "<sha256> <size> <mtime_ns> <inode>": reused while the file's
/// size, mtime and inode are unchanged, rewritten (best effort) otherwise.
/// Hashing the 79 GB Flash-Next takes tens of seconds; the cache makes
/// repeated loads free. The cache trusts local file metadata, not content.
[[nodiscard]] std::optional<std::string> BaseContentHashCached(std::span<const std::string> files,
                                                               std::string* error);

/// A loaded, verified variant. Immutable once loaded; shared so a request can
/// keep using one while it is unloaded.
struct LoadedVariant {
  VariantManifest manifest;
  std::filesystem::path dir;
  std::unique_ptr<core::GgufReader> lora_file, head_file;
  std::optional<LoraAdapter> lora;  ///< bound against the registry's base
  [[nodiscard]] const core::GgufReader* head() const { return head_file.get(); }
};

/// Every variant of one loaded base model. Thread-safe.
class VariantRegistry {
 public:
  /// `base` must outlive the registry; `base_sha256` is BaseContentHash of
  /// its files, `base_architecture` its general.architecture.
  VariantRegistry(const core::GgufReader& base, std::string base_sha256);

  /// Loads and verifies a manifest; the variant is registered under its
  /// manifest name, replacing nothing (a loaded name is an error).
  [[nodiscard]] std::shared_ptr<const LoadedVariant> Load(const std::string& manifest_path, std::string* error);
  bool Unload(std::string_view name);
  [[nodiscard]] std::shared_ptr<const LoadedVariant> Find(std::string_view name) const;
  [[nodiscard]] std::vector<std::string> Names() const;
  [[nodiscard]] const std::string& base_sha256() const noexcept { return base_sha256_; }
  [[nodiscard]] const std::string& base_architecture() const noexcept { return base_arch_; }

 private:
  const core::GgufReader& base_;
  std::string base_sha256_, base_arch_;
  mutable std::mutex mu_;
  std::map<std::string, std::shared_ptr<const LoadedVariant>, std::less<>> variants_;
};

/// W + scale * B A as F32 rows [rows][cols], for merged-vs-unmerged checks
/// and for exporting a merged model. `base` may be any format cpu::
/// DequantizeRow reads.
[[nodiscard]] std::vector<float> MergeLora(const TensorRef& base, const LoraPair& pair, float scale);

}  // namespace halo::model

#endif  // HALO_MODEL_VARIANT_HPP_
