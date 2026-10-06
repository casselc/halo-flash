#include "train/export.hpp"

#include <cstring>
#include <filesystem>

#include "core/gguf_writer.hpp"
#include "core/sha256.hpp"

namespace halo::train {

namespace {

std::span<const std::uint8_t> Bytes(const float* p, std::size_t n) {
  return {reinterpret_cast<const std::uint8_t*>(p), n * sizeof(float)};
}

std::optional<model::VariantPart> Part(model::VariantPartKind kind, const std::string& dir, const std::string& file,
                                       std::string* error) {
  const std::string path = (std::filesystem::path(dir) / file).string();
  const auto h = core::Sha256File(path, error);
  if (!h) return std::nullopt;
  return model::VariantPart{kind, file, *h, std::filesystem::file_size(path)};
}

}  // namespace

std::optional<std::string> ExportVariant(const LastK& m, const ExportInfo& info, std::string* error) {
  std::error_code ec;
  std::filesystem::create_directories(info.out_dir, ec);
  model::VariantManifest man;
  man.name = info.name;
  man.version = info.version;
  man.base_architecture = info.base_architecture;
  man.base_name = info.base_name;
  man.base_sha256 = info.base_sha256;
  man.provenance = info.provenance;
  man.eval = info.eval;
  const auto& cfg = m.config();
  const auto& p = m.params();
  if (cfg.lora_rank > 0) {
    core::GgufWriter w;
    w.Set("general.architecture", info.base_architecture);
    w.Set("general.type", std::string("adapter"));
    w.Set("adapter.type", std::string("lora"));
    w.Set("adapter.lora.alpha", cfg.lora_alpha);
    w.Set("general.name", info.name);
    for (const auto& s : m.slices()) {
      if (s.is_head) continue;
      // A: [rank][in] -> GGUF dims (in, rank); B: [out][rank] -> (rank, out).
      w.AddTensor(s.name, {s.cols, s.rows}, core::GgmlType::kF32, Bytes(p.data() + s.offset, s.size()));
    }
    if (!w.WriteFile((std::filesystem::path(info.out_dir) / "lora.gguf").string(), error)) return std::nullopt;
    auto part = Part(model::VariantPartKind::kLora, info.out_dir, "lora.gguf", error);
    if (!part) return std::nullopt;
    man.parts.push_back(*part);
  }
  if (cfg.head == HeadKind::kClassifier) {
    core::GgufWriter w;
    w.Set("general.architecture", std::string("halo-head"));
    w.Set("head.type", std::string("classifier"));
    w.Set("head.num_classes", cfg.num_classes);
    w.Set("head.position", std::string("last"));
    w.Set("head.layer_norm_eps", 1e-5F);
    for (const auto& s : m.slices()) {
      if (!s.is_head) continue;
      std::vector<std::uint64_t> dims = s.rows == 1 ? std::vector<std::uint64_t>{s.cols}
                                                    : std::vector<std::uint64_t>{s.cols, s.rows};
      w.AddTensor(s.name, dims, core::GgmlType::kF32, Bytes(p.data() + s.offset, s.size()));
    }
    if (!w.WriteFile((std::filesystem::path(info.out_dir) / "head.gguf").string(), error)) return std::nullopt;
    auto part = Part(model::VariantPartKind::kClassifierHead, info.out_dir, "head.gguf", error);
    if (!part) return std::nullopt;
    man.parts.push_back(*part);
  }
  if (man.parts.empty()) {
    if (error) *error = "nothing to export (no LoRA, no head)";
    return std::nullopt;
  }
  const std::string path = (std::filesystem::path(info.out_dir) / "manifest.json").string();
  if (!man.WriteFile(path, error)) return std::nullopt;
  return path;
}

bool LoadClassifierHead(const core::GgufReader& head, LastK& m, std::string* error) {
  if (head.GetMetadataString("head.type") != std::optional<std::string_view>{"classifier"}) {
    if (error) *error = "not a classifier head GGUF";
    return false;
  }
  for (const auto& s : m.slices()) {
    if (!s.is_head) continue;
    const auto* t = head.FindTensor(s.name);
    if (!t || t->type != core::GgmlType::kF32 || t->ElementCount() != s.size()) {
      if (error) *error = "head tensor " + s.name + " is missing or has the wrong shape";
      return false;
    }
    std::memcpy(m.params().data() + s.offset, t->data, s.size() * sizeof(float));
  }
  return true;
}

}  // namespace halo::train
