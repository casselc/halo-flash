#include "model/lora.hpp"

#include <set>

namespace halo::model {

const LoraPair* LoraAdapter::Find(std::string_view base_name) const {
  const auto it = pairs.find(std::string(base_name));
  return it == pairs.end() ? nullptr : &it->second;
}

std::optional<LoraAdapter> LoraAdapter::Bind(const core::GgufReader& adapter,
                                             const core::GgufReader& base, std::string* error) {
  const auto fail = [&](std::string m) -> std::optional<LoraAdapter> {
    if (error) *error = std::move(m);
    return std::nullopt;
  };
  if (adapter.GetMetadataString("general.type") != std::optional<std::string_view>{"adapter"} ||
      adapter.GetMetadataString("adapter.type") != std::optional<std::string_view>{"lora"}) {
    return fail("not a LoRA adapter GGUF");
  }
  if (adapter.GetMetadataString("general.architecture") != base.GetMetadataString("general.architecture")) {
    return fail("adapter architecture does not match the base model");
  }
  const auto alpha = adapter.GetMetadataFloat32("adapter.lora.alpha");
  if (!alpha || !(*alpha > 0)) return fail("missing or invalid adapter.lora.alpha");

  LoraAdapter out;
  out.alpha = *alpha;
  std::set<std::string> seen;
  for (const auto& t : adapter.GetTensors()) {
    const std::string name(t.name);
    const bool is_a = name.ends_with(".lora_a"), is_b = name.ends_with(".lora_b");
    if (!is_a && !is_b) return fail("unexpected adapter tensor " + name);
    const std::string base_name = name.substr(0, name.size() - 7);
    if (!seen.insert(base_name).second) continue;
    const auto* a = adapter.FindTensor(base_name + ".lora_a");
    const auto* b = adapter.FindTensor(base_name + ".lora_b");
    const auto* w = base.FindTensor(base_name);
    if (!a || !b) return fail("unpaired LoRA tensor for " + base_name);
    if (!w) return fail("adapter targets a tensor the base model lacks: " + base_name);
    if (w->dimensions.size() != 2) return fail("expert/3-D LoRA targets are not supported: " + base_name);
    if (a->type != core::GgmlType::kF32 || b->type != core::GgmlType::kF32 ||
        a->dimensions.size() != 2 || b->dimensions.size() != 2) {
      return fail("LoRA tensors must be 2-D F32: " + base_name);
    }
    const std::uint64_t in = w->dimensions[0], out_dim = w->dimensions[1];
    const std::uint64_t rank = a->dimensions[1];
    if (rank == 0 || a->dimensions[0] != in || b->dimensions[0] != rank || b->dimensions[1] != out_dim) {
      return fail("LoRA shape does not match base tensor " + base_name);
    }
    LoraPair p;
    p.rank = rank;
    p.a = {a->data, a->type, in, rank, 1, 0, 0, a->name};
    p.b = {b->data, b->type, rank, out_dim, 1, 0, 0, b->name};
    out.pairs.emplace(base_name, p);
  }
  if (out.pairs.empty()) return fail("adapter has no tensors");
  return out;
}

}  // namespace halo::model
