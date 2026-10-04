#ifndef HALO_ENGINE_FLASH_NEXT_GPU_HPP_
#define HALO_ENGINE_FLASH_NEXT_GPU_HPP_

// Qwen3.8-Flash-Next decode on the GPU (P2: correctness first), mirroring
// halo::model::ReferenceModel step by step: hyper-connection mixing, PLE
// n-gram injection (rows gathered on the CPU from the on-disk table), Gated
// DeltaNet, gated GQA with QSA block selection, and softmax top-k MoE with the
// shared expert. Routing and block selection run on the host with the
// reference's exact tie rules; P3 moves them on device.

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "model/lora.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"

namespace halo::engine {

class FlashNextGpu {
 public:
  [[nodiscard]] static std::unique_ptr<FlashNextGpu> Create(rt::Device& device, const model::ModelWeights& weights,
                                                            model::NgramTable* ngram, const model::LoraAdapter* lora,
                                                            std::uint32_t max_context, std::string* error);
  ~FlashNextGpu();
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits, std::string* error);
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }

 private:
  struct DevTensor {
    rt::Buffer buf;
    rt::KernelId matvec = -1, embed = -1;
  };
  struct DevLora {
    const model::LoraPair* pair = nullptr;
    float scale = 0;
  };
  struct LinearState {
    rt::Buffer conv, state;
  };
  struct AttnState {
    rt::Buffer k, v, index_k, block_k;
    std::uint32_t blocks = 0;
  };

  FlashNextGpu(rt::Device& d, const model::ModelWeights& w, model::NgramTable* ngram)
      : dev_(d), w_(w), ngram_(ngram) {}
  bool Init(const model::LoraAdapter* lora, std::uint32_t max_context, std::string* error);
  bool Upload(const model::TensorRef& t, std::string* error);
  rt::Buffer Scratch(std::size_t floats, std::string* error);
  bool Launch(const char* kernel, rt::Dim3 grid, rt::Dim3 block, std::initializer_list<rt::Arg> args,
              std::string* error);
  bool MatVec(const model::TensorRef& t, std::uint64_t expert, const rt::Buffer& x, const rt::Buffer& y,
              std::string* error);
  bool GroupedNorm(const rt::Buffer& x, const model::TensorRef& w, const rt::Buffer& y, std::string* error);
  bool HcMix(const model::HcMixer& m, bool inject, std::string* error);
  bool Ple(const model::LayerWeights& l, std::int32_t token, std::string* error);
  bool Linear(const model::LayerWeights& l, LinearState& s, std::string* error);
  bool Attention(const model::LayerWeights& l, AttnState& s, std::string* error);
  bool Moe(const model::LayerWeights& l, std::string* error);

  rt::Device& dev_;
  const model::ModelWeights& w_;
  model::NgramTable* ngram_;
  std::uint32_t max_context_ = 0, position_ = 0;
  model::NgramHistory ngram_history_;
  std::unordered_map<const void*, DevTensor> tensors_;
  std::unordered_map<const void*, DevLora> lora_;
  std::unordered_map<std::string, rt::KernelId> kernels_;
  std::vector<LinearState> linear_;
  std::vector<AttnState> attn_;
  std::vector<rt::Buffer> owned_;
  rt::Buffer res_, xn_, lo_, gate_hc_, mixed_, inject_, block_out_, embd_;
  rt::Buffer ple_emb_, ple_key_, ple_value_, ple_query_, ple_gated_, ple_norm_, ple_history_;
  rt::Buffer qkv_, conv_out_, z_, alpha_, beta_, gdn_out_;
  rt::Buffer qg_, q_, gate_, k_, v_, ctx_, scores_, iq_, ik_, pooled_, block_scores_, visible_;
  rt::Buffer router_, eg_, eu_, ed_, sgate_, logits_, lora_tmp_, lora_out_;
};

}  // namespace halo::engine

#endif  // HALO_ENGINE_FLASH_NEXT_GPU_HPP_
