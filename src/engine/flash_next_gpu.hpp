#ifndef HALO_ENGINE_FLASH_NEXT_GPU_HPP_
#define HALO_ENGINE_FLASH_NEXT_GPU_HPP_

// Qwen3.8-Flash-Next decode on the GPU (P2: correctness first), mirroring
// halo::model::ReferenceModel step by step: hyper-connection mixing, PLE
// n-gram injection (rows gathered on the CPU from the on-disk table), Gated
// DeltaNet, gated GQA with QSA block selection, and softmax top-k MoE with the
// shared expert. Routing and QSA block selection run on device with the
// reference's tie rules (equal scores keep the lower index), so a step
// needs no host round trip except the n-gram rows and the final logits.

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
  /// Runs `tokens` in order; `logits` (optional) receives the last position's.
  /// Token at a time for now (P4 batches it); any chunking is equivalent.
  [[nodiscard]] bool Prefill(std::span<const std::int32_t> tokens, std::span<float> logits, std::string* error);
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }

  /// A device-resident copy of the whole sequence state: DeltaNet recurrent
  /// and conv state, attention KV, QSA indexer keys and pooled blocks, PLE
  /// conv history, n-gram window and position. Many may exist (forks).
  class Snapshot;
  [[nodiscard]] std::unique_ptr<Snapshot> NewSnapshot(std::string* error);
  [[nodiscard]] bool Save(Snapshot& snapshot, std::string* error);
  [[nodiscard]] bool Restore(const Snapshot& snapshot, std::string* error);
  /// Rewinds to an empty sequence.
  [[nodiscard]] bool Reset(std::string* error);

 private:
  struct DevTensor {
    rt::Buffer buf;
    rt::KernelId matvec = -1, embed = -1, expert = -1;
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
  /// Matvec on the expert that routing slot `slot` selected (ids on device).
  bool ExpertMatVec(const model::TensorRef& t, std::uint32_t slot, const rt::Buffer& x, const rt::Buffer& y,
                    std::string* error);
  /// Visits every state buffer with the float count currently meaningful.
  template <typename F>
  bool ForEachState(F&& f);
  bool CopyFloats(const rt::Buffer& dst, const rt::Buffer& src, std::size_t n, std::string* error);

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
  rt::Buffer router_, expert_ids_, expert_weights_, eg_, eu_, ed_, sgate_, logits_, lora_tmp_, lora_out_;
};

class FlashNextGpu::Snapshot {
 public:
  ~Snapshot() {
    for (auto& b : buffers_) dev_.Free(b);
  }

 private:
  friend class FlashNextGpu;
  explicit Snapshot(rt::Device& d) : dev_(d) {}
  rt::Device& dev_;
  std::vector<rt::Buffer> buffers_;  ///< ForEachState order
  std::vector<std::uint32_t> blocks_;
  model::NgramHistory ngram_history_;
  std::uint32_t position_ = 0;
  bool valid_ = false;
};

}  // namespace halo::engine

#endif  // HALO_ENGINE_FLASH_NEXT_GPU_HPP_
