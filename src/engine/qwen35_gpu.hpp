#ifndef HALO_ENGINE_QWEN35_GPU_HPP_
#define HALO_ENGINE_QWEN35_GPU_HPP_

// Qwen3.5 dense decode on the GPU (P2: correctness first). One token per
// Step, every operation a kernel from src/kernels/core.hip, checked against
// halo::model::Qwen35Reference. Runs on either rt backend (HIP or HRX).

#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "model/lora.hpp"
#include "model/qwen35.hpp"
#include "rt/device.hpp"

namespace halo::engine {

class Qwen35Gpu {
 public:
  [[nodiscard]] static std::unique_ptr<Qwen35Gpu> Create(rt::Device& device, const model::Qwen35Weights& weights,
                                                         const model::LoraAdapter* lora, std::uint32_t max_context,
                                                         std::string* error);
  ~Qwen35Gpu();

  /// Runs `token` at the next position; `hidden` / `logits` (optional) receive
  /// the final normalized hidden state and the LM head output.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> hidden, std::span<float> logits, std::string* error);
  /// Runs `tokens` at the next positions as one chunk (P4): projections as
  /// T-token GEMMs (BF16 weights on FP16 WMMA, fp32 accumulation), DeltaNet
  /// with the chunk kernels, attention a token at a time. `hidden`
  /// (optional, tokens.size() x hidden_size) receives every position's final
  /// normalized hidden state. Long inputs run in chunks of kMaxChunk.
  [[nodiscard]] bool Prefill(std::span<const std::int32_t> tokens, std::span<float> hidden, std::string* error);
  /// The stage-1 trainer's frozen trunk (P11): runs `tokens` as Prefill does
  /// but only through layers [0, layer), writing the residual stream
  /// entering `layer` for every position (tokens.size() x hidden_size).
  /// Layers from `layer` on are not advanced: Reset before any other use.
  [[nodiscard]] bool PrefillToLayer(std::span<const std::int32_t> tokens, std::uint32_t layer,
                                    std::span<float> residual, std::string* error);

  /// Several LoRA adapters resident at once (P10 variants), selected per
  /// call: AddAdapter uploads one under `id` (rank <= kMaxLoraRank; its
  /// tensors must target this model's projections), SelectAdapter makes it
  /// apply to the following Step/Prefill calls ("" = the base model).
  /// The adapter given to Create is resident as "default" and selected.
  /// Switching adapters mid-sequence mixes states; callers Reset between
  /// requests of different variants.
  static constexpr std::uint32_t kMaxLoraRank = 256;
  [[nodiscard]] bool AddAdapter(const std::string& id, const model::LoraAdapter& lora, std::string* error);
  bool RemoveAdapter(const std::string& id);
  [[nodiscard]] bool SelectAdapter(const std::string& id, std::string* error);
  [[nodiscard]] const std::string& selected_adapter() const noexcept { return selected_; }
  static constexpr std::uint32_t kMaxChunk = 256;
  /// Up to this context the chunk's attention runs in one launch (its score
  /// scratch is kMaxChunk x heads x max_context floats); beyond, a token at a time.
  static constexpr std::uint32_t kMaxBatchedAttention = 8192;
  /// Saves / restores the whole sequence state (recurrent state, conv
  /// history, KV cache, position) in device memory, for prefix forking.
  /// Rewinds to an empty sequence (position 0, zero recurrent state).
  [[nodiscard]] bool Reset(std::string* error);
  [[nodiscard]] bool SaveState(std::string* error);
  [[nodiscard]] bool RestoreState(std::string* error);
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t vocab_size() const noexcept { return w_.config.vocab_size; }

 private:
  struct DevTensor {
    rt::Buffer buf;
    const model::TensorRef* ref = nullptr;
    rt::KernelId matvec = -1, embed = -1;
  };
  struct DevLora {
    rt::Buffer a, b;
    std::uint32_t rank = 0;
    float scale = 0;
  };
  struct LayerState {
    rt::Buffer conv, state, k, v;
  };

  Qwen35Gpu(rt::Device& d, const model::Qwen35Weights& w) : dev_(d), w_(w) {}
  bool Init(const model::LoraAdapter* lora, std::uint32_t max_context, std::string* error);
  bool Upload(const model::TensorRef& t, std::string* error);
  rt::Buffer Scratch(std::size_t floats, std::string* error);
  bool MatVec(const model::TensorRef& t, const rt::Buffer& x, const rt::Buffer& y, std::string* error);
  /// y[T][rows] = x[T][cols] W^T (+ LoRA), for T <= kMaxChunk.
  bool Gemm(const model::TensorRef& t, const rt::Buffer& x, const rt::Buffer& y, std::uint32_t tokens,
            std::string* error);
  bool PrefillChunk(std::span<const std::int32_t> tokens, std::span<float> hidden, std::string* error,
                    std::uint32_t stop_layer = UINT32_MAX);
  using AdapterSet = std::unordered_map<const void*, DevLora>;  // keyed by base tensor host pointer
  bool AllocChunk(std::string* error);
  bool Launch(const char* kernel, rt::Dim3 grid, rt::Dim3 block, std::initializer_list<rt::Arg> args,
              std::string* error);

  rt::Device& dev_;
  const model::Qwen35Weights& w_;
  std::uint32_t max_context_ = 0, position_ = 0, saved_position_ = 0;
  std::unordered_map<const void*, DevTensor> tensors_;  // keyed by host data pointer
  std::unordered_map<std::string, AdapterSet> adapters_;
  const AdapterSet* lora_ = nullptr;  // the selected adapter, or null
  std::string selected_;
  std::unordered_map<std::string, rt::KernelId> kernels_;
  std::vector<LayerState> layers_, saved_;
  std::vector<rt::Buffer> owned_;
  // Activations.
  rt::Buffer res_, xn_, out_, g_, up_, qkv_, conv_out_, z_, alpha_, beta_, gdn_out_, qg_, q_, gate_, k_, v_, ctx_,
      scores_, logits_, lora_tmp_, lora_out_;
  // Chunk activations, [kMaxChunk][dim] (allocated on the first Prefill).
  struct Chunk {
    rt::Buffer res, xn, out, g, up, qkv, conv, z, alpha, beta, gdn, qg, q, gate, k, v, ctx, lora_tmp, lora_out;
    rt::Buffer scores;  // [kMaxChunk][heads][max_context], when max_context <= kMaxBatchedAttention
  } chunk_;
  bool chunk_ready_ = false;
};

}  // namespace halo::engine

#endif  // HALO_ENGINE_QWEN35_GPU_HPP_
