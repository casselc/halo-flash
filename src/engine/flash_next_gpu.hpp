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

#include "core/sampling.hpp"
#include "model/lora.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "rt/file_upload.hpp"

namespace halo::engine {

class FlashNextGpu {
 public:
  [[nodiscard]] static std::unique_ptr<FlashNextGpu> Create(rt::Device& device, const model::ModelWeights& weights,
                                                            model::NgramTable* ngram, const model::LoraAdapter* lora,
                                                            std::uint32_t max_context, std::string* error,
                                                            std::span<const rt::FileSource> sources = {});
  ~FlashNextGpu();
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits, std::string* error);

  /// Runs `token` and samples the next one on the device; only the token
  /// and the advanced RNG state come back. `penalties` is the request's
  /// state (sampling::SamplerState::penalties(), sorted by token, at most
  /// 8192 entries); `rng` its RNG state.
  struct SampleRequest {
    const sampling::SamplingConfig* config = nullptr;
    std::span<const sampling::TokenPenalty> penalties;
    std::uint64_t rng = 0;
  };
  struct SampleResult {
    std::int32_t token = -1;
    std::uint64_t rng = 0;
  };
  [[nodiscard]] bool StepSample(std::int32_t token, const SampleRequest& request, SampleResult* result,
                                std::string* error);
  /// Fast (default): distribution-exact, falls back to the exact kernel
  /// when needed. Exact: draw-identical to core/sampling (debugging).
  enum class SamplingMode { kFast, kExact };
  void SetSamplingMode(SamplingMode m) noexcept { sampling_mode_ = m; }
  /// P3 kernels: repack Q8_0 weights at load for the bandwidth GEMV
  /// (default). Must be set before Create's upload, so it is a static knob.
  static inline bool repack_q8_ = true;
  static inline bool block_gemv_ = true;  ///< P3 Q4_K / Q5_1 GEMVs
  static inline bool fused_moe_ = true;   ///< P3 routed experts in three launches
  static inline bool small_row_kernels_ = true;  ///< P3 block-per-row F32 matvec for few rows
  /// P3 expert gate+up with int8 activations (llama.cpp-style Q8_1); false
  /// keeps the float kernel. A quality-gated choice (docs/BENCHMARKS.md).
  static inline bool int8_act_moe_ = true;
  static inline bool fused_hc_ = true;  ///< P3 hyper-connection mixer in two launches
  static inline bool fused_shexp_ = true;  ///< P3 shared expert in two launches
  static inline bool fused_linear_ = true;  ///< P3 DeltaNet block in five launches
  static constexpr std::uint32_t kHcParts = 8;  ///< column chunks of the fused mixer's down projection
  static constexpr std::uint32_t kPrefillChunk = 256;
  static inline bool batched_prefill_ = true;
  /// Debug: chunk projections through the decode GEMVs (token by token)
  /// instead of the int8 GEMMs, so a chunk equals stepping up to float order.
  static inline bool prefill_exact_projections_ = false;
  /// Repacked Q8_0 projections as int8 WMMA GEMMs (activations quantized per
  /// token and 32 values); off, they run through the decode GEMVs per token.
  static inline bool prefill_int8_gemm_ = true;
  /// Runs `tokens` in order; `logits` (optional) receives the last position's.
  /// P4: in chunks of kPrefillChunk tokens, the dense projections as T-token
  /// GEMMs and the rest row-batched or token by token (prefill_chunk.cpp);
  /// a token at a time when batched_prefill_ is off or the model's shapes
  /// do not fit the chunk kernels.
  [[nodiscard]] bool Prefill(std::span<const std::int32_t> tokens, std::span<float> logits, std::string* error);
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  /// Graph replay (default) or eager launches; both run the same sequence.
  void SetUseGraphs(bool on) noexcept { use_graphs_ = on; }
  /// Kernels launched so far by name (not the per-type embedding lookups), so
  /// tests can assert which paths a model's shapes and types select.
  [[nodiscard]] bool LaunchedKernel(const std::string& name) const { return kernels_.contains(name); }

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
    // P3: Q8_0 repacked into aligned int8 rows (`qs`) and fp16 scales (`d`).
    rt::Buffer qs, d;
    std::uint32_t qstride = 0;
  };
  bool RepackQ8(const model::TensorRef& t, DevTensor& d, std::string* error);
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
  bool Init(const model::LoraAdapter* lora, std::uint32_t max_context, std::string* error,
            std::span<const rt::FileSource> sources);
  bool Upload(const model::TensorRef& t, std::string* error);
  rt::Buffer Scratch(std::size_t floats, std::string* error);
  bool Launch(const char* kernel, rt::Dim3 grid, rt::Dim3 block, std::initializer_list<rt::Arg> args,
              std::string* error);
  bool MatVec(const model::TensorRef& t, std::uint64_t expert, const rt::Buffer& x, const rt::Buffer& y,
              std::string* error, std::size_t x_off = 0, std::size_t y_off = 0);
  bool GroupedNorm(const rt::Buffer& x, const model::TensorRef& w, const rt::Buffer& y, std::string* error);
  bool HcMix(const model::HcMixer& m, bool inject, std::string* error);
  bool Ple(const model::LayerWeights& l, std::string* error);
  /// Emits one whole step (everything after the host upload) as device
  /// work; run eagerly or captured into a graph.
  enum class Head { kNone, kLogits, kSample };
  bool Record(Head head, std::string* error);
  bool Run(std::int32_t token, Head head, const SampleRequest* request, std::span<float> logits,
           SampleResult* result, std::string* error);
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

  // ---- P4 chunk prefill (prefill_chunk.cpp) ----
  struct ChunkBuffers {
    rt::Buffer res, xn, lo, gate, mixed, inject, block_out, embd, xq, xd, host;
    rt::Buffer qkv, conv, z, alpha, beta, gdn, qg, q, gate_a, k, v, ctx, iq, ik;
    rt::Buffer router, ids, wts, eg, eu, ed, sgate;
    std::size_t params_offset = 0;  // in `host`: per-token step params after the n-gram rows
  };
  ChunkBuffers ck_;
  bool ck_ready_ = false;
  std::vector<std::uint8_t> ck_host_;
  bool ChunkSupported() const;
  bool AllocChunk(std::string* error);
  bool PrefillChunk(std::span<const std::int32_t> tokens, std::span<float> logits, std::string* error);
  /// y[T][rows] = W x[T][cols]: int8 GEMM for repacked Q8 (x quantized here),
  /// BF16 / F32 GEMMs, else the decode GEMV per token.
  bool Gemm(const model::TensorRef& t, const rt::Buffer& x, const rt::Buffer& y, std::uint32_t tokens,
            std::string* error);
  bool HcMixChunk(const model::HcMixer& m, bool inject, std::uint32_t tokens, std::string* error);
  bool PleChunk(const model::LayerWeights& l, std::uint32_t tokens, std::string* error);
  bool LinearChunk(const model::LayerWeights& l, LinearState& s, std::uint32_t tokens, std::string* error);
  bool AttentionChunk(const model::LayerWeights& l, AttnState& s, std::uint32_t tokens, std::string* error);
  bool MoeChunk(const model::LayerWeights& l, std::uint32_t tokens, std::string* error);

  rt::Device& dev_;
  const model::ModelWeights& w_;
  model::NgramTable* ngram_;
  std::uint32_t max_context_ = 0, position_ = 0;
  bool use_graphs_ = true;
  rt::Graph graphs_[3];  ///< by Head: none, logits, sample
  SamplingMode sampling_mode_ = SamplingMode::kFast;
  /// One upload per step: n-gram rows, step parameters, sampling parameters,
  /// penalties (only the used prefix is sent).
  std::size_t params_offset_ = 0, sample_offset_ = 0, penalty_offset_ = 0;
  std::vector<std::uint8_t> step_host_;
  std::uint32_t sample_blocks_ = 0;
  rt::Buffer sample_block_out_, sample_candidates_, sample_key_, sample_rid_, sample_w_, sample_out_;
  model::NgramHistory ngram_history_;
  std::unordered_map<const void*, DevTensor> tensors_;
  rt::FileUploader* loader_ = nullptr;  ///< during Init: weights read from their files
  std::unordered_map<const void*, DevLora> lora_;
  std::unordered_map<std::string, rt::KernelId> kernels_;
  std::vector<LinearState> linear_;
  std::vector<AttnState> attn_;
  std::vector<rt::Buffer> owned_;
  rt::Buffer res_, xn_, lo_, gate_hc_, mixed_, inject_, block_out_, embd_;
  rt::Buffer ple_emb_, ple_key_, ple_value_, ple_query_, ple_gated_, ple_norm_, ple_history_;
  rt::Buffer qkv_, conv_out_, z_, alpha_, beta_, gdn_out_;
  rt::Buffer qg_, q_, gate_, k_, v_, ctx_, scores_, iq_, ik_, pooled_, block_scores_, visible_;
  rt::Buffer router_, expert_ids_, expert_weights_, moe_h_, moe_part_, hc_lo_part_, hc_inject_part_, hc_scale_, eg_, eu_, ed_, sgate_, logits_, lora_tmp_, lora_out_;
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
