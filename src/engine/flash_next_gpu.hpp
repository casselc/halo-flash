#ifndef HALO_ENGINE_FLASH_NEXT_GPU_HPP_
#define HALO_ENGINE_FLASH_NEXT_GPU_HPP_

// Qwen3.8-Flash-Next decode on the GPU (P2: correctness first), mirroring
// halo::model::ReferenceModel step by step: hyper-connection mixing, PLE
// n-gram injection (rows gathered on the CPU from the on-disk table), Gated
// DeltaNet, gated GQA with QSA block selection, and softmax top-k MoE with the
// shared expert. Routing and QSA block selection run on device with the
// reference's tie rules (equal scores keep the lower index), so a step
// needs no host round trip except the n-gram rows and the final logits.

#include <functional>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/sampling.hpp"
#include "model/lora.hpp"
#include "model/ngram.hpp"
#include "model/weights.hpp"
#include "rt/device.hpp"
#include "rt/file_upload.hpp"

namespace halo::engine {
struct MtpProposal;  // mtp_sampling.hpp


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
  /// Decode attention with batched loads (halo_attention_select_b; bit-identical to halo_attention_select).
  static inline bool decode_attn_batched_ = true;
  /// Stream D2 decode options, one bit each, every one bit-identical to the launches it replaces (0 is main's
  /// decode; docs/ROOFLINE.md section 10 has the measured bits that were not kept and their numbers).
  enum DecodeOpt : unsigned {
    kDecodeQ5kExperts = 1u,       ///< the Q5_K layer's experts in 3 launches (halo_moe_gate_up_q5_k_rows)
    kDecodeIndexerDual = 2u,      ///< QSA indexer q + k projections in one launch (halo_matvec_bf16_dual)
    kDecodeHcMixLds = 8u,         ///< hc_mix_a stages only its column chunk (halo_hc_mix_a_c)
    kDecodeQsaScores = 64u,       ///< QSA block scores with q in LDS and the heads' chains side by side
    kDecodeShexpCombine = 256u,   ///< routed combine folded into the shared expert's down launch
    kDecodeSplitGraph = 2048u,    ///< graph split before the PLE layer: the n-gram read overlaps the layers before it
    kDecodeQ8ActOnce = 32768u,    ///< routed gate/up activations quantized once, in the router launch
    kDecodeHoistRowLoads = 65536u,  ///< block-per-row F32 dots (router, alpha/beta, shexp gate) issue all loads first
    kDecodeHoistSmall = 131072u,    ///< norm scales, combines, the mixers' prologues: loop loads issued first
    kDecodeGdnNoSpill = 262144u,    ///< DeltaNet step reads its state column twice instead of spilling it
    kDecodeTopkWarp = 524288u,      ///< router top-k by warp shuffles (halo_router_topk_w)
    /// Stream C: QSA selection on one 256-thread block (halo_qsa_select256: the prefill's wave-digit radix
    /// select and ballot compaction, same lists) instead of the 1024-thread kernel's serial digit walk and
    /// Hillis-Steele scans (20 barriers per 1024 blocks: grows with the context).
    kDecodeSelect256 = 8388608u,
    /// Stream C, not kept: QSA block scores on one thread per block of the whole context (grid ceil(max_blocks /
    /// 256)) instead of at most 64 x 256 threads looping over the blocks. 327 -> 439 us a call at 256K (slower).
    kDecodeScoresWide = 16777216u,
  };
  static inline unsigned decode_opt_ = kDecodeQ5kExperts | kDecodeIndexerDual | kDecodeHcMixLds | kDecodeQsaScores |
                                       kDecodeShexpCombine | kDecodeSplitGraph | kDecodeQ8ActOnce | kDecodeHoistRowLoads |
                                       kDecodeHoistSmall | kDecodeGdnNoSpill | kDecodeTopkWarp | kDecodeSelect256;
  static inline bool small_row_kernels_ = true;  ///< P3 block-per-row F32 matvec for few rows
  /// P3 expert gate+up with int8 activations (llama.cpp-style Q8_1); false
  /// keeps the float kernel. A quality-gated choice (docs/BENCHMARKS.md).
  static inline bool int8_act_moe_ = true;
  static inline bool fused_hc_ = true;  ///< P3 hyper-connection mixer in two launches
  static inline bool fused_shexp_ = true;  ///< P3 shared expert in two launches
  static inline bool fused_linear_ = true;  ///< P3 DeltaNet block in five launches
  static constexpr std::uint32_t kHcParts = 8;  ///< column chunks of the fused mixer's down projection
  /// Prompt tokens per prefill chunk (buffers are sized for the smaller of
  /// this and the context when first needed). Large chunks let each routed
  /// expert see enough tokens to reuse its weights.
  static inline std::uint32_t prefill_chunk_ = 2048;
  /// A prompt longer than prefill_chunk_ is prefilled in chunks of this many tokens instead (0: off). The chunk
  /// buffers grow to it on the first such prompt (~4.3 GB more at 8192), so short prompts never pay for it. With
  /// prefill_gemm_slice_ and prefill_dense_prefix_ the logits are bit-identical to 2048-token chunks; at 8K
  /// tokens the prefill is ~370 ms (7%) faster (stream L, docs/ROOFLINE.md section 11).
  static inline std::uint32_t prefill_long_chunk_ = 8192;
  /// GEMMs whose per-token cost rises with the token count (their activations stop fitting the MALL past
  /// ~2048 tokens: `_l`, `_s`, the router and the small fp32 projections) run on slices of this many tokens
  /// when a chunk is longer (0: never). Per-token results do not depend on the slicing.
  static inline std::uint32_t prefill_gemm_slice_ = 2048;
  /// A chunk crossing the end of the dense range runs its dense prefix on the flash kernel (as 2048-token
  /// chunks do) instead of all of it on the QSA path.
  static inline bool prefill_dense_prefix_ = true;
  static inline bool batched_prefill_ = true;
  /// Debug: chunk projections through the decode GEMVs (token by token)
  /// instead of the int8 GEMMs, so a chunk equals stepping up to float order.
  static inline bool prefill_exact_projections_ = false;
  /// Repacked Q8_0 projections as int8 WMMA GEMMs (activations quantized per
  /// token and 32 values); off, they run through the decode GEMVs per token.
  static inline bool prefill_int8_gemm_ = true;
  /// Repacked Q8_0 projections on FP16 WMMA (weights dequantized while
  /// staging, fp16 activations); takes precedence over prefill_int8_gemm_.
  static inline bool prefill_f16_gemm_ = true;
  /// Routed experts of a chunk grouped by expert (one GEMM per expert's
  /// tokens) instead of token by token.
  static inline bool prefill_grouped_moe_ = true;
  /// Expert bucketing in three parallel passes instead of one block.
  static inline bool prefill_parallel_bucket_ = true;
  /// F32 projections of <= 16 rows (mixer inject, shared-expert gate): one
  /// warp per token instead of 64-row tiles. Off: predicted -100 ms,
  /// measured +490 ms per 2048-token prefill (~3.4 ms per call).
  static inline bool prefill_skinny_f32_ = false;
  /// The combine after a block fused with the next mixer's norm (bit-identical).
  static inline bool prefill_fused_combine_ = true;
  /// The mixers' F32 inject projection inside the norm that produces xn
  /// (per-stream partial products, then a reduce) instead of a GEMM.
  static inline bool prefill_fused_inject_ = true;
  // combine_norm skips the fp32 xn the mixer's mix epilogue would re-read;
  // the epilogue rebuilds it from res (same arithmetic, bit-identical).
  static inline bool prefill_mix_rebuild_ = true;
  static inline bool prefill_blocked_xh_ = true;  // the mixers' fp16 streams K-blocked for the down GEMM
  static inline bool prefill_banded_gemm_ = true;  // large dense GEMMs in bands of 8 row tiles
  static inline bool prefill_reg_gemm_ = true;  // banded GEMMs with weights in registers (halo_gemm_q8r_f16_lr8)
  static inline bool prefill_fused_moe_combine_ = true;  // routed combine + shared expert in one pass
  /// combine_norm with one block per token (all four streams; halo_hc_combine_norm_tok, bit-identical).
  static inline bool prefill_tok_combine_ = false;  // attention half: 56.9 vs 55.3 ms per prefill, not kept
  /// ... which after the MoE block also builds the routed + shared experts' sum itself (_tok_moe).
  static inline bool prefill_moe_in_combine_ = true;
  static inline bool prefill_moe_vec_ = true;  // ... reading the expert rows 16 bytes at a time (_tok_moev)
  /// Attention q: split, norm and rope in one kernel (halo_split_qgate_norm_rope, bit-identical).
  static inline bool prefill_q_fused_ = true;
  static inline bool prefill_kq_fused_ = true;
  static inline bool prefill_gated_wave_ = true;
  static inline bool prefill_gate_in_place_ = true;  // flash attention reads the gate from qg (halo_attention_fa_hg)  // DeltaNet gated norm one wave per row (halo_gated_norm_h128w)  // k and indexer q: norm + rope (+ k cache append) in one kernel
  /// 64-row tiles (halo_gemm_q8r_f16_s) up to 640 rows instead of 512: the shared expert's 640-row gate/up
  /// otherwise take three 256-row tiles (the last half empty).
  static inline bool prefill_s640_ = true;
  /// Producers whose fp32 output only feeds an FP16 GEMM write the fp16 copy directly (attention context,
  /// mixer SiLU, shared-expert SiLU product) instead of a separate halo_f16_rows pass.
  static inline bool prefill_f16_out_ = true;
  static std::uint32_t SmallRows() { return prefill_s640_ ? 640 : 512; }
  static inline bool prefill_ple_overlap_ = true;
  /// Prototype: Q4_K gate/up experts repacked K-blocked on first prefill use (decode does not read that layout).
  static inline bool prefill_moe_blocked_ = false;
  /// Span tracing (docs/TRACING.md): when set, the engine loads this code object (core_trace.hsaco) instead
  /// of the embedded one, and after SpanArm() brackets the span_occurrence_-th launch (from 0) of kernel
  /// span_kernel_ with halo_span_set(1) / halo_span_set(0).
  static inline std::vector<std::uint8_t> span_hsaco_;
  static inline std::string span_kernel_;
  static inline std::int64_t span_occurrence_ = 0;
  void SpanArm() { span_seen_ = 0; }
  [[nodiscard]] std::int64_t SpanSeen() const { return span_seen_; }  // read the n-gram rows while the layers before the PLE run
  /// DeltaNet over 16-token chunks (prep + scan) instead of token by token.
  static inline bool prefill_gdn_chunked_ = true;
  static inline bool prefill_gdn_chunk_fp32_ = false;  // study: the fp32 chunked kernels instead of c16
  static inline bool prefill_gdn_c16_bf16_ = false;    // study: the two-way bf16 scan (fails the long-set gate)
  static inline bool prefill_gdn_c16_u3_ = true;       // scan3u (update's K three-way); false: scan3, +0.0013 NLL
  /// Read the next prefill chunk's n-gram rows on a host thread while the
  /// current chunk runs on the device.
  static inline bool prefill_ngram_prefetch_ = true;
  /// The chunked QSA attention reads an fp16 copy of the K/V cache (same
  /// results; ~24.6 KB more per cached token).
  static inline bool prefill_kv16_ = true;
  /// Stream C (precision option, default off): the fp16 K/V rows are the only K/V store. No fp32 K/V is kept
  /// (48 KB a token less: ~32 KB instead of ~81 KB with today's indexer keys); decode attention reads fp16 rows
  /// (halo_attention_select_bh), prefill appends fp16 rows directly and reads them as it does with prefill_kv16_.
  /// Read at Create (it decides the sessions' KV kinds).
  static inline bool kv_f16_only_ = false;
  /// Stream C (exact): raw indexer keys are read only while their QSA block is incomplete (pooling), so pages of
  /// them wholly below the current block are released (paged KV): 6.1 KB a token less (~1.6 GB at 262K).
  /// Snapshots keep the live rows only. Read at Create.
  static inline bool kv_index_window_ = true;
  /// QSA block scores of a chunk as one bf16x3 WMMA GEMM (near-fp32).
  static inline bool prefill_scores_wmma_ = true;
  /// Stream C (exact): the block scores with each block-key tile split once per block in LDS and q split once
  /// (halo_qsa_scores_wmma_b; same WMMAs per score in the same order).
  static inline bool prefill_scores_b_ = true;
  /// Past the QSA budget, attention for 4 tokens at a time over the union of
  /// their selections (shared K/V reads) instead of token by token.
  static inline bool prefill_qsel_attention_ = true;
  // QSA top-k selection on 256-thread blocks (halo_qsa_select_rows256; same lists as the 1024-thread kernel).
  static inline bool prefill_select256_ = true;
  // QSA attention with V packed in registers and 16-byte softmax LDS accesses (halo_attention_wmma_qsel_h2; bit-identical).
  static inline bool prefill_qsel_h2_ = true;
  /// The router GEMM on bf16 WMMA with hi/lo-split operands (~2^-15 per
  /// product) instead of fp32 VALU.
  static inline bool prefill_router_bf16x3_ = true;
  /// Precision option (stream P): the router GEMM on plain FP16 WMMA (weights
  /// rounded to fp16 while staging, the fp16 copy of `mixed`, fp32 sums)
  /// instead of bf16x3. Takes precedence over prefill_router_bf16x3_.
  /// 1: plain fp16; 2: weights split hi/lo (two WMMAs), fp16 `mh`; 3: fp16
  /// weights, fp32 `mixed` split hi/lo.
  /// 4 (stream R): mode 3's numbers, with fp16 weights converted once and the activation's hi / lo halves
  /// written by the mix epilogue (mh, ml), on 128 x 64 tiles (halo_gemm_h_hl_128x64).
  static inline int prefill_router_f16_ = 3;
  /// Stream R study: the mode-4 router kernel's tile (0: 128 x 64, BK 64; 1: 128 x 64, BK 32; 2: 64 x 64).
  static inline int prefill_router_tile_ = 0;
  /// Stream R (exact, with router mode 4): the MoE-half mix epilogue skips the fp32 `mixed` nobody reads.
  static inline bool prefill_mixed32_skip_ = true;
  /// Precision option (stream P): the small F32 projections over `mixed`
  /// (DeltaNet alpha/beta, shared-expert gate) on FP16 WMMA like the router
  /// option, instead of the fp32 tiled GEMM.
  static inline bool prefill_small_f16_ = true;
  /// Precision option (stream P): the mix epilogue reads xn as the norm's fp16
  /// K-blocked copy (xh) instead of rebuilding it in fp32 from the residual.
  static inline bool prefill_mix_h16_ = true;
  /// Precision option (stream P): the shared expert's gate and up as W8A8
  /// (halo_gemm_q8a8_128x128, activations quantized per token and 32 values).
  static inline bool prefill_shexp_w8a8_ = false;
  /// Precision option (stream P): the mixers' down projection as W8A8
  /// (halo_gemm_q8a8_160x64 over an int8 copy of xn written by combine_norm).
  static inline bool prefill_down_w8a8_ = false;
  /// Precision option (stream P): DeltaNet qkv and z GEMM outputs in fp16,
  /// read as fp16 by the conv, the conv state advance and the gated norm.
  static inline bool prefill_gdn_io16_ = false;
  /// Precision option (stream R): the whole DeltaNet data path in fp16 (implies prefill_gdn_io16_'s GEMMs):
  /// the conv writes fp16 q/k/v, prep and scan read them (K and Q exact in fp16: 2 WMMAs a product), the
  /// scan writes o in fp16, the one-wave gated norm reads fp16 o and z. 1: halo_gdn_c16_scanf (K^T
  /// gathered), 2: halo_gdn_c16_scanft (K^T from a transposed copy).
  static inline int prefill_gdn_f16_ = 1;
  /// Precision option (stream P): the chunk's hyper-connection residual
  /// streams stored in 16 bits (1: fp16, 2: bf16) instead of fp32; every
  /// residual reader and writer of the prefill takes the matching variant.
  static inline int prefill_res16_ = 1;
  /// Precision option (stream P): the chunked DeltaNet scan with fp16 two-way splits for K S0, Q S0
  /// and the update (halo_gdn_c16_scanh) instead of scan3u's bf16 three-way.
  static inline bool prefill_gdn_scan_h_ = true;
  /// Attention of a chunk's tokens in batched launches (same arithmetic as
  /// stepping) instead of token by token.
  static inline bool prefill_batched_attention_ = true;
  /// The batched attention on FP16 WMMA tiles with an online softmax (head
  /// dim 256); off, the decode kernel's arithmetic per token.
  static inline bool prefill_wmma_attention_ = true;
  /// While every token sees every earlier position (under the QSA budget),
  /// 4 tokens per attention block share K/V and QSA selection is skipped.
  static inline bool prefill_mq_attention_ = true;
  static inline bool prefill_fa_attention_ = true;  // dense causal: halo_attention_fa (needs the fp16 K/V copy)
  /// DeltaNet recurrence with each state column over 4 lanes (2 blocks per
  /// head), its inputs staged and prefetched per 8 tokens, and a separate
  /// gated norm; exact mode keeps the fused kernel, bit-identical to
  /// stepping. In the real workload 389 -> ~206 ms per 2048-token prefill
  /// (A/B on device time); a lone-kernel bench understates this, running at
  /// ~950 MHz instead of ~2.7 GHz.
  static inline bool prefill_split_gdn_ = true;
  /// Test only (positive controls for FlashNextGpu.ChunkedPrefillMatchesStepping): 1 embeds the last row of every
  /// prefill chunk but a call's final one as the next token id (a one-row error at each chunk boundary inside the
  /// prompt); 2 moves every element of every prefill row's token embedding by prefill_test_ulps_ ulps, away from or
  /// toward zero by a hashed choice per column (a systematic tiny error). 0 (default): off.
  static inline int prefill_test_perturb_ = 0;
  static inline int prefill_test_ulps_ = 4;
  /// Runs `tokens` in order; `logits` (optional) receives the last position's.
  /// P4: in chunks of prefill_chunk_ tokens, the dense projections as T-token
  /// GEMMs and the rest row-batched or token by token (prefill_chunk.cpp);
  /// a token at a time when batched_prefill_ is off or the model's shapes
  /// do not fit the chunk kernels. `cancelled` (optional) is polled between
  /// chunks (each chunk ends synchronized, so at most one chunk runs after
  /// it turns true); then the call fails with error "cancelled" and the
  /// bound session's state is partly advanced (drop or reset it).
  [[nodiscard]] bool Prefill(std::span<const std::int32_t> tokens, std::span<float> logits, std::string* error,
                             const std::function<bool()>& cancelled = {});
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  /// Graph replay (default) or eager launches; both run the same sequence.
  void SetUseGraphs(bool on) noexcept { use_graphs_ = on; }

  /// Sync-free per-kernel timing (profiling). Between Start and Stop, every
  /// kernel the engine launches eagerly is followed by a one-thread kernel
  /// that writes the GPU's constant-rate counter; Stop synchronizes, reads
  /// the stamps and returns each (kernel name, grid) key's total time and
  /// call count, longest first. Unlike a synchronize after every launch,
  /// this keeps the clocks and back-to-back execution of the real workload
  /// (about 5 us per stamp is added). Use around Prefill; decode graphs are
  /// not captured while timing is on. `ticks_per_ms` converts counter ticks
  /// (measured against host time when 0: pass the device's known rate when
  /// it is, 100 MHz on gfx11 -> 1e5).
  struct KernelTime {
    std::string key;
    double ms = 0;
    std::uint64_t calls = 0;
  };
  [[nodiscard]] bool StartKernelTiming(std::size_t capacity, std::string* error);
  [[nodiscard]] std::vector<KernelTime> StopKernelTiming(double* span_ms, double* ticks_per_ms, std::string* error);
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

  /// Routing trace (analysis): when on before the first step (graphs are captured with it), every step
  /// copies each layer's routed expert ids into a device buffer; RouteTrace() downloads the last step's
  /// ids, [num_layers][num_experts_used].
  static inline bool route_trace_ = false;

  // ---- P6 sessions (docs/SESSIONS.md) ----
  /// The engine holds any number of sequences. One is bound at a time: Step, StepSample, Prefill, Save,
  /// Restore, Reset and Position act on it. Binding swaps host-side handles only (no device copies): each
  /// session owns its recurrent state, its KV pages and its captured decode graphs. Session 0 exists from
  /// Create and is bound initially, so single-sequence callers see no change.
  using SessionId = std::uint32_t;
  [[nodiscard]] bool NewSession(SessionId* id, std::string* error);
  /// An exact copy of `src` at its current position: recurrent state copied, KV pages below the position
  /// shared copy-on-write (the page holding the position is copied; later writes go to fresh pages).
  [[nodiscard]] bool ForkSession(SessionId src, SessionId* id, std::string* error);
  [[nodiscard]] bool BindSession(SessionId id, std::string* error);
  /// Frees a session's state, graphs and its references to shared KV pages. The bound session cannot be dropped.
  [[nodiscard]] bool DropSession(SessionId id, std::string* error);
  [[nodiscard]] SessionId BoundSession() const noexcept { return bound_; }
  [[nodiscard]] bool HasSession(SessionId id) const noexcept;
  [[nodiscard]] std::uint32_t SessionPositionOf(SessionId id) const;
  /// Device bytes of physical memory the session's state holds, counting shared KV pages once per holder
  /// (`exclusive`: pages only this session references) - for the memory tables.
  struct SessionBytes {
    std::size_t recurrent = 0, kv_mapped = 0, kv_exclusive = 0;
  };
  [[nodiscard]] SessionBytes SessionMemory(SessionId id) const;
  /// Stream C: whether the bound session can grow to `positions` (prompt and generation) in the memory the device
  /// reports left (rt::Device::AvailableBytes), counting its KV kinds' bytes per token beyond the pages it already
  /// maps and the long-prompt chunk buffers a prefill of `prefill_tokens` would add; `why` says what is short.
  [[nodiscard]] bool KvRoomFor(std::uint32_t positions, std::uint32_t prefill_tokens, std::string* why) const;
  /// Device bytes one more position costs the bound session (its KV kind table; indexer keys windowed).
  [[nodiscard]] double KvBytesPerToken() const;

  // ---- P5 MTP draft layer (mtp.cpp), measurement stage: no verify path, the trunk's numerics untouched ----
  /// Uploads the draft block (the `shared-*` sidecar). Its attention cache holds max_context positions.
  [[nodiscard]] bool LoadMtp(const model::MtpWeights& mtp, std::string* error);
  /// One draft-block step at MtpPosition() for `token` (the token after the hidden's position): the hidden input
  /// is the bound session's trunk residual after its last Step (`from_trunk`) or the draft block's own previous
  /// output (chained drafts), as ReferenceModel::MtpStep. Logits of the drafted next token when non-empty.
  /// Advances MtpPosition(); MtpRewind(p) moves it back (later steps overwrite the cache rows).
  [[nodiscard]] bool MtpStep(std::int32_t token, bool from_trunk, std::span<float> logits, std::string* error);
  void MtpRewind(std::uint32_t position) noexcept { mtp_pos_ = position; }

  // ---- P5 speculative decoding (spec_decode.cpp; docs/MTP.md) ----
  /// Keeps the draft block's history in step with the bound session from now on: every Step / StepSample /
  /// Prefill also feeds the draft block (its attention cache trails the trunk by one position: the last trunk
  /// position's hidden waits for the next token). Turn on right after Reset (history starts at position 0).
  /// The trunk's results do not change.
  [[nodiscard]] bool MtpTrack(bool on, std::string* error);
  /// Tracking for the bound session with a history window (the server, docs/MTP.md item 2b): the draft block
  /// follows from trunk position `start` (>= the session's position; earlier positions' hiddens are gone), as
  /// if the sequence began there (its own positions count from `start`: attention is relative under RoPE). Prefill
  /// rows and steps before `start` do not feed it, so a prompt costs draft steps only for its last positions.
  /// Drafting resumes once the draft block holds the hidden of the position before the one being run.
  [[nodiscard]] bool MtpTrackFrom(std::uint32_t start, std::string* error);
  [[nodiscard]] bool MtpTracking() const noexcept { return mtp_track_; }
  /// Trunk position where the tracked draft history starts (kMtpNever: not following).
  [[nodiscard]] std::uint32_t MtpBase() const noexcept { return mtp_base_; }
  static constexpr std::uint32_t kMtpNever = 0x7fffffffu;
  struct SpecResult {
    std::vector<std::int32_t> tokens;  ///< emitted after `token`, in order; the last one is not yet run
    std::uint32_t drafted = 0, accepted = 0;
    std::uint64_t rng = 0;  ///< greedy: the request's RNG passed through
    std::vector<std::int32_t> drafts;  ///< the proposed drafts d1 .. dk (diagnostics, the sampling audit)
  };
  /// One speculative cycle. Like StepSample(token): runs `token` and returns what follows it, here 1 + accepted
  /// tokens. `drafts` tokens are drafted with the MTP block (trimmed head when SetDraftVocab was called) and
  /// verified in one batched pass whose columns are byte-identical to single steps; rejected rows' recurrent
  /// state is rolled back from per-row snapshots (no replay).
  /// Greedy (`sampler` null): device greedy sampling per verified row: the emitted tokens are exactly
  /// StepSample's. Sampled (`sampler` set, no penalties): host p / q acceptance with residual correction.
  /// drafts == 0, or no draft history yet: an ordinary step (the draft block still catches up).
  [[nodiscard]] bool SpecStep(std::int32_t token, std::uint32_t drafts, sampling::SamplerState* sampler,
                              SpecResult* result, std::string* error);
  static constexpr std::uint32_t kMaxDrafts = 7;
  /// Test hook (greedy): one cycle like SpecStep with the caller's drafts instead of the draft block's; the same
  /// verify, acceptance and rollback. Needs the batch path and the draft block not tracking this session.
  [[nodiscard]] bool SpecStepDrafts(std::int32_t token, std::span<const std::int32_t> drafts, SpecResult* result,
                                    std::string* error);
  /// The draft block's per-session state (position and pending hidden) for Save / Restore around SpecStep:
  /// Snapshot covers the trunk only.
  struct MtpCheckpoint {
    std::uint32_t pos = 0, base = 0;
    std::vector<float> hpend;
  };
  [[nodiscard]] bool MtpSave(MtpCheckpoint* cp, std::string* error);
  [[nodiscard]] bool MtpRestore(const MtpCheckpoint& cp, std::string* error);
  /// Draft head restricted to `ids` (FR-Spec style, draft logits only): MtpStep then fills logits.size() ==
  /// ids.size() entries, entry i the logit of token ids[i] (the same rows of the LM head, same GEMV). Empty: full.
  [[nodiscard]] bool SetDraftVocab(std::span<const std::int32_t> ids, std::string* error);
  [[nodiscard]] std::uint32_t MtpPosition() const noexcept { return mtp_pos_; }
  /// The bound session's trunk residual (hc_dim floats) after its last Step.
  [[nodiscard]] bool TrunkHidden(std::vector<float>* out, std::string* error);

  // ---- PROBE ONLY (probe.cpp): batch-shaped WMMA kernels timed in the engine; never part of a decode ----
  /// Routed expert ids for the probe's B columns: ids[layer][b * k + slot].
  [[nodiscard]] bool ProbeSetRouting(std::uint32_t B, const std::vector<std::vector<std::uint32_t>>& ids,
                                     std::string* error);
  /// Launches every layer's batch-shaped dense GEMVs (+ split-K reduces), routed experts (grouped, routing from
  /// ProbeSetRouting), shared expert, mixers' projections and the LM head for B columns. Results are discarded.
  /// `check`: also compares the first qkv projection's column 0 with the decode GEMV (relative L2 printed).
  [[nodiscard]] bool ProbeRun(std::uint32_t B, bool check, std::string* error);

  /// One decode step for several sessions at once (P6 batched decode): every weight is read once for the
  /// batch, routed experts grouped across sessions. Each entry's logits / sampled token / state are byte-identical
  /// to a Step / StepSample of that session alone. Sessions must be distinct; 1..kMaxBatch entries; every
  /// entry the same kind (logits wanted, sampled, or neither). Shapes or options the batched kernels do not
  /// cover run the entries one after another (same results).
  static constexpr std::uint32_t kMaxBatch = 8;
  struct BatchEntry {
    SessionId session = 0;
    std::int32_t token = 0;
    std::span<float> logits;                   ///< filled when non-empty
    const SampleRequest* sample = nullptr;     ///< sampled on the device when set (then `result` too)
    SampleResult* result = nullptr;
  };
  [[nodiscard]] bool StepBatch(std::span<const BatchEntry> entries, std::string* error);
  /// Whether StepBatch would run the batched kernels (else it steps the entries in turn).
  [[nodiscard]] bool BatchSupported() const;
  [[nodiscard]] std::string BatchUnsupported() const;  ///< why not (empty: supported)
  static inline bool batch_decode_ = true;  ///< false: StepBatch always steps the entries in turn
  [[nodiscard]] bool RouteTrace(std::vector<std::uint32_t>* ids, std::string* error);

 private:
  struct DevTensor {
    rt::Buffer buf;
    rt::KernelId matvec = -1, embed = -1, expert = -1, embed_rows = -1;
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
  /// One KV tensor of one attention layer: a virtual range for max_context positions whose physical pages are
  /// mapped on demand and shared between forks (rt::Device paging), or one allocation at full capacity when the
  /// backend cannot map pages. `row_bytes` per `row_tokens` positions (block_k: one row per compress_ratio).
  struct KvPage;
  /// What a KV region holds (the kind table of a session; docs/SESSIONS.md).
  enum class KvKind : std::uint8_t { kK, kV, kIndexK, kBlockK, kK16, kV16 };
  struct KvRegion {
    KvKind kind = KvKind::kK;
    std::uint32_t layer = 0;
    rt::Buffer va;  // device address range of the whole capacity
    std::size_t row_bytes = 0;
    std::uint32_t row_tokens = 1;
    std::vector<std::shared_ptr<KvPage>> pages;  // mapped pages, from offset 0 (paged mode); null: released
    bool paged = false;
  };
  struct AttnState {
    rt::Buffer k, v, index_k, block_k;
    std::uint32_t blocks = 0;
    // fp16 copy of k and v for the chunked QSA attention (which rounds K/V
    // to fp16 anyway): positions [0, valid16) are current. Filled lazily
    // before use; Restore and Reset set valid16 to 0.
    rt::Buffer k16, v16;
    std::uint32_t valid16 = 0;
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
  /// part 0: the whole step; 1: embedding and the layers before the PLE layer; 2: the rest.
  bool Record(Head head, std::string* error, int part = 0);
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
    rt::Buffer sorted, pos, tiles, ntiles, hp, yp, x16, bcounts, bbase;
    rt::Buffer pa, pb;  // PLE key and normalized rows
    rt::Buffer ipart;   // inject partial products [T][hc][rows]
    rt::Buffer rscale;  // per-row norm scale [T][hc] when combine_norm skipped writing xn
    rt::Buffer xh, mh, lh, ml;  // ml: fp16 lo of `mixed` (router mode 4); fp16 activations: per GEMM, the shared `mixed`, the mixer lo (mix_h16)  // grouped experts (hp, x16 fp16)
    rt::Buffer bscores, vis, ascores;                // batched attention, kAttnSub tokens at a time
    rt::Buffer ublk, umask, ucount;                  // QSA unions of 4 tokens (halo_qsa_union4)
    rt::Buffer qsh, qsl;                             // a sub-chunk's indexer q split to bf16 hi / lo (scores_b)
    rt::Buffer gdnprep;                              // chunked DeltaNet: per (head, 16-token chunk) halo_gdn_c16_prep
    std::size_t params_offset = 0;  // in `host`: per-token step params after the n-gram rows
    std::uint32_t cap = 0;  // tokens the buffers hold
    std::uint32_t bstride = 0;  // bscores row stride (QSA blocks): grown with the position, not max_context
    std::size_t bytes = 0;      // device bytes of the chunk buffers (for KvRoomFor)
  };
  static constexpr std::uint32_t kAttnSub = 512;
  std::uint32_t AttnVisibleStride() const;
  ChunkBuffers ck_;
  /// Grows ck_.bscores so a row holds `blocks` block scores (stream C: sized by the position reached).
  bool GrowBlockScores(std::uint32_t blocks, std::string* error);
  bool ck_ready_ = false;
  std::vector<std::uint8_t> ck_host_;
  bool ChunkSupported() const;
  /// Chunk buffers for `tokens`-token chunks (capped at the context); grows them when they hold fewer.
  bool AllocChunk(std::uint32_t tokens, std::string* error);
  void FreeChunk();
  // pre_emb: this chunk's n-gram rows already read (by the previous chunk's
  // prefetch); after_prep(history) runs once the chunk's host data is
  // uploaded, with the n-gram history after its tokens (for the next
  // chunk's prefetch, overlapping this chunk's device work).
  bool PrefillChunk(std::span<const std::int32_t> tokens, std::span<float> logits, std::string* error,
                    const std::vector<float>* pre_emb = nullptr,
                    const std::function<void(const model::NgramHistory&)>& after_prep = {});
  /// y[T][rows] = W x[T][cols]: int8 GEMM for repacked Q8 (x quantized here),
  /// BF16 / F32 GEMMs, else the decode GEMV per token.
  /// x16: x already converted to fp16 rows (shared by several GEMMs), or null.
  // An F32 projection of `mixed` on FP16 WMMA when prefill_small_f16_ allows it, else Gemm.
  bool MixedGemmF16(const model::TensorRef& t, const rt::Buffer& y, std::uint32_t tokens, std::string* error);
  bool Gemm(const model::TensorRef& t, const rt::Buffer& x, const rt::Buffer& y, std::uint32_t tokens,
            std::string* error, const rt::Buffer* x16 = nullptr, bool x16_blocked = false);
  bool HcMixChunk(const model::HcMixer& m, bool inject, std::uint32_t tokens, std::string* error, bool normed = false);
  bool PleChunk(const model::LayerWeights& l, std::uint32_t tokens, std::string* error);
  bool LinearChunk(const model::LayerWeights& l, LinearState& s, std::uint32_t tokens, std::string* error);
  bool AttentionBatched(const model::LayerWeights& l, AttnState& s, std::uint32_t T, std::string* error);
  bool AttentionChunk(const model::LayerWeights& l, AttnState& s, std::uint32_t tokens, std::string* error);
  bool F16Route(const model::TensorRef& t) const;
  bool TokCombineFits(const model::HcMixer& m) const;
  bool TokCombine(const model::HcMixer& m) const;
  bool MixH(const model::HcMixer& m) const;
  bool BlockedXh(const model::HcMixer& m) const;
  bool FuseInject(const model::HcMixer& m) const;
  bool MixFused(const model::HcMixer& m) const;
  bool MixUp(const model::HcMixer& m, std::uint32_t T, std::string* error);
  bool Half(const rt::Buffer& x, const rt::Buffer& out, std::uint32_t n, std::string* error);
  /// f(t0, n) over the token slices of a chunk (prefill_gemm_slice_); one call for short chunks.
  template <class F>
  bool ForGemmSlices(std::uint32_t T, F&& f) {
    const std::uint32_t S = prefill_gemm_slice_;
    if (S == 0 || T <= S) return f(0u, T);
    for (std::uint32_t t0 = 0; t0 < T; t0 += S)
      if (!f(t0, T - t0 < S ? T - t0 : S)) return false;
    return true;
  }
  bool MixedHalf(std::uint32_t T, std::string* error);
  bool RouterHL(const model::LayerWeights& l);
  const rt::Buffer* MixedH() const;
  bool Bucket(std::uint32_t pairs, std::uint32_t E, std::string* error);
  bool GroupedMoeFits(const model::LayerWeights& l) const;
  bool GroupedMoe(const model::LayerWeights& l, std::uint32_t T, std::string* error, bool combine = true);
  bool SharedExpertChunk(const model::LayerWeights& l, std::uint32_t T, std::string* error,
                         bool fused_combine = false);
  bool MoeChunk(const model::LayerWeights& l, std::uint32_t tokens, std::string* error);

std::unordered_set<const void*> moe_blocked_;
  std::unordered_map<const void*, rt::Buffer> router16_;  // fp16 router weights (prefill_router_f16_ 4)
  bool want_ml_ = false;
  bool router_hl_ok_ = false;
  bool mixed32_dead_ = false;  // this layer's MoE half reads no fp32 `mixed` (RouterHL)  // RouterHL held for this layer: mh / ml are the router's input  // the next mix epilogue also writes ml (the MoE-half mixer, router mode 4)  // tensors repacked K-blocked (prefill_moe_blocked_)

rt::Buffer moe_blocked_tmp_;
  rt::Buffer route_trace_buf_;
  std::uint32_t cur_layer_ = 0;

std::int64_t span_seen_ = -1;  // launches of span_kernel_ since SpanArm(); -1: not armed
    rt::Device& dev_;
  const model::ModelWeights& w_;
  model::NgramTable* ngram_;
  std::uint32_t max_context_ = 0, position_ = 0;
  bool kv16_only_ = false;  // kv_f16_only_ at Create
  bool index_window_ = false;  // kv_index_window_ at Create
  /// First raw indexer-key row any later step or chunk reads at `position` (its QSA block's start).
  std::uint32_t IndexLiveRow(std::uint32_t position) const {
    return index_window_ ? position / w_.config.compress_ratio * w_.config.compress_ratio : 0;
  }
  bool use_graphs_ = true;
  void Stamp(const std::string& key);  // after a kernel, while timing
  bool timing_ = false;
  rt::Buffer stamps_;
  std::vector<std::string> stamp_keys_;  // key of the kernel each stamp follows (0: the start)
  std::size_t stamp_cap_ = 0;
  std::chrono::steady_clock::time_point timing_start_;
  rt::Graph graphs_[3];  ///< by Head: none, logits, sample
  rt::Buffer q8act_;  ///< kDecodeQ8ActOnce: the layer's int8 activations (5 KB)
  rt::Graph pre_graph_, post_graphs_[3];  ///< kDecodeSplitGraph: Record parts 1 and 2
  float* ple_rows_host_ = nullptr;        ///< pinned n-gram rows for the split step's async upload
  SamplingMode sampling_mode_ = SamplingMode::kFast;
  /// One upload per step: n-gram rows, step parameters, sampling parameters,
  /// penalties (only the used prefix is sent).
  std::size_t params_offset_ = 0, sample_offset_ = 0, penalty_offset_ = 0;
  std::vector<std::uint8_t> step_host_;
  std::uint32_t sample_blocks_ = 0;
  rt::Buffer sample_block_out_, sample_candidates_, sample_key_, sample_rid_, sample_w_, sample_out_;
  model::NgramHistory ngram_history_;
  std::unordered_map<const void*, DevTensor> tensors_;
  // P4: the mixers' up projections with rows index-major (4 i + s), for the
  // GEMM fused with the stream mix; and whether ck_.mh holds `mixed` in fp16.
  struct PermutedUp {
    rt::Buffer qs, d;
  };
  std::unordered_map<const void*, PermutedUp> hc_up_perm_;
  bool mh_valid_ = false;
  bool xn_skipped_ = false;  // the last combine_norm wrote rscale instead of xn
  bool xq_valid_ = false;    // the last combine_norm wrote the int8 copy of xn (prefill_down_w8a8_)
  bool moe_defer_ = false;    // the next SharedExpertChunk leaves the routed + shared sum to combine_norm
  bool moe_pending_ = false;  // ... and it did: block_out was not written
  bool q_roped_ = false;
  bool kq_roped_ = false;
  bool gate_in_qg_ = false;   // this layer's attention gate is read in place from qg (gate_a not written)     // ... and k (cache appended) and the indexer q from halo_norm_rope_rows
  bool lo_half_ = false;      // the mixer's SiLU pass already left lo in fp16 (in xh)      // this layer's q rows came out of halo_split_qgate_norm_rope
  rt::FileUploader* loader_ = nullptr;  ///< during Init: weights read from their files
  std::unordered_map<const void*, DevLora> lora_;
  std::unordered_map<std::string, rt::KernelId> kernels_;
  // ---- P6: sessions. The bound session's state lives in the members below (linear_, attn_, ple_history_,
  // ngram_history_, position_ and the decode graphs); the others are parked in sessions_. Binding swaps.
  struct SeqState {
    std::vector<LinearState> linear;
    std::vector<AttnState> attn;
    rt::Buffer ple_history;
    model::NgramHistory ngram_history;
    std::uint32_t position = 0;
    rt::Graph graphs[3], pre_graph, post_graphs[3];
    std::vector<KvRegion> kv;  // every KV region of the session (attention layers x kinds)
  };
  std::vector<std::unique_ptr<SeqState>> sessions_;  // by id; null: dropped; sessions_[bound_] is the swapped-out shell
  SessionId bound_ = 0;
  std::vector<KvRegion> kv_;  // the bound session's KV regions
  void SwapBound(SeqState& s) noexcept;
  bool AllocSeq(SeqState& s, std::string* error);
  void FreeSeq(SeqState& s) noexcept;
  /// Maps KV pages so the bound session can hold `tokens` positions, read from `live` on (indexer-key pages wholly
  /// below IndexLiveRow(live) are released, those at or above it mapped).
  bool EnsureKv(std::uint32_t tokens, std::uint32_t live, std::string* error);
  /// Unmaps the bound session's KV pages that a fork also holds (Reset, Restore): those stay with the fork; the
  /// pages it alone holds stay mapped (rewritten in place).
  bool ReleaseKvPages(std::string* error);
  // ---- P6 batched decode (batch_decode.cpp) ----
  struct BatchBuffers {
    rt::Buffer res, xn, mixed, block_out, embd, inject, scale, lo_part, inject_part;
    rt::Buffer qkv, conv_out, z, alpha, beta, gdn_out;
    rt::Buffer qg, q, gate, k, v, ctx, iq, ik, pooled;
    rt::Buffer router, ids, wts, q8act, ucount, uexp, unum, upair, moe_h, moe_part, eg, sgate;
    rt::Buffer logits, sample_out, params;  // params: per entry, the single step's upload layout (entry_stride)
    std::size_t entry_stride = 0;
    std::vector<std::uint8_t> host;
  };
  BatchBuffers bb_;
  bool bb_ready_ = false;
  bool AllocBatch(std::string* error);
  bool BatchedMoe(const model::LayerWeights& l) const;
  struct BatchGraph {
    std::vector<SessionId> sessions;
    int head = 0;
    bool verify = false;
    rt::Graph graph;
    std::uint64_t used = 0;
  };
  std::vector<BatchGraph> batch_graphs_;
  std::uint64_t batch_clock_ = 0;
  bool RecordBatch(std::span<const SessionId> sessions, Head head, std::string* error);
  /// Points the per-token scratch members at entry b's slices of bb_ and binds its session (recording only).
  void BatchEntryScope(std::uint32_t b, std::span<const SessionId> sessions, bool enter);
  bool skip_attn_proj_ = false;  // Attention(): projections done by the batched caller
  rt::Buffer probe_x_, probe_part_, probe_y_, probe_h_, probe_ep_, probe_tab_;
  std::uint32_t probe_B_ = 0;
  std::size_t probe_tab_stride_ = 0;
  const model::MtpWeights* mtp_ = nullptr;
  AttnState mtp_attn_;
  rt::Buffer mtp_res_, mtp_in_, mtp_hn_, mtp_cat_;
  std::uint32_t mtp_pos_ = 0;
  bool MtpRun(std::int32_t token, const rt::Buffer* hidden, std::span<float> logits, std::string* error);
  // Speculative state: tracking flag, the session it tracks, the last trunk position's hidden (pending), and the
  // per-row recurrent snapshots of a verify pass (verify_snap_[row]: one buffer per DeltaNet layer's state and
  // conv, then the PLE history).
  bool mtp_track_ = false;
  SessionId mtp_owner_ = 0;
  rt::Buffer mtp_hpend_;
  std::vector<std::vector<rt::Buffer>> verify_snap_;
  bool verify_ = false;  // RecordBatch: the entries are consecutive rows of one session; snapshot after each
  // The draft block has run trunk positions [mtp_base_, mtp_base_ + mtp_pos_) and holds the hidden of the next
  // one in mtp_hpend_: pending when that is position_ - 1 (it waits for the token at position_), following
  // (the hidden of position_ is the next to keep) when mtp_base_ + mtp_pos_ == position_.
  std::uint32_t mtp_base_ = 0;
  bool MtpPending() const noexcept {
    return mtp_track_ && bound_ == mtp_owner_ && position_ > 0 && mtp_base_ + mtp_pos_ + 1 == position_;
  }
  bool MtpKeepsHidden() const noexcept { return mtp_track_ && bound_ == mtp_owner_ && mtp_base_ + mtp_pos_ == position_; }
  bool MtpAfterPrefillRows(std::span<const std::int32_t> tokens, std::string* error);
  bool AllocVerifySnaps(std::uint32_t rows, std::string* error);
  /// SpecStep / SpecStepDrafts after drafting: verify rows_tok = [token, d1 .. dk] at position_ .. + k, accept,
  /// commit, roll back, and (when tracking) the draft block's catch-up. `props`: sampled mode's proposals.
  bool SpecVerify(std::span<const std::int32_t> rows_tok, std::span<const MtpProposal> props,
                  sampling::SamplerState* sampler, SpecResult* result, std::string* error);
  rt::Buffer draft_qs_, draft_d_;
  std::uint32_t draft_rows_ = 0;
  std::vector<std::int32_t> draft_ids_;
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
