# Hardware/model co-design for Strix Halo

The questions:
1. What LLM architecture would be optimal for this machine?
2. How far is Qwen3.8-Flash-Next from it, and where does that cost time?
3. Which adaptation paths get an existing model closer without pretraining, and at what cost?

Research and arithmetic only: nothing in this document was run on the GPU.

**Labels.**
- **[M]** measured on this machine. The source is the halo-flash doc or file named.
- **[D]** derived by arithmetic from [M] numbers and model shapes.
- **[P]** published. A link is given.
- **[I]** inferred or estimated by me. Treat these as hypotheses to test.

Where a number is a guess, the text says so.

Inputs:
- `docs/HARDWARE.md`, `docs/ROOFLINE.md`, `docs/BENCHMARKS.md`, `docs/HALOGEN-NOTES.md`,
  `docs/N-LANE.md`, `docs/P0-BASELINES.md`, `docs/PLAN.md`, `src/model/config.hpp`;
- the per-kernel table of a 2048-token prefill at `/data/halo-work/kt_head.txt` (span 1481 ms);
- the user's BitNet/NPU work in `../bitnet-jolt` (evidence branches listed in its
  `artifacts/EVIDENCE_INDEX.md`).

---

## 0. Summary

- **This machine sells FLOPs dearly and capacity cheaply.**
  - About 55 TFLOP/s of WMMA, the same rate for f16, bf16 and int8 [M].
  - 240 GB/s of DRAM [M], shared by the GPU, CPU and NPU.
  - 128 GB of capacity, nearly free to fill.
  - An ideal model for it has:
    - few active FLOPs per prompt token (prefill);
    - few weight bytes per generated token (decode);
    - as much of the rest as possible parked in capacity: sparse experts and lookup tables.
- **Where Qwen3.8-Flash-Next spends its time against that ideal** (2048-token prefill of
  1481 ms):
  - **4-stream hyper-connections:** ~300 ms of prefill (~20%) and ~10% of decode bytes.
    - An ideal model would spend ~30 ms here.
  - **Dense weights at Q8_0:** 74% of decode bytes [M]. This is a quantization choice, not
    architecture, and the cheapest fix.
  - **Top-10 routing:** sets the expert FLOPs and bytes.
  - **The 2.08 B-parameter DeltaNet projections:** the largest single block of active FLOPs.
- **Low bits buy bytes only.** No WMMA here is faster than f16; the iu4 rate is unmeasured,
  see 1.4.
  - So 2-bit or ternary weights help decode and capacity.
  - They help prefill only where the kernel is bandwidth-bound and the dequantization is
    cheap.
  - **Ternary has no advantage over a 2-bit codebook on this GPU.** Pick by quality per byte.
- **Ranked adaptation paths** (section 4):
  1. **Byte diet, no training.** Rotated 4-bit dense, a 4-bit LM head and a bf16 router.
     - Decode bytes 6.53 -> 4.1 GB: roofline ~31 -> ~50 tok/s at 85% efficiency.
     - Cost: $0-50.
  2. **MTP speculative decoding with a vocabulary-trimmed draft head.** Engine work only.
  3. **fp16/bf16 hyper-connection streams.** Free; −80 to −150 ms of prefill if the quality
     gate passes.
  4. **2-bit experts by block-wise QAT/distillation, together with top-k 10 -> 6-8.**
     - Today's experts sit at both their achieved compute rate (~28 TFLOP/s) and their
       bandwidth (~174 GB/s) [D]. Either change alone buys little prefill; together they
       take the MoE from ~420 to ~250 ms.
     - Decode 4.1 -> 3.3 GB per token or less.
     - Speculation needs this: expert bytes scale with the number of positions verified.
       After rank 1 alone, MTP gains only ~1.16x by the bytes model; with 2-bit experts,
       ~1.5x.
     - Cost: about $0.5-3k rented.
  5. **Collapsing the hyper-connections to one stream by distillation.**
     - The largest remaining prefill prize.
     - No published method, so the highest risk.
- **Hybrid GPU+CPU+NPU (section 6).** Splitting this model's critical path across engines buys
  little.
  - The GPU's bandwidth-bound phases leave little to share.
  - The NPU's per-command cost (0.39 ms) [M] rules out per-layer handoffs.
  - Co-tenancy costs the GPU 8-18% [M].
  - The CPU earns its keep on lookups (the PLE n-gram table), n-gram drafting and host work.
  - The NPU earns it on separate small models (controller, embeddings), not on this model's
    layers.
- **First experiment (section 5):** a free inference-only sensitivity sweep, scored with the
  existing quality gates. Rotated 4-bit dense, top-k 8/6, fp16 hyper-connection streams, and
  2-bit experts in a band of layers. It decides which training money is worth spending.

---

## 1. The hardware's economics on one page

### 1.1 Rates

| Resource | Rate | Notes |
|---|---:|---|
| WMMA f16 / bf16 / iu8, 16x16x16 | ~34 cycles per SIMD per WMMA; 51.5-56 T/s register-only peak | [M] `HARDWARE.md`, `BENCHMARKS.md`. **iu8 is not faster than f16** |
| VALU | ~1 wave-instruction per cycle per SIMD; 7.2 T lane-ops/s fp32 | [M]. **Does not overlap WMMA:** every dequant instruction adds to GEMM time [M] |
| DRAM | 241 GB/s read, 224 write [M] (`P0-BASELINES.md`) | 10-33 GB/s for scattered small segments [M]; 239 GB/s needs ≥4 KB runs |
| MALL | 32 MB at ~800-900 GB/s [M] | L2 is invalidated between kernels [M]; only the MALL and fusion carry data across launches |
| Ridge point | 55 T / 240 G ≈ **229 FLOP/byte** | [D] |
| Dispatch | 1.7 µs dependent, 1.0 µs independent [M] | ~1,145 dispatches per decode token today [M] |
| CPU, 16 Zen 5 | full-width AVX-512 FPU on Strix Halo [P] ([chipsandcheese](https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix)); 32 B/cycle per CCD to the fabric [P] ([chipsandcheese](https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling)) | BitNet-2B ternary on 16 cores: pp2048 1030 tok/s, tg 79.8 tok/s [M] (bitnet-jolt `artifacts/e2e/cpu_baseline.md`), so ≥94 GB/s effective CPU read [D] |
| NPU, XDNA2 (8x6 array, 32 compute tiles) | 50 TOPS nominal | Hand-written int8 GEMM: 8.14 TMAC/s marginal, **2.40 TMAC/s end to end**; ~55 GB/s DRAM; 0.39 ms per command [M] (`N-LANE.md`) |

### 1.2 What a token costs at today's shapes [D from `config.hpp`, ROOFLINE §1.1]

| Component | Params read per token | Prefill FLOP per 2048-token chunk | Decode bytes per token (today's quantization) |
|---|---:|---:|---:|
| DeltaNet projections (36 layers: qkv 10240, z 6144, out 6144 x 2560) | 2.08 B | 8.5 T | ~2.2 GB (Q8_0) |
| Gated-attention projections (12 layers) | 0.60 B | 2.4 T | ~0.63 GB |
| Hyper-connection mixers (2 x 2 x 320 x 10240 x 48) | 0.63 B | 2.6 T | 0.67 GB |
| Shared expert | 0.24 B | 1.0 T | 0.25 GB |
| Routed experts, top-10 of 512, ff 640 | 2.36 B active (120.8 B total) | 9.7 T (11.6 T with 16-token padding) | 1.47 GB |
| Router (fp32) | 0.06 B | 0.26 T | 0.25 GB |
| LM head, 248,320 x 2560 | 0.64 B | last token only | 0.68 GB |
| DeltaNet state, read and written | — | — | 0.23 GB |
| **Total** | **~6.6 B** | **~25 T** | **6.53 GB** [M: BENCHMARKS + ROOFLINE §5] |

- **Prefill floor.** 25 TFLOP at peak WMMA is 0.49 s, or 4,200 tok/s.
  - At realistic efficiencies the ROOFLINE target is ~1.1 s (1,800-1,900 tok/s).
  - Measured today: 1.48 s of kernels [M] and 1.58 s wall (1,294 tok/s) [M].
  - gufo measures 1,599 tok/s [M]. halogen publishes 1,584 tok/s on an 8K prompt [P, their
    README].
- **Decode floor.** 6.53 GB at 241 GB/s is 27 ms, or 37 tok/s. Measured: 44.7 ms (22.4 tok/s)
  [M]. halogen publishes 37.6 tok/s at a smaller quantization mix [P].

### 1.3 What is free and what is expensive

| Free or nearly free | Expensive |
|---|---|
| Capacity: 128 GB holds 120 B of experts at 4-5 bits, plus a 27 GB n-gram table | Every active FLOP in prefill: no int8 or fp8 speed-up |
| Lookups: the PLE table reads ~2.9 MB per 2048 tokens [D: 32,768 rows x 90 B] | Every weight byte in decode, and every expert byte at small T |
| Weights touched by many tokens per chunk (dense: 2048 tokens of reuse) | Wide fp32 activations that miss the 32 MB MALL. The 84 MB HC residual is read and written ~3x per half-layer [M] |
| Data in the MALL (≤32 MB) between kernels | VALU dequant inside a GEMM (serial with WMMA) |
| | Small scattered reads (10-33 GB/s) |
| | Kernel boundaries: write-back is paid by the next kernel [M] |
| | Per-layer handoffs to the NPU (≥0.39 ms) or the CPU |

### 1.4 Low-bit formats on this GPU, quantified

- **WMMA types.** gfx1151 has f16, bf16, iu8 and iu4 WMMA. It has **no fp8/bf8 WMMA** and no
  fp8 conversion instructions [P].
  - LLVM gives gfx1151 `FeatureWMMA256bInsts` but not the gfx12 fp8 WMMA:
    [issue](https://github.com/anthony-chaudhary/fak/issues/13542),
    [RDNA3 WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/).
  - fp8 and fp4 are therefore storage formats only.
- **iu4.** `v_wmma_i32_16x16x16_iu4` exists. Secondary sources quote RDNA3 iu4 at 1024 ops per
  clock per CU, 2x iu8 [P] ([zolotukhin.ai reference](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
  - **`halo_hwprobe wmma` measures f16, bf16 and iu8 only:** the iu4 rate here is unmeasured.
  - iu4 needs *both* operands in 4 bits (W4A4). Even with learned rotations that costs 4.1
    points of zero-shot accuracy on Llama-3-8B (SpinQuant 65.5 vs FP 69.6; QuaRot 63.3) [P]
    ([SpinQuant](https://arxiv.org/abs/2405.16406)).
  - So iu4 is a probe item (one hour of `hwprobe`), not a plan.
- **Dequantization budget, prefill (GEMM, T = 2048).**
  - One 16x16x16 WMMA consumes 256 weights, 8 per lane. It costs 34 cycles; each VALU
    instruction adds ~1 cycle [M].
  - A weight fragment is reused across `subs` = n/16 token tiles: 2.5 at n ≈ 40 tokens per
    expert; 128 for dense layers.
  - So VALU per WMMA ≈ 8·c / subs, where c is VALU instructions per weight per lane [D].
  - **Experts (subs 2.5): c = 2 adds +19%, c = 4 adds +38%, c = 8 adds +75%** [D].
    - Measured: removing the Q4_K dequant arithmetic saves 81 ms of 664 [M, ROOFLINE 4l].
    - halogen's LUT + `v_perm` dequant uses as much VALU as ours [M, HALOGEN-NOTES].
  - Dense GEMMs (subs ~8-16 per block stage) tolerate almost any code.
- **Dequantization budget, decode (GEMV).**
  - VALU lane-ops per DRAM byte at full bandwidth: 7.2 T / 240 G ≈ 30 [D].
  - At 2 bits per weight that is ~7.5 lane-ops per weight before VALU becomes the limit.
  - So decode can afford trellis or lattice codes (QTIP, QuIP# E8). Prefill experts cannot,
    unless the codes decode in ≤2-3 VALU ops per weight (plain 2-bit with a LUT, ternary via
    `v_perm`).
- **Consequence.** The format for experts should be a *cheap-decode* 2-bit code: a 4-entry or
  16-entry LUT with group scales, after an incoherence rotation.
  - Ternary is a special case of it with no hardware advantage here.
  - The bitnet-jolt NPU work found the same on XDNA2: ternary expanded to int8 for compute,
    4x the weight bytes (461 -> 1843 MiB) [M, bitnet-jolt `VERDICT.md`].

---

## 2. Sketch of the optimal architecture for this machine

This is a design target, not something to pretrain. Numbers are [D]/[I]; the reasoning comes
from section 1.

### 2.1 Principles, each with the measurement behind it

1. **Total parameters ≫ active parameters.** Capacity is free and FLOPs are not.
   - Keep ~100-120 B parameters resident; spend 2-3 B active per token.
   - Fine-grained MoE is right for this machine. The Ling MoE scaling law puts the
     efficiency optimum at granularity 8-12 and a low activation ratio [P]
     ([arXiv 2507.17702](https://arxiv.org/abs/2507.17702)).
   - Qwen3.8-Flash-Next is already at granularity 8 (2·2560/640) and a 2% activation ratio.
2. **Expert bits, not expert count, set prefill bandwidth.**
   - At T = 2048 every expert is touched [D: (1 - 10/512)^2048 ≈ 3e-18], so a chunk streams
     *all* expert bytes: 75.5 GB today (315 ms at 240 GB/s), 32 GB at 2.125 bits (134 ms).
   - An expert's weights are reused by n = T·k/E tokens: 40 today.
   - The bandwidth/compute balance needs n ≳ 229·b/16 at b bits [D]: 59 at 4.125 bits, 30
     at 2 bits.
   - **At 2 bits, n = 40 makes experts compute-bound.** That is the right side to be on,
     because compute is the shared floor.
   - **Measured today they are co-bound** [D from kt_head]: 11.6 TFLOP padded in 418 ms is
     ~28 TFLOP/s achieved; 75.5 GB in 435 ms is ~174 GB/s. halogen's kernels keep 81-87% of
     their time with weights cache-resident [M].
     - So cutting bytes alone (2-bit, or pruning to 256 experts) saves only the exposed
       streaming, ~40-65 ms.
     - Cutting FLOPs alone (lower k) hits the 75.5 GB streaming time (~420 ms at the achieved
       174 GB/s).
     - Both together pay (§4).
   - Fewer, larger experts raise n but do not change per-token FLOPs. Granularity is not a
     hardware lever here.
3. **Top-k is a direct FLOP and byte lever.** Expert FLOPs and decode expert bytes scale
   linearly with k.
   - k = 6 instead of 10 saves 3.9 TFLOP per chunk and 0.59 GB per decode token at today's
     bits [D].
   - In prefill the FLOP saving shows only once expert bytes are also cut: at today's bits
     the 75.5 GB stream is the next floor (point 2).
4. **One residual stream, or a narrow 16-bit one.**
   - fp32 4-stream HC is 84 MB per 2048 tokens. That misses the MALL and is read and written
     several times per half-layer.
   - A single fp32 stream is 21 MB: it fits the MALL.
   - HC's published benefit at expansion 4 [P] ([Hyper-Connections](https://arxiv.org/abs/2409.19606)):
     - 7B dense: −0.022 loss and +0.9 points average downstream (70.1 -> 71.0);
     - OLMoE: 1.8x faster convergence.
   - Its cost here is ~20% of prefill [D, §3]. If HC is kept, mHC-style fused kernels
     [P] ([mHC](https://arxiv.org/abs/2512.24880): 6.7% training overhead at n = 4 with
     TileLang fusion) and bf16 streams are the minimum.
5. **Linear-to-full attention at 3:1 or sparser.** With QSA top-2048 selection on the full
   layers, attention stays bounded at long context.
   - At 2K, full attention is 74 ms (5%) and DeltaNet 145 ms (10%) [M].
   - The ratio is not the bottleneck at 2-8K. Long context depends on the indexer and the
     KV format.
   - Hybrid evidence:
     - [Jet-Nemotron](https://arxiv.org/abs/2508.15884) keeps a few full-attention layers
       and gets 47x generation throughput at 64K on H100 [P].
     - Qwen3-Next uses the same 3:1 Gated DeltaNet hybrid [P]
       ([Qwen3-Next](https://huggingface.co/docs/transformers/en/model_doc/qwen3_next)).
6. **Trim the DeltaNet inner width.**
   - 48 value heads x 128 = 6144 is 2.4x the hidden size. The DeltaNet projections are the
     largest dense block (2.08 B active, 8.5 TFLOP per chunk).
   - 32 value heads would save 0.5 B active parameters and ~2 TFLOP per chunk (~40-60 ms)
     [D]. This only makes sense when pretraining.
7. **Shapes in multiples of 16, ideally 128.**
   - Today's shapes are aligned: head dims 128 and 256; ff 640 = 40 tiles of 16.
   - Head dim 256 limits attention to ~48 query rows per block by LDS (64 KB), per ROOFLINE
     §3.6. 128 would double the reuse.
   - At T = 2048 the 16-token padding of expert tiles costs ~20% of expert WMMA [M].
     Larger chunks reduce it.
8. **16-bit activations, 2-4-bit weights.**
   - int8 activations buy nothing in WMMA [M].
   - fp32 intermediates cost bytes: the DeltaNet fp32 round trips are ~24 GB per chunk [M,
     ROOFLINE 4g].
   - Recurrent state stays fp32. A two-way bf16 split of `K·S0` failed the long-set gate
     [M, ROOFLINE §8].
9. **Lookup memory sized to capacity.**
   - Hashed n-gram tables (PLE here; DeepSeek's Engram [P]
     ([arXiv 2601.07372](https://arxiv.org/abs/2601.07372)) offloads a 100 B table at <3%
     overhead) cost capacity and a few MB per chunk, not FLOPs.
   - On a 128 GB box this is the cheapest quality there is, provided the table is resident
     or on fast NVMe. It is 38 ms per 2048 tokens from XFS vs 374 ms from ZFS [M].
10. **Speculation-aware sparsity.**
    - Verifying m tokens touches ~m·k distinct experts per layer:
      E(1 - (1 - k/E)^m) = 48 at m = 5, k = 10, E = 512 [D, independent routing; correlated
      routing would touch fewer, [I]].
    - Expert bytes per verify step therefore scale with m while dense bytes do not.
    - A model built for speculative decoding on a bandwidth-bound box wants **a small expert
      share of decode bytes**: low k and low expert bits. It also wants a draft head with a
      **trimmed vocabulary**: the 0.68 GB LM head costs as much as 5 experts' layers per draft
      token. FR-Spec cuts LM-head draft cost by 75% without retraining [P]
      ([arXiv 2502.14856](https://arxiv.org/abs/2502.14856)).

### 2.2 The sketch, with numbers [D/I]

| Knob | Qwen3.8-Flash-Next | Sketch |
|---|---|---|
| Hidden size, layers | 2560, 48 | 2560, 48 (or 40) |
| Residual | 4 HC streams fp32, rank-320 mixers | 1 stream fp32 (or 2 streams bf16) |
| Mixer | 3:1 Gated DeltaNet : gated attention, QSA | same; DeltaNet v-heads 48 -> 32 if pretraining |
| MoE | 512 x ff 640, top-10 + shared | 512 x ff 640, **top-6** + shared |
| Expert weights | Q4_K / Q5_1 (~5.0 b average, 75.5 GB) | 2.125-bit LUT code + rotation (32 GB) |
| Dense weights | Q8_0 (4.5 GiB) | ~4.25-bit, Hadamard-rotated (halogen's choice), with an fp16/Q8 prefill copy if dequant VALU hurts |
| Router | fp32 (0.25 GB per token) | bf16 |
| LM head | Q8_0 248K vocab | 4-bit; draft heads over a ~32K frequency-ranked subset |
| KV | fp32 + fp16 copy | fp16 only (24.6 KB per token; 6.4 GB at 262K) |
| PLE / n-gram memory | 27 GB, 1 layer | keep; consider more injection points (Engram) |
| MTP | 1 layer | 1-2 layers, trimmed-vocabulary head |

**What the sketch would run at** [D, using ROOFLINE efficiencies; treat as ±20%]:

| | Today | Sketch |
|---|---:|---:|
| Prefill, 2048 tokens (kernel ms) | 1481 (HC 330, MoE 435, dense 390, DeltaNet 145, attention 74, router 40, rest ~70) | ~990: HC -> ~35, MoE -> ~250 (k = 6 **and** 2-bit), router -> ~15, rest as today |
| Prefill tok/s | 1,381 (kernel span) | ~2,000 |
| Decode bytes per token | 6.53 GB | 2.64 GB (dense 1.62 + head 0.34 + router 0.08 + experts 0.38 + state 0.23) |
| AR decode at 85% of 240 GB/s | 31 tok/s roofline-85% (22.4 measured) | ~77 tok/s |
| MTP, 4 drafts (5 positions verified), ~3 tokens emitted per step [I] | 13.1 GB per step, 4.4 GB per token (1.5x AR) | 2.27 + 5 x 0.38 + 4 x 0.17 draft ≈ 4.85 GB per step, ~1.6 GB per token, ~125 tok/s at 85% |

These figures are a conditional budget, not evidence that the sketch is optimal at retained
quality. Its changes (top-6, one stream, 2-bit experts, narrower DeltaNet) are jointly
unvalidated. The adaptation paths in section 4 test them one at a time.

Quality per byte and per FLOP:
- **Per byte.** The 2-3-bit codebook formats are on the Pareto front. ParetoQ: ternary, 2-bit
  and 3-bit tie and often beat 4-bit at equal bytes, and 2-bit is recommended for hardware [P]
  ([ParetoQ](https://arxiv.org/abs/2502.02631)). QTIP's 2-bit models scale better than 4-bit
  per byte [P] ([QTIP](https://arxiv.org/abs/2406.11235)).
- **Per FLOP.** The activation ratio dominates [P, Ling]. Each top-k step costs ~10% of
  expert FLOPs.
- MoE experts tolerate low bits better than dense FFNs [P]
  ([MoQE](https://arxiv.org/abs/2310.02410)). This is the reason to push bits down in the
  experts first and keep dense layers at ~4 bits.

---

## 3. Qwen3.8-Flash-Next against the sketch: costs by component

Measured prefill times come from `/data/halo-work/kt_head.txt` (2048 tokens, 1481 ms span, at
HEAD as of 2026-10-05) and ROOFLINE §8. "Structural" means the cost would remain with ideal
kernels.

| Component | Prefill measured | Structural part [D] | Decode share | What the structure costs |
|---|---:|---:|---:|---|
| **HC mixers + combine** (4 x fp32 streams, rank-320) | ~330 ms (combine_norm 107, mix 108, `_mb` 60, `_l` share, inject) | ~200-270 ms: 2.6 TFLOP of mixer GEMMs (~50-80 ms) plus 84 MB residual passes (combine floor 67 ms + mix re-read 33 ms + write-back) against ~30 ms for a single stream | 0.67 GB (10%) + ~200 dispatches | **~20% of prefill, ~10% of decode bytes.** gufo pays ~245 ms on the same structure [M], so even ideal kernels leave ~200 ms |
| **Routed experts** | ~435 ms (gate/up 260, down 158, combine 48) | Floor max(9.7 T padded 11.6 T, 75.5 GB) ≈ 225-315 ms at peak rates. Achieved: ~28 TFLOP/s **and** ~174 GB/s, co-bound. gufo ~433 ms [M] | 1.47 GB (23%) | k = 10 and 4.5-6-bit codes together. Reducing either alone hits the other |
| **Dense projections** | ~390 ms | 12.1 T at 65% ≈ 360 ms | 3.9 GB at Q8 (60%, excluding head and router) | Q8_0 is a quantization choice: rotated 4-bit halves it. DeltaNet inner width 2.4x hidden |
| **DeltaNet** (36 layers, chunked) | ~145 ms (scan 76, prep 15, conv 27, norm 26) | ~75 ms target (ROOFLINE §2) | state 0.23 GB | fp32 state is required for quality [M]. Not a structural problem |
| **Full attention** (12 layers) | ~74 ms (FA 40 + glue) | ~24-30 ms | small | fine at 2K. At long context the indexer and KV format matter, not the ratio |
| **Router** (fp32) | ~40 ms | ~15 ms in bf16 | 0.25 GB (4%) | fp32 weights: a format choice |
| **LM head** (248K vocab) | 3 ms | — | 0.68 GB (10%) | costs ~0.7 GB per **draft** token unless the vocabulary is trimmed |
| **PLE n-gram** (27 GB IQ4_NL) | host 5-38 ms per chunk | — | ~1.4 KB per token | a capacity feature that suits this machine |

**The gap in one line.** Against the sketch, the structure costs ~300 ms of prefill in HC and
~170 ms in top-10 plus ~5-bit experts (they are co-bound, so they count together). In decode it costs ~3.9 GB of bytes: 2.0 GB Q8 dense + 0.34 head + 0.17
router + 0.67 HC + 0.85 expert bits and k.
- Roughly half of that decode gap is quantization (fixable without training).
- The other half is architecture: HC, k and expert bits, which need training.

---

## 4. Adaptation paths, ranked

**Ranking metric:** (expected speed gain × expected quality retention) / cost. Free paths
come first, ordered by gain. Paid paths are weighted toward decode, the interactive
bottleneck; reorder for a prefill-heavy workload. Speed gains are
[D] from section 3. Retention is [P] where a paper measured a comparable model, otherwise [I].

**Cost basis** (rented, October 2026) [P]:
- **Prices** ([gpuperhour](https://gpuperhour.com/),
  [spheron](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)):
  - H100: $1.5-2 marketplace, ~$2.5-5 on-demand.
  - H200: ~$3.5-4 typical.
  - B200: $4-6.
  - MI300X: $2.4-3.4.
- **Training FLOPs** [I]:
  - 6·N_active·D for full training of the student;
  - ~4·N_active·D with frozen weights and LoRA (no weight gradients);
  - plus 2·N_active·D for teacher forwards, unless teacher logits are cached.
- **Effective MoE throughput** [I, not measured for these kernels]: 50-200 TFLOP/s per H100
  (5-20% MFU; fine-grained MoE with quantized frozen experts is often at the low end),
  ~2.5x that on B200. **Every dollar figure below spans this 4x range.** Run a 30-minute
  throughput probe on the rented node before committing a budget.
- **Memory:** the bf16 student is ~250 GB (125 B params): 8xH100 (640 GB) or 8xH200.
  - Avoid a second copy for the teacher: cache teacher top-k logits in a first pass (§7).
    Alternatively, for ranks 4b-5, run the teacher as the same frozen weights with the
    adaptation switched off (k = 10, original HC).
  - An 8-bit student plus a resident bf16 teacher would be ~375 GB before activations.

N_active ≈ 6.6 B, so **1 B tokens of frozen-base KD is ~4e19 + 1.3e19 FLOP ≈ 75-300
H100-hours, about $200-1,500** [D/I].

### Rank 1: byte diet, no training (PTQ with rotations)

- **Method.**
  - Dense projections (DeltaNet, attention, shared expert, mixers): Q8_0 -> ~4.25-bit with
    Hadamard incoherence and GPTQ/QTIP-style rounding.
  - LM head -> 4-6 bit.
  - Router fp32 -> bf16 (the hi/lo split is already measured as quality-neutral [M]).
  - halogen ships exactly this class of trunk (Hadamard-rotated 4-bit, 6-bit HC weights) [M,
    HALOGEN-NOTES].
- **Evidence.**
  - QuIP# and AQLM are "close to lossless" at 3-4 bits, and QTIP halves the remaining gap [P]
    ([QTIP](https://arxiv.org/abs/2406.11235),
    [QuIP#](https://arxiv.org/abs/2402.04396)).
  - Rotations (QuaRot/SpinQuant) make 4-bit weight-only nearly free [P].
- **Gain.**
  - Decode bytes 6.53 -> 4.11 GB [D]: roofline-85% 31 -> 50 tok/s.
  - On today's kernels (31 ms streaming + 14 ms overhead), 44.7 -> ~34 ms (~29 tok/s) [D].
  - Prefill: neutral if the dense GEMMs keep an fp16/Q8 copy. That is +2.4 GB of RAM, which
    is affordable.
- **Retention:** ≥99% [I], checked by the existing gates.
- **Cost.**
  - Calibration only. Layer-wise PTQ of 3.9 B dense parameters runs locally, or in < 4
    rented GPU-hours: $0-50.
  - halo-flash needs a 4-bit dense GEMV with an online FWHT. 2560 = 5 x 512, so a block
    Hadamard of 128 or 512. halogen does a 128-point in-register FWHT [M].
- **Risk:** low.

### Rank 2: MTP speculative decoding, with a trimmed-vocabulary draft head (engine only)

- **Method.**
  - Use the shipped MTP layer.
  - Restrict draft logits to the ~32K most frequent tokens (FR-Spec, no retraining [P]).
  - Verify with the prefill path.
- **Evidence.**
  - gufo goes 26.5 -> 43.0 tok/s with MTP on this machine [M].
  - halogen goes 37.6 -> 46-56 [P].
  - Qwen3-Next reports a high MTP acceptance rate [P].
- **The cost structure here** [D]:
  - Verifying m tokens reads dense bytes once and expert bytes ~m times.
  - Today: 5.06 + m x 1.47 GB.
  - After rank 1: 2.64 + m x 1.47.
  - After 2-bit experts (rank 4): 2.64 + m x 0.63.
  - Four drafts mean five positions verified (the current input plus four). Drafting costs
    ~0.8 GB per draft token with the full head, ~0.17 GB with a 32K head (used below).
  - With ~3 tokens emitted per step [I], under independent routing:

    | Weights | GB per step | GB per emitted token | Gain vs AR |
    |---|---:|---:|---:|
    | today | 13.1 | 4.4 | 1.5x |
    | after rank 1 | 10.7 | 3.6 | **1.16x** |
    | after rank 1 + 2-bit experts | 6.5 | 2.2 | 1.5x |

  - Consecutive tokens' routing is correlated, so the union is smaller [I]. gufo's measured
    1.62x (26.5 -> 43 tok/s) [M] is above the 1.5x this model gives for today's bytes.
    Measure the real union per layer with the engine's router before trusting the table.
  - **The co-design point:** once dense bytes are cut, speculative decoding on this box is
    limited by *expert* bytes per verified position. That is why ranks 4 and 4b matter for
    decode, not only for capacity.
- **DeltaNet rollback.** Recurrent state cannot roll back a rejected token. bitnet-jolt hit
  exactly this: a one-token rollback made llama.cpp discard the whole hybrid prefix [M,
  bitnet-jolt `exact-spine-runtime-gate/RESULTS.md`].
  - The engine must snapshot DeltaNet state per draft position: 56.6 MB fp32 per position
    [D]. Alternatively, re-run the accepted prefix.
- **Cost:** engine work; $0.
- **Risk:** low to medium (acceptance on real prompts).
- Upgrading the draft head (EAGLE-3-style, trained on teacher features
  [P, [EAGLE-3](https://arxiv.org/abs/2503.01840)]) costs ~0.5-2 B tokens of frozen-base
  training, $100-600 [I].

### Rank 3: fp16/bf16 hyper-connection streams (no training)

- Halves the residual traffic: 84 -> 42 MB per pass.
- **Gain:** −80 to −150 ms of prefill [D, ROOFLINE §3.3 and §8], for $0 of rental.
- **Retention:** unknown; the long-set gate decides. The fp32 residual is the conservative
  original choice.
- **Risk:** medium.
- It ranks above the paid paths: ~1.2-3.8x the prefill gain of 2-bit experts alone, with no
  training. (Repeated as 5a below.)

### Rank 4: 2-bit experts by block-wise QAT/distillation (pairs with rank 4b)

- **Method.** EfficientQAT-style, applied to the expert matrices only:
  1. **Block-AP:** per-layer reconstruction of all parameters, on ~4K calibration samples.
  2. **E2E-QP:** train only the quantization scales against the teacher's logits.
  - Format: a cheap-decode 2.125-bit LUT code after one shared input rotation per layer. One
    shared rotation for all 512 experts is what halogen does [M].
- **Evidence.**
  - EfficientQAT: Llama-2-70B at 2 bits in 41 h on one A100; average zero-shot 69.48 vs
    72.41 (96%) [P] ([arXiv 2407.11062](https://arxiv.org/abs/2407.11062)).
  - MoE experts are more robust to low bits than dense FFNs [P, MoQE].
  - QTIP 2-bit Llama-2-70B: C4 perplexity 5.48 (QuIP# 5.71, AQLM 5.62) [P].
  - ParetoQ: 2-bit is a different regime from ≥3 bits ("representations change
    drastically"), so QAT rather than PTQ [P].
- **Gain** [D]:
  - Expert bytes 75.5 -> 32 GB: frees ~43 GB, so the n-gram table and long KV stay resident.
  - Decode 4.11 -> 3.27 GB (+26% after rank 1).
  - Speculation gain rises from 1.5x to 1.9x (rank 2 table).
  - Prefill alone: experts become compute-bound. Saves the exposed-streaming part, ~0.4-0.7 ms
    per call over ~95 calls ≈ 40-65 ms, **if** the dequant stays ≤2-3 VALU per weight.
    Trellis/E8 codes would *lose* prefill time (c = 8: +75% VALU per WMMA, §1.4).
  - With top-k 6 (4b): MoE ~420 -> ~250 ms [D, at today's achieved 28 TFLOP/s and
    174 GB/s].
- **Retention:** 96-99% [I, from EfficientQAT 70B dense at 96% and MoE robustness].
- **Cost.** Block-AP over 120 B expert parameters is ~2x EfficientQAT's 70B: ~80 A100-h, so
  ~40-50 H100-h. Add E2E-QP over ~0.5 B tokens with a frozen trunk (~40 H100-h). Total
  **~100-250 H100-h ≈ $0.3-1.2k**, up to ~$3k with re-runs and evaluation.
- **Data:** RedPajama/DCLM samples for Block-AP; the teacher's own chat outputs for E2E.
- **halo-flash work:** grouped gate/up and down kernels for the 2-bit code, on the existing
  K-blocked layout.
- **Risk:** medium. Hold out the long set; check that agreement with the API ≥ 91%.

### Rank 4b: top-k reduction (10 -> 8 or 6) with router distillation

- **Method.**
  1. First measure sensitivity with renormalized top-k at inference (free; section 5).
  2. Then distill: student at k = 6 with the router and LoRA on the shared expert and norms
     trainable; teacher at k = 10 (same weights); KL on logits.
  - Optionally use dynamic k (top-p routing) to keep difficult tokens at 10.
- **Evidence.**
  - Little direct published evidence for cutting k after training at this scale. This is
    [I], which is why the sweep comes first.
  - Expert pruning shows the experts are redundant in *count*: REAP removes 50% of experts
    with >96% retained on Qwen3-Coder-480B [P]
    ([REAP](https://arxiv.org/abs/2510.13999)). Removing a *per-token* compute share is a
    different question.
- **Gain** [D]:
  - Prefill alone: ~0. At today's bits the 75.5 GB stream (~420 ms at 174 GB/s) becomes the
    floor.
  - With 2-bit experts: MoE ~420 -> ~250 ms (k = 6) or ~335 (k = 8).
  - Decode expert bytes −40%: 0.59 GB at 4-5 bits, 0.25 at 2 bits.
  - Speculation verify bytes fall proportionally: ×5 positions.
- **Retention:** unknown. Guess 97-99% at k = 8 and 93-98% at k = 6 after KD [I].
- **Cost.** KD on 1-2 B tokens ≈ 75-600 H100-h ≈ **$0.3-1.5k** (share the run with rank 4).
- **Data:** DCLM/FineWeb-Edu plus prompts from Tulu-3/UltraChat with teacher responses.
  Cached teacher top-32 logits are ~0.5 MB per 2048-token sequence, ~250 GB per 1 B tokens.
- **halo-flash work:** k as a runtime parameter. The kernels assume k = 10 in bucketing and
  combine.
- **Risk:** medium.

### Rank 5: simplifying the hyper-connections

Two options, which differ in cost:

**5a. bf16 or fp16 HC streams, no training.**
- Residual traffic halves: 84 -> 42 MB per pass.
- The mix-epilogue fp16 probe and the combine floors suggest -80 to -150 ms of prefill [D,
  ROOFLINE §3.3 and §8].
- Retention: unknown; the long-set gate decides. The fp32 residual is the conservative
  original choice.
- Cost: $0. Engine work only.

**5b. Collapse 4 streams to 1 by distillation.**
- **Method** [I; no published recipe for removing HC from a trained model]:
  1. Initialize the single stream as the HC read-out at each layer.
  2. Replace each layer's width/depth mixing by its expected (static) value.
  3. Align hidden states layer by layer (RADLADS step 1 / MOHAWK stage 2 style).
  4. Distill logits.
- **Gain:** ~200-270 ms of prefill (-15 to -18%), 0.67 GB of decode bytes (-16% after rank 1),
  and ~200 fewer dispatches per token [D].
- **Retention:** unknown. The HC gain at 7B is ~1 point of average downstream accuracy, so a
  well-distilled collapse may lose about that [I].
- **Token budget by analogy with RADLADS:** 100 M alignment + 250-700 M KD tokens [P,
  RADLADS], but trainable parameters now touch every layer [I]. ~1 B tokens with a frozen
  MoE: ~100-500 H100-h, **$0.5-3k**.
- **Risk:** high; novel.

### Rank 6: depth pruning plus distillation (Minitron / ShortGPT)

- **Method:** score layers (ShortGPT block influence), drop 8 of 48 layers, distill.
- **Evidence.**
  - Minitron: 15B -> 8B with up to 40x fewer tokens than from scratch [P]
    ([arXiv 2407.14679](https://arxiv.org/abs/2407.14679)).
  - Llama-3.1-Minitron-4B used 94 B tokens [P].
  - ShortGPT drops ~25% of layers with moderate MMLU loss but larger generation loss [P]
    ([arXiv 2403.03853](https://arxiv.org/abs/2403.03853)).
- **Gain:** ~17% across the board [D].
- **Cost:** recovery needs ≥10-90 B tokens: 1-9k H100-h, **$3-30k**.
- **Retention:** 90-97% [P/I].
- **Risk:** medium; the cost is high. Worse ratio than ranks 1-5.

### Rank 7: ternary trunk via BitDistill / BitNet continued pretraining

- **Method:** BitDistill.
  1. SubLN refinement.
  2. Continued pretraining (10 B tokens).
  3. Logit + MiniLM attention-relation distillation on the task.
- **Evidence.**
  - Qwen3 0.6-4B on classification and summarization reach parity (MNLI 91.40 vs 91.48 at
    4B) [P] ([arXiv 2510.13998](https://arxiv.org/abs/2510.13998)). This is
    **task-specific** fine-tuning, not general chat.
  - General-purpose conversion is weaker. HF's Llama3-8B-1.58 at 100 B tokens is "close in
    some metrics", but lags on average [P]
    ([HF blog](https://github.com/huggingface/blog/blob/main/1_58_llm_extreme_quantization.md)).
  - Falcon notes that converting existing checkpoints to ternary "often leads to poor
    performance" [P] ([Axolotl/TII](https://huggingface.co/blog/axolotl-ai-co/finetuning-ternary-llms-tii-axolotl)).
  - bitnet-jolt established that BitNet's BF16 master trains with stock PEFT (671 tok/s at
    seq 2048, 35.8 GiB on this GPU) [M]. Tooling is not the obstacle; quality and tokens
    are.
- **Gain over ranks 1 + 4:** dense 1.62 -> ~0.6 GB per token. No compute gain on this GPU
  (§1.4).
- **Cost:** ≥10 B tokens of continued pretraining over 6.6 B active: ~5e20-1e21 FLOP,
  ~1.4-5k H100-h, **~$3-20k**.
- **Retention:** low confidence, 85-95% general [I].
- **Risk:** high. Dominated by 2-bit codebook QAT at equal bytes on this hardware.

### Rank 8: converting the 12 remaining attention layers to linear (RADLADS / LoLCATs / Liger)

- **Evidence.**
  - RADLADS converts Qwen2.5-72B with 350-700 M tokens for < $2,000. MMLU retention is 90-92%
    and LAMBADA ~98-100% (7-72B), on 8xMI300X (7B in ~7 h) [P]
    ([arXiv 2505.03005](https://arxiv.org/abs/2505.03005)).
  - LoLCATs uses 40 M tokens with LoRA and closes 78% of the MMLU gap at 70B/405B [P]
    ([arXiv 2410.10254](https://arxiv.org/abs/2410.10254)).
  - Liger recovers 93% with 0.02% of pretraining tokens [P]
    ([arXiv 2503.01496](https://arxiv.org/abs/2503.01496)).
  - Mamba-in-the-Llama keeps a quarter of the attention layers [P]
    ([arXiv 2408.15237](https://arxiv.org/abs/2408.15237)).
- **Gain here:** ≤74 ms at 2K (5%), and it hurts long-context retrieval, which is what the 12
  layers are for.
- **Rank:** low. These methods matter for converting a *dense-attention* model; this one is
  already 75% linear.

### Rank 9: expert pruning or merging (REAP, HC-SMoE)

- **Gain.**
  - 512 -> 256 experts halves expert storage and the bytes streamed per prefill chunk
    (75.5 -> 38 GB), doubling reuse to n = 80.
  - Like 2-bit experts, that saves only the exposed streaming at today's achieved compute
    rate: ~40-65 ms. It becomes larger if the expert kernels get faster.
  - Decode bytes per token and the speculation multiplier (~m for any E ≫ k) are unchanged
    [D].
- **Retention:** REAP keeps >96% at 50% on code [P]. General chat is untested.
- **Cost:** one-shot, ~$0-100.
- It is an alternative to 2-bit for prefill and capacity, not a complement for decode.

### Rank 10: fewer, larger experts (upcycling-style re-granularization)

- No speed gain at fixed active parameters (§2.1.2), and it moves away from the scaling-law
  optimum.
- Not recommended.

### Summary table

| Rank | Path | Prefill | Decode bytes per token | Retention | Cost | Risk |
|---:|---|---:|---:|---:|---:|---|
| 1 | Byte diet (rotated 4-bit dense, head, bf16 router) | 0 | 6.53 -> 4.11 GB | ≥99% [I] | $0-50 | low |
| 2 | MTP + trimmed draft head | 0 | /1.16 after rank 1 alone; /1.5 with 2-bit experts (bytes model; gufo measured 1.62x) | 100% (exact verify) | $0 (+$100-600 draft upgrade) | low to medium |
| 3 | fp16/bf16 HC streams | -80..150 ms | ~0 | gate decides | $0 | medium |
| 4 | 2-bit experts (block QAT + E2E KD) | -40..65 ms alone | 4.11 -> 3.27 GB | 96-99% [I] | $0.3-3k | medium |
| 4b | Top-k 10 -> 6 + router KD | ~0 alone; with 4: MoE 420 -> 250 ms | -0.25..0.59 GB | 93-99% [I] | $0.3-1.5k | medium |
| 5b | HC 4 -> 1 by distillation | -200..270 ms | -0.67 GB | ~97-99% [I] | $0.5-3k | high |
| 6 | Depth pruning + KD | -17% | -17% | 90-97% | $3-30k | medium |
| 7 | Ternary trunk (BitDistill) | 0 | -1 GB vs rank 1 | 85-95% [I] | $3-20k | high |
| 8 | Linearize the 12 attention layers | ≤ -74 ms | ~0 | 90-98% | $1-2k | medium; hurts long context |
| 9 | Expert pruning | ~0 | 0 | 96% at 50% | $0-100 | low; memory only |

**Combined (1 + 2 + 3 + 4 + 4b at k = 8)** [D/I]:
- Prefill ~1481 -> ~1250-1330 ms (+11-18%). With k = 6, ~1170-1250 ms.
- AR decode ~3.0 GB per token (~68 tok/s at 85% of 240 GB/s); ~2 GB per emitted token with
  MTP.
- About $1-5k of rented compute, depending on the achieved training MFU.

---

## 5. The recommended first experiment

**Inference-only sensitivity sweep, no training: four knobs, each scored by the existing
gates.**
- Short set: NLL ≤ 0.305, API top-token agreement ≥ 91%.
- Long set: paired per case against the current build. Shift within ±0.001 NLL ±SE, as in
  BENCHMARKS.

| Arm | Change | Predicted speed [D] | What it decides |
|---|---|---|---|
| A | Dense projections + LM head to rotated 4-bit (PTQ, local) | decode 44.7 -> ~34 ms | rank 1: ship or not |
| B | Top-k 8 and 6, renormalized, no training | prefill ~0 at today's bits (co-bound); decode -0.3 / -0.6 GB | whether rank 4b's KD is needed at all (k = 8 might pass untrained), and its budget |
| C | HC residual streams stored fp16 (combine, mix, `_mb` read fp16) | prefill -80..-150 ms | rank 3; and how fragile the HC path is, as a proxy for 5b's risk |
| D | Experts of layers 12-23 at 2-bit PTQ: llama.cpp IQ2_S-class or a plain 2-bit LUT | none yet (kernel needed); quality only | the slope of rank 4: if 2-bit PTQ on a quarter of the layers stays within the gate, QAT will pass on all of them |

Why this first:
- **Cheap.** $0 of rental. Each arm is a flag or a requantized GGUF plus a scoring run of
  ~1-2 h on this machine, when the GPU is free.
- **Decisive.** Each arm is a go/no-go for a paid path, with a predicted number written down
  before measuring (the house rule in ROOFLINE).
- **Measurable** with tools that exist: `halo_score`, `ab_bench.py`, `--kernel-timing`.

**Decision rules, fixed now:**
- **A** passes the gate -> build the 4-bit dense GEMV; that is rank 1.
- **B:** if k = 8 passes untrained, ship it with A. If k = 6 fails by < 3x the k = 8 shift,
  buy the KD (rank 4b, ~$0.3-1.5k), bundled with rank 4's run.
- **C** passes -> ship it, and consider 5b only after 1-4.
- **D:** if the 12-layer 2-bit band moves NLL by < 0.01, fund rank 4 (~$0.3-3k).

**Also queue (GPU time, an hour):** `halo_hwprobe wmma` with an iu4 variant. If iu4 runs at 2x
iu8, W4A4 for the dense trunk becomes worth a QuaRot/SpinQuant study. Its accuracy cost (4
points on Llama-3-8B) makes it a long shot.

---

## First experiment: results

Arms A, B and D and the iu4 probe of section 5 (stream Q). Arm C belongs to stream P and is
not covered here. **Status:** phase 1 done: the flag, the variant GGUFs, the probe, and the
rules and predictions below, fixed before any GPU run. The GPU phase (scoring, timing, the
probe) waits for stream P to finish.

### What was built (phase 1, CPU and disk only)

**Arm B: experts per token as a runtime parameter.**
- **Flag:** `--moe-topk N` in `halo_score`, `halo_run` and `halo_serve`, through
  `ModelWeights::SetExpertsUsed`. The router keeps its softmax over all 512 experts and
  renormalizes the N largest to sum to 1.
- **Nothing in the kernels assumes k = 10.** Every place takes k from
  `config.num_experts_used`:
  - router top-k, single row and row-wise;
  - bucketing, one-block and three-pass;
  - grouped gate/up and down tiles (the token of a pair is `pair / k`);
  - `combine_grouped`, `combine_shared`, and the `combine_norm_tok_moe` epilogue;
  - the decode fused and unfused expert loops;
  - every buffer size.

  The GPU kernel tests already draw k in 1..10 (`GroupedExpertsMatchCpuReference`,
  `FusedMoeMatchesCpuComposition`, `CombineNormTokEqualsRows`, `RouterTopkMatchesReferenceRule`).
  The only check of 10 is `Config::IsFlashNextGeometry`, which `halo_run` prints and nothing
  gates on.
- **Tests:**
  - CPU, Hegel: the reference's routing (`cpu::RouteTopK`) equals a stable-sort oracle; a
    smaller k is a renormalized prefix of a larger one; `SetExpertsUsed` validates its range.
  - GPU, gated on `HALO_TINY_HF_DIR`: the engine at every k below the trained count against the
    reference.
  - Real model (phase 2): `--moe-topk 10` must give logits bit-identical to no flag.

**Arm A: `halo_requant A`, rotated 4-bit dense trunk and LM head.**
- **Tensors:** 497. Every 2-D Q8_0 tensor except `token_embd` and the five Q8_0 routed-expert
  `ffn_down_exps`:
  - DeltaNet `attn_qkv`, `attn_gate`, `ssm_out`;
  - attention `attn_q/k/v/output`;
  - the shared expert;
  - the hyper-connection mixers, including the output mixer;
  - the PLE projections;
  - the LM head.

  The BF16 indexer, the F32 router and the injects are unchanged.
- **Code, 4.125 bpw** (halogen's trunk format, HALOGEN-NOTES):
  - a randomized block Hadamard R = diag(±1) H₁₂₈/√128 on the input dimension (64 for the
    mixer `up` rows of 320); signs seeded by the tensor name;
  - the 16-entry IQ4_NL codebook times one fp16 scale per 128 rotated values (per 64 for those
    rows).
- **Rounding: RTN with rotation, not GPTQ.** Each group's scale minimizes its squared error:
  a grid over clipping, each point refit by least squares.
  - GPTQ needs each layer's input Hessian from real activations. That means a GPU pass over
    calibration text, which phase 1 does not have.
  - So arm A measures the pessimistic end of rank 1. Calibrated rounding can only improve on it.
- **Simulation:** each row is stored as dequant(Q(W·R))·Rᵀ, quantized in the rotated basis.
  Rotating W and back without quantizing in between would return W.
- **Container: Q8_0, not BF16.** The loader accepts BF16 for these tensors, but the engine's
  fast paths need Q8_0: the FP16-WMMA prefill route (`F16Route` needs the repacked int8 copy),
  the fused mixer, DeltaNet and shared-expert decode kernels. BF16 would change which kernels
  run, not only the weights, and would add ~9 GB. Q8_0 keeps every kernel of the baseline.
- **Error the container adds** (measured by the tool per tensor, relative Frobenius error
  against today's Q8_0 weights, over all 497 tensors):
  - the 4-bit code: 0.0863 (0.082-0.090 per tensor class);
  - Q8_0 re-rounding: +0.0064;
  - total: 0.0866.

  So the container adds 0.56% of the code's error energy: Q8_0 simulates the 4-bit code
  faithfully.
- **Time and memory:** 4.5 min on 12 threads, 166 MB resident.

**Arm D: `halo_requant D`, 2-bit routed experts in layers 12-23, stored exactly.**
- **Tensors:** 36. In this band gate/up are Q4_K and down is Q5_1 in every layer.
- **Code, 2.5 bpw** (2.425 for the down rows of 640):
  - every 32 values hold the four levels a + s·l (l = 0..3, a ≤ 0), fit by least squares
    (llama.cpp make_qkx2-style);
  - s and −a quantized to 6 bits against the maximum of the super-block, as Q4_K quantizes
    its scales;
  - no rotation, no calibration (RTN).
- **Exact in the tensors' own containers:**
  - Q4_K with q = 5l: d·sc·q − dmin·m holds exactly these levels.
  - Q5_1 with q = 10l: d = s/10 and m = a in fp16; the fp16 rounding is ≤ 2⁻¹¹ of the step,
    and the stored tensor is still exactly a 4-level uniform code.

  Sizes and the fast kernels (`gate_up_q4_k`, `down_q5_1`, grouped) are unchanged.
- **Test:** a Hegel property decodes each container with the engine's own dequantizers and
  checks at most 4 levels on one uniform grid per 32 values. It covers Q4_K, Q5_K, Q5_1 and
  Q8_0 (Q8_0 for the five Q8_0 down layers outside this band).
- **Error:** 0.3076 relative to today's 4-5-bit experts (gate 0.307, up 0.307, down 0.309).
  The container adds nothing.
- **Checked on the written file:** a pure-Python decode of 40 random rows from each of three
  tensors found only q ∈ {0, 5, 10, 15} (Q4_K) or {0, 10, 20, 30} (Q5_1), 7,200 groups in all.
- **Time and memory:** 5.3 min, 165 MB resident.

**iu4 probe.** `halo_hwprobe wmma` now has an iu4 row set: `v_wmma_i32_16x16x16_iu4`, both
operands 4-bit, confirmed in the device assembly.

**Variant files** (in `/data/halo-work/codesign/`):
- The tensor types and sizes are unchanged, so each variant is a reflink copy of the four
  original shards with the chosen tensors rewritten in place. Only rewritten extents use disk.
- `manifest.tsv` gives each tensor's error terms and an FNV-64 of its bytes.
- `halo_requant` streams 64 MB row chunks, ~170 MB resident. It resumes from `progress.tsv`
  under a header bound to the arm and the source shards' sizes and mtimes.
- Crash test: one band killed mid-tensor and resumed gave shards byte-identical to an
  uninterrupted run. A progress file with a different header is refused.
- Variants on disk: `A-rot4/` and `D-exp2/` (+5 GB and +18 GB of unshared extents).
- GPU-phase scripts: `tools/codesign/` (`q_sweep.sh` scores, `q_speed.sh` for
  timing/sanity/iu4). Each chunk takes `flock /data/halo-work/gpu.lock` on its own, pins and
  unpins inside, stays under 14 min, and refuses to start below 85 GiB available.

### Rules and predictions, fixed before measuring

**Protocol.**
- **One binary for every arm:** this branch merged with main at the time of the GPU phase.
  - Without flags its GPU paths are main's: the branch changes only the CPU reference's
    routing and adds flags.
  - The baseline is the original UD-Q4_K_XL on that binary.
- **Scoring:** `halo_score --prefill 1`, default n-gram cache, in chunks (`--start/--limit`)
  under the GPU lock.
  - Short set: 98 cases.
  - Long set: 12 cases, 2K-24K-token prompts.
- **Paired comparison:** `tools/paired_nll.py`, per case against the baseline.
  - The shift is the token-weighted NLL difference.
  - Also reported: the per-case mean ± SE, cases up/down, and the largest |d|.
- **Gate (BENCHMARKS):** short-set NLL ≤ 0.305 and API top-token agreement ≥ 91%.
- **Expected baseline:** NLL 0.2859-0.2864, agreement 92.5-92.7%.
- **Sanity checks before any arm counts:**
  - `--moe-topk 10` logits bit-identical to no flag (`halo_run --logits-out`);
  - the baseline on the gate within its expected range.

**Arm A (rank 1: ship or not).**
- **Prediction:**
  - short-set shift +0.004..+0.012 (central +0.007);
  - agreement −0.2..−0.8 points;
  - long-set shift +0.002..+0.010;
  - the gate passes.
- **Decision:**
  - **Go:** the gate passes and the long-set shift is ≤ +0.010. Build the 4-bit dense GEMV
    (rank 1).
  - **Inconclusive:** the gate passes but the long-set shift is > +0.010. Re-run A with
    calibrated (GPTQ-style) rounding before any kernel work.
  - **No-go for RTN:** the gate fails. Rank 1 needs calibrated rounding; re-run A with it.

**Arm B (rank 4b: is KD needed, and its budget).**
- **Prediction:**
  - short-set shift: k = 8 +0.003..+0.020 (central +0.008); k = 6 +0.012..+0.060 (central
    +0.030);
  - so k = 8 passes the gate and k = 6 is marginal to failing.
  - Ratio r = shift(6) / shift(8) ≈ 3-4.
  - Prefill, 2048 tokens, pinned: k = 8 −5..−25 ms, k = 6 −10..−50 ms. The MoE stays co-bound
    by the 68 GB expert stream; the compute share falls 20% / 40%.
  - Decode: −1.4 ms (k = 8) and −2.8 ms (k = 6) per token. Expert bytes fall 0.29 / 0.59 GB at
    today's ~210 GB/s effective.
- **Decision:**
  - **k = 8:**
    - Passes untrained: the gate passes and the long-set shift is ≤ +0.010. Ship it with A
      (the default stays 10 until then).
    - Fails: KD is needed for k = 8 too.
  - **k = 6:**
    - Passes untrained: ship k = 6, no KD.
    - Fails, with r < 3: buy the KD (rank 4b, ~$0.3-1.5k), bundled with rank 4's run.
    - Fails, with r ≥ 3: a cliff. Don't fund k = 6 KD; at most k = 8.
  - **Inconclusive** when shift(8) < 0.002: r is then a ratio of noise. Decide k = 6 on its
    own shift against the gate.

**Arm D (the slope of rank 4).**
- **Prediction:**
  - short-set shift +0.006..+0.040 (central +0.015), so likely inconclusive;
  - long-set shift similar.
  - The code's relative error is ~3.5x that of Q4_K. One quarter of the layers' experts
    carries it.
- **Decision:**
  - **Fund rank 4 (~$0.3-3k):** shift < 0.010 on the short set, and also on the long set.
  - **Inconclusive:** 0.010-0.030. Before the full spend, run a one-band QAT pilot: layers
    12-23, Block-AP only, a few H100-hours.
  - **Don't fund rank 4 as costed:** > 0.030. 2-bit is too far from PTQ here; consider 3-bit
    experts instead.

**iu4 probe.**
- **Prediction:** 40% that iu4 runs at 2x iu8 (~17 cycles per WMMA per SIMD, as RDNA3's
  1024 vs 512 ops/clk/CU sheet says); otherwise the same rate as iu8 (~34 cycles).
- **Decision:**
  - **≥ 1.8x iu8:** W4A4 for the dense trunk goes on the list for a QuaRot/SpinQuant study.
    Still a long shot: 4 points on Llama-3-8B.
  - **≤ 1.2x:** close the item.

### Results

Pending the GPU phase.

---

## 6. Hybrid GPU + CPU + NPU over unified memory

The three engines share one 128 GB LPDDR5X pool with no copies. They also share **one 240 GB/s
of bandwidth**, and they synchronize at very different costs.

### 6.1 The co-processors as measured

| | GPU (gfx1151) | CPU (16 x Zen 5) | NPU (XDNA2) |
|---|---|---|---|
| Dense int8/f16 throughput | 25-30 T achieved on dense GEMMs [M] (55 peak) | ~4.3 T effective on BitNet ternary/int8 [M, bitnet-jolt VERDICT]; full-width AVX-512 [P] | 8.14 TMAC/s marginal, 2.40 end to end, hand-written; 0.9 TMAC/s from the compiler [M, N-LANE] |
| DRAM read | 241 GB/s [M] | ≥94 GB/s effective [D from 79.8 tok/s x 1.1 GiB]; fabric limit 32 B/cycle per CCD [P] | ~55 GB/s [M] (13.7 per channel) |
| Synchronization | 1.7 µs per dependent kernel; 6.6 µs launch + sync [M] | host thread: µs | 63-115 µs per command (1-8 workers); 0.39 ms with 32 tiles [M]; ≥4 commands per transformer layer today [M] |
| Programming paths | HIP / HRX | C++ / AVX-512 | IRON/MLIR-AIE (bitnet-jolt, bit-exact int8 GEMM, 9.3 TOPS stock `whole_array`) [M]; Loom/libamdf in hrx-system [M]; Ryzen AI SW / ONNX Runtime with the VitisAI EP (not tried here) |
| What fits | everything | lookups, sampling, grammar, scheduling, small models | fixed-shape int8 GEMMs with N ≤ 4096 per `aie.dma_bd` stride [M, bitnet-jolt]; on-chip memory of ~6 MB vs ~57 MB per decider layer [M] |

**Co-tenancy is measured, and it is not free** [M, bitnet-jolt `gpu-cotenancy`]. With a BitNet
controller on NPU+CPU running beside a GPU LLM worker:
- the GPU worker lost 13-18% of prefill and 8-10% of decode;
- offloading the controller's linears to the NPU, rather than the CPU, *reduced* the GPU's
  loss by 6.6% and saved 9.5 W.
- A separate measurement: GPU LoRA training raised a co-resident controller's TTFT by 48%
  [M, `model-candidate-halo`].
- Strix Halo "lets its GPU squeeze out the CPU" under bandwidth pressure [P, chipsandcheese].

**Rule that follows.** Concurrent CPU or NPU work can only gain while the GPU is compute- or
issue-bound, never while it streams DRAM. From the traces [M, ROOFLINE §4.2 and §8]:

| GPU phase (2048-token prefill) | Bound | DRAM headroom for others |
|---|---|---|
| HC combine / mix epilogue (~215 ms) | DRAM (90% / 78% vmcnt) | none |
| Experts (~435 ms) | dequant VALU + DRAM (~114-150 GB/s) | little |
| Dense GEMMs (~390 ms) | WMMA + staging, weights reused across tokens | yes |
| DeltaNet scan (~76 ms) | LDS/barrier issue | yes |
| Attention (~40 ms) | barriers, MALL | yes |
| **Decode** | DRAM in the streaming kernels. On average only 146 GB/s (6.53 GB in 44.7 ms) | little: the idle DRAM sits in µs-scale gaps between small kernels and launches, which a co-processor at 0.07-0.39 ms per command cannot target. Its bytes land on the streaming kernels |

### 6.2 Candidate splits, evaluated

| Idea | Expected gain | Preconditions | Verdict |
|---|---|---|---|
| **Router on NPU or CPU** | negative | The router sits on the serial path before each layer's experts. 48 handoffs x ≥0.07-0.4 ms = 3-19 ms, plus compute: 2.7 GMAC per layer at 2.4 TMAC/s = 1.1 ms vs 0.83 ms on the GPU [D] | **No.** Make the router bf16 on the GPU (-25 ms) |
| **MTP / draft on NPU, concurrent with GPU verify** | ≤ ~10% of decode | Drafting with a 32K-vocabulary head costs the GPU only ~0.17 GB, ~0.8 ms per draft token [D]: ~3 ms of a ~30 ms step. The NPU needs ≥1 command per token (≥0.39 ms), multi-phase commands (not available [M]), and the model's 0.1-0.2 GB per token at ≤55 GB/s, ~2-4 ms. Its bytes come out of the GPU's verify bandwidth | **Not yet.** Do vocabulary trimming on the GPU first; it captures most of the gain |
| **Small dense draft model on NPU** (e.g. 0.15-0.3 B, d 1024, int8, 32K head) | 0-10% [I] | As above. Today ~4 commands per layer x 0.39 ms makes a 12-layer draft ~19 ms per token [D]: slower than the GPU's own MTP | **Gated** on Loom memory-tile staging + continuation commands (N-LANE blockers 1-2) |
| **Prompt-lookup / suffix n-gram drafting on CPU** | 1.3-2.5x on repetitive text (code edits, RAG, agent loops) [I]; ~1x on fresh prose | No model bytes: a CPU suffix index over context and history. Verify on the GPU, same rollback machinery as MTP | **Yes, cheap.** Combine with MTP: take the longer draft |
| **PLE / n-gram lookup on CPU** | already done | Host reads with prefetch (0647d6b, 21a0490 [M]); 2.9 MB per chunk; resident table 5 ms vs 38 ms from XFS | **Yes.** This is the model feature that suits the box. Open: the ~5% device slowdown with the table resident [M] |
| **DeltaNet heads split CPU/GPU in prefill** | ≤ 2% | The scan is issue-bound on the GPU (145 ms). The CPU's ~4 T would take ≤25% of heads in the same time; 36 per-layer syncs; fp32 state stays in shared memory | **No.** The complexity outweighs ≤35 ms |
| **DeltaNet state update on CPU in decode** | negative | Decode is DRAM-bound; 0.23 GB per token moves from GPU bandwidth to slower CPU bandwidth; 36 syncs per token | **No** |
| **Dense GEMM share on NPU in prefill** | ≤ 3-4% at the measured end-to-end rate. Up to ~7-10% (−100..150 ms) if the 16.3 TOPS marginal rate became end to end [D] | The dense phase has DRAM headroom. Today 4.8 TOPS end to end is ~15% of the GPU's achieved 25-30 T. Needs memory-tile staging, cascade and continuation commands (N-LANE blockers 1-2), int8 activations, a column split per layer with one sync per GEMM, and ≤55 GB/s for its weight share | **Not today.** The largest NPU prize for this model, so rerun N-LANE's gate when the Loom features land |
| **Experts on CPU in prefill (llama.cpp-style offload)** | ≤ 3% | Built for VRAM-limited dGPUs; here the CPU would take ~5% of experts and contend for DRAM in the bandwidth-bound expert phase | **No** |
| **Long-context attention or indexer on CPU** | ~0 | QSA bounds attention to 2048 selected tokens. The indexer reads ~17 MB per layer per token at 262K [D], ~1 ms in total. No gain from moving it | **No** |
| **Separate small models on the NPU** (controller, embeddings, classifier, reranker, ASR) | frees GPU and CPU time; measured cheaper for the GPU worker than running the same model on the CPU [M] | Fixed shapes, int8, N ≤ 4096; resident weights; admission control for concurrency [M, bitnet-jolt] | **Yes.** This is where the NPU has paid off here (bitnet-jolt) |

### 6.3 A model co-designed for the split [I]

If the model were designed for three engines rather than adapted, the components would be sized
like this:
- **GPU:** the trunk, with everything in §2. All bandwidth-bound work stays here, because
  splitting bandwidth-bound work only moves bytes between engines.
- **CPU:**
  - an Engram-style lookup memory sized to capacity (tens of GB; reads of a few MB per
    chunk);
  - a suffix-automaton drafter;
  - tokenization, sampling, grammar constraints and scheduling.
  - The CPU's ~94 GB/s is spent where the GPU is issue-bound (prefill) and kept idle during
    decode.
- **NPU:** a separate, *distilled*, fixed-shape int8 model for side tasks: routing requests,
  classification, embeddings. bitnet-jolt's 2B controller is the existing example.
  - A dense **draft** model for the NPU becomes worth it only after the runtime supports one
    command per token. Then the shape would be d 1024, ~12 layers, N ≤ 4096 everywhere, int8
    weights ~0.15-0.3 GB, and a 32K frequency-ranked head.
  - **Distilling it:** EAGLE/DistillSpec-style KD from the trunk on ~2-5 B tokens of
    teacher outputs: 6 x 0.3 B x 5 B + 2 x 6.6 B x 5 B ≈ 7.5e19 FLOP ≈ 100-150 H100-h,
    **$300-700** [I]. It is cheap to train; the blocker is the NPU runtime, not the model.
- **How the adaptation paths make the split cleaner.**
  - 2-bit experts (rank 4) and lower k (rank 4b) shrink the expert share of verify bandwidth.
    That makes *any* drafter more valuable, the NPU one included.
  - HC removal (5b) cuts ~200 dispatches per token. That makes the GPU's decode step shorter
    and more regular, so a concurrent drafter's window (the verify step) has to fit a tighter
    budget.
  - A trimmed vocabulary for draft heads removes the single largest draft cost on every
    engine.

**Bottom line on hybrids.** On this machine the honest gains come from:
- capacity (lookup tables);
- CPU drafting (no bytes);
- the NPU running *other* models.

Splitting this model's layers across engines saves under ~5% at measured co-processor rates.
- Up to ~10% of prefill if the NPU's marginal GEMM rate becomes end to end.
- 8-18% is lost to co-tenancy when done carelessly [M].

---

## 7. Data and teacher logits for the paid paths

- **Pretraining-style text** (for KD, alignment and Block-AP calibration):
  - [DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0), which
    RADLADS chose over FineWeb and FineWeb-Edu [P];
  - [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu);
  - RedPajama/SlimPajama, which EfficientQAT's Block-AP uses [P].
- **Chat behaviour** (Qwen3.8-Flash-Next is an instruct model):
  - prompts from Tulu-3 SFT, UltraChat and LMSYS-Chat;
  - responses generated by the teacher itself (self-distillation keeps the output
    distribution);
  - a mix of ~50/50 with the text above [I].
- **Teacher logits.**
  - Cache top-32 (id, logit) pairs per position: ~0.5 MB per 2048 tokens, ~250 GB per 1 B
    tokens [D]. On rented GPUs the teacher forward costs ~2·N_active·D ≈ 1.3e19 FLOP per
    1 B tokens, ~20 H100-h [D].
  - This machine prefills at ~1,300 tok/s, so 1 B tokens would take ~9 days of its GPU [D].
    Too slow, and it collides with serving. bitnet-jolt measured +48% controller TTFT under
    co-resident training [M]. Rent it.
- **Local training is not an option for these paths.**
  - bitnet-jolt measured 650-1,700 tok/s of LoRA on 1-3 B dense models on this GPU [M].
  - A 125 B MoE does not fit in bf16 (250 GB), and it would be far slower even at 4-bit
    frozen weights [I].

---

## 8. What bitnet-jolt already established that this builds on

- **BitNet b1.58-2B-4T on XDNA2.**
  - Bit-exact NPU int8 GEMMs; perplexity identical to 4 decimal places.
  - 1.12x prefill at 2048 tokens only when the NPU and CPU run concurrently on disjoint
    halves.
  - Ternary is expanded to int8 for the NPU (4x the bytes).
  - **Lesson reused here:** low-bit formats pay only where the compute path consumes them
    natively or dequantizes cheaply.
- **NPU programming realities.**
  - 0.197 ms dispatch round trip.
  - Hardware-context switching costs +53% to +210% per dispatch.
  - The `aie.dma_bd` N ≤ 4096 limit.
  - Concurrent contexts segfault, hence a single-flight lease.
  - A warm controller engages the NPU 0% of the time.
- **Tri-device co-tenancy is measured:** the GPU worker loses 8-18%; NPU offload of a side
  model beats CPU offload for everyone (§6.1).
- **PEFT works on BitNet's BF16 master and on the hybrids.** The only BitNet obstacle was a
  stale `auto_map`.
  - Adapting the recurrent `Conv1d` adds +0.55% parameters on LFM2.5 and +6.1% on Qwen3.5.
  - Hybrids train faster and need less memory than full attention: LFM2.5-2.6B 784 tok/s at
    22.7 GiB vs BitNet-2B 671 tok/s at 35.8 GiB.
- **Recurrent state cannot roll back one token.** The exact-spine work fixed prefix reuse by
  saving at the token-exact boundary. Speculative decoding on DeltaNet needs the same care
  (rank 2).
- **What it did not do:** no ternary *conversion* or distillation of a pretrained model. It
  used Microsoft's natively trained BitNet. Ranks 3 and 7 here would be new work.

---

## Sources

Published: [BitDistill](https://arxiv.org/abs/2510.13998) ·
[BitNet b1.58](https://arxiv.org/abs/2402.17764) · [BitNet 2B4T](https://arxiv.org/abs/2504.12285) ·
[Continual 1.58-bit pretraining](https://arxiv.org/abs/2502.11895) ·
[HF 1.58-bit fine-tuning](https://github.com/huggingface/blog/blob/main/1_58_llm_extreme_quantization.md) ·
[LLM-QAT](https://arxiv.org/abs/2305.17888) · [EfficientQAT](https://arxiv.org/abs/2407.11062) ·
[ParetoQ](https://arxiv.org/abs/2502.02631) · [QuIP#](https://arxiv.org/abs/2402.04396) ·
[AQLM](https://arxiv.org/abs/2401.06118) · [QTIP](https://arxiv.org/abs/2406.11235) ·
[QuaRot](https://arxiv.org/abs/2404.00456) · [SpinQuant](https://arxiv.org/abs/2405.16406) ·
[MoQE](https://arxiv.org/abs/2310.02410) · [QMoE](https://arxiv.org/abs/2310.16795) ·
[REAP](https://arxiv.org/abs/2510.13999) · [Sparse upcycling](https://arxiv.org/abs/2212.05055) ·
[Fine-grained MoE scaling](https://arxiv.org/abs/2402.07871) · [Ling MoE scaling](https://arxiv.org/abs/2507.17702) ·
[LoLCATs](https://arxiv.org/abs/2410.10254) · [Mamba in the Llama](https://arxiv.org/abs/2408.15237) ·
[MOHAWK](https://arxiv.org/abs/2408.10189) · [Llamba](https://arxiv.org/abs/2502.14458) ·
[RADLADS](https://arxiv.org/abs/2505.03005) · [Liger](https://arxiv.org/abs/2503.01496) ·
[Jet-Nemotron](https://arxiv.org/abs/2508.15884) · [Minitron](https://arxiv.org/abs/2407.14679) ·
[Minitron in practice](https://arxiv.org/abs/2408.11796) · [ShortGPT](https://arxiv.org/abs/2403.03853) ·
[Hyper-Connections](https://arxiv.org/abs/2409.19606) · [mHC](https://arxiv.org/abs/2512.24880) ·
[Engram](https://arxiv.org/abs/2601.07372) · [EAGLE-3](https://arxiv.org/abs/2503.01840) ·
[FR-Spec](https://arxiv.org/abs/2502.14856) · [Gated DeltaNet](https://arxiv.org/abs/2412.06464) ·
[Qwen3-Next (HF docs)](https://huggingface.co/docs/transformers/en/model_doc/qwen3_next) ·
[RDNA3 WMMA (GPUOpen)](https://gpuopen.com/learn/wmma_on_rdna3/) ·
[RDNA3/4 reference (zolotukhin.ai)](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/) ·
[Strix Halo memory subsystem (Chips and Cheese)](https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling) ·
[GPU prices (gpuperhour)](https://gpuperhour.com/) ·
[GPU prices (Spheron)](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/).
