# Roofline: pp2048 prefill (and decode), worked backwards

This document works backwards from hardware limits. It derives what one 2048-token prefill chunk
(T = 2048, depth 0) must cost at minimum and what is realistic. It then states what each
operation's kernel has to look like to get there, and ranks the gaps between that and what we
run today.
- **Analysis only.** Nothing here was run on the GPU.
- **Where the numbers come from.** Each number is derived from model dimensions
  (`src/model/config.hpp`, PLAN "Model facts"), from measured hardware facts, or from an
  assumption labelled **A*n***.
- **Measured times** come from a sync-free per-kernel timing of one prefill
  (`kernel_timing_nix_pre_dpp.txt`, span 2719 ms, clock pinned about 9% below boost).
  - That snapshot predates three changes: DPP/float4 DeltaNet (cbb4a18), gate/up and down weight
    prefetch (0a7f2e7, e708ca5, 9119be5), and the fused inject (791ab3f).
  - Where those changed a row, the gap table says so.
  - The snapshot's per-kernel times add up to the whole span (2719.37 ms over about 3,990
    launches), so launch gaps are included in the kernel rows.

## 0. Hardware facts and assumptions

| | Value | Source |
|---|---|---|
| FP16 WMMA peak (register-only) | 51.5 TFLOPS (1.29 TFLOPS/CU; one 16x16x16 ≈ 12.7 ns per SIMD) | measured |
| INT8 WMMA peak | 56 TOPS | measured |
| DRAM read ceiling | 241 GB/s | measured (P0) |
| VALU fp32, single issue | 80 SIMD x 32 lanes x 2.8 GHz = 7.2 T lane-ops/s (14.3 TFLOPS FMA) | derived; dual-issue VOPD ignored |
| A1 MALL (Infinity Cache) | 32 MB, **~900 GB/s** read, ~135-200 ns unloaded latency | measured (`halo_gemm_bench memory`, clock pinned): 850-945 GB/s for 12-32 MB working sets, 242 GB/s from 48 MB; loaded latency unmeasured. The figures below at 600 GB/s are conservative |
| A2 L2 | 2 MB, ~80 ns unloaded | latency measured (dependent loads); bandwidth unmeasured |
| A3 DRAM loaded latency | ~1.0 µs | assumption; unloaded latency measured at 250-380 ns (rising with TLB reach, 48-512 MB) |
| A4 LDS | 128 B/clk/CU, ~64 clk latency; 64 KB per workgroup, 128 KB per WGP | LDS sizes given; bandwidth and latency are assumptions |
| A5 attainable efficiency | large dense GEMMs 65% of WMMA (33.5 TFLOPS; gufo does 30-37); skinny/odd GEMMs and attention 50%; streaming 75% of DRAM (181 GB/s; gufo experts 160-190); simple elementwise 80% (193 GB/s); VALU-bound loops 55-60% of issue | assumption |
| A6 dependent-launch cost | ~5-10 µs drain plus ramp per dependent kernel boundary | assumption |

Q8_0 is 34 B per 32 values (1.0625 B/param), Q4_K 144 B/256 (0.5625), Q5_1 24 B/32 (0.75).
T x D fp32 (D = 4 x 2560 = 10240) is **83.9 MB**; T x H fp32 (H = 2560) is 21 MB. At 241 GB/s,
one 84 MB pass takes **0.35 ms**.

## 1. Work inventory for pp2048

### 1.1 Weights and FLOPs (FLOP = 2 x params x tokens)

| Operation (layers) | Shape per layer | FLOP | Weight bytes read | t at WMMA peak | t at 241 GB/s | Bound |
|---|---|---:|---:|---:|---:|---|
| Mixer down, 2 per layer (48) | 320 x 10240 Q8_0 | 1.29 T | 334 MB | 25.0 ms | 1.4 ms | compute (but see 1.2: the residual traffic dominates) |
| Mixer up + stream mix, 2 per layer (48) | 10240 x 320 Q8_0 | 1.29 T | 334 MB | 25.0 | 1.4 | compute and activations |
| Mixer inject (48 x 2) | 4 x 10240 F32 | 0.02 T | 16 MB | VALU 1 | 0.1 | activations (reads xn) |
| GDN qkv (36) | 10240 x 2560 | 3.87 T | 1003 MB | 75.1 | 4.2 | compute |
| GDN z (36) | 6144 x 2560 | 2.32 T | 602 MB | 45.0 | 2.5 | compute |
| GDN out (36) | 2560 x 6144 | 2.32 T | 602 MB | 45.0 | 2.5 | compute |
| GDN alpha+beta (36) | 96 x 2560 F32 | 0.04 T | 35 MB | VALU 5 | 0.1 | VALU |
| Attn q+gate (12) | 12288 x 2560 | 1.55 T | 401 MB | 30.0 | 1.7 | compute |
| Attn k+v (12) | 1024 x 2560 | 0.13 T | 33 MB | 2.5 | 0.1 | compute |
| Attn out (12) | 2560 x 6144 | 0.77 T | 201 MB | 15.0 | 0.8 | compute |
| Indexer q+k (12) | 640 x 2560 BF16 | 0.08 T | 39 MB | 1.6 | 0.2 | compute |
| Router (48) | 512 x 2560 F32 | 0.26 T | 252 MB | VALU 18 | 1.0 | VALU (F32) |
| Shared expert gate+up, down (48) | 1280 x 2560, 2560 x 640 | 0.97 T | 251 MB | 18.8 | 1.0 | compute |
| PLE key+value (1) | 12800 x 2560 | 0.13 T | 35 MB | 2.6 | 0.1 | compute |
| **Dense total** | | **15.0 T** | **4.14 GB** | **292 ms** | **17 ms** | compute |
| **Routed experts** (48) | 512 x (2 x 640 x 2560 Q4_K + 2560 x 640 Q5_1) | 9.66 T (x1.2 padded) | **75.5 GB** | 188 (225 padded) | **313 ms** | **bandwidth** |
| Attention core, dense causal (12) | 24 heads, d 256, T²/2 pairs | 0.62 T | K/V on chip or MALL | 12.0 | — | compute |
| DeltaNet recurrence (36) | 48 heads, 128 x 128 state | 0.46 T fp32 VALU | state 3.1 MB per layer | VALU **32-40 ms** | 0.5 | VALU issue |
| LM head, last token only | 248320 x 2560 Q8_0 | — | 675 MB | — | 2.8 | bandwidth |

**Routed experts: every expert's weights are read on every chunk.**
- With 2048 x 10 = 20,480 (token, expert) pairs over 512 experts, an expert gets **~40 tokens
  on average**.
- An expert is left untouched with probability (1 - 10/512)^2048 ≈ 3e-18. So all 75.5 GB are
  streamed: 1.84 MB of gate/up plus 1.23 MB of down per expert per layer.
- Weight reuse is therefore n ≈ 40, and the arithmetic intensity is fixed by the routing:
  - gate/up: 2 x 40 / 0.5625 B = **142 FLOP/B**;
  - down: 2 x 40 / 0.75 B = **107 FLOP/B**.
- Against the ridge of 51.5 T / 241 G = 214 FLOP/B (or 185 at the A5 efficiencies), experts are
  **bandwidth-bound by 1.5-2x**.
- WMMA work is 16-token granular (`subs = ceil(n/16)`). That pads by about 8 tokens per expert,
  about 1.2x, so 11.6 TFLOP effective.

**DeltaNet recurrence floor.**
- **VALU work.** Per state element per token the recurrence does four VALU ops: S·k (FMA),
  S·q (FMA), and the update `s = decay*s + delta*k` (mul + FMA). Over 128² x 48 heads x 2048 x 36
  layers x 4 that is 232 G lane-ops, or 32 ms at 7.2 T lane-ops/s. With LDS reads, DPP and the
  output that becomes **~40 ms**.
- **Dependent chain per token**, per wave: decay → 32-term dot split into 4 accumulators
  (~8 dependent FMAs plus 3 adds) → 2 DPP adds → delta (2 ops) → update FMA. That is ~12 dependent
  ops, **~85 clk**.
  - Over 2048 tokens: 2048 x 85 clk = 62 µs per layer, **2.2 ms in total**.
- **So the recurrence is issue-bound, not serial-latency-bound**, provided nothing slower (an LDS
  round trip, a barrier, a global load) sits on the per-token chain.
  - A chunked (WY) form on WMMA could go below 32 ms. It isn't needed to reach the target.

**Attention.**
- At depth 0 every token sees every earlier one (QSA selects nothing), so the cost is dense
  causal: 2 x 2 x (T²/2) x 256 x 24 heads x 12 layers = 0.62 TFLOP, or 12 ms at peak.
- GQA has 12 q heads per KV head. A block holding R query rows reuses each K/V row R times.

### 1.2 Activations that cannot stay on chip

The residual stream is 4 x 2560 fp32 per token, 84 MB per chunk. That is larger than the MALL
(A1), so each hyper-connection combine must read and write it from DRAM: **168 MB per half-layer
x 96 = 16.1 GB, 67 ms**.

The mixer's stream mix needs `xn = res·scale·w` again after the down/up projections. Unless the
whole mixer runs per sub-chunk while that sub-chunk's residual is in the MALL, this is one more
84 MB read: +33 ms.

What remains of a block's own activations can be written in fp16 by GEMM epilogues and read by
the next kernel while that kernel is compute- or VALU-bound, so it hides:
- DeltaNet: q/k/v 42 MB, z 25 MB, out 25 MB;
- attention: q 25 MB, gate 25 MB, ctx 25 MB.

`mixed` (10.5 MB fp16) and `block_out` (21 MB) fit the MALL.

### 1.3 Lower bound

Taking each operation at max(compute, bytes), in series (layers are dependent):

| Part | Floor |
|---|---:|
| Dense non-mixer GEMMs (12.1 TFLOP at peak) | 236 ms |
| Mixers: 96 x (combine+down: 168 MB at 241 GB/s, 0.70 ms; up+mix: 84 MB re-read 0.35 ms vs 0.26 ms compute) | 101 ms |
| Routed experts (75.5 GB at 241 GB/s) | 313 ms |
| Attention core | 12 ms |
| DeltaNet recurrence (VALU) | 40 ms |
| F32 router + alpha/beta + shared gate (VALU) | 20 ms |
| LM head, embedding, PLE, top-k | 5 ms |
| **Serial lower bound** | **~727 ms (≈2,820 tok/s)** |

- **Perfect overlap.** If compute and memory overlapped perfectly across the whole chunk, the
  bound would be max(25.3 TFLOP of WMMA = 491 ms, ~96 GB of DRAM = 400 ms). That is about
  0.49 s, unreachable with dependent layers.
- **Where the floor is.** It is half compute (the dense GEMMs) and half bandwidth (the experts).
  No single trick removes both halves.

## 2. Realistic target

These apply the A5 efficiencies.

| Part | Derivation | Target |
|---|---|---:|
| Dense non-mixer GEMMs | 12.13 TFLOP / 33.5 TFLOPS | 362 ms |
| Mixers + combines | 96 x (A: 168 MB / 193 GB/s = 0.87 ms; B: max(84 MB / 193 GB/s = 0.43, 13.4 GFLOP / 25.7 TFLOPS = 0.52)) | 134 → **145 ms** with small kernels |
| Routed experts gate/up + down | 75.5 GB / 181 GB/s; needs WMMA ≥ 11.6 TFLOP / 0.42 s = 54% of peak at the same time | **420 ms** (gate/up 252, down 168) |
| MoE glue: top-k, bucket, fp16 perm, yp write/read + combine | yp 105 MB written and 105 MB read per layer at 241 GB/s = 0.87 ms x 48 = 42 | **45 ms** |
| F32 router + alpha/beta + sgate | 0.295 TFLOP fp32 at ~60% VALU | 35 ms |
| DeltaNet recurrence | 40 ms at ~55% issue | 75 ms |
| DeltaNet glue (conv+L2 fused into the qkv epilogue, gated norm fused) | | 10 ms |
| Attention core | 0.62 TFLOP at 50% | 24 ms |
| Attention glue (q/k norm, RoPE, split, KV write in the GEMM epilogue) | | 3 ms |
| Shared-expert glue, embedding, PLE, LM head | | 6 ms |
| Launch / dependency gaps | ~1,500 dependent launches x ~10 µs (A6) | 15 ms |
| **Total** | | **≈1,140 ms → 1,800 tok/s** |
| Same, with router + shared expert overlapped under the bandwidth-bound routed experts | -(29 + 30) ms | ≈1,080 ms → 1,900 tok/s |

- **Against the reference points.** gufo is 1.28 s (kernel time 1.36 s). The stretch target is
  1.14 s, which needs essentially every row above at its target.
- **Dense GEMMs are the stiffest row.** At 33.5 TFLOPS they are 362 ms; at gufo's low end of
  30 TFLOPS they are 404 ms.
- **The MoE row trades directly against bandwidth.** At 160 GB/s the experts cost 472 ms
  (+52 ms).

## 3. Required structure per operation

### 3.1 Bytes in flight (Little's law: in-flight = bandwidth x latency)

| Source | Rate needed | Latency (assumed) | In flight, chip | Per CU (of 40) |
|---|---|---|---:|---:|
| DRAM, experts at 181 GB/s | 181 GB/s | 1.0 µs (A3) | 181 KB | **4.5 KB** |
| DRAM, full 241 GB/s | 241 GB/s | 1.0 µs | 241 KB | 6 KB |
| MALL, attention K/V at ~600 GB/s | 600 GB/s | 0.5 µs (A1) | 300 KB | 7.5 KB |
| L2/MALL, dense GEMM operands (see below) | 262 GB/s | 0.25-0.5 µs | 65-130 KB | 1.6-3.3 KB |
| LDS, per CU | 358 GB/s | 23 ns (A4) | — | 8 KB |

Two things follow.
1. **Byte volume in flight is not the problem.** Every current kernel issues more than the
   per-CU need: dense GEMM 32 KB per stage per block; gate/up 2.3 KB per wave per stage with
   16 waves per CU.
2. **The problem is *when* the loads are issued.** A one-stage-ahead prefetch with two barriers
   per stage puts all 8 waves of a block in lock-step. Each time the stage's math runs shorter than
   the latency, the whole block waits. The thread traces' 65-82% `s_waitcnt` is that collapse,
   not a lack of parallelism. A scratch spill of the prefetch registers (dense GEMM) makes the load
   synchronous outright.

### 3.2 Dense GEMMs (Q8_0 → FP16 WMMA): `GemmQ8rF16`, core.hip:4250

**Tile.** BM 256 x BN 128 x KS 64, 8 waves, each wave 32 rows x 128 tokens (TM 2, TN 8).
- Per block and stage: A is 16 KB int8 plus 0.5 KB of scales; B is 16 KB fp16. The math is
  4.19 MFLOP.
- Arithmetic intensity against L2/MALL is 2·256·128 / (256·1 + 128·2) = **128 FLOP/B**. At
  33.5 TFLOPS that needs 262 GB/s from L2/MALL.
- Each 256-row weight slab is shared by the 16 token-tile blocks launched back to back, so DRAM
  sees only the 4.1 GB of weights.
- A stage takes 4.19 MFLOP / (0.84 TFLOPS/CU at 65%) = 5 µs. That is far longer than any latency
  in 3.1, so **one stage ahead is enough, if it is real**.

**VGPR budget.** The accumulators are fixed at 16 tiles x 8 = **128**. A and B fragments add 16 + 8.
The prefetch adds int4 x 4 (16), uint4 x 4 (16) and 4 scales (core.hip:4268-4270). 64-bit row
addresses for 8 load streams take about 16 more, plus indices and temporaries. That totals
228-256, with spills; the trace shows the prefetch landing in scratch.

Occupancy is fixed by LDS, not VGPRs: (256 + 128) x 72 x 2 B = 55 KB per block, so 2 blocks per WGP
and 4 waves per SIMD. **Any count ≤ 256 VGPRs with zero scratch gives the same occupancy.** "≤ 200"
matters only for tiles small enough for 2 blocks per CU.

Ways to cut the ~30 VGPRs:
- buffer loads with an SGPR base and a 32-bit VGPR offset (about 16 saved);
- loop-invariant offsets bumped by KS instead of recomputed;
- scales packed 2 per VGPR.

gufo runs the same tile shape (256 x 128, 8 waves, one-stage register prefetch, one LDS buffer) at
30-37 TFLOPS. So this is register allocation, not design.

**Measured.** At the profile clock: qkv 28.2 TFLOPS, z 25.3, attn q 27.0, out 23.9. The small-row
shapes run at 14: shexp gate/up (3 row blocks of 256 for 640 rows, 17% padding) and k/v.

**Fusions owed** (gufo has all of them):
- q/k RMSNorm, RoPE, the q/gate split and the K/V cache write (fp16) in the q/k/v epilogue;
- conv + SiLU + per-head L2 in the GDN qkv epilogue. `halo_conv_l2norm_prefill`'s 128-channel
  block is exactly one head, so a 256-row tile with a 3-token halo covers it.
- SwiGLU in a single shared gate+up GEMM epilogue (1280 rows);
- the sigmoid-gated add in the shared down epilogue.

### 3.3 Hyper-connection mixers: the largest structural gap

**Current per half-layer** (from prefill_chunk.cpp:226-247 and 754-776):

| Kernel | DRAM traffic | Snapshot ms per call |
|---|---|---:|
| `halo_hc_combine_norm_rows` | reads res 84 + block_out 21; writes res 84, **xn fp32 84**, **xh fp16 42** = 315 MB (1.31 ms at 241 GB/s) | 1.55 |
| `halo_gemm_q8r_f16_m` (down, 320 x 10240) | xh 42 MB; **grid 32 x 1 = 32 workgroups on 40 CUs**; K = 10240 in 160 stages of 8 µs (math 3.1 µs) | 1.29 |
| `halo_gemm_q8r_f16_mix` (up + mix) | K = 320 is only 5 stages; the epilogue re-reads **xn fp32 84 MB** in 32-byte pieces (8 consecutive hidden indices per token per wave, core.hip:4343-4357); writes mixed 21 + 10.5 | 1.89 |
| inject GEMM (snapshot only) | re-read xn 84 MB, 32 workgroups (`f32_tiled [32x1]`) | ~1.30 |

**Required structure: two kernels per half-layer.**

- **Kernel A** (combine + norm + down):
  - Token-tiled; split-K by **stream**: 4 x 2560, so 4 x 32 = 128 workgroups.
  - Each split reads its stream's res slice and block_out, writes the combined res, and squares
    for the norm.
  - It runs WMMA on `res·w` and applies the per-(token, stream) RMS scale in the epilogue:
    `lo = Σ_s scale_{t,s}·partial_s`, which is linear, so no second pass is needed.
  - Inject partials are as now (791ab3f).
  - Traffic is 168 MB; xn and xh are never written. Target **0.87 ms**.
- **Kernel B** (up + mix):
  - The waves are laid out along tokens (WM 1, WN 8, TM 16), so a wave's epilogue covers 64
    consecutive hidden indices x 4 streams: 256 B pieces.
  - It rebuilds `xn` from res x scale x w (84 MB, or 42 MB if it reads an fp16 copy; the latter
    must pass the quality gate).
  - Target **0.52 ms**.
- **Optional:** run A and B over 256-token sub-chunks (10.5 MB of res) so that B's re-read hits
  the MALL. That gets to ~1.05 ms per half-layer (101 ms in total).

### 3.4 Routed experts (`MoeGateUp` / `MoeDown`, core.hip:3718-3870)

**Shape facts.**
- n ≈ 40 tokens per expert; WMMA work is padded to 16 (`subs`), but activation staging always
  loads 64 rows (`kXp = 4`, unconditional).
- Gate/up: 5 row-blocks of 128 per expert tile, K = 2560 in 20 stages.
- Down: 20 row-blocks, **K = 640 in only 5 stages**.

**Measured (snapshot).**
- gate/up: 7.46 ms per layer = **126 GB/s**, 18 TFLOPS unpadded.
- down: 6.25 ms = **101 GB/s**, 10.7 TFLOPS.
- The target is 181 GB/s for both: 5.2 and 3.5 ms per layer.

**What is wrong with down.**
- The activation (hp) staging, core.hip:3838-3845, is a **synchronous global→LDS copy between two
  barriers on every stage**. Only the weights are fetched early (`load` at :3837).
- Each of 5 stages exposes a MALL/L2 round trip, about 0.5 µs (A1), against about 1.2 µs of math
  per SIMD.
- hp for one expert tile is staged again by each of 20 row-blocks: 20 x 26 MB = 0.5 GB per layer
  from the MALL.
- The epilogue writes `yp` in fp16, 105 MB per layer (0.44 ms of the 6.25).

**Required structure.**
- **Gate/up.** Keep one wave per 16 weight rows. Each weight byte is read once from DRAM and
  feeds `subs` WMMAs, which is the right reuse. Then:
  - prefetch **2 stages** (2 x 2.3 KB per wave), or let independent blocks interleave (3-4 per CU),
    so DRAM sees ≥ 4.5 KB per CU at every instant;
  - stage only `16·subs` activation rows.
- **Down.** Make a block own one expert tile across many rows (persistent over its 20
  row-blocks). It stages hp once (40 x 640 x 2 = 51 KB, which fits the 64 KB LDS) and then streams
  weights with a 2-3-stage register ring and no per-stage barrier.
- **yp.** The 210 MB per layer write+read (42 ms) is the remaining structural cost. Fusing the
  weighted combine into the next HC combine (gufo) removes the `block_out` pass, not the yp read.
  Fp32 atomics into `block_out` (21 MB, MALL-resident) would remove it but cost 52 M atomics per
  layer; that is unmeasured and likely not cheaper.
- **The VALU dequant shares the SIMD with WMMA** (RDNA3 has no separate matrix pipe). It is about
  16 VALU per 16-value fragment against 3 x 32-clock WMMAs, roughly +17% on WMMA time. That is why
  the target needs ≥ 54% WMMA *and* 75% DRAM at the same time.

### 3.5 DeltaNet recurrence (`halo_gdn_prefill_split`, core.hip:4132)

**Layout.** Grid [2 x 48]: 96 blocks x 8 waves = 9.6 waves per SIMD.
- Each thread holds 32 state values (rows i0..i0+31 of column j).
- Per token: 16 `ds_read_b128` for k and q, 96 FMAs for `kq`, A and B, 6 DPP ops, then a 32-wide
  update.
- **Issue floor:** about 1,250 clk per token per SIMD, so 0.9 ms per layer, about 33 ms in all.

**Measured.** The snapshot shows **15.4 ms per layer** (553 ms), which is 17x that floor. The
in-code comment (core.hip:4196-4199) attributes 553 vs 308 ms to clang 22 splitting the float4
reads. With DPP replacing `ds_bpermute` (73% LDS waits in the trace), the current figure is
estimated at 200-300 ms; not re-measured here.

**Required structure.**
- **Nothing but VALU and DPP between consecutive tokens' updates.** k, q, v, decay and beta for
  the next 8-16 tokens are already in LDS or registers before the token loop (as now).
- **Compute `kq = k·q` once per token** during staging rather than in all 64 columns
  (core.hip:4208). That removes a quarter of the FMAs.
- **Barriers no more than once per 8-16 tokens.**
- **Fusions:** the gated RMSNorm, which needs the head's 128 columns and so a split-to-1-block
  layout or a per-head sum-of-squares exchange; and fp16 output for the out GEMM.

Target: 75 ms. gufo: 130 ms.

### 3.6 Attention (`halo_attention_wmma_mq`, core.hip:4429)

**Current.** One block per (KV head, 4 tokens): 48 rows, 1,024 blocks per layer.
- Each block streams **fp32** K and V up to its last token and converts them to fp16 in LDS,
  32 keys per chunk with at least 3 barriers (:4459).
- K/V traffic is 1,024 blocks x ~1,024 keys x 2 KB = **2.1 GB per layer** from the MALL.
  At ~600 GB/s (A1) that is 3.5 ms per layer of the 9.1 measured.
- LDS is 52 KB per block, so 1 block per CU.

**Required.**
- Write K/V as fp16 in the k/v GEMM epilogue, after norm and RoPE.
- Rows per block are capped by LDS at head dim 256: Q 48 rows (24 KB) + K 16 KB + V 16 KB + P ≈
  59 KB. So reuse stays at about 48. The traffic is then 51.5 GFLOP / (48 FLOP/B) = **1.07 GB per
  layer**, 1.8 ms at 600 GB/s.
- Keep S and P in registers, with the softmax per wave-owned row tile and P shared through LDS once
  per chunk.
- Use 64-key chunks to halve the barriers per key.

Target: **~2 ms per layer**. If the MALL bandwidth is only 400 GB/s, ~2.7 ms.

### 3.7 Activation round trips in the current pipeline (DRAM, ≥ 40 MB)

| Round trip | Bytes per call | Calls | ms at 241 GB/s | Avoidable by |
|---|---:|---:|---:|---|
| `xn` fp32 written by combine_norm, re-read by the mix epilogue | 168 MB | 96 | **67** | kernel A/B (3.3) |
| `xh` fp16 written by combine_norm, read by the mixer down | 84 MB | 96 | 33 | down reads res directly |
| inject GEMM re-read of xn (snapshot only) | 84 MB | 96 | 33 | done: 791ab3f |
| GDN qkv fp32 → conv → conv fp32 → recurrence | 336 MB | 36 | 50 | conv in the qkv epilogue, fp16: 84 MB (12.5) |
| GDN z fp32 write + read | 100 MB | 36 | 15 | fp16 (7.5) |
| GDN out fp32 → gated norm → xh → out GEMM | 150 MB | 36 | 22 | norm fused, fp16 (7.5) |
| Attn qg write → split → norm → RoPE → attention → ctx → f16 → out | ~680 MB | 12 | 34 | epilogue fusion: 150 MB (7.5) |
| MoE yp fp16 write + combine read | 210 MB | 48 | 42 | atomics / fused accumulation (unproven) |
| MoE hp fp16 write + read | 52 MB | 48 | 10 | fused gate/up→down per expert |
| **Total at the DRAM floor** | | | **~306** | **≈ 215 ms recoverable** |

The unavoidable residual combine read and write (168 MB, 67 ms) is not in this table.

## 4. Gap table (snapshot measured vs target, ranked by ms recoverable)

The snapshot ran at a clock about 9% below boost, so the gaps on compute-bound rows are overstated
by about 9%.

| # | Operation | Measured (kernels) | Target | Gap | Structural cause |
|---|---|---:|---:|---:|---|
| 1 | DeltaNet recurrence | 553 (`gdn_prefill_split`) | 75 | **478**; ~125-225 after cbb4a18 | per-token LDS round trips (`ds_bpermute`, narrow LDS reads under clang 22) on the serial chain; redundant `kq` (core.hip:4208); 17x over issue floor |
| 2 | HC mixers + combines | 583: combine_norm 146, mix 181, m 124, inject ~125, misc 7 | 145 | **438**; ~330 after 791ab3f | xn fp32 + xh written then re-read (3.7, 100 ms floor alone); the down GEMM runs 32 workgroups on 40 CUs (prefill_chunk.cpp:173,177); mix epilogue with 32-byte reads and 5-stage K (core.hip:4343) |
| 3 | Routed experts + glue | 720: gate/up 363, down 302, combine 39, top-k/bucket/perm 16 | 465 | **255** (part may be taken by the prefetch commits) | 101-126 GB/s against 181: one-stage prefetch in lock-step; down stages hp synchronously, 5 short stages, 20x re-staged (core.hip:3837-3845); 64-row activation staging with n ≈ 40 |
| 4 | Dense non-mixer GEMMs | 500: qkv 137, out 129*, z 92, q 57, shexp 60*, k/v/idx 19, PLE 5 | 362 | **138** (~95 at boost) | prefetch in scratch at 228-256 VGPRs, so the load is synchronous; 24-28 TFLOPS vs 33.5; 640-row shexp tiles 17% padded |
| 5 | Attention core | 109 (`attention_wmma_mq`) | 24 | **85** | fp32 K/V re-streamed and converted per 4-token block: 2.1 GB per layer from the MALL; 32-key chunks with ≥ 3 barriers |
| 6 | Router + alpha/beta + sgate (F32) | 118: router 79, alpha/beta 23*, sgate 16* | 35 | **83** | fp32 VALU at 3.3 TFLOPS (23%); alpha/beta/sgate as 32-workgroup launches (`f32_tiled [32x1]`); `mixed` read 4 times |
| 7 | DeltaNet glue | 70: conv_l2norm 46, gated_norm_h 24 | 10 | **60** | 336 + 150 MB of fp32 round trips per layer (3.7) |
| 8 | Attention glue | 31: split 10, norms 9.4, RoPE 6.5, f16 4, copies 1.3 | 3 | **28** | not fused into the q/k/v GEMM epilogue |
| 9 | Shared-expert glue | 14: axpy_sigmoid 11, silu_mul 1.4, f16 1.6 | 1 | 13 | not in the GEMM epilogues |
| 10 | Token embedding | 11.5 (2,048 launches, prefill_chunk.cpp:733-738) | 0.2 | 11 | one launch per token |
| 11 | PLE (device) | ~6 | 2 | 4 | small |
| — | LM head (last token) | 2.9 | 2.8 | 0 | at floor |
| | **Total** | **2,719** | **~1,140** | **~1,590** | |

\* These are split by K among calls sharing a grid shape:
- `f16_l [16x10]`: ssm/attn out 129.4, shexp down 13.5, PLE value 1.1.
- `f32_tiled [32x1]`: inject 124.6, alpha/beta 23.4, sgate 15.6.

**Ranking against today's code** (estimated, since the snapshot predates the three changes):
1. mixers ~330;
2. experts ~200-255;
3. DeltaNet ~125-225;
4. dense ~95-138;
5. attention 85;
6. router/F32 83;
7. DeltaNet glue 60;
8. attention glue 28.

The first two (~530-585 ms) are structural fusion and pipelining fixes. Without them 1.14 s is
not reachable even with perfect GEMMs.

### 4.1 Refreshed at 01743f7 (pinned, `halo_run --kernel-timing 1`, device 2,170 ms)

| Row | Now | Target | Gap | Snapshot gap |
|---|---:|---:|---:|---:|
| HC mixers: combine_norm 198, mix 166, `_m` 110, small 3 | ~477 | 145 | **~332** | 438 |
| DeltaNet recurrence (`gdn_prefill_split`, 8.3 ms per call) | 298 | 75 | **~223** | 478 |
| Routed experts: gate/up 283 + 11, down 226 + 33, combine 40, top-k/bucket/perm ~17 | ~610 | 465 | **~145** | 255 |
| F32 router 86 (`f32_tiled [32x8]`, 1.79 ms per call for 2.7 GFLOP), alpha/beta/sgate 44 | 130 | 35 | **~95** | 83 |
| Dense FP16 GEMMs (`_l` 420, `_s` 7, bf16 10) | ~437 | 362 | ~75 | 138 |
| Attention core | 87 | 24 | 63 | 85 |
| DeltaNet glue: conv_l2norm 46.5, gated_norm_h 24 | 70 | 10 | 60 | 60 |
| Attention glue | ~31 | 3 | 28 | 28 |
| Shared-expert glue | 14 | 1 | 13 | 13 |
| Token embedding | 11 | 0.2 | 11 | 11 |

The router row is new to the ranking. At 1.79 ms per call it runs about 9x slower than fp32 VALU
issue allows (~0.2 ms). That is a kernel problem, not a precision one, so it can be fixed without
touching routing numerics.

### 4.2 Trace survey at 4116d65 (`tools/trace_kernel.sh`, 3rd in-engine dispatch of each)

Shares of the traced waves' cycles (one CU's SIMD). "vmcnt" is waiting on global memory and
"lgkmcnt" on LDS or scalar loads.

| Kernel (ms per prefill) | vmcnt | lgkmcnt + LDS | barrier | VALU | WMMA | Reading |
|---|---:|---:|---:|---:|---:|---|
| `hc_combine_norm_rows` (198) | 90% | 2% | 4% | 1% | — | streaming-bound: fix structurally (kernel A, 3.3), not in-kernel |
| `gdn_prefill_split` (298) | 1% | 64% | 26% | 5% | — | serial LDS round trips plus barriers, and SIMDs idle as long as busy (96 workgroups): keep the chain in registers/DPP and raise parallelism |
| `moe_gate_up_grouped_q4_k` (283) | 29% | 7% | 9% | 39% | 10% | dequant VALU-bound as much as memory: why the 6th wave (item 2) bought nothing. Cheaper dequant, or a pre-dequantized layout, is the lever |
| `moe_down_grouped_q5_1` (226) | 31% | 3% | 16% | 36% | 2% | same as gate/up, plus barriers |
| `gemm_q8r_f16_mix` (166) | 86% | 5% | 2% | 4% | 0.4% | the epilogue's `xn` reads (`vmcnt(3)` x248), as section 3.3 predicted |
| `gemm_q8r_f16_m` (110) | 73% | 5% | 17% | 3% | 0.3% | one `vmcnt(0)` per K stage: the prefetch of 1 stage covers too little latency (why split-K didn't help) |
| `attention_wmma_mq` (87) | 48% | 11% | 28% | 7% | 0.7% | K/V streaming and barriers |
| `moe_combine_grouped` (40) | 88% | 4% | — | — | — | streaming |
| `conv_l2norm_prefill` (47) | 69% | 2% | 5% | 3% | — | streaming, plus 20% scalar/branch |

Across the board: WMMA issue is under 10% everywhere. No kernel is compute-bound. The common
remedies are deeper prefetch (more bytes in flight per wave), less traffic (fusion), and for
the experts, cheaper dequant.

## 5. Decode (single token)

**Bytes per token** (BENCHMARKS):
- dense 4.5 GiB (4.83 GB, including the 0.68 GB LM head and 0.25 GB F32 router);
- routed experts 10 x 3.07 MB x 48 = 1.47 GB;
- GDN state read and written, 2 x 113 MB.

That is **6.53 GB**: **27.1 ms at 241 GB/s** (the docs' 26 ms excludes the state), and 32 ms at
85%.

**Where 44.7 ms (HIP) goes.** At their benchmarked rates, the streaming kernels account for about
31 ms: dense 4.83 GB at ~230 GB/s = 21 ms, experts at 161-180 GB/s = 8.7 ms, state about 1 ms.
The other ~14 ms is:
- 1,145 dispatches x 1.8 µs replay = 2 ms;
- one drain/ramp per dependent kernel boundary (A6), which leaves DRAM idle;
- small kernels below bandwidth (norms, combines, top-k, attention decode);
- a GDN step kernel with 153 spills.

**Required structure.**
1. **≤ ~300 dependent launches per token.** Fuse:
   - combine + norm + mixer down + inject;
   - mixer up + mix;
   - qkv + conv + GDN step + gated norm;
   - router + top-k;
   - expert gate/up + SwiGLU, and down + weighted combine;
   - shared expert into the same launch.
2. **Every streaming kernel at ≥ 85% of 241 GB/s.** By Little's law that is ≥ 5 KB in flight per
   CU at all times. For a GEMV that means 2+ independent row-groups per wave with loads issued a
   full row ahead.
3. **The GDN step without spills.**
4. **To move the floor itself:** a cheaper dense quantization (Q8 dense is 74% of the bytes) and
   MTP (PLAN).

## 6. Assumptions that could change the ranking

- **MALL size and bandwidth (A1).** If the MALL is smaller than ~20 MB:
  - `mixed`/`block_out`/x16 re-reads go to DRAM;
  - the GEMM grid order matters: the qkv GEMM's 40 row-blocks would each re-read 10.5 MB of
    activations, 420 MB or 1.7 ms;
  - the attention K/V stream (2.1 GB per layer now) becomes DRAM-bound. Attention would then rise
    to #3 and the dense GEMMs would need token-outer ordering.
  - If the MALL bandwidth is ~400 rather than 600 GB/s, the attention target goes from 24 to
    ~32 ms.
- **The current DeltaNet time.** The snapshot's 553 ms is the clang-22 regression. If today it is
  ≤ 200 ms, DeltaNet drops below experts and dense. One re-profile settles the ranking of rows 1-4.
- **DRAM loaded latency (A3) and WMMA/VALU co-issue.**
  - If the latency is ≥ 2 µs, the in-flight needs double (9-12 KB per CU), and the experts'
    one-stage prefetch can't cover it at any occupancy.
  - If dequant VALU and WMMA serialize worse than +17%, experts become compute-co-bound at about
    360-450 ms, and int8 or pre-dequantized formats become the lever.
- **Profile clock.** The compute-bound rows (dense, attention, DeltaNet) are about 9% slower in the
  snapshot than at boost. Bandwidth-bound rows (experts, mixers) barely change. This favours the
  memory-structure fixes in the ranking.
- **Apportioning shared grid shapes by K (\*).** Small-K GEMMs carry relatively more fixed
  epilogue cost, so shexp down is probably larger and out smaller than shown. The totals are
  unaffected.

## 7. Work queue (prefill first, ranked)

Every item states its hypothesis, a prediction, and the measurement that decides it
(`tools/ab_bench.py` with the clock pinned, plus `halo_run --kernel-timing 1` for attribution).
Predictions are against the snapshot in section 4 and get revised after step 0.

**0. Measure before changing anything.**
1. ~~Run `halo_gemm_bench memory` with the clock pinned.~~ Done. The MALL holds the full 32 MB at
   ~900 GB/s (A1). The "cliff at 12-16 MB" was a benchmark bug: unwritten pages, and a repeat
   loop inside the kernel that let resident blocks re-read their slice from cache.
2. ~~Paired quality verdict on the 12 long cases.~~ Done: prefill scores like stepping (BENCHMARKS).
3. ~~A/B against the last timed commit~~: superseded by a fresh per-kernel profile at HEAD
   (section 4.1). Attributing the nine commits one by one would not change the queue.
4. ~~Refresh section 4.~~ Done: 4.1.

**1. Native-vector prefetch arrays in `GemmQ8rF16` (`_l`, `_s`, `_mix`).** DONE (01743f7):
measured **-83.5 ms** (2256 -> 2173 ms, pinned), at the low end of the prediction.
- Hypothesis: the 80-144 B of scratch is an alloca that AMDGPUPromoteAlloca rejects. HIP's
  `uint4` is a struct, and arrays of structs are not promoted before LLVM PR #217055. That keeps
  the prefetch in scratch and the load synchronous: the thread trace shows `vmcnt(4)` as about 24%
  of stalls. Declaring the arrays as `ext_vector_type(4)` should let the pass promote them.
- Prediction: scratch drops to 0 with VGPRs ≤ 256. The FP16 GEMMs (dense row 4 plus the mixer
  GEMMs in row 2, about 800 ms in the snapshot) recover 10-20%, so **-80 to -160 ms**.
  Occupancy does not change: these kernels are LDS-limited at 4 waves.
- Check first: dump device IR (`-mllvm -print-after=amdgpu-promote-alloca`) to confirm that this
  alloca is the one that survives. If the kernel then spills real VGPRs, shrink the prefetch depth
  rather than accept spills.

**2. Grouped MoE gate/up at ≤ 240 VGPRs (Q4_K 241, Q5_1 243).** TRIED, NOT KEPT: measured
-1.7 ms against item 1 (noise; 3 rounds x 3 reps, pinned). One more wave does not help, so gate/up
is not latency-hiding-bound: look at the stage structure and issue instead (row 3).
- Hypothesis: on gfx1151 VGPRs are allocated in blocks of 24, so 241-256 VGPRs give 5 waves per
  SIMD and ≤ 240 give 6. Gate/up is bandwidth-bound at 101-126 GB/s against 181. One more wave
  adds about 20% more bytes in flight.
- Prediction: **-20 to -50 ms** of gate/up's 363 (part of row 3). Under 10 ms means latency
  hiding is not the limit, so look at the stage structure instead.
- Means, in order: trim live ranges in the source; then `"amdgpu-waves-per-eu"="6"` on that
  kernel; then `"amdgpu-sched-strategy"`, trying `iterative-minreg` per kernel. Reject any form
  that adds scratch.

**3. LDS width audit.** Done (disassembly at HEAD, nix build): all LDS loads in the dense GEMMs,
grouped MoE gate/up and down, and `gemm_f32_tiled` are already `ds_load_b128`. Two exceptions
remain:
- `halo_attention_wmma_mq`: 64 `ds_load_u16_d16`/`_hi` and 66 `ds_load_b32`, probably the
  per-element fp16 conversion of K/V. Fold this into attention (row 5).
- The FP16 GEMM's 64 `ds_store_2addr_b32` per `_l` are the epilogue's accumulator writes,
  once per output tile, not per K stage; the A stage already stores `ds_store_b128`. Tried
  packing the dequant writes anyway: no instruction change. Nothing to gain.

**4a. Split K in the mixer down GEMM (`halo_gemm_q8r_f16_m`, 320 x 10240).** TRIED, NOT KEPT:
measured +21.6 ms end to end (within noise; pinned, 3 rounds). The GEMM itself went from 1146 to
1267 µs per call, plus 28 µs in the summing pass. The model was wrong: LDS limits `_m` to 2
workgroups per WGP, so 40 slots, and the plain launch already filled 32. Its 1.15 ms against a
0.41 ms math floor is not a parallelism problem. Next step: a thread trace of `_m` to separate
per-stage overhead (dequant and two barriers per 64-K stage, 160 stages) from cache misses. The
split quadrupled the weight working set across concurrent K ranges. Kernel A (3.3) removes this
GEMM's separate pass anyway.
- Hypothesis: it runs 32 workgroups on 40 CUs. The snapshot has it at 1.29 ms per call, against
  ~0.41 ms of math and ~0.2 ms of traffic. Splitting K four ways, one split per stream (4 x
  2560), gives 128 workgroups. The four partials (2.6 MB each) are summed in the existing
  scale-SiLU pass.
- Prediction: 0.5-0.65 ms per call, so **-60 to -75 ms** over 96 calls. This is a step toward
  kernel A in 3.3.

**4c. Mixer restructuring (R1-R3, 2026-10-04).** Done so far, each with identical long-set
quality:

| Step | Predicted | Measured | Kept |
|---|---|---|---|
| R1: combine_norm issues all loads first (e2b034b) | -55..-65 | -7 | yes: real; the kernel was already near bandwidth (~190 GB/s) |
| R2a: combine_norm skips fp32 xn; the mix epilogue rebuilds it (b507480) | -35..-45 | **-34.8** | yes |
| R2b: block-wide mix epilogue with contiguous reads, then with all loads batched | -85..-105 | +0.3, +12.6 | no |
| Mix GEMM KS 64 -> 32: LDS 55 -> 30 KB, 4 -> 7 waves (32127d0) | -35..-40 | **-38.1** | yes |
| R3: `_m` two stages in flight (clamped refill, LDS-only barrier) | -35..-45 | -1.4, +1.5 | no |

What settled R2b and R3 was ablation inside the real kernel, not traces:
- **Mix:** without its epilogue loads, mix runs at 0.60 ms per call instead of 1.92. The same
  reads run at ~230 GB/s in a microbenchmark (`halo_gemm_bench memory`, column blocks), so the
  limit was bytes in flight at 2 blocks per WGP, which KS = 32 fixed.
- **`_m`:** 1044 us per call; 667 without weight loads, 466 without the `xh` loads. The `xh`
  read (42 MB, 128-byte pieces per token per stage, 32 blocks) costs ~0.58 ms and does not
  respond to prefetch depth. Next: kernel A of section 3.3 (combine + norm + down), so xh is
  neither written (42 MB) nor read back. Expected about -50 to -70 ms; trace and ablate a
  prototype before predicting more closely.

Also found:
- `__syncthreads()` on gfx11 waits for all outstanding global loads (its workgroup fence), so a
  prefetch survives only until the next barrier. An LDS-only fence
  (`__builtin_amdgcn_fence(..., "workgroup", "local")`) avoids that.
- A GEMM tile whose stage does not split evenly over 256 threads silently skipped rows; it is
  now a `static_assert`.

**4d. Library-derived GEMM targets** (user decision 2026-10-04: use these as targets for our
own kernels; a ROCm upgrade is acceptable if needed). hipBLASLt 7.2.3, pinned clock, 2048
tokens (`tools/blaslt_bench.cpp`):

| Shape | Ours | Target (hipBLASLt 7.2.3) | Halogen (ROCm 7.14, tuned) |
|---|---:|---:|---:|
| 320 x 10240 (`_m`) | 1.048 ms | 0.638 | 0.410 |
| 10240 x 320 (mix, GEMM part) | ~0.6 of 1.460 | 0.672 | 0.471 |
| 512 x 2560 (router, F32) | 1.276 | 0.193-0.225 (16-bit) | ~0.40 (hi/lo split, inferred) |
| 10240 x 2560 (dense) | 3.457 | 3.530 | 2.79 |

How the library gets there on the skinny K = 10240 shape (kernel names from a rocprofv3
trace):
- tiles of 96 x 96 to 128 x 96 with 128 threads, and two stages of global prefetch (`PGR2`);
- roughly 25-way split K, so thousands of short-K workgroups rather than our 32 long ones.

First attempt at that shape, not kept (pinned, in-engine): 64 x 64 tiles with split K over
`gridDim.z` (`halo_gemm_q8r_f16_k`), partials summed in the SiLU pass. Per call: 2.58 ms at 4
splits, 3.51 at 8, 3.92 at 16, 4.28 at 20, against 1.05 for the single 320-row tile. Small row
tiles re-read xh once per row tile (5x, ~210 MB), and the cost grew with the split. Copying the
library's split without its tile economics does not work; next is a microbenchmark study of
`_m` tile shapes with a trace, not more in-engine guesses.

**4i. QSA attention over 4-token unions (173b279).** Attention for an 8K-token prefill: 1578 -> 654
ms (pinned), 13.46 -> 6.11 ms per 512-token call, as predicted. Quality is unchanged at prompt
lengths up to 24K. The next long-prompt target is the indexer's scoring: `halo_qsa_scores_rows`
takes 1.39 s per 8K prefill, as much as attention did before. It scores every complete block for
every token. Then the serial n-gram read per chunk (4h).

**4j. Long-prompt attention block (2026-10-04, 8K-token prefill, pinned).**
- V prefetch in `qsel` (6c61579): 653 -> 502 ms.
- Indexer scores as a bf16x3 WMMA GEMM: 1605 -> ~24 ms (-1.52 s); 8K prefill 697 -> 869 tok/s on
  random tokens.

What remains in attention: `qsel` at ~500 ms. The trace shows 43% barrier and 25% K gathers.

**K staged through LDS, measured, not kept:** 4835 -> 4810 us per call (-0.5%; predicted -20 to
-35%). The redundant per-lane K fetches it removed were cache hits. The cost is the one exposed
K round trip per chunk, and hiding it needs K prefetched in registers like V: 32 more VGPRs of
fp32 on top of 253.

**The lever is an fp16 copy of the K/V cache.** The prefill attention kernels already round K/V
to fp16, so prefill results would be bit-identical. It would halve the gather bytes and the
prefetch registers, so both K and V fit in flight. The cost is ~24.6 KB more per cached token:
~6.4 GB at a 262K context, ~200 MB at 8K. That needs a decision on memory.

Then the serial n-gram read per chunk.

**4h. Chunk size and long prompts (2026-10-04, pinned, pseudo-random tokens, n-gram not
resident).** For an 8192-token prompt:

| Chunk size | 2048 | 4096 | 8192 |
|---|---:|---:|---:|
| tok/s | 640 | 619 | 596 |

The expert kernels amortize only ~15% (gate/up 1207 -> 1021 ms, down 935 -> 776 ms): they are
dequant/VALU-bound per token, so 4g's "75 GB weight-stream floor" overstates them. With
2048-token chunks, the device time per chunk climbs with context:

| Chunk | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| device time | 2.06 s | 2.88 s | 3.40 s | 4.12 s |

The growth is almost all `halo_attention_wmma_rows`: the attention path once the context exceeds
one chunk. It takes 13.5 ms per call, 1.94 s of 13.0 s. That is the long-prompt bottleneck, and
most of the distance to halogen's 1,584 tok/s on an 8K-token prompt.

Also: the n-gram read for each chunk (~115 ms of random tokens, not resident) runs serially
before that chunk's device work. Overlapping it with the previous chunk saves ~0.35-0.45 s at 8K.

**4g. Bytes audit (2026-10-04).** Per 2048-token prefill, analytic from tensor shapes. rocprofv3
7.2.3's FETCH_SIZE/WRITE_SIZE read 0 for every prefill dispatch (with or without graphs) while
counting the load-time repack, so they could not be used. Writes are weighted like reads (4f),
at ~200 GB/s.

| Area | Traffic | Cost | Avoidable | Potential |
|---|---:|---:|---|---:|
| Routed expert weights (all 512 experts stream every chunk) | ~75 GB | ~375 ms | none at T = 2048; amortized only by larger chunks | - |
| Mixers (96 half-layers): res r/w, out, xh, mix re-read, mixed | ~37 GB | ~185 ms | xh round trip, 84 MB per call | ~40 ms (kernel A broke even) |
| DeltaNet (36): qkv 84, conv r/w 168, recurrence 84 + 50, z 50 + 50, gdn 50, xh 25 + 25, out 21 MB per layer | ~24 GB | ~120 ms | qkv, conv, z, gdn stored fp16: ~270 MB per layer | **~45-50 ms** |
| Expert outputs `yp` fp16 write + combine read, 210 MB per layer | ~10 GB | ~50 ms | fused accumulation | ~40 ms, unproven |
| Attention (12): qg 100, split, norms, RoPE, ctx, f16 | ~9 GB | ~45 ms | epilogue fusion | ~30 ms |

Order: DeltaNet fp16 intermediates first (mechanical; the quality gate decides), then the
attention epilogues.

**First item measured, not kept:** `conv` stored in fp16 (conv kernel writes fp16, the split
recurrence widens on load).
- Quality: unchanged.
- Speed: -1.0 ms end to end (noise); recurrence 300.6 -> 299.7 ms, conv 46.5 -> 45.9 ms.
- Why: neither kernel is bandwidth-bound. The recurrence waits on LDS round trips and barriers
  (4.2), and the conv kernel is latency-bound at ~130 GB/s. The write saved has no
  bandwidth-bound successor.

**Rule for the rest of this list:** a byte saving pays only where the writer or the next kernel is
bandwidth-bound. Check that in the trace (vmcnt share) and in achieved GB/s first. Candidates that
pass: the GEMM outputs feeding other GEMMs (qkv -> z, qg -> split), the experts' `yp` (combine
streams at ~88% vmcnt), and the mixer chain. Halogen's 32K-token chunks amortize the ~75 GB of expert weights over 16x
the tokens; for long prompts, the chunk size is the lever (memory, not code).

**4f. GEMM-core latency study: the in-engine gap is write-back, not the GEMM** (2026-10-04,
`halo_gemm_bench study`, pinned). `_m` on its real shape (320 x 10240, 2048 tokens):

| Condition | `_m` per call |
|---|---:|
| warm (repeated launches) | 0.729 ms |
| cold (96 MB read in between) | 0.768 ms |
| **right after a 96 MB write** | **1.359 ms** |
| in-engine (after combine_norm's ~126 MB of writes; rocprofv3 and our timestamps agree) | 1.02-1.05 ms |

hipBLASLt's 0.638 ms was measured warm, so like for like our core is about 12% behind, not
1.6x. Dirty lines drain to DRAM during the next kernel, which pays for its predecessor's
writes. So:
- per-kernel attribution in the engine charges write-back to the following kernel;
- removing writes anywhere speeds up whatever runs next (R2a's 84 MB also took 10 ms off
  `_m`);
- the lever is total bytes moved, writes included (section 1.2), more than per-kernel latency.

The same probably holds for the other "slow" kernels that follow heavy writers (`mix` after
the down GEMM and SiLU, the experts after the router and bucketing). Check in the harness before
optimizing any of them alone.

**4e. Router on bf16 WMMA, hi/lo split (781449a).** -7.4 ms (router 1.30 -> 1.02 ms per call,
pinned) against a predicted -35..-40; quality unchanged within noise. The trace shows 59% of
cycles in one `vmcnt(0)` per K stage at ~330 GB/s from cache, so it is not bandwidth-bound.
An LDS-only barrier with a clamped refill changed nothing (1005 vs 1012 us), as for `_m`: with
one stage in flight, the data is needed at the next store anyway. The levers left are fewer
bytes per product and fewer re-reads:
- split W once at load into stored bf16 hi/lo (the same 4 bytes per element, no per-block
  split VALU);
- tiles of 128 tokens, halving the W re-reads;
- 2 WMMAs per product (x in bf16 hi only), if the quality gate allows it.

**4b. Router F32 GEMM (`f32_tiled [32x8]`, 512 x 2560 x 2048 per layer).** DONE, partly. A
trace first: 52% of cycles in `s_waitcnt vmcnt`, 33% at barriers, ~4% in FMAs. With a 32-wide K
step and a one-stage register prefetch it measured **-36.5 ms** against a predicted -65 to -85:
router 87.9 -> 61.4 ms, alpha/beta/sgate 43.8 -> 32.4 ms. The long-set scores are identical on
all 12 cases (same summation order).

A second trace explains the shortfall: 63% of cycles still wait on `vmcnt(0)` once per stage.
One stage of math (~600 cycles) covers too little of the load latency. Each 64 x 64 fp32 tile
re-reads its operands, about 335 MB per call, or ~260 GB/s at 1.28 ms. Next: a deeper prefetch,
or 128-row tiles to halve the traffic. Trace that version before predicting.
- Hypothesis: 1.79 ms per call against ~0.2 ms of fp32 issue. Its 64 x 64 tiles with
  `ds_load_b128` and ~2 waves per SIMD leave it latency-bound. A register-blocked fp32 tile, or
  WMMA on a hi/lo fp16 split for near-fp32 sums, keeps the routing exact enough. The route must
  match the current one on the property tests and the quality sets.
- Prediction: **-60 to -75 ms**.

**4. Then the structural gaps, in the order step 0 gives** (currently mixers, experts, DeltaNet,
dense, attention, F32 router; section 4).

**5. Decode: the spilling DeltaNet step kernels (not item 1's cause).**
- The device IR shows no surviving alloca in these kernels. Their scratch is real register
  spilling at 256 VGPRs:
  - `halo_gdn_step_d128`: 109 VGPRs spilled, 440 B;
  - `_norm`: 152 spilled, 612 B;
  - `halo_gdn_prefill_d128_norm`: 25 spilled;
  - `halo_moe_gate_up_grouped_q6_k`: 442 spilled.
- The lever is less live state per thread: split the 128 x 128 state slice across more lanes or
  waves, or keep part of it in LDS.
- Prediction: 1-3 ms of the 44.7 ms per token. First check whether any Q6_K expert tensors exist
  in the weights; if not, the q6_k gate/up spills never run.

**6. Fallback flags, only if items 1-2 fail at the source level.**
- `-mllvm -amdgpu-promote-alloca-to-vector-max-regs=N` and `-limit=`. Clang 22 already defaults
  max-regs to 32, so this is mostly for the clang 21 comparison build.

**7. Tooling: keep the occupancy model honest.**
- A ctest that compiles with `-Rpass-analysis=kernel-resource-usage` and checks
  `kernel_report.py` against the compiler's waves/SIMD for every kernel. The tool's 16-register
  granule was wrong until a65df43.
- The remaining LDS-limited mismatches come from the compiler using the declared
  `__launch_bounds__`; settle them as part of that test.
