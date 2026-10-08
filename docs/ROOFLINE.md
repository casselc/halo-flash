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

**fp16 K/V copy (fb4a7b5):** attention 608 -> 449 ms per 8K prefill (-188 ms in total, pinned),
bit-identical results, ~24.6 KB more memory per cached token. It also fixed out-of-bounds K/V
reads in the union kernels (tail-block slots past the causal end).

**N-gram prefetch (21a0490):** the next chunk's rows are read during this chunk's device work.
Host prep 115 -> 1.3 ms on later chunks; 8K prefill 8.40 -> 8.08 s.

**State (2026-10-04, default clock, n-gram from /data, not resident):** pp2048 1081 tok/s; an
8K-token prefill 1014 tok/s (from ~632 before the long-prompt work).

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

**4m. Chunked gated DeltaNet, first prototype (branch `proto/gdn-chunked`, 47199e4).**
- The chunked form (cumulative log decays G, strictly lower A = b (k.k) e^(G_t-G_s),
  (I + A) Delta = bV - b e^G K S0, outputs and end state from S0 and Delta) was verified exact
  against stepping (5.6e-16, double).
- The fp32 per-lane kernels (16-token chunks) pass the property test, but run at 16.9 ms (scan)
  + 2.6 ms (prep) per layer, against the split kernel's 8.1 ms.
- Traces: the scan is LDS-bound (54%: every lane re-reads the chunk's K and Q), its staging is
  unprefetched, and SIMDs are idle 2.5x as long as busy at 96 blocks. The prep's 6144 small
  blocks are load-latency bound.
- WMMA redesign (e95425e, same branch): every product on bf16 WMMA with the hi/lo split; the
  state in accumulator tiles, which double as B operands after one lane ^ 16 exchange; 4 waves
  per block (2 column groups x 2 key halves). Correct within relative L2 1e-4, both controls
  caught.
  - 9.39 ms per layer, against the split kernel's 8.18 (pinned, 2K).
  - Trace: WMMA 0.6% of cycles, barrier 43%, chunk staging loads 29%, at 256 VGPRs and 4 waves
    per SIMD.
  - Batching the staging loads made it worse (10.08 ms, 32 spills): the register budget leaves
    no room to prefetch.
- What it would take: larger chunks (C = 64) so each barrier and staging round trip covers 4x the
  math; the state in LDS instead of registers (a head's 128 x 128 fp32 is 64 KB, so per key half
  or in bf16 hi/lo); and a prep pass that computes T and the QK masks ahead of the scan. Halogen's
  scan runs at 254 VGPRs and 61.9 KB LDS with 128 threads: the same corner, with bigger chunks.

**4l. Routed experts: ablations (2026-10-04, `halo_gemm_bench moe`, one layer, pinned).** Base
13.83 ms per layer (x48 = 664 ms). Each probe removes one cost with wrong numerics:

| Probe | Layer | Saving | x48 |
|---|---:|---:|---:|
| gate/up (Q4_K) without dequant arithmetic | 12.14 ms | -1.69 | -81 ms |
| gate/up without weight loads | 12.48 ms | -1.35 | -65 ms |
| down (Q5_1) without dequant arithmetic | 13.51 ms | -0.32 | -15 ms |
| down without weight loads | 11.91 ms | -1.92 | -92 ms |

Tried and not kept: prefetching down's next weight stage with LDS-only barriers and unconditional
loads made the layer 14.18 ms (+0.35). VGPRs went 106 -> 130 and occupancy 12 -> 10 waves, so
the load saving above is probably DRAM bandwidth freed for the other kernels, not latency.
Expert weights stream at ~114 GB/s overall (1.57 GB per layer).

Gate/up does ~134 GFLOP per layer, plus ~20% sub-tile padding, at ~23 TFLOP/s: it is near
WMMA + dequant-VALU bound. The open lever is cheaper dequant (bounded by -81 ms), e.g. a lookup
table (halogen). Activation re-staging (x restaged per 128-row block, ~1.4 GB per layer from the
MALL) is the other candidate.

**4k. Mixer down GEMM `_m`, closed for now (2026-10-04, `halo_gemm_bench study`, pinned).**

| Variant (320 x 10240, T 2048) | warm | cold | after 96 MB of writes |
|---|---:|---:|---:|
| `_m` (320 x 64 tiles, 32 blocks) | 0.74 ms | 0.77 | 1.34 |
| 32-token tiles (64 blocks, all CUs) | 0.85-0.90 | 0.91-0.93 | 1.53-1.55 |
| two stages in flight (LDS-only barriers, clamped refills) | 0.77 | 0.79 | 1.35 |
| no dequant at all (probe, wrong numerics) | 0.67 | 0.73 | 1.42 |
| int8 WMMA path (`wmma_t`, without quantize) | 1.36 | - | - |

Split K (4-20) was worse still (4f). Like for like, `_m` is within ~13% of hipBLASLt (0.64 ms
warm). Its in-engine 1.02 ms is mostly combine_norm's ~126 MB of writes draining during it
(4f). Levers that remain are outside the kernel: fewer bytes written before it (kernel A
removed xh but broke even, branch `proto/kernel-a`), or ordering independent compute-bound work
in between.

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

## 8. 2026-10-05 (branch `proto/gdn-chunked`): chunked DeltaNet in, and where gufo is ahead

All numbers: `tools/ab_bench.py`, 2048 tokens, clock pinned, 3 rounds x 3 reps, in-engine
per-kernel attribution. Device time went 2046 -> 1866 ms (1097.5 tok/s pinned).

**DeltaNet, chunked (16-token chunks), kept.**

| Step | Predicted | Measured | Notes |
|---|---|---|---|
| Skeleton floor (loads, staging, barriers, no math) | 0.4-1.0 ms/layer | 1.25 | 64-token chunks with fp32 K and Q do not fit 64 KB of LDS |
| prep + bf16 hi/lo WMMA scan (4033919) | -100..-140 | -120.8 | **failed the long-set gate**: +0.0020 +- 0.0009 NLL |
| scan3: K S0 three-way, split once at staging (90c134e) | -140..-170 | -155.5 | +0.0013 +- 0.0009 |
| scan3u: the update's K three-way too (e92b8eb) | ~-150 | **-140.5** | -0.0001 +- 0.0008: no shift |
| prep2: chunks walked a block, prefetched (1c74cd9) | -40 | **-35.1** | prep 57.1 -> 15.3 ms, bit-identical |

Why the bf16 scan failed while random-data error was 6e-6 (flat in length, also with
correlated keys): once the state predicts the values, V - S^T k cancels, and a two-way
split's ~2^-17 error in K S0 is large against the residual. The fp32 chunked kernels scored
+0.0007 +- 0.0008 on the same set, which isolated precision from the chunked math. A
TheRock nightly clang (24.0.0git, `HALO_HSACO_COMPILER`) removed the scan's spills
(256 -> 227 VGPRs) but made the whole engine 71 ms slower (the `q8r_f16_l` GEMMs +86 ms,
grouped gate/up -14): ROCm 7.2.3 stays.

**gufo, profiled the same way** (`tools/prof/prof.py`, one timed 2048-token pass, 1245 ms),
gaps per prefill by role (ours - gufo):

| Role | ours | gufo | gap |
|---|---|---|---|
| MoE routed (gate/up 6.0 ms/call vs ~5.5; down 5.18 vs ~4.1; our separate combine 0.82) | ~593 | ~433 | ~160 |
| HC mixers (combine, mix GEMM + epilogue, `_m` down: fp16 1.04 ms vs W8A8 int8 0.50) | ~396 | ~245 | ~150 |
| Attention, dense 2K (7.15 ms/call vs 4.07) | 86 | 49 | 37 |
| DeltaNet (now) | 108 | ~114 | -6 |

**Mix epilogue probes, none kept** (`halo_gemm_q8r_f16_mix`, 1.45 ms/call; trace: 78% of
cycles in `vmcnt` waits, one stream's loads at a time):

| Probe | Measured mix ms (from 140) |
|---|---|
| Read combine_norm's fp16 `xh` instead of rebuilding from fp32 `res` (half the bytes) | 140.4 |
| Rebuild as a template choice, all of a lane's loads issued first | 154.9 |
| Block-wide epilogue, float4 loads, 256 contiguous bytes per (token, stream) | 138.4 |
| Ablation: epilogue loads replaced by constants | **72.7** |
| Ablation: epilogue stores removed | 114.5 |

The loads cost ~0.7 ms/call, but not through bytes, batching or access width. The likely
cause: `res` / `xh` were written by `combine_norm` two kernels earlier (126 MB still draining:
writes are paid by the next kernels). The lever is upstream, in what combine_norm writes.
A runtime branch (`xh ?` / `rs ?`) inside the template cost the plain mix 39 VGPRs and 388 B
of scratch: keep such choices compile-time.

**Attention at 2K:** routing the dense case through QSA (`--mq-attention 0`, fp16 K/V,
prefetch) measured -9.6 ms (noise; predicted -30..-45): `qsel_h` 75.4 vs `mq` 87.1. Trace of
`qsel_h`: 43% barriers, 21% `vmcnt`, 18% `lgkmcnt`, WMMA 1%: per-chunk block-wide phases
(scores on 6 waves, softmax on 192 threads, P.V on 8) wait on each other. gufo's dense kernel:
32 rows a block (16 queries x 2 heads), 16-key tiles, 8 waves = S tiles x head-dim halves,
K/V prefetched into registers, V transposed in LDS. Next: a kernel whose waves each own their
rows and softmax, so only K/V staging needs a block barrier.

**Later the same night (main line of the branch, each A/B 3 rounds x 3 reps, pinned):**

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| n-gram rows read while layer 0 runs (PLE is layer 1; 41-110 ms host read) (0647d6b) | -35..-40 wall | **-42 ms wall** (1.733 -> 1.691 s) | yes |
| token embedding in one launch instead of T | (the 45 ms stamp was the read above) | neutral | yes (0647d6b) |
| MoE down: stage pipeline + multiply bit spread for Q5_1 (9343638) | -40..-55 | down 224 -> 207 ms | yes |
| `Unpack4` via byte permutes | -25..-40 | not built: its element order already avoids the interleave | — |
| W8A8 for the mixer down (`_m`): tall `GemmQ8rWmma<5,2,4,2>` | -35..-60 | microbench 976 us vs fp16 725 us | no |

The down pipeline missed its prediction for a reason worth keeping: the trace showed 28% of
wave cycles in `vmcnt(0)`, but at 12 waves/SIMD other waves covered most of it; the kernel is
VALU-bound (40%) on dequant. A wave-level stall share says little at high occupancy.
The int8 GEMM template is staging-limited (~20 of 56 TOPS); W8A8 needs a better int8 kernel
first (gufo's runs the same shape in 0.50 ms in-engine).
| Flash-style dense attention `halo_attention_fa` (0cdb694) | -25..-45 | **-31.8** (attention 86.6 -> 47.0 ms; gufo 49) | yes |

0647d6b broke every multi-chunk prompt (the deferred n-gram read collided with the next chunk's
async prefetch, which after_prep starts); fixed in 0cdb694, found by the long quality set. Host
or control-flow changes to prefill need a multi-chunk run before they are committed.
First version of the attention kernel: 95 ms, 40% of cycles in `ds_store_b16` (a wave's
transposed V stores 640 bytes apart); one key per lane fixed it.
| Router weights split to bf16 once (`halo_gemm_f32_bf16x3w`, bit-identical) | -12..-20 | -1.9 (49.2 -> 47.3) | no (252 MB for 2 ms) |

**Night total, wall clock** (`halo_run --prefill-bench`, default clock, `prose_long.txt`, same
argmax): 2048 tokens 1.896 -> 1.651 s (1080 -> 1240 tok/s, +15%); 6940 tokens 6.82 -> 5.98 s
(1017 -> 1160 tok/s, +14%). Over the 8K prefill the kernel sums fell 972 ms (DeltaNet -698,
host n-gram read -262, MoE down -58, first-chunk attention -42). gufo pp2048: 1599 tok/s.
The router probe repeated the high-occupancy mistake above: at 10 waves/SIMD its trace's
26% `vmcnt` and 24% split VALU were covered by other waves.
| Routed combine + shared-expert add in one pass (f4ab695, bit-identical) | -10..-15 | kernels -10, device -5.2 (noise) | yes |

**Queue after this night, ranked by measured gap to gufo (2K, per prefill):**
1. HC mixers (~150 ms): the mix epilogue's ~0.7 ms/call of loads did not respond to bytes,
   batching or access width; test the write-back hypothesis directly (e.g. a probe kernel that
   only reads `res` right after combine_norm vs after an unrelated kernel), then decide whether
   combine_norm should write less (int8 / fp16 streams, as gufo does).
2. Mixer down `_m` (~50 ms): needs an int8 GEMM faster than the fp16 one (ours is
   staging-limited at ~20 of 56 TOPS); gufo's W8A8 runs the shape in 0.50 ms.
3. MoE gate/up (~24 ms) and down (~10 ms left): VALU-bound on dequant at high occupancy.
4. Dense GEMMs: gufo fuses the DeltaNet qkv and z projections (one 16384-row GEMM, ~21 ms).
Rules learned tonight: check occupancy before trusting a stall share; run a multi-chunk prompt
for any host or control-flow change; keep kernel variants compile-time.

**2026-10-05 day, on main:**

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| Mix GEMM with row tiles fastest (`halo_gemm_q8r_f16_mixt`, f4d8504, bit-identical) | -30..-40 | mix 140.9 -> 116.3 ms, device -22.5 | yes |
| Same order for the `_l` GEMMs (study mode) | — | 10240x2560 equal, 2560x6144 +0.3 ms worse, 6144x2560 -0.1 ms | no |

The mix probe sequence that found it: `halo_gemm_bench mixstudy` showed the kernel as slow warm
(1.41 ms) as in-engine (write-back refuted); ablations put ~1 ms in the epilogue alone (~117 GB/s);
a GPT-6.1 review of the code and numbers ranked DRAM locality of the dispatch order first. With
token tiles fastest, resident blocks read 256-byte pieces of many tokens' 40 KB rows. The `_l`
GEMMs depend more on sharing weight rows across token tiles than on output-write locality.
| DeltaNet conv + L2 norm, 8 tokens a block (c25cdb5, bit-identical) | -15..-20 | conv 46.8 -> 27.2 ms, device -9.4 | yes |
| qkv + z as one 16384-row GEMM (study) | -10..-20 | 6.66 ms vs 4.02 + 2.48 separately: no gain | no |
| Dense GEMMs in bands of 8 row tiles (0841284, bit-identical) | -15..-20 | -5.8 kernel (study promised -0.17..-0.36 ms a call) | yes |

Wall clock after these (default clock, `prose_long.txt`): 2048 tokens 1.582 s (1294 tok/s),
6940 tokens 5.78 s (1200 tok/s). gufo's dense GEMM is ~15% faster than ours at the same shapes
(16384 x 2560: ~5.15 vs 5.9-6.6 ms); fusing shapes or reordering blocks does not close that.
| `combine_norm` loads batched + one barrier for the inject sums (fe82591, bit-identical; perf model + `combinestudy` ablation) | -40 | kernel 158.5 -> 108.5 ms, device -40.5 | yes |
| Mix epilogue (`mixt`) loads batched (study) | -40 | 1.10 -> 1.22 ms a call (worse: VGPRs); the epilogue is bound by its 84 MB residual read | no |
| combine + norm fused into the down GEMM (`halo_hc_combine_down`, bit-identical, study) | -45 | 2.13 ms vs 1.68 for combine_norm + `_mb` | no |
| combine_norm without y16 + down GEMM rebuilding operands from `res` (`_mr`, bit-identical, study) | -12 | 0.92 + 1.04 = 2.13 ms vs 1.68 | no |

Why the exact two-kernel mixer loses (patch kept at `/data/halo-work/patches/combine_down_and_mr.diff`):
the down GEMM has 32 blocks (64-token tiles). Fused, each block alternates a latency-bound combine
phase with its GEMM phase, all blocks in the same phase at once, so neither overlaps the other. Unfused
but reading the fp32 residual, each stage takes 256-byte pieces of 64 different 40 KB rows, the DRAM
pattern the K-blocked fp16 copy exists to avoid. What remains open: a K-blocked residual layout
([cols / 64][T][64] fp32, every residual kernel changed; makes the GEMM's and the mix epilogue's
reads contiguous), or the numerics-changing options (mix reading the fp16 copy; an fp16/bf16 residual),
which need a quality eval first.
| `combine_norm` residual stores after BlockSum (bit-identical; GLM's review suggestion) | -3..-8 | study 1.29 -> 1.26 ms; engine -2.7 ms within noise (spread 7.5); trace: wave lifetime unchanged (the load wait moved from 1->2 to 2->3) | no |
| Mix epilogue: row scales and gammas staged in LDS, the residual loads of each 16-token subtile issued as one batch (bit-identical; found with the span tracer: the epilogue was 89 of a wave's 118 us) | -35..-50 (epilogue to ~43 us) | engine kernel 121.5 -> 108.0 ms (-13.4, spread 3.7); end to end -16 ms (paired over 6 interleaved rounds: 5 faster, 1 tie); trace: epilogue 89 -> 74 us | yes |

The study harness (`mixstudy`, after 96 MB of writes) said the opposite (1.10 -> 1.21 ms): its condition
does not reproduce the engine's, as noted before; decide by the engine (A/B and in-engine traces).
| Mix epilogue lanes over 64 columns of the block's tile (128-byte coalesced residual loads; bit-identical) | lower request count | trace: epilogue 74 -> 70 us, kernel extent 1175 -> 1196 us (VGPRs 216 -> 225: occupancy 7 -> 6) | no |

What remains in the mix epilogue fits the rounds model as DRAM-bound: ~500 waves in it at once, a 16-load
round each ~1.1 MB / 185 GB/s ~ 6 us, 8 rounds + the sigmoids ~ 65-70 us (traced 70-74). Fewer bytes (the
fp16 copy instead of the fp32 residual) is the remaining lever; it changes rounding (quality eval first).
| MoE down: output tile staged in LDS, written as 16-byte pieces of contiguous token rows (bit-identical: logits compared bitwise; found with the span tracer: the epilogue was 7-19 us of a 35 us wave) | -18..-25 | MoE down 179.8 -> 133.1 ms (-46.8), q8r -4.9; end to end -60.2 ms (1541 -> 1481, significant; 1383 tok/s); trace: epilogue 8.6 -> 2.0 us, wave 35.4 -> 27.5 us | yes |
| MoE gate/up: the same LDS-staged output epilogue (bit-identical) | similar to down | gate/up 248.3 -> 245.1 ms (-3.3, spread 2.5); end to end -6.5 ms within noise | no |
| Register-weight GEMM in bands of 5 for the 2560-row shapes (`lr5`, bit-identical) | -20..-25 (lr8 runs 75% of its floor, `_l` 60%) | `_l` 118.6 -> lr5 172.0 ms; end to end +57 ms | no |
| Down GEMM `_mb` with two stages of operands in flight (kDepth 2, bit-identical) | -11 (in-engine vmcnt waits were 64%) | `_mb` 59.2 -> 61.2 ms, end to end -1.3 within noise: the waits are bandwidth after combine_norm's writes, not latency | no |
| `moe_combine_shared`: the k expert values loaded before the sum (bit-identical) | ~-5 (rounds model: near bandwidth-bound at 1280 waves) | not among the kernels that moved; end to end within noise (spreads 20-28 ms) | no |

DeltaNet scan (`halo_gdn_c16_scan3u`, 76 ms, issue floor ~60): 96 blocks x 4 waves, all resident in one
generation (~4.8 waves/SIMD), each walking 128 chunk steps in series at ~16.5 us per step; per SIMD a step
issues ~6.7 us of WMMA and ~6.3 us of VALU, so it runs at ~80% of issue. Its 6 spilled VGPRs are reloaded
4 times a step from scratch (~0.3 us of 16.5, ~1.3 ms per prefill). No cheap lever: the remaining 20% is
issue packing across the steps' phases.
| MoE gate/up dequant ablated (study harness, fixed routing; wrong results, timing only) | upper bound of an exact VALU-cutting repack | 6.75 -> 7.45-7.79 ms (slower, at 6 vs 5 waves/SIMD): the dequant VALU is hidden under memory waits; repacking to cut VALU would not help | n/a |
| Q4_K gate/up experts K-blocked (exact one-time repack; prototype `--moe-blocked 1`, decode not adapted) | -30..-47 (sweep: 191 -> 240 GB/s for one weight stream) | gate/up 248.1 -> 236.5 ms (-11.6, spreads 3.2/0.3); end to end -4.5 within noise; logits bit-identical | prototype only |

The sweep probe over-predicted the repack ~3x: it models one weight stream; the kernel reads gate and up,
16-byte headers and activations together. Integrating it means teaching the decode expert kernels the
blocked layout (decode is weight-bandwidth-bound: risk) for ~12 ms of prefill.
| Mix epilogue reads combine_norm's fp16 `xh` (retried after 881b573 made the epilogue batch its loads: the rounds model then called it DRAM-bound, so half the bytes should have helped) | -15..-30 | mix 107.6 -> 117.8 ms (slower), end to end +6.1; logits drift max 0.32 of 16.8 (argmax same) | no; quality eval not run (no speed gain) |

Both byte halving (fp16) and request coalescing (128-byte) now failed on the mix epilogue after its loads were
batched: neither bytes nor request count is its limiter, so the rounds model's "DRAM-bound" reading is wrong
for it. Remaining suspects: the 31 MB of `mixed` stores and the write-back of combine_norm's 126 MB (see the
2026-10-04 probes above); an ATT trace of the current epilogue is the next step if it is revisited.

Dense GEMM redesign (speculation from GPT-6.1-sol and GLM 5.3, 2026-10-05, not built): both propose the same
discriminating probe first: the `_l` kernel with pre-dequantized fp16 weights (exact: the same fp16 values,
2x the bytes, ~1.4 GB more for these shapes) or with dequantization bypassed, and read stage time from the span
trace (4.0 -> ~3.3 us would mean serial dequant is the ceiling). Then: GLM ranks warp-specialized producer waves
(KS 32 two-deep ring) and a persistent 40-block grid; sol ranks balanced stream-K / persistent scheduling for the
tail and doubts warp specialization on this hardware (VALU and WMMA serialize per SIMD). Both: skip split-K
(fp32 partial traffic), int8 W8A8 changes numerics.
| Few-row F32 projections with rows on threads (alpha + beta fused in one pass; bit-identical) | -19 | 28.0 -> 31.7 ms (rows128 0.55 ms per layer vs tiled 0.47; rows16 0.25 vs 0.23): one LDS read per FMA vs the tiled 4x4 outer product's 0.5 | no |

**Against gufo (its 2-pass rocprofv3 profile, `/data/halo-work/gufo_prof`, vs `kt_head` at f1380a8), 2026-10-05.**
Per 2048-token pass, gufo's kernels sum to 1354 ms. By role we are at or ahead: routed MoE 416 vs 455,
attention 40 vs 49, HC combine ~155 vs 145, DeltaNet ~144 vs ~130. The big dense GEMMs are matched or better
(qkv+z 181.0 vs 183.7; attention q/gate/k/v ~59 vs ~60; DeltaNet out 2560 x 6144 2.17 vs 2.10 ms a call). The
~100 ms gap in the dense class is:

| Role | gufo | ours | gap | why |
|---|---|---|---|---|
| Mix GEMM + epilogue | 60.0 + 23.7 | 108.1 | +24 | gufo: separate GEMM (0.64 ms) and vectorized epilogue kernel |
| Small F32 projections (alpha, beta, shared-expert gate) | ~3.6 | 27.9 | +24 | gufo: fp16 WMMA; ours exact fp32 VALU |
| Router (512 rows) | 17.6 | 39.5 | +22 | gufo: fp16; ours bf16x3 (fp32-like) |
| Mixer down (`_mb`) | 46.3 | 59.9 | +14 | gufo: W8A8 (int8 activations) |
| Shared-expert gate/up | 22.4 | 34.3 | +12 | gufo: W8A8 |
| Attention out | 23.9 | ~30 | +6 | gufo: W8A8 (K-quant weights, int8 activations) |

About 75 of the ~100 ms are precision choices (fp16 or int8 activations where we keep fp32 or fp16 exact),
i.e. quality-gate decisions, not kernel engineering; the exact candidate is the mix GEMM's split (~24).

**gufo's kernels under ATT and counters (2026-10-05; `/data/halo-work/gufo_cmp`, per call at T 2048):**
- **Mix:** gufo's fused mix GEMM (`DenseF16GEMMKernel<256,128,1,4,2,8,true>`) takes 0.54 ms and reads 61 MB.
  Ours (`mixt`) takes 1.06 ms and reads 156 MB. gufo mixes from the **fp16** normalized copy, stored
  row-major (contiguous), where ours rebuilds from the fp32 residual. That is a precision choice, and our
  fp16 attempt read the K-blocked copy with 2-byte gathers, not contiguous rows.
- **MoE combine:** gufo fuses it into the HC combine (`HcCombineMoeF16Kernel`): 1.28 ms against our
  `moe_combine_shared` 0.79 + `combine_norm` 1.03 = 1.82 ms per MoE layer. Exact in principle, and
  ~25 ms per prefill.
- **Plain HC combine** is about equal (1.09 vs 1.03 ms), but under ATT it waits 84% in `vmcnt` against
  our 44% `vmcnt` + 32% barrier.
- **DeltaNet scan:** gufo's row-split takes 1.42 ms plus a 0.6 ms epilogue, against our 2.03 ms. The
  DeltaNet gap is in conv / norm kernels, not the scan.
- **Attention:** ours is faster (3.19 vs 3.46 ms).
**2026-10-05 evening, stream E (exact efficiency; branch `stream/exact`).** Every row bit-identical (logits
compared with `cmp` against the unmodified build; gpu_tests property tests bitwise). A/B: `tools/ab_bench.py`,
4 rounds x 3 reps, pinned, 2048 tokens.

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| Routed + shared experts' sum built inside the next combine_norm, one block per token for all four streams (`halo_hc_combine_norm_tok_moe`; block_out's 21 MB write + read gone; gufo's HcCombineMoeF16 does the same) | -15..-26 (bytes: 336 MB at ~200 GB/s = 1.68 vs 2.16 ms a layer) | combine_norm (MoE half) + moe_combine_shared 53.1 + 46.1 -> 87.3 ms (-11.9 kernel; 1.86 ms a layer, ~180 GB/s, ATT 80% vmcnt); end to end -13.6 (within noise) | yes |
| First build of the above: the shared `__device__` template body was not inlined (an ABI call: flat loads, 166-200 VGPRs, a branch per load) | — | 141 ms (3.0 ms a layer); `__forceinline__` fixed it (87 VGPRs, 16 waves) | fixed |
| The same per-token kernel for the attention-half combine (`halo_hc_combine_norm_tok`, out read once instead of once per stream) | -2..-5 (L2 re-reads only) | 55.3 -> 56.9 ms | no (flag `--tok-combine`, off) |
| Attention q: split, q norm and RoPE in one pass (`halo_split_qgate_norm_rope`; q written once instead of three times, ~200 MB a layer) | -9..-13 | split + norm + rope 22.6 -> 11.2 ms; end to end -14.5 (significant) | yes |
| Shared expert gate/up (640 rows) on 64-row tiles (`_s`, 160 blocks) instead of three 256-row tiles (48 blocks, last half empty); stream P's finding | -5 (study: 0.405 -> 0.349 ms after writes) | 34.3 -> 24.9 ms (-9.4 kernel); end to end -17.5 (off arm spread 18) | yes |

| Producers write the fp16 copy their FP16 GEMM reads instead of fp32 + `halo_f16_rows` (`halo_attention_fa_h` context, `halo_scale_silu_h` mixer lo, `halo_silu_mul_h` shared expert; PLE value GEMM reuses the key GEMM's copy); `--f16-out` | -8..-10 | conversions 4.42 + 1.90 + (lo) gone, producers +0.2..+0.5; kernel sum -7.3, end to end -5.5 (within noise, spreads 10) | yes |
| tok_moe inject phase ablated (irows 0; timing only) | — | 87.4 -> 82.4 ms: the inject partials are ~5 ms of it | n/a |
| Deferred n-gram read vs layer 0 (`HALO_PREFILL_TIMING`) | — | read 67-73 ms warm (155 cold) while layer 0 takes ~30: ~35 ms of GPU idle per 2048-token prefill, charged to the PLE `halo_f16_rows`. `halo_disk_bench`: 4K random reads saturate at ~217K IOPS from QD 16 (PCIe Gen3 x4): IOPS-bound, so the levers are a larger row cache or a Gen4 link, not kernels | open (policy) |

| MoE down in 256-row blocks: two row tiles a wave, the activation stage and its barriers serve twice the rows (`MoeDown<RT>`; halogen's down runs 256 rows a workgroup and spends 10% at barriers vs our 29-34%) | -15..-23 | 133.2 -> 131.0 ms (-2.2; end to end within noise); bit-identical. ATT: barrier 23 -> 17%, but VGPRs 158 -> 240 (9 -> 6 waves) and VALU (dequant) 33 -> 38% with 30% VALU-stalled: the kernel is issue-bound on Q5_1 dequant once barriers shrink. The templated RT = 1 copy also lost 3 ms (166 VGPRs) | no (reverted) |
| Gate/up compiled for 7 waves (`amdgpu_waves_per_eu(7)`, the same source) | -15..-25 (halogen at 210 VGPRs) | compiler reaches 216 VGPRs only with 167 spills: not run. Note: a second caller of `MoeGateUp` stops it being inlined (420 B scratch), and forcing inline gives 256 VGPRs instead of 241: register count here depends on inlining order | no |

Exactness trap found on the way: with one element per thread, the backend fused the norm's `x * x` into
WarpSum's first butterfly add (`v_fmac` across the `ds_bpermute`), a different rounding from `halo_rmsnorm`
(whose loop keeps the product rounded). An empty `asm volatile("" : "+v"(ss))` keeps the rounding. A per-element
bitwise property test against the separate kernels caught it (2-3 ulp in the norm scale, rows > 0 only).

**DeltaNet scan, exact restructure (stream E, 2026-10-05 night; prompted by halogen's fused scan at 0.76 ms/layer).**

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| Probe: the update's 16-bit K gathers (384 `ds_load_u16` a wave a chunk) replaced by contiguous 128-bit loads (wrong operands, timing only) | ceiling | scan3u 83.6 -> 66.2 ms: the gathers cost ~17 ms | n/a |
| K hi / mid copied transposed into Q's LDS once Q S0 is done (two extra barriers a chunk), update reads them as 128-bit fragments; kl still gathered (`halo_gdn_c16_scan3ut`, bit-identical) | -6..-9 (2/3 of the probe, minus barriers) | 83.7 -> 81.3 ms (-2.4); end to end within noise. Spills 6 -> 15 at 256 VGPRs | no (reverted) |

Why the exact side is capped: the 96 scan blocks must all be resident (5 per WGP: 5 x 24 KB of the 128 KB LDS), so a
dedicated transposed K copy (12 KB) does not fit, and reusing Q's space costs two barriers a chunk on the serial
chain. Fusing prep2 into the scan moves its per-chunk work onto that chain too. Moving the q/k L2 norm into the scan
leaves the conv's bytes unchanged. Halogen's 0.76 ms comes mainly from an f16 state mirror (fewer WMMAs and splits per
chunk), which is a precision change (stream P).

**Stream E, remaining exact items (2026-10-05 night).** Bit-identical (logits `cmp`, bitwise property tests); A/B flag off vs on,
4 rounds x 3 reps, pinned, 2048 tokens.

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| tok_moe reads the expert rows 16 bytes at a time: each thread sums 8 contiguous indices (same per-element order), the row goes through LDS to the per-stream mapping (`halo_hc_combine_norm_tok_moev`, `--moe-vec`) | -5..-12 | 87.7 -> 79.6 ms (-8.1); end to end -20.7 (significant) | yes |
| k norm + RoPE + KV-cache append, and indexer-q norm + RoPE, one kernel each (`halo_norm_rope_rows`, `--kq-fused`) | -1.5..-2 | the five kernels 2.80 -> 2.24 ms (-0.56); end to end -2.9 (noise) | yes (marginal) |
| DeltaNet gated norm one wave per row, 8 rows a block, no barriers; the block sum reproduced as (p0 + p2) + (p1 + p3) (`halo_gated_norm_h128w`, `--gated-wave`) | -2..-4 (0.73 ms a call vs ~0.6 floor) | 26.4 -> 20.4 ms (-6.0; 0.57 ms a call, at the floor); end to end -3.3 (noise) | yes |
| Flash attention reads the gate in place from the q / gate projection rows; the q split no longer writes gate_a (`halo_attention_fa_hg`, `--gate-in-place`; only where the dense flash branch runs) | -2..-3 | split 11.25 -> 9.12 ms, fa +0.1 (-2.0 kernel); end to end -14.8 (within noise; the next GEMM's spread is ~20) | yes |

Multi-chunk check for this set (control flow per chunk): all new flags on vs off give identical logits for 2048 tokens in
512- and 768-token chunks, and for a 3485-token prompt in 2048-token chunks (second chunk on the QSA path, gate_a written).

**Candidate A (halogen-style dense attention: one workgroup per KV head x 32 tokens over all 12 q heads), stopped at the
gate (2026-10-05 night, stream E).** The case for it was halogen's dense kernel at T = 8192 (36.3 vs our 43.1 ms a
call). In our engine the dense flash kernel only ever runs while the context is within the indexer budget
(top-k 2048 tokens / ratio 4 = 512 blocks), i.e. for the first 2048 positions of a prompt; beyond that the model's
QSA selection applies and attention is `halo_attention_wmma_qsel_h`, which already covers all 12 heads of a KV head
per block (48 rows). Kernel timing of a 6940-token prefill (pinned): `fa_hg` 12 calls, 39.4 ms (first chunk only);
`qsel_h` 120 calls, 357 ms; `qsa_select_rows` 41.5 ms. So the 8K saving (-70..-90 ms) does not exist for this model, and
at 2048 tokens the ceiling is 12 x (3.28 - ~2.9) = ~-4.5 ms (0.3% of the prefill, below the A/B noise of 10-20 ms).
Not built. The long-prompt attention lead is `qsel_h` (3.08 ms a 512-token call, ~6% of an 8K prefill).
**Precision options (stream P, 2026-10-05, branch stream/precision).** The roles in the table above where gufo
rounds more than we do, one flag each, gated by the quality sets (docs/BENCHMARKS.md, "Precision options")
and an in-engine A/B (`ab_bench.py`, 4 rounds x 3 reps, pinned, prose_1 at 2048 tokens):

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| Router on plain fp16 WMMA (`halo_gemm_f32h_f16_k128`: fp32 W rounded to fp16 while staging, the fp16 `mh`; `--router-f16 1`) | -15..-20 kernel; quality neutral | router 40.7 -> 21.9 ms (-18.8); end to end -20.6 ms; long set **+0.0021 ± 0.0007** (9 of 12 cases worse) | no (quality) |
| DeltaNet alpha/beta + shared-expert gate on the same kernel (`--small-f16`) | -18..-24 kernel; quality neutral | 27.9 -> 7.6 ms (-20.3, spreads 0.9 / 0.2); end to end -14.8 ms (inside ab_bench's spread rule: new spread 22.7); quality passes | yes (default on) |
| Mix epilogue reads xn as combine_norm's fp16 K-blocked copy, one 8-byte load per (token, stream) and lane, float4 / half2 stores (`halo_gemm_q8r_f16_mixth`, `--mix-h16`; coordinator's candidate from gufo's ATT: gufo mixes from fp16 xn) | -10..-25 kernel (the earlier fp16 probe, with per-element gathers, was +10) | mix 107.9 -> 71.9 ms (-36.0); end to end **-36.0 ms, significant** (spreads 10.9 / 1.8); max \|Δlogit\| 0.57 of 16.8 on prose_1, same argmax; quality passes | yes (default on) |
| Shared-expert gate/up as W8A8: `halo_quantize_q8_kblocked` (mixed -> int8 K-blocked, per-32 scales) + 2x `halo_gemm_q8a8_128x128` (new int8 GEMM with GemmQ8rF16's register prefetch; study: 0.236 ms after writes vs `_l` 0.405) (`--shexp-w8a8`) | -7..-11 kernel | gate/up 34.2 -> 22.6 ms, but the quantize pass costs 7.5 ms (0.16 ms a call, ~134 GB/s after mixth's writes): net -4.1 kernel (spreads <= 0.6); end to end -8.7 ms within noise; quality passes | yes (marginal; next: quantize in the mix epilogue or from `mh`) |
| Router with fp16 W hi/lo + fp16 activations (`halo_gemm_f32h_f16_w2`, `--router-f16 2`) | -11..-15 kernel; long ~+0.001 | router 43.4 -> 24.4 ms; end to end -19.1 (noise); scores bit-identical to `--router-f16 1`: the weights are exact bf16 and 99.97% exact fp16, so the lo half is ~0 | no (quality) |
| Router with fp16 W + fp32 `mixed` split hi/lo (`halo_gemm_f32_f16_x2`, two WMMAs, K stage 64; `--router-f16 3`) | -11..-15 kernel; long ~+0.001 | router 42.9 -> 30.3 ms (-12.6); end to end **-23.7 ms, significant**; long +0.00009 ± 0.00057 | yes (default `--router-f16 3`) |
| Mixer down as W8A8: combine_norm (template kQ8) writes xn as int8 K-blocked + per-32 scales, `halo_gemm_q8a8_160x64` replaces `_mb` (study: 0.583 vs 0.644 ms after writes) (`--down-w8a8`) | -8..-15 | `_mb` 59.1 -> 50.8 ms (-8.3), combine_norm unchanged (108.3 -> 108.5: the bytes it no longer wrote bought nothing), kernel sum -3.5, end to end -0.6 (noise); quality passes. With `--mix-h16` on, combine_norm must write the fp16 copy anyway | no |
| DeltaNet qkv and z GEMMs write fp16 (`halo_gemm_q8r_f16_lr8h`), conv8 / state advance / gated norm read fp16 (`--gdn-io16`; coordinator's halogen finding, input side only, the scan untouched; ROOFLINE 4g's fp16 conv output was -1 ms with the old conv) | -8..-20 (134 MB fewer per layer) | conv 28.8 -> 23.2, gated norm 26.5 -> 25.5, the GEMMs unchanged (177.6 -> 177.7: their fp16 stores are not what bounds them); end to end +9.3 ms (noise, spreads 15 / 50); quality passes | no |
| Hyper-connection residual stored in 16 bits: every prefill reader / writer (broadcast, combine, combine_norm and E's per-token variants, both row norms, PLE norm + conv, last-token widening) templated on the element type, the norm fed the stored value (`--res16 1` fp16, `2` bf16; user-assigned from CODESIGN / halogen) | -30..-60 end to end (CODESIGN: -80..-150, which counted the mix's residual reads that `--mix-h16` had already removed) | fp16: combine_norm 107.8 -> 84.3 ms (-23.5, spreads 1.6 / 5.0), end to end -24.2; bf16: the same (-24.5, significant). No overflow (max \|Δlogit\| 0.75 fp16, 0.52 bf16, all finite); `--res16 0` bit-identical to the fp32 build. Long set: fp16 -0.0006, bf16 +0.0015 | yes, fp16 (default `--res16 1`) |
| Shared-expert W8A8 re-measured after merging main (stream E runs these GEMMs on `_s` 64-row tiles, exact) | — | `_s` 24.9 ms vs W8A8 22.5 + quantize 7.6 = 30.1; kernel sum +0.1, end to end -9.4 (noise, base spread 43) | no (default off again) |
| DeltaNet scan with K, Q and the state split two-way in **fp16** (~22 bits) for K S0, Q S0 and the update, 3 WMMAs each instead of 6 / 3 / 5; Delta and O keep bf16 hi / lo (`halo_gdn_c16_scanh`, `--gdn-scan-h`; coordinator's candidate from halogen's f16 state mirror, which as a single fp16 copy would be ~2^-11, far below the failed two-way bf16) | -15..-25 (WMMAs a chunk 118 -> 78, issue-bound ~80%) | scan 77.2 -> 42.7 ms (-34.5, spreads 0.3 / 0.1); end to end -34.2 (inside ab_bench's spread rule: spreads 36 / 33); 256 VGPRs, 0 spills (scan3u 6); quality passes (long +0.0006 ± 0.0009) | yes (default on) |
| fp16 residual re-measured on the merged build (stream E's per-token combine kernels) | — | `combine_norm_tok_moe` 89.2 -> 76.4 ms, rows 55.6 -> 42.3; end to end -27.4 ms, significant | yes |

Stream P total, everything kept, against main 4ad6410 (ab_bench, 4 rounds, pinned, 2048 tokens): 1433.2 -> 1295.1 ms,
**-138.1 ms, significant** (1429 -> 1581 tok/s). By role (kernel-timed rep): mix 108.2 -> 70.2, MoE-half combine
87.7 -> 76.4, attention-half combine 55.6 -> 42.3, DeltaNet scan 80.3 -> 42.7, router 41.3 -> 32.0, small F32
projections 28 -> 9. Cumulative quality passes the gate against the original baseline (docs/BENCHMARKS.md).
Not kept: plain fp16 router (long set), fp16-weight-split router (= plain), mixer down W8A8 and shared-expert W8A8
(after the merge), DeltaNet qkv/z fp16 I/O (no speed). Open: the quantize pass that sank shared-expert W8A8
(fold it into the mix epilogue), and bf16 instead of fp16 residual (same speed, worse long set).


**Main after merging streams E and P (2026-10-06).** Stream E's vectorized MoE sum now also has 16-bit residual variants
(`halo_hc_combine_norm_tok_moev_r16` / `_rb`), bit-identical to the scalar `tok_moe` ones; with the precision options
off, logits are bit-identical to the pre-stream baseline; gpu_tests 42 passed. ab_bench (4 rounds x 3 reps, pinned,
prose_1 at 2048 tokens): precision options off 1406.3 ms, defaults **1279.7 ms (1600 tok/s), -126.7 ms, significant**.

## 9. 2026-10-06 (stream D, branch `stream/decode`): decode, the K-blocked expert layout, and attention

Measured on main 56693d8 with `halo_run --device-sample 1` (steady median of 76 steps after 4 warm-up steps)
and `--decode-timing 1` (sync-free per-kernel stamps, eager: ~2.5 ms/step more than graphs). Long context:
`--prompt-file prose_long.txt --prefill-prompt 1` (6,939 tokens prefilled, then stepping).

| Context | ms/step (tok/s), default clock | kernel sum (eager) |
|---|---:|---:|
| 5 tokens ("The capital of France is") | 43.5 (23.0) | 46.2 |
| 6,939 tokens | 55.4 (18.1) | 58.1 |

**Classification (short context, per step).** Bytes per call against the call time:

| Kernel | ms/step | calls | GB/s | Gap to ~220 GB/s |
|---|---:|---:|---:|---:|
| `gemv_q8r_dual` (DeltaNet qkv + z, 44.6 MB) | 7.24 | 36 | 222 | 0 |
| `moe_gate_up_q4_k_dot` (10 experts, 18.4 MB) | 6.07 | 47 | 143 | 2.1 |
| `gemv_q8r_r4`, 80 blocks (2560 x 6144 out projections, 16.7 MB) | 5.08 | 49 | 161 | 1.4 |
| `hc_mix_a` (320 x 10240 down + inject, 3.7 MB) | 3.60 | 96 | 99 | 2.0 |
| `moe_down_q5_1_r4` (12.3 MB) | 3.27 | 43 | 162 | 0.9 |
| LM head (`gemv_q8r_r4`, 7760 blocks, 676 MB) | 2.80 | 1 | 242 | 0 |
| `hc_mix_b` (10240 x 320 up, 3.5 MB) | 2.69 | 97 | 126 | 1.2 |
| F32 router (`matvec_f32_rowblock`, 5.2 MB) | 1.79 | 48 | 141 | 0.6 |
| small kernels (norm scales, combines, top-k at 15.8 us, rope, norms, copies; < 16 us each) | ~3.5 | ~600 | — | most of it |
| host gap between steps (n-gram row read + upload + launch; charged to `halo_broadcast`) | ~1.0 | 1 | — | ~1 |
| `attention_select` at 5 tokens / **at 6,939 tokens** | 0.35 / **12.39** | 12 | — | **~11 at 7K** |

At long context one kernel takes over: `halo_attention_select` (24 blocks, one per head, 17 VGPRs) walks the ~2K
visible positions one dependent load at a time; 1,033 us a call.

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| Q4_K gate/up experts K-blocked for **decode** (`halo_moe_gate_up_q4_kb_dot`: the `_dot` kernel reading the blocked layout, same sums, outputs bit-identical over 51 id sets; patch `/data/halo-work/streams/D/blocked_decode_and_attn.diff`) | 0 ± 0.3 ms/token (-0.3..+0.5): a decode block already reads two contiguous 23 KB runs; the blocked layout only shortens them to 2.3 KB | `halo_hwprobe sweep` at DRAM size (4000 blocks, 184 MB): row-major 224.5 / 232.3 GB/s, blocked 229.2 / 231.5: the pattern costs nothing at these run lengths. Kernel (gemv_bench, 3 rounds): 114.9 -> 123.1 us (+7%), ~+0.4 ms/token in engine. Prefill on current main (ab_bench 4 x 3, pinned): gate/up 249.8 -> 237.3 ms (-12.5), end to end -16.3 ms within noise (spreads 10.7 / 17.3) | **no** |
| Decode attention with batched loads (`halo_attention_select_b`: q in LDS, key rows as 8 float4 at a time, the V sum reading 16 positions before folding them in order, scores and visible list through 1024-entry LDS tiles; every dot product and sum in the original order, 51 VGPRs; `--decode-attn-b`, default on) | 1,033 -> 100-250 us a call at 7K; step -8.5..-10.5 ms at 7K, noise at 5 tokens | 1,033 -> 162 us a call (kernel sum 58.1 -> 47.6 ms/step); pinned A/B 53.75 -> 44.22 ms/step at 7K (below). Logits bit-identical over 80 decode steps at 7K context (flag 0 vs 1), tokens identical; property test bitwise vs the serial kernel | **yes** |

Decode A/B of the kept change (`--decode-attn-b 0` vs `1`, same binary, interleaved, 3 rounds, clock pinned,
steady median of 76 steps): 6,939-token context **53.75 -> 44.22 ms/step (-9.5 ms, 18.6 -> 22.6 tok/s)**, spreads
0.03 / 0.02; 5-token context 42.93 -> 42.64 (-0.3), spreads 0.01. Predicted -8.5..-10.5 at 7K, noise at 5 tokens.
Exactness: decode logits bitwise equal over 80 steps at 7K context; every pre-existing kernel's ISA is identical to
main 56693d8's (only constant-table offsets move), so `--decode-attn-b 0` is main's decode; greedy tokens and
printed margins equal main's binary over 80 steps; prefill logits (prose_1, 2048) equal main defaults'
(`logits_pmerge_def.bin`); gpu_tests 43 passed, plus the small-quant engine tests.

Why the blocked layout is not kept: it would have to replace the row-major experts (two copies do not fit), so
decode would pay +0.4 ms a token for -12.5 ms a 2,048-token prefill: it breaks even at ~30 decoded tokens a
prefill, and the prefill gain is inside end-to-end noise. Decode's gate/up gap (143 vs ~220 GB/s) is not the read
pattern: the sweep reads both patterns at ~230 GB/s. Its likely causes (not yet probed): each of the 400 blocks
quantizes the activations before issuing any weight load (a serial prologue per block generation) and the
~110 us kernel's ramp and tail.

Remaining decode levers, by size (short context, ms/token): the hyper-connection mixers' two kernels at
99 / 126 GB/s (~3.2); ~600 small launches (~3.5, fusion: section 5's list); MoE gate/up prologue (~2.1); the
80-block 2560-row GEMVs at 161 GB/s (~1.4, more row groups or split-K); host gap ~1 (prefetching the n-gram rows
is not possible before the token is known; overlapping it with the LM head and sampling would need the token
earlier); attention at long context after this change ~1.9 at 7K (floor ~0.5).

**Stream S (exact efficiency 2, branch `stream/exact2`, 2026-10-06).** 8K = 8192 tokens of prose_long + prose_8888
(`streams/S/prose_10k.txt`; prose_8888 alone is 3485 tokens), pinned, kernel timing of the last rep.

*Long-prompt attention.* Span trace of `halo_attention_wmma_qsel_h` (launch 130 of 144, sample 7): waves live
480 us, 102 chunks of 32 keys at ~4.0 us (P.V + barrier 1.02, staging 0.76, scores 0.71, score write + barriers 0.73,
softmax 0.70), plus a 26.6 us prologue and a 26.2 us epilogue; ATT: barrier 45.7%, lgkmcnt 20.0%, vmcnt 18.1%,
WMMA 0.9%. Every row bit-identical (8K and 2048 logits `cmp`, property test bitwise):

| Change | Predicted | Measured (qsel_h per 8K prefill, 144 calls) | Kept |
|---|---|---|---|
| Q staged a row per wave, all loads first; the gate loads batched before the outputs | -40..-50 | 460.0 -> 424.6 ms | yes |
| `halo_attention_wmma_qsel_w`: 6 waves own their rows (wave = row tile x key tile for scores, x dim half for P.V), scores swapped between the pair through LDS, qsel_h's softmax replayed in registers (DPP sequential 8-key sums), V transposed, Q/K swizzled to fit 64 KB; 3 barriers a chunk instead of 5 | -150..-210 | first build 662 ms; after the next row's fix 520 ms (3.61 ms a call vs 2.95): slower | no (642bd91) |
| The union list (ublk / umask) read a chunk ahead, one key per lane, raw values (each staged piece and each prefetch address re-read it: dependent load rounds a chunk; found in qsel_w's trace: 2.2 of 5.6 us a chunk in vmcnt) | not written before measuring | 424.6 -> 396.0 ms | yes |
| V staged transposed ([key/16][dim][key%16]): P.V fragments are 32 bytes instead of 16 two-byte gathers | -20..-35 | 396.0 -> 386.3 ms | yes |

Why qsel_w lost (trace after the list fix not taken; first-build trace): its score phase took 1.5 us for 16 WMMAs a
wave at 3 waves/SIMD (0.57 us of issue; qsel_h's took 0.71 with the same chain), lgkmcnt 33.8%: six waves at
256 VGPRs, the duplicated softmax and the exchange add LDS round trips that the five-barrier kernel's 8 waves
overlapped. Same lesson as `fa`: owning rows removes barriers but not the LDS-latency-bound fragment traffic.
Halogen's sparse path (`k_sel_tilemask` + `k_attn_fa2` with a tile mask) was not run: its FaSel layout is not known.

End to end, merged main 5e5466e vs this branch (ab_bench, 4 rounds x 3 reps, pinned): 8192 tokens, ab_bench's device
metric (it keeps the last `device` line of each rep, i.e. the last 2048-token chunk, positions 6144-8191)
1346.7 -> 1324.0 ms, **-22.7 ms, significant** (spreads 8.0 / 5.2); over the whole 8K prefill the kernel-timed rep
gives qsel_h 460.4 -> 386.7 ms (-73.7, spread 0.6). 2048 tokens (qsel_h does not run): 1243.7 -> 1245.3, +1.7
within noise. Logits byte-identical to main at 8K and 2048 (`logits_main_5e5466e.bin`); gpu_tests 45 passed.

*Dense GEMMs over pre-dequantized fp16 weights (probe `--w16`, 642bd91, removed).* The fp16 copy holds the values
the staging computes (fp16(q x scale) by the same fp16 operations; q x scale itself is not always representable),
so all `_w16` GEMMs are bit-identical (property test over the whole fp16 scale range; 2048 logits identical).
6.67 GiB more (400 tensors). At 2048 tokens, kernel ms (Q8 -> fp16 weights): lr8 qkv 110.8 -> 138.8, lr8 z
66.6 -> 84.5, lr8 q/gate 43.2 -> 54.8, `_l` 119.5 -> 136.9, `_mb` 57.3 -> 90.4, mixth 69.1 -> 66.3, `_s` 30.1 -> 28.9;
dense GEMMs 496.5 -> 600.5 ms (+104), prefill 1.275 -> 1.376 s. Predicted -12..-30. The dequantization is not
on the critical path; the doubled weight bytes and the prefetch registers (lr8 6 -> 5 waves/SIMD) are. Against
gufo's profile (per call, 2048 tokens): qkv + z 4.84 vs 5.15 ms (16384 x 2560 fused), q/gate/k/v 4.07 vs 4.62,
mix + epilogue 0.72 vs 0.89; the remaining dense-class gap is the mixer down (`_mb` 0.60 vs W8A8 0.50) and
attention out, both precision choices (W8A8), not dequantization.

**Precision options, second round (stream R, 2026-10-06, branch stream/precision2).** Same gate and A/B protocol as
stream P (`ab_bench.py`, 4 rounds x 3 reps, pinned, prose_1 at 2048 tokens; quality: docs/BENCHMARKS.md).

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| Shared-expert W8A8 with the quantize folded into the mix epilogue (P's open item) | ~0 (-1..+1): the ceiling is the GEMMs' 22.5 vs `_s` 24.5 ms = -2.0, the producer pays ~+1.9..2.4 (5.9 MB more written a call x 48 by a DRAM-bound epilogue, per-32 absmax across 32 rows); even a 64-row W8A8 tile gives -3..-5 | not built (below the ~10 ms noise, not free). The int8 copy of `mixed` cannot feed the mixer down (it reads xn, another tensor); for the routed Q4_K gate/up an iu8 path needs a cvt + scale + fma per int32 accumulator per 32 K (~12 VALU a WMMA vs ~3-5 dequant today; iu8 runs at the f16 rate) | no |
| DeltaNet data path in fp16 (`--gdn-f16 1`): P's fp16 qkv / z GEMMs, the conv writes fp16 q / k / v (`halo_conv_l2norm_prefill8_hh`), prep2 reads them (`_h`), the scan takes K and Q exact in fp16 so K S0, Q S0 and the update drop the lo halves (2 WMMAs instead of 3, 78 -> 54 a chunk; `halo_gdn_c16_scanf`, 232 VGPRs), reads V and writes O in fp16, and E's one-wave gated norm reads fp16 o and z (`halo_gated_norm_h128w_hh`). Differs from P's `--gdn-io16` (no gain: it also turned E's one-wave norm off) by the output side and the scan | -27..-40 kernel (conv -12..-15, norm -7..-10, scan -7..-13, prep -1..-2) | conv 28.9 -> 20.7, gated norm 20.5 -> 12.2, scan 42.6 -> 34.0, prep2 13.3 -> 9.7, GEMMs unchanged: -35.3 kernel; end to end **-35.1 ms, significant** (1278.0 -> 1242.9); max \|Δlogit\| 0.64, same argmax; quality passes | yes (default on) |
| The same with the update's K^T from a transposed fp16 copy staged once a chunk (`halo_gdn_c16_scanft`, `--gdn-f16 2`; LDS freed by the dropped lo halves) | -3..-6 on the scan | scan 38.5 -> 45.9 ms (unpinned, one rep): slower (16 bank-conflicting 16-bit stores a thread, 236 VGPRs) | no |
| Router mode 4 (`--router-f16 4`): the MoE-half mix epilogue writes the fp16 lo of `mixed` (`halo_gemm_q8r_f16_mixthl`) and skips the fp32 copy nobody then reads, the experts' permuted copy comes from mh (`halo_f16_perm_rows_h`), the router reads fp16 hi / lo with fp16 W converted once on 128 x 64 tiles (`halo_gemm_h_hl_128x64`; L2 traffic 335 -> 168 MB a call). Bit-identical to mode 3 (same products, same order) | -6..-10 kernel; tiles BK 32 <= BK 64 < 64 x 64 | router 31.1 -> 25.5 (BK 64; BK 32 48.6, 64 x 64 33.8), mixthl -0.5 against mixth, perm 6.0 -> 7.1; kernel sum +1.6 (downstream GEMMs moved within their spreads), end to end +0.6 ms (noise) | no (code kept, default mode 3) |


Stream R total (stream/precision2 with main a9e0af3 merged, against main 56693d8's binary, same A/B): main 1280.7, R with
`--gdn-f16 0` 1278.2 (within noise; logits byte-identical to main's defaults), R defaults **1246.0 ms, -34.7 ms
significant** (1599 -> 1644 tok/s). Cumulative quality = the `--gdn-f16` row (the only kept change).
**Experts per token as an option (`--moe-topk N`, stream Q, CODESIGN first experiment arm B; not a default).**
- ab_bench, 4 rounds x 3 reps, pinned, prose_1 at 2048 tokens, main defaults: 1281.0 ms at k = 10, k = 8
  **-78.8 ms**, k = 6 **-154.1 ms**, both significant.
- Predicted -15 / -35 from "experts stream every weight once whatever k is". Wrong: the grouped gate/up
  (251 -> 208 -> 162 ms) and down (133 -> 111 -> 86 ms) scale with pairs. At prefill they are bound by per-pair work,
  not by the expert byte stream, so time per pair is the lever in those kernels.
- Decode -1.58 / -3.45 ms per token.
- Quality: k = 8 passes the gate (short +0.0105), k = 6 fails (+0.0565). Default stays 10.

## 10. 2026-10-06 (stream D2, branch `stream/decode2`): decode at 36.9 ms, and what the memory system punishes

Measured with `halo_run --device-sample 1` (steady median of 76 steps; 36 for the per-bit sessions), clock pinned,
interleaved arms of one binary (`--decode-opt MASK`; 0 is main's decode: its logits equal main's binary).
Long context: `--prompt-file prose_long.txt --prefill-prompt 1` (6,939 tokens). Working files and every session's
raw output: `/data/halo-work/streams/D2/` (`PREDICTIONS.md` holds each prediction, written before its run).

**Byte floor per token (short context, 240 GB/s): 6.55 GB = 27.3 ms.** Dual qkv + z 1.605 GB, ssm_out 0.601,
attn_q 0.401, attn k/v 0.033, attn_out 0.200, hyper-connection mixers 0.690, routed gate/up 0.865 (Q4_K) +
0.023 (Q5_K layer), routed down 0.529 (Q5_1) + 0.087 (Q8), shared expert 0.250, F32 router 0.252, alpha/beta 0.035,
LM head 0.676, indexer 0.039, DeltaNet state read + write 0.226, PLE 0.035. Main: 42.7 ms = 64% of the floor
rate; this branch: 36.9 ms = 74%.

| Bit | Change (all bit-identical; a property test per kernel against the launch it replaces) | Predicted (ms/step) | Measured | Kept |
|---|---|---|---|---|
| 1 | Layer 2's experts are Q5_K: the generic per-expert path (5 launches an expert, byte-at-a-time `halo_matvec_q5_k_expert`, ~1.05 ms). One launch for all 10 experts' gate + up + SiLU (`halo_moe_gate_up_q5_k_rows`: a lane's super-block pieces loaded first, LoadQ5_K's arithmetic, matvec_q5_k's order), then `halo_moe_down_q8r_r4` + `halo_moe_combine` (same sums as the r4 down + axpy chain) | -0.75..-0.9 | -0.75 | yes |
| 2 | QSA indexer q/k (BF16, generic matvec: 37 + 26 us a layer) in one launch with 16 loads in flight a lane (`halo_matvec_bf16_dual`) | -0.5..-0.6 | -0.5 (24 us a layer) | yes |
| 8 | `hc_mix_a` declared 48 KB of LDS for a 1536-column chunk: 40 of its 168 blocks resident. Chunk-sized LDS (`halo_hc_mix_a_c`) | -0.8..-1.3 | -0.85 (37.5 -> 28.6 us) | yes |
| 64 | `halo_qsa_scores`: one thread a block, 512 dependent FMA + L1 loads (35 us even at 1 block). q in LDS, rows as float4, the heads' chains side by side (`_b`) | -0.37..-0.4 | -0.19 (36.9 -> 7.9 us eager) | yes |
| 256 | Routed combine folded into the shared-expert down launch (`halo_shexp_down_combine`) | -0.2..-0.25 | -0.12 | yes |
| 2048 | Graph split before the PLE layer (layer 1): params uploaded, part 1 launched, the n-gram rows read meanwhile into pinned memory and uploaded stream-ordered, part 2 launched. The host gap (D2H token, O_DIRECT n-gram read on a cache miss, H2D, launch) was 0.23-1.4 ms in the trace and the reason step medians sat ~0.85 ms over the minimum | -0.5..-0.9 | -0.15..-0.6; median now on the minimum | yes |
| 32768 | The routed gate/up quantized x in every one of its 400 blocks (ISA: 8 warps x 10 serial rounds of an L2 load + ~20 dependent shuffles) before its first weight load. The router launch (same x) now quantizes it once (80 of its 512 blocks, one warp each) into a global Q8Act; gate/up copies 5 KB into LDS (`halo_moe_gate_up_q4_k_dot_g`). D's P1 hypothesis, confirmed | -0.5..-1.4 | **-1.02** (130 -> 104 us) | yes |
| 65536 | Block-per-row F32 dots (router, alpha/beta, shared-expert gate logit) compile to load, wait, FMA per iteration (10 serial DRAM rounds a thread). Loads hoisted (`_u`, `RowDotHoisted`), the chain explicitly `fmaf` | -0.6..-0.9 | -0.43 | yes |
| 131072 | The same hoisting in the norm scales, combines, mix_b's partial sums, mix_a's staging and inject rows | -0.6..-0.9 | -0.24 | yes |
| 262144 | `halo_gdn_step_d128_norm` held its 128-float state column in registers (256 VGPRs, 152 spilled). `_r`: read it in chunks of 32 for A and B, again (L2) for the update: 87 VGPRs, no spills | -0.2..-0.35 | -0.06 / -0.92 (39.3 -> 25.4 us eager) | yes |
| 524288 | Router top-k selection by warp shuffles (`halo_router_topk_w`) instead of 10 block rounds | -0.4..-0.5 | -0.12 (15.7 -> 12.8 us: the softmax is the rest) | yes |
| 4 | Q8 GEMVs with cols <= 6144 staging x in 24 KB (r4n / r2n): all 80 blocks of the 2560 x 6144 projections resident | -0.9..-1.5 | **+1.15** (103.7 -> 121 us; LM head 2796 -> 2896) | no |
| 16 | DeltaNet qkv + z GEMV with 16 KB LDS: 160 blocks resident instead of 40 | 0..-0.4 | **+2.4** (201 -> 267 us) | no |
| 32 | Gate/up with each lane's 15 weight loads issued before quantizing (`_dot_p5`) | -0.9..-1.6 | **+2.6** (129 -> 179 us) | no |
| 128 | Combine + the next mixer's norm scales in one launch (4 blocks) | -0.3..-0.4 | ~0 | no |
| 512 | DeltaNet alpha/beta rows as extra blocks of the qkv + z launch | -0.25..-0.35 | ~0 | no |
| 1024 | One-chunk register prefetch in the Q8 GEMVs (two load rounds in flight a wave) | -1.2..-2.1 | **+0.45** (80-block 103 -> 107.5, dual 200 -> 209) | no |
| 4096 | Weights before the prologue in the Q5_1 expert down and mix_b | -0.8..-1.3 | **+2.2** (down 75 -> 117 us) | no |
| 8192 / 16384 | 2 / 1 rows a wave for the 2560 x 6144 GEMVs | -0.5..+0.5 | -0.04 / +0.6 | no |
| 1048576 | mix_a's three weight rounds a lane issued as one (pinned with `sched_barrier`; the scheduler sinks hoisted loads back otherwise) | -0.35..-0.7 | **+0.25** | no |
| 2097152 | mix_b's weights before its (hoisted) partial sums | -0.2..-0.3 | **+0.3** | no |
| 4194304 | Shared-expert gate and up rows in one column loop (4 loads a round, 5 rounds instead of 2 x 5) | -0.25..-0.35 | -0.01 | no |

**End to end (final A/B, 3 rounds, pinned, 76 steps; main's binary, this binary at `--decode-opt 0`, default):**
short context 42.73 / 42.77 / **36.91 ms (23.4 -> 27.1 tok/s, -5.82 ms, -13.6%)**; 6,939-token context 44.25 /
44.24 / **38.95 ms (22.6 -> 25.7 tok/s, -5.29 ms, -12.0%)**; round spreads <= 0.46 ms (short default 37.37 /
36.91 / 36.90). Exactness: decode logits byte-identical to main's binary over 65 steps at short and 66 steps at 7K
context, greedy text identical in all 9 device-sampled runs, prefill logits (prose_1, 2048) identical to
`logits_main_5e5466e.bin`, every new kernel bitwise against its original (Hegel properties, both backends); gpu_tests 57 passed with the small-model fixture, and SmallQuantizedModelRunsFastPathsAndMatchesReference, ChunkedPrefillMatchesStepping and GreedyMatchesEngineStepsAndReuseIsExact pass with both fixtures.

**What the measurements say about this memory system.** Five changes that issue a kernel's *weight* loads earlier
or more of them at once (more resident blocks, a register prefetch, weights before a prologue, one-round loads)
each made the kernel slower, by 4-50%, even with the same bytes per wave; the hwprobe sweep reproduces the
residency direction in steady state (6 KB rows: 235 GB/s at 40 resident blocks, 213 at 100; 2.5 KB rows: 245 vs
223) but not its size. Hoisting a thread's *loop* loads (small, L2/MALL-resident or few-page reads that the
compiler had serialized with a wait each) helped every time. Concurrent distinct DRAM pages (or their address
translation: all 79 GiB live in GTT, 4 KB pages, VRAM is 512 MB) is the working hypothesis; HARDWARE.md's flat
~34 GB/s for random 4 KB pages fits it. Not tested further here. Launch fusion is worth ~0-1 us a launch inside the
graph (rocprof's 2.4 us gaps are the profiler's); bits 128 and 512 removed 96 and 36 launches for nothing.

**Where the 36.9 ms go (final, eager kernel timing; floor at 240 GB/s):**

| Kernel | ms/step | calls | us/call | MB/call | GB/s | floor ms | gap ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gemv_q8r_r4` 2560 x 6144 (ssm_out, attn_out, 80 blocks) | 5.05 | 49 | 103.1 | 16.7 | 162 | 3.41 | 1.64 |
| `moe_gate_up_q4_k_dot_g` | 4.89 | 47 | 104.1 | 18.4 | 177 | 3.61 | 1.28 |
| `hc_mix_a_cu` | 2.65 | 96 | 27.6 | 3.65 | 132 | 1.46 | 1.19 |
| `hc_mix_b_u` | 2.54 | 97 | 26.2 | 3.48 | 133 | 1.41 | 1.13 |
| `moe_down_q5_1_r4` | 3.15 | 43 | 73.2 | 12.3 | 168 | 2.20 | 0.95 |
| `gemv_q8r_dual` (qkv + z) | 7.21 | 36 | 200.2 | 44.6 | 223 | 6.68 | 0.52 |
| `shexp_gate_up_u` + `shexp_down_combine_u` | 1.85 | 96 | 23.1 / 15.4 | 3.48 / 1.74 | 151 / 113 | 1.05 | 0.80 |
| `router_rowblock_q8act` | 1.35 | 47 | 28.7 | 5.24 | 183 | 1.03 | 0.32 |
| LM head, attn_q, DeltaNet step, Q8 / Q5_K experts, indexer, alpha/beta, k/v, PLE | 7.22 | | | | 94-248 | 6.44 | 0.78 |
| small kernels (norm scales, combine, top-k 12.8 us, attention prep, copies) + eager stamps | ~4.1 | | | | | | |

Streaming kernels 35.9 ms against a 27.3 ms floor (eager; the graph runs ~3 ms faster than the stamps).

**Remaining levers (ms/step, short context):** the 80-block 2560 x 6144 GEMVs 1.6 (two lock-stepped rounds of 40
blocks, one 2 KB load round in flight a wave; more residency and prefetch both lost, so the next try is a different
row-to-block mapping, measured, not predicted from bandwidth); routed gate/up 1.3 and down 0.95 (same family); the
mixers 2.3 (lock-stepped one-to-three rounds of ~1.4 MB); shared expert 0.8; small kernels ~2 in the graph
(attention prep chain of ~15 launches a layer, top-k softmax 12.8 us x 48); 7K attention 1.8 (floor ~0.5). A
page-locality / translation probe (same bytes in flight over 4 KB vs 64 KB vs 2 MB spans, at 40 / 100 / 160
resident blocks) should come before any of these.
## 11. 2026-10-06 (stream L, branch `stream/longctx`): long-prompt prefill (8K)

8K = the first 8192 tokens of `streams/S/prose_10k.txt`, pinned unless noted. End-to-end metric: `ab_bench.py
--chunks 0` sums every chunk's device time of a rep (whole prefill; the old metric kept only the last chunk).

**Profile of main 19d80bb.** Device per 2048-token chunk 1243 / 1319 / 1348 / 1425 ms (5335 total; the 2048 prompt
alone 1243); wall 5.38 s pinned (1522 tok/s), 5.37 s default clock (1527 tok/s). The QSA path (positions >= 2048)
costs `qsel_h` 386.6, `qsa_select_rows` 52.3, `qsa_scores_wmma` ~21, `union4` 3.4 = 463 ms, in place of what dense
`fa_hg` would cost at 2048 (3 x 39.8): ~+345 ms long-context-specific; everything else is the per-chunk cost a
2048 prompt also pays. Halogen claims 1584 tok/s at 8K (one 32K chunk); no gufo 8K number in the docs.

**Selection statistics** (debug dump of every token's selection, 12 layers x 6144 tokens): the union of 4 tokens
is 3042 keys (1.49 x 2048) at block granularity; a tile mask with 16-key tiles already covers 0.89 of the causal
context, 32 tokens x 64-key tiles 1.00. Halogen's sparse path (`k_sel_tilemask` + `k_attn_fa2`) is therefore dense
attention at 8K: at our dense efficiency ~594 ms for chunks 2-4 (halogen's 1.19x per-key edge: ~500) vs the QSA
path's 463. Not a lever for this model at 8K; the FaSel layout was not needed to decide it.

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| `halo_qsa_select_rows256`: 256-thread blocks; each radix pass's threshold digit found by one wave (8 digits a lane, shuffle prefix sum, ballot) instead of thread 0 walking up to 256 bins one dependent LDS read at a time; compaction by wave ballots + 8 per-wave totals (2 barriers an iteration) instead of Hillis-Steele scans (20 barriers each at 1024 threads). Same lists (ties, order); property test vs the reference rule (heavy ties, up to 70K blocks) | 52.3 -> 4-7 ms | 52.3 -> 10.3-10.9 ms (71 us a call); end to end inside noise (-24.7, spreads 90 / 127) | yes |
| `halo_attention_wmma_qsel_h2`: V loaded 4 bytes (2 dims) of one key a lane and instruction, 8 keys a task, packed by v_perm into the transposed tile (2 x 16-byte stores instead of 32 two-byte stores a thread); softmax reads its 8 scores as 2 x 16 bytes and stores its 8 p as one 16-byte store. Hypothesis: qsel_h LDS-throughput bound (~80% by HARDWARE.md rates) | -60..-105 (fallback -10..-25) | 386.5 -> 372.0 ms (-14.5); span: softmax 0.99 -> 0.60 us, staging 0.51 -> 0.63, others unchanged; ATT barrier 61 -> 64% | yes (small) |
| qsel_h3: h2 with 3 scoring waves (row tile x both key tiles, each Q fragment read once for 2 WMMAs) | -17..-27 | 387.2 ms (+15 vs h2) | no (removed) |
| Long chunks, exact: `--prefill-chunk 8192` with the `_l` / `_s` / router / k128 GEMMs on 2048-token slices (`--gemm-slice`, default 2048) and a chunk crossing position 2048 running its prefix on `fa_h` and the QSA loop from position 2048 (`--dense-prefix`, default on) | -300..-400 kernel at 8K | **-371.9 ms end to end, significant** (5329 -> 4957, 1537 -> 1653 tok/s); logits byte-identical to 2048-token chunks | flags kept; default chunk unchanged (+4.3 GB, below) |

*Why qsel_h2 missed:* the hypothesis that the kernel is LDS-throughput bound overall is refuted. Each phase is bound
by its own fragment stream and latency and the phases do not overlap within a block (barrier 64%); removing the
16-bit V stores did not shorten the staging (the 16 four-byte V loads with uniform per-key addresses replaced 4
16-byte ones). The remaining design that differs from qsel_w (wave-owned rows) and from these: warp specialization
across chunks (scorers holding Q in registers, P.V waves separate, scores(c+1) overlapping P.V(c), 2 barriers a
chunk, no Q tile in LDS: ~42.6 KB, 3 blocks a WGP). Not built: no skeleton probe yet, and the floor-style estimate
(~2x) is the kind that failed for qsel_w.

*Long chunks.* Unmodified, chunk 4096 / 8192 gave 5.39 / 5.34 s (no gain) and different logits: the MoE kernels
amortize per-pair work (gate/up -162, down -105, the q8r / q5_k variants -54 ms at 8192) and lr8 -52, but `_l`
+208, `_s` +120, router +80, k128 +36: their T x cols fp16 activations stop fitting the 32 MB MALL past 2048
tokens and are re-read per row tile from DRAM; and the first 2048 positions went through the QSA path. Slicing those
GEMMs keeps their per-token cost (per-token results do not depend on the slice) and the dense prefix keeps the flash
kernel for positions < 2048, so chunk 4096 and 8192 give logits byte-identical to main's 2048-token chunks (8K and a
3485-token prompt checked; new engine test `LongChunksMatchDenseRangeChunks`, negative control: with the prefix off
it fails). Kernel-timed rep at 8192: 5286 -> 4958 ms; residual losses `_l` +30, k128 +17, `_s` +14 after slicing.
Memory: chunk scratch ~0.7 MB a token: MemAvailable minimum during an 8K prefill 29.3 GB at chunk 2048, 25.0 GB at
8192 (+4.3 GB). Chunk 4096 (one run): 5.10 s.

Stream L total, against main 19d80bb (ab_bench, 4 rounds x 3 reps, pinned, whole-prefill device sum, 8192 tokens):
main 5329.2 ms (1537 tok/s); L defaults 5285.9 (-43.3, within noise; qsel_h2 + select256 -57 kernel); L with
`--prefill-chunk 8192` **4957.3 ms, -371.9, significant (1653 tok/s)**.
2048 tokens (same A/B): main 1249.6, L 1242.2 (-7.4), L with `--prefill-chunk 8192` 1243.2 (-6.4), both within noise;
logits byte-identical to `logits_main_5e5466e.bin`; gpu_tests 49 passed with the small-model fixtures.
Default clock, wall, 8K (6 reps over two processes each, after a warm prefill; other streams idle but the host
was not quiet): main median 5.39 s (min 5.34, 1534 tok/s), L chunk 8192 median 5.07 s (min 5.02, **1631 tok/s**),
above halogen's claimed 1584.

Remaining long-prompt levers, by size: (1) chunk 8192 as the default for long prompts (-372 ms at 8K, +4.3 GB; the
scratch is sized min(chunk, max context), so a 262K server pays it once); (2) qsel at 372 ms against a ~0.74 ms a
call WMMA floor (2.58 ms measured): warp specialization across chunks (above), prove with a skeleton first;
(3) `_l` / `_s` / k128 still +60 ms at chunk 8192 after slicing (a MALL-aware tile order for T > 2048 instead of
slices); (4) select256 at 71 us a call writes 4 MB of visible lists that only `union4` reads: fusing selection
into the union (bitmaps straight from the radix threshold) ~-8 ms; (5) decode: `halo_qsa_select` (1024 threads,
serial digit walk) runs 12 times a decode step, ~28 us each at 7K: the 256-thread kernel would save ~0.25 ms a
token (stream D2's area, not changed here).

## 12. 2026-10-06 (stream M, branch `stream/sessions-mtp`): batched decode (P6)

Design and memory tables: docs/SESSIONS.md. Pinned, graphs, 4 prompts (300-1,000 tokens), `halo_run --batch B
--device-sample 1`; every row byte-identical per entry to the session stepped alone (`--batch-check 1`).

| Change | Predicted | Measured (ms/step B=2 / 4 / 8) | Kept |
|---|---|---|---|
| v1: weight kernels with NB columns in-block (one weight read), grouped experts across sessions, per-session kernels once per entry | ~50 / ~75 / ~123 (dense x1.08/1.25/1.7, experts by measured union, 4.3 ms per entry) | 55.6 / 90.7 / 165.8 | yes (base) |
| Lone-kernel probe of the NB-column Q8 GEMV (`halo_gemv_bench`) | t(NB8)/t(1) 1.25-1.5 | 2.5-3.9 (VALU / latency at ~950 MHz) | (probe) |
| v2: norm scales, combine, top-k one launch for all entries (grid y); Q5_1 down converts weights once per expert; mixers one column per block; GEMV column groups (G 4) interleaved with row blocks | B8 -46, B4 -19 | 52.9 / 80.3 / 137.2 (-3 / -10 / -29): the Q5_1 down did not move (bandwidth by union, not dequant), mixers -6 | yes |
| GEMV column groups G 1 / 2 / 4 / 8 at B=8 | G 4 best by ~15% on the GEMVs | 146.4 / 139.9 / 138.3 / 153.2 | G 4 (B4: G 2) |
| GEMV weights prefetched a piece ahead (`_p`) | -3..-6 at B8 | +4.3 (B8), +3.0 (B4) | no |
| GEMV with 16 KB LDS (`_s`, more resident blocks) | -2..-5 | -2.9 (B8), within noise at B4 | option (`HALO_BATCH_GEMV_VAR=_s`), not default |
| Mixer columns interleaved with row blocks instead of grid z | -5..-10 at B8 | -0.3 | kept (harmless) |

Remaining at B=8 (eager): experts 46 (union bandwidth, expected), mixers 23, dense GEMVs ~35, per-entry ~20.

## 13. 2026-10-06 (stream C, branch `stream/fullctx`): full context

Pinned, real model. Prefill 32K 21.7 s (1,504 tok/s), 128K 95.6 s (1,369), 256K 201.7-203.4 s (1,288-1,300, fp16
KV). 256K kernel profile: QSA path 52.1 of 204 s (qsel_h2 21.5, qsa_scores_wmma 20.6, select_rows256 8.8, union4
1.3). Decode: 7K 38.6, 32K 39.2, 128K 42.0, 256K 44.5 ms/token; at 256K qsa_scores_b 3.6-3.9, select 2.7,
attention 1.7 ms.

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| Decode QSA select on `halo_qsa_select256` (`kDecodeSelect256`, same lists) | 7K -0.2; 128K -1.5..-3; 256K -3..-6 | -0.21 / -0.20 / -0.27 (128K) / -0.27 (256K): the 1024-thread select costs only 136-274 us a call | yes (default) |
| Decode qsa_scores_b on a full grid (one row a thread, `kDecodeScoresWide`) | 256K -1.8..-2.2 | +1.7 ms at 256K (327 -> 439 us a call) | no |
| `halo_qsa_scores_wmma_b`: block-key tile split once per block in LDS, q pre-split (bit-identical) | 256K -7..-11 s, 128K -1.5..-2.5 s | 128K prefill (pinned, 2 rounds): 91.65 / 92.33 -> 88.71 / 89.03 s (-2.9 / -3.3 s); logits bit-identical (128K `cmp`, 2K / 8K references, property test bitwise) | yes (default) |
| Engine KV page pool (released pages kept and remapped at another address; for the server's memory guard, not speed) | logits unchanged; server admits 160K after 128K | GPU aperture violation in the swarm at 2 MB pages, 2 of 3 runs (0 of 3 without): remapping a released handle is unsafe here. Replaced by Reset/Restore keeping exclusive pages in place (59752a2; exact, 68 tests at 2 MB and 4 KB) | no |

## 14. 2026-10-06 (stream L2, branch `stream/pageloc`): page locality at fixed bytes in flight (probe only)

`halo_hwprobe pageloc` (raw: `/data/halo-work/streams/L2/probe1.txt`; predictions written first in
`streams/L2/PREDICTIONS.md`). Every wave: 12 iterations of 2 KB in flight (4 x b128 a lane, ISA checked, no
prefetch), 24 KB a wave; K streams a wave-iteration (each a contiguous 2048/K-byte piece) spread over a Wb window;
48 KB LDS (40 resident blocks, as `gemv_q8r_r4`) unless noted; fresh lines every launch (rotation past the MALL);
hipMalloc buffer (lands in GTT, see D); clock pinned (2859-2900 MHz by the in-kernel clock probe before and after;
the table's own per-row `clk` column is invalid: the clock read is scheduled before the load wait). GB/s:

| K streams a wave (Wb = 24 KB, packed rows) | NB 16 | 40 | 80 | 168 | 400 | NB 80, all blocks resident (no LDS) |
|---|---:|---:|---:|---:|---:|---:|
| 1 (contiguous 2 KB) | 113 | 162 | 175 | 183 | 195 | 181 |
| 2 (rows 12 KB apart) | 134 | 120 | **137** | 155 | 182 | 125 |
| 4 (= `gemv_q8r_r4` pattern) | 158 | 167 | **195** | 215 | 231 | 152 |
| 8 | 162 | 153 | 189 | 220 | 233 | 134 |
| 16 (WMMA-like) | 161 | 152 | 181 | 207 | 223 | 159 |

| Window Wb (NB 80 / NB 400) | 4 KB | 24 KB | 64 KB | 2 MB | 32 MB |
|---|---:|---:|---:|---:|---:|
| K 2 | 223 / 239 | 137 / 184 | 137 / 183 | 136 / 183 | 137 / 179 |
| K 4 | **228** / 239 | 196 / 231 | 177 / 213 | 173 / 209 | 177 / 211 |
| K 16 | 225 / 235 | 179 / 222 | 158 / 191 | 154 / 189 | 152 / 184 |

- Regions padded apart, same bytes (NB 80, 24 KB a wave at 24 KB / 64 KB / 2 MB pitch: 640 waves in 640 distinct
  2 MB fragments at the last): K 1 176 / 172 / 168, K 4 196 / 194 / 192 (-2..-4%). (The NB 40 rows at 2 MB pitch,
  ~550-600 GB/s, reused lines from the MALL: invalid.)
- Memory type: a 192 MB hipMalloc taken first also landed in GTT (vram_used +0, gtt_used +334 MB): hipMalloc never
  returns the carve-out here. Same patterns: identical within 1% (the "vram" rows are GTT too, so this compares two
  GTT buffers, not VRAM; a carve-out allocation needs another allocator: not done).
- Engine-like shapes: dual qkv + z (rows 2560 B, K 4, 544 blocks) **240**; r4 2560 x 6144 (K 4, NB 80) **195**; the same
  r4 bytes as 160 blocks of 3072-byte rows (split-K 2 shape) **228**; 16 rows a wave at NB 80 182.
- Counters: rocprofv3 on gfx1151 exposes no UTCL1 / UTCL2 / TLB counters and no EA read latency (only
  GL2C_EA_RDREQ_* sizes, GL2C_MC_*, TA_BUSY); none collected.

**Conclusion.** The hypothesis as stated does not hold. (1) Translation / TLB reach is not the limiter: padding the
same bytes over 640 distinct 2 MB fragments costs 2-4%, the 2 MB and 32 MB windows equal the 64 KB one (no step at
2 MB or 32 MB), and the two buffers match. (2) GB/s does not fall monotonically with streams at fixed bytes: at the
GEMV's packing K 4 is the *fastest* (195 vs 175 for one contiguous stream, 137 for K 2), and K 16 is only 7% below K 4.
(3) What matters is how a wave's in-flight 2 KB is spread over DRAM: pieces inside one 4 KB window read 223-239 GB/s
(~peak) for every K and block count, while the same pieces spread over >= 64 KB lose 10-15% (K 4) to 15-20% (K 16),
and some strides alias badly (K 2 at any spacing >= 12 KB: -30%). Block count still matters (NB 80 -> 400: +18% at
K 4), and with every block resident (no LDS limit) NB 80 gets slower (152 vs 195), the residency direction D2 saw.
This matches HARDWARE.md (row locality, not translation) refined: locality within ~4 KB of the bytes a wave has in
flight at the same moment, not the number of rows.

**Implied GEMV change (to be A/B'd in-engine, not yet built):** for the 80-block 2560 x 6144 projections (5.05
ms/token, 162 GB/s in engine), keep 4 rows a wave and 40 resident blocks but (a) give each block half the columns
(split-K 2, 160 blocks, 3 KB per row a wave; probe 195 -> 228 GB/s, ~-0.7 ms/token if the engine follows the probe
ratio; not bit-exact unless the two partial sums are combined in the original order, which the lane-strided
order prevents -> a precision-class change), or, exact, (b) store each 4-row group's Q8 rows interleaved so that a
wave's two consecutive 2 KB load rounds fall in one 4 KB page (row r's 512-byte piece of round i at r x 1 KB +
(i mod 2) x 512 inside the page; same lane arithmetic, same order; probe 195 -> 228 GB/s): a weight relayout that the
prefill GEMMs must also read or a second copy (2560 x 6144 x 49 layers ~0.77 GB). Do not order a block's rows
further apart (the 64 KB+ windows lose) and do not raise rows a wave to 16 (no gain).

## 15. 2026-10-06 (stream M, branch `stream/sessions-mtp`): MTP speculative decode (P5)

docs/MTP.md. Pinned, graphs, real model + Q8_0 MTP sidecar, 9 prompts x 256 tokens (six GSM8K / HumanEval chat,
open-ended chat, raw code, raw doc); AR 26.8 tok/s (37.3 ms). Greedy output identical to AR in every row.

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| Trimmed 32K draft head (`SetDraftVocab`) | draft step 4.2 -> ~1.7 ms, greedy acceptance unchanged | 1.77 ms; greedy E[tokens/cycle] unchanged within noise; T 0.8 loses the mass outside the subset (doc d1 0.82 -> 0.69) | yes |
| Batch-shaped WMMA verify path (probe only, 4 designs) | X (B=1 step) 37-41 ms | X 60-66 ms: dense GEMVs 43-48 ms vs 21.5 today | no |
| SpecStep, fixed depth 3 (exact batched verify, snapshot rollback) | 1.16x raw greedy, ~1.50x paper chat | 1.22x total; paper chat mean ~1.38x; raw code / doc 0.97x | base |
| gufo's adaptive policy (ported) | ~depth 3 | 1.24x total (33.18 tok/s): backs off to AR for 16 tokens after a drop; raw text 1.03-1.05x | no (comparison only) |
| Verify cost model C[k] (k = 1..5) | 53.4 / 67 / 80.6 / 97 / 111 ms | 54.15 / 69.73 / 84.75 / 102.25 / 113.14 (+1-5%); catch-up 1.37 / accepted draft; rollback 1.46 | (calibration) |
| Measured-cost depth chooser (`MtpDepthChooser`, argmax E[tokens]/E[cost]) | >= best fixed depth (1.25x), each prompt within 2% of its best | **1.294x** (34.70 tok/s), +3.4% over d2, +0.8% over the per-prompt oracle, each prompt within 1.8%; raw text 1.12x | yes (`--spec-depth -2`) |

## 16. 2026-10-07 (stream MS, branch `stream/mtp-serve`): MTP in halo_serve

docs/MTP.md "Server integration". Pinned, chooser limit 4, 32K draft head; greedy output identical to AR throughout.

| Change | Predicted | Measured | Kept |
|---|---|---|---|
| Draft history window W (`MtpTrackFrom`), halo_run, 2K / 8K ctx, doc, code | W matters little past a short window; tokens/cycle at W 0 ~0.90 of full | ctx flat; doc / code 0.61-0.76 of full tokens/cycle at W 0, still 0.75 at W 2048 (8K doc); catch-up 1.33-1.42 ms a position | yes (W 128) |
| Serving, W 0 (eager catch-up none) | paper greedy 1.35x, T 0.8 1.20x, agent 1.20x; TTFT = AR | 1.21x / 1.22x / 1.27x (turn 2 1.37x); TTFT = AR +-3% | no (W 128 2% faster end to end) |
| Serving, W 128 | W 0 + 0-3% decode, +0.17 s TTFT cold | 1.26x / 1.23x / 1.24x (turn 2 1.35x); TTFT +0.13-0.18 s cold | yes (default) |
| MTP serving with decode graphs off (fault workaround) | fault gone, speed unchanged | speed unchanged (+-1%) but faulted in 1 of 9 runs: not the cause | no |

## 17. 2026-10-07 (stream SE, branch `stream/spill-crypt`): encrypted spill files (format v2)

Not a kernel change: the cost of AES-256-GCM on the SR4 spill path (docs/SESSIONS.md section 10). Real model,
default KV, `halo_run --spill-bench --spill-format 12` (v1 and v2 from the same prefill, restores interleaved),
unpinned, `/data` XFS on the SN5000 (Gen3 x4). Prediction (SE PREDICTIONS P1): within +-3% (GCM at 14.3 GB/s a core
replaces CRC32C at ~13 GB/s, 4 I/O threads, disk-bound).

| Change | Predicted | Measured (v1 -> v2) | Kept |
|---|---|---|---|
| v2: per-piece AES-256-GCM (VAES/AVX-512 path) instead of CRC32C, wrapped DEK, sealed meta | +-3% spill and restore | restore 2K 81.2 -> 79.7 ms (-1.8%), 12K 266.9 -> 267.1 ms (+0.1%), median of 3; spill (1 sample each) 2K 75 -> 68 ms, 12K 229 -> 213 ms, 128K 1984 -> 1962 ms and 1978 -> 2030 ms: noise, disk-bound. 128K restore not measured: `hipMemCreate` out of memory on main efaece7's own halo_run too (b0.txt), not SE's | yes (default; within noise) |
| Fix round: GCM context reset after every call, fail-closed key memory, checkpoint id, restore sync on failure (SE P2) | 12K restore +-2% of 267.1 ms | restore 2K v1 80.0 / v2 80.6 ms, 12K v1 268.7 / v2 266.6 ms (medians of 5, b2.txt); spill 12K v1 227 / v2 242 ms (1 sample each: disk noise) | yes |
