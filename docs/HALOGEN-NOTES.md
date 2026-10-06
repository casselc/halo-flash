# Notes on halogen-flash-server (external reference, learn-only)

halogen-flash-server (peonist-ai) is a closed-source Qwen3.8-Flash-Next engine for the same
hardware. Its EULA allows use and does not prohibit reverse engineering, but asks that derived
source or kernels not be redistributed. **Nothing from it is copied into halo-flash.** These
notes record measured facts and design choices to test independently.

We inspected v0.16.2 statically on 2026-10-04: the container image's `flash_serve` binary, its
gfx1151 kernel metadata and disassembly, its tuned hipBLASLt plan, the README and CHANGELOG, and
the HF checkpoint headers. Nothing was executed. Working files are in
`/data/halogen-research` (not in the repo).

## Their numbers (their README; not comparable to our pp2048)
- **Prefill:** 1,584 tok/s on an 8K prompt, 1,567 at 32K, 1,517 at 131K. Cold, one
  32K-token call.
  - They publish no pp2048 figure. They report that a 1,024-token chunk costs about half the
    throughput.
- **Decode:** 37.6 tok/s serial, 46-56 with the draft head.
- **Setup:** ROCm 7.14, Linux 6.18.4+, about 85 W, median clock 2,229 MHz. IOMMU is off, which
  they credit with 13-16% of prefill.

## Per-GEMM times from their tuned plan
hipBLASLt bf16, M = 2048 tokens:

| Shape (rows x K) | Use | Their time | Ours (same role, 2026-10-04) |
|---|---|---:|---:|
| 320 x 10240 | HC down | 0.410 ms | `_m` 1.04 ms (fp16 WMMA over Q8_0) |
| 336 x 10240 | HC down + 4 inject rows, padded (inferred) | 0.483 ms | — |
| 10240 x 320 | HC up | 0.471 ms | `mix` 1.40 ms (includes the mix epilogue) |
| 10240 x 2560 | dense | 2.79 ms | — |
| 512 x 5120 | router; bf16 hi/lo split of fp32 (inferred) | 0.40 ms | `f32_tiled` 1.28 ms (512 x 2560) |

They spend about 0.9 ms per HC mixer in GEMMs; we spend about 4 ms (combine_norm 1.58,
`_m` 1.04, `mix` 1.40).

## Design choices to test (independently written)
1. **HC mixers as plain GEMMs at library speed.** Their combine + scatter + RMSNorm is one
   kernel and still writes a normalized bf16 copy. The win is in the GEMMs, not in avoiding
   that write; our kernel A prototype, which avoided the write, broke even (branch
   `proto/kernel-a`).
2. **Experts: weights global -> VGPR, never through LDS.** Only activations are staged in LDS.
   - Dequant goes through a per-group 16-entry bf16 lookup (scale x codebook, `v_perm_b32`).
   - `v_permlanex16` forms the B operand.
   - 8 waves at about 250 VGPRs, relying on ILP rather than occupancy.
   - This matches our finding that a sixth wave did not help gate/up.
3. **DeltaNet:** chunked, with the state in WMMA accumulators and fp16 mirrors of q/k/v. The
   L2 norm is folded into per-row scalars instead of writing normalized q/k.
4. **Router:** bf16 WMMA with fp32 accumulation, either on bf16-clean weights or on a hi/lo
   split.
5. **Their weights:** a Hadamard-rotated 4-bit trunk (QuIP#-style sign vectors), 6-bit HC
   weights, and one shared input rotation for all 512 experts. GGUFs are repacked at load.

## Kernels run in our harness (2026-10-05)

Their gfx1151 code object (`k000.co`) was loaded into our own host program with
`hipModuleLoadData` and its expert kernels were launched on synthetic inputs. No halogen binary,
script or service was run. The lab code and data are in `/data/halogen-research/lab`:
- `hlab.cpp` launches their kernels;
- `ours.cpp` launches our `core.hsaco` the way `halo_gemm_bench moe` does;
- `att_offline.cpp` decodes their thread traces;
- `run*.sh` are the scripts and `out/` holds the results.

**Setup for every number below:**
- T = 2048 tokens, top-10 of 512 experts, routing from random normal logits (as in `halo_gemm_bench moe`): all 512 experts used, at most 57 tokens per expert.
- Clock pinned at 2.9 GHz, each kernel run alone and back to back, median of 30 calls.
- Our in-engine times are about 0.8x these isolated, pinned times: gate/up 5.29 vs 6.7 ms, down 3.10 vs 3.69 ms.

### Which kernels, and their arguments
- **Kernels:** the v2 checkpoint's experts run `q4moe::k_q4moe<A, B, M, Tune<0,128>, C>(q4moe::Args)`.
  - A: 1 = bf16 WMMA, 2 = f16.
  - B: 0 = gate/up, 2 = down.
  - M: 64 or 128 tokens per tile.
  - C: epilogue. C = 1 uses the per-row output scale vectors and an in-register 128-point Hadamard (DPP butterflies plus `v_permlanex16`). C = 0 does not.
- **Args (152 bytes, read from the kernarg loads):**

  | Offset | Field |
  |---|---|
  | 0x00 | weight blob |
  | 0x08 | offset of the 4-bit codes (row-major) |
  | 0x10 | offset of the f16 scales (one per 128 weights) |
  | 0x18 | scale row stride |
  | 0x20 | 16 float codebook values |
  | 0x60, 0x68 | f16 scale vectors for the C = 1 epilogue |
  | 0x70 | activations: x [T][2560] bf16 for gate/up; h [P][640] for down |
  | 0x78 | pair ids sorted by expert; the kernel computes token = id / 10 |
  | 0x80 | items |
  | 0x88 | device item count |
  | 0x90 | output [P][640] or [P][2560] bf16, one row per sorted position |

- **Items:** 16 bytes each: expert, row block, start, count.
  - Gate/up has 5 row blocks per expert, each 128 gate rows plus the matching 128 up rows. Down has 10 blocks of 256 rows.
  - Grid = number of items, 256 threads. Workgroups past the device count exit.
- **Byte sizes:** these match the checkpoint exactly.
  - Gate/up: 512 x 1280 x (1280 + 40) = 865,075,200 bytes.
  - Down: 512 x 2560 x (320 + 10) = 432,537,600 bytes.
  - That is 4.125 bits per weight, against our Q4_K 4.5 (944 MB) and Q5_1 6.0 (503 MB).
- **Outputs:** every variant gave finite, non-zero outputs. Correctness against a reference was not checked; only the timing is used.

### How they read the weights
- **Layout:** one lane per weight row; 32 rows per wave (lanes 0-15 gate rows, lanes 16-31 the matching up rows); 256 rows per workgroup.
- **Per stage (one 128-weight group):**
  - Each lane loads its row's 64 contiguous bytes with 4 x `global_load_b128` (+0/16/32/48), plus one `global_load_u16` for the next group's scale.
  - That is 2 KB per wave and 16 KB per workgroup per stage, with a row stride of 1280 B (gate/up) or 320 B (down).
  - So this is the same row-strided 64-byte pattern as ours. No LDS is used for weights.
- **Prefetch:** the next stage's weights load at the top of each iteration (one stage of register double buffering).
  - The same iteration then waits with `s_waitcnt vmcnt(2)` for its activation loads.
  - vmcnt counts loads in order, so that wait also waits for the prefetched weights, and the effective prefetch distance is about zero.
  - Latency is covered by occupancy instead: 210 VGPRs, 34 KB LDS, giving 3 workgroups per WGP (6 waves per SIMD).
  - In the M = 64 gate/up loop, that wait is the most stalled instruction (13.2 M of 45 M stall cycles).
- **Dequant:**
  - Per group, 16 `v_fma_mix_f32` (scale x codebook value) build a 16-entry table.
  - Then `v_perm_b32` byte-selects bf16 values by nibble: 252 `v_perm_b32` per iteration.
  - `v_permlanex16` forms the B fragment.
- **Activations:** staged into LDS through VGPRs.

### Measured: theirs vs ours (isolated, same harness style)

| | Theirs (A=1, M=64, C=1) | Ours | Δ |
|---|---:|---:|---:|
| Gate/up time per call | 5.68-5.72 ms | 6.65-6.72 ms (q4_k) | -15% |
| ... weight bytes | 865 MB (152 GB/s) | 944 MB (141 GB/s) | -8% bytes |
| ... with weights cache-resident (every item uses expert 0/1) | 4.62 ms | 4.91 ms | -6% |
| ... added by streaming weights from DRAM | 1.1 ms | 1.8 ms | -0.7 ms |
| Down time per call | 2.96 ms | 3.69-3.70 ms (q5_1) | -20% |
| ... weight bytes | 433 MB (146 GB/s) | 503 MB (136 GB/s) | -14% bytes |
| ... with weights cache-resident | 2.57 ms | 2.73 ms | -6% |
| ... added by streaming weights from DRAM | 0.39 ms | 0.97 ms | -0.58 ms |

- **Other variants:**
  - Gate/up: M = 128 takes 6.15 ms and C = 0 takes 6.09 ms.
  - Down: C = 2 takes 2.74 ms and C = 0 takes 2.84 ms.
  - A = 2 (f16) gate/up takes 5.67 ms, the same as bf16.
- **Neither kernel is DRAM-bound at T = 2048.** With the weights cache-resident, theirs keeps 81% (gate/up) and 87% (down) of its time.
  - Gate/up issues about 19.5 M WMMA, about 2.7-3.3 ms of WMMA issue at 2.9 GHz.
  - On top of that, 357 M VALU wave-instructions (1.5-1.8 ms).
  - 16-token padding adds 19%: 7,625 16-token tiles cover 122,000 rows for 102,400 real ones.
- **The gap comes from three places.** The 6% compute difference is small; the large differences are:
  - how much weight streaming the kernel exposes on top of its compute (0.7 and 0.58 ms);
  - fewer bytes (8% and 14%);
  - in down, our barriers.
- **Counters per call (theirs / ours):**

  | Counter | Gate/up | Down |
  |---|---|---|
  | VALU | 357 M / 313 M | 207 M / 201 M |
  | LDS | 21.4 M / 22.7 M | 11.8 M / 24.1 M |
  | FLAT | 3.74 M / 6.23 M | 2.45 M / 6.68 M |
  | SALU | 10.2 M / 30.0 M | 6.1 M / 35.9 M |
  | Waves | 20.5 K / 33.3 K | 41 K / 133 K |
  | L2-to-fabric reads (x2) | 1.93 / 1.97 GB | 1.20 / 1.73 GB |

  - Their dequant is not cheaper in VALU. Their wider loads (four 16-byte loads per lane per stage) and exact item list cut FLAT, SALU and wave counts.
- **ATT (share of traced wave latency):**

  | Kernel | vmcnt | VALU | Barrier | WMMA | Other |
  |---|---:|---:|---:|---:|---|
  | Their gate/up | 37% | 43% (18% is `v_perm`) | 10.5% | 4.7% | |
  | Our gate/up | 40% | 31% | 11% | 7.5% | lgkmcnt 6% |
  | Their down | 30% | 49% | 10% | | |
  | Our down | 12% | 25-30% | **29-34%** | | lgkmcnt 9% |

- **Occupancy inside their own code:** their own 252/256-VGPR variants run 4 waves per SIMD instead of 6, and are slower by:
  - 7% on gate/up (M = 128 against M = 64);
  - 17% on down (M = 128 against M = 64, same C = 1).

  The M = 128 path differs in code too, so this comparison is confounded.
- **Item order:** listing items by row block instead of by expert (so the 5 row blocks that share an expert's tokens no longer run together) costs +15% (6.53 ms). Our grid already places them together: x varies fastest.

### What to try in our own design (predictions, to be checked in-engine)
1. **Gate/up occupancy.** Ours runs at 241 VGPRs (4 waves per SIMD); theirs at 210 (6 per SIMD).
   - Get our gate/up to 216 VGPRs or fewer, giving 3 workgroups per WGP.
   - Expected: most of the 0.7 ms per call of exposed streaming, so roughly -0.4 to -0.7 ms isolated, or about -15 to -25 ms per pp2048 over 47 calls.
   - Caveat: an earlier note says a sixth wave did not help gate/up. Decide it by ablation: re-measure the weights-cached gap after the change.
2. **Down: 256 rows per workgroup and an exact item list.**
   - Their down covers 256 rows over all of K = 640 (5 stages) with 8 waves, and launches only real items.
   - Ours: 128-row blocks, grid 20 x 833 with early exits, 3.3x the waves, and 29-34% of latency in barriers.
   - Expected: up to -0.6 to -0.7 ms per call isolated, about -25 ms per pp2048 over 43 calls.
3. **Exact work lists in general.** A device item count plus an (expert, row block, start, count) list, so no grid of mostly empty tiles. SALU was 3-6x lower, and the FLAT count 1.7-2.7x lower. Expected: small directly, a prerequisite for 2.
4. **A 4.125-bpw format.** 4-bit codes, one f16 scale per 128, a 16-value codebook in kernarg, dequant by building a LUT and byte-permuting.
   - It saves 8% (gate/up) and 14% (down) of bytes. Only the exposed-streaming part speeds up from that, so about -0.1 to -0.15 ms per call.
   - It is a requantization with a quality cost. They pair it with Hadamard rotation, which we do not have. Low priority.
5. **The WMMA floor (~3 ms per gate/up call at T = 2048) is the real limit for both engines.** Further gains need:
   - less 16-token padding (19% here);
   - fewer dequant VALU per weight per tile, e.g. taller token tiles per weight load when an expert has more than 64 tokens.

Not done in the time box: their DeltaNet chunked kernels and the HC combine.

**ATT note:** rocprofv3 7.2.3 (comgr/LLVM 21-22) cannot decode their code. It contains true16 `v_mov_b16`, and VOP3 two-operand ops with src2 = 0x80. Its traces silently stop at the first such instruction.

The workaround (in `lab/att_offline.cpp`):
1. Take the instruction listing from TheRock's LLVM 24 `llvm-objdump`.
2. Feed the raw `.att` files to `librocprof-trace-decoder` directly, with an ISA callback over that listing.
