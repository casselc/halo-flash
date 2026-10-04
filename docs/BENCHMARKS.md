# Benchmarks and quality gates

Each section records what was measured, against which weights and build, so
that numbers stay comparable. Weights are unsloth Qwen3.8-Flash-Next
UD-Q4_K_XL (HF revision 38bb39ee), four shards verified against the HF
sha256s.

## Quality

### Short set (P1/P2 gate), 2026-10-04

Teacher-forced scoring on ds4's `qwen38-flash-alibaba-100` fixtures
(`tools/score.cpp`, `halo_score`).
- **Rendering:** prompts are rendered as ds4 renders non-thinking Qwen3.8
  chat.
- **Agreement:** counted where local tokens align byte for byte with the
  API's.
- **Gate (fixed before measuring):** NLL ≤ 0.305 and API top-token
  agreement ≥ 91%.

| Engine | Kernels | Cases | Target tokens | NLL | API top-token agreement | Gate |
|---|---|---:|---:|---:|---:|---|
| halo-flash, HIP, eager | P2 (correctness) | 98 | 5,568 | **0.2859** | **92.46%** | pass |
| halo-flash, HIP, graphs | P3 GEMVs, float expert activations | 98 | 5,568 | 0.285956 | 92.47% | pass |
| halo-flash, HIP, graphs | P3 + int8 expert activations | 98 | 5,568 | 0.285943 | 92.67% | pass |
| halo-flash, HIP, graphs | P3 + int8 + fused mixer / shared expert / DeltaNet (66dae81) | 98 | 5,568 | 0.286366 | 92.47% | pass |
| ds4 Q4 on DGX Spark (ds4's own GGUF) | — | 100 | 5,696 | 0.2903 | 92.08% | reference |

Paired by case, P2 and the fused build differ by +0.0005 ± 0.0006 per case (standard error),
with 51 cases up and 47 down: noise from float reordering, not a systematic
shift.

ds4's NLL row also includes the two cases that have damaged emoji metadata.
Its agreement column uses the same 5,568 tokens as ours. ds4's Q4 is a
different quantization from UD-Q4_K_XL, so the comparison is a sanity check,
not an identity.

### Greedy identity with gufo

`halo_run --device hip` against `gufo prompt --raw -t 0 --speculative off`
(gufo 8bdde80), with the same weights.
- "The capital of France is", 24 tokens: **identical**, including one step
  with a top-2 logit margin of 0.069.
- token-trace sha256 (gufo):
  `6e29a188870a63130d14c840b32d7a59d561f81c612ba57e62031dadb98c33e2`.

## Speed

### Decode, real model (2026-10-04, idle machine, greedy, `halo_run --device-sample 1`)

Median of 60 steady steps after warm-up (the "Capital of France" prompt).

| Build | HIP | HRX |
|---|---:|---:|
| P2 kernels, eager | ~180 ms (5.6 tok/s) | — |
| First P3 GEMVs + graphs (019be74) | ~110 ms | — |
| Multi-row GEMVs, tuned experts (95245c0) | 51.1 ms (19.6 tok/s) | 93.7 ms |
| + int8 expert activations | 47.4 ms (21.1 tok/s) | — |
| + fused mixer, shared expert, DeltaNet block (66dae81) | **44.7 ms (22.4 tok/s)** | 64.0 ms (15.6 tok/s) |

The commit message of 66dae81 quotes 46.7 and 84 ms; those were measured
with a stale `halo_run` binary built before the fusions.

At this quant mix the roofline is about 26 ms per token (38 tok/s). gufo
reports 25.9 tok/s on the same machine (its own benchmark protocol, not yet
rerun here).

Measured 2026-10-03/04 while other jobs were running, so the absolute
numbers are noisy.

| Measurement | Value |
|---|---|
| Model load (79 GiB resident) | 24 s with the parallel O_DIRECT loader; > 15 min with per-tensor copies of mapped memory |
| Decode, P2 kernels, eager | ~5.6 steps/s; gufo `prompt` reported 5.2 tok/s in its own verbose output |
| Graph instantiation | 8,000 nodes: ~14 ms on HIP, ~2 ms on HRX |
| Graph replay per node (trivial kernels) | ~3 µs on HIP, ~5 µs on HRX (loaded machine) |

### gufo on this machine (2026-10-04, idle, same weights)

gufo 8bdde80, its own `model_bench` protocol (C1 only: the bench supports
one sequence), UD-Q4_K_XL as above.

| Test | gufo |
|---|---:|
| pp2048 | **1599.35 ± 2.24 tok/s** |
| tg128, AR | 26.53 ± 0.21 tok/s |
| tg128, MTP (after pp512 at 1009.47 tok/s) | 43.00 tok/s |

### Prefill, real model (P4, 2026-10-04, idle, `halo_run --prefill-bench 2048`)

2048 pseudo-random tokens (fixed seed) from a reset engine, median of 3,
HIP. Stepping a token at a time runs at about 22 tok/s.

| Build | tok/s | Where the time went (profiled) |
|---|---:|---|
| Stage A: dense projections as GEMMs, 256-token chunks (40f22de) | 64 | routed experts per token 62%, attention per token 25% |
| + routed experts grouped by expert, FP16 WMMA, 2048-token chunks | 124 | per-token attention 7.3 s of 16 |
| + attention batched on FP16 WMMA, online softmax | 223 | 5 layers with repacked Q8_0 down still per token |
| + grouped down over repacked Q8_0 | 363 | F32 router/alpha/beta/inject row GEMMs 1.1 s |
| + tiled FP32 GEMM (1db436f) | **407** | int8 GEMMs 1.0 s, grouped experts 1.4 s, DeltaNet 0.43 s |

gufo is at 1599. The dense int8 GEMMs (about 13.5 TFLOP per 2048 tokens)
run near 13 TOPS and the grouped experts near 5-7 TFLOPS; both have to
roughly double for parity.

### Bytes per decoded token at this quant mix

| Part | Size |
|---|---:|
| Dense, mostly Q8_0: `attn_qkv` 956 MiB, `ssm_out` 574, `attn_gate` 574, hyper-connection mixers 638, LM head 644, router (F32) 240 | ~4.5 GiB |
| Routed experts, 10 of 512 per layer (Q4_K gate/up, Q5_1 down) | ~1.4 GiB |
| **Total per token** | **~5.9 GiB** |

At the P0 read ceiling of 241 GB/s, that is about 26 ms per token, a decode
roofline of about 38 tok/s at this mix.

### Prefill GEMM (P4 kernel, 2026-10-04, idle machine, `halo_gemm_bench`)

Int8 WMMA over repacked Q8_0 weights, activations quantized per token and 32
values (`halo_quantize_q8_rows` + `halo_gemm_q8r_wmma_t`), timed together.
The register-only WMMA ceiling (`halo_gemm_bench peak`) is **56 TOPS int8,
51.5 TOPS fp16** on this gfx1151.

| Shape (rows x cols) | T = 64 | T = 512 | T = 2048 |
|---|---:|---:|---:|
| 10240 x 2560 (DeltaNet qkv) | 9.5 TOPS | 20.8 TOPS (397k tok/s) | 18.1 TOPS |
| 2560 x 6144 (DeltaNet out) | 6.8 | 13.1 | 12.6 |
| 320 x 10240 (mixer down) | 1.5 | 8.3 | 8.0 |

HRX within a few percent of HIP. About 36% of the int8 peak on the large
shape; gufo's dense projections reach about 55%. What was tried:
- per-wave tiles 32 x 32 to 64 x 32, two K blocks per stage, register
  prefetch of the next stage, LDS row padding, interleaved WMMAs: each
  within +-10%;
- FP16 WMMA with weights dequantized while staging (no per-block scale
  epilogue): 13 TOPS, slower;
- the int8 kernel with the scale epilogue removed entirely (wrong results,
  as a probe): 23 TOPS. The limit is staging (global to LDS and barriers),
  not the math; the next step is a deeper pipeline or direct-to-register
  operand loads.

### DeltaNet over a prompt chunk (P4 kernels, 2026-10-04, `halo_gemm_bench deltanet`)

One layer's conv + L2 and recurrence at the real sizes (16 q/k heads, 48 v
heads, d 128, kernel 4), HIP:

| T | T decode steps | chunked kernels | per token |
|---:|---:|---:|---:|
| 64 | 1.63 ms | 0.37 ms (4.5x) | 5.7 us |
| 512 | 13.0 ms | 2.96 ms (4.4x) | 5.8 us |
| 2048 | 51.2 ms | 12.0 ms (4.3x) | 5.8 us |

Bit-identical to stepping (outputs and both states). Across 36 DeltaNet
layers that is about 0.21 ms per prompt token, a ceiling near 4,800 tok/s
for this part alone; the recurrence is still serial in T with one block per
head (48 blocks), so the chunked parallel (WY) form is the next lever.

### BF16 prefill GEMM (Qwen3.5 torso, 2026-10-04, `halo_gemm_bench bf16`)

FP16 WMMA over BF16 weights, eight weight copies cycled so they come from
DRAM (the single-copy numbers were 2-4x better and misleading: the matrix
stayed in the cache).

| Kernel | 6144 x 2048, T = 16 | T = 64 | T = 256 |
|---|---:|---:|---:|
| 32-value K stages (`halo_gemm_bf16_f16`) | 937 us (27 GB/s) | 902 us | 1459 us (4.4 TFLOPS) |
| 128-value K stages, loads in flight (`_k128`) | 500 us (50 GB/s) | 617 us | 641 us (10.1 TFLOPS) |

Split-K over 4 was slower in every case. Short chunks remain latency-bound.
A first skinny kernel (VALU, 4 rows per wave, 16 tokens' activations staged
in LDS 256 columns at a time) read only 44 GB/s at T = 16: the staging and
two barriers per 256 columns leave too few weight loads in flight. Not
adopted; holding activations in registers for T <= 8 with more rows per
wave (as llama.cpp's small-batch kernels do) is the next design.

### HRX vs HIP decode gap: what it is not (2026-10-04, idle machine)

Decode is 44.8 ms/token on HIP and 63.7 ms on HRX. Measured:
- **Not graphs.** Eager and graph replay are within 2 ms on both backends
  (HIP 44.82 vs 44.84; HRX 65.8 eager vs 63.7 graph).
- **Not dispatch overhead.** `halo_graph_bench`: 1.8 us per node on HIP,
  3.2 on HRX; at 1,145 dispatches per token that is 1.6 ms of the ~19 ms.
- **Kernels run slower under HRX**, unevenly (`halo_gemv_bench`, HBM-resident
  data):

  | Kernel | HIP | HRX |
  |---|---:|---:|
  | `gemv_q8r_r4`, 10240 x 2560 | 222 GB/s | 147 GB/s |
  | `gemv_q8r_r4`, LM head 248320 x 2560 | 242 GB/s | 160 GB/s |
  | `moe_gate_up_q4_k_dot`, 10 experts | 161 GB/s | 159 GB/s |
  | `moe_down_q5_1_r4`, 10 experts | 180 GB/s | **28 GB/s** |
  | `moe_down_q5_1`, 10 experts | 121 GB/s | **10 GB/s** |

- **Not memory placement or the TLB:** with the same 10 experts every
  launch (cache-hot), `moe_down_q5_1_r4` is still 28 GB/s on HRX and 180 on
  HIP. No kernel uses scratch; the slow kernels' metadata is unremarkable
  (89 VGPRs, 8 KB LDS).
- rocprofv3 does not see HRX's dispatches (libhrx opens its own HSA
  queues), so per-kernel HRX timing needs HRX's own profiling.

Open: something in how HRX dispatches these kernels (kernel descriptor
handling, LDS or kernarg setup, CU masking) differs from ROCm's. HIP stays
the default backend until it is found.
