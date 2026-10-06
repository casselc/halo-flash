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

### Prefill vs stepping (P4), 2026-10-04

Chunked prefill must score like token-by-token stepping. The P4 prefill
replaces the decode kernels with FP16 WMMA GEMMs, grouped experts, split
DeltaNet and batched attention, so its float order differs. Scored with
`halo_score --prefill 1 --ngram-resident 1`. The stepping runs were resumed
across three invocations (`--start`); the per-case results are unaffected.

**Long set** (`qwen38-flash-alibaba-long`, 12 cases, prompts of 2K-24K
tokens, 766 target tokens):

| Mode | Build | NLL | API top-token agreement |
|---|---|---:|---:|
| stepping | fa47edf | 0.1227 | 97.26% |
| prefill, FP16 GEMMs | fa47edf | 0.12268 | 97.13% |
| prefill, before the FP16 route | 96d4f5a (inferred from the run time) | 0.12270 | 97.39% |
| prefill, FP16 GEMMs | 0ab257c | 0.12277 | 97.13% |

Paired by case against stepping:
- FP16 prefill at fa47edf: +0.0000 ± 0.0010 per case (standard error),
  4 cases up and 7 down, largest |difference| 0.0065.
- The earlier prefill: +0.0000 ± 0.0015, 7 up and 5 down.

No shift, at any prompt length up to 24K.

**Short set**, prefill at 0ab257c: NLL 0.285929, agreement 92.67% over 98
cases. The gate (NLL ≤ 0.305, ≥ 91%) passes, and this matches the stepping
builds above (0.2859-0.2864).

### Precision options (stream P), 2026-10-05

Changes that alter rounding, each behind a flag, scored against the unmodified build (45501ff
plus per-case output in `halo_score`) with `halo_score --prefill 1` (the n-gram table not resident:
the 79 GiB model plus its 28.8 GB left too little headroom next to the other stream's builds; the
table's values are the same either way). Two baseline runs are bit-identical (20 short cases), so
every difference below is the change itself. Gate, fixed before measuring: short set NLL ≤ 0.305,
agreement ≥ 91%; paired by case, short mean ΔNLL ≤ +0.001 and mean − 2·SE ≤ 0, agreement drop ≤ 0.3
pp; long mean ΔNLL ≤ +0.002, mean − 2·SE ≤ 0, no case worse by more than 0.02. Paired ΔNLL is the
mean over cases of the per-case NLL difference ± its standard error.

| Change (flag) | Short NLL / agreement | Short paired ΔNLL (up/down) | Long NLL / agreement | Long paired ΔNLL (up/down), worst case | Verdict |
|---|---|---|---|---|---|
| baseline | 0.286079 / 92.58% | — | 0.121277 / 97.26% | — | — |
| Router on plain fp16 WMMA (`--router-f16 1`) | 0.285933 / 92.62% | −0.00058 ± 0.00043 (45/53) | 0.123398 / 97.00% | **+0.00213 ± 0.00072** (9/3), +0.0074 | fails (long) |
| Router, weights split hi/lo, fp16 activations (`--router-f16 2`) | bit-identical to `--router-f16 1` on every case (the weights are exact in fp16, see below) | | | | fails (long) |
| Router, fp16 weights, fp32 activations split hi/lo (`--router-f16 3`) | 0.286079 / 92.55% | −0.00026 ± 0.00040 (48/50) | 0.121366 / 97.26% | +0.00009 ± 0.00057 (8/4), +0.0027 | **passes** (default) |
| DeltaNet alpha/beta + shared-expert gate on fp16 WMMA (`--small-f16 1`) | 0.285613 / 92.60% | −0.00038 ± 0.00041 (44/54) | 0.122459 / 96.87% | +0.00118 ± 0.00105 (6/6), +0.0089 | **passes** (default on) |
| Mix epilogue reads the norm's fp16 K-blocked xn (`--mix-h16 1`) | 0.285795 / 92.58% | −0.00024 ± 0.00041 (46/52) | 0.122384 / 97.52% | +0.00111 ± 0.00076 (7/5), +0.0046 | **passes** (default on) |
| Mixer down as W8A8 (`--down-w8a8 1`) | 0.285805 / 92.62% | −0.00029 ± 0.00048 (49/49) | 0.122210 / 97.65% | +0.00095 ± 0.00109 (7/5), +0.0120 | passes (not kept: no speed gain) |
| DeltaNet qkv / z written in fp16, read by conv, state advance, gated norm (`--gdn-io16 1`; paired against the kept set of a3ac848, short 0.286147 / 92.53%, long 0.121658 / 97.52%) | 0.286039 / 92.58% | −0.00043 ± 0.00094 (39/59; worst +0.074) | 0.121563 / 97.65% | −0.00010 ± 0.00077 (6/6), +0.0056 | passes (not kept: no speed gain) |
| Residual streams in fp16 (`--res16 1`; paired against the kept set of 8a18bf3, short 0.285504 / 92.64%, long 0.122071 / 97.00%) | 0.285031 / 92.47% | −0.00011 ± 0.00042 (49/49) | 0.121496 / 97.78% | −0.00058 ± 0.00068 (5/7), +0.0027 | **passes** (default on) |
| Residual streams in bf16 (`--res16 2`, same baseline; short set not run: fp16 measured the same speed) | — | — | 0.123549 / 97.26% | +0.00147 ± 0.00102 (9/3), +0.0063 | long within the rule, not chosen |
| DeltaNet scan with fp16 two-way splits (`--gdn-scan-h 1`; paired against the merged kept set, short 0.285504 / 92.64%, long 0.122071 / 97.00%) | 0.285501 / 92.51% | +0.00036 ± 0.00045 (47/51) | 0.122639 / 97.39% | +0.00056 ± 0.00091 (7/5), +0.0060 | **passes** (default on) |
| Shared-expert gate/up as W8A8 (`--shexp-w8a8 1`) | 0.285309 / 92.60% | −0.00081 ± 0.00093 (56/42) | 0.121921 / 97.26% | +0.00064 ± 0.00095 (7/5), +0.0058 | passes (kept, then turned off after merging stream E's `_s` tiles: no speed gain left) |
| DeltaNet fp16 data path, stream R (`--gdn-f16 1`: qkv / z GEMMs, conv output, prep / scan inputs, scan output and gated norm in fp16; paired against main 56693d8 defaults, scored as stream P's cumulative set: 6 long cases re-scored on this build matched it exactly) | 0.285022 / 92.60% | −0.00104 ± 0.00107 (45/53) | 0.122013 / 97.65% | +0.00034 ± 0.00054 (8/4), +0.0027 | **passes** (default on); against the original baseline: short −0.00102 ± 0.00106, long +0.00074 ± 0.00052 (9/3), worst +0.0043: passes |

**Cumulative, everything kept** (stream/precision after merging main: `--router-f16 3`, `--small-f16`, `--mix-h16`,
`--res16 1`, `--gdn-scan-h`), paired against the original baseline: short 0.286185 / 92.51% (paired
+0.000015 ± 0.000453, 56/42, agreement −0.07 pp), long 0.121673 / 96.87% (paired +0.00039 ± 0.00072, 6/6, worst case
+0.0053, agreement −0.39 pp): passes the same gate. Speed (`ab_bench.py`, 4 rounds, pinned, 2048 tokens, against main
4ad6410): 1433.2 -> 1295.1 ms, **-138.1 ms (-9.6%), significant** (1429 -> 1581 tok/s).

The F32 router, DeltaNet alpha/beta and shared-expert gate weights hold exact bf16 values (all of them),
and 99.95-99.97% are exact in fp16 (the rest have fp16-subnormal magnitudes): rounding these weights to fp16
loses (almost) nothing. The plain-fp16 router's long-set loss came from rounding the activations; keeping them
as an fp16 hi/lo pair (`--router-f16 3`) removes it (`/data/halo-work/streams/P/f16exact.py`).

### Chunked-prefill test outliers on the small model (stream F), 2026-10-06

`FlashNextGpu.ChunkedPrefillMatchesStepping` reported fp16-mode worst drift from stepping of 0.104 on main 19d80bb,
against 0.0033 / 0.0042 in earlier runs. Not a regression and not a bug: the worst is a tail. Forcing fp16 mode over
60 cases (213 checks): median 0.0023, 6 checks above 0.02, worst 0.134. Seven outliers (0.026-0.134) were replayed
deterministically and compared with the CPU reference (`model::ReferenceModel`, fp32); each is one top-k choice
near a tie that flipped in one engine:

| Outlier | Engine that flipped | Choice | Gap at the flip | Engine vs reference: as run / reference with that one choice swapped |
|---|---|---|---|---|
| fp16, 19 tokens | stepping | router, layer 7 | 1.3e-3 (logit 0.28) | 0.0256 / 0.0021 |
| fp16, 400 tokens | stepping | router, layer 5 | 3.6e-5 (logit 0.23) | 0.125 / 0.0028 |
| fp16, 152 tokens | stepping | QSA block, layer 7 | 0.44 (score 18.4; block score read back 0.70 low) | 0.134 / 0.0027 |
| fp16, 272 tokens | stepping | QSA block, layer 7 | 0.024 (score 34.9) | 0.073 / 0.0023 |
| fp16, 594 tokens | stepping | router, layer 4 | 6.3e-4 (logit 0.12) | 0.044 / 0.0021 |
| fp16, 6 tokens | fp16 prefill (`--router-f16 3`; 0.0025 with mode 0) | router, layer 3 | 1.4e-4 (logit 0.66) | 0.028 / 0.00056 |
| int8, 82 tokens | int8 prefill (float-activation oracle) | QSA block, layer 7 | 0.055 (score 20.7) | 0.086 / 0.0026 |

The stepped oracle flips because the test runs it with int8 expert activations (the decode default) in the fp16 and
int8 modes: with float expert activations stepping matches the reference to 2.7e-7 on every replayed case and the
fp16 prefill stays within 6e-4 of it (chunks of 1, 17 and 64 tokens, up to 594 tokens, across chunk boundaries).
Pre-stream c95db3e gives the same stepped outliers on the same cases. Positive control: one changed token at a chunk
boundary moves the chunked engine 0.07-0.11 from the reference, so the comparison sees a one-row error -- which is
also why a near-tie flip between the chunked and split engines (the int8 case) is indistinguishable from one by an
L2 bound alone. The test now prints how many checks exceed 0.01.

Real model: the mechanism (top-k routing and QSA selection amplifying small arithmetic differences) is the same, and
its cost is what the NLL gates above measure for int8 expert activations and `--router-f16 3`. No fp16 range
problem was involved (all six precision options off left the stepped outliers unchanged, and `--res16 0` alone left
the fp16-prefill flip in place), and no chunk-boundary or indexing error was found.
Tools: `/data/halo-work/streams/F/` (replay and CPU-flip harness as a patch, cases, logs).

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

Main 56693d8 and stream D (2026-10-06, `halo_run --device-sample 1`, steady median of 76 steps; long context:
`--prompt-file prose_long.txt --prefill-prompt 1`, 6,939 tokens prefilled first; ROOFLINE section 9):

| Build | 5-token context | 6,939-token context |
|---|---:|---:|
| main 56693d8, default clock | 43.5 ms (23.0 tok/s) | 55.4 ms (18.1 tok/s) |
| main 56693d8, clock pinned | 42.9 ms (23.3 tok/s) | 53.8 ms (18.6 tok/s) |
| + batched decode attention (`halo_attention_select_b`, bit-identical), pinned | 42.6 ms (23.5 tok/s) | 44.2 ms (22.6 tok/s) |
| Stream D2 (`--decode-opt`, all bit-identical; ROOFLINE section 10), pinned, final A/B vs main 19d80bb's 42.73 / 44.25 | **36.9 ms (27.1 tok/s)** | **39.0 ms (25.7 tok/s)** |

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

### The n-gram table on this machine's storage (2026-10-04)

The PLE n-gram table is IQ4_NL inside the first GGUF shard: 320M rows of
90 bytes, 28.8 GB. A 2048-token prompt touches 32,768 rows (16 heads),
28,559 distinct for gufo's synthetic prose. The shard lives on ZFS (128K
records, lz4, `direct=standard`), so a direct read of one 90-byte row
costs a record read; ARC never holds the table.

| n-gram rows for 2048 tokens | Host time before the device starts |
|---|---:|
| Direct reads, cold (fresh text) | 1,058-1,252 ms |
| Direct reads, warm (row cache, 16K entries) | 373-482 ms |
| Resident in RAM (`--ngram-resident 1`, 8 s to load) | 5 ms |

gufo's `model_bench` uses the same reader and the same file, so its pp2048
pays this too; its synthetic prose (96 words) repeats rows across its warm-up
and measured prompts. Comparisons below use its protocol
(`halo_run --prefill-file` with `synthetic_text(1)` after a
`synthetic_text(8888)` warm-up).

Open: with the table resident, device time per prefill rises by about 5%
(3,230 -> 3,405 ms, steady over 8 reps). Not transient compaction (flat over
reps), not reclaim (kswapd idle, 12 GB free), not allocation order (loading
it after the engine changes nothing).

| pp2048, gufo's prose protocol | tok/s |
|---|---:|
| direct n-gram reads | 567 |
| resident n-gram table | 600 |

### Dedicated NVMe vs ZFS (2026-10-04)

A WD Blue SN5000 4 TB as XFS (`noatime`) at `/data`, against ZFS `/home` (128 KiB records) on the
system drive. Both links train at PCIe Gen3 x4 (8 GT/s), though both drives and slots support
Gen4; that is a platform setting (candidate: BIOS `AMD CBS > NBIO Common Options > PSPP Policy`).

`halo_disk_bench`: random `O_DIRECT` reads over a 46 GiB shard. ZFS 2.4 Direct I/O was on; the ARC
stayed at 1.1 GiB.

| Block, reads in flight | ZFS IOPS | XFS IOPS | ZFS median | XFS median |
|---|---:|---:|---:|---:|
| 4K, 1 | 6,968 | 21,605 | 137 µs | 45 µs |
| 4K, 16 | 26,430 | **218,199** | 596 µs | 72 µs |
| 4K, 64 | 26,477 | 222,118 | 2,398 µs | 267 µs |
| 64K, 16 | 26,459 | 53,074 | 597 µs | 267 µs |
| 1M, 64 | 3,399 (3.56 GB/s) | 3,373 (3.54 GB/s) | 18.9 ms | 17.5 ms |

On ZFS a 4 KiB read costs a 128 KiB record: 26.4K IOPS x 128 KiB is the same 3.5 GB/s as the
1 MiB reads.

**In the engine** (`halo_run --prefill-bench 2048`, n-gram table read from disk, default clock):

| | ZFS | XFS |
|---|---:|---:|
| n-gram read, first prefill | 1,103 ms | 121 ms |
| n-gram read, later prefills | 374 ms | 38 ms |
| prefill end to end | 888 tok/s | 1,041 tok/s |
| model load | 28.8 s | 26.7 s |

The weights (sha256-verified against the HF ETags) live at
`/data/models/qwen3.8-flash-next/`. With them there, the resident table (28.8 GB of RAM) buys
about 32 ms per 2048 tokens.

### Prefill on gufo's protocol, resident n-gram table (P4, 2026-10-04, idle)

Each step's prediction was written down before measuring.

| Change | Predicted | Measured |
|---|---|---|
| (start) | | 600 tok/s, device 3.41 s |
| FP16-route dense GEMMs (Q8_0 dequantized while staging, 3 tile shapes), fused mixer up + mix, lane-split DeltaNet (first version) | 750-770 | 706 |
| DeltaNet split: inputs staged per 8 tokens, next stage prefetched; on by default | ~760 | **739** |

How each was decided (the measurement that misled first, then the one that decided):
- DeltaNet: a lone-kernel bench said 1.02-1.4x; it runs at ~950 MHz while the real workload holds ~2.7 GHz. An in-engine A/B of device time decided it: fused 2,954 ms, split 2,771 ms (the recurrence 389 -> ~206 ms). Register usage: the fused prefill kernel uses 256 VGPRs with 29 spills; the decode step kernel 256 with 153 spills (a decode lead).
- Dense GEMMs, per shape at T = 2048 (idle `halo_gemm_bench model`): FP16 route 25-30 TFLOPS on the large shapes against int8's 13-19 TOPS; the mixer's 320 x 10240 down projection 0.76 ms with all 320 rows in one tile (activations read once) against 1.85 ms in 64-row tiles. Best of the two routes per shape: 595 ms per prefill against int8's 1,026.
- Ablation (`HALO_PREFILL_SKIP`, device time with the kernels not launched): routed experts 604 ms, DeltaNet (fused) 389, attention 238, large dense GEMM tiles 194 (the synchronizing profiler said 443).

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
