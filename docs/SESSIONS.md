# P6: sessions, paged KV, batched decode (stream M, 2026-10-06)

Working files, raw logs and every prediction (written before its run): `/data/halo-work/streams/M/`
(`PREDICTIONS.md`). All numbers: HIP, ROCm 7.2.3, clock pinned (`tools/gpu_clock.sh pin`) unless stated, real
model UD-Q4_K_XL.

## 1. What a session is

The engine (`FlashNextGpu`) holds any number of sequences. One is **bound**; `Step`, `StepSample`, `Prefill`,
`Save` / `Restore` / `Reset` and `Position` act on it, exactly as before (session 0 exists from `Create`, so
single-sequence callers see no change). A session owns:

| State | Size | Kept as |
|---|---:|---|
| DeltaNet recurrent state, 36 layers x 48 heads x 128 x 128 fp32 | 113.2 MB | one allocation per layer |
| DeltaNet conv state, 36 x 3 x 10240 fp32 | 4.4 MB | one allocation per layer |
| PLE conv history | 0.4 MB | one allocation |
| n-gram history, position, QSA block counts, fp16-copy validity | host | in the session object |
| KV per attention layer: K, V fp32; indexer keys fp32; pooled QSA blocks; fp16 K/V copy | 81.4 KB / token | paged regions (section 2) |
| its captured decode graphs (they bake the state's addresses) | host / driver | in the session object |

Binding swaps host-side handles only (`SwapBound`): no device copy, 0.000 ms measured. Per-token scratch
(activations, the prefill chunk buffers `ck_`, ~4.3 GB at the 8192-token long-prompt chunk) belongs to the
engine and is shared by all sessions.

API (`src/engine/flash_next_gpu.hpp`): `NewSession`, `ForkSession(src)`, `BindSession`, `DropSession`,
`SessionPositionOf`, `SessionMemory`, `StepBatch`.

## 2. KV storage: virtual ranges, pages on demand, copy-on-write forks

Each KV tensor of each attention layer is a **region**: a reserved virtual range sized for `max_context`
positions, into which 2 MB physical pages are mapped as the sequence grows (HIP virtual memory management:
`hipMemAddressReserve`, `hipMemCreate`, `hipMemMap`). Kernels see one contiguous buffer per tensor, so no kernel
changed and every result is byte-identical by construction (decode and prefill logits equal main's).

- **On demand.** `EnsureKv(tokens)` maps pages for the positions a step or chunk writes plus 512 positions of
  slack (tiled kernels may load rows a tile past the last valid one). Memory follows the context actually used,
  not `max_context`: a 256K-capable session that holds 2K tokens holds 2K tokens' pages.
- **Forks share pages.** A fork maps the parent's physical pages below the parent's immutable bound (positions
  written; complete QSA blocks; fp16 rows already converted) into its own range, refcounted; the page holding
  the bound and anything mapped above it are copied. Sessions only append, so neither side ever writes a shared
  page: no fault handling is needed. Reset and Restore rewrite positions from 0, so they first unmap the bound
  session's pages that a fork also holds (stream C: before that fix a reset parent wrote into pages its forks
  shared; regression test `Sessions.ForkSurvivesParentResetAndRestore`). Pages the session alone holds stay mapped
  and are rewritten in place (`Sessions.ResetKeepsExclusivePagesExactly`): released pages stay charged to GTT by
  the HIP runtime, so releasing them made the server refuse prompts it could hold (section 7).
- **Pluggable layout.** A region is `(row_bytes, row_tokens, pages)`; today's kinds are K / V (2048 B a token),
  indexer keys (512 B), pooled blocks (512 B per 4 tokens), fp16 K / V (1024 B). An fp16-only KV layout (stream
  for full context) is a different kind table with the same paging.
- **Fallback.** A backend without page mapping (HRX today; or `HALO_NO_PAGING=1`) allocates each region whole
  at `max_context` and a fork copies the written rows.
- `HALO_PAGE_BYTES` overrides the page size (tests use 64 KB so short sequences span several pages).
- **Mapping changes settle before use.** On this stack the GPU sees a page mapping tens of microseconds after
  `hipMemMap` / `hipMemSetAccess` / `hipMemUnmap` return (a write right after a remap reached the old page; right
  after a fresh map it faulted: section 11). `rt::HipDevice` holds the next launch, fill or copy back until 1 ms
  after the last map or unmap (`HALO_VMM_SETTLE_US`); callers batch their maps before their writes.

**Measured** (VMM probe, `streams/M/probe/vmm.hip`): granule 4 KB; a 2 MB page costs 84 us to create and
46 us to map the first time, 7.3 us to map again (sharing); reads through a shared mapping run at 217.6 GB/s
(hipMalloc: 216.8).

### Per-session memory and costs (`halo_run --session-bench N`, prose prefilled)

| Context | Recurrent | KV mapped | Fork | Fork's own memory (pages it does not share) | Bind | Snapshot Save / Restore (whole copy) | Drop 2 forks |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 118.0 MB | 302 MB | 16.8 ms | 118 + 151 MB | 0 | 3.8 / 2.3 ms | 12.4 ms |
| 8,192 | 118.0 MB | 805 MB | 18.9 ms | 118 + 151 MB | 0 | 6.8 / 5.4 ms | 17.7 ms |
| 32,768 | 118.0 MB | 2,819 MB | 25.9-28.7 ms | 118 + 151 MB | 0 | 18.9 / 17.4 ms | 37.4 ms |

The fork's 151 MB exclusive is the partly filled boundary page of each of the 72 regions (pooled-block pages
cover 16K tokens, indexer pages 4K) plus the slack: a per-fork constant, not per token. Smaller pages for the
small-row kinds would cut it (open item). Restore of any shared prefix is a fork: **< 30 ms at 32K** (target
< 1 s). Decode on a 32K session: 40 ms/token (37 at short context).

### What fits (GTT 104.9 GB, model ~85 GB; 4.3 GB long-prompt chunk scratch reserved; ~14.6 GB for sessions)

Per session = 118 MB recurrent + KV per token x (context + 512) + ~70 MB page rounding.

| Context | Today's KV (81.4 KB/token) | Sessions that fit | fp16-only KV (~32 KB/token: fp16 K/V 24.6 KB + fp32 indexer 6.1 KB + blocks 1.5 KB) | Sessions that fit |
|---:|---:|---:|---:|---:|
| 2K | 0.42 GB (measured) | 34 | 0.25 GB | 58 |
| 8K | 0.92 GB (measured) | 15 | 0.45 GB | 32 |
| 32K | 2.94 GB (measured) | 4-5 | 1.24 GB | 11 |
| 128K | 10.9 GB | 1 | 4.4 GB | 3 |
| 256K (262,144) | 21.5 GB | **0** (exceeds the ~19.9 GB left even without chunk scratch) | 8.6 GB | **1** (2 without the 8192 chunk scratch) |

Shared prefixes change the table: N forks of one 32K conversation cost 2.94 GB + N x 0.27 GB + their own new
tokens (8 forks: ~5.1 GB instead of 23.5 GB).

## 3. Batched decode

`StepBatch` steps up to 8 sessions at once. **Contract: every entry's logits, sampled token, RNG and state are
byte-identical to that session stepped alone, and the single-session path stays byte-identical to main.** The
second clause fixes the arithmetic: a batched kernel must give each column the single-token kernel's fp32 FMA
chain in the same order. So batching shares weight *loads*, never changes a sum (no WMMA, no reordering):
- Weight kernels (dense GEMVs, the DeltaNet qkv+z dual, alpha/beta, router + int8 activations, both mixer
  kernels, the shared expert, the LM head): one launch reads each weight once and computes G columns from it
  (G = 4 default); column groups are interleaved with the row blocks so neighbouring blocks share rows.
- Routed experts: `halo_moe_bucket_b` builds the union of the batch's experts on the device; the grouped gate/up
  and down kernels read each expert once and apply it to every (session, slot) pair routed to it.
- Per-session work (norm scales, combines, top-k: one launch for all entries on grid y; DeltaNet conv and
  step, the attention chain, PLE, embedding, sampling, the Q5_K-expert layer: the single kernels per entry, on
  the entry's slice, its session bound while recording).
- Graphs are captured per session list (they bake each session's state addresses) and cached, LRU 16.

### Prediction vs measurement (pinned, graphs, 4 prompts, contexts 300-1,000 tokens)

Expert union per layer, measured with `halo_run --route-trace` over 2,391 decode steps of four texts
(`union.py`): random sessions B=2/4/8 touch 18.8 / 34.7 / 59.7 distinct experts per layer (independent-routing
model: 19.8 / 38.8 / 74.7); consecutive positions of one sequence (an MTP verify) 15.8 / 24.7 / 38.0.

| B | Predicted v1 | v1 measured | v2/v3 measured | tok/s aggregate | vs B=1 | gufo C-B (its own protocol) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.9 | 37.0 | 37.0 | 27.0 | 1.00x | 26.0 (C1) |
| 2 | ~50 | 55.6 | 52.9 | 37.8 | 1.40x | 43.4 |
| 4 | ~75 | 90.7 | 80.3 | 49.8 | 1.84x | 69.4 |
| 8 | ~123 | 165.8 | 134.3 (`_s` GEMVs; 137.2 default) | 59.6 | 2.20x | 101.6 |

Exactness: real model, every entry-step byte-identical to the session stepped alone at B=3 (36/36), B=4
(96/96), B=8 (128/128 and 96/96 after the merge); small model: B=2..8, logits and device sampling, every group
size and kernel variant; the session swarm (below).

Where B=8 goes (eager kernel timing, ms/step, B=1 -> B=8): routed experts 8.1 -> 46.4 (union x6.0: bandwidth,
~190 GB/s, the expected cost); mixers 5.3 -> 22.8; DeltaNet qkv+z 7.4 -> 12.7; 2560 x 6144 outputs
5.2 -> 11.8; LM head 2.9 -> 4.8; per-entry work (DeltaNet step 7.3, attention chain ~8, conv 2, embedding /
n-gram host 2.7) ~0.9 -> ~20.

### Why it is below the prediction and below gufo's batching

1. **Exact columns cost VALU per column.** Every column needs one fp32 FMA per weight (4.9 G for the dense
   weights) plus LDS reads; lone-kernel ratios at NB = 8 were 2.5-3.9x (bench at ~950 MHz) and the in-engine
   kernels still cost 1.7-2.3x their single-token time at B = 8. Variants that change only scheduling (column
   groups 1/2/4/8, weight prefetch a piece ahead, 16 KB LDS for more resident blocks, mixer columns on separate
   blocks) moved B=8 by <= 3 ms. Measured marginal cost: ~8 ms per extra column at B=2 besides the experts,
   against an FMA-throughput floor of ~1.3 ms.
2. **Small latency-bound kernels scale with the columns.** The mixers (3.6 MB, 28 us at B=1) do not get faster
   per column when their columns share a launch or run on separate blocks; their cost tracks the work per block.
3. **gufo's "exact C2-C8"** is batch-invariance within gufo's own batched kernels (its single request runs the
   same kernels). Our gate is stricter: equality with main's specialized single-token kernels. Reaching
   gufo-class batching probably means running the single path through batch-shaped kernels (WMMA or
   reordered sums), a numerics change for the single path that would need the precision gate and the user's
   decision.

### Relaxing exactness for B >= 2 (user decision, 2026-10-11)

The contract above (a batched entry byte-identical to the session stepped alone) is what keeps B = 8 at 59.6 tok/s
against gufo's 101.6. Measured before relaxing it (`halo_score`, main 2aa35f5, ds4 alibaba fixtures; streams/BQ):
- **Batch-shaped arithmetic costs no quality.** Scoring every continuation token as a one-token Prefill (the chunk
  path's batch-shaped GEMMs) instead of a decode step changes every case's sum and passes the gate: short dNLL
  -0.00088 +- 0.00035 (0.2850 -> 0.2841, agreement 92.60 -> 92.65%), long -0.00020 +- 0.00052; the greedy argmax
  changes at 0.79% (short) and 0.65% (long) of positions, near-ties both ways.
- **The single path keeps its kernels.** A batch-shaped B = 1 step measured 60-66 ms against 37 (ROOFLINE, "batch-shaped
  WMMA verify path"), so only B >= 2 takes relaxed kernels. Consequence: a session's greedy output may differ (at about
  that rate) depending on whether it shared a batch.
- **The gate for relaxed kernels:** `halo_score --prefill 1 --batch 8` scores the fixtures through StepBatch. On
  today's exact kernels it reproduces the single-token scores case for case (98/98, 12/12), the control; a relaxed
  kernel must pass `streams/P/compare.py` against the single-token scores.

### Relaxed dense kernels (B >= 2, default; `--batch-relaxed 0` for exact)
StepBatch with 2-8 independent sessions (never an MTP verify, whose greedy equality with stepping needs exact rows)
runs the dense Q8 projections (DeltaNet qkv + z, outputs, attention q / k / v / out, the LM head) as int8-activation
v_dot4 GEMVs: each column's activations quantized per 32 values in two int8 levels (x ~ q s + q2 s2,
`halo_q8x_cols`), a wave a row, 4 columns a launch group (the B = 8 sweep's best for every shape). Measured (pinned,
4 prompts, device sampling; streams/BD):

| B | exact ms/step | relaxed ms/step | tok/s aggregate (exact -> relaxed) |
|---:|---:|---:|---:|
| 2 | 53.48 | 50.84 | 37.4 -> 39.3 |
| 4 | 82.80 | 75.80 | 48.3 -> 52.8 |
| 8 | 136.94 | 123.46 | 58.4 -> 64.8 |

B = 8 dense kernels 33.1 -> ~18 ms (DeltaNet dual 12.5 -> 7.0, 2560 x 6144 outputs 11.9 -> 6.1, LM head 4.8 -> 2.9).
Quality (`halo_score --prefill 1 --batch 8` against the single-token scores): short dNLL -0.00012 +- 0.00037,
agreement +0.07 pp; long +0.00074 +- 0.00048; the greedy argmax changes at 0.79% / 0.52% of positions, the floor any
non-identical arithmetic showed (the one-token Prefill proxy: 0.79%). One int8 level was 2 ms faster at B = 8 but
lost 0.27 pp of agreement (short dNLL +0.0004): two levels are the default (`HALO_RELAXED_RESID=0`: one). Keeping the
LM head exact recovered nothing (argmax 0.95%) for +2.8 ms. Not yet relaxed: the mixers (22 ms at B = 8), the shared
expert and router (7.5), the routed experts (51, bandwidth on the union), per-entry work (~20).

**Relaxed mixers (B >= 4).** The hyper-connection mixers' down (320 x 10240) and up (4 x 2560 x 320) projections:
per column once, the norm and its two-level quantization (`halo_hc_norm_q8x`), the down projection as a split-K
int8 GEMV (7 parts, the dense kernel with `chunk`), the inject rows in fp32 (`halo_hc_inject_dot`), SiLU of the
parts' sum quantized (`halo_hc_ls_q8`), and the up projection with the gated combine (`halo_hc_mix_up_i8`). The
first try kept the exact kernels' shape (each row block normalizing and quantizing its whole chunk in 38 KB of
LDS: one block a CU) and gained 3 ms; per-column work done once gained 9.7. B=8 123.46 -> 113.73 ms (70.3 tok/s),
B=4 76.36 -> 72.54; B=2 none (not used there). Quality: short dNLL -0.00039, agreement +0.00 pp; long -0.00031;
argmax 0.86% / 0.39%. Left: the inject rows (17 us a call, latency-bound) and the up projection (~60 GB/s).

**Per-session launches merged over the entries (exact and relaxed).** Attention's small per-session kernels (q / gate
split, the q / k / indexer / pooled norms and ropes, the K / V / indexer / block appends, the indexer projections,
pooling, QSA scores and selection, attention itself) and the DeltaNet conv ran once per entry, a few blocks each.
Each now runs once a layer for all entries (`_y` kernels: grid.y = entry), the single kernel's body unchanged on
entry e's scratch slice and step parameters, its KV cache and conv state from a per-entry pointer table (`bb_.tab`,
uploaded when the session list changes). Byte-identical; not for a verify (its entries are one session's
consecutive rows). `HALO_BATCH_MERGE=0` for the per-entry launches. B=8 113.83 -> 107.35 ms (74.5 tok/s), B=4
72.51 -> 69.69, B=2 51.70 -> 50.66. Left per entry: the DeltaNet step (bandwidth-bound per session) and sampling.

## 4. Tests

- `Sessions.SwarmMatchesFreshReplay` (Hegel, swarm: each run enables a random subset of the rules): new / fork /
  bind / step / prefill / StepBatch over a random subset in random order / drop / reset / graphs on-off; after
  every step, prefill and batch entry the session's logits are compared with `memcmp` against a fresh engine
  replaying that session's own operations. 60 cases, at 2 MB and 64 KB pages.
- `Sessions.ForkedSessionsDivergeIndependently`: 300-token prefix, fork, both diverge past the shared pages;
  memory accounting shows shared pages (64 KB pages: 14.0 MB mapped, 10.4 MB exclusive).
- `Sessions.BatchedStepsEqualSingleSteps`: B = 2..8, logits and device sampling (temperature 0.8, top-p 0.9),
  12 steps, each entry against the same session on a second engine.
- `Sessions.ForkSurvivesParentResetAndRestore`, `Sessions.ResetKeepsExclusivePagesExactly` (stream C, 4 KB pages):
  a parent reset or restored under a fork, and a reset over kept dirty pages, each against a fresh engine bitwise.
- gpu_tests 62 pass with `HALO_SMALL_QUANT_DIR` (68 at stream C's end, at both 2 MB and 4 KB pages); decode logits (66 steps, short and 7K context) and prefill logits
  (2K vs `logits_main_5e5466e.bin`, 8K vs `l8k_ref.bin`) byte-identical to main after every change.

## 5. Open items

- Per-entry kernels in one launch (DeltaNet conv / step, attention chain) through a per-graph state-pointer table:
  ~-8 ms at B=8.
- Batched dense kernels: the 1.7-2.3x per-column cost is the lever for both batching and MTP verification
  (section 6). Next probe: why a 16-FMA chain per 16 weights runs at ~1/3 of the VALU rate (ISA shows the
  int8 conversions shared and few dual-issue FMAs).
- Fork's 151 MB boundary copies: smaller pages for the pooled-block and indexer regions.
- Radix prefix cache (PLAN addendum item 1) on top of forks: checkpoints = parked forks; disk tier.
- Prefill chunks interleaved with active decodes; server / C ABI session binding.

## 6. Consequence for P5 (MTP)

Verifying m positions of one sequence is a batch of m columns with consecutive routing (union 15.8 / 20.6 /
24.7 / 28.3 experts per layer for m = 2..5). With today's batched kernels: verify(4) ~ 37 + 3 x ~11.5 ~ 72 ms;
three drafts ~ 3 x 4.5 ms (MTP layer + LM head); at an assumed 0.7 acceptance (2.53 tokens a cycle) that is
~34 ms/token, **~1.1x AR**. The bytes-only floor (experts' union ~3.1 ms per added position + ~1.3 ms dense FMA)
would give ~1.5x. CODESIGN's table (1.5x "today", 1.16x after the dense byte diet) assumed bandwidth-bound
verification; the measured per-column cost, not bytes, decides it here. Acceptance must be measured first.

## 7. Full context (stream C, 2026-10-06; working files `/data/halo-work/streams/C/`)

**KV kinds.** Every region carries its kind (`KvKind`: K, V, indexer keys, pooled blocks, fp16 K, fp16 V) and
layer; forks, snapshots and paging go by the kind table. `--kv-f16 1` (`kv_f16_only_`, default off; `halo_serve` turns it on above 150000 positions, see below) drops the
fp32 K/V: decode attention reads fp16 rows (`halo_attention_select_bh`, bit-identical to the fp32 kernel on the
rounded cache), prefill appends fp16 rows directly; prefill logits are bit-identical to the default kinds.
Raw indexer keys are read only while their QSA block is incomplete, so pages wholly below the current block
are released (`kv_index_window_`, default on, exact; `HALO_KV_INDEX_WINDOW=0` disables).

**Measured** (SessionMemory after prefill; GTT limit 104.86 GB, 83.27 GB after load; the 8192-token chunk
buffers and the rest ~6.3 GB):

| Context | Default KV (window off / on) | fp16-only KV (window off / on) | GTT used after prefill |
|---:|---:|---:|---:|
| 32,768 | 2.82 GB / — | 1.16 GB / — | 92.5 (default) |
| 131,072 | 10.67 / 9.92 GB | 4.23 / 3.47 GB | 100.4 (default), 93.9 (fp16) |
| 262,144 | does not fit (~19.7 GB) | 8.48 GB / ~6.9 GB | 98.2 (fp16) |

Per token: default 81.4 KB (75.3 with the window), fp16-only 32.3 KB (26.2). One session fits up to ~185-200K
positions with default KV; 262,144 needs fp16-only KV. Also fixed: snapshots were sized for max_context
(~20 GB each at 256K; now sized at Save), the block-score scratch grows with the position, the decode
attention scores are sized by the visible stride. `halo_serve` refuses a prompt the memory left cannot hold
(`KvRoomFor`) before prefilling it.

Quality of `--kv-f16 1` (docs/BENCHMARKS.md gate): passes vs main defaults and vs the original baseline; on a
128K-token document its NLL equals the default kinds' (paired -0.0007 +- 0.0030). Needles, NLL by window and the
split / fork checks: docs/BENCHMARKS.md "Long context". Raw results: `/data/halo-work/streams/C/`.

**Serving end to end** (`halo_serve --max-context 262144`, one request at a time, greedy, thinking off;
`streams/C/serve/`):

| KV | Request sequence | Result |
|---|---|---|
| fp16-only | 261,499-token needle prompt, then a follow-up turn | correct; turn 1 200.1 s end to end; turn 2 reused 261,506 cached tokens, 0.79 s; GTT max 98.2 GB, MemAvailable min 25.3 GB |
| default | fresh server: 192K, 208K, 224K, 256K | each refused before prefill in < 0.2 s (192K: 14.8 GB KV + 5.9 GB chunk buffers + 2 GB reserve > 21.6 GB free); server healthy after each |
| default | 128K, then 160K, 176K (before the fix below) | 128K correct (94.5 s); 160K and 176K refused ("4.4 GB available") |
| default | 128K, 160K, 176K, 128K (after the fix) | 128K correct (93.8 s); 160K refused by 0.1 GB (needs 2.5 GB more, was 12.4; + 2 GB reserve vs 4.4 free); 176K refused; 128K again correct over the kept pages (95.4 s) |

**Largest reliable context per KV kind** (one session, the 8,192-token prefill chunk): default KV **~150K
positions** (prompt + max_tokens + 512 slack) whatever was served before: 104.86 GB GTT - 83.27 model - 7.2 chunk
buffers and scratch (measured) - ~0.8 released indexer pages - 2 reserve = 11.6 GB / 75.3 KB. A fresh server
admits up to ~170K (the guard estimates the chunk buffers at 5.9 GB, 1.3 GB under the measured growth: open item).
fp16-only KV: the full 262,144, with ~6 GB of GTT to spare.

**Released pages stay charged.** The HIP runtime keeps a released VMM page charged to GTT for the life of the
process (GTT used stays flat across resets; a later hipMemCreate reuses it), so rt::Device::AvailableBytes cannot
see it. With Reset releasing every page, a server that had served 128K refused 160K. Reset and Restore now keep
the pages the session alone holds (only fork-shared pages are unmapped, which the COW fix needs), so the guard's
"pages already held" covers them. Not kept: an engine-owned pool remapping released handles at another address
(`streams/C/pool_c14.patch`): the swarm faulted with a GPU aperture violation at 2 MB pages in 2 of 3 runs (0 of
3 without the pool). Still uncounted: indexer-key pages released below the live window (~6 KB a token, reused by
the runtime on the next prompt, so the guard under-reports by up to ~0.8 GB after a 128K prompt).

Measured on the eviction path (stream F, 2026-10-07, `SessionCache.ReleasedPageChargeProbe`, ROCm 7.2.3): the
charge lasts only while a range the page was mapped into stays reserved. 4.29 GB of pages unmapped and released
stayed charged (and a new create reused them) until `hipMemAddressFree` of their range gave all of it back.
`DropSession` frees the session's ranges, so dropping a session holding 1.28 GB of KV pages gave 1.38 GB back to
`AvailableBytes` at once: `SessionCache` eviction sees what it frees (the audit's F3 does not occur).
`SessionCache.EvictionStopsOnceThePromptFits` keeps it so, on an `rt::Device` ledger that models this runtime
(mutant `guard-ignores-released`). The charge shows where a range stays reserved: Reset/Restore (fixed above)
and the indexer-key pages below the live window.

**`halo_serve` default (2026-10-07, option (b) with the threshold rule).** Without `--kv-f16`, `halo_serve` uses
fp16-only KV when `--max-context` is above `--kv-f16-above` (default 150000, about what default KV reliably holds;
`0` disables the automatic choice) and the default kinds at or below it. An explicit `--kv-f16 1` or `--kv-f16 0`
always wins. Startup logs one line, e.g. `kv: fp16-only (auto: --max-context 262144 > --kv-f16-above 150000)`.
`halo_run`, `halo_score` and the library default (`kv_f16_only_ = false`) are unchanged, so their references stay
valid. Resolution: `server::ResolveKvKind` (`src/server/kv_choice.hpp`, tests `KvChoice.*` in server_tests).

**The case for `--kv-f16 1` as the long-context serving default** (decided above: option (b)).

- Capacity: 26.2 KB a token against 75.3 (window on), 65% less. One default-KV session tops out near
  150K; fp16-only serves the model's full 262,144 positions. At 32K, 11 sessions fit against 4-5.
- Quality: the precision gate passes against main defaults (short -0.00045 +- 0.00040, long -0.00019 +- 0.00053)
  and against the original baseline; 128K-document NLL equals the default kinds' (-0.0007 +- 0.0030); needles
  are correct at every depth tried, in both kinds at 32K and 128K (11/11 each) and in fp16 at 256K (7/7).
- Speed: prefill is unchanged (prefill reads only fp16 rows in both kinds; prefill logits are bit-identical).
  Decode is the same or slightly faster: 128K needle steps 42.1-42.4 ms (fp16) vs 42.6-42.8 (default), 32K
  39.1-39.6 vs 39.6-41.1 (unpinned, different sessions; not an A/B).
- Costs: decode logits are no longer byte-identical to main's references (a precision change, quality-gated),
  so decode exactness checks would need fp16-KV references, and the gain does not show at short context, so the
  rule "default only if significant end to end" is not met by speed or quality; the case is capacity.
- Options: (a) default on everywhere; (b) default on in `halo_serve` only, or only when `--max-context` exceeds
  what default KV can hold (~150K), keeping `halo_run` / `halo_score` references unchanged; (c) keep off and
  document `--kv-f16 1` for long contexts. Recommendation: (b), with the threshold rule, since it changes nothing
  below 150K and makes the advertised context actually servable.
- gufo on the same needle prompts (its own server, `-c` at the length): 32K 11/11, 128K 11/11, 256K
  7/7; cold prefill 25.1-25.5 s / 104.0-106.0 s / 218-220 s against ours 22.2-24.6 / 92.9-96.1 / 198.7-202.0 s.


## 8. Serving: prompt-prefix reuse across retained sessions (stream H, 2026-10-06; working files `/data/halo-work/streams/H/`)

`halo_serve` used one engine state, extended when a prompt extended everything run and reset otherwise. Two
agent harnesses defeated it (yard Phase 0): opencode's concurrent title request (another prompt) reset the
conversation, which then re-prefilled 7.3K tokens; Hermes drops `reasoning_content` from its history, so its
next prompt is never an extension of what the engine generated and it re-prefilled ~12.5K tokens every turn.

`server::SessionCache` (`src/server/session_cache.hpp`) keeps several sessions and starts each request from the
retained state with the longest prefix of its prompt:
- **Checkpoints.** During each prefill a fork is parked where the prompt's generation prompt starts (the last
  `<|im_start|>`): the next turn renders everything before it again, however it renders this turn's output.
  So reasoning dropped from history costs nothing: it sits after the checkpoint, never mid-history (the engine's
  copy need not be re-rendered without reasoning).
- **LRU** under `--prefix-cache-sessions` (8) and `--prefix-cache-gb` (8 GB of recurrent + exclusive KV pages),
  and whenever `KvRoomFor` says a prompt does not fit. An unrelated request gets its own session.
- **One request at a time** still (the mutex): the title request no longer evicts the conversation, but the
  conversation's request still waits for it. Overlap needs the scheduler (G3) and batched decode.

**Exactness.** Where a prefill is split changes the arithmetic. Real model (`SessionCache.RealModelSplitProbe`,
prose, 5000 and 10000 tokens): a split at any multiple of 16 positions leaves the logits of the prefill and of
4 steps after it bit-identical to one unsplit prefill (24 points from 16 to 9008, across the 8192-token long-chunk
boundary); every split at a multiple of 8 or 4 only moves them by 1.2e-2 to 6.4e-2 relative (L2). 16 is the
DeltaNet scan's piece. A decoded token's state also differs from a prefilled one's.

| `--prefix-cache` | Reuses | Results vs one cold prefill of the prompt |
|---|---|---|
| `exact` (default) | checkpoints at multiples of 64 (`--prefix-cache-align`) taken in aligned prefills | byte-identical (test below) |
| `fast` | also finished sessions (the client echoed what was generated) and unaligned checkpoints | equal up to the prefill-vs-decode rounding: 1 of 123 small-model requests differed |
| `off` | nothing | the reference |

`SessionCache.ReuseIsByteIdenticalToColdPrefill` (Hegel, small model, 40 conversations, 123 requests): random
agent conversations whose next prompt keeps the previous prompt to its generation prompt and then echoes or
re-renders the turn, with unrelated requests interleaved; every request's greedy or seeded-sampled tokens from
the exact cache equal a cache-off engine's; 28% of prompt tokens were reused. Positive control:
`HALO_TEST_CACHE_ALIGN=1` (checkpoints anywhere) fails it on the first request.

**Cost per request** (real model, `HALO_CACHE_TRACE=1`, 1.4-1.6K-token conversation, stream H c1/c2): a
`Prefill` call has ~140 ms fixed cost (113 tokens 279 ms; marginal ~0.65 ms/token), and a checkpoint splits the
prefill into two calls plus a fork (~10-20 ms). A later turn's prepare: exact 400-427 ms (64 + ~12 tokens),
fast 290-300 ms, off 1.06-1.17 s; the first decode step is 38-43 ms in every mode (no graph-capture cost). So
exact mode costs clients that echo their history (pi, opencode) ~0.2 s TTFC a turn against the old single-session
extension, and saves Hermes ~8 s a turn. Fast mode skips the checkpoint when it extends a finished session.
Removing the split cost belongs to the engine: a lower per-call fixed cost, or capturing the recurrent state at an
aligned position inside a chunk.

**Agent harnesses** (yard Phase 0 battery, 5 tasks each, real model; `platform/bin/analyze.py`; "later" = every
request of a task after its first; re-prefill = prompt tokens not served from reused state):

| Harness | Later-turn TTFC median: main / exact / fast | Re-prefill median, exact | Task wall, t1+t4+t5: main / exact / fast | Pass: main / exact |
|---|---|---|---|---|
| Hermes 0.21.5 | 9.5 / 0.76 / 0.63 s | 329 tokens (main: the whole ~12.5K prompt) | 136 / 75 / 70 s | 5/5 / 5/5 |
| opencode 1.18.35 | 0.42 / 0.81 / 0.44 s | 436 | 88 / 78 / 125 s | 5/5 / 5/5 |
| pi 1.0.4 (chat) | 0.35 / 0.65 / 0.24 s | 270 | 49 / 43 / 43 s | 5/5 / 5/5 |
| pi 1.0.4 (responses) | 0.33 / 0.66 / 0.28 s | 243 | 53 / 36 / 38 s | 5/5 / 5/5 |
| Codex 0.160.1 | rejected (400) / 0.96 / 0.50 s | 636 | - / 109 / 93 s | 0/1 / 4/5 |

Main's opencode p90 (15.8 s: the title request's ~11 s plus a 7.3K re-prefill) became 10.5 s: the wait for the
title request remains (one request at a time), the re-prefill is gone. Codex's multi-agent tools work end to
end (namespace tools, server/openai_request.hpp): asked to delegate, it spawned, waited for and closed a sub-agent
through `multi_agent_v1.*` calls returned with `"namespace"`, and answered correctly. Single runs per task: task walls move with
the model's output length (opencode t5 24.7 s exact, 43.5 s fast).

## 9. Serving when the client leaves, stalls or retries (stream F, 2026-10-07; working files `/data/halo-work/streams/F/notes/`)

Every boundary a request crosses now has a done / not done outcome, and the unknowns have a bound
(`research/agent-platform/ref-observability/AMBIGUITY.md`):

| Where | Done | Not done | Unknown, and its bound |
|---|---|---|---|
| Client receives the response | every event, the terminal one included, written | a write failed, the connection closed (`POLLRDHUP`), or the server is stopping: **cancelled** | bytes the kernel accepted after the peer left: found at the next write or poll (one frame) |
| Client still reading | the kernel keeps sending | no progress for `--send-timeout` (default 30 s): a blocked send, or written bytes the kernel cannot send because the client's window stayed closed (`SIOCOUTQNSD`) | a client whose own buffers still take data is indistinguishable from a reader until they fill |
| Engine work for the request | ran in full | stopped: while queued (the generator's timed lock is retried every 20 ms; Flash-Next and Qwen3.5), before each prefill chunk after the first (chunks end synchronized: at most one more runs), before each decode step | none past one chunk or step |

What a cancelled request leaves:
- **Responses object**: status `cancelled` (an API status; `incomplete_details` stays null, its reasons being
  `max_output_tokens` and `content_filter`). A response is `completed` only when it was written in full.
- **Store**: a streamed request keeps only the cancelled object (conversation null; its client may hold the id from
  `response.created`), so `GET /v1/responses/{id}` answers definitely and `previous_response_id` on it is a 400
  naming the parameter, never a continuation of a turn the client did not get. A non-streaming request whose
  client left stores nothing (nobody received its id): `GET` and `previous_response_id` on it are 404.
- **Store write that fails** (audit F7): the terminal event (`response.completed` / `incomplete`) is held back
  until `ResponseStore::Put` returns, and a non-streaming body is built after it. Put settles its own outcome (a
  failed SQLite COMMIT is rolled back, so false means nothing was written). Not stored: non-streaming 500, streaming
  an `error` event in place of the terminal one, observer `finish_reason` `error`; nothing is kept and a retry is
  safe (each attempt is a new response node). A stream whose held-back terminal event cannot be delivered replaces
  the stored turn with the cancelled object.
- **Observer** (halo-scope GENERATION): `finish_reason` `cancelled`, so a truncated generation never looks like a
  finished one in traces or training data.
- **Engine state**: a request stopped mid-prefill drops its live session (partly advanced); checkpoints parked
  before the stop stay (each a cold prefill of its tokens). A stop between decode steps keeps exactly what ran,
  the state of the same request with `max_tokens` equal to the tokens emitted (fast mode retains it, exact mode
  drops it as usual). A request cancelled while queued never touches the cache.
- **Non-streaming**: a 503 (only a client of a stopping server can still read it), decided by the execution's own
  cancelled outcome, not a second poll. A pipelined next request on a keep-alive connection is not a departure
  (the liveness check reads `POLLRDHUP`/`POLLHUP`/`POLLERR`, never a readable socket).

Tests: `OpenAiService.ServingModelUnderDeparturesStallsAndRetries` (Hegel rules complete / leave while queued,
mid-prefill, mid-decode / stall / retry / continue over loopback HTTP, with the invariants above),
`OpenAiService.ADeparted*`, `APipelinedNextRequestIsNotADeparture`, `AFailedStoreWriteIsAnErrorAndKeepsNothing`
(the Hegel machine also has a store-fails rule), `KeepaliveCommentsWhileAStreamWaitsAndClientsIgnoreThem`,
`ScopeResponses.AFailedWriteIsReportedAndLeavesNothing`, `Http.AStalledReaderIsCutOffAfterTheSendTimeout`, `Http.ASlowStalledReaderIsCutOffToo`;
GPU (small model): `SessionCache.CancelledRequestsLeaveStateEqualToAColdReplay` (cancels at any poll; exact mode
stays byte-identical to a cold engine, fast mode after a k-token stop equals `max_tokens` = k),
`ACancelStopsThePrefillAtTheNextChunk`, `AQueuedRequestWhoseClientLeftNeverRuns`. Mutants (tests/mutants.toml):
`svc-departure-as-stop`, `svc-no-liveness-check`, `gen-queue-ignores-departure`, `engine-prefill-ignores-cancel`,
`http-send-no-deadline`, `http-stall-ignores-unsent`, `guard-ignores-released`, and (stream A2)
`svc-departed-nonstream-stored`, `http-readable-is-departure`, `svc-store-failure-ignored`,
`svc-terminal-before-store`, `scope-commit-failure-left-open`, `svc-keepalive-never-sent`.

End to end (real model, `halo_serve --send-timeout 5`, `HALO_CACHE_TRACE=1`): a streamed Responses request cut
after 20 deltas stopped at 35 tokens, `GET` said `cancelled`, continuing from it was a 400, and a completed
response continued normally. A client that read nothing of a 6000-token stream held the engine 5.8 s (156 tokens)
before a second client was served; with the send deadline alone the kernel took ~1 MB of frames first (4434 tokens,
172 s). A non-streaming request whose client left after 3 s stopped at 76 tokens of 3000.

Still open: the engine lock is held while a send waits (another client waits up to `--send-timeout` per stalled
request; a bounded per-request output buffer drained by the connection thread would remove that); by default
nothing is sent while a request is queued or prefilling, so a client's idle timeout can still fire first:
`halo_serve --sse-keepalive S` (default 15 s since 2026-10-07, 0 turns it off; stream A2) writes an SSE comment `: keepalive` from the liveness
poll when a stream has written nothing for S seconds (SSE parsers ignore comments; checked with the openai 2.24
SDK, `tests/functional/sse_keepalive_sdk_check.py`), which also turns a departure into a failed write; a
half-closed connection (`shutdown(SHUT_WR)` after the request) counts as gone; `Qwen35Generator` polls while
queued, before its prefill and each step, but its prefill is one engine call; the first prefill chunk after a
checkpoint fork starts without a poll (the generation prompt only, a few tokens).

## 10. Suspend and resume: a session's state spilled to NVMe (stream SR4, 2026-10-07; working files `/data/halo-work/streams/SR4/`)

HARNESS.md SR4: an idle session costs disk, not device memory; a resume restores its state instead of re-prefilling.
`FlashNextGpu::SpillSession(path, blob)` writes the bound session's state to a file; `RestoreSession(path)` puts it
back into the bound session (any session, any engine of the same model and options). Code:
`src/engine/session_spill.cpp`; cache and server hooks in `server::SessionCache` and `halo_serve`.

**What a file holds.** Snapshot's rows, nothing derived: the DeltaNet recurrent state (36 x 48 x 128 x 128 fp32) and
conv state, the PLE conv history, per attention layer the K and V rows (fp32, or fp16 with `--kv-f16 1`), the live
indexer-key rows (`kv_index_window_`: only the current QSA block's), the pooled block rows, the per-layer block counts,
the n-gram window and the position. The fp16 K/V copy of the default kinds is not stored: Restore marks it invalid and
the chunked prefill re-derives it from the fp32 rows on first use (the same conversion, so the same bits; mutant
`spill-fp16-copy-kept-valid`). Captured decode graphs are not state: a restored session captures its own.

**Bytes per token** (measured on the files, real model, `halo_run --spill-bench`): **default KV 50.7 KB a token, fp16-only
KV 26.1 KB a token, plus 118.0 MB per session** (recurrent + conv + PLE) and < 1 MB of header, live indexer rows and
4 KB section padding. That is 12 attention layers x (K + V 4096 B + pooled blocks 128 B) and x (2048 + 128) B. The
file is smaller than the session's device footprint (default 75.3 KB / token mapped: the fp16 copy and the indexer
keys are not stored).

**Draft block (MTP).** The draft block's history is engine-wide state owned by one session. When the spilled session
is that owner with an unbroken history (`MtpPending`), the file also holds the draft block's position, its pending
hidden and its K / V / indexer / block rows; a restore into an engine that has the draft block loaded and tracking
takes ownership and resumes drafting exactly where it was (`MtpSpec.SpillRestoreKeepsDraftHistoryOnRealModel`: the
same drafts, acceptances and tokens as the uninterrupted session). Otherwise (not the owner, history broken, or an
engine without the draft block) nothing of it is stored or restored, and the rule is the one `MtpTrack` already has
mid-sequence: the restored session does not draft until its next reset (`mtp_pos_` past the position). The trunk's
results never depend on the draft block (greedy verification emits AR's tokens), so this costs speed only.

### File format (version 1)

| Bytes | Content |
|---|---|
| 0-95 | header: magic `HALOSPL1`, version, header size, **model fingerprint**, **options fingerprint**, **KV kind**, indexer window, **position**, layer count, task count, draft-block flag / position / blocks, n-gram size, blob size, file size, meta size, meta CRC, piece size, **header CRC** |
| 96 .. 4096k | meta: block counts, n-gram window, the task table (file offset, bytes, CRC32C per piece), the caller's blob |
| payload | the sections in order, each from a 4096-byte boundary, cut into pieces of at most 16 MB (the I/O and checksum unit) |

- **Model fingerprint**: FNV-1a over every uploaded tensor's name, type, shape, file offset and first 64 bytes, and the
  LoRA count (fixed at the end of Create). Not a content hash of 79 GB (that is `model::BaseContentHashCached`, minutes
  cold); it tells apart other models, other quantizations and other files of the same shapes.
- **Options fingerprint**: every static option that changes a state's numbers (prefill precision options, chunk sizes,
  router modes, decode kernel masks, `prefill_kv16_`, the KV kind and index window, LoRA scales), strictly: a state
  computed under other options is not what this engine's own prefill would hold, so it is refused even where the
  option is documented exact. Graphs and timing are not in it.
- **Checks on restore, before the session is touched**: magic, version, header CRC, sizes, file size = recorded size
  (truncated or extended files), meta CRC, model, options, KV kind, index window, layer count, position <= max_context,
  block counts, and the whole task table against the layout this engine computes for that position. A refused file
  leaves the session as it was. **While reading**: each piece's CRC32C; a mismatch resets the session to empty
  (`RestoreSession` reports `reset`), never a partly restored state. The caller then re-prefills, which is exactly a
  cold prefill.
- The blob (the cache's token list) is covered by the meta CRC; the caller trusts the blob, not its own record, for what
  the state holds.

### File format v2: encrypted (stream SE, 2026-10-07; working files `/data/halo-work/streams/SE/`)

Spill files hold whole conversations (the tokens, and states from which text can be recovered). Since SE they are
encrypted at the application level (envelope encryption; the user chose it over a LUKS volume), and **v2 is the only
format halo_serve writes unless the operator asks for plaintext**.

| Bytes | Content |
|---|---|
| 0-447 | header, all of it the **AAD** of everything below: magic `HALOSPL2`, version 2, header size, model fingerprint, options fingerprint, KV kind, index window, position, layer and task counts, draft-block fields, n-gram size, blob size, file size, meta size, piece size (v1's offsets 0-83), the **file salt** (8 random bytes), the **key id** (<= 128 printable bytes), the wrap nonce, the **checkpoint id** (16 random bytes, bytes 244-259), zero reserved bytes |
| 448-511 | the **wrapped DEK** (32), its GCM tag (16), the meta's GCM tag (16) |
| 512 .. 4096k | meta, AES-256-GCM sealed: block counts, n-gram window, the task table (offset, bytes, section, **GCM tag** per piece), the caller's blob |
| payload | the same sections and 16 MB pieces as v1, each AES-256-GCM sealed in place |

- **Keys.** Each file gets a fresh random 256-bit data key (DEK), stored only wrapped (AES-256-GCM, random nonce) under
  a key-encryption key: `KEK = HKDF-SHA256(ikm, info = "halo-spill-v2" || be32(len) || session || be64(epoch))`, the
  ikm being the provider's master key (phase A) or the session key yard-ctl registered (phase B). The key id names
  provider, master (or session key) name, epoch and session (`tpm:<16 hex>:e1:s<pid>-<key>`, `yard:<16 hex>:e7:<sess>`);
  a key name is the first 8 bytes of HMAC-SHA256(key, "halo-spill-master-id"), never the key.
- **Nonces.** Piece i: `salt || be32(i)`; the meta: `salt || 0xffffffff` (pieces are < 2^32 - 1). Unique within a file
  because indices are; across files the DEK differs (and the salt is a second guard). The DEK-wrap nonce is random
  under a KEK: a yard KEK wraps one DEK (one checkpoint); a master KEK (session, epoch) wraps one per file of that
  session at that epoch (re-spills after restores): q wraps collide with probability about q^2 / 2^97, negligible at
  a few thousand files.
- **Integrity.** The GCM tags replace v1's CRC32Cs. Restore: magic and version; sizes (file size = recorded); the
  key id against the caller's KEK (another session, epoch, provider or master: refused with both ids in the message);
  the DEK unwrap, which authenticates the whole header; the meta's tag; then v1's identity and layout checks; each
  piece's tag before its bytes go to the device (a failure resets the session, as v1's checksum did). Every refusal
  before the payload leaves the session untouched. A failed GCM open zeroes its buffer (never unauthenticated
  plaintext left behind); every EVP context is reset after each call; the DEK is encrypted from its `SecretBytes`
  straight into the header (never copied there in the clear).
- **Which checkpoint.** GCM proves a file authentic, not that it is the latest: two files of one session, key and
  epoch (a re-spill after a restore) would both authenticate. The writer draws a random **checkpoint id** into the
  authenticated header, the cache keeps it in the entry, and restore refuses an authentic file of another checkpoint
  before touching the session (`SpillProtection::expect_checkpoint`).
- **Partial restores.** A restore goes into a fresh session that the cache drops on any failure; inside the engine
  every restore worker joins (`RunPool`) and the device is synchronized before the state is reset, so no write of a
  failed restore lands after `RestoreSession` returns (`Spill.CheckpointIdentityDowngradeAndNoLateWrites`). So a
  partial restore is never exposed or used; it may physically exist in a session that is then discarded. If that
  synchronize itself fails, quiescence is not established: the engine **poisons** itself (no reset; every session,
  step, prefill, spill and restore call is refused until restart; `Spill.FailedSyncInRestoreCleanupPoisonsTheEngine`).
  Round 4: a poisoned engine also **quarantines** its device memory: `DropSession`, `ReleaseKvPages`, the session
  free path and the destructor release nothing for reuse (queued writes may still target it; the process exit
  releases it with the device context), so the cache's own failure cleanup cannot hand those pages to another
  session (`SessionCache.PoisonedRestoreQuarantinesItsMemory`). `SpillPeek` touches no device memory. The
  destructor's early return leaves the members to their own destructors: buffers and graphs are plain handles
  (static_assert), and the KV pages and sessions, whose destructors would release pages, are leaked on purpose
  (round 5, tested with a page-release probe). halo_serve does not recover in the same device context: it notices the
  poison, stops serving and exits with status 3 for its supervisor to restart it. The flag is an atomic
  (`core::PoisonFlag`, release / acquire), written by the restore and polled by the main loop without a lock (round 6;
  ThreadSanitizer test). Integration: `streams/SE/s6.txt` (a real-model halo_serve with
  `HALO_TEST_RESTORE_SYNC_FAULT=1` exits 3 after a failed restore). That test also found a shutdown-order bug from
before SE: halo_serve's generator (its session cache) was destroyed after the engine it uses (a use-after-free at
every shutdown, harmless by luck until the poisoned engine's quarantine left null sessions); halo_serve now destroys
the generator first.
- **Spill outcomes.** `SpillInfo::published` tells "nothing published" (the path is what it was) from "published but
  the directory fsync failed" (durability unknown). The cache reports the latter to the key provider as
  `Outcome::kUncertain`: the key counts as used (never handed back for another file), and the file is removed.
- **v1** (plaintext, CRC32C) is still written when there is no provider and `--spill-plaintext-ok 1`, and read only
  with `--spill-accept-v1 1`, and then only for an entry this process recorded as v1: an entry recorded as v2 never
  takes a v1 file (`expect_version`), whatever the flag. Each v1 restore logs a warning. The flag only covers v1
  files this same process wrote (with `--spill-plaintext-ok 1`); it does not import files from before SE or from
  another process (the cache index is in memory). It goes with format v1 (sunset: when v1 writing is removed).
- Crypto: OpenSSL 3.6.3 libcrypto from the pinned nixpkgs (`pkgs.openssl` in flake.nix; no new fetch: the store path
  was already in the closure), EVP AES-256-GCM. On this CPU (Zen 5, `vaes vpclmulqdq avx512*`) it takes the
  VAES/AVX-512 path: 14.3 GB/s a core for 16 MB buffers; with VAES/VPCLMULQDQ or AVX-512F masked
  (`OPENSSL_ia32cap`) 3.0 GB/s, without AES-NI 0.44 GB/s. Code: `src/core/spill_crypt.*` (primitives, the envelope),
  `src/engine/session_spill.cpp` (the format), `src/server/spill_keys.*` (providers).

### Key providers and flags (phase A)

| `--spill-key` | Master key | Notes |
|---|---|---|
| `tpm` / `tpm:CRED` | the systemd credential `halo-spill-master`: `$CREDENTIALS_DIRECTORY/halo-spill-master` when run as a service (`LoadCredentialEncrypted=`), else `systemd-creds decrypt --user --name=halo-spill-master CRED -` (default CRED `$XDG_CONFIG_HOME/halo-flash/spill-master.cred`) | made once by `/data/halo-work/platform/ops/spill-key-seal.sh`, **no root**: a user-scoped credential (systemd 259) is sealed and decrypted unprivileged through the `io.systemd.Credentials` socket; key type host + TPM2 (the file holds a TPM2 blob; user scope ignores PCR binding). chuck is not in group `tss` and need not be |
| `file:PATH` | 32 raw bytes or 64 hex digits in a 0600 regular file (no symlink) | tests, machines without a TPM |
| `none` (default) | - | spills are **refused** (the admin route answers 409, eviction drops) unless `--spill-plaintext-ok 1` |

The master (and every derived KEK and DEK, and every yard key) lives in `SecretBytes`: own mmap pages, `mlock`ed,
`MADV_DONTDUMP`, `MADV_WIPEONFORK`, zeroed (`OPENSSL_cleanse`) before unmapping. **Fail closed:** if `mlock` or either
`madvise` fails the allocation is empty and the caller refuses (no key is ever held in swappable or dumpable memory;
the service needs `RLIMIT_MEMLOCK` of a page per live key, a few hundred KB at most). The master is read once at
startup (before the model loads: a missing key fails fast) and zeroed at exit. `systemd-creds` is run by absolute path
(`/usr/bin/systemd-creds`) with a scrubbed environment (`PATH=/usr/bin:/bin LANG=C`, nothing inherited). Nothing logs
key bytes; halo_serve logs `spill: encrypted (v2), key tpm:<name> epoch N`.

**Default: refuse, not plaintext.** Without a key, a spill is refused. A refused spill costs a re-prefill later (latency,
exactly a cold prefill); a plaintext default would silently write whole conversations to plain XFS. So plaintext
needs the explicitly named `--spill-plaintext-ok 1`; and when a provider is configured but has no key for a session,
the spill is refused even then (never a plaintext fallback: mutant `cache-plaintext-on-key-error`).

**Epochs: an access fence, not crypto-shredding.** The master provider's epoch is persisted in
`<spill dir>/master-<key name>.epoch` (0600, written durably before it takes effect; created at 1; a malformed file
refuses the start) and never goes back across restarts (`--spill-epoch` was removed). `POST /admin/spill/fence-epoch`
(old name `/admin/spill/rotate`) moves it on: every file sealed before names an old key id and no longer opens through
this server; the cache erases those entries and deletes their files (`DropUnreadable`). This is **not** erasure: the
master still derives every old KEK, so a surviving copy of an old file opens for anyone who later holds the master.

**Where true crypto-shredding holds.**
- *Phase A (master):* only when the master itself is destroyed: the sealed credential and every copy of it deleted
  (or the TPM's key gone) **and** the running halo_serve exited (the master is in its memory until then).
- *Phase B (yard keys):* per checkpoint, once no copy of K is left anywhere: yard-ctl deleted K (IDENTITY.md), no
  token carrying K can still be delivered (the last token's `iat + 300 s`), and halo_serve holds no K for it: either
  `+ key TTL` passed (halo zeroes keys at `min(token exp, registration + TTL)`; the sweeper zeroes them within a
  second), or a revoke sent after the last token's expiry was confirmed by `GET /admin/keys`. That is yard SR1's
  `shredSettle` + `shredVerify` (spec/keys, finding 1). Plaintext the live process had in memory before is outside it.

**Cost** (ROOFLINE section 22): restore 2K 81.2 -> 79.7 ms, 12K 266.9 -> 267.1 ms (v1 -> v2, medians); spills within
disk noise. Predicted +-3%: the GCM pass replaces the CRC32C pass at about the same bytes a second.

### The key route: per-checkpoint keys from yard-ctl (phase B)

`--yard-jwks PATH --yard-issuer ISS [--yard-audience halo_serve] [--yard-key-ttl 600] [--yard-state PATH]
[--spill-yard-fallback master|refuse]` adds the provider "yard" (`src/server/yard_keys.*`). yard-ctl registers a
256-bit key K per session and checkpoint; the file's KEK is `HKDF(K, "halo-spill-v2" || sess || E)`. The yard-ctl
side (custody of K, the token, the call sequence) is specified in the research repo's IDENTITY.md, "Spill keys"; the
shared key-service model of yard SR1 (yard `spec/keys/`) covers both consumers (spill keys, ZFS workspace keys) and
this route implements its halo side (`staleCheck`, `usedCheck`, `haloDurable`).

- **Transport.** The route is served on the **admin Unix socket** (`--admin-socket`, default
  `$XDG_RUNTIME_DIR/halo-flash/admin.sock`: its directory must be ours and 0700, the socket is 0600, and every
  connection's peer uid (`SO_PEERCRED`) must be halo_serve's own). **Same uid only** (round 3): the cross-uid option
  `--admin-socket-peer-uid` was removed (a 0700 directory and a 0600 socket deny another uid before `SO_PEERCRED` is
  even read, so it never worked) and is refused. A yard-ctl of another uid uses the encrypted-transport route below.
  The path is probed with a bounded non-blocking connect: only a refusal (`ECONNREFUSED`, nobody listens) or the
  path vanishing (`ENOENT`) proves it stale and lets it be replaced; a live peer, a full backlog, a timeout (500 ms)
  or any other error refuses the start (round 4: fail closed); `Stop()` unlinks the path only while it is still this server's socket (same inode). That is the same-host
  default. The key is in the body and halo_serve has no TLS: `--yard-keys-tcp 1` also serves it on
  `--host:--port`, which is allowed only behind an authenticated, encrypted transport (the tailnet's WireGuard to
  that host, nothing in between); a signed token does not keep K confidential.
- `POST /admin/keys/{sess}` body `{"token","use":"seal|open|revoke","key":"<base64url 32 bytes>"}`. The token: RS256
  only, explicitly PKCS#1 v1.5 + SHA-256 (yard's issuer signs RS256: internal/sts/jwt.go); `kid` in the JWKS file;
  `aud` halo_serve (a string, or an array containing it), `iss`; integer `iat`, `exp` (and `nbf` if present: any other
  type is refused), `0 < exp - iat <= 300`, no post-`exp` leeway (iat / nbf may be 30 s ahead for clock skew); `jti`;
  `yard_sess` = `{sess}`; `yard_epoch`; `halo_spill: {use, key_sha256, entry (seal: the entry id
  "<incarnation>.<key>" from `GET /admin/sessions`), through (revoke, optional)}`. A JSON document with a repeated member is refused; JWT segments are strict base64url. Every
  token refusal is the same `401 {"ok":false,"error":"token refused"}`; the reason goes to the log only. 409s (stale,
  fenced, used, replay) are answered only to a verified token. Errors never echo the key or the token; the request
  body is overwritten after the call (the parsed copies too). The HTTP layer keeps request bytes in a `SecretBuffer`:
  consuming a request moves the rest down and zeroes the vacated tail, growth zeroes the old storage before freeing
  it, and dropping the connection (a client gone mid-body, a parse error) zeroes everything; every request's body
  string is zeroed after its dispatch, whatever the path (an API-key refusal included).
- **JWKS.** A key is used only if `kty` RSA, `alg` absent or RS256, `use` absent or `sig`, `key_ops` absent or
  containing `verify`, a canonical modulus of 2048..16384 bits (`BN_num_bits`, not encoded bytes) and `e` = 65537. A
  duplicate `kid` makes the whole document invalid: an empty key set is swapped in (deny-all) until a valid
  document appears. The file is opened `O_NOFOLLOW` and must be a regular file of at most 1 MB, not writable by
  group or others, owned by halo's uid or root (yard publishes it as one of them); otherwise the same deny-all. It
  is re-read when it changes (inode, size, mtime, ctime: a chmod is noticed) and at least every 60 s; an unknown kid forces a re-read at most once per 5 s (a flood cannot
  keep halo loading keys). The file's size is checked on the opened descriptor (<= 1 MB) before it is read. Loading
  and validating a new set happens outside the provider lock and the result is swapped in (an immutable snapshot),
  so leases and the expiry sweeper never wait for it. A removed signing key stops verifying at the next registration;
  keys already registered under it keep their (short) deadline, and a verification already running on the old snapshot
  completes on it. A missing, unreadable, malformed or empty document
  denies everything until fixed. Startup
  refuses an empty issuer or audience, a TTL outside 1..3600 s, and a JWKS with no usable key.
- `GET /admin/keys`: the server `incarnation` and whether the provider is `stopped`; per session the highest epoch, the revoke fence (`revoked_through`), the last seal, spilled and
  restored epochs, and the live slots (epoch, key name, entry, in use; never the key). An expired key is gone, not
  hidden.
- `POST /admin/sessions/{key}/spill` (no query: `?session=` was removed). The **seal registration names the entry**
  (`halo_spill.entry` = its `entry_id`, `"<incarnation>.<key>"`, shown by `GET /admin/sessions`; the incarnation is
  32 random hex digits, 128 bits, drawn at every start; round 3, widened in round 4; an entry id must be exactly the
  incarnation, a dot and a canonical decimal key): a numeric key is reused by the next process for another
  conversation, so a seal for an entry of another incarnation (a token minted before a restart) is refused (409).
  Only a seal registration for that entry seals it, under its own session's label, so an admin
  caller can neither relabel an entry nor spend another session's registration. From its first yard seal an entry is
  **yard-bound** for good: it is never sealed under the master again, fallback or not (no custody downgrade); a
  re-spill needs a fresh seal registration. `restore` needs the open key: until it is registered the restore answers
  409 and the entry and its file stay untouched (the cache's `kNoKey`).
- **Epoch rules**, per session, all persisted (below):
  - seal at E: `E > fence`, `E >= highest`, `E > last spill`; for exactly one entry;
  - open at E: `E - 1 > fence`, `E >= highest`, `E == last seal + 1`, `E > last restore`; opens exactly the file of
    `E - 1`;
  - revoke at token epoch T: **a token cannot expand revocation beyond its authority T** (round 4: a delayed revoke
    never fences, erases or zeroes newer state; cleanup may still apply a fence an earlier revoke already set).
    The fence R is the claim `through` (1 <= R <= T; `through: 0` is refused), else min(last spilled epoch, T), none if nothing was spilled (round 3,
    GLM NL3: an abort before any spill does not poison its epoch). With R: `fence = max(fence, R)`. Keys zeroed: a
    slot registered at an epoch <= T (a pending operation the token may cancel) or whose file epoch is fenced (it could
    never be used); a registration after T survives. The last seal is forgotten if fenced. A revoke whose T is below
    the highest epoch is not refused as stale (yard may legitimately revoke an older checkpoint): it is applied, bounded
    by its T. The route then
    erases that session's entries (spilled or resident) whose key epoch is <= fence (`DropRevoked`; a newer
    checkpoint survives a late cleanup). A revoke is never stale and repeating it is a no-op 200
    (`"duplicate":true`). yard-ctl sends `through` = the epoch of the last checkpoint it wants gone when it knows it.
  - a registration below the highest epoch is stale (409); a seen jti is a replay (409).
- **Operation identity (round 5).** A seal or open token may carry `halo_spill.op`, a signed operation id
  ([A-Za-z0-9._-]{1,64}) that yard's key service mints per checkpoint operation. When the request and the record
  both have one, the op id decides which operation a token belongs to (a new op id is a newly authorized operation,
  even with the same key, as an open that reopens the same file); otherwise (use, epoch, key hash) decides (K is
  fresh per checkpoint). An op id never names two operations (another key or epoch: 409).
  - **Cancelled stays cancelled.** A revoke that cancels an unused operation (registered, or expired unused) without
    fencing a file marks it revoked; any later token of that operation (a delayed one minted before the revoke) is
    refused (409 "cancelled"). Round 6: one identity predicate serves lookup, live-slot deduplication, recording and
    cancellation (two distinct op ids with the same key and epoch are two tracked operations; an id-less retry of an
    id-bearing operation is that operation and never relabels it). Every operation leaves a non-secret tombstone (op
    id, use, epoch, key hash, first-registration cap, state) kept at least 330 s (300 s token lifetime + 30 s skew)
    past both its last use and its first registration + TTL, whatever else the session does; while it lives,
    cancelled and consumed are final. At most 1024 tombstones a session: a full table refuses new operations (409,
    retry later) instead of evicting history. A new checkpoint at the same epoch (new op id and key) is allowed.
  - **Expired, never used.** An operation whose key expired unused may be registered again with a fresh token while
    its first registration + TTL lasts (its slot died with a token, not the operation); after that it is over (409:
    register a new operation, a new op id). A revoked or consumed operation is never registered again.
  - Records are memory only. After a restart a delayed seal token dies on the incarnation check; a delayed open token
    may register its key, but nothing can be opened with it (the cache's entries and files did not survive the
    restart), and the key dies at its TTL.
  - A repeated default revoke is recomputed when it is delivered: after a new checkpoint was spilled at the same
    epoch it covers that one too; yard retries with an explicit `through`.
- **One-shot, really.** halo keeps a non-secret record per operation (use, epoch, key hash, entry, op id;
  registered / consumed / expired / revoked). The same registration again (fresh token, same use, epoch, key and entry) is a no-op
  200: `"state":"registered"` while the key is live, `"state":"consumed"` after its spill or restore, and then the key
  is **not** re-armed. Another key at a used epoch is refused (409 "used"). A persisted used fence means "an attempt
  started", not "a checkpoint is available": a 409 "used" is answered both when the spill or restore happened and
  when its key was disabled after a failure (nothing published, the rollback not durable); either way the caller
  re-prefills (the entry stayed resident or its file is gone).
- **Expiry is erasure.** A key's authorization ends at `min(token exp, registration + TTL)` (steady clock for the
  TTL). The token is checked again at the transition boundary, after the provider lock is taken (another
  registration's fsync may have held it past exp): an expired one is refused, and the key's deadline is computed from
  that in-lock wall-clock sample in milliseconds, so it never outlives the token's absolute exp (round 4). The steady
  anchor is taken before the wall sample, so a pause between them can only shorten the key's life (round 5). A later
  forward jump of the wall clock does not shorten an installed key: its deadline is on the steady clock; a retry with a fresh token may extend it to that token's exp, but never past the first registration + TTL; a sweeper thread zeroes and unmaps expired keys once a second (a key reserved by a running spill or restore is
  zeroed when that finishes).
- **Files that can never open again.** `MayOpen` (and so `DropUnreadable` and a failed restore) treats a yard file of
  epoch E as unreadable for good when it is fenced, already restored, behind a newer seal (the last seal is not E),
  or behind the highest epoch (an open at E + 1 would be stale): its entry and file are dropped instead of waiting for
  a key that can never be registered (round 5).
- **Leases and races.** A spill or restore takes a lease on the exact registration (its generation) under the
  provider's lock; finishing consumes that registration and no newer one; a registration cannot replace a slot a
  lease holds, the other use's slot included (a new seal would end a reserved open, an open a reserved seal: 409,
  retry). A lease is granted only while its epoch is unused and unfenced. A seal that published nothing gives its used
  fence back (also when its slot was revoked meanwhile). If that rollback is not published, the stricter fence stays
  and the key is zeroed; if its publication is uncertain, memory takes the lowered fence (what the disk may show) and
  the provider stops, so safety comes from the fail-stop and the zeroed slots, not from the fence. A seal published
  with uncertain durability keeps the fence and consumes the key.
  Lock order: the cache's generation lock, then the provider's lock; never the reverse
  (a registration runs outside the generation lock; only its revoke cleanup enters it afterwards). **In-flight restore
  at a revoke:** the revoke zeroes the slot at once; a restore that already holds its lease finishes with its own copy
  of the KEK; the revoke's cleanup then runs under the generation lock, after it, and erases the restored entry. So a
  revoke takes effect for every operation that starts after it, and erases the results of the ones in flight.
- **Restart (incarnation).** Per session the highest epoch, revoke fence, last seal, last spilled and last restored
  epochs are kept in `--yard-state` (default `<spill dir>/yard-state`, 0600, written durably; at most 4096 sessions,
  rewritten whole under the lock). A registration is acknowledged only after they are on disk; a spill or restore
  raises its used fence on disk **before** the key is used (a failed spill puts it back).
- **State files (round 3, round 4; `src/server/state_file.*`).** halo_serve owns the whole spill directory for its
  lifetime (`spill-dir.lock`, taken before any key or the model loads): a second halo_serve on the same directory
  refuses to start whatever its keys. Each state file (yard's and the master's epoch) is also owned exclusively
  (`<file>.lock`). The state directory is opened once and pinned (`O_DIRECTORY | O_NOFOLLOW`, ours, not
  group/other-writable); the lock, the reads, every publish (`openat`, `renameat`) and the directory fsync go through
  that descriptor, so a renamed or replaced path cannot split a write from its fsync. Its ancestors are trusted. Publication is typed:
  *unpublished* (nothing changed), *durable*, or *uncertain* (renamed into place, directory fsync failed). A
  transition is staged: memory takes the new epochs only after a durable publish, and no live key is destroyed before
  that (except a revoke's keys: fail-closed, documented). On *uncertain* the outcome is unknown, not "not done": the
  yard provider **stops** (its key slots are zeroed; leases already handed out keep their own KEK copies and finish;
  every registration answers 503 and every new lease is refused until a restart reloads what the disk holds; round 5:
  through the master fallback too, seal and open alike, one policy for the composite provider); the
  master takes the new epoch (what the disk shows, the stricter fence), reports the failure, and seals nothing at that
  epoch until a later durable write confirms it (each seal attempt retries that write), since a crash could revert
  the epoch and orphan files sealed at it. Loading opens the directory and then
  the file relative to it with `O_NOFOLLOW`, requires a regular file owned by us with no group/other bits, reads it
  to EOF bounded by its size, and refuses junk, unterminated lines, duplicate sessions and inconsistent epochs.
  `<file>.init` marks a state that was initialized: a missing state with its marker refuses the start ("recovery
  required"). Only `ENOENT` counts as absent when the marker is looked up; any other error refuses (round 5). State
  file names are single plain names (no `/`, not `.` or `..`). The first initialization writes the state, then the marker; a state without its marker (from before
  round 3, or a first start interrupted between the two writes) gets the marker on its next load, under the lock.
  Restoring an older copy of both from a backup is out of scope (no external monotonic anchor).
  **Recovery** when a start says "recovery required":
  - *aborted first initialization*: cannot happen any more (state first, marker second); a marker without any state
    from an older build may be deleted;
  - *unexpected loss of an established state*: do not just delete the marker (that silently re-opens every fenced
    epoch). Either restore the authoritative fences (yard's view of each session's epochs and revokes), or start a new
    authorization namespace: delete the spill files and the marker, and have yard re-register only at epochs above
    everything it ever issued (and wait out the token lifetime, 300 s, plus the key TTL first);
  - *deliberate reset*: the same as a new namespace. Keys, leases and seen jtis are memory only: a restart forgets every key (yard
  re-registers after verifying `GET /admin/keys`); a replayed token after a restart is either refused by the
  persisted fences or re-registers the same, unused operation (harmless). Cache entries are memory only too, so after
  a restart old spill files are not restored at all (they are re-prefilled).
- Fallback: an entry that was never yard-bound and has no seal registration is sealed under the `--spill-key` master
  (`master`) or not spilled (`refuse`, the default); a fallback-sealed file opens with the fallback, a yard-sealed file
  only with yard's open key.

**Ambiguity of the registration call** (TEMPLATE.md: request, effect, answer; the caller is yard-ctl):

| Outcome | What halo_serve did | What yard-ctl sees | Caller rule |
|---|---|---|---|
| lost request | nothing | no answer / timeout | **verify** `GET /admin/keys` (slot at E with K's name?) then **idempotent retry** with a fresh token (same use, epoch, key, entry) |
| lost answer | registered (persisted first) | no answer or a 5xx | verify shows the slot; a retry is a no-op 200 (`duplicate`) |
| late (a copy lands after a newer registration) | refused: stale epoch, fenced epoch or replayed jti; a late copy of the same operation is a no-op | the late answer, if any, is 409 | none needed (fence by epoch and jti) |
| duplicate after the key was used | not re-armed | 200 `"state":"consumed"` | done |
| stale or fenced epoch (yard moved on, or revoked) | refused 409, nothing changed | 409 | the file of the old epoch is unopenable: re-prefill; never retry at the old epoch |
| spill without a seal key (not landed, expired, or another entry's) | spill refused (409), nothing written | 409 from the spill route | register again (verify first), then spill again (SR4: a spill is safe to repeat) |
| restore without an open key | 409, entry kept | 409 | register the open key, then restore again |
| revoke lost | keys held until `min(token exp, registration + TTL)` (<= 300 s from the last token's iat); an unexpired open token can still register until its exp | no answer | **verify** (`revoked_through` >= R) and retry; the shred completes only by the settle bound or a verified revoke (above) |
| halo restart | keys and leases gone; epochs and fences kept | `GET /admin/keys` shows no slots | re-register (fresh token) before the spill or restore; a 409 "used" means an attempt started (the operation happened, or its key was disabled after a failure): re-prefill either way |

### Threat model

| Threat | Phase A (TPM master) | Phase B (yard session keys) | Not covered |
|---|---|---|---|
| The disk, a backup or a copied file leaves the machine | protected: files are unreadable without the master, which exists only TPM-sealed (host + TPM2 key) and in halo_serve's locked memory | protected, and also without yard's K | - |
| Tampering, bit rot, truncation, a file swapped between sessions, epochs or models, an older checkpoint of the same session | refused (GCM tags, header as AAD, key id, identity checks, checkpoint id); a partial restore is never exposed or used | same | denial of service: a deleted or damaged file costs a re-prefill |
| A v1 file substituted for a v2 checkpoint | refused (the entry's recorded format), even with `--spill-accept-v1 1` | same | - |
| Old files after a key change | `fence-epoch`: no longer opened by this server (an access fence; the master still derives them); destroyed only with the master (credential and its copies deleted, process exited) | revoke (fence) or the settle bound: halo holds no K; with yard deleting K: crypto-shredded | copies of plaintext a process had in memory before |
| Another local user | the credential is scoped to chuck's uid; files and dir 0600 / 0700 | same | root |
| A process running as chuck, or root, while halo_serve runs | **not protected**: it can run `systemd-creds decrypt --user` itself, or read halo_serve's memory (ptrace as the same uid) | a process as chuck cannot mint yard tokens, so it cannot make halo open a yard-sealed file; but it can read keys halo holds at that moment, and plaintext state in device memory | memory and GPU memory of the live process; swap is not involved (key memory is mlocked or refused; the 16 MB staging buffers are pinned) |
| A stolen admin API key | can spill / restore phase-A entries (halo opens them itself) | cannot open yard-sealed files without a fresh yard token bound to the key; cannot relabel an entry or move it to the master (custody) | - |
| The key in transit | - | the Unix socket (0600, peer uid) or an authenticated encrypted transport only (above) | a misconfigured `--yard-keys-tcp 1` on a plain network |
| Residual copies of K in halo_serve | - | the body, its parsed JSON strings, every header value (the API key), the HTTP layer's buffer (consumed, grown, dropped) and receive buffer are overwritten (cleansed, not mlocked: live request bytes can be swapped); K is decoded straight into `SecretBytes` after a length check | copies a library made internally (the JSON parser's reallocation of strings it was growing, the kernel's socket buffers) |
| Two halo_serve processes on one spill directory | the second refuses to start (`spill-dir.lock`, whatever its keys) | same | - |
| A delayed or replayed revoke | - | bounded by its token epoch: never fences, erases or zeroes state after it | - |

### I/O path

Spill: `Synchronize`, then 4 threads (`spill_threads_`), each with one pinned 16 MB staging buffer (`AllocHost`):
`Download` of a piece (serialized under one mutex: one HIP stream), CRC32C (SSE4.2), `pwrite` with `O_DIRECT` (no page
cache: multi-GB files would evict the n-gram row cache) to a temp file; then the header, `fsync`, `rename`, `fsync` of
the directory. Restore: header and meta read buffered, then 4 threads `pread` `O_DIRECT` pieces, check them, `Upload`.
Disk reads of different pieces overlap each other and the uploads; overlap with other sessions' decode was not built
(the server is one request at a time; the uploads hold the engine's stream for ~3 ms per 16 MB). `O_DIRECT` refused
(tmpfs): buffered fallback.

### Measured: restore against re-prefill (real model, `halo_run --spill-bench`, `/data` XFS on the SN5000, PCIe Gen3 x4, unpinned)

Restore = a new session (the suspended one was dropped) + `RestoreSession`, median of 3, each followed by a step
whose logits are compared bitwise with the uninterrupted session's (all 18 identical). Re-prefill = one `Prefill` of
the same tokens after a warm-up prefill.

| Context | KV | File | Spill (write + fsync) | Restore: predicted / measured | Re-prefill | Ratio |
|---:|---|---:|---:|---:|---:|---:|
| 2,048 | default | 221.9 MB | 67 ms | 0.10 / **0.094 s** | 1.33 s | 14x |
| 12,288 | default | 740.9 MB | 213 ms | 0.31 / **0.289 s** | 8.56 s | 30x |
| 131,072 | default | 6.76 GB | 1.97 s | 2.7 / **2.56 s** | 98.7 s | 39x |
| 2,048 | fp16-only | 171.5 MB | 54 ms | 0.09 / **0.072 s** | 1.32 s | 18x |
| 12,288 | fp16-only | 438.9 MB | 144 ms | 0.18 / **0.162 s** | 8.51 s | 53x |
| 131,072 | fp16-only | 3.54 GB | 1.05 s | 1.35 / **1.22 s** | 97.6 s | 80x |

HARNESS predicted < 0.5 s at 12K against ~8 s: 0.29 s (default) and 0.16 s (fp16) against 8.5 s. The model (file /
3.4 GB/s + 130 us per new 2 MB page + ~20 ms) was within 10%: restores read at 2.7-3.0 GB/s (the page mapping and the
single-stream uploads sit inside the measured time), spills write at 3.0-3.5 GB/s. At 128K the read is the whole cost:
a Gen4 link (BIOS setting, docs/BENCHMARKS.md) would roughly halve it.

**Exactness.** Bitwise: logits after spill + restore equal the uninterrupted session's (gpu_tests below; the bench's
18 restores; `halo_run --prefill-prompt 1 --spill-roundtrip DIR --decode-logits-out` vs the same run without it, 16
decode steps after prose_8888, `cmp`-identical for both KV kinds). The drop path (no file, or a refused one) is a
re-prefill from empty, which equals a cold prefill (`Spill.LongContextRoundTripAndDropPath`).

### Where it fits in SessionCache

`SessionCacheOptions::spill_dir` (halo_serve `--spill-dir`, default `/data/halo-work/spill`) adds a disk tier:
- `List()`, `Spill(key)`, `Restore(key)`: retained entries by a stable key; halo_serve exposes them as
  `GET /admin/sessions`, `POST /admin/sessions/{key}/spill`, `POST /admin/sessions/{key}/restore` (under the generation
  lock, so between requests; behind `--api-key` like every route). This is the hook yard-ctl drives; the policy (which
  session, when) is SR5's.
- `spill_on_evict` (halo_serve `--spill-evict 1`, default off): eviction (count, budget, `KvRoomFor`) spills the least
  recently used resident entry instead of dropping it; a spill that is not done drops it as before.
- `Prepare` treats a spilled entry like a resident one when it holds the longest usable prefix: it restores it (making
  room by evicting others first), checks the blob equals the entry's tokens, deletes the file, then forks or extends it
  as usual. A restore that fails erases the entry and the search runs again (those tokens are re-prefilled).
- Bounds: `max_spilled` (64) and `spill_budget_bytes` (64 GB); beyond them the oldest files are deleted. Files are
  named `s<pid>-<key>.spill`; at startup the cache deletes crash-left temp files and the files of processes that no
  longer run (their index lived in that process); the destructor deletes its own.
- **Memory guard (audit F3).** A spill ends in `DropSession`, which frees the session's ranges, and stream F measured
  that this returns the pages to `AvailableBytes` at once (section 7): `KvRoomFor` sees what a spill frees, exactly as
  with a drop. A spill briefly allocates 4 x 16 MB of pinned staging (freed before it returns).
- Exactness: a restored entry is bit-identical to the spilled one, so exact mode's contract holds
  (`SessionCache.SpilledStatesServeByteIdenticalToColdPrefill`).
- End to end (real model, `streams/SR4/s1.txt`): a 1,237-token chat request parked a 1,216-token checkpoint;
  `POST /admin/sessions/1/spill` wrote a 179.7 MB file (mode 0600, directory 0700) and freed 319 MB of device memory;
  the same request again restored it (`cached_tokens` 1216, prepare 258 ms against 1,335 ms cold) and the entry was
  resident again; an unknown key is a 409.

### Ambiguity (done / not done / unknown) and the caller rules

Every call is synchronous: when it returns, no I/O of it is in flight (threads joined, `fsync` returned), so the
settle bound is the return, or the process's death. One exception: a restore whose failure cleanup could not
synchronize the device poisons the engine; its device writes may still be queued, so its settle bound is the process's
death (halo_serve then stops and exits with status 3; see "Partial restores").

| Call | Done | Not done | Unknown, and the caller rule |
|---|---|---|---|
| `SpillSession` | true: the complete file is at `path` and durable (fsync before rename, rename, fsync of the directory) | false before the rename: `path` is unchanged (absent or the older complete file), the temp file removed | (1) crash during the call: `path` holds the old file or none, plus a temp file no reader opens (deleted at the next start) -> **verify**: a later restore reads whatever complete file is there and its blob says which state it is. (2) crash between rename and the directory fsync, or that fsync failing (false, "may not be durable"): after a power loss `path` holds either the old or the new file, each complete -> **verify** the same way. (3) a torn or bit-rotted file at `path` (not this protocol's doing) -> the checksums refuse it -> **re-prefill**. Cache rule: an entry is marked spilled only on done; otherwise it stays resident (its device state is intact) |
| `RestoreSession` | true: the session's state is the file's, bit for bit; the blob says what it holds | false, `reset` false: refused before any write, the session unchanged; false, `reset` true: a piece failed its checksum, the session is empty | crash mid-restore: the process's state is gone; the file is only read, so a retry is **idempotent** (and if the file is gone, re-prefill). Cache rule: any false erases the entry and its file; the tokens are re-prefilled (a cold prefill) |
| Evict with spill (`spill_on_evict`) | spill done, then the device copy dropped | spill not done: dropped (the state is recomputed if asked for again) | as `SpillSession`; never two resident copies, never a spilled entry without a complete file |
| Delete a file (restore done, trim, destructor) | removed | unlink failed: the file stays, unreferenced | **fence** by name: a dead pid's files are deleted at the next start; a live process never reads a file its index does not name |
| Admin route | 200 `{"ok":true}` | 409 with the reason: nothing changed (spill) or the entry is gone (restore) | the HTTP answer lost: **verify** with `GET /admin/sessions` (spilled flag and key); both operations are safe to repeat (spill of a spilled entry and restore of a resident one are no-ops) |

Fault injection: `spill_fault_` 1 (crash after half the payload, no rename) and 2 (after the fsync, before the
rename) in `Spill.CrashMidWriteLeavesTheOldFileOrNone` and the Hegel machine; bit flips, truncation and extension of
files in `Spill.RefusesDamagedTruncatedAndMismatchedFiles` and the machines. Counters: `SessionCache::spill_counters()`
(spilled, spill_failed, restored, restore_failed, files_deleted); not yet exported as Prometheus metrics (open).

### Permissions and storage

Files are created `0600` (`O_EXCL`), the spill directory `0700`, under `--spill-dir` (default `/data/halo-work/spill`,
not `/tmp`: tmpfs would hold multi-GB files in RAM, the memory a spill exists to free). A spill file holds the whole
conversation's state (its tokens in the blob, and states from which text can be recovered): since SE they are
encrypted (format v2 above; HARNESS SR0 invariant 4).

### Tests and mutants

SE (format v2, keys): `Spill.*` and the cache test above run on v2 (a random test master, KEK per "test" session);
`Spill.RefusesDamagedTruncatedAndMismatchedFiles` runs its cases for both formats, and for v2 adds flips in the model
fingerprint, salt, reserved (authenticated) bytes, key id, wrap nonce, wrapped key, wrap tag, meta tag and meta
ciphertext, no key, wrong epoch, wrong session, the right id with another key, another master, and authentic files
of another model / options / KV kind / position (test hook `spill_test_identity_`), and a v1 file without
`accept_v1`; `Spill.EncryptedFilesNeverReuseANonceOrHoldKeys` reads real files with the format's own rules (every
piece opens under salt || be32(i), no keystream block repeats within or across files though thousands of pieces are
all-zero plaintext, no DEK / KEK / master / blob / `HALOSPL1` in the file); `SessionCache.SpilledStates...` also
rotates the epoch (shred: `DropUnreadable`) and checks every file is v2; `SessionCache.SpillKeysRefuseKeepAndShred`:
no key -> no file (also on eviction), no provider -> refused, plaintext only when chosen, v1 restored only with
accept_v1, a withheld open key keeps the entry, rotation shreds. CPU (`crypt_tests`, no GPU): GCM refuses every
changed bit (Hegel), nonces are salt || be32(i) and never repeat (Hegel), HKDF RFC 5869 A.1, the KEK separates
sessions / epochs / masters, the DEK wrap authenticates the header's AAD (bytes 0-447) and its own ciphertext and tag (448-495; the meta tag at 496 is the meta's), and hides the DEK (Hegel), providers (file modes,
symlinks, the `systemd-creds` path, rotation; opt-in `HALO_TEST_SYSTEMD_CREDS=1`: the real user-scoped TPM
credential). Mutants (tests/mutants.toml, all killed: `streams/SE/m1_cpu.txt`, `m1g_a.txt`, `m1g_b.txt`,
`m1g_c.txt`): `crypt-` / `spill-nonce-constant`, `-header-not-aad`, `-tag-unchecked`, `-dek-in-clear` (CPU and GPU
each), `crypt-kek-ignores-epoch`, `cache-plaintext-on-key-error`; SR4's nine re-run on v2 code (the v1 ones through
the v1 cases).
Phase B: `YardKeys.*` (CPU, a test RSA key and JWKS, tokens in internal/sts/jwt.go's shape): refusals each for its
own reason (expired, audience, issuer, another session, lifetime, future, no `halo_spill`, use, key binding, no epoch,
alg none / HS256, an unpublished signer, a forged signature, malformed bodies); seal / open / revoke epochs, replay,
duplicates, a skipped epoch shreds, fallback to the master; `SessionCache.YardKeysAuthoriseResume` (GPU): no key ->
no file, the entry waits for the open key, open at the seal epoch refused, restore at E + 1, one-shot seal, revoke
erases. Mutants (all killed, `streams/SE/m2_cpu.txt`): `yard-open-at-seal-epoch`, `yard-no-stale-check`,
`yard-jti-replay`, `yard-open-recorded-epoch`, `yard-audience-unchecked`, `yard-key-unbound`,
`cache-yard-open-key-ignored`. End to end (`streams/SE/s1.txt`, real model): `--spill-key tpm` (the user-scoped
credential) and `--yard-jwks`: master spill (v2, 0600) and reuse (cached 1216 tokens); yard seal (and a duplicate),
spill under `yard:` id, restore refused without the open key, open at the seal epoch refused, open at E + 1,
restore, reuse; a fallback spill, rotate (`dropped: 1`); key bytes not in the log.
`MtpSpec.SpillRestoreKeepsDraftHistoryOnRealModel` passes on v2 (`t5.txt`).

Fix round (reviews by GPT-6.1 and GLM 5.3, `streams/SE/reviews/`): `YardKeys.*` rewritten (13 tests):
`Base64UrlIsStrict`; `TokenRefusals` (24 cases, one normalized 401 body, the reason checked in the log field: at-exp
with no leeway, audience arrays, exp <= iat, non-integer or fractional nbf, a duplicate claim, seal without an entry,
revoke through a later epoch, PS256; the body wiped); `ConfigurationFailsClosed`; `JwksKeyChecks` (1024-bit, e = 3,
zero-padded modulus, use enc, key_ops without verify, duplicate kid); `JwksRefreshRemovesKeys` (a removed known kid,
a broken and an empty document deny); `EpochRulesReplayDuplicatesAndShredding`; `RevokeFencesTheSession` (a fresh
seal at E, a fresh open at E + 1, a delayed pre-revoke open token, idempotent repeat, the fence never goes back, GLM's
M1 repro); `ConsumedKeysAreNeverRearmed`; `ExpiryZeroesKeys` (TTL, token exp before TTL, the sweeper thread alone;
`SecretBytes::Live()` proves the key unmapped); `EpochsSurviveARestart` (replayed seal token, open after seal, revoke
fence, a crash between lease and finish, malformed and group-readable state refused); `LeasesAreGenerationBound`
(late finish after revoke + re-registration; restore in flight at a revoke); `NoCustodyDowngrade`.
`SpillKeys.MasterProviderEpochsAndFence` (persisted epoch, never back, malformed sidecar refused, yard-bound refused),
the TPM path by absolute path with the environment checked; `SpillCrypt.SecretBytesFailClosed`,
`FailedOpenWipesAndSealCanWriteElsewhere`, `Sha256Vector`; `Http.UnixSocketIsPrivate`. GPU:
`Spill.CheckpointIdentityDowngradeAndNoLateWrites` (an older authentic checkpoint refused, v1 for a v2 checkpoint
refused with accept_v1, a late-piece failure with 4 workers on 64 KB pieces: uploads happened before it and none after
`RestoreSession` returned); `SessionCache.YardKeysAuthoriseResume` rewritten (entry-bound seal, master fallback only
before the first yard seal, custody after a restore, an in-flight restore at a revoke erased by the fenced cleanup, a
newer checkpoint surviving a late cleanup); `SpillKeysRefuseKeepAndShred` (a fenced file is erased, not kept waiting).
Mutants: 41 CPU (`streams/SE/m3_cpu.txt`, `m3b_cpu.txt`) and 10 GPU (`m3g_a.txt`, `m3g_b.txt`) new or re-targeted, all
killed; SR4's eight small-model mutants re-run, killed (`m3g_c.txt`). Not mutated (no test can observe them):
`EVP_CIPHER_CTX_reset` after each call, the non-in-place DEK wrap on the success path, the full-key (not key-name)
duplicate comparison short of a 64-bit name collision, and `RunPool` joining (the mutant, detaching, is undefined
behaviour). 12K restore after the fix round: v2 median 266.6 ms (v1 268.7; P1 267.1).

Round 3 (second-pass reviews, `streams/SE/reviews/*round2.md`): `YardKeys.SealEntriesAreIncarnationBound`,
`StateHasOneOwner`, `PersistenceOutcomesAreTyped` (injected failure before the rename and at the directory fsync, for
a registration, a new session, a lease, a revoke and the master's fence), `FailedSealRollback` (rollback not durable:
the key disabled, neither a lease nor a fresh-token duplicate uses it; uncertain publish consumes; the revoked-slot
case), `OppositeReservedSlotsSurvive` (both orders), `RevokeDefaultFencesWhatWasSpilled`,
`StateLoadingRefusesUnsafeOrPartialState`, `UnknownKidFloodIsRateLimited`, `RetriesNeverPassTheFirstTtl`,
`KeyNamesNeverDecideEquality` (a forced 64-bit name collision, `KeyNameCollideForTest`);
`SpillCrypt.GcmContextIsResetAfterEveryCall` (`Gcm::IdleForTest`); `Http.SecretBufferLeavesNoCopies` (pipelined key
request, growth, a request cut off mid-body), `Http.UnixSocketOwnership` (live socket refused, successor not
unlinked, stale replaced); GPU `Spill.FailedSyncInRestoreCleanupPoisonsTheEngine`, and the yard resume test gains an
uncertain spill (`spill_fault_ = 3`) and incarnation entry ids. Mutants: 23 new CPU and 2 new GPU (`r3-*`), 13 earlier
ones re-targeted; all killed (`streams/SE/m4_cpu.txt`, `m4b_cpu.txt`, `m4g_a.txt`, `m4g_b.txt`). Still not mutated:
the lease eligibility check and the duplicate branch's used guard (each masked by the rollback's key disabling: defense
in depth), and the out-of-place DEK wrap on the success path. 12K restore: v2 271.3 ms median (v1 268.8; b3.txt).

Round 4 (third-pass reviews): `YardKeys.RevokeIsBoundedByItsTokenEpoch` (a delayed revoke at T after a spill and after
a registration at T + 1: both survive), `ExpiryIsCheckedUnderTheLock` (the lock held past exp: refused; held but not
past exp: the key still ends at exp), `MarkerlessStateIsUpgraded`, `StateDirectoryIsPinned`, `JwksFileIsChecked`,
`ConcurrentMasterAndYardPersistence` (also under ThreadSanitizer: `build/cpu-tsan`, crypt_tests 39/39 clean), the
master's unconfirmed epoch, a 128-bit incarnation and canonical entry ids; `Http.UnixSocketProbeFailsClosed` (full
backlog); GPU `SessionCache.PoisonedRestoreQuarantinesItsMemory`. Mutants (`streams/SE/m5_cpu.txt`, `m5b_cpu.txt`,
`m5g.txt`, `m5g2.txt`): 13 new `r4-*` killed (12 CPU, 1 GPU), the re-targeted ones and every earlier CPU mutant of
these files re-killed. A ThreadSanitizer positive control for the old plain fault-hook global could not be made: TSan
treats file-descriptor number reuse as synchronization, which orders the two writers, so no planted race in that path
is reported; the fix (an atomic exchange) stands on its own.

SR4:

- `Spill.RoundTripIsBitIdentical` (Hegel, 40): random histories of prefill chunks and steps, both KV kinds, pieces 16 KB
  (sections cut into many pieces) or 16 MB, 1-4 threads; restored into the same session (after rewriting it), another
  session, or a fresh engine; then steps and prefill chunks memcmp-equal to the uninterrupted engine; file mode 0600.
- `Spill.LongContextRoundTripAndDropPath`: 3,333 positions (indexer window, many blocks), both kinds, into a fresh
  engine; the drop path equals the cold prefill.
- `Spill.RefusesDamagedTruncatedAndMismatchedFiles`: truncated (a piece, to the header, empty), extended, magic, header
  bit flips (position; a field only the header CRC protects), meta bit flip, payload bit flips (first and last piece:
  session reset), other model, options, KV kind, position, version (re-checksummed headers), an fp16-only engine, a
  precision option flipped, a smaller max context; refused ones leave the session stepping bitwise like the oracle.
- `Spill.CrashMidWriteLeavesTheOldFileOrNone`: both faults, with and without an older file (which still restores the
  older state exactly); `SpillCleanTemps` removes the temp.
- `Spill.SuspendResumeEvictCrashMachine` (Hegel, swarm, 64 KB pages): three logical sessions; step, prefill, suspend,
  crash while suspending, resume (restore or re-prefill), evict, flip a file bit. After every resume the state is what
  the last done spill's blob says, an undamaged file always restores, and the next step equals a fresh replay.
- `SessionCache.SpilledStatesServeByteIdenticalToColdPrefill` (Hegel, 40): exact cache, max_retained 1-2 with
  spill-on-evict, admin spills / restores / bit flips between turns; every request's tokens equal a cache-off engine's.
- `MtpSpec.SpillRestoreKeepsDraftHistoryOnRealModel` (real model + MTP sidecar, 300 and 2,600 tokens).
- Mutants (tests/mutants.toml, 9 of 9 killed; `streams/SR4/m1a.txt`, `m1.txt`): `spill-skips-recurrent`,
  `spill-wrong-position`, `spill-fp16-copy-kept-valid`, `spill-no-payload-checksum`, `spill-no-header-checksum`,
  `spill-options-unbound`, `spill-writes-in-place`, `spill-mtp-rows-skipped` (new suite `gpu-real`: the real model and
  the MTP sidecar), `cache-uses-unrestored-entry`. The Hegel tests here run without a shrink phase (as the other
  session machines), so a mutant fails in seconds; `tools/mutants.py` builds with `HALO_BUILD_JOBS` jobs when set.

### Open

- Uploads under one mutex on the engine's stream: a second stream (or `UploadAsync` with per-thread staging) would let
  a restore run beside another session's decode; and would lift 2.8 GB/s toward the disk's 3.4.
- Spill files outlive nothing: the index is in the process. Resuming after a halo_serve restart needs the index (token
  lists) persisted next to the files (the blob already holds the tokens; the canonical / checkpoint flags would join it).
- Prometheus counters for the spill outcomes; SR5's policy (what to spill, when) and yard-ctl's one-shot resume token.

## 11. Sessions under MTP speculative decoding (stream MS, 2026-10-07; working files `/data/halo-work/streams/MS/`)

`halo_serve --mtp-model` (docs/MTP.md "Server integration") decodes a qualifying request in cycles that run the
step's token plus up to k drafts and emit 1 + accepted tokens. What that changes for a session:
- **The session holds what the engine ran.** A cycle commits the token and its accepted drafts; when the
  generation ends inside that run (a stop token, a stop string or a departure between a cycle's tokens), the
  engine has run drafts that were never emitted. `RunDecode` (src/server/decode_loop.cpp) records exactly the
  run tokens, so the session's token list may extend past the reply; the next request's prefix check decides reuse
  as before (a mismatch only loses reuse). Exact mode drops the live session anyway; fast mode retains it with
  those tokens. `SpecDecodeStats::ran_past_end` counts such endings.
- **KV room**: `KvRoomFor` is asked for prompt + budget + `kMaxDrafts` positions (a verify writes up to k rows past
  the last emitted one). The draft block's own cache (max-context rows, 2.74 GB with its weights) is reserved at
  startup.
- **Departures** (section 9): checked before every cycle and between a cycle's tokens; a departure leaves the
  session as a request with `max_tokens` = tokens emitted would, plus the cycle's accepted drafts. The Hegel
  machine `ServingModelUnderDeparturesStallsAndRetries` runs with and without speculative cycles.
- **The draft block follows one session** (`mtp_owner_`): each request re-anchors it with `MtpTrackFrom` before its
  prefill, at max(reused, prompt - W); batched sessions (StepBatch) stay AR.
- **A fault this exposed: the GPU's address translation lags the HIP mapping calls** (root cause: stream RC,
  2026-10-07; working files `/data/halo-work/streams/RC/`, experiments and predicted vs actual in its PREDICTIONS.md).
  MTP serving hit an intermittent GPU write fault (amdgpu permission fault, then a sticky `hipErrorIllegalAddress`
  that failed every later request) in the next request's `ForkSession`: the copy kernel, writing a fresh page it had
  mapped microseconds before, through the correct address. Stream MS's fix (6240e69: `ReleaseRange` keeps ranges
  reserved and reuses them, no `hipMemAddressFree`) removed it, by correlation. What RC established:
  - **Not a stale pointer** (the feared alternative, which the pool would have turned into silent corruption): in
    all 5 logged faults the faulting page is the fork's freshly mapped copy target and the writer the copy kernel
    (address log with monotonic timestamps against the kernel's fault addresses). With released ranges quarantined
    and a sentinel page mapped wherever they had pages (`HALO_DIAG_RANGE=poison`, checked at every reserve), 3 bench
    runs at the faulting configuration gave no fault and no write into any released range; MS's never-free run
    (released ranges left unmapped, so a stale write would fault) gave 0 of 4. The detector is positive-controlled
    by `SessionCache.PoisonModeReportsAWriteIntoAReleasedRange`. (Stale reads are not covered.)
  - **A runtime / driver race (A).** A 2 ms wait after every page mapping removed the fault with the runtime still
    freeing and reusing addresses (0 of 4 runs, against 2 of 3 without at W 0). A minimal program
    (`streams/RC/ana/vmm_stale.cpp`, plain HIP, no engine) shows the same race deterministically: map page A at an
    address, write it, unmap it, release it, map page S at the same address, and write at once (a kernel with a raw
    pointer, or `hipMemsetAsync`): 100 of 100 writes land in the old, released page A, and reads through the address
    return them; S is untouched. `hipStreamSynchronize(0)`, `hipDeviceSynchronize`, a blocking stream or a yield do
    not help; time does (>= 60 us between unmap and remap, or >= 100 us after the remap: 0 of 100; a 50 us spin 1 of
    100). ROCm 7.2.3 (nix), kernel 7.0, gfx1151. Not narrowed below the HIP API (ROCr `hsa_amd_vmem_*`, KFD and the
    amdgpu TLB flush are candidates); worth an upstream report with that program. Stream C's not-kept handle pool
    (section 7: released handles remapped at another address, GPU faults at 2 MB pages in 2 of 3 swarm runs) was
    probably the same lag.
  - **Fix (rt::HipDevice, "VMM settle")**: no GPU work (launch, graph launch, fill, upload, download) is submitted
    until `HALO_VMM_SETTLE_US` (default 1000) after the last page map or unmap; `ForkSession` maps all its pages
    before launching its copies, so a fork waits once. With the settle and the runtime freeing and reusing ranges
    (the pool off): 0 of 4 runs at W 0 (and the fork's TTFT unchanged within run-to-run noise: greedy TTFT sums
    14.6-17.0 s vs 17.4 s). Test `SessionCache.RemappedPageSeesTheNewPageAtOnce` (20 of 20 misdirected with
    `HALO_VMM_SETTLE_US=0`), mutant `rt-vmm-settle-off`.
  - **The range pool is gone** (released ranges go back to the runtime again). It removed the fault by keeping
    forks off freshly freed addresses, but it leaked: the memory of every page slot a range ever mapped stays
    charged until the range is freed (ROCm 7.2.3, `streams/RC/ana/vmm_gtt.cpp`: 4 GB of unmapped, released pages in
    a kept range stayed in GTT; a page mapped at the same slot of the same range reuses the charge, any other slot
    adds to it), so pooled ranges held their high-water pages and 128K contexts ran out of memory in `hipMemCreate`
    on main (stream SE). With ranges freed and the settle: the 128K spill bench restores 6.76 GB in 2571 ms with
    bit-identical next-step logits (SR4 before the pool: 2.56 s), and MTP serving at W 0 shows no fault (stream RC
    E8: 0 of 4; E10, the final defaults with no diagnostics: 0 of 4, greedy TTFT unchanged). Test
    `SessionCache.ReleasedRangesGiveTheirMemoryBack` (24 ranges of different sizes, 128 MB each: within 512 MB; the
    pool keeps 3072 MB), mutant `rt-range-kept-on-release`. The pool survives only
    as `HALO_DIAG_RANGE=pool` for A/B.
  - **Why only under MTP, and mostly at W 0 with graphs.** Every fault was on an address last backed by one of the
    850 MB raw expert copies freed after repacking, then only by reserved-and-freed ranges that never mapped it. MTP
    serving repacks the draft block's experts and frees three such buffers just before serving, so serving-time KV
    ranges land there: ~100 fork copy pages a run in that class (all 5 faults were in it, while it is ~7% of fork
    copy pages); AR serving, whose raw copies are freed earlier and whose addresses are taken by long-lived
    allocations, has ~1 a run (RC: 0 of 3 AR runs with the runtime freeing ranges). Why that address history makes
    the lag show as a fault (an invalid translation) rather than a misdirected write is not established. The rate
    was higher at W 0 with graphs (MS 5 of 6, RC 2 of 3) than at W 128 (RC 0 of 4, logging heavily) or eager (1 of 8):
    timing.
  - Diagnostics kept, off by default: `HALO_DIAG_VA=1` (every allocation, range, page and graph event on stderr with
    `CLOCK_MONOTONIC` microseconds, the clock of the kernel log and of `AMD_LOG_LEVEL`), `HALO_DIAG_RANGE=free|pool|
    poison` (free is the default), `HALO_DIAG_MAP_DELAY_US`, and `halo_vmm_remap_probe` (tools/vmm_remap_probe.cpp:
    the runtime race in 80 lines of plain HIP, for an upstream report). Mutant `rt-poison-check-blind`.

**Prior art (searched 2026-10-07).** The same class is public on gfx1201 (RDNA4): ROCm/rocm-systems#11693 (freshly
mapped VM memory reads zeros; map returns before the mapping is visible to GPU work; ~2 ms host delay works around it;
open) and ROCm/legacy-rocm-build#6692 (reusing a VA after hipMemUnmap corrupts live VMM buffers). Our gfx1151 data
(writes land in the released page; ~60-100 us suffice) is drafted as a comment for #11693 in
/data/halo-work/platform/upstream/rocm-vmm-remap-gfx1151.md, not posted.

## 12. What a request added to the cache (stream G2, 2026-10-07; docs/ANTHROPIC.md)

Reuse semantics are unchanged. The Anthropic routes report the cache in the API's usage fields, so
`TextGenerator::Result` gained `cache_write_tokens` beside `cached_tokens`: after `SessionCache::Finish`,
`RetainedPrefix(prompt)` (the longest prefix of the prompt a retained state, resident or spilled, holds whole: a
checkpoint whose tokens are a prefix of the prompt, or a finished fast-mode session that extends it) minus the
tokens the request reused. Exact mode: the checkpoint parked at the generation prompt's start, rounded down to 64,
minus the reuse (0 when the prompt is shorter than 64 tokens or eviction dropped the checkpoint); fast mode: the
whole prompt. `cache_control` breakpoints do not steer it. Tested by `FlashNextGenerator.CacheUsageCountsWhatIs
ReusedAndWhatIsRetained` (mutant `anth-cache-write-is-prefilled`).
