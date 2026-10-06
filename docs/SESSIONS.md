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
  session's pages (stream C: before that fix a reset parent wrote into pages its forks shared; regression test
  `Sessions.ForkSurvivesParentResetAndRestore`).
- **Pluggable layout.** A region is `(row_bytes, row_tokens, pages)`; today's kinds are K / V (2048 B a token),
  indexer keys (512 B), pooled blocks (512 B per 4 tokens), fp16 K / V (1024 B). An fp16-only KV layout (stream
  for full context) is a different kind table with the same paging.
- **Fallback.** A backend without page mapping (HRX today; or `HALO_NO_PAGING=1`) allocates each region whole
  at `max_context` and a fork copies the written rows.
- `HALO_PAGE_BYTES` overrides the page size (tests use 64 KB so short sequences span several pages).

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

## 4. Tests

- `Sessions.SwarmMatchesFreshReplay` (Hegel, swarm: each run enables a random subset of the rules): new / fork /
  bind / step / prefill / StepBatch over a random subset in random order / drop / reset / graphs on-off; after
  every step, prefill and batch entry the session's logits are compared with `memcmp` against a fresh engine
  replaying that session's own operations. 60 cases, at 2 MB and 64 KB pages.
- `Sessions.ForkedSessionsDivergeIndependently`: 300-token prefix, fork, both diverge past the shared pages;
  memory accounting shows shared pages (64 KB pages: 14.0 MB mapped, 10.4 MB exclusive).
- `Sessions.BatchedStepsEqualSingleSteps`: B = 2..8, logits and device sampling (temperature 0.8, top-p 0.9),
  12 steps, each entry against the same session on a second engine.
- gpu_tests 62 pass with `HALO_SMALL_QUANT_DIR`; decode logits (66 steps, short and 7K context) and prefill logits
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
layer; forks, snapshots and paging go by the kind table. `--kv-f16 1` (`kv_f16_only_`, default off) drops the
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
128K-token document its NLL equals the default kinds' (paired -0.0007 +- 0.0030). Long-range checks, speed and
the open items: `/data/halo-work/streams/C/STATE.md`.

