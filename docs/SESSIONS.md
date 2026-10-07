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

**The case for `--kv-f16 1` as the long-context serving default** (not flipped: the user decides).

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
