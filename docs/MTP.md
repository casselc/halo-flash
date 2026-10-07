# P5 MTP: acceptance measured, speedup projected (stream M, 2026-10-06)

Measurement stage only: the draft block runs on the GPU (`FlashNextGpu::LoadMtp` / `MtpStep`, src/engine/mtp.cpp:
ReferenceModel::MtpStep's graph with the trunk's helpers on the sidecar's weights, eager). There is no verify
path; the trunk is unchanged (decode and prefill logits byte-identical to main). Raw logs:
`/data/halo-work/streams/M/m1_*.txt`, `m3_*.txt`, projection `project.py`.

**Draft block vs CPU reference** (same trunk hidden): relative L2 4e-7 .. 1.9e-5, same argmax, 4 positions.
Note: the reference block runs its own QSA indexer (its own weights); the plan's "reuse the target's QSA indices"
is not what the reference does, so the draft follows the reference.

**Protocol** (`halo_run --mtp-model ... --mtp-prompts ... --n 200 [--mtp-temp 0.8]`): pass 1 decodes 200 tokens AR
and keeps every target distribution; pass 2 replays token by token, runs the block's base step at every position
(its attention cache is filled exactly as a live decoder would) and at each of 196 generated positions drafts 4
tokens, then rewinds. Greedy: the block's own argmax chain against the AR tokens (an exact simulation of greedy
verification). T = 0.8: the chain teacher-forced on the sampled target path; acceptance per depth sum_x min(p, q)
(the speculative-sampling acceptance probability), E[tokens] from their running products (an estimate: it treats
depths as independent and conditions on the target path).

| Text | Greedy conditional acceptance d1 / d2 / d3 / d4 | E[tokens/cycle] k = 1..4 | T 0.8 acceptance d1..d4 | E k = 1..4 |
|---|---|---|---|---|
| code (engine source) | .67 .67 .62 .64 | 1.67 2.12 2.40 2.58 | .82 .76 .73 .71 | 1.82 2.47 3.00 3.42 |
| chat (template, thinking on) | .77 .76 .76 .69 | 1.77 2.35 2.79 3.10 | .77 .73 .67 .63 | 1.77 2.35 2.77 3.06 |
| doc (PLAN.md excerpt) | .68 .67 .64 .62 | 1.68 2.14 2.44 2.62 | .82 .76 .74 .70 | 1.82 2.45 2.96 3.34 |
| **mixed (mean of the three)** | | **1.71 2.20 2.54 2.77** | | **1.80 2.42 2.91 3.27** |
| prose, prose2 (word-salad prompts; greedy continuation repeats) | .92-.96 | 1.9 2.8 3.6 4.4 | .44-.61 | 1.6 1.9 2.1 2.1 |

**Costs** (pinned): AR 37.0 ms/token. Draft step: block 1.29 ms headless, 4.2 ms with the full 248K head and
logits download (a 32K trimmed head: ~1.7 ms, estimated). Verify of m positions, (a) today's exact batched kernels:
StepBatch measured at B = 1..5 (37.0, 52.9, 66.7, 80.3, 97.4 ms), corrected for the smaller expert union of
consecutive positions (0.77 ms per expert-union unit; union 15.8 / 20.6 / 24.7 / 28.3 vs random 18.8 / 27.0 / 34.7
/ 41.8) plus 0.9 ms per extra position for the DeltaNet rollback snapshot: 51.5 / 63.6 / 75.3 / 90.6 ms for m = 2..5.
(b) a batch-shaped single path: B = 1 cost X, each extra column = its expert union increment x 0.77 + 2.5 ms per-entry
work (measured at B = 8) + 0.3 ms dense compute (WMMA rate) + 0.9 ms snapshot: X + 8.2 / 15.6 / 22.4 / 28.9 ms.
Cycle = verify(k + 1) + k drafts + catch-up (one headless block step per emitted token).

| Projected speedup vs AR (mixed text) | greedy | T 0.8 |
|---|---:|---:|
| (a) exact batched, full head | 1.09x (k 1-2) | 1.19x (k 2) |
| (a) exact batched, 32K head | 1.17x (k 2) | 1.28x (k 2-3) |
| (b) batch-shaped, X = 37, full head | 1.28x (k 2) | 1.42x (k 3) |
| (b) batch-shaped, X = 37, 32K head | 1.39x (k 3) | 1.58x (k 3) |
| (b) batch-shaped, X = 40, 32K head (speedup vs today's 37.0 AR) | 1.33x (k 3) | 1.52x (k 4) |

**B = 1 cost of a batch-shaped single path (estimate, not measured):** 37-41 ms, central ~39 ms (0-10% over today's
37.0). Basis: it reads the same bytes; gufo's batch-capable decode runs C1 at 38.5 ms on this machine; WMMA GEMVs at
one column waste tile rows but stay bandwidth-bound. It changes the single path's arithmetic (fp16 activations or
reordered sums), so it needs the precision gate. A skeleton probe (WMMA GEMV, qkv and 2560 x 6144 shapes, T = 1..8,
in the engine) should precede any build.

## Round 2 (2026-10-06): batch-shaped probe, trimmed draft head, why acceptance is low

### Trimmed draft head (draft logits only; verification would keep the full head)
32,768 ids: the most frequent tokens of a 4.4M-token corpus (ds4's Qwen quality-fixture prompts and responses, gufo
and ds4 sources and docs; none of the test texts), filled by lowest id, plus the chat-control tokens
(`halo_run --count-files`, `streams/M/vocab32k.bin`). The draft head is the same Q8 rows gathered into a 32K-row
matrix (`SetDraftVocab`), same GEMV.
- Draft step with head and logits download: **1.77 ms** (full head 4.2 ms); the block alone 1.29 ms.
- Acceptance, greedy, depth 1..4 E[tokens/cycle] (full -> trimmed): code 2.58 -> 2.57, chat 3.10 -> 3.03, doc
  2.62 -> 2.62: unchanged within noise (the argmax almost never leaves the subset).
- T 0.8 (sum min(p, q), q renormalized over the subset): code 3.42 -> 3.49, chat 3.06 -> 2.80, doc 3.34 -> 2.67:
  the probability mass outside the subset is lost under sampling (doc depth-1 0.82 -> 0.69). A larger or
  domain-matched subset would trade back some of the 2.4 ms saved.

### Batch-shaped single path: skeleton probe in the engine (PROBE ONLY, `halo_run --wmma-probe`)
Each decode step (eager, kernel timing on) is followed by the batch-shaped kernels for every layer on the real weights
(dense Q8 projections incl. mixers, shared expert, LM head; Q4_K / Q5_1 routed experts grouped by expert with routing
from real traces), B = 1..8, timed in the same window as today's kernels. Today's dense kernels in that window:
21.5 ms/step, routed experts 8.0, the rest 6.7 (eager 36.2-37.7).

| Probe kernel (dense) | Numerics vs today | Dense ms at B=1 | per added column |
|---|---|---:|---:|
| v1 fp16 WMMA, lane = row, 32 B a lane a step | fp16 weights fp16(q d) and activations, WMMA sum order, split-K + ordered reduce | 47.4 | +0.7..0.85 |
| v2 tile through LDS, 256 B row-contiguous loads, next tile prefetched | same | 43.3 | +1.8..2.2 |
| v3 v2 + scales loaded with the tile | same | 43.3 | +1.7..1.9 |
| v4 W8A8 iu8 WMMA (no weight dequant; activations int8 per 32 like the routed experts' Q8Act) | int8 activations (precision change), exact int32 block sums, fixed fp32 block order | 47.8 | +0.2..0.74 |
| today's decode GEMVs | | 21.5 | (exact batched: ~5.5) |

Routed experts, batch-shaped fp16 WMMA grouped by expert: 10.1-11.0 ms at B=1 (today 8.0); per added column the
union's bytes: +5.2..9.0 ms (0.74 ms per expert-union unit), the same as today's exact grouped kernels.
Composed whole step: **X = 60-66 ms** graph-equivalent at B=1 (today 37.0); measured per-column increments B=2..8: dense
+0.2..2 ms, experts by union, per-entry work +2.5 (from the batched decode measurements), snapshot +0.9.
None of four designs got a dense batch-shaped GEMV near today's single-token speed. Not VALU (W8A8 removes the
dequantization and is no faster), not the scale loads (v3). Working hypothesis, untested: a wave reads 16 rows at
once (16 concurrent DRAM pages) where today's GEMVs read 2-4: ROOFLINE section 10's page-locality effect. A design
that keeps 2-4 rows a wave would have to fake the 16-row WMMA tile (e.g. columns on the M side), not probed.
**At the measured X the batch-shaped path loses** (re-projection below); it is attractive only if X comes within
~10% of 37.

### Re-projection (verify models as before; drafts at the trimmed head's 1.77 ms; catch-up 1.29 ms a token)

| Text / decoding | (a) exact batched, full head | (a) exact batched, 32K head | (b) measured X 61.4 (v3) | (b) measured X 66.2 (v4) | (b) X 37 (hypothetical) |
|---|---:|---:|---:|---:|---:|
| mixed code/chat/doc, greedy | 1.09x | 1.16x | 0.96x | 0.95x | 1.37x |
| mixed, T 0.8 | 1.19x | 1.21x | 1.03x | 1.03x | 1.44x |
| GSM8K / HumanEval-style chat, greedy | 1.38x | **1.50x** (k 3) | 1.38x | 1.39x | 1.90x |

### Why acceptance on our texts is low (diagnostics)
1. **gufo on identical prompts** (same weights, greedy, its own MTP, draft limit 1, counts from its /metrics): chat 85
   of 112 = 0.759 (ours 0.765), doc 80 / 120 = 0.667 (ours 0.684), GSM8K-1 98 / 102 = 0.961 (ours 0.954),
   HumanEval-1 96 / 103 = 0.932 (ours 0.883). gufo stops the raw code continuation at once (EOS first). With draft
   limit 4 (its adaptive policy; gufo refuses a fixed depth) it emits per verification round: chat 1.89, doc 1.77,
   GSM8K-1 4.17, HumanEval-1 3.75 tokens (ours, depth 4: 3.10, 2.62, 4.38, 3.88). Same level and the same text
   dependence:
   no wiring bug of ours alone. gufo's draft block is the same port as our reference, so this does not clear a bug
   both share (item 2).
2. **Wiring audit.** HF transformers' Qwen4Exp modeling has no MTP module (`_keys_to_ignore_on_load_unexpected =
   [r"^mtp.*"]`), so the audit used the converters (ds4 `qwen4_exp_convert.py`, `qwen4_pack_to_qwen4exp.py`) and
   the checkpoint names: eh_proj = [fc_embedding | fc_hidden] (embedding first, confirmed), hidden input is the
   10,240-wide trunk residual (`pre_fc_norm_hidden` is 10,240 wide; before the head mixer), position: MTP position i
   takes trunk hidden i and token i+1 (gufo `DraftReplay` agrees). One open point: whether `pre_fc_norm_hidden`
   normalizes the four streams together (our reference and gufo) or each stream on its own (as every trunk mixer
   norm does). Ablation (`HALO_MTP_HNORM_GROUPED=1`, one RMS per stream): depth-1 acceptance higher on 8 of 9 texts
   (chat 0.765 -> 0.837, doc 0.684 -> 0.719, GSM8K 0.954/0.913/0.944 -> 0.964/0.954/0.964, HumanEval
   0.883/0.872/0.913 -> 0.939/0.929/0.959; code 0.668 -> 0.653); E[tokens/cycle] at depth 4 on the six paper-style
   prompts 4.06 -> 4.28. Suggestive (sign test p ~ 0.02), not proof; needs the authors' MTP code or a larger eval.
3. **MTP KV and indexer history.** Covered: the measurement replays every prompt position through the draft block
   (base step at each position, `halo_run` pass 2), so its attention KV, indexer keys and pooled blocks hold the
   whole prompt. gufo does the same (`DraftCatchUp` on prefill). At these lengths (< 2,048 tokens) QSA keeps every
   position, so the indexer cannot drop history either.
4. **Precision.** The sidecar is Q8_0 (19 tensors: projections, experts, eh_proj), F32 (11: norms, router), BF16 (2:
   indexer). The draft block's arithmetic is not the cause: the CPU reference (fp32 on the dequantized weights) on
   the GPU run's trunk hiddens drafts the same tokens as the GPU at every depth (160 / 160 on chat, 240 / 240 on
   code). The trunk with float expert activations (`--int8-moe 0`, the exact path): code 0.668 -> 0.694, chat 0.765
   -> 0.765, doc 0.684 -> 0.750 at depth 1 (E4 2.58 -> 2.72, 3.10 -> 3.02, 2.62 -> 2.78): within noise to slightly
   higher; its generated text differs, so this is not paired. The unquantized (BF16) sidecar is not on this machine.
5. **Paper-style prompts** (GSM8K and HumanEval in the chat template, thinking on, greedy): depth-1 0.87-0.95,
   E[tokens/cycle] at depth 4 **3.73-4.38 (mean 4.06)**, the paper's ~4.1. Our earlier texts were raw continuations of
   a C++ file and a design doc (no chat template) and one open-ended chat question: the low acceptance is the text,
   not the engine.

## Decode path (2026-10-06): SpecStep, greedy output identical to AR

`FlashNextGpu::MtpTrack` / `SpecStep` (src/engine/spec_decode.cpp), off unless a caller uses them (`halo_run --spec
1`). One cycle: the draft block proposes k tokens (its pending base step on the trunk hidden of p - 1, then its own
chain), the trunk verifies [token, d1 .. dk] as k + 1 rows of one RecordBatch pass with every entry the bound session
(each row's arithmetic is the single step's), the device greedy sample of each row (StepSample's kernels) is compared
with the drafts and the longest agreeing prefix is kept. Rollback without replay: DeltaNet state, conv windows and the
PLE history are copied after every row but the last (`verify_snap_`); KV and indexer rows past the committed position
are overwritten later. The draft block then catches up on the committed rows from their true hiddens (`bb_.res`).
Sampled cycles draw drafts from the top 64 of the trimmed draft logits and accept with p / q and a residual draw
(gufo's `mtp_sampling`, ported; host unit tests in `tests/core/mtp_sampling_test.cpp`); penalties are not supported
there.

First measurements (pinned, 9 prompts: six GSM8K / HumanEval chat prompts, one open-ended chat, raw code, raw doc):
fixed depth 3, 256 tokens each: 32.99 vs AR 27.03 tok/s = 1.22x, paper-style chat 1.12-1.55x (mean ~1.38x), raw
code and doc 0.97x (acceptance 0.41-0.43); gufo's adaptive policy (ported) 1.19x: it backs off to AR for 16 tokens
after any drop in its depth estimate. 7K prompt (`prose_long`, 384 tokens) 1.10x. Greedy output identical to AR in
every run (9 x 512, 7K x 384, 9 x 256).

## Round 3 (2026-10-06): measured-cost depth chooser, tests, sampling audit

### Cost calibration (fixed depths 0..5, 9 prompts x 256, pinned, per-cycle traces, `streams/M/r1_*`)
Fit ms = C[k] + c x accepted + r x [a rejection happened] over 4,284 cycles (`fit_costs.py`):

| | k 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| C[k] predicted (verify model above + 1.77 ms trimmed drafts) | 38.4 | 53.4 | 67 | 80.6 | 97 | 111 |
| C[k] measured | 37.3 | 54.15 | 69.73 | 84.75 | 102.25 | 113.14 |
| total tok/s (AR 26.78) | 25.88 | 32.40 | **33.55** | 32.32 | 30.69 | 29.51 |

Catch-up per accepted draft c = 1.37 ms (predicted 1.3), rollback copy r = 1.46 ms (not modelled before). Depth 0
costs AR + 1.37 ms (the draft block follows every step). The best fixed depth depends on the text: depth 1 on raw
code / doc (1.10x, where depth 3 loses), depth 2-5 on the chat prompts; no fixed depth is best overall (d2 1.25x).

### Depth chooser (`MtpDepthChooser`, src/engine/mtp_depth.hpp; `halo_run --spec-depth -2`, the default there)
Per session: depth = argmax over k of E[tokens](k) / E[cost](k) with the measured C[k], c, r; per-depth acceptance as a
Beta with forgetting (lambda 0.9) around a persistent prior inherited from the shallower depth; depth 0 when no k beats
AR, with a depth-1 probe every 8 AR tokens (backing off while AR keeps winning); limit 4.
Prediction: total >= the best single fixed depth (1.25x), each prompt within 2% of its best fixed depth.
Result (pinned, 9 x 256, `streams/M/r2_chooser.txt`): **34.70 vs AR 26.81 tok/s = 1.294x**, greedy output identical to
AR on all nine.

| Prompt | AR tok/s | chooser | x AR | best fixed depth (r1) | gufo policy |
|---|---:|---:|---:|---:|---:|
| gsm1 | 26.92 | 34.47 | 1.28 | 32.05 (d2) | 37.05 |
| gsm2 | 26.91 | 38.44 | 1.43 | 37.84 (d3) | 36.60 |
| gsm3 | 26.93 | 40.55 | 1.51 | 41.30 (d3) | 37.26 |
| he1 | 26.87 | 37.96 | 1.41 | 38.54 (d5) | 37.08 |
| he2 | 26.87 | 34.90 | 1.30 | 35.51 (d2) | 34.30 |
| he3 | 26.89 | 39.32 | 1.46 | 39.93 (d4) | 38.85 |
| chat (open-ended) | 26.93 | 31.37 | 1.17 | 31.44 (d2) | 27.88 |
| raw code | 26.63 | 29.73 | 1.12 | 29.47 (d1) | 27.99 |
| raw doc | 26.39 | 29.61 | 1.12 | 29.50 (d1) | 27.29 |
| **total** | 26.81 | **34.70** | **1.294** | 33.55 (d2 everywhere); 34.43 per-prompt oracle | 33.18 |

The prediction held: +3.4% over the best single fixed depth, +0.8% over the per-prompt oracle, every prompt within
1.8% of its best fixed depth; raw code / doc go from a loss at depth 3 (0.97x) to 1.12x. Paper-style chat: mean
1.40x (1.28-1.51x), short of the ~1.50x projected for the trimmed head at depth 3. Depth histogram: 0: 44, 1: 322,
2: 259, 3: 275, 4: 40 cycles.

**P5 gate "at least 1.5x AR on mixed text": not met** (1.29x over the set; 1.12x on raw text). The verify pass is
the limit: k + 1 rows cost 1.45-2.7x a single step (C[k] / C[0]) with today's exact per-column kernels (docs/
SESSIONS.md); a cheaper multi-row verify (round 2's batch-shaped probe lost at X 60-66 ms) is the only lever left
at this acceptance.

### Server integration: proposal only (not built)
Today `FlashNextGenerator` (src/server/flash_next_generator.cpp) runs one sequence: prefill, then StepSample per
token with device sampling (penalties and RNG in a `SamplerState`), prefix reuse when the new prompt extends
`history_`. Proposed, behind `halo_serve --mtp-model F [--mtp-vocab V]`, **off by default**:
1. **Load and track.** Load the sidecar (2.8 GB) and the trimmed head at startup; `MtpTrack(true)` after every
   `Reset`, so the draft block follows from position 0 and prefix reuse keeps it in step (`MtpPending` holds while
   the engine is only ever advanced by Prefill / Step / SpecStep).
2. **Prompt cost first.** The draft block's prompt history runs as eager single steps, 1.3 ms a prompt token: +2.7 s
   TTFT at 2K, +10 s at 8K, against 1.25 s / 5 s of trunk prefill. Before serving, either (a) a batched draft
   prefill (the block's projections as GEMMs over the chunk, like the trunk's), or (b) a history window: catch the
   block up on only the last W prompt tokens (e.g. 512). (b) changes acceptance only, never the output (every token
   is verified), so it is safe to ship first; measure acceptance against W.
3. **Decode loop.** Replace StepSample by `SpecStep(token, chooser.Depth(), ...)` while the request qualifies:
   greedy (temperature 0) at any setting; sampled only without penalties (repeat / frequency / presence), else
   StepSample as today. A `MtpDepthChooser` per generator, reset per request (its prior is the persistent part).
   Cap drafts at `budget - completion_tokens - 1` so a cycle never emits past max_tokens.
4. **Stops and history.** A cycle can emit several tokens; on a stop at emitted index j, emit nothing after it.
   The engine has run the cycle's committed rows (token, d1 .. da) past the stop, so `history_` must record exactly
   what the engine ran (token plus the accepted drafts), not what was emitted; the next request's prefix check
   then decides reuse as today (a mismatch resets: correct, only the reuse is lost).
5. **KV room.** `KvRoomFor` must reserve `kMaxDrafts` rows beyond prompt + budget (the verify writes k rows ahead).
6. **Sampling reproducibility.** Sampled spec mode draws on the host from the request's RNG stream in a different
   order than device StepSample, so a seeded request gives a different (equally distributed, per the audit) text with
   MTP on. Document it; keep `--mtp-model` off when seed reproducibility against AR matters.
7. **Metrics.** Per request drafted / accepted / cycles and the chooser's depth histogram in `/metrics` and the
   observer.
8. **Tests.** A server-level test on the small fixture is impossible (no sidecar); gate a real-model test like
   `MtpSpec.GreedyEqualsArOnRealModel`: the same request with and without `--mtp-model`, greedy, identical text,
   including a stop inside an accepted run and a max_tokens cut mid-cycle.
Batched sessions (StepBatch) stay AR: the draft block follows one session (`mtp_owner_`), and B-session x (k + 1)-row
verify batches are a P6/P5 follow-up.
