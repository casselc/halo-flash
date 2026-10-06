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
