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
