# K2: measured prefill-tail baseline

## Result

The extra tail pass exists and has a measurable fixed cost. On the unchanged
engine, a **one-token tail takes 112.3 ms** and a **256-token tail takes
347.7 ms**, using the tool's reported chunk-device span. No merged variant was
run, so these results are **calibration, not measured speedups**.

| Direct Prefill length | Chunks | Wall-prefill median | Wall range across 3 reps | Tail-span median | Tail-span range |
|---:|---|---:|---:|---:|---:|
| 8192 | 8192 | 4.938 s | 66 ms | — | — |
| 8193 | 8192 + 1 | 5.058 s | 72 ms | 112.3 ms | 4.9 ms |
| 8448 | 8192 + 256 | 5.296 s | 67 ms | 347.7 ms | 1.1 ms |
| 8449 | 8192 + 257 | 5.344 s | 68 ms | 363.9 ms | 0.7 ms |

The 257-token tail is 4.66% more expensive than the 256-token tail. The measured
whole-prefill differences from 8192 are 120/358/406 ms; these are different input
lengths and sequential processes, not paired variant A/B gains.

`HALO_PREFILL_TIMING` labels the interval after upload until synchronized chunk
completion as "device". This is a host wall-clock span including dispatch and
waiting, not a pure sum of kernel event times. Total wall-prefill also includes
host preparation/upload. Model loading is outside the reported prefill timer.

## Protocol and predictions

- Fresh `halo_run` build in K2's isolated worktree. Production `src/` and
  `tools/run.cpp` match frozen engine `0810614`; branch starts at its docs-only
  successor `055ac0b`.
- HIP/gfx1151, pinned minimum SCLK 2900 MHz, fclk level 7 (2000 MHz).
- Explicit fp32 KV (`--kv-f16 0`), chunk 2048 / long chunk 8192, GEMM slice 2048,
  dense prefix enabled, context 16384, 4 GiB n-gram cache. No MTP loaded.
- First tokens of `streams/S/prose_10k.txt`; one process per length, three timed
  prefills per process. Warm-up file `prose_8888.txt` actually encodes **3485
  tokens**, as recorded in every log; its name is not a token-count promise.
- Prespecified calibration ranges all held: 8192 wall 4.5–5.8 s (actual 4.938),
  one-token tail 25–200 ms (112.3), 256-token tail 180–650 ms (347.7), and
  257-vs-256 within 10% (4.66%). These broad ranges were prior-based estimates,
  not a calibrated merge-cost model.

## What this justifies

The measured tail span is an optimistic ceiling on avoidable work, not a
prediction that it can all disappear. Relative to whole-prefill wall time it is
about **2.22%** for one token and **6.57%** for 256 tokens. A merge still pays for
the additional tokens, can change kernel tiling/cache behavior, and must retain
exactness and cancellation semantics.

Before a runtime variant: obtain actual tail expert routing and a marginal-cost
projection; resolve the cancellation-point constraint; then use prescribed
pinned in-engine alternating A/B and byte-identical logits/state checks. The
existing two-log workload sample does not establish production prevalence.
No kept/not-kept kernel row or precision policy changes follow from this sweep.

## Artifacts and release

Under `/data/halo-work/streams/K2`:

- `run_baseline.py`: guarded, bounded runner with clock restoration and lock log.
- `20261009T150959Z-baseline.json`: commands, hashes, clock table, all wall/chunk
  samples. `20261009T150959Z-summary.json`: independently derived summary.
- `20261009T150959Z-{8192,8193,8448,8449}.log` and matching logits binaries.
- `PREDICTIONS.md`: pre-run ranges and actual outcomes; its pre-run hash is in
  the baseline JSON.

The GPU lock was released after the sweep. Verified no model/probe processes,
0% GPU utilization, 18,661,376 bytes GTT use, automatic clock policy restored.
No further GPU jobs are running or queued by this stream.

## Re-measure after the short-call GEMMs (main 9acbec4, 2026-10-10)

The 112 ms one-token tail was mostly the dense GEMMs' fixed cost at small T, which every short Prefill call
paid (ROOFLINE section 19). With the exact short-call tiles merged, the same flags and lengths give (pinned,
3 reps, logits identical with the tiles off):

| Direct length | Tail | Tail span, tiles off | Tail span, default | Wall, default |
|---:|---:|---:|---:|---:|
| 8193 | 1 | 114.2 ms | **63.2 ms** | 5.031 s (off 5.078) |
| 8448 | 256 | (K2 baseline 347.7) | **326 ms** | 5.349 s |
| 8449 | 257 | (K2 baseline 363.9) | 363 ms (past the 256 limit) | 5.332 s |

## Verdict

Not worth a runtime merge variant. A merge could save at most the tail span (63 ms, 1.3% of an 8193-token
prefill), only for calls whose length is just past a multiple of 8192, and it would have to keep the
cancellation poll it removes and the checkpoint boundaries. The general fix (short Prefill calls) already took
the larger share for every short call, not only tails. K2 closes here; the queue row records it.
