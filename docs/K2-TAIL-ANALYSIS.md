# K2: CPU preflight for bounded prefill tails

## Decision

The queued probe needs corrected boundaries. Engine defaults already promote a
Prefill call longer than 2048 tokens to **8192-token chunks**. Direct lengths
2049, 2304 and 4097 each make one chunk, not a full chunk plus a tail. The
structural opportunity starts at direct length 8193. A cold server request holds
back one token for decode, so its first corresponding boundary is **8194 request
tokens**. Cache reuse and checkpoint splits change individual call lengths.

Proceed to designing a **baseline-only routing/timing probe**, after K0. A naive
merge is not eligible for a runtime prototype: it removes an existing
inter-chunk cancellation poll, violating the queue's cancellation constraint.
The CPU preflight produced no GPU result, speedup, expert histogram or exactness verdict.

Follow-up: K0's bounded classification is complete (K0-KV-CLASSIFICATION.md), and
the unchanged halo_run baseline has been freshly built. The first measurement
attempt stopped before model load: current GPU is card1 but passwordless clock
control permits only card0. No timing samples were collected; lock released,
clock policy remained auto. The prepared runner is streams/K2/run_baseline.py.
The user subsequently updated permissions and the unchanged-baseline sweep completed;
see [K2-BASELINE-CALIBRATION.md](K2-BASELINE-CALIBRATION.md) for measured tail spans.

## Frozen source and prior verdicts

Worktree branch: `experiment/kyojin-tail-analysis`, starting at `055ac0b`, the
docs-only successor to engine baseline `0810614`.

- `flash_next_gpu.hpp:105-117`: base 2048, long 8192, slice 2048, dense prefix on.
- `flash_next_gpu.cpp:1524-1553`: per-call promotion, allocation and chunk loop;
  cancellation polled before each chunk after the first.
- `flash_next_generator.cpp:106` and `session_cache.cpp:227`: final prompt token
  held back. Cache `run(ck)` / `run(n)` at `session_cache.cpp:298-321` are separate
  Prefill calls; a merge must never span the fork/checkpoint.
- `prefill_chunk.cpp:1453`: MTP catch-up also runs per chunk and must be included
  in the eventual cost/exactness investigation.
- ROOFLINE section 11 kept **sliced** 8192-token chunks; unsliced long chunks
  failed both performance and exactness. This idea concerns only bounded
  remainders beyond the current 8192 capacity, not another uniform chunk increase.

## CPU results

`tools/prefill_tail_analysis.py` mirrors chunk selection and AllocChunk's
allocation formulas with the documented Qwen Config shape. This is a static
capacity estimate, not measured resident memory or actual GGUF metadata.

| Direct Prefill tokens | Current chunks | Hypothetical chunks | Extra device bytes | Extra host bytes |
|---|---|---|---:|---:|
| 2049 | 2049 | unchanged | 0 | 0 |
| 2304 | 2304 | unchanged | 0 | 0 |
| 4097 | 4097 | unchanged | 0 | 0 |
| 8193 | 8192 + 1 | 8193 | 814,468 | 10,304 |
| 8448 | 8192 + 256 | 8448 | 181,444,064 | 2,637,824 |
| 16385 | 8192 + 8192 + 1 | 8192 + 8193 | 814,468 | 10,304 |

The hypothetical rows save one trunk chunk traversal **and remove one
cancellation poll**. They describe a cost/opportunity bound, not an approved
policy. Both pre-run memory predictions hold (<1 MiB device/<16 KiB host for
one token; <200 MiB device for 256 tokens).

The estimate includes the device copy of `ck_host_`; excludes KV, weights,
one-time mixer permutations, MTP buffers, PLE prefetch/hash vectors and allocator
overhead. `kAttnSub=512` is declared at `flash_next_gpu.hpp:590`.
Block-score runtime growth is excluded; it does not change between the cold
8192/8448 cases. A warm request starting beyond 64K needs a different absolute
accounting. Additional allocation may persist across later requests.

The preserved `agent_dp_w128` and `bench_full_r1` cache logs provide a bounded
prevalence check, not production traffic. Checkpoint summaries report aggregate
prefill counts without the split positions; large checkpoint aggregates remain
unclassified. See the JSON for per-record classification. Do not infer a task
win from total prompt lengths or tiny benchmark samples.
Projected onto the current default chunk settings, the sample has 19 records:
17 are definitely ineligible and 2 large checkpoint
aggregates lack split positions; none is positively identified as eligible.
This establishes no workload-level demand for the optimization yet.

## Review disposition

GLM 5.3 independently reproduced the byte deltas and supported designing the
probe, conditional on cancellation and source-validation gates.

- Its alleged broken test was a source misread: line 10 tests **2049**, not
  8193. The original four tests passed. An additional explicit 8193/long-off
  case now verifies `[2048,2048,2048,2048,1]`; the extended five tests pass.
- Cancellation constraint retained as absolute. Polling before the merged chunk
  does not restore the intermediate poll; a runtime design remains blocked.
- Added the sub-cap definition citation and flat-buffer boundary checks.
- Scratch exclusions and bounded log prevalence are explicit. Real routing is
  still required; a short tail must not be priced as all expert weights.

## Next deciding experiment

After K0 classifies the known fp16-KV tolerance failure, collect **unchanged
baseline** per-chunk timing/routing at direct lengths 8192, 8193, 8208, 8448,
8449 and 16385; use server lengths shifted by one for cold/no-checkpoint runs.
Also retain 2049/2304/4097 as promotion controls and `--prefill-long-chunk 0`
as a distinct 2048-chunk control. Record exact flags and Prefill-call splits.

Measure actual tail-touched experts, kernel/host/MTP catch-up costs and scratch
high-water marks. Then calibrate a numeric end-to-end prediction before any
variant A/B. A later runtime proposal must preserve checkpoint boundaries and
cancellation semantics, prove byte-identical logits/state at boundary lengths,
and cover cache reuse, departures, rollback and spill/restore. Use the shared
GPU lock, bounded holds and prescribed pinned A/B protocol.

## Reproduction

Run from the K2 worktree:

```sh
python3 tools/test_prefill_tail_analysis.py
python3 tools/prefill_tail_analysis.py
python3 tools/prefill_tail_analysis.py --server --lengths 8192 8193 8194 8449 8450
```

Archived results: `K2-CPU-DIRECT.json` and `K2-CPU-SERVER.json`. The direct JSON
also names the two preserved benchmark logs used for the prevalence check.
