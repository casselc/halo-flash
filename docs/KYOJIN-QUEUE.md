# Kyojin-inspired GPU experiment queue

Queued 2026-10-08. Status: K0 classification and K2 unchanged-baseline calibration complete;
no merged/kernel variants tested or speedups claimed.
Audience: the next halo-flash performance stream and its coordinator.

## Baseline and evidence

- Source inspected: [Kyojin 9b43d671640f227277dad68002679f9857aba336](https://github.com/Yamz-Labs/kyojin/tree/9b43d671640f227277dad68002679f9857aba336).
- Published benchmark revision: `39dc956edc1d72444240d2ac07b530df8d949d5a`.
- Hardware matches our family: Strix Halo / gfx1151. Kyojin's EXL3 weight pack,
  speculative policy, prompts and measurement conventions differ. Its headline
  results are not a matched A/B against halo-flash.
- Intended development baseline: `stream/mtp-draft-prefill` at `0810614`, including
  state-only/exact-column catch-up, W=all, and generator lifetime ordering.
  Changing to an integrated release requires a new frozen baseline and checks.
- Main-based shutdown fix: `fix/serve-shutdown` at `ef578a1`.
- Existing evidence: `MTP.md`, `ROOFLINE.md`, and DP's kept logs. Full closeout GPU
  run had 88 passes, 16 fixture skips and one unresolved fp16-KV tolerance failure.
  Do not label that baseline fully green or erase the failing replay.

## Execution order and ownership

| ID | Experiment | State | Dependency | Next deciding action |
|---|---|---|---|---|
| K0 | Freeze baseline and classify the tolerance failure | bounded classification complete; test still fails | none | See K0-KV-CLASSIFICATION.md; preserve strict bound and open precision investigation |
| K1 | History-copy proposals alongside MTP | queued | K0 before GPU work | CPU shadow analysis of actual token histories; proceed only on projected end-to-end benefit |
| K2 | Merge a short prefill tail | closed: not worth a runtime variant | — | The tail's cost was the short-call GEMM overhead; exact short tiles (main 9acbec4) took the 1-token tail 114 -> 63 ms for every short call. See K2-BASELINE-CALIBRATION.md |
| K3 | Single-wave QSA attention | queued | K0 before GPU work | Bank/register/traffic analysis against the current union-attention path |
| K4 | Expert-multiplicity-specific MoE dispatch | queued | K0 before GPU work | Expert multiplicity histogram and compiler resource comparison |

K0 reproduced both archived failures on main 61e3df0 and DP 0810614, with MTP off and
stable repeats. [K0-KV-CLASSIFICATION.md](K0-KV-CLASSIFICATION.md) records controls, provenance
and the limits of the pre-existing precision-interaction classification. This does not make the full suite green.

CPU analyses K1–K4 may proceed independently. First GPU deciding probe: K2's
small boundary-length sweep. K1's prototype follows if its shadow analysis wins;
K3 and K4 require their resource analyses to justify kernel work. A rejected idea
gets a not-kept row and closes; it does not silently become another tuning round.

The coordinator assigns one isolated worktree per experiment, branched from the
frozen baseline. Each stream owns its STATE.md, PREDICTIONS.md, scripts and raw
results under its stream directory. This document queues work; it creates no
background agent, GPU process, cloud job or recurring schedule.

## Common gates

1. Re-read ROOFLINE's kept **and not-kept** rows before implementing. Name the
   earlier experiment and explain the material difference.
2. Review the bottleneck hypothesis with GLM 5.3 on the Z.AI Coding Plan. Use
   resumable review sessions; conclusions are hypotheses to test.
3. Before every new GPU/performance measurement, write a numeric prediction
   from the calibrated performance model plus a skeleton/resource probe. The
   expected win is currently **unknown** for all four ideas; fill the prediction
   after CPU analysis, not by borrowing Kyojin's headline speedup.
4. GPU runs use the enforced shared lock, HALO_GPU_LOCKED=1, take/release logs,
   at most ~15-minute holds and one large model instance. Builds use at most -j8.
5. Decide by pinned in-engine A/B: four alternating rounds × three reps, prose_1
   at 2K with prose_8888 warm-up, plus the experiment's targeted workload. Retain
   wall TTFT, decode and end-to-end time, not just kernel microbenchmarks.
6. Exact paths require byte-identical baseline logits and kernel property tests.
   Precision-changing paths are separate, default-off experiments using
   BENCHMARKS.md's paired short/long gates against defaults and the original
   baseline. Greedy equality alone is not the precision quality gate.
7. Preserve session-cache, departures, rollback, spill/restore and MTP state-hash
   invariants as applicable. Plant a mutant for each new state/indexing rule.
8. Report predicted versus actual, uncertainty, memory cost and kept/not-kept.
   Use task throughput/latency as the final metric for agent-oriented changes.

## K1 — History-copy proposals alongside MTP

Source: [spec_policy.py](https://github.com/Yamz-Labs/kyojin/blob/9b43d671640f227277dad68002679f9857aba336/tools/qwen/spec_policy.py),
`NgramIndex`, `SeqFollower`, `install`; comparison machinery in `spec_bench.py`.
Kyojin incrementally indexes a token suffix, verifies hash hits against tokens,
and proposes the continuation of an earlier occurrence. Its policy maintains
draft-cache continuity even when a copied proposal wins.

Hypothesis: code/tool-output repetition can produce more accurate, cheaper
proposals than another MTP draft forward. Halo's PLE n-gram embedding table is
not this proposal mechanism. `SpecStepDrafts` exists, but currently refuses
externally supplied drafts while MTP owns the tracking state; arbitration and
state maintenance are the new work.

First action: shadow captured prose/code/agent token histories on CPU. Measure
lookup CPU time, hit length, verified prefix accuracy, acceptance by depth,
memory growth, truncation and collision handling. Build a cycle-cost projection
using the current catch-up/verify costs. Do not use the old 1.37-ms catch-up
coefficient as a calibrated cost for DP's state-only path; its full chooser
cost model needs a new fixed-depth sweep before retuning.

Advance only if proposals project a task-level win including lookup, drafting,
verification, rollback and catch-up. Prototype greedy mode first. Test both
accepted and rejected chains, same-prefix different sessions, rollback/reset,
hash collisions, cache reuse, departures and budget cuts. Preserve exact trunk
output and state continuity; sampled proposal arbitration needs its own design
and distribution tests before enabling it.

Deliverable: CPU shadow summary, cost projection, then optional A/B against
MTP-only on copy-heavy agent/code requests and negative-control prose.

## K2 — Short-tail prefill merging

Source: [generator/job.py](https://github.com/Yamz-Labs/kyojin/blob/9b43d671640f227277dad68002679f9857aba336/exllamav3/generator/job.py),
`EXL3_PF_TAIL_MERGE`, `EXL3_PF_NO_TAIL`.
Application: `FlashNextGpu::Prefill` / `prefill_chunk.cpp`.

Hypothesis: a prompt just beyond a chunk boundary pays a poorly amortized MoE
tail. Merging that remainder may save a weight pass without allocating the
largest chunk for every request. This differs from the already tried optional
8192-token chunk. Halo does not share every Kyojin last-page split, so first
verify that the proposed extra pass is actually present here.

CPU preflight at engine baseline 0810614 found that long chunks already default to 8192:
direct 2049/2304/4097 are single-chunk promotion controls. Probe direct 8192/8193/8208/8448/8449/16385;
the cold server holds back one token, shifting request boundaries by one. Inspect timing and actual routing
with aligned-neighbor controls and existing chunk settings. Price the
extra scratch and maximum context impact. Preserve canonical checkpoint/slice
boundaries; a tail optimization may not remove a scheduler cancellation point
or alter the prefix-cache exactness contract.

Advance on repeatable full-TTFT benefit exceeding run-to-run noise and an
acceptable memory increase. Test residual formats, GDN/PLE state, QSA pooled
blocks, cache reuse, cancellation, spill and DP catch-up hashes.

Deliverable: boundary-length A/B, ordinary-prompt negative controls, scratch
high-water mark, exactness results and a bounded tail policy if kept.

CPU results and GLM review disposition: [K2-TAIL-ANALYSIS.md](K2-TAIL-ANALYSIS.md).
Hypothetical merges add 814,468 device bytes for a one-token tail and 181,444,064 for 256 tokens,
but remove an inter-chunk cancellation poll. They are not approved runtime policies.
The CPU preflight alone produced no GPU result; the unchanged-baseline calibration below is now complete.

Update: [K2-BASELINE-CALIBRATION.md](K2-BASELINE-CALIBRATION.md) records the completed
unchanged-baseline sweep. One-token/256-token tail spans are112.3/347.7ms; no merge
variant or speedup was tested. Next deciding work is routing/marginal-cost analysis and cancellation design.

## K3 — Single-wave sparse attention

Source: [qsa_prefill.hip](https://github.com/Yamz-Labs/kyojin/blob/9b43d671640f227277dad68002679f9857aba336/exllamav3/modules/attention_fn/qsa_prefill.hip),
`qsa_pf`; dispatch in `qsa_prefill_hip.py` / `qsa_indexer.py`.
Application: `halo_attention_wmma_qsel_h2` / `AttentionChunk`.

Hypothesis: one wave handling the query heads associated with one KV head can
remove cross-wave LDS/barrier phases. Halo's rejected multi-wave `qsel_w` is a
different design. Our current query unions reuse KV across tokens; losing that
reuse may cost more than the barriers save.

First action: compare traffic, LDS bank addresses, VGPR/SGPR occupancy and spills
on the exact Qwen head shapes. Include a union-aware variant in the analysis.
Build a skeleton only if the bound remains attractive after extra KV traffic.

Test page/selection indexing, boundary lengths, empty/partial selections and
the softmax/output reference. If reduction or softmax order changes, classify
the variant as precision-changing and use the quality gates above.

Deliverable: resource/traffic report, optional exact skeleton and whole-prefill
A/B at 8K/32K. Isolated kernel time is not a keep decision.

## K4 — Actual expert multiplicity specialization

Source: [exl3_dec.cu](https://github.com/Yamz-Labs/kyojin/blob/9b43d671640f227277dad68002679f9857aba336/exllamav3/exllamav3_ext/quant/exl3_dec.cu),
`exl3_dec_moe_union`, `launch_moe_gu_r`, `launch_moe_down_r`.
Application: routing/bucket and grouped gate/up/down kernels in
`src/kernels/batch_decode.hip`.

Hypothesis: bucket experts by actual `unum[u]`, allowing smaller-capacity
workgroups inside a wider verification batch, rather than reserving NB-sized
activation/LDS arrays for every expert. Halo already instantiates NB=2..8, including NB=3;
copying Kyojin's RM=3 switch would be a duplicate. Its reported GLM-specific
kernel gain is not a prediction for Qwen on our engine.

First action: collect actual multiplicity histograms from preserved routing
data. Compare compiler resources for capacity 1/2/3 versus the existing batch
width; include bucket, launch and padding costs. Preserve expert/pair ordering,
per-pair sums and weighted combines.

Deliverable: histogram, bounded resource/cost estimate, optional grouped
prototype, property tests and verification-cycle/end-to-end A/B.

## Deliberately excluded duplicates

Generic prefetching/residency, K-blocked experts, mixer fusion, larger uniform
chunks and state-only draft catch-up already have Halo verdicts. EXL3 decoding
is a weight-format/quality project, not a small kernel transplant. Packed KV
and precision changes remain behind quality gates. Source reuse requires the
applicable MIT notices and file-level provenance; source inspection alone
does not authorize treating published performance as our measured result.

## Acceptance record

For each K1–K4, fill: baseline and variant SHAs, calibrated prediction, decisive
probe, exactness/quality result, end-to-end A/B, memory cost, reviewer findings,
kept/not-kept decision and reproducible script/raw-result paths. Update this
queue and ROOFLINE when a gate closes.

GLM 5.3 reader review (2026-10-08): no planning blocker; verified the duplicate
exclusions and open baseline-test boundary against the Halo source/evidence.
This is a review of the queue, not a performance result or a green release gate.
