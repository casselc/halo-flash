# Span tracing: where a kernel's time goes, per wave

Per-kernel timestamps and counters average a kernel away. Following Sites (*Understanding Software
Dynamics*, KUtrace): write down the expected time of each phase, record every phase transition
cheaply, rebuild the timeline, and explain each difference.

## Pieces

- `src/kernels/span_trace.h`: `HALO_SPAN_BEGIN(tag, max_events)` at kernel start,
  `HALO_SPAN(event)` at phase boundaries, `HALO_SPAN_RT(event)` to also read the realtime clock.
  The macros are compiled in only with `-DHALO_TRACE`, into `build/nix-gpu/gen/core_trace.hsaco`
  (CMake target `core_trace`). The engine's `core.hsaco` contains none of it.
- **A record:** 16 bytes, written by lane 0 of the wave. It holds:
  - SHADER_CYCLES (20 bits), with where the wave ran (SE, SA, WGP, SIMD) in the spare high bits;
  - the block, the wave, the event and the kernel tag.

  Every 64th block's wave 0 also records the 100 MHz realtime clock as an anchor.
- **Slots:** one region per tag, slot = (block × waves per block + wave) × max_events. There are no
  atomics. A later launch of the same tag overwrites.
- **Sampling:** `halo_span_sample` (host-set) records only every Nth block. The other blocks run the
  same code without storing.
- **Capture:** `halo_gemm_bench trace CORE_TRACE_HSACO CASE OUT.bin [SAMPLE]`, where CASE is
  `control`, `combine` or `combine+mb`. It runs the case warm, clears the buffer, runs it once,
  writes the records, then times the plain and traced code objects interleaved (the overhead check).
- **Report:** `tools/span_report.py OUT.bin [--bins N] [--control]`. It gives:
  - per tag: active waves over time, busy share per WGP, and the distribution of every
    event-to-event transition (p10/p50/p90/max), plus wave lifetimes;
  - the shader clock, fitted to the anchors when they span more than 50 µs, otherwise the pinned
    2.88 GHz.

## Why it is built this way (each choice measured)

| Problem | Measured | Fix |
|---|---|---|
| Realtime clock read on every event | `s_sendmsg_rtn` GET_REALTIME takes about 0.6 µs | Every event reads SHADER_CYCLES only (an `s_getreg`) |
| Per-wave `atomicAdd` slot reservation | Combined with the clock reads: +70% on `combine_norm` | Deterministic slots |
| Plain stores of trace records | A barrier's release fence (`s_waitcnt_vscnt 0`) then waits for the trace stores: GEMM +37% | Records stored through inline asm, which the compiler's wait insertion does not track |
| Is SHADER_CYCLES one GPU-wide counter? (first answer, refuted below) | Yes. The anchors' (cycles − realtime × clock) agree across all 20 WGPs to within about 100 cycles | Real time from anchors only. Each wave is placed against the anchor of the nearest earlier block index; nearest-in-cycles picks the wrong 20-bit wrap |

Review fixes (GPT-6.1-sol and GLM 5.3 reviewed the first version):
- **Wave 0 started late.** It started 3-4 µs after the other waves in every block. Wave 0 evaluated
  the anchor condition, a vector load of the sampling setting waited on under a saturated DRAM
  queue, before its first stamp.

  Fix: the start stamp is the first instruction; the sampling settings are read as scalar loads in
  uniform control flow; an anchor is its own record (event 255) with the realtime and cycle reads
  back to back.

  Check in `span_report.py`: no wave may start more than 2 µs before its block's wave 0 (0 now).
- **Sampling aliasing.** Use a stride coprime with the 20 WGPs (7 or 11). The report prints
  coverage over (SE, SA, WGP); stride 7 covers all 20.
- **Stalls.** The cycle counter keeps counting while a wave is parked at `s_barrier`: control
  events 6 -> 7 (wave 0 waits while waves 1-7 run a loop). Cycle-derived 11.75 µs vs realtime
  11.88 µs (-1.1%).

**SHADER_CYCLES is per SIMD.** In the engine, not under the study harness's continuous load, the
4 SIMDs of one WGP held counters hundreds of thousands of cycles apart. Waves on SIMD 0 agreed with
their block's wave 0 to about 16 cycles; waves on SIMDs 1-3 did not. Most likely each SIMD's counter
restarts when it powers up after idle. The study's global-looking agreement held only under its own
conditions. Fix: every traced wave writes its own anchor (realtime and cycles read together) after
its start stamp, about 0.6 µs on traced waves only. Wave 0 also writes a closing anchor, which gives
the block's clock: 2.88-2.91 GHz in every block, in the study and in the engine.

**In the engine** (`halo_run ... --span-hsaco build/nix-gpu/gen/core_trace.hsaco --span-kernel NAME
--span-occurrence N --span-out FILE [--span-sample 7]`):
- `halo_run` loads the traced code object with recording off.
- In the last prefill rep it brackets the N-th launch of NAME with stream-ordered
  `halo_span_set(1)` / `halo_span_set(0)` kernels, so no host synchronization changes the
  surrounding cache or write-drain state.
- Cross-check against `--kernel-timing`: the `_mb` trace spans 631 µs (kernel timing 0.619 ms per
  call), `combine_norm` 1,170 µs (1.15 ms). Placement: 0 waves before their block's wave 0.
- Realtime messages jitter: a ~10 µs span measured -0.4% to -5% against realtime across runs.

Positive control (`halo_span_control`, `span_report.py --control`): a loop bracketed by both
clocks. The cycle-derived duration matches the realtime one within 1.1% (9.73 vs 9.84 µs), which
is about the realtime message's latency.

Overhead at T 2048, pinned, plain vs traced code object, three runs each:

| Case | Every block | Every 8th block | Every 32nd |
|---|---|---|---|
| `combine_norm` (6 events per wave, 20 µs waves) | +6.6..7.6% | +0.5..2.7% | +1.5..2.4% |
| `combine_norm` + down GEMM (`_mb`, 3 events per stage) | +5.2..6.9% | +0.7..2.5% | −0.1..+0.7% |

After the review fixes, every 7th block (three runs each): `combine_norm` +1.0..2.8%,
`combine_norm` + down GEMM −0.7..+1.5%.

The events also change the generated code: about +0.5% even when no block stores. Use sampling
(8 to 32) for kernels with many short waves, and report the overhead with every trace. A full
trace (every block) distorted the GEMM's barrier waits about 8× (0.46 vs 0.06 µs per stage).

## First readings (2026-10-05, T 2048, pinned, sampled by 8)

- **`halo_hc_combine_norm_rows`:** about 1,100 waves resident (the cap is 1,280); waves live
  20.6 µs (p50).
  - **The inject phase (4 → 5) takes 9.5 µs of that (46%).** It is four rounds of 16 L2 loads of
    the inject weights per wave. That is what `halo_hwprobe rounds` predicts for L2 at this
    concurrency (about 2.6 µs per round): bandwidth-bound L2 traffic, re-read by every block.
  - The block reduction (2 → 3) takes 3.5 µs. It sits right after the residual stores, and the
    barrier's release fence waits for them (`s_waitcnt_vscnt 0` in the plain kernel's ISA).
  - Since wave slots are capped, every µs a wave spends there costs throughput.
- **`halo_gemm_q8r_f16_mb`:** a stage takes about 2.8 µs.
  - The math phase (3 → 1) is 2.49 µs, against 1.89 µs of WMMA in the model: the LDS fragment
    reads (about 0.3 µs a stage by bandwidth) are not hidden.
  - The store phase is 0.25 µs and the barrier 0.06 µs.
  - Prologue 7.7 µs, epilogue 7.6 µs.
  - 32 blocks on 20 WGPs leave some WGPs with one block, some with two.

## Readings in the engine (2026-10-05, T 2048, pinned)

- **Mix GEMM (`mixt`): the epilogue was 89 of a wave's 118 µs.** Batching its residual loads per
  subtile cut that to 74 (kept: -13.4 ms kernel, -16 ms end to end). What remains fits the rounds
  model as DRAM-bound. Coalescing the loads to 128 bytes changed nothing (70 µs), which rules out
  request rate.
- **MoE down (Q5_1): 4.19 ms a call against a ~2.6-3.0 ms floor for its 629 MB of weights.**
  - ~640 waves resident (cap 720); waves live 35 µs (p50): math stages 18.6, output stores 8.6
    (p90 18.9), first-stage loads 6.1. This fits 123 generations × 35 µs ~ 4.35 ms.
  - ATT on the same dispatch: VALU 30% (dequantization), barriers 21%, memory ~23% (issue stalls on
    a full memory queue, store stalls, vmcnt), WMMA 3%.
  - The grid sized for 833 tiles is not a cost: empty blocks take ~2 ns each (`halo_hwprobe empty`).
- **Tracer bug the MoE trace exposed:** slots were indexed by block, so a 16,660-block grid
  overflowed the region and every occurrence looked like 285 tiles. Slots now count sampled blocks,
  and dropped waves are counted and reported.
