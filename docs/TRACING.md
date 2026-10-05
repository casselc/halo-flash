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
| Is SHADER_CYCLES one GPU-wide counter? | Yes. The anchors' (cycles − realtime × clock) agree across all 20 WGPs to within about 100 cycles | Real time from anchors only. Each wave is placed against the anchor of the nearest earlier block index; nearest-in-cycles picks the wrong 20-bit wrap |

Positive control (`halo_span_control`, `span_report.py --control`): a loop bracketed by both
clocks. The cycle-derived duration matches the realtime one within 1.1% (9.73 vs 9.84 µs), which
is about the realtime message's latency.

Overhead at T 2048, pinned, plain vs traced code object, three runs each:

| Case | Every block | Every 8th block | Every 32nd |
|---|---|---|---|
| `combine_norm` (6 events per wave, 20 µs waves) | +6.6..7.6% | +0.5..2.7% | +1.5..2.4% |
| `combine_norm` + down GEMM (`_mb`, 3 events per stage) | +5.2..6.9% | +0.7..2.5% | −0.1..+0.7% |

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
