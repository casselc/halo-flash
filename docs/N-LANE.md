# NPU research lane: results

## N1 — bring-up (2026-10-03): works

- Strix Halo NPU5 (`RyzenAI-npu5`, firmware 1.1.2.65, in-tree amdxdna,
  kernel 7.0). hrx-system libamdf + Loom built with XDNA enabled in a
  separate build directory, with flags as `ci.py iree-cmake-xdna`.
- Conformance: CI host-only 32/32; hardware 16/17 (88/2/4 gtest cases).
  The failure is `umd/drm/device_test`, which expects a numeric device
  address that this driver mode does not report (`AMDXDNA_INVALID_ADDR`
  under PASID/SVA; libamdf deliberately uses the host pointer there,
  `libamdf/src/xdna/umd/drm/buffer.c`). It is a test expectation only.
  Two manual suites fail on stale compile-report goldens; run by hand they
  are bit-exact.
- An int8 GEMM tile written for this lane is bit-exact against an
  independent oracle, and wrong oracles are rejected.

## N2 — strands-decider torso on the NPU (2026-10-03): SHELVED

Gate, fixed before measuring: shelve if a multi-tile int8 GEMM at K=2048
stays under 1 TMAC/s, if a decider layer cannot run in one or a few
commands, or if the projected per-question latency exceeds 3x an optimized
GPU prefill.

| Measurement | Result |
|---|---|
| Array (queried from the device) | 8 columns x 6 rows: 32 compute tiles, 8 memory tiles, 8 shims |
| One tile, int8, weights resident in tile memory | ~91 GMAC/s (~10 cycles per `mmul`) |
| Scale-out, 1/2/4/8/10 workers | 93/185/364/717/906 GMAC/s marginal; 659 end-to-end at 10 |
| Above 10 workers | does not compile (routing at 11-16; out of DRAM channels at 32) |
| DRAM to array | ~13.7 GB/s per channel, ~55 GB/s total |
| Fixed cost per command | 63 us (1 worker) to 115 us (8) |
| Several GEMMs in one command | works, bit-exact |
| Weights resident across commands | not available; a layer is ~57 MB int8 against ~6 MB on chip anyway |

Every kernel was bit-exact against an independent oracle, with poisoned
outputs and output guards checked; two wrong oracles mismatched on more
than 99.9% of outputs.

Projected cost per question (the torso is ~1.4 GMAC per token, matmuls only):

| Question length | Optimized GPU | NPU as measured | NPU at 32 tiles, today's codegen |
|---|---|---|---|
| 50 tokens | 10-20 ms | 100-150 ms | ~55 ms |
| 300 tokens | 30-45 ms | 0.5-0.7 s | ~145 ms |

Verdict:
- GEMM throughput (at least 1 TMAC/s): **fail**.
- Latency (within 3x of the GPU): **fail**, even with 32 tiles.
- Layer in one or a few commands: partial.

The limits are in the toolchain, not the silicon.

**To reopen.** Both depend on the Loom AIE2P backend:
1. An `mmul` inner loop at 2 cycles or fewer per instruction (VLIW
   co-issue, post-increment addressing).
2. Memory-tile staging and multicast, so that all 32 tiles can be fed.

Together these put the arithmetic above 10 TMAC/s. Rerun the same gate
when either lands. Until then the decider runs on the GPU, scheduled
between LLM decode steps (P9).

## N2b — hand-written Low kernels (2026-10-03): INCONCLUSIVE, stays shelved

Question: would better code generation pass the N2 gate? Hand-written Low
code, no compiler changes. Every configuration was bit-exact against an
independent oracle. The wrong oracles (swapped K slices, transposed weight
blocks, a missing chain slice) each mismatched more than 99.9% of the
2,097,152 outputs.

- **Inner loop: about 1.29 cycles per `mmul` statically, versus ~10 from
  the compiler.**
  - Straight-line code; post-increment A loads co-issued with `mmul`; each
    A fragment reused across 4 accumulators.
  - About 300 GMAC/s per tile measured.
  - Loops are unusable: the bundle planner drains the pipeline at every
    backedge, which is 3.3x slower.
  - The floor is one 512-bit load per cycle. Undetermined whether that is
    the hardware or the planner's model of it.
- **All 32 tiles: 8.14 TMAC/s marginal, 2.40 end to end** (int8, K=2048).
  - Layout: a K-chain, 4 tiles per column, each holding a 512-wide K slice
    and seeding its accumulators from the previous tile's partial sum.
  - Inputs: four A streams, each multicast across the 8 columns.
  - Fixed cost per command: 0.39 ms with 32 tiles.
- **Gate:**
  - GEMM of at least 1 TMAC/s: **pass**.
  - A decider layer in one or a few commands: **blocked**. It needs at
    least 4 commands per layer at 0.39 ms or more each.
  - Latency within 3x of the GPU: **fail** today, by about 62-75 ms
    against a budget of 30-60 ms at 50 tokens, and about 250-300 ms against
    90-135 ms at 300 tokens. With the missing features below, the
    projection is about 34 ms and 61 ms, which passes.

**Blockers, by owner.**

1. **Loom AIE2P array planner.**
   - Memory-tile buffering. Weights must stream (about 57 MB per layer
     against 512 KB of tile memory per command), and only 16 DRAM stream
     channels exist for 32 tiles.
   - Cascade wiring between tiles.
2. **XDNA runtime, image and firmware contract.** Multi-phase programs,
   or continuation commands that keep tile state. Today the non-GEMM ops
   between linears force separate commands, and each command reloads its
   program and weights.
3. **Lowering (non-blocking).** Post-increment `vector.mma` lowering, full
   unrolling within 16 KB of program memory, 1x4 A reuse, and a fix for the
   backedge drain. That would give about 3x per tile from the compiler alone.

**Reopen when 1 and 2 land.** Then rebuild the K-chain with streamed
weights and rerun the gate. Note that even with them, the decider would
take 36-55 GB/s of the memory bandwidth the GPU LLM uses.
