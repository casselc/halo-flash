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
