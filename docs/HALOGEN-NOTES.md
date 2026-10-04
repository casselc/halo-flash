# Notes on halogen-flash-server (external reference, learn-only)

halogen-flash-server (peonist-ai) is a closed-source Qwen3.8-Flash-Next engine for the same
hardware. Its EULA allows use and does not prohibit reverse engineering, but asks that derived
source or kernels not be redistributed. **Nothing from it is copied into halo-flash.** These
notes record measured facts and design choices to test independently.

We inspected v0.16.2 statically on 2026-10-04: the container image's `flash_serve` binary, its
gfx1151 kernel metadata and disassembly, its tuned hipBLASLt plan, the README and CHANGELOG, and
the HF checkpoint headers. Nothing was executed. Working files are in
`/data/halogen-research` (not in the repo).

## Their numbers (their README; not comparable to our pp2048)
- **Prefill:** 1,584 tok/s on an 8K prompt, 1,567 at 32K, 1,517 at 131K. Cold, one
  32K-token call.
  - They publish no pp2048 figure. They report that a 1,024-token chunk costs about half the
    throughput.
- **Decode:** 37.6 tok/s serial, 46-56 with the draft head.
- **Setup:** ROCm 7.14, Linux 6.18.4+, about 85 W, median clock 2,229 MHz. IOMMU is off, which
  they credit with 13-16% of prefill.

## Per-GEMM times from their tuned plan
hipBLASLt bf16, M = 2048 tokens:

| Shape (rows x K) | Use | Their time | Ours (same role, 2026-10-04) |
|---|---|---:|---:|
| 320 x 10240 | HC down | 0.410 ms | `_m` 1.04 ms (fp16 WMMA over Q8_0) |
| 336 x 10240 | HC down + 4 inject rows, padded (inferred) | 0.483 ms | — |
| 10240 x 320 | HC up | 0.471 ms | `mix` 1.40 ms (includes the mix epilogue) |
| 10240 x 2560 | dense | 2.79 ms | — |
| 512 x 5120 | router; bf16 hi/lo split of fp32 (inferred) | 0.40 ms | `f32_tiled` 1.28 ms (512 x 2560) |

They spend about 0.9 ms per HC mixer in GEMMs; we spend about 4 ms (combine_norm 1.58,
`_m` 1.04, `mix` 1.40).

## Design choices to test (independently written)
1. **HC mixers as plain GEMMs at library speed.** Their combine + scatter + RMSNorm is one
   kernel and still writes a normalized bf16 copy. The win is in the GEMMs, not in avoiding
   that write; our kernel A prototype, which avoided the write, broke even (branch
   `proto/kernel-a`).
2. **Experts: weights global -> VGPR, never through LDS.** Only activations are staged in LDS.
   - Dequant goes through a per-group 16-entry bf16 lookup (scale x codebook, `v_perm_b32`).
   - `v_permlanex16` forms the B operand.
   - 8 waves at about 250 VGPRs, relying on ILP rather than occupancy.
   - This matches our finding that a sixth wave did not help gate/up.
3. **DeltaNet:** chunked, with the state in WMMA accumulators and fp16 mirrors of q/k/v. The
   L2 norm is folded into per-row scalars instead of writing normalized q/k.
4. **Router:** bf16 WMMA with fp32 accumulation, either on bf16-clean weights or on a hi/lo
   split.
5. **Their weights:** a Hadamard-rotated 4-bit trunk (QuIP#-style sign vectors), 6-bit HC
   weights, and one shared input rotation for all 512 experts. GGUFs are repacked at load.
