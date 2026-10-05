# Hardware model: gfx1151 (Strix Halo) as measured

Measured facts the performance work relies on, each with the probe that produced it
(`tools/hwprobe.hip`, target `hwprobe`, binary `halo_hwprobe <probe>`). Numbers are at the
pinned clock unless noted (`tools/gpu_clock.sh pin`). When a kernel's behaviour disagrees with
this model, the model is wrong or the kernel has a defect; find out which before optimizing.

Machine: Ryzen AI Max+ 395, Radeon 8060S (gfx1151, RDNA 3.5), 20 WGPs = 40 CUs = 80 SIMD32,
128 KB LDS per WGP (64 KB per workgroup), 1536 VGPRs per SIMD (24-register granule),
LPDDR5X-8000 256-bit unified memory, 32 MB MALL. ROCm 7.2.3.

## Clocks (`halo_hwprobe clock`; prefill timings from `halo_run --prefill-bench 2048`)

- Truth comes from inside kernels: `clock64()` reads SHADER_CYCLES, **20 bits** (wraps every
  ~0.35 ms), so sum short windows; `wall_clock64()` is a 100 MHz realtime counter. sysfs
  (`hwmon/freq1_input`, `pp_dpm_sclk`) reports the *requested* clock, not the delivered one.
- `auto`: ~2.88 GHz delivered under full VALU load (2.79-2.90 by load shape).
- `profile_peak`: ~2.51 GHz delivered under full load while sysfs says 2900; a 2048-token
  prefill takes 1.72-1.73 s vs 1.56-1.57 at auto (~10% slower). Do not measure with it.
- Manual level + overdrive minimum sclk 2900 MHz + top fclk (`tools/gpu_clock.sh pin`):
  ~2.88 GHz, prefill 1.557-1.558 s (1 ms spread over 5 warm reps; auto: 4 ms). `ab_bench` uses
  this. On this APU `pp_dpm_sclk` and `pp_dpm_mclk` reject writes; `pp_dpm_fclk`, `pp_dpm_socclk`
  and `pp_od_clk_voltage` accept them.

## Execution

- VALU: dependent FMA latency 5.2 cycles; throughput ~1 wave-instruction per cycle per SIMD at
  high occupancy (16.5 cycles per dependent FMA per wave with ~16 waves per SIMD).
- WMMA 16x16x16 (`halo_hwprobe wmma`): **f16, bf16 and iu8 all ~34 cycles per WMMA per SIMD at
  throughput (~55 T/s for the GPU)**. iu8 is not faster than f16 here: int8 activations can only
  win by removing dequantization VALU and bytes, never by matrix throughput. One dependent chain in
  one wave per SIMD already runs at ~35-36 cycles: WMMA latency is about its issue interval, so a
  single accumulator chain is not a bottleneck. (A first reading of 69 cycles for f16 was the clock
  ramping from idle: the first kernel after idle can run ~2x slow, so probes warm up with a
  full-length run first.)
- **VALU does not overlap WMMA.** Each independent VALU instruction interleaved adds ~1 cycle per
  WMMA (4 FMAs: +4.2, 16 FMAs: +16). Per SIMD, time ~ 34 x WMMAs + VALU instructions + other
  issue. A GEMM that dequantizes in VALU pays for every dequant instruction directly.

## LDS (`halo_hwprobe lds`)

- Peak ~128-130 bytes per cycle per WGP, only with 8- or 16-byte accesses and conflict-free
  addresses (7.4 TB/s for the GPU at 2.88 GHz).
- **16-bit accesses: ~13 B/cycle (a tenth of peak)**; 4-byte: ~100 (load) / 66 (store). The cost
  is per instruction: avoid `ds_load_u16` / `ds_store_b16` gathers in hot loops.
- Bank conflicts: power-of-two strides serialize (4-byte elements at stride 32 or 64: 4 B/cycle,
  32x slower); stride 33 is nearly conflict-free.

## Memory (`halo_hwprobe dram`, `halo_hwprobe latency`)

- Streaming: ~240 GB/s read, ~222 write, ~208 copy (counting both directions).
- **Access pattern dominates.** 480 MB as 40 KB rows, each block reading `seg` bytes from 128
  rows: if consecutive (concurrently resident) blocks take neighbouring segments of the same
  rows, 157 GB/s at 32 B segments, 216 at 256 B, 239 at >= 4 KB. If consecutive blocks take
  different rows: **10 GB/s at 32 B, 33 at 256 B, 119 at 1 KB**, 239 only from 4 KB. Small
  segments scattered across concurrent blocks collapse DRAM bandwidth up to 20x. Random 4 KB
  pages read 512 bytes at a time: ~34 GB/s flat, whatever the loads in flight (1-8 per lane) or
  waves (1-16 per SIMD): parallelism does not fix locality.
- Inside the MALL the order does not matter: the same sweep over a 15 MB footprint reads ~800 GB/s
  for both orders (64 B segments and up; ~400-470 at 32 B). The collapse above is DRAM row (page)
  locality, not cache or translation. MALL-resident working sets (<= ~32 MB) stream at ~800 GB/s.
- Dependent-load latency by footprint: 16 KB 39 ns, 256 KB 62, 2 MB 92, 8 MB 134, 24 MB 180,
  64 MB 256, 256 MB 295 ns (L1 / L2 / MALL / DRAM).
- Writes are paid by the next kernels (their write-back drains during them); a microbenchmark's
  "after 96 MB of writes" condition does not reproduce the engine's cache state for large GEMMs.

## Overheads (`halo_hwprobe sync`)

- Barrier round (LDS write, barrier, LDS read): ~33-42 ns (100-120 cycles) for blocks of up to
  256 threads, 52 ns at 512, 123 ns at 1024.
- A loop that issues one dependent memory round per barrier step costs the memory latency per step
  (~0.4 us for random DRAM addresses) whichever barrier is used; only prefetching several steps
  ahead hides it.
- Back-to-back empty kernels: 1.7-2.1 us per launch.

## Validation against our kernels (2048-token prefill, in-engine)

Floor = max(WMMA time at 34 cycles/WMMA/SIMD, bytes at 240 GB/s), per call:

| Kernel | Floor | Measured | Of floor | What the model points at |
|---|---|---|---|---|
| Dense qkv GEMM (10240 x 2560) | 1.94 ms (WMMA) | 3.34 | 58% | dequant VALU (~3.5 per WMMA, serial with WMMA), stage barriers |
| MoE gate/up (Q4_K) | 3.9 ms (weights) | 6.0 | 65% | dequant VALU at 45% |
| MoE down (Q5_1) | 2.6 ms (weights) | 4.3 | 61% | dequant VALU |
| Mixer down `_m` (320 x 10240) | 0.24 ms | 1.04 | 23% | activations read as 128 B pieces of 64 rows per stage: the 20-30 GB/s pattern |
| Attention, dense 2K | 0.93 ms (WMMA) | 3.9 | 24% | per-tile barriers and softmax |

The `_m` row led to the first change derived from this model: the mixers' fp16 streams stored
K-blocked ([cols / 64][T][64]) so a stage reads one contiguous 8 KB run. Study: 0.67 -> 0.44 ms
warm, 1.33 -> 0.62 after writes (predicted 0.35-0.45 warm). In-engine: 101.9 -> 55.2 ms per
prefill (predicted -40 to -50), the norm kernel writing it +10.6, net -31.7 ms.

## Open questions (from a GPT-6.1 review of the probes, 2026-10-05)

- Same-rows vs different-rows bandwidth: answered, DRAM row locality (the same pattern inside the
  MALL shows no order effect).
- Random 4 KB pages at a flat ~34 GB/s for every loads-in-flight / occupancy setting: verify in the
  ISA that the sweep really varied outstanding loads; vary page size and locality.
- Pointer-chase knees are effective latencies, not cache sizes (translation, associativity and
  residency mix in).
- Harness: ISA-checked kernels with observable outputs, randomized run order, distributions over
  repeats, and real producer -> consumer sequences in place of the generic "after 96 MB of writes".
- The cost model built on this must be an empirically calibrated search, validated on held-out
  shapes and fused sequences, with launch amortization, occupancy cliffs, transaction amplification,
  layout conversions, MoE routing distributions and one-time transformation costs included.

## Consequences for kernel design

1. Order dispatch so that concurrently resident blocks touch contiguous memory (the mixer's mix
   GEMM: -18% from walking row tiles fastest).
2. Count VALU per WMMA before choosing a dequantization scheme; it adds directly.
3. Stage LDS in 8- or 16-byte accesses with padded strides; treat 16-bit LDS gathers as a cost.
4. Per-wave stall shares from thread traces are not SIMD idle time at high occupancy: check the
   VALU/WMMA issue share and occupancy, or ablate.
5. Measure at the overdrive pin, and read the clock from inside kernels.
