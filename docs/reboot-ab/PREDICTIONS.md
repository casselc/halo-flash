# Stream L2 predictions (written before each measurement)

Tried-list check (docs/ROOFLINE.md): no page-locality / translation probe has been run; D2 sec. 10 lists it as the
next step. Related not-kept rows (D2 bits 4, 16, 32, 1024, 4096, 1048576; M's `_p` prefetch) changed kernels; this is a
probe only (no engine change), so no repeat.

## P1. pageloc probe (halo_hwprobe pageloc all), pinned clock, 2026-10-06 ~16:05

Setup: 2 KB in flight per wave (4 x b128 a lane), 24 KB per wave, K streams per wave-iteration, window Wb, 48 KB LDS
(40 resident blocks, as gemv_q8r_r4), GTT buffer (hipMalloc 2 GB) unless noted.

Hypothesis H (concurrent DRAM pages / TLB reach limit GEMV-like streams):
- A (K at Wb = 24 KB packed rows, NB 80): GB/s falls with K. Prediction if H holds: K1 ~205, K4 ~180 (the r4
  kernel's 162 includes x staging + compute), K16 <= 155 (>= 20% below K1). Null: K1..K16 within 5%.
  My prior: H partly holds -- K16 10-20% below K1, K4 within 3-8% of K1.
- Block count: 16 blocks are latency-bound (~128 waves x 2 KB / 450 ns ~ 580 GB/s cap, but per-wave serial rounds:
  16 x 8 x 2 KB per ~0.6 us round = ~55 GB/s); 40-80 ~200; 168/400 (more rounds of 40 resident) ~ same as 80 or a bit
  higher (ramp amortized). K effect, if real, should be largest at high resident wave counts.
- B (window at fixed K): if TLB reach: 2 MB and 32 MB windows lower than 24 KB by >= 10% at K 16; 4 KB best.
  If DRAM-row only: the 32 MB window with all waves sharing K slot-major regions has FEWER concurrent DRAM rows
  globally and could be faster than packed 24 KB. I predict 32 MB window ~ equal or faster than packed (H as DRAM
  pages, not TLB), +0..10%.
- C (regions padded rp 24 KB -> 64 KB -> 2 MB, same bytes): TLB-reach version of H predicts rp 2 MB drops >= 15%
  (640 live 2 MB fragments); DRAM-only predicts small (<5%) change. Prior: < 8% drop.
- D (VRAM carve-out vs GTT, same DRAM): VRAM >= GTT by 0-10% for K 16, equal (+-3%) for K 1.
- E: dual-like shape (544 blocks, rows 2560) faster than r4-like (80 blocks) by >= 15% (as in engine: 223 vs 162).

## P2. reboot A/B pre baseline (reboot_ab.sh pre), written 16:20 before any run

Pre values expected from RULES.md / main: prefill 2048 ~1243 ms (pinned), 8K ~4956 ms (long chunks default);
decode 36.9 short / 39.0 at 7K (graphs, device-sample). Expect within +-1% of those.
Post (after iommu=pt + Gen4) prediction: the GPU's IOMMU group is already `identity` before the reboot, so GPU
memory traffic does not go through IOMMU translation either way: prefill, decode and the probe unchanged (+-0.5%,
inside round spreads). NVMe: Gen3 x4 -> Gen4 x4 and DMA-FQ -> identity: O_DIRECT dd ~3.2-3.5 -> ~5-7 GB/s; the
n-gram cold read for 2048 tokens (latency-bound small random reads, mostly) drops less than 2x, maybe -10..-40%.
Decode's host gap n-gram read is per-token tiny; no measurable decode change.
