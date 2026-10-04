# P0 baselines: runtime microbenchmarks

Measured 2026-10-03 on a Ryzen AI MAX+ 395 (Radeon 8060S, gfx1151) with
128 GB of unified memory. Kernel 7.0.0-34. `bench/p0/run.sh` reproduces
the numbers.

## Backends

- **ROCm HIP**: distro ROCm 7.1 (`libamdhip64` 7.1.52801).
- **HRX native**: libhrx v0.1.0 at hrx-system 1618350.
- **HRX HIP shim**: the same `bench_hip` binary run with
  `LD_PRELOAD=libamdhip64.so.7.9999.0` from the same HRX build.

HRX's GPU driver needs a TheRock nightly HSA runtime, because it calls
`hsa_amd_queue_create`. Neither the distro ROCm 7.1 nor nix's ROCm 7.2.3
provides that symbol. These runs used TheRock run 37080189834, the "core"
artifact set, fetched with hrx-system's
`build_tools/ci/ci_core_linux.py fetch-rocm`.

All three backends load one HSACO, compiled from `bench/p0/kernels.c` with
`clang -target amdgcn-amd-amdhsa -mcpu=gfx1151`, so they run identical
machine code. Each binary first checks argument passing with a store kernel.

## Results

| Metric | ROCm HIP | HRX native | HRX HIP shim |
|---|---:|---:|---:|
| Noop launch, async back-to-back (ns/launch) | 1740 | 1782 | 2003 |
| Noop launch + stream sync, p50 / p99 (µs) | 6.6 / 7.2 | 8.0 / 11.9 | 7.0 / 14.5 |
| Graph, 1 node, launch + sync p50 (µs) | 12.1 | **6.0** | 6.5 |
| Graph, 1000 chained nodes (ns/node) | 1733 | 1735 | 1740 |
| Graph, 1000 independent nodes (ns/node) | 988 | 919 | fails (HRX bug) |
| Read bandwidth, 8 GiB (GB/s) | 241.7 | 241.8 (wall) | 241.8 |
| Write bandwidth, 8 GiB (GB/s) | 225.9 | 223.7 (wall) | 223.8 |
| 66 GiB single allocation | ok, 2.98 s | ok, 2.52 s | not run |
| Read / write bandwidth over 66 GiB (GB/s) | 241.1 / 223.5 | 241.2 / 222.4 (wall) | not run |

HRX native bandwidth is computed from wall-clock time around the dispatch and
an event synchronize. HRX's `hrx_event_elapsed_time` currently reports near-zero
durations; see "HRX findings" below. For an 8 GiB kernel, wall-clock and
event timing agree to within about 1%, as the HIP and shim rows show.

## What this means for the design

1. **Bandwidth ceiling: 241 GB/s read.** This is about 12% above the 215 GB/s
   assumed in the plan. The AR decode roofline at gufo's quantization mix
   (about 5.2 GB per token) rises to about 46 tok/s. The P3 target of at least
   85% of roofline therefore becomes about 39 tok/s.
2. **The per-dispatch floor is the GPU, not the runtime.** A dependent kernel
   costs about 1.7 µs and an independent one about 0.9–1.0 µs on every
   backend.
   - A decode step that issues about 700 dependent kernels would spend about
     1.2 ms per token on dispatch alone, which is about 5% at 40 tok/s.
   - Kernel fusion therefore matters more than which runtime we choose. P3's
     fused per-layer kernels, or a persistent per-layer kernel, are the lever.
   - HRX graphs don't change the per-node cost.
3. **HRX's advantage is single-shot latency.** Launching a one-node graph and
   syncing takes 6.0 µs, against 12.1 µs on ROCm. For dispatch throughput HRX
   matches ROCm, and a single launch plus sync is slightly slower (8.0 µs
   against 6.6 µs).
4. **Allocations above 64 GiB work** on both runtimes, at full bandwidth.
   Resident Q4 weights of about 70 GiB fit in a single buffer.

**Decision per subsystem.**
- **Weights, activations and bandwidth kernels:** either runtime. They go
  through `rt::Device`.
- **Decode-step submission:** HRX native graphs, because single-shot latency
  is lower. Re-measure once real fused kernels exist.
- **Timing:** HIP events or wall clock, until HRX's event timing is fixed.
- **Fallback:** keep the ROCm HIP backend first-class. HRX is v0.1.0, and
  this run already found three defects.

## HRX findings (reported upstream-ready; investigation in progress)

1. `hrx_event_elapsed_time` returns about 1e-6 ms for an approximately 35 ms
   kernel, with default event flags.
2. HIP shim: a graph of independent kernel nodes (`hipGraphAddKernelNode` with
   no dependencies) fails at launch with
   `graph_exec.c:2344: INTERNAL; graph block signals no timeline value`.
   Captured chains work, and so do native HRX graphs with independent nodes.
3. `HRX_DEVICE_PROPERTY_COMPUTE_UNITS` returns OK but leaves 0 on gfx1151.

Still to measure in P0: host-call round trip, and NVMe random-read IOPS on the
dedicated drive, once it is installed.
