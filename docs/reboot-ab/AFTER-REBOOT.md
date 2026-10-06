# After the reboot (iommu=pt, PCIe Gen4): run `post` and compare

Stream L2's `pre` baseline was recorded on 2026-10-06 before the reboot (results: `reboot_pre.json`,
`reboot_pre.txt`, raw logs in `reboot_pre/`). The script, the binary and the probe are fixed, so `post` measures the
same thing:

- binary: `/data/halo-work/halo_run_50d5a35` (main 50d5a35, built in `/data/halo-work/wt-pageloc`)
- probe: `/data/halo-work/streams/L2/halo_hwprobe_pl` (`halo_hwprobe` + the `pageloc` mode, branch `stream/pageloc`)
- tools (`ab_bench.py`, `gpu_clock.sh`, `.nix-env.sh`): `/data/halo-work/wt-pageloc`

## 1. Check the reboot took effect (no GPU, seconds)

    cat /proc/cmdline                                   # expect iommu=pt
    cat /sys/kernel/iommu_groups/*/type | sort | uniq -c   # expect identity for the NVMe groups (pre: 33 DMA-FQ, 1 DMA, 2 identity)
    cat /sys/class/nvme/nvme*/device/current_link_speed     # expect 16.0 GT/s PCIe (pre: 8.0 GT/s on both)

Before the reboot the GPU's own group was already `identity` (amdgpu), so iommu=pt should change the NVMe and
other devices' DMA path, not the GPU's memory accesses. Expect the n-gram cold read, `nvme dd` and model load to move;
prefill, decode and the probe should not (a move there is a finding).

## 2. Run post (needs ~25-35 min of wall time, three GPU lock takes of ~3-8 min each)

Quiet machine first: no other model instance; MemAvailable >= 90 GB (ab_bench refuses below that).

    cd /data/halo-work/streams/L2
    ./reboot_ab.sh post            # phases: sys prefill decode ngram probe summary

Each GPU phase takes `/data/halo-work/gpu.lock` itself (flock), logs take / release in `streams/LOG.md`, and writes
a `.done` marker; rerunning skips finished phases, so an interrupted run is resumed by running the same command.
Single phases: `./reboot_ab.sh post decode`, etc. To redo a phase delete `reboot_post/<phase>.done` (prefill:
`reboot_post/prefill_*.json.done` too).

The first run after boot reads the model from disk (page cache empty): the `model ready` times of the first runs
are cold-cache loads, not comparable with `pre`'s (pre had ~11 GB of page cache). Prefill / decode are unaffected
(ab_bench and decode medians exclude load). The n-gram reads are O_DIRECT (XFS), page-cache independent.

## 3. Compare

    python3 reboot_summary.py compare pre post     # writes reboot_compare_pre_post.txt

It prints, per metric, pre -> post with the difference and the "significant only if it exceeds both arms' round
spreads" rule (ab_bench's), plus per-kernel decode us/call and every pageloc / dram probe row side by side.

What each part measures:
- prefill 2048 / 8192 (ab_bench, pinned, 3 rounds x 3 reps, warm prose_8888; 8192 = whole prefill, `--chunks 0`)
- decode short / 7K (`--device-sample 1 --n 80`, steady median of 76 steps, pinned, 3 rounds; plus one eager
  `--decode-timing 1` run per context for per-kernel times)
- n-gram cold read: `HALO_PREFILL_TIMING` "n-gram read" + "deferred" ms for a novel 2048-token prompt (random
  dictionary words, fresh each run), `--ngram-cache-gb 0` (every rep from the NVMe) and default cache (first rep
  cold); plus `dd` of 2 GiB O_DIRECT from the model's 4th shard
- `halo_hwprobe pageloc key` (stream / window / padding / VRAM-vs-GTT GB/s), `dram`, `latency`, `clock`
