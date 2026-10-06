#!/usr/bin/env python3
"""Interleaved A/B prefill benchmark with noise statistics.

Runs each arm (a halo_run binary plus flags) in rounds, interleaved
(A B C A B C ...), so slow drifts (thermal state, background load) hit every
arm alike. Each run is `halo_run --prefill-bench` with HALO_PREFILL_TIMING;
its value is the median device time of its reps (the warm-up excluded).
Reports per arm the median over rounds and the spread (min..max), and each
arm's difference from the first, marked significant only when it exceeds
both arms' spreads.

Guards: --build runs `cmake --build` first and refuses to measure if it
fails (a failed build would otherwise leave a stale binary); the GPU clock
is pinned for the measurement (manual level, overdrive minimum shader clock
2900 MHz, top fabric clock: tools/gpu_clock.sh pin) and always set back to
auto; profile_peak delivered only ~2.5 GHz under load and made prefill ~10%
slower than auto, the overdrive pin ~0.5% faster with a 1 ms spread; a run starts only if MemAvailable covers its need;
results are written to --out after every run.

Attribution: unless --no-kernels, each run adds one more rep with
`--kernel-timing 1` (sync-free per-kernel GPU timestamps). That rep is left
out of the end-to-end median; its per-kernel times are kept per arm, and the
report lists the kernels whose median time moved most against the first arm,
so a surprising result shows where the time went without a re-run.

  ab_bench.py --model shard-00001.gguf --prompt prose_1.txt --warm prose_8888.txt \\
      --arm base=/path/halo_run_old --arm new=build/gpu/halo_run \\
      --arm 'new-noskinny=build/gpu/halo_run --skinny-f32 0' \\
      --rounds 3 --build build/gpu --out ab.json [--resident]
"""

import argparse
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time

# Through the DRM class link, which survives PCI bus renumbering (adding an
# NVMe moved the GPU from c6:00.0 to c7:00.0). The sudoers rule must name
# this exact path.
DEV = "/sys/class/drm/card0/device"
CLOCK = DEV + "/power_dpm_force_performance_level"


def write_sysfs(path, value):
    try:
        subprocess.run(["sudo", "-n", "tee", path], input=value + "\n", text=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def set_clock(level):
    """'pin': manual level, overdrive minimum sclk 2900 MHz, top fclk; 'auto': undo."""
    if level == "pin":
        return (write_sysfs(CLOCK, "manual") and write_sysfs(DEV + "/pp_od_clk_voltage", "s 0 2900")
                and write_sysfs(DEV + "/pp_od_clk_voltage", "c") and write_sysfs(DEV + "/pp_dpm_fclk", "7"))
    write_sysfs(DEV + "/pp_od_clk_voltage", "r")
    write_sysfs(DEV + "/pp_od_clk_voltage", "c")
    return write_sysfs(CLOCK, "auto")


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


KERNEL_LINE = re.compile(r"^  (\S.*?)\s{2,}([0-9.]+) ms\s+[0-9.]+%\s+(\d+) calls", re.M)


def run_once(cmd, args, env):
    kernels = not args.no_kernels
    full = shlex.split(cmd) + ["--model", args.model, "--prefill-bench", str(args.tokens), "--max-context",
                               str(args.tokens), "--reps", str(args.reps + (1 if kernels else 0)),
                               "--prefill-file", args.prompt]
    if kernels:
        full += ["--kernel-timing", "1"]
    if args.warm:
        full += ["--prefill-warm", args.warm]
    if args.resident:
        full += ["--ngram-resident", "1"]
    p = subprocess.run(full, env=env, capture_output=True, text=True, timeout=args.timeout)
    out = p.stdout + p.stderr
    device = [float(m.group(1)) for m in re.finditer(r"device ([0-9.]+) ms", out)]
    n = args.reps + (1 if kernels else 0)
    if args.chunks > 1:
        # Whole prefill: each rep prints one device line per chunk; sum them per rep.
        tail = device[-n * args.chunks:] if len(device) >= n * args.chunks else []
        device = [sum(tail[i * args.chunks:(i + 1) * args.chunks]) for i in range(len(tail) // args.chunks)]
    if p.returncode != 0 or len(device) < n:
        raise RuntimeError(f"run failed ({p.returncode}): {' '.join(full)}\n{out[-2000:]}")
    # With --prefill-warm the first timed line is the warm-up prefill; the
    # kernel-timed rep (last) is excluded from the end-to-end numbers.
    reps = device[-n:][:args.reps]
    per_kernel = {}
    if kernels:
        for m in KERNEL_LINE.finditer(out):
            per_kernel[m.group(1).strip()] = float(m.group(2))
        if not per_kernel:
            raise RuntimeError(f"no kernel timing in the output of: {' '.join(full)}")
    return statistics.median(reps), reps, per_kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--warm")
    ap.add_argument("--arm", action="append", required=True, help="name=command [flags]")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=1,
                    help="device lines per rep to sum (whole-prefill device time; default: the last chunk only)")
    ap.add_argument("--resident", action="store_true", help="--ngram-resident 1 (needs ~30 GB more)")
    ap.add_argument("--build", help="cmake build dir to build first")
    ap.add_argument("--no-pin", action="store_true", help="leave the GPU clock on auto")
    ap.add_argument("--need-gb", type=float, default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="ab_bench.json")
    ap.add_argument("--no-kernels", action="store_true", help="skip the extra kernel-timed rep")
    ap.add_argument("--top", type=int, default=12, help="kernels listed per arm in the attribution")
    args = ap.parse_args()

    arms = []
    for a in args.arm:
        name, _, cmd = a.partition("=")
        if not cmd:
            sys.exit(f"--arm {a!r}: expected name=command")
        arms.append((name, cmd))
    if args.build:
        b = subprocess.run(["cmake", "--build", args.build, "-j32"], capture_output=True, text=True)
        if b.returncode != 0 or " error" in b.stdout + b.stderr:
            sys.exit("build failed; refusing to measure a stale binary:\n" + (b.stdout + b.stderr)[-3000:])
    need = args.need_gb if args.need_gb is not None else 90.0 + (30.0 if args.resident else 0.0)

    env = dict(os.environ, HALO_PREFILL_TIMING="1")
    pinned = not args.no_pin and set_clock("pin")
    if not args.no_pin and not pinned:
        print(f"warning: could not pin the GPU clock ({CLOCK}); measuring at auto", file=sys.stderr, flush=True)

    def restore(*_):
        if pinned:
            set_clock("auto")
        sys.exit(130)

    signal.signal(signal.SIGINT, restore)
    signal.signal(signal.SIGTERM, restore)
    results = {"arms": {n: {"command": c, "runs": []} for n, c in arms}, "clock": "od2900" if pinned else "auto",
               "tokens": args.tokens, "reps": args.reps, "resident": args.resident, "started": time.time()}
    try:
        for rnd in range(args.rounds):
            for name, cmd in arms:
                avail = mem_available_gb()
                if avail < need:
                    sys.exit(f"MemAvailable {avail:.0f} GB < {need:.0f} GB needed; refusing to start a run")
                med, reps, per_kernel = run_once(cmd, args, env)
                results["arms"][name]["runs"].append({"round": rnd, "device_ms": med, "reps": reps,
                                                      "kernels_ms": per_kernel})
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=1)
                print(f"round {rnd} {name:24} device {med:8.1f} ms  (reps {', '.join(f'{r:.1f}' for r in reps)})",
                      flush=True)
    finally:
        if pinned:
            set_clock("auto")

    print(f"\nclock {results['clock']}, {args.rounds} rounds x {args.reps} reps, {args.tokens} tokens")
    base_name = arms[0][0]
    summary = {}
    for name, _ in arms:
        v = [r["device_ms"] for r in results["arms"][name]["runs"]]
        summary[name] = (statistics.median(v), min(v), max(v))
    b_med, b_lo, b_hi = summary[base_name]
    for name, _ in arms:
        med, lo, hi = summary[name]
        line = f"{name:24} median {med:8.1f} ms  spread {lo:.1f}..{hi:.1f} ({hi - lo:.1f})  {args.tokens / med * 1e3:7.1f} tok/s"
        if name != base_name:
            diff = med - b_med
            noise = max(hi - lo, b_hi - b_lo)
            line += f"  vs {base_name}: {diff:+.1f} ms ({100 * diff / b_med:+.1f}%) {'significant' if abs(diff) > noise else 'within noise'}"
        print(line)
    results["summary"] = {n: {"median": s[0], "min": s[1], "max": s[2]} for n, s in summary.items()}
    if not args.no_kernels:
        # Per kernel: median over rounds of the kernel-timed rep, and the
        # round-to-round spread, against the first arm.
        def per_kernel(name):
            runs = results["arms"][name]["runs"]
            keys = set().union(*(r["kernels_ms"] for r in runs))
            return {k: [r["kernels_ms"].get(k, 0.0) for r in runs] for k in keys}

        base_k = per_kernel(base_name)
        results["kernel_summary"] = {}
        for name, _ in arms:
            if name == base_name:
                continue
            arm_k = per_kernel(name)
            rows = []
            for k in set(base_k) | set(arm_k):
                b = base_k.get(k, [0.0])
                a = arm_k.get(k, [0.0])
                spread = max(max(b) - min(b), max(a) - min(a))
                rows.append((statistics.median(a) - statistics.median(b), k, statistics.median(b),
                             statistics.median(a), spread))
            rows.sort(key=lambda r: -abs(r[0]))
            results["kernel_summary"][name] = [
                {"kernel": k, "base_ms": b, "arm_ms": a, "diff_ms": d, "spread_ms": sp} for d, k, b, a, sp in rows]
            total = sum(r[0] for r in rows)
            print(f"\nper-kernel, {name} vs {base_name} (kernel-timed rep; sum of differences {total:+.1f} ms):")
            for d, k, b, a, sp in rows[:args.top]:
                print(f"  {k:58} {b:8.2f} -> {a:8.2f} ms  {d:+7.2f}  (spread {sp:.2f})")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
