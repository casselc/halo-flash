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
is pinned to profile_peak through the sudoers rule for the measurement and
always set back to auto; a run starts only if MemAvailable covers its need;
results are written to --out after every run.

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

CLOCK = "/sys/devices/pci0000:00/0000:00:08.1/0000:c6:00.0/power_dpm_force_performance_level"


def set_clock(level):
    try:
        subprocess.run(["sudo", "-n", "tee", CLOCK], input=level + "\n", text=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def run_once(cmd, args, env):
    full = shlex.split(cmd) + ["--model", args.model, "--prefill-bench", str(args.tokens), "--max-context",
                               str(args.tokens), "--reps", str(args.reps), "--prefill-file", args.prompt]
    if args.warm:
        full += ["--prefill-warm", args.warm]
    if args.resident:
        full += ["--ngram-resident", "1"]
    p = subprocess.run(full, env=env, capture_output=True, text=True, timeout=args.timeout)
    device = [float(m.group(1)) for m in re.finditer(r"device ([0-9.]+) ms", p.stdout + p.stderr)]
    if p.returncode != 0 or len(device) < args.reps:
        raise RuntimeError(f"run failed ({p.returncode}): {' '.join(full)}\n{(p.stdout + p.stderr)[-2000:]}")
    # With --prefill-warm the first timed line is the warm-up prefill.
    reps = device[-args.reps:]
    return statistics.median(reps), reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--warm")
    ap.add_argument("--arm", action="append", required=True, help="name=command [flags]")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--resident", action="store_true", help="--ngram-resident 1 (needs ~30 GB more)")
    ap.add_argument("--build", help="cmake build dir to build first")
    ap.add_argument("--no-pin", action="store_true", help="leave the GPU clock on auto")
    ap.add_argument("--need-gb", type=float, default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="ab_bench.json")
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
    pinned = not args.no_pin and set_clock("profile_peak")

    def restore(*_):
        if pinned:
            set_clock("auto")
        sys.exit(130)

    signal.signal(signal.SIGINT, restore)
    signal.signal(signal.SIGTERM, restore)
    results = {"arms": {n: {"command": c, "runs": []} for n, c in arms}, "clock": "profile_peak" if pinned else "auto",
               "tokens": args.tokens, "reps": args.reps, "resident": args.resident, "started": time.time()}
    try:
        for rnd in range(args.rounds):
            for name, cmd in arms:
                avail = mem_available_gb()
                if avail < need:
                    sys.exit(f"MemAvailable {avail:.0f} GB < {need:.0f} GB needed; refusing to start a run")
                med, reps = run_once(cmd, args, env)
                results["arms"][name]["runs"].append({"round": rnd, "device_ms": med, "reps": reps})
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
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
