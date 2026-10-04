#!/usr/bin/env python3
"""Summarize a rocprofv3 thread trace (--att) per traced dispatch.

  att_summary.py OUTDIR [--top N]

For every `stats_*.csv` rocprofv3 wrote (one per traced dispatch): the total
traced latency (cycles summed over instructions of the traced waves), the
share stalled, latency and stall by instruction class (s_waitcnt, VMEM,
LDS, WMMA, VALU, barrier, ...), and the instructions with the most stall
cycles. Read it as where one SIMD's waves spent their cycles, not as a
kernel time: the trace covers the target CU/SIMD only.
"""

import argparse
import collections
import csv
import glob
import os
import sys


def classify(ins):
    op = ins.split()[0]
    if op.startswith("s_waitcnt"):
        return "s_waitcnt " + (ins.split()[1].split("(")[0] if len(ins.split()) > 1 else "")
    for prefix, name in (("v_wmma", "wmma"), ("global_load", "vmem load"), ("global_store", "vmem store"),
                         ("buffer_", "vmem"), ("scratch_", "scratch"), ("ds_load", "lds load"),
                         ("ds_store", "lds store"), ("ds_", "lds other"), ("s_barrier", "barrier"),
                         ("v_dual", "valu dual"), ("v_", "valu"), ("s_", "salu/branch")):
        if op.startswith(prefix):
            return name
    return op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.outdir, "**", "stats_*.csv"), recursive=True))
    if not files:
        sys.exit(f"no stats_*.csv under {args.outdir} (did the trace capture a dispatch?)")
    for f in files:
        rows = list(csv.DictReader(open(f)))
        for r in rows:
            for k in ("Hitcount", "Latency", "Stall", "Idle"):
                r[k] = int(r.get(k) or 0)
        rows = [r for r in rows if r.get("Instruction") and not r["Instruction"].startswith(";")]
        tl = sum(r["Latency"] for r in rows)
        ts = sum(r["Stall"] for r in rows)
        print(f"{os.path.basename(f)}: latency {tl:,} cycles, stalled {100 * ts / max(tl, 1):.0f}%, "
              f"idle {sum(r['Idle'] for r in rows):,}")
        lat, st = collections.Counter(), collections.Counter()
        for r in rows:
            c = classify(r["Instruction"])
            lat[c] += r["Latency"]
            st[c] += r["Stall"]
        for c, v in lat.most_common(10):
            print(f"  {c:26} {100 * v / max(tl, 1):5.1f}% of latency, {100 * st[c] / max(tl, 1):5.1f}% stalled")
        print("  most stalled instructions:")
        for r in sorted(rows, key=lambda r: -r["Stall"])[:args.top]:
            print(f"    {r['Stall']:>12,} stall  x{r['Hitcount']:<7} {r['Instruction'][:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
