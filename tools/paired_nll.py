#!/usr/bin/env python3
"""Paired per-case comparison of halo_score runs (stdlib only).

Each arm is one or more halo_score logs (chunks of one run, e.g. --start /
--limit pieces), joined by commas. The first arm is the baseline.

  paired_nll.py base=b0.log,b1.log A=a0.log,a1.log [B8=...]

Per arm: cases, target tokens, NLL (token-weighted, recomputed from the
cases), API top-token agreement (from each chunk's final line, weighted by
its aligned tokens). Against the baseline: the per-case NLL difference,
mean +- standard error, cases up / down, largest |difference|, and the
token-weighted difference. Cases are matched by id; a case missing from
either arm is reported and left out.
"""
import math
import re
import sys

CASE = re.compile(r"^(\S+)\s+prompt\s+(\d+)\s+targets\s+(\d+)\s+nll\s+([0-9.eE+-]+)")
TOTAL = re.compile(r"^cases (\d+), target tokens (\d+), aligned (\d+)")
AGREE = re.compile(r"API top-token agreement ([0-9.]+)%")


def load(paths):
    cases, aligned, agree = {}, 0, 0.0
    for p in paths:
        chunk_aligned = None
        with open(p) as f:
            for line in f:
                m = CASE.match(line)
                if m:
                    cases[m.group(1)] = (int(m.group(3)), float(m.group(4)))
                    continue
                m = TOTAL.match(line)
                if m:
                    chunk_aligned = int(m.group(3))
                    continue
                m = AGREE.search(line)
                if m and chunk_aligned is not None:
                    aligned += chunk_aligned
                    agree += float(m.group(1)) / 100.0 * chunk_aligned
                    chunk_aligned = None
    return cases, aligned, agree


def main():
    arms = []
    for a in sys.argv[1:]:
        name, _, paths = a.partition("=")
        arms.append((name, load(paths.split(","))))
    if not arms:
        sys.exit(__doc__)
    base_name, (base, _, _) = arms[0]
    for name, (cases, aligned, agree) in arms:
        toks = sum(t for t, _ in cases.values())
        nll = sum(t * v for t, v in cases.values()) / toks if toks else float("nan")
        print(f"{name:10s} cases {len(cases):4d}  tokens {toks:6d}  NLL {nll:.6f}  "
              f"agreement {100 * agree / aligned if aligned else float('nan'):.2f}% ({aligned} aligned)")
    for name, (cases, _, _) in arms[1:]:
        ids = sorted(set(base) & set(cases))
        missing = sorted(set(base) ^ set(cases))
        d = [cases[i][1] - base[i][1] for i in ids]
        n = len(d)
        if n == 0:
            print(f"{name} vs {base_name}: no common cases")
            continue
        mean = sum(d) / n
        se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
        toks = sum(base[i][0] for i in ids)
        wd = sum(base[i][0] * (cases[i][1] - base[i][1]) for i in ids) / toks
        up = sum(x > 0 for x in d)
        down = sum(x < 0 for x in d)
        print(f"{name} vs {base_name}: {n} cases, per-case mean {mean:+.4f} +- {se:.4f} (SE), "
              f"{up} up / {down} down, largest |d| {max(abs(x) for x in d):.4f}; token-weighted {wd:+.5f}"
              + (f"; unmatched {missing}" if missing else ""))


if __name__ == "__main__":
    main()
