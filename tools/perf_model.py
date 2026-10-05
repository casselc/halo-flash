#!/usr/bin/env python3
"""Per-kernel performance model for a prefill, from measured hardware constants
(docs/HARDWARE.md) and per-kernel hardware counters (rocprofv3, see
tools/counter_check.py for what the counters can be trusted with).

  tools/perf_model.py table PMC_DIR    per-kernel counters + kernel-trace time, aggregated by name+grid

PMC_DIR holds pass directories p0..pN, each a rocprofv3 --pmc ... --kernel-trace run of the same
prefill (identical dispatch sequence). Counter facts applied here:
  - WMMA is in no counter; SQ_INSTS reads 0: WMMA counts come from shapes.
  - global loads/stores are SQ_INSTS_FLAT (TEX_LOAD misses them).
  - GL2C_EA_*REQ bytes are half the true traffic: x2. L2 is invalidated between launches.
"""
import collections, csv, glob, os, sys

BYTES_X = 2.0  # GL2C EA request bytes read half the true traffic (counter_check.py)


def load(pmc_dir):
    passes = sorted(d for d in glob.glob(os.path.join(pmc_dir, "p*")) if os.path.isdir(d))
    per = collections.defaultdict(lambda: collections.defaultdict(float))  # (name, grid, wg) -> counters
    calls = collections.Counter()
    dur = collections.defaultdict(list)  # per pass: summed kernel-trace ns, to check passes agree
    for pi, d in enumerate(passes):
        trace = {}
        tp = os.path.join(d, "run_kernel_trace.csv")
        if os.path.exists(tp):
            for r in csv.DictReader(open(tp)):
                trace[r["Correlation_Id"]] = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        seen = set()
        pass_dur = collections.defaultdict(float)
        for r in csv.DictReader(open(os.path.join(d, "run_counter_collection.csv"))):
            key = (r["Kernel_Name"].split("(")[0], int(r["Grid_Size"]), int(r["Workgroup_Size"]))
            per[key][r["Counter_Name"]] += float(r["Counter_Value"])
            did = (r["Dispatch_Id"])
            if did not in seen:
                seen.add(did)
                if pi == 0:
                    calls[key] += 1
                pass_dur[key] += int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        for k, v in pass_dur.items():
            dur[k].append(v / 1e6)
    return per, calls, dur


def table(pmc_dir):
    per, calls, dur = load(pmc_dir)
    rows = []
    for k, c in per.items():
        ms = dur[k]
        rd = (c.get("GL2C_EA_RDREQ_128B", 0) * 128 + c.get("GL2C_EA_RDREQ_64B", 0) * 64) * BYTES_X
        wr = c.get("GL2C_EA_WRREQ_64B", 0) * 64 * BYTES_X
        rows.append((min(ms) if ms else 0, max(ms) if ms else 0, k, calls[k], c, rd, wr))
    rows.sort(key=lambda r: -r[0])
    print(f"{'kernel':42} {'grid':>9} {'calls':>5} {'ms(min..max over passes)':>24} {'VALU M':>8} {'LDS M':>7} "
          f"{'FLAT M':>7} {'SALU M':>7} {'rd MB':>8} {'wr MB':>7}")
    for lo, hi, (name, grid, wg), n, c, rd, wr in rows[:40]:
        print(f"{name[:42]:42} {grid // wg:9d} {n:5d} {lo:11.1f} .. {hi:9.1f} {c.get('SQ_INSTS_VALU', 0) / 1e6:8.1f} "
              f"{c.get('SQ_INSTS_LDS', 0) / 1e6:7.1f} {c.get('SQ_INSTS_FLAT', 0) / 1e6:7.2f} {c.get('SQ_INSTS_SALU', 0) / 1e6:7.1f} "
              f"{rd / 2**20:8.0f} {wr / 2**20:7.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "table":
        print(__doc__)
        sys.exit(2)
    table(sys.argv[2])
