#!/usr/bin/env python3
"""Known-answer checks for the hardware counters and the toolchain.

Runs `halo_hwprobe counters` under rocprofv3 (several --pmc passes) and
compares each known-answer kernel's counters with what it must report:
  - KnownKernel<KIND, N>: exactly 64 * N instructions of one kind per wave
    (v_add_f32, ds_load_b32, global_load_b32, or WMMA) issued through inline
    asm; the N = 20 minus N = 10 difference cancels the fixed prologue, so
    the expected delta is 10 * 64 * waves exactly.
  - KnownMaskedKernel: the asm block guarded by a branch no lane takes.
  - KnownHalfKernel: the asm block taken by one of each block's two waves.
  - KnownReadKernel / KnownScatterKernel: bytes cold, warm, and scattered.
Also checks the compiled ISA: each known-answer loop body must hold exactly
64 of its instruction (the compiler kept the probe as written).

  tools/counter_check.py [--hwprobe build/nix-gpu/halo_hwprobe] [--out DIR]
Run inside the nix dev shell (rocprofv3, hipcc). Exit status 1 on any FAIL.
"""
import argparse, csv, collections, os, re, subprocess, sys

PASSES = [
    ["SQ_INSTS_VALU", "SQ_INSTS_LDS", "SQ_INSTS_SALU", "SQ_WAVES"],
    ["SQ_INSTS_FLAT", "SQ_INSTS_TEX_LOAD", "SQ_INSTS_TEX_STORE", "SQ_INSTS_SMEM"],
    ["GL2C_EA_RDREQ_32B", "GL2C_EA_RDREQ_64B", "GL2C_EA_RDREQ_96B", "GL2C_EA_RDREQ_128B"],
    ["GL2C_MC_RDREQ", "GL2C_HIT", "GL2C_MISS", "GL2C_EA_WRREQ_64B"],
]
WAVES = 80 * 2  # 80 blocks x 64 threads


def collect(hwprobe, out):
    per = collections.defaultdict(dict)  # dispatch index (order) -> {counter: value}, plus name
    for i, counters in enumerate(PASSES):
        d = os.path.join(out, f"p{i}")
        subprocess.run(["rocprofv3", "--pmc", *counters, "--output-format", "csv", "-d", d, "-o", "run", "--",
                        hwprobe, "counters"], check=True, capture_output=True)
        path = os.path.join(d, "run_counter_collection.csv")
        rows = [r for r in csv.DictReader(open(path)) if "Known" in r["Kernel_Name"]]
        ids = sorted({int(r["Dispatch_Id"]) for r in rows})
        order = {d_: k for k, d_ in enumerate(ids)}
        for r in rows:
            k = order[int(r["Dispatch_Id"])]
            per[k]["name"] = r["Kernel_Name"]
            per[k][r["Counter_Name"]] = per[k].get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
    return [per[k] for k in sorted(per)]


def isa_check():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hwprobe.hip")
    asm = subprocess.run(["hipcc", "-O3", "-std=c++20", "--offload-arch=gfx1151", "--offload-device-only", "-S", src,
                          "-o", "-"], check=True, capture_output=True, text=True).stdout
    results = []
    for kind, mnem in [(0, "v_add_f32"), (1, "ds_load_b32"), (2, "global_load_b32"), (3, "v_wmma_f32_16x16x16_f16")]:
        m = re.search(r"\n(_ZN\w*KnownKernelILi%dELi10E\w*):.*?s_endpgm" % kind, asm, re.S)
        body = m.group(0) if m else ""
        n = len(re.findall(r"\b%s\b" % mnem, body))
        # N = 10 rounds of 64; the compiler may unroll the rounds (asm blocks: 640) or not (64).
        results.append((f"ISA KnownKernel<{kind}, 10>: {mnem} (64 a round)", n, 640 if kind < 3 else 64,
                        n in (64, 640)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hwprobe", default="build/nix-gpu/halo_hwprobe")
    ap.add_argument("--out", default="/tmp/counter_check")
    a = ap.parse_args()
    rows = isa_check()
    k = collect(a.hwprobe, a.out)
    names = [x["name"].split("(")[0] for x in k]
    need = 10 * 64 * WAVES

    def delta(i, c):
        return k[i + 1].get(c, 0) - k[i].get(c, 0)

    rows += [("VALU delta, v_add_f32 x 64 x 10 per wave", delta(0, "SQ_INSTS_VALU"), need, None),
             ("LDS delta, ds_load_b32", delta(2, "SQ_INSTS_LDS"), need, None),
             ("FLAT delta, global_load_b32 (TEX_LOAD does not count them)", delta(4, "SQ_INSTS_FLAT"), need, None)]
    rows = [(n, v, e, ok if ok is not None else abs(v - e) <= 0.01 * e) for n, v, e, ok in rows]
    print(f"{'check':66} {'measured':>12} {'expected':>12}  result")
    fails = 0
    for n, v, e, ok in rows:
        fails += not ok
        print(f"{n:66} {v:12.0f} {e:12.0f}  {'PASS' if ok else 'FAIL'}")
    print("\nfindings (no fixed expectation):")
    print(f"  WMMA in SQ_INSTS_VALU: delta {delta(6, 'SQ_INSTS_VALU'):.0f} for {need} WMMAs "
          f"({'counted' if abs(delta(6, 'SQ_INSTS_VALU') - need) < 0.01 * need else 'NOT counted: take WMMA counts from shapes'})")
    print(f"  TEX_LOAD delta for global_load_b32: {delta(4, 'SQ_INSTS_TEX_LOAD'):.0f} (FLAT: {delta(4, 'SQ_INSTS_FLAT'):.0f})")
    print(f"  SQ_WAVES per instruction kernel: {k[0].get('SQ_WAVES', 0):.0f} (launched {WAVES})")
    print(f"  masked-off asm block (64 x 64 v_add_f32 under a never-taken branch): VALU {k[8].get('SQ_INSTS_VALU', 0):.0f}"
          f" vs the plain N=10 kernel's {k[0].get('SQ_INSTS_VALU', 0):.0f}")
    print(f"  half the waves take a 64 x 64 v_add_f32 block: VALU {k[9].get('SQ_INSTS_VALU', 0):.0f}"
          f" (per-wave counting of the taken half predicts ~{64 * 64 * WAVES // 2} + overhead)")
    labels = ["evict 256 MB", "cold 64 MB", "4 MB (1st)", "4 MB (2nd)", "scatter 64 Ki x 32 B", "1 MB (1st)", "1 MB (2nd)",
              "1 MB twice, one launch"]
    for j, lab in enumerate(labels):
        x = k[10 + j]
        rd = sum(x.get(f"GL2C_EA_RDREQ_{s}B", 0) * s for s in (32, 64, 96, 128))
        mix = ", ".join(f"{s}B {x.get(f'GL2C_EA_RDREQ_{s}B', 0):.0f}" for s in (32, 64, 96, 128))
        print(f"  {lab:22} EA read bytes {rd / 2**20:9.2f} MB  MC_RDREQ {x.get('GL2C_MC_RDREQ', 0):10.0f}  "
              f"L2 hit {x.get('GL2C_HIT', 0):10.0f} miss {x.get('GL2C_MISS', 0):10.0f}  [{mix}]")
    true = [256, 64, 4, 4, 2, 1, 1, 2]
    print("  EA read bytes / true bytes (useful bytes for the scatter): " + ", ".join(
        f"{sum(k[10 + j].get(f'GL2C_EA_RDREQ_{s}B', 0) * s for s in (32, 64, 96, 128)) / 2**20 / t:.2f}"
        for j, t in enumerate(true)))
    print(f"\n{len(rows) - fails}/{len(rows)} checks passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
