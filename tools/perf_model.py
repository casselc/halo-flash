#!/usr/bin/env python3
"""Per-kernel performance model for a prefill, from measured hardware constants
(docs/HARDWARE.md) and per-kernel hardware counters (rocprofv3, see
tools/counter_check.py for what the counters can be trusted with).

  tools/perf_model.py table PMC_DIR            per-kernel counters + kernel-trace time, aggregated by name+grid
  tools/perf_model.py floors PMC_DIR TIMING    per-kernel floors (issue, memory) vs measured time
      TIMING: a `halo_run --kernel-timing 1` log (the last rep's table is used)

PMC_DIR holds pass directories p0..pN, each a rocprofv3 --pmc ... --kernel-trace run of the same
prefill (identical dispatch sequence). Counter facts applied here:
  - WMMA is in no counter; SQ_INSTS reads 0: WMMA counts come from shapes.
  - global loads/stores are SQ_INSTS_FLAT (TEX_LOAD misses them).
  - GL2C_EA_*REQ bytes are half the true traffic: x2. L2 is invalidated between launches.
"""
import collections, csv, glob, os, re, sys

BYTES_X = 2.0  # GL2C EA read requests count half the true bytes; 64 B write requests count exactly (counter_check.py)


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
        wr = c.get("GL2C_EA_WRREQ_64B", 0) * 64
        rows.append((min(ms) if ms else 0, max(ms) if ms else 0, k, calls[k], c, rd, wr))
    rows.sort(key=lambda r: -r[0])
    print(f"{'kernel':42} {'grid':>9} {'calls':>5} {'ms(min..max over passes)':>24} {'VALU M':>8} {'LDS M':>7} "
          f"{'FLAT M':>7} {'SALU M':>7} {'rd MB':>8} {'wr MB':>7}")
    for lo, hi, (name, grid, wg), n, c, rd, wr in rows[:40]:
        print(f"{name[:42]:42} {grid // wg:9d} {n:5d} {lo:11.1f} .. {hi:9.1f} {c.get('SQ_INSTS_VALU', 0) / 1e6:8.1f} "
              f"{c.get('SQ_INSTS_LDS', 0) / 1e6:7.1f} {c.get('SQ_INSTS_FLAT', 0) / 1e6:7.2f} {c.get('SQ_INSTS_SALU', 0) / 1e6:7.1f} "
              f"{rd / 2**20:8.0f} {wr / 2**20:7.0f}")


# ---- hardware constants (docs/HARDWARE.md) ----------------------------------
CLOCK_HZ = 2.88e9
SIMDS = 80
WMMA_CYCLES = 34.0      # per 16x16x16 WMMA per SIMD, f16 / bf16 / iu8 alike
VALU_CYCLES = 1.0       # per wave instruction; serial with WMMA
LDS_CYCLES = 1.0        # issue per wave instruction (bandwidth checked separately)
DRAM_BPS = 240e9

# ---- WMMA counts from shapes (2048-token prefill of this model) --------------
T = 2048


def gemm_tiles(rows, cols, bm, bn, tokens=T):
    """WMMAs of a tiled GEMM including padding: (token tiles x BN/16) x (row tiles x BM/16) x cols/16."""
    gx, gy = -(-tokens // bn), -(-rows // bm)
    return gx * (bn // 16) * gy * (bm // 16) * (cols // 16)


def moe_subs():
    # sum over experts of ceil(n_e / 16) for 20480 (token, expert) pairs over 512 experts in tiles of 64:
    # ~ pairs / 16 plus half a sub-tile per expert of padding.
    return 20480 / 16 + 512 * 0.5


def attention_fa_wmma():
    # per block (16 tokens x 4 heads, 8 waves): 32 WMMAs per wave per 32-key tile; keys 0 .. t0 + 15
    tiles = sum(-(-(16 * i + 16) // 32) for i in range(T // 16))
    return tiles * 6 * 8 * 32


# (kernel, blocks) -> WMMAs per call; shapes from the model config.
WMMA = {
    ("halo_gemm_q8r_f16_lr8", 640): gemm_tiles(10240, 2560, 256, 128),   # DeltaNet qkv
    ("halo_gemm_q8r_f16_lr8", 384): gemm_tiles(6144, 2560, 256, 128),    # DeltaNet z
    ("halo_gemm_q8r_f16_lr8", 768): gemm_tiles(12288, 2560, 256, 128),   # attention q + gate
    ("halo_gemm_q8r_f16_l", 160): (48 * gemm_tiles(2560, 6144, 256, 128) + 48 * gemm_tiles(2560, 640, 256, 128)) / 97,
    ("halo_gemm_q8r_f16_l", 48): gemm_tiles(640, 2560, 256, 128),        # shared expert gate / up
    ("halo_gemm_q8r_f16_s", 128): gemm_tiles(512, 2560, 64, 128),        # attention k / v
    ("halo_gemm_q8r_f16_mb", 32): gemm_tiles(320, 10240, 320, 64),       # mixer down
    ("halo_gemm_q8r_f16_mixt", 640): gemm_tiles(10240, 320, 256, 128),   # mixer up + mix
    ("halo_gemm_f32_bf16x3", 256): 3 * gemm_tiles(512, 2560, 64, 64),    # router, hi/lo x3
    ("halo_gemm_bf16_f16_k128", 256): gemm_tiles(512, 2560, 64, 64),
    ("halo_moe_gate_up_grouped_q4_k", 4165): moe_subs() * (640 // 16) * 2 * (2560 // 16),
    ("halo_moe_gate_up_grouped_q5_k", 4165): moe_subs() * (640 // 16) * 2 * (2560 // 16),
    ("halo_moe_down_grouped_q5_1", 16660): moe_subs() * (2560 // 16) * (640 // 16),
    ("halo_moe_down_grouped_q8r", 16660): moe_subs() * (2560 // 16) * (640 // 16),
    ("halo_gdn_c16_scan3u", 96): 96 * 4 * (T // 16) * (8 * 6 + 8 * 3 + 3 + 3 + 8 * 5),
    ("halo_attention_fa", 768): attention_fa_wmma(),
}


MALL_BPS = 800e9
MB = 1e6
# Unique weight bytes per call (read once from DRAM): routed experts 512 x (gate+up 2 x 640 x 2560 | down 2560 x 640)
# at 4.5 (Q4_K), 5.5 (Q5_K), 6 (Q5_1) and 8.5 (Q8_0 repacked) bits; dense Q8_0 weights at 8.5 bits.
Q8 = 34 / 32
WEIGHT_BYTES = {
    ("halo_moe_gate_up_grouped_q4_k", 4165): 512 * 2 * 640 * 2560 * 144 / 256,
    ("halo_moe_gate_up_grouped_q5_k", 4165): 512 * 2 * 640 * 2560 * 176 / 256,
    ("halo_moe_down_grouped_q5_1", 16660): 512 * 2560 * 640 * 24 / 32,
    ("halo_moe_down_grouped_q8r", 16660): 512 * 2560 * 640 * Q8,
    ("halo_gemm_q8r_f16_lr8", 640): 10240 * 2560 * Q8,
    ("halo_gemm_q8r_f16_lr8", 384): 6144 * 2560 * Q8,
    ("halo_gemm_q8r_f16_lr8", 768): 12288 * 2560 * Q8,
    ("halo_gemm_q8r_f16_mb", 32): 320 * 10240 * Q8,
    ("halo_gemm_q8r_f16_mixt", 640): 10240 * 320 * Q8,
}


# Unique activation bytes per call that cannot stay in the 32 MB MALL (read + written once from DRAM), for the
# hyper-connection chain: the residual is 4 streams x T x 2560 fp32 = 84 MB.
RES = 4 * T * 2560 * 4
STREAM_BYTES = {
    # combine: read res + block_out, write res + the K-blocked fp16 copy (xn itself is skipped)
    ("halo_hc_combine_norm_rows", 8192): RES + T * 2560 * 4 + RES + RES // 2,
    # down projection: the fp16 copy
    ("halo_gemm_q8r_f16_mb", 32): RES // 2,
    # up + mix epilogue: rebuilds xn from res, writes `mixed` in fp32 and fp16
    ("halo_gemm_q8r_f16_mixt", 640): RES + T * 2560 * 6,
}


def timing(path):
    """Last `kernel timing` table of a halo_run --kernel-timing log: name [grid] -> (ms, calls)."""
    lines = open(path).read().split("kernel timing")[-1].splitlines()
    out = {}
    for l in lines:
        m = re.match(r"\s+(\S+) \[(\d+)x(\d+)x(\d+)\]\s+([\d.]+) ms\s+[\d.]+%\s+(\d+) calls", l)
        if m:
            name, gx, gy, gz, ms, calls = m.groups()
            out[(name, int(gx) * int(gy) * int(gz))] = (float(ms), int(calls))
    return out


def per_call(pmc_dir):
    """Mean counters per dispatch, keyed (name, blocks). Collect with tools/pmc.sh: without its keepalive,
    instruction counters read 0 after any > ~70 ms GPU idle gap (docs/HARDWARE.md)."""
    passes = sorted(d for d in glob.glob(os.path.join(pmc_dir, "p*")) if os.path.isdir(d))
    sums, ids = collections.defaultdict(lambda: collections.defaultdict(float)), collections.defaultdict(set)
    for d in passes:
        for r in csv.DictReader(open(os.path.join(d, "run_counter_collection.csv"))):
            key = (r["Kernel_Name"].split("(")[0], int(r["Grid_Size"]) // int(r["Workgroup_Size"]))
            sums[key][r["Counter_Name"]] += float(r["Counter_Value"])
            ids[(key, r["Counter_Name"])].add(r["Dispatch_Id"])
    return {k: {c: v / len(ids[(k, c)]) for c, v in cs.items()} for k, cs in sums.items()}


def floors(pmc_dir, timing_path):
    """Per kernel: issue = (34 x WMMA + VALU + LDS) cycles over 80 SIMDs, WMMA from shapes and VALU / LDS from
    per-call counters; memory = max(unique weight bytes at
    DRAM rate, L2-to-fabric bytes (counters x 2) at MALL rate for GEMMs, DRAM rate for streaming kernels). Floor = max(issue, memory)."""
    tm, fd = timing(timing_path), per_call(pmc_dir)
    rows = []
    for key, (ms, calls) in tm.items():
        c = fd.get(key, {})
        have = c.get("SQ_INSTS_VALU", 0) > 0
        wmma = WMMA.get(key, 0.0)
        valu, lds = c.get("SQ_INSTS_VALU", 0.0), c.get("SQ_INSTS_LDS", 0.0)
        ea = ((c.get("GL2C_EA_RDREQ_128B", 0) * 128 + c.get("GL2C_EA_RDREQ_64B", 0) * 64) * BYTES_X
              + c.get("GL2C_EA_WRREQ_64B", 0) * 64)
        per = lambda cyc: cyc / SIMDS / CLOCK_HZ * 1e3 * calls
        wmma_ms, valu_ms = per(WMMA_CYCLES * wmma), per(VALU_CYCLES * valu + LDS_CYCLES * lds)
        dram = (WEIGHT_BYTES.get(key, 0.0) + STREAM_BYTES.get(key, 0.0)) / DRAM_BPS * 1e3 * calls
        # GEMMs re-read MALL-resident tiles; streaming kernels touch each byte once, mostly over > 32 MB
        mall = ea / (MALL_BPS if wmma else DRAM_BPS) * 1e3 * calls
        issue, mem = wmma_ms + valu_ms, max(dram, mall)
        rows.append(dict(ms=ms, key=key, calls=calls, wmma=wmma_ms, valu=valu_ms, issue=issue, dram=dram,
                         mall=mall, floor=max(issue, mem), have=have))
    rows.sort(key=lambda r: -r["ms"])
    fmt = lambda r, k: f"{r[k]:6.1f}" if r["have"] else "     ?"
    tot = sum(r["ms"] for r in rows)
    print(f"{'kernel':34} {'blocks':>6} {'calls':>5} {'meas':>7} {'WMMA':>6} {'VALU+L':>6} {'DRAMw':>6} {'fabric':>6} "
          f"{'floor':>6} {'gap':>6} {'eff':>4}")
    for r in rows[:30]:
        name, blocks = r["key"]
        print(f"{name[:34]:34} {blocks:6d} {r['calls']:5d} {r['ms']:7.1f} {r['wmma']:6.1f} "
              f"{fmt(r, 'valu')} {r['dram']:6.1f} "
              f"{fmt(r, 'mall')} {r['floor']:6.1f} "
              f"{r['ms'] - r['floor']:6.1f} {r['floor'] / r['ms'] * 100:3.0f}%")
    fl = sum(r["floor"] for r in rows)
    known = [r for r in rows if r["have"] or r["key"] in WMMA]
    print(f"\ntotal measured {tot:.0f} ms; sum of floors {fl:.0f} ms ({fl / tot * 100:.0f}%); "
          f"kernels with no counters and no shape model: {sum(r['ms'] for r in rows if r not in known):.0f} ms")
    chain = [r for r in rows if r["key"][0] in ("halo_hc_combine_norm_rows", "halo_gemm_q8r_f16_mb",
                                                 "halo_gemm_q8r_f16_mixt", "halo_scale_silu", "halo_inject_reduce")
             or r["key"] == ("halo_f16_rows", 2560)]
    fused = 96 * (2 * RES + T * 2560 * 4 + T * 2560 * 6) / DRAM_BPS * 1e3
    print(f"hyper-connection mixer chain: {sum(r['ms'] for r in chain):.0f} ms measured, "
          f"{sum(r['floor'] for r in chain):.0f} ms sum of floors; one fused pass per mixer (res read + written once): "
          f"DRAM floor {fused:.0f} ms")
    print("largest gaps (measured - floor):")
    for r in sorted(rows, key=lambda r: -(r["ms"] - r["floor"]))[:10]:
        print(f"  {r['key'][0]} [{r['key'][1]}]: {r['ms'] - r['floor']:.1f} ms")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "floors":
        floors(sys.argv[2], sys.argv[3])
        sys.exit(0)

    if len(sys.argv) < 3 or sys.argv[1] != "table":
        print(__doc__)
        sys.exit(2)
    table(sys.argv[2])
