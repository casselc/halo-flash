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


# ---- v2: design-level terms (2026-10-05) -------------------------------------
# v1 priced bytes at peak bandwidth and issue at peak rate, per kernel. Two exact designs predicted that way at
# 1.25 and 1.55 ms measured 2.13. v2 adds the three terms that decided them, every constant from a probe
# (never fitted to the kernels it is validated on):
#   1. access-pattern bandwidth: `halo_hwprobe sweep` measures each activation pattern at the kernel's own grid
#      and residency (ms for the whole read);
#   2. parallelism: a wave's dependent load round takes max(latency, bytes in flight GPU-wide / bandwidth)
#      (`halo_hwprobe rounds`: DRAM ~440 ns and ~185 GB/s, L2 ~240 ns and ~1000 GB/s);
#   3. serialization: phases every block runs in lockstep add; inside a GEMM stage only WMMA overlaps the next
#      stage's loads (one-stage prefetch), VALU and LDS issue and the barriers do not.
# And one interference term: a kernel after one that wrote >= 32 MB also pays the MALL's dirty lines (32 MB)
# draining at its own pattern's bandwidth ("writes are paid by the next kernel").
US = 1e-6
BARRIER = 0.04 * US
DRAM_LAT, DRAM_DEP_BPS = 0.44 * US, 185e9
L2_LAT, L2_BPS = 0.24 * US, 1000e9
MALL_DIRTY = 32 * 2**20

# halo_hwprobe sweep, pinned, 2026-10-05 (ms for the whole read at the kernel's grid and residency)
SWEEP_MS = {
    "res fp32 row-major, 32 blocks x 64 tokens (_mr)": 0.660,
    "fp16 row-major, 32 x 64 (_m)": 0.360,
    "fp16 K-blocked, 32 x 64 (_mb)": 0.153,
    "res fp32 K-blocked, 32 x 64": 0.351,
    "mix epilogue res, row tiles fastest": 0.459,
    "mix epilogue res, token tiles fastest": 1.653,
}
COMBINE_BPS = 204e9  # halo_hwprobe stream: the combine's traffic (res += out in place + fp16 copy)


def dram_round(waves, loads):
    """One dependent round of `loads` 128-byte loads per wave, `waves` waves in the same phase."""
    return max(DRAM_LAT, waves * loads * 128 / DRAM_DEP_BPS)


def l2_round(waves, loads):
    return max(L2_LAT, waves * loads * 128 / L2_BPS)


def simd_s(wave_instr, simds):
    """Seconds to issue `wave_instr` wave-instructions (VALU / LDS, ~1 cycle each) spread over `simds` SIMDs."""
    return wave_instr / simds / CLOCK_HZ


def gemm_stages(stages, valu_lds_per_simd_s, wmma_per_simd_s, mem_s):
    """GemmQ8rF16's loop: per stage the VALU/LDS issue and two barriers in series, then WMMA overlapping the next
    stage's loads (one-stage prefetch)."""
    return stages * (valu_lds_per_simd_s / stages + 2 * BARRIER + max(wmma_per_simd_s, mem_s) / stages)


def down_gemm(sweep_key, valu_lds, after_writes=False):
    """The mixer's down projection (320 x 10240, T 2048): 32 blocks of 64 tokens, LDS 55 KB (two blocks a WGP:
    a block's 8 waves effectively share 2 SIMDs), 160 stages of 64."""
    wmma = 64 // 16 * (320 // 16) * (10240 // 16)  # per block
    mem = SWEEP_MS[sweep_key] * 1e-3
    t = gemm_stages(160, simd_s(valu_lds / 32, 2), WMMA_CYCLES * wmma / 2 / CLOCK_HZ, mem)
    if after_writes:
        t += MALL_DIRTY / (84e6 / mem if "fp32" in sweep_key else 42e6 / mem)
    return t


def combine_norm(dram_bytes, l2_bytes, valu_lds):
    """halo_hc_combine_norm_rows: 8192 short blocks at 16 waves/SIMD, not in lockstep, so DRAM traffic, L2 traffic
    and issue overlap; the slowest of them."""
    return max(dram_bytes / COMBINE_BPS, l2_bytes / L2_BPS, simd_s(valu_lds, SIMDS))


def combine_down(valu_lds):
    """halo_hc_combine_down (study): per stream, a lockstep combine phase in every block (each of 256 waves takes
    8 rows; per row 10 DRAM rounds of 16 loads, then 10 L2 rounds of 48: residual, gamma, 4 inject rows), then the
    stream's 40 GEMM stages reading the rows just written (21 MB live: MALL, ~800 GB/s)."""
    waves = 256
    phase_a = 4 * 8 * (10 * dram_round(waves, 16) + 10 * l2_round(waves, 48))
    gemm_valu = 15.3e6  # the GEMM part's issue, as _mb's
    wmma = 64 // 16 * (320 // 16) * (10240 // 16)
    phase_b = gemm_stages(160, simd_s(gemm_valu / 32, 2), WMMA_CYCLES * wmma / 2 / CLOCK_HZ, 84e6 / 800e9)
    return phase_a + simd_s((valu_lds - gemm_valu) / 32, 2) + phase_b


def mixt(epilogue_key, valu_lds):
    """halo_gemm_q8r_f16_mixt: 640 blocks (2 a CU), K 320 (10 short stages); the epilogue reads the residual
    (pattern measured by the sweep) and writes `mixed` (31 MB). A block's GEMM and epilogue run in series."""
    wmma_total = 640 * (128 // 16) * (256 // 16) * (320 // 16)
    return (WMMA_CYCLES * wmma_total / SIMDS / CLOCK_HZ + simd_s(valu_lds, SIMDS)
            + SWEEP_MS[epilogue_key] * 1e-3 + 31e6 / 222e9)


def validate():
    """Predictions against measurements the model was not fitted to (study harness = halo_gemm_bench, warm,
    pinned; engine = per-call kernel timing). Instruction counts per call from tools/pmc.sh."""
    RES = 4 * 2048 * 2560 * 4
    cases = [
        # (case, predicted s, measured s, source of the measurement)
        ("down GEMM _mb (K-blocked fp16), study", down_gemm("fp16 K-blocked, 32 x 64 (_mb)", 15.27e6), 0.442e-3,
         "combinedown"),
        ("down GEMM _m (row-major fp16), study", down_gemm("fp16 row-major, 32 x 64 (_m)", 15.27e6), 0.67e-3,
         "HARDWARE.md"),
        ("down GEMM _mr (rebuild from res), study", down_gemm("res fp32 row-major, 32 blocks x 64 tokens (_mr)",
                                                              18.10e6), 1.038e-3, "combinedown"),
        ("down GEMM _mb after 96 MB of writes, study", down_gemm("fp16 K-blocked, 32 x 64 (_mb)", 15.27e6, True),
         0.62e-3, "HARDWARE.md"),
        ("down GEMM _m after 96 MB of writes, study", down_gemm("fp16 row-major, 32 x 64 (_m)", 15.27e6, True),
         1.33e-3, "HARDWARE.md"),
        ("down GEMM _mb in engine", down_gemm("fp16 K-blocked, 32 x 64 (_mb)", 15.27e6, True), 59.4e-3 / 96,
         "kt_model2"),
        ("down GEMM _m in engine (before blocked xh)", down_gemm("fp16 row-major, 32 x 64 (_m)", 15.27e6, True),
         101.9e-3 / 96, "ROOFLINE"),
        ("combine_norm (y16, inject), study", combine_norm(2 * RES + 2048 * 2560 * 4 + RES // 2, 8192 * 50e3, 39.6e6),
         1.289e-3, "combinestudy"),
        ("combine_norm, no y16, study", combine_norm(2 * RES + 2048 * 2560 * 4, 8192 * 50e3, 37e6), 0.939e-3,
         "combinestudy"),
        ("combine_norm before the load batching, study", combine_norm(2 * RES + 2048 * 2560 * 4 + RES // 2,
                                                                      8192 * 50e3, 39.6e6), 1.704e-3, "combinestudy"),
        ("combine + down fused (combine_down), study", combine_down(44.36e6), 2.138e-3, "combinedown"),
        ("mixt, row tiles fastest, study", mixt("mix epilogue res, row tiles fastest", 0.15e-3 * SIMDS * CLOCK_HZ),
         1.097e-3, "mixstudy"),
        ("mixt, token tiles fastest, study", mixt("mix epilogue res, token tiles fastest", 0.15e-3 * SIMDS * CLOCK_HZ),
         1.41e-3, "core.hip comment (older build: weak)"),
    ]
    print(f"{'case':48} {'predicted':>9} {'measured':>9} {'error':>7}  source")
    errs = []
    for name, p, m, src in cases:
        errs.append(abs(p - m) / m)
        print(f"{name:48} {p * 1e3:8.3f}  {m * 1e3:8.3f}  {(p - m) / m * 100:+6.0f}%  {src}")
    print(f"\nmedian |error| {sorted(errs)[len(errs) // 2] * 100:.0f}%, max {max(errs) * 100:.0f}%")
    # The two design decisions of 2026-10-05: did the model say "loses"?
    base = combine_norm(2 * RES + 2048 * 2560 * 4 + RES // 2, 8192 * 50e3, 39.6e6) + down_gemm(
        "fp16 K-blocked, 32 x 64 (_mb)", 15.27e6)
    fused = combine_down(44.36e6)
    rebuild = combine_norm(2 * RES + 2048 * 2560 * 4, 8192 * 50e3, 37e6) + down_gemm(
        "res fp32 row-major, 32 blocks x 64 tokens (_mr)", 18.10e6)
    print(f"decisions: current pair {base * 1e3:.2f} ms (measured 1.68); fused {fused * 1e3:.2f} (2.13); "
          f"rebuild pair {rebuild * 1e3:.2f} (2.13)")
    # Past in-engine A/B deltas (ms per prefill).
    xh = 96 * (down_gemm("fp16 K-blocked, 32 x 64 (_mb)", 15.27e6, True)
               - down_gemm("fp16 row-major, 32 x 64 (_m)", 15.27e6, True)) * 1e3
    print(f"A/B: blocked xh (down GEMM kernel) predicted {xh:+.1f} ms, measured -46.7; combine_norm load batching "
          f"predicted +0.0 (the model has no per-iteration wait term), measured -40.5 (device)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "validate":
        validate()
        sys.exit(0)
    if len(sys.argv) >= 4 and sys.argv[1] == "floors":
        floors(sys.argv[2], sys.argv[3])
        sys.exit(0)

    if len(sys.argv) < 3 or sys.argv[1] != "table":
        print(__doc__)
        sys.exit(2)
    table(sys.argv[2])
