#!/usr/bin/env python3
"""Summarize a reboot_ab.sh run: reboot_<label>.json + reboot_<label>.txt; with two labels, compare them.

  reboot_summary.py pre            -> reboot_pre.json, reboot_pre.txt
  reboot_summary.py compare pre post -> reboot_compare_pre_post.txt
"""
import glob
import json
import os
import re
import statistics
import sys

L2 = "/data/halo-work/streams/L2"


def read(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return ""


def med(v):
    return statistics.median(v) if v else None


def summarize(label):
    d = os.path.join(L2, f"reboot_{label}")
    r = {"label": label}
    sys_txt = read(os.path.join(d, "sys.txt"))
    s = {}
    for key in ("date", "cmdline", "gpu_iommu_group_type", "gpu_link", "iommu_groups"):
        m = re.search(rf"^{key}: (.*)$", sys_txt, re.M)
        s[key] = m.group(1).strip() if m else None
    s["iommu_types"] = {t: int(n) for n, t in re.findall(r"^\s+(\d+) (\S+)$", sys_txt, re.M)}
    s["nvme"] = re.findall(r"^nvme .*$", sys_txt, re.M)
    r["system"] = s
    pf = {}
    for t in (2048, 8192):
        j = os.path.join(d, f"prefill_{t}.json")
        if os.path.exists(j):
            x = json.load(open(j))
            runs = x["arms"]["main50d5a35"]["runs"]
            v = [q["device_ms"] for q in runs]
            pf[str(t)] = {"rounds_ms": v, "median_ms": med(v), "min_ms": min(v) if v else None,
                          "max_ms": max(v) if v else None, "tok_s": t / med(v) * 1e3 if v else None,
                          "clock": x.get("clock")}
    r["prefill"] = pf
    dec = {}
    loads = []
    for ctx in ("short", "7k"):
        v, mins = [], []
        for f in sorted(glob.glob(os.path.join(d, f"decode_{ctx}_r*.txt"))):
            t = read(f)
            m = re.search(r"steady decode: median ([0-9.]+) ms/step .*min ([0-9.]+) ms, over (\d+) steps", t)
            if m:
                v.append(float(m.group(1)))
                mins.append(float(m.group(2)))
            loads += [float(x) for x in re.findall(r"ready in ([0-9.]+) s", t)]
        kt = read(os.path.join(d, f"decode_{ctx}_kt.txt"))
        top = []
        for m in re.finditer(r"^  (\S+)\s+([0-9.]+) ms/step\s+([0-9.]+)%\s+([0-9.]+) calls/step\s+([0-9.]+) us/call", kt, re.M):
            top.append({"kernel": m.group(1), "ms_step": float(m.group(2)), "us_call": float(m.group(5))})
        top.sort(key=lambda k: -k["ms_step"])
        m = re.search(r"kernels ([0-9.]+) ms/step", kt)
        ms = re.search(r"steady decode: median ([0-9.]+)", kt)
        dec[ctx] = {"rounds_ms": v, "median_ms": med(v), "mins_ms": mins,
                    "eager_kernel_sum_ms": float(m.group(1)) if m else None,
                    "eager_steady_ms": float(ms.group(1)) if ms else None, "top_kernels": top[:10]}
    r["decode"] = dec
    r["model_ready_s"] = {"values": loads, "median": med(loads)}
    ng = {}
    pat = re.compile(r"prefill chunk (\d+) tokens: host prep ([0-9.]+) ms \(hash ([0-9.]+), n-gram read ([0-9.]+), "
                     r"deferred ([0-9.]+)\), upload ([0-9.]+) ms, device ([0-9.]+) ms")
    for name in ("ngram_novel_nocache", "ngram_novel_cache"):
        t = read(os.path.join(d, name + ".txt"))
        rows = [dict(tokens=int(a), host_prep=float(b), hash=float(c), read=float(e), deferred=float(f),
                     upload=float(g), device=float(h)) for a, b, c, e, f, g, h in pat.findall(t)]
        ng[name] = {"chunks": rows, "read_plus_deferred_ms": [x["read"] + x["deferred"] for x in rows]}
    dd = read(os.path.join(d, "nvme_dd.txt"))
    m = re.search(r"copied, ([0-9.]+) s, ([0-9.,]+ [GM]B/s)", dd)
    ng["nvme_dd_2GiB_odirect"] = m.group(2) if m else None
    r["ngram"] = ng
    pr = read(os.path.join(d, "probe_key.txt"))
    rows = []
    for m in re.finditer(r"^\s+(\w+)\s+K\s+(\d+) Wb\s+(\S+) rp\s+(\S+) NB\s+(\d+) lds\s+(\d+)\s+([0-9.]+) us\s+([0-9.]+) GB/s\s+clk\s+(\d+) MHz", pr, re.M):
        rows.append({"mem": m.group(1), "K": int(m.group(2)), "Wb": m.group(3), "rp": m.group(4), "NB": int(m.group(5)),
                     "lds": int(m.group(6)), "us": float(m.group(7)), "gbs": float(m.group(8)), "mhz": int(m.group(9))})
    r["probe_pageloc_key"] = rows
    m = re.findall(r"^\s+(\d+)\s+(cols-fast|rows-fast)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+) GB/s", pr, re.M)
    r["probe_dram"] = [{"seg": int(a), "order": b, "read": float(c), "write": float(e), "copy": float(f)} for a, b, c, e, f in m]
    r["probe_clock"] = re.findall(r"^\s+\d+ blocks x\s+\d+ threads: .*$", pr, re.M)
    r["probe_latency"] = [l.strip() for l in pr.splitlines() if re.search(r"\bns\b", l)][:20]
    json.dump(r, open(os.path.join(L2, f"reboot_{label}.json"), "w"), indent=1)
    out = [f"reboot A/B baseline '{label}' ({s['date']})", f"cmdline: {s['cmdline']}",
           f"iommu: {s['iommu_types']} (gpu group: {s['gpu_iommu_group_type']}); gpu link {s['gpu_link']}"]
    out += [f"  {n}" for n in s["nvme"]]
    for t, p in pf.items():
        out.append(f"prefill {t}: median {p['median_ms']:.1f} ms ({p['tok_s']:.0f} tok/s), rounds {p['rounds_ms']}, clock {p['clock']}")
    for ctx, p in dec.items():
        if p["median_ms"]:
            out.append(f"decode {ctx}: median {p['median_ms']:.2f} ms/step, rounds {p['rounds_ms']}; eager kernel sum "
                       f"{p['eager_kernel_sum_ms']}, eager steady {p['eager_steady_ms']}")
    out.append(f"model ready (s): {loads}")
    for name in ("ngram_novel_nocache", "ngram_novel_cache"):
        out.append(f"{name}: read+deferred ms per chunk {ng[name]['read_plus_deferred_ms']}")
    out.append(f"nvme dd 2 GiB O_DIRECT: {ng['nvme_dd_2GiB_odirect']}")
    out.append("pageloc key rows:")
    for x in rows:
        out.append(f"  {x['mem']:5} K {x['K']:2} Wb {x['Wb']:>5} rp {x['rp']:>5} NB {x['NB']:4} lds {x['lds']:5}  {x['gbs']:6.1f} GB/s  {x['mhz']} MHz")
    open(os.path.join(L2, f"reboot_{label}.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    return r


def compare(a, b):
    A = json.load(open(os.path.join(L2, f"reboot_{a}.json")))
    B = json.load(open(os.path.join(L2, f"reboot_{b}.json")))
    out = [f"compare {a} -> {b}", f"cmdline {A['system']['cmdline']}  ->  {B['system']['cmdline']}",
           f"iommu {A['system']['iommu_types']} -> {B['system']['iommu_types']}"]
    out += [f"nvme {x} -> {y}" for x, y in zip(A["system"]["nvme"], B["system"]["nvme"])]

    def line(name, pa, pb):
        if not pa or not pb or pa.get("median_ms") is None or pb.get("median_ms") is None:
            return f"{name}: missing"
        sa = max(pa["rounds_ms"]) - min(pa["rounds_ms"])
        sb = max(pb["rounds_ms"]) - min(pb["rounds_ms"])
        d = pb["median_ms"] - pa["median_ms"]
        sig = "significant" if abs(d) > max(sa, sb) else "within noise"
        return (f"{name}: {pa['median_ms']:.2f} -> {pb['median_ms']:.2f} ms ({d:+.2f}, {100 * d / pa['median_ms']:+.1f}%), "
                f"spreads {sa:.2f} / {sb:.2f}: {sig}")
    for t in ("2048", "8192"):
        out.append(line(f"prefill {t}", A["prefill"].get(t), B["prefill"].get(t)))
    for c in ("short", "7k"):
        out.append(line(f"decode {c}", A["decode"].get(c), B["decode"].get(c)))
        ka = {k["kernel"]: k["us_call"] for k in A["decode"][c]["top_kernels"]}
        for k in B["decode"][c]["top_kernels"]:
            if k["kernel"] in ka:
                out.append(f"    {k['kernel']:50} {ka[k['kernel']]:8.1f} -> {k['us_call']:8.1f} us/call")
    out.append(f"model ready s: {A['model_ready_s']['median']} -> {B['model_ready_s']['median']}")
    for n in ("ngram_novel_nocache", "ngram_novel_cache"):
        out.append(f"{n} read+deferred ms: {A['ngram'][n]['read_plus_deferred_ms']} -> {B['ngram'][n]['read_plus_deferred_ms']}")
    out.append(f"nvme dd: {A['ngram']['nvme_dd_2GiB_odirect']} -> {B['ngram']['nvme_dd_2GiB_odirect']}")
    pa = {(x["mem"], x["K"], x["Wb"], x["rp"], x["NB"], x["lds"]): x["gbs"] for x in A["probe_pageloc_key"]}
    out.append("pageloc key (GB/s):")
    for x in B["probe_pageloc_key"]:
        k = (x["mem"], x["K"], x["Wb"], x["rp"], x["NB"], x["lds"])
        if k in pa:
            out.append(f"  {x['mem']:5} K {x['K']:2} Wb {x['Wb']:>5} rp {x['rp']:>5} NB {x['NB']:4} lds {x['lds']:5}  "
                       f"{pa[k]:6.1f} -> {x['gbs']:6.1f} ({100 * (x['gbs'] / pa[k] - 1):+.1f}%)")
    ra = {(x["seg"], x["order"]): x["read"] for x in A["probe_dram"]}
    out.append("dram read GB/s:")
    for x in B["probe_dram"]:
        if (x["seg"], x["order"]) in ra:
            out.append(f"  seg {x['seg']:6} {x['order']:9} {ra[(x['seg'], x['order'])]:6.1f} -> {x['read']:6.1f}")
    txt = "\n".join(out) + "\n"
    open(os.path.join(L2, f"reboot_compare_{a}_{b}.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        summarize(sys.argv[1])
