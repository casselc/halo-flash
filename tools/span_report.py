#!/usr/bin/env python3
"""Report span traces (src/kernels/span_trace.h; captured by `halo_gemm_bench trace`).

  span_report.py TRACE.bin [--bins N]

Per kernel tag: the kernel's extent on the 100 MHz realtime clock, how many
waves were active over time (ramp-up, steady state, tail), the busy share of
each WGP, and for every event-to-event transition the distribution over
waves of its duration (shader cycles, and realtime), with the effective clock
(cycles / realtime) per transition. Read with the expected per-phase times
written down first (Sites: estimate, observe, explain the difference).
"""
import argparse
import collections
import struct
import sys

TAGS = {1: "span_control", 2: "hc_combine_norm_rows", 3: "GemmQ8rF16", 4: "MoeDown", 5: "MoeGateUp", 6: "qsel_h"}
RT_HZ = 100e6
WRAP = 1 << 20  # SHADER_CYCLES width
ANCHOR = 255  # anchor records: realtime and cycles read together


def load(path):
    data = open(path, "rb").read()
    (n,) = struct.unpack_from("<I", data, 0)
    return [struct.unpack_from("<IIIBBBB", data, 4 + 16 * i) for i in range(n)]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else 0


def timelines(recs):
    """(tag, block, wave) -> [(event, seconds)], where each wave ran, and per tag the blocks' clocks (Hz).
    SHADER_CYCLES is effectively per SIMD in the engine, so every traced wave writes an anchor record (event 255:
    realtime + cycles read together) after its start stamp, and wave 0 another at its end. Each wave's readings
    are unwrapped from its own anchor (within half a wrap, ~180 us, then cumulatively) and converted at the
    block's clock (wave 0's anchors: cycles over realtime; 2.88 GHz without an end anchor)."""
    waves = collections.defaultdict(list)
    anchors = collections.defaultdict(list)  # (tag, block, wave) -> [(rt, cycles)] in record order
    for rt, cy, block, wave, event, tag, valid in recs:
        if not valid:
            continue
        if event == ANCHOR:
            anchors[(tag, block, wave)].append((rt, cy & (WRAP - 1)))
        else:
            waves[(tag, block, wave)].append((event, rt, cy & (WRAP - 1), cy >> 20))
    clock = {}
    for (tag, block, wave), an in anchors.items():
        if wave != 0:
            continue
        hz = 2.88e9
        if len(an) >= 2:
            (r0, c0), (r1, c1) = an[0], an[-1]
            dt = (r1 - r0) / RT_HZ
            if dt > 0:
                d = (c1 - c0) % WRAP
                d += round((dt * 2.88e9 - d) / WRAP) * WRAP
                hz = d / dt if d > 0 else hz
        clock[(tag, block)] = hz
    out, where = {}, {}
    for (tag, block, wave), ev in waves.items():
        an = anchors.get((tag, block, wave))
        if not an:
            continue
        r0, c0 = an[0]
        hz = clock.get((tag, block), 2.88e9)
        d = (ev[0][2] - c0) % WRAP
        if d >= WRAP // 2:
            d -= WRAP
        t, prev, times = d, ev[0][2], []
        for k, (e, _, cy, _) in enumerate(ev):
            if k:
                t += (cy - prev) % WRAP
                prev = cy
            times.append((e, r0 / RT_HZ + t / hz))
        out[(tag, block, wave)] = times
        hw = ev[0][3]
        where[(tag, block, wave)] = {"simd": hw & 3, "wgp": (hw >> 2) & 15, "sa": (hw >> 6) & 1, "se": (hw >> 7) & 7}
    per_tag = collections.defaultdict(list)
    for (tag, block), hz in clock.items():
        per_tag[tag].append(hz)
    return out, where, {t: sorted(v)[len(v) // 2] for t, v in per_tag.items()}, per_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--control", action="store_true",
                    help="check: realtime vs cycle-derived durations of the control kernel's 4 -> 5")
    a = ap.parse_args()
    recs = load(a.trace)
    tl, where, clock, clocks = timelines(recs)
    if a.control:
        rt = {}
        for r, cy, block, wave, event, tag, valid in recs:
            if valid and tag == 1 and event in (4, 5):
                rt[(block, wave, event)] = r
        d_rt = [(rt[(b, w, 5)] - rt[(b, w, 4)]) / RT_HZ for (b, w, e) in rt if e == 4 and (b, w, 5) in rt]
        d_cy = []
        for (tag, block, wave), ev in tl.items():
            t = dict(ev)
            if tag == 1 and 4 in t and 5 in t:
                d_cy.append(t[5] - t[4])
        print(f"control 4 -> 5: realtime p50 {pct(d_rt, .5) * 1e6:.3f} us, cycle-derived p50 {pct(d_cy, .5) * 1e6:.3f} us "
              f"({(pct(d_cy, .5) / pct(d_rt, .5) - 1) * 100:+.2f}%), {len(d_rt)} waves")
        # Stall check: wave 0 parked at s_barrier across 6 -> 7 (the others run a loop).
        rt67 = {}
        for r, cy, block, wave, event, tag, valid in recs:
            if valid and tag == 1 and event in (6, 7):
                rt67[(block, wave, event)] = r
        for label, sel in (("wave 0 (parked at the barrier)", lambda w: w == 0), ("waves 1-7 (running)", lambda w: w > 0)):
            dr = [(rt67[(b, w, 7)] - rt67[(b, w, 6)]) / RT_HZ for (b, w, e) in rt67 if e == 6 and sel(w) and (b, w, 7) in rt67]
            dc = [dict(ev)[7] - dict(ev)[6] for (tg, b, w), ev in tl.items()
                  if tg == 1 and sel(w) and 6 in dict(ev) and 7 in dict(ev)]
            if dr and dc:
                print(f"control 6 -> 7, {label}: realtime p50 {pct(dr, .5) * 1e6:.3f} us, cycle-derived "
                      f"{pct(dc, .5) * 1e6:.3f} us ({(pct(dc, .5) / pct(dr, .5) - 1) * 100:+.2f}%)")
    by_tag = collections.defaultdict(dict)
    for (tag, block, wave), ev in tl.items():
        by_tag[tag][(block, wave)] = ev
    for tag, ws in sorted(by_tag.items()):
        t0 = min(ev[0][1] for ev in ws.values())
        t1 = max(ev[-1][1] for ev in ws.values())
        print(f"== {TAGS.get(tag, tag)}: {len(ws)} waves, {(t1 - t0) * 1e6:.1f} us from the first wave's start to "
              f"the last's end")
        cl = sorted(clocks[tag])
        print(f"  block clocks (start/end anchors) GHz: p10 {pct(cl, .1) / 1e9:.3f} p50 {pct(cl, .5) / 1e9:.3f} "
              f"p90 {pct(cl, .9) / 1e9:.3f} ({len(cl)} blocks)")
        nb = a.bins
        width = (t1 - t0) / nb or 1e-9
        active = [0.0] * nb
        for ev in ws.values():
            s_, e_ = ev[0][1] - t0, ev[-1][1] - t0
            for b in range(nb):
                lo, hi = b * width, (b + 1) * width
                active[b] += max(0.0, min(e_, hi) - max(s_, lo)) / width
        peak = max(active) or 1
        print(f"  active waves over time ({nb} slices of {width * 1e6:.1f} us):")
        for b in range(nb):
            print(f"    {b * width * 1e6:8.1f} us {active[b]:7.0f} {'#' * int(50 * active[b] / peak)}")
        # Busy share per WGP (union of its waves' lifetimes over the kernel's extent).
        per = collections.defaultdict(list)
        for (block, wave), ev in ws.items():
            w = where[(tag, block, wave)]
            per[(w["se"], w["sa"], w["wgp"])].append((ev[0][1], ev[-1][1]))
        busy = []
        for k, iv in per.items():
            iv.sort()
            tot, cs, ce = 0.0, None, None
            for s_, e_ in iv:
                if cs is None or s_ > ce:
                    tot += (ce - cs) if cs is not None else 0
                    cs, ce = s_, e_
                else:
                    ce = max(ce, e_)
            tot += ce - cs
            busy.append((tot / (t1 - t0), k, len(iv)))
        busy.sort()
        print(f"  WGPs used {len(busy)}; busy share min {busy[0][0] * 100:.0f}% median "
              f"{busy[len(busy) // 2][0] * 100:.0f}% max {busy[-1][0] * 100:.0f}%; waves per WGP "
              f"{min(b[2] for b in busy)}..{max(b[2] for b in busy)}")
        # Coverage and placement checks: sampled waves per SE/WGP, waves placed before their block's anchor.
        cover = collections.Counter((where[(tag, b, w)]["se"], where[(tag, b, w)]["sa"], where[(tag, b, w)]["wgp"])
                                    for (b, w) in ws)
        print(f"  coverage: {len(cover)} WGPs (SE, SA, WGP); waves per slot {min(cover.values())}..{max(cover.values())}")
        early = sum(1 for (b, w), ev in ws.items() if w > 0 and (b, 0) in ws and ev[0][1] < ws[(b, 0)][0][1] - 2e-6)
        print(f"  placement: {early} waves start > 2 us before their block's wave 0 (expect 0)")
        trans = collections.defaultdict(list)
        for ev in ws.values():
            for (e0, s0), (e1, s1) in zip(ev, ev[1:]):
                trans[(e0, e1)].append(s1 - s0)
        nw = len(ws)
        print(f"  {'transition':>10} {'count':>8} {'us p10':>8} {'p50':>8} {'p90':>8} {'max':>8} {'us/wave':>8}")
        for (e0, e1), d in sorted(trans.items(), key=lambda kv: -sum(kv[1]))[:16]:
            print(f"  {e0:>4} -> {e1:<3} {len(d):8d} {pct(d, .1) * 1e6:8.2f} {pct(d, .5) * 1e6:8.2f} "
                  f"{pct(d, .9) * 1e6:8.2f} {max(d) * 1e6:8.2f} {sum(d) * 1e6 / nw:8.2f}")
        life = [(ev[-1][1] - ev[0][1]) * 1e6 for ev in ws.values()]
        print(f"  wave lifetime us: p10 {pct(life, .1):.2f} p50 {pct(life, .5):.2f} p90 {pct(life, .9):.2f} "
              f"max {max(life):.2f}")


if __name__ == "__main__":
    sys.exit(main())
