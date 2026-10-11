#!/usr/bin/env python3
"""G3 M5: turn `quint run --mbt --out-itf` traces of the scheduler model's design instance into the server-side
events tests/server/scheduler_replay_test.cpp replays against the Scheduler class (docs/SCHEDULER.md section 9).

  itf_events.py OUT_DIR TRACE.itf.json...

One JSON object a line, in trace order; model request ids ("id"), the model's state after the step where the replay
compares it:
  {"e":"arrive", "t", "id", "dec", "p", "root", "conv", "turn", "prompt", "max", "conn", "out", "target",
   "prio", "held"}       an admission; out: queued|replayed|adopted|duplicate|limit|memory; conn: what the server
                         knows (not known gone); held: the retained prefix of each conversation afterwards
  {"e":"depart", "t", "id"}   the server learns that the request's client left (FIN, or a vanished peer after ACK_TO)
  {"e":"fence", "t", "ids", "parks"}   a boundary's fences (consecutive model fence steps); parks: fenced id ->
                         parked prefix length (of every fence up to the next pick)
  {"e":"pick", "t", "kind", "ids", "binds", "vroot", "vprin"}
                         the engine's next unit; binds: id -> reused prompt (first unit); vroot / vprin: virtual
                         times after the charge
  {"e":"end", "t", "elapsed", "prog", "held"}   the unit ended; prog: per id {id, pre, dec, pdone, fin, comp}
"""
from __future__ import annotations

import json
import os
import sys

ACK_TO = 2  # scheduler.qnt
COST = {"Slice": 2, "Batch": 1, "Decide": 1}  # SLICE, DECODE, DEC_COST
PREFIX = "design::scheduler::"


def val(v):
    if isinstance(v, dict):
        if "#bigint" in v:
            return int(v["#bigint"])
        if "#map" in v:
            return {_key(val(k)): val(x) for k, x in v["#map"]}
        if "#set" in v:
            return [val(x) for x in v["#set"]]
        if "#tup" in v:
            return tuple(val(x) for x in v["#tup"])
        if "tag" in v:
            return v["tag"]
        return {k: val(x) for k, x in v.items()}
    if isinstance(v, list):
        return [val(x) for x in v]
    return v


def _key(k):
    return k if not isinstance(k, list) else tuple(k)


def state(s):
    return {k[len(PREFIX):]: val(v) for k, v in s.items() if k.startswith(PREFIX)}


def active(r):
    return r["ph"] in ("Queued", "Running")


def known_gone(r, time):
    return not r["conn"] and (r["fin"] or time >= r["leftAt"] + ACK_TO)


def held_of(st):
    return {str(c): h for c, h in st["srv"]["held"].items()}


def convert(path: str) -> list:
    doc = json.load(open(path))
    raw = doc["states"]
    out = []
    pending_fence: list = []
    pending_parks: dict = {}
    told_gone: set = set()

    def flush_fences(t):
        nonlocal pending_fence, pending_parks
        if pending_fence:
            out.append({"e": "fence", "t": t, "ids": pending_fence, "parks": pending_parks})
            pending_fence, pending_parks = [], {}

    for i in range(1, len(raw)):
        pre, post = state(raw[i - 1]), state(raw[i])
        act = raw[i].get("mbt::actionTaken")
        t = post["time"]
        # consecutive fence steps are one boundary (Scheduler::Boundary)
        if act != "fence":
            flush_fences(pre["time"])
        if act == "arrive":
            ids = [k for k in post["reqs"] if pre["reqs"][k]["ph"] == "Waiting" and post["reqs"][k]["ph"] != "Waiting"]
            assert len(ids) == 1, ids
            mid = ids[0]
            r0, r1 = pre["reqs"][mid], post["reqs"][mid]
            target = None
            if r1["ph"] == "Queued":
                o = "queued"
            elif r1["ph"] == "Rejected":
                mine = [x for x in pre["reqs"].values()
                        if active(x) and x["p"] == r0["p"] and not x["dec"] and not known_gone(x, pre["time"])]
                o = "limit" if mine else "memory"
            elif r1["ph"] == "Finished":
                o = "replayed"
                stored = [x for x in pre["reqs"].values() if not x["dec"] and x["id"] != mid and
                          x["conv"] == r0["conv"] and x["turn"] == r0["turn"] and x["ph"] == "Finished" and x["served"]]
                target = min(x["id"] for x in stored)
            elif r1["ph"] == "Merged":
                adopted = [k for k in post["reqs"] if k != mid and pre["reqs"][k] != post["reqs"][k]]
                if adopted:
                    o, target = "adopted", adopted[0]
                else:
                    o = "duplicate"
                    inflight = [x for x in pre["reqs"].values() if not x["dec"] and x["id"] != mid and
                                x["conv"] == r0["conv"] and x["turn"] == r0["turn"] and active(x)]
                    target = min(x["id"] for x in inflight)
            else:
                raise SystemExit(f"{path}: arrive to {r1['ph']}")
            ev = {"e": "arrive", "t": t, "id": mid, "dec": r0["dec"], "p": r0["p"], "root": r0["root"],
                  "conv": r0["conv"], "turn": r0["turn"], "prompt": r0["prompt"], "max": r0["maxTok"],
                  "conn": not known_gone(r0, pre["time"]), "out": o, "prio": r1["prio"], "held": held_of(post)}
            if target is not None:
                ev["target"] = target
            out.append(ev)
        elif act == "fence":
            ids = sorted(k for k in post["reqs"] if active(pre["reqs"][k]) and post["reqs"][k]["ph"] == "Cancelled")
            assert ids
            pending_fence.extend(ids)
            for mid in ids:
                c = pre["reqs"][mid]["conv"]
                if c in post["srv"]["held"] and post["srv"]["held"][c] != pre["srv"]["held"][c]:
                    pending_parks[str(mid)] = post["srv"]["held"][c]
        elif act == "pick":
            eng = post["eng"]
            ids = sorted(eng["rs"])
            binds = {str(k): post["reqs"][k]["reused"] for k in ids
                     if not pre["reqs"][k]["started"] and not pre["reqs"][k]["dec"]}
            out.append({"e": "pick", "t": t, "kind": eng["kind"].lower(), "ids": ids, "binds": binds,
                        "vroot": post["srv"]["vtime"], "vprin": post["srv"]["pvt"]})
        elif act == "unitEnd":
            eng = pre["eng"]
            prog = []
            for k in sorted(eng["rs"]):
                r0, r1 = pre["reqs"][k], post["reqs"][k]
                fin = r1["ph"] == "Finished"
                prog.append({"id": k, "pre": r1["done"] - r0["done"], "dec": r1["decoded"] - r0["decoded"],
                             "pdone": r1["done"] >= r1["prompt"], "fin": fin, "comp": fin})
            out.append({"e": "end", "t": t, "elapsed": COST[eng["kind"]],
                        "kind": eng["kind"].lower(), "prog": prog, "held": held_of(post)})
        # what the server learns after this step: departures (FIN at once, a vanished peer after ACK_TO)
        for k, r in post["reqs"].items():
            if active(r) and known_gone(r, t) and k not in told_gone:
                told_gone.add(k)
                out.append({"e": "depart", "t": t, "id": k})
            if not known_gone(r, t):
                told_gone.discard(k)
    flush_fences(state(raw[-1])["time"])
    # the code fences every request known gone at a boundary, the model one a step: each fence event carries the
    # parks of every fence until the next pick
    group: list = []
    for e in out + [{"e": "pick"}]:
        if e["e"] == "fence":
            group.append(e)
        elif e["e"] == "pick" and group:
            parks: dict = {}
            for g in group:
                parks.update(g["parks"])
            for g in group:
                g["parks"] = parks
            group = []
    return out


def main() -> int:
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    for p in sys.argv[2:]:
        evs = convert(p)
        name = os.path.basename(p).replace(".itf.json", ".jsonl")
        with open(os.path.join(outdir, name), "w") as f:
            for e in evs:
                f.write(json.dumps(e, separators=(",", ":"), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
