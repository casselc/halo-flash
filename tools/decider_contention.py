#!/usr/bin/env python3
"""G3 D2 (docs/SCHEDULER.md M4): how the strands-decider-2B (/v1/systemone, its own mutex, concurrent on the GPU)
and Flash-Next slow each other.

  decider_contention.py --out R.json --text-dir DIR [--rate 2] [--calls 20] [--port N] -- <halo_serve --decider-dir ...>

Phases (one server): deciders alone (1 question, 3 questions); a warm Flash-Next decode alone, then with deciders at
--rate per second; a cold prefill (--cold-tokens, ~4 characters a token) alone, then with deciders. Reports each
phase's decider latency (median, p90, max: client-side ms) and Flash-Next's decode tok/s or prefill time.
The caller holds the GPU lock.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_ab  # noqa: E402

Q1 = {"state": "Help! My payouts have been failing for 3 days and the support chat keeps closing on me. I run a small "
               "shop and this is the third time this month. What is going on?",
      "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}
Q3 = {"state": Q1["state"],
      "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                    "topic": {"type": "choice", "instructions": "What is it about?",
                              "criteria": {"payments": "Payments or payouts", "access": "Account access",
                                           "shipping": "Shipping", "other": "Something else"}},
                    "anger": {"type": "score", "instructions": "How angry is the writer?",
                              "criteria": ["calm", "annoyed", "angry", "furious"]}}}


def decide(port: int, body: dict) -> float:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    t0 = time.perf_counter()
    c.request("POST", "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = c.getresponse()
    body_out = r.read()
    if r.status != 200:
        raise SystemExit(f"systemone: {r.status} {body_out[:400]!r}")
    return (time.perf_counter() - t0) * 1000


def stats(ms: list) -> dict:
    if not ms:
        return {}
    s = sorted(ms)
    return {"n": len(s), "median_ms": statistics.median(s), "p90_ms": s[min(len(s) - 1, int(0.9 * len(s)))],
            "max_ms": s[-1]}


def deciders_while(port: int, rate: float, stop: threading.Event, out: list, body: dict) -> threading.Thread:
    def run():
        nxt = time.perf_counter()
        while not stop.is_set():
            out.append(decide(port, body))
            nxt += 1.0 / rate
            time.sleep(max(0.0, nxt - time.perf_counter()))
    t = threading.Thread(target=run)
    t.start()
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--rate", type=float, default=2)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--decode-tokens", type=int, default=400)
    ap.add_argument("--cold-tokens", type=int, default=16384)
    ap.add_argument("--port", type=int, default=18089)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
    texts = serve_ab.chunk_texts(a.text_dir, 4, 8192 * 4)
    # the cold texts lie past the decode phases' texts (chunk_texts slices one concatenation: overlapping texts
    # would make the "cold" prefill partly a cache hit)
    colds = serve_ab.chunk_texts(a.text_dir, 6, a.cold_tokens * 4)[4:6]
    ask = "\n\nThink step by step about the text above, then write a long, detailed analysis of it."
    res: dict = {}
    proc = serve_ab.start_server(cmd, a.port, a.out)
    try:
        for _ in range(3):
            decide(a.port, Q1)  # warm-up
        res["decider_alone_1q"] = stats([decide(a.port, Q1) for _ in range(a.calls)])
        res["decider_alone_3q"] = stats([decide(a.port, Q3) for _ in range(a.calls)])
        print("alone:", res["decider_alone_1q"], res["decider_alone_3q"], flush=True)
        for phase, with_dec in (("decode_alone", False), ("decode_with_deciders", True)):
            i = 0 if not with_dec else 1
            first = serve_ab.send(a.port, serve_ab.turn(texts[i] + ask, 16), {})
            req = serve_ab.turn(texts[i + 2][:2000] + ask, a.decode_tokens, first["id"])
            lat: list = []
            stop = threading.Event()
            t = deciders_while(a.port, a.rate, stop, lat, Q1) if with_dec else None
            r = serve_ab.send(a.port, req, {})
            stop.set()
            if t:
                t.join()
            res[phase] = {"tok_s": serve_ab.tps(r), "tokens": (r.get("usage") or {}).get("completion_tokens"),
                          "decider": stats(lat)}
            print(phase, res[phase], flush=True)
        for phase, with_dec in (("prefill_alone", False), ("prefill_with_deciders", True)):
            lat = []
            stop = threading.Event()
            t = deciders_while(a.port, a.rate, stop, lat, Q1) if with_dec else None
            body = colds[1 if with_dec else 0]  # a different text each: no prefix reuse
            r = serve_ab.send(a.port, serve_ab.turn(body + "\n\nSummarize the text above in one sentence.", 8), {})
            stop.set()
            if t:
                t.join()
            res[phase] = {"ttft_s": r["ttft_s"], "prompt": (r.get("usage") or {}).get("prompt_tokens"),
                          "cached": (r.get("usage") or {}).get("cached_tokens"),
                          "decider": stats(lat)}
            print(phase, res[phase], flush=True)
        c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
        c.request("GET", "/metrics")
        res["metrics"] = c.getresponse().read().decode(errors="replace")
    finally:
        serve_ab.stop_server(proc)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
