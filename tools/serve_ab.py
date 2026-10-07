#!/usr/bin/env python3
"""Replay a request set against halo_serve and compare two configurations (e.g. --mtp-model on vs off).

  serve_ab.py run --out R.jsonl --requests Q.jsonl [--port N] -- <halo_serve command line>
      Starts the server, waits for /health, sends each request of Q.jsonl in order (streamed, so the time to the
      first output text and the decode time are measured client side), stops the server (SIGINT) and writes one
      result line per request: name, text (reasoning + content + tool calls as the client saw them),
      finish_reason, usage, ttft_s, decode_s (first to last output piece), e2e_s, and the response id.
  serve_ab.py compare A.jsonl B.jsonl [--only greedy]
      Requests whose "check" is "identical" must give the same text and finish reason in A and B.
  serve_ab.py table A.jsonl B.jsonl
      Per request: TTFT and decode tokens/s (completion_tokens - 1 over decode_s) of A and B, and B / A.

Q.jsonl lines: {"name": ..., "path": "/v1/chat/completions" | "/v1/responses", "body": {...}, "check": "identical" |
"none", "chain": NAME (optional: body.previous_response_id = the id of the earlier request NAME)}.
One server at a time (the model needs most of the memory); the caller holds the GPU lock.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import time


def wait_health(port: int, proc: subprocess.Popen, timeout_s: float = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise SystemExit(f"halo_serve exited with {proc.returncode} before it was ready")
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit("halo_serve did not become ready")


def sse_events(resp):
    """Yields (arrival time, parsed JSON) per `data:` line of a text/event-stream response."""
    buf = b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            now = time.perf_counter()
            for line in frame.split(b"\n"):
                if line.startswith(b"data: "):
                    data = line[6:]
                    if data.strip() == b"[DONE]":
                        continue
                    yield now, json.loads(data)


def send(port: int, req: dict, ids: dict) -> dict:
    body = dict(req["body"])
    body["stream"] = True
    if req["path"].endswith("/completions"):
        body["stream_options"] = {"include_usage": True}
    if "chain" in req:
        body["previous_response_id"] = ids[req["chain"]]
    payload = json.dumps(body).encode()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    t0 = time.perf_counter()
    c.request("POST", req["path"], payload, {"Content-Type": "application/json"})
    resp = c.getresponse()
    out = {"name": req["name"], "check": req.get("check", "none"), "status": resp.status, "text": "", "id": "",
           "finish_reason": None, "usage": None, "ttft_s": None, "decode_s": None}
    if resp.status != 200:
        out["error"] = resp.read().decode(errors="replace")[:500]
        return out
    first = last = None
    tool_args: dict = {}
    for now, e in sse_events(resp):
        piece = ""
        if "error" in e and "choices" not in e:  # a generation that failed after the stream started
            out["error"] = json.dumps(e["error"])[:500]
            continue
        if "choices" in e:  # chat.completion.chunk, text_completion
            out["id"] = e.get("id", out["id"])
            for ch in e["choices"]:
                d = ch.get("delta", {})
                piece += (ch.get("text") or "") + (d.get("reasoning_content") or "") + (d.get("content") or "")
                for tc in d.get("tool_calls") or []:
                    f = tc.get("function", {})
                    piece += (f.get("name") or "") + (f.get("arguments") or "")
                if ch.get("finish_reason"):
                    out["finish_reason"] = ch["finish_reason"]
            if e.get("usage"):
                out["usage"] = e["usage"]
        else:  # Responses events
            t = e.get("type", "")
            if t in ("response.output_text.delta", "response.reasoning_text.delta",
                     "response.function_call_arguments.delta"):
                piece += e.get("delta", "")
            if t == "response.created":
                out["id"] = e["response"]["id"]
            if t in ("response.completed", "response.incomplete"):
                r = e["response"]
                out["finish_reason"] = r.get("status")
                u = r.get("usage") or {}
                out["usage"] = {"prompt_tokens": u.get("input_tokens"), "completion_tokens": u.get("output_tokens"),
                                "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens")}
        if piece:
            out["text"] += piece
            first = first if first is not None else now
            last = now
    end = time.perf_counter()
    out["e2e_s"] = end - t0
    if first is not None:
        out["ttft_s"] = first - t0
        out["decode_s"] = last - first
    ids[req["name"]] = out["id"]
    return out


def cmd_run(a) -> int:
    reqs = [json.loads(l) for l in open(a.requests) if l.strip()]
    cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
    cmd = cmd + ["--port", str(a.port)]
    log = open(a.out + ".server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_health(a.port, proc)
        ids: dict = {}
        with open(a.out, "w") as f:
            for r in reqs:
                res = send(a.port, r, ids)
                f.write(json.dumps(res) + "\n")
                f.flush()
                u = res.get("usage") or {}
                print(f"{res['name']}: {res['status']} {res['finish_reason']} {u.get('completion_tokens')} tokens, "
                      f"ttft {res['ttft_s']}, decode {res['decode_s']}" + (f", ERROR {res['error']}" if res.get("error") else ""),
                      flush=True)
        c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)  # the counters after the set
        c.request("GET", "/metrics")
        with open(a.out + ".metrics", "w") as f:
            f.write(c.getresponse().read().decode(errors="replace"))
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return 0


def load(p):
    return {r["name"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


def cmd_compare(a) -> int:
    A, B = load(a.a), load(a.b)
    bad = 0
    n = 0
    for name, ra in A.items():
        if ra["check"] != "identical":
            continue
        rb = B.get(name)
        n += 1
        same = (rb is not None and ra["text"] == rb["text"] and ra["finish_reason"] == rb["finish_reason"]
                and not ra.get("error") and not rb.get("error"))
        if not same:
            bad += 1
            pos = next((i for i, (x, y) in enumerate(zip(ra["text"], (rb or {}).get("text", ""))) if x != y), None)
            print(f"DIFFERS {name}: finish {ra['finish_reason']} vs {(rb or {}).get('finish_reason')}, first char diff {pos}")
        else:
            print(f"identical {name} ({ra['finish_reason']}, {len(ra['text'])} chars)")
    print(f"{n - bad} of {n} identical")
    return 1 if bad else 0


def tps(r):
    u = r.get("usage") or {}
    n = u.get("completion_tokens") or 0
    return (n - 1) / r["decode_s"] if r.get("decode_s") and n > 1 else None


def cmd_table(a) -> int:
    A, B = load(a.a), load(a.b)
    print(f"| request | tokens | TTFT A s | TTFT B s | decode A tok/s | decode B tok/s | B/A |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, ra in A.items():
        rb = B.get(name)
        if not rb:
            continue
        ta, tb = tps(ra), tps(rb)
        n = (rb.get("usage") or {}).get("completion_tokens")
        f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
        print(f"| {name} | {n} | {f(ra['ttft_s'], 3)} | {f(rb['ttft_s'], 3)} | {f(ta)} | {f(tb)} | "
              f"{f(tb / ta if ta and tb else None, 3)} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--requests", required=True)
    r.add_argument("--port", type=int, default=18089)
    r.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("compare", "table"):
        c = sub.add_parser(name)
        c.add_argument("a")
        c.add_argument("b")
    a = ap.parse_args()
    return {"run": cmd_run, "compare": cmd_compare, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
