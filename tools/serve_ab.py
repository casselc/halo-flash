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
  serve_ab.py conc --out R.json --text-dir DIR [--sessions 1,2,4] [--rounds 3] [--context 8192] [--new 500]
                   [--output 300] [--cold PATH --cold-after 3 --decoder-output 1500] -- <halo_serve command line>
      G3 (docs/SCHEDULER.md 9): N sandboxes (X-Halo-Principal s1..sN) each build an agent conversation of about
      --context tokens (warm-up turn, not measured), then all N send their next turn at once (--new tokens of new
      input continuing with previous_response_id, --output tokens): per round, each turn's TTFT and decode tok/s,
      and the aggregate tok/s (all tokens over the time from the first send to the last token). With --cold: one
      sandbox decodes a long answer (--decoder-output tokens) and --cold-after s later another sends PATH's text as
      a cold prompt: its TTFT (the prefill beside the decoder) and the decoder's tok/s before, during and after it.

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


def send(port: int, req: dict, ids: dict, headers: dict | None = None, on_piece=None) -> dict:
    body = dict(req["body"])
    body["stream"] = True
    if req["path"].endswith("/completions"):
        body["stream_options"] = {"include_usage": True}
    if "chain" in req:
        body["previous_response_id"] = ids[req["chain"]]
    payload = json.dumps(body).encode()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    t0 = time.perf_counter()
    c.request("POST", req["path"], payload, {"Content-Type": "application/json", **(headers or {})})
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
            if on_piece:
                on_piece(now)
    end = time.perf_counter()
    out["e2e_s"] = end - t0
    out["t0"], out["t_first"], out["t_last"] = t0, first, last
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


def start_server(cmd: list, port: int, out: str):
    log = open(out + ".server.log", "w")
    proc = subprocess.Popen(cmd + ["--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
    with open(out + ".pid", "w") as f:  # the caller can stop it if this script is killed
        f.write(str(proc.pid))
    wait_health(port, proc)
    return proc


def stop_server(proc) -> None:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def chunk_texts(text_dir: str, n: int, chars: int) -> list:
    """n distinct texts of about `chars` characters from the files of text_dir (sorted, concatenated)."""
    blob = ""
    for name in sorted(os.listdir(text_dir)):
        path = os.path.join(text_dir, name)
        if os.path.isfile(path):
            blob += open(path, errors="replace").read() + "\n"
        if len(blob) >= n * chars + chars:
            break
    if len(blob) < n * chars:
        raise SystemExit(f"{text_dir}: {len(blob)} chars, need {n * chars}")
    return [blob[i * chars:(i + 1) * chars] for i in range(n)]


def turn(content: str, max_out: int, prev: str | None = None) -> dict:
    body = {"input": [{"role": "user", "content": content}], "max_output_tokens": max_out, "temperature": 0,
            "store": True}
    if prev:
        body["previous_response_id"] = prev
    return {"name": "t", "path": "/v1/responses", "body": body}


def concurrently(fns: list) -> list:
    import threading
    out = [None] * len(fns)
    start = threading.Barrier(len(fns))

    def run(i):
        start.wait()
        out[i] = fns[i]()

    ts = [threading.Thread(target=run, args=(i,)) for i in range(len(fns))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def cmd_conc(a) -> int:
    cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
    sessions = [int(x) for x in a.sessions.split(",") if x]
    nmax = max(sessions) if sessions else 1
    # ~4 characters a token (prose and docs)
    texts = chunk_texts(a.text_dir, nmax + a.rounds * nmax + 1, a.context * 4)
    news = [t[: a.new * 4] for t in texts[nmax:]]
    ask = ("\n\nRead the text above carefully. Then think step by step about what it says, and write a long, "
           "detailed analysis of it.")
    proc = start_server(cmd, a.port, a.out)
    results = {"sessions": {}, "cold": None}
    try:
        for n in sessions:
            rows = []
            ids = {}
            hdr = [{"X-Halo-Principal": f"s{n}-{i}"} for i in range(n)]
            # warm-up turns, one after another (not measured): each sandbox's conversation of ~context tokens
            for i in range(n):
                r = send(a.port, turn(texts[i] + ask, 16), {}, hdr[i])
                ids[i] = r["id"]
                print(f"warm-up s{n}-{i}: {r['status']} prompt {(r.get('usage') or {}).get('prompt_tokens')}", flush=True)
            for rnd in range(a.rounds):
                reqs = [turn(news[(rnd * n + i) % len(news)] + ask, a.output, ids[i]) for i in range(n)]
                res = concurrently([lambda i=i: send(a.port, reqs[i], {}, hdr[i]) for i in range(n)])
                t0 = min(r["t0"] for r in res)
                tend = max(r["t_last"] or r["t0"] for r in res)
                toks = sum(((r.get("usage") or {}).get("completion_tokens") or 0) for r in res)
                for i, r in enumerate(res):
                    ids[i] = r["id"] or ids[i]
                    u = r.get("usage") or {}
                    rows.append({"round": rnd, "session": i, "status": r["status"], "ttft_s": r["ttft_s"],
                                 "tok_s": tps(r), "tokens": u.get("completion_tokens"), "prompt": u.get("prompt_tokens"),
                                 "cached": u.get("cached_tokens"), "error": r.get("error")})
                agg = toks / (tend - t0) if tend > t0 else None
                rows.append({"round": rnd, "aggregate_tok_s": agg, "tokens": toks, "wall_s": tend - t0})
                print(f"N={n} round {rnd}: " + ", ".join(
                    f"s{i} ttft {r['ttft_s'] or 0:.2f}s {tps(r) or 0:.1f} tok/s ({(r.get('usage') or {}).get('completion_tokens')} tok, "
                    f"cached {(r.get('usage') or {}).get('cached_tokens')})" for i, r in enumerate(res)) +
                      f"; aggregate {agg or 0:.1f} tok/s", flush=True)
            results["sessions"][str(n)] = rows
        if a.cold:
            cold_text = open(a.cold, errors="replace").read()
            dec_id = send(a.port, turn(texts[0] + ask, 16), {}, {"X-Halo-Principal": "dec"})["id"]
            stamps = []
            dec_req = turn(news[0] + ask, a.decoder_output, dec_id)
            cold_req = turn(cold_text + "\n\nSummarize the text above in one sentence.", 16)

            def decoder():
                return send(a.port, dec_req, {}, {"X-Halo-Principal": "dec"}, on_piece=stamps.append)

            def cold():
                time.sleep(a.cold_after)
                return send(a.port, cold_req, {}, {"X-Halo-Principal": "cold"})

            dr, cr = concurrently([decoder, cold])
            cs, cf = cr["t0"], cr["t_first"] or cr["t0"]
            during = [t for t in stamps if cs <= t <= cf]
            before = [t for t in stamps if t < cs]
            after = [t for t in stamps if t > cf]
            rate = lambda ts: (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else None
            results["cold"] = {"cold_ttft_s": cr["ttft_s"], "cold_prompt": (cr.get("usage") or {}).get("prompt_tokens"),
                               "decoder_tok_s_before": rate(before), "decoder_tok_s_during": rate(during),
                               "decoder_pieces_during": len(during),
                               "decoder_tok_s_during_wall": (len(during) / (cf - cs)) if cf > cs else None,
                               "decoder_tok_s_after": rate(after), "decoder_tokens": (dr.get("usage") or {}).get("completion_tokens"),
                               "decoder_e2e_s": dr["e2e_s"], "cold_status": cr["status"], "cold_error": cr.get("error")}
            print("cold:", json.dumps(results["cold"]), flush=True)
        c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
        c.request("GET", "/metrics")
        results["metrics"] = c.getresponse().read().decode(errors="replace")
    finally:
        stop_server(proc)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--requests", required=True)
    r.add_argument("--port", type=int, default=18089)
    r.add_argument("command", nargs=argparse.REMAINDER)
    cc = sub.add_parser("conc")
    cc.add_argument("--out", required=True)
    cc.add_argument("--text-dir", required=True)
    cc.add_argument("--sessions", default="1,2,4")
    cc.add_argument("--rounds", type=int, default=3)
    cc.add_argument("--context", type=int, default=8192)
    cc.add_argument("--new", type=int, default=500)
    cc.add_argument("--output", type=int, default=300)
    cc.add_argument("--cold")
    cc.add_argument("--cold-after", type=float, default=3)
    cc.add_argument("--decoder-output", type=int, default=1500)
    cc.add_argument("--port", type=int, default=18089)
    cc.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("compare", "table"):
        c = sub.add_parser(name)
        c.add_argument("a")
        c.add_argument("b")
    a = ap.parse_args()
    return {"run": cmd_run, "compare": cmd_compare, "table": cmd_table, "conc": cmd_conc}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
