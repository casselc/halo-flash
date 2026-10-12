#!/usr/bin/env python3
"""Operation-contract fault probe: a minimal outbox against a live halo_serve (docs/OPERATIONS.md; cordis RFC 0007).

The probe plays the platform. It keeps the contract's rules:
  - commit before ack;
  - a refusal closes only its copy;
  - settle never-accepted only through a terminal answer;
  - resend exactly as first sent;
  - never re-aim an attempt at a new incarnation.

It injects the faults the models cover:
  - lost submits, duplicated submits and late copies;
  - cancels before acceptance and while running;
  - lost acks, and a platform that forgets what it fetched;
  - results left past a short capture window (expiry);
  - server restarts (new incarnations).

It checks every answer against the contract and reports the outcomes it reached.

Checks:
  - an attempt this probe saw accepted (202) is never answered never_accepted, closed or cancelled_before_accept
    by the incarnation that accepted it;
  - `unknown` only comes from another incarnation;
  - every done/expired/released answer carries usage, and a committed result's usage never changes;
  - a duplicate submit never starts a second run (its answer is the attempt's status, not 202);
  - `expired` comes no earlier than the capture window after the attempt finished;
  - every operation ends committed, or with all its attempts settled.

Usage (under the GPU lock):
  op_fault_probe.py --serve build/nix-gpu/halo_serve --model small-q.gguf [--ops 12] [--seed 1] [--restarts 2]
"""

import argparse
import hashlib
import http.client
import json
import os
import random
import signal
import subprocess
import sys
import time


class Server:
    def __init__(self, args):
        self.args = args
        self.proc = None
        self.inc = None
        self.starts = 0

    def start(self):
        log = open(os.path.join(self.args.out, f"serve.{self.starts}.log"), "w")
        self.proc = subprocess.Popen(
            [self.args.serve, "--model", self.args.model, "--port", str(self.args.port), "--max-context", "4096",
             "--op-capture-s", str(self.args.capture_s)], stdout=log, stderr=subprocess.STDOUT)
        self.starts += 1
        for _ in range(300):
            try:
                _, _, body = self.call("GET", "/v1/halo/contract")
                self.inc = json.loads(body)["incarnation"]
                return
            except OSError:
                time.sleep(0.2)
        raise SystemExit("halo_serve did not start")

    def stop(self):
        if self.proc:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None

    def call(self, method, path, body=b"", headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.args.port, timeout=120)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        data = r.read()
        h = {k.lower(): v for k, v in r.getheaders()}
        c.close()
        return r.status, h, data

    def now(self):
        _, h, _ = self.call("GET", "/v1/halo/contract")
        return int(h["halo-time"])


class Probe:
    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)
        self.srv = Server(args)
        self.failures = []
        self.events = {}
        self.ops = []          # per operation: {"id", "attempts": [..], "committed": usage or None}
        self.durable = {}      # the platform's committed receipts: (op, attempt) -> usage
        self.fetched = {}      # volatile: lost in a platform crash

    # ---- bookkeeping
    def event(self, e):
        self.events[e] = self.events.get(e, 0) + 1

    def fail(self, why):
        self.failures.append(why)
        print("FAIL:", why, flush=True)

    # ---- verbs
    def submit(self, a):
        st, h, data = self.srv.call("POST", "/v1/operations", a["body"],
                                    {"Halo-Incarnation": a["inc"], "X-Halo-Principal": "probe"})
        if h.get("halo-incarnation") != self.srv.inc:
            self.fail("an answer without the current incarnation")
        return st, h, json.loads(data) if data else {}

    def verb(self, method, a, what=""):
        path = f"/v1/operations/{a['op']}/{a['n']}" + (f"/{what}" if what else "")
        st, h, data = self.srv.call(method, path, b"", {
            "Halo-Incarnation": a["inc"], "Halo-Not-After": str(a["not_after"]), "Halo-Digest": a["digest"],
            "X-Halo-Principal": "probe"})
        return st, h, data

    def new_attempt(self, op, window_ms=None):
        n = len(op["attempts"])
        not_after = self.srv.now() + (window_ms or self.args.window_ms)
        max_tokens = self.rng.choice([16, 64, 400])
        body = json.dumps({"operation_id": op["id"], "attempt": n, "not_after": not_after, "api": "completions",
                           "mode": "relaxed",
                           "request": {"prompt": f"Story {op['id']}: once upon a time", "max_tokens": max_tokens}}
                          ).encode()
        a = {"op": op["id"], "n": n, "inc": self.srv.inc, "not_after": not_after, "body": body,
             "digest": hashlib.sha256(body).hexdigest(), "accepted": False, "accepted_inc": None,
             "st": "pending", "usage": None, "done_ms": None, "reserve": max_tokens}
        op["attempts"].append(a)
        return a

    # ---- one operation, to the end
    def run_op(self, op):
        while op["committed"] is None and len(op["attempts"]) < self.args.max_attempts:
            abandon = self.rng.random() < 0.1  # lost submit, never resent: it must end closed at its not-after
            a = self.new_attempt(op, 1500 if abandon else None)
            self.drive(op, a, abandon)
            if op["committed"] is None and a["st"] == "pending":
                self.fail(f"{a['op']}/{a['n']} left pending")

    def drive(self, op, a, abandon=False):
        r = self.rng.random
        if r() < 0.15:
            # cancel before any copy is sent: the later copy must be a no-op
            st, _, data = self.verb("POST", a, "cancel")
            if st == 200 and json.loads(data)["status"] == "cancelled_before_accept":
                self.event("cancelled before accept")
                st, _, j = self.submit(a)  # the delayed copy
                if st == 202:
                    self.fail("a copy was accepted after cancelled_before_accept")
                self.settle(a, "closed", 0)
                return
        lose_first = abandon or r() < 0.2
        if lose_first:
            self.event("lost submit")
        else:
            self.send(a)
            if r() < 0.3:
                self.event("duplicate submit")
                self.send(a, duplicate=True)
        cancel_running = r() < 0.15
        forget_fetch = r() < 0.15
        drop_ack = r() < 0.25
        let_expire = r() < 0.15
        restart_now = self.srv.starts <= self.args.restarts and r() < 0.08
        if restart_now:
            self.restart()
        deadline = time.time() + 120
        while time.time() < deadline and a["st"] == "pending":
            st, h, data = self.verb("GET", a)
            if st != 200:
                self.fail(f"status {st} for {a['op']}/{a['n']}: {data[:200]!r}")
                return
            j = json.loads(data)
            s = j["status"]
            if s == "unknown":
                if a["inc"] == self.srv.inc:
                    self.fail("unknown from the incarnation the attempt was sent to")
                self.event("unknown")
                self.settle(a, "lost", a["reserve"])  # worst case
            elif s in ("never_accepted", "closed", "cancelled_before_accept"):
                if a["accepted"] and a["accepted_inc"] == self.srv.inc:
                    self.fail(f"{s} for an attempt accepted by this incarnation ({a['op']}/{a['n']})")
                if s == "never_accepted":
                    self.event("never accepted")
                    if abandon:
                        time.sleep(0.1)  # wait for closed
                    else:
                        self.event("resent after never_accepted")
                        self.send(a)  # exactly as first sent (the first copy was lost or refused)
                        time.sleep(0.05)
                else:
                    self.event(s)
                    self.settle(a, "closed", 0)
            elif s == "running":
                if cancel_running:
                    cancel_running = False
                    st, _, d = self.verb("POST", a, "cancel")
                    if st == 200:
                        self.event("cancel while running")
                time.sleep(0.05)
            elif s == "done":
                if "usage" not in j:
                    self.fail("done without usage")
                if a["done_ms"] is None:
                    a["done_ms"] = int(j["expires_at"]) - self.args.capture_s * 1000
                if let_expire:
                    let_expire = False
                    self.event("left to expire")
                    time.sleep(self.args.capture_s + 0.5)
                    continue
                st, _, body = self.verb("GET", a, "result")
                if st != 200:
                    continue  # expired meanwhile: the next status says so
                self.fetched[(a["op"], a["n"])] = j["usage"]
                if forget_fetch:
                    forget_fetch = False
                    self.fetched.clear()  # the platform crashed before committing
                    self.event("platform forgot a fetch")
                    continue
                self.commit(op, a, j["usage"])
                if drop_ack:
                    self.event("ack lost")
                else:
                    st, _, d = self.verb("POST", a, "ack")
                    if st != 200 or json.loads(d)["status"] != "released":
                        self.fail(f"ack answered {st} {d[:200]!r}")
                if j.get("finish") == "cancelled":
                    self.event("cancelled attempt committed")
            elif s == "expired":
                now = int(h["halo-time"])
                if a["done_ms"] is not None and now < a["done_ms"] + self.args.capture_s * 1000:
                    self.fail("expired inside its capture window")
                if "usage" not in j:
                    self.fail("expired without usage")
                self.event("expired settled at usage")
                self.settle(a, "lost", j["usage"]["completion_tokens"])
                self.verb("POST", a, "ack")
            elif s == "released":
                self.fail("released before the platform committed it")
                return
            else:
                self.fail("unexpected status " + s)
                return
        if a["st"] == "pending":
            self.fail(f"{a['op']}/{a['n']} did not settle in time")

    def send(self, a, duplicate=False):
        if a["inc"] != self.srv.inc:
            return  # never re-aimed at another incarnation; its status will say unknown
        st, h, j = self.submit(a)
        if st == 202:
            if a["accepted"]:
                self.fail(f"a second 202 for {a['op']}/{a['n']}: a duplicate started a run")
            a["accepted"], a["accepted_inc"] = True, self.srv.inc
        elif st == 200:
            self.event("duplicate answered with its status")
        elif st in (429, 503):
            self.event("refused copy")
        elif st == 410:
            self.event("late copy rejected")
        elif st == 409:
            self.event("stale or conflict")
        else:
            self.fail(f"submit answered {st}: {j}")

    def commit(self, op, a, usage):
        key = (a["op"], a["n"])
        if key in self.durable and self.durable[key] != usage:
            self.fail("a committed result's usage changed")
        self.durable[key] = usage
        a["st"], a["usage"] = "committed", usage
        op["committed"] = usage

    def settle(self, a, st, cost):
        a["st"], a["usage"] = st, cost

    def restart(self):
        self.event("restart")
        old = self.srv.inc
        self.srv.stop()
        self.srv.start()
        if self.srv.inc == old:
            self.fail("a restart kept its incarnation")

    def run(self):
        os.makedirs(self.args.out, exist_ok=True)
        self.srv.start()
        try:
            for i in range(self.args.ops):
                op = {"id": f"p{self.args.seed}-{i}", "attempts": [], "committed": None}
                self.ops.append(op)
                self.run_op(op)
                # a late copy of a settled attempt
                if self.rng.random() < 0.3 and op["attempts"]:
                    a = op["attempts"][0]
                    if a["inc"] == self.srv.inc:
                        st, _, j = self.submit(a)
                        if st == 202:
                            self.fail(f"a late copy of {a['op']}/{a['n']} was accepted again")
                        self.event("late copy of a settled attempt")
        finally:
            self.srv.stop()
        committed = sum(1 for o in self.ops if o["committed"] is not None)
        settled = sum(len(o["attempts"]) for o in self.ops)
        print(json.dumps({"ops": len(self.ops), "committed": committed, "attempts": settled,
                          "incarnations": self.srv.starts, "events": self.events,
                          "failures": self.failures}, indent=1))
        return 1 if self.failures else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=18124)
    p.add_argument("--ops", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--capture-s", type=int, default=3)
    p.add_argument("--window-ms", type=int, default=20000)
    p.add_argument("--max-attempts", type=int, default=6)
    p.add_argument("--out", default="op_fault_probe.out")
    sys.exit(Probe(p.parse_args()).run())


if __name__ == "__main__":
    main()
