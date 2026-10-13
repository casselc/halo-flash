#!/usr/bin/env python3
"""A fault-injecting proxy in front of a real halo_serve, speaking the stand-in's admin interface.

cordis's outbox/test/stub_ops.py stands in for halo with the operation API plus faults a test switches on. This proxy
gives a test the same faults against the real server: the operation verbs go to a halo_serve it owns, and
/admin/restart restarts that process for real (a new incarnation, everything held gone). A milestone written against
the stand-in can then run its fault schedule through the actual adapter against the real model.

    op_fault_proxy.py PORT_FILE COUNT_FILE [ignored stand-in modes...]

The halo_serve it runs, from the environment:
  OP_PROXY_SERVE   the halo_serve binary (required)
  OP_PROXY_MODEL   --model (required)
  OP_PROXY_PORT    its port (default 18126)
  OP_PROXY_ARGS    more halo_serve arguments (e.g. "--max-context 32768 --op-capture-s 2")
  OP_PROXY_LOG     where its output goes (default: PORT_FILE.serve.log)
If a halo_serve already answers on OP_PROXY_PORT, the proxy uses it, and restarts it as its own the first time a
test asks. The port file is written once the upstream is ready.

Faults (POST /admin/fault, JSON; each applies once, to the next matching request), as the stand-in's:
  {"lose_submit": true}          the next submit is not forwarded; its connection is dropped
  {"lose_submit_answer": true}   the next submit is forwarded and its answer dropped
  {"delay_submit": S}            the next submit is held S seconds before it is forwarded (a late copy)
  {"dup_submit": true}           the next submit is forwarded twice; the first answer is returned
  {"lose_ack_answer": true}      the next ack is forwarded and its answer dropped
  {"lose_cancel": true}          the next cancel is not forwarded
and one of the proxy's own:
  {"expire": true}               the next attempt seen done is reported running until halo says it expired (start
                                 halo with a short --op-capture-s): the observer waits past the capture window
  POST /admin/restart            restarts halo_serve (answers once it is serving again)
  GET  /admin/state              {incarnation, records, unacked_outcomes, generated, attempts: {}} from halo's contract
"""

import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

port_file, count_file = sys.argv[1], sys.argv[2]
SERVE = os.environ["OP_PROXY_SERVE"]
MODEL = os.environ["OP_PROXY_MODEL"]
UP_PORT = int(os.environ.get("OP_PROXY_PORT", "18126"))
UP_ARGS = os.environ.get("OP_PROXY_ARGS", "").split()
UP_LOG = os.environ.get("OP_PROXY_LOG", port_file + ".serve.log")

lock = threading.Lock()
ready = threading.Event()
faults = {}
expiring = set()  # (op, attempt) reported running until halo says expired
proc = None
submits = 0
generated = 0
restarts = 0
fired = {}  # fault -> times it applied


def fire(name):
    fired[name] = fired.get(name, 0) + 1


def upstream(method, path, body=b"", headers=None, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", UP_PORT, timeout=timeout)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    out = (r.status, [(k, v) for k, v in r.getheaders() if k.lower() not in ("connection", "transfer-encoding", "content-length")], data)
    c.close()
    return out


def serving():
    try:
        return upstream("GET", "/v1/halo/contract", timeout=2)[0] == 200
    except OSError:
        return False


def start():
    global proc
    log = open(UP_LOG, "a")
    proc = subprocess.Popen([SERVE, "--model", MODEL, "--port", str(UP_PORT)] + UP_ARGS, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(1200):
        if serving():
            return
        if proc.poll() is not None:
            raise SystemExit(f"halo_serve exited ({proc.returncode}); see {UP_LOG}")
        time.sleep(0.5)
    raise SystemExit("halo_serve did not start")


def stop():
    global proc
    if proc is not None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc = None
    else:  # one we did not start: the operator's, on our port
        out = subprocess.run(["fuser", "-k", "-TERM", f"{UP_PORT}/tcp"], capture_output=True)
        for _ in range(240):
            if not serving():
                break
            time.sleep(0.5)


def restart():
    global restarts
    ready.clear()
    try:
        stop()
        start()
        restarts += 1
    finally:
        ready.set()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, status, headers, data):
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def json(self, status, obj):
        self.reply(status, [("Content-Type", "application/json")], json.dumps(obj).encode())

    def drop(self):
        self.close_connection = True
        try:
            self.connection.shutdown(2)
        except OSError:
            pass

    def forward(self, body=b""):
        h = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection", "content-length")}
        return upstream(self.command, self.path, body, h)

    def do_GET(self):
        if self.path == "/admin/state":
            ready.wait()
            c = json.loads(upstream("GET", "/v1/halo/contract")[2])
            r = c["operations"]["records"]
            return self.json(200, {"incarnation": c["incarnation"], "records": r["in_use"], "unacked_outcomes": r["unacked_outcomes"],
                                   "generated": generated, "restarts": restarts, "fired": fired, "attempts": {}})
        ready.wait()
        status, headers, data = self.forward()
        parts = self.path.split("?")[0].split("/")
        if status == 200 and len(parts) == 5 and parts[1:3] == ["v1", "operations"]:  # a status
            key = (parts[3], parts[4])
            view = json.loads(data)
            with lock:
                if view.get("status") == "done" and faults.pop("expire", None):
                    expiring.add(key)
                    fire("expire")
                if key in expiring:
                    if view.get("status") == "done":  # hold it running until halo releases its body
                        view = {k: view[k] for k in ("operation_id", "attempt", "digest")}
                        view.update(status="running", cancel_requested=False)
                        data = json.dumps(view).encode()
                    else:
                        expiring.discard(key)
                        fire("expired:" + view.get("status", "?"))
        self.reply(status, headers, data)

    def do_POST(self):
        global submits, generated
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        if self.path == "/admin/fault":
            with lock:
                faults.update(json.loads(body or b"{}"))
                return self.json(200, faults)
        if self.path == "/admin/restart":
            restart()
            fire("restart")
            return self.json(200, {"restarts": restarts})
        ready.wait()
        if self.path == "/v1/operations":
            with lock:
                submits += 1
                with open(count_file, "w") as f:
                    f.write(str(submits))
                lose = faults.pop("lose_submit", None)
                late = faults.pop("delay_submit", None)
                dup = faults.pop("dup_submit", None)
                lose_answer = faults.pop("lose_submit_answer", None)
                for name, on in (("lose_submit", lose), ("delay_submit", late), ("dup_submit", dup), ("lose_submit_answer", lose_answer)):
                    if on:
                        fire(name)
            if lose:
                return self.drop()
            if late:
                time.sleep(float(late))
            status, headers, data = self.forward(body)
            if dup:
                self.forward(body)
            if status == 202:
                with lock:
                    generated += 1
            if lose_answer:
                return self.drop()
            return self.reply(status, headers, data)
        what = self.path.rsplit("/", 1)[-1]
        with lock:
            lose_cancel = what == "cancel" and faults.pop("lose_cancel", None)
            lose_ack = what == "ack" and faults.pop("lose_ack_answer", None)
            if lose_cancel:
                fire("lose_cancel")
            if lose_ack:
                fire("lose_ack_answer")
        if lose_cancel:
            return self.drop()
        status, headers, data = self.forward(body)
        if lose_ack:
            return self.drop()
        self.reply(status, headers, data)


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    if not serving():
        start()
    ready.set()
    srv = Server(("127.0.0.1", 0), Handler)
    with open(port_file + ".tmp", "w") as f:
        f.write(str(srv.server_address[1]))
    os.replace(port_file + ".tmp", port_file)

    def bye(*_):
        stop()
        os._exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    srv.serve_forever()


if __name__ == "__main__":
    main()
