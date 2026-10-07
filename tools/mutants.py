#!/usr/bin/env python3
"""Planted-mutant gate: every mutant in tests/mutants.toml must be killed.

Each mutant is an exact, once-only text edit to one source file (kernels,
engine code, anything Mull cannot reach) and the test binary + gtest filter
that must fail when it is applied. For every mutant the runner applies the
edit, rebuilds the binary, runs the filter, and restores the file (always,
also on SIGINT/SIGTERM).

Outcomes:
  killed    tests failed or crashed
  timeout   tests exceeded the mutant's timeout (counts as killed)
  SURVIVED  tests passed: the suite cannot see this defect
  INVALID   the edit did not apply exactly once, or did not compile

Before any mutant runs, each distinct (suite, binary, filter) is run
unmutated and must pass with no skipped tests; otherwise every mutant would
look killed (or, when skipped, survived) for the wrong reason.

  tools/mutants.py                 run all
  tools/mutants.py -k snapshot     run mutants whose name contains "snapshot"
  tools/mutants.py --list
Exit status is non-zero unless every selected mutant is killed.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import signal
import subprocess
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclasses.dataclass(frozen=True)
class Suite:
    name: str
    build_dir: str
    env: tuple[str, ...]  # variables that must be set, or the tests skip


@dataclasses.dataclass(frozen=True)
class Mutant:
    name: str
    file: str
    find: str
    replace: str
    suite: str
    binary: str
    filter: str
    why: str
    timeout: int


def load(path: str) -> tuple[dict[str, Suite], list[Mutant]]:
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    suites = {
        name: Suite(name, s["build_dir"], tuple(s.get("env", [])))
        for name, s in doc.get("suite", {}).items()
    }
    mutants, seen = [], set()
    for m in doc.get("mutant", []):
        if m["name"] in seen:
            raise ValueError(f"duplicate mutant name {m['name']}")
        seen.add(m["name"])
        if m["suite"] not in suites:
            raise ValueError(f"{m['name']}: unknown suite {m['suite']}")
        if m["find"] == m["replace"]:
            raise ValueError(f"{m['name']}: find == replace")
        mutants.append(Mutant(m["name"], m["file"], m["find"], m["replace"], m["suite"],
                              m["binary"], m["filter"], m["why"], int(m.get("timeout", 900))))
    return suites, mutants


def apply(text: str, m: Mutant) -> str:
    """The mutated text; raises if `find` does not occur exactly once."""
    n = text.count(m.find)
    if n != 1:
        raise ValueError(f"{m.name}: `find` occurs {n} times in {m.file}")
    return text.replace(m.find, m.replace, 1)


def skipped(output: str) -> bool:
    return "[  SKIPPED ]" in output


class Runner:
    def __init__(self, suites: dict[str, Suite], verbose: bool):
        self.suites = suites
        self.verbose = verbose
        self.pending_restore: tuple[str, bytes] | None = None

    def restore(self) -> None:
        if self.pending_restore:
            path, data = self.pending_restore
            with open(path, "wb") as f:
                f.write(data)
            self.pending_restore = None

    def build(self, suite: Suite, binary: str) -> tuple[bool, str]:
        p = subprocess.run(["cmake", "--build", os.path.join(ROOT, suite.build_dir), "--target", binary, "-j",
                            os.environ.get("HALO_BUILD_JOBS", str(os.cpu_count() or 8))], capture_output=True, text=True)
        return p.returncode == 0, p.stdout + p.stderr

    def test(self, suite: Suite, binary: str, filt: str, timeout: int) -> tuple[str, str]:
        exe = os.path.join(ROOT, suite.build_dir, binary)
        try:
            p = subprocess.run([exe, f"--gtest_filter={filt}", "--gtest_fail_fast=1"], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            return "timeout", out
        return ("pass" if p.returncode == 0 else "fail"), p.stdout + p.stderr

    def baseline(self, mutants: list[Mutant]) -> bool:
        ok = True
        for key in sorted({(m.suite, m.binary, m.filter) for m in mutants}):
            suite = self.suites[key[0]]
            missing = [v for v in suite.env if not os.environ.get(v)]
            if missing:
                print(f"baseline {key}: missing env {', '.join(missing)}")
                ok = False
                continue
            built, log = self.build(suite, key[1])
            if not built:
                print(f"baseline {key}: build failed\n{log[-2000:]}")
                ok = False
                continue
            status, out = self.test(suite, key[1], key[2], 3600)
            if status != "pass" or skipped(out):
                print(f"baseline {key}: {status}{' with skipped tests' if skipped(out) else ''}\n{out[-2000:]}")
                ok = False
            else:
                print(f"baseline {key[1]} {key[2]}: pass")
        return ok

    def run(self, m: Mutant) -> tuple[str, float, str]:
        start = time.monotonic()
        suite = self.suites[m.suite]
        path = os.path.join(ROOT, m.file)
        with open(path, "rb") as f:
            original = f.read()
        try:
            mutated = apply(original.decode(), m)
        except ValueError as e:
            return "INVALID", 0.0, str(e)
        self.pending_restore = (path, original)
        try:
            with open(path, "w") as f:
                f.write(mutated)
            built, log = self.build(suite, m.binary)
            if not built:
                return "INVALID", time.monotonic() - start, "does not compile:\n" + log[-1500:]
            status, out = self.test(suite, m.binary, m.filter, m.timeout)
        finally:
            self.restore()
        verdict = {"fail": "killed", "timeout": "timeout", "pass": "SURVIVED"}[status]
        return verdict, time.monotonic() - start, out[-1500:] if self.verbose else ""


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(ROOT, "tests", "mutants.toml"))
    ap.add_argument("-k", dest="select", help="only mutants whose name contains this")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true", help="trust that the unmutated suites pass")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    suites, mutants = load(args.manifest)
    if args.select:
        mutants = [m for m in mutants if args.select in m.name]
    if args.list:
        for m in mutants:
            print(f"{m.name:32} {m.file}: {m.why}")
        return 0
    if not mutants:
        print("no mutants selected")
        return 1

    runner = Runner(suites, args.verbose)

    def on_signal(signum, _frame):
        runner.restore()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if not args.skip_baseline and not runner.baseline(mutants):
        print("unmutated baseline is not clean; no mutant verdict would mean anything")
        return 2

    results = []
    try:
        for m in mutants:
            verdict, secs, detail = runner.run(m)
            results.append((m, verdict))
            print(f"{verdict:9} {secs:7.1f}s  {m.name}  ({m.file})", flush=True)
            if detail and (verdict in ("SURVIVED", "INVALID") or args.verbose):
                print("    " + detail.replace("\n", "\n    "))
    finally:
        runner.restore()
        # Leave the binaries built from the unmutated sources.
        for suite, binary in sorted({(m.suite, m.binary) for m in mutants}):
            runner.build(suites[suite], binary)

    bad = [(m, v) for m, v in results if v in ("SURVIVED", "INVALID")]
    killed = len(results) - len(bad)
    print(f"\n{killed}/{len(results)} killed")
    for m, v in bad:
        print(f"  {v}: {m.name} -- {m.why}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
