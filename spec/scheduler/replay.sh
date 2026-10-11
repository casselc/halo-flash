#!/usr/bin/env bash
# replay.sh (G3 M5): N traces of the design instance (`quint run --mbt --out-itf`; `quint run` only, never verify),
# turned into the server-side events tests/server/scheduler_replay_test.cpp replays against the Scheduler class.
#   spec/scheduler/replay.sh            # N=24 SEED=0x5eed STEPS=160 -> tests/server/data/sched_traces/*.jsonl
set -eu
N=${N:-24}; SEED=${SEED:-0x5eed}; STEPS=${STEPS:-160}
OUT=${OUT:-../../tests/server/data/sched_traces}
cd "$(dirname "$0")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
quint run schedTest.qnt --main design --invariant safety --max-samples "$N" --n-traces "$N" --max-steps "$STEPS" \
  --seed "$SEED" --mbt --out-itf "$tmp/t{seq}.itf.json" > "$tmp/run.log" 2>&1 || { cat "$tmp/run.log"; exit 1; }
mkdir -p "$OUT"
rm -f "$OUT"/*.jsonl
python3 itf_events.py "$OUT" "$tmp"/t*.itf.json
echo "$(ls "$OUT"/*.jsonl | wc -l) traces, $(cat "$OUT"/*.jsonl | wc -l) events in $OUT"
