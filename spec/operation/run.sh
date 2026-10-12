#!/usr/bin/env bash
# run.sh: the checks of the operation model (docs/OPERATIONS.md; `quint run` only, never `quint verify`).
#  - sim rows: an instance against a property (design: ok; worlds and mutants as named);
#  - wit row: every witness of the design is reached.
set -u
OUT=${OUT:-/dev/null}
SEED=${SEED:-0x5eed}
SAMPLES=${SAMPLES:-10000}
STEPS=${STEPS:-40}
cd "$(dirname "$0")" || exit 1
F=operation_modes.qnt
echo "quint $(quint --version), seed $SEED, steps $STEPS, samples $SAMPLES" | tee "$OUT"
fails=0
sim() { # main invariant expect
  local out v
  out=$(quint run "$F" --main "$1" --invariant "$2" --max-samples "$SAMPLES" --max-steps "$STEPS" --seed "$SEED" --verbosity 1 2>&1)
  v=ok; echo "$out" | grep -q "\[violation\]" && v=VIOLATED
  echo "$out" | grep -q -E "\[ok\]|\[violation\]" || v=ERROR
  [ "$v" = "$3" ] || fails=$((fails + 1))
  printf "sim   %-16s %-18s %-8s (expect %s)\n" "$1" "$2" "$v" "$3" | tee -a "$OUT"
}
sim design          safety           ok
sim promised        safety           ok
sim promised        noOvercharge     ok
sim cacheLike       safety           ok
sim cacheLike       noOvercharge     VIOLATED
sim noFence         runsOnce         VIOLATED
sim noFence         accounted        VIOLATED
sim noTomb          runsOnce         VIOLATED
sim abandonUntombed accounted        VIOLATED
sim cacheNoTomb     neverAcceptedTrue VIOLATED
W="wDone,wRestartedThenDone,wNeverThenDone,wDeduped,wStaleRejected,wRefused,wAbandonedThenDone,wUnknown"
out=$(quint run "$F" --main design --witnesses ${W//,/ } --max-samples "$SAMPLES" --max-steps "$STEPS" --seed "$SEED" --verbosity 1 2>&1)
echo "$out" | grep -E "witnessed" | sed 's/^/wit   design /' | tee -a "$OUT"
miss=$(echo "$out" | grep -c "in 0 trace")
[ "$miss" = 0 ] || fails=$((fails + 1))
out=$(quint run "$F" --main cacheLike --witnesses wForgotten --max-samples "$SAMPLES" --max-steps "$STEPS" --seed "$SEED" --verbosity 1 2>&1)
echo "$out" | grep -E "witnessed" | sed 's/^/wit   cacheLike /' | tee -a "$OUT"
echo "$out" | grep -q "in 0 trace" && fails=$((fails + 1))
out=$(quint run "$F" --main promised --witnesses wDone --max-samples "$SAMPLES" --max-steps "$STEPS" --seed "$SEED" --verbosity 1 2>&1)
echo "$out" | grep -E "witnessed" | sed "s/^/wit   promised /" | tee -a "$OUT"
echo "$out" | grep -q "in 0 trace" && fails=$((fails + 1))
echo "failures: $fails" | tee -a "$OUT"
exit $((fails > 0))
