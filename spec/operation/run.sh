#!/usr/bin/env bash
# run.sh: the checks of the operation model (docs/OPERATIONS.md; `quint run` only, never `quint verify`).
#  - sim rows: an instance against a property (the design: ok; worlds and mutants as named in operation_modes.qnt);
#  - wit rows: every witness of the design is reached.
#   OUT=results.txt spec/operation/run.sh       # SAMPLES=10000 STEPS=60 SEED=0x5eed
set -u
cd "$(dirname "$0")" || exit 1
OUT=${OUT:-/dev/null}
SEED=${SEED:-0x5eed}
SAMPLES=${SAMPLES:-10000}
STEPS=${STEPS:-60}
F=operation_modes.qnt
echo "quint $(quint --version), seed $SEED, steps $STEPS, samples $SAMPLES" | tee "$OUT"
fails=0
sim() { # main invariant expect
  local out v
  out=$(quint run "$F" --main "$1" --invariant "$2" --max-samples "$SAMPLES" --max-steps "$STEPS" --seed "$SEED" --verbosity 1 2>&1)
  v=ok; echo "$out" | grep -q "\[violation\]" && v=VIOLATED
  echo "$out" | grep -q -E "\[ok\]|\[violation\]" || v=ERROR
  [ "$v" = "$3" ] || fails=$((fails + 1))
  printf "sim   %-19s %-18s %-8s (expect %s)\n" "$1" "$2" "$v" "$3" | tee -a "$OUT"
}
wit() { # main witnesses...
  local m=$1 out; shift
  out=$(quint run "$F" --main "$m" --witnesses "$@" --max-samples "$SAMPLES" --max-steps "$STEPS" --seed "$SEED" --verbosity 1 2>&1)
  echo "$out" | grep -E "witnessed" | sed "s/^/wit   $m /" | tee -a "$OUT"
  [ "$(echo "$out" | grep -c witnessed)" = "$#" ] || fails=$((fails + 1))
  echo "$out" | grep -q "in 0 trace" && fails=$((fails + 1))
}
sim design             safety            ok
sim promised           safety            ok
sim promised           noOvercharge      ok
sim cacheLike          core              ok
sim cacheLike          noOvercharge      VIOLATED
sim noFence            runsOnce          VIOLATED
sim noDeadline         runsOnce          VIOLATED
sim pruneEarly         runsOnce          VIOLATED
sim dropExpiredOutcome neverAcceptedTrue VIOLATED
sim noReserve          promiseKept       VIOLATED
sim ackBeforeCommit    noReceiptLost     VIOLATED
sim cancelUntombed     accounted         VIOLATED
sim refusalCloses      accounted         VIOLATED
sim noReceiptDedup     appliedOnce       VIOLATED
sim eraseForRoom       neverAcceptedTrue VIOLATED
wit design wDone wRestartedThenDone wCrashedThenCommitted wAckLostThenDone wDupReceipt wNeverThenDone \
  wClosedByDeadline wCancelledBefore wCancelledRunningCommitted wExpiredSettled wRefused wStale wLate wDeduped \
  wPruned wUnknown
wit cacheLike wForgotten
echo "failures: $fails" | tee -a "$OUT"
exit $((fails > 0))
