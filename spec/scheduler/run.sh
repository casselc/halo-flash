#!/usr/bin/env bash
# run.sh: the checks of the G3 scheduler model (docs/SCHEDULER.md; quint 0.32, `quint run` only, never
# `quint verify`). As in yard spec/warden/run.sh:
#  - `sim` rows: the design instance by simulation (expect ok), and each mutant against the property it is named
#    for (expect VIOLATED); `today` is today's halo_serve scheduling (one mutex, FIFO, to the end);
#  - `wit` row: every witness of the design is reached in at least one trace.
# One quint process at a time.
#
#   spec/scheduler/run.sh                       # SAMPLES=20000 MSAMPLES=5000 STEPS=120 SEED=0x5eed
#   OUT=results.txt spec/scheduler/run.sh
set -u
OUT=${OUT:-/dev/stdout}
SEED=${SEED:-0x5eed}
SAMPLES=${SAMPLES:-20000}    # the design and its witnesses
MSAMPLES=${MSAMPLES:-5000}   # mutants
STEPS=${STEPS:-120}
cd "$(dirname "$0")" || exit 1
F=schedTest.qnt
echo "quint $(quint --version), seed $SEED, steps $STEPS, samples $SAMPLES (design) / $MSAMPLES (mutants)" | tee "$OUT"
fails=0
sim() { # main invariant samples expect [note]
  local t0 out v; t0=$(date +%s.%N)
  out=$(quint run "$F" --main "$1" --invariant "$2" --max-samples "$3" --max-steps "$STEPS" --seed "$SEED" --verbosity 1 2>&1)
  v=ok; echo "$out" | grep -q "\[violation\]" && v=VIOLATED
  echo "$out" | grep -q -E "\[ok\]|\[violation\]" || v=ERROR
  [ "$v" = "$4" ] || fails=$((fails + 1))
  printf "sim   %-21s %-21s %6s x %-3s %-8s (expect %-8s) %6.1fs %s\n" "$1" "$2" "$3" "$STEPS" "$v" "$4" \
    "$(echo "$(date +%s.%N) - $t0" | bc)" "${5:-}" | tee -a "$OUT"
}
wit() { # main witnesses...
  local m=$1; shift
  local out; out=$(quint run "$F" --main "$m" --max-samples "$SAMPLES" --max-steps "$STEPS" --seed "$SEED" --verbosity 1 --witnesses "$@" 2>&1)
  echo "$out" | grep -E "witnessed in" | sed "s/^/wit   $m /" | tee -a "$OUT"
  if echo "$out" | grep -q "witnessed in 0 trace" || ! echo "$out" | grep -q "witnessed in"; then
    fails=$((fails + 1)); echo "wit   $m: a witness was not reached (or the run failed)" | tee -a "$OUT"
  fi
}

# ---- the design ----
sim design safety "$SAMPLES" ok "(all 11 properties)"
SEED=0xbeef sim design safety "$SAMPLES" ok "(all 11 properties, second seed 0xbeef)"
wit design wPreempted wBatched wDeciderBetweenSlices wDeciderFair wParkedOnCancel wEvicted wMem429 \
  wRetryAfter429 wTimeout wFencedFin wFencedVanished wLateArrival wReplayed wAdopted wTurn2Done
# keyless clients (no idempotency key): the rest of the design still holds what does not need keys
sim noIdempotency noRecompute "$MSAMPLES" ok "(keyless: park on cancel + bind at start still avoid recompute)"
sim noIdempotency boundedWait "$MSAMPLES" ok "(keyless)"
SAMPLES=$MSAMPLES wit noIdempotency wLimit429 wBoundLater

# ---- today's halo_serve (one generator mutex, FIFO, each request to its end) ----
sim today boundedWait    "$MSAMPLES" VIOLATED "a turn waits for another sandbox's whole request"
sim today deciderLatency "$MSAMPLES" VIOLATED "a decider waits for a whole agent request"

# ---- mutants: one design rule off each ----
sim noPreempt            boundedWait          "$MSAMPLES" VIOLATED
sim noDeciderPrio        deciderLatency       "$MSAMPLES" VIOLATED
sim strictDeciderPrio    boundedWait          "$MSAMPLES" VIOLATED "a decider flood starves the sandboxes"
sim noFence              workAfterGoneBounded "$MSAMPLES" VIOLATED
sim noDeadPeerProbe      workAfterGoneBounded "$MSAMPLES" VIOLATED "a peer gone without FIN"
sim noHeartbeat          noSpuriousTimeout    "$MSAMPLES" VIOLATED "AUDIT D3"
sim dropOnCancel         noRecompute          "$MSAMPLES" VIOLATED
sim bindAtAdmission      noRecompute          "$MSAMPLES" VIOLATED "(keyless; with keys it is equivalent)"
sim memoryReadingOnly    noOvercommit         "$MSAMPLES" VIOLATED "AUDIT D1/E3"
sim releasedStaysCharged refuseOnlyIfShort    "$MSAMPLES" VIOLATED "AUDIT F3"
sim evictPlanned         withinReservation    "$MSAMPLES" VIOLATED
sim queued429            rejectedNeverRuns    "$MSAMPLES" VIOLATED "AUDIT N4"
sim noIdempotency        generatedOnce        "$MSAMPLES" VIOLATED "a retry after a lost answer"
sim noIdempotency        noSelfInflicted429   "$MSAMPLES" VIOLATED "a retry refused for its own attempt"

echo "failures: $fails" | tee -a "$OUT"
exit $((fails > 0))
