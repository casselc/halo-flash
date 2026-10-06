#!/usr/bin/env bash
# Stream Q: one halo_score chunk. Run only under the GPU lock:
#   flock /data/halo-work/gpu.lock /data/halo-work/wt-codesign/tools/codesign/q_score.sh <label> <shard1> short|long <start> <limit> [flags...]
# Writes scores/<label>_<set>_<start>.log (via .tmp, renamed only when the run completed).
set -uo pipefail
LOG=/data/halo-work/streams/LOG.md
WT=/data/halo-work/wt-codesign
BIN=/data/halo-work/codesign/bin/halo_score
D=${HALO_FIXTURES:-$HOME/opencode/src/ds4/gguf-tools/quality-testing/data}
label=$1 model=$2 set=$3 start=$4 limit=$5; shift 5
case $set in short) fix=$D/qwen38-flash-alibaba-100 ;; long) fix=$D/qwen38-flash-alibaba-long ;; *) exit 2 ;; esac
out=/data/halo-work/codesign/scores/${label}_${set}_${start}.log
# Each run maps ~80 GB of weights: refuse (exit 75, the driver retries) rather than OOM anyone.
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
[ "$avail" -ge 85 ] || { echo "$(date '+%F %H:%M:%S') only ${avail} GiB available, not starting" >&2; exit 75; }
echo "$(date '+%F %H:%M') Q take lock: score $label $set start $start limit $limit" >> "$LOG"
"$WT/tools/gpu_clock.sh" pin > /dev/null 2>&1
trap '"$WT/tools/gpu_clock.sh" unpin > /dev/null 2>&1' EXIT
timeout 840 "$BIN" --model "$model" --fixtures "$fix" --prefill 1 --start "$start" --limit "$limit" "$@" > "$out.tmp" 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q "^NLL " "$out.tmp"; then mv "$out.tmp" "$out"; fi
"$WT/tools/gpu_clock.sh" unpin > /dev/null 2>&1
trap - EXIT
echo "$(date '+%F %H:%M') Q release lock: score $label $set start $start (rc $rc)" >> "$LOG"
exit $rc
