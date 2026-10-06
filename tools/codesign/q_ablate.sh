#!/usr/bin/env bash
# Stream Q: short-set scores of extra variants (resumable; lock per chunk).
#   q_ablate.sh <label>=<variant dir> ...
set -uo pipefail
R=/data/halo-work/wt-codesign/tools/codesign
F=Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
for a in "$@"; do
  label=${a%%=*} dir=${a#*=}
  [ -f /data/halo-work/codesign/scores/${label}_short_0.log ] && continue
  while :; do
    flock /data/halo-work/gpu.lock "$R/q_score.sh" "$label" "$dir/$F" short 0 98
    rc=$?
    [ $rc -eq 75 ] || break
    sleep 60
  done
  [ $rc -eq 0 ] || { echo "$label rc $rc"; exit 1; }
done
