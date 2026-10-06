#!/usr/bin/env bash
# Stream Q: resumable driver. Each chunk takes the GPU lock on its own (other
# streams interleave); finished chunks (scores/*.log present) are skipped.
#   q_sweep.sh [arm ...]     arms: base A B8 B6 D (default: all, in that order)
set -uo pipefail
R=/data/halo-work/wt-codesign/tools/codesign
M=/data/models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
F=Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
arms=("$@"); [ ${#arms[@]} -eq 0 ] && arms=(base A B8 B6 D)
for arm in "${arms[@]}"; do
  case $arm in
    base) model=$M flags=() ;;
    A) model=/data/halo-work/codesign/A-rot4/$F flags=() ;;
    B8) model=$M flags=(--moe-topk 8) ;;
    B6) model=$M flags=(--moe-topk 6) ;;
    D) model=/data/halo-work/codesign/D-exp2/$F flags=() ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  for chunk in "short 0 98" "long 0 6" "long 6 6"; do
    set -- $chunk
    [ -f /data/halo-work/codesign/scores/${arm}_$1_$2.log ] && continue
    while :; do
      flock /data/halo-work/gpu.lock "$R/q_score.sh" "$arm" "$model" "$1" "$2" "$3" "${flags[@]}"
      rc=$?
      [ $rc -eq 75 ] || break
      sleep 60  # low memory: lock released, retry
    done
    [ $rc -eq 0 ] || { echo "chunk $arm $chunk rc $rc: stopping"; exit 1; }
  done
done
