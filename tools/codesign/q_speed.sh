#!/usr/bin/env bash
# Stream Q: arm B speed. Run only under the GPU lock:
#   flock /data/halo-work/gpu.lock /data/halo-work/wt-codesign/tools/codesign/q_speed.sh prefill|decode
set -uo pipefail
LOG=/data/halo-work/streams/LOG.md
WT=/data/halo-work/wt-codesign
B=/data/halo-work/codesign/bin
S=/data/halo-work/codesign/scores
M=/data/models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
[ "$avail" -ge 85 ] || { echo "only ${avail} GiB available, not starting" >&2; exit 75; }
echo "$(date '+%F %H:%M') Q take lock: $1 timing, --moe-topk 10/8/6" >> "$LOG"
rc=0
case $1 in
  prefill)  # ab_bench pins and unpins the clock itself
    timeout 840 python3 "$WT/tools/ab_bench.py" --model "$M" --prompt /data/halo-work/prose_1.txt \
      --warm /data/halo-work/prose_8888.txt --arm "k10=$B/halo_run" --arm "k8=$B/halo_run --moe-topk 8" \
      --arm "k6=$B/halo_run --moe-topk 6" --rounds 4 --reps 3 --out "$S/ab_topk.json" > "$S/ab_topk.log" 2>&1 || rc=$?
    ;;
  decode)
    "$WT/tools/gpu_clock.sh" pin > /dev/null 2>&1
    trap '"$WT/tools/gpu_clock.sh" unpin > /dev/null 2>&1' EXIT
    : > "$S/decode_topk.log"
    for round in 1 2 3; do
      for k in 10 8 6; do
        echo "round $round k $k" >> "$S/decode_topk.log"
        timeout 240 "$B/halo_run" --model "$M" --prompt "The capital of France is" --n 128 --device-sample 1 \
          --moe-topk $k 2>&1 | grep -E "steady decode|argmax|^error" >> "$S/decode_topk.log" || rc=$?
      done
    done
    "$WT/tools/gpu_clock.sh" unpin > /dev/null 2>&1
    trap - EXIT
    ;;
  sanity)  # --moe-topk 10 must equal no flag, bit for bit (prefill logits; greedy decode text)
    for a in none k10 k8; do
      f=(); [ $a = k10 ] && f=(--moe-topk 10); [ $a = k8 ] && f=(--moe-topk 8)
      timeout 300 "$B/halo_run" --model "$M" --prefill-bench 1024 --max-context 4096 --reps 1 --logits-out "$S/logits_$a.bin" "${f[@]}" \
        > "$S/sanity_pf_$a.log" 2>&1 || rc=$?
      timeout 300 "$B/halo_run" --model "$M" --prompt "The capital of France is" --n 32 "${f[@]}" \
        > "$S/sanity_dec_$a.log" 2>&1 || rc=$?
    done
    {
      cmp "$S/logits_none.bin" "$S/logits_k10.bin" && echo "prefill logits: k10 == none, identical"
      cmp -s "$S/logits_none.bin" "$S/logits_k8.bin" || echo "prefill logits: k8 differs from none (expected)"
      diff <(grep -v -E "ms|tok/s|steps/s| s$|load|toolchain|model:|ready in" "$S/sanity_dec_none.log") \
           <(grep -v -E "ms|tok/s|steps/s| s$|load|toolchain|model:|ready in" "$S/sanity_dec_k10.log") && echo "decode text: k10 == none"
    } > "$S/sanity_cmp.txt" 2>&1
    ;;
  iu4)
    "$WT/tools/gpu_clock.sh" pin > /dev/null 2>&1
    timeout 300 "$B/halo_hwprobe" wmma > "$S/hwprobe_wmma.log" 2>&1 || rc=$?
    "$WT/tools/gpu_clock.sh" unpin > /dev/null 2>&1
    ;;
  *) rc=2 ;;
esac
echo "$(date '+%F %H:%M') Q release lock: $1 timing (rc $rc)" >> "$LOG"
exit $rc
