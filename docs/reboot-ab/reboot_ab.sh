#!/usr/bin/env bash
# Stream L2: labelled baseline for the IOMMU (iommu=pt) / PCIe (Gen4) reboot A/B.
#
#   reboot_ab.sh <label> [phase ...]     label: pre | post (any [a-z0-9_]+)
#   phases (default: all, in order): sys prefill decode ngram probe summary
#
# Idempotent and resumable: each phase writes its raw output under reboot_<label>/ and a .done marker; a
# finished phase is skipped (delete its .done to redo it). GPU phases take /data/halo-work/gpu.lock themselves
# (flock on this script) and log take / release in streams/LOG.md; each holds it for ~3-6 min.
# Fixed inputs so pre and post are comparable: binary /data/halo-work/halo_run_50d5a35 (main 50d5a35),
# probe /data/halo-work/streams/L2/halo_hwprobe_pl, tools from the wt-pageloc worktree (stream/pageloc).
set -u
LABEL=${1:?usage: reboot_ab.sh <label> [phase ...]}
shift
[[ $LABEL =~ ^[a-z0-9_]+$ ]] || { echo "bad label $LABEL" >&2; exit 2; }
L2=/data/halo-work/streams/L2
W=/data/halo-work/wt-pageloc
OUT=$L2/reboot_$LABEL
BIN=/data/halo-work/halo_run_50d5a35
PROBE=$L2/halo_hwprobe_pl
M=/data/models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
LOCK=/data/halo-work/gpu.lock
LOG=/data/halo-work/streams/LOG.md
mkdir -p "$OUT"

log() { echo "$(date '+%Y-%m-%d %H:%M') L2 $*" >> $LOG; }

# Internal: run one GPU phase body while holding the lock (called through flock).
if [ "$LABEL" = "_locked" ]; then
  LABEL=$1 PHASE=$2
  OUT=$L2/reboot_$LABEL
  cd "$W" || exit 1
  source .nix-env.sh >/dev/null 2>&1
  log "take lock: reboot_ab $LABEL $PHASE"
  trap 'tools/gpu_clock.sh unpin >/dev/null 2>&1; log "release lock: reboot_ab $LABEL $PHASE"' EXIT
  case $PHASE in
    prefill)
      # ab_bench single arm (pins the clock itself, 3 rounds x 3 reps, warm prefill prose_8888 first).
      for t in 2048 8192; do
        f=$OUT/prefill_$t.json
        [ -s $f.done ] && continue
        if [ $t = 2048 ]; then P=/data/halo-work/prose_1.txt; C=1; else P=/data/halo-work/streams/S/prose_10k.txt; C=0; fi
        python3 tools/ab_bench.py --model $M --prompt $P --warm /data/halo-work/prose_8888.txt --tokens $t --chunks $C \
          --arm main50d5a35=$BIN --rounds 3 --reps 3 --no-kernels --out $f > $OUT/prefill_$t.txt 2>&1
        echo "rc $?" >> $OUT/prefill_$t.txt
        grep -q "median" $OUT/prefill_$t.txt && echo ok > $f.done
      done ;;
    decode)
      tools/gpu_clock.sh pin > $OUT/decode_clock.txt 2>&1
      for r in 1 2 3; do
        for ctx in short 7k; do
          f=$OUT/decode_${ctx}_r$r.txt
          [ -s $f ] && grep -q steady $f && continue
          if [ $ctx = short ]; then A="--max-context 9000"; else A="--prompt-file /data/halo-work/prose_long.txt --prefill-prompt 1 --max-context 9000"; fi
          $BIN --model $M $A --n 80 --device-sample 1 2>&1 | grep -v "Basic " > $f
        done
      done
      for ctx in short 7k; do  # one eager kernel-timed run per context (per-kernel us/call)
        f=$OUT/decode_${ctx}_kt.txt
        [ -s $f ] && grep -q steady $f && continue
        if [ $ctx = short ]; then A="--max-context 9000"; else A="--prompt-file /data/halo-work/prose_long.txt --prefill-prompt 1 --max-context 9000"; fi
        $BIN --model $M $A --n 80 --decode-timing 1 2>&1 | grep -v "Basic " > $f
      done
      tools/gpu_clock.sh unpin >> $OUT/decode_clock.txt 2>&1
      echo ok > $OUT/decode.done ;;
    ngram)
      # Non-resident n-gram rows (O_DIRECT from the GGUF on XFS: the page cache is not involved, so no drop is
      # needed; --ngram-cache-gb 0 turns off the in-process row cache, so every rep reads from the NVMe).
      # Prompt: shuffled dictionary words, seeded by the label + time, never seen before (novel n-grams).
      tools/gpu_clock.sh pin > $OUT/ngram_clock.txt 2>&1
      python3 - "$OUT/ngram_prompt.txt" "$LABEL$(date +%s%N)" <<'PY'
import random, sys
w = [x.strip() for x in open("/usr/share/dict/words") if x.strip().isalpha() and x.strip().islower()]
r = random.Random(sys.argv[2])
open(sys.argv[1], "w").write(" ".join(r.choice(w) for _ in range(3000)) + "\n")
PY
      HALO_PREFILL_TIMING=1 $BIN --model $M --prefill-bench 2048 --max-context 2048 --reps 3 \
        --prefill-file $OUT/ngram_prompt.txt --ngram-cache-gb 0 2>&1 | grep -v "Basic " > $OUT/ngram_novel_nocache.txt
      # Default cache (4 GiB): the first prefill of the process is cold, later reps hit the process cache.
      python3 - "$OUT/ngram_prompt2.txt" "$LABEL$(date +%s%N)b" <<'PY'
import random, sys
w = [x.strip() for x in open("/usr/share/dict/words") if x.strip().isalpha() and x.strip().islower()]
r = random.Random(sys.argv[2])
open(sys.argv[1], "w").write(" ".join(r.choice(w) for _ in range(3000)) + "\n")
PY
      HALO_PREFILL_TIMING=1 $BIN --model $M --prefill-bench 2048 --max-context 2048 --reps 3 \
        --prefill-file $OUT/ngram_prompt2.txt 2>&1 | grep -v "Basic " > $OUT/ngram_novel_cache.txt
      # Raw sequential read of the GGUF shard tail (O_DIRECT, 2 GiB): NVMe / PCIe throughput.
      dd if=${M%-00001-of-00004.gguf}-00004-of-00004.gguf of=/dev/null bs=4M count=512 iflag=direct > $OUT/nvme_dd.txt 2>&1
      tools/gpu_clock.sh unpin >> $OUT/ngram_clock.txt 2>&1
      echo ok > $OUT/ngram.done ;;
    probe)
      tools/gpu_clock.sh pin > $OUT/probe_key.txt 2>&1
      $PROBE clock >> $OUT/probe_key.txt 2>&1
      $PROBE pageloc key >> $OUT/probe_key.txt 2>&1
      $PROBE dram >> $OUT/probe_key.txt 2>&1
      $PROBE latency >> $OUT/probe_key.txt 2>&1
      tools/gpu_clock.sh unpin >> $OUT/probe_key.txt 2>&1
      echo ok > $OUT/probe.done ;;
  esac
  exit 0
fi

phases=("$@")
[ ${#phases[@]} -eq 0 ] && phases=(sys prefill decode ngram probe summary)
for ph in "${phases[@]}"; do
  case $ph in
    sys)
      {
        echo "date: $(date -Is)"
        echo "uname: $(uname -a)"
        echo "cmdline: $(cat /proc/cmdline)"
        echo "iommu_groups: $(ls /sys/kernel/iommu_groups | wc -l)"
        echo "iommu_types:"; cat /sys/kernel/iommu_groups/*/type | sort | uniq -c
        GPU=$(for c in /sys/class/drm/card[0-9]*; do case $c in *-*) ;; *) [ "$(basename "$(readlink -f "$c/device/driver")")" = amdgpu ] && echo "$c/device" && break ;; esac; done)
        echo "gpu_sysfs: $GPU"
        echo "gpu_iommu_group_type: $(cat $GPU/iommu_group/type 2>/dev/null)"
        for n in /sys/class/nvme/nvme*; do
          echo "nvme $(basename $n): $(cat $n/model 2>/dev/null | xargs) speed $(cat $n/device/current_link_speed) width $(cat $n/device/current_link_width) (max $(cat $n/device/max_link_speed) x$(cat $n/device/max_link_width)) group_type $(cat $n/device/iommu_group/type 2>/dev/null)"
        done
        echo "gpu_link: $(cat $GPU/current_link_speed) x$(cat $GPU/current_link_width)"
        echo "vram_total: $(cat $GPU/mem_info_vram_total) gtt_total: $(cat $GPU/mem_info_gtt_total)"
        grep -E "MemTotal|MemAvailable" /proc/meminfo
        echo "model_df: $(df -T /data | tail -1)"
      } > $OUT/sys.txt 2>&1
      echo ok > $OUT/sys.done ;;
    prefill|decode|ngram|probe)
      if [ -s $OUT/$ph.done ] || { [ $ph = prefill ] && [ -s $OUT/prefill_2048.json.done ] && [ -s $OUT/prefill_8192.json.done ]; }; then
        echo "phase $ph done (rm $OUT/$ph.done to redo)"; continue
      fi
      flock $LOCK "$0" _locked "$LABEL" "$ph"
      [ $ph = prefill ] && [ -s $OUT/prefill_2048.json.done ] && [ -s $OUT/prefill_8192.json.done ] && echo ok > $OUT/prefill.done ;;
    summary)
      python3 $L2/reboot_summary.py "$LABEL" ;;
    *) echo "unknown phase $ph" >&2; exit 2 ;;
  esac
done
