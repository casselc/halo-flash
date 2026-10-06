#!/usr/bin/env bash
# Crash-tested resume for halo_train (the standing rule for long runs): a run
# killed at arbitrary points -- after a step but before its checkpoint, in the
# middle of a checkpoint write (temp file synced, never renamed), by an
# external SIGKILL at a random moment, and stopped by its time budget -- and
# resumed each time must end with a checkpoint byte-identical to an
# uninterrupted run's, and an identical per-step loss log. A config change
# must be refused, not resumed.
#
# Usage: tools/train_resume_test.sh <build-dir> <tiny-qwen35-dir> <task-dir> <work-dir>
#   tiny-qwen35-dir: make_tiny_qwen35_reference.py output with tiny-qwen35-f32.gguf
#   task-dir:        make_sentiment_task.py output
set -euo pipefail
BUILD=$1 TINY=$2 TASK=$3 WORK=$4
BIN=$BUILD/halo_train
rm -rf "$WORK" && mkdir -p "$WORK"
head -96 "$TASK/train.jsonl" > "$WORK/train.jsonl"
head -32 "$TASK/heldout.jsonl" > "$WORK/heldout.jsonl"
cfg() {  # $1 = file, $2 = lr
  cat > "$1" <<EOF
{"base": "$TINY/tiny-qwen35-f32.gguf", "train_data": "train.jsonl", "eval_data": "heldout.jsonl",
 "work_dir": "w", "engine": "cpu", "last_k": 4, "head": "lm",
 "lora": {"rank": 4, "alpha": 8, "targets": ["attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv",
          "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out", "ffn_gate", "ffn_up", "ffn_down"]},
 "steps": 60, "batch_size": 4, "lr": $2, "head_lr": 1e-3, "weight_decay": 0.01, "warmup_steps": 5,
 "max_grad_norm": 1.0, "seed": 3, "checkpoint_every": 5, "eval_every": 20, "max_tokens": 128,
 "variant": {"name": "tiny-resume", "version": 1, "base_name": "tiny"}}
EOF
}
cfg "$WORK/run.json" 1e-3
cfg "$WORK/other.json" 2e-3
"$BIN" cache --config "$WORK/run.json" > /dev/null
steps() { grep '"step"' "$1" | grep -v '"event"' | sed 's/, "seconds": [0-9.e+-]*//'; }

# A: uninterrupted, with a different thread count than B.
"$BIN" train --config "$WORK/run.json" --threads 3 > /dev/null 2>&1
mv "$WORK/w/run" "$WORK/runA"

# B: killed four ways, resumed each time.
expect_kill() {
  set +e
  "$@" > /dev/null 2>&1
  local rc=$?
  set -e
  [ "$rc" -eq 137 ] || { echo "expected SIGKILL (137), got $rc"; exit 1; }
}
HALO_TRAIN_KILL_AT_STEP=7 expect_kill "$BIN" train --config "$WORK/run.json" --threads 5
HALO_TRAIN_KILL_IN_CHECKPOINT=15 expect_kill "$BIN" train --config "$WORK/run.json" --threads 5
ls "$WORK"/w/run/checkpoint.bin.tmp.* > /dev/null  # the torn write is left behind, never read
# A different config must not resume this run.
if "$BIN" train --config "$WORK/other.json" > "$WORK/refused.log" 2>&1; then echo "resume with another config was not refused"; exit 1; fi
grep -q "belongs to another run" "$WORK/refused.log"
# External SIGKILL at a random moment after step 22.
"$BIN" train --config "$WORK/run.json" --threads 5 > /dev/null 2>&1 &
pid=$!
until [ "$(grep -c '"loss"' "$WORK/w/run/progress.jsonl")" -ge 22 ]; do sleep 0.01; done
sleep "0.0$((RANDOM % 9))"
kill -9 "$pid"
wait "$pid" || true
# Time-budgeted chunks until done.
chunks=0
until "$BIN" train --config "$WORK/run.json" --threads 5 --max-seconds 0.3 2>/dev/null | grep -q '^done'; do
  chunks=$((chunks + 1))
  [ "$chunks" -lt 1000 ] || { echo "no progress"; exit 1; }
done

cmp "$WORK/runA/checkpoint.bin" "$WORK/w/run/checkpoint.bin"
diff <(steps "$WORK/runA/progress.jsonl") <(steps "$WORK/w/run/progress.jsonl")
echo "resume test: checkpoint byte-identical to the uninterrupted run after 3 kills (mid-step, mid-checkpoint-write, external), a refused foreign config and $chunks budgeted chunks;"
echo "  sha256 $(sha256sum "$WORK/w/run/checkpoint.bin" | cut -c1-16); per-step losses identical ($(steps "$WORK/w/run/progress.jsonl" | wc -l) steps)"
grep -c '"event": "resume"' "$WORK/w/run/progress.jsonl" | sed 's/^/  resume events logged: /'
