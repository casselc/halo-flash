#!/usr/bin/env bash
# Thread-trace chosen dispatches of one kernel inside a real run, then
# summarize where its waves spent their cycles (tools/att_summary.py).
#
#   tools/trace_kernel.sh KERNEL_REGEX ITERATIONS OUTDIR -- command [args...]
#
#   KERNEL_REGEX  rocprofv3 --kernel-include-regex (e.g. '^halo_gemm_f32_tiled$')
#   ITERATIONS    rocprofv3 --kernel-iteration-range, counted per matching
#                 kernel from 0 (e.g. '[3-4]': its 4th and 5th dispatch)
#
# Example (the router GEMM is the f32_tiled dispatch with 256 workgroups):
#   tools/trace_kernel.sh '^halo_gemm_f32_tiled$' '[2-5]' /tmp/att_router -- \
#     build/nix-gpu/halo_run --model M --prefill-bench 2048 --max-context 2048 --reps 1 --prefill-file F
#
# Trace inside the engine, not a microbenchmark: a lone kernel runs at a
# low clock and without the cache state the real pipeline leaves behind.
# Needs rocprofv3 matching the HIP runtime (the nix dev shell provides it).
set -euo pipefail
if [[ $# -lt 5 || $4 != "--" ]]; then
  sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi
regex=$1 iters=$2 out=$3
shift 4
command -v rocprofv3 >/dev/null || { echo "trace_kernel.sh: rocprofv3 not on PATH (run inside nix develop)" >&2; exit 1; }
rm -rf "$out"
rocprofv3 --att --kernel-include-regex "$regex" --kernel-iteration-range "$iters" -d "$out" -o run -- "$@" \
  > "$out.log" 2>&1 || { echo "rocprofv3 failed; see $out.log" >&2; tail -5 "$out.log" >&2; exit 1; }
python3 "$(dirname "${BASH_SOURCE[0]}")/att_summary.py" "$out"
