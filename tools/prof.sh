#!/usr/bin/env bash
# Kernel-level GPU profile of a command with rocprofv3 (rocprofiler-sdk).
#
#   tools/prof.sh [rocprofv3 options --] command [args...]
#
# Examples:
#   tools/prof.sh ./build/gpu/halo_gemv_bench 20
#   tools/prof.sh --pmc FETCH_SIZE -- ./build/gpu/halo_gemv_bench 2
#
# Runs the command under `rocprofv3 --kernel-trace --stats` and prints the top
# kernels by total GPU time. With --pmc, also prints the per-kernel mean of each
# counter. gfx1151 collects only a few counters per pass (e.g. FETCH_SIZE and
# WRITE_SIZE together fail with "exceeds the capabilities"): run one at a time.
#
# Environment:
#   HRX_ROOT      hrx-system checkout          (default: <repo>/../hrx-system)
#   THEROCK       TheRock runtime prefix       (default: $HRX_ROOT/build/therock)
#   THEROCK_PROF  rocprofiler-sdk prefix       (default: $HRX_ROOT/build/therock-prof)
#   PROF_OUT      output directory             (default: a fresh mktemp dir)
#   PROF_TOP      number of kernels to print   (default: 20)
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HRX_ROOT=${HRX_ROOT:-$repo/../hrx-system}
THEROCK=${THEROCK:-$HRX_ROOT/build/therock}
THEROCK_PROF=${THEROCK_PROF:-$HRX_ROOT/build/therock-prof}
PROF_TOP=${PROF_TOP:-20}

if [[ $# -eq 0 ]]; then
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi

rocprofv3=$THEROCK_PROF/bin/rocprofv3
[[ -x $rocprofv3 ]] || { echo "prof.sh: no rocprofv3 at $rocprofv3 (set THEROCK_PROF)" >&2; exit 1; }
[[ -d $THEROCK/lib ]] || { echo "prof.sh: no TheRock runtime at $THEROCK (set THEROCK or HRX_ROOT)" >&2; exit 1; }

# "[rocprofv3 options] -- command": options are taken only when the first
# argument starts with "-", so a command's own "--" is left alone.
extra=()
if [[ $1 == -* ]]; then
  while [[ $# -gt 0 && $1 != -- ]]; do extra+=("$1"); shift; done
  [[ $# -gt 0 ]] || { echo "prof.sh: rocprofv3 options must be followed by -- command" >&2; exit 2; }
  shift
fi
[[ $# -gt 0 ]] || { echo "prof.sh: no command given" >&2; exit 2; }

out=${PROF_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/halo-prof.XXXXXX")}
mkdir -p "$out"

export ROCM_PATH=$THEROCK_PROF
export ROCPROF_TMPDIR=${ROCPROF_TMPDIR:-$out/tmp}   # else --pmc leaves .rocprofv3/ in $PWD
export LD_LIBRARY_PATH=$THEROCK_PROF/lib:$THEROCK/lib:$THEROCK/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

rc=0
"$rocprofv3" --kernel-trace --stats --output-format csv -d "$out" -o run \
  "${extra[@]}" -- "$@" || rc=$?

stats=$(find "$out" -name 'run_kernel_stats.csv' -print -quit)
if [[ -z $stats ]]; then
  echo "prof.sh: rocprofv3 exited $rc and wrote no kernel stats under $out" >&2
  exit $(( rc ? rc : 1 ))
fi
counters=$(find "$out" -name 'run_counter_collection.csv' -print -quit)

python3 - "$stats" "$PROF_TOP" "$counters" <<'EOF'
import collections, csv, sys
stats, top, counters = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = sorted(csv.DictReader(open(stats)), key=lambda r: -float(r["TotalDurationNs"]))
total = sum(float(r["TotalDurationNs"]) for r in rows)
print(f"\n== top kernels by total GPU time ({len(rows)} kernels, {total/1e6:.3f} ms total)")
print(f"{'kernel':<56} {'calls':>6} {'total ms':>10} {'avg us':>10} {'min us':>10} {'max us':>10} {'%':>6}")
for r in rows[:top]:
    name = r["Name"] if len(r["Name"]) <= 56 else r["Name"][:53] + "..."
    print(f"{name:<56} {int(r['Calls']):>6} {float(r['TotalDurationNs'])/1e6:>10.3f} "
          f"{float(r['AverageNs'])/1e3:>10.1f} {float(r['MinNs'])/1e3:>10.1f} "
          f"{float(r['MaxNs'])/1e3:>10.1f} {float(r['Percentage']):>6.2f}")
if counters:
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(counters)):
        agg[r["Counter_Name"]][r["Kernel_Name"]].append(float(r["Counter_Value"]))
    for c, by_kernel in sorted(agg.items()):
        print(f"\n== {c}: per-kernel mean over dispatches (zeros = dispatches reading 0)")
        for k, v in sorted(by_kernel.items(), key=lambda kv: -sum(kv[1])):
            print(f"{k[:56]:<56} {len(v):>6} {sum(v)/len(v):>16.1f}   zeros={sum(x == 0 for x in v)}")
EOF
echo "prof.sh: CSVs in $(dirname "$stats")"
exit $rc
