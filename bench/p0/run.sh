#!/usr/bin/env bash
# Builds and runs the P0 runtime microbenchmarks on all three backends.
# Requires: HRX_ROOT (an hrx-system checkout built into build/cmake, with a
# TheRock runtime in build/therock), clang with amdgcn support, hipcc.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${OUT:-$here/../../build/p0}"
: "${HRX_ROOT:?set HRX_ROOT to the hrx-system checkout}"
mkdir -p "$out"
therock="$HRX_ROOT/build/therock/lib:$HRX_ROOT/build/therock/lib/rocm_sysdeps/lib"
libhrx="$HRX_ROOT/build/cmake/libhrx/src/libhrx"
shim="$(ls "$HRX_ROOT"/build/cmake/libhrx/src/binding/hip/libamdhip64.so.* | head -1)"

clang -x c -std=c23 -nogpulib -target amdgcn-amd-amdhsa -mcpu=gfx1151 -O3 \
  -fvisibility=hidden -o "$out/kernels.hsaco" "$here/kernels.c"
hipcc -O2 -std=c++20 -o "$out/bench_hip" "$here/bench_hip.cpp"
clang++ -O2 -std=c++20 -I"$HRX_ROOT/libhrx/include" -o "$out/bench_hrx" \
  "$here/bench_hrx.cpp" -L"$libhrx" -lhrx -Wl,-rpath,"$libhrx"

args=(--hsaco "$out/kernels.hsaco" "$@")
"$out/bench_hip" "${args[@]}"
LD_LIBRARY_PATH="$therock" "$out/bench_hrx" "${args[@]}"
LD_LIBRARY_PATH="$therock:$libhrx" LD_PRELOAD="$shim" \
  HALO_BENCH_BACKEND=hrx-hip-shim HALO_BENCH_SKIP_INDEPENDENT=1 \
  "$out/bench_hip" "${args[@]}"
