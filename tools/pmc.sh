#!/usr/bin/env bash
# rocprofv3 with the instruction counters kept armed.
#   tools/pmc.sh <rocprofv3 args...> -- <app> <args...>
# On this machine (kernel 7.0, ROCm 7.2.3) SQ_INSTS_* read 0 on every dispatch after the GPU idles for more
# than ~60-70 ms during counter collection (SQ_WAVES keeps counting) until some process creates a GPU queue
# (docs/HARDWARE.md). A side process submitting an empty marker every 10 ms keeps the GPU from idling; it
# launches no waves, so it disturbs no counter. Take the GPU lock around this as for any GPU run.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
probe=${HALO_HWPROBE:-$here/build/nix-gpu/halo_hwprobe}
"$probe" keepalive 10 100000 &
ka=$!
sleep 1
rocprofv3 "$@"
rc=$?
kill "$ka" 2>/dev/null
wait "$ka" 2>/dev/null
exit $rc
