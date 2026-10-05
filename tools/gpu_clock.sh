#!/usr/bin/env bash
# Pin or release the GPU clocks for measurements (sudoers rules for these
# sysfs files are required; see docs/HARDWARE.md).
#   tools/gpu_clock.sh pin     manual level, overdrive minimum sclk 2900 MHz, top fclk
#   tools/gpu_clock.sh unpin   overdrive table reset, auto level
#   tools/gpu_clock.sh status
# profile_peak is not used: it delivers ~2.5 GHz under load (in-kernel
# counters; sysfs shows the requested 2900) and makes prefill ~10% slower.
set -euo pipefail
D=/sys/class/drm/card0/device
w() { echo "$2" | sudo -n tee "$D/$1" >/dev/null; }
case "${1:-status}" in
  pin) w power_dpm_force_performance_level manual; w pp_od_clk_voltage "s 0 2900"; w pp_od_clk_voltage c; w pp_dpm_fclk 7 ;;
  unpin) w pp_od_clk_voltage r; w pp_od_clk_voltage c; w power_dpm_force_performance_level auto ;;
  status) ;;
  *) echo "usage: $0 pin|unpin|status" >&2; exit 2 ;;
esac
echo "level $(cat $D/power_dpm_force_performance_level); $(head -3 $D/pp_od_clk_voltage | tr '\n' ' ')"
