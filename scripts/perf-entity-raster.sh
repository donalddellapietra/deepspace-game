#!/usr/bin/env bash
# Perf sweep for the entity render modes across 1 / 1k / 10k / 100k
# soldier entities, at 640×360 for a fixed 300-frame render-harness
# run. Prints a compact summary table so ray-march vs raster can be
# compared at a glance.
#
# Usage:
#   scripts/perf-entity-raster.sh              # both modes, all counts
#   scripts/perf-entity-raster.sh raster       # raster only
#   scripts/perf-entity-raster.sh ray-march    # ray-march only
#   scripts/perf-entity-raster.sh raster 10000 # single-count drill-down
#
# Frame 0 is always the slowest in raster mode (mesh extraction runs
# once per unique subtree NodeId). 300 frames is enough that the
# warm-up frame averages out; steady-state is what to compare. Worst-
# frame stats are also printed so you can see the extract spike.

set -euo pipefail

WORKTREE="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="${WORKTREE}/target/release/deepspace-game"
SOLDIER="${WORKTREE}/assets/vox/soldier.vox"
TMP="${WORKTREE}/tmp"
mkdir -p "$TMP"

MODES=("ray-march" "raster")
COUNTS=(1 1000 10000 100000)

if [[ $# -ge 1 ]]; then
  MODES=("$1")
fi
if [[ $# -ge 2 ]]; then
  COUNTS=("$2")
fi

cargo build --release --bin deepspace-game --quiet

printf "%-10s %-8s  %-10s  %-10s  %-10s  %-10s  %-12s\n" \
  "mode" "count" "total(avg)" "render" "gpu_pass" "upload" "worst_total"

for mode in "${MODES[@]}"; do
  for count in "${COUNTS[@]}"; do
    # 100k under ray-march is not expected to fit; skip it to keep
    # the loop from hitting the test-runner's wall-clock timeout.
    if [[ "$mode" == "ray-march" && "$count" -gt 10000 ]]; then
      printf "%-10s %-8s  %-10s  %-10s  %-10s  %-10s  %-12s\n" \
        "$mode" "$count" "skip" "skip" "skip" "skip" "skip"
      continue
    fi
    timeout_secs=25
    if [[ "$count" -ge 100000 ]]; then timeout_secs=55; fi
    out="$(
      timeout $((timeout_secs + 5)) "$BINARY" \
        --render-harness --disable-overlay --disable-highlight \
        --plain-world --plain-layers 40 \
        --spawn-depth 6 --spawn-xyz 1.5 1.5 1.8 \
        --spawn-yaw 0 --spawn-pitch 0 \
        --spawn-entity "$SOLDIER" --spawn-entity-count "$count" \
        --entity-render "$mode" \
        --harness-width 640 --harness-height 360 \
        --exit-after-frames 300 --timeout-secs $timeout_secs \
        2>&1 || true
    )"
    timing=$(echo "$out" | grep "render_harness_timing" | tail -1)
    worst=$(echo "$out" | grep "render_harness_worst" | tail -1)
    total=$(echo "$timing" | grep -oE 'total=[0-9.]+'      | cut -d= -f2)
    render=$(echo "$timing" | grep -oE 'render=[0-9.]+'    | cut -d= -f2)
    gpu=$(echo "$timing"    | grep -oE 'gpu_pass=[0-9.]+'  | cut -d= -f2)
    upload=$(echo "$timing" | grep -oE 'upload=[0-9.]+'    | cut -d= -f2)
    worst_total=$(echo "$worst" | grep -oE 'total_ms=[0-9.]+@frame[0-9]+' | head -1)
    printf "%-10s %-8s  %-10s  %-10s  %-10s  %-10s  %-12s\n" \
      "$mode" "$count" \
      "${total:-na}" "${render:-na}" "${gpu:-na}" "${upload:-na}" \
      "${worst_total:-na}"
  done
done
