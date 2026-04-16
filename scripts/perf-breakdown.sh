#!/usr/bin/env bash
# Perf breakdown harness. Runs a matrix of harness configurations
# and emits one summary line per run, plus a CSV trace per run under
# tmp/perf/ for deeper analysis.
#
# Usage:
#   scripts/perf-breakdown.sh [matrix]
#
# Matrix (default: all):
#   resolution    — sweep harness dims at fixed spawn_depth
#   depth         — sweep spawn_depth at fixed 1280x720
#   world         — plain vs sphere at fixed spawn_depth + 1280x720
#
# Each run:
#   - 60 rendered frames, warmup=10
#   - CSV trace: tmp/perf/<label>.csv
#   - One "RUN <label>" header + the three summary lines from the
#     harness: avg (timing), worst, workload.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p tmp/perf

# Build once; silent unless failed.
echo "[build]"
if ! cargo build --bin deepspace-game 2>&1 | tail -3; then
  echo "build failed" >&2
  exit 1
fi

run() {
  local label="$1"; shift
  local csv="tmp/perf/${label}.csv"
  echo "=== RUN ${label} ==="
  # --timeout-secs 12 is the inner per-run kill switch; outer
  # `timeout 15` catches cargo hangs before the harness starts.
  set +e
  timeout 15 cargo run --bin deepspace-game -- \
    --render-harness --plain-world --plain-layers 20 \
    --exit-after-frames 60 --timeout-secs 12 \
    --perf-trace "$csv" --perf-trace-warmup 10 \
    --suppress-startup-logs \
    "$@" 2>&1 | grep -E "^render_harness_timing|^render_harness_worst|^render_harness_workload|^render_harness_shader|^renderer_features|^perf_trace:"
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "  (run exited non-zero: $rc)"
  fi
}

SEL="${1:-all}"

if [ "$SEL" = "resolution" ] || [ "$SEL" = "all" ]; then
  echo "### MATRIX: resolution (plain, spawn_depth=6) ###"
  for res in "64 64" "320 180" "640 360" "1280 720" "1920 1080"; do
    set -- $res
    run "res_${1}x${2}" --harness-width "$1" --harness-height "$2" --spawn-depth 6
  done
fi

if [ "$SEL" = "depth" ] || [ "$SEL" = "all" ]; then
  echo "### MATRIX: depth (plain, 1280x720) ###"
  for d in 3 6 10 14 17; do
    run "depth_${d}" --harness-width 1280 --harness-height 720 --spawn-depth "$d"
  done
fi

if [ "$SEL" = "world" ] || [ "$SEL" = "all" ]; then
  echo "### MATRIX: world preset (1280x720, spawn_depth=6) ###"
  run "world_plain" --harness-width 1280 --harness-height 720 --spawn-depth 6
  run "world_sphere" --sphere-world --harness-width 1280 --harness-height 720 --spawn-depth 6
fi

if [ "$SEL" = "room" ] || [ "$SEL" = "all" ]; then
  echo "### MATRIX: room / openness (1280x720, plain, --shader-stats) ###"
  # Baseline: flat ground, camera one cell up. Every ray hits the
  # surface immediately — closest-case workload.
  run "room_baseline" \
    --harness-width 1280 --harness-height 720 --spawn-depth 6 --shader-stats
  # One break: carves a single anchor-sized cell directly below the
  # camera. A few rays near the crosshair pass into the void before
  # hitting the deeper surface — longer marches through empty space.
  run "room_break_one" \
    --harness-width 1280 --harness-height 720 --spawn-depth 6 --shader-stats \
    --script "wait:15,break,wait:40"
  # Zoom out, break a big cell, zoom back in. The carved volume
  # spans 3^2 anchor cells; most rays now traverse a large void.
  run "room_break_big" \
    --harness-width 1280 --harness-height 720 --spawn-depth 6 --shader-stats \
    --script "wait:10,zoom_out:2,wait:5,break,wait:5,zoom_in:2,wait:30"
  # Pitch upward inside the carved volume so rays travel along the
  # open horizontal axis rather than into the floor immediately.
  run "room_break_big_horiz" \
    --harness-width 1280 --harness-height 720 --spawn-depth 6 --shader-stats \
    --script "wait:10,zoom_out:2,wait:5,break,wait:5,zoom_in:2,wait:5,pitch:-0.3,wait:25"
fi

echo "[done] CSVs under tmp/perf/"
