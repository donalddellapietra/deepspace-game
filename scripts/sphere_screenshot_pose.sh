#!/usr/bin/env bash
# Variant of sphere_screenshot.sh with explicit camera pose control.
#
# Usage:
#   scripts/sphere_screenshot_pose.sh <label> <depth> <x> <y> <z> <pitch> <yaw>
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "usage: $0 <label> <depth> <x> <y> <z> <pitch> <yaw>" >&2
  exit 2
fi

label="$1" depth="$2" x="$3" y="$4" z="$5" pitch="$6" yaw="$7"
WORKTREE="$(git rev-parse --show-toplevel)"
cd "$WORKTREE"
outdir="$WORKTREE/tmp/$label"
mkdir -p "$outdir"
outpng="$outdir/depth-$depth.png"

echo "[sphere_screenshot_pose] label=$label d=$depth xyz=$x,$y,$z p=$pitch y=$yaw" >&2
cargo run --quiet -- \
  --render-harness --disable-overlay --sphere-world \
  --spawn-depth "$depth" \
  --spawn-xyz "$x" "$y" "$z" \
  --spawn-pitch "$pitch" --spawn-yaw "$yaw" \
  --harness-width 480 --harness-height 320 \
  --exit-after-frames 60 --timeout-secs 30 \
  --suppress-startup-logs \
  --screenshot "$outpng"
[[ -f "$outpng" ]] || { echo "ERROR: $outpng not produced" >&2; exit 1; }
echo "[sphere_screenshot_pose] ok: $outpng" >&2
