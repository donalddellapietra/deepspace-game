#!/usr/bin/env bash
# Capture a single sphere-world screenshot at a given spawn depth.
#
# Usage:
#   scripts/sphere_screenshot.sh <label> <depth> [extra cargo args...]
#
# Output:  tmp/<label>/depth-<N>.png
#
# Designed for the unified-DDA rewrite: invoke at each plan step to
# verify the sphere still renders. Camera pose mirrors
# sphere_zoom_seamless: looking straight down, surface at y=1.80,
# gap = 0.6 * 12 * (3 / 3^depth).
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <label> <depth> [extra args...]" >&2
  exit 2
fi

label="$1"
depth="$2"
shift 2

WORKTREE="$(git rev-parse --show-toplevel)"
cd "$WORKTREE"

outdir="$WORKTREE/tmp/$label"
mkdir -p "$outdir"
outpng="$outdir/depth-$depth.png"

# Camera pose: sphere centered at body-local (0.5,0.5,0.5) with
# outer_r=0.45 in body-local. Body cell at world slot 13 = world
# [1,2)³ → sphere center world (1.5,1.5,1.5), outer surface top at
# world y=1.95.
#
# (Earlier the base height was 1.80 — that put the camera INSIDE
# the shell at any depth ≥ ~7 because the "+gap" term shrinks below
# the 0.15-unit gap-to-actual-surface. The screenshots looked like
# the body march walling out at depth 9-10 when it was actually the
# camera being inside the shell looking at the inner stone surface
# from very close. Fixed to 1.95 + small static nudge so the camera
# is always JUST ABOVE the outer shell, with the depth-scaled gap
# preserved.)
cam_y=$(python3 - <<EOF
d = $depth
gap = 12.0 * (3.0 / (3.0 ** d)) * 0.6
print(f"{1.95 + 0.01 + gap:.6f}")
EOF
)

echo "[sphere_screenshot] label=$label depth=$depth cam_y=$cam_y out=$outpng" >&2

cargo run --quiet -- \
  --render-harness \
  --disable-overlay \
  --sphere-world \
  --spawn-depth "$depth" \
  --spawn-xyz 1.5 "$cam_y" 1.5 \
  --spawn-pitch -1.5707 \
  --spawn-yaw 0 \
  --harness-width 480 \
  --harness-height 320 \
  --exit-after-frames 60 \
  --timeout-secs 30 \
  --suppress-startup-logs \
  --screenshot "$outpng" \
  "$@"

if [[ ! -f "$outpng" ]]; then
  echo "[sphere_screenshot] ERROR: $outpng not produced" >&2
  exit 1
fi

echo "[sphere_screenshot] ok: $outpng ($(stat -f%z "$outpng") bytes)" >&2
