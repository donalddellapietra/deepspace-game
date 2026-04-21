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

# Camera pose: sphere centered at 1.5, surface at r=0.30 → top at 1.80.
# Gap shrinks with depth so the interaction envelope always reaches
# the surface. Match sphere_zoom_seamless exactly.
cam_y=$(python3 - <<EOF
d = $depth
gap = 12.0 * (3.0 / (3.0 ** d)) * 0.6
print(f"{1.80 + gap:.6f}")
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
