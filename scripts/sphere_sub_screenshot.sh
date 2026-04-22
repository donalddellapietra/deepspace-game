#!/usr/bin/env bash
# Capture a sphere-world screenshot with the SphereSub path active
# (camera promoted to sphere-anchored state via force_sphere_state).
#
# Usage:
#   scripts/sphere_sub_screenshot.sh <label> <depth>
#
# Output:  tmp/<label>/sub-depth-<N>.png
#
# Without force_sphere_state, the freshly-spawned harness camera has
# no UVR descent, so compute_render_frame always falls back to Body
# kind and the GPU's SphereSub branch (sphere_in_sub_frame) never
# fires. This script enables it explicitly.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <label> <depth>" >&2
  exit 2
fi

label="$1"
depth="$2"

WORKTREE="$(git rev-parse --show-toplevel)"
cd "$WORKTREE"

outdir="$WORKTREE/tmp/$label"
mkdir -p "$outdir"
outpng="$outdir/sub-depth-$depth.png"

cam_y=$(python3 - <<EOF
d = $depth
gap = 12.0 * (3.0 / (3.0 ** d)) * 0.6
print(f"{1.80 + gap:.6f}")
EOF
)

echo "[sphere_sub_screenshot] label=$label depth=$depth cam_y=$cam_y out=$outpng" >&2

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
  --script "force_sphere_state,wait:5,screenshot:$outpng"

[[ -f "$outpng" ]] || { echo "ERROR: $outpng not produced" >&2; exit 1; }
echo "[sphere_sub_screenshot] ok: $outpng ($(stat -f%z "$outpng") bytes)" >&2
