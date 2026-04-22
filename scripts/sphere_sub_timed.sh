#!/usr/bin/env bash
# Capture a sphere-sub screenshot AND emit the render_harness_timing
# line from the harness's stderr log. Use to compare render time
# across shader changes.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <label> <depth>" >&2
  exit 2
fi

label="$1" depth="$2"
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

stderr_log="$outdir/sub-depth-$depth.log"
cargo run --quiet -- \
  --render-harness --disable-overlay --sphere-world \
  --spawn-depth "$depth" \
  --spawn-xyz 1.5 "$cam_y" 1.5 \
  --spawn-pitch -1.5707 --spawn-yaw 0 \
  --harness-width 480 --harness-height 320 \
  --exit-after-frames 60 --timeout-secs 30 \
  --suppress-startup-logs \
  --script "force_sphere_state,wait:5,screenshot:$outpng" 2>"$stderr_log" >/dev/null

# Extract the most interesting timing lines.
grep -E "render_harness_timing|render_harness_worst" "$stderr_log" | head -2
