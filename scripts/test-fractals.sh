#!/usr/bin/env bash
#
# Capture a reference screenshot of every fractal preset via the
# render harness. Outputs land in `tmp/fractals/` relative to the
# repo root so they never leak into system /tmp.
#
# Usage:
#   scripts/test-fractals.sh                 # depth 5, 960x540 each
#   FRACTAL_DEPTH=8 scripts/test-fractals.sh # override depth
#   FRACTAL_WIDTH=1920 FRACTAL_HEIGHT=1080 scripts/test-fractals.sh
#
# Framing:
#   All 5 presets share a "far diagonal corner" pose
#   (spawn_xyz=2.8,2.8,2.8, yaw=π/4, pitch≈-arcsin(1/√3)) which points
#   the camera from the (+X,+Y,+Z) corner back toward the root-cell
#   origin. This shows each fractal's 3D silhouette clearly and
#   matches the baked-in bootstrap defaults, so
#   `scripts/dev.sh -- --menger-world` lands in the same framing.
#
# Depth 5 is the sweet spot for visual verification: at depth 8 the
# 3^-8 sub-cells average down into muddy LOD tones; at depth 5 each
# structural-role block type reads as a distinct color.

set -euo pipefail

cd "$(dirname "$0")/.."

DEPTH="${FRACTAL_DEPTH:-5}"
WIDTH="${FRACTAL_WIDTH:-960}"
HEIGHT="${FRACTAL_HEIGHT:-540}"
OUT_DIR="tmp/fractals"

# Far-diagonal pose: from +X+Y+Z corner of the root cell looking back
# along the body diagonal to (0,0,0). yaw=π/4 (≈0.785), pitch=-arcsin(1/√3)
# (≈-0.615). Verified in ScheduleWakeup-free iterative harness runs.
SPAWN_ARGS=(
  --spawn-xyz 2.8 2.8 2.8
  --spawn-yaw 0.785
  --spawn-pitch -0.615
)

FRACTALS=(
  menger
  sierpinski-tet
  cantor-dust
  jerusalem-cross
  sierpinski-pyramid
)

mkdir -p "$OUT_DIR"

# Ensure the binary is current. Build once up front so the per-fractal
# invocations don't each pay a compile check.
cargo build --bin deepspace-game --quiet

for kind in "${FRACTALS[@]}"; do
  out="$OUT_DIR/${kind}.png"
  printf 'capturing %-20s -> %s\n' "$kind" "$out"
  timeout 8 ./target/debug/deepspace-game \
    --render-harness \
    --disable-overlay \
    --disable-highlight \
    --"${kind}"-world \
    --plain-layers "$DEPTH" \
    "${SPAWN_ARGS[@]}" \
    --harness-width "$WIDTH" \
    --harness-height "$HEIGHT" \
    --screenshot "$out" \
    --exit-after-frames 2 \
    --timeout-secs 5 \
    >/dev/null 2>&1
  if [[ ! -s "$out" ]]; then
    echo "  WARN: empty screenshot for $kind" >&2
  fi
done

echo
echo "Screenshots written to $OUT_DIR/:"
ls -la "$OUT_DIR"/
