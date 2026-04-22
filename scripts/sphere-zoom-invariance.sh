#!/usr/bin/env bash
#
# Capture screenshots of the DemoSphere at a sequence of progressively
# closer camera positions, all looking at the same surface point on
# the planet. The rendered silhouette / texture should stay visually
# continuous — no sudden geometry shift when the camera crosses the
# anchor-depth threshold that moves render_frame into the body cell.
#
# Body geometry:
#   The body lives at CORE_SLOT (slot 13 = (1,1,1)) of the root cell.
#   Root coords `[0, 3)³`, body cell `[1, 2)³`, center (1.5, 1.5, 1.5).
#   Outer shell radius = 0.45 (cell-local) = 0.45 in root coords, so
#   the planet surface on the +Z side is at z ≈ 1.95.
#
# The camera sits on the +Z side of the planet (X=1.5, Y=1.5, Z=var),
# always looking toward -Z (yaw=0, pitch=0). As Z decreases from 2.7
# (far) to 1.955 (touching surface), we sweep through the exact
# anchor-depth range where the old code broke (render_frame descending
# past the body into the face subtree).
#
# Output:
#   tmp/stage3c/{TAG}_zoom_{N}_{NAME}.png
# where TAG is the caller-supplied label (typically "before" / "after")
# so both passes survive side-by-side for comparison.
#
# Usage:
#   scripts/sphere-zoom-invariance.sh after       # tag outputs "after"
#   scripts/sphere-zoom-invariance.sh before      # tag outputs "before"

set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${1:-after}"
DEPTH="${ZOOM_DEPTH:-8}"
WIDTH="${ZOOM_WIDTH:-960}"
HEIGHT="${ZOOM_HEIGHT:-540}"
OUT_DIR="tmp/stage3c"

mkdir -p "$OUT_DIR"

# Build once up front.
cargo build --bin deepspace-game --quiet

# Each row: NAME SPAWN_X SPAWN_Y SPAWN_Z SPAWN_DEPTH
# All views use yaw=0, pitch=0 (looking toward -Z).
# Camera starts on +Z side and walks inward toward the planet.
# The "spawn_depth" moves with the zoom-in: at shallow depth the
# render frame is rooted high in the tree (root cell); at deep
# depth the camera's anchor path passes the body cell, which used
# to push render_frame into the face subtree and break geometry.
VIEWS=(
  # 1. Very far: 0.75 units from surface, shallow anchor.
  "zoom_1_veryfar   1.5 1.5 2.70   3"
  # 2. Far: 0.50 units from surface, still outside body cell.
  "zoom_2_far       1.5 1.5 2.45   4"
  # 3. Medium: 0.15 units, just outside body cell (z=2.0 boundary).
  "zoom_3_medium    1.5 1.5 2.15   6"
  # 4. Close: 0.10 units, INSIDE body cell now. Old code: broken.
  "zoom_4_close     1.5 1.5 2.05   8"
  # 5. Very close: 0.05 units from outer shell, deep inside body cell.
  "zoom_5_veryclose 1.5 1.5 2.00  10"
  # 6. Touching: right at the surface, deep anchor — the worst case
  # for the old code (render_frame several levels into the face subtree).
  "zoom_6_touching  1.5 1.5 1.975 12"
)

for row in "${VIEWS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  NAME="$1"; SX="$2"; SY="$3"; SZ="$4"; SD="$5"
  out="$OUT_DIR/${TAG}_${NAME}.png"
  printf 'capturing %-22s from (%s,%s,%s) depth=%s -> %s\n' \
    "$NAME" "$SX" "$SY" "$SZ" "$SD" "$out"
  timeout 20 ./target/debug/deepspace-game \
    --render-harness \
    --disable-overlay \
    --disable-highlight \
    --sphere-world \
    --plain-layers "$DEPTH" \
    --spawn-xyz "$SX" "$SY" "$SZ" \
    --spawn-depth "$SD" \
    --spawn-yaw 0.0 \
    --spawn-pitch 0.0 \
    --harness-width "$WIDTH" \
    --harness-height "$HEIGHT" \
    --screenshot "$out" \
    --exit-after-frames 3 \
    --timeout-secs 10 \
    >"$OUT_DIR/${TAG}_${NAME}.log" 2>&1 || {
      echo "  (timed out or exited non-zero; see $OUT_DIR/${TAG}_${NAME}.log)"
    }
  if [[ ! -s "$out" ]]; then
    echo "  WARN: empty screenshot for $NAME" >&2
  fi
done

echo
echo "Screenshots written to $OUT_DIR/ (tag=$TAG):"
ls -la "$OUT_DIR"/${TAG}_*.png 2>/dev/null || true
