#!/usr/bin/env bash
#
# Capture ~7 screenshots of the DemoSphere planet from distinct orbital
# positions. Output lands in `tmp/stage3b/` relative to the worktree
# root (never in system /tmp).
#
# Camera geometry:
#   The body lives at the CORE_SLOT (slot 13 = (1,1,1)) of the root
#   cell. Root coords are `[0, 3)³`. So the body cell is `[1, 2)³`
#   and the body center is at (1.5, 1.5, 1.5).
#
#   The body's outer shell in cell-local `[0, 1)` coords has r=0.45,
#   so in root coords the shell radius is 0.45. Cameras at distance
#   ~1.1 frame the whole planet; ~0.65 gives a close-up.
#
# Yaw/pitch convention (camera.rs): smoothed_up = +Y at spawn; tangent
# basis produces t_right=+X, t_fwd=-Z (fwd is "into the screen"). So
# forward = (-sin(yaw)*cos(pitch), sin(pitch), -cos(yaw)*cos(pitch)).
# yaw=0 looks toward -Z. pitch>0 looks up.
#
# Usage:
#   scripts/sphere-orbit.sh                   # default depth 8, 960x540
#   ORBIT_DEPTH=16 scripts/sphere-orbit.sh    # deeper SDF tree
#   ORBIT_WIDTH=1920 ORBIT_HEIGHT=1080 scripts/sphere-orbit.sh

set -euo pipefail

cd "$(dirname "$0")/.."

DEPTH="${ORBIT_DEPTH:-8}"
WIDTH="${ORBIT_WIDTH:-960}"
HEIGHT="${ORBIT_HEIGHT:-540}"
OUT_DIR="tmp/stage3b"

mkdir -p "$OUT_DIR"

# Build once up front.
cargo build --bin deepspace-game --quiet

# Each row: NAME SPAWN_X SPAWN_Y SPAWN_Z YAW PITCH
VIEWS=(
  # Above: cam at (1.5, 2.7, 1.5) looking almost straight down.
  # pitch=-1.4 (below horizon); yaw=0 (arbitrary near pole).
  "above         1.5 2.7 1.5   0.0      -1.4"
  # Below: cam at (1.5, 0.3, 1.5) looking almost straight up.
  "below         1.5 0.3 1.5   0.0       1.4"
  # Equatorial -Z side: cam at (1.5, 1.5, 2.7) looking toward -Z
  # (yaw=0, pitch=0 since default forward is -Z).
  "equator_pz    1.5 1.5 2.7   0.0       0.0"
  # Equatorial +X side: cam at (2.7, 1.5, 1.5) looking toward -X
  # (yaw=π/2).
  "equator_px    2.7 1.5 1.5   1.5708    0.0"
  # Isometric +X+Y+Z corner: cam at (2.6,2.6,2.6) toward center.
  # pitch=-0.615, yaw=π/4 (since fwd=(-0.577,-0.577,-0.577) with
  # our convention means sy=+0.707, cy=+0.707).
  "iso_pxyz      2.6 2.6 2.6   0.7854   -0.615"
  # Isometric -X-Y+Z corner: cam at (0.4,0.4,2.6) toward center.
  # fwd=(+0.577,+0.577,-0.577): pitch=+0.615, yaw=-π/4.
  "iso_nxyz      0.4 0.4 2.6  -0.7854    0.615"
  # Close-up along +Z: cam at (1.5, 1.5, 2.175), shell radius 0.45,
  # distance 0.675 ≈ 1.5× outer_r.
  "closeup_pz    1.5 1.5 2.175 0.0       0.0"
)

for row in "${VIEWS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  NAME="$1"; SX="$2"; SY="$3"; SZ="$4"; YAW="$5"; PITCH="$6"
  out="$OUT_DIR/${NAME}.png"
  printf 'capturing %-14s from (%s,%s,%s) yaw=%s pitch=%s -> %s\n' \
    "$NAME" "$SX" "$SY" "$SZ" "$YAW" "$PITCH" "$out"
  timeout 20 ./target/debug/deepspace-game \
    --render-harness \
    --disable-overlay \
    --disable-highlight \
    --sphere-world \
    --plain-layers "$DEPTH" \
    --spawn-xyz "$SX" "$SY" "$SZ" \
    --spawn-yaw "$YAW" \
    --spawn-pitch "$PITCH" \
    --harness-width "$WIDTH" \
    --harness-height "$HEIGHT" \
    --screenshot "$out" \
    --exit-after-frames 3 \
    --timeout-secs 10 \
    >"$OUT_DIR/${NAME}.log" 2>&1 || {
      echo "  (timed out or exited non-zero; see $OUT_DIR/${NAME}.log)"
    }
  if [[ ! -s "$out" ]]; then
    echo "  WARN: empty screenshot for $NAME" >&2
  fi
done

echo
echo "Screenshots written to $OUT_DIR/:"
ls -la "$OUT_DIR"/*.png 2>/dev/null || true
