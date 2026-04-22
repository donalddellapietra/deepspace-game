#!/usr/bin/env bash
#
# Deep-depth visual gate for the Stage 3d slot-path + residual face
# walker. Captures screenshots of the DemoSphere at progressively
# deeper `--spawn-depth` targeting the same surface point. The
# render-frame clamp (Stage 3c) keeps the shader at the body cell,
# so each capture should render the SAME sphere surface from the
# camera's perspective — if the precision model is correct, depth 5
# and depth 30 produce visually indistinguishable output.
#
# Follows the `repro-sphere-d10-bug.sh` pattern: explicit
# `--spawn-xyz` + `--spawn-depth`, no in-game zoom (there is none).
#
# Acceptance: all 6 PNGs must show a clean sphere surface (no
# warping, no axis-aligned-grid fallback, no shift, no striping,
# no wedge artifacts). Read each PNG and confirm visually; do NOT
# rely on `--exit-after-frames` exit code.
#
# Usage:
#   scripts/sphere-deep-depth-check.sh
#   DEEP_DEPTH_OUT=tmp/stage3d_shader scripts/sphere-deep-depth-check.sh
#
# Output: tmp/stage3d_shader/depth_NN.png  (NN = 05, 10, 15, 20, 25, 30).

set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${DEEP_DEPTH_OUT:-tmp/stage3d_shader}"
mkdir -p "$OUT_DIR"

# Build once up front.
cargo build --bin deepspace-game --quiet

# Camera pose: on the +Z side of the planet, looking straight at
# the surface along -Z. Planet surface at z≈1.95 for the +Z pole;
# camera at z=2.15 is 0.2 root-units of standoff (close-up). Zoom-
# invariance means the Stage 3c render-frame clamp keeps the shader
# at the body cell for ALL `--spawn-depth` values, so the visible
# scene is identical no matter how deep the anchor is.
#
# Task spec asked for xyz=(1.5, 1.7, 1.5), but y=1.7 is INSIDE the
# planet (surface at y=1.95) — both the Stage 3b baseline and this
# walker render gray (camera-inside-rock) for that config, which
# doesn't exercise the precision model. Using an outside-surface
# config instead lets the deep-depth capture actually show the
# rendered sphere.
SPAWN_X=1.5
SPAWN_Y=1.5
SPAWN_Z=2.15
SPAWN_PITCH=0.0
INTERACTION_RADIUS=10000

# Spawn depths covering the precision ladder: 5 (shallow, trivial),
# 10 (Stage 3b's cs_raycast wall), 15 (around the absolute-f32 wall),
# 20, 25, 30 (the 30-layer target). Zoom-invariance: render-frame
# clamp keeps the shader at the body cell, so visually all depths
# should render identically.
DEPTHS=(5 10 15 20 25 30)

for d in "${DEPTHS[@]}"; do
    out="$OUT_DIR/$(printf 'depth_%02d.png' "$d")"
    log="$OUT_DIR/$(printf 'depth_%02d.log' "$d")"
    printf 'capturing depth=%-2s -> %s\n' "$d" "$out"
    timeout 20 ./target/debug/deepspace-game \
        --render-harness \
        --disable-overlay \
        --disable-highlight \
        --sphere-world \
        --plain-layers 8 \
        --spawn-xyz "$SPAWN_X" "$SPAWN_Y" "$SPAWN_Z" \
        --spawn-depth "$d" \
        --spawn-pitch "$SPAWN_PITCH" \
        --interaction-radius "$INTERACTION_RADIUS" \
        --harness-width 600 \
        --harness-height 400 \
        --screenshot "$out" \
        --exit-after-frames 80 \
        --timeout-secs 18 \
        >"$log" 2>&1 || {
            echo "  (timed out or exited non-zero; see $log)"
        }
    if [[ ! -s "$out" ]]; then
        echo "  WARN: empty screenshot for depth=$d" >&2
    fi
done

echo
echo "Screenshots written to $OUT_DIR/:"
ls -la "$OUT_DIR"/*.png 2>/dev/null || true
