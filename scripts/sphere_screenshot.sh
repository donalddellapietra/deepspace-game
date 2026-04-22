#!/usr/bin/env bash
# Screenshot the shell-only sphere from outside (silhouette view) and
# zoomed into one shell block (Cartesian voxel content view).
#
# Outputs:
#   tmp/shell_outside.png  — sphere silhouette from above
#   tmp/shell_zoomed.png   — close-up showing shell-block content
#
# Usage:
#   scripts/sphere_screenshot.sh
#   SKIP_BUILD=1 scripts/sphere_screenshot.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${SKIP_BUILD:-}" ]]; then
    cargo build >&2
fi

BIN="./target/debug/deepspace-game"
mkdir -p tmp

# View 1: from outside, well above the sphere — sphere silhouette
# visible in the lower portion of the frame.
echo "=== sphere from outside ==="
timeout 40 "$BIN" \
    --sphere-world \
    --spawn-xyz 1.5 2.5 1.5 --spawn-depth 1 --spawn-pitch -1.2 \
    --interaction-radius 10000 \
    --harness-width 600 --harness-height 400 \
    --disable-overlay \
    --timeout-secs 30 \
    --screenshot tmp/shell_outside.png \
    --exit-after-frames 30 \
    2>&1 | grep -E "render_harness_timing|Demo sphere" | head -3 \
    || true

# View 2: zoomed into a shell block — at deep depth, near the surface.
echo "=== zoomed into shell block ==="
timeout 40 "$BIN" \
    --sphere-world \
    --spawn-depth 12 --spawn-elevation-cells 3 --spawn-pitch -0.4 \
    --interaction-radius 10000 \
    --harness-width 600 --harness-height 400 \
    --disable-overlay \
    --timeout-secs 30 \
    --screenshot tmp/shell_zoomed.png \
    --exit-after-frames 30 \
    2>&1 | grep -E "render_harness_timing|Demo sphere" | head -3 \
    || true

echo ""
echo "Outputs: tmp/shell_outside.png, tmp/shell_zoomed.png"
