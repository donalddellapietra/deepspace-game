#!/usr/bin/env bash
# Reproduces the d=10 sphere "hollow block + striped ground" bug in the
# headless harness. Output PNGs go to `tmp/bug_mN.png` for mode N.
#
# This script uses the USER'S EXACT COORDS from a screenshot where the
# bug was captured in the live game. Those coords are:
#   - root_xyz       = (1.5, 1.7993, 1.4988)  — off-PosY-face-axis (z ≠ 1.5)
#   - anchor_depth   = 10 (layer 21)
#   - pitch          = -0.5
#
# The off-axis z (1.4988 not 1.5) matters: `pick_face(n)` flips between
# PosY and PosZ subtrees for adjacent rays, producing the mode-4 stripe
# pattern. Camera at EXACT face center (z = 1.5) makes the stripes
# disappear — use that variant (`repro-sphere-d10-elevation.sh`) to
# test close-range d=10 rendering without the face-boundary confound.
#
# `--interaction-radius 10000` lets the cursor raycast reach the
# surface from a distance the default 6-cell cap would block.
#
# Usage:
#   scripts/repro-sphere-d10-bug.sh            # cycles modes 0..7
#   scripts/repro-sphere-d10-bug.sh 4          # single mode
#   SKIP_BUILD=1 scripts/repro-sphere-d10-bug.sh   # skip cargo build

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${SKIP_BUILD:-}" ]]; then
    cargo build >&2
fi

BIN="./target/debug/deepspace-game"
mkdir -p tmp

MODES=("$@")
if [[ ${#MODES[@]} -eq 0 ]]; then
    MODES=(0 1 2 3 4 5 6 7)
fi

for m in "${MODES[@]}"; do
    echo "=== mode $m ==="
    timeout 20 "$BIN" \
        --sphere-world \
        --spawn-xyz 1.5 1.7993 1.4988 \
        --spawn-depth 10 \
        --spawn-pitch -0.5 \
        --interaction-radius 10000 \
        --sphere-debug-mode "$m" \
        --screenshot "tmp/bug_init_m${m}.png" \
        --script "wait:10,place,wait:30,screenshot:tmp/bug_m${m}.png" \
        --exit-after-frames 80 \
        --timeout-secs 18 \
        --harness-width 600 \
        --harness-height 400 \
        --disable-overlay 2>&1 \
        | grep -E "HARNESS_EDIT|render_harness_shader" \
        | head -2
done

echo ""
echo "Outputs:"
ls -la tmp/bug_m*.png 2>/dev/null || echo "  (no screenshots saved)"
