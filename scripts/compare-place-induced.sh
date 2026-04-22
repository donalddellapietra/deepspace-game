#!/usr/bin/env bash
# Capture sphere debug modes BEFORE a d=10 block is placed and AFTER,
# side-by-side. Writes `tmp/before_mN.png` and `tmp/after_mN.png` for
# each mode.
#
# Purpose: the bug visually manifests ONLY after a place. Walker state
# before (uniform ground, single-plane winner) vs after (striped
# alternating winners) tells us whether the change is:
#   - in walker termination depth (mode 2 diff),
#   - in walker content-vs-empty result (mode 3 diff),
#   - in winning-plane stability (mode 4 diff),
#   - in ratio of landed cell (mode 6 diff),
# without having to re-navigate between places.
#
# Usage:
#   scripts/compare-place-induced.sh                  # all modes 0..6
#   scripts/compare-place-induced.sh 3 4              # selected modes
#   SKIP_BUILD=1 scripts/compare-place-induced.sh 4   # iterate after edit

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${SKIP_BUILD:-}" ]]; then
    cargo build >&2
fi

BIN="./target/debug/deepspace-game"
mkdir -p tmp

MODES=("$@")
if [[ ${#MODES[@]} -eq 0 ]]; then
    MODES=(0 1 2 3 4 5 6)
fi

CAM_ARGS=(
    --sphere-world
    --spawn-xyz 1.5 1.7993 1.4988
    --spawn-depth 10
    --spawn-pitch -0.5
    --interaction-radius 10000
    --harness-width 600
    --harness-height 400
    --disable-overlay
    --timeout-secs 18
)

for m in "${MODES[@]}"; do
    echo "=== mode $m ==="

    # BEFORE: render, no script, exit. Clean scene.
    timeout 20 "$BIN" "${CAM_ARGS[@]}" \
        --sphere-debug-mode "$m" \
        --screenshot "tmp/before_m${m}.png" \
        --exit-after-frames 30 2>&1 \
        | grep -E "HARNESS_EDIT|render_harness_shader" \
        | head -1 || true

    # AFTER: render, place a d=10 block, wait, screenshot.
    timeout 20 "$BIN" "${CAM_ARGS[@]}" \
        --sphere-debug-mode "$m" \
        --screenshot "tmp/after_m${m}_tail.png" \
        --script "wait:10,place,wait:30,screenshot:tmp/after_m${m}.png" \
        --exit-after-frames 80 2>&1 \
        | grep -E "HARNESS_EDIT|render_harness_shader" \
        | head -1 || true
done

echo ""
echo "Before outputs:"
ls -la tmp/before_m*.png 2>/dev/null | awk '{print "  " $NF}'
echo "After outputs:"
ls -la tmp/after_m*.png 2>/dev/null | grep -v tail | awk '{print "  " $NF}'
