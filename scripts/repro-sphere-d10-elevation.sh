#!/usr/bin/env bash
# Repro harness for close-range d=10 sphere rendering, parameterized
# by elevation (in anchor-cell units above the terrain surface).
#
# Differs from `repro-sphere-d10-bug.sh`:
#   - Camera positioned via `--spawn-elevation-cells <N>`, so the
#     distance above surface scales with anchor depth (SAME visual
#     altitude regardless of depth chosen).
#   - Camera is on the PosY face axis (x=1.5, z=1.5 implied via the
#     default raycast from (1.5, 2.0, 1.5)) — eliminates the
#     `pick_face` flip that causes the mode-4 stripe pattern in the
#     off-axis repro. Pure close-range rendering test.
#
# Output PNGs: tmp/elev_${cells}_m${mode}.png (initial) and
# tmp/elev_${cells}_m${mode}_placed.png (after place).
#
# Usage:
#   scripts/repro-sphere-d10-elevation.sh 50          # cells=50, all modes
#   scripts/repro-sphere-d10-elevation.sh 50 4        # cells=50, mode 4
#   scripts/repro-sphere-d10-elevation.sh 50 4 -0.3   # cells, mode, pitch
#
# Common elevations:
#   e=1     — camera AT terrain surface (often renders gray, close-
#             range sphere-DDA degenerates)
#   e=30    — ~terrain surface, slight above (tree stripes visible
#             in mode 4 even at face center)
#   e=1000  — camera 0.017 root above surface; walker LOD-terminates
#             at shallow ancestors; d=10 cells render as d=6-ish
#             representative — not the actual d=10 leaf

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${SKIP_BUILD:-}" ]]; then
    cargo build >&2
fi

BIN="./target/debug/deepspace-game"
mkdir -p tmp

CELLS="${1:-30}"
MODES=()
PITCH="-0.5"

if [[ $# -ge 2 ]]; then
    MODES=("$2")
fi
if [[ $# -ge 3 ]]; then
    PITCH="$3"
fi

if [[ ${#MODES[@]} -eq 0 ]]; then
    MODES=(0 1 2 3 4 5 6 7)
fi

CAM_ARGS=(
    --sphere-world
    --spawn-depth 10
    --spawn-elevation-cells "$CELLS"
    --spawn-pitch "$PITCH"
    --interaction-radius 10000
    --harness-width 600
    --harness-height 400
    --disable-overlay
    --timeout-secs 18
)

for m in "${MODES[@]}"; do
    echo "=== cells=$CELLS pitch=$PITCH mode=$m ==="
    timeout 20 "$BIN" "${CAM_ARGS[@]}" \
        --sphere-debug-mode "$m" \
        --screenshot "tmp/elev_${CELLS}_m${m}.png" \
        --script "wait:10,place,wait:30,screenshot:tmp/elev_${CELLS}_m${m}_placed.png" \
        --exit-after-frames 80 2>&1 \
        | grep -E "above_y|HARNESS_EDIT" \
        | head -2 \
        || true
done

echo ""
echo "Outputs: tmp/elev_${CELLS}_m*.png"
