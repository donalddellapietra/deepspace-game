#!/bin/bash
#
# Cursor-detail regression harness.
#
# User-reported symptom: "you're on the surface of a very big block,
# right on it, but ~50% of the time the cursor highlight doesn't show,
# and when it does it only outlines the seams of the BIG collapsed
# block — not the small block you're actually touching."
#
# Design (post-rewrite):
#   - Raycast walker descends to tree_depth so the cursor ALWAYS finds
#     a concrete leaf cell, never bails out on an "empty" coarse
#     representative.
#   - Hit path is truncated to `edit_depth` (= anchor_depth) before
#     the cursor highlight / break / place fires — user edits at
#     their current layer.
#   - Interaction radius = 12 anchor cells (capped reach, scales with
#     zoom).
#
# Expectation in this harness:
#   probe_depth == anchor_depth (the truncated layer-N path length),
#   hit=true whenever the surface is within 12 anchor cells.
#
# Fast iteration: one binary invocation per test case, probe_down in
# script, parse `HARNESS_PROBE` stdout. No PNG reads.
#
# Usage:
#   scripts/cursor_detail_probe.sh                # both world presets
#   WORLDS="sphere" scripts/cursor_detail_probe.sh
#
# Env:
#   WORLDS        "plain sphere" by default (space-separated)
#   DEPTHS        anchor depths to sweep
#   PLAIN_LAYERS  plain-world tree depth (default 20)

set -e
cd "$(dirname "$0")/.."

WORLDS="${WORLDS:-plain sphere}"
DEPTHS="${DEPTHS:-3 5 8 12 16 20}"
PLAIN_LAYERS="${PLAIN_LAYERS:-20}"

cargo build --bin deepspace-game 2>&1 | grep -E "error" || true

run_probe() {
    local world=$1
    local depth=$2
    local spawn_pitch=$3
    local world_args=$4
    local tree_depth=$5

    # `--spawn-on-surface` dispatches on world preset to a
    # path-based spawn that tracks the surface at any depth (plain
    # → dirt/grass boundary, sphere → SDF terrain at the north pole).
    # Avoids the f32-quantization failure mode of `--spawn-xyz` at
    # deep zooms.

    # shellcheck disable=SC2086
    out=$(timeout 20 ./target/debug/deepspace-game --render-harness \
        $world_args \
        --spawn-on-surface \
        --spawn-pitch "$spawn_pitch" \
        --spawn-yaw 0 \
        --spawn-depth "$depth" \
        --disable-overlay --disable-highlight \
        --harness-width 320 --harness-height 180 \
        --script 'wait:10,probe_down' \
        --exit-after-frames 60 --timeout-secs 15 \
        --suppress-startup-logs 2>/dev/null)

    # Parse HARNESS_PROBE stdout line.
    line=$(printf '%s\n' "$out" | grep "HARNESS_PROBE direction=down" | head -1)
    hit=$(printf '%s\n' "$line" | awk '{for(i=1;i<=NF;i++) if($i~/^hit=/) print substr($i,5)}')
    anchor=$(printf '%s\n' "$line" | awk '{for(i=1;i<=NF;i++) if($i~/^anchor=/) print substr($i,8)}')

    # Strip "[..]" and count commas+1 to get path depth; handle empty.
    if [ "$anchor" = "[]" ] || [ -z "$anchor" ]; then
        probe_depth=0
    else
        # count slots by commas
        inner=${anchor#[}
        inner=${inner%]}
        probe_depth=$(printf '%s\n' "$inner" | awk -F',' '{print NF}')
    fi

    # Report PASS/FAIL versus the expectation:
    # probe_depth should equal anchor_depth AND hit should be true
    # (the walker found the surface and the hit is within interaction
    # radius). Anything else is the bug.
    if [ "$hit" = "true" ] && [ "$probe_depth" = "$depth" ]; then
        status="ok"
    else
        status="FAIL"
    fi
    printf "    %-5s  d=%-3s  hit=%-5s  probe_depth=%-3s  (expected=%s)  anchor=%s\n" \
        "$status" "$depth" "$hit" "$probe_depth" "$depth" "$anchor"
}

echo "=== cursor-detail probe sweep ==="
echo "    Expectation: probe_depth == anchor_depth (walker found a hit,"
echo "    then the truncate-to-edit_depth step left a layer-N path). A"
echo "    'hit=false' or 'probe_depth < anchor_depth' means the walker"
echo "    missed the surface OR the interaction-radius gate rejected it."
echo

for world in $WORLDS; do
    case "$world" in
        plain)
            echo "-- plain world, $PLAIN_LAYERS layers --"
            for d in $DEPTHS; do
                run_probe "plain" "$d" "-1.5707963" \
                    "--plain-world --plain-layers $PLAIN_LAYERS" \
                    "$PLAIN_LAYERS"
            done
            ;;
        sphere)
            echo "-- sphere world (tree_depth=30) --"
            for d in $DEPTHS; do
                run_probe "sphere" "$d" "-1.5707963" \
                    "--sphere-world" "30"
            done
            ;;
    esac
    echo
done
