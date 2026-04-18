#!/bin/bash
#
# Cursor-detail regression harness.
#
# User-reported symptom: "you're on the surface of a very big block,
# right on it, but ~50% of the time the cursor highlight doesn't show,
# and when it does it only outlines the seams of the BIG collapsed
# block — not the small block you're actually touching."
#
# Root cause hypothesis: CPU-side LOD. `cs_edit_depth() == anchor_depth`
# so the raycast walker stops descending at the current zoom level and
# returns a coarse "representative block" instead of walking to the
# real leaf cell. At a shallow anchor, that terminal IS the big block.
#
# Fast iteration: one binary invocation per test case, probe_down in
# script, parse `HARNESS_PROBE` stdout. No PNG reads. Each run emits a
# single line: "(preset, depth, camera) → probe_depth (expected: X)".
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
    local spawn_xyz=$3
    local spawn_pitch=$4
    local world_args=$5
    local tree_depth=$6

    # shellcheck disable=SC2086
    out=$(timeout 20 ./target/debug/deepspace-game --render-harness \
        $world_args \
        --spawn-xyz $spawn_xyz \
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

    printf "    d=%-3s  hit=%-5s  probe_depth=%-3s  (tree_depth=%s)  anchor=%s\n" \
        "$depth" "$hit" "$probe_depth" "$tree_depth" "$anchor"
}

echo "=== cursor-detail probe sweep ==="
echo "    Expectation: probe_depth should be ~= tree_depth (probe reaches"
echo "    the actual leaf cell). If it equals the anchor depth, CPU LOD is"
echo "    stopping the walker at zoom granularity — the user-reported bug."
echo

for world in $WORLDS; do
    case "$world" in
        plain)
            echo "-- plain world, $PLAIN_LAYERS layers, camera at dirt/grass boundary --"
            # Plain world has a dirt/grass boundary at y≈0.95. Put camera
            # one cell above the surface at every anchor.
            for d in $DEPTHS; do
                run_probe "plain" "$d" \
                    "1.5 0.96 1.5" "-1.5707963" \
                    "--plain-world --plain-layers $PLAIN_LAYERS" \
                    "$PLAIN_LAYERS"
            done
            ;;
        sphere)
            echo "-- sphere world (tree_depth=30), camera above north pole --"
            # Sphere outer shell top at y=1.95. Camera 0.03 above.
            for d in $DEPTHS; do
                run_probe "sphere" "$d" \
                    "1.5 1.98 1.5" "-1.5707963" \
                    "--sphere-world" \
                    "30"
            done
            ;;
    esac
    echo
done
