#!/bin/bash
#
# Sphere break + raycast probe regression harness.
#
# User-reported symptom: raycast works fine in the Cartesian world but
# fails for the sphere starting "1 layer below surface" or at increased
# zoom resolutions — `break` either hits nothing, or hits but the
# render doesn't show the broken cell.
#
# Fast iteration strategy: one harness process per --spawn-depth,
# running the script
#     emit:pre → probe_down → break → wait:8 → probe_down → emit:post
# and then dumping the stderr lines this flow emits:
#     HARNESS_MARK label=pre/post  ui_layer=... anchor_depth=... frame=N
#     HARNESS_PROBE direction=down hit=true|false anchor=[...] ui_layer=...
#     HARNESS_EDIT  action=broke  anchor=[...] changed=true|false
# Three programmatic signals come out:
#     (a) changed  — did the tree actually mutate? (oracle for "break applied")
#     (b) pre-probe vs post-probe anchor — did raycast hit a different cell
#         after the break? (oracle for "render / raycast sees the edit")
#     (c) pre vs post hit bool              — does the probe still hit?
# The iteration is fast because we never grep pixels or read PNGs — all
# signals are short stderr lines; the whole sweep runs in a few seconds.
# PNGs are captured as tail-end artifacts for visual sanity when a
# depth fails.
#
# Usage:
#   scripts/sphere_break_probe.sh           # default sweep
#   DEPTHS="3 5 8" scripts/sphere_break_probe.sh
#
# Tunables (env):
#   SPAWN_XYZ   default "1.5 1.98 1.5" (right above the sphere's north pole)
#   SPAWN_PITCH default -1.5
#   SPAWN_YAW   default 0
#   DEPTHS      default "3 5 8 12 16 20 24 28"
#   WIDTH, HEIGHT  default 640 360 (screenshots are for eyeballs only)

set -e
cd "$(dirname "$0")/.."

SPAWN_XYZ="${SPAWN_XYZ:-1.5 1.98 1.5}"
SPAWN_PITCH="${SPAWN_PITCH:--1.5}"
SPAWN_YAW="${SPAWN_YAW:-0}"
DEPTHS="${DEPTHS:-3 5 8 12 16 20 24 28}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-360}"

SHOT_DIR="tmp/shot/sphere_break"
mkdir -p "$SHOT_DIR"

cargo build --bin deepspace-game 2>&1 | grep -E "error" || true

echo "=== sphere break + probe sweep ==="
echo "    cam xyz=$SPAWN_XYZ pitch=$SPAWN_PITCH yaw=$SPAWN_YAW"
echo "    depths: $DEPTHS"
echo "    resolution: ${WIDTH}x${HEIGHT}"
echo

printf "  %-4s %-8s %-9s %-46s %-46s %-46s\n" \
    "d" "edited" "pre→post" "pre_probe anchor" "edit anchor" "post_probe anchor"
printf "  %-4s %-8s %-9s %-46s %-46s %-46s\n" \
    "--" "------" "---------" "--------" "-----------" "----------"

for d in $DEPTHS; do
    pre_png="$SHOT_DIR/d${d}_pre.png"
    post_png="$SHOT_DIR/d${d}_post.png"
    rm -f "$pre_png" "$post_png"

    script="wait:10,emit:pre,screenshot:${pre_png},probe_down,break,wait:10,probe_down,screenshot:${post_png},emit:post"

    # shellcheck disable=SC2086
    out=$(timeout 25 ./target/debug/deepspace-game --render-harness \
        --sphere-world \
        --spawn-xyz $SPAWN_XYZ \
        --spawn-pitch "$SPAWN_PITCH" \
        --spawn-yaw "$SPAWN_YAW" \
        --spawn-depth "$d" \
        --disable-overlay --disable-highlight \
        --harness-width "$WIDTH" --harness-height "$HEIGHT" \
        --script "$script" \
        --exit-after-frames 80 --timeout-secs 20 \
        --suppress-startup-logs 2>&1)

    # Pull the structured lines we care about. printf '%s\n' is safer
    # than echo here since the output may contain backslashes.
    edited=$(printf '%s\n' "$out" | awk '
        /HARNESS_EDIT action=broke/ {
            # HARNESS_EDIT action=broke anchor=[...] changed=true ui_layer=N anchor_depth=M
            for (i=1;i<=NF;i++) if ($i ~ /^changed=/) { print substr($i,9); exit }
        }')
    if [ -z "$edited" ]; then edited="N/A"; fi

    edit_anchor=$(printf '%s\n' "$out" | awk '
        /HARNESS_EDIT action=broke/ {
            for (i=1;i<=NF;i++) if ($i ~ /^anchor=/) { print substr($i,8); exit }
        }')
    if [ -z "$edit_anchor" ]; then edit_anchor="(no edit)"; fi

    # Two HARNESS_PROBE lines: first is pre-break, second is post-break.
    probes=$(printf '%s\n' "$out" | grep "HARNESS_PROBE direction=down" || true)
    pre_hit="?"; pre_anchor="(no probe)"; pre_layer="?"
    post_hit="?"; post_anchor="(no probe)"; post_layer="?"

    parse_probe() {
        printf '%s\n' "$1" | awk '{
            hit=""; anchor=""; layer=""
            for (i=1;i<=NF;i++) {
                if ($i ~ /^hit=/)       hit=substr($i,5)
                if ($i ~ /^anchor=/)    anchor=substr($i,8)
                if ($i ~ /^ui_layer=/)  layer=substr($i,10)
            }
            print hit"|"anchor"|"layer
        }'
    }

    probe1=$(printf '%s\n' "$probes" | sed -n '1p')
    probe2=$(printf '%s\n' "$probes" | sed -n '2p')
    if [ -n "$probe1" ]; then
        IFS='|' read -r pre_hit pre_anchor pre_layer <<EOF
$(parse_probe "$probe1")
EOF
    fi
    if [ -n "$probe2" ]; then
        IFS='|' read -r post_hit post_anchor post_layer <<EOF
$(parse_probe "$probe2")
EOF
    fi

    pre_disp="${pre_anchor} (L${pre_layer} hit=${pre_hit})"
    post_disp="${post_anchor} (L${post_layer} hit=${post_hit})"

    # Indicator column: did probe anchor CHANGE after the break?
    if [ "$pre_anchor" = "$post_anchor" ]; then
        if [ "$edited" = "true" ]; then
            delta="SAME!"   # edit applied but probe unchanged — render/raycast stale
        else
            delta="same"
        fi
    else
        delta="changed"
    fi

    png_note="pre+post"
    [ -f "$pre_png"  ] || png_note="${png_note}-MISSING"
    [ -f "$post_png" ] || png_note="${png_note}-MISSING"

    printf "  %-4s %-8s %-9s %-46s %-46s %-46s\n" \
        "$d" "$edited" "$delta" "$pre_anchor" "$edit_anchor" "$post_anchor"
done

echo
echo "  PNGs → $SHOT_DIR/d<depth>_{pre,post}.png"
