#!/bin/bash
#
# Repro for the ray-march entity cutoff at chunk borders.
#
# Symptom: in `--entity-render ray-march` (the default), an entity's
# visible silhouette is progressively clipped as its sub-cell offset
# drifts away from zero. Raster mode (`--entity-render raster`) is
# unaffected. Over many ticks a soldier "slithers between blocks" —
# half-body visible, other half missing — because the shader only
# places an EntityRef cell at the entity's anchor slot, while the
# entity's bbox (= anchor_cell_size) extends into the +1 neighbor
# cell on any axis where offset > 0. Rays entering from the
# neighbor side never test the bbox → that portion of the entity
# is invisible.
#
# What this script does:
# - Spawn one soldier entity with a known deterministic velocity
#   (from `entity_velocity(0) = [0.0, 0.05, 0.30]`).
# - Render from a fixed camera angle looking at the entity.
# - Capture frames at N = {30, 60, 90, 120, 150} so you can watch
#   offset drift from ~0 to ~0.75 on the z-axis and see the bbox
#   clip more severely at each step.
# - Save PNGs under tmp/entity_cutoff/ and print the paths.
#
# Usage:
#   scripts/replicate_entity_cutoff.sh
#
# Optional env:
#   ENTITY_CUTOFF_MODE   — `ray-march` (default) or `raster`
#   ENTITY_CUTOFF_FRAMES — space-separated frame counts (default
#                          "30 60 90 120 150")
#   ENTITY_CUTOFF_DIR    — output directory (default tmp/entity_cutoff)

set -e
cd "$(dirname "$0")/.."

cargo build --bin deepspace-game 2>&1 | tail -1

MODE="${ENTITY_CUTOFF_MODE:-ray-march}"
FRAMES="${ENTITY_CUTOFF_FRAMES:-30 60 90 120 150}"
OUT="${ENTITY_CUTOFF_DIR:-tmp/entity_cutoff}"

rm -rf "$OUT"
mkdir -p "$OUT"

echo "=== Entity-cutoff repro (mode=$MODE) ==="
echo "Single entity, velocity=[0.00, 0.05, 0.30]. z-offset rolls over"
echo "at ~200 frames at 60fps; cutoff visible from ~frame 60 onward."
echo

for n in $FRAMES; do
    timeout 12 target/debug/deepspace-game \
        --render-harness \
        --disable-overlay --disable-highlight \
        --plain-world --plain-layers 10 \
        --spawn-depth 3 --spawn-xyz 1.5 1.0 2.8 \
        --spawn-yaw 0 --spawn-pitch -0.3 \
        --harness-width 320 --harness-height 180 \
        --exit-after-frames "$n" --timeout-secs 10 \
        --spawn-entity assets/vox/soldier.vox --spawn-entity-count 9 \
        --entity-render "$MODE" \
        --screenshot "$OUT/f${n}.png" \
        >/dev/null 2>&1
    printf "  frame=%-4s  %s\n" "$n" "$OUT/f${n}.png"
done

echo
echo "Open the PNGs side by side. In ray-march mode (default) the"
echo "soldier's silhouette shrinks from ~frame 60 onward as the"
echo "sub-cell z-offset drifts — that's the cutoff. A correct fix"
echo "leaves the silhouette intact at every frame count. Raster mode"
echo "has a different spawn-positioning path (scene overlay is skipped),"
echo "so it's not a useful visual control with these harness flags."
