#!/usr/bin/env bash
# Like proto-screenshot.sh but BREAKS the proto target cell first
# so it becomes non-uniform (tag=2). Then screenshot. This exercises
# the v1 dispatch (cartesian_voxels_in_cell → march_cartesian on
# the cell's actual subtree) instead of the v0 solid-cube fallback.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-tmp/proto_break.png}"
mkdir -p "$(dirname "$OUT")"

# Script schedule starts at frame 30 (after GPU uploads settle);
# wait:5 → break at frame 35, wait:3 → screenshot at frame 38.
# --exit-after-frames must be > 38; using 45 for headroom.
# break — break the cell at the crosshair (= screen-center, which
# is the proto target cell from this camera position).
# Camera at root-y 1.575 puts it at WP-local y ≈ 2.18 (vs target
# cell surface at WP-local y ≈ 1.85); distance ≈ 0.33 in WP-local,
# well within the default 6-cell × (1/9 cell-size) = 0.67 reach.
#
# --force-edit-depth 8: anchor stays at depth 5 (so interaction
# range stays at normal scale), but `break` edits cells at depth 8
# (= 3 levels into the slab cell's anchor chain). The slab cell
# becomes non-uniform (tag=2) after the break — its subtree now
# has a hole at the broken sub-voxel — and the v1 dispatch
# (cartesian_voxels_in_cell → march_cartesian on the cell's
# subtree) fires, rendering the cell as actual Cartesian voxels
# (BLUE in the current diagnostic mode) instead of the v0 solid
# red box.
timeout 15 ./target/debug/deepspace-game \
    --wrapped-planet --planet-render-sphere \
    --spawn-xyz 1.506 1.575 1.464 \
    --spawn-yaw 0 --spawn-pitch -1.57 \
    --interaction-radius 12 \
    --force-edit-depth 8 \
    --disable-highlight \
    --render-harness \
    --script "wait:5,break,wait:3,screenshot:$OUT" \
    --exit-after-frames 45 \
    --timeout-secs 10 \
    --suppress-startup-logs 2>&1 | tail -3
