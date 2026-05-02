#!/usr/bin/env bash
# Take a screenshot of the wrapped-planet with the harness camera
# positioned over the prototype target cell (slab grid (7, 1, 11)).
# Reliable, self-terminating, no bash-timeout games.
#
# Usage:
#   scripts/proto-screenshot.sh OUT.png
#
# The cell sits at lat≈0.81, lon≈-1.40 (radians), r=r_outer. Sphere
# surface point in WP-local = (1.556, 1.845, 1.176). Mapped to
# root-local (= --spawn-xyz coord space) via the WP at root path
# [13, 13]: WP frame [0, 3)³ scales to root-local [1.333, 1.667)³.
#
# CAMERA MUST BE INSIDE THE WP CELL ([1.333, 1.667)³) so the
# render frame's compute_render_frame descent reaches WP-path
# [13, 13] and dispatches sphere_uv_in_cell. Outside the WP, the
# descent diverges (slot[1] != 13) and the dispatch falls back to
# march_cartesian, where my prototype check (in sphere_uv_in_cell)
# never fires — the screenshot then shows march_cartesian's view
# of the WP's slab cells as axis-aligned cubes inside the WP frame
# (looks like a green tiled rectangle, not the sphere).
#
# Surface point of the target cell in root-local = (1.506, 1.538,
# 1.464). Camera at root-local y = 1.65 (inside WP at [1.333,
# 1.667), above surface y=1.538) looking straight down.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-tmp/proto.png}"
mkdir -p "$(dirname "$OUT")"

# --timeout-secs 5: harness self-terminates if it stalls.
# bash timeout 12: outer guard in case the harness itself wedges.
# --exit-after-frames 3: harness exits cleanly after 3 frames.
timeout 12 ./target/debug/deepspace-game \
    --wrapped-planet --planet-render-sphere \
    --spawn-xyz 1.506 1.575 1.464 \
    --spawn-yaw 0 --spawn-pitch -1.57 \
    --interaction-radius 12 \
    --disable-highlight \
    --render-harness \
    --screenshot "$OUT" \
    --exit-after-frames 3 \
    --timeout-secs 5 \
    --suppress-startup-logs 2>&1 | tail -3
