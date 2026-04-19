#!/usr/bin/env bash
# Capture PNG screenshots of the soldier test scene under both entity
# render modes at 1 / 100 / 10000 count, into <worktree>/tmp/. Useful
# for eyeball-diff'ing the raster path against the ray-march baseline
# after a change.
#
# Usage: scripts/visual-entity-raster.sh

set -euo pipefail

WORKTREE="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="${WORKTREE}/target/release/deepspace-game"
SOLDIER="${WORKTREE}/assets/vox/soldier.vox"
TMP="${WORKTREE}/tmp"
mkdir -p "$TMP"

cargo build --release --bin deepspace-game --quiet

for mode in ray-march raster; do
  for count in 1 100 10000; do
    out="${TMP}/entity_${mode}_${count}.png"
    timeout 60 "$BINARY" \
      --render-harness --disable-overlay --disable-highlight \
      --plain-world --plain-layers 40 \
      --spawn-depth 6 --spawn-xyz 1.5 1.5 1.8 \
      --spawn-yaw 0 --spawn-pitch 0 \
      --spawn-entity "$SOLDIER" --spawn-entity-count "$count" \
      --entity-render "$mode" \
      --harness-width 640 --harness-height 360 \
      --exit-after-frames 60 --timeout-secs 55 \
      --screenshot "$out" \
      >/dev/null 2>&1 || echo "FAILED: $mode $count"
    echo "wrote $out"
  done
done
