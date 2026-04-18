#!/usr/bin/env bash
#
# One iteration of the LOD zoom-invariance sweep.
#
# For a given iteration tag, renders the menger fractal at 3 spawn
# depths (shallow / mid / deep zoom) from the *same* physical camera
# pose, extracts perf + visibility metrics from the harness, and
# saves a PNG per depth + a CSV row to `tmp/lod-sweep/<iter>.csv`.
#
# Compare the three PNGs: if zoom is ZOOM-INVARIANT, they should all
# look (nearly) identical. If the old LOD gate is active, the
# shallow-zoom render is coarser.
#
# Usage:
#   scripts/lod-iter.sh <iter-name>
#   scripts/lod-iter.sh baseline
#   scripts/lod-iter.sh focus8
#   scripts/lod-iter.sh deep-desired

set -euo pipefail

cd "$(dirname "$0")/.."

ITER="${1:?iteration tag required}"
DEPTHS=(3 8 17)
FRACTAL="menger"
OUT_DIR="tmp/lod-sweep"
CSV="$OUT_DIR/$ITER.csv"

mkdir -p "$OUT_DIR"

cargo build --bin deepspace-game --quiet

echo "iter,spawn_depth,submitted_done_avg_ms,gpu_pass_avg_ms,avg_steps,avg_descend,avg_lod_terminal,hit_fraction" > "$CSV"

for depth in "${DEPTHS[@]}"; do
  out_png="$OUT_DIR/${ITER}_d${depth}.png"
  log=$(mktemp)
  # NOTE: deliberately no --spawn-xyz. The bootstrap's baked spawn
  # is the real-game scenario; overriding xyz takes a different
  # code branch that skips auto-carve and therefore hid this class
  # of bug in earlier harness runs.
  timeout 8 ./target/debug/deepspace-game \
    --render-harness \
    --disable-overlay \
    --disable-highlight \
    --shader-stats \
    --"${FRACTAL}"-world \
    --plain-layers 20 \
    --spawn-depth "$depth" \
    --spawn-yaw 0.785 \
    --spawn-pitch -0.615 \
    --harness-width 960 \
    --harness-height 540 \
    --screenshot "$out_png" \
    --exit-after-frames 4 \
    --timeout-secs 6 \
    >"$log" 2>&1 || true

  # Extract from render_harness_timing avg_ms line (per-metric avg across frames)
  timing=$(grep -E '^render_harness_timing ' "$log" | tail -1 || true)
  submitted=$(echo "$timing" | grep -oE 'submitted_done=[0-9.]+' | cut -d= -f2 | head -1)
  # gpu_pass on Apple Silicon may be disabled; try to extract anyway
  gpu_pass=$(echo "$timing" | grep -oE 'gpu_pass=[0-9.]+' | cut -d= -f2 | head -1)

  # Extract from render_harness_shader line
  shader=$(grep -E '^render_harness_shader ' "$log" | tail -1 || true)
  avg_steps=$(echo "$shader" | grep -oE 'avg_steps=[0-9.]+' | cut -d= -f2 | head -1)
  avg_descend=$(echo "$shader" | grep -oE 'avg_descend=[0-9.]+' | cut -d= -f2 | head -1)
  avg_lod_terminal=$(echo "$shader" | grep -oE 'avg_lod_terminal=[0-9.]+' | cut -d= -f2 | head -1)
  hit_fraction=$(echo "$shader" | grep -oE 'hit_fraction=[0-9.]+' | cut -d= -f2 | head -1)

  echo "$ITER,$depth,${submitted:-NA},${gpu_pass:-NA},${avg_steps:-NA},${avg_descend:-NA},${avg_lod_terminal:-NA},${hit_fraction:-NA}" >> "$CSV"
  printf '  d=%-3d  submitted=%-6s  gpu_pass=%-6s  avg_steps=%-6s  avg_descend=%-6s  avg_lodterm=%-6s  hit_frac=%s\n' \
    "$depth" "${submitted:-NA}" "${gpu_pass:-NA}" "${avg_steps:-NA}" "${avg_descend:-NA}" "${avg_lod_terminal:-NA}" "${hit_fraction:-NA}"

  rm -f "$log"
done

echo
echo "CSV: $CSV"
echo "PNGs: $OUT_DIR/${ITER}_d{3,8,17}.png"
