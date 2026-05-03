#!/usr/bin/env bash
#
# Reproduce the extreme-lag case on Jerusalem cross:
# camera at (1.5, 1.5, 1.5) — inside the body-centre nucleus rod —
# with anchor_depth 7 on a plain_layers=20 world. At native 2560×1440
# this hits ~62 ms avg submitted_done and ~91 ms worst-frame,
# matching the ~80 ms user report.
#
# The lazy diagonal spawn (2.8, 2.8, 2.8) at the same anchor is
# much cheaper (~12 ms) because the ray exits the sparse structure
# quickly. The (1.5, 1.5, 1.5) nucleus-interior spawn is the
# pathological case — every ray direction has to traverse the
# cross's 74 %-empty recursion before hitting solid.
#
# Use this when benchmarking LOD / empty-skip / ribbon-related
# changes: the numbers here tell you whether you actually reduced
# sparse-fractal cost or just shifted noise around.

set -euo pipefail

cd "$(dirname "$0")/.."

cargo build --bin deepspace-game --quiet

echo "=== Jerusalem cross: body-centre nucleus spawn, anchor 7 ==="
echo "    Expected worst-case: ~60-90 ms submitted_done at 2560x1440"
echo
timeout 15 ./target/debug/deepspace-game \
  --render-harness \
  --disable-overlay \
  --disable-highlight \
  --shader-stats \
  --jerusalem-cross-world \
  --plain-layers 20 \
  --spawn-xyz 1.5 1.5 1.5 \
  --spawn-depth 7 \
  --harness-width 2560 \
  --harness-height 1440 \
  --exit-after-frames 30 \
  --timeout-secs 12 \
  --suppress-startup-logs 2>&1 \
  | grep -E '^(render_harness_timing avg_ms|render_harness_shader|render_harness_worst)'

echo
echo "Key metrics to watch:"
echo "  submitted_done=N        - authoritative GPU-bound time on Metal"
echo "                            (submit -> on_submitted_work_done). Target < 30 ms."
echo "  avg_steps=N             - DDA steps per ray (currently ~136)"
echo "  avg_empty / avg_oob /   - cost decomposition: empty-cell advances,"
echo "    avg_descend             ribbon pops, tree descents"
echo "  hit_fraction=N          - fraction of rays that hit content (~0.12)"
