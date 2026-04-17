#!/bin/bash
#
# Replicate the "movement lag inside the soldier" issue.
#
# Symptom: live FPS drops from 60 to ~40 when moving (WASD) or placing
# blocks inside dense voxel content. Rotation (looking around) stays
# at 60 FPS.
#
# Setup: spawn at the same coords the user reported from a live play
# session (camera lx=1.3889 ly=2.8333 lz=0.3889 at anchor_depth=7),
# then script a continuous WASD-style movement pattern via the
# `step:x+` / `step:x-` script commands.
#
# Output: a per-frame CSV trace at /tmp/lag-repro.csv. The CSV has
# pack_ms, upload_total_ms, tree_write_ms etc. per frame. Frames where
# the script fires a step command will show the cache-miss repack
# cost; wait frames show cache-hit cost (near-zero upload).
#
# On the original occupancy-stack-slim branch, step-frame pack_ms was
# 85-92 ms (10 FPS spike). The implicit-ribbon refactor removes the
# per-frame ribbon rebuild and shrinks update_tree to just the four
# storage-buffer uploads — measure the new step-frame pack_ms here.
#
# Usage: scripts/replicate_movement_lag.sh
#   (then inspect /tmp/lag-repro.csv)

set -e
cd "$(dirname "$0")/.."

# Always use debug build — matches what `scripts/dev.sh` runs live.
cargo build --bin deepspace-game 2>&1 | grep -E "(error|warning: unused)" || true

SCRIPT=""
for i in {1..16}; do
    SCRIPT="${SCRIPT}step:x+,wait:5,step:x-,wait:5,"
done
# Trim trailing comma
SCRIPT="${SCRIPT%,}"

./target/debug/deepspace-game --render-harness \
    --vox-model assets/vox/soldier_729.vxs --plain-layers 8 \
    --spawn-xyz 1.3889 2.8333 0.3889 --spawn-depth 7 \
    --disable-overlay --harness-width 600 --harness-height 400 \
    --script "wait:30,${SCRIPT}" \
    --perf-trace /tmp/lag-repro.csv --perf-trace-warmup 10 \
    --exit-after-frames 300 --timeout-secs 60 --suppress-startup-logs \
    2>&1 | tail -5

echo ""
echo "=== Summary of step vs wait frame costs ==="
python3 <<'PY'
import csv
with open('/tmp/lag-repro.csv') as f:
    rows = list(csv.DictReader(f))
step_packs = [float(r['pack_ms']) for r in rows if float(r['pack_ms']) > 1.0]
wait_packs = [float(r['pack_ms']) for r in rows if float(r['pack_ms']) <= 1.0 and int(r['frame']) > 30]
if step_packs:
    print(f"  step frames: count={len(step_packs)}  mean pack={sum(step_packs)/len(step_packs):.2f} ms  max={max(step_packs):.2f} ms")
if wait_packs:
    print(f"  wait frames: count={len(wait_packs)}  mean pack={sum(wait_packs)/len(wait_packs):.2f} ms")
worst = max(rows, key=lambda r: float(r['wall_ms']))
print(f"  worst frame: frame={worst['frame']} wall={float(worst['wall_ms']):.1f} ms upload={float(worst['upload_total_ms']):.1f} ms pack={float(worst['pack_ms']):.1f} ms")
PY
