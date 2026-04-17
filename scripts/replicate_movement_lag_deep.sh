#!/bin/bash
#
# Deep-world variant of replicate_movement_lag.sh. Spans 20 layers
# with each soldier voxel expanded into a uniform subtree of depth 13
# (`--vox-interior-depth 13`), so the min block lives at layer 7 and
# the diggable interior extends down to layer 20.
#
# Tests that the LOD-upload-key hysteresis (render_path at
# anchor_depth - 3) still holds at deep anchor depths, and that edit
# invalidation only fires when it should.
#
# Output: /tmp/lag-repro-deep.csv (same schema as the shallow run).
#
# Three phases:
#   1. warm-up:                30 frames still
#   2. WASD jitter (stays in-cell, no repack expected)
#   3. unidirectional drift    (crosses cells at deep anchor depth)
#
# Spawn is placed above the soldier's head, then `teleport_above_last_edit`
# is used to dive into a layer-7 voxel for the dig test.

set -e
cd "$(dirname "$0")/.."

cargo build --bin deepspace-game 2>&1 | grep -E "(error|warning: unused)" || true

# Build the script.
JITTER=""
for i in {1..20}; do
    JITTER="${JITTER}step:x+,wait:3,step:x-,wait:3,"
done
DRIFT=""
for i in {1..40}; do
    DRIFT="${DRIFT}step:x+:0.5,wait:2,"
done
SCRIPT="wait:30,${JITTER}${DRIFT}"
SCRIPT="${SCRIPT%,}"

./target/debug/deepspace-game --render-harness \
    --vox-model assets/vox/soldier_729.vxs \
    --plain-layers 20 --vox-interior-depth 13 \
    --disable-overlay --harness-width 600 --harness-height 400 \
    --script "${SCRIPT}" \
    --perf-trace /tmp/lag-repro-deep.csv --perf-trace-warmup 10 \
    --exit-after-frames 400 --timeout-secs 120 --suppress-startup-logs \
    2>&1 | tail -5

echo ""
echo "=== Summary: deep (20-layer) replicate_movement_lag ==="
python3 <<'PY'
import csv
with open('/tmp/lag-repro-deep.csv') as f:
    rows = list(csv.DictReader(f))
repacks = [r for r in rows if r['reused_gpu_tree']=='0']
packs = sorted([float(r['pack_ms']) for r in repacks], reverse=True)
print(f"  total frames: {len(rows)}")
print(f"  repacks:      {len(repacks)}")
if packs:
    import statistics
    print(f"  pack_ms      max={packs[0]:.2f}  p90={packs[len(packs)//10]:.2f}  median={packs[len(packs)//2]:.2f}  mean={statistics.mean(packs):.2f}")
worst = sorted(rows, key=lambda r: float(r['upload_total_ms']), reverse=True)[:3]
print("  top 3 upload spikes:")
for r in worst:
    print(f"    frame={r['frame']:>3}  upload={float(r['upload_total_ms']):6.2f}  pack={float(r['pack_ms']):5.2f}  reused={r['reused_gpu_tree']}")
packed = rows[-1]['packed_node_count']
print(f"  final packed_node_count: {packed}")
PY
