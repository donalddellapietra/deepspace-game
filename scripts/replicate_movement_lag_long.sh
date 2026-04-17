#!/bin/bash
#
# Stress test for movement + block-placement lag.
#
# Walks 10+ cells in one direction (crossing the LOD-quantize boundary
# at LOD_CAMERA_QUANTIZE_LEVELS), then alternates placements and
# breaks, then walks back. Emits per-frame CSV at /tmp/lag-long.csv.
#
# What it should look like with implicit-ribbon + quantized LOD key:
#   - Most frames: pack_ms ≈ 0 (cache hit).
#   - A few frames per ~9-cell quantum boundary crossing: pack_ms = the
#     full re-pack cost. Should still be ≤ ~80 ms per miss, but they
#     should fire 1×/quantum, not 1×/cell.
#   - Place/break frames: the edit dirties the LOD key, so they will
#     re-pack — should be at most as expensive as the worst movement
#     re-pack.

set -e
cd "$(dirname "$0")/.."

cargo build --bin deepspace-game 2>&1 | grep -E "(error|warning: unused)" || true

# Walk 12 cells forward, place a block every 3 cells; then walk back.
SCRIPT="wait:30,"
for i in {1..12}; do
    SCRIPT="${SCRIPT}step:x+,wait:2,"
    if (( i % 3 == 0 )); then
        SCRIPT="${SCRIPT}place,wait:2,"
    fi
done
for i in {1..12}; do
    SCRIPT="${SCRIPT}step:x-,wait:2,"
done
SCRIPT="${SCRIPT}break,wait:5,break,wait:5,break,wait:5"

./target/debug/deepspace-game --render-harness \
    --vox-model assets/vox/soldier_729.vxs --plain-layers 8 \
    --spawn-xyz 1.3889 2.8333 0.3889 --spawn-depth 7 \
    --disable-overlay --harness-width 600 --harness-height 400 \
    --script "$SCRIPT" \
    --perf-trace /tmp/lag-long.csv --perf-trace-warmup 10 \
    --exit-after-frames 350 --timeout-secs 60 --suppress-startup-logs \
    2>&1 | tail -3

echo ""
echo "=== Summary ==="
python3 <<'PY'
import csv
with open('/tmp/lag-long.csv') as f:
    rows = list(csv.DictReader(f))
miss = [r for r in rows if int(r['reused_gpu_tree']) == 0]
hit  = [r for r in rows if int(r['reused_gpu_tree']) == 1]
print(f"  frames: {len(rows)}, cache hits: {len(hit)}, misses: {len(miss)}")
if miss:
    pms = [float(r['pack_ms']) for r in miss]
    upms = [float(r['upload_total_ms']) for r in miss]
    print(f"  cache-miss pack: mean={sum(pms)/len(pms):.2f}ms max={max(pms):.2f}ms")
    print(f"  cache-miss upload: mean={sum(upms)/len(upms):.2f}ms max={max(upms):.2f}ms")
non_startup = [r for r in rows if int(r['frame']) >= 30]
worst = max(non_startup, key=lambda r: float(r['wall_ms']))
print(f"  worst non-startup wall: frame={worst['frame']} wall={float(worst['wall_ms']):.1f}ms pack={float(worst['pack_ms']):.1f}ms reused={worst['reused_gpu_tree']}")
PY
