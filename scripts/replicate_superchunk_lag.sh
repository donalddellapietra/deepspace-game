#!/bin/bash
#
# Replicate the "superchunk border" lag: crossing a render_path cell
# boundary triggers a full pack (~100ms spike on the legacy branch).
#
# "Superchunk" = the cell at `anchor_depth - RENDER_FRAME_K` (=3), i.e.
# one level shallower than the render frame's finest content. Inside
# that cell the LOD-upload key is stable (see the anchor-drop fix);
# leaving it forced a full CachedTree re-emit + ribbon rebuild before
# the content-memoized pack landed.
#
# Setup: spawn in the 20-layer interior soldier world at anchor_depth=7
# so render_path sits at depth 4 (cell size = 3^-4 ≈ 0.037 root units).
# Steps of magnitude 20 (= 20 anchor-cells in one shot) blow past cell
# boundaries quickly.
#
# Output: /tmp/superchunk-lag.csv — a CSV row per frame. Repack frames
# (reused_gpu_tree=0) show the spike cost.
#
# Expected output on current HEAD (anchor-drop fix):
#   repacks: ≥1
#   pack_ms: ≈80–110 ms on crossings
#
# Target after a successful fix:
#   pack_ms median ≤ 5 ms (or 0 if packed async)
#   worst-case upload_total ≤ 10 ms post-warmup

set -e
cd "$(dirname "$0")/.."

cargo build --bin deepspace-game 2>&1 | grep -E "(error|warning: unused)" || true

# 80 × (step:x+:20, wait:2) = big unidirectional drift, crosses many
# render_path cells. wait:20 warms the cache first.
SCRIPT=""
for i in $(seq 1 80); do
  SCRIPT="${SCRIPT}step:x+:20,wait:2,"
done
SCRIPT="${SCRIPT%,}"

./target/debug/deepspace-game --render-harness \
    --vox-model assets/vox/soldier_729.vxs \
    --plain-layers 20 --vox-interior-depth 13 \
    --spawn-depth 7 \
    --disable-overlay --harness-width 600 --harness-height 400 \
    --script "wait:20,${SCRIPT}" \
    --perf-trace /tmp/superchunk-lag.csv --perf-trace-warmup 10 \
    --exit-after-frames 400 --timeout-secs 120 --suppress-startup-logs \
    2>&1 | grep -E "render_harness_(timing|worst|workload)" | head -5

echo ""
echo "=== Summary: render_path (superchunk) crossings ==="
python3 <<'PY'
import csv, statistics
rows = list(csv.DictReader(open('/tmp/superchunk-lag.csv')))
repacks = [r for r in rows if r['reused_gpu_tree']=='0']
packs = sorted([float(r['pack_ms']) for r in repacks], reverse=True)
print(f"  total frames: {len(rows)}")
print(f"  repacks:      {len(repacks)}")
if packs:
    print(f"  pack_ms       max={packs[0]:.2f}  median={packs[len(packs)//2]:.2f}  mean={statistics.mean(packs):.2f}")
worst_upload = sorted(rows, key=lambda r: float(r['upload_total_ms']), reverse=True)[:5]
print("  top 5 upload spikes (post-warmup frames):")
for r in worst_upload:
    if int(r['frame']) > 10:
        print(f"    frame={r['frame']:>3}  upload={float(r['upload_total_ms']):6.2f}  pack={float(r['pack_ms']):6.2f}  reused={r['reused_gpu_tree']}")
PY
