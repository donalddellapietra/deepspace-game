#!/bin/bash
#
# Replicate edit lag: placing/breaking blocks stalls the frame because
# re-packing used to be O(visible_nodes) from scratch.
#
# Incremental pack (CachedTree::update_root memoizes by NodeId) makes
# each edit O(edit_path_depth) — a few hundred u32s appended to the
# existing buffer. Initial world load is a full pack (one-shot).
#
# Setup: 20-layer soldier world, spawn above model, fly_to_surface to
# land on it, then break/place a bunch of times.
#
# Expected:
#   initial pack:     ~50-100 ms (one-shot, 2M u32s for soldier_729)
#   per-edit pack:    <1 ms      (~300 u32s appended)

set -e
cd "$(dirname "$0")/.."

cargo build --bin deepspace-game 2>&1 | grep -E "(error|warning: unused)" || true

SCRIPT=""
for i in $(seq 1 20); do
  SCRIPT="${SCRIPT}break,wait:3,place,wait:3,"
done
SCRIPT="${SCRIPT%,}"

./target/debug/deepspace-game --render-harness \
    --vox-model assets/vox/soldier_729.vxs \
    --plain-layers 20 --vox-interior-depth 13 \
    --spawn-xyz 0.5 1.5 0.12 --spawn-depth 10 --spawn-pitch -1.5 \
    --disable-overlay --harness-width 600 --harness-height 400 \
    --script "wait:10,fly_to_surface,wait:5,${SCRIPT}" \
    --exit-after-frames 300 --timeout-secs 120 --suppress-startup-logs \
    2>&1 | grep -E "render_harness_pack" > /tmp/edit-packs.txt

echo "=== render_harness_pack lines ==="
cat /tmp/edit-packs.txt | head -5
echo ..
tail -3 /tmp/edit-packs.txt
echo

echo "=== Summary: initial pack vs incremental edit packs ==="
python3 <<'PY'
import re, statistics
rows = [l.strip() for l in open('/tmp/edit-packs.txt') if l.strip()]
def extract(line):
    m_appended = re.search(r'appended_u32s=(\d+)', line)
    m_pack = re.search(r'pack_ms=([\d.]+)', line)
    return int(m_appended.group(1)) if m_appended else 0, float(m_pack.group(1)) if m_pack else 0.0

if not rows:
    print("  no render_harness_pack lines found")
else:
    initial = rows[0]
    edit_rows = rows[1:]
    ia, ip = extract(initial)
    print(f"  initial pack:   appended_u32s={ia:>8}  pack_ms={ip:6.2f}")
    if edit_rows:
        packs = [extract(r) for r in edit_rows]
        appended = [a for a, _ in packs]
        pack_ms = [p for _, p in packs]
        print(f"  edit packs x{len(packs)}:")
        print(f"    appended_u32s  max={max(appended):>5}  median={statistics.median(appended):.0f}  mean={statistics.mean(appended):.0f}")
        print(f"    pack_ms        max={max(pack_ms):6.3f}  median={statistics.median(pack_ms):6.3f}  mean={statistics.mean(pack_ms):6.3f}")
        print(f"  speedup: initial/median-edit = {ip/statistics.median(pack_ms):.0f}x")
PY
