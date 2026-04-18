#!/bin/bash
#
# Sphere-surface zoom-invariance regression harness.
#
# Pins the camera close to the sphere surface and renders the SAME view
# at a sweep of `--spawn-depth` values (i.e. different anchor depths,
# different LOD contexts, but physically identical camera/scene). On a
# correctly LOD'd renderer these frames should be *nearly identical* —
# the only differences come from sub-pixel cells where the terminal
# block color differs.
#
# The Cartesian path already has this invariance (distance-based LOD
# stops descent at sub-pixel cell size, march.wgsl:266-278). The sphere
# path currently does not: it samples to `uniforms.max_depth` regardless
# of cell-to-pixel size, so close-up frames flicker as anchor_depth
# changes — the symptom the user sees as "surface fluctuates wildly."
#
# Output: tmp/shot/sphere_zoom/<depth>.png per run + a pairwise pixel-
# diff summary. The script exits 0 always (it's a diagnostic, not a
# pass/fail gate — interpret the numbers in context).
#
# Usage: scripts/sphere_zoom_invariance.sh
#
# Tunables via env:
#   SPAWN_XYZ     camera world position (default "1.5 2.0 1.5" — just above the sphere's north pole)
#   SPAWN_PITCH   camera pitch in rad   (default -1.5, near straight-down)
#   SPAWN_YAW     camera yaw in rad     (default 0)
#   DEPTHS        space-separated list of anchor depths (default "3 5 8 12 16 20 25")
#   WIDTH/HEIGHT  harness resolution    (default 640 360)

set -e
cd "$(dirname "$0")/.."

SPAWN_XYZ="${SPAWN_XYZ:-1.5 2.0 1.5}"
SPAWN_PITCH="${SPAWN_PITCH:--1.5}"
SPAWN_YAW="${SPAWN_YAW:-0}"
DEPTHS="${DEPTHS:-3 5 8 12 16 20 25}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-360}"

SHOT_DIR="tmp/shot/sphere_zoom"
mkdir -p "$SHOT_DIR"
rm -f "$SHOT_DIR"/*.png

cargo build --bin deepspace-game 2>&1 | grep -E "error" || true

echo "=== Capturing sphere surface at fixed camera, varying --spawn-depth ==="
echo "    camera xyz = $SPAWN_XYZ   pitch = $SPAWN_PITCH   yaw = $SPAWN_YAW"
echo "    resolution = ${WIDTH}x${HEIGHT}"
echo "    depths     = $DEPTHS"
echo

for d in $DEPTHS; do
    out="$SHOT_DIR/d${d}.png"
    # shellcheck disable=SC2086
    timeout 20 ./target/debug/deepspace-game --render-harness \
        --sphere-world \
        --spawn-xyz $SPAWN_XYZ \
        --spawn-pitch "$SPAWN_PITCH" \
        --spawn-yaw "$SPAWN_YAW" \
        --spawn-depth "$d" \
        --disable-overlay --disable-highlight \
        --harness-width "$WIDTH" --harness-height "$HEIGHT" \
        --screenshot "$out" \
        --exit-after-frames 60 --timeout-secs 15 \
        --shader-stats --suppress-startup-logs \
        2>&1 | grep -E "render_harness_shader" | head -1 | \
        awk -v d="$d" '{
            # Pull avg_steps, hit_fraction, and max_steps out of the line.
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^avg_steps=/)     a = substr($i, 11)
                if ($i ~ /^hit_fraction=/)  h = substr($i, 14)
                if ($i ~ /^max_steps=/)     m = substr($i, 11)
            }
            printf "  d=%-3s  avg_steps=%s  max_steps=%s  hit_fraction=%s\n", d, a, m, h
        }'
done

echo
echo "=== Pairwise pixel diff between adjacent depths ==="
python3 - "$SHOT_DIR" $DEPTHS <<'PY'
import sys, os
from PIL import Image, ImageChops

shot_dir = sys.argv[1]
depths   = sys.argv[2:]

def load(d):
    p = os.path.join(shot_dir, f"d{d}.png")
    if not os.path.exists(p):
        return None
    return Image.open(p).convert("RGB")

imgs = {d: load(d) for d in depths}
missing = [d for d, im in imgs.items() if im is None]
if missing:
    print(f"  missing captures: {missing}")

def diff_stats(a, b):
    if a is None or b is None: return None
    if a.size != b.size: return None
    d = ImageChops.difference(a, b)
    # Per-channel absolute diff, averaged over all pixels.
    hist = d.histogram()
    total_pixels = a.size[0] * a.size[1]
    # Channels: R, G, B contribute 3 histograms of 256 bins each.
    weighted_sum = 0
    differing_pixels = 0
    for ch in range(3):
        bins = hist[ch*256:(ch+1)*256]
        for v, count in enumerate(bins):
            weighted_sum += v * count
        differing_pixels += sum(bins[1:])  # any non-zero diff
    mean_abs_diff = weighted_sum / (3 * total_pixels)
    # differing_pixels as computed is 3x (once per channel) — normalize
    # to "fraction of pixels where ANY channel differed" is hard without
    # per-pixel work. Approximate: report per-channel frac instead.
    per_channel_differ_frac = differing_pixels / (3 * total_pixels)
    return mean_abs_diff, per_channel_differ_frac

print(f"  {'pair':<12} {'mean_abs':>10} {'frac_differing':>16}")
print(f"  {'-'*12} {'-'*10} {'-'*16}")
pairs = list(zip(depths, depths[1:]))
# Also compute diffs against the DEEPEST capture as a reference
# (conceptually: the "ground truth" for LOD-invariant rendering is
# the highest-detail render — everything else should match it up to
# LOD-driven block simplification).
deepest = depths[-1]
for a, b in pairs:
    stats = diff_stats(imgs[a], imgs[b])
    if stats is None:
        print(f"  d{a} vs d{b}: skipped")
        continue
    mean_abs, frac = stats
    print(f"  d{a:<3} vs d{b:<3} {mean_abs:>10.2f} {frac:>15.2%}")
print()
for d in depths[:-1]:
    stats = diff_stats(imgs[d], imgs[deepest])
    if stats is None: continue
    mean_abs, frac = stats
    print(f"  d{d:<3} vs d{deepest:<3} (ref) {mean_abs:>10.2f} {frac:>15.2%}")

print()
print(f"  artifacts in {shot_dir}/")
PY
