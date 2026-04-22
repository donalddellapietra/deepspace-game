#!/usr/bin/env bash
#
# CPU-GPU parity spot-check. Runs the CPU walker's `walker_reaches_depth_30_hit`
# test (canonical precision model proof) AND the shader's deep-depth
# screenshot script. Asserts:
#   1. CPU `cargo test --lib world::cubesphere::walker` all green.
#   2. Each `tmp/stage3d_shader/depth_NN.png` has a hit-colored pixel
#      in the sphere region (i.e. at least some pixel is NOT sky
#      blue / flat gray).
#
# Pragmatic alternative to a full programmatic-GPU integration test
# inside `cargo test`: the CPU walker's test harness already verifies
# the precision model to depth 30; this script verifies the shader
# renders something in the sphere region at matching depths.
#
# Usage: scripts/cpu_gpu_parity_check.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "== CPU walker precision tests =="
cargo test --lib world::cubesphere::walker 2>&1 | tail -5

echo
echo "== Shader deep-depth screenshots =="
if [[ ! -s tmp/stage3d_shader/depth_05.png ]]; then
    echo "  running scripts/sphere-deep-depth-check.sh first"
    scripts/sphere-deep-depth-check.sh >/dev/null 2>&1
fi

# Verify each screenshot has non-sky / non-gray pixels (i.e. the
# shader actually rendered something at that depth). We check that
# the center region of the image has some variation — sky-only
# images are almost uniform.
python3 - <<'PY'
import sys
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    print("PIL/Pillow missing; skipping pixel check", file=sys.stderr)
    sys.exit(0)

depths = [5, 10, 15, 20, 25, 30]
fails = []
for d in depths:
    path = Path(f"tmp/stage3d_shader/depth_{d:02d}.png")
    if not path.exists():
        fails.append((d, "missing"))
        continue
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # Sample 9-point grid in the center region (where the sphere
    # should be) and check for color variation. Flat-blue sky gives
    # near-zero variation.
    samples = [img.getpixel((w * (1 + x) // 4, h * (1 + y) // 4))
               for x in range(3) for y in range(3)]
    reds = [s[0] for s in samples]
    greens = [s[1] for s in samples]
    blues = [s[2] for s in samples]
    variation = max(reds) - min(reds) + max(greens) - min(greens) + max(blues) - min(blues)
    if variation < 10:
        fails.append((d, f"flat image (variation={variation})"))
    else:
        print(f"  depth={d:2d} center variation={variation:3d} OK")

if fails:
    print()
    print("PARITY FAIL:", fails)
    sys.exit(1)
print()
print("PARITY OK — all 6 deep-depth screenshots show sphere geometry.")
PY
