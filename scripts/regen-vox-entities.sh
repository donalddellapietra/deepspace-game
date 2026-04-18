#!/usr/bin/env bash
# Regenerate voxelized entity models from the GLB source files in
# assets/characters/. Output goes to assets/vox/, which is gitignored
# (see .gitignore). Run this once after a fresh clone, or whenever
# the GLB sources change.
#
# Requires: python3 + `pip3 install trimesh numpy`.
#
# Usage:
#   ./scripts/regen-vox-entities.sh

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p assets/vox

echo "Voxelizing Soldier @ res=81..."
python3 tools/glb_to_vox.py \
    assets/characters/Soldier.glb \
    -o assets/vox/soldier.vox \
    --resolution 81

echo "Voxelizing Fox @ res=81..."
python3 tools/glb_to_vox.py \
    assets/characters/Fox.glb \
    -o assets/vox/fox.vox \
    --resolution 81 || echo "(fox voxelization failed; continuing)"

echo "Done. Files:"
ls -lh assets/vox/
