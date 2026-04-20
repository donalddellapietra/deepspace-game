# Content Pipeline

## 3D Model → Playable World

```
Source GLB/glTF (Sponza, San Miguel, Bistro, character models, …)
        │
        │  scripts/fetch-glb-presets.sh         (downloads the three
        │                                        canonical scenes)
        ▼
assets/scenes/<name>.glb   (gitignored; multi-GB)
        │
        │  tools/scene_voxelize/                (GPU voxelizer;
        │   cargo run --release                  MIT-licensed copy of
        │     -- --models <name> \               voxel-raymarching's
        │        --scale 16                      pipeline — see
        │                                        ATTRIBUTION.md)
        ▼
assets/scenes/<name>.vxs   (sparse palette-indexed voxel grid)
        │
        │  src/import/vxs.rs                    (loads .vxs,
        │                                        merges palette into
        │                                        ColorRegistry)
        ▼
  src/world/tree               (content-addressed base-3 tree)
        │
        │  WorldPreset::Scene | WorldPreset::VoxModel
        ▼
   Playable, editable voxel world
```

## scene_voxelize

Rust + WGPU GPU voxelizer ported from James Catania's
[voxel-raymarching](https://github.com/jamescatania/voxel-raymarching)
under MIT. Triangle-cube intersection in a compute shader, per-triangle
UV sampling at the voxel/triangle hit (so colors follow the actual
texel, not a nearest-vertex approximation), and k-means palette
quantization in Oklab. Up to 200 palette entries.

See `tools/scene_voxelize/ATTRIBUTION.md` for the full upstream license
and a list of what was removed/modified.

### Usage

```bash
# one-time: fetch the three canonical benchmark scenes
scripts/fetch-glb-presets.sh

# voxelize Sponza at 16 voxels/meter (default)
cd tools/scene_voxelize
cargo run --release -- --models sponza

# all three at higher res
cargo run --release -- --models sponza san_miguel bistro --scale 24
```

Output lands at `assets/scenes/<name>.vxs`. Both the source GLBs and
the generated `.vxs` files live in `assets/scenes/` and are gitignored
(too large for git) — rebuild with `fetch-glb-presets.sh` +
`cargo run` whenever the repo is cloned fresh.

### Load in-game

```bash
# high-level scene presets (pick depth automatically)
scripts/dev.sh -- --sponza-world
scripts/dev.sh -- --san-miguel-world
scripts/dev.sh -- --bistro-world

# or the generic .vxs path for anything custom
scripts/dev.sh -- --vox-model assets/scenes/sponza.vxs --plain-layers 9
```

## Character GLBs

Small rigged characters (`assets/characters/Fox.glb`,
`Soldier.glb`) are used by the npc pipeline — see
`build_npc_blueprint.py` / `extract_skeleton.py`. These are separate
from the scene voxelizer.
