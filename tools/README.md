# Content Pipeline

## 3D Model → Playable World

```
Source model (.glb, .obj, .fbx)
       │
       ▼
  FileToVox (--scale N)
       │
       ▼
  .vox file (assets/vox/)
       │
       ▼
  Game import (import::vox → stamp into WorldState)
       │
       ▼
  Playable, editable voxel world
```

## FileToVox

Converts meshes to MagicaVoxel `.vox` format with color preservation.

**Install:** https://github.com/Zarbuz/FileToVox

**Usage:**
```bash
./tools/filetovox.sh model.glb assets/vox/model.vox --scale 128
```

The `--scale` parameter controls resolution:
- `64` — fast preview, blocky
- `128` — good balance for most models
- `256` — maximum MagicaVoxel resolution, high detail

## Color Handling

The game registers exact RGBA colors from the .vox palette into its runtime `Palette`. Each unique color in the .vox file becomes a distinct material. The palette supports up to 255 entries total (shared across all imported models and the 10 built-in block types).

## Test Assets

`assets/vox/` contains test models:
- `monu1.vox`, `monu3.vox` — monuments (from MagicaVoxel samples)
- `castle.vox` — colorful castle
- `chr_knight.vox` — character model
- `monu9.vox` — large monument
