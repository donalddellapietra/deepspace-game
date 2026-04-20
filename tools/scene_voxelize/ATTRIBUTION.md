# Attribution

This directory is a modified copy of the voxelization pipeline from:

- **Project:** voxel-raymarching
- **Author:** James Catania
- **License:** MIT (included below, copied verbatim from upstream)
- **Upstream repo snapshot:** `external/voxel-raymarching/` in this repo

## What was copied

Wholesale copy of:
- `generate/` — the GLB → voxel pipeline
- `utils/` — GPU helper crate it depends on

## What was removed

- `generate/src/lightmaps.rs` and associated shaders (`brdf.wgsl`,
  `cubemap.wgsl`, `downsample.wgsl`, `prefilter.wgsl`) — HDR environment
  baking, not needed.
- `generate/src/shaders/tree.wgsl`, `tree_64.wgsl` — their 4³ sparse
  brickmap packer. Dropped because we emit `.vxs` (our existing sparse
  format, read by `src/import/vxs.rs`), not their brickmap.
- `utils/src/tree.rs` — raymarcher-side tree type, only used by their
  app at runtime.
- `utils/src/texture/`, `utils/src/textures.rs` — swap-chain helpers
  used by their raymarcher app, not the voxelize path.

## What was modified

- `generate/src/lib.rs`, `main.rs` — dropped lightmap paths; renamed
  CLI.
- `generate/src/models.rs` — replaced the 4³ tree packing (3 compute
  passes + serialization) with a readback-to-CPU + `.vxs` writer.

## Original MIT license

```
MIT License

Copyright (c) 2026 James Catania

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
