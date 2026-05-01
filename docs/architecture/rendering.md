# Rendering

The renderer is a per-pixel GPU ray marcher that walks the world tree
inside a *frame-local* coordinate system, chosen fresh each frame from
the camera's anchor path. There is no global world transform, no mesh
buffer, and no CPU geometry stage.

Source of truth:
- `src/renderer.rs` — wgpu pipeline, buffer management, present loop.
- `src/app/frame.rs` — frame selection (pure; unit-testable).
- `src/world/gpu/` — packing, ribbon, GPU-side types.
- `assets/shaders/*.wgsl` — the ray marcher itself.

## One-frame walkthrough

1. **Pick the render frame.** `compute_render_frame` walks up the
   camera's anchor path until it finds a Cartesian ancestor at
   `camera_depth - K`. Returns an `ActiveFrame { render_path,
   logical_path, kind, node_id }`.

2. **Project the camera.** `Camera::gpu_camera_at(frame)` produces a
   `GpuCamera` whose position is in the frame's `[0, 3)³` local box.
   All f32 math is safe: the frame box is bounded.

3. **Pack the tree.** `gpu::pack::pack_tree_lod_preserving` BFS-packs
   the subtree rooted at the frame into two flat buffers:
   - `tree_buffer` — one `GpuChild` per child slot (8B: tag, block,
     node_index).
   - `node_kinds_buffer` — one `GpuNodeKind` per packed node (16B:
     kind discriminant, face, inner/outer radii).

   Cartesian subtrees that subtend fewer than LOD_THRESHOLD pixels are
   flattened to their `representative_block`.

4. **Build the ribbon.** `gpu::ribbon::build_ribbon` emits the chain
   of ancestors from the frame's *parent* up to the world root. When a
   ray exits the frame's `[0, 3)³` box, the shader pops up this ribbon
   to find a containing sibling or terminal — this is how the rendered
   volume extends "past the walls" without the packer having to
   pre-flatten every ancestor.

5. **Upload buffers.** `renderer.update_tree`, `update_ribbon`,
   `update_camera`, `update_uniforms` write to wgpu buffers. An
   `LodUploadKey` (root, camera anchor, offset bits, render path,
   logical path, visual depth, kind tag) gates repacking — if the key
   is unchanged, the upload is skipped.

6. **Render.** A full-screen quad triggers `fs_main`. The fragment
   shader reconstructs a ray from NDC, calls `march()`, shades the hit,
   and composites the cursor highlight AABB.

## Shaders

All WGSL in `assets/shaders/` is stitched together by
`src/shader_compose.rs` (minimal `#include` resolver) starting from
`main.wgsl`:

- **`main.wgsl`** — entry. Vertex stage draws the quad; fragment
  stage builds the primary ray and calls `march`.
- **`march.wgsl`** — Cartesian tree walker.
- **`tree.wgsl`** — child access, slot math, representative-block
  LOD fallback when a subtree can't be descended further.
- **`ray_prim.wgsl`** — ray–box intersection.
- **`bindings.wgsl`** — bind-group layouts shared by Rust and WGSL.

All shaders march in the frame's local coordinates. None of them know
the frame's *depth in the world tree* — that's the point.

## GPU buffers

| Buffer | Size | Purpose |
|---|---|---|
| `tree` | N × 8 B | packed children (tag, block, node_index) |
| `node_kinds` | M × 16 B | per-node kind |
| `ribbon` | R × 8 B | ancestor chain for frame-exit pops |
| `camera` | 96 B | frame-local pos, basis vectors, fov |
| `palette` | 256 × 4 B | block-type colors |
| `uniforms` | 96 B | screen size, root index/kind, max_depth, highlight AABB, ribbon count |

## Frame selection

`compute_render_frame` in `src/app/frame.rs` returns an
`ActiveFrameKind::Cartesian` whose render root is a plain 3×3×3
Cartesian node. `render_path == logical_path`.

## Precision budget

f32 at depth 14 hits its ulp limit (1/3¹⁴ ≈ 2×10⁻⁷). This is why the
renderer never runs in world-root coordinates. `K = 3` levels up from
the camera's anchor gives a 27³ = ~20k-cell viewport that stays well
inside f32. Deeper than that is rendered via *pop up the ribbon*, not
via more precision.

If you need the theoretical backdrop, see
[../principles/scaling-deep-trees.md](../principles/scaling-deep-trees.md)
and [../history/camera-rewrite-first-principles.md](../history/camera-rewrite-first-principles.md).

## Cache keys

Two structs gate expensive work:

- `LodUploadKey` — (root, camera anchor, offset bits, render path,
  logical path, visual depth, kind tag). Equal to last frame ⇒ skip
  the packer + upload.
- `HighlightCacheKey` — (`LodUploadKey`, yaw/pitch bits, cursor
  locked, epoch). Equal ⇒ skip the cursor raycast.

## Offscreen / screenshot rendering

`renderer.render_offscreen()` renders to a target texture;
`capture_to_png` reads it back. Used by the test harness for
deterministic screenshots. See [../testing/harness.md](../testing/harness.md).
