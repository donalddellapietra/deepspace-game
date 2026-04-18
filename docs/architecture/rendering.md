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
   camera's anchor path until it finds a path whose node can serve
   as a frame root — either a Cartesian ancestor at `camera_depth - K`,
   or a `CubedSphereBody` cell the camera sits inside. Returns an
   `ActiveFrame { render_path, logical_path, kind, node_id }`.

2. **Project the camera.** `Camera::gpu_camera_at(frame)` produces a
   `GpuCamera` whose position is in the frame's `[0, 3)³` local box.
   All f32 math is safe: the frame box is bounded.

3. **Pack the tree.** `gpu::CachedTree::update_root` emits the
   world tree into three flat buffers via content-addressed memo:
   - `tree` — interleaved header + inline children slab (u32s).
   - `node_kinds` — one `GpuNodeKind` per packed node (16B:
     kind discriminant, face, inner/outer radii).
   - `node_offsets` — BFS-idx → tree[] offset.

   Edits reuse previously-packed subtrees via `bfs_by_nid` (O(1) per
   unchanged subtree); only the N+1 new ancestors get appended. LOD
   lives in the shader — pack does no view-dependent flattening
   beyond collapsing uniform subtrees (safe at any view). Spheres
   stay as Node children regardless of uniformity —
   their DDA is cheap and their silhouette must be preserved.

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
- **`march.wgsl`** — unified tree walker. Switches on `NodeKind` at
  each descent: Cartesian DDA, or cubed-sphere dispatch.
- **`tree.wgsl`** — child access, slot math, representative-block
  LOD fallback when a subtree can't be descended further.
- **`sphere.wgsl`**, **`face_math.wgsl`**, **`face_walk.wgsl`** —
  cubed-sphere DDA: one face at a time, with face-seam crossings
  handled by axis remapping. See [cubed-sphere.md](cubed-sphere.md).
- **`ray_prim.wgsl`** — ray–box, ray–sphere intersections.
- **`bindings.wgsl`** — bind-group layouts shared by Rust and WGSL.

All shaders march in the frame's local coordinates. None of them know
the frame's *depth in the world tree* — that's the point.

## GPU buffers

| Buffer | Size | Purpose |
|---|---|---|
| `tree` | N × 8 B | packed children (tag, block, node_index) |
| `node_kinds` | M × 16 B | per-node kind + sphere radii / face |
| `ribbon` | R × 8 B | ancestor chain for frame-exit pops |
| `camera` | 96 B | frame-local pos, basis vectors, fov |
| `palette` | 256 × 4 B | block-type colors |
| `uniforms` | 96 B | screen size, root index/kind, max_depth, highlight AABB, ribbon count |

## Frame selection

`compute_render_frame` in `src/app/frame.rs` returns one of three
`ActiveFrameKind`s:

- **`Cartesian`** — render root is a plain 3×3×3 Cartesian node.
  `render_path == logical_path`.
- **`Body { inner_r, outer_r }`** — render root is a
  `CubedSphereBody`. Rays first intersect the body sphere; inside,
  the shader dispatches into face DDA.
- **`Sphere(SphereFrame)`** — render root stays at the containing
  body, but the player's `logical_path` is inside a specific face
  subtree. The shader walks the face with an explicit `(u_min, v_min,
  r_min, size)` window. This is what lets the player stand on a
  planet surface without the body cell blowing the camera's f32
  budget.

`render_path` drives the GPU ribbon. `logical_path` drives editing,
highlight, and UI — it can be deeper than `render_path` when the
player is inside a face subtree.

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
