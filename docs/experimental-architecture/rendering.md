# Rendering: GPU Ray Marching

## Overview

The renderer casts one ray per pixel through the recursive base-3 tree. No meshes are generated, cached, or streamed. The GPU traverses the tree directly, descending through nodes until hitting a terminal block or reaching a screen-space LOD cutoff. The entire mesh pipeline (greedy meshing, mesh cache, mesh streaming, cold bakes, entity reconciliation) is replaced by a single compute/fragment shader.

## The Ray March Algorithm

For each pixel on screen:

1. Compute a ray from the camera through the pixel.
2. Starting at the root node, determine which of 27 children the ray enters (DDA step into a 3x3x3 grid).
3. If the child is `Empty`: advance the ray to the next cell boundary, continue stepping.
4. If the child is `Block(type)`: hit. Compute the surface normal from which face was crossed. Shade and write the pixel.
5. If the child is `Node(id)`: descend into that node. The ray is now in a 3x smaller space. Repeat from step 2 with the child's 27 children.
6. LOD cutoff: if the cell's screen-space size is below a threshold (e.g., smaller than a pixel), treat the node as a solid block of its dominant color instead of descending further.

The ray walks down the tree, picking 1 of 27 children at each level. At each level, the coordinate space subdivides by 3x per axis. The worst case is ~63 levels of descent (root to atom), but the LOD cutoff means most rays descend only 5-15 levels.

## DDA Stepping (Digital Differential Analyzer)

Within each node's 3x3x3 grid, the ray steps from cell to cell using DDA — the same algorithm used for 2D line rasterization, extended to 3D:

1. Compute `tMax` for each axis: how far along the ray until it crosses the next cell boundary on that axis.
2. Advance along whichever axis has the smallest `tMax`.
3. Update the cell index on that axis (+1 or -1 depending on ray direction).
4. Repeat until the ray exits the 3x3x3 grid or hits a non-empty cell.

For a 3x3x3 grid, this is at most ~9 steps (diagonal traversal). The total DDA steps per pixel across all tree levels is bounded by roughly 9 x (number of levels descended) — typically under 100.

## GPU Data Layout

### Tree Buffer

The visible portion of the tree is uploaded as a GPU storage buffer. Each node is packed as 27 child entries:

```
struct GpuChild {
    tag: u8,        // 0 = Empty, 1 = Block, 2 = Node
    block_type: u8, // valid when tag == 1
    node_index: u32 // index into the buffer, valid when tag == 2
}
```

Node indices are local to the buffer (not NodeIds). A mapping from NodeId to buffer index is maintained CPU-side when building the upload buffer each frame.

The visible scene typically requires ~10,000 nodes near the player. At ~160 bytes per node (27 x 6 bytes), that's ~1.6MB — easily fits in GPU memory. Distant regions need fewer nodes (coarser LOD), so the buffer stays small regardless of world size.

### Camera Uniforms

```
struct Camera {
    position: vec3<f32>,    // in tree-local coordinates
    forward: vec3<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    fov: f32,
    aspect: f32,
    near: f32,
}
```

### Output

A full-screen texture (the framebuffer). The shader writes one color per pixel. No intermediate geometry, no vertex buffers, no index buffers.

## LOD: Automatic and Per-Pixel

The ray decides how deep to descend based on the screen-space size of the current cell:

```
cell_screen_size = cell_world_size / ray_distance * screen_height / (2 * tan(fov/2))

if cell_screen_size < 1.0:
    // this cell is sub-pixel, stop descending
    // shade as a solid block of the node's dominant color
```

This means:
- Near the player: rays descend to layer 0 (millimeter voxels). Full detail.
- 10 meters away: rays stop at layer 3. Block-level detail.
- 100 meters away: rays stop at layer 6. Furniture-scale blobs.
- 1 km away: rays stop at layer 9. Tree-scale silhouettes.
- Horizon: rays stop at layer 12+. Terrain-scale shapes.

No LOD management code. No LOD transitions. No popping. Every pixel independently resolves exactly the detail it needs.

## Lighting

### Basic: Directional Light + AO

The simplest approach:

1. **Surface normal**: known from which face the ray crossed when it hit a block. No normal maps or smoothing needed — voxels have axis-aligned faces.
2. **Directional light**: `dot(normal, sun_direction)`. One multiply.
3. **Ambient occlusion**: cast a few short secondary rays from the hit point into the tree. Count how many hit solid blocks nearby. Darken accordingly.

AO can be precomputed per-node and stored as a small per-child value (1 byte per child = 27 bytes per node), or computed live in the shader at the cost of secondary ray traversals.

### Advanced: Full Ray-Traced Lighting

Since we're already traversing the tree per pixel, adding shadows and reflections is architecturally simple:

- **Shadows**: cast a secondary ray from the hit point toward the sun. If it hits a solid block, the pixel is in shadow.
- **Reflections**: cast a reflected ray. Traverse the tree again.
- **Global illumination**: multiple bounces of secondary rays (expensive but possible).

Each secondary ray uses the same tree traversal code. The cost is linear in the number of secondary rays per pixel.

## Edits Are Instant

When the player breaks a block:

1. CPU: create 63 new nodes (one per ancestor to root). Microseconds.
2. CPU: update the upload buffer — swap out the ~63 changed nodes.
3. GPU: next frame, the ray marches through the updated tree. The change is visible immediately.

No mesh re-baking. No cold bake stalls. No mesh cache invalidation. The tree IS the renderable data.

## Integration with Bevy

Keep Bevy for windowing, input, audio, and the React overlay bridge. Replace only the rendering:

1. Remove `render_world`, `RenderState`, `MeshStore`, `mesh_cache.rs`, `mesh_stream.rs`, `walk.rs` (the render walk), entity reconciliation, sub-mesh spawning.
2. Add a custom Bevy render plugin that:
   - Collects visible tree nodes into a flat buffer (CPU, per frame)
   - Uploads the buffer to a GPU storage buffer
   - Dispatches a full-screen ray march shader (WGSL compute or fragment)
   - Writes to the swapchain texture

The game logic (player movement, collision, editing, inventory, UI overlay) stays in Bevy unchanged. The renderer becomes a single system that uploads nodes and dispatches a shader.

## What Disappears

The following modules/systems are no longer needed:

- `mesh_cache.rs` — no meshes
- `mesh_stream.rs` — no mesh streaming
- `walk.rs` (render walk) — the GPU traverses the tree, not the CPU
- `model/mesher.rs` (greedy meshing) — no meshes
- `serial.rs` (mesh serialization) — no meshes to serialize
- `bin/gen_world.rs` (offline mesh baking) — no meshes to bake
- Entity reconciliation in `render.rs` — no entities
- `Palette` / material system — block colors are a simple lookup table in the shader
- `BakedSubMesh`, `WorldRenderedNode`, `SubMeshBlock` components — no entities

What remains on the rendering side: one system that builds the GPU buffer, one shader that ray marches, and a color palette uniform.
