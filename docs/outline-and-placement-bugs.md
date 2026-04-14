# Block Outline and Cross-Node Placement Bugs

## Context

The sphere-planet branch uses a pure ray-marched voxel renderer (wgpu, no Bevy).
The world is a base-3 recursive tree: each node has 27 children (3x3x3). Blocks
are rendered by a full-screen fragment shader that traverses this tree via iterative
DDA. There is no geometry pipeline — everything is ray-marched.

A React overlay (via wry WebView) provides the UI (hotbar, inventory, color picker).
The game has a CPU raycast (`edit.rs`) that mirrors the GPU traversal to determine
which block the crosshair targets, and editing functions (`break_block`, `place_block`)
that modify the tree.

## Bug 1: Block Outline

### Problem

The block outline (wireframe highlight on the targeted block) has two issues:

1. **Not visible on all faces**: When looking at the top face of a block from
   above, the outline edges don't appear. It only shows on side faces where
   edges are visible. This means looking straight down at terrain shows no
   outline at all.

2. **Not occluded by geometry**: The outline shows through solid blocks. If a
   highlighted block is behind a wall, the wireframe still renders.

### Current Implementation (working, but with above issues)

The outline is rendered entirely in the ray-march fragment shader. The CPU
computes the AABB of the targeted block (`hit_aabb()` in `edit.rs`) and uploads
it as `highlight_min`/`highlight_max` uniforms. The shader intersects each pixel's
ray with this AABB and draws white on edges:

```wgsl
let hb = ray_box(camera.pos, inv_dir, h_min, h_max);
if hb.t_enter < hb.t_exit && hb.t_exit > 0.0 {
    let t = max(hb.t_enter, 0.0);
    let hit_pos = camera.pos + ray_dir * t;
    let from_min = hit_pos - h_min;
    let from_max = h_max - hit_pos;
    let box_size = h_max - h_min;
    let edge_width = box_size.x * 0.04;
    // Check if near any edge (where two faces meet)
    let edge_count = u32(near_x) + u32(near_y) + u32(near_z);
    if edge_count >= 2u { color = mix(color, white, 0.8); }
}
```

The `edge_count >= 2` approach only draws where two face-planes intersect —
i.e., the 12 edges of the cube. This works for side faces but NOT for a face
viewed head-on, because on the top face (Y-max), the hit point is near Y-max
but not near any X or Z boundary unless you're at the very corner. The edges
of the top face (the 4 lines forming the square) require being near BOTH an
X/Z boundary AND the Y boundary — which is exactly what `edge_count >= 2`
checks. So the edges should be visible... but they're only 4% of the block
wide, which at grazing angles from above becomes sub-pixel.

### Attempted Fix 1: Per-face edge detection

Determined which face the ray entered through, then checked only the two
in-face axes for edge proximity:

```wgsl
// Determine entry face via per-slab t_lo values
var face_axis = 0u;
if t_lo.y > t_lo.x && t_lo.y > t_lo.z { face_axis = 1u; }
else if t_lo.z > t_lo.x { face_axis = 2u; }
// On the entry face, check the other two axes
if face_axis == 1u {
    on_edge = d_lo.x < ew || d_hi.x < ew || d_lo.z < ew || d_hi.z < ew;
}
```

**Result**: Broke the outline entirely at higher zoom layers. The face detection
was unreliable, especially when the camera was inside or very close to the
expanded AABB. The `t_lo` comparison failed at edge cases.

### Attempted Fix 2: Occlusion via HitResult.t

Added a `t: f32` field to the shader's `HitResult` struct. At each block hit
in the march, computed `t` from a ray-box intersection with the hit cell's AABB.
Then in the outline code, only drew the outline if `t_enter <= result.t + epsilon`.

**Result**: Broke the outline at higher zoom layers because the shader marches
deeper than the CPU raycast (`visual_depth = edit_depth + 3`). The shader's
`result.t` hits a fine sub-block, but the outline's `t_enter` is for the coarse
node AABB. Even with a scaled epsilon (`length(box_size)`), the interaction
between LOD levels made the threshold unreliable.

### Attempted Fix 3: Expanded AABB (Bevy style)

Expanded the AABB by 1% each side (2% total, matching Bevy's `gizmos.cube()`
with `Vec3::splat(cell_size * 1.02)`) and used per-slab face detection with
camera-inside-box fallback.

**Result**: Same face detection issues as Fix 1. The expansion helped with
z-fighting but didn't solve the visibility or occlusion problems.

### What the Bevy version does

The main branch (Bevy-based) uses `gizmos.cube()` — Bevy's built-in wireframe
cube gizmo. This is a proper line-drawing system with its own depth buffer
handling, separate from the voxel renderer. It doesn't have occlusion or
face-detection issues because it draws actual line geometry with correct depth
testing.

### Key constraints

- No geometry pipeline available (pure ray-march renderer)
- The shader marches to `visual_depth` (finer than CPU's `edit_depth`)
- At higher zoom layers, the highlight AABB covers a large coarse node while
  the shader renders fine detail inside it
- The outline must work at all zoom levels and from all viewing angles

## Bug 2: Cross-Node Block Placement

### Problem

Cannot place blocks when the adjacent cell would be outside the current 3x3x3
node. For example, standing on top of a block at cell (1, 2, 1) and trying to
place above gives target (1, 3, 1) which is outside the [0,2] range. The
current code returns `false`.

This is the most common case — you can't build upward past node boundaries,
which happens every 3 blocks.

### Current Implementation

```rust
let nx = x as i32 + dx;
let ny = y as i32 + dy;
let nz = z as i32 + dz;
if nx < 0 || nx > 2 || ny < 0 || ny > 2 || nz < 0 || nz > 2 {
    return false;  // <-- blocks all cross-node placement
}
```

### Attempted Fix: Walk up and descend

Walked up the hit path carrying the overflow offset until finding a parent
level where the neighbor is in-bounds, then descended back down into the
neighbor subtree picking face-adjacent cells.

```rust
// Walk up
carry = [if nx < 0 { -1 } else if nx > 2 { 1 } else { 0 }, ...];
// At parent level where in-bounds:
// Descend back down, picking child_x = if dx > 0 { 0 } else if dx < 0 { 2 } else { 1 }
```

**Result**: The logic was complex and had issues:
- When descending into a neighbor subtree, the neighbor cell might be a `Node`
  (containing blocks), `Empty`, or `Block`. Each case needs different handling.
- The "pick face-adjacent cell" heuristic (e.g., dx=+1 → child x=0) doesn't
  account for the actual ray hit position within the face.
- At coarse zoom levels with short paths, the walk-up immediately reaches root
  and the descent re-enters the same area or misses.
- The path construction for `propagate_edit` was fragile — the new path needs
  correct node IDs at each level, but we're navigating into nodes we didn't
  visit during the original raycast.

### What needs to happen

The correct approach for cross-node placement:
1. Compute the world-space position of the placement target (hit point + face normal * cell_size)
2. Perform a NEW raycast or tree lookup from root to find which cell that world
   position falls in
3. If that cell is empty, build a proper path from root to it and place the block

This avoids the fragile walk-up/descend approach entirely by converting the
problem to a point-in-tree lookup, which the tree structure already supports.

## Files involved

- `assets/shaders/ray_march.wgsl` — outline rendering (fragment shader)
- `src/world/edit.rs` — `place_block`, `hit_aabb`, `cpu_raycast`
- `src/renderer.rs` — `GpuUniforms` with `highlight_min/max`, `set_highlight()`
- `src/main.rs` — `update_highlight()` called each frame
