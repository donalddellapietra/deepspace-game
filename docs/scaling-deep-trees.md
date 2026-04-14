# Scaling the Ray Marcher to Deep Trees (>8 levels)

## Status: BLOCKED — grey screen at >8 levels

The ray marcher works correctly at 6 levels (depth 6, 38 nodes). When we scale to 20+ levels (depth 21, ~80-125 nodes), the screen renders as a solid grey cube. The underlying tree structure is correct (tests pass, content-addressing works, node counts are right), but the rendering pipeline breaks.

## The Problem

The shader has `MAX_STACK_DEPTH = 8` (fixed WGSL arrays). A 21-level tree needs the ray to descend 21 levels near the camera to reach actual blocks. At depth 8, the ray hits a Node child, can't descend further, and renders `vec3(0.5)` (grey) because `dominant_block = 255` at that intermediate depth.

"Just increase MAX_STACK_DEPTH to 24" was tried — **it didn't fix it.** The screen remained solid grey even with stack depth 24, max_iterations 512, and visual_depth capped at 24.

## What We Tried

### 1. Increase shader stack depth (8 → 24)
**Result: Still grey.** The arrays were enlarged, max_iterations increased to 512, renderer and visual_depth caps raised to 24. No change. The ray either still can't reach blocks, or the LOD/packing pipeline prevents it.

### 2. Screen-space LOD cutoff in shader
Added per-pixel check: `cell_world_size / ray_distance * screen_height / (2 * tan(fov/2))`. If sub-pixel, stop descending and use dominant color. This helps performance for distant terrain in the 6-level tree but didn't fix the deep-tree grey screen.

### 3. Distance-aware GPU packing (`pack_tree_lod`)
CPU walks the tree with camera position, only uploads nodes large enough on screen. Distant nodes flattened to Block(dominant_color). Near nodes get full children. This runs every frame.

**Potential issue:** Content-addressed nodes appear at multiple world positions. The BFS visits each NodeId once, using the FIRST occurrence's position for LOD decisions. A node first seen nearby gets full detail, but the same node far away also gets full detail (because it's already visited). Conversely, a node first seen far away gets flattened, even if it also appears nearby.

### 4. Uniform node flattening
Nodes where the entire subtree is one block type (`uniform_type`) get packed as Block/Empty. All-air volumes skip instantly. All-stone mountains render as one block. Works correctly at 6 levels.

### 5. Dominant color propagation
Nodes track `dominant_block` recursively through children. GPU packs this into the `block_type` field for Node entries. Shader uses `palette.colors[bt]` at LOD cutoff instead of grey. Works at 6 levels.

### 6. Memoized tree_depth()
Prevented exponential recursive walks through content-addressed tree. Reduced from potentially millions of calls to ~N (one per unique node).

## Why It's Hard

### Content-addressing vs. spatial LOD
The fundamental tension: content-addressed nodes are **position-independent** (same NodeId appears at many world positions), but LOD decisions are **position-dependent** (a node near the camera needs more detail than the same node far away).

The GPU buffer can only store ONE version of each node. If the same NodeId appears both near and far from the camera, it must be packed the same way in both locations. This means:
- If we pack it with full children (for nearby): distant occurrences waste shader time descending
- If we flatten it (for distant): nearby occurrences lose detail

The 6-level tree doesn't hit this hard because content sharing is mostly between nodes at the same depth (same-sized things). At 20+ levels, nodes are reused across many depths and positions.

### Camera position vs. tree depth
At zoom_level 0 with depth 21, the finest blocks are ~10^-10 units wide. The camera is at [1.5, 1.75, 1.5] — a macro position. The ray must descend 21 levels of 3×3×3 DDA to reach a block. Even with LOD, nearby rays need full depth.

The `y-structure` (y=0 underground, y=1 surface, y=2 air) must be maintained at EVERY level or the camera ends up inside solid terrain. This was fixed for 6 levels but the 20-level version had the camera alternating between air/terrain at different depths due to the base-3 expansion of y=1.75.

### Shader register pressure
WGSL arrays of size 24 × 5 arrays = 120 registers per pixel. GPU occupancy drops, reducing parallelism. This may explain why increasing stack depth alone didn't help — the shader may have hit register limits and produced incorrect results silently.

## What Might Actually Work

### A. Virtual root / camera-relative packing
Instead of always starting the ray from the absolute root, find the deepest node containing the camera and make THAT the GPU root. Upload only its subtree. The ray starts at depth 0 in a local coordinate system.

**Challenge:** Rays that exit this node need to pop up to siblings/parents. This requires uploading the path from absolute root to the virtual root (one node per level = 21 nodes), plus siblings at each level.

### B. Two-pass rendering
1. **Coarse pass:** Render from root with low max_depth (3-4 levels). This shows the large-scale terrain structure — mountains, biomes, etc. Each pixel gets a coarse block color.
2. **Detail pass:** For pixels near the camera, re-render starting from a deeper node that's been pre-identified as containing the camera. This pass uses full depth but only covers a portion of the screen.

### C. Hybrid approach: pre-resolved GPU buffer
Instead of uploading the raw tree and letting the shader traverse it, resolve the tree on CPU for the camera's viewport. Walk the tree from root, and for each node the camera can see:
- If it's small enough on screen (< N pixels): store its dominant color as a Block entry
- If it's large: store its 27 children

This produces a FLAT buffer where every entry is either Block or a reference to a pre-resolved child. The shader's traversal depth equals the number of nodes that are "large enough to see" in a straight line from the camera — typically 5-8 regardless of tree depth.

**This is essentially what `pack_tree_lod` tries to do, but it needs to handle the content-addressing problem:** the same NodeId at different positions needs different resolutions. The fix is to pack by POSITION (world-space path), not by NodeId. Two occurrences of the same NodeId at different distances become separate entries in the GPU buffer.

### D. Completely different shader architecture
Replace the iterative stack-based DDA with a different traversal:
- **Beam marching:** Instead of per-pixel rays, march beams (groups of pixels). Coarse beams for distant terrain, fine beams near camera.
- **Cone tracing:** March a cone that widens with distance. The cone's width determines LOD.
- **Precomputed distance fields:** For each node, store max empty-space skip distance. Rays jump through empty regions.

## Current State (Working)

- 6-level tree with 38 unique nodes renders correctly
- Zoom (scroll wheel) changes edit scale and movement speed
- Left-click break, right-click place with clone-on-write propagation
- Dominant colors, uniform flattening, distance-aware packing, screen-space LOD
- All 13 tests pass

## Files

| File | Role |
|------|------|
| `assets/shaders/ray_march.wgsl` | GPU ray marcher (DDA through base-3 tree) |
| `src/world/worldgen.rs` | World generator (currently 6 levels) |
| `src/world/gpu.rs` | `pack_tree` and `pack_tree_lod` (tree → GPU buffer) |
| `src/world/tree.rs` | Node, NodeLibrary, content-addressing, dominant_block, uniform_type |
| `src/world/edit.rs` | CPU raycast, break/place, clone-on-write propagation |
| `src/world/state.rs` | WorldState, tree_depth() |
| `src/renderer.rs` | wgpu renderer, uniforms (max_depth), buffer upload |
| `src/main.rs` | App loop, zoom, input, LOD packing per frame |
