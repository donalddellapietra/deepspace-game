# Scaling the Ray Marcher to Deep Trees

## The Architecture

The world is a recursive base-3 tree. Every node has 27 children (3×3×3). The GPU ray marcher casts one ray per pixel, stepping through the tree via DDA at each level. Per-pixel LOD stops descent when a cell is sub-pixel. See `docs/experimental-architecture/rendering.md` for the full design.

## What "Deep" Means

A 63-layer tree spans quarks to galaxies. But the ray marcher doesn't care about absolute depth. It only cares about **visual depth** — the number of levels a ray actually descends before hitting a terminal or reaching LOD cutoff. Visual depth is bounded by screen resolution, not tree depth:

- A sub-pixel cell stops at depth 0 (relative to where the ray enters it)
- A cell filling the screen stops at ~log₃(screen_width) ≈ 7 levels
- The worst case (a ray from camera through nearby terrain to the horizon) crosses maybe 10-15 visually-significant levels

The challenge is not "how do we traverse 63 levels" — it's "how do we start the ray near the camera instead of at the absolute root."

## The Real Problem: Absolute Root Traversal

The old implementation started every ray at the tree's absolute root (layer 63) and descended toward the camera. Even with LOD cutoff, the ray must descend through every intermediate layer between root and camera to reach the local neighborhood. At depth 21, that's 21 levels of descent just to arrive at the terrain the player is standing on — before any actual rendering work begins.

This is why increasing `MAX_STACK_DEPTH` from 8 to 24 didn't help. The ray could descend further, but it was still doing useless work traversing empty intermediate nodes. And the GPU buffer still packed the tree root-down, wasting slots on nodes between root and camera that every ray must pass through identically.

### Why the Grey Screen

The grey screen at >8 levels was a symptom, not the disease. `dominant_block = 255` (unset) at intermediate nodes caused the shader to render `vec3(0.5)` when it couldn't descend further. But even with correct dominant block values, the approach is fundamentally wrong — the ray shouldn't be traversing those intermediate nodes at all.

## The Solution: Camera-Relative Packing (Virtual Root)

The old Bevy mesh-based implementation solved this with `WorldAnchor` — a floating origin that keeps all Bevy coordinates near zero. The ray marcher needs the equivalent: **the GPU buffer should be rooted at the deepest node containing the camera, not the absolute root.**

### How It Works

1. **CPU: Walk from root to camera.** Follow the camera's path through the tree to find the deepest node that contains the camera and is large enough to be the rendering root. This node becomes the **virtual root**. The walk is O(depth) — 63 node lookups, microseconds.

2. **CPU: Pack the virtual root's subtree.** Upload the virtual root and its visible descendants into the GPU buffer. Nodes near the camera get full children. Nodes far from the camera get flattened (children replaced with their representative block type). This is spatial packing — the same NodeId at two different distances becomes two different entries with different LOD levels.

3. **CPU: Pack the ancestor path.** For rays that exit the virtual root (looking up at the sky, or toward the horizon), upload one node per ancestor level from the virtual root back to the absolute root, plus siblings at each level. This is ~63 extra nodes — trivial.

4. **GPU: Ray starts at the virtual root.** The camera position is in virtual-root-local coordinates (always in [0, 3)³, perfect f32 precision). Rays descend from the virtual root, not the absolute root. Visual depth is 5-15 levels regardless of absolute tree depth.

5. **GPU: Rays that exit the virtual root pop upward.** The ancestor path provides the context. The ray enters the virtual root's parent, steps through siblings via DDA, and may descend into a sibling's subtree. This is rare (only for rays pointing away from the camera's local neighborhood) and adds at most one level per ancestor.

### Why This Fixes Everything

- **Stack depth**: Visual depth is bounded by screen resolution (~7-15 levels), not tree depth. `MAX_STACK_DEPTH = 16` is generous.
- **Register pressure**: 16 × 5 arrays = 80 registers. Well within GPU limits.
- **Performance**: No wasted traversal through empty intermediate nodes. Every DDA step is visually meaningful.
- **Precision**: Camera coordinates are always near zero in virtual-root space. No f32 precision issues.
- **Correctness**: No grey screen — every ray reaches a terminal or LOD cutoff within the visual depth budget.

## LOD at the Ray Level: Per-Child Representative Type

When the ray hits a `Node(id)` child and the LOD cutoff says "don't descend," the ray needs a color for that child. This is NOT a per-node "dominant block" — it's a **per-child representative block type**.

### Presence-Preserving, Not Majority Vote

The old Bevy implementation used presence-preserving downsample: if ANY child voxel in a 3×3×3 block is non-empty, the parent voxel is non-empty. The representative block type among the non-empty children is chosen by frequency.

This matters because majority vote destroys thin features:

- A tree trunk (1 voxel wide, surrounded by 26 air voxels) has majority type = Air. The trunk vanishes after one level of cascaded downsample.
- Presence-preserving: the trunk survives because any non-empty child keeps the parent non-empty. The representative type is Wood (the most common non-empty type), not Air.

For the ray marcher, each `Node(id)` in the library should cache its **representative block type** — the most common non-empty block type among all terminals in its subtree, computed bottom-up. This is O(1) per node during tree construction and gives the correct LOD color at any depth.

### Why Per-Child, Not Per-Node

The ray marcher steps through a node's 3×3×3 children via DDA. When it hits a child that's sub-pixel, it needs THAT CHILD's representative type. Different children of the same node have different types — one is stone, another is air, another is a tree. The spatial structure at the 3×3×3 level is always preserved; the LOD cutoff only collapses the structure WITHIN each child.

This means the 3×3×3 DDA always runs (trivial — at most 9 steps), and LOD only affects whether the ray descends INTO a child. The visual result is that distant terrain has blocky 3×3×3 structure at each visible level, not flat single-color nodes.

## Content-Addressing and Spatial Packing

The `scaling-deep-trees` doc previously described a "fundamental tension" between content-addressed nodes (position-independent) and LOD (position-dependent). This is real but solvable:

### The Library Is Content-Addressed; the GPU Buffer Is Spatial

The `NodeLibrary` stores nodes by content (same children = same NodeId). This is correct and essential for memory efficiency and dedup.

The GPU buffer is packed **spatially** — by camera-relative position, not by NodeId. The same NodeId appearing near and far from the camera becomes **two separate entries** in the GPU buffer:
- The near entry has full children (Node tags with buffer indices)
- The far entry is flattened to `Block(representative_type)`

This means the GPU buffer has no content-addressing conflicts. Each entry is a unique spatial position with the correct LOD for its distance. The CPU packer walks the tree spatially (BFS/DFS from virtual root with distance-based LOD cutoff) and emits one entry per visible spatial position.

Content-addressed dedup still works in the library — 50 forests sharing one oak tree NodeId store it once. But the GPU buffer may contain 50 entries for that NodeId at different LOD levels depending on distance. The buffer is O(visible nodes), not O(unique nodes).

## What Was Tried and Why It Failed

### Increasing stack depth (8 → 24)
**Why it failed:** The ray was still starting at the absolute root. More stack depth just meant more useless traversal through empty intermediate nodes. The ray reached deeper but was still doing the wrong thing.

### Distance-aware GPU packing (`pack_tree_lod`)
**Why it partially worked, partially failed:** The right idea (flatten distant nodes) but wrong execution. It packed by NodeId, not by spatial position. A NodeId first encountered near the camera got full detail everywhere; a NodeId first encountered far away got flattened everywhere. The fix: pack by position (the virtual root approach).

### Screen-space LOD cutoff in shader
**Why it worked but didn't fix the real problem:** Correctly stopped descent for distant terrain. But the ray still started at the absolute root, so nearby terrain (which CAN'T be LOD-cutoff because it fills the screen) still needed to descend through all intermediate levels.

### Uniform flattening / dominant color propagation
**Why they worked at 6 levels:** At 6 levels, the ray's total descent (6 levels) fits in the stack. The optimizations reduced wasted work within those 6 levels. At 21 levels, the optimizations don't help because the ray can't even reach the interesting nodes within the stack budget.

## Implementation Plan

### Phase 1: Virtual Root Packing
1. CPU: Walk from root to camera → find virtual root node
2. CPU: Pack virtual root's subtree with distance-based LOD flattening, by spatial position
3. CPU: Pack ancestor path (virtual root → absolute root) with siblings
4. GPU: Start ray at virtual root. Camera position in virtual-root-local coords.

### Phase 2: Ancestor Traversal
1. GPU: When ray exits the virtual root, pop to parent using the ancestor path
2. GPU: Step through parent's siblings, potentially descending into them
3. This enables sky, horizon, and distant terrain rendering

### Phase 3: Representative Block Types
1. Compute per-node representative block type (most common non-empty terminal, presence-preserving)
2. Cache on each node during tree construction / edit propagation
3. GPU uses representative type at LOD cutoff instead of grey

## Files

| File | Role |
|------|------|
| `assets/shaders/ray_march.wgsl` | GPU ray marcher — virtual root traversal, per-child LOD |
| `src/world/gpu.rs` | Camera-relative spatial packing, virtual root selection |
| `src/world/tree.rs` | Node, NodeLibrary, representative block type computation |
| `src/world/worldgen.rs` | World generator |
| `src/world/edit.rs` | CPU raycast, break/place, clone-on-write propagation |
| `src/world/state.rs` | WorldState |
| `src/renderer.rs` | wgpu renderer, uniforms, buffer upload |
| `src/main.rs` | App loop, zoom, input |
