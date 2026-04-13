# Seam Fixes Between Emit-Level Blocks

## Problem

Visible seams (thin lines) appear between blocks at zoom layer 10 and below. At layers 11-12 no seams are visible. The seams are at the boundaries between emit-level entities — separate Bevy entities with independent meshes that should tile seamlessly.

### Why layer 10?

At zoom layers 10-12 the renderer uses identical parameters (emit=11, target=12, scale=1.0). The only difference is camera height and view radius:

- **Layer 12**: radius ~32 Bevy units. Camera inside one emit node (125 units wide). Zero entity boundaries visible.
- **Layer 11**: radius ~160. ~1-2 emit nodes per axis. Few boundaries visible.
- **Layer 10**: radius ~800. ~6+ emit nodes per axis. Many boundaries visible as a grid.

The seams always exist but are invisible at higher layers because the camera doesn't see enough entity boundaries.

### Root cause

The GPU does not guarantee consistent edge rasterization across separate draw calls. Two adjacent entity meshes that share an exact edge can produce sub-pixel gaps where the `ClearColor` (dark unlit green) bleeds through, appearing as thin dark lines against the PBR-lit grass surface.

---

## Attempted Fixes

### 1. Cache BakedNode by SmallPath instead of NodeId

**Theory**: In a uniform world all emit nodes share one NodeId. The first bake (possibly at a parent edge with `None` border planes) determines the mesh for all positions. Interior nodes reuse incorrect boundary face culling, causing spurious faces at seams.

**Change**: Changed `baked: HashMap<NodeId, BakedNode>` to `baked: HashMap<SmallPath, BakedNode>` so each position gets its own correctly-bordered mesh.

**Result**: Fixed the caching correctness issue but caused massive lag — every visible path now triggers a full `flatten_children` + `bake_all_children` (hundreds of 125^3 bakes per frame) instead of sharing one cached result.

**Status**: Reverted.

### 2. Cross-parent neighbor lookup in `compute_border_slabs`

**Theory**: `compute_border_slabs` returned `None` for neighbors outside the current parent (at parent edges), causing incorrect face culling. Extending the lookup to walk up to the grandparent and find the adjacent parent's child would provide correct border data at all positions.

**Change**: Modified `compute_border_slabs` to walk ancestor chain; when a sibling is out of bounds, look up the grandparent, find the adjacent parent, and sample the opposite-edge child.

**Result**: No visible difference. The seams persisted. This confirmed the seams are not caused by border-plane correctness.

**Status**: Kept (correct behavior, just not the fix for this issue).

### 3. Cache key `(NodeId, [bool; 6])` with border existence

**Theory**: Combine the NodeId cache efficiency with border-awareness. Interior nodes share `[true; 6]` (one mesh). Edge nodes with missing borders get separate cache entries.

**Change**: Added `compute_border_existence` that cheaply checks which of 6 neighbors exist (with cross-parent lookup). Cache key is `(NodeId, [bool; 6])`. `path_key: HashMap<SmallPath, (NodeId, [bool; 6])>` tracks each path's key for incremental updates.

**Result**: Correctly separates edge vs interior meshes without per-path baking overhead. However, seams persist because the root cause is GPU rasterization gaps, not mesh content.

**Status**: Kept (correct caching, just not the fix for this visual issue).

### 4. Boundary vertex bloat in `merge_child_faces`

**Theory**: Push vertices at the mesh boundary (position 0 or 125 on any axis) outward by a tiny epsilon so adjacent entity meshes overlap slightly, sealing the sub-pixel gap.

**Change**: After merging child face data, iterate positions and shift boundary vertices by `SEAM_BLOAT = 0.005`.

**Result**: Not tested — replaced by approach 5 (simpler to implement).

**Status**: Abandoned in favor of transform-based approach.

### 5. Entity transform scale bias (`TILE_SCALE_BIAS`)

**Theory**: Scale each emit-level entity's transform slightly larger than nominal (e.g., `scale * 1.002`) so the mesh extends fractionally beyond its tile, overlapping with neighbors. The overlap is sub-pixel and uses the same material, so it's invisible.

**Change**: Added `const TILE_SCALE_BIAS: f32 = 0.002` and applied `visit.scale * (1.0 + TILE_SCALE_BIAS)` to both Transform sites (entity update and entity spawn).

**Result**: Made seams WORSE. Scaling from the mesh origin (corner 0,0,0) inflates the mesh asymmetrically — the +X/+Y/+Z edges push outward but the 0,0,0 corner stays fixed. This creates visible misalignment at tile boundaries where the scaled-up geometry of one tile doesn't match the non-scaled origin edge of its neighbor. The 0.2% distortion across 125 units (0.25 voxels) was large enough to make gaps more visible, not less.

**Status**: Reverted.

### 6. Disable shadows (`shadows_enabled: false`)

**Theory**: Overlapping co-planar boundary faces from adjacent entities could cause shadow acne — the shadow map can't resolve which face is in front, so fragments self-shadow, creating dark lines at chunk boundaries.

**Change**: Set `shadows_enabled: false` on the `DirectionalLight` in `main.rs`.

**Result**: Seams persist unchanged. Confirms the dark lines are not from shadow acne but from GPU rasterization gaps (clear color bleed).

**Status**: Reverted.

---

## Key Insights

- The `ClearColor` is already matched to grass (`Color::srgb(0.3, 0.6, 0.2)`) to hide seams, but the PBR-lit surface is brighter than the flat clear color, so gaps are still visible.
- Border slab extraction (`extract_border_slab`) was optimized from full 125^3 flat grid copies down to thin 125x125 slabs to reduce memory.
- The seams are a fundamental GPU rasterization issue, not a mesh content or face-culling bug.

### Diagnostic logging added (session 2026-04-12)

Added per-view-layer-change logging in `render_world()`:
- Fires on `zoom.layer` change (not just `emit_layer` change — important because layers 10-12 share the same emit_layer=11).
- Logs: zoom.layer, target_layer, emit_layer, scale, radius_bevy.
- Per-visit (first 10): node_id, has_children, origin, scale, path_depth, and child structure (how many children are leaves vs branches vs empty).
- Total visit count.

**Key finding from logs**: All three zoom layers (10, 11, 12) use identical render parameters:
- emit_layer=11, target_layer=12, scale=1.0
- Every emitted node has 125 leaf children, 0 branch children
- There are no "meshes of meshes" at any layer — all children are leaves with raw voxels

The ONLY difference is visit count driven by radius:
| Zoom layer | Visits | Radius (Bevy units) |
|-----------|--------|---------------------|
| 12 | 2 | 32 |
| 11 | 35 | 160 |
| 10 | 1530 | 800 |

At layer 10, 1530 separate mesh entities tile the world. The seams are between these entities. They're invisible at layer 11 (35 entities, closer camera) and layer 12 (2 entities).

### Prior `SEAM_BLOAT` in `merge_child_faces` (pre-existing)

There is already a `SEAM_BLOAT = 0.005` mechanism in `merge_child_faces()` in `mesher.rs:468-504`. It pushes boundary vertices (at position 0.0 or `grid_size` on any axis) outward by 0.005. This was added with a `grid_size: usize` parameter, but the call sites in `render.rs` do NOT pass this parameter (they pass only 2 args). This code path may not be compiling/active — needs investigation.

## Untried approaches

### A. Boundary vertex bloat (mesh-level, not transform-level)
Push boundary vertices outward in the mesh data itself (in `merge_child_faces` or `bake_faces`), not via Transform scale. Unlike the Transform approach, this only moves boundary faces and doesn't distort interior geometry. The existing `SEAM_BLOAT` code in `merge_child_faces` attempts this but may not be wired up correctly.

### B. Reduce entity count at wider zoom
Instead of emitting 1530 entities at layer 10, emit fewer larger entities. This would require baking larger meshes (e.g., emitting at depth 10 instead of 11, covering 625^3 voxels per entity). Currently infeasible due to memory/compute cost of baking 625^3 grids.

### C. Match ClearColor to PBR-lit surface
The current ClearColor is flat `srgb(0.3, 0.6, 0.2)` but the PBR-lit grass is brighter. Matching more closely would make gaps less visible (the original intent per `main.rs:35-38` comment). However, the lit surface brightness varies with sun angle, so this can't fully solve it.

### D. Backface rendering at tile boundaries
Always emit both-sided faces at tile boundaries so that even if a gap exists, the backface of the adjacent tile's boundary geometry fills it in. Would increase triangle count at boundaries.

### E. Shared-edge degenerate triangles
Add thin degenerate triangles (skirts) along tile boundaries that connect the two meshes' edges, guaranteeing watertight geometry. Common in terrain LOD systems. More complex to implement but geometrically correct.
