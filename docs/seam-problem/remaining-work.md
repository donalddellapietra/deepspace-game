# Remaining Work: Fixing AO at Child Boundaries

## The constraint

Any fix must operate **post-merge** (in `merge_child_faces` or later). Modifying AO during `bake_faces` / `compute_face_ao` changes the greedy merge output, which breaks the per-child incremental baking cache — the optimization that makes block placement instant.

## Why simple post-merge fixes don't work

- **Force to 1.0**: removes legitimate AO at corners/concavities — looks wrong
- **Clamp to 0.9**: AO levels go as low as 0.6, so 0.9 clamp still leaves visible discontinuities
- **Max-brightness matching**: only works when both sides of the boundary have vertices at matching positions, which the greedy merge doesn't guarantee on all axes

## The core problem restated

The greedy merge runs per-child (25³). At child boundaries, adjacent children produce separate quads that meet edge-to-edge. These quads can have **different AO levels** because:

1. One child's merge produces a large uniform-AO quad (all vertices at brightness X)
2. The adjacent child's merge produces a different-sized quad (vertices at brightness Y)
3. At the shared edge, X ≠ Y → visible brightness jump

The AO values themselves are correct — the problem is that the greedy merge locks each child's AO into a uniform level per quad, and adjacent children pick different levels.

## Promising approaches not yet tried

### A. Cross-child AO smoothing (post-merge vertex blending)

For each vertex at a child boundary, look at ALL nearby vertices (within 1–2 voxels on both sides of the boundary) and blend their colors. This is more robust than exact-position matching because it doesn't require vertices to be at the same position — it spatially blends across the boundary.

Implementation: build a spatial index of boundary vertex positions and colors, then for each boundary vertex, find neighbors within a small radius and set color to the weighted average.

**Pro**: handles mismatched vertex positions from different greedy merge structures.
**Con**: O(n log n) per merge, adds complexity.

### B. Run the greedy merge across children

Instead of `bake_child_faces` (per-child 25³), run `bake_faces` on the full 125³ grid. This eliminates child boundaries entirely — no seams possible.

The per-child baking is needed for incremental updates (only re-bake dirty children). But `merge_child_faces` already concatenates all children's face data. The alternative: bake faces on the full 125³ grid, but cache the FLAT GRID per-child for incremental updates (patch the flat grid when a child changes, then re-bake faces on the full grid).

**Pro**: cleanest fix, eliminates the problem entirely.
**Con**: re-baking the full 125³ on each edit is slower than re-baking one 25³ child. Need to measure if it's acceptable.

### C. Dual-grid AO

Compute AO on a separate pass that operates on the full 125³ grid (not per-child). The greedy merge still runs per-child for geometry, but AO values are sampled from the full-grid AO. This decouples AO from the per-child merge boundary.

**Pro**: AO is always continuous across child boundaries.
**Con**: requires a separate AO computation pass and storage.

### D. Shader-based AO instead of vertex colors

Move AO from vertex colors to a shader that computes it per-fragment based on the world-space position. This eliminates vertex-color discontinuities entirely because AO is computed continuously.

**Pro**: eliminates the problem at the source, looks better (smooth gradients instead of per-vertex).
**Con**: significant rendering architecture change, higher GPU cost per fragment.

### E. Force AO uniformity at child boundaries during bake, with a separate "display" cache

Keep two sets of child faces: one for the incremental cache (with original AO, used for dirty-checking and merge decisions), and one for display (with smoothed AO at child boundaries). When a child changes, re-bake the original faces (preserving cache validity) then apply boundary smoothing for the display copy.

**Pro**: preserves incremental baking.
**Con**: doubles memory for child face data.
