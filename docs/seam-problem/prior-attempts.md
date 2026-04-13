# Prior Attempts to Fix Seams

See [root-cause.md](root-cause.md) for what the problem actually is.

Note: attempts 1–6 were based on the incorrect assumption that seams were between emit-level entities (tile boundaries). The actual cause is AO discontinuity at child boundaries within each tile. Attempts 7–10 targeted the correct cause.

---

## Attempts targeting tile boundaries (incorrect theory)

### 1. Cache BakedNode by SmallPath instead of NodeId

**Theory**: In a uniform world all emit nodes share one NodeId. The first bake determines the mesh for all positions. Interior nodes reuse incorrect boundary face culling.

**Change**: `baked: HashMap<SmallPath, BakedNode>` so each position gets its own mesh.

**Result**: Fixed caching correctness but caused massive lag — every visible path triggers a full bake instead of sharing cached results.

**Status**: Reverted.

### 2. Cross-parent neighbor lookup in `compute_border_slabs`

**Theory**: `compute_border_slabs` returned `None` at parent edges, causing incorrect face culling.

**Change**: Walk ancestor chain to find adjacent parent's child for border data.

**Result**: No visible difference. Seams are not caused by border-plane correctness.

**Status**: Kept (correct behavior, just not the fix).

### 3. Cache key `(NodeId, [bool; 6])` with border existence

**Theory**: Combine NodeId cache efficiency with border-awareness.

**Change**: Cache key includes which of 6 borders exist. Interior nodes share one mesh, edge nodes get separate entries.

**Result**: Correct caching, but seams persist (wrong root cause).

**Status**: Kept.

### 4. Boundary vertex bloat (`SEAM_BLOAT`)

**Theory**: Push tile-edge vertices outward by epsilon so adjacent entity meshes overlap.

**Change**: In `merge_child_faces`, shift vertices at position 0 or 125 by `SEAM_BLOAT = 0.005–0.01`.

**Result**: Ineffective — the seams are not between tiles, they're within each tile at child boundaries.

**Status**: Removed.

### 5. Entity transform scale bias (`TILE_SCALE_BIAS`)

**Theory**: Scale each entity slightly larger (e.g., `scale * 1.002`) so meshes overlap at tile edges.

**Change**: `tile_transform()` shifts origin and inflates scale.

**Result**: Made seams WORSE. Scaling from the corner (0,0,0) inflates asymmetrically. Also had a spawn/update mismatch bug where spawn used raw transform but update used biased transform.

**Status**: Reverted.

### 6. Disable shadows

**Theory**: Shadow acne at mesh boundaries.

**Change**: `shadows_enabled: false` on DirectionalLight.

**Result**: No change. Seams are not shadow-related.

**Status**: Reverted.

---

## Attempts targeting AO at child boundaries (correct theory)

### 7. Disable all AO (`merge_child_faces` post-merge)

**Theory**: AO vertex colors cause the dark lines.

**Change**: Set all vertex colors to `[1.0, 1.0, 1.0, 1.0]` in `merge_child_faces` after merging.

**Result**: **Seams disappear completely.** This proved AO is the sole cause. But obviously removes all AO shading.

**Status**: Diagnostic only.

### 8. Disable AO at child boundaries only (post-merge)

**Theory**: AO discontinuities are specifically at child boundaries (every 25 voxels).

**Change**: Set vertex colors to `[1.0, 1.0, 1.0, 1.0]` only for vertices where `pos % 25 < epsilon`.

**Result**: **Seams disappear.** Confirmed child boundaries are the specific location. But setting to full white makes edges look unnaturally bright — removes legitimate AO at corners and concavities.

**Status**: Works but looks wrong.

### 9. Force uniform AO in `compute_face_ao` at child edges (pre-merge)

**Theory**: Fix AO values at the source in `bake_faces` rather than post-merge.

**Change**: Force `ao = [3,3,3,3]` for any face at a child boundary position during `compute_face_ao`.

**Result**: Fixed underground seams but **broke incremental baking** — changing which faces have uniform vs per-vertex AO changes the greedy merge output, invalidating the per-child cache. Block placement became slow again. Also didn't fix surface edge seams.

**Status**: Reverted. Cannot modify AO computation without breaking the incremental bake optimization.

### 10. Max-brightness matching at child boundary positions (post-merge)

**Theory**: Find all vertices at the same child-boundary position and set them all to the max brightness among them, so both sides of the edge agree.

**Change**: Two-pass in `merge_child_faces`: pass 1 collects max brightness per quantized position, pass 2 applies it.

**Result**: Fixed seams on ONE axis but not the other two. Adjacent children's greedy merges produce different quad structures, so boundary vertices don't always exist at matching positions on all axes.

**Status**: Reverted. Only partially effective.

### 11. Clamp minimum brightness at child boundaries (post-merge)

**Theory**: Instead of matching, just clamp the minimum brightness to 0.9 (AO_CURVE[2]) at child boundaries.

**Change**: `merged.colors[i][0].max(0.9)` for child-boundary vertices.

**Result**: Ineffective — the AO discontinuities are too dark (brightness as low as 0.6, AO level 0). Clamping to 0.9 still leaves visible jumps from 0.9 to 1.0.

**Status**: Reverted.
