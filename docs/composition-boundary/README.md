# Composition Boundary Face Waste: Analysis and Approaches

## Problem Statement

With DETAIL_DEPTH=3, the renderer uses "hierarchical mesh composition" (Approach B):

- Entity emitted at `emit_layer = target_layer - 2` (i.e., `view + 1`)
- Entity has 125 children at `view + 2`
- Each child is **independently** baked from ITS 125 children at `view + 3` (flatten into 125-cubed, greedy mesh)
- The entity's mesh is the composition of 125 pre-baked child meshes with position offsets
- Entity count stays at ~1,000 (same as old DETAIL_DEPTH=2)

**The problem**: each child is baked in isolation by `pre_bake_child()`. The greedy mesher's `get` closure in `bake_faces()` returns `None` for coordinates outside the 125-cubed grid (line 231 of `mesher.rs`). Two adjacent solid children each emit faces on their shared boundary -- invisible but wasted geometry.

### Quantifying the Waste

- Underground entities are uniform-solid. Each has 125 children.
- `is_interior_uniform()` skips children whose 6 immediate neighbors (within the parent) are uniform-same. That eliminates the 27 (3-cubed) truly interior children.
- That leaves 98 boundary children (faces of the 5-cubed cube).
- Each boundary child's pre-baked mesh includes outer faces on every exposed side. For a uniform-solid child, the `bake_faces` call produces faces on all outer surfaces where the neighbor lookup returns `None` (out of the child's own 125-cubed grid).
- A boundary child at a corner has 3 exposed outer sides, an edge has 2, a face has 1. But the critical waste is at **inter-child boundaries within the parent**: each shared face produces **two** sets of quads (one from each child), both invisible.
- Estimated: ~588 wasted quads per entity x ~550 underground entities = ~323,000 invisible quads per frame.

### What the Old System (DETAIL_DEPTH=2) Does Right

With DETAIL_DEPTH=2, `BakedNode::new_cold()` in `render.rs`:

1. Flattens 125 children's 25-cubed grids into ONE 125-cubed grid via `flatten_children()`
2. Greedy-meshes the entire 125-cubed grid with `bake_all_children()`, which calls `bake_child_faces()` using `make_get()` -- a closure over the **full flat grid**
3. The neighbor lookup for child slot X can see into adjacent child slot Y through the flat grid
4. `is_interior_uniform()` skips children entirely surrounded by same-material siblings -- zero faces
5. Only true surface faces are produced, and greedy merging consolidates them efficiently

The key insight: in DETAIL_DEPTH=2, the 125-cubed flat grid provides **cross-child neighbor information** at zero extra cost because it's a single contiguous array. This is exactly what DETAIL_DEPTH=3's composition pipeline lacks.

### Why Previous Fixes Failed

1. **Skipping uniform entities at emit time**: A surface entity (the one straddling air/ground) has a uniform-solid downsample but needs its +Y face visible. Can't skip it.

2. **Tower fast path**: Only helps when ALL 125 children share the same NodeId. Only applies deep underground where content-addressed dedup already collapsed everything. Doesn't help the ~98 boundary children of a non-tower entity.

3. **Parent flat grid for extended boundary queries**: Required building a 125-cubed grid PER CHILD PER ENTITY (to give each child's mesher access to neighbor data). 125 children x 125-cubed = 244 million voxels per entity. Even with caching, the flat grid depends on which parent the child is in, so it can't be shared across parents that happen to share a child NodeId. The content-addressing advantage is lost.

---

## Approach A: Boundary Face Mask (Post-Bake Culling)

### Mechanism

After pre-baking each child in isolation (as today), add a **post-processing pass** at composition time that removes boundary faces between adjacent solid children.

For each pair of adjacent children within the parent's 5-cubed grid, check whether a face between them should be culled:

1. In `compose_node()`, after collecting `children_faces`, iterate over all 300 internal face-pairs (5x5x5 grid, 3 axes, (5-1) boundaries per axis per row = 300 pairs).
2. For each pair (child A, child B) sharing a face on axis `d`:
   - If both children are non-empty AND both are uniform-same-material: the entire shared face can be culled from both children's `FaceData`.
   - If one or both are mixed: need per-voxel boundary check. For each of the 125x125 = 15,625 voxels on the shared face, check if child A's boundary voxel AND child B's boundary voxel are both solid. If so, both face quads at that position are wasted.
3. Filter the `FaceData` to remove the identified quads before GPU upload.

**Optimization**: For uniform-uniform pairs (the underground case), skip the per-voxel check entirely. Just remove all quads from both children's FaceData that lie on the shared face. This is the common case underground.

For mixed children at the surface, the per-voxel check is needed but only for the ~6 surface entities (not the ~550 underground ones).

### Per-Frame and First-Frame Cost

- **First frame**: Same as today for pre-baking. The post-processing pass adds: for each entity, iterate 300 internal boundaries. For uniform-uniform pairs (underground): O(1) check per pair. For mixed pairs (surface): scan up to 15,625 voxels per pair, but only ~6 entities have mixed children at the surface.
- **Per-frame (steady state)**: Post-processing is part of `compose_node()`, which runs once per unique NodeId and is cached in `render_state.baked`. So it only runs on first bake, not every frame.
- **Estimated overhead**: Negligible for uniform pairs. For mixed surface entities, ~6 entities x ~50 mixed-mixed boundaries x 15,625 voxels = ~4.7M voxel checks. At branch-free u8 comparisons, this is ~5ms worst case. In practice most mixed pairs are air-solid (one side empty, no quads to cull), so it's much less.

### Cacheability

- Pre-baked child meshes remain content-addressed and cached by NodeId (same as today).
- The culling pass runs inside `compose_node()`, whose result is cached per parent NodeId in `render_state.baked`.
- **Key problem**: pre-baked meshes are shared across parents. If child C appears in parent P1 (where its +X neighbor is solid) and parent P2 (where its +X neighbor is air), the pre-baked mesh for C has boundary faces in both cases. The culling must happen per-parent, not per-child. This means we can't mutate the cached pre-baked `FaceData` -- we need to **filter during composition**.

**Implementation detail**: Instead of removing quads from `FaceData` (expensive, requires rebuilding index arrays), add a **face mask** bitmap to each child's composition. During `compose_children_meshes()`, skip quads whose face normal points toward a known-solid neighbor child. This can be done by checking each quad's position against the child's bounding box and the neighbor child's classification.

### Correctness

- **No visual holes**: Only culls faces between two solid voxels. If one side is air, the face is kept. If terrain noise creates a hollow pocket at the boundary, the mixed child's voxels correctly report air, and the face is kept.
- **Works with terrain noise**: Terrain noise affects which voxels are solid at the leaf level. By the time we reach composition, the pre-baked `FaceData` already reflects the actual voxel content. The boundary check operates on the actual solid/air state, not the downsample.
- **Edge case -- AO**: AO values at boundary voxels are computed during pre-baking without neighbor info, so they use `is_solid = false` for out-of-bounds. This means even faces that survive culling have slightly wrong AO at child boundaries. This is a visual artifact but not a correctness issue (no holes).

### Implementation Complexity

**Medium**. Requires:
- Modifying `compose_children_meshes()` to accept neighbor classification info
- Adding a per-quad filter during composition (check position against boundary, check neighbor classification)
- The quad-position-to-boundary-face mapping is non-trivial: need to determine which face of which child each quad belongs to by inspecting its position and normal within the child's coordinate range

**Risk**: Filtering quads by position requires floating-point comparisons against child boundary coordinates. The greedy mesher produces merged quads that may span multiple voxels -- these need careful handling to determine if they lie exactly on a boundary.

---

## Approach B: Boundary Slab Exchange (One-Voxel-Deep Neighbor Sharing)

### Mechanism

Give each child's mesher access to **one voxel-deep slabs** of its 6 neighboring children within the parent, so it can make correct face-culling and AO decisions at boundaries without needing the full 125-cubed neighbor grid.

For a child at slot (sx, sy, sz) within the parent's 5-cubed grid, when meshing face direction +X (for example), the mesher needs to know whether voxel (125, y, z) is solid. That voxel lives in the neighbor child at slot (sx+1, sy, sz), at local coordinate (0, y, z).

**The slab**: for each child, extract the 6 boundary slabs from its 125-cubed baked grid:
- +X slab: voxels at x=0 of the +X neighbor (125 x 125 = 15,625 voxels)
- -X slab: voxels at x=124 of the -X neighbor
- (similarly for Y and Z)

But wait -- the children are at `view + 2`, and their internal grids are 125-cubed (the flattened result of THEIR 125 children at `view + 3`). We don't cache the flat grid per child; `pre_bake_child()` builds it, meshes it, and discards it.

**Revised mechanism**: Cache a **boundary descriptor** per child NodeId alongside the pre-baked `FaceData`. The boundary descriptor is 6 slabs, each 125x125 voxels (or a compressed version). For uniform children, the descriptor is trivially "all voxel V on every face."

During composition, `pre_bake_child()` returns both `FaceData` and the boundary descriptor. Then, before composing, we could re-bake boundary children with extended `get` closures that consult neighbor slabs. But this re-bake is exactly what we want to avoid.

**Better variant**: Don't re-bake. Instead, modify `pre_bake_child()` to accept optional neighbor slabs, and use them in the `get` closure to answer queries for coordinates just outside the child's [0, 125) range. This way, the child's mesh is baked once with correct boundary information.

**Problem**: This breaks content-addressed sharing. Child C at slot (2,2,2) in parent P1 might have different neighbors than child C at the same slot in parent P2. The pre-baked mesh for C depends on its neighbors, so it's no longer keyed solely by C's NodeId.

**Solution**: Two-phase baking:
1. **Phase 1 (cacheable)**: Pre-bake each child's interior faces (everything not on the outer 125-cubed boundary). These are independent of neighbors and cacheable by NodeId.
2. **Phase 2 (per-parent)**: For each boundary child, compute the boundary slabs from its neighbors, and bake only the boundary face-layer (the outermost voxel ring of each child). This is a thin shell: 6 faces x 125x125 voxels each, but many are already handled by interior meshing.

This two-phase split is complex and error-prone. The greedy mesher fundamentally wants to merge across the interior/boundary divide.

**Simpler variant (recommended if choosing this approach)**: Accept the cache miss. Key pre-baked meshes by `(NodeId, neighbor_classification)` where `neighbor_classification` is a 6-tuple of `ChildClass` values for the 6 neighbors. Most underground children have the same neighbor classification (all Uniform-same), so dedup still works well. Only surface entities produce unique keys.

### Per-Frame and First-Frame Cost

- **First frame**: Same pre-baking pipeline, but each child's `get` closure now extends 1 voxel beyond the [0, 125) range using neighbor slabs. Extracting a 125x125 slab from a neighbor's flat grid: ~15,625 byte copies per slab, 6 slabs per child, 125 children = ~11.7M byte copies per entity. At memcpy speeds, ~1ms per entity, ~1s for ~1000 entities on first frame.
- **Steady state**: Cached by `(NodeId, neighbor_classification)`. Cache hits eliminate re-baking. Underground entities all share the same key, so only ~6 surface entities need unique bakes.
- **Slab extraction cost**: Need to either cache the 125-cubed flat grids per child (15,625 x 125 = ~1.95M bytes each, 125 children = ~244M per entity -- too much), or re-flatten just the boundary slab from the child's 25-cubed children (cheaper, ~15K per slab).

**Slab from downsample shortcut**: For uniform children, the slab is trivially all-same-voxel. For mixed children, the boundary slab at full resolution requires descending into the child's children at the boundary. If the child's child at the boundary edge is uniform, the slab for that 25x25 region is trivial. Only truly mixed boundary children of mixed children need per-voxel extraction.

### Cacheability

- Pre-baked `FaceData` keyed by `(NodeId, [ChildClass; 6])` -- the 6 neighbor classifications.
- Underground: all neighbors are Uniform(stone), so all children share one cache key per NodeId. Excellent dedup.
- Surface: neighbors vary. ~6 entities x 125 children = ~750 unique keys. Manageable.

### Correctness

- **No visual holes**: Each child's mesher sees the actual voxel content at the boundary. Faces are emitted only where one side is solid and the other is air.
- **Correct AO**: AO computation at boundaries uses the neighbor slab data, producing correct lighting.
- **Works with terrain noise**: The slab data reflects the actual leaf-level voxel content (post-noise), not a downsample. As long as the slab extraction reads from the correct layer's voxel data, terrain features are preserved.

### Implementation Complexity

**High**. Requires:
- Modifying `pre_bake_child()` to accept 6 optional neighbor slabs
- Modifying `bake_faces()` `get` closure to consult slabs for out-of-range queries
- Building slab extraction from child's children at the boundary
- Changing the pre-bake cache key from `NodeId` to `(NodeId, [ChildClass; 6])`
- Handling the slab memory lifecycle (extract, pass, drop)
- Testing correctness at all boundary configurations (corner children with 3 slabs, edge with 2, face with 1)

**Risk**: Slab extraction for mixed children is the most complex part. Each child's boundary voxels live in specific grandchildren (the child's children at the face of its 5-cubed grid). Extracting a 125x125 slab means reading from up to 25 grandchildren's voxel grids. This is feasible but fiddly.

---

## Approach C: Hybrid Flatten (Flatten at Composition Level with Lazy Boundary Regions)

### Mechanism

Return to the DETAIL_DEPTH=2 strategy of flattening into a single contiguous grid, but at the composition level. Instead of each child being pre-baked independently, flatten ALL 125 children's 125-cubed grids into one 625-cubed grid, then greedy-mesh the whole thing.

**The problem**: a 625-cubed grid is 244 million voxels (244 MB). Too large to build or mesh in a frame.

**The hybrid**: Don't actually build the full 625-cubed grid. Instead, use a **virtual grid** backed by lazy per-child lookups:

```
get(x, y, z) -> Option<u8>:
    child_slot = (x/125, y/125, z/125)  // which of 125 children
    local = (x%125, y%125, z%125)        // within that child
    return child_flat_grids[child_slot][local]
```

This requires keeping each child's 125-cubed flat grid in memory (~1.95M per child, 125 children = 244M per entity). That's too much.

**Optimization**: Only keep the flat grids for non-empty, non-interior-uniform children. Underground, that's ~98 boundary children. Still ~191M. Too much.

**Further optimization**: Don't keep flat grids at all. Use the tree structure directly:

```
get(x, y, z) -> Option<u8>:
    child_slot = (x/125, y/125, z/125)
    local = (x%125, y%125, z%125)
    grandchild_slot = (local_x/25, local_y/25, local_z/25)
    voxel = (local_x%25, local_y%25, local_z%25)
    return library[children[child_slot]].children[grandchild_slot].voxels[voxel]
```

This is a tree lookup per voxel query. The greedy mesher calls `get` for every voxel in the region plus every neighbor check. For a 625-cubed grid, that's ~244M primary lookups + ~1.5B neighbor lookups = ~1.7 billion tree traversals. Even at 2 levels deep (parent -> child -> grandchild -> voxels), each traversal is 2 HashMap lookups + array indexing. At ~20ns per lookup, that's ~34 seconds. Way too slow.

**Real optimization -- two-level virtual flat grid with boundary caching**:

Build a **sparse 625-cubed virtual grid** that:
1. For each child, builds the 125-cubed flat grid only once (cached by NodeId in `pre_baked`)
2. The `get(x, y, z)` closure indexes into the appropriate child's cached flat grid

Memory: only the currently-meshed children's flat grids need to be in memory. Using the existing `pre_bake_child` path but caching the flat grid alongside the FaceData.

**Wait**: this IS feasible. Each child's flat grid is 125-cubed = 1.95M bytes. But we don't need all 125 simultaneously -- we need at most 2 adjacent children's flat grids at a time for boundary queries. However, the greedy mesher processes one depth-slice at a time across the full 625-cubed grid, which touches all 5 children along that axis.

Actually, we should step back. The mesher in `bake_faces()` iterates `d` from 0 to `size-1` (here 0 to 624), and for each slice, iterates all `u x v` in that slice. For slice `d=124` (the +X boundary of child 0), it needs neighbor data from `d=125` (the -X face of child 1). The `get` closure routes to the correct child's flat grid.

**Memory**: at any moment, only the flat grids of children that are currently being accessed are needed. But the mesher's random access pattern (AO queries reach +-1 in all directions) means that for any voxel near a child boundary, we need up to 8 neighboring children's flat grids. Worst case, all 125 flat grids are accessed. Total: 125 x 1.95M = 244M. This is the same 244M problem.

**Fallback to the approach that actually works**: Don't greedy-mesh the full 625-cubed. Instead, keep the per-child meshing (as today), but give the mesher access to neighboring children's flat grids for boundary queries only. This is Approach B, reformulated.

### Revised Approach C: Keep Per-Child Meshing, Cache and Share Flat Grids

Cache each child's 125-cubed flat grid by NodeId in the existing `pre_baked` cache (alongside FaceData). During composition:

1. For each non-skippable child, ensure its flat grid is cached.
2. Build a `get` closure that, for coordinates outside [0, 125), routes to the neighbor child's cached flat grid.
3. Re-bake the child using this extended `get`.
4. Cache the result by `(NodeId, neighbor_node_ids_on_boundary)`.

Memory per child flat grid: 1.95M bytes. Total cached: one per unique non-leaf NodeId that appears as a composition child. Underground, content-addressing means there might be only ~10-20 unique NodeIds among all composition children, so ~20-40M of flat grids. Surface entities add more, but the total is bounded by the number of unique NodeIds in the render set.

**This is actually Approach B with flat-grid-based slabs.** The slab is the neighbor's flat grid, not a 1-voxel-deep extraction. The advantage: no slab extraction code, the mesher's `get` closure simply indexes into the neighbor's cached flat grid for out-of-range queries.

### Per-Frame and First-Frame Cost

- **First frame**: Build and cache flat grids for all unique child NodeIds (~50ms for 20 unique nodes, each requiring flatten of 125 x 25-cubed). Then, for each entity, bake 125 children with extended get closures (~2ms per child x 98 boundary children = ~196ms per entity). With ~1000 entities but heavy dedup (most underground entities share the same composition), effective cost is ~10-20 unique compositions x 196ms = 2-4 seconds.
- **Steady state**: All cached. Zero re-bake cost unless the world is edited.
- **Memory**: ~20-40M for flat grids (acceptable).

### Cacheability

- Flat grids: cached by child NodeId. Content-addressed, excellent dedup.
- FaceData: cached by `(NodeId, [Option<NodeId>; 6])` -- the 6 neighbor NodeIds. Underground: all same, so one entry per NodeId. Good dedup.

### Correctness

- Same as Approach B. Full neighbor access means correct face culling and correct AO.
- Works with terrain noise: flat grids contain actual leaf-level voxel data.

### Implementation Complexity

**Medium-High**. Requires:
- Adding flat grid caching alongside FaceData in `pre_baked`
- Modifying the mesher's `get` closure to route out-of-range queries to neighbor flat grids
- Changing the pre-bake cache key
- Managing flat grid memory (eviction when NodeId leaves the render set)

---

## Comparison Table

| Criterion | A: Boundary Face Mask | B: Boundary Slab Exchange | C: Hybrid Flatten + Cache |
|---|---|---|---|
| **Eliminates boundary waste** | Yes (post-hoc) | Yes (at bake time) | Yes (at bake time) |
| **Correct AO at boundaries** | No (AO still wrong) | Yes | Yes |
| **Implementation complexity** | Medium | High | Medium-High |
| **Memory overhead** | None | ~6 x 125x125 bytes per boundary config | ~2M per unique child NodeId |
| **Cache-friendly** | Yes (existing cache structure) | Needs new cache key | Needs new cache key + flat grid storage |
| **First-frame cost** | Same as today + negligible post-process | Same + slab extraction (~1ms/entity) | Same + flat grid build (~50ms total) |
| **Steady-state cost** | Same as today | Same as today | Same as today |
| **Works with terrain** | Yes | Yes | Yes |
| **Risk of visual holes** | Low (conservative: only culls provably-occluded) | Very low (correct by construction) | Very low (correct by construction) |
| **Greedy merge quality** | Same as today (no cross-child merging) | Same as today | Same as today |

---

## Recommendation: Approach A (Boundary Face Mask)

### Why Approach A

1. **Lowest implementation risk**. The existing pipeline doesn't change. Pre-baking stays content-addressed by NodeId. Composition stays the same. We add a filtering step that removes known-wasted geometry.

2. **Solves the actual problem**. The ~323,000 wasted quads are overwhelmingly from uniform-uniform child boundaries underground. For a uniform child, ALL quads on a given face are on the boundary. If the neighbor is also uniform-same, ALL of those quads are wasted. The uniform-uniform check is O(1): if `child_class[slot] == Uniform(v)` and `child_class[neighbor_slot] == Uniform(v)`, remove all quads from child `slot` whose normal points toward `neighbor_slot`. This covers ~95% of the waste with zero per-voxel work.

3. **Doesn't break the cache model**. Pre-baked FaceData remains keyed by NodeId. The filtering happens during composition, which is already per-parent. No new cache keys needed.

4. **AO imperfection is acceptable**. At child boundaries, AO is slightly wrong (neighboring voxels appear as air during AO computation). This is a subtle lighting artifact at the seam between two 125-cubed regions. At DETAIL_DEPTH=3 zoom levels, these seams are far from the camera and the AO error is invisible. If it later becomes a problem, it can be fixed independently.

5. **Incremental path to Approach B/C**. If boundary AO becomes visible at close zoom, the filtering approach can coexist with selective boundary re-baking for surface entities only. The underground entities (which are the bulk of the waste) would still use the fast uniform-uniform filter.

### Implementation Sketch

In `compose_node()`, after collecting `children_faces` (line 521 of render.rs), add:

1. Build a `cull_mask: Vec<[bool; 6]>` for each child slot. `cull_mask[slot][face_dir] = true` means "all quads on this face can be removed."
2. For each internal boundary pair: if both children are `Uniform(v)` with the same `v`, set `cull_mask` for both children on the shared face.
3. Pass `cull_mask` to a modified `compose_children_meshes()` that, when copying quads from a child's FaceData, skips quads whose normal matches a culled face direction.

For the uniform-uniform case (95%+ of underground waste), this is ~10 lines of code in the compose path. No mesher changes. No cache key changes. No memory overhead.

For mixed-boundary children (surface entities), the current waste is small (~6 entities x ~20 mixed-mixed boundaries x ~4 wasted quads per boundary = ~480 quads). This is negligible and doesn't need to be optimized in v1.

### Why Not B or C

- **Approach B** is the theoretically cleanest solution but requires significant plumbing: slab extraction from grandchildren's voxel grids, modified mesher get closures, new cache key types, and careful handling of the 6-neighbor slab lifecycle. The implementation surface area is large and the benefit over Approach A is marginal (correct AO at boundaries that are barely visible).

- **Approach C** is Approach B with flat-grid caching instead of slabs. It's slightly simpler than B (no slab extraction, just cache the full flat grid and index into it) but adds ~20-40M of memory for flat grid storage. The complexity is similar to B, and the memory cost is non-trivial on constrained platforms.

Both B and C solve a problem (boundary AO) that Approach A doesn't, but that problem is not the problem being reported. The reported problem is ~323K wasted invisible quads per frame, and Approach A eliminates them with minimal code change.
