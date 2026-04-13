# DETAIL_DEPTH=3 & Hierarchical Mesh Composition: Findings

## What We Built

Added a third layer of sub-cell detail (`DETAIL_DEPTH=3`) to the voxel renderer. At view layer `L`, the renderer now shows detail from layer `L+3` instead of `L+2`, giving 125 voxels per view cell per axis (up from 25).

The implementation uses **hierarchical mesh composition** (Approach B):
- Entity emits at `view+1` (~1,000 entities, same as old system)
- Each entity's 125 children are independently pre-baked (flatten grandchildren into 125³, greedy mesh)
- Entity mesh = composition of 125 pre-baked child meshes with position offsets
- Falls back to single-level flatten for view layers near MAX_LAYER where 3 layers aren't available

## Performance Issues Encountered (In Order)

### 1. Entity Count Explosion (200k entities)

**Symptom**: First naive approach (LOD cascade with fine/coarse emit layers) produced ~200k entities at layer 9.

**Cause**: Emitting at `target-1` (the fine layer) put one entity per view cell. With radius 32 view cells, that's 64³ × π/6 ≈ 137k entities.

**Fix**: This led us to Approach B (composition) — emit at `view+1` with ~1,000 entities, compose finer detail into each entity's mesh.

### 2. Bevy Coordinate Range (5× larger)

**Symptom**: Looking around was laggy at layer 9 and below.

**Cause**: `anchor.norm` was set to `scale_for_layer(target)`. With `target = view+3` instead of `view+2`, norm was 5× smaller, making Bevy coordinates 5× larger (~4000 units instead of ~800). Bevy's GPU frustum culling became less effective — entities behind the camera still clipped the frustum.

**Fix**: Changed norm to `scale_for_layer(view+1)` via `norm_for_layer()`. Entity scale compensates: `scale = scale_for_layer(target) / norm`. Bevy coordinates stay bounded at ~160 units regardless of DETAIL_DEPTH.

**Key learning**: The norm controls the Bevy coordinate range. Deeper target layers → smaller norm → larger coordinates → worse GPU culling. Always normalize by the emit layer, not the target.

### 3. Underground Boundary Face Waste (323k invisible quads)

**Symptom**: Still laggy at layer 9. Each underground entity produced ~588 wasted quads.

**Cause**: Each child is baked in isolation — its mesher returns `None` for out-of-bounds, generating faces on ALL outer surfaces. Two adjacent solid children produce double faces at their shared boundary. With ~550 underground entities × 98 boundary children × 6 outer quads = ~323k invisible quads per frame.

**Failed fixes**:
- Skipping all uniform entities at emit time → removed the ground (surface entity has uniform-solid downsample but needs visible +Y face)
- Tower fast path (box mesh for entities with all-identical children) → produced visible floating cubes underground, and didn't generalize to terrain with material layers
- Extended boundary queries using parent flat grid → infinitely slow (baking 125³ per child per entity, not cacheable since mesh depends on neighbors)

**Working fix**: **Boundary Face Mask (Approach A)** — post-bake culling at composition time. For each pair of adjacent children within the parent's 5³ grid: if both are `Uniform(v)` with the same non-empty `v`, mark their shared face for culling. During `compose_children_meshes`, skip quads whose normal matches a culled face. This is O(1) per boundary pair for the uniform-uniform case (95%+ of underground waste). See `docs/composition-boundary/README.md` for the full analysis of three approaches considered.

### 4. Per-Frame Library Lookups (the final culprit)

**Symptom**: Slow at ALL layers including layer 12 (where composition isn't even active). Slower than the original code before any of our changes.

**Cause**: We added `world.library.get(node_id).map_or(false, |n| n.uniform_empty)` at emit time in the walk to skip uniform-empty nodes. This runs **every frame for every entity** (~1100 HashMap lookups). Each lookup touches a ~15KB Node struct (the 25³ voxel grid), thrashing the CPU cache. The original code had no library access in the walk's emit path.

**Fix**: Removed the emit-time check entirely. Empty entities just bake to zero-face meshes (cached after first frame, zero ongoing cost). The bake cache means the library is only accessed on the FIRST encounter of each NodeId, not every frame.

**Key learning**: **Never add per-frame HashMap lookups to the hot walk path.** The walk runs every frame and touches every visible entity. Even a "cheap" lookup becomes expensive × 1100 entities × 60fps. Move classification to bake time (cached) instead of walk time (per-frame).

## Architecture Summary

```
DETAIL_DEPTH=2 (old):
  emit at view+1 → flatten 125 children (25³ each) into 125³ → greedy mesh
  One flat grid = full neighbor info = only surface faces

DETAIL_DEPTH=3 (new, composition):
  emit at view+1 → pre-bake 125 children (each flattens ITS 125 children into 125³)
  → compose with position offsets + boundary face cull mask
  → result cached per entity NodeId

Boundary cull mask:
  For each adjacent pair of Uniform(v) children with same v:
    mark shared face for culling on both sides
  During composition: skip quads whose normal matches a culled face
  Result: ~95% of underground boundary waste eliminated

Norm fix:
  anchor.norm = scale_for_layer(view+1)  // NOT target
  entity_scale = scale_for_layer(target) / norm
  Keeps Bevy coordinates bounded regardless of DETAIL_DEPTH
```

## Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `DETAIL_DEPTH` | 3 | Layers of sub-cell detail |
| `RADIUS_VIEW_CELLS` | 32.0 | View distance in cells |
| `BRANCH_FACTOR` | 5 | Children per axis per node |
| `NODE_VOXELS_PER_AXIS` | 25 | Voxels per axis per node |
| `MAX_LAYER` | 12 | Tree depth |

## Files Changed

| File | Changes |
|------|---------|
| `src/world/tree.rs` | Added `DETAIL_DEPTH`, `uniform_empty` flag on Node |
| `src/world/view.rs` | `target_layer_for` uses `DETAIL_DEPTH`, added `norm_for_layer` |
| `src/world/render.rs` | Composition pipeline (`compose_node`, `pre_bake_child`), simplified walk, `entity_scale` |
| `src/model/mesher.rs` | `compose_children_meshes` with cull masks, `merge_child_faces_raw`, `build_sub_meshes`, `bake_faces_raw` |
| `src/player.rs` | Norm uses `norm_for_layer` |
| `src/camera.rs` | Zoom transition uses `norm_for_layer` |
| `src/editor/save_mode.rs` | `resolve_emit_node_at_lp` adjusts for composition emit depth |
