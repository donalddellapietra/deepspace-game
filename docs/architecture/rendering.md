# Rendering

## Model: uniform-layer camera (Option A)

The camera carries one integer: which tree layer to render. Every
visible entity in the frame is emitted at that layer. There is no
distance-based LOD mixing. Zooming the camera in or out changes the
layer for the whole frame at once.

```rust
pub struct CameraZoom {
    /// Which tree layer the camera renders. 0..=MAX_LAYER.
    /// Clamped to a UX-friendly range so the player never sees the
    /// whole world as one voxel (layer 0) or walks into sub-voxel
    /// noise (layer MAX_LAYER).
    layer: u8,
}
```

- **Zoom in** → `layer += 1` → finer nodes, smaller visible area.
- **Zoom out** → `layer -= 1` → coarser nodes, larger visible area.

The player's `Position` does not change when the camera zooms. The
camera is a pure view state.

## Target layer: sample two layers below the view layer

A naive renderer at view layer `L` would emit one entity per node at
layer `L`, textured from that node's majority-vote 25³ downsample. That
discards most of the tree — at view layer 5, every on-screen cell is
averaging `5^(MAX_LAYER - 5) × 5³` leaf voxels into one value.

Instead, the renderer emits one entity per **layer-`(L + 2)` node**
(clamped to `MAX_LAYER`). The reason is the same `(c / 5, c % 5)`
decomposition the editor uses: inside a layer-`L` node's 25³ grid, one
cell `(cx, cy, cz)` decomposes into exactly two more slot steps, so one
view cell corresponds to exactly *one* layer-`(L + 2)` subtree. The
renderer can hand that subtree's raw voxel grid straight to the greedy
mesher without involving the downsample at all — detail never gets
smoothed away by zooming out, because we're always reading from two
layers below whatever the camera is showing.

At `L = MAX_LAYER` and `L = MAX_LAYER - 1` the clamp degenerates into
"just emit at the leaf layer," which is the right behaviour: there's no
deeper layer to sample. This rule was ported from the 2D prototype's
`subtexture_25` / `subtexture_5` helpers.

## Per-frame tree walk

Rendering walks the tree from the root to the target layer, emitting
one entity per visited node at `target_layer = (zoom.layer + 2).min(MAX_LAYER)`.
The walk is a recursive descent with culling:

```
render_walk(node_id, path, layer, camera_frustum, out):
    aabb = world_aabb_for(path, layer)
    if aabb does not intersect camera_frustum:
        return
    if layer == zoom.layer:
        out.push((path, node_id, aabb))
        return
    let node = library.get(node_id)
    for slot in 0..125:
        child_id = node.children[slot]
        if child_id == EMPTY_NODE:
            continue
        render_walk(child_id, path.push(slot), layer + 1, ...)
```

For each position collected by the walk, the renderer spawns one
Bevy entity with the node's baked mesh handle, translated to the
node's world-space origin.

## Why this is cheap even at large zoom numbers

A naive walk to layer K visits up to `125^K` nodes — astronomically
too many. Two things cut the cost to something reasonable:

1. **Cell-radius culling (v1).** Proper frustum culling is deferred;
   the current walker drops any node whose AABB is more than
   `RADIUS_VIEW_CELLS` cells from the camera, measured at the current
   view layer. That is, the radius is
   `RADIUS_VIEW_CELLS * cell_size_at_layer(view_layer)` Bevy units,
   which grows by 5× per layer as you zoom out. The visible world
   covers a roughly constant number of cells regardless of zoom —
   ported from the 2D prototype's "viewport counts cells, not pixels"
   behaviour. A real frustum test will replace this once it shows up
   in a profile.
2. **Content dedup.** Multiple entities at different world positions
   reference the same `NodeId` and share the same mesh handle. GPU
   upload is paid once per unique pattern, not once per entity. An
   infinite grassland world emits thousands of entities per frame but
   they all point at one mesh.

The frame cost is dominated by entity spawn/despawn bookkeeping and
the per-node AABB check, not by mesh data.

## Mesh cache keyed on NodeId

`RenderState.meshes: HashMap<NodeId, Vec<BakedSubMesh>>` caches the
baked mesh for every `NodeId` that has ever been rendered. An entry
survives across frames *and* across zoom layer changes — the mesh is
a function of the node's voxel grid, nothing layer-specific.

Because the cache is keyed on `NodeId`, it inherits the library's
dedup guarantee for free: every grass leaf in the world shares the
same `NodeId`, so grassland bakes exactly one mesh regardless of how
many entities are on screen. The same "sub-texture cache" trick the
2D prototype uses.

v1 never evicts from this cache. Long edit sessions will grow it by
roughly one entry per unique node ever seen; not a concern yet.

## Entity management across frames

The renderer keeps a map of `path → entity, node_id` like the current
`RenderState`. On each frame:

1. Compute the set of layer-K positions in the frustum (tree walk).
2. For each position in the new set:
   - If it's already rendered and the `NodeId` matches, reuse.
   - If `NodeId` changed (ancestor edit propagation replaced this
     subtree), despawn the old entity and spawn a new one.
   - If it's brand new (camera moved into view), spawn.
3. For each position in the old set but not the new set (camera
   moved out of view, or layer changed), despawn.

When `zoom.layer` changes, the whole entity set is rebuilt from
scratch — the previous set becomes stale because every position is at
a different layer now.

## World transforms

Each emitted entity needs a `Transform` in Bevy's float world space.
Transforms accumulate as the walk descends:

- The root occupies some fixed region in Bevy space (decide the exact
  scale at implementation time — most natural is "one Bevy unit = one
  leaf voxel," but this depends on how the camera is set up).
- Each descent multiplies the child's extent by `1/5` and adds the
  slot's offset.
- At layer K, a node is `(1/5)^K` of the root's extent.

Precision is not a problem because the frustum is always local to the
camera — even at deep zoom, the visible nodes are near the camera's
world position, not out at the root's corner.

## What happens when the player edits

Edits change node content via the propagation walk in `editing.md`.
Every affected ancestor mints a new `NodeId` and gets a fresh mesh. On
the next frame, the tree walk sees the new `NodeId` at the affected
positions and respawns the corresponding entities.

The renderer does not need to be informed of edits explicitly — the
tree walk naturally picks up the new ids because it always reads the
current tree from the root on every frame.

## What rendering doesn't need

- No "dirty" flag. Meshes are always current (edit walk keeps them
  so). The renderer never asks "has this changed?"
- No explicit LOD management per region. Every node at `zoom.layer`
  renders at the same scale.
- No special handling for procedurally-generated vs edited nodes —
  they look identical from the tree's perspective.
- No material per entity — each `BakedSubMesh` carries its block type
  and the renderer picks the material from a `BlockMaterials`
  resource (same pattern as the current code).

## Deferred

- **Proper frustum culling.** v1 uses a "within N view cells of the
  camera" radius test instead of a real frustum intersection. Replace
  once it shows up in a profile.
- **Smooth zoom transitions.** Changing `zoom.layer` is currently a
  hard swap (the entity set is rebuilt from scratch when
  `target_layer` changes). A cross-fade or a short interpolation frame
  would feel nicer.
- **Background mesh baking.** Bakes currently happen on the main
  thread. For procedurally-rich worlds, move them to worker threads.
  Not needed for grassland v1.
- **Mesh cache eviction.** `RenderState.meshes` grows monotonically.
  Fine for grassland; revisit when edit-heavy workloads start pushing
  its size.
- **Partial tree walks.** The walk currently re-descends from the root
  every frame. A cached "last frame's visits" + delta-walk can limit
  work to only the nodes entering or leaving view.
