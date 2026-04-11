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

## Per-frame tree walk

Rendering walks the tree from the root to the target layer, emitting
one entity per visited node at `zoom.layer`. The walk is a recursive
descent with frustum culling:

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

1. **Frustum culling.** At any layer, only the nodes inside the
   camera's view frustum survive the walk. A typical frustum contains
   a bounded number of layer-K nodes roughly proportional to
   `frustum_volume / node_size_at_K`. For a standard camera, that's
   ~hundreds to low thousands of nodes per frame.
2. **Content dedup.** Multiple entities at different world positions
   reference the same `NodeId` and share the same mesh handle. GPU
   upload is paid once per unique pattern, not once per entity. An
   infinite grassland world emits thousands of entities per frame but
   they all point at one mesh.

The frame cost is dominated by entity spawn/despawn bookkeeping and
the per-node frustum check, not by mesh data.

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

- **Smooth zoom transitions.** Changing `zoom.layer` is currently a
  hard swap. A cross-fade or a short interpolation frame would feel
  nicer; skip until the base renderer works.
- **Background mesh baking.** Bakes currently happen on the main
  thread. For procedurally-rich worlds, move them to worker threads.
  Not needed for grassland v1.
- **Partial frustum walks.** The walk currently re-descends the tree
  from the root every frame. A cached "last frame's frustum" +
  delta-walk can limit work to only the nodes entering or leaving
  view. Skip until the base walk shows up in a profile.
