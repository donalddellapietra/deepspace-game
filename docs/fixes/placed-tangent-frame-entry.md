# Placed Tangent Frame Entry

## What not to fix

Do not make `compute_render_frame` stop before `Child::PlacedNode` tangent blocks.

That change seems plausible because a placed tangent child is the boundary where the
renderer must apply the tangent transform. However, stopping frame descent there
breaks the current frame-local architecture: the camera can no longer enter the
deduped content subtree as its render frame, so low-depth frame transitions jump
back to the containing Cartesian parent.

## Why entering the placed node is intentional

`Child::PlacedNode` separates content identity from placement orientation:

- The child `node` is deduped terrain/content.
- The child `kind` carries the per-placement tangent transform.
- GPU packing gives each placed tangent edge its own virtual BFS kind while
  aliasing the underlying content buffer.

The render frame is allowed to descend through the placed node so that deep
camera/render work stays frame-local inside the content subtree. The tangent
boundary must be preserved by the placement kind, ribbon/GPU kind metadata, and
rotation-aware CPU transforms. Blocking descent at `compute_render_frame` throws
away the local-frame benefit and reintroduces the visible frame-entry jump.

## Correct direction

If camera motion or frame transitions break around placed tangent cells, inspect
the systems that consume the chosen frame:

- camera local projection (`WorldPos::in_frame_rot`)
- movement direction conversion (`frame_path_chain`)
- GPU ribbon entries and placed-node BFS kind preservation
- CPU raycast/edit frame transforms

The fix should keep frame descent through `Child::PlacedNode`; it should repair
the transform path that failed to honor the placed tangent boundary.
