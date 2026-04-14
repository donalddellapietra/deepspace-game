# Editing and Block Targeting

## Block Targeting via GPU Readback

The GPU ray marcher already computes exactly which block every pixel hits. Instead of running a separate CPU raycast, we read the hit information back from the shader.

The ray march shader writes to a secondary output buffer (in addition to the color framebuffer):

```wgsl
struct HitInfo {
    node_path: array<u32, 63>,  // path to the hit node
    depth: u32,                  // depth of hit
    child_index: u32,            // which child was hit (0..26)
    face: u32,                   // which face (0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z)
    distance: f32,               // ray distance to hit
}
```

On click, the CPU reads back the `HitInfo` for the pixel under the crosshair. This gives the exact node and child that was hit, plus which face — everything needed to break or place a block.

This is one GPU readback per click (not per frame). Latency is 1-2 frames (the GPU pipeline delay). For a click action, this is imperceptible.

### Highlight / Crosshair Preview

For the block highlight (the wireframe showing which block the crosshair is pointing at), the readback needs to happen every frame. Two options:

1. **Write hit info for the center pixel only** — a 1×1 readback, very cheap. The shader checks `if (pixel == screen_center)` and writes to a small buffer.
2. **Compute the highlight on GPU** — the shader detects when it's rendering the targeted block and adds a wireframe overlay directly. No readback needed.

Option 2 is better — zero CPU involvement for the highlight.

## Edit Mechanics

When the player clicks to break a block:

1. Read the `HitInfo` for the crosshair pixel. This gives the path to the node containing the targeted child, plus the child index.
2. Clone the node's 27 children array.
3. Replace `children[child_index]` with `Child::Empty`.
4. Insert the new children array into the library (content-addressed — may dedup to an existing node).
5. The parent now has a different child at this slot → new parent node → new grandparent → ... → new root. This propagation is O(63) node insertions, each just a 27-element array copy + hash. Microseconds.
6. Update the world's root NodeId.
7. Next frame, the ray marcher traverses the updated tree. The broken block is gone.

No mesh re-baking. No cache invalidation. The tree changed, the ray marcher sees the change, done.

## Placing Blocks

Same flow, but the target is the empty cell adjacent to the hit face:

1. Read `HitInfo` — gives the hit block's path + face.
2. Compute the adjacent cell on the hit face (step one child in the face direction).
3. If the adjacent cell is `Empty`, replace it with `Block(selected_type)`.
4. Propagate upward to root.

## Edit Scale

The player edits at their current gameplay layer. At layer 9, they break individual blocks (~0.74m). At layer 12, they break tree-sized structures (~20m). At layer 6, they break furniture-sized objects (~2.7cm).

The edit code is identical at every layer — it's always "replace one child in one node." The layer determines the scale of what "one child" means.

## Undo

Content addressing makes undo trivial. The old root NodeId still references a valid tree (nodes are refcounted, not deleted until unreferenced). To undo:

1. Push the current root onto an undo stack.
2. On undo, pop the previous root and set it as the current root.
3. Ref_inc the restored root, ref_dec the abandoned root.

The old tree's nodes still exist in the library (they haven't been evicted because the undo stack holds a reference). Undo is O(1) — a single pointer swap.
