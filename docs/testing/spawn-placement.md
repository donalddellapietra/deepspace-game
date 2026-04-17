# Spawning Next to Objects

Placing the camera so an object is visible at spawn is surprisingly subtle because the world tree is recursive and the object's position depends on *where it lives in the tree*, not just the voxel indices in the model file.

This doc records the working recipe, derived while building the `--vox-model` preset for imported `.vox`/`.vxs` content. Use it anytime you're adding a new `WorldPreset` or positioning a camera at a fresh content-bearing cell.

## The core mistake

Naive spawn: `WorldPos::from_frame_local(&Path::root(), [2.5, 1.5, 2.5], 2).deepened_to(3)` + yaw pointing at `(1.5, 1.5, 1.5)`. Almost always **shows empty sky**, because:

1. The model isn't physically at the world center.
2. Models wrapped at `CENTER_SLOT` occupy a *fraction* of their wrap cell — the rest of the cell is empty air.
3. The camera's ray direction may never cross the cells that contain the model.

## The working recipe

### Step 1: compute the model's world-space bounds

When you place a voxel model inside a larger tree via wrapping:

```rust
// Each wrap nests the model inside CENTER_SLOT of a new 27-child node.
let wrap_size = 3.0 * (1.0 / BRANCH as f32).powi(wraps as i32);
let wrap_origin = 1.5 - wrap_size / 2.0;   // world-space origin of the wrap cell

// The model's own tree is padded to the next power of 3.
let padded = BRANCH.pow(model_depth as u32) as f32;
let extent_x = wrap_size * (model.size_x as f32 / padded);
let extent_y = wrap_size * (model.size_y as f32 / padded);
let extent_z = wrap_size * (model.size_z as f32 / padded);

// World-space bounding box of the actual filled voxels:
let x_min = wrap_origin;
let x_max = wrap_origin + extent_x;
// (same for y, z)

let center_x = wrap_origin + extent_x / 2.0;
let center_y = wrap_origin + extent_y / 2.0;
let center_z = wrap_origin + extent_z / 2.0;
```

**Always print this out.** The extents often surprise you — a "243-cube" Soldier voxelization is actually `730×724×176` (humanoid) which after padding to `3^7=2187` lands in `[1.000..1.334, 1.000..1.331, 1.000..1.081]` of the world. Thin slab, not a cube.

### Step 2: place the camera to guarantee ray–model intersection

The most reliable spawn:

```rust
let cam_x = center_x.clamp(0.05, 2.95);                 // inside model's x footprint
let cam_y = (wrap_origin + wrap_size + 0.5).min(2.95);  // outside the wrap cell
let cam_z = center_z.clamp(0.05, 2.95);                 // inside model's z footprint
let yaw = 0.0;
let pitch = -1.5;  // nearly straight down (−π/2)
```

Why each part matters:

- **`cam_x`, `cam_z` inside the model's x,z footprint**: when the camera pitch is straight down, rays go down the y axis. They need to pass through cells that actually contain model voxels.
- **`cam_y` outside the wrap cell** (`> wrap_origin + wrap_size`): puts the camera in a root cell one level above the wrap. Rays traverse empty air, then enter the wrap cell, then descend into the model tree. This avoids the "camera is inside the model tree but outside the voxel footprint" case, which is the #1 source of empty-view surprises.
- **Pitch ≈ `-1.5`** (almost straight down): humanoid GLBs like Soldier/Fox have a thin z axis (depth). Top-down views give the clearest silhouette in one frame.
- **Yaw `0.0` is fine** under straight-down pitch — the up vector is what orients the image, and it's unambiguous.

### Step 3: anchor depth determines how much of the world is in-frame

`spawn_pos.deepened_to(N)` sets the camera's anchor depth. The renderer picks a render frame at `render_depth = anchor_depth - K` (where `K = RENDER_FRAME_K = 3` today).

For a "see the whole model from outside" shot:

```rust
let spawn_pos = WorldPos::from_frame_local(&Path::root(), [cam_x, cam_y, cam_z], 2)
    .deepened_to(3);
// anchor_depth = 3 → render_depth = 0 → frame = world root.
// The whole [0,3)³ root is in the render frame. No ribbon.
```

For a "see the model from inside" / "zoomed-in exploration" shot, raise the anchor depth, but verify the render frame still contains the model — otherwise you'll need the ribbon to pop out, and that pathway has its own quirks (see `docs/architecture/rendering.md`).

## Debugging checklist when the model is invisible

When you're staring at blue sky and expected to see content:

1. **`grep "render_harness_shader"` with `--shader-stats`**.
   - `hit_fraction > 0` means the scene has renderable content the camera can see. You just might not be looking at it — tweak yaw/pitch.
   - `hit_fraction == 0` means no ray ever found content. The camera placement is wrong (or the content isn't where you think).

2. **Print the model's world bounds** from the bootstrap. Make sure the camera is *on the correct side* of those bounds given the pitch.

3. **Check `avg_empty`**. Many empty-cell advances per ray means rays are flying through air for a long time before giving up. Either the camera is far from the model (rays exit the frame before reaching it) or pitch is wrong.

4. **Check `avg_descend` vs `avg_lod_terminal`**. If both are near zero, rays aren't entering the content tree at all — the camera's ray cone misses the slot where the model lives.

5. **Sanity-check with an explicit `--spawn-xyz` override.** If the default doesn't work, try `--spawn-xyz <model_center_x> <above_model_y> <model_center_z> --spawn-pitch -1.5`. If *that* works but the bootstrap-computed default doesn't, the bootstrap math is wrong — compare the computed values against the ones that worked.

6. **Don't trust "the model tree is there" — trust ray hits**. A 91K-node library is visible in `render_harness_workload` but that only tells you the packer ran; it doesn't tell you the camera sees it.

## Other ways to place models (not used here, for reference)

- **Place at a bottom slot** (e.g. `slot_index(1, 0, 1) = 10` instead of `CENTER_SLOT = 13`) so the model sits on the "floor" of the wrap cell. Works well for plain-world hybrids where the model should be on the ground.
- **Multiple instances** in different slots of the same wrap — good for A/B comparing two models side by side.
- **No wraps** (`total_depth == model_depth`): the model *is* the root. Simplest arithmetic; the model fills `[0..size/padded * 3]` of the root directly. Use this when you want the full model to fill the view with no surrounding air.

## Shortcuts to avoid

- **Don't use `--spawn-depth` with a content bootstrap unless you know the render frame is still content-bearing at that depth.** Deep anchors shrink the render frame aggressively (`render_depth = anchor_depth - K`), and a model that fits in a shallow root may be entirely out of frame at a deep anchor.
- **Don't compute spawn from `WORLD_SIZE`** (= 3.0) alone. That gives you *root frame* coordinates, which mean nothing about where an individual imported model landed inside the tree.
- **Don't trust your memory of a camera direction.** yaw/pitch conventions vary; always verify empirically with a screenshot or `hit_fraction > 0`.

## See also

- [rendering.md](../architecture/rendering.md) — render frame selection, ribbon semantics.
- [coordinates.md](../architecture/coordinates.md) — world vs frame-local coordinates.
- [cookbook.md](cookbook.md) — general testing commands.
- `src/world/bootstrap.rs::bootstrap_vox_model_world` — the reference implementation of the spawn math described here.
