# Heightmap-based GPU collisions for entities

Goal: entities walk on terrain that follows the voxel ground, not a
fixed sea-level plane — and do it on GPU so 10k–100k entities stay
well under the 10 ms frame budget.

Adopts the `npc-instancing` branch's progression (heightmap →
spatial cache with proper edit invalidation → integer keys), but
re-expressed in base 3 to match this repo's recursive tree.

## Collision granularity: `anchor_depth - 1`

> If the entity is at layer 25, collision tests against layer-24
> blocks.

Concretely:

- `entity_anchor_depth` = D (e.g. 25).
- `collision_depth` = D - 1 (e.g. 24). One layer coarser than the
  entity's own cell; a collision block is 3× the size of the
  entity's anchor cell along each axis.
- The heightmap stores one Y value per collision cell, i.e. per
  (collision_depth, x-slot, z-slot) pair within the render frame.

**Why D-1:** an entity fits inside a D-1 cell exactly (its anchor is
one of the 27 children of that D-1 cell). Ground-following and wall
collision both reduce to "what's the top Y of the D-1 cell at this
(x, z)?" — one texel read. Any finer (D itself) and the heightmap
has to resolve per-entity-cell detail that the entity doesn't use;
any coarser (D-2) and the entity moves *within* a collision cell,
which means collision accuracy is worse than entity size.

## Base-3 sizing

Heightmap resolution is derived, not chosen:

```
heightmap_side = 3 ^ (collision_depth - frame_depth)
```

- Frame at depth F, collision at depth C = D - 1.
- Side length = 3 ^ (C - F) texels per axis.
- Each texel **is** exactly one collision cell projected to the XZ
  plane. No fractional mapping, no aliasing.

Example sizes:

| C - F | side | texels | bytes (R32F) |
|-------|------|--------|--------------|
| 2     | 9    | 81     | 324          |
| 3     | 27   | 729    | 2.9 KB       |
| 4     | 81   | 6561   | 26 KB        |
| 5     | 243  | 59049  | 236 KB       |
| 6     | 729  | 531441 | 2.1 MB       |

At zoom levels where the entity is many tree-layers below the render
frame (C - F ≥ 7), the heightmap would blow up. **We cap at (C - F) ≤ 6**
and skip collision for entities below the capped depth — those
entities are sub-pixel anyway and wouldn't visually benefit.

## Workgroup shape

- Compute shader workgroup size = `9 × 9` (81 threads).
- Every base-3 heightmap size (27, 81, 243, 729, 2187) divides
  evenly by 9, so dispatch is exact — zero idle threads at edges.
- Well under Apple silicon's 1024-threads/workgroup limit.

If we ever need a 3^7 = 2187 heightmap, 9×9 still fits: 243×243
workgroups.

## Data layout

```wgsl
// Binding 0: heightmap texture, R32Float, size = 3^(C-F) per axis.
// Each texel stores the world-Y of the top of the highest solid
// block within that collision cell's XZ column, scanning downward
// from the top of the render frame. 1.0 (sentinel "no ground") if
// the column is empty.
@group(N) @binding(0) var heightmap: texture_storage_2d<r32float, read_write>;
```

CPU-side mirror isn't needed — entity positions live in the entity
buffer which is already a storage buffer, and the compute shader
reads/writes both in-place.

## Stage 1: GPU heightmap generation

Run once per **(frame_root_id, collision_depth)** change and on
explicit edit-invalidation. Persists across frames otherwise.

**Shader:** `assets/shaders/heightmap_gen.wgsl`

Per-texel logic:
1. Thread index → texel (u, v). Convert to (x, z) in frame-local
   `[0, 3)^2` coords: `x = (u + 0.5) * 3.0 / side`.
2. Ray-march **straight down** from `y = 3.0` through the same
   `tree[]` buffer the render pass uses.
3. Early-terminate at the first tag=1 (Block) child. Write
   `hit_world_y` (top of that block) into the texel.
4. If the ray exits the frame without hitting anything, write
   the sentinel `1.0` (i.e. the frame's Y floor — entities fall
   off the edge, handled by a separate bounds check).

Termination depth: march down as far as `collision_depth` below
frame root. Below that, we don't care — entities at that depth have
less-than-texel-size ground features, so we can treat their ground
as the coarsest Y hit.

Cost: 3^(C-F) × 3^(C-F) texels × ~20 DDA steps each. For a 243×243
heightmap, that's ~1.2M texture writes + DDA ops. Well under 1 ms
on Apple silicon.

## Stage 2: Per-entity physics compute shader

Run once per frame, just before (or at the start of) the render
pass. Dispatches `ceil(entity_count / 64)` workgroups of 64.

**Shader:** `assets/shaders/entity_physics.wgsl`

Per-entity logic:
1. Load entity transform (translate + scale) from `entities[]`.
2. Convert translate.x / translate.z to texel coords:
   `tex = vec2<i32>(floor(translate.xz / cell_size));`
3. `ground_y = textureLoad(heightmap, tex, 0).r;`
4. Apply velocity, clamp Y to `ground_y + entity_height`,
   write back.

Edge cases: entity at texel boundary uses its cell's texel (no
bilinear — integer-coord lookup, matches base-3 grid exactly).

## Edit invalidation

On tree edit at `edit_path`:
1. Walk `edit_path` down to `collision_depth`. If shallower, the
   edit affects a multi-cell heightmap region; if deeper, the edit
   is sub-collision-cell and doesn't affect the heightmap at all
   (entities collide with a D-1 block; editing a D-2 cell inside
   doesn't change the block's top).
2. Compute the dirty rectangle in texel coords from `edit_path`'s
   XZ slot prefix.
3. Re-dispatch `heightmap_gen` **only for that rectangle** via a
   sub-dispatch (one workgroup per 9×9 texel chunk covering the
   dirty rect).

Dirty rectangles are always base-3 aligned (3×3, 9×9, 27×27, …)
because the invalidated depth aligns to the heightmap's base-3 grid.

## Frame-root changes

When the render frame shifts (zoom, teleport, ribbon pop), the
heightmap is fully stale — different frame-local coords, different
collision depth possibly, different node ids underneath. Rebuild the
whole heightmap in one dispatch. Cheap (~1 ms); runs inside the
frame-change stall we already eat.

## Integer coords (from the `npc-instancing` lesson)

The old branch's heightmap bug was float XZ keys drifting over
time — NPCs "sank" into terrain or "floated" above. We avoid this
by keying everything in **integer texel coords derived from the
entity's anchor path**, not from floating-point XZ:

```rust
// Entity's (x, z) texel in heightmap coords:
let depth_delta = collision_depth - frame_depth;
let x_slot = entity_anchor.slot_x_at_depth(frame_depth + depth_delta);
let z_slot = entity_anchor.slot_z_at_depth(frame_depth + depth_delta);
let tex = ivec2(x_slot, z_slot);
```

No float division, no rounding drift. An entity never changes which
texel it looks up until its anchor cell actually crosses a
collision-cell boundary — the tree's slot arithmetic guarantees it.

## What we drop

- CPU-side `entities.tick()` Y clamping (current "drop vy" hack
  becomes unnecessary once GPU physics takes over).
- `entity_surface_y` on App — heightmap replaces the flat-world
  sea-level fallback. Flat worlds naturally show a uniform
  heightmap value; sphere worlds get their radial surface by
  ray-marching down toward the body center instead of straight
  down (Stage 4).

## Stages

**Stage 1 (this task):** Heightmap gen + entity Y clamp via compute
shader. Replaces the flat-world Y drop. Works for any terrain shape
that has a clear "top" per XZ column. CPU still owns entity spawn
and removal; GPU owns motion.

**Stage 2:** Wall collisions. Same heightmap — entity rejects motion
whose destination texel has a ground_y > current_y + step_threshold.

**Stage 3:** Entity-entity (GPU spatial hash + push-apart).

**Stage 4:** Non-vertical gravity (sphere worlds). Heightmap stays
2D but in the body's surface-parametrized space (UV on face +
radius), not XZ in Cartesian frame.

## Perf budget

At 10k entities, 243×243 heightmap, 60 fps:

- Heightmap gen (on invalidation only): ~1 ms per rebuild. Typical
  frame does zero rebuilds.
- Entity physics dispatch: 10k entities / 64 threads = 157
  workgroups. Each thread: 1 texture load + a few muladds + 1
  buffer write. ~0.1 ms.
- CPU-side cost: zero, once motion moves to GPU.

Total entity-physics budget per frame: **~0.1 ms at steady state,
+1 ms once on invalidation events** (mostly frame-root changes and
edits). Leaves the other ~9 ms for ray-march + raster entity pass.

## Open decisions

1. Compute shader in its own pass, or folded into the frame's
   `record_frame_passes`? Separate is simpler; folding saves one
   command-buffer boundary.
2. Heightmap texture is a global (per-frame) or per-frame-root
   binding? Global means rebinding on frame-root change; per-root
   means N textures cached. Start global, measure.
3. Entity buffer becomes `storage, read_write` — currently
   `storage, read` for the ray-march. Two bindings or alias the
   same buffer with two usages? wgpu supports both; alias is
   simpler.
