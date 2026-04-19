# Heightmap-based GPU collisions for entities

Goal: entities walk on terrain that follows the voxel ground, not a
fixed sea-level plane — and do it on GPU so 10k–100k entities stay
well under the 10 ms frame budget.

Adopts the `npc-instancing` branch's progression (heightmap →
spatial cache with proper edit invalidation → integer keys), but
re-expressed in base 3 to match this repo's recursive tree.

## Collision granularity: `anchor_depth + 1` (one layer finer)

Minecraft-style collision, but resolved at 1/3 the entity's own
block size along each axis — so collisions are precise to a
sub-anchor-cell level.

Concretely:

- `entity_anchor_depth` = D (e.g. 25).
- `collision_depth` = D + 1 (e.g. 26). One layer **deeper** than the
  entity's anchor; a collision cell is 1/3 the size of the entity's
  own cell along each axis.
- Each entity anchor cell contains 3×3×3 = 27 collision sub-cells.
- The heightmap stores one Y value per collision cell's XZ column.

**Why D+1:** the entity is 1 anchor cell wide. Collision resolution
of 1/3 of that gives 3 discrete standing heights within a single
entity cell — enough precision to land on a ledge that's 1/3 of the
way up a block, to side-step a bump at 1/3-block scale, and to
walk naturally up terrain with 1/3-block steps. Finer (D+2, 1/9 of
entity size) is overkill for visual quality and triples heightmap
storage per axis without visible benefit. Coarser (D itself) means
the entity can't rest on anything smaller than its own cell — too
chunky, feels like Minecraft with 1-meter blocks.

## Base-3 sizing

Heightmap resolution is derived, not chosen:

```
heightmap_side = 3 ^ (collision_depth - frame_depth)
              = 3 ^ (entity_anchor_depth - frame_depth + 1)
```

Each texel **is** exactly one collision cell projected to the XZ
plane. No fractional mapping, no aliasing.

Example sizes (entity-depth-relative-to-frame shown as `Δ`):

| Δ = D - F | collision side = 3^(Δ+1) | texels   | bytes (R32F) |
|-----------|--------------------------|----------|--------------|
| 2         | 27                       | 729      | 2.9 KB       |
| 3         | 81                       | 6561     | 26 KB        |
| 4         | 243                      | 59049    | 236 KB       |
| 5         | 729                      | 531441   | 2.1 MB       |
| 6         | 2187                     | 4.78M    | 19 MB        |

Typical gameplay has Δ = 3–5 (entity a few layers below the render
frame), so 26 KB – 2 MB. Fine. **We cap at Δ ≤ 5** (2 MB heightmap);
entities deeper than that are sub-pixel anyway and fall back to
frame-level Y clamp. The cap only triggers at extreme zoom-out
views.

## Workgroup shape

- Compute shader workgroup size = `9 × 9` (81 threads).
- Every base-3 heightmap size (27, 81, 243, 729, 2187) divides
  evenly by 9, so dispatch is exact — zero idle threads at edges.
- Well under Apple silicon's 1024-threads/workgroup limit.

243×243 → 27×27 workgroups. 729×729 → 81×81. No waste.

## Data layout

```wgsl
// Binding N: heightmap texture, R32Float, size = 3^(D+1 - F) per axis.
// Each texel stores the world-Y of the top of the highest solid
// block within that collision cell's XZ column, scanning downward
// from the top of the render frame. Frame's Y_min sentinel when
// the column is empty (entity has no ground there).
@group(M) @binding(N) var heightmap: texture_storage_2d<r32float, read_write>;
```

CPU-side mirror isn't needed — entity positions live in the entity
buffer which is already a storage buffer; the compute shader reads
and writes both in-place.

## Stage 1: GPU heightmap generation

Run once per **(frame_root_id, collision_depth)** change and on
explicit edit-invalidation. Persists across frames otherwise.

**Shader:** `assets/shaders/heightmap_gen.wgsl`

Per-texel logic:
1. Thread index → texel `(u, v)`. Convert to XZ in frame-local
   `[0, 3)^2` coords: `x = (u + 0.5) * 3.0 / side`.
2. Ray-march **straight down** from `y = 3.0` through the same
   `tree[]` buffer the render pass uses.
3. Early-terminate at the first tag=1 (Block) hit. Write
   `hit_world_y` (top of that block) into the texel.
4. If the ray exits the frame without hitting anything, write the
   sentinel (frame's Y floor). Entities over that texel "fall off
   the edge" — handled by a separate bounds check in the physics
   pass, not here.

Termination depth: march down as far as `collision_depth`. Sub-
collision-cell features (D+2 and beyond) are averaged or ignored —
entities can't resolve them anyway.

Cost: 3^(Δ+1) × 3^(Δ+1) texels × ~20 DDA steps each. 243×243
heightmap ≈ 1.2M DDA ops. Well under 1 ms on Apple silicon.

## Stage 2: Per-entity physics compute shader

Run once per frame, just before (or at the start of) the render
pass. Dispatches `ceil(entity_count / 64)` workgroups of 64.

**Shader:** `assets/shaders/entity_physics.wgsl`

Per-entity logic:
1. Load entity transform (translate + scale) from `entities[]`.
2. Convert `translate.xz` to collision-texel coords. Because the
   entity's anchor path encodes a full base-3 slot trail, this is
   integer slot arithmetic — no float division (see §"Integer
   coords" below).
3. `ground_y = textureLoad(heightmap, tex, 0).r;`
4. Apply velocity (XZ), clamp Y to `ground_y`, write back.

Edge cases: entity crosses a collision-cell boundary mid-frame → the
physics pass reads the new cell's texel naturally, since the texel
lookup is derived from the post-motion position.

## Edit invalidation

On tree edit at `edit_path`:
1. If `edit_path.depth() > collision_depth`, edit is sub-cell →
   no heightmap effect. Skip.
2. Otherwise, walk `edit_path`'s XZ slot prefix to `collision_depth`.
   The dirty rect in texel coords is a `3^(collision_depth - edit_depth) × 3^(collision_depth - edit_depth)`
   aligned block — always base-3 aligned.
3. Re-dispatch `heightmap_gen` **only for that rectangle** via a
   sub-dispatch (one 9×9 workgroup per chunk covering the dirty
   rect).

Dirty rectangles are always base-3 aligned (3×3, 9×9, 27×27, …)
because the invalidated depth aligns to the heightmap's base-3
grid. Zero box-clipping math.

## Frame-root changes

When the render frame shifts (zoom, teleport, ribbon pop), the
heightmap is fully stale — different frame-local coords, possibly
different `collision_depth`, different node ids underneath. Rebuild
the whole heightmap in one dispatch. Cheap (~1 ms); runs inside the
frame-change stall we already eat.

## Integer coords (from the `npc-instancing` lesson)

The old branch's heightmap bug was float XZ keys drifting over
time — NPCs "sank" into terrain or "floated" above. We avoid this
by keying everything in **integer texel coords derived from the
entity's anchor path**, not from floating-point XZ:

```rust
// Entity's (x, z) texel in heightmap coords:
// walk the entity's anchor down to collision_depth; collect the
// x-slot and z-slot at each level, flatten base-3 -> texel index.
let depth_delta = collision_depth - frame_depth;
let mut tex_x: u32 = 0;
let mut tex_z: u32 = 0;
for k in 0..depth_delta {
    let slot = entity.anchor.slot(frame_depth + k);
    let (sx, _sy, sz) = slot_coords(slot);
    tex_x = tex_x * 3 + sx as u32;
    tex_z = tex_z * 3 + sz as u32;
}
// Entity's own cell sits one layer ABOVE collision depth; append
// the entity's anchor-cell slot to get the specific collision
// sub-cell the entity is standing in:
let entity_slot = entity.anchor.slot(collision_depth); // wait — the
// entity's anchor ends at D; the collision cell at D+1 is inferred
// from the entity's CURRENT sub-cell Y offset (`offset[1]` tells us
// which vertical third of its anchor cell the entity is in, and
// together with its XZ third position gives the D+1 slot).
```

No float division, no rounding drift. An entity never changes which
texel it looks up until its anchor's XZ slot *or* its offset's
1/3-of-cell bin actually changes — the tree's slot arithmetic + the
coarse-binned offset guarantee it.

## What we drop

- CPU-side `entities.tick()` Y clamping (current "drop vy" hack
  becomes unnecessary once GPU physics takes over).
- `entity_surface_y` on App — heightmap replaces the flat-world
  sea-level fallback. Flat worlds naturally produce a uniform
  heightmap value; sphere worlds get their radial surface by
  ray-marching down toward the body center instead of straight
  down (Stage 4).

## Stages

**Stage 1 (this task):** Heightmap gen + entity Y clamp via compute
shader. Replaces the flat-world Y drop. Works for any terrain shape
that has a clear "top" per XZ column. CPU still owns entity spawn
and removal; GPU owns motion.

**Stage 2:** Wall collisions. Same heightmap — entity rejects motion
whose destination texel has a `ground_y > current_y + step_threshold`.
Step threshold = one collision cell = 1/3 entity cell = natural
"climb over ledge" height.

**Stage 3:** Entity-entity (GPU spatial hash + push-apart).

**Stage 4:** Non-vertical gravity (sphere worlds). Heightmap stays
2D but in the body's surface-parametrized space (UV on face +
radius), not XZ in Cartesian frame.

## Perf budget

At 10k entities, 243×243 heightmap (Δ=4), 60 fps:

- Heightmap gen (on invalidation only): ~1 ms per rebuild. Typical
  frame does zero rebuilds.
- Entity physics dispatch: 10k entities / 64 threads = 157
  workgroups. Each thread: 1 texture load + a few muladds + 1
  buffer write. ~0.1 ms.
- CPU-side cost: zero, once motion moves to GPU.

Total entity-physics budget per frame: **~0.1 ms at steady state,
+1 ms once on invalidation events** (mostly frame-root changes and
edits). Leaves the other ~9 ms for ray-march + raster entity pass.

At 729×729 heightmap (Δ=5): gen ≈ 4 ms per rebuild; steady state
unchanged (per-entity cost is the same texel load). Still fine.

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
