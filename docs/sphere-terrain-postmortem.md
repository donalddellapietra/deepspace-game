# Sphere Terrain: What Was Tried and Why It's Wrong

## What works

The basic sphere generation is sound:
- Recursive tree builder evaluates a density field (`distance(p, center) < radius`)
- AABB culling skips exterior (air tower) and deep interior (solid tower)
- Content-addressed dedup collapses uniform regions
- Smooth sphere (no noise) generates in ~3 seconds, 9k library entries
- Renders correctly at all zoom layers
- Downsample cascade makes the sphere rounder at coarser layers

## What was attempted for terrain

Added 3D noise (OpenSimplex2) to the sphere density function:
```
surface_radius = radius + noise3D(p) * amplitude
```

Three octaves (mountains 80, hills 20, detail 5) with early-exit
optimization for voxels clearly inside/outside the surface band.

## Why it's fundamentally wrong

### 1. Violates the layer architecture

The game's core principle: **same UX and code at every layer**. The
terrain approach hacks around layer transitions instead of working
with them:

- "Don't make terrain at layer 8 or below" — this creates an
  artificial boundary where smooth sphere AABB checks at layers 0-8
  transition to noise-aware checks at layers 9+. Nodes at the
  boundary can be incorrectly classified.
- The bake budget (MAX_COLD_BAKES) is a band-aid that makes terrain
  "pop in" over multiple frames instead of being instant. This
  violates the principle that zoom transitions should feel identical
  at every layer.
- Clamping dt to 0.1s hides the real problem (too much work per
  frame) instead of fixing it.

### 2. Startup generation is too slow

With terrain noise, every surface leaf is unique (no dedup). The
sphere surface has ~5000 leaves. Each needs 25^3 = 15,625 voxel
evaluations with noise calls. At debug opt-level 1, this takes
10-55 seconds depending on octave count. Unacceptable for startup.

Reducing to 1 octave and lower amplitude helped (55s -> 10s) but
this trades terrain quality for speed — the wrong tradeoff.

### 3. Cold bake burst at zoom transitions

At zoom 10, the render radius (800 Bevy units) covers most of the
sphere. Every surface node at emit layer 11 is unique (terrain
noise), so the renderer must cold-bake hundreds of meshes.

Each cold bake allocates a flatten_children grid of 125^3 = ~2MB.
Even with a budget of 16/frame, that's 32MB of allocations per
frame, and it takes many frames to bake all visible nodes.

The smooth sphere didn't have this problem because dedup meant
~25 unique bakes regardless of view distance.

### 4. Physics feedback loop (addressed but with a hack)

A slow bake frame produces a large `time.delta_secs()`. Gravity
applies for that entire dt, giving the player enormous velocity.
The collision sweep expands to cover thousands of blocks. The
next frame is even slower. Clamping dt to 0.1s breaks the loop
but doesn't fix the underlying cause.

## What the failed terrain commit (28e3d59) did differently

That approach started with a complete grassland (instant generation)
and lazily replaced nearby nodes with terrain content. This meant:
- Startup was instant (grassland is 25 library entries)
- At every zoom layer, the tree was always valid grassland
- Only ~8 nodes per frame were replaced with terrain
- Distant terrain was uniform grassland (efficient dedup/baking)

It failed for other reasons (fragile pristine detection, coupling
to emit layer), but its lazy approach correctly respected the
layer architecture.

## What should be done instead

The terrain system needs to work WITH the layer architecture:

1. **Start with a valid base tree** (grassland or smooth sphere)
   that renders instantly at all layers.

2. **Generate terrain lazily** — only for nodes the renderer is
   about to display, within a per-frame budget. Distant/unvisited
   nodes stay as the base tree.

3. **Terrain at every layer should be the same operation** — no
   special cases for "layer 8 and below" vs "layer 9 and above."
   The noise amplitude relative to the cell size determines whether
   terrain is visible at that layer, not a hardcoded cutoff.

4. **Cold bakes should never burst** — if a node needs terrain AND
   baking, both should be budgeted together as one operation per
   frame, not two separate unbounded loops.

5. **The tree should always be renderable** — at no point should a
   zoom transition encounter a node that hasn't been built yet.
   Grassland/smooth-sphere acts as the fallback until terrain
   generation catches up.

## Files changed in this branch

- `src/world/generator.rs` — SphereParams, TerrainNoise, density evaluation, AABB checks
- `src/world/state.rs` — sphere builder (new_sphere, build_sphere_root, build_sphere_node)
- `src/player.rs` — spawn on sphere, dt clamp
- `src/world/render.rs` — zoom.layer cache tracking, bake budget
- `src/import/stamp.rs` — test fix (WorldState::new_grassland() instead of default())
