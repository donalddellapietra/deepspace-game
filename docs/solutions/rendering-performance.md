# Rendering Performance: 88M → 40k triangles

## The problem

At view layer 10 (zoomed out), the game ran at 30-40 FPS with frame
time flickering between 16ms and 33ms. At layers 11-12 (zoomed in),
performance was perfect at 60 FPS. The visual content was identical —
the same infinite grass field — yet layer 10 was orders of magnitude
more expensive.

## Investigation

### Red herring: entity count

Initial diagnosis pointed at entity count. At layer 10, the renderer
produced ~2,200 Bevy entities (one per tree visit × sub-meshes per
visit). We tried:

1. **Emit-layer batching** — walk the tree one level coarser, bake
   125³ meshes from children's voxels. Reduced visits from 137k to
   ~1,100. (Committed, kept — this is correct regardless.)

2. **GPU instancing** — custom render pipeline with instance buffers.
   Reduced entities to ~6. But froze after a few seconds because
   `write_buffer` accumulated staging buffers every frame, and
   disabling `NoFrustumCulling` meant the GPU drew all 1,100
   instances including those behind the camera. (Branch preserved,
   reverted.)

3. **Automatic Bevy batching** — per-entity with shared mesh/material
   handles. Bevy batches into ~4 draw calls and frustum-culls
   automatically. ~2,200 entities but GPU only draws visible ones.

None of these fixed the performance. Entity count was never the
bottleneck.

### Real bottleneck #1: collision

`move_and_collide` sampled blocks at `target_layer_for(view_layer)`,
which at view layer 10 meant leaf-resolution blocks (block_size = 1
Bevy unit). The player AABB was 25 units wide (one view cell). The
block sweep was 27³ = **19,683 tree lookups per frame** — plus
`on_ground` doing it again. ~40k tree lookups per frame just for
collision.

**Fix**: collision samples at `(view_layer + 1).min(MAX_LAYER)`
instead of `target_layer_for(view_layer)`. At view layer 10,
block_size = 5 (one layer below view). Block sweep = 7³ = 343.
Collision time dropped from ~15ms to <1ms. The 5-leaf collision
granularity is 1/5th of the player's width — fine for gameplay.
Small placed blocks still register as solid via the
presence-preserving downsample.

### Real bottleneck #2: triangle count

After fixing collision, the game was smoother but still stuttered at
low layers. Profiling showed the GPU was the bottleneck:

- Layer 12: ~80k triangles. 1 chunk in the cull sphere.
- Layer 10: **88M triangles**. 1,100 chunks in the sphere.

Each chunk's 125³ mesh generated ~15,625 quads for the ground
surface (one per voxel face). All coplanar, same material, same AO.
A flat grass surface that could be 1 quad was 15,625 quads.

**Fix**: greedy meshing. Adjacent coplanar faces with uniform AO
merge into larger rectangles. A 125×125 ground surface collapses to
~1 quad. Triangle count dropped from 88M to ~40k across all layers.

## The solution: greedy meshing

### Algorithm

For each of 6 face directions, for each depth slice:

1. Build a 2D grid of exposed faces: `(voxel_type, ao_level)` or
   `None`
2. Only faces with **uniform AO** (all 4 vertices equal) enter the
   grid. Non-uniform faces are emitted immediately as individual
   quads.
3. Greedy rectangle scan: extend width, then height, matching
   `(voxel_type, ao_level)`.
4. Emit one quad per merged rectangle.

### Why uniform-AO is the merge condition

When faces merge, interior vertices are removed. The GPU interpolates
AO linearly between the merged quad's corners. This is only correct
when AO is constant — otherwise the interpolated values differ from
the original per-face values, producing visible shading changes.

For flat grassland with no nearby blocks, all AO is `[3,3,3,3]`
(fully lit). Everything merges. Near placed blocks, AO varies and
faces stay separate. The output is pixel-identical.

### Results

| Metric | Before | After |
|--------|--------|-------|
| Triangles at layer 10 | 88,000,000 | ~40,000 |
| Triangles at layer 12 | ~80,000 | ~40,000 |
| Render distance 64 cells, layer 10 | Unplayable | 60 FPS |
| Collision blocks per frame (layer 10) | 19,683 | 343 |

### Future considerations

- **Textures/tints**: if per-voxel visual variation is added, extend
  the merge eligibility check. Non-eligible faces automatically fall
  through to the individual quad path.
- **GPU instancing**: the `gpu-instancing` branch preserves the
  custom render pipeline. With greedy meshing reducing triangle count,
  the GPU instancing approach becomes viable again — the frustum
  culling issue matters less when each instance has ~36 triangles
  instead of 80k.
- **Neighbor-aware face culling**: chunks emit boundary faces at
  edges because the mesher doesn't know about neighboring chunks.
  Passing neighbor data would eliminate these invisible faces,
  further reducing triangle count for underground chunks.

## Files changed

- `src/model/mesher.rs` — greedy meshing algorithm
- `src/world/collision.rs` — `view_layer + 1` collision sampling
- `src/world/render.rs` — emit-layer batching, diagnostics timings
- `src/diagnostics.rs` — HUD shows walk/reconcile/collision/triangles
