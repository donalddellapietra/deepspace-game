# d=10 sphere bug — live investigation progress

## Confirmed facts (clean baseline, HEAD = 988f980)

Running `scripts/compare-place-induced.sh` produces `tmp/before_m*.png`
and `tmp/after_m*.png`. All comparisons use identical `--spawn-xyz
1.5 1.7993 1.4988 --spawn-depth 10 --spawn-pitch -0.5` with a
`wait:10,place,wait:30` script between them (matched timing so
camera physics doesn't drift).

| Mode | Before place | After place | Same? |
|---|---|---|---|
| 0 (normal) | grey ground, yellow grass dome | grey ground **with horizontal stripes** + hollow-looking yellow cube | **NO** |
| 1 (step count) | uniform blue (low steps) on ground | uniform blue on ground, yellow cube center | **YES on ground** |
| 2 (walker terminal depth) | yellow-green on ground | yellow-green on ground, cyan cube | **YES on ground** |
| 3 (walker result) | green everywhere on ground | green everywhere on ground | YES |
| 4 (winning plane) | **uniform light-blue** (r_lo) on ground | **alternating r_lo / v_lo / u_lo stripes** | **NO** |
| 5 (cell size) | uniform cyan on ground | uniform cyan on ground, magenta cube | **YES on ground** |
| 6 (ratio bucket) | uniform blue on ground | uniform blue on ground | **YES on ground** |
| 7 (face_node_idx hash) | (unstable per earlier test — stripes) | (unstable) | NO |

**Mode-4 is the only mode whose GROUND output differs pre vs post
place.** Every other mode's ground output is identical. That means:

- Walker's terminal cell → same depth, same size, same ratio bucket
- Walker's result → same (block found)
- Number of DDA steps → same
- Only the `last_side` value at hit time → different

## The confounder

If walker returns THE SAME CELL for the same ray, the plane-DDA
computes the same 6 `t` values and picks the same arg-min. So
`last_side` should be the same. Unless one of:

1. **Walker's returned `u_lo / v_lo / size` values differ at bit level
   even though the mod-8 hash matches.** (Mode 6 only captures
   low bits.)
2. **`t` at hit time differs**, because the walker at a PREVIOUS
   iteration returned a slightly-different cell.
3. **Some non-walker input to the DDA changes** (body_origin,
   body_size, window_bounds, face_node_idx, pinned_face).

## Ruled out (previous rounds)

See `d10-hollow-block-debug-state.md` for full list. Includes:
- pick_face flipping (pinned — mode 7 uniform, mode 4 still stripes)
- Walker integer-slot-pick jitter (already-stable walker used)
- Pack uniform-flatten (disabled, stripes persist)
- Body dimension drift (hardcoded, stripes persist)
- LOD-binning (face_lod_depth = 10u forced, stripes persist)
- Cell bound quantization to fixed d=9 (stripes persist)
- Plane-normal cancellation rewrite (mid ± delta form, pixel-identical)

## Next concrete experiment

Instrument walker state in an SSBO for the center pixel, read back
CPU-side. Dump:
- `w.depth`, `w.block`
- `w.ratio_u`, `w.ratio_v`, `w.ratio_r`, `w.ratio_depth`
- `steps`, `final_last_side`, `t_at_hit`

Run the repro with instrumentation before/after place. If
ratio/block are identical and t_at_hit differs, hypothesis 2 holds.
If walker output itself differs, hypothesis 1.

## Running notes

### 2026-04-22 late — probe_at diagnostic

Added `probe_at:<pitch>:<yaw>` script cmd to CPU-raycast arbitrary
directions. Compared CPU walker output pre vs post place for 4
adjacent pitches (-0.494 through -0.503, pixel-adjacent deltas):

- **Pre-place at pitch -0.5**: `ratio_u=3280 ratio_v=3298 ratio_r=3563
  ratio_depth=8`, `path=[13,16,13,13,22,13,25,19,22,22]`.
- **Post-place at pitch -0.5**: `ratio_u=3280 ratio_v=3298 ratio_r=3564`,
  `path=[13,16,13,13,22,22,7,1,4,4]`. **Different cell one r-cell outward.**
- All adjacent pitches (-0.494, -0.497, -0.5, -0.503) behave IDENTICALLY
  pre to pre and post to post — not ray-precision.

Probe at distant pitch -0.7 (far from placement): **unchanged** pre
vs post. So the divergence is LOCALIZED to rays that pass through
the placed cell's region.

### Attempted fix: single-leaf place_block

Changed `place_block` to insert `Child::Block(block_type)` directly
instead of `build_uniform_subtree`. Hypothesis: uniform-subtree gets
flattened by pack to a shallow-depth ancestor. But mode 0 is visually
UNCHANGED. Reverted.

### Current hypothesis re-aligned

Probing pre/post confirms: walker returns SAME cells for rays that
don't pass through the placed block's path, and DIFFERENT cells for
rays that do. That's correct behavior — walker correctly detects
the new content.

**The mode-4 stripes on distant ground must come from something
else**: the DDA's `last_side` evolves per iteration. For rays that
pass through newly-non-empty ancestors (on their way to deeper
cells), the empty-advance step count changes, altering the final
`last_side`. Adjacent rays that take different numbers of advances
see different final last_sides → stripes.

Investigation path: where exactly does the DDA's advance sequence
change between pre/post for a ray going past-but-not-through the
placed cell? This is the remaining precision question.

### 2026-04-22 late-late — GPU SSBO probe landed, root cause identified

Built `probe_gpu:<x>:<y>` script cmd that writes GPU walker state
into a dedicated 64-byte SSBO for the matching pixel, reads back
CPU-side, and prints. Ground truth, not inferred.

**Pre-place at y=300 across x=100..500**: every probed pixel returns
`depth=4 face_node_idx=7 winning=r_lo ratio=(40,40,43,4)`. All rays
hit the SAME uniform-flat leaf at a shallow depth. Uniform shading.

**Post-place at y=300 across x=100..500**:
- `x=100,200`: walker terminates at **depth 4**, same shallow cell
  as before, winning=r_lo. No change from pre-place.
- `x=300,400,500`: walker descends to **depth 8** (affected by
  placement), different cell, winning=v_hi, face_node_idx=18
  (face subtree got a new BFS idx due to the placement's new
  NodeId ancestor chain).

So the "stripes" are actually a **spatial LOD boundary**: some
pixels hit the new deep cell at d=8 (winning=v_hi=dark green),
others hit the old shallow uniform-flat leaf at d=4 (winning=r_lo
=light blue). The seam between these two regions is what reads as
a stripe.

**The walker's behavior is correct.** Placing a block un-flattens
the affected ancestor's uniform-type, pack no longer flattens it,
walker descends deeper in that spatial region and returns the
actual deep cell. Rays outside the affected region still hit the
original uniform-flat ancestor at shallower depth.

**The visual bug** comes from shading discontinuity across the LOD
boundary: the d=4 region uses a big cell's plane normal, the d=8
region uses a small cell's plane normal, and those normals differ
in direction. Adjacent pixels straddling the boundary get different
shading.

### Fix direction

To eliminate the shading seam, one of:
1. Prevent the LOD boundary from forming — force consistent depth
   across the face subtree (disable uniform-flatten on face-subtree
   descendants; tried in this worktree earlier, partial, see git).
2. Make the shading CONTINUOUS across LOD boundaries — the hit
   normal at depth 4 should blend smoothly into the hit normal at
   depth 8 across the boundary. Requires LOD-aware normal interpo-
   lation.
3. Change the shading-source entirely — use the sphere's radial
   normal everywhere regardless of cell depth. Tried, had downstream
   issues with `cube_face_bevel` UV.

Option 1 is the proven-working direction — ensuring consistent
walker depth eliminates the boundary. But it disables an
optimization. Next iteration: re-test with uniform-flatten disabled
ONLY for sphere face-subtree descendants, leaving Cartesian flatten
intact.

### 2026-04-22 post-compact — SDF_DETAIL_LEVELS and close-camera

After session restart, re-ran bracket experiment
`repro-sphere-d10-elevation.sh` at elev=30, 300, 3000 cells. Result:

| Elev (cells) | Cam rn | Walker@step1 | Image |
|---|---|---|---|
| 30  | 0.504 | d=3 block=STONE | pure grey |
| 300 | 0.509 | d=3 block=STONE | pure grey |
| 3000| 0.647 | d=4 then empty, hit at d=4 step 5 | clean sphere |

Forcing `--force-visual-depth 10` at elev=300: still
`packed_nodes=11`, walker still terminates at depth=3. The GPU tree
is simply NOT deeper than 4 in the central face chain, because the
SDF worldgen caps at `cubesphere.rs:327 SDF_DETAIL_LEVELS = 4`.
Below that, cells commit to solid/empty from their center sample
and uniform-dedupe. So `--force-visual-depth` can't create detail
the library doesn't hold.

At elev=300 the spawn's `fly_to_surface_elevation` placed the
camera 300 cells above what the CPU RAYCAST thinks is the surface,
but the SDF-built tree represents that same region as a uniform
STONE cell at d=3 (ratio 13,13,13 covers rn 0.481–0.518, camera
rn=0.509 falls inside). Camera is embedded in stone per the GPU's
tree. That is NOT the striped-ground bug — it is the "SDF detail
< spawn depth" mismatch.

### The actual striped bug, re-confirmed via GPU probe

`repro-sphere-d10-bug.sh 0` (spawn world 1.5,1.7993,1.4988, pitch
-0.5, place one block). Probed post-place pixels across the visible
seam:

- cube-face pixels (x=200 y=80, x=300 y=100/150/250):
  `depth=8 ratio=(3280,3298,3564,8) winning=3 face_node_idx=18`
- ground pixels (x=300 y=300, x=450 y=300):
  `depth=4 ratio=(40,40,43,4) winning=4 face_node_idx=18`

So post-place, adjacent regions terminate at different walker
depths AND different winning faces. d=4 cell's r-face vs d=8 cell's
v-face give different hit_normals → different shading. That seam IS
the stripe.

Placement never descends below d=8 even when anchor is d=10,
because pack flattens the uniform-stone sub-subtrees into a single
tag=1 leaf at the shallowest level where it becomes uniform.
`build_child_entry` line 219: `if is_cart && uniform_type !=
UNIFORM_MIXED → tag=1 leaf`.

### User's framing restated

"d≤8 intentional and works, d≥10 geometry breaks":
- d≤8 works = the flattened-leaf cells at d≤8 render with clean
  face-aligned shading. Every pixel on that region agrees on
  winning-face.
- d≥10 breaks = rays that should descend to d=10 (the anchor where
  placed blocks live) don't, because pack has already flattened to
  d=8. But adjacent rays that hit the tag=1 leaf at d=8 disagree
  on winning face from pixels that hit a d=4 leaf → stripes.

### Principled fix candidates (not yet tried, re-ranked)

1. **Don't flatten uniform-stone subtrees inside a face subtree**
   (sphere-only surgery to `build_child_entry`). Walker reaches the
   anchor depth in the entire region. Cost: more packed nodes; may
   blow past the 1M cap if a large stone region.
2. **Pack all siblings along a placed-block's path at the anchor
   depth**, not just the path cells. Unifies LOD in a neighborhood
   of the placement. Lighter than (1) but needs spatial "nearby
   placement" detection.
3. **Smooth-radial shading only for sphere-body-face cells, axial
   for Cartesian cells**. Tried `hit_normal = n` globally and it
   destroyed the ground (plus hung). Scoped version has not been
   tried.

