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

