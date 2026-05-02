# Sphere-mode precision: what we tried and why it didn't work

A running log of attempts to fix the **camera-jitter / CPU-raycast-off**
symptoms reported at deep zoom (anchor depths ≈ 18+, "layers 8 and below"
in user-facing terminology) in `--wrapped-planet --wrapped-planet-tangent
--planet-render-sphere` mode. Each section is a self-contained
post-mortem so the next attempt doesn't repeat the same mistake.

The branch state these were tried on: `sphere-mercator-1-2`, with the
`TangentBlock` `NodeKind` + `march_in_tangent_cube` shader walker
(commits up through `ce7b83a`).

## Symptoms (the ground truth)

User reports, in order seen:

1. Visible cells flicker / appear noisy at deep zoom — bottom-of-screen
   sub-cells in the screenshot at `Screenshot 2026-05-02 at 1.32.08 PM.png`
   are tiny and jittery.
2. Camera "jitters" at layers 8 and below.
3. CPU raycast hit positions "slightly off at layers 8 and below
   (depths 18+ or something)" — break/place lands one or more cells
   away from where the cursor visually points.

Reference behaviour: **plain Cartesian (`--plain-world`) renders cleanly
to 40+ layers without these artifacts.** Whatever it does, sphere mode
should match it.

## Attempt 1: `compute_render_frame` no longer breaks at `WrappedPlane`

Commit `879f596` (reverted by `3801e76`).

### Premise

`WorldPos::in_frame` (`world_pos.rs:229`) is precision-clean **iff** the
render frame is a long prefix of the camera anchor — the loop walks
`[common_prefix..anchor_depth)`, so the tail walk is short and every
slot contribution is bounded by `WORLD_SIZE`.

In plain Cartesian, `compute_render_frame` descends through every
Cartesian Node it can reach along the camera anchor, so `common_prefix`
is roughly `anchor_depth − render_margin` and the tail walk is ~3-4
levels regardless of how deep the anchor sits.

In sphere mode, `compute_render_frame` (`frame.rs:69-85`) **explicitly
breaks the descent at `WrappedPlane`** with the comment "the wrap branch
in the shader fires at depth==0 of the marcher's local frame". Render
frame stays pinned at WP depth (≈ 2). When the camera anchor goes to
18+, `in_frame` walks 16+ levels — accumulator noise ≈ 2e-7 in
WP-local. That's the precision wall.

The fix should be: remove the WP-break so descent goes as deep as the
camera anchor allows, same as Cartesian.

### What I changed

- `compute_render_frame` no longer stops at WrappedPlane. Kind is
  derived from the FINAL node landed on (so it's still
  `WrappedPlane` when the descent literally ends there, but
  `Cartesian` once descent goes past).

### What broke / why it didn't help

The render frame *can* deepen past WP — but only if the camera anchor's
slot path actually finds Nodes to descend into. The wrapped-planet
bootstrap fills empty slots of the WP subgrid (and the embedding
non-centre slots) with `Child::Empty`. For an above-slab camera, the
deepened anchor's slot at WP-children level (typically `slot_y = 2`,
above the 2-row slab footprint) lands on `Child::Empty` — descent
stops at WP. **Render frame still pinned at depth 2 for the typical
view.** Smoke-test confirmed: `target_frame stable render_path=[13, 13]
kind=WrappedPlane` even after the WP-break removal.

### Lesson

Plain Cartesian's deep descent is enabled by **fully populating the
world tree with Node references everywhere**. See `plain.rs`'s
`air_l1`, `air_l2`, the `air_subtree()` helper, and `carve_air_pocket`
which auto-installs Nodes at any anchor-path slot that would otherwise
be Empty. Without that, removing the WP-break is necessary but not
sufficient.

## Attempt 2: air-subtree wrapping for empty slots in the bootstrap

Commit `879f596` (same WIP commit as Attempt 1; reverted by `3801e76`).

### Premise

Mirror plain Cartesian's `air_subtree` trick: in
`wrapped_planet_world`, fill every otherwise-Empty slot (slab
subgrid leaves outside the footprint, embedding non-centre slots)
with `Child::Node(air_subtree)` of the right depth. Then any camera
anchor path finds Nodes all the way down. `compute_render_frame`
(now no longer breaking at WP) descends to the deepest reachable
Node. `WorldPos::in_frame`'s tail walk is short. f32 precision
matches plain Cartesian's 40+ layer discipline.

### What I changed

- Pre-computed `air_subtrees: Vec<Child>` indexed by depth. Each
  entry is a uniform-air Node chain whose `Node.depth` matches the
  index (so it dedups symmetrically with material chains of the
  same depth at the same slot, and `tree_depth` bookkeeping is
  preserved).
- Slab subgrid: `leaf_at(...)` returns `air_node_of_depth(
  cell_subtree_depth)` for cells outside the slab footprint instead
  of `Child::Empty`. Falls back to `Child::Empty` when
  `cell_subtree_depth == 0` to match the material's `Child::Block`
  depth-0 leaf.
- Embedding loop: each iteration's non-centre 26 slots get
  `air_node_of_depth(library.get(current).depth)` — air sibling
  matches the wrapped content's depth, so embedding `Node.depth`
  growth stays linear (= 1 per wrap, same as before).

### What broke / why it didn't help

Smoke test confirmed the change worked **at the data-structure level**:

```
target_frame stable render_path=[13, 13, 7, 19, 16, 7, 7, 1, 1, 7, 7, 1, 1, 7, 7, 1, 1, 7, 25, 4]
                    logical_path=[13, 13, 7, 19, 16, 7, 7, 1, 1, 7, 7, 1, 1, 7, 7, 1, 1, 7, 25, 4, 25, 10, 10, 19]
                    kind=Cartesian cam_local=[0.0, 0.0, 2.5262766]
```

20-deep render path (vs the previous depth 2). `kind = Cartesian` (not
`WrappedPlane`), so the shader dispatches `march_cartesian`, not the
sphere DDA. Camera local `[0.0, 0.0, 2.5262766]` is in the deep
frame's local coords, magnitudes well within `[0, 3)`.

But the user reported **no visible improvement** — the same jitter and
raycast-off at deep anchor.

### Lesson (open question)

The render frame deepening + air-subtree fix addresses the
`in_frame`-tail-walk precision noise, but the user's symptom is not
gone. Hypotheses for the next attempt:

1. **The shader still dispatches sphere DDA at WP frame via ribbon
   pop.** With `kind = Cartesian` at the deep render frame, the shader
   uses `march_cartesian`. When the ray exits the deep frame, ribbon
   pops upward. At the WP ancestor, `march_cartesian` doesn't know how
   to dispatch into the sphere visualization — only `sphere_uv_in_cell`
   does. So the planet itself doesn't render through the deep-frame
   ray at all; it gets rendered by ribbon-pop into WP, which in turn
   re-runs sphere DDA at WP local with the camera implicitly at
   WP-local position via the ribbon transform chain. That chain might
   re-introduce the same precision noise it was supposed to fix.
   **Not verified.**
2. **`march_in_tangent_cube`'s ray transform is the noise source, not
   `in_frame`.** Even with a precise camera local, the per-cube TBN
   matrix multiply (with scale ≈ 27) amplifies the input by an order
   of magnitude. Deep cube cells (size `3 / 3^N` in cube-local) sit
   below the amplified eps for N ≥ 13 — exactly where the user
   reports breakdown. **Plausible but not isolated.**
3. **The reported jitter is not in the camera-local at all.** Could be
   sub-pixel cell-selection flicker driven by mouse / TAA / per-frame
   noise that the deep frame doesn't address. The user explicitly
   ruled out TAA but didn't rule out mouse micro-motion. **Not
   tested.**
4. **`with_render_margin` (`mod.rs:519-523`, called per frame) might
   re-pin the frame at WP some other way.** Did not verify the
   actual `frame_kind` going to the GPU on the user's run vs the
   smoke-test run; the user might have had a slightly different
   config (e.g. the grass cube view) that triggers a code path I
   didn't smoke-test. **Not verified end-to-end.**

### What to investigate next

- Walk the **end-to-end ray path** for a single pixel in the user's
  reported scenario, both before and after the air-subtree fix:
  - Where does the camera's `cam_local` actually originate?
  - Which shader dispatch fires? sphere DDA at ribbon-pop, or
    march_cartesian at deep frame?
  - At the cube boundary, what's the camera position the ray
    transformer actually receives, and what's its precision?
- Specifically: enable `--shader-stats` and `walker_probe` for one
  pixel at the user's deep-zoom configuration with both branches
  (revert + apply) to see if `cam_local`, `cell_min`, and `hit_t`
  change.
- Consider whether the precision issue is in the cube transform
  itself (per-cube TBN `* scale` amplification), independent of the
  camera local — that might mean **the architectural fix has to be
  per-cube render frame** (rasterisation or analogous), not just air
  subtrees + render-frame following.

## Attempts NOT to repeat

- "Just bump `MAX_STACK_DEPTH`" — register-pressure cliff at 8 on
  Apple Silicon (per the existing comment in `bindings.wgsl`); spills
  to threadgroup memory and wrecks every fragment's perf. The
  separate `march_in_tangent_cube` walker with its own stack-24
  arrays is the right escape hatch for the tangent path; don't push
  the global cap.
- "Add LOD termination to `march_in_tangent_cube` so sub-pixel cells
  splat the rep" — the user explicitly rejected this approach
  ("we never see sub-pixel cells anyway"). Sub-pixel masking hides
  the symptom but loses deep-edit visibility, which the user wants
  preserved.
- "Drop spherical coords entirely and iterate a flat cube list per
  ray" — the precision wall is at the per-cube TBN transform, not
  at `sphere_descend_anchor`'s lat/lon math. Iterating cubes
  cartesian-only doesn't move the wall.
- "Cube rasterisation as primitives (per-cube fragment march)" —
  proposed but the user pushed back hard ("WHAT THE FUCK ARE YOU
  TALKING ABOUT … THE CARTESIAN ALREADY GOES UP TO 40+ LAYERS").
  The user's belief is that Cartesian's mechanism IS the right model
  for the planet, not a from-scratch rewrite. So far Attempt 2 was
  the most direct application of that mechanism, and it didn't
  visibly help — but the right next step is probably tighter
  diagnosis of WHY Attempt 2 didn't help, not rewriting again.
