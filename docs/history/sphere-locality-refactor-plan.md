# Sphere locality refactor — findings

This is a working document, not a plan. It captures the precision
bug, the traps that two failed implementation attempts hit, and the
open questions a future session will need to answer. The goal is
that a new contributor (or a new Claude session) can start from
zero, read this once, and avoid repeating the walls the last two
sessions walked into.

**Companion documents:**
- `docs/principles/no-absolute-coordinates.md` — the rule being
  enforced.
- `docs/principles/locality-prime-directive.md` — why the rule
  matters end-to-end.
- `docs/architecture/coordinates.md` — the `WorldPos`/`Path`
  primitives.
- `docs/architecture/cubed-sphere.md` — how face subtrees are
  built and what `(u, v, r)` means.

## The concrete symptom

At `anchor_depth ≥ ~20`, the CPU-side highlight AABB for a sphere
hit collapses to a zero-extent box:

1. `hit_aabb_body_local` samples 8 corners via
   `face_space_to_body_point(face, un, vn, rn, inner_r, outer_r,
   body_size=3.0)`.
2. Each corner lands at body-frame coordinates near `(1.5, 1.5,
   1.5)`.
3. f32 ULP near `1.5` is `~2e-7`. A depth-26 cell has face-
   normalized size `~3⁻²⁴ ≈ 5e-12`, which scales to `~5e-12 × 3` ≈
   `1.5e-11` in body-frame — five orders of magnitude below ULP.
4. All 8 corners round to the same `f32` triple. `AABB.min ==
   AABB.max`. The shader's `hit_pos ∈ AABB` check can't fire.

## Why this is fundamentally a locality violation

The AABB collapse is one observable. The deeper rule being broken
is that the sphere pipeline pins multiple runtime decisions at
absolute body-frame coordinates — the hardcoded `vec3<f32>(1.5)`
in `sphere.wgsl::march_face_root`, the `body_size=3.0` in every
`cs_raycast_in_body` call, the `uniforms.root_radii * 3.0` scaling,
and the face-normalized `(un_abs, vn_abs, rn_abs)` that
`walk_face_subtree` uses as absolute coordinates in `[0, 1]`. Every
one of those bakes in "the body is always at the same place, at the
same scale" and the f32 representation of that becomes the
precision wall.

The Cartesian pipeline doesn't have this problem because it
already lives in render-frame-local coordinates
(`camera.position.in_frame(&render_path)`), with bounds in
`[0, WORLD_SIZE)³` at any anchor depth.

## Sites that still use body-frame absolute coordinates

Grep targets for a future session — these are the sites to audit,
not a prescriptive change list. The fix at each site depends on
decisions downstream.

- `src/world/aabb.rs::hit_aabb_body_local` — samples 8 body-frame
  corners.
- `src/world/aabb.rs::hit_aabb` — unused at runtime (one test).
- `src/app/mod.rs::gpu_camera_for_frame` — Sphere uses
  `sphere.body_path`; Cartesian uses `frame.render_path`.
- `src/app/edit_actions/upload.rs::set_root_kind_face` — ships
  `in_frame(body_path)` as the "pop pos" uniform.
- `src/app/edit_actions/mod.rs::frame_aware_raycast` — Sphere
  dispatches to `cpu_raycast_in_sphere_frame`; Cartesian to
  `cpu_raycast_in_frame`. Asymmetric.
- `src/world/raycast/mod.rs::cpu_raycast_in_sphere_frame` — calls
  `sphere::cs_raycast_in_body` with `body_origin=[0,0,0]`,
  `body_size=3.0`.
- `src/world/raycast/sphere.rs::cs_raycast_in_body` — hardcoded
  `cs_center = body_origin + body_size*0.5` with the above.
- `assets/shaders/sphere.wgsl::march_face_root` — hardcoded
  `cs_center = vec3<f32>(1.5)`, `cs_outer = radii.y * 3.0`.
- `assets/shaders/sphere.wgsl::sphere_in_cell` &
  `sphere_in_face_window` — take `body_cell_origin` and
  `body_cell_size` as parameters. Callers currently supply body-
  frame values; these could become frame-relative.
- `assets/shaders/face_walk.wgsl::walk_face_subtree` — takes
  `(un_in, vn_in, rn_in)` in absolute face `[0, 1]`. Precision in
  `[0, 1]` is `~1 / ULP ≈ 1e7`; cells past face depth ~14 round
  into their neighbours.

## What two failed attempts taught us

**Attempt 1 (one session earlier): "just swap `body_path` →
`render_path` in the existing sphere raycast."**
Result: the Cartesian walker can't navigate a face subtree.
`slot_index(us, vs, rs)` uses the same axis ordering as
`slot_index(x, y, z)` numerically, but the **semantic axis
mapping differs per face**. For PosY, `body_y` corresponds to
`face_r`, but the walker — picking `slot_index(floor(x),
floor(y), floor(z))` from ray render-frame position — treats
`body_y` as `face_v`. The descent test wandered off after six
iterations.

The per-face mapping, for reference:

| Face | `u_axis` (body) | `v_axis` (body) | `r_axis` (body) |
|---|---|---|---|
| PosX | −Z | +Y | +X |
| NegX | +Z | +Y | −X |
| PosY | +X | −Z | +Y |
| NegY | +X | +Z | −Y |
| PosZ | +X | +Y | +Z |
| NegZ | −X | +Y | −Z |

**Attempt 2 (this session): "ship `root_sphere_center` in render-
frame to the shader, change `gpu_camera_for_frame` to
`render_path`, keep everything else."**
Result: the shader's `BODY_SIZE_IN_FACE_ROOT_FRAME = 9.0` constant
is correct *only* at `face_depth == 0`. At deeper `face_depth` it
needs to be `3 × 3^(face_depth + 1)`. The renderer silently
produced pixel-stable (consistent with itself) but geometrically
wrong output; `sphere_zoom_invariance.sh` passed because it
compares frames across anchor depths rather than to a ground truth.
Also, leaving the shader's highlight check in body-frame while
moving CPU AABB to render-frame breaks the live-game highlight at
every depth, not just the deep ones. Both attempts were reverted.

## The user's guidance: "lose the sphere curvature after a couple
of layers"

The operative design point. The face subtree's `(u, v, r)` cube-
to-equal-area mapping only matters visually at `face_depth ≈ 0`,
where the render frame spans a significant fraction of the whole
face. Past that, the render frame is a tiny sub-face where the
curvature over its extent is below the pixel grid, and treating
the face subtree as a Cartesian tree in render-frame coordinates
is *visually indistinguishable* from the curved version.

This opens the door to unifying the deep-face path with the
Cartesian walker — the face subtree gets walked "as if it were
Cartesian," slot by slot, with no curvature math at all. At
`face_depth == 0` the sphere-in-cell math stays (that case is
body-scale, precision is fine), but at `face_depth ≥ 1` the
walker becomes symmetric with the Cartesian renderer.

The open question is the *crossover behaviour* between
`face_depth == 0` (sphere math, curved cells) and `face_depth ==
1` (Cartesian math, axis-aligned cells). The cells on either side
of the transition have to agree on physical position or the
player sees a pop as they zoom in. There are two approaches:

1. **Pick one interpretation throughout.** Either render the face
   as axis-aligned cubes at *all* depths (visible difference at
   very shallow zoom, invisible at deep), or render as curved
   sphere at all depths (visible AABB collapse at deep as today).
   The user has accepted option A with the "lose curvature after
   a couple of layers" guidance.

2. **Render axis-aligned consistently, but only starting at
   `face_depth ≥ 1`.** The `face_depth == 0` case renders via the
   existing sphere-in-cell path, but only when the camera is
   zoomed out enough that this case actually holds — which in
   practice means just the initial shallow view of a fresh
   planet. Any zoom-in transitions to the cartesian walker. The
   visual transition point is at the `face_depth == 0 → 1`
   boundary. Whether this is visible or jarring depends on the
   face window size at that transition — an empirical question.

Neither path is obviously correct. Running the zoom_invariance
regression at the boundary is the experiment the next session
needs.

## Open questions a future session should answer before coding

*Answered 2026-04-19. Kept as a record of the state before the
option-B refactor landed; update or delete if the shape changes.*

1. **What does `sphere_zoom_invariance.sh` actually compare?**
   It compares pairwise PNGs across depths and each depth against
   the deepest as a "reference." Neither is a ground truth — both
   are self-comparisons. The script therefore detects *consistency*
   across anchor depth, not geometric correctness. A refactor that
   changes the geometry consistently still passes.

2. **Where does `gpu_camera_for_frame` get the camera into
   render-frame for Cartesian today?** `in_frame(&frame.render_path)`.
   Sphere uses `in_frame(&sphere.body_path)` instead — the body cell
   path is strictly shallower than the render path, which is why
   sphere-frame precision is inherited from the body cell regardless
   of how far the face subtree descends. That asymmetry IS the
   anomaly.

3. **What does `WorldPos::in_frame` do for a `frame_path` that
   descends past a face root?** It walks Cartesian-style
   (`slot_coords` interprets the slot as `(x, y, z)`). Because the
   tree's slot numeric layout is shared between `slot_index(x,y,z)`
   and `slot_index(us,vs,rs)`, the *slot sequence* is correct either
   way. The produced float coordinates aren't the face's physical
   `(u,v,r)` — but for "walk the face subtree as an axis-aligned
   Cartesian tree in render-frame coords" that's exactly the
   behavior we want.

4. **Sphere shader functions:** `sphere_in_cell` is called from
   `march_cartesian` on DDA descent into a body cell, and from
   `march()` when `ROOT_KIND_BODY`. `march_face_root` is called
   from `march()` when `ROOT_KIND_FACE`. `sphere_in_face_window`
   is **dead code** — defined but no callers in the shader. The
   helper functions in `face_walk.wgsl::face_point_to_body_with_bounds`
   / `face_root_point_to_body` / `face_dir_to_body` /
   `face_local_normal_to_body` / `face_box_to_body_bounds` are
   also dead code. Safe to delete in a follow-up cleanup.

5. **What test drives live-highlight correctness?** No pixel-level
   test exists. The CPU test pair
   (`sphere_cursor_hit_point_is_inside_aabb_after_wall_dig`) tests
   the raycast/AABB contract in isolation. The shader's AABB check
   has no automated regression — mismatch between CPU-frame-AABB
   and shader-frame-camera-pos is a silent bug today.

6. **`render_path` selection:**
   - `desired_depth = anchor_depth − RENDER_FRAME_K (= 3)`, clamped
     to `MAX_DEPTH`.
   - For sphere frames, `camera_local_sphere_focus_path(desired_depth)`
     builds a path INTO the face subtree by slot-locating the camera's
     body-frame position, walking to `desired_depth` slots deep.
   - Then `target_render_frame` shrinks the render path while
     `!camera_fits_frame || frame_projected_pixels < FRAME_FOCUS_MIN_PIXELS (192)`.
   - Net effect for `--spawn-depth 25` at the stock sphere test xyz:
     render_path lands at depth 5 (`face_depth = 3`), because the
     frame window at depth 6 would project < 192 pixels from the
     camera's altitude. The clamp is driven by on-screen projected
     pixel coverage of the current face window — not by any
     face-depth-aware heuristic.
   - Consequence: `face_depth == 0` only happens at shallow anchor
     depths in practice (where the full face projects to ≥ 192
     pixels). `face_depth ≥ 1` is the deep-anchor regime. The
     precision wall is specific to `face_depth ≥ 1` with
     `anchor_depth ≥ 20`.

## Testing stance for any attempted implementation

Before a commit to main gets pushed, all of these must pass at
their current baselines (no regressions) **plus** the baseline for
the sphere suite must hold at depth ≥ 20, which is where the
precision bug manifests:

- `cargo test --test e2e_layer_descent` (plain world; baseline
  has one pre-existing failure at d=22 sky-dominance that is
  unrelated and present on this branch before the refactor).
- `cargo test --test e2e_layer_descent_sphere` (4 tests, all
  currently pass).
- `cargo test --test render_visibility` (2 tests).
- `scripts/sphere_zoom_invariance.sh` (invariance across depth).
- `scripts/sphere_break_probe.sh` (break+probe at depths 5..28).
- `scripts/cursor_detail_probe.sh` (cursor probe plain + sphere).
- `scripts/live_perf_sphere_vs_plain.sh` (perf comparison — a
  regression here means the new path is slower than body-frame
  sphere math; the user has been explicit about not accepting
  perf hits).
- `scripts/replicate_edit_spike.sh` (live-loop edit perf
  invariance).

Recommended ordering: land AABB and CPU raycast changes first
with a narrow regression test that exercises `anchor_depth = 30`
(past the current pre-existing precision wall). When that passes
on CPU, move on to shader.

## Don't land intermediate "always-green" states

Per the auto-memory `feedback_dont_force_incremental_green`:

> For interconnected architectural rewrites, splitting into
> "always-green" commits adds shim layers that themselves become
> the bug source; one big broken-in-the-middle commit is fine.

Concretely: the camera frame, the AABB computation, the shader
sphere constants, and the shader dispatch all have to land
together or the cursor highlight rendering is wrong for sphere
throughout the transition. Expect one large commit that touches
~8-12 files and breaks pixel-identity on the sphere briefly
before the dust settles. Do not split into "CPU only" and "shader
only" commits.

## What's landed vs. outstanding

**Landed** (earlier commits on this branch, not this session):
- `ribbon-pop t-scaling` fix: `hit.t` returned from
  `cpu_raycast_in_sphere_frame` is now scaled by `3^pops` so
  deeply-popped raycast hits measure `t` in the caller's frame
  units. Fixes the "cursor hit-point outside AABB" bug the user
  originally reported. Commit `780f589`.

**Outstanding** (the content of this document):
- The whole body-frame-absolute → render-frame-local unification
  of the sphere pipeline.

## Starting point for the next session

1. Run the baseline tests listed above and record numbers. (Pixel
   counts, perf ms-per-frame, test durations.)
2. Write a narrow CPU-side regression test at
   `anchor_depth = 30` for sphere that asserts
   `hit_aabb_in_frame_local` produces a non-zero-extent box.
   Make sure it *fails* on today's code — that's the precision
   bug pinpointed.
3. Read `compute_render_frame` carefully. Understand the
   `face_depth` calculation and the clamping in
   `target_render_frame`. This was the bit I didn't fully
   understand in either attempt.
4. Read `WorldPos::offset_from` and satisfy yourself that it's
   precision-safe for computing sphere-center-relative-to-camera
   without ever going through absolute body-frame.
5. Only then start editing.

If you find any of these open questions have clear answers in
code I missed, update this doc as you go. The point of the
document is to reduce the cost of the next attempt, not to
prescribe its shape.
