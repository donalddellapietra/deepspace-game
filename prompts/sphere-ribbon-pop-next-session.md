# Sphere ribbon-pop architecture — brief for next session

You're picking up a partially-landed architectural rewrite on the
`sphere-attempt-2-1` worktree branch. This document describes **what
was done, what's left, and what decisions the previous session made
(or didn't)**. Treat everything here as evidence, not doctrine —
several earlier attempts at this problem have misdiagnosed the issue,
so verify before acting on any claim below.

## 1. The bug

The user reports that cubed-sphere rendering fails at deep zoom —
40+ anchor layers into a face subtree. Symptoms: the CPU raycast and
GPU shader both produce garbage because the face-subtree DDA hits
an f32 precision wall at depth ≥ ~15.

**Root cause** (confirmed by the `ribbon_pop_feasibility` tests):
the existing sphere DDA computes cell bounding planes from
`ea_to_cube(u_lo_face)` and `ea_to_cube(u_hi_face)` where `u_lo_face,
u_hi_face` are stored as absolute face-normalized coords. At
face-subtree depth N the cell is `1/3^N` wide in those coords. By
N=15 the two values round to the same f32; the plane normals
collapse; ray-plane intersections return garbage.

This is **not** an inherent f32 limit. Cartesian subtrees handle
60+ layers by ribbon-popping: at each descent the ray is rewritten
into child-local `[0, 3)³` coords, so all math stays O(1)
regardless of absolute depth. The sphere DDA violates that
discipline — it walks the face subtree monolithically in
face-root-normalized coords.

The design doc for the fix is at
**`docs/design/sphere-ribbon-pop-proposal.md`** (branch `main`
commit `ba95cf2`). Read it. It's ~200 lines; the important bits are
"The frame descent" and "Per-level warp handling".

## 2. What the feasibility gate proved

Before touching any DDA code, the previous session built a numeric
gate in `src/world/cubesphere/frame.rs` (module
`ribbon_pop_feasibility`). It tests three candidate Δt formulas at
face-subtree depth 30, 40, 60:

| form                                        | deep-depth behavior |
|---------------------------------------------|---------------------|
| Naive: `tan(u_lo)` vs `tan(u_hi)` subtract  | collapses (Δt = 0)  |
| Factored absolute: `−(A+K·a)/(B+K·b)`       | collapses (Δt = 0)  |
| Cross-product Δt: `K·(A·b−B·a)/(B·(B+K·b))` | f32-precision stable |

The "factored absolute" trap is important — it's the formula that
*looks* right (because it uses `n_delta`), but since A is O(1) and
`K·a` is O(1/3^N), the f32 addition `A + K·a` rounds to `A` at
depth ≥ 20 and the formula silently degrades to the naive result.
**The DDA must compute per-cell relative Δt via cross-product — not
absolute `t(K)` followed by subtraction.**

Radial axis (ray-sphere) has a structurally identical fix using the
rationalized sqrt-difference `(D_b − D_a) / (√D_b + √D_a)`.

**Concrete sweep data** (handoff depth = where ribbon-pop takes
over from exact-warp evaluation). Each row is the depth at which
exact `(n_base, n_delta)` is computed from the warp (tan, sec²)
before ribbon-popping the rest of the way:

```
k_start= 1 rel_err=5.4e-2   (face root — linearization residual dominates)
k_start= 3 rel_err=1.1e-2
k_start= 5 rel_err=1.3e-3
k_start= 7 rel_err=1.5e-4
k_start= 9 rel_err=1.6e-5   (f32-eps floor)
```

**Implication:** the proposal's tentative "handoff at depth 2 or
3" is too aggressive. For sub-percent fidelity, the exact →
linearized handoff should be **depth 5+**, floor at depth 9. At
depth 1–4, use the existing curved face-root march; at depth ≥ 5,
use ribbon-popped `(n_base, n_delta)`. The shader / CPU should
dispatch on this handoff threshold somehow.

The feasibility gate data at depth 30, 40, 60 is *identical* to
four sig figs, which means the extra descents past the handoff add
**zero** accumulated drift. That's the architectural claim the
proposal makes — verified.

One important caveat on the test itself: at depth ≥ ~45 even the
naive f64 reference `tan(u+δ) − tan(u)` collapses in f64. The
tests use the identity
```
tan(u + δ) − tan(u) = tan(δ)·sec²(u) / (1 − tan(u)·tan(δ))
```
to evaluate the reference without cancellation. Any future test
of the proposition at deep depth needs the same trick — the f64
reference *itself* is precision-limited at these scales.

## 3. Architectural decision

The user explicitly chose the "full proposal" (architecture 1) over
a surgical fix:

- **Architecture 1 (chosen):** `compute_render_frame` descends
  through the face subtree slot-by-slot; render frame is a specific
  face-subtree cell at depth N; ray math in frame-local coords;
  same structural shape as Cartesian ribbon-pop.
- **Architecture 2 (rejected):** frame stays at the body; the
  existing DDAs gain ribbon-pop state internally. Smaller blast
  radius but doesn't make sphere frames structurally like
  Cartesian frames.

The decision was made with awareness that architecture 1 is
1000–1500 LoC across 8+ files and must land as a single
visually-coherent commit per the `feedback_no_intermediate_visual_states`
memory. Respect that constraint.

Note the memory entry:
> `project_recursive_architecture.md` — Every layer identical, no
> special leaf layer; sphere-related objects exempted from recursive
> symmetry

So the sphere is *allowed* to differ from Cartesian — but the user
still chose the structurally-aligned architecture. Proceed on that
basis unless the user revises.

## 4. What's already landed (branch `sphere-attempt-2-1`)

Commits in order, newest first:

| sha      | what                                                       |
|----------|------------------------------------------------------------|
| `a537bad` | `CellBoundaries` + cross-product Δt + radial Δt on `FaceFrame` |
| `4805e12` | Split `cubesphere.rs` monolith into `{geometry,frame,worldgen}.rs` + `FaceFrame` type |
| `73ed23a` | Explicit depth-40 sweep + radial axis feasibility tests    |
| `86f986e` | Factored-vs-cross-product test (confirms the trap)         |
| `cf4bf36` | Feasibility gate for ribbon-pop linearization              |
| `ba95cf2` | Proposal doc `docs/design/sphere-ribbon-pop-proposal.md`   |

All 144 lib tests pass. `cargo test --lib` is green on this branch.

### The foundation you have

**`src/world/cubesphere/`** (split of the old monolith):
- `mod.rs` — module declarations + re-exports
- `geometry.rs` — `Face`, `FACE_SLOTS`, `CORE_SLOT`, `ea_to_cube`,
  `face_uv_to_dir`, `body_point_to_face_space`,
  `face_space_to_body_point`, `ray_outer_sphere_hit`,
  `find_body_ancestor_in_path`. Pure geometry — untouched.
- `worldgen.rs` — `PlanetSetup`, `insert_spherical_body`,
  `install_at_root_center`. Also unchanged.
- `frame.rs` — **the new work.** Contains:
  - `FaceFrame` struct with per-axis `(n_base, n_delta)` in
    body-local XYZ + scalar `(r_base, r_delta)` for radial.
  - `FaceFrame::at_face_root(face, inner_r, outer_r)` — build the
    face-root frame from the warp (tan, sec²).
  - `FaceFrame::descend(us, vs, rs)` — one-slot ribbon-pop:
    `n_base_child = n_base + slot · n_delta; n_delta_child = n_delta / 3`.
  - `FaceFrame::descend_path(&slots)` — iterated descend.
  - `FaceFrame::absolute_boundary_ts(o, d, us, vs, rs) -> CellBoundaries`
    — build the 6-face cell boundary set.
  - `CellBoundaries::all_ts(t_after)` / `next_boundary_absolute` —
    shallow-depth absolute-t form. **Collapses at deep depth;
    documented to only use at shallow levels.**
  - `delta_t_cross_product(n_base, n_delta, o, d, k0, k1) -> f32`
    — the precision-stable Δt for a u/v plane cell step.
  - `delta_t_radial(o, d, r_base, r_delta, k0, k1, is_exit_root)`
    — radial analogue using the rationalized sqrt-difference.
  - Three modules of tests: `ribbon_pop_feasibility` (the gate),
    `face_frame_tests` (descend invariants), `dda_helper_tests`
    (validates the primitives at depth 40).

These primitives are the DDA atoms. The subsequent DDA rewrite
should invoke them and not re-derive the math.

## 5. What's not done

The proposal's architecture 1 requires changes across roughly these
files (counts are rough):

| file                                        | change                                   | ~LoC |
|---------------------------------------------|------------------------------------------|------|
| `src/app/frame.rs`                          | `SphereFrame` carries `FaceFrame` + face-descent slots; `compute_render_frame` descends through face subtree | 150 |
| `src/world/raycast/sphere.rs`               | rewrite DDA to consume `FaceFrame` + cross-product Δt | 400+ |
| `src/world/raycast/mod.rs`                  | dispatch signature update                | 50  |
| `assets/shaders/sphere.wgsl`                | rewrite DDA (mirror CPU)                 | 400+ |
| `assets/shaders/bindings.wgsl`              | uniform layout: add `FaceFrame` packed fields | 30  |
| `src/renderer/mod.rs`                       | `set_root_kind_face` signature + uniform mirror | 80  |
| `src/app/edit_actions/upload.rs`            | pass `FaceFrame` uniforms to renderer    | 30  |
| `src/app/edit_actions/mod.rs`               | `frame_aware_raycast` constructs and passes `FaceFrame` | 40  |
| `src/app/mod.rs`                            | `render_frame_kind`, `gpu_camera_for_frame` | 30  |
| `src/app/event_loop.rs`                     | debug overlay sphere-frame branch        | 20  |
| `src/world/aabb.rs`                         | `hit_aabb_body_local` — possibly frame-local for deep hits | 50  |
| `assets/shaders/main.wgsl`, `march.wgsl`    | top-level sphere dispatch                | 30  |
| tests                                       | visual regression, depth descent         | 100+ |

Per `feedback_no_intermediate_visual_states`, all of this needs
to land **as one commit**. Per `feedback_file_rewrite_procedure`,
**use Write, not Edit stacks**, and split further monoliths if any
file grows past 500 LoC.

## 6. Recommended execution order

The last session tried to split this across multiple sessions but
that's risky — the user may want it all together. A plausible order
once you're ready to land the big diff:

1. **`src/app/frame.rs`** — rewrite `SphereFrame` to carry:
   - `body_path: Path` (unchanged)
   - `face_descent: Vec<(u32, u32, u32)>` (UVR slots below face root)
   - `frame: FaceFrame` (the ribbon-popped state at the end of
     `face_descent`)
   - Drop the old `face_u_min, face_v_min, face_r_min, face_size`
     fields — replaced by `FaceFrame` entirely.

2. **`compute_render_frame`** — walk the anchor path past the body.
   When you hit the face-root (`NodeKind::CubedSphereFace`), start
   building a `FaceFrame` via `at_face_root`. For every subsequent
   slot in the anchor path, decode it with `slot_coords` (which
   returns `(us, vs, rs)` — inside a face subtree those ARE the UVR
   slots, so `slot_coords` "just works"), append to
   `face_descent`, `FaceFrame::descend(us, vs, rs)`.

3. **Uniforms** — extend `bindings.wgsl` and `renderer/mod.rs` with
   packed FaceFrame fields. Reasonable packing:
   ```
   root_radii:         vec4<f32> = (inner_r, outer_r, r_base, r_delta)
   root_face_meta:     vec4<u32> = (face_id, descent_depth, _, _)
   root_face_nbuv:     vec4<f32> = (n_base_u.xyz, n_base_v.x)
   root_face_nbv_ndu:  vec4<f32> = (n_base_v.yz, n_delta_u.xy)
   root_face_ndu_ndv:  vec4<f32> = (n_delta_u.z, n_delta_v.xyz)
   ```
   Adds 1 vec4 over current; reuses the 2 old face-window fields.

4. **CPU `cs_raycast`** — new DDA:
   - Enter the render frame at the ray's first intersection with
     the shell / frame boundaries (compute initial `(us, vs, rs)`).
   - Loop: call `FaceFrame::absolute_boundary_ts` only to structure
     the 6 boundary pairs, then compute Δt's via
     `delta_t_cross_product` / `delta_t_radial` — **not** the
     absolute-t forms.
   - Pick min Δt, advance cell slot, descend into child node if
     the tree has one, pop out of frame if the ray exits.
   - Return `HitInfo` with `SphereHitCell` populated. The
     `SphereHitCell` fields `(u_lo, v_lo, r_lo, size)` collapse at
     deep depth — consider adding a `FaceFrame` to
     `SphereHitCell` too, so `aabb.rs` can reconstruct the hit
     cell's corners via ribbon-pop rather than absolute face
     coords.

5. **Shader `sphere.wgsl`** — mirror. WGSL is f32-only (no f64),
   so the same cross-product form is required. The shader has to
   maintain a cell stack (`(us, vs, rs)` per descent level) in
   registers. Depth is bounded by `MAX_FACE_DEPTH` (currently 63)
   but deep-zoom-realistic is ≤ 60. Watch register pressure.

6. **AABB (`aabb.rs`)** — at depth 40 the highlight cell's 8
   corners in body-local coords all round to the same f32 value.
   Current `hit_aabb_body_local` uses absolute face coords —
   replace with a `FaceFrame`-aware form that reconstructs corner
   positions via the ribbon-pop basis, or use frame-local coords
   for the highlight box.

7. **Camera transform** — `WorldPos::in_frame(&body_path)` still
   works for sphere frames (camera stays expressed in body-local
   for ray origin), but keep in mind its **precision is
   f32-limited at body-local magnitude**. The DDA doesn't need
   precise absolute camera position — it needs precise Δt per
   step, which the ribbon-pop form delivers.

8. **Tests**:
   - `render_visibility` headless at sphere-world + anchor depth
     30/40/50. Verify the silhouette is round and grid lines align.
   - `e2e_layer_descent` (existing) should pass; it currently caps
     at depth 38 due to the precision wall. Extend to depth 60 as
     part of the verification.
   - Add a visual regression test in `tests/` that measures
     silhouette curvature at depth 20 vs depth 50 (should be
     identical — infinite-zoom renders LOCAL).

## 7. Known gotchas / subtle points

* **`WorldPos::in_frame` treats all slots as XYZ.** Inside a face
  subtree the slot encoding is `slot_index(us, vs, rs)`, which
  decodes via `slot_coords` as `(us, vs, rs) = (sx, sy, sz)`. The
  *numbers* match, but the *semantics* are UVR not XYZ. For
  `in_frame(&body_path)` this is fine — you stop at the body and
  don't recurse into UVR. For `in_frame(&deeper_face_path)` this
  is wrong: it'd give a UVR-interpreted XYZ position, which is
  not body-local XYZ. **Don't transform camera with a face-descent
  path.** Keep sphere-frame camera transforms rooted at
  `body_path`.

* **`slot_coords` and UVR slot ordering.** `FACE_SLOTS` maps
  `Face as usize → slot_index in the body cell`. Inside a face
  subtree, `slot_index(us, vs, rs)` is UVR where u=first, v=second,
  r=third. `slot_coords` returns `(first, second, third)`.
  Symmetric with Cartesian by coincidence, not by design — don't
  assume it holds for other coord systems.

* **`FaceFrame` in body-local XYZ, ray likewise.** Ray origin
  `ray_o_centered` in `CellBoundaries` is body-local with the body
  sphere center subtracted. All plane normals `(n_base, n_delta)`
  are body-local XYZ directions. Don't re-interpret these in
  face-local coords — the ribbon-pop linearization is valid only
  in body-local.

* **Radial direction ambiguity.** `delta_t_radial(..., is_exit_root)`
  has a sign flag because a ray can cross a concentric sphere
  twice (enter / exit). Pass `false` for the entry root (ray going
  inward), `true` for the exit root. The existing
  `sphere::ray_sphere_after` helper picks whichever root is > t,
  which is typically right but may not be at very shallow grazing
  angles near the inner radius — watch for this in the DDA.

* **`face_lod_depth` is a Nyquist gate.** The old walker uses it
  to truncate descent when the projected cell size is sub-pixel.
  In the new architecture, `compute_render_frame` already picks
  the render frame based on anchor depth; the LOD gate inside
  the DDA should still fire as a safety net but isn't the primary
  mechanism.

* **The `"One big commit"` memory comes from recent scars.** See
  `/Users/donalddellapietra/.claude/projects/-Users-donalddellapietra-GitHub-deepspace-game/memory/feedback_no_intermediate_visual_states.md`
  and `feedback_dont_force_incremental_green.md`. The previous
  session partially wired FaceFrame without populating it from
  `compute_render_frame` — that's as far as you can go while
  staying green. Anything further means committing a state where
  the visuals may be wrong until all pieces land.

* **Don't cargo clean.** Incremental rebuild from this branch
  takes ~3–5s. A clean rebuild takes ~20 minutes. Use
  `rm -rf target/debug/incremental/deepspace_game-*` if
  incrementals get stuck.

* **Verify pwd when running cargo.** This is a worktree at
  `.claude/worktrees/sphere-attempt-2-1`. The shell can silently
  revert to the main repo path; always `pwd` before `cargo`.

* **Temp files in `tmp/` inside the worktree**, not `/tmp`.

## 8. How to verify progress

After any change, expected sanity checks:

```bash
cd /Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/sphere-attempt-2-1
pwd  # must match the above
cargo test --lib                          # 144 tests baseline
cargo test --lib ribbon_pop_feasibility   # precision gate
cargo test --lib dda_helper_tests         # FaceFrame primitives
cargo test --lib planet_world             # existing sphere integration
```

For visual verification:
```bash
./scripts/dev.sh -- --sphere-world --render-harness \
  --screenshot tmp/sphere_d30.png --exit-after-frames 60 --anchor-depth 30
```

Compare silhouettes across depths — should look identical
(infinite-zoom = local coords at any depth).

## 9. What I'd do differently if starting over

Looking at the session history:

* **The feasibility gate was the right first step.** Without it,
  we'd have been tempted to go straight to shader work. The gate
  caught the factored-vs-cross-product trap — and the f64
  reference collapse issue, which would've been a nasty bug in any
  test of the implementation.

* **The file split before the DDA rewrite was probably
  premature.** It was cheap but it mixed reorganization with
  architectural prep. If I were starting over I might keep the
  split for its own commit (as happened) and keep subsequent
  commits focused on semantic changes.

* **I should have been more honest about session scope earlier.**
  I oscillated between "full rewrite in one session" and "land
  scaffolding and pick up next session" instead of picking and
  committing. The final state — primitives landed, DDAs not
  wired — is a reasonable checkpoint but I spent context
  deliberating.

* **Memory entries are load-bearing.** `feedback_capability_trust`
  and `feedback_dont_get_intimidated` push toward committing to
  the full architecturally-correct solution. Respect that when you
  pick up this thread — don't scope down out of caution.

## 10. Ultimate test of your work

The user's observable is a sphere that renders correctly at anchor
depth 40+ with all three paths (CPU raycast, GPU shader, highlight
AABB) agreeing pixel-precisely on which cell is under the crosshair
— same cell for break, same cell for the highlight box, same cell
for what's rendered. When you think you're done, load a sphere
world, zoom to depth 45, aim at the surface, and break a block. If
the break lands where you aimed, and the highlight box outlines the
visible cell, and the visual looks round and grid-aligned, the
architecture works.

Be skeptical of green test suites as the sole success criterion —
per `feedback_green_tests_isnt_fix`, unit tests can pass while the
user's actual observable is broken. Always measure the user's
observable directly at the end.

Good luck. The primitives under `FaceFrame` are trusted; build on
them.
