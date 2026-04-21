# Unified slot‑path + residual DDA

Single raycast primitive for both Cartesian and sphere subtrees.
Retires `sphere::cs_raycast` (body‑XYZ march) entirely. Extends the
local‑frame pattern that already makes `cartesian::cpu_raycast_with_face_depth`
(CPU) and `march.wgsl::march_cartesian` (GPU) scale to layer 60, with
a Jacobian for sphere‑cell curvature and a face‑seam rotation step
at cube edges.

## Why

Today’s build has two sphere raycast paths:

| Path | Where | Status |
|------|-------|--------|
| Body march | `src/world/raycast/sphere.rs` + `assets/shaders/sphere.wgsl` (`cs_raycast`) | Works to ~layer 10, then f32 body‑XYZ cell AABBs fuse. |
| Sub‑frame  | `src/world/raycast/sphere_sub.rs` + `sphere.wgsl::sphere_in_sub_frame` | Built but inert in practice — `compute_render_frame` (`src/app/frame.rs:282`) gates on `m_truncated ≥ MIN_SPHERE_SUB_DEPTH` and the build almost always resolves to `Body`. |

Cartesian scales to ~60 because the render frame is rebased to a
local `[0,3)³` at the camera’s layer (`node_origin`, `cell_size`
never accumulate across global layers — they only cover the bounded
render‑local subtree). The sphere has to adopt the same pattern;
there is no way to keep body‑XYZ as a DDA state variable past layer
10.

The sub‑frame already has the right *shape* (integer `render_path` +
per‑cell Jacobian, rebuilt fresh per neighbor per
`SphereSubFrame::with_neighbor_stepped`, `src/app/frame.rs:95`), but
it was built as a second rendering mode next to body march, so:

- the body↔sub activation handshake (`compute_render_frame`) is
  fragile and rarely takes the sub branch;
- face‑seam crossings terminate (`sphere_sub.rs:499‑503`:
  `with_neighbor_stepped → None ⇒ break`);
- the shader carries two full DDAs (`cs_raycast` + `sphere_in_sub_frame`)
  that must stay in sync for shading, LOD, bevels, atmosphere.

Option B collapses this into one primitive that handles Cartesian
*and* sphere cells *and* face seams uniformly. Matches the
`every layer identical` principle in memory.

## Architecture

### State carried along the ray

    slot_path:  Vec<u8>        // integer chain from world root → current cell
    residual:   [f32; 3]       // position inside current cell ∈ [0, 1)³
    rd_local:   [f32; 3]       // ray direction expressed in current cell's local basis
    cell_kind:  CellKind       // Cartesian | SphereBody | SphereFace(face_id, J)
    t_world:    f32            // distance along ray in world units — for sorting / hit reports only, never for geometry

`slot_path` above the render‑root is precision‑free (`u8` per level).
`residual` stays in `[0, 1)³`, so f32 always has ≥6 decimal digits
of precision inside the current cell regardless of global depth.
`rd_local` is **recomputed** at every basis change from the invariant
`rd_body` (or `rd_world` for pure Cartesian) — never composed from
prior `rd_local` — to avoid rotation accumulation error.

### Transitions (one primitive, four cases)

1. **Descend** (`slot_path.push(child_slot)`): compute `residual` in
   child basis as `(residual - child_offset) * 3.0`, rotate `rd_local`
   into child basis if basis changes (sphere subtree ⇒ rebuild `J`
   at child corner; flat Cartesian ⇒ no change). `cell_kind` updates
   from the child node’s kind.

2. **Advance inside cell**:

       t_exit_local[k] = if rd_local[k] > 0 { (1 - residual[k]) / rd_local[k] }
                         else                { -residual[k] / rd_local[k] }
       axis    = argmin(t_exit_local)
       t_local = t_exit_local[axis]
       residual += rd_local * t_local
       // snap the exiting axis to kill rounding drift
       residual[axis] = if rd_local[axis] > 0 { 1.0 - eps } else { eps }
       t_world += t_local * cell_world_size  // cell_world_size from slot_path length + kind

   All operands are `[0,1)`‑bounded. Safe in f32.

3. **Step neighbor (same parent)**: `slot_path.step_neighbor(axis, sign)`
   — integer carry on the top frame. Residual flips on the crossed
   axis (`0 → 1 - eps` or `1 → eps`). `rd_local` unchanged.

4. **Bubble up past parent**: pop `slot_path` until the neighbor
   step is valid, then `step_neighbor` at that level, then descend
   back down to the target leaf, rebuilding `residual` and `rd_local`
   at each descent. Because each descent division shrinks the
   relevant operands by 3× but `residual` is renormalized to
   `[0,1)`, there is no precision loss at any global depth.

### Face‑seam crossing (novel)

When case 4 tries to bubble past the face‑root of a sphere body,
instead of returning `None` like today’s `with_neighbor_stepped`,
we consult a static cube‑face adjacency table:

    face_neighbor: [[(Face, RotUVR); 4]; 6]
      // faces[from][edge_dir] = (to_face, rotation of (u,v,r) into to_face's basis)

At a face seam the transition is:

- `slot_path`: pop the face‑root slot, push the sibling face slot
  (+ mirror the ancestor `body_path` — unchanged).
- `residual`: map via `RotUVR` (permutation + sign flip, integer),
  then snap the entering axis to `eps` or `1 - eps`.
- `rd_local`: rotate through `RotUVR`. Because `rd_local` is always
  rederived from `rd_body` via the current cell’s `J_inv`,
  alternative implementation is to simply recompute
  `rd_local = J_inv_new · rd_body` at the new face’s corner.
  Prefer the recompute — composition‑free.
- `cell_kind`: `SphereFace(to_face, J_new)` where `J_new` comes from
  `cubesphere::face_frame_jacobian(to_face, …new corner…)`.

The adjacency table is 24 entries (6 faces × 4 edges) of integer
rotation + sign data, plus the `Face → Face` topology from
`cubesphere::FACE_SLOTS`. All static, all testable in isolation.

### What `cell_kind` carries

    enum CellKind {
        Cartesian,
        SphereBody { inner_r, outer_r },      // inside body AABB but not yet in a face subtree
        SphereFace { face: u8, j: Mat3, j_inv: Mat3, un, vn, rn, frame_size },
    }

`SphereBody` is the handoff step between Cartesian above and
`SphereFace` below: when the DDA descends through the
`CubedSphereBody` node, the first descent picks a face slot and
transitions to `SphereFace`. Before picking a face slot the cell is
still Cartesian‑rectilinear (a `3³` grid of face‑root children), so
traversal there is `Cartesian`‑style DDA with no Jacobian.

## Work items (ordered)

The memory `No intermediate visual states` applies: the CPU + GPU
rewrite must land as one diff. Intermediate commits inside the
worktree can be broken; the PR must not be.

### 1. State types + CPU primitive (no shader changes)

- New module `src/world/raycast/unified.rs`.
- Define `CellKind`, `RayState`, `unified_raycast(library, root, ray_origin, ray_dir, max_depth, lod) -> Option<HitInfo>`.
- Cases 1–3 only (descend / advance / same‑parent neighbor). No
  face seam yet. No bubble‑up past face‑root yet.
- Unit tests against `cartesian::cpu_raycast_with_face_depth` on
  flat subtrees: same hit for thousands of random rays at depths
  0 through 60.

### 2. Sphere face‑subtree support

- Extend `unified_raycast` to recognise `CubedSphereBody`: descent
  into a face slot constructs `SphereFace` with fresh `J` via
  `cubesphere::face_frame_jacobian` at the child corner.
- Cases 1–3 only inside a face subtree. Face seam still terminates.
- Parity test against `sphere_sub::cs_raycast_local` at depths
  0 through the current sub‑frame `MIN_SPHERE_SUB_DEPTH` ceiling,
  then extrapolate past it — the unified primitive should keep
  returning hits where `cs_raycast_local` works today.

### 3. Bubble‑up past parent (within one face)

- Add case 4: pop `slot_path` on `cell[axis] ∉ [0,2]`, step parent,
  descend back down rebuilding residual. Mirror of
  `cartesian.rs:88‑97`.
- Tests: random rays that cross arbitrarily many face‑subtree
  internal boundaries in one march.

### 4. Face‑seam table + crossing (novel)

- New `src/world/cubesphere/face_adjacency.rs` with the 24‑entry
  static table. Generator tests that derive it from `FACE_SLOTS`
  and verify round‑trips (`A →edge→ B →edge→ A` is identity).
- Hook into case 4: when `slot_path.pop_to(face_root)` would bubble
  past the face root, consult the table, rotate residual + rebuild
  `cell_kind`, continue the march.
- Tests:
  - Ray tangent to a cube edge crossing all 4 adjacent faces in
    sequence returns one connected hit‑set, not 4 terminated marches.
  - Ray from outside the body, grazing an edge, hits on exactly one
    face (no double‑hit at the seam).

### 5. GPU mirror (`march.wgsl` + `sphere.wgsl` rewrite)

- New shader function `unified_dda(state, rd_body, ...)` with the
  same state machine. Reuses `cs_face_frame_jacobian` and bevel /
  bound / shade helpers already in `sphere.wgsl`.
- Replace the `march_cartesian → sphere_in_sub_frame` dispatch
  (`march.wgsl:864`) with a single `unified_dda` call.
- Visual regression screenshots at layers 0, 5, 10, 15, 20, 30, 45, 60:
  – cartesian cube  (baseline from main)
  – sphere outside  (shallow m)
  – sphere near surface  (m ~ 10)
  – sphere deep inside  (m ~ 30 and m ~ 60)
  – sphere spanning a cube edge  (seam test)

### 6. Retire body march + sub‑frame scaffolding

- Delete `sphere::cs_raycast` (Rust) and `sphere_in_sub_frame` +
  `cs_raycast` body march branch (WGSL).
- Delete `ActiveFrameKind::SphereSub` (the render‑frame kind
  becomes Cartesian everywhere; sphere cells are just a cell kind
  inside the tree). Simplify `compute_render_frame` and
  `SphereSubFrame` in `src/app/frame.rs`.
- Delete `src/world/raycast/sphere_sub.rs` and the `sphere.rs`
  body‑march functions. Keep the `cubesphere::` geometry helpers
  (`face_frame_jacobian`, `FACE_SLOTS`, etc.) — they’re reused.
- Shrink `assets/shaders/sphere.wgsl` to just shading + Jacobian
  helpers (~400 LoC, down from 1570).

### 7. Regression + performance sweep

- Run `tests/e2e_sphere_descent.rs`, `sphere_zoom_seamless.rs`,
  existing Cartesian DDA tests. All must pass without `#[ignore]`.
- Render perf at layers 0, 10, 20, 40, 60 — must be within noise
  of current Cartesian perf at equivalent depth (memory: `no
  performance hits`, `“fundamentally expensive” is never acceptable`).

## Precision invariants

At every DDA step:

- `residual ∈ [0, 1)³`, bounded by construction.
- `rd_local` is a rotation of the original `rd_body` (or `rd_world`)
  through the current cell’s `J_inv`. Never composed from the
  previous step’s `rd_local`.
- `t_world` is advanced monotonically by `t_local * cell_world_size`,
  where `cell_world_size = render_root_size * (1/3)^levels_below_root`.
  Levels below render‑root is bounded (render frame is already
  rebased locally), so `cell_world_size` stays in a sane f32 range.
- `slot_path` above the render‑root is integer.
- `slot_path` below the render‑root is bounded in length by render
  depth budget (same as Cartesian today).

There is no f32 quantity whose magnitude grows or shrinks with
global layer depth. That’s the whole point.

## What gets deleted

    src/world/raycast/sphere.rs              (536 LoC — cs_raycast)
    src/world/raycast/sphere_sub.rs          (1147 LoC — cs_raycast_local)
    assets/shaders/sphere.wgsl               (~1100 LoC of body march + sub_frame; keep ~400 LoC helpers)
    src/app/frame.rs: SphereSubFrame         + ActiveFrameKind::SphereSub
    src/world/aabb.rs: SphereSub‑specific Jacobian projection branch

Net: ~2500 LoC removed, ~700 LoC added (`unified.rs` + shader
`unified_dda` + face adjacency table + tests). One primitive to
reason about, one primitive to debug.

## Risks / open questions

1. **Bevels and shading at face seams.** Today, `shade_pixel` has
   special cases for `sphere_in_sub_frame` hits
   (`sphere.wgsl::shade_pixel`’s `cube_face_bevel` neutralisation,
   commit `0123745`). At a seam, the hit’s local basis changes —
   shading needs either to reconstruct body‑XYZ normals from
   `(slot_path, residual, cell_kind.j)` at hit time, or to carry
   `body_normal` explicitly. **Decision: reconstruct at hit time**;
   shading already has body‑XYZ coords today via `body_point_to_face_space`.

2. **Atmosphere / horizon ray‑sphere entry.** Body march today
   does one ray‑sphere intersection up front to clip to
   `[inner_r, outer_r]`. Unified primitive does this during the
   Cartesian → SphereBody descent: when entering the
   `CubedSphereBody` cell, intersect the ray against the outer
   shell AABB *and* the `outer_r` sphere, pick the later entry.
   Same for atmosphere halo — compute once at outer entry, fade
   based on `t_world`.

3. **LOD depth consistency across seams.** LOD today is a function
   of `slot_path.len()` and distance. At a seam, `slot_path` length
   can differ between old‑face and new‑face by 0 (same face‑subtree
   depth) or more (if the seam crossing bubbles up several levels
   on one side). Keep LOD = `f(slot_path.len(), t_world)` — length
   is continuous across a seam because we only push/pop equal
   counts.

4. **Face‑seam rotation at pathological angles.** Rays nearly
   tangent to two face seams simultaneously (hitting a cube
   vertex). Resolve by defining vertex handling as "step through
   one seam then immediately through the next" — i.e. two case‑4
   applications in one loop iteration with `t_local ≈ 0`. Cap at
   3 zero‑advance steps per iteration to avoid infinite loops at
   degenerate angles.

5. **Scope discipline.** Memory `Don't force incremental green`
   and `No intermediate visual states` mean items 1–6 above land
   as one logical change. Use the worktree to commit intermediate
   broken states; squash or leave them when landing on main. Do
   not route `unified_dda` through a feature flag alongside the
   old paths — that’s the shim the `no shortcuts` memory
   specifically rejects.

## Acceptance

- Sphere renders at layer 60 with the same visual quality as
  Cartesian at layer 60.
- Ray grazing a cube edge renders continuously across the seam
  (test: screenshot along the seam line shows no gap or double
  image).
- All existing sphere / cartesian tests pass without `#[ignore]`.
- Render perf at every depth is within noise of Cartesian at
  equivalent depth.
- `src/world/raycast/sphere.rs` and `sphere_sub.rs` deleted;
  `ActiveFrameKind::SphereSub` removed; `sphere.wgsl` contains
  only shading/Jacobian helpers.
