# Wrapped Cartesian Planet — Detailed Implementation Plan

Companion to `docs/design/wrapped-cartesian-planet-plan.md`. Refines every
phase to file:line resolution. Where the high-level plan was wrong about a
file path or contract, the correction is called out inline.

Scope assumptions cross-referenced against the actual code:

- `src/world/tree.rs:84-94` — `NodeKind` is `Cartesian | CubedSphereBody |
  CubedSphereFace`. `WrappedPlane` will be a fourth variant.
- `src/world/anchor.rs:104-128` — `step_neighbor_cartesian` already bubbles
  up; no kind dispatch yet. Wrap-aware step is a localized change at one
  recursion depth.
- `src/world/gpu/pack.rs:208-227` — uniform-flatten gating for non-Cartesian
  kinds is a single `matches!(node.kind, NodeKind::Cartesian)` check. Wrap
  metadata sits orthogonally on a Cartesian node, so the gating does NOT
  need to grow a new branch unless we choose a separate kind.
- `assets/shaders/main.wgsl:5-11` — `#include "sphere.wgsl"` is the only
  WGSL coupling between the legacy sphere path and the marcher; deleting
  the include + the dispatch in `march.wgsl:558-581` excises the entire
  sphere render path.
- `assets/shaders/march.wgsl:271-796` — `march_cartesian` is the complete
  Cartesian DDA. Wrap and curvature both hook into it without changing
  signatures.

---

## Phase 0b — Remove legacy cubed-sphere code

### Files to delete entirely

- `src/world/cubesphere.rs` (478 LoC).
- `src/world/cubesphere_local.rs` (size unknown but it's a sibling helper).
- `src/world/raycast/sphere.rs` (258 LoC).
- `src/world/spherical_worldgen.rs` (only call sites: `bootstrap.rs:614,
  616` and `raycast/mod.rs:403, 423`).
- `assets/shaders/sphere.wgsl`.
- `assets/shaders/face_math.wgsl`.
- `assets/shaders/face_walk.wgsl` (verify it's unused outside sphere.wgsl).
- `src/app/frame.rs` is **mostly** deleted: the `SphereFrame` struct
  (`src/app/frame.rs:18-32`), `ActiveFrameKind::Body` and `Sphere` variants
  (`src/app/frame.rs:34-39`), `frame_point_to_body`
  (`src/app/frame.rs:54-70`), and the sphere branches inside
  `compute_render_frame` (`src/app/frame.rs:96-156`) all go. What survives:
  the slim `ActiveFrame { render_path, logical_path, node_id, kind:
  Cartesian }` carrier and `frame_from_slots` (line 74-80). Reduce file to
  ~80 LoC.

### Files to edit

- `src/world/tree.rs:84-118` — drop `NodeKind::CubedSphereBody` and
  `CubedSphereFace`. The `Hash for NodeKind` impl
  (`src/world/tree.rs:104-118`) collapses to the two unit-discriminant
  variants we end up with. `use super::cubesphere::Face` (line 12) is
  removed.
- `src/world/tree.rs:449-461` — delete `dedup_respects_node_kind` test (it
  references `CubedSphereBody`).
- `src/world/gpu/types.rs:79-90` — `GpuNodeKind::from_node_kind` collapses
  to the two surviving variants. `inner_r/outer_r` fields and `face` field
  on the GPU struct can stay (16-byte alignment is required) but become
  zero-initialized junk. Plan: rename them to `param_a/param_b/param_c`
  generic slots — Phase 1 will repurpose them for slab dims.
- `src/world/gpu/types.rs:148-186` — drop the
  `from_node_kind_body_carries_radii` and `from_node_kind_face_carries_face_id`
  tests.
- `src/world/gpu/pack.rs:208-227` — the `is_cart` predicate folds away
  (every node is now Cartesian or WrappedPlane). Replace
  `matches!(node.kind, NodeKind::Cartesian)` with
  `node.kind.allows_uniform_flatten()` — see Phase 1.
- `src/world/gpu/pack.rs:355-405` — delete `planet_world` test fixture and
  `pack_includes_body_kind_and_radii` / `pack_flattens_uniform_empty_siblings`.
- `assets/shaders/main.wgsl:10` — remove `#include "sphere.wgsl"`.
- `assets/shaders/march.wgsl:4` — remove `#include "sphere.wgsl"`.
- `assets/shaders/march.wgsl:558-581` — delete the `kind == 1u` (body)
  dispatch. The remaining path is the unified Cartesian DDA + (Phase 1's)
  WrappedPlane wrap test.
- `assets/shaders/march.wgsl:808-988` — `march()` collapses: delete every
  `current_kind == ROOT_KIND_BODY/FACE` branch and the
  `body_pop_level/face_root_meta` ribbon machinery
  (`assets/shaders/march.wgsl:878-912`). What remains is the Cartesian
  pop-loop only.
- `assets/shaders/bindings.wgsl:60-67` — remove `root_radii`,
  `root_face_meta`, `root_face_bounds`, `root_face_pop_pos`. Keep the
  `root_kind` u32 — Phase 2 reuses it as a "WrappedPlane root" flag.
- `src/renderer/buffers.rs:295-308` — the GpuUniforms layout drops the four
  removed fields. Pad to 16-byte boundaries explicitly with
  `_pad_face: [u32; N]` slots.
- `src/world/raycast/mod.rs:1-` — the `cpu_raycast_in_sphere_frame` entry
  point (referenced from `edit_actions/mod.rs:69-85`) is removed. The
  `frame_aware_raycast` match arm collapses to its Cartesian branch.
- `src/world/edit.rs` — anywhere it dispatches on
  `NodeKind::CubedSphereBody/Face` (grep before deletion). All such dispatch
  goes; edits propagate Cartesian-style only.
- `src/world/aabb.rs` — delete `hit_aabb_body_local` (sphere-only path
  referenced from `edit_actions/mod.rs:148-150`).
- `src/world/bootstrap.rs:614-616` — delete the planet-installation block.
- `src/world/raycast/mod.rs:399-440` — delete the spherical_worldgen test
  module gates.
- `src/app/edit_actions/mod.rs:65-86, 119-136, 148-153` — collapse the
  sphere arm of `frame_aware_raycast`. The Cartesian arm becomes the only
  arm.
- `src/app/edit_actions/zoom.rs` — drop sphere-frame branches; verify zoom
  still works for Cartesian.
- `src/app/mod.rs`, `src/shader_compose.rs`, `src/renderer/mod.rs` —
  drop unused imports, sphere-feature gates, sphere-shader composition.
- `src/world/mod.rs:15` — remove `pub mod spherical_worldgen;`. Also
  remove `pub mod cubesphere;` and `pub mod cubesphere_local;`.

### Validation

- `cargo build --release` from worktree pwd succeeds.
- `cargo test --release` passes; tests that referenced sphere worlds either
  go away or assert the new `--wrapped-planet` worldgen (Phase 1).
- `grep -r "CubedSphere\|sphere_in_cell\|march_face_root\|SphereFrame" src
  assets` returns zero. (Acceptable matches: docs/, comments referencing
  past attempts.)
- Render harness with `--plain-world --plain-layers 8 --spawn-depth 6`
  produces a non-sky screenshot to confirm Cartesian path still works.

### Risks

- `aabb.rs` and `edit.rs` may have non-sphere call sites that still drag
  in sphere helpers. Scan before deletion.
- `src/app/frame.rs::compute_render_frame` is called from many places; the
  trimmed signature stays the same (returns `ActiveFrame`) but its kind
  variant set shrinks. Callers that match on `Sphere/Body` arms must drop
  those arms.
- WASM build (`scripts/dev-wasm.sh`) references shader files via
  `#include` resolution at WGSL-composition time; double-check
  `src/shader_compose.rs` doesn't break.

### Estimated commit size

~1100 LoC removed, ~30 LoC added (just where deletions leave a hole that
needs a stub). One commit.

---

## Phase 0c — Visual harness port

The reference harness lives in
`/Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/sphere-attempt-2-2-3-2/tests/sphere_zoom_seamless.rs`
(read it for the pattern; do not depend on its sphere-specific args).

### Files to add

- `tests/wrapped_planet_visual.rs` — top-down + edge-on screenshots,
  silhouette analysis. Pattern lifted from `sphere_zoom_seamless.rs:60-90`
  (`planet_fraction`) and `render_visibility.rs:233-258` (`image_diff`).
- `tests/wrapped_planet_visual/harness.rs` — symlink or `#[path]` to
  `tests/e2e_layer_descent/harness.rs` so we reuse `ScriptBuilder`, `run`,
  `tmp_dir`. (Existing file referenced from `e2e_layer_descent.rs:18-19`.)

### Concrete additions

In `tests/wrapped_planet_visual.rs`:

```rust
fn altitude_args(altitude_world_units: f32, depth: u8, png: &str) -> Vec<String> {
    // Camera at (1.5, 1.5 + altitude, 1.5) looking straight down at the
    // slab embedded near root center. depth = camera anchor depth.
    let cam_y = 1.5 + altitude_world_units;
    vec![
        "--render-harness".into(), "--wrapped-planet".into(),
        "--spawn-depth".into(), depth.to_string(),
        "--spawn-xyz".into(), "1.5".into(), format!("{cam_y:.6}"), "1.5".into(),
        "--spawn-pitch".into(), "-1.5707".into(),
        "--spawn-yaw".into(), "0".into(),
        "--harness-width".into(), "480".into(),
        "--harness-height".into(), "320".into(),
        "--exit-after-frames".into(), "60".into(),
        "--timeout-secs".into(), "30".into(),
        "--suppress-startup-logs".into(),
        "--screenshot".into(), png.into(),
    ]
}
```

Three test functions:

1. `top_down_at_altitude_steps` — sweep altitudes
   `[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]` (slab-radius units relative to the
   embedded planet radius computed in Phase 3). Compute `planet_fraction`
   per shot. Assert monotonic decrease in coverage as altitude rises and
   coverage stays > 0.005 at every step.
2. `edge_on_silhouette_circular_at_orbit` — camera at `--spawn-pitch 0`
   looking horizontally at the slab from a high orbital altitude; compute
   silhouette **circularity score**: detect non-sky pixels, find the
   bounding box, fit a least-squares circle (or measure
   `aspect = bbox.height/bbox.width`). Assert `0.85 < aspect < 1.15` for
   a true sphere. Pre-Phase-3 this test is `#[ignore]` and gets enabled in
   Phase 3.
3. `slab_at_low_altitude_renders_flat` — altitude < 0.01 slab-radius units;
   silhouette is a horizontal band. Asserts `bbox.height < 0.4 * frame_h`
   and the band is approximately rectangular.

### CLI plumbing

- Add `--wrapped-planet` CLI flag in the harness arg parser. Its body in
  `bootstrap.rs` builds a slab at depth 22 (Phase 1 handles this).
- Add `--spawn-pitch`, `--spawn-yaw`, `--spawn-xyz` are already present
  per `e2e_layer_descent.rs:39-46`. No new flags needed.

### Validation

- `cargo test --release wrapped_planet_visual::slab_at_low_altitude` after
  Phase 1 lands.
- `cargo test --release wrapped_planet_visual::top_down_at_altitude_steps`
  after Phase 1 (no curvature) — coverage should still decrease with
  altitude purely from perspective.
- Phase 3 enables `edge_on_silhouette_circular_at_orbit`.
- Screenshots dropped to `<worktree>/tmp/wrapped_planet_visual/` — never
  `/tmp` (memory rule).

### Risks

- Headless test harness needs `--render-harness` flag working correctly;
  `tests/render_visibility.rs:269-282` shows the sandbox-skip pattern.
  Reproduce that.
- Image-analysis must be tolerant of TAA jitter. Use `is_sky` heuristics
  (planet color is not blue) rather than golden-image diff.
- Estimated commit size: ~250 LoC test code, no source changes.

---

## Phase 1 — Slab as Cartesian content (no wrap, no curvature)

### Architectural decision

The high-level plan offers two options: a new `NodeKind::WrappedPlane {
dims }` or wrap-metadata on a Cartesian root. **Recommendation: new
NodeKind variant.** Reasons:

1. Pack-time uniform-flatten gating already keys on `NodeKind` (see
   `src/world/gpu/pack.rs:208-219`). A NodeKind discriminant is the
   minimum-friction extension point.
2. The shader's marcher will need a kind-dispatched wrap rule (Phase 2);
   running it on the **slab root** specifically (not on every Cartesian
   node) requires a NodeKind tag.
3. We're paying for the `GpuNodeKind` 16 bytes/node anyway.

### Files to touch

- `src/world/tree.rs:84-94` — extend the enum:

  ```rust
  pub enum NodeKind {
      Cartesian,
      WrappedPlane {
          /// Slab extent in cells along (x, y, z) at SLAB_DEPTH levels
          /// below this node. The slab root sits at the top of the
          /// flat Cartesian subtree; the actual cells live at
          /// (slab_root.depth + slab_depth) in the world tree.
          dims: [u32; 3],
          /// Depth of the slab below this NodeKind (typically 2-4 for
          /// a slab inside a 27³ at 22).
          slab_depth: u8,
      },
  }
  ```
- `src/world/tree.rs:104-118` — `Hash for NodeKind` adds the variant:
  hash discriminant, then `dims[0..2].hash(state)` and `slab_depth.hash`.
- `src/world/tree.rs:96-100` — `Default` impl unchanged (Cartesian).
- `src/world/tree.rs` — add helper:

  ```rust
  impl NodeKind {
      pub fn allows_uniform_flatten(self) -> bool {
          matches!(self, NodeKind::Cartesian)
      }
  }
  ```

  WrappedPlane must NOT be uniform-flattened or its slab dims metadata
  vanishes.

- `src/world/gpu/types.rs:69-89` — `GpuNodeKind`:

  ```rust
  pub struct GpuNodeKind {
      pub kind: u32,                  // 0=Cartesian, 1=WrappedPlane
      pub dims_x: u32,                // unused for Cartesian
      pub dims_y: u32,
      pub dims_z: u32,
      // 16-byte total — alignment preserved.
  }
  ```

  `from_node_kind` returns `kind=1` and the dims for WrappedPlane.

- `src/world/gpu/pack.rs:217-219` — replace `is_cart && uniform_type ...`
  with `node.kind.allows_uniform_flatten() && uniform_type ...`.

- `src/world/bootstrap.rs` — add `pub fn wrapped_planet_world(
  embedding_depth: u8, slab_dims: [u32; 3], slab_depth: u8) -> WorldState`.
  The slab is built as:
  - Embedding: at root, descend (e.g.) 20 levels of uniform-empty
    Cartesian nodes following a chosen path (slot 13 every level).
  - At depth 20 (`embedding_depth`), insert a `NodeKind::WrappedPlane {
    dims: [20, 10, 2], slab_depth: 2 }` node. Its 27 children are
    constructed as a **flat Cartesian subtree** that, when descended
    `slab_depth` levels, produces a `dims.x × dims.y × dims.z` grid of
    leaf blocks (grass at the surface, dirt below, stone at the floor —
    same surface palette as `plain_world`).
  - Outside `dims.x × dims.y × dims.z` cells inside the 27³ slab volume,
    children are `Empty`. This is "sparse occupancy = unset = absent"
    per the architecture.

  Concrete: at `slab_depth=2`, the root node's 3³=27 children point to
  9 child nodes (`3 × 3 × 1 = 9`) covering the slab footprint, plus
  18 Empty slots. Each child node holds `~3 × 3 × 2 = 18` non-empty
  grandchildren plus 9 empty.

- `src/world/bootstrap.rs` — wire `WorldPreset::WrappedPlane` so
  `--wrapped-planet` CLI dispatches to it.

- `src/main.rs` (or wherever CLI args parse) — add `--wrapped-planet` flag.

- `assets/shaders/march.wgsl` — **no shader change needed in Phase 1**.
  WrappedPlane has `kind=1` but the marcher's tag-2 descent path doesn't
  branch on it yet. The slab renders identically to a Cartesian subtree;
  wrap and curvature land in Phases 2 and 3.

### Validation

- `cargo build --release`.
- `cargo test --release wrapped_planet_visual::slab_at_low_altitude_renders_flat`.
- Render harness: `--wrapped-planet --spawn-depth 4 --spawn-xyz 1.5 1.55
  1.5 --spawn-pitch -1.5707 --screenshot /tmp/slab_top.png` — should show
  the rectangular slab from above.
- Screenshot from spawn-pitch 0 at the slab surface — should look like
  `--plain-world` flat ground.
- `cargo test --release` — full suite green.

### Risks

- Slab at depth 22 inside 27³ at depth 20 means the embedding cell is
  `WORLD_SIZE / 3^20 ≈ 8e-10` world units. Camera anchor at the slab uses
  the WorldPos symbolic-anchor discipline (`src/world/anchor.rs:181-348`),
  which is precision-safe at any depth (the existing Cartesian render
  already passes deep-zoom tests). No new precision issue here — but
  verify the slab's BFS-pack size doesn't blow up due to many distinct
  slot patterns. Use `pack_tree` regression tests as a guard.
- `--spawn-xyz` arithmetic must produce a WorldPos whose anchor
  prefix matches the slab's anchor prefix. The harness's existing flag
  uses `WorldPos::from_frame_local(&Path::root(), xyz, anchor_depth)`
  which handles this.

### Estimated commit size

~350 LoC: 100 in tree.rs/gpu/types.rs/pack.rs, 200 in bootstrap.rs (the
slab-builder), 50 plumbing.

---

## Phase 2 — X-wrap

The defining feature: walking east → returning west. Implemented at the
slab root (the WrappedPlane NodeKind boundary), NOT at every Cartesian
ancestor.

### CPU side: `step_neighbor_cartesian`

`src/world/anchor.rs:104-128` is where Cartesian neighbor stepping
bubbles up. Currently it has no kind awareness (the comment says
"Cartesian interpretation only" — line 236). Wrap stepping needs to
consult the parent node's `NodeKind`.

The wrap rule is: when stepping past the X boundary inside a
WrappedPlane subtree, the bubble-up is **truncated at the WrappedPlane
root** and the slot index wraps modulo `dims.x` (in slot units at the
slab-root depth).

Two implementation options:

(A) Library-aware step: add `step_neighbor(library, axis, direction)` —
the new WorldPos primitive consults the library to walk slot kinds.

(B) Caller-side wrap: every place that calls `step_neighbor_cartesian`
near the slab boundary calls a wrap-aware variant.

**Recommendation: (A).** It's a one-time cost in the primitive, and
every consumer (`add_local`, `renormalize_cartesian`, the editing
pipeline, etc.) gets the wrap correctly without per-call-site
rewrites. Note that `WorldPos::add_local`
(`src/world/anchor.rs:268-274`) takes `_lib: &NodeLibrary`; it already
plumbs the library through, so kind-dispatching here is a one-line
threading.

### Files to touch (CPU)

- `src/world/anchor.rs:104-128` — `Path::step_neighbor_cartesian` keeps
  its current signature (kind-agnostic) but gains a sibling:

  ```rust
  pub fn step_neighbor_in_world(
      &mut self,
      library: &NodeLibrary,
      world_root: NodeId,
      axis: usize,
      direction: i32,
  )
  ```

  Recursive logic:
  1. Walk `self` from root, tracking the `NodeId` at each depth.
  2. When we reach `self.depth-1`, attempt the local step.
  3. On overflow, look at the **parent's NodeKind**. If parent is
     `Cartesian`, bubble up as today. If parent is `WrappedPlane { dims }`
     and the overflow axis matches the wrap axis (X), wrap the slot
     index modulo `dims.x` at the slab-root level instead of bubbling.
- `src/world/anchor.rs:237-262` — `renormalize_cartesian` becomes
  `renormalize_world(&mut self, library: &NodeLibrary, root: NodeId)`,
  calls the new step. The existing free function stays as an internal
  fallback for kind-agnostic callers.
- `src/world/anchor.rs:268-274` — `add_local` passes `library` plus the
  caller-supplied root through to `renormalize_world`.

  **Correction to the high-level plan:** `step_neighbor_cartesian` does
  not get edited; we add a sibling method. Renaming would force every
  call site to thread the library; not all of them need it.

- Every call site of `add_local` must already have the library (most do
  — `entities.rs:154-156` already does). Audit and thread `world_root`
  where missing.

### Files to touch (GPU)

- `assets/shaders/march.wgsl:386-419` — the OOB pop branch in
  `march_cartesian`. When `depth == 0` and `current_kind ==
  ROOT_KIND_WRAPPED_PLANE` (Phase 0b kept `root_kind`; Phase 2 reuses
  it), the OOB on the wrap axis becomes a **wrap** instead of a pop:

  ```wgsl
  // Pseudo: at depth 0 with WrappedPlane root
  if depth == 0u && current_kind == ROOT_KIND_WRAPPED_PLANE {
      let dims_x = uniforms.slab_dims.x;
      // cell.x went to -1 or dims_x. Wrap modulo dims_x.
      let new_x = (cell.x + i32(dims_x)) % i32(dims_x);
      let dx = new_x - cell.x;
      // Translate ray_origin by ±(dims_x in slab-root local units).
      // Slab-root spans [0, 3) in local coords; one cell at slab_depth
      // in the slab-local frame is 3.0 / 3^slab_depth. The wrap shift
      // is dims_x * cell_size_at_slab_depth, applied to ray_origin.x.
      let wrap_shift = f32(dims_x) * (3.0 / pow(3.0, f32(uniforms.slab_depth)));
      ray_origin.x = ray_origin.x - f32(dx) * wrap_shift;
      // Re-init the DDA at the new cell.
      // ... reuse the descent reset code.
      continue;
  }
  ```

  The wrap is applied in the slab-root's `[0, 3)³` local frame so f32
  precision is bounded — this is the same precision discipline as the
  ribbon pop. **Important:** do NOT translate in absolute world
  coords; that would lose precision at deep camera anchors.

- `assets/shaders/bindings.wgsl:60-67` — extend `Uniforms`:
  ```wgsl
  slab_dims: vec4<u32>,    // (x, y, z, slab_depth)
  ```
- `src/world/gpu/types.rs::GpuUniforms` — corresponding CPU side.
- `src/renderer/buffers.rs:286-311` — populate `slab_dims` from
  `app.active_frame` when the WrappedPlane root is the render frame.
- `src/app/frame.rs::compute_render_frame` — when the descent encounters
  a `NodeKind::WrappedPlane`, set `ActiveFrameKind::WrappedPlane { dims,
  slab_depth }` and stop descending into the slab subtree (the render
  frame should be the slab root, not a sub-cell, so the marcher's wrap
  branch fires at the right depth).

### Pop-from-WrappedPlane semantics

When the ray exits the slab on the **non-wrap** axes (Y or Z) and the
slab is the render frame root, normal ribbon-pop applies — the ray
exits to the surrounding empty 27³ cells. No special case.

### Validation

- Walk-east-return-west test:
  ```rust
  // Camera at (cell 0, surface, mid-z) at slab-root depth.
  // Step east dims_x times. Camera anchor must be back to cell 0.
  ```
  Add `tests/wrapped_planet_wrap.rs` with a unit test on `add_local`
  using a hand-built WrappedPlane world.
- Visual test: `tests/wrapped_planet_visual::ray_east_hits_west_side`.
  Spawn camera at `(slab_x_min + 0.1, surface + 0.5, mid_z)` looking
  east (yaw=π/2). Place a marker block at `(slab_x_max - 0.5, surface +
  0.5, mid_z)` — the ray launched east should hit it via wrap.
- Visual: `--wrapped-planet --spawn-pitch 0 --spawn-yaw 0 --screenshot`
  shows continuous terrain at the X-wrap seam (no missing column).

### Risks

- Wrap on the GPU must reset the DDA's `s_cell`, `cur_side_dist`, and
  cached header state. The branch is non-trivial — the cleanest
  pattern is to factor the descent-reset block (`assets/shaders/march.wgsl:751-790`)
  into a helper and call it from both the descent path and the new wrap
  path.
- Ribbon-pop interaction: when the camera is OUTSIDE the slab and the
  ribbon pops UP through the WrappedPlane root, no wrap should apply
  (we're outside the wrap regime). The wrap branch must gate on
  `current_kind == ROOT_KIND_WRAPPED_PLANE && depth == 0` AND the OOB
  axis being the wrap axis.
- Iteration cap: a ray bouncing along the wrap axis indefinitely could
  exhaust `max_iterations = 2048` (`assets/shaders/march.wgsl:377`). The
  cap already prevents infinite loops; verify wraps don't pathologically
  multiply iterations.

### Estimated commit size

~400 LoC: 80 in `anchor.rs` (new step primitive + tests), 50 in
`tree.rs` (kind threading), 100 in `march.wgsl` (wrap branch),
40 in `bindings.wgsl` + buffers + types, 130 in tests.

---

## Phase 3 — Render-time curvature (coordinator-owned)

This phase is render-only. Storage and simulation never see curvature.

### 3a — Camera altitude and `k(altitude)`

#### What is altitude?

In the slab's local frame the slab top surface lives at some Y
coordinate `slab_surface_y`. Define
`altitude = (camera_y - slab_surface_y) / R` where `R` is the
**effective planet radius** (defined in 3b). Both `camera_y` and
`slab_surface_y` are in the slab-root's local `[0, 3)` frame, so this
ratio is precision-stable at any anchor depth.

`altitude = 0` means camera at the slab surface (k=0 → flat march, no
bending).
`altitude = 1` means camera at one planet radius above the surface
(k≈0.5 — see curve below).
`altitude → ∞` means k saturates at 1 (full spherical mode).

#### Curve choice

Three candidates, evaluated:

- **Linear `k = clamp(altitude, 0, 1)`** — sharp transition at altitude=1.
  Rejected: visible "horizon just bent" at the saturation point.
- **Sigmoid `k = 1 / (1 + exp(-(altitude - 0.5) * 6))`** — smooth, S-shaped.
  Rejected: too aggressive at the low end (k ≈ 0.05 at altitude=0,
  noticeable parabolic bend on a "flat" surface walk).
- **Power `k = 1 - 1 / (1 + altitude^p)`** with `p = 1.5` — the
  recommendation. Properties:
  - `k(0) = 0` exactly (flat march on the surface).
  - `k(1) ≈ 0.5` (half-curved at one radius up).
  - `k(3) ≈ 0.84`.
  - `k(10) ≈ 0.97`.
  - Saturates smoothly without an obvious knee.

  WGSL:
  ```wgsl
  fn altitude_to_k(altitude: f32) -> f32 {
      let a = max(altitude, 0.0);
      let p = 1.5;
      return 1.0 - 1.0 / (1.0 + pow(a, p));
  }
  ```

  Compute `altitude` once per frame on CPU, pass `k` as uniform.

### 3b — Planet radius from slab circumference

The wrap circumference at the slab equator is
`C = dims.x * cell_size_at_slab_depth_in_slab_local_units`. In the slab
root's `[0, 3)` frame, one slab-depth cell is
`cell_size = 3.0 / 3^slab_depth`. So
`C = dims.x * 3.0 / 3^slab_depth`.

The implied planet radius for the curvature math is `R = C / (2π)`. In
slab-local units. This is the value the shader uses in the bending
formula.

For `dims.x = 20, slab_depth = 2`: `cell_size = 3/9 ≈ 0.333`,
`C = 20 * 0.333 ≈ 6.67`, `R ≈ 1.06`. Same units as ray_origin/ray_dir
in the slab frame — **consistent**.

CPU computes `R` once per frame, passes as uniform `slab_R: f32`.

### 3c — Per-step ray bending math

The architecture doc (line 45) prescribes:

> a per-step adjustment to sample position based on accumulated distance
> and the curvature parameter ... subtracting `(distance² · k) / (2R)`
> from the height coordinate

Derivation. A great-circle arc on a sphere of radius R, parameterized by
arclength s, has tangent slope falling at rate 1/R. A flat-ground ray
travelling distance s along the surface drops below the curved surface
by `s² / (2R)` (small-angle Taylor expansion of `R(1 - cos(s/R))`).
Multiplying by `k ∈ [0, 1]` interpolates between flat (k=0) and full
parabolic (k=1).

#### Dimensional check

`s` has units of slab-local length. `R` has slab-local length. `s²/R`
has slab-local length. The drop is a vertical offset added to the ray
sample's slab-local Y. **Dimensionally consistent.**

#### Where in the shader

Two strategies:

(i) **Once at ray entry.** Pre-curve the ray direction by a small
downward tilt; the marcher walks a straight ray. Cheap (one extra
vector op), but the curve isn't quadratic — it's linear after the
tilt. Falls apart for distances > R.

(ii) **Per DDA step.** At each `cur_side_dist` advance in
`march_cartesian`, offset the sampled position by `(t_cumulative² *
k) / (2R)` along the local "down" direction. Accurate for any t,
costs one FMA per inner-loop iteration.

**Recommendation: per DDA step.** The 1-FMA cost is negligible against
the existing per-iteration loads (see `shader_stats.avg_loads_total`
from `src/renderer/draw.rs:116-119`). Visual quality matters more than
shaving a few cycles in a path that's already memory-bound.

#### Insertion point

`assets/shaders/march.wgsl:271-796` — `march_cartesian`. The bending
hook lives at the **side_dist update step**: each axis-min advance
along the ray, after `cur_side_dist += m * delta_dist *
cur_cell_size`, recompute the **effective Y** of the cell-center
sample by subtracting `(t² · k) / (2R)`.

But the DDA as written tests `if ((cur_occupancy & slot_bit) == 0u)`
against the integer cell coords — which were computed assuming a
straight ray. Two options to reconcile:

(α) **Bend the ray, not the data.** Maintain a virtual `t_curved`. At
each side_dist crossing, transform the ray's sample position by `pos_y
+= -t² * k / (2R)`. The DDA cell-walk continues against the
**unbent** logical ray, but the sampled cell index uses the bent
position to look up `(cell.x, bent_cell_y, cell.z)`. This decouples
DDA topology from curvature.

(β) **Lazy correction at hit time.** DDA finds a Cartesian "would-be"
hit; before declaring success, re-shoot the bent ray locally to
adjust the hit point by the parabolic offset. Cheaper but less
accurate at oblique angles.

**Recommendation: (α).** It's the literal reading of the architecture
doc. Implementation:

```wgsl
// In march_cartesian's loop, just before the slot check:
let t_now = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
let drop = (t_now * t_now * uniforms.curvature_k) / (2.0 * uniforms.slab_R);
let sample_pos_y = (ray_origin + ray_dir * t_now).y - drop;
let sample_cell_y = i32(floor((sample_pos_y - cur_node_origin.y) / cur_cell_size));
// Use sample_cell_y in place of cell.y for the slot test.
```

**Caveat:** with bent sampling, the DDA's cell-y index must follow the
bent path. This means the OOB pop check (cell.y outside [0, 2]) must
also use `sample_cell_y`. Practically: introduce a derived `cell_eff =
vec3(cell.x, sample_cell_y, cell.z)` and use it for the slot lookup
and the OOB check. The DDA's `s_cell[depth]` keeps the unbent
y-tracker (so cur_side_dist progression is monotonic on Y); cell_eff
is recomputed each iteration.

### 3d — Transition between parabolic and full-spherical modes

For altitudes where the parabolic approximation breaks down (drop
becomes comparable to R itself), switch to the **exact spherical**
sampling. Threshold: `drop > 0.1 * R`, equivalently `t > sqrt(0.2 *
R²/k) ≈ 0.45 * R / sqrt(k)`. Below this t, use parabolic; above, use
the exact arc:

```wgsl
// Exact spherical: ray sample at parametric distance s along ray
// projects onto a sphere of radius R. Drop below the tangent plane is
//   drop_exact = R * (1 - cos(s / R))     for k=1.
// At intermediate k, interpolate the angle:
//   theta = (s / R) * sqrt(k)             (heuristic blend)
//   drop_blended = R * (1 - cos(theta)) * sign_match
```

This is approximate for `k < 1` but matches at k=1 (full sphere) and
k→0 (flat). The threshold check is one branch per DDA iteration,
acceptable.

**Threshold k value for the switch:** when k > 0.7 AND t > 0.5*R, use
the spherical formula; otherwise parabolic. Tunable in Phase 5.

### 3e — Two-ray gameplay picking

Block placement / breaking uses
`src/app/edit_actions/break_place.rs:11-65` → `frame_aware_raycast()`
in `src/app/edit_actions/mod.rs:59`. The ray direction comes from
`self.camera.forward()` (line 35) which is straight (Camera basis is
unrelated to render curvature; see `src/camera.rs:45-62`).

**The CPU raycast is already straight.** Picking sees the simulation
ray, not the rendered ray. Phase 3 needs **no change** to the picking
subsystem — both code paths (CPU pick and shader render) live with
their natural ray, and they're naturally divergent at high altitude.

The reticle / crosshair sits at screen center; at high altitude the
visual under the reticle may not exactly match the picked cell because
the rendered ray was bent. This is acceptable per architecture line 59:
"gameplay actions driven by camera rays — block placement and breaking
by clicking, ranged targeting, screen-space picking — must be evaluated
against the unwarped ray, not the rendered (warped) one."

**However**, one subtlety: the reticle highlight comes from
`do_break`'s frame-aware raycast hitting `aabb::hit_aabb_in_frame_local`
— the highlighted cell is the **picked** cell. At high altitude this
will visibly desync from the rendered crosshair location. If
gameplay-relevance demands aligning visual reticle with pick, run **two
raycasts**: one straight (canonical pick) and one bent (visual reticle).
**Defer to Phase 5** unless playtest reveals it's a problem at the
altitudes the player typically interacts at.

API change if needed:
- Add `WorldState::cpu_raycast_curved(origin, dir, k, R)` mirroring the
  shader's bend.
- `highlight.rs` calls `cpu_raycast_curved` for the highlight; existing
  break/place keep the straight `frame_aware_raycast`.

### 3f — Visual altitude-step harness

`tests/wrapped_planet_visual::edge_on_silhouette_circular_at_orbit`
(written in Phase 0c, ignored until now) gets enabled. It runs at
`altitude = 5 * R`, pitch 0, yaw 0; computes silhouette circularity:
- Detect non-sky pixels.
- Find tight bounding box.
- For pixels on the bbox edge, fit a circle (center + radius) by
  least-squares.
- Assert RMS pixel error < 5% of bbox diameter.

### 3g — Edge-on horizon validation test

Beyond the bbox-circularity test, add
`tests/wrapped_planet_visual::horizon_falls_off_at_altitude` — a sweep
over altitudes `[0.1, 0.5, 1.0, 2.0, 5.0]`, each producing a horizon
profile (y-coordinate of the slab-sky boundary as a function of x).
At low altitude the horizon is a horizontal line; at orbit it's a
near-perfect circle. Assert:
- altitude=0.1: max-min y of horizon < 5% of frame height.
- altitude=5.0: max-min y of horizon > 30% of frame height (curvature
  visible).
- The transition between is monotonic.

### 3h — Files to touch

- `assets/shaders/bindings.wgsl:29-68` — extend Uniforms:
  ```wgsl
  curvature_k: f32,
  slab_R: f32,
  slab_surface_y: f32,
  curvature_axis_y: vec3<f32>,  // local "down" in slab frame, usually (0,-1,0)
  ```
- `assets/shaders/march.wgsl:271-796` — add bending in the inner loop
  (~30 LoC). Specifically: introduce `cell_eff` derived from
  `sample_cell_y`, use it for the slot bit test (line 421-426) and the
  OOB check (line 386). Touch `cur_side_dist` math too — when the bent
  Y exits a cell while the unbent Y hasn't, force a Y-axis pop.
  **Open question:** bent DDA correctness near oblique rays — needs
  thorough testing.
- `src/renderer/buffers.rs:286-311` — populate the new uniform fields.
- `src/app/edit_actions/upload.rs` (or wherever the per-frame upload
  lives) — compute `altitude`, derive `k`, write to uniforms.
- `tests/wrapped_planet_visual.rs` — enable the ignored tests, add the
  horizon-falloff test.

### Risks

- The bent-DDA approach (option α) is novel. The simpler fallback is
  option β (lazy correction at hit time): march straight, then warp the
  hit point by the parabolic offset. Option β preserves DDA correctness
  by construction but visibly distorts cell boundaries (cells appear
  sheared). Decide based on a Phase 3 prototype.
- Curvature interacts with X-wrap: a ray traveling east bends downward;
  on wrap the ray re-enters from the west still bent. The wrap
  preserves direction (architecture line 88) but the parabolic state
  (`t_cumulative`) must persist across the wrap so the bend continues
  monotonically. The wrap branch in `march_cartesian` keeps
  `t_cumulative` unchanged — it's a coordinate translation, not a
  ray-state reset.
- Apple Silicon GPU register pressure: the bent-DDA adds ~2 floats of
  per-thread state. Verify `MAX_STACK_DEPTH = 8` arrays still fit
  (`assets/shaders/bindings.wgsl:267`).

### Estimated commit size

~600 LoC: 200 shader, 100 CPU plumbing + uniform, 300 tests + tuning
infrastructure. One commit (per `feedback_no_intermediate_visual_states`
— don't split a renderer rewrite that spans CPU pack + GPU uniforms +
WGSL).

---

## Phase 4 — Poles

### Banned-cell representation

**Confirmed: sparse occupancy already serves.** The architecture
document (this worktree, line 17) says "the top and bottom rows are
non-buildable — banned, decorative, or hidden". The high-level plan
restates this. Reading
`src/world/gpu/pack.rs:159-168`:

> Compute occupancy from the slab. ... `occ |= 1u32 << i` only when the
> slot is `Some(GpuChild)`.

Empty slots produce zero bits. The shader's tag-0 fast-path
(`assets/shaders/march.wgsl:421-433`) treats them as void. **No new
"banned" primitive is needed for the simulation/render layer** — the
worldgen simply leaves the polar Y-rows as `Child::Empty` in the slab.
Edits at those cells naturally fail because the editing pipeline only
modifies cells reachable through a non-Empty parent slot chain.

### Edits-rejection enforcement

The high-level plan calls for "edits in those regions are rejected".
Two enforcement points:

- **Soft (recommended).** The slab worldgen leaves polar rows Empty;
  ray-cast hits at those Y indices return Empty parents that
  `edit::break_block` / `place_block` already handle by returning
  `false` ("nothing to break"). Done.
- **Hard.** Add a check in `edit::place_block` that rejects placement
  whose target cell's Y index in the slab is `< pole_y_min ||
  >= pole_y_max`. Required if the player can place blocks "into" empty
  pole airspace — without the check they'd build a polar pillar.

Plan: implement the hard check.

```rust
// In src/world/edit.rs, place_block:
if let Some(active_frame_kind) = ... {  // WrappedPlane
    let slab_y = compute_slab_y_index(&hit, &active_frame_kind);
    if slab_y < POLE_BAND_HEIGHT || slab_y >= dims_y - POLE_BAND_HEIGHT {
        return false;  // banned region
    }
}
```

### Pole render impostor

For visual completeness from orbit, draw a **flat polar disk** at the
top and bottom of the slab. Two implementation options:

(A) **Geometry impostor.** Add a polar disk mesh in `entity_raster`. The
disk lives at `y = slab_surface_y`, radius = slab circumference / 2π
(matching the implied planet R). At low altitude the disk is below
the player's feet and behind the cliff edge of the slab; at orbit it
fills the visual gap.

(B) **Shader-side analytic.** In the marcher's tag=0 fast-path, when
the ray exits the slab Y bounds AND `curvature_k > 0.5`, draw a
colored impostor based on the ray's angle to the slab Y-axis. Cheap,
no extra geometry, but limited to a flat color.

**Recommendation: (B)** for v1. The polar visual treatment is
architectural-decoration only; a flat ice color is enough to read as
"polar cap" from orbit.

Concrete shader hook in `assets/shaders/main.wgsl::shade_pixel`
(line 56-137): on a miss, before the sky color, check if the unbent
ray's Y-tangent points into the slab Y exclusion zone — if yes, splat
the ice color (e.g. RGB 0.9, 0.95, 1.0) modulated by altitude-fall-off.

### Validation

- `tests/wrapped_planet_visual::orbit_view_shows_polar_caps` — high
  altitude, look at slab pole; expect non-sky region at the top of the
  visible disk.
- `tests/wrapped_planet_visual::ground_view_at_high_latitude` — camera
  inside slab near the polar boundary, looking outward toward the pole;
  expect the impostor to be visually consistent.
- `tests/edit_pole_rejection.rs` — try to place a block in a polar Y
  index, assert `place_block` returns false.

### Risks

- The impostor is purely cosmetic; getting it visually consistent across
  altitude transitions is a tuning task (Phase 5).
- "Soft" pole rejection has the gotcha that the worldgen MUST
  consistently leave polar rows empty — a one-off slab-builder bug
  that fills them would let players build there. Add a worldgen
  invariant assertion.

### Estimated commit size

~250 LoC: 80 shader (impostor), 100 worldgen + edit-rejection,
70 tests.

---

## Phase 5 — Polish

### 5a — `k(altitude)` tuning

Stepwise visual harness: take the altitude-step screenshots from Phase
3g, render them at a range of `p` exponent values (`[1.2, 1.5, 1.8,
2.0]`), pick the curve with the smoothest transition by eye plus
machine score (silhouette circularity + horizon-falloff smoothness).
Files: tunable in `bindings.wgsl::Uniforms.curvature_k_p` if we expose
it; otherwise hard-coded `p = 1.5` after tuning.

### 5b — Deep-depth render on the slab

`tests/wrapped_planet_deep_zoom.rs` — zoom into a single voxel on the
slab surface (anchor depth 30+, comparable to existing deep-plain
tests). Verify:
- `cargo test --release wrapped_planet_visual::deep_zoom_on_slab`.
- `frame_raycast_hit t > 0` at depths 25, 30, 35 (single voxel, deep
  anchor inside slab).
- No precision artifacts (mirror `tests/render_visibility.rs:31-67`'s
  pattern).

### 5c — Entities on the slab

Spawn entities; they live in normal Cartesian flow inside the slab.
Wrap should apply to entities crossing the X-boundary. `entities.rs`'s
`tick` (line 142-158) calls `add_local`, which (via Phase 2's
WrappedPlane-aware step) wraps automatically. Test: spawn an entity at
the east edge with eastward velocity, advance a tick, assert entity's
anchor crossed to the west.

### 5d — Edits / placement / pole rejection

Comprehensive end-to-end: break, place, save mesh, place mesh on the
slab. All must succeed in non-pole regions and be rejected in pole
regions. Use the existing `e2e_layer_descent.rs` harness pattern.

### Estimated commit size

~150 LoC tuning + 200 LoC tests.

---

## Cross-cutting validation gates (every phase)

- `cargo build --release` from the worktree.
- `cargo test --release` from the worktree.
- New visual screenshots committed to `tmp/wrapped_planet_visual/`
  inside the worktree (NEVER `/tmp` per memory rule
  `feedback_tmp_in_worktree`).
- Coordinator review pass (per the high-level plan's gates section).
- For ANY rendering/geometry change: per-step screenshot comparison to
  the prior step's baseline (memory rule
  `feedback_screenshot_along_the_way`).

## Risks the high-level plan misses

1. **Bent DDA correctness.** The architecture's "subtract `t² · k / 2R`"
   prescription is unambiguous for *sample position*, but ambiguous for
   the DDA's cell-traversal logic. Phase 3's recommendation (bend
   sample-y, not DDA-y) needs a prototype before committing. Fallback
   (option β: warp at hit time) is uglier but bounded-risk.
2. **Wrap + ribbon-pop interaction.** When the camera is well above the
   slab and a ray exits the slab in X, it should NOT wrap — it should
   pop UP via the ribbon to the surrounding empty 27³, just like Y/Z
   exits. The wrap branch must be tightly gated on
   `current_kind == ROOT_KIND_WRAPPED_PLANE && depth == 0`.
3. **WorldPos library threading for wrap.** `step_neighbor_cartesian`
   currently takes no library. Wrap awareness needs library access at
   step time — a few public-API call sites need the library passed
   through. `add_local` already plumbs it, but other call sites
   (anywhere in `edit.rs`, the harness, etc.) may not. Audit before
   Phase 2 lands.
4. **Pole impostor + curvature interaction.** The pole impostor in (B)
   uses `curvature_k > 0.5` as the "from orbit" gate. At intermediate
   altitudes the impostor pops in/out. Phase 5a tuning includes the
   impostor blend.
5. **Reticle-pick desync at altitude.** Phase 3's recommendation is to
   keep CPU pick straight and shader render bent; the reticle visual
   under the player's crosshair won't always match the picked cell at
   high altitude. May surface in playtest.

## Anything I couldn't ground in the code

- The exact slab dimensions (`dims = [20, 10, 2]`) and embedding depth
  (22) are placeholder choices from the high-level plan; they need to
  be revisited after Phase 1 once visual prototypes show how big the
  slab "feels" at typical camera altitudes. The architecture doc
  (line 81) explicitly leaves these choices open.
- `face_walk.wgsl` mentioned for deletion — I located it via
  `grep` but didn't read it. Verify before deletion that it's not
  reused outside the sphere path.
- The `--wrapped-planet` CLI plumbing — the harness already supports
  per-preset args (`--plain-world`, `--sphere-world`); the addition is
  mechanical and matches existing patterns. I didn't read the CLI
  parser to confirm the exact dispatch site.
- The TAA path interactions with curvature: TAA reprojects via
  `(camera.pos, ray_dir, t)` (`assets/shaders/main.wgsl:267-282`). With
  bent rays, the historical ray for a pixel is not the same as the
  current ray for the same pixel — TAA may smear at altitude
  transitions. Mitigation: clamp TAA history weight to 0 when
  `curvature_k` changes by more than ~0.05 between frames. Treat as
  a Phase 5b polish item.
