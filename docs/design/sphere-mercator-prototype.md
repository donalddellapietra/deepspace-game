# Sphere-Mercator Prototype (Branch `sphere-mercator-1-2-1`)

> Status: design + reference math only. No renderer wiring.
> Existing `TangentBlock` / sphere DDA paths are untouched.

## One-line summary

The world is a **flat wrapped Cartesian slab**. The "sphere" is a render-time
camera lens: a per-pixel transform that converts a screen-space ray into a
slab-space ray, and a per-hit transform that rotates the slab-space normal back
into the apparent UV-sphere tangent frame. **Nothing in the world is curved.**

## Why the previous hybrid failed

`sphere-mercator-1-2` runs a sphere DDA at the top of the descent. Even with
`TangentBlock` escapes into Cartesian for the deep tail, the sphere DDA itself
threads `(lon, lat, r)` state through the walker. Those coordinates lose
precision well before the deepest cells are reached — the residual jitter at
high depth comes from the spherical step, not the cells themselves.

Pure Cartesian DDA on the wrapped plane has no such failure mode: it inherits
the existing anchor/offset precision discipline that already renders correctly
at any depth.

## Architecture

```
                  screen pixel
                       │
                       ▼
              ┌────────────────────┐
              │  PlanetLens.project │   sphere intersection (analytic)
              └────────────────────┘   (one sphere primitive, far from origin)
                       │
            (slab_origin, slab_dir)   in flat wrapped-plane coords
                       │
                       ▼
              ┌────────────────────┐
              │  cartesian DDA     │   precision-stable at any depth
              │  on wrapped slab   │   (existing march_cartesian)
              └────────────────────┘
                       │
            (cell_min, slab_normal, t)
                       │
                       ▼
              ┌────────────────────┐
              │  PlanetLens.shade  │   rotate normal by per-hit tangent frame
              └────────────────────┘   (TangentFrame::at, computed locally)
                       │
                       ▼
                   shaded pixel
```

The walker, edits, collision, and physics never see anything but the flat
wrapped-Cartesian slab. The sphere only exists in two render-only places:

1. **The camera lens** (one analytic sphere intersection per pixel).
2. **Per-hit normal rotation** (one tangent-frame build per visible cell).

Both are O(1), bounded, and operate on small numbers — no precision blow-up.

## Coordinate systems

- **World** (camera/sphere space): standard `(x, y, z)`. Planet center `C`,
  surface radius `R`. Sphere lives in this space; **no voxel data does**.
- **Slab** (wrapped Cartesian): `(slab_x, slab_y, slab_z)` in cell units, where
  - `slab_x` ∈ `[0, W)` wraps; `W = dims[0]` cells wide
  - `slab_y` ∈ `[0, H)` is radial depth (Y=0 = bottom of slab, Y=H = top)
  - `slab_z` ∈ `[0, L)` is latitude, no wrap; ends are pole strips
- **Bridge**: longitude `θ = 2π · slab_x / W`, latitude `φ = π · (slab_z / L − ½)`,
  altitude `a = (slab_y − H_surf) · cell_size` (where `H_surf` is the row whose
  top is the surface).

The bridge is one direction only when projecting in (world → slab) and the
other when shading (slab → world). At no point does the WORLD store
`(θ, φ, a)`.

## The lens math

### Project (world ray → slab ray)

```
project(camera_world, ray_world, lens) -> Option<(slab_origin, slab_dir)>
```

1. Solve sphere intersection: closest point of `(camera + t·ray)` to the
   conceptual sphere of radius `R` at `C`. Pick the entry hit if the camera
   is outside; otherwise the camera's projection.
2. From that anchor point `P_w`, read `(θ_a, φ_a, a_a)` via the bridge.
3. Build the local tangent frame at `(θ_a, φ_a)` (existing
   `TangentFrame::at(C, R, φ_a, θ_a)`).
4. Decompose `ray_world` into `(east, normal, north)` components in that
   frame. Those components ARE `(slab_dx, slab_dy, slab_dz)` after scaling
   by the slab's cell-per-radian and cell-per-meter rates.
5. Anchor the slab origin at `P_w`'s slab coords: `(slab_x_a, slab_y_a, slab_z_a)`.

The transformation is **linear in a small neighborhood of the anchor** — which
is fine, because the cartesian DDA only marches a few cells before the
component frames are recomputed (or just for the visible patch — see "scope"
below).

### Shade (slab hit → world normal)

```
shade(cell_min_slab, slab_normal, lens) -> world_normal
```

1. From `cell_min_slab` derive `(θ, φ)` for the cell.
2. Build `TangentFrame::at(C, R, φ, θ)` — local→world rotation.
3. `slab_normal` is axis-aligned in slab space (`±X` = ±east, `±Y` = ±radial,
   `±Z` = ±north). Multiply by the frame.

This is the **per-block rotation** the user asked for: each visible block's
local `(±X, ±Y, ±Z)` faces are rotated by its own tangent frame, so the block
appears to sit on the sphere even though its data is on the flat slab.

## Why this is precision-stable at deep depth

Every quantity that has to be exact at depth `N` (cell size `1/3^N`) lives in
**slab space**, where it is represented exactly by the existing `WorldPos`
anchor + offset.

The **only** quantities in world (sphere) space are:
- The camera position relative to its own anchor (small f32, ≤ a few cells).
- The lens anchor `P_w` — derived per-pixel from one sphere intersection.
- The tangent frame at the anchor — built from `(sin/cos)(θ_a)` /
  `(sin/cos)(φ_a)` of bounded inputs (`|θ| ≤ π`, `|φ| ≤ π/2`).

None of these scale with `1/3^N`. The depth-dependent quantities never leave
slab space.

The shading rotation also uses bounded inputs: `(θ, φ)` of the visible cell.
A cell at depth 25 has `θ` precise to `2π / (W · 3^25)` — well within f32
when expressed as a single `sin/cos` of a small relative angle from the lens
anchor (see the offset-angle trick under "Open questions" → "frame banding").

## Scope of the prototype

This commit lands:

- `src/world/sphere/uv_lens.rs` — `PlanetLens`, `project_ray`, `shade_normal`,
  `slab_to_world`, `world_to_slab`. CPU only. Pure math + unit tests.
- This design doc.
- Additive only: no existing file changed besides `src/world/sphere/mod.rs`
  (one `pub mod` line). `TangentBlock`, sphere DDA, `cs_raycast` untouched.

This commit does NOT:

- Wire the lens into `march()` or `shade_pixel()`.
- Touch the GPU pack or shader.
- Remove the existing TangentBlock / sphere-DDA paths.

Those are the next branch: `sphere-mercator-1-2-2` will replace
`shade_pixel`'s ray construction with `lens.project_ray()`, run the
**unmodified** `march_cartesian` on the resulting slab ray, and rotate the
returned normal via `lens.shade_normal()` before lighting.

## Open questions (deferred)

- **Frame banding**: a single tangent frame for the whole slab ray is fine
  for the patch the camera sees, but for grazing horizon rays the patch can
  span enough longitude that the linearization error becomes visible (the
  back of the patch is in a frame the front doesn't share). Two responses:
  (a) accept it for the prototype; (b) refine the anchor along the ray
  every K slab cells. (b) is straightforward — the slab ray re-anchors and
  shading still works because `shade_normal` reads the cell's own `(θ, φ)`.
- **Pole strips**: the lens degenerates at `|φ| → π/2`. The slab already
  bans pole strips by worldgen — the lens just needs to clip rays whose
  sphere-intersection latitude lies outside `[-π/2 + ε, π/2 − ε]`.
- **Edits / picking**: the inverse path. A click in screen space goes
  through the lens to a slab cell. The lens already provides `world_to_slab`;
  highlight rendering needs the same `shade` rotation as terrain.

## Testing

Unit tests in `src/world/sphere/uv_lens.rs`:

- Round trip: `world → slab → world` recovers the input within 1e-5 of `R`.
- Wrap: `θ = π` and `θ = −π` map to the same world point.
- Tangent-frame orthogonality at every test sample (reuses
  `TangentFrame`'s invariants).
- **Precision at depth 25**: a cell sized `1/3^25` of the slab still
  produces an orthonormal tangent frame, normal error < 1e-5.
- Lens projection at planet north pole, looking down → slab ray is
  `(slab_x_pole, top_of_slab, slab_z_pole)` going `−Y`.

Visual harness wiring is deferred to the wiring branch.

## How to validate the prototype

```bash
cd .claude/worktrees/sphere-mercator-1-2-1
cargo test --release world::sphere::uv_lens
```

All tests must pass. Shader / harness validation is the next branch's job.
