# Sphere raycast at arbitrary depth — problem and proposed approach

Status: proposal, not implemented. Architectural sketch for
discussion — details may need revision during implementation.

## What we're trying to do

The engine is an infinite-zoom voxel world. Users zoom into any
cell recursively — supported depth is at least 60 levels. At every
depth, the render frame is the cell the camera is inside, expressed
in its own local `[0, 3)³` coordinate system; cells at the
render-frame level are O(1) in local coords and get drawn at
full screen resolution.

On the sphere-world preset, the world contains a cubed-sphere: a
cube whose 6 faces are warped outward into a round shape via
`ea_to_cube(u) = tan(u · π/4)`. Each face has a 27-child subtree
that represents the face's voxels in `(u, v, r)` coordinates. We
want:

1. **Three-way agreement** between (a) the GPU shader that renders
   a pixel, (b) the CPU raycast that picks cells for break/place/
   highlight, and (c) the highlight AABB outline. All three must
   resolve to the same cell for the same ray.

2. **Arbitrary zoom depth** — the user can descend to 60+ layers
   and still edit, render, and highlight cells at that level.

3. **No approximation artifacts** at any zoom level — the sphere
   should look spherical, the grid should look aligned, transitions
   between zoom levels should be visually seamless.

All of this is straightforward for the Cartesian portions of the
world. Cartesian nodes cleanly support infinite zoom. The sphere
has so far not matched that bar.

## What's known about the difficulty

### f32 precision and the warp

The warp `ea_to_cube` is nonlinear: to compute a cell's boundary
plane in body-XYZ, we evaluate `tan(u_ea · π/4)` at the cell's
`u_lo` and `u_hi`. Cells are partitioned uniformly in face-
normalized coords (u ∈ [0, 1]), so a cell at depth N has size
`1/3^N` in face-normalized coords. At depth 30, adjacent cell
edges in face coords differ by ~1e-14 — below f32 precision (eps
≈ 1e-7 at magnitude 1). Naïvely computing `ea_to_cube(u_hi) -
ea_to_cube(u_lo)` loses the delta, and the cell's two bounding
planes become numerically identical.

This precision wall is not a statement about depth per se — it's
a statement about arithmetic done with absolute face-normalized
coords at deep levels. Cartesian doesn't hit this wall because it
rescales the ray at every level (ribbon-pop), so all math stays in
O(1) magnitudes regardless of absolute depth.

The sphere's walker, as currently written, descends from the
face-root through all face-subtree levels in a single monolithic
loop and reports cell bounds in face-root-normalized coords.
That's where the precision issue enters. This is an architectural
mismatch with Cartesian, not an inherent limit of the sphere
geometry.

### The warp applies at every layer, not just the face root

An earlier draft of the fix assumed the face subtree nests
affinely in body-XYZ once we're past the face root — i.e., that
"the warp is applied once, at the root, and everything below is
Cartesian." That is not correct in general. The warp is a
non-linear function; its derivative varies across the face. Each
cell's body-XYZ shape depends on *where* on the face it sits,
not just how deep in the face subtree.

However, the warp's nonlinearity *within a small cell* is bounded
by the second-order term of its Taylor expansion, which decays
as O(size²). At deep levels that error is beneath f32 eps and
beneath any perceptual threshold. At shallow levels it matters,
but at shallow levels we already have an exact face-root march.

## Proposed approach

### The frame descent

`compute_render_frame` descends through the face subtree the same
way it descends through Cartesian subtrees — slot by slot, using
the anchor's base-3 path. At each descent the frame accumulates:

- The body-XYZ origin of the cell's corner.
- The body-XYZ basis vectors for the cell's `u`, `v`, `r` axes —
  evaluated from the warp's derivative at the cell's corner.
- The cell's face-normalized `(u_lo, v_lo, r_lo, size)` — kept as
  a parallel exact representation alongside the body-XYZ basis.

The render frame at depth N is a specific face-subtree cell at
depth N. Ray + cell math happens in this frame's local coords.

### Per-level warp handling

At each descent, plane normals for the cell's 6 bounding faces are
stored as a `(base, delta_per_local_unit)` pair:

- `n_base` = plane normal at the cell's corner, computed from the
  full warp (`u_axis − ea_to_cube(u_corner_ea) · n_axis`).
- `n_delta` = how that normal changes per unit of local-u
  traversal, computed from the warp's derivative
  (`−(π/4) · sec²(u_corner_ea · π/4) · (size_ea / 3) · n_axis`).

Within the frame, a ray-plane intersection at local_u = K is
`t = −(n_base + K · n_delta) · ray_origin / (n_base + K · n_delta) · ray_dir`,
which expands to a numerically well-conditioned expression in
`(A, B, a, b) = (n_base · ro, n_base · rd, n_delta · ro, n_delta · rd)`,
all of which are O(1) regardless of absolute depth.

This captures the warp's first-order behavior exactly at every
level, preserves f32 precision at arbitrary depth, and reduces to
the ordinary Cartesian ray-plane intersection when the delta
collapses to zero.

### Transition between zoom levels

When the render frame descends from depth N to N+1, the new frame
computes its own `n_base, n_delta` at its own corner. The shared
boundary between N and N+1 agrees to first order in the warp.
The second-order residual is O(size²) — imperceptible past depth 3
and mathematically beneath f32 precision past depth ~10.

At the face root (depth 1, cell is the whole face), second-order
is not negligible — this is where the sphere's macro curvature
lives. That level uses the existing exact sphere-in-cell march
(numerical integration over the curved shell, not linearized).
Deeper levels use the ribbon-pop linearized march.

The transition between "face-root exact march" and "descendant
linearized march" happens at a depth where the two agree to
within f32 precision — likely depth 2 or 3. Past that, the
linearized form is mathematically equivalent to the exact form in
the precision we can represent.

### What this shares with Cartesian

The architectural skeleton — ribbon-pop per level, render frame
descends to anchor depth, local math stays O(1) — is identical.
The sphere-specific piece is the `(n_base, n_delta)` pair carried
per frame, which factors in the warp's derivative at that level.
In Cartesian `n_delta` is trivially zero; the sphere walker
becomes Cartesian at extreme depth as `n_delta` decays to below
f32 eps.

## What we don't know yet

- **Exact implementation cost.** The shader side needs the same
  `(n_base, n_delta)` scheme. WGSL is f32 only (no native f64),
  so the CPU and shader use the same decomposition — good for
  three-way agreement, but requires both to implement it.

- **Visual boundary between exact face-root march and linearized
  descendant march.** In principle the boundary is at a depth
  where second-order warp is below the pixel threshold. Whether
  that's depth 2, 3, or 5 depends on camera distance; we'd want
  to set this empirically with a test that renders both modes on
  either side of the boundary and checks pixel agreement.

- **Rendering quality of the linearized form at the face root's
  immediate sub-cells.** The first-order approximation at depth 2
  might produce a visible "faceting" of the sphere if the
  transition is mishandled. A visual regression test (sphere
  silhouette curvature) would catch this.

- **Interaction with anchor paths that cross a body boundary.**
  The anchor path descends from world_root through Cartesian
  ancestors into the body, then into a face subtree. The frame
  computation needs to handle this transition — specifically, it
  needs the body's radii metadata at the point where it starts
  descending through `CubedSphereFace` nodes.

- **Slot-index semantics inside the face subtree.** Today
  `CubedSphereFace` nodes have 27 children in `(u_slot, v_slot,
  r_slot)` order. For anchor-path descent to work via
  `from_frame_local`, either (a) the anchor path through a face
  subtree uses UVR slot semantics (affecting the `anchor.rs`
  coord module), or (b) we compute the anchor's face-subtree
  slots separately from its face-root entry. Option (b) is
  probably cleaner but hasn't been prototyped.

## Why this is a proposal and not a spec

I've cycled through several wrong diagnoses of the underlying
issue during development (see `docs/history/sphere-precision-
confusion.md`). Before implementing, I want to confirm:

- The per-level `(n_base, n_delta)` scheme actually delivers the
  precision it claims — this should be verified with a numeric
  test that intersects a ray with a sphere face at depth 30+ and
  checks it lands within f32 precision of the exact answer.

- The face-subtree descent in `compute_render_frame` maps cleanly
  onto the existing Cartesian path-walk infrastructure, or
  whether it requires a parallel path that knows about sphere
  coords.

- The shader side can mirror the scheme without WGSL features we
  don't have access to.

If any of those don't pan out, the architecture may need to shift.
This doc is the current best understanding, not a final design.
