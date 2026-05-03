# SphericalWrappedPlane — UV sphere via stored per-cell tangent bases

A node kind that takes the existing `WrappedPlane` storage layout
(flat 2D slab with X-wrap) and renders it as a UV sphere by
attaching a precomputed tangent basis to every `(lon_idx, lat_idx)`
cell at world-build time. No per-fragment angle math; cell-local
arithmetic everywhere; one cell-boundary transform rule applied
symmetrically across shader, CPU raycast, anchor descent.

## Why not the existing approaches

- **`sphere-mercator-1`**: rotations *recomputed every fragment* from
  `(lon_c, lat_c)` via `sin`/`cos` on absolute angles. Same anti-pattern
  as BROKEN STATE 2 from the TB work (world-absolute math, no stored
  transform). Hits an f32 precision wall at deep zoom.
- **Cubed sphere via dodecahedron-style TBs**: 26 distinct rotations,
  but the *layout* is a flat 3³ grid — the cells don't lie on a sphere.
- **Per-cell TB on a flat WP slab**: rotates each cell's interior but
  the cells themselves stay on a flat 2D grid in world space. Looks
  like a flat slab of tilted cubes, not a sphere.

## Design rules — what TB taught us

1. **Store the transform.** Per-cell tangent basis computed once at
   world build, stored on the cell. Never recomputed in the hot path.
2. **One boundary rule, applied symmetrically.** Like
   `TbBoundary::{enter,exit}_point` — fires in shader, CPU raycast,
   anchor descent, all reading the same stored data with the same math.
3. **Cell-local arithmetic.** Inside a cell, magnitudes bounded by
   cell size. No world-absolute coords.
4. **Same code path on every consumer.** No render-time
   reinterpretation that diverges from edit-time math.

## Tree shape

```
NodeKind::SphericalWrappedPlane {
    dims: [u32; 3],         // [N_lng, N_r, N_lat] same as WrappedPlane
    slab_depth: u8,         // dims[0] == 3^slab_depth (lon fully wraps)
    body_radius_cells: f32, // sphere radius in WP-local cell units;
                            // typical = dims[0] / (2π) for arc-length
                            // -consistent cell sizes
}
```

Children: same flat layout as `WrappedPlane`. Each non-empty cell
slot at `(lon_idx, r_idx, lat_idx)` is a `Child::Node` pointing to a
`NodeKind::TangentBlock { rotation: R(lon_c, lat_c) }`. The rotation
is the **tangent basis at the cell centre**, mapping cell storage
axes `(local_x, local_y, local_z)` to WP-local axes `(lon_tangent,
radial, lat_tangent)` at that cell's position.

The TB's content (uniform stone / dirt / grass subtree) is shared by
NodeId across all cells with the same content — material dedup
survives the rotation diversity. Library entries: 1 per unique
rotation per material = ~378 × 3 = ~1100 nodes for default dims.

## Cell rotation — the one nontrivial bit

Each cell's rotation `R(lon_c, lat_c)` maps storage frame to a tangent
frame at the cell centre:

```
storage_x  →  east (longitude tangent)
storage_y  →  up (radial outward)
storage_z  →  north (latitude tangent)
```

Constructed from two axis rotations composed:

```rust
let r_lon = rotation_y(lon_c);              // about world Y (polar axis)
let r_lat = rotation_x_local(-lat_c);       // about local X (rotated east)
let r = matmul(&r_lon, &r_lat);             // column-major composition
```

The rotation chain is *exact* (TB algebra composes without drift —
same property the dodecahedron preset relies on). 12 distinct face
normals in dodecahedron → 378 distinct cell rotations here, identical
storage / dispatch.

## Cell-entry transform

When a ray crosses from WP-local frame into cell-`(lon_i, r_i, lat_i)`
storage frame:

```
cell_origin_world  = sphere_position(lon_c, lat_c, r_c)   // precomputed
local_in_storage   = R^T · (p_wp - cell_origin_world) · (3 / cell_size)
                                                    + 1.5 · (something)
```

The rule mirrors `TbBoundary::enter_point` exactly, with two extras:
1. **Translation**: subtract `cell_origin_world` (precomputed sphere
   position) before the rotation. TB has no translation because TB
   cells live at slot-centred parent-frame positions.
2. **Cell-size rescale**: same `3 / cell_size` factor every WP-cell
   entry already does.

The TB inscribed-cube shrink (`tb_scale`) is **not** needed — adjacent
cells share an edge in tangent space, so there's no rotation-extent
overflow. `tb_scale = 1.0`.

Exit transform (storage → WP-local) is the inverse, same as
`TbBoundary::exit_point`.

## Where the per-cell precompute lives

Two layers:

**Per-cell `TangentBlock`** (stores `R`): existing `NodeKind::TangentBlock
{ rotation }`. Already plumbed through `GpuNodeKind`'s
`rot_col0/rot_col1/rot_col2`, packed by `pack_node_kind`. No changes.

**Per-cell sphere position** (stores `cell_origin_world`): NEW. Either
- (a) Stored on the parent `SphericalWrappedPlane` node as a flat
  `Vec<[f32; 3]>` indexed by `lat_idx * dims[0] + lon_idx`, uploaded
  to a new GPU buffer (think: like the ribbon, side-buffer indexed
  by parent + slot). Size: 27 × 14 × 12 bytes = 4.5 KB.
- (b) Stored *inline* on the cell's `GpuNodeKind` — extend
  `GpuNodeKind` with `cell_world_pos: vec4<f32>` (x/y/z + spare).
  Costs 16 bytes per library entry, but cells already have unique
  rotations so they don't dedup; no waste.

I'd go with (b) — keeps the cell-entry math purely "read what's on the
node, apply unified rule", same shape as `TbBoundary`. (a) introduces
a side buffer dispatch, more like the wrapped_planet wrap path.

## Build-time precompute

```rust
fn spherical_wrapped_plane_world(
    dims: [u32; 3],
    slab_depth: u8,
    body_radius_cells: f32,
    lat_max: f32,
    cell_subtree_depth: u8,
) -> WorldState {
    let mut library = NodeLibrary::default();
    let n_lng = dims[0];
    let n_r   = dims[1];
    let n_lat = dims[2];
    let cell_size_wp = 3.0 / (3.0_f32.powi(slab_depth as i32));

    let mut wp_children = empty_children_2d(dims);   // flat slab layout

    for lon_idx in 0..n_lng {
        let lon_c = (lon_idx as f32 + 0.5) / n_lng as f32 * std::f32::consts::TAU;
        for lat_idx in 0..n_lat {
            let lat_norm = (lat_idx as f32 + 0.5) / n_lat as f32 * 2.0 - 1.0;
            let lat_c = lat_norm * lat_max;
            let rotation = matmul(&rotation_y(lon_c), &rotation_x(-lat_c));

            for r_idx in 0..n_r {
                let r_c = body_radius_cells + (r_idx as f32 + 0.5) * cell_size_wp;
                let cell_world = [
                    r_c * lat_c.cos() * lon_c.cos(),
                    r_c * lat_c.sin(),
                    r_c * lat_c.cos() * lon_c.sin(),
                ];
                let material = material_for(r_idx);  // grass / dirt / stone

                // Inner content: uniform-material subtree of `cell_subtree_depth`.
                let content = build_uniform_cartesian_subtree(
                    &mut library, material, cell_subtree_depth,
                );
                // Wrap in TB carrying both rotation and (NEW) world position:
                let cell = library.insert_with_kind(
                    uniform_children(content),
                    NodeKind::TangentBlock { rotation },
                    cell_world,                      // <-- NEW field
                );
                wp_children[slot_idx_for(lon_idx, r_idx, lat_idx)] =
                    Child::Node(cell);
            }
        }
    }

    let wp = library.insert_with_kind(
        wp_children,
        NodeKind::SphericalWrappedPlane {
            dims, slab_depth, body_radius_cells,
        },
    );
    WorldState { root: wp, library }
}
```

(`cell_world` is the new "translation" field; for plain `TangentBlock`
it stays `[0; 3]` and contributes nothing.)

## Shader sketch

`march.wgsl` already has TB child dispatch around line 760. Extend
its entry transform to include translation when the parent is a
`SphericalWrappedPlane`:

```wgsl
if node_kinds[child_idx].kind == NODE_KIND_TANGENT_BLOCK {
    // Old WP-cell entry: (p - slot_center_in_wp) * (3 / cell_size)
    let local_pre_origin = (ray_origin - cell_origin_in_wp) * scale;

    // NEW: if we're a SphericalWP child, the cell's *world* position
    // overrides slot_center_in_wp. Read from the cell's GpuNodeKind:
    let cell_world_pos = node_kinds[child_idx].cell_world_pos.xyz;
    let local_pre_origin = (ray_origin - cell_world_pos) * scale_wp;

    // Then existing TB enter_point rule:
    let local_origin = tb_enter_point(child_idx, local_pre_origin, 1.5);
    let local_dir    = tb_enter_dir(child_idx, local_pre_dir);

    // Recurse into cell's [0,3)³ via existing march_in_tangent_cube.
}
```

`cell_world_pos` defaults to `[0; 3]` for plain TBs, in which case
the entry math reduces to the existing TB rule (subtracting zero is
identity). Same code path; SphericalWP is just "TB with non-zero
translation".

## CPU raycast / anchor descent

Mirror the shader: extend `cpu_raycast/cartesian.rs`'s TB child
dispatch to subtract `cell_world_pos` before applying `enter_point`.
Anchor `pop_one_level_rot_aware` / `descend_one_level_rot_aware`
similarly add/subtract translation.

The unified rule: every system applies
`R^T · (p - cell_world_pos - pivot) / tb_scale + pivot` on cell entry,
inverse on exit. Translation is one extra `vec3` subtraction at one
boundary — no other code changes.

## Wrap

`SphericalWrappedPlane` reuses `WrappedPlane`'s X-wrap mechanism: when
`lon_idx` overflows from `dims[0] - 1` to 0, the path stays inside the
SphericalWP subtree. The cell at `lon_idx = 0` has the right rotation
(`R_y(0)` modulo lat) and world position, so the ray enters seamlessly.

## What stays unchanged

- `WrappedPlane` itself (flat slab) — keep existing preset for
  Cartesian-wrap tests.
- `TangentBlock` algebra — `TbBoundary::{enter,exit}` extended with
  optional translation, default zero, no behavior change for current
  callers.
- Shader's TB child dispatch + ribbon pop — same code, one extra
  `vec3` subtraction read from `GpuNodeKind`.
- CPU raycast TB sites — same.
- Anchor descent — same.

## What's new

- `NodeKind::SphericalWrappedPlane` variant.
- `GpuNodeKind`'s `cell_world_pos` field (12 bytes per node, was 4
  bytes of padding).
- `NodeKind::TangentBlock` extended to carry an optional
  `cell_world_pos` (default zero); the variant becomes
  `TangentBlock { rotation, cell_world_pos }`. Dedup hash extended
  to include the world-pos bits.
- `TbBoundary` extended with translation: `enter_point(p) = R^T · (p
  - translation - pivot) / tb_scale + pivot`. Existing callers pass
  `translation = [0; 0; 0]`.
- Bootstrap `spherical_wrapped_planet_world` (the example above).
- CLI flag `--spherical-wrapped-planet`.

## Open questions

- **Cell-edge stitching**: tangent patches don't perfectly cover a
  curved sphere — there are tiny gaps/overlaps at cell boundaries
  (~ `cell_arc² / r`). For default dims ([27, 2, 14]) those are
  ~ 0.5% of cell size. Probably invisible. Stress with
  small `body_radius_cells`.
- **Pole gap**: lat-cap of ±72° leaves a ~36° polar disc empty per
  pole. Same as `sphere-mercator-1`. User said don't care.
- **GPU cost of `cell_world_pos` in every `GpuNodeKind`**: 16 bytes
  × library size. For default ~1100 entries that's 17 KB extra.
  Trivial.
- **f64 build-time precompute?** The sphere positions are computed
  with `f32` `cos`/`sin`. For default dims with subtree depth K, the
  smallest meaningful sphere displacement is `cell_size / 3^K = 3 /
  3^(slab_depth + K)`. For slab_depth=3, K=22 that's ~3·10⁻¹¹, near
  f32 precision floor. Build time once: use f64, downcast at storage.

## Implementation order

1. Extend `NodeKind::TangentBlock` with `cell_world_pos: [f32; 3]`,
   default zero. Dedup hash. Add to `GpuNodeKind`. Lib tests pass —
   no behavior change with default zero.
2. Extend `TbBoundary` with translation. Existing call sites pass
   zero. Lib tests pass.
3. Plumb `cell_world_pos` through shader's TB child dispatch + ribbon
   pop. Existing TB-using presets render identically (translation = 0).
4. Add `NodeKind::SphericalWrappedPlane` variant + dedup.
5. Build the bootstrap. Visual smoke test.
6. CPU raycast translation.
7. Anchor descent translation (relevant if camera enters a cell).
8. Wrap test (camera moves around the sphere).
