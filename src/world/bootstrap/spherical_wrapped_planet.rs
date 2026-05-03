//! Spherical wrapped-planet bootstrap — proof-of-concept UV sphere
//! built entirely on top of the existing `WrappedPlane` + per-cell
//! `TangentBlock` infrastructure.
//!
//! Construction: identical to `wrapped_planet` (a `[27, 2, 14]` slab
//! at `slab_depth=3` with grass/dirt/stone material rows) except each
//! slab cell carries:
//!
//! 1. A **rotation** `R(lon_c, lat_c) = R_y(lon_c) · R_x(-lat_c)`
//!    aligning storage `+Y` to the sphere's radial direction at the
//!    cell centre (same algebra as `dodecahedron_test`).
//! 2. A **`cell_offset`** = `(sphere_pos − natural_slot_centre) / cell_size`
//!    in direct-parent slot units, repositioning the cell from its
//!    natural slab-grid slot onto the sphere surface.
//!
//! The renderer's existing TB child dispatch + ribbon pop reads both
//! pieces from the cell's `GpuNodeKind` and applies the unified
//! transform — same code path as ordinary TBs, just with non-zero
//! offsets.
//!
//! **Open issue (rendering correctness)**: the shader's slot DDA
//! still walks the flat slab grid. Cells whose `cell_offset` is
//! large (typical here — offsets of ~5+ slot units) sit far from
//! their natural DDA slot, so the DDA misses them on rays whose
//! actual geometric path doesn't intersect the natural slot. Result:
//! gaps and wrong-cell hits. A sphere-DDA dispatch path is needed
//! for correct rendering. This preset exists to **visualise** what
//! the cell-offset plumbing produces, not to render correctly yet.

use super::WorldBootstrap;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, uniform_children, Child, NodeKind, NodeLibrary, BRANCH, MAX_DEPTH,
};

/// Same dims as wrapped_planet for direct comparison.
pub const DEFAULT_SPHERICAL_SLAB_DIMS: [u32; 3] = [27, 2, 14];
pub const DEFAULT_SPHERICAL_SLAB_DEPTH: u8 = 3;
pub const DEFAULT_SPHERICAL_EMBEDDING_DEPTH: u8 = 1;
pub const DEFAULT_SPHERICAL_CELL_SUBTREE_DEPTH: u8 = 20;
/// Latitude clamp — same value as `sphere-mercator-1` to skip the
/// poles entirely.
pub const DEFAULT_SPHERICAL_LAT_MAX: f32 = 1.26;

pub fn spherical_wrapped_planet_world(
    embedding_depth: u8,
    slab_dims: [u32; 3],
    slab_depth: u8,
    cell_subtree_depth: u8,
    lat_max: f32,
) -> WorldState {
    assert!(embedding_depth > 0);
    assert!(slab_depth > 0);
    assert!(cell_subtree_depth >= 1);
    let total_depth = (embedding_depth as usize)
        .saturating_add(slab_depth as usize)
        .saturating_add(cell_subtree_depth as usize);
    assert!(total_depth <= MAX_DEPTH);

    let mut subgrid: u32 = 1;
    for _ in 0..slab_depth {
        subgrid = subgrid.checked_mul(BRANCH as u32).unwrap();
    }
    assert!(slab_dims[0] <= subgrid && slab_dims[1] <= subgrid && slab_dims[2] <= subgrid);

    let mut library = NodeLibrary::default();

    // Inner uniform-stone subtree shared by all cells (depth =
    // cell_subtree_depth - 1 since the TB head consumes one level).
    fn build_uniform_anchor(library: &mut NodeLibrary, block: u16, depth: u8) -> Child {
        if depth == 0 {
            return Child::Block(block);
        }
        let inner = build_uniform_anchor(library, block, depth - 1);
        Child::Node(library.insert(uniform_children(inner)))
    }

    let n_lng = slab_dims[0];
    let n_r = slab_dims[1];
    let n_lat = slab_dims[2];
    // body_radius in WrappedPlane's [0, 3)³ units. r = N_lng·cell_size
    // / (2π) keeps the equator's arc-cell-size consistent with the
    // slab cell extent: 2π·r / N_lng = cell_size = 3/N_lng → r = 3/(2π).
    let body_radius_wp: f32 = 3.0 / (2.0 * std::f32::consts::PI);
    // Cell size in WP's [0, 3)³ units.
    let cell_size_wp: f32 = 3.0 / subgrid as f32;
    // Sphere centre in WP's [0, 3)³.
    let centre = [1.5_f32, 1.5_f32, 1.5_f32];

    // Per-cell builder: TangentBlock head with rotation+cell_offset
    // pointing the cell at its sphere position. NO dedup (each cell
    // unique).
    let make_cell = |library: &mut NodeLibrary,
                     lon_idx: u32,
                     r_idx: u32,
                     lat_idx: u32,
                     material: u16|
     -> Child {
        // Lon convention matches the shader: cell 0 is at lon=-π
        // (=−X axis) and cell N_lng-1 is at lon=+π−ε. The shader's
        // `u = (lon + π) / (2π)` inverts to `lon = -π + u · 2π`,
        // which gives cell-centre `lon_c = -π + (lon_idx + 0.5) · (2π/N_lng)`.
        let lon_c = -std::f32::consts::PI
            + (lon_idx as f32 + 0.5) / n_lng as f32 * std::f32::consts::TAU;
        let lat_c = -lat_max + (lat_idx as f32 + 0.5) / n_lat as f32 * 2.0 * lat_max;
        // Radial position: cells stack INWARD from body_radius.
        // r_idx=N_r-1 is the OUTERMOST shell (at body_radius − ½·cell_size);
        // r_idx=0 is the INNERMOST (at body_radius − (N_r−½)·cell_size).
        // This matches the shader's `r_p` → `cy` inverse mapping
        // `r_p = r_inner + (cy + 0.5)·cell_size_render`, so the cell
        // the shader looks for at cy=r_idx is the SAME cell the
        // bootstrap places at r_idx. (Earlier the formula was
        // `body_radius + (r_idx + 0.5)·cell_size` — pointing OUTWARD,
        // which placed cells outside the shader's expected radial
        // band; cells rendered but at world positions inconsistent
        // with the ray's spherical intersection, making blocks
        // appear to drift as the camera moved.)
        let r_c = body_radius_wp
            - (n_r as f32 - r_idx as f32 - 0.5) * cell_size_wp;
        let cos_lat = lat_c.cos();
        let sin_lat = lat_c.sin();
        let cos_lon = lon_c.cos();
        let sin_lon = lon_c.sin();
        let sphere_pos = [
            centre[0] + r_c * cos_lat * cos_lon,
            centre[1] + r_c * sin_lat,
            centre[2] + r_c * cos_lat * sin_lon,
        ];
        // Natural slot centre in WP's [0, 3)³: each slab subgrid cell
        // is at (lon_idx + 0.5)·cell_size_wp etc.
        let natural_centre = [
            (lon_idx as f32 + 0.5) * cell_size_wp,
            (r_idx as f32 + 0.5) * cell_size_wp,
            (lat_idx as f32 + 0.5) * cell_size_wp,
        ];
        // cell_offset is in DIRECT-PARENT slot units (1 unit = 1
        // cell_size). Convert via division by cell_size_wp.
        let cell_offset = [
            (sphere_pos[0] - natural_centre[0]) / cell_size_wp,
            (sphere_pos[1] - natural_centre[1]) / cell_size_wp,
            (sphere_pos[2] - natural_centre[2]) / cell_size_wp,
        ];
        // Tangent rotation: storage +Y → radial outward at this cell.
        // R = R_y(lon_c) · R_x(-lat_c). Same construction as
        // dodecahedron's `rotation_align_y_to`, but expressed via
        // axis rotations since lat/lon parameterise the basis.
        let r_lon = crate::world::tree::rotation_y(lon_c);
        let r_lat = rotation_x_local(-lat_c);
        let rotation = crate::world::mat3::matmul(&r_lon, &r_lat);

        let inner = build_uniform_anchor(library, material, cell_subtree_depth - 1);
        Child::Node(library.insert_with_kind(
            uniform_children(inner),
            NodeKind::TangentBlock { rotation, cell_offset },
        ))
    };

    // Material rule: y=0 stone, y=top grass, otherwise dirt.
    let material_at = |r_idx: u32| -> u16 {
        if r_idx == 0 {
            block::STONE
        } else if r_idx + 1 == n_r {
            block::GRASS
        } else {
            block::DIRT
        }
    };

    // Layer 0: leaves at full subgrid resolution. Per-cell unique TBs.
    let n0 = subgrid as usize;
    let mut layer: Vec<Vec<Vec<Child>>> = vec![vec![vec![Child::Empty; n0]; n0]; n0];
    for lat_idx in 0..n_lat {
        for r_idx in 0..n_r {
            for lon_idx in 0..n_lng {
                let cell =
                    make_cell(&mut library, lon_idx, r_idx, lat_idx, material_at(r_idx));
                layer[lat_idx as usize][r_idx as usize][lon_idx as usize] = cell;
            }
        }
    }

    // Bottom-up 3³ grouping into Cartesian nodes (slab_depth - 1
    // rounds), exactly mirroring wrapped_planet.
    let mut size = n0;
    for _round in 0..(slab_depth as usize - 1) {
        let new_size = size / 3;
        let mut next: Vec<Vec<Vec<Child>>> = (0..new_size)
            .map(|_| (0..new_size).map(|_| vec![Child::Empty; new_size]).collect())
            .collect();
        for nz in 0..new_size {
            for ny in 0..new_size {
                for nx in 0..new_size {
                    let mut children = empty_children();
                    let mut all_empty = true;
                    for cz in 0..BRANCH {
                        for cy in 0..BRANCH {
                            for cx in 0..BRANCH {
                                let x = nx * BRANCH + cx;
                                let y = ny * BRANCH + cy;
                                let z = nz * BRANCH + cz;
                                let c = layer[z][y][x];
                                if !c.is_empty() {
                                    all_empty = false;
                                }
                                children[slot_index(cx, cy, cz)] = c;
                            }
                        }
                    }
                    next[nz][ny][nx] = if all_empty {
                        Child::Empty
                    } else {
                        Child::Node(library.insert_with_kind(children, NodeKind::Cartesian))
                    };
                }
            }
        }
        layer = next;
        size = new_size;
    }
    debug_assert_eq!(size, BRANCH as usize);

    // Final 3³ → WrappedPlane root.
    let mut slab_children = empty_children();
    for cz in 0..BRANCH {
        for cy in 0..BRANCH {
            for cx in 0..BRANCH {
                slab_children[slot_index(cx, cy, cz)] = layer[cz][cy][cx];
            }
        }
    }
    let wrapped_plane_root = library.insert_with_kind(
        slab_children,
        NodeKind::SphericalWrappedPlane {
            dims: slab_dims,
            slab_depth,
            body_radius_cells: body_radius_wp,
            lat_max,
        },
    );

    // Embed in `embedding_depth` Cartesian layers, slot 13 each level.
    let mut current = Child::Node(wrapped_plane_root);
    for _ in 0..(embedding_depth as usize) {
        let mut children = empty_children();
        children[slot_index(1, 1, 1)] = current;
        current = Child::Node(library.insert_with_kind(children, NodeKind::Cartesian));
    }
    let root = match current {
        Child::Node(id) => id,
        _ => unreachable!(),
    };
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "spherical_wrapped_planet world: dims={:?}, slab_depth={}, body_radius_wp={:.4}, library_entries={}, tree_depth={}",
        slab_dims, slab_depth, body_radius_wp, world.library.len(), world.tree_depth(),
    );
    world
}

/// Rotation about the LOCAL +X axis by `radians`, column-major.
/// (Equivalent to `tree::rotation_x` if it existed.) Used for the
/// latitude tilt in the per-cell tangent basis.
fn rotation_x_local(radians: f32) -> [[f32; 3]; 3] {
    let (s, c) = radians.sin_cos();
    [
        [1.0, 0.0, 0.0],
        [0.0, c, s],
        [0.0, -s, c],
    ]
}

/// Spawn INSIDE the slab embedding's box but outside the sphere
/// itself, looking diagonally at the sphere centre.
///
/// **Why slot 13 (centre) rather than slot 0 (corner)**: zoom_in_in_world
/// descends the anchor through slot floor(offset * 3) at each level.
/// If we spawn in slot 0, every zoom step descends the corner chain
/// — the render frame zooms toward the (0, 0, 0) corner of the
/// world, AWAY from the sphere at (1.5, 1.5, 1.5). Cells "disappear"
/// because the render frame's box no longer contains them.
///
/// Spawning in slot 13 (=embedding cell, which contains the
/// WrappedPlane) keeps the anchor descent chain pointed at the
/// sphere. From (1.1, 1.1, 1.1) the camera is inside slot 13 (=
/// world `(1, 2)³`) but outside the sphere (= world ≈
/// `(1.34, 1.66)³` for embedding_depth=1).
pub fn spherical_wrapped_planet_spawn(embedding_depth: u8) -> WorldPos {
    // Anchor at slot 13 (= SphericalWP node) with offset (0.1, 0.1,
    // 0.1) → world (1.1, 1.1, 1.1). Then deepen by RENDER_FRAME_K=3
    // levels so the render frame truncates to depth=embedding_depth
    // (= the SphericalWP itself). That fires `march_spherical_wrapped_plane`.
    // Need anchor depth ≥ embedding_depth + RENDER_FRAME_K (=3) so
    // the render frame settles at the SphericalWP node, where the
    // sphere DDA dispatch fires. Deeper is also fine — the
    // truncate-while-strict-descendant rule keeps render_path at
    // SphericalWP regardless of how deep the anchor goes.
    WorldPos::uniform_column(slot_index(1, 1, 1) as u8, 1, [0.1, 0.1, 0.1])
        .deepened_to(embedding_depth + 3)
}

pub(super) fn bootstrap_spherical_wrapped_planet_world(
    embedding_depth: u8,
    slab_dims: [u32; 3],
    slab_depth: u8,
    cell_subtree_depth: u8,
) -> WorldBootstrap {
    let world = spherical_wrapped_planet_world(
        embedding_depth,
        slab_dims,
        slab_depth,
        cell_subtree_depth,
        DEFAULT_SPHERICAL_LAT_MAX,
    );
    let spawn_pos = spherical_wrapped_planet_spawn(embedding_depth);
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        // From (0.4, 0.4, 0.4) looking up-and-diagonally at the
        // sphere centre (1.5, 1.5, 1.5). Engine yaw convention has
        // π/4 pointing -X-Z (verified empirically), so 5π/4 points
        // +X+Z. Pitch +0.616 looks up.
        default_spawn_yaw: 5.0 * std::f32::consts::FRAC_PI_4,
        default_spawn_pitch: 0.616,
        plain_layers: 0,
        color_registry: crate::world::palette::ColorRegistry::new(),
    }
}
