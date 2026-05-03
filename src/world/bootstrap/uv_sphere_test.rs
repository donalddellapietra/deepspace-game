//! UV-sphere tangent-cell test world.
//!
//! This is the UV analogue of `dodecahedron_test`: many tangent
//! cells are placed as real Cartesian tree cells on a spherical shell,
//! and each cell is a placed tangent instance carrying the lat/lon
//! tangent basis that maps storage +Y to the local radial normal.
//! The sphere shape therefore comes from actual tree placement, not
//! from a special WrappedPlane shader reinterpretation.

use super::WorldBootstrap;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, uniform_children, Child, NodeKind, NodeLibrary,
};

const GRID_DEPTH: u8 = 4; // 3^4 = 81 cells per axis.
const GRID_SIZE: usize = 81;
const CELL_SUBTREE_DEPTH: u8 = 20;
const SPHERE_RADIUS_CELLS: f32 = 36.0;
const TARGET_CENTER_SPACING: f32 = 0.85;
const LAT_RINGS: u32 = 107;
const GLOBAL_LONGITUDES: u32 = 256;
const LAT_MAX: f32 = 1.26;

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / m, v[1] / m, v[2] / m]
}

#[inline]
fn ring_longitude_cells(lat: f32) -> u32 {
    let circumference = 2.0 * std::f32::consts::PI * SPHERE_RADIUS_CELLS * lat.cos().abs();
    (circumference / TARGET_CENTER_SPACING).ceil().max(8.0) as u32
}

#[inline]
fn tangent_rotation_for_lat_lon(lat: f32, lon: f32) -> [[f32; 3]; 3] {
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    let east = normalize([-so, 0.0, co]);
    let radial = normalize([cl * co, sl, cl * so]);
    let north = normalize([-sl * co, cl, -sl * so]);
    [east, radial, north]
}

#[inline]
fn uv_lat_for_ring(ring: u32) -> f32 {
    let lat_step = 2.0 * LAT_MAX / LAT_RINGS as f32;
    -LAT_MAX + (ring as f32 + 0.5) * lat_step
}

#[inline]
fn uv_lon_for_lattice_cell(cell_x: u32) -> f32 {
    let pi = std::f32::consts::PI;
    -pi + (cell_x as f32 + 0.5) * (2.0 * pi / GLOBAL_LONGITUDES as f32)
}

#[inline]
fn longitude_stride_for_ring(lat: f32) -> u32 {
    let target = ring_longitude_cells(lat).min(GLOBAL_LONGITUDES);
    GLOBAL_LONGITUDES.div_ceil(target).max(1)
}

fn build_uniform_cartesian_subtree(
    library: &mut NodeLibrary,
    block_id: u16,
    depth: u8,
) -> Child {
    if depth == 0 {
        return Child::Block(block_id);
    }
    let inner = build_uniform_cartesian_subtree(library, block_id, depth - 1);
    Child::Node(library.insert(uniform_children(inner)))
}

fn build_tangent_cell(content: Child, rotation: [[f32; 3]; 3]) -> Child {
    let Child::Node(node) = content else {
        panic!("tangent cell content must be a node");
    };
    Child::PlacedNode {
        node,
        kind: NodeKind::TangentBlock { rotation },
    }
}

#[inline]
fn grid_index(x: usize, y: usize, z: usize) -> usize {
    (z * GRID_SIZE + y) * GRID_SIZE + x
}

fn build_grid_tree(library: &mut NodeLibrary, leaves: Vec<Child>) -> Child {
    let mut size = GRID_SIZE;
    let mut layer = leaves;
    while size > 1 {
        debug_assert_eq!(size % 3, 0);
        let next_size = size / 3;
        let mut next = vec![Child::Empty; next_size * next_size * next_size];
        for z in 0..next_size {
            for y in 0..next_size {
                for x in 0..next_size {
                    let mut children = empty_children();
                    for dz in 0..3 {
                        for dy in 0..3 {
                            for dx in 0..3 {
                                let src_x = x * 3 + dx;
                                let src_y = y * 3 + dy;
                                let src_z = z * 3 + dz;
                                children[slot_index(dx, dy, dz)] =
                                    layer[(src_z * size + src_y) * size + src_x];
                            }
                        }
                    }
                    next[(z * next_size + y) * next_size + x] = if children.iter().all(|c| c.is_empty()) {
                        Child::Empty
                    } else {
                        Child::Node(library.insert(children))
                    };
                }
            }
        }
        layer = next;
        size = next_size;
    }
    layer[0]
}

pub(super) fn uv_sphere_test_world() -> WorldState {
    debug_assert_eq!(GRID_SIZE, 3usize.pow(GRID_DEPTH as u32));
    let mut library = NodeLibrary::default();
    let mut leaves = vec![Child::Empty; GRID_SIZE * GRID_SIZE * GRID_SIZE];
    let tangent_content =
        build_uniform_cartesian_subtree(&mut library, block::GRASS, CELL_SUBTREE_DEPTH);
    let centre = (GRID_SIZE as f32 - 1.0) * 0.5;
    let radius = SPHERE_RADIUS_CELLS;
    let mut occupied = std::collections::HashSet::new();

    for v in 0..LAT_RINGS {
        let lat = uv_lat_for_ring(v);
        let stride = longitude_stride_for_ring(lat);
        let mut u = 0;
        while u < GLOBAL_LONGITUDES {
            let cell_x = u;
            u += stride;
            let lon = uv_lon_for_lattice_cell(cell_x);
            let rotation = tangent_rotation_for_lat_lon(lat, lon);
            let radial = rotation[1];
            let gx = (centre + radial[0] * radius).round() as i32;
            let gy = (centre + radial[1] * radius).round() as i32;
            let gz = (centre + radial[2] * radius).round() as i32;
            if !(0..GRID_SIZE as i32).contains(&gx)
                || !(0..GRID_SIZE as i32).contains(&gy)
                || !(0..GRID_SIZE as i32).contains(&gz)
            {
                continue;
            }
            let idx = grid_index(gx as usize, gy as usize, gz as usize);
            if !occupied.insert(idx) {
                continue;
            }
            leaves[idx] = build_tangent_cell(tangent_content, rotation);
        }
    }

    let root_child = build_grid_tree(&mut library, leaves);
    let root = match root_child {
        Child::Node(id) => id,
        _ => library.insert(empty_children()),
    };
    library.ref_inc(root);
    let world = WorldState { root, library };
    eprintln!(
        "uv_sphere_test world: tree_depth={} library_entries={} lat_rings={} global_longitudes={} radius_cells={}",
        world.tree_depth(),
        world.library.len(),
        LAT_RINGS,
        GLOBAL_LONGITUDES,
        SPHERE_RADIUS_CELLS,
    );
    world
}

pub(super) fn uv_sphere_test_spawn() -> WorldPos {
    // Empty +Z side looking back at the centre. Depth 2 places the
    // camera near the shell while keeping the initial view framed.
    let mut pos = WorldPos::uniform_column(slot_index(1, 1, 2) as u8, 1, [0.5, 0.5, 0.85]);
    pos = pos.deepened_to(2);
    pos
}

pub(super) fn bootstrap_uv_sphere_test_world() -> WorldBootstrap {
    WorldBootstrap {
        world: uv_sphere_test_world(),
        planet_path: None,
        default_spawn_pos: uv_sphere_test_spawn(),
        default_spawn_yaw: std::f32::consts::PI,
        default_spawn_pitch: 0.0,
        plain_layers: 0,
        color_registry: crate::world::palette::ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_has_expected_depth() {
        let world = uv_sphere_test_world();
        assert_eq!(world.tree_depth(), GRID_DEPTH as u32 + CELL_SUBTREE_DEPTH as u32);
    }

    #[test]
    fn rotations_are_lat_lon_dependent_and_radial() {
        let lat = uv_lat_for_ring(LAT_RINGS / 2);
        let stride = longitude_stride_for_ring(lat);
        let a = tangent_rotation_for_lat_lon(lat, uv_lon_for_lattice_cell(0));
        let b = tangent_rotation_for_lat_lon(lat, uv_lon_for_lattice_cell(stride));
        assert_ne!(a, b);
        let radial = a[1];
        let len = (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ring_sampling_targets_cell_sized_spacing() {
        let equator = uv_lat_for_ring(LAT_RINGS / 2);
        let equator_stride = longitude_stride_for_ring(equator);
        let equator_count = GLOBAL_LONGITUDES.div_ceil(equator_stride);
        let equator_spacing =
            2.0 * std::f32::consts::PI * SPHERE_RADIUS_CELLS * equator.cos().abs()
                / equator_count as f32;
        let lat_spacing = 2.0 * LAT_MAX * SPHERE_RADIUS_CELLS / LAT_RINGS as f32;
        assert!(
            (0.6..=1.1).contains(&equator_spacing),
            "equator spacing should be near target cell spacing, got {equator_spacing}",
        );
        assert!(
            (0.6..=1.1).contains(&lat_spacing),
            "latitude spacing should be near target cell spacing, got {lat_spacing}",
        );
    }

    #[test]
    fn sampled_longitudes_stay_on_global_meridians() {
        let low_lat = uv_lat_for_ring(0);
        let mid_lat = uv_lat_for_ring(LAT_RINGS / 2);
        let low_stride = longitude_stride_for_ring(low_lat);
        let mid_stride = longitude_stride_for_ring(mid_lat);
        assert_eq!(low_stride % mid_stride, 0, "sparser rings should use meridian subsets");
    }

    #[test]
    fn sphere_cells_are_deduped_placed_tangent_blocks() {
        let world = uv_sphere_test_world();
        fn count_placed_tangent_blocks(
            library: &NodeLibrary,
            child: Child,
            seen: &mut std::collections::HashSet<crate::world::tree::NodeId>,
        ) -> usize {
            let placed = usize::from(matches!(
                child,
                Child::PlacedNode { kind: NodeKind::TangentBlock { .. }, .. }
            ));
            let Some(id) = child.node_id() else { return placed };
            if !seen.insert(id) {
                return placed;
            }
            let Some(node) = library.get(id) else { return 0 };
            placed + node.children
                .iter()
                .copied()
                .map(|c| count_placed_tangent_blocks(library, c, seen))
                .sum::<usize>()
        }

        let tangent_blocks = count_placed_tangent_blocks(
            &world.library,
            Child::Node(world.root),
            &mut std::collections::HashSet::new(),
        );
        assert!(
            tangent_blocks > 100,
            "expected many tangent sphere cells, got {tangent_blocks}",
        );
        assert!(
            world.library.len() < tangent_blocks,
            "content nodes should be deduped below placed rotations: library={} tangent_blocks={}",
            world.library.len(),
            tangent_blocks,
        );
    }

    #[test]
    fn no_tangent_plane_nodes_in_geometric_uv_sphere() {
        let world = uv_sphere_test_world();
        fn has_tangent_plane(
            library: &NodeLibrary,
            child: Child,
            seen: &mut std::collections::HashSet<crate::world::tree::NodeId>,
        ) -> bool {
            if matches!(child, Child::PlacedNode { kind: NodeKind::TangentPlane { .. }, .. }) {
                return true;
            }
            let Some(id) = child.node_id() else { return false };
            if !seen.insert(id) {
                return false;
            }
            let Some(node) = library.get(id) else { return false };
            node.kind.is_tangent_plane()
                || node.children
                    .iter()
                    .copied()
                    .any(|c| has_tangent_plane(library, c, seen))
        }

        assert!(!has_tangent_plane(
            &world.library,
            Child::Node(world.root),
            &mut std::collections::HashSet::new(),
        ));
    }
}
