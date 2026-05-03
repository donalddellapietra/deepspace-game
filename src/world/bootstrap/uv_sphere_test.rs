//! UV-sphere tangent-cell test world.
//!
//! This is the UV analogue of `dodecahedron_test`: many tangent
//! cells are placed as real Cartesian tree cells on a spherical shell,
//! and each cell carries the lat/lon tangent basis that maps storage
//! +Y to the local radial normal. The sphere shape therefore comes
//! from actual tree placement, not from a special WrappedPlane shader
//! reinterpretation.

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
const UV_DIMS: [u32; 2] = [54, 27];
const LAT_MAX: f32 = 1.26;

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / m, v[1] / m, v[2] / m]
}

#[inline]
fn tangent_rotation_for_uv_cell(cell_x: u32, cell_z: u32) -> [[f32; 3]; 3] {
    let pi = std::f32::consts::PI;
    let lon_step = 2.0 * pi / UV_DIMS[0] as f32;
    let lat_step = 2.0 * LAT_MAX / UV_DIMS[1] as f32;
    let lon = -pi + (cell_x as f32 + 0.5) * lon_step;
    let lat = -LAT_MAX + (cell_z as f32 + 0.5) * lat_step;
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    let east = normalize([-so, 0.0, co]);
    let radial = normalize([cl * co, sl, cl * so]);
    let north = normalize([-sl * co, cl, -sl * so]);
    [east, radial, north]
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

fn build_tangent_cell(
    library: &mut NodeLibrary,
    block_id: u16,
    depth: u8,
    rotation: [[f32; 3]; 3],
) -> Child {
    assert!(depth >= 1, "tangent cell subtree depth must be >= 1");
    let inner = build_uniform_cartesian_subtree(library, block_id, depth - 1);
    Child::Node(library.insert_with_kind(
        uniform_children(inner),
        NodeKind::TangentPlane { rotation },
    ))
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
                    next[(z * next_size + y) * next_size + x] =
                        Child::Node(library.insert(children));
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
    let centre = (GRID_SIZE as f32 - 1.0) * 0.5;
    let radius = 29.0_f32;

    for v in 0..UV_DIMS[1] {
        for u in 0..UV_DIMS[0] {
            let rotation = tangent_rotation_for_uv_cell(u, v);
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
            leaves[idx] = build_tangent_cell(
                &mut library,
                block::GRASS,
                CELL_SUBTREE_DEPTH,
                rotation,
            );
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
        "uv_sphere_test world: tree_depth={} library_entries={} uv_dims={:?}",
        world.tree_depth(),
        world.library.len(),
        UV_DIMS,
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
        let a = tangent_rotation_for_uv_cell(0, 13);
        let b = tangent_rotation_for_uv_cell(1, 13);
        assert_ne!(a, b);
        let radial = a[1];
        let len = (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }
}
