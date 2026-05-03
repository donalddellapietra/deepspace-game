//! UV-sphere tangent-cell test world.
//!
//! Diagnostic UV ring world.
//!
//! Content is stored as a straight `[27, 1, 1]` UV lattice under a
//! `UvRing` root. The render path maps that one row into a ring, so
//! placement and tangent rotation come from the same UV coordinate
//! instead of rounded Cartesian sphere samples.

use super::WorldBootstrap;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, uniform_children, Child, NodeKind, NodeLibrary,
};

const GRID_DEPTH: u8 = super::DEFAULT_WRAPPED_PLANET_SLAB_DEPTH;
const GRID_SIZE: usize = 27;
const CELL_SUBTREE_DEPTH: u8 = 20;
const RING_DIMS: [u32; 3] = [super::DEFAULT_WRAPPED_PLANET_SLAB_DIMS[0], 1, 1];

#[inline]
#[cfg(test)]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / m, v[1] / m, v[2] / m]
}

#[inline]
#[cfg(test)]
fn tangent_rotation_for_ring_cell(cell_x: u32) -> [[f32; 3]; 3] {
    let lat = 0.0f32;
    let lon = -std::f32::consts::PI
        + (cell_x as f32 + 0.5) * (2.0 * std::f32::consts::PI / RING_DIMS[0] as f32);
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

#[inline]
fn grid_index(x: usize, y: usize, z: usize) -> usize {
    (z * GRID_SIZE + y) * GRID_SIZE + x
}

fn build_grid_tree(library: &mut NodeLibrary, leaves: Vec<Child>, root_kind: NodeKind) -> Child {
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
                    let kind = if next_size == 1 { root_kind } else { NodeKind::Cartesian };
                    next[(z * next_size + y) * next_size + x] = if children.iter().all(|c| c.is_empty()) {
                        Child::Empty
                    } else {
                        Child::Node(library.insert_with_kind(children, kind))
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
    for x in 0..RING_DIMS[0] as usize {
        let idx = grid_index(x, 0, 0);
        leaves[idx] = tangent_content;
    }

    let root_child = build_grid_tree(
        &mut library,
        leaves,
        NodeKind::UvRing { dims: RING_DIMS, slab_depth: GRID_DEPTH },
    );
    let root = match root_child {
        Child::Node(id) => id,
        _ => library.insert(empty_children()),
    };
    library.ref_inc(root);
    let world = WorldState { root, library };
    eprintln!(
        "uv_sphere_test uv-ring world: tree_depth={} library_entries={} ring_dims={:?} slab_depth={}",
        world.tree_depth(),
        world.library.len(),
        RING_DIMS,
        GRID_DEPTH,
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
        let a = tangent_rotation_for_ring_cell(0);
        let b = tangent_rotation_for_ring_cell(1);
        assert_ne!(a, b);
        let radial = a[1];
        let len = (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ring_lattice_matches_wrapped_plane_row() {
        assert_eq!(RING_DIMS, [super::super::DEFAULT_WRAPPED_PLANET_SLAB_DIMS[0], 1, 1]);
    }

    #[test]
    fn ring_uses_one_fixed_uv_latitude() {
        for cell in 0..RING_DIMS[0] {
            let rotation = tangent_rotation_for_ring_cell(cell);
            let radial = rotation[1];
            assert!(radial[1].abs() < 1e-6, "radial y drifted at cell {cell}: {}", radial[1]);
        }
    }

    #[test]
    fn root_is_uv_ring() {
        let world = uv_sphere_test_world();
        let root = world.library.get(world.root).expect("root exists");
        match root.kind {
            NodeKind::UvRing { dims, slab_depth } => {
                assert_eq!(dims, RING_DIMS);
                assert_eq!(slab_depth, GRID_DEPTH);
            }
            other => panic!("expected UvRing root, got {other:?}"),
        }
    }

    #[test]
    fn ring_row_cells_are_populated_in_uv_lattice() {
        let world = uv_sphere_test_world();
        for x in 0..RING_DIMS[0] as usize {
            let mut node_id = world.root;
            for level in (0..GRID_DEPTH as u32).rev() {
                let div = 3usize.pow(level);
                let sx = (x / div) % 3;
                let slot = slot_index(sx, 0, 0);
                let node = world.library.get(node_id).expect("node exists");
                match node.children[slot] {
                    Child::Node(child) => node_id = child,
                    other => panic!("ring x={x} missing at slot {slot}: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn ring_content_is_deduped_under_wrapped_plane() {
        let world = uv_sphere_test_world();
        fn count_nonempty_nodes(
            library: &NodeLibrary,
            child: Child,
            seen: &mut std::collections::HashSet<crate::world::tree::NodeId>,
        ) -> usize {
            let Some(id) = child.node_id() else { return 0 };
            if !seen.insert(id) {
                return 0;
            }
            let Some(node) = library.get(id) else { return 0 };
            1 + node.children
                .iter()
                .copied()
                .map(|c| count_nonempty_nodes(library, c, seen))
                .sum::<usize>()
        }

        let nodes = count_nonempty_nodes(
            &world.library,
            Child::Node(world.root),
            &mut std::collections::HashSet::new(),
        );
        assert!(
            world.library.len() < RING_DIMS[0] as usize + CELL_SUBTREE_DEPTH as usize + 10,
            "ring content should stay deduped: library={} reachable_nodes={nodes}",
            world.library.len(),
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
