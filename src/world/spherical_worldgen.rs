//! Insert a spherical body into the world tree.
//!
//! The planet IS a `NodeKind::CubedSphereBody` node, placed at a
//! chosen path inside the world's Cartesian tree. There's no
//! external `SphericalPlanet` handle — the tree is the single
//! source of truth, and the path returned here is just a hint for
//! spawn-position computation.

use super::anchor::Path;
use super::cubesphere::insert_spherical_body;
use super::palette::block;
use super::sdf::{Planet, Vec3};
use super::tree::{slot_index, Child, NodeId, NodeLibrary};

/// Declarative planet description. Radii are in **the body cell's
/// local `[0, 1)` frame** (per spec §1d). The body cell's actual
/// world-space size is determined by where in the tree it lives.
#[derive(Clone, Debug)]
pub struct PlanetSetup {
    /// Radii in body cell local `[0, 1)`.
    pub inner_r: f32,
    pub outer_r: f32,
    /// Face subtree depth.
    pub depth: u32,
    /// SDF in body cell local frame (center = `(0.5, 0.5, 0.5)`).
    pub sdf: Planet,
}

/// The demo / starter planet. Body cell-local: `outer_r ≤ 0.5` so
/// the sphere fits cleanly in one cell of its parent.
pub fn demo_planet() -> PlanetSetup {
    let center: Vec3 = [0.5, 0.5, 0.5];
    let inner_r = 0.12_f32;
    let outer_r = 0.45_f32;
    PlanetSetup {
        inner_r,
        outer_r,
        depth: 25,
        sdf: Planet {
            center,
            radius: 0.30,
            noise_scale: 0.015,
            noise_freq: 8.0,
            noise_seed: 2024,
            gravity: 9.8,
            influence_radius: outer_r * 2.0,
            surface_block: block::GRASS,
            core_block: block::STONE,
        },
    }
}

/// Build a body node and place it at the given world-tree path,
/// returning the body's `NodeId` and the full path to it from
/// `world_root`.
///
/// `host_path` is the slots from `world_root` to the cell that
/// will become the body. The cell at the end of `host_path`
/// becomes a `CubedSphereBody` child of its parent.
///
/// For the simplest case (planet at root center), pass a single
/// `slot_index(1, 1, 1) = 13` to put the body in the central
/// depth-1 cell.
pub fn insert_into_tree(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    host_slots: &[u8],
    setup: &PlanetSetup,
) -> (NodeId, Path, Path) {
    assert!(!host_slots.is_empty(), "host_slots must point at a child");

    let body_id = insert_spherical_body(
        lib, setup.inner_r, setup.outer_r, setup.depth, &setup.sdf,
    );

    // Rebuild the path from world_root downward, replacing the
    // target slot with the body. Only the path levels are
    // touched; all other siblings are preserved.
    let new_root = install_body(lib, world_root, host_slots, body_id);

    let mut body_path = Path::root();
    for &s in host_slots { body_path.push(s); }
    (new_root, body_path, body_path)
}

/// Walk down `slots`, expanding any uniform terminals on the path
/// into Node children, then install `new_node` at the leaf and
/// rebuild parents on the way up. Returns the new world root id.
fn install_body(
    lib: &mut NodeLibrary,
    root: NodeId,
    slots: &[u8],
    new_node: NodeId,
) -> NodeId {
    fn rebuild(
        lib: &mut NodeLibrary,
        current: NodeId,
        slots: &[u8],
        level: usize,
        new_node: NodeId,
    ) -> NodeId {
        let target = slots[level] as usize;
        let node = lib.get(current).expect("install path must exist in library");
        let mut new_children = node.children;
        if level + 1 == slots.len() {
            new_children[target] = Child::Node(new_node);
        } else {
            let next_id = match node.children[target] {
                Child::Node(nid) => rebuild(lib, nid, slots, level + 1, new_node),
                Child::Empty | Child::Block(_) => {
                    use super::tree::empty_children;
                    let expanded = lib.insert(empty_children());
                    rebuild(lib, expanded, slots, level + 1, new_node)
                }
            };
            new_children[target] = Child::Node(next_id);
        }
        lib.insert(new_children)
    }
    rebuild(lib, root, slots, 0, new_node)
}

/// Convenience — install at slot 13 (depth-1 center cell of the root).
pub fn install_at_root_center(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    setup: &PlanetSetup,
) -> (NodeId, Path) {
    let (new_root, body_path, _) = insert_into_tree(
        lib, world_root, &[slot_index(1, 1, 1) as u8], setup,
    );
    (new_root, body_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::NodeKind;

    #[test]
    fn demo_planet_setup_valid() {
        let s = demo_planet();
        assert!(0.0 < s.inner_r && s.inner_r < s.outer_r);
        assert!(s.outer_r <= 0.5);
    }

    #[test]
    fn install_at_root_center_creates_body_at_slot_13() {
        let mut lib = NodeLibrary::default();
        // Empty world root.
        let root = lib.insert(super::super::tree::empty_children());
        let setup = demo_planet();
        let (new_root, body_path) = install_at_root_center(&mut lib, root, &setup);
        assert_eq!(body_path.depth(), 1);
        assert_eq!(body_path.slot(0), slot_index(1, 1, 1) as u8);

        let new_root_node = lib.get(new_root).unwrap();
        let body_child = match new_root_node.children[slot_index(1, 1, 1)] {
            Child::Node(id) => id,
            _ => panic!("slot 13 not a Node"),
        };
        let body = lib.get(body_child).unwrap();
        assert!(matches!(body.kind, NodeKind::CubedSphereBody { .. }));
    }
}
