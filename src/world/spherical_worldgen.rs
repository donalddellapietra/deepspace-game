//! Install a spherical body into the world tree + spawn helpers.
//!
//! The planet IS a `NodeKind::CubedSphereBody` node at a chosen
//! path inside the world's tree. No external `SphericalPlanet`
//! handle — the tree is the source of truth; the returned path
//! just helps callers compute a surface spawn position.

use super::anchor::{Path, WorldPos};
use super::cubesphere::{insert_spherical_body, Face, FACE_SLOTS};
use super::palette::block;
use super::sdf::{Planet, Vec3};
use super::tree::{empty_children, slot_index, Child, NodeId, NodeLibrary};

/// Declarative planet description. Radii live in the body cell's
/// local `[0, 1)` frame — the body cell's actual world size is
/// determined by where the tree places it.
#[derive(Clone, Debug)]
pub struct PlanetSetup {
    pub inner_r: f32,
    pub outer_r: f32,
    pub depth: u32,
    pub sdf: Planet,
}

/// Demo planet used by `--preset DemoSphere`. The sphere fits in one
/// cell of its parent (`outer_r <= 0.5`), so the body can live at any
/// cell-center slot without straddling a cube boundary.
pub fn demo_planet() -> PlanetSetup {
    let center: Vec3 = [0.5, 0.5, 0.5];
    let inner_r = 0.12_f32;
    let outer_r = 0.45_f32;
    PlanetSetup {
        inner_r,
        outer_r,
        depth: 28,
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

/// Build a body node and install it at `host_slots` inside
/// `world_root`. Returns the new world-root `NodeId` (content-
/// addressed rebuild) and the `Path` to the body from world root.
pub fn insert_into_tree(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    host_slots: &[u8],
    setup: &PlanetSetup,
) -> (NodeId, Path) {
    assert!(!host_slots.is_empty(), "host_slots must point at a child");
    let body_id = insert_spherical_body(
        lib, setup.inner_r, setup.outer_r, setup.depth, &setup.sdf,
    );
    let new_root = install_body(lib, world_root, host_slots, body_id);
    let mut body_path = Path::root();
    for &s in host_slots { body_path.push(s); }
    (new_root, body_path)
}

/// Walk down `slots`, expanding any uniform terminals into empty
/// Nodes, then install `new_node` at the leaf and rebuild parents.
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
        let node = lib.get(current).expect("install path exists");
        let mut new_children = node.children;
        if level + 1 == slots.len() {
            new_children[target] = Child::Node(new_node);
        } else {
            let next_id = match node.children[target] {
                Child::Node(nid) => rebuild(lib, nid, slots, level + 1, new_node),
                _ => {
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

/// Convenience — install at depth-1 center slot (body_path = `[13]`).
pub fn install_at_root_center(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    setup: &PlanetSetup,
) -> (NodeId, Path) {
    insert_into_tree(lib, world_root, &[slot_index(1, 1, 1) as u8], setup)
}

/// Build a spawn `WorldPos` that lands on the planet surface at the
/// given anchor depth. The camera will be just above the outer shell,
/// looking radially inward (down toward the planet).
///
/// The path is built precision-safely by walking face-subtree slots:
/// at each level, pick the `(u, v, r)` slot that contains the surface
/// point at `r = outer_r`. At the requested anchor depth, the cell
/// is arbitrarily small but all coordinates stay in `[0, 1)³` — no
/// 3^depth f32 quantization. This is what keeps the spawn stable at
/// anchor_depth = 20+.
pub fn demo_sphere_surface_spawn(
    body_path: &Path,
    setup: &PlanetSetup,
    anchor_depth: u8,
    face: Face,
) -> WorldPos {
    // Anchor below body: first prepend body_path, then the face-center
    // slot, then descend into the face subtree along `(u=0.5, v=0.5,
    // r=outer_r_of_face)`. "Outer" in face-normalized coords is r=1-ε
    // (just below the outer shell). u=v=0.5 puts us at face center.
    let mut path = *body_path;
    path.push(FACE_SLOTS[face as usize] as u8);

    let mut un = 0.5f32;
    let mut vn = 0.5f32;
    let mut rn = 1.0 - 1e-6; // just inside the outer shell

    // Descend the face subtree one level per remaining depth step.
    let remaining = (anchor_depth as i32 - path.depth() as i32).max(0) as usize;
    for _ in 0..remaining {
        let us = ((un * 3.0).floor() as usize).min(2);
        let vs = ((vn * 3.0).floor() as usize).min(2);
        let rs = ((rn * 3.0).floor() as usize).min(2);
        path.push(slot_index(us, vs, rs) as u8);
        un = un * 3.0 - us as f32;
        vn = vn * 3.0 - vs as f32;
        rn = rn * 3.0 - rs as f32;
    }

    // Offset within the deepest cell: the point where r barely breaks
    // the outer shell, with u/v at face center.
    WorldPos::new(path, [
        un.clamp(0.0, 1.0 - f32::EPSILON),
        vn.clamp(0.0, 1.0 - f32::EPSILON),
        rn.clamp(0.0, 1.0 - f32::EPSILON),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::NodeKind;

    #[test]
    fn demo_planet_radii_valid() {
        let s = demo_planet();
        assert!(0.0 < s.inner_r && s.inner_r < s.outer_r && s.outer_r <= 0.5);
    }

    #[test]
    fn install_at_root_center_creates_body() {
        let mut lib = NodeLibrary::default();
        let root = lib.insert(empty_children());
        let (new_root, body_path) = install_at_root_center(&mut lib, root, &demo_planet());
        assert_eq!(body_path.depth(), 1);
        assert_eq!(body_path.slot(0), slot_index(1, 1, 1) as u8);
        let body_child = match lib.get(new_root).unwrap().children[slot_index(1, 1, 1)] {
            Child::Node(id) => id,
            _ => panic!("slot 13 not a Node"),
        };
        assert!(matches!(lib.get(body_child).unwrap().kind, NodeKind::CubedSphereBody { .. }));
    }

    #[test]
    fn surface_spawn_depth_matches_request() {
        let mut body_path = Path::root();
        body_path.push(slot_index(1, 1, 1) as u8);
        for target in [8u8, 14, 20, 28, 33] {
            let pos = demo_sphere_surface_spawn(&body_path, &demo_planet(), target, Face::PosY);
            assert_eq!(pos.anchor.depth(), target);
        }
    }
}
