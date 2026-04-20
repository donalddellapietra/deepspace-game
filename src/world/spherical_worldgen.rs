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
        depth: 20,
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

/// Build a spawn `WorldPos` on the outer shell of the planet at the
/// given anchor depth. The camera will sit just outside the body's
/// outer shell on `face`, ready to look radially inward.
///
/// The anchor is built via pure Cartesian descent through the body
/// cell. That's load-bearing: the tree's face subtree interprets
/// child slots as `(u, v, r)`, but `WorldPos::in_frame` walks the
/// anchor with Cartesian `(x, y, z)` semantics. If we built the
/// anchor with UVR-slot math, `in_frame(&body_path)` would return
/// a Cartesian-misinterpreted position (e.g., deep inside the shell
/// instead of on top of it), and the camera would spawn inside the
/// planet. Cartesian descent keeps the camera's world position
/// consistent with how `in_frame` reads it.
///
/// Side effect: the anchor's path doesn't correspond to the tree's
/// UVR cell at that depth — it's a purely positional address. The
/// shader projects from camera *position* (not *anchor*) and the
/// raycast hits real tree cells regardless.
pub fn demo_sphere_surface_spawn(
    body_path: &Path,
    setup: &PlanetSetup,
    anchor_depth: u8,
    face: Face,
) -> WorldPos {
    // Surface point in the body cell's local `[0, 1)³` frame, then
    // scaled to the anchor.rs `WORLD_SIZE = 3.0` convention used by
    // `from_frame_local`. The outer shell sits at
    // `body_center + face_normal * outer_r`.
    let n = face.normal();
    // Step slightly inside the outer shell so the camera is
    // guaranteed within the body cell (avoids clamp fallback in
    // `from_frame_local`).
    let r = setup.outer_r - f32::EPSILON;
    let pos_in_body_01 = [
        0.5 + n[0] * r,
        0.5 + n[1] * r,
        0.5 + n[2] * r,
    ];
    // `from_frame_local` expects coords in the frame's `[0, 3)³`
    // cell — body cell local.
    let pos_in_body_03 = [
        pos_in_body_01[0] * 3.0,
        pos_in_body_01[1] * 3.0,
        pos_in_body_01[2] * 3.0,
    ];
    let target_depth = anchor_depth.max(body_path.depth());
    WorldPos::from_frame_local(body_path, pos_in_body_03, target_depth)
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
