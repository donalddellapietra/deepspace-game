//! Planet worldgen: insert a cubed-sphere body into the tree from a
//! `Planet` SDF. Six face subtrees are built by recursive SDF
//! sampling; the core slot holds a uniform block; the remaining 20
//! body slots are empty. The caller is responsible for placing the
//! body node inside a parent and bumping its refcount.

use super::geometry::{Face, CORE_SLOT, FACE_SLOTS, face_space_to_body_point};
use crate::world::sdf::{Planet, Vec3};
use crate::world::tree::{
    empty_children, slot_index, uniform_children, Child, NodeId, NodeKind, NodeLibrary,
};

/// The demo planet. Inner/outer radii in the body cell's local
/// `[0, 1)` frame — `0 < inner_r < outer_r ≤ 0.5` so the sphere fits
/// cleanly inside one Cartesian cell. A smooth surface (no noise),
/// stone core, grass surface.
#[derive(Clone, Debug)]
pub struct PlanetSetup {
    pub inner_r: f32,
    pub outer_r: f32,
    /// Face subtree depth. Internal SDF sampling caps at
    /// `SDF_DETAIL_LEVELS` past which uniform-stone/uniform-empty
    /// fillers extend the subtree via dedup.
    pub depth: u32,
    pub sdf: Planet,
}

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
            noise_scale: 0.0,
            noise_freq: 1.0,
            noise_seed: 0,
            gravity: 9.8,
            influence_radius: outer_r * 2.0,
            surface_block: crate::world::palette::block::GRASS,
            core_block: crate::world::palette::block::STONE,
        },
    }
}

/// Max levels of SDF recursion into a face subtree. Below this, each
/// cell commits to solid-or-empty from its center sample and extends
/// via uniform dedup. Limits worldgen cost without visibly changing
/// a smooth sphere.
const SDF_DETAIL_LEVELS: u32 = 4;

/// Build a spherical body node and return its `NodeId`. Caller is
/// responsible for placing it inside a parent (e.g., world root's
/// center slot) and bumping its refcount.
pub fn insert_spherical_body(
    lib: &mut NodeLibrary,
    inner_r: f32,
    outer_r: f32,
    depth: u32,
    sdf: &Planet,
) -> NodeId {
    debug_assert!(0.0 < inner_r && inner_r < outer_r && outer_r <= 0.5);

    // Build each face subtree, tagging only the root with
    // CubedSphereFace — internal nodes stay Cartesian for maximal
    // dedup (slot-index UVR convention is established at the root).
    let mut body_children = empty_children();
    for &face in &Face::ALL {
        let child = build_face_subtree(
            lib, face, inner_r, outer_r,
            -1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
            depth, depth.min(SDF_DETAIL_LEVELS), sdf,
        );
        let face_root = match child {
            Child::Node(id) => {
                let children = lib.get(id).expect("face root just inserted").children;
                lib.insert_with_kind(children, NodeKind::CubedSphereFace { face })
            }
            Child::Empty => lib.insert_with_kind(empty_children(), NodeKind::CubedSphereFace { face }),
            Child::Block(b) => {
                lib.insert_with_kind(uniform_children(Child::Block(b)), NodeKind::CubedSphereFace { face })
            }
            Child::EntityRef(_) => unreachable!("worldgen never emits entity refs"),
        };
        body_children[FACE_SLOTS[face as usize]] = Child::Node(face_root);
    }
    body_children[CORE_SLOT] = lib.build_uniform_subtree(sdf.core_block, depth);

    lib.insert_with_kind(body_children, NodeKind::CubedSphereBody { inner_r, outer_r })
}

/// Recursive build of one face subtree. Returns a `Child` so the
/// caller can collapse uniform subtrees. Emit cells by sampling the
/// SDF at the cell center under the equal-angle UVR-to-world map.
#[allow(clippy::too_many_arguments)]
fn build_face_subtree(
    lib: &mut NodeLibrary,
    face: Face,
    inner_r: f32,
    outer_r: f32,
    u_lo: f32, u_hi: f32,
    v_lo: f32, v_hi: f32,
    rn_lo: f32, rn_hi: f32,
    depth: u32,
    sdf_budget: u32,
    sdf: &Planet,
) -> Child {
    let body_size = 1.0f32; // sampled in body-local [0, 1)³
    let u_c = 0.5 * (u_lo + u_hi);
    let v_c = 0.5 * (v_lo + v_hi);
    let rn_c = 0.5 * (rn_lo + rn_hi);
    let p_center = face_space_to_body_point(
        face,
        (u_c + 1.0) * 0.5, (v_c + 1.0) * 0.5, rn_c,
        inner_r, outer_r, body_size,
    );
    let d_center = sdf.distance(p_center);
    let radial_half = 0.5 * (rn_hi - rn_lo) * (outer_r - inner_r);
    let lateral_half = 0.5 * (u_hi - u_lo).max(v_hi - v_lo) * outer_r;
    let cell_rad = (lateral_half * lateral_half + radial_half * radial_half).sqrt();

    if d_center > cell_rad {
        return if depth == 0 { Child::Empty } else { Child::Node(uniform_empty_chain(lib, depth)) };
    }
    if d_center < -cell_rad {
        let b = sdf.block_at(p_center);
        return if depth == 0 { Child::Block(b) } else { lib.build_uniform_subtree(b, depth) };
    }
    if depth == 0 {
        return if d_center < 0.0 { Child::Block(sdf.block_at(p_center)) } else { Child::Empty };
    }
    if sdf_budget == 0 {
        return if d_center < 0.0 {
            lib.build_uniform_subtree(sdf.block_at(p_center), depth)
        } else {
            Child::Node(uniform_empty_chain(lib, depth))
        };
    }

    let mut children = empty_children();
    let du = (u_hi - u_lo) / 3.0;
    let dv = (v_hi - v_lo) / 3.0;
    let drn = (rn_hi - rn_lo) / 3.0;
    for rs in 0..3 {
        for vs in 0..3 {
            for us in 0..3 {
                children[slot_index(us, vs, rs)] = build_face_subtree(
                    lib, face, inner_r, outer_r,
                    u_lo + du * us as f32, u_lo + du * (us + 1) as f32,
                    v_lo + dv * vs as f32, v_lo + dv * (vs + 1) as f32,
                    rn_lo + drn * rs as f32, rn_lo + drn * (rs + 1) as f32,
                    depth - 1, sdf_budget - 1, sdf,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

fn uniform_empty_chain(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert(uniform_children(Child::Node(id)));
    }
    id
}

/// Install a body into the world tree at `host_slots`, returning the
/// new world root and the body's path from the new root.
pub fn install_at_root_center(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    setup: &PlanetSetup,
) -> (NodeId, crate::world::anchor::Path) {
    let body_id = insert_spherical_body(lib, setup.inner_r, setup.outer_r, setup.depth, &setup.sdf);
    let host_slot = slot_index(1, 1, 1) as u8;
    let root_node = lib.get(world_root).expect("world root exists");
    let mut children = root_node.children;
    children[host_slot as usize] = Child::Node(body_id);
    let new_root = lib.insert(children);
    let mut body_path = crate::world::anchor::Path::root();
    body_path.push(host_slot);
    (new_root, body_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::palette::block;

    #[test]
    fn insert_body_creates_structured_children() {
        let mut lib = NodeLibrary::default();
        let sdf = Planet {
            center: [0.5; 3], radius: 0.30,
            noise_scale: 0.0, noise_freq: 1.0, noise_seed: 0,
            gravity: 0.0, influence_radius: 1.0,
            surface_block: block::GRASS, core_block: block::STONE,
        };
        let body = insert_spherical_body(&mut lib, 0.12, 0.45, 6, &sdf);
        let body_node = lib.get(body).unwrap();
        assert!(matches!(body_node.kind, NodeKind::CubedSphereBody { .. }));
        for &face in &Face::ALL {
            let slot = FACE_SLOTS[face as usize];
            match body_node.children[slot] {
                Child::Node(id) => {
                    let n = lib.get(id).unwrap();
                    assert!(matches!(n.kind, NodeKind::CubedSphereFace { face: f } if f == face));
                }
                _ => panic!("face slot {slot} not a Node"),
            }
        }
        match body_node.children[CORE_SLOT] {
            Child::Node(id) => {
                assert_eq!(lib.get(id).unwrap().uniform_type, block::STONE);
            }
            Child::Block(b) => assert_eq!(b, block::STONE),
            _ => panic!("core slot empty"),
        }
        for s in 0..27 {
            if s == CORE_SLOT || FACE_SLOTS.contains(&s) { continue; }
            assert!(matches!(body_node.children[s], Child::Empty));
        }
    }
}
