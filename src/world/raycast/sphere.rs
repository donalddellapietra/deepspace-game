//! CPU sphere-body raycast. Mirror of the shader's
//! `march_sphere_body` (in `assets/shaders/sphere.wgsl`). One
//! function, one walker, one cell convention — called whenever the
//! cartesian DDA descends into a `CubedSphereBody` cell, with that
//! cell's origin and size in the current render-frame coordinates.
//!
//! Nothing here hardcodes a body-absolute constant: body position
//! and size are parameters, so the same code works whether the body
//! is at the world root or nested deeper via the ribbon-pop chain.

use super::{HitInfo, MAX_FACE_DEPTH};
use crate::world::cubesphere::FACE_SLOTS;
use crate::world::cubesphere_local;
use crate::world::sdf;
use crate::world::tree::{
    slot_index, Child, NodeId, NodeLibrary, EMPTY_NODE, REPRESENTATIVE_EMPTY, UNIFORM_EMPTY,
    UNIFORM_MIXED,
};

/// CPU sphere raycast inside one `CubedSphereBody` cell. Returns
/// `Some(hit)` on a block terminal; `None` on miss (so the caller's
/// cartesian DDA continues past this body cell).
///
/// `body_origin`, `body_size` are in render-frame-local coords.
/// `ancestor_path` is the slot chain from the caller's frame root
/// to the cell containing this body — the returned `HitInfo::path`
/// extends `ancestor_path + (body_id, face_slot) + face-subtree
/// descent`, mirroring what the generic edit pipeline expects.
///
/// On a block hit, `place_path` is populated with the last empty
/// cell's path so `place_block` can put a cell adjacent to the hit
/// without needing the cartesian xyz-delta logic (which doesn't
/// apply to face-subtree slots, which are `(u, v, r)`).
pub fn cs_raycast_in_body(
    library: &NodeLibrary,
    body_id: NodeId,
    body_origin: [f32; 3],
    body_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    ancestor_path: &[(NodeId, usize)],
    max_face_depth: u32,
) -> Option<HitInfo> {
    // Ribbon-pop shrinks ray magnitude by 1/3 per level; renormalize
    // so the ray-sphere disc computation uses |dir|=1.
    let ray_dir = sdf::normalize(ray_dir);

    let cs_center = [
        body_origin[0] + body_size * 0.5,
        body_origin[1] + body_size * 0.5,
        body_origin[2] + body_size * 0.5,
    ];
    let cs_outer = outer_r_local * body_size;
    let cs_inner = inner_r_local * body_size;
    let shell = cs_outer - cs_inner;
    if shell <= 0.0 { return None; }

    let oc = sdf::sub(ray_origin, cs_center);
    let b = sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c;
    if disc <= 0.0 { return None; }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return None; }

    let eps = (shell * 1e-5).max(1e-7);
    let mut t = t_enter + eps;
    // Step floor: at least 1/3 of the nominal face-subtree cell
    // width. Prevents stall when the ray grazes a cell near its edge
    // and each step lands inside the same empty cell.
    let mut step_world = shell * 0.33;
    let mut prev_place_path: Option<Vec<(NodeId, usize)>> = None;

    for _ in 0..8_000usize {
        if t >= t_exit { break; }
        let p = sdf::add(ray_origin, sdf::scale(ray_dir, t));
        let local = sdf::sub(p, cs_center);
        let r = sdf::length(local);
        if r > cs_outer { t += eps * 8.0; continue; }
        if r < cs_inner { break; }

        // Project into the body cell's local frame for face lookup.
        let p_body = [
            p[0] - body_origin[0],
            p[1] - body_origin[1],
            p[2] - body_origin[2],
        ];
        let fp = cubesphere_local::body_point_to_face_space(
            p_body, inner_r_local, outer_r_local, body_size,
        )?;

        let face_slot = FACE_SLOTS[fp.face as usize];
        let body_node = library.get(body_id)?;
        let face_root_id = match body_node.children[face_slot] {
            Child::Node(id) => id,
            _ => {
                // Face subtree missing (can only happen for degenerate
                // bodies): continue the sphere march through empty
                // face territory.
                t += step_world;
                continue;
            }
        };

        let walk = walk_face_subtree_with_path(
            library, face_root_id, fp.un, fp.vn, fp.rn, max_face_depth,
        );
        if let Some((block_id, term_depth, mut face_path)) = walk {
            if block_id != 0 {
                let mut full = ancestor_path.to_vec();
                full.push((body_id, face_slot));
                full.append(&mut face_path);
                return Some(HitInfo {
                    path: full,
                    face: 4, // unused for sphere hits — place_path drives placement
                    t,
                    place_path: prev_place_path,
                });
            }
            // Empty cell — record full path as placement target.
            let mut empty_full = ancestor_path.to_vec();
            empty_full.push((body_id, face_slot));
            empty_full.append(&mut face_path);
            prev_place_path = Some(empty_full);
            // Step by the cell's nominal radial extent.
            let cells_d = 3.0_f32.powi(term_depth as i32);
            let nominal = shell / cells_d * 0.33;
            let floor_step = shell * 0.01;
            step_world = nominal.max(floor_step).max(eps * 4.0);
        }
        t += step_world;
    }

    None
}

/// CPU mirror of the shader's `walk_face_subtree`. Descends the face
/// subtree along `(un, vn, rn)` returning `(block_id, term_depth,
/// path_slots)`. Below SDF detail, the face subtree collapses to
/// uniform chains that we must descend into to preserve edit
/// precision — breaking a small cell inside a uniform stone region
/// materializes the expected sub-cell.
///
/// Empty terminals above `max_depth` are padded with
/// `EMPTY_NODE`-tagged entries so placement path depth is uniform:
/// without padding, block size on placement would depend on where
/// the empty chain ends.
pub fn walk_face_subtree_with_path(
    library: &NodeLibrary,
    face_root_id: NodeId,
    un_in: f32, vn_in: f32, rn_in: f32,
    max_depth: u32,
) -> Option<(u16, u32, Vec<(NodeId, usize)>)> {
    let mut node_id = face_root_id;
    let mut un = un_in.clamp(0.0, 0.9999999);
    let mut vn = vn_in.clamp(0.0, 0.9999999);
    let mut rn = rn_in.clamp(0.0, 0.9999999);
    let mut path: Vec<(NodeId, usize)> = Vec::new();
    let limit = max_depth.min(MAX_FACE_DEPTH);

    let pad_to_limit = |path: &mut Vec<(NodeId, usize)>,
                        mut un: f32, mut vn: f32, mut rn: f32,
                        from_d: u32| {
        for _ in from_d..limit {
            let us = ((un * 3.0) as usize).min(2);
            let vs = ((vn * 3.0) as usize).min(2);
            let rs = ((rn * 3.0) as usize).min(2);
            path.push((EMPTY_NODE, slot_index(us, vs, rs)));
            un = un * 3.0 - us as f32;
            vn = vn * 3.0 - vs as f32;
            rn = rn * 3.0 - rs as f32;
        }
    };

    for d in 1u32..=limit {
        let node = library.get(node_id)?;
        let us = ((un * 3.0) as usize).min(2);
        let vs = ((vn * 3.0) as usize).min(2);
        let rs = ((rn * 3.0) as usize).min(2);
        let slot = slot_index(us, vs, rs);
        path.push((node_id, slot));
        match node.children[slot] {
            // EntityRef shouldn't appear in a sphere face subtree
            // (they're only in ephemeral scene overlays), but treat
            // as empty if it does so the sphere march can continue.
            Child::Empty | Child::EntityRef(_) => {
                let sun = un * 3.0 - us as f32;
                let svn = vn * 3.0 - vs as f32;
                let srn = rn * 3.0 - rs as f32;
                pad_to_limit(&mut path, sun, svn, srn, d);
                return Some((0, limit, path));
            }
            Child::Block(b) => return Some((b, d, path)),
            Child::Node(nid) => {
                if d == limit {
                    // Hit max depth on a Node child: materialize its
                    // effective block for LOD-terminal presentation.
                    let Some(child) = library.get(nid) else {
                        return Some((0, d, path));
                    };
                    let block = match child.uniform_type {
                        UNIFORM_MIXED => {
                            let rep = child.representative_block;
                            if rep != REPRESENTATIVE_EMPTY { rep } else { 0 }
                        }
                        UNIFORM_EMPTY => 0,
                        b => b,
                    };
                    return Some((block, d, path));
                }
                node_id = nid;
                un = un * 3.0 - us as f32;
                vn = vn * 3.0 - vs as f32;
                rn = rn * 3.0 - rs as f32;
            }
        }
    }
    Some((0, limit, path))
}
