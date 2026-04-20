//! Sphere-shell CPU raycast. One function, one path.
//!
//! `cs_raycast` steps the ray through the outer sphere's interior,
//! finds the (face, u, v, r) of each sample, walks the corresponding
//! face subtree to the configured terminal depth, and reports the
//! hit. A `FaceWindow`, when provided, restricts the march to one
//! face's sub-window — used when the render frame root lives inside
//! a face subtree.

use super::{HitInfo, MAX_FACE_DEPTH};
use crate::world::cubesphere::{body_point_to_face_space, FACE_SLOTS};
use crate::world::sdf;
use crate::world::tree::{
    slot_index, Child, NodeId, NodeLibrary, EMPTY_NODE, REPRESENTATIVE_EMPTY,
    UNIFORM_EMPTY, UNIFORM_MIXED,
};

/// Sentinel returned by `walk_face_subtree` when the terminal cell
/// is empty. Using a u16 sentinel (not `0`) because palette index 0
/// is a real block type (`block::STONE`).
const EMPTY_CELL: u16 = REPRESENTATIVE_EMPTY;

/// Restricts `cs_raycast` to a sub-region of a single face. Used when
/// the render frame root lives deep inside a face subtree: the face-
/// window bounds tell the raycaster which absolute UVR region is
/// currently being marched.
#[derive(Copy, Clone, Debug)]
pub struct FaceWindow {
    pub face: u32,
    pub u_min: f32,
    pub v_min: f32,
    pub r_min: f32,
    pub size: f32,
}

/// Sphere DDA in a single body cell.
///
/// Parameters are in the caller's frame. `ray_origin`/`ray_dir` are
/// raw (not renormalised). `body_origin`/`body_size` specify the
/// body cell's extent. `inner_r_local`/`outer_r_local` are in the
/// body cell's local `[0, 1)` frame (so `radius_world = r_local *
/// body_size`).
///
/// `ancestor_path` is prepended to every returned path so the hit
/// info's path is absolute (rooted at the caller's world root).
///
/// `window`, when `Some`, restricts hits to that face's sub-window.
/// When `None`, any face is acceptable (whole-body march).
pub(super) fn cs_raycast(
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
    window: Option<FaceWindow>,
) -> Option<HitInfo> {
    let ray_dir = sdf::normalize(ray_dir);
    let center = [
        body_origin[0] + body_size * 0.5,
        body_origin[1] + body_size * 0.5,
        body_origin[2] + body_size * 0.5,
    ];
    let outer = outer_r_local * body_size;
    let inner = inner_r_local * body_size;
    let shell = outer - inner;
    if shell <= 0.0 { return None; }

    // Ray–outer-sphere entry.
    let oc = sdf::sub(ray_origin, center);
    let b = sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - outer * outer;
    let disc = b * b - c;
    if disc <= 0.0 { return None; }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return None; }

    let window_scale = window.map_or(1.0, |w| w.size);
    let eps = (shell * 1e-5).max(1e-7);
    let mut t = t_enter + eps;
    let mut step_world = shell * window_scale * 0.33;
    let mut prev_place_path: Option<Vec<(NodeId, usize)>> = None;

    for _ in 0..8_000usize {
        if t >= t_exit { break; }
        let p = sdf::add(ray_origin, sdf::scale(ray_dir, t));
        let local = sdf::sub(p, center);
        let r = sdf::length(local);
        if r > outer { t += eps * 8.0; continue; }
        if r < inner { break; }

        let p_body = [
            p[0] - body_origin[0],
            p[1] - body_origin[1],
            p[2] - body_origin[2],
        ];
        let fp = body_point_to_face_space(p_body, inner_r_local, outer_r_local, body_size)?;

        // Window filter — skip samples outside the active face
        // subtree sub-region, if one is configured.
        let (un, vn, rn) = if let Some(w) = window {
            if fp.face as u32 != w.face { break; }
            if fp.un < w.u_min || fp.un >= w.u_min + w.size ||
               fp.vn < w.v_min || fp.vn >= w.v_min + w.size ||
               fp.rn < w.r_min || fp.rn >= w.r_min + w.size {
                break;
            }
            (
                ((fp.un - w.u_min) / w.size).clamp(0.0, 0.9999999),
                ((fp.vn - w.v_min) / w.size).clamp(0.0, 0.9999999),
                ((fp.rn - w.r_min) / w.size).clamp(0.0, 0.9999999),
            )
        } else {
            (fp.un, fp.vn, fp.rn)
        };

        let face_slot = FACE_SLOTS[fp.face as usize];
        let body_node = library.get(body_id)?;
        let face_root_id = match body_node.children[face_slot] {
            Child::Node(id) => id,
            _ => {
                if window.is_some() { return None; }
                t += step_world;
                continue;
            }
        };

        let (block_id, term_depth, mut face_path) =
            walk_face_subtree(library, face_root_id, un, vn, rn, max_face_depth);

        if block_id != EMPTY_CELL {
            let mut full_path = ancestor_path.to_vec();
            full_path.push((body_id, face_slot));
            full_path.append(&mut face_path);
            return Some(HitInfo {
                path: full_path,
                face: 4, // unused for sphere hits; place_path drives placement
                t,
                place_path: prev_place_path,
            });
        }

        // Empty cell: record full path as placement target, adapt
        // step size to the cell's nominal width so a huge empty
        // region doesn't eat the step budget one sub-cell at a time.
        let mut empty_full = ancestor_path.to_vec();
        empty_full.push((body_id, face_slot));
        empty_full.append(&mut face_path);
        prev_place_path = Some(empty_full);
        let cells = 3.0_f32.powi(term_depth as i32);
        let nominal = shell * window_scale / cells * 0.33;
        let floor = shell * window_scale * 0.01;
        step_world = nominal.max(floor).max(eps * 4.0);
        t += step_world;
    }

    None
}

/// Descend a face subtree along `(un, vn, rn)` to the terminal cell,
/// returning `(block, term_depth, path)`. After empty terminals above
/// `max_depth`, synthesises `EMPTY_NODE` path entries so placement
/// depth matches `cs_edit_depth` regardless of where the empty chain
/// was truncated.
fn walk_face_subtree(
    library: &NodeLibrary,
    face_root_id: NodeId,
    un_in: f32, vn_in: f32, rn_in: f32,
    max_depth: u32,
) -> (u16, u32, Vec<(NodeId, usize)>) {
    let limit = max_depth.min(MAX_FACE_DEPTH);
    let mut node_id = face_root_id;
    let mut un = un_in.clamp(0.0, 0.9999999);
    let mut vn = vn_in.clamp(0.0, 0.9999999);
    let mut rn = rn_in.clamp(0.0, 0.9999999);
    let mut path: Vec<(NodeId, usize)> = Vec::with_capacity(limit as usize);

    for d in 1u32..=limit {
        let Some(node) = library.get(node_id) else {
            return (EMPTY_CELL, d.saturating_sub(1), path);
        };
        let us = ((un * 3.0) as usize).min(2);
        let vs = ((vn * 3.0) as usize).min(2);
        let rs = ((rn * 3.0) as usize).min(2);
        let slot = slot_index(us, vs, rs);
        path.push((node_id, slot));
        match node.children[slot] {
            Child::Empty => {
                // Synthesise a placement chain down to `limit` so
                // editing places a cell of uniform size regardless
                // of how deep the tree really represents the empty.
                let mut sub_un = un * 3.0 - us as f32;
                let mut sub_vn = vn * 3.0 - vs as f32;
                let mut sub_rn = rn * 3.0 - rs as f32;
                for _ in d..limit {
                    let us2 = ((sub_un * 3.0) as usize).min(2);
                    let vs2 = ((sub_vn * 3.0) as usize).min(2);
                    let rs2 = ((sub_rn * 3.0) as usize).min(2);
                    path.push((EMPTY_NODE, slot_index(us2, vs2, rs2)));
                    sub_un = sub_un * 3.0 - us2 as f32;
                    sub_vn = sub_vn * 3.0 - vs2 as f32;
                    sub_rn = sub_rn * 3.0 - rs2 as f32;
                }
                return (EMPTY_CELL, limit, path);
            }
            Child::Block(b) => return (b, d, path),
            Child::EntityRef(_) => return (EMPTY_CELL, d, path),
            Child::Node(nid) => {
                if d == limit {
                    // LOD-terminal: flatten uniform subtrees to a
                    // block; mixed subtrees fall back to their
                    // representative (or EMPTY_CELL if all-empty).
                    let Some(child) = library.get(nid) else { return (EMPTY_CELL, d, path); };
                    let bt = match child.uniform_type {
                        UNIFORM_MIXED => {
                            let rep = child.representative_block;
                            if rep == REPRESENTATIVE_EMPTY { EMPTY_CELL } else { rep }
                        }
                        UNIFORM_EMPTY => EMPTY_CELL,
                        b => b,
                    };
                    return (bt, d, path);
                }
                node_id = nid;
                un = un * 3.0 - us as f32;
                vn = vn * 3.0 - vs as f32;
                rn = rn * 3.0 - rs as f32;
            }
        }
    }
    (0, limit, path)
}
