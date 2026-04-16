//! Sphere-body DDA. CPU mirror of the shader's `sphere_in_cell` /
//! `sphere_in_face_window`. Step-based march — accuracy is tuned for
//! cursor targeting, not rendering fidelity.

use super::{HitInfo, MAX_FACE_DEPTH};
use crate::world::cubesphere::FACE_SLOTS;
use crate::world::cubesphere_local;
use crate::world::sdf;
use crate::world::tree::{slot_index, Child, NodeId, NodeLibrary, EMPTY_NODE, UNIFORM_EMPTY, UNIFORM_MIXED};

/// Restricts the sphere march to a sub-region of a single face.
/// `Some(...)` = the sphere-frame caller's face-window; `None` = a
/// whole-body march issued from a Cartesian DDA that descended into
/// a body cell.
pub(super) struct FaceBounds {
    pub face: u32,
    pub u_min: f32,
    pub v_min: f32,
    pub r_min: f32,
    pub size: f32,
}

/// Unified sphere-in-body raycast. When `bounds == None`, the march
/// spans the entire outer sphere and accepts a hit on any face. When
/// `bounds == Some(...)`, it breaks out the moment the ray leaves
/// the given face-window.
///
/// On a block terminal, returns a `HitInfo` whose `path` extends
/// through `(body_id, face_slot) + face-subtree descent`, so the
/// generic `propagate_edit` pipeline can break/place that cell. When
/// the ray traverses empty cells before the terminal, the last such
/// cell's full path is reported in `place_path` — face-subtree slots
/// are `(u, v, r)` not `(x, y, z)`, so the usual xyz-delta placement
/// would land in the wrong cell (or into solid rock).
pub(super) fn cs_raycast_in_body(
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
    bounds: Option<FaceBounds>,
) -> Option<HitInfo> {
    // The ray-sphere intersection (disc = b²-c) assumes |ray_dir| = 1.
    // Callers in the pop loop divide by 3.0 per ribbon level, so the
    // incoming ray_dir may not be unit.
    let ray_dir = sdf::normalize(ray_dir);
    let window_scale = bounds.as_ref().map_or(1.0, |b| b.size);

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
    let mut step_world = shell * window_scale * 0.33;
    let mut prev_place_path: Option<Vec<(NodeId, usize)>> = None;

    for _ in 0..8_000usize {
        if t >= t_exit { break; }
        let p = sdf::add(ray_origin, sdf::scale(ray_dir, t));
        let local = sdf::sub(p, cs_center);
        let r = sdf::length(local);
        if r > cs_outer {
            t += eps * 8.0;
            continue;
        }
        if r < cs_inner { break; }

        let p_body = [
            p[0] - body_origin[0],
            p[1] - body_origin[1],
            p[2] - body_origin[2],
        ];
        let face_point = cubesphere_local::body_point_to_face_space(
            p_body, inner_r_local, outer_r_local, body_size,
        )?;

        // Face filter + window clip for the bounded case.
        let (un, vn, rn) = if let Some(ref b) = bounds {
            if face_point.face as u32 != b.face { break; }
            let un_abs = face_point.un;
            let vn_abs = face_point.vn;
            let rn_abs = face_point.rn;
            if un_abs < b.u_min || un_abs >= b.u_min + b.size
                || vn_abs < b.v_min || vn_abs >= b.v_min + b.size
                || rn_abs < b.r_min || rn_abs >= b.r_min + b.size
            {
                break;
            }
            (
                ((un_abs - b.u_min) / b.size).clamp(0.0, 0.9999999),
                ((vn_abs - b.v_min) / b.size).clamp(0.0, 0.9999999),
                ((rn_abs - b.r_min) / b.size).clamp(0.0, 0.9999999),
            )
        } else {
            (face_point.un, face_point.vn, face_point.rn)
        };

        let face = face_point.face;
        let face_slot = FACE_SLOTS[face as usize];
        let body_node = library.get(body_id)?;
        let face_root_id = match body_node.children[face_slot] {
            Child::Node(id) => id,
            _ => {
                if bounds.is_some() {
                    return None;
                }
                t += step_world;
                continue;
            }
        };

        let walk = walk_face_subtree_with_path(library, face_root_id, un, vn, rn, max_face_depth);
        if let Some((block_id, term_depth, mut face_path)) = walk {
            if block_id != 0 {
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
            // Empty cell — record its full path as the placement target.
            // Step refinement uses the cell's nominal width, floored so
            // a huge empty region doesn't eat the step budget one
            // sub-cell at a time.
            let mut empty_full = ancestor_path.to_vec();
            empty_full.push((body_id, face_slot));
            empty_full.append(&mut face_path);
            prev_place_path = Some(empty_full);
            let cells_d = 3.0_f32.powi(term_depth as i32);
            let nominal = shell * window_scale / cells_d * 0.33;
            let coarse_floor = shell * window_scale * 0.01;
            step_world = nominal.max(coarse_floor).max(eps * 4.0);
        }
        t += step_world;
    }

    None
}

/// CPU walker mirror of the shader's `walk_face_subtree`. Descends
/// the face subtree along `(un, vn, rn)`, returning
/// `(block_id, term_depth, path)`. After empty terminals above
/// `max_depth`, synthesizes `EMPTY_NODE`-tagged entries so the
/// placement path has uniform depth — without that, block size on
/// placement depends on where the empty chain happens to end (tiny
/// above SDF detail, huge over uniform-empty regions).
pub(super) fn walk_face_subtree_with_path(
    library: &NodeLibrary,
    face_root_id: NodeId,
    un_in: f32, vn_in: f32, rn_in: f32,
    max_depth: u32,
) -> Option<(u8, u32, Vec<(NodeId, usize)>)> {
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
            let slot = slot_index(us, vs, rs);
            path.push((EMPTY_NODE, slot));
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
            Child::Empty => {
                let sub_un = un * 3.0 - us as f32;
                let sub_vn = vn * 3.0 - vs as f32;
                let sub_rn = rn * 3.0 - rs as f32;
                pad_to_limit(&mut path, sub_un, sub_vn, sub_rn, d);
                return Some((0, limit, path));
            }
            Child::Block(b) => return Some((b, d, path)),
            Child::Node(nid) => {
                // Descend to `limit` (= cs_edit_depth). Do NOT stop at
                // uniform-content subtrees: below SDF_DETAIL_LEVELS the
                // subtree collapses to a uniform chain (stone or empty),
                // and stopping there would cap the editable cell size
                // at the SDF detail level regardless of zoom — "edits
                // stop at layer 15." The packer still flattens these
                // chains for rendering, but editing a sub-cell forces
                // re-materialization on the next upload.
                if d == limit {
                    let Some(child_node) = library.get(nid) else {
                        return Some((0, d, path));
                    };
                    let block = match child_node.uniform_type {
                        UNIFORM_MIXED => {
                            let rep = child_node.representative_block;
                            if rep < 255 { rep } else { 0 }
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
