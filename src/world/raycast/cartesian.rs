//! Cartesian stack-based DDA. CPU mirror of the shader's
//! `march_cartesian`. Walks the unified tree in XYZ slot order.

use super::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeKind, NodeLibrary};

/// 1/√3 — the AABB-fit shrink factor for an arbitrarily-oriented
/// cube to fit inside its parent's axis-aligned slot. Matches the
/// shader's TangentBlock dispatch.
const INV_SQRT_3: f32 = 0.577_350_3;

/// Convert a unit quaternion `(x, y, z, w)` to the columns of its
/// rotation matrix (east, normal, north). Matches the shader's
/// quat-to-basis math byte-for-byte so CPU and GPU dispatches
/// produce identical hits.
fn quat_to_basis(q: [f32; 4]) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;
    let east   = [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy)];
    let normal = [2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)];
    let north  = [2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy)];
    (east, normal, north)
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Apply R⁻¹ (basis as ROWS) to a vector around the origin. Used to
/// transform a ray into a TangentBlock's local rotated frame.
fn apply_inv_basis(v: [f32; 3], basis: ([f32; 3], [f32; 3], [f32; 3])) -> [f32; 3] {
    let (east, normal, north) = basis;
    [dot3(east, v), dot3(normal, v), dot3(north, v)]
}

/// Public entry: transform `(ray_origin, ray_dir)` into the cube's
/// local rotated frame using the cube's quaternion. The frame is
/// `[0, 3)³` with rotation around its centre `(1.5, 1.5, 1.5)`.
/// Used by `cpu_raycast_in_frame` when the active render frame IS a
/// TangentBlock — so the camera ray is in cube-local coords before
/// the inner DDA runs.
pub fn rotate_camera_into_tangent_frame(
    rotation: [f32; 4],
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let basis = quat_to_basis(rotation);
    let centre = [1.5_f32, 1.5, 1.5];
    let d = [ray_origin[0] - centre[0], ray_origin[1] - centre[1], ray_origin[2] - centre[2]];
    let local_d = apply_inv_basis(d, basis);
    let local_origin = [local_d[0] + centre[0], local_d[1] + centre[1], local_d[2] + centre[2]];
    let local_dir = apply_inv_basis(ray_dir, basis);
    (local_origin, local_dir)
}

/// Rotate a hit normal returned from `cpu_raycast_inner` (in the
/// cube's local frame, axis-aligned face direction) back to world
/// frame using R (basis as COLUMNS). Inverse of
/// `rotate_camera_into_tangent_frame`.
#[allow(dead_code)]
pub fn rotate_normal_out_of_tangent_frame(
    rotation: [f32; 4],
    local_normal: [f32; 3],
) -> [f32; 3] {
    let (east, normal, north) = quat_to_basis(rotation);
    [
        east[0] * local_normal[0] + normal[0] * local_normal[1] + north[0] * local_normal[2],
        east[1] * local_normal[0] + normal[1] * local_normal[1] + north[1] * local_normal[2],
        east[2] * local_normal[0] + normal[2] * local_normal[1] + north[2] * local_normal[2],
    ]
}

/// Stack frame for iterative DDA traversal.
pub(super) struct Frame {
    pub node_id: NodeId,
    pub cell: [i32; 3],
    pub side_dist: [f32; 3],
    pub node_origin: [f32; 3],
    pub cell_size: f32,
}

/// Stack-based Cartesian DDA over the unified tree. `max_depth`
/// caps how deep the walker descends; the deepest cell at that
/// depth is the hit granularity.
pub(super) fn cpu_raycast_inner(
    library: &NodeLibrary,
    root: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
) -> Option<HitInfo> {
    let inv_dir = [
        if ray_dir[0].abs() > 1e-8 { 1.0 / ray_dir[0] } else { 1e10 },
        if ray_dir[1].abs() > 1e-8 { 1.0 / ray_dir[1] } else { 1e10 },
        if ray_dir[2].abs() > 1e-8 { 1.0 / ray_dir[2] } else { 1e10 },
    ];
    let step = [
        if ray_dir[0] >= 0.0 { 1i32 } else { -1 },
        if ray_dir[1] >= 0.0 { 1i32 } else { -1 },
        if ray_dir[2] >= 0.0 { 1i32 } else { -1 },
    ];
    let delta_dist = [inv_dir[0].abs(), inv_dir[1].abs(), inv_dir[2].abs()];

    let (t_enter, t_exit) = ray_aabb(ray_origin, inv_dir, [0.0; 3], [3.0; 3]);
    if t_enter >= t_exit || t_exit < 0.0 {
        return None;
    }

    let t_start = t_enter.max(0.0) + 0.001;
    let entry_pos = [
        ray_origin[0] + ray_dir[0] * t_start,
        ray_origin[1] + ray_dir[1] * t_start,
        ray_origin[2] + ray_dir[2] * t_start,
    ];

    let initial_cell = [
        (entry_pos[0].floor() as i32).clamp(0, 2),
        (entry_pos[1].floor() as i32).clamp(0, 2),
        (entry_pos[2].floor() as i32).clamp(0, 2),
    ];
    let cell_f = [initial_cell[0] as f32, initial_cell[1] as f32, initial_cell[2] as f32];

    let mut stack: Vec<Frame> = Vec::with_capacity(max_depth as usize + 1);
    let mut path: Vec<(NodeId, usize)> = Vec::with_capacity(max_depth as usize + 1);

    stack.push(Frame {
        node_id: root,
        cell: initial_cell,
        side_dist: compute_initial_side_dist(&entry_pos, &cell_f, &inv_dir, &ray_dir, 1.0, &[0.0; 3]),
        node_origin: [0.0; 3],
        cell_size: 1.0,
    });

    let mut normal_face: u32 = 2;
    let mut iterations = 0u32;
    let max_iterations = (max_depth.max(1) * 4096).max(8192);

    loop {
        if iterations >= max_iterations || stack.is_empty() {
            break;
        }
        iterations += 1;

        let depth = stack.len() - 1;
        let cell = stack[depth].cell;

        if cell[0] < 0 || cell[0] > 2 || cell[1] < 0 || cell[1] > 2 || cell[2] < 0 || cell[2] > 2 {
            stack.pop();
            if path.len() > depth {
                path.truncate(depth);
            }
            if stack.is_empty() { break; }
            let d = stack.len() - 1;
            advance_dda(&mut stack[d], &step, &delta_dist, &mut normal_face);
            continue;
        }

        let slot = slot_index(cell[0] as usize, cell[1] as usize, cell[2] as usize);
        let node_id = stack[depth].node_id;
        let node = library.get(node_id)?;
        let child = node.children[slot];

        if path.len() > depth {
            path[depth] = (node_id, slot);
        } else {
            path.push((node_id, slot));
        }

        match child {
            Child::Empty => {
                advance_dda(&mut stack[depth], &step, &delta_dist, &mut normal_face);
            }
            // Entity cells aren't selectable by the cursor raycast —
            // editing entities goes through a different path that
            // descends into the entity's own voxel subtree.
            Child::EntityRef(_) => {
                advance_dda(&mut stack[depth], &step, &delta_dist, &mut normal_face);
            }
            Child::Block(_) => {
                return Some(HitInfo {
                    path: path.clone(),
                    face: normal_face,
                    t: cell_entry_t(&stack[depth], &ray_origin, &inv_dir),
                    place_path: None,
                });
            }
            Child::Node(child_id) => {
                let child_node = library.get(child_id)?;

                // Short-circuit fully-empty subtrees at any depth.
                // Without this, the DDA descends into uniform-air
                // nodes recursively, visiting O(3^depth) leaf cells
                // before escaping — easily exceeding the iteration
                // budget for deep carved cavities.
                if child_node.representative_block
                    == crate::world::tree::REPRESENTATIVE_EMPTY
                {
                    advance_dda(&mut stack[depth], &step, &delta_dist, &mut normal_face);
                    continue;
                }

                // TangentBlock dispatch — CPU mirror of the shader's
                // outside dispatch in `march_cartesian`. When the parent
                // DDA descends into a TangentBlock child, transform the
                // ray into the cube's local rotated frame, recursively
                // raycast the cube interior, and prepend our path to
                // the resulting hit. Without this, the cursor reports
                // axis-aligned cells inside the cube while the shader
                // renders rotated cells — cursor and visual disagree.
                if let NodeKind::TangentBlock { rotation } = child_node.kind {
                    let parent_origin = stack[depth].node_origin;
                    let parent_cell_size = stack[depth].cell_size;
                    let cube_centre = [
                        parent_origin[0] + (cell[0] as f32 + 0.5) * parent_cell_size,
                        parent_origin[1] + (cell[1] as f32 + 0.5) * parent_cell_size,
                        parent_origin[2] + (cell[2] as f32 + 0.5) * parent_cell_size,
                    ];
                    let cube_side = parent_cell_size * INV_SQRT_3;
                    let scale = 3.0 / cube_side;
                    let basis = quat_to_basis(rotation);
                    let d_origin = [
                        ray_origin[0] - cube_centre[0],
                        ray_origin[1] - cube_centre[1],
                        ray_origin[2] - cube_centre[2],
                    ];
                    let local_d = apply_inv_basis(d_origin, basis);
                    let local_origin = [
                        local_d[0] * scale + 1.5,
                        local_d[1] * scale + 1.5,
                        local_d[2] * scale + 1.5,
                    ];
                    let inv_dir_local = apply_inv_basis(ray_dir, basis);
                    let local_dir = [
                        inv_dir_local[0] * scale,
                        inv_dir_local[1] * scale,
                        inv_dir_local[2] * scale,
                    ];

                    let inner_max = max_depth.saturating_sub(depth as u32 + 1);
                    if let Some(sub_hit) = cpu_raycast_inner(
                        library, child_id, local_origin, local_dir, inner_max,
                    ) {
                        // Prepend the path up through this TangentBlock
                        // to the inner hit's path. local_t == world_t
                        // because the dir scale absorbs into the t
                        // parameterisation (same as the shader).
                        let mut full_path = path[..=depth].to_vec();
                        full_path.extend(sub_hit.path);
                        let full_place_path = sub_hit.place_path.map(|pp| {
                            let mut combined = path[..=depth].to_vec();
                            combined.extend(pp);
                            combined
                        });
                        return Some(HitInfo {
                            path: full_path,
                            // Face is in cube-local axis-aligned coords,
                            // matching how break/place treats faces of any
                            // cell — local to the cell's parent frame.
                            face: sub_hit.face,
                            t: sub_hit.t,
                            place_path: full_place_path,
                        });
                    }
                    // Miss inside cube — advance the parent DDA past
                    // this cell, same as for a regular Cartesian Node
                    // miss. (`advance_dda` updates side_dist + normal_face.)
                    advance_dda(&mut stack[depth], &step, &delta_dist, &mut normal_face);
                    continue;
                }

                if (depth as u32 + 1) >= max_depth {
                    return Some(HitInfo {
                        path: path.clone(),
                        face: normal_face,
                        t: cell_entry_t(&stack[depth], &ray_origin, &inv_dir),
                        place_path: None,
                    });
                }

                let parent_origin = stack[depth].node_origin;
                let parent_cell_size = stack[depth].cell_size;
                let child_origin = [
                    parent_origin[0] + cell[0] as f32 * parent_cell_size,
                    parent_origin[1] + cell[1] as f32 * parent_cell_size,
                    parent_origin[2] + cell[2] as f32 * parent_cell_size,
                ];
                let child_cell_size = parent_cell_size / 3.0;

                let child_max = [
                    child_origin[0] + parent_cell_size,
                    child_origin[1] + parent_cell_size,
                    child_origin[2] + parent_cell_size,
                ];
                let (ct_enter, _) = ray_aabb(ray_origin, inv_dir, child_origin, child_max);
                let ct_start = ct_enter.max(0.0) + 0.0001 * child_cell_size;
                let child_entry = [
                    ray_origin[0] + ray_dir[0] * ct_start,
                    ray_origin[1] + ray_dir[1] * ct_start,
                    ray_origin[2] + ray_dir[2] * ct_start,
                ];
                let local_entry = [
                    (child_entry[0] - child_origin[0]) / child_cell_size,
                    (child_entry[1] - child_origin[1]) / child_cell_size,
                    (child_entry[2] - child_origin[2]) / child_cell_size,
                ];
                let child_cell = [
                    (local_entry[0].floor() as i32).clamp(0, 2),
                    (local_entry[1].floor() as i32).clamp(0, 2),
                    (local_entry[2].floor() as i32).clamp(0, 2),
                ];
                let lc = [child_cell[0] as f32, child_cell[1] as f32, child_cell[2] as f32];

                stack.push(Frame {
                    node_id: child_id,
                    cell: child_cell,
                    side_dist: compute_initial_side_dist(
                        &ray_origin, &lc, &inv_dir, &ray_dir,
                        child_cell_size, &child_origin,
                    ),
                    node_origin: child_origin,
                    cell_size: child_cell_size,
                });
            }
        }
    }

    None
}

pub(super) fn advance_dda(
    frame: &mut Frame,
    step: &[i32; 3],
    delta_dist: &[f32; 3],
    normal_face: &mut u32,
) {
    if frame.side_dist[0] < frame.side_dist[1] && frame.side_dist[0] < frame.side_dist[2] {
        frame.cell[0] += step[0];
        frame.side_dist[0] += delta_dist[0] * frame.cell_size;
        *normal_face = if step[0] > 0 { 1 } else { 0 };
    } else if frame.side_dist[1] < frame.side_dist[2] {
        frame.cell[1] += step[1];
        frame.side_dist[1] += delta_dist[1] * frame.cell_size;
        *normal_face = if step[1] > 0 { 3 } else { 2 };
    } else {
        frame.cell[2] += step[2];
        frame.side_dist[2] += delta_dist[2] * frame.cell_size;
        *normal_face = if step[2] > 0 { 5 } else { 4 };
    }
}

pub(super) fn ray_aabb(
    origin: [f32; 3],
    inv_dir: [f32; 3],
    bmin: [f32; 3],
    bmax: [f32; 3],
) -> (f32, f32) {
    let t1 = [
        (bmin[0] - origin[0]) * inv_dir[0],
        (bmin[1] - origin[1]) * inv_dir[1],
        (bmin[2] - origin[2]) * inv_dir[2],
    ];
    let t2 = [
        (bmax[0] - origin[0]) * inv_dir[0],
        (bmax[1] - origin[1]) * inv_dir[1],
        (bmax[2] - origin[2]) * inv_dir[2],
    ];
    let t_enter = t1[0].min(t2[0]).max(t1[1].min(t2[1])).max(t1[2].min(t2[2]));
    let t_exit = t1[0].max(t2[0]).min(t1[1].max(t2[1])).min(t1[2].max(t2[2]));
    (t_enter, t_exit)
}

pub(super) fn compute_initial_side_dist(
    ray_origin: &[f32; 3],
    cell: &[f32; 3],
    inv_dir: &[f32; 3],
    ray_dir: &[f32; 3],
    cell_size: f32,
    node_origin: &[f32; 3],
) -> [f32; 3] {
    [
        if ray_dir[0] >= 0.0 {
            (node_origin[0] + (cell[0] + 1.0) * cell_size - ray_origin[0]) * inv_dir[0]
        } else {
            (node_origin[0] + cell[0] * cell_size - ray_origin[0]) * inv_dir[0]
        },
        if ray_dir[1] >= 0.0 {
            (node_origin[1] + (cell[1] + 1.0) * cell_size - ray_origin[1]) * inv_dir[1]
        } else {
            (node_origin[1] + cell[1] * cell_size - ray_origin[1]) * inv_dir[1]
        },
        if ray_dir[2] >= 0.0 {
            (node_origin[2] + (cell[2] + 1.0) * cell_size - ray_origin[2]) * inv_dir[2]
        } else {
            (node_origin[2] + cell[2] * cell_size - ray_origin[2]) * inv_dir[2]
        },
    ]
}

pub(super) fn cell_entry_t(frame: &Frame, ray_origin: &[f32; 3], inv_dir: &[f32; 3]) -> f32 {
    let cell_min = [
        frame.node_origin[0] + frame.cell[0] as f32 * frame.cell_size,
        frame.node_origin[1] + frame.cell[1] as f32 * frame.cell_size,
        frame.node_origin[2] + frame.cell[2] as f32 * frame.cell_size,
    ];
    let cell_max = [
        cell_min[0] + frame.cell_size,
        cell_min[1] + frame.cell_size,
        cell_min[2] + frame.cell_size,
    ];
    let (t_enter, _) = ray_aabb(*ray_origin, *inv_dir, cell_min, cell_max);
    t_enter
}
