//! Cartesian stack-based DDA. CPU mirror of the shader's
//! `march_cartesian`. Walks the unified tree in XYZ slot order.

use super::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeKind, NodeLibrary};

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

                // TangentBlock dispatch — frame-local rotation around
                // (1.5, 1.5, 1.5). Mirrors the shader's march_cartesian.
                // NO world-absolute coordinates.
                if let NodeKind::TangentBlock { rotation } = child_node.kind {
                    // Scale: the slot extent (size parent_cell_size in
                    // parent frame) maps to the child's [0, 3)³ local
                    // frame, so scale = 3 / parent_cell_size.
                    let scale = 3.0 / parent_cell_size;
                    let lp_origin = [
                        (ray_origin[0] - child_origin[0]) * scale,
                        (ray_origin[1] - child_origin[1]) * scale,
                        (ray_origin[2] - child_origin[2]) * scale,
                    ];
                    let lp_dir = [
                        ray_dir[0] * scale,
                        ray_dir[1] * scale,
                        ray_dir[2] * scale,
                    ];
                    // Centred R^T around (1.5, 1.5, 1.5). Mirrors the
                    // shader's TB entry — rotating origin and direction
                    // by R^T about the same pivot is t-preserving, so
                    // the world parameter from the inner DDA matches
                    // the world ray. Direction-only here makes
                    // break/place targeting drift with camera position.
                    let centered = [lp_origin[0] - 1.5, lp_origin[1] - 1.5, lp_origin[2] - 1.5];
                    let rotated_origin = [
                        rotation[0][0] * centered[0] + rotation[0][1] * centered[1] + rotation[0][2] * centered[2],
                        rotation[1][0] * centered[0] + rotation[1][1] * centered[1] + rotation[1][2] * centered[2],
                        rotation[2][0] * centered[0] + rotation[2][1] * centered[1] + rotation[2][2] * centered[2],
                    ];
                    let local_origin = [
                        rotated_origin[0] + 1.5,
                        rotated_origin[1] + 1.5,
                        rotated_origin[2] + 1.5,
                    ];
                    let local_dir = [
                        rotation[0][0] * lp_dir[0] + rotation[0][1] * lp_dir[1] + rotation[0][2] * lp_dir[2],
                        rotation[1][0] * lp_dir[0] + rotation[1][1] * lp_dir[1] + rotation[1][2] * lp_dir[2],
                        rotation[2][0] * lp_dir[0] + rotation[2][1] * lp_dir[1] + rotation[2][2] * lp_dir[2],
                    ];
                    let sub_max_depth = max_depth.saturating_sub(depth as u32 + 1);
                    if let Some(sub_hit) = cpu_raycast_inner(
                        library, child_id, local_origin, local_dir, sub_max_depth,
                    ) {
                        let mut combined_path = path[..=depth].to_vec();
                        combined_path.extend(sub_hit.path);
                        return Some(HitInfo {
                            path: combined_path,
                            face: sub_hit.face,
                            t: sub_hit.t,
                            place_path: None,
                        });
                    }
                    advance_dda(&mut stack[depth], &step, &delta_dist, &mut normal_face);
                    continue;
                }

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
