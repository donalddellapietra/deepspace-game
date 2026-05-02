//! Cartesian stack-based DDA. CPU mirror of the shader's
//! `march_cartesian`. Walks the unified tree in XYZ slot order.

use super::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeKind, NodeLibrary};

/// Stack frame for iterative DDA traversal. Each level carries its
/// own ray + derived DDA state so a `Rotated45Y` descent can rotate
/// the ray and `cur_node_origin` once at the boundary, then walk
/// the rotated subtree with ordinary cartesian DDA. On pop, the
/// parent's frame still holds its un-rotated ray verbatim — no
/// inverse-T arithmetic needed.
pub(super) struct Frame {
    pub node_id: NodeId,
    pub cell: [i32; 3],
    pub side_dist: [f32; 3],
    pub node_origin: [f32; 3],
    pub cell_size: f32,
    pub ray_origin: [f32; 3],
    pub ray_dir: [f32; 3],
    pub inv_dir: [f32; 3],
    pub step: [i32; 3],
    pub delta_dist: [f32; 3],
}

/// Stack-based Cartesian DDA over the unified tree. `max_depth`
/// caps how deep the walker descends; the deepest cell at that
/// depth is the hit granularity.
///
/// On every `tag == Node` descent the walker reads the child's
/// `NodeKind` and, for `Rotated45Y`, transforms the ray by `T(p) =
/// (p.x − p.z, p.y, p.x + p.z)` (45° Y rotation + √2 XZ stretch)
/// relative to the cell center. The transformed ray plus a
/// `node_origin` reset to `[0, 0, 0]` makes the rotated child fill
/// `[0, parent_cell_size]³` axis-aligned, so the rest of the DDA
/// runs unchanged in the rotated frame. Cartesian descents skip
/// the transform — one branch on data, no shadow code path.
pub(super) fn cpu_raycast_inner(
    library: &NodeLibrary,
    root: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
) -> Option<HitInfo> {
    let initial = make_frame(root, ray_origin, ray_dir, [0.0; 3], 1.0)?;

    let mut stack: Vec<Frame> = Vec::with_capacity(max_depth as usize + 1);
    let mut path: Vec<(NodeId, usize)> = Vec::with_capacity(max_depth as usize + 1);
    stack.push(initial);

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
            let (step_p, delta_p) = (stack[d].step, stack[d].delta_dist);
            advance_dda(&mut stack[d], &step_p, &delta_p, &mut normal_face);
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
            Child::Empty | Child::EntityRef(_) => {
                let (step_e, delta_e) = (stack[depth].step, stack[depth].delta_dist);
                advance_dda(&mut stack[depth], &step_e, &delta_e, &mut normal_face);
            }
            Child::Block(_) => {
                let f = &stack[depth];
                let t = cell_entry_t(f, &f.ray_origin, &f.inv_dir);
                return Some(HitInfo {
                    path: path.clone(),
                    face: normal_face,
                    t,
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
                    let (step_e, delta_e) = (stack[depth].step, stack[depth].delta_dist);
                    advance_dda(&mut stack[depth], &step_e, &delta_e, &mut normal_face);
                    continue;
                }

                if (depth as u32 + 1) >= max_depth {
                    let f = &stack[depth];
                    let t = cell_entry_t(f, &f.ray_origin, &f.inv_dir);
                    return Some(HitInfo {
                        path: path.clone(),
                        face: normal_face,
                        t,
                        place_path: None,
                    });
                }

                let parent = &stack[depth];
                let parent_origin = parent.node_origin;
                let parent_cell_size = parent.cell_size;
                let parent_ray_origin = parent.ray_origin;
                let parent_ray_dir = parent.ray_dir;
                let parent_inv_dir = parent.inv_dir;
                let child_origin = [
                    parent_origin[0] + cell[0] as f32 * parent_cell_size,
                    parent_origin[1] + cell[1] as f32 * parent_cell_size,
                    parent_origin[2] + cell[2] as f32 * parent_cell_size,
                ];
                let child_cell_size = parent_cell_size / 3.0;

                // Per-descent rotation: read the child's NodeKind. For
                // `Rotated45Y`, transform the parent's ray into the
                // rotated child's local frame. Cartesian (and any
                // other non-rotated kind) takes the cheap path —
                // copy the parent ray and `node_origin = child_origin`.
                let (
                    child_ray_origin,
                    child_ray_dir,
                    child_node_origin,
                ) = if matches!(child_node.kind, NodeKind::Rotated45Y) {
                    let cell_center = [
                        child_origin[0] + parent_cell_size * 0.5,
                        child_origin[1] + parent_cell_size * 0.5,
                        child_origin[2] + parent_cell_size * 0.5,
                    ];
                    let pc = [
                        parent_ray_origin[0] - cell_center[0],
                        parent_ray_origin[1] - cell_center[1],
                        parent_ray_origin[2] - cell_center[2],
                    ];
                    let pc_t = [pc[0] - pc[2], pc[1], pc[0] + pc[2]];
                    let dc_t = [
                        parent_ray_dir[0] - parent_ray_dir[2],
                        parent_ray_dir[1],
                        parent_ray_dir[0] + parent_ray_dir[2],
                    ];
                    let new_origin = [
                        pc_t[0] + parent_cell_size * 0.5,
                        pc_t[1] + parent_cell_size * 0.5,
                        pc_t[2] + parent_cell_size * 0.5,
                    ];
                    (new_origin, dc_t, [0.0f32; 3])
                } else {
                    (parent_ray_origin, parent_ray_dir, child_origin)
                };

                // Find the entry cell within the child node using the
                // child's frame ray (rotated for Rotated45Y, identical
                // to parent for cartesian).
                let child_inv = if matches!(child_node.kind, NodeKind::Rotated45Y) {
                    [
                        if child_ray_dir[0].abs() > 1e-8 { 1.0 / child_ray_dir[0] } else { 1e10 },
                        if child_ray_dir[1].abs() > 1e-8 { 1.0 / child_ray_dir[1] } else { 1e10 },
                        if child_ray_dir[2].abs() > 1e-8 { 1.0 / child_ray_dir[2] } else { 1e10 },
                    ]
                } else {
                    parent_inv_dir
                };
                let child_box_min = child_node_origin;
                let child_box_max = [
                    child_box_min[0] + parent_cell_size,
                    child_box_min[1] + parent_cell_size,
                    child_box_min[2] + parent_cell_size,
                ];
                let (ct_enter, _) = ray_aabb(
                    child_ray_origin, child_inv, child_box_min, child_box_max,
                );
                let ct_start = ct_enter.max(0.0) + 0.0001 * child_cell_size;
                let child_entry = [
                    child_ray_origin[0] + child_ray_dir[0] * ct_start,
                    child_ray_origin[1] + child_ray_dir[1] * ct_start,
                    child_ray_origin[2] + child_ray_dir[2] * ct_start,
                ];
                let local_entry = [
                    (child_entry[0] - child_node_origin[0]) / child_cell_size,
                    (child_entry[1] - child_node_origin[1]) / child_cell_size,
                    (child_entry[2] - child_node_origin[2]) / child_cell_size,
                ];
                let child_cell = [
                    (local_entry[0].floor() as i32).clamp(0, 2),
                    (local_entry[1].floor() as i32).clamp(0, 2),
                    (local_entry[2].floor() as i32).clamp(0, 2),
                ];
                let lc = [child_cell[0] as f32, child_cell[1] as f32, child_cell[2] as f32];

                let child_step = [
                    if child_ray_dir[0] >= 0.0 { 1i32 } else { -1 },
                    if child_ray_dir[1] >= 0.0 { 1i32 } else { -1 },
                    if child_ray_dir[2] >= 0.0 { 1i32 } else { -1 },
                ];
                let child_delta = [
                    child_inv[0].abs(),
                    child_inv[1].abs(),
                    child_inv[2].abs(),
                ];

                stack.push(Frame {
                    node_id: child_id,
                    cell: child_cell,
                    side_dist: compute_initial_side_dist(
                        &child_ray_origin, &lc, &child_inv, &child_ray_dir,
                        child_cell_size, &child_node_origin,
                    ),
                    node_origin: child_node_origin,
                    cell_size: child_cell_size,
                    ray_origin: child_ray_origin,
                    ray_dir: child_ray_dir,
                    inv_dir: child_inv,
                    step: child_step,
                    delta_dist: child_delta,
                });
            }
        }
    }

    None
}

/// Build the initial root-frame state for `cpu_raycast_inner`. Returns
/// `None` if the ray misses the world's `[0, 3)³` root box entirely.
fn make_frame(
    root: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    node_origin: [f32; 3],
    cell_size: f32,
) -> Option<Frame> {
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
    let bmax = [
        node_origin[0] + 3.0 * cell_size,
        node_origin[1] + 3.0 * cell_size,
        node_origin[2] + 3.0 * cell_size,
    ];
    let (t_enter, t_exit) = ray_aabb(ray_origin, inv_dir, node_origin, bmax);
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
        (((entry_pos[0] - node_origin[0]) / cell_size).floor() as i32).clamp(0, 2),
        (((entry_pos[1] - node_origin[1]) / cell_size).floor() as i32).clamp(0, 2),
        (((entry_pos[2] - node_origin[2]) / cell_size).floor() as i32).clamp(0, 2),
    ];
    let cell_f = [initial_cell[0] as f32, initial_cell[1] as f32, initial_cell[2] as f32];
    Some(Frame {
        node_id: root,
        cell: initial_cell,
        side_dist: compute_initial_side_dist(
            &entry_pos, &cell_f, &inv_dir, &ray_dir, cell_size, &node_origin,
        ),
        node_origin,
        cell_size,
        ray_origin,
        ray_dir,
        inv_dir,
        step,
        delta_dist,
    })
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
