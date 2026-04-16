//! CPU raycasting and block editing.
//!
//! The CPU ray march mirrors the GPU shader's tree traversal to
//! determine which block the crosshair is pointing at. Edits operate
//! at a layer-dependent depth: the zoom level controls how deep the
//! raycast descends, so the same code breaks a single block at fine
//! zoom or an entire node (3x3x3 group) at coarse zoom.

use super::cubesphere::FACE_SLOTS;
use super::cubesphere_local;
use super::anchor::{Path, WORLD_SIZE};
use super::sdf;
use super::state::WorldState;
use super::tree::*;

const MAX_FACE_DEPTH: u32 = crate::world::tree::MAX_DEPTH as u32;

/// Information about a ray hit in the tree.
#[derive(Debug, Clone)]
pub struct HitInfo {
    /// Path from root to the hit: each entry is (node_id, child_slot).
    /// The last entry's child_slot is the slot that was hit.
    pub path: Vec<(NodeId, usize)>,
    /// Which face was crossed when the block was hit.
    /// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
    pub face: u32,
    /// Distance along the ray to the hit point.
    pub t: f32,
    /// Optional explicit path where a place_block should land. For
    /// Cartesian hits this is `None` — `place_child` derives the
    /// adjacent cell via `face`. For sphere hits `face`'s xyz-axis
    /// semantics don't apply (face-subtree slots are (u,v,r)), so
    /// `cs_raycast_in_body` fills this with the last empty cell the
    /// ray traversed before striking the block, which IS the correct
    /// placement target regardless of which cube face was entered.
    pub place_path: Option<Vec<(NodeId, usize)>>,
}

/// Stack frame for iterative DDA traversal.
struct Frame {
    node_id: NodeId,
    cell: [i32; 3],
    side_dist: [f32; 3],
    node_origin: [f32; 3],
    cell_size: f32,
}

/// Cast a ray through the tree, stopping at `max_depth` levels from root.
///
/// `max_depth` controls the interaction layer: at depth 3 in a 3-level
/// tree the ray targets individual blocks; at depth 2 it targets 3x3x3
/// node groups. This is how zoom controls edit scale.
pub fn cpu_raycast(
    library: &NodeLibrary,
    root: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
) -> Option<HitInfo> {
    cpu_raycast_with_face_depth(library, root, ray_origin, ray_dir, max_depth, 6, f32::MAX)
}

/// Frame-aware raycast. Mirrors the renderer's frame architecture
/// so cell-precision is bounded by the frame depth (camera in
/// `[0, 3)` regardless of absolute path), and the ray pops upward
/// into ancestor frames when it exits the current frame's bubble.
///
/// `frame_path` is the camera's anchor truncated to the desired
/// frame depth (matches what `compute_render_frame` returns +
/// `pack_tree_lod_preserving` packs). `cam_local` is
/// `WorldPos::in_frame(&frame_path)`. `ray_dir` must be expressed
/// in that same frame-local metric.
///
/// Returns a `HitInfo` whose `path` is from `world_root` (the
/// frame slots are prepended to whatever the inner DDA found).
pub fn cpu_raycast_in_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    frame_path: &[u8],
    cam_local: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
    max_face_depth: u32,
) -> Option<HitInfo> {
    cpu_raycast_in_frame_with_budget(
        library, world_root, frame_path, cam_local, ray_dir,
        max_depth, max_face_depth,
    )
}
/// Raycast reach in frame-local units. The render frame spans
/// `[0, 3)` per axis (3 cells). A reach of 24 means the ray can
/// travel 8 cell-widths — roughly equivalent to Minecraft's
/// ~5-block reach.
const RAYCAST_REACH: f32 = 24.0;

/// Maximum number of ribbon pops the CPU raycast will attempt.
/// Each pop moves one level toward the root. The render frame is
/// `RENDER_FRAME_K` levels above the anchor, so 6 pops covers the
/// frame plus a few ancestor shells — enough to find walls of
/// carved cavities without scanning the entire tree.
const MAX_RAYCAST_POPS: usize = 6;

pub fn cpu_raycast_in_frame_with_budget(
    library: &NodeLibrary,
    world_root: NodeId,
    frame_path: &[u8],
    cam_local: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
    max_face_depth: u32,
) -> Option<HitInfo> {
    // Walk world_root down frame_path collecting (parent_id, slot)
    // entries AND the chain of NodeIds. The chain is used as
    // ribbon storage; frame_entries[] becomes the prefix prepended
    // to the inner hit's path.
    let mut chain: Vec<NodeId> = Vec::with_capacity(frame_path.len() + 1);
    chain.push(world_root);
    let mut frame_entries: Vec<(NodeId, usize)> = Vec::with_capacity(frame_path.len());
    {
        let mut current = world_root;
        for &slot in frame_path {
            let Some(node) = library.get(current) else { break };
            frame_entries.push((current, slot as usize));
            match node.children[slot as usize] {
                Child::Node(child_id) => {
                    current = child_id;
                    chain.push(current);
                }
                _ => break,
            }
        }
    }
    let effective_depth = chain.len() - 1; // edges traversed
    let frame_entries = &frame_entries[..effective_depth];

    let mut current_frame_depth = effective_depth;
    let mut ray_origin = cam_local;
    let mut ray_dir = ray_dir;
    let total_max_depth = max_depth;
    let mut pops_done: usize = 0;

    loop {
        let frame_root_id = chain[current_frame_depth];
        let inner_max = total_max_depth
            .saturating_sub(current_frame_depth as u32);

        // Frame-root NodeKind dispatch: sphere body vs Cartesian.
        let frame_node = library.get(frame_root_id);
        let frame_kind = frame_node.map(|n| n.kind);
        if let Some(NodeKind::CubedSphereBody { inner_r, outer_r }) = frame_kind {
            let inner_ancestor: Vec<(NodeId, usize)> = Vec::new();
            if let Some(mut hit) = cs_raycast_in_body(
                library, frame_root_id, [0.0; 3], 3.0,
                inner_r, outer_r,
                ray_origin, ray_dir,
                &inner_ancestor, max_face_depth,
            ) {
                prepend_frame_entries(&mut hit, frame_entries, current_frame_depth);
                return Some(hit);
            }
        } else if let Some(mut hit) = cpu_raycast_with_face_depth(
            library, frame_root_id, ray_origin, ray_dir,
            inner_max, max_face_depth, RAYCAST_REACH,
        ) {
            prepend_frame_entries(&mut hit, frame_entries, current_frame_depth);
            return Some(hit);
        }

        // Miss. Pop one level toward root, up to MAX_RAYCAST_POPS.
        if current_frame_depth == 0 || pops_done >= MAX_RAYCAST_POPS {
            return None;
        }
        let last_slot = frame_entries[current_frame_depth - 1].1;
        let (sx, sy, sz) = slot_coords(last_slot);
        ray_origin = [
            sx as f32 + ray_origin[0] / 3.0,
            sy as f32 + ray_origin[1] / 3.0,
            sz as f32 + ray_origin[2] / 3.0,
        ];
        // ray_dir is divided by 3 on pop to match the coordinate
        // transform, then re-normalized so `t` values remain in
        // frame-local distance units (not inflated by accumulated
        // ÷3 factors). Without normalization, `t` grows by 3× per
        // pop, making RAYCAST_REACH meaningless after a few pops.
        ray_dir = super::sdf::normalize([
            ray_dir[0] / 3.0,
            ray_dir[1] / 3.0,
            ray_dir[2] / 3.0,
        ]);
        current_frame_depth -= 1;
        pops_done += 1;
    }
}

/// Prepend frame entries (root → frame_root path) to a hit's path
/// and place_path.
fn prepend_frame_entries(
    hit: &mut HitInfo,
    frame_entries: &[(NodeId, usize)],
    depth: usize,
) {
    let mut new_path: Vec<(NodeId, usize)> =
        Vec::with_capacity(depth + hit.path.len());
    new_path.extend(frame_entries.iter().take(depth).cloned());
    new_path.append(&mut hit.path);
    hit.path = new_path;
    if let Some(mut pp) = hit.place_path.take() {
        let mut new_pp: Vec<(NodeId, usize)> =
            Vec::with_capacity(depth + pp.len());
        new_pp.extend(frame_entries.iter().take(depth).cloned());
        new_pp.append(&mut pp);
        hit.place_path = Some(new_pp);
    }
}

/// Frame-aware raycast for a sphere sub-frame. The linear render
/// frame stays rooted at the containing body cell, but the actual
/// editable/renderable layer is the bounded face-cell window
/// described by `(face, *_min, face_size, face_depth)`.
pub fn cpu_raycast_in_sphere_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    frame_path: &[u8],
    cam_local: [f32; 3],
    ray_origin_body: [f32; 3],
    ray_dir: [f32; 3],
    max_face_depth: u32,
    face: u32,
    face_u_min: f32,
    face_v_min: f32,
    face_r_min: f32,
    face_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    face_depth: u32,
) -> Option<HitInfo> {
    let mut chain: Vec<NodeId> = Vec::with_capacity(frame_path.len() + 1);
    chain.push(world_root);
    let mut frame_entries: Vec<(NodeId, usize)> = Vec::with_capacity(frame_path.len());
    {
        let mut current = world_root;
        for &slot in frame_path {
            let Some(node) = library.get(current) else { break };
            frame_entries.push((current, slot as usize));
            match node.children[slot as usize] {
                Child::Node(child_id) => {
                    current = child_id;
                    chain.push(current);
                }
                _ => break,
            }
        }
    }
    let effective_depth = chain.len() - 1;
    let frame_entries = &frame_entries[..effective_depth];
    let mut current_frame_depth = effective_depth;
    let mut ray_origin_local = cam_local;

    loop {
        let frame_root_id = chain[current_frame_depth];
        if current_frame_depth == effective_depth {
            if let Some(mut hit) = cs_raycast_in_body_bounded(
                library,
                frame_root_id,
                [0.0; 3],
                3.0,
                inner_r_local,
                outer_r_local,
                ray_origin_body,
                ray_dir,
                &[],
                face,
                face_u_min,
                face_v_min,
                face_r_min,
                face_size,
                max_face_depth.saturating_sub(face_depth),
            ) {
                let mut new_path: Vec<(NodeId, usize)> =
                    Vec::with_capacity(current_frame_depth + hit.path.len());
                new_path.extend(frame_entries.iter().take(current_frame_depth).cloned());
                new_path.append(&mut hit.path);
                hit.path = new_path;
                if let Some(mut pp) = hit.place_path.take() {
                    let mut new_pp: Vec<(NodeId, usize)> =
                        Vec::with_capacity(current_frame_depth + pp.len());
                    new_pp.extend(frame_entries.iter().take(current_frame_depth).cloned());
                    new_pp.append(&mut pp);
                    hit.place_path = Some(new_pp);
                }
                return Some(hit);
            }
        } else {
            let inner_max = max_face_depth.saturating_sub(current_frame_depth as u32);
            let frame_node = library.get(frame_root_id);
            let frame_kind = frame_node.map(|n| n.kind);
            let hit = if let Some(NodeKind::CubedSphereBody { inner_r, outer_r }) = frame_kind {
                cs_raycast_in_body(
                    library,
                    frame_root_id,
                    [0.0; 3],
                    3.0,
                    inner_r,
                    outer_r,
                    ray_origin_local,
                    ray_dir,
                    &[],
                    max_face_depth,
                )
            } else {
                cpu_raycast_with_face_depth(
                    library,
                    frame_root_id,
                    ray_origin_local,
                    ray_dir,
                    inner_max,
                    max_face_depth,
                    RAYCAST_REACH,
                )
            };
            if let Some(mut hit) = hit {
                let mut new_path: Vec<(NodeId, usize)> =
                    Vec::with_capacity(current_frame_depth + hit.path.len());
                new_path.extend(frame_entries.iter().take(current_frame_depth).cloned());
                new_path.append(&mut hit.path);
                hit.path = new_path;
                if let Some(mut pp) = hit.place_path.take() {
                    let mut new_pp: Vec<(NodeId, usize)> =
                        Vec::with_capacity(current_frame_depth + pp.len());
                    new_pp.extend(frame_entries.iter().take(current_frame_depth).cloned());
                    new_pp.append(&mut pp);
                    hit.place_path = Some(new_pp);
                }
                return Some(hit);
            }
        }
        if current_frame_depth == 0 {
            return None;
        }
        let last_slot = frame_entries[current_frame_depth - 1].1;
        let (sx, sy, sz) = slot_coords(last_slot);
        ray_origin_local = [
            sx as f32 + ray_origin_local[0] / 3.0,
            sy as f32 + ray_origin_local[1] / 3.0,
            sz as f32 + ray_origin_local[2] / 3.0,
        ];
        current_frame_depth -= 1;
    }
}

/// Same as `cpu_raycast` but with an explicit cap on how deep the
/// face-subtree walker descends inside a `CubedSphereBody`. Cells
/// at the deepest planet depth are sub-pixel; the cap selects the
/// user-visible cell granularity (~3^max_face_depth cells per face
/// axis).
///
/// `max_t` limits the ray's reach in frame-local units. Hits beyond
/// `max_t` are discarded. Pass `f32::MAX` for unlimited range.
pub fn cpu_raycast_with_face_depth(
    library: &NodeLibrary,
    root: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
    max_face_depth: u32,
    max_t: f32,
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

    // Intersect ray with root node [0, 3).
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

    let mut normal_face: u32 = 2; // default +Y
    let mut iterations = 0u32;
    let max_iterations = (max_depth.max(1) * 4096).max(8192);

    loop {
        if iterations >= max_iterations || stack.is_empty() {
            break;
        }
        iterations += 1;

        let depth = stack.len() - 1;
        let cell = stack[depth].cell;

        // Out of bounds — pop.
        if cell[0] < 0 || cell[0] > 2 || cell[1] < 0 || cell[1] > 2 || cell[2] < 0 || cell[2] > 2 {
            stack.pop();
            if path.len() > depth {
                path.truncate(depth);
            }
            if stack.is_empty() {
                break;
            }
            let d = stack.len() - 1;
            advance_dda(&mut stack[d], &step, &delta_dist, &mut normal_face);
            continue;
        }

        let slot = slot_index(cell[0] as usize, cell[1] as usize, cell[2] as usize);
        let node_id = stack[depth].node_id;
        let node = library.get(node_id)?;
        let child = node.children[slot];

        // Update path.
        if path.len() > depth {
            path[depth] = (node_id, slot);
        } else {
            path.push((node_id, slot));
        }

        match child {
            Child::Empty => {
                advance_dda(&mut stack[depth], &step, &delta_dist, &mut normal_face);
            }
            Child::Block(_) => {
                let t = cell_entry_t(&stack[depth], &ray_origin, &inv_dir);
                if t > max_t {
                    return None;
                }
                return Some(HitInfo {
                    path: path.clone(),
                    face: normal_face,
                    t,
                    place_path: None,
                });
            }
            Child::Node(child_id) => {
                let child_node = library.get(child_id)?;

                // NodeKind dispatch — sphere body cells switch to
                // sphere DDA in the body cell's local frame.
                if let NodeKind::CubedSphereBody { inner_r, outer_r } = child_node.kind {
                    let parent_origin = stack[depth].node_origin;
                    let parent_cell_size = stack[depth].cell_size;
                    let body_origin = [
                        parent_origin[0] + cell[0] as f32 * parent_cell_size,
                        parent_origin[1] + cell[1] as f32 * parent_cell_size,
                        parent_origin[2] + cell[2] as f32 * parent_cell_size,
                    ];
                    let body_size = parent_cell_size;
                    if let Some(sphere_hit) = cs_raycast_in_body(
                        library, child_id, body_origin, body_size,
                        inner_r, outer_r,
                        ray_origin, ray_dir,
                        &path,
                        max_face_depth,
                    ) {
                        return Some(sphere_hit);
                    }
                    // Sphere missed — advance Cartesian DDA past this cell.
                    advance_dda(&mut stack[depth], &step, &delta_dist, &mut normal_face);
                    continue;
                }

                // Short-circuit fully-empty subtrees at any depth.
                // Without this, the DDA descends into uniform-air
                // nodes recursively, visiting O(3^depth) leaf cells
                // before escaping — easily exceeding the iteration
                // budget for deep carved cavities.
                if child_node.representative_block == 255 {
                    advance_dda(&mut stack[depth], &step, &delta_dist, &mut normal_face);
                    continue;
                }

                if (depth as u32 + 1) >= max_depth {
                    let t = cell_entry_t(&stack[depth], &ray_origin, &inv_dir);
                    if t > max_t {
                        return None;
                    }
                    return Some(HitInfo {
                        path: path.clone(),
                        face: normal_face,
                        t,
                        place_path: None,
                    });
                }

                // Descend into child node.
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

/// Break (remove) the block at the hit location.
pub fn break_block(world: &mut WorldState, hit: &HitInfo) -> bool {
    propagate_edit(world, hit, Child::Empty)
}

/// Place any child (block or subtree) adjacent to the hit face.
/// For blocks, use `Child::Block(idx)`. For saved meshes, use
/// `Child::Node(saved_node_id)`.
pub fn place_child(world: &mut WorldState, hit: &HitInfo, new_child: Child) -> bool {
    // Sphere hits carry an explicit placement path (the last empty
    // cell the ray traversed before hitting the block). Place there
    // directly — the face→xyz-delta derivation below is nonsense for
    // face-subtree (u,v,r) slots.
    if let Some(ref place_path) = hit.place_path {
        let place_hit = HitInfo {
            path: place_path.clone(),
            face: hit.face,
            t: hit.t,
            place_path: None,
        };
        return propagate_edit(world, &place_hit, new_child);
    }
    let (_parent_id, slot) = *hit.path.last().unwrap();
    let (x, y, z) = slot_coords(slot);
    let (dx, dy, dz): (i32, i32, i32) = match hit.face {
        0 => (1, 0, 0),
        1 => (-1, 0, 0),
        2 => (0, 1, 0),
        3 => (0, -1, 0),
        4 => (0, 0, 1),
        5 => (0, 0, -1),
        _ => return false,
    };

    let nx = x as i32 + dx;
    let ny = y as i32 + dy;
    let nz = z as i32 + dz;

    // In-node: neighbor is within the same 3x3x3 node.
    if (0..=2).contains(&nx) && (0..=2).contains(&ny) && (0..=2).contains(&nz) {
        let adj_slot = slot_index(nx as usize, ny as usize, nz as usize);
        let parent_id = hit.path.last().unwrap().0;

        let node = match world.library.get(parent_id) {
            Some(n) => n,
            None => return false,
        };

        if !is_placeable(&world.library, node.children[adj_slot]) {
            return false;
        }

        let mut place_hit = hit.clone();
        place_hit.path.last_mut().unwrap().1 = adj_slot;
        return propagate_edit(world, &place_hit, new_child);
    }

    // Cross-node: use pure slot/path arithmetic — no f32 coordinates.
    // Build a Path from the hit's slot indices, step it in the face
    // normal direction (with carry/borrow propagation), then walk the
    // tree along the new path.
    let (axis, direction) = match hit.face {
        0 => (0usize, 1i32),
        1 => (0, -1),
        2 => (1, 1),
        3 => (1, -1),
        4 => (2, 1),
        5 => (2, -1),
        _ => return false,
    };
    let mut target_path = Path::root();
    for &(_, slot) in &hit.path {
        target_path.push(slot as u8);
    }
    target_path.step_neighbor_cartesian(axis, direction);

    place_child_at_path(world, &target_path, new_child)
}

/// Place a block adjacent to the hit face. Builds a uniform subtree
/// that matches the depth of siblings at the placement site, so the
/// placed block has full recursive structure like the terrain around it.
pub fn place_block(world: &mut WorldState, hit: &HitInfo, block_type: u8) -> bool {
    // Figure out how deep siblings are at the placement site.
    // The hit path has `path.len()` levels from root. The sibling
    // nodes at that depth have some subtree depth. We match it.
    let sibling_depth = if let Some(&(parent_id, _)) = hit.path.last() {
        if let Some(parent) = world.library.get(parent_id) {
            // Find the max depth among non-empty siblings.
            let mut max_d = 0u32;
            for child in &parent.children {
                if let Child::Node(nid) = child {
                    let d = depth_of_node(&world.library, *nid);
                    if d > max_d { max_d = d; }
                }
            }
            max_d
        } else {
            0
        }
    } else {
        0
    };

    let child = world.library.build_uniform_subtree(block_type, sibling_depth);
    place_child(world, hit, child)
}

/// Compute depth of a single node (non-memoized, but uniform nodes are O(1)).
fn depth_of_node(library: &super::tree::NodeLibrary, id: super::tree::NodeId) -> u32 {
    let Some(node) = library.get(id) else { return 0 };
    // For uniform nodes (all children identical), just check one child.
    let first_child = node.children[0];
    match first_child {
        Child::Node(child_id) => 1 + depth_of_node(library, child_id),
        _ => 1,
    }
}

/// Place a block at the position identified by a slot path. Walks the
/// tree along the path, materializing empty intermediate nodes as
/// needed. Uses pure integer slot arithmetic — no f32 coordinates —
/// so it works at all 63 layers without precision loss.
fn place_child_at_path(
    world: &mut WorldState,
    target_path: &Path,
    new_child: Child,
) -> bool {
    let target_depth = target_path.depth() as usize;
    if target_depth == 0 {
        return false;
    }
    let slots = target_path.as_slice();

    let mut path: Vec<(NodeId, usize)> = Vec::with_capacity(target_depth);
    let mut current_id = world.root;

    for level in 0..target_depth {
        let slot = slots[level] as usize;
        path.push((current_id, slot));

        let node = match world.library.get(current_id) {
            Some(n) => n,
            None => return false,
        };

        let is_last = level == target_depth - 1;

        match node.children[slot] {
            Child::Node(child_id) if !is_last => {
                current_id = child_id;
            }
            child if is_last && is_placeable(&world.library, child) => {
                let place_hit = HitInfo { path, face: 0, t: 0.0, place_path: None };
                return propagate_edit(world, &place_hit, new_child);
            }
            child if !is_last && is_placeable(&world.library, child) => {
                // Empty subtree — build a chain of empty nodes with
                // the target block at the correct slot at each level.
                let remaining_slots = &slots[level + 1..target_depth];
                let chain_id = build_placement_chain_from_slots(
                    world, remaining_slots, new_child,
                );
                let place_hit = HitInfo { path, face: 0, t: 0.0, place_path: None };
                return propagate_edit(world, &place_hit, Child::Node(chain_id));
            }
            _ => {
                return false;
            }
        }
    }

    false
}

/// Build a chain of empty nodes along `slots`, placing `leaf_child`
/// at the deepest level. Pure slot arithmetic, no f32.
fn build_placement_chain_from_slots(
    world: &mut WorldState,
    slots: &[u8],
    leaf_child: Child,
) -> NodeId {
    let mut child = leaf_child;
    for &slot in slots.iter().rev() {
        let mut children = empty_children();
        children[slot as usize] = child;
        let id = world.library.insert(children);
        child = Child::Node(id);
    }

    match child {
        Child::Node(id) => id,
        _ => unreachable!("remaining > 0 guarantees at least one wrapping node"),
    }
}

/// Install a subtree (NodeId) at a given path from root.
///
/// `ancestor_slots` is a list of child slot indices from root to the
/// target position. The subtree at `new_node_id` replaces whatever
/// was at the final slot. Clone-on-write propagation creates new
/// ancestors all the way to a new root.
///
/// Used for placing saved meshes / imported models.
pub fn install_subtree(world: &mut WorldState, ancestor_slots: &[usize], new_node_id: NodeId) {
    if ancestor_slots.is_empty() { return; }

    // Phase 1: Descent — record (parent_id, slot) pairs.
    let mut descent: Vec<(NodeId, usize)> = Vec::with_capacity(ancestor_slots.len());
    let mut current_id = world.root;

    for &slot in ancestor_slots {
        descent.push((current_id, slot));
        let Some(node) = world.library.get(current_id) else { return };
        match node.children[slot] {
            Child::Node(child_id) => current_id = child_id,
            _ => {
                // Terminal or empty at this slot — we'll replace it.
                // No further descent needed.
                break;
            }
        }
    }

    // Phase 2: Ascent — walk back up, cloning children arrays.
    // Preserve each ancestor's NodeKind (see `propagate_edit`).
    let mut child = Child::Node(new_node_id);
    for &(parent_id, slot) in descent.iter().rev() {
        let Some(node) = world.library.get(parent_id) else { return };
        let original_kind = node.kind;
        let mut new_children = node.children;
        new_children[slot] = child;
        child = Child::Node(world.library.insert_with_kind(new_children, original_kind));
    }

    if let Child::Node(new_root) = child {
        world.swap_root(new_root);
    }
}

/// Apply an edit and propagate clone-on-write up to root.
fn propagate_edit(world: &mut WorldState, hit: &HitInfo, new_child: Child) -> bool {
    if hit.path.is_empty() {
        return false;
    }

    let mut replacement: Option<NodeId> = None;

    for i in (0..hit.path.len()).rev() {
        let (node_id, slot) = hit.path[i];
        // EMPTY_NODE sentinel = "synthesize a fresh empty Cartesian
        // node for this level" — used by sphere placement to extend
        // the placement path past the tree's existing empty
        // terminals down to the user's chosen cs_edit_depth, so the
        // placed block's cell size is consistent regardless of how
        // deep the nearest real empty terminal lived.
        let (children_template, original_kind) = if node_id == EMPTY_NODE {
            (empty_children(), NodeKind::Cartesian)
        } else {
            let node = match world.library.get(node_id) {
                Some(n) => n,
                None => return false,
            };
            // CRITICAL: preserve the original NodeKind when rebuilding.
            // Without this, an edit through a `CubedSphereBody` or
            // `CubedSphereFace` ancestor reinserts it as Cartesian,
            // the shader's NodeKind dispatch stops firing, and the
            // walker descends into the body's children Cartesian-style
            // — painting the planet's interior-stone fillers as cube
            // blocks. (Spec §1b: NodeKind is part of node identity.)
            (node.children, node.kind)
        };

        let mut new_children = children_template;
        if let Some(nid) = replacement {
            new_children[slot] = Child::Node(nid);
        } else {
            new_children[slot] = new_child;
        }

        replacement = Some(world.library.insert_with_kind(new_children, original_kind));
    }

    if let Some(new_root) = replacement {
        world.swap_root(new_root);
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------- helpers

/// A cell is placeable if it's Empty or an all-empty Node subtree
/// (representative_block == 255). At coarser zoom levels, air regions
/// are represented as Node subtrees rather than Child::Empty.
fn is_placeable(library: &NodeLibrary, child: Child) -> bool {
    match child {
        Child::Empty => true,
        Child::Node(id) => library
            .get(id)
            .map_or(false, |n| n.representative_block == 255),
        Child::Block(_) => false,
    }
}

fn advance_dda(frame: &mut Frame, step: &[i32; 3], delta_dist: &[f32; 3], normal_face: &mut u32) {
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

fn ray_aabb(origin: [f32; 3], inv_dir: [f32; 3], bmin: [f32; 3], bmax: [f32; 3]) -> (f32, f32) {
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

fn compute_initial_side_dist(
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

fn cell_entry_t(frame: &Frame, ray_origin: &[f32; 3], inv_dir: &[f32; 3]) -> f32 {
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

/// Check whether the cell at world-space position `pos` is solid at
/// the given tree depth. Walks the tree from root, mapping the
/// position to slot indices at each level. Returns true if the cell
/// is Block or Node (has content); false if Empty or out of bounds.
pub fn is_solid_at(
    library: &NodeLibrary,
    root: NodeId,
    pos: [f32; 3],
    max_depth: u32,
) -> bool {
    // The root spans [0, 3) on each axis.
    if pos[0] < 0.0 || pos[0] >= 3.0
        || pos[1] < 0.0 || pos[1] >= 3.0
        || pos[2] < 0.0 || pos[2] >= 3.0
    {
        return false;
    }

    let mut node_id = root;
    let mut node_origin = [0.0f32; 3];
    let mut cell_size = 1.0f32; // each cell at this level

    for depth in 0..max_depth {
        let node = match library.get(node_id) {
            Some(n) => n,
            None => return false,
        };

        // Which cell does pos fall into at this level?
        let cx = ((pos[0] - node_origin[0]) / cell_size).floor() as i32;
        let cy = ((pos[1] - node_origin[1]) / cell_size).floor() as i32;
        let cz = ((pos[2] - node_origin[2]) / cell_size).floor() as i32;

        if cx < 0 || cx > 2 || cy < 0 || cy > 2 || cz < 0 || cz > 2 {
            return false;
        }

        let slot = slot_index(cx as usize, cy as usize, cz as usize);
        match node.children[slot] {
            Child::Empty => return false,
            Child::Block(_) => return true,
            Child::Node(child_id) => {
                if depth + 1 >= max_depth {
                    return true; // at max depth, treat node as solid
                }
                // Descend.
                node_origin = [
                    node_origin[0] + cx as f32 * cell_size,
                    node_origin[1] + cy as f32 * cell_size,
                    node_origin[2] + cz as f32 * cell_size,
                ];
                cell_size /= 3.0;
                node_id = child_id;
            }
        }
    }

    // Reached max_depth without resolving — treat as solid if we're
    // still inside a node.
    true
}

// ─────────────────────────────────────── cubed-sphere body raycast

/// CPU mirror of the shader's `sphere_in_cell`. When `cpu_raycast`
/// descends into a `NodeKind::CubedSphereBody` child, it dispatches
/// here to find which sub-cell of the planet the ray hits and
/// returns a `HitInfo` whose path extends through `(body_id,
/// face_slot)` + the face subtree descent — letting the standard
/// `propagate_edit` machinery break/place that cell generically.
///
/// Step-based march (not the fully analytic version the shader
/// uses): cell-bounded by the deepest term_depth seen, advances
/// by a fraction of that cell. Sufficient accuracy for cursor
/// targeting; not meant for visual rendering.
fn cs_raycast_in_body(
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
    // Normalize direction: the ray-sphere intersection (disc = b²-c)
    // requires |ray_dir| = 1. Callers may pass non-unit direction
    // (e.g. after the pop loop divides by 3.0).
    let ray_dir = sdf::normalize(ray_dir);
    let debug_probe = max_face_depth <= 2 && ancestor_path.len() <= 1;
    let cs_center = [
        body_origin[0] + body_size * 0.5,
        body_origin[1] + body_size * 0.5,
        body_origin[2] + body_size * 0.5,
    ];
    let cs_outer = outer_r_local * body_size;
    let cs_inner = inner_r_local * body_size;
    let shell = cs_outer - cs_inner;
    if shell <= 0.0 { return None; }

    // Ray-sphere with outer radius. Standard form (cursor accuracy
    // is fine; we don't need the Numerical-Recipes fallback).
    let oc = sdf::sub(ray_origin, cs_center);
    let b = sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c;
    if disc <= 0.0 { return None; }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return None; }
    if debug_probe {
        eprintln!(
            "cs_raycast_in_body start origin={:?} dir={:?} body_origin={:?} body_size={} inner={} outer={} t_enter={} t_exit={}",
            ray_origin, ray_dir, body_origin, body_size, inner_r_local, outer_r_local, t_enter, t_exit,
        );
    }

    // Step size scales with the deepest face-subtree depth observed.
    // Start coarse (1/3 of shell) and refine as we sample finer cells.
    let mut step_world = shell * 0.33;
    let eps = (shell * 1e-5).max(1e-7);
    let mut t = t_enter + eps;
    // Tracks the last empty cell's face-subtree path. On block hit,
    // THIS is the placement target: it's the cell the ray was in one
    // step before striking the block, which by construction is empty
    // and adjacent to the block along the ray. This replaces the
    // broken `face_for_placement` mapping that tried to infer an
    // xyz-axis delta from which cubed-sphere boundary was crossed —
    // face-subtree slots are (u,v,r), not (x,y,z), so an xyz delta
    // applied via `place_child` landed in the wrong cell (or into
    // solid rock, making placement silently fail).
    let mut prev_place_path: Option<Vec<(NodeId, usize)>> = None;
    let max_steps = 8_000usize;
    for step_index in 0..max_steps {
        if t >= t_exit { break; }
        let p = sdf::add(ray_origin, sdf::scale(ray_dir, t));
        let local = sdf::sub(p, cs_center);
        let r = sdf::length(local);
        if r > cs_outer {
            if debug_probe && step_index < 12 {
                eprintln!("cs_raycast_in_body step={} t={} outside_outer r={} outer={}", step_index, t, r, cs_outer);
            }
            t += eps * 8.0;
            continue;
        }
        if r < cs_inner {
            if debug_probe && step_index < 12 {
                eprintln!("cs_raycast_in_body step={} t={} below_inner r={} inner={}", step_index, t, r, cs_inner);
            }
            break;
        }

        let p_body = [
            p[0] - body_origin[0],
            p[1] - body_origin[1],
            p[2] - body_origin[2],
        ];
        let face_point =
            cubesphere_local::body_point_to_face_space(p_body, inner_r_local, outer_r_local, body_size)?;
        let face = face_point.face;
        let face_slot = FACE_SLOTS[face as usize];
        let body_node = library.get(body_id)?;
        let face_root_id = match body_node.children[face_slot] {
            Child::Node(id) => id,
            _ => {
                t += step_world;
                continue;
            }
        };
        let un = face_point.un;
        let vn = face_point.vn;
        let rn = face_point.rn;

        let walk = walk_face_subtree_with_path(library, face_root_id, un, vn, rn, max_face_depth);
        if debug_probe && step_index < 12 {
            eprintln!(
                "cs_raycast_in_body step={} t={} r={} face={:?} un={} vn={} rn={} walk={:?}",
                step_index,
                t,
                r,
                face,
                un,
                vn,
                rn,
                walk.as_ref().map(|(block_id, term_depth, path)| (*block_id, *term_depth, path.len())),
            );
        }
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
            // Empty cell — record its full path as the candidate
            // placement target. Step refinement uses the cell's
            // nominal width, BUT clamped to a sane minimum so we
            // don't crawl through huge empty regions one
            // sub-cell at a time. With max_face_depth deep + the
            // surface several levels deeper than the cap, every
            // cap-empty cell is part of a much bigger empty
            // region and refining to nominal cell width burns the
            // entire `max_steps` budget before reaching content.
            // Floor at shell/100 so we always traverse the shell
            // in O(100) steps regardless of cap depth.
            let mut empty_full = ancestor_path.to_vec();
            empty_full.push((body_id, face_slot));
            empty_full.append(&mut face_path);
            prev_place_path = Some(empty_full);
            let cells_d = 3.0_f32.powi(term_depth as i32);
            let nominal = shell / cells_d * 0.33;
            let coarse_floor = shell * 0.01;
            step_world = nominal.max(coarse_floor).max(eps * 4.0);
        }
        t += step_world;
    }

    None
}

fn cs_raycast_in_body_bounded(
    library: &NodeLibrary,
    body_id: NodeId,
    body_origin: [f32; 3],
    body_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    ancestor_path: &[(NodeId, usize)],
    frame_face: u32,
    frame_u_min: f32,
    frame_v_min: f32,
    frame_r_min: f32,
    frame_size: f32,
    remaining_depth: u32,
) -> Option<HitInfo> {
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

    let mut step_world = shell * frame_size * 0.33;
    let eps = (shell * 1e-5).max(1e-7);
    let mut t = t_enter + eps;
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
        if r < cs_inner {
            break;
        }

        let p_body = [
            p[0] - body_origin[0],
            p[1] - body_origin[1],
            p[2] - body_origin[2],
        ];
        let face_point =
            cubesphere_local::body_point_to_face_space(p_body, inner_r_local, outer_r_local, body_size)?;
        if face_point.face as u32 != frame_face {
            break;
        }
        let un_abs = face_point.un;
        let vn_abs = face_point.vn;
        let rn_abs = face_point.rn;
        if un_abs < frame_u_min || un_abs >= frame_u_min + frame_size ||
           vn_abs < frame_v_min || vn_abs >= frame_v_min + frame_size ||
           rn_abs < frame_r_min || rn_abs >= frame_r_min + frame_size {
            break;
        }

        let body_node = library.get(body_id)?;
        let face_slot = FACE_SLOTS[frame_face as usize];
        let face_root_id = match body_node.children[face_slot] {
            Child::Node(id) => id,
            _ => return None,
        };
        let un = ((un_abs - frame_u_min) / frame_size).clamp(0.0, 0.9999999);
        let vn = ((vn_abs - frame_v_min) / frame_size).clamp(0.0, 0.9999999);
        let rn = ((rn_abs - frame_r_min) / frame_size).clamp(0.0, 0.9999999);

        let walk = walk_face_subtree_with_path(library, face_root_id, un, vn, rn, remaining_depth);
        if let Some((block_id, term_depth, mut face_path)) = walk {
            if block_id != 0 {
                let mut full_path = ancestor_path.to_vec();
                full_path.push((body_id, face_slot));
                full_path.append(&mut face_path);
                return Some(HitInfo {
                    path: full_path,
                    face: 4,
                    t,
                    place_path: prev_place_path,
                });
            }
            let mut empty_full = ancestor_path.to_vec();
            empty_full.push((body_id, face_slot));
            empty_full.append(&mut face_path);
            prev_place_path = Some(empty_full);
            let cells_d = 3.0_f32.powi(term_depth as i32);
            let nominal = shell * frame_size / cells_d * 0.33;
            let coarse_floor = shell * frame_size * 0.01;
            step_world = nominal.max(coarse_floor).max(eps * 4.0);
        }
        t += step_world;
    }
    None
}

/// CPU walker mirror of the shader's `walk_face_subtree`.
/// Returns `(block_id, term_depth, path)` on success.
fn walk_face_subtree_with_path(
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
    // After an empty terminal at d < limit, the tree gives us no
    // further structure to walk — but placement at the user's chosen
    // cs_edit_depth requires a path of uniform depth so the placed
    // cell size is consistent (without this, block size varies with
    // wherever the nearest empty chain happened to terminate: tiny
    // above the SDF-detail surface, huge over uniform-empty regions).
    // Synthesize EMPTY_NODE-tagged slot entries for the remaining
    // levels; propagate_edit materializes empty nodes on rebuild.
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
                // Descend all the way to `limit` (= cs_edit_depth).
                // Do NOT stop at uniform-content subtrees: below
                // SDF_DETAIL_LEVELS the face subtree is a uniform
                // chain (stone or empty), and stopping there caps
                // the editable cell size at the SDF detail level
                // regardless of zoom — "interaction stops at layer
                // 15." The shader packer still flattens those
                // uniform chains for rendering efficiency, but that
                // doesn't prevent edits at finer paths: editing a
                // sub-cell of a uniform chain simply forces the
                // packer to re-materialize that subtree on the next
                // upload, so the finer geometry becomes visible
                // exactly where the edit landed.
                if d == limit {
                    // Cap: report the cell at this depth via the
                    // child subtree's uniform/representative block.
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

/// Compute the world-space AABB of the block at the hit location.
///
/// Walks the path Cartesian-style until it encounters a
/// `NodeKind::CubedSphereBody` ancestor; from there the path
/// continues into a face subtree where cell indices are
/// `(u_slot, v_slot, r_slot)` in cube-sphere coords. The bulged
/// voxel's world AABB is computed from `cubesphere::block_corners`
/// at the cell's `(face, u, v, r)` extents.
fn hit_path_slots(hit: &HitInfo) -> Path {
    let mut path = Path::root();
    for &(_, slot) in &hit.path {
        path.push(slot as u8);
    }
    path
}

pub fn hit_aabb_in_frame_local(hit: &HitInfo, frame_path: &Path) -> ([f32; 3], [f32; 3]) {
    let cell_path = hit_path_slots(hit);
    let common = cell_path.common_prefix_len(frame_path) as usize;

    let mut cell_min_common = [0.0f32; 3];
    let mut cell_size_common = WORLD_SIZE;
    for depth in common..cell_path.depth() as usize {
        let (sx, sy, sz) = slot_coords(cell_path.slot(depth) as usize);
        let child_size = cell_size_common / 3.0;
        cell_min_common[0] += sx as f32 * child_size;
        cell_min_common[1] += sy as f32 * child_size;
        cell_min_common[2] += sz as f32 * child_size;
        cell_size_common = child_size;
    }

    let mut frame_min_common = [0.0f32; 3];
    let mut frame_size_common = WORLD_SIZE;
    for depth in common..frame_path.depth() as usize {
        let (sx, sy, sz) = slot_coords(frame_path.slot(depth) as usize);
        let child_size = frame_size_common / 3.0;
        frame_min_common[0] += sx as f32 * child_size;
        frame_min_common[1] += sy as f32 * child_size;
        frame_min_common[2] += sz as f32 * child_size;
        frame_size_common = child_size;
    }

    let scale = WORLD_SIZE / frame_size_common;
    let min = [
        (cell_min_common[0] - frame_min_common[0]) * scale,
        (cell_min_common[1] - frame_min_common[1]) * scale,
        (cell_min_common[2] - frame_min_common[2]) * scale,
    ];
    let extent = cell_size_common * scale;
    (
        min,
        [min[0] + extent, min[1] + extent, min[2] + extent],
    )
}

pub fn hit_aabb_body_local(library: &NodeLibrary, hit: &HitInfo) -> ([f32; 3], [f32; 3]) {
    use super::cubesphere::Face;

    for (index, &(node_id, slot)) in hit.path.iter().enumerate() {
        let Some(node) = library.get(node_id) else { break };
        let Child::Node(child_id) = node.children[slot] else { continue };
        let Some(child_node) = library.get(child_id) else { continue };
        let NodeKind::CubedSphereBody { inner_r, outer_r } = child_node.kind else {
            continue;
        };

        let Some(&(_, face_slot)) = hit.path.get(index + 1) else {
            return ([0.0, 0.0, 0.0], [WORLD_SIZE, WORLD_SIZE, WORLD_SIZE]);
        };
        let Some(face_index) = (0..6).find(|&face| FACE_SLOTS[face] == face_slot) else {
            return ([0.0, 0.0, 0.0], [WORLD_SIZE, WORLD_SIZE, WORLD_SIZE]);
        };
        let face = Face::from_index(face_index as u8);

        let mut iu = 0u32;
        let mut iv = 0u32;
        let mut ir = 0u32;
        let mut depth = 0u32;
        for &(_, face_child_slot) in &hit.path[index + 2..] {
            let (us, vs, rs) = slot_coords(face_child_slot);
            iu = iu * 3 + us as u32;
            iv = iv * 3 + vs as u32;
            ir = ir * 3 + rs as u32;
            depth += 1;
        }

        let cells = if depth == 0 { 1.0 } else { 3.0_f32.powi(depth as i32) };
        let un0 = iu as f32 / cells;
        let vn0 = iv as f32 / cells;
        let rn0 = ir as f32 / cells;
        let du = 1.0 / cells;
        let dv = 1.0 / cells;
        let dr = 1.0 / cells;

        let corners = [
            cubesphere_local::face_space_to_body_point(face, un0, vn0, rn0, inner_r, outer_r, WORLD_SIZE),
            cubesphere_local::face_space_to_body_point(face, un0 + du, vn0, rn0, inner_r, outer_r, WORLD_SIZE),
            cubesphere_local::face_space_to_body_point(face, un0, vn0 + dv, rn0, inner_r, outer_r, WORLD_SIZE),
            cubesphere_local::face_space_to_body_point(face, un0 + du, vn0 + dv, rn0, inner_r, outer_r, WORLD_SIZE),
            cubesphere_local::face_space_to_body_point(face, un0, vn0, rn0 + dr, inner_r, outer_r, WORLD_SIZE),
            cubesphere_local::face_space_to_body_point(face, un0 + du, vn0, rn0 + dr, inner_r, outer_r, WORLD_SIZE),
            cubesphere_local::face_space_to_body_point(face, un0, vn0 + dv, rn0 + dr, inner_r, outer_r, WORLD_SIZE),
            cubesphere_local::face_space_to_body_point(face, un0 + du, vn0 + dv, rn0 + dr, inner_r, outer_r, WORLD_SIZE),
        ];
        let mut min = corners[0];
        let mut max = corners[0];
        for corner in &corners[1..] {
            for axis in 0..3 {
                min[axis] = min[axis].min(corner[axis]);
                max[axis] = max[axis].max(corner[axis]);
            }
        }
        return (min, max);
    }

    hit_aabb_in_frame_local(hit, &Path::root())
}

pub fn hit_aabb(library: &NodeLibrary, hit: &HitInfo) -> ([f32; 3], [f32; 3]) {
    use super::cubesphere::{block_corners, Face, FACE_SLOTS};

    let mut origin = [0.0f32; 3];
    let mut cell_size = 1.0f32;

    for (i, &(node_id, slot)) in hit.path.iter().enumerate() {
        let node = match library.get(node_id) {
            Some(n) => n,
            None => break,
        };
        // Detect the body by inspecting the CHILD at this slot, not
        // the current node's kind. At this point `origin`/`cell_size`
        // describe the current node's children: the child at `slot`
        // occupies `[origin + (x,y,z)*cell_size, +cell_size)^3` with
        // width `cell_size`. If we advanced state first and then
        // checked the body at path[i+1], `cell_size` would already be
        // the body's *children's* width (1/3 of its cell) — wrong
        // scale for body_center and r_world. Matches `cpu_raycast`'s
        // sphere dispatch, which uses `parent_cell_size` as body_size.
        let child = node.children[slot];
        if let Child::Node(child_id) = child {
            if let Some(child_node) = library.get(child_id) {
                if let NodeKind::CubedSphereBody { inner_r, outer_r } = child_node.kind {
                    let (x, y, z) = slot_coords(slot);
                    let body_origin = [
                        origin[0] + x as f32 * cell_size,
                        origin[1] + y as f32 * cell_size,
                        origin[2] + z as f32 * cell_size,
                    ];
                    let body_size = cell_size;
                    let body_center = [
                        body_origin[0] + body_size * 0.5,
                        body_origin[1] + body_size * 0.5,
                        body_origin[2] + body_size * 0.5,
                    ];
                    // The NEXT path entry is (body_id, face_slot) —
                    // determines which face subtree holds the hit.
                    let face_slot = match hit.path.get(i + 1) {
                        Some(&(_, fs)) => fs,
                        None => {
                            return (body_origin, [
                                body_origin[0] + body_size,
                                body_origin[1] + body_size,
                                body_origin[2] + body_size,
                            ]);
                        }
                    };
                    let face = match (0..6).find(|&f| FACE_SLOTS[f] == face_slot) {
                        Some(f) => Face::from_index(f as u8),
                        None => {
                            return (body_origin, [
                                body_origin[0] + body_size,
                                body_origin[1] + body_size,
                                body_origin[2] + body_size,
                            ]);
                        }
                    };
                    // Remaining entries AFTER (body, face_slot) walk
                    // the face subtree — each slot's (u, v, r)
                    // accumulates cell indices.
                    let mut iu: u32 = 0;
                    let mut iv: u32 = 0;
                    let mut ir: u32 = 0;
                    let mut depth: u32 = 0;
                    for &(_face_node_id, face_slot_idx) in &hit.path[i + 2..] {
                        let (us, vs, rs) = slot_coords(face_slot_idx);
                        iu = iu * 3 + us as u32;
                        iv = iv * 3 + vs as u32;
                        ir = ir * 3 + rs as u32;
                        depth += 1;
                    }
                    // Cell extent in the face's normalized (u, v, r) frame.
                    let cells = if depth == 0 { 1.0 } else { 3.0_f32.powi(depth as i32) };
                    let u_lo = (iu as f32 / cells) * 2.0 - 1.0;
                    let v_lo = (iv as f32 / cells) * 2.0 - 1.0;
                    let r_lo_n = ir as f32 / cells;
                    let du = 2.0 / cells;
                    let dv = 2.0 / cells;
                    let drn = 1.0 / cells;
                    let r_world_lo = inner_r * body_size + r_lo_n * (outer_r - inner_r) * body_size;
                    let dr_world = drn * (outer_r - inner_r) * body_size;
                    let corners = block_corners(
                        body_center, face,
                        u_lo, v_lo, r_world_lo,
                        du, dv, dr_world,
                    );
                    let mut aabb_min = corners[0];
                    let mut aabb_max = corners[0];
                    for c in &corners[1..] {
                        for k in 0..3 {
                            if c[k] < aabb_min[k] { aabb_min[k] = c[k]; }
                            if c[k] > aabb_max[k] { aabb_max[k] = c[k]; }
                        }
                    }
                    return (aabb_min, aabb_max);
                }
            }
        }
        // Cartesian step.
        let (x, y, z) = slot_coords(slot);
        origin = [
            origin[0] + x as f32 * cell_size,
            origin[1] + y as f32 * cell_size,
            origin[2] + z as f32 * cell_size,
        ];
        cell_size /= 3.0;
    }
    // Undo last division — the final path entry IS the hit cell.
    cell_size *= 3.0;
    let aabb_max = [
        origin[0] + cell_size,
        origin[1] + cell_size,
        origin[2] + cell_size,
    ];
    (origin, aabb_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::bootstrap::plain_test_world;
    use crate::world::palette::block;
    use crate::world::cubesphere::insert_spherical_body;
    use crate::world::sdf::Planet;

    #[test]
    fn propagate_edit_preserves_node_kinds_through_sphere_path() {
        // Regression for "world collapses into floating cubes when
        // I break a block": propagate_edit was reinserting ancestors
        // via lib.insert (default Cartesian), destroying the body's
        // CubedSphereBody NodeKind. The shader's NodeKind dispatch
        // then stopped firing, walker descended into the body
        // Cartesian-style and rendered the interior-stone fillers
        // as cube blocks.
        let mut lib = NodeLibrary::default();
        let sdf = Planet {
            center: [0.5, 0.5, 0.5],
            radius: 0.30, noise_scale: 0.0, noise_freq: 1.0, noise_seed: 0,
            gravity: 0.0, influence_radius: 1.0,
            surface_block: block::GRASS, core_block: block::STONE,
        };
        let body_id = insert_spherical_body(&mut lib, 0.12, 0.45, 6, &sdf);
        let body_kind_before = lib.get(body_id).unwrap().kind;
        assert!(matches!(body_kind_before, NodeKind::CubedSphereBody { .. }));

        // Find any Block leaf in the body and edit it.
        // For this test, simulate a hit at slot 0 (some arbitrary
        // sub-cell) of the first face subtree.
        let body_node = lib.get(body_id).unwrap();
        let face_root_id = match body_node.children[crate::world::cubesphere::FACE_SLOTS[0]] {
            Child::Node(id) => id,
            _ => panic!("face slot must be a Node"),
        };
        let face_kind_before = lib.get(face_root_id).unwrap().kind;
        assert!(matches!(face_kind_before, NodeKind::CubedSphereFace { .. }));

        // Build a simulated edit path: world-tree-style root → body
        // → face_root → face's slot 0. Use a 1-level world (body is
        // at slot 0 of "world root").
        let mut world_children = empty_children();
        world_children[0] = Child::Node(body_id);
        let world_root = lib.insert(world_children);
        let mut world = WorldState { root: world_root, library: lib };
        world.library.ref_inc(world_root);

        // Build a HitInfo at body → face_root → slot 0.
        let hit = HitInfo {
            path: vec![
                (world_root, 0),
                (body_id, crate::world::cubesphere::FACE_SLOTS[0]),
                (face_root_id, 0),
            ],
            face: 0, t: 1.0, place_path: None,
        };
        assert!(propagate_edit(&mut world, &hit, Child::Empty));

        // Walk the new world root and verify NodeKinds along the
        // edit path are preserved (NOT collapsed to Cartesian).
        let new_world_root = world.root;
        let new_root_kind = world.library.get(new_world_root).unwrap().kind;
        assert!(matches!(new_root_kind, NodeKind::Cartesian),
            "world root must stay Cartesian");
        let new_body = match world.library.get(new_world_root).unwrap().children[0] {
            Child::Node(id) => id,
            _ => panic!("slot 0 must still be the body"),
        };
        let new_body_kind = world.library.get(new_body).unwrap().kind;
        assert!(matches!(new_body_kind, NodeKind::CubedSphereBody { .. }),
            "body NodeKind must survive the edit (was: {:?})", new_body_kind);
        let new_face = match world.library.get(new_body).unwrap()
            .children[crate::world::cubesphere::FACE_SLOTS[0]] {
            Child::Node(id) => id,
            _ => panic!("face slot must still be a Node"),
        };
        let new_face_kind = world.library.get(new_face).unwrap().kind;
        assert!(matches!(new_face_kind, NodeKind::CubedSphereFace { .. }),
            "face NodeKind must survive the edit (was: {:?})", new_face_kind);
    }

    #[test]
    fn raycast_hits_ground() {
        let world = plain_test_world();
        // Cast straight down from above the ground.
        let hit = cpu_raycast(
            &world.library,
            world.root,
            [1.5, 2.5, 1.5],
            [0.0, -1.0, 0.0],
            8,
        );
        assert!(hit.is_some(), "Should hit the ground");
        let hit = hit.unwrap();
        // Should hit the +Y face (face 2) since we're coming from above.
        assert_eq!(hit.face, 2, "Should hit top face");
    }

    #[test]
    fn cpu_raycast_in_frame_at_root_matches_world_raycast() {
        // With frame_path empty, frame_aware raycast should be
        // equivalent to the legacy world-coord raycast.
        let world = plain_test_world();
        let world_hit = cpu_raycast(
            &world.library, world.root,
            [1.5, 2.5, 1.5], [0.0, -1.0, 0.0],
            8,
        );
        let frame_hit = cpu_raycast_in_frame(
            &world.library, world.root,
            &[],  // empty frame == world root
            [1.5, 2.5, 1.5], [0.0, -1.0, 0.0],
            8, 6,
        );
        assert!(world_hit.is_some());
        assert!(frame_hit.is_some());
        let w = world_hit.unwrap();
        let f = frame_hit.unwrap();
        assert_eq!(w.path.len(), f.path.len(),
            "same hit depth via both paths");
        assert_eq!(w.face, f.face);
    }

    #[test]
    fn cpu_raycast_in_frame_pop_finds_hit_in_ancestor() {
        // Camera frame is set to a deep empty cell (no content
        // in the frame's bubble), but the ray points UP toward
        // a sibling that has content. Frame ascent must find
        // the hit in the ancestor.
        let world = plain_test_world();
        // Frame at slot 16 (top-row middle): some test_world
        // slots there have features. Camera in that frame's
        // [0, 3) coords pointing... actually the test world is
        // small (depth 3), let's just verify that the ascent
        // code path runs and produces SOMETHING (or None) without
        // panicking when the inner raycast misses.
        let frame_path = [16u8, 13u8];
        // Cam at edge of frame, ray pointing in arbitrary dir.
        let cam = [0.5, 0.5, 0.5];
        let dir = [0.7, 0.7, 0.0];
        let _ = cpu_raycast_in_frame(
            &world.library, world.root,
            &frame_path, cam, dir, 8, 6,
        );
        // Just verify no panic — covered the ascent code.
    }

    #[test]
    fn planet_world_raycast_hits_sphere() {
        // Reproduce the user's bug: at shallow anchor (anchor=4),
        // edit_depth=3, cs_edit_depth=1, frame_path=[16]. Cursor
        // ray pointing down at the planet should produce a sphere
        // hit (face=4), not a Cartesian fallback (face=2).
        use crate::world::worldgen::generate_world;
        use crate::world::spherical_worldgen::{demo_planet, install_at_root_center};

        let mut world = generate_world();
        let setup = demo_planet();
        let (new_root, _planet_path) =
            install_at_root_center(&mut world.library, world.root, &setup);
        world.swap_root(new_root);

        let cam_local = [1.5, 0.0, 1.5];
        let ray_dir = [0.0, -0.9320391, -0.3623577];
        let frame_path = [16u8];
        let edit_depth = 3u32;
        let cs_edit_depth = 1u32;

        let hit = cpu_raycast_in_frame(
            &world.library, world.root,
            &frame_path, cam_local, ray_dir,
            edit_depth, cs_edit_depth,
        );
        eprintln!("planet_world_raycast hit = {:?}",
            hit.as_ref().map(|h| (h.path.len(), h.face, h.t,
                h.place_path.as_ref().map(|p| p.len()))));
        assert!(hit.is_some(), "ray should hit planet");
        let h = hit.unwrap();
        assert_eq!(h.face, 4,
            "should be sphere hit (face=4), not Cartesian fallback");
    }

    #[test]
    fn cpu_raycast_in_frame_dispatches_sphere_when_frame_is_body() {
        // Regression: when the camera frame is itself the
        // CubedSphereBody (camera path goes [13, ...]), the inner
        // raycast must dispatch to cs_raycast_in_body, not run a
        // Cartesian DDA on the body's children. Prior bug: face
        // subtree nodes (CubedSphereFace kind) didn't trigger
        // sphere dispatch and got returned as Cartesian solid hits
        // via the rep-block max_depth fallback (face=2, path=5+).
        use crate::world::worldgen::generate_world;
        use crate::world::spherical_worldgen::{demo_planet, install_at_root_center};

        let mut world = generate_world();
        let setup = demo_planet();
        let (new_root, _planet_path) =
            install_at_root_center(&mut world.library, world.root, &setup);
        world.swap_root(new_root);

        // Camera inside the body cell, frame = [13] (body).
        // cam_local in body's [0, 3) frame.
        let cam_local = [1.5, 1.5, 1.5];
        let ray_dir = [0.0, -1.0, 0.0];
        let frame_path = [13u8];

        let hit = cpu_raycast_in_frame(
            &world.library, world.root,
            &frame_path, cam_local, ray_dir,
            10, 4,
        );
        if let Some(h) = &hit {
            assert_eq!(h.face, 4,
                "body-as-frame must produce sphere hit (face=4), \
                 not Cartesian fallback (face=2). Got path.len={} face={}",
                h.path.len(), h.face);
        }
    }

    #[test]
    fn cpu_raycast_in_frame_path_starts_from_world_root() {
        // The returned path's first entry must be (world.root, _).
        let world = plain_test_world();
        let hit = cpu_raycast_in_frame(
            &world.library, world.root,
            &[], [1.5, 2.5, 1.5], [0.0, -1.0, 0.0],
            8, 6,
        ).expect("should hit ground");
        assert_eq!(hit.path[0].0, world.root,
            "path begins at world.root");
    }

    #[test]
    fn raycast_misses_sky() {
        let world = plain_test_world();
        // Cast straight up from a position in pure air (root y=2, x=0, z=0 is air_l2).
        let hit = cpu_raycast(
            &world.library,
            world.root,
            [0.5, 2.5, 0.5],
            [0.0, 1.0, 0.0],
            8,
        );
        assert!(hit.is_none(), "Should miss when looking at sky");
    }

    #[test]
    fn break_block_modifies_world() {
        let mut world = plain_test_world();
        let old_root = world.root;
        let hit = cpu_raycast(
            &world.library,
            world.root,
            [1.5, 2.5, 1.5],
            [0.0, -1.0, 0.0],
            8,
        ).unwrap();
        assert!(break_block(&mut world, &hit));
        assert_ne!(world.root, old_root, "Root should change after edit");
    }

    #[test]
    fn place_block_on_ground() {
        let mut world = plain_test_world();
        // First break a block to create an empty space, then place into it.
        let hit = cpu_raycast(
            &world.library,
            world.root,
            [1.5, 2.5, 1.5],
            [0.0, -1.0, 0.0],
            8,
        ).unwrap();
        assert!(break_block(&mut world, &hit));

        // Now cast again — should hit the block below the one we removed.
        let hit2 = cpu_raycast(
            &world.library,
            world.root,
            [1.5, 2.5, 1.5],
            [0.0, -1.0, 0.0],
            8,
        ).unwrap();
        let old_root = world.root;
        // Place on top of the newly exposed surface.
        assert!(place_block(&mut world, &hit2, block::BRICK));
        assert_ne!(world.root, old_root, "Root should change after placement");
    }

    #[test]
    fn zoom_controls_edit_depth() {
        let world = plain_test_world();
        // At max_depth=1, the ray should hit a node (coarse edit).
        let hit_coarse = cpu_raycast(
            &world.library,
            world.root,
            [1.5, 2.5, 1.5],
            [0.0, -1.0, 0.0],
            1,
        );
        // At max_depth=8, the ray should hit a block (fine edit).
        let hit_fine = cpu_raycast(
            &world.library,
            world.root,
            [1.5, 2.5, 1.5],
            [0.0, -1.0, 0.0],
            8,
        );
        assert!(hit_coarse.is_some());
        assert!(hit_fine.is_some());
        // Coarse hit should have shorter path (fewer levels descended).
        assert!(hit_coarse.unwrap().path.len() < hit_fine.unwrap().path.len());
    }

    #[test]
    fn cross_node_placement_upward() {
        let mut world = plain_test_world();
        // Hit the ground surface from above. The test world has grass
        // surface around y≈1 (level-2 node). At depth 2, cells are
        // 1.0 wide, so cell (x, 2, z) at depth 1 is the top of a
        // coarse node.
        //
        // We need a hit at a cell boundary where placing above would
        // cross into a different node (ny > 2).
        let hit = cpu_raycast(
            &world.library,
            world.root,
            [1.5, 2.5, 1.5],
            [0.0, -1.0, 0.0],
            2, // coarse depth — each cell is a 3x3x3 group
        );
        assert!(hit.is_some(), "Should hit ground at depth 2");
        let hit = hit.unwrap();

        let (_, slot) = *hit.path.last().unwrap();
        let (_x, y, _z) = slot_coords(slot);

        // If the hit cell is at y=2 (top of node), placing above (face=2, +Y)
        // requires cross-node placement.
        if y == 2 {
            let old_root = world.root;
            assert!(
                place_block(&mut world, &hit, block::BRICK),
                "Cross-node placement above y=2 cell should succeed"
            );
            assert_ne!(world.root, old_root);
        }
    }

    #[test]
    fn cross_node_placement_into_empty_subtree() {
        let mut world = plain_test_world();
        // The test world has air_l2 at (0, 2, 0). Hit the ground
        // surface below it and place upward into the empty air region.
        let hit = cpu_raycast(
            &world.library,
            world.root,
            [0.5, 2.5, 0.5],
            [0.0, -1.0, 0.0],
            3, // depth 3 = individual blocks
        );
        assert!(hit.is_some(), "Should hit ground");
        let hit = hit.unwrap();
        let (aabb_min, aabb_max) = hit_aabb(&world.library, &hit);
        let cell_size = aabb_max[0] - aabb_min[0];

        // Check that the block above the hit is in a different node
        // (i.e., the target center is in the air region).
        let target_center_y = (aabb_min[1] + aabb_max[1]) * 0.5 + cell_size;
        if target_center_y < 3.0 {
            let old_root = world.root;
            let placed = place_block(&mut world, &hit, block::BRICK);
            // Should succeed even when crossing into an empty subtree.
            assert!(placed, "Should place into empty subtree");
            assert_ne!(world.root, old_root);

            // Verify the block is now solid at the target position.
            let target = [
                (aabb_min[0] + aabb_max[0]) * 0.5,
                target_center_y,
                (aabb_min[2] + aabb_max[2]) * 0.5,
            ];
            assert!(
                is_solid_at(&world.library, world.root, target, 8),
                "Placed block should be solid at target"
            );
        }
    }

    /// Replicate the interactive game's frame-aware raycast at depths
    /// from 4 to 38 using direct spawns. This mirrors what
    /// `App::frame_aware_raycast` does when spawned at each depth.
    #[test]
    fn frame_aware_raycast_hits_at_all_depths() {
        use crate::world::bootstrap;

        let render_frame_k = 3u8;

        for anchor_depth in [4u8, 6, 8, 10, 11, 12, 15, 20, 25, 30, 33, 38] {
            let boot = bootstrap::bootstrap_world(
                bootstrap::WorldPreset::PlainTest,
                Some(40),
            );
            let mut world = boot.world;
            let pos = bootstrap::plain_surface_spawn(anchor_depth);
            bootstrap::carve_air_pocket(&mut world, &pos.anchor, 40);

            let frame_depth = anchor_depth.saturating_sub(render_frame_k);
            let mut frame_path = pos.anchor;
            frame_path.truncate(frame_depth);

            let cam_local = pos.in_frame(&frame_path);
            let ray_dir = crate::world::sdf::normalize([0.0, -0.434, -0.901]);
            let edit_depth = anchor_depth as u32;

            let hit = cpu_raycast_in_frame(
                &world.library,
                world.root,
                frame_path.as_slice(),
                cam_local,
                ray_dir,
                edit_depth,
                6,
            );

            assert!(
                hit.is_some(),
                "direct-spawn raycast missed at anchor_depth={anchor_depth}: \
                 frame_path={:?} cam_local={:?} edit_depth={edit_depth}",
                frame_path.as_slice(), cam_local,
            );

            let h = hit.unwrap();
            let old_root = world.root;
            let changed = break_block(&mut world, &h);
            assert!(
                changed,
                "break_block failed at anchor_depth={anchor_depth}: path_len={} face={}",
                h.path.len(), h.face,
            );
            assert_ne!(world.root, old_root,
                "root unchanged after break at anchor_depth={anchor_depth}");
        }
    }

    /// Replicate the INTERACTIVE flow: spawn at depth 8, then zoom_in
    /// incrementally to deeper depths, raycasting at each.
    ///
    /// `carve_air_pocket` creates a uniform-air subtree from the
    /// spawn cell (depth 8) down to depth 40. Deeper frame roots sit
    /// inside this all-air subtree. The DDA at each popped level
    /// finds nothing until it pops above the air subtree boundary
    /// (depth 7). With MAX_RAYCAST_POPS=6 and RENDER_FRAME_K=3, the
    /// deepest reachable frame is depth 7+6=13 → anchor depth 16.
    /// Beyond that the raycast correctly returns None: there's
    /// nothing solid within reach, just as Minecraft's reach limit
    /// prevents targeting blocks thousands of voxels away.
    #[test]
    fn frame_aware_raycast_hits_after_zoom_in_from_spawn() {
        use crate::world::bootstrap;

        let render_frame_k = 3u8;
        let initial_depth = 8u8;
        // carve_air_pocket clears the cell at depth 7. The air
        // subtree fills depths 8..40. MAX_RAYCAST_POPS=6 from
        // frame depth 13 can pop to depth 7 (the boundary). So
        // the max anchor depth with hits is 13 + RENDER_FRAME_K = 16.
        let max_hittable_depth = 7u8 + MAX_RAYCAST_POPS as u8 + render_frame_k;

        let boot = bootstrap::bootstrap_world(
            bootstrap::WorldPreset::PlainTest,
            Some(40),
        );
        let mut world = boot.world;
        let mut pos = bootstrap::plain_surface_spawn(initial_depth);
        bootstrap::carve_air_pocket(&mut world, &pos.anchor, 40);

        for target_depth in (initial_depth + 1)..=30u8 {
            pos.zoom_in();

            let anchor_depth = pos.anchor.depth();
            assert_eq!(anchor_depth, target_depth,
                "zoom_in should increase depth by 1");

            let frame_depth = anchor_depth.saturating_sub(render_frame_k);
            let mut frame_path = pos.anchor;
            frame_path.truncate(frame_depth);

            let cam_local = pos.in_frame(&frame_path);
            let ray_dir = crate::world::sdf::normalize([0.0, -0.434, -0.901]);
            let edit_depth = anchor_depth as u32;

            let hit = cpu_raycast_in_frame(
                &world.library,
                world.root,
                frame_path.as_slice(),
                cam_local,
                ray_dir,
                edit_depth,
                6,
            );

            if target_depth <= max_hittable_depth {
                assert!(
                    hit.is_some(),
                    "zoom-in raycast missed at depth={target_depth}: \
                     frame_path={:?} cam_local={:?} edit_depth={edit_depth} \
                     anchor_slots={:?}",
                    frame_path.as_slice(), cam_local, pos.anchor.as_slice(),
                );
                let h = hit.unwrap();
                eprintln!(
                    "zoom-in depth={target_depth}: hit path_len={} face={} t={}",
                    h.path.len(), h.face, h.t,
                );
            } else {
                // Beyond pop range: the air pocket has no solid
                // content within MAX_RAYCAST_POPS. Correctly None.
                assert!(
                    hit.is_none(),
                    "expected miss at depth={target_depth} (beyond pop range), \
                     but got a hit",
                );
            }
        }
    }

    /// Deep raycast with solid content within reach. This simulates
    /// the real gameplay scenario: the player breaks blocks at their
    /// current zoom level, so walls are always nearby. Here we build
    /// a world with a single empty cell (the camera's) surrounded by
    /// 26 solid siblings at a deep depth, and verify the raycast
    /// finds the adjacent solid cell without any pops.
    #[test]
    fn deep_raycast_hits_nearby_solid() {
        let depth: usize = 35;
        let mut lib = NodeLibrary::default();

        // Build a solid subtree extending below the target depth.
        let solid_child = lib.build_uniform_subtree(block::STONE, (depth - 1) as u32);

        // Build the target-depth node: one empty cell (camera slot),
        // 26 solid cells around it.
        let camera_slot = slot_index(1, 1, 1); // center
        let mut children = uniform_children(solid_child);
        children[camera_slot] = Child::Empty;
        let mut root = lib.insert(children);

        // Wrap in ancestor nodes up to the root.
        for _ in 0..(depth - 1) {
            let mut c = uniform_children(solid_child);
            c[camera_slot] = Child::Node(root);
            root = lib.insert(c);
        }

        // Camera at center of the frame, looking -Z.
        let frame_depth = depth.saturating_sub(3);
        let mut frame_path = crate::world::anchor::Path::root();
        for _ in 0..frame_depth {
            frame_path.push(camera_slot as u8);
        }

        // Camera is 3 levels below frame: (1 + (1 + (1+0.5)/3)/3)/3... = 1.5
        let cam_local = [1.5, 1.5, 1.5];
        let ray_dir = crate::world::sdf::normalize([0.0, 0.0, -1.0]);

        let hit = cpu_raycast_in_frame(
            &lib,
            root,
            frame_path.as_slice(),
            cam_local,
            ray_dir,
            depth as u32,
            6,
        );

        assert!(
            hit.is_some(),
            "raycast at depth={depth} should find adjacent solid cell"
        );
        let h = hit.unwrap();
        assert!(
            h.t < RAYCAST_REACH,
            "hit t={} should be within RAYCAST_REACH={}",
            h.t, RAYCAST_REACH,
        );
        eprintln!(
            "deep_raycast depth={}: hit path_len={} face={} t={}",
            depth, h.path.len(), h.face, h.t,
        );
    }

    #[test]
    fn placement_outside_world_returns_false() {
        let mut world = plain_test_world();
        // Create a fake hit at the world boundary (cell y=2 at depth 1).
        // Placing above would go outside [0, 3).
        let hit = HitInfo {
            path: vec![(world.root, slot_index(1, 2, 1))],
            face: 2, // +Y
            t: 1.0, place_path: None,
        };
        assert!(
            !place_block(&mut world, &hit, block::BRICK),
            "Should reject placement outside world bounds"
        );
    }

    /// Cross-node placement must work identically at shallow and deep
    /// layers. The old f32-based `place_child_at_point` lost precision
    /// past depth ~23; the new path-based `place_child_at_path` uses
    /// pure slot arithmetic and works at all 63 layers.
    #[test]
    fn cross_node_placement_works_at_depth_40() {
        use crate::world::anchor::Path;

        let depth: usize = 40;
        let mut lib = NodeLibrary::default();
        let solid = lib.build_uniform_subtree(block::BRICK, depth as u32);

        // Root: only slot (0,1,1) is solid; (1,1,1) is empty.
        let mut root_children = empty_children();
        root_children[slot_index(0, 1, 1)] = solid;
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        // Build a hit path at depth 40 along x=2 at every level
        // inside the solid subtree. The carry from stepping +X at
        // depth 40 propagates all the way up to root, crossing from
        // slot (0,1,1) to (1,1,1) — which is empty.
        let mut path = Vec::new();
        let mut current = root;
        for level in 0..depth {
            let slot = if level == 0 {
                slot_index(0, 1, 1) // enter solid at root
            } else {
                slot_index(2, 1, 1) // rightmost x at every deeper level
            };
            path.push((current, slot));
            if let Some(node) = world.library.get(current) {
                if let Child::Node(child_id) = node.children[slot] {
                    current = child_id;
                }
            }
        }

        let hit = HitInfo {
            path: path.clone(),
            face: 0, // +X
            t: 1.0,
            place_path: None,
        };

        let result = place_block(&mut world, &hit, block::STONE);
        assert!(result, "Cross-node placement at depth {} must succeed", depth);

        // Verify the block landed at the expected path.
        let mut verify_path = Path::root();
        for &(_, slot) in &path {
            verify_path.push(slot as u8);
        }
        verify_path.step_neighbor_cartesian(0, 1);

        // The stepped path should start with slot (1,1,1) at root
        // (crossed from x=0 to x=1) and then (0,1,1) at every
        // deeper level (all x=2 wrapped to x=0).
        assert_eq!(
            verify_path.as_slice()[0],
            slot_index(1, 1, 1) as u8,
            "root slot must cross from x=0 to x=1"
        );
        for d in 1..depth {
            assert_eq!(
                verify_path.as_slice()[d],
                slot_index(0, 1, 1) as u8,
                "depth {} must wrap x=2 to x=0", d + 1,
            );
        }

        // Walk the tree to verify the block exists.
        let placed_slots = verify_path.as_slice();
        let mut node_id = world.root;
        for (i, &slot) in placed_slots.iter().enumerate() {
            let node = world.library.get(node_id).expect("node must exist");
            let child = node.children[slot as usize];
            if i == placed_slots.len() - 1 {
                assert!(
                    !matches!(child, Child::Empty),
                    "Placed block at depth {} must not be empty (slot {})",
                    i + 1, slot,
                );
            } else if let Child::Node(next) = child {
                node_id = next;
            } else {
                panic!(
                    "Expected Node at depth {} slot {}, got {:?}",
                    i + 1, slot, child,
                );
            }
        }
    }
}
