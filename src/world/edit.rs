//! CPU raycasting and block editing.
//!
//! The CPU ray march mirrors the GPU shader's tree traversal to
//! determine which block the crosshair is pointing at. Edits operate
//! at a layer-dependent depth: the zoom level controls how deep the
//! raycast descends, so the same code breaks a single block at fine
//! zoom or an entire node (3x3x3 group) at coarse zoom.

use super::state::WorldState;
use super::tree::*;

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

    loop {
        if iterations >= 512 || stack.is_empty() {
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
                return Some(HitInfo {
                    path: path.clone(),
                    face: normal_face,
                    t: cell_entry_t(&stack[depth], &ray_origin, &inv_dir),
                });
            }
            Child::Node(child_id) => {
                if (depth as u32 + 1) >= max_depth {
                    // At max depth, treat node as solid.
                    return Some(HitInfo {
                        path: path.clone(),
                        face: normal_face,
                        t: cell_entry_t(&stack[depth], &ray_origin, &inv_dir),
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

/// Place a block adjacent to the hit face.
pub fn place_block(world: &mut WorldState, hit: &HitInfo, block_type: BlockType) -> bool {
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

    if nx < 0 || nx > 2 || ny < 0 || ny > 2 || nz < 0 || nz > 2 {
        return false;
    }

    let adj_slot = slot_index(nx as usize, ny as usize, nz as usize);
    let parent_id = hit.path.last().unwrap().0;

    let node = match world.library.get(parent_id) {
        Some(n) => n,
        None => return false,
    };

    if !node.children[adj_slot].is_empty() {
        return false;
    }

    let mut place_hit = hit.clone();
    place_hit.path.last_mut().unwrap().1 = adj_slot;
    propagate_edit(world, &place_hit, Child::Block(block_type))
}

/// Apply an edit and propagate clone-on-write up to root.
fn propagate_edit(world: &mut WorldState, hit: &HitInfo, new_child: Child) -> bool {
    if hit.path.is_empty() {
        return false;
    }

    let mut replacement: Option<NodeId> = None;

    for i in (0..hit.path.len()).rev() {
        let (node_id, slot) = hit.path[i];
        let node = match world.library.get(node_id) {
            Some(n) => n,
            None => return false,
        };

        let mut new_children = node.children;
        if let Some(nid) = replacement {
            new_children[slot] = Child::Node(nid);
        } else {
            new_children[slot] = new_child;
        }

        replacement = Some(world.library.insert(new_children));
    }

    if let Some(new_root) = replacement {
        world.swap_root(new_root);
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------- helpers

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

/// Compute the world-space AABB of the block at the hit location.
pub fn hit_aabb(hit: &HitInfo) -> ([f32; 3], [f32; 3]) {
    let mut origin = [0.0f32; 3];
    let mut cell_size = 1.0f32;

    for &(_node_id, slot) in &hit.path {
        let (x, y, z) = slot_coords(slot);
        origin = [
            origin[0] + x as f32 * cell_size,
            origin[1] + y as f32 * cell_size,
            origin[2] + z as f32 * cell_size,
        ];
        cell_size /= 3.0;
    }
    // Undo the last division — the final path entry IS the hit cell.
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

    #[test]
    fn raycast_hits_ground() {
        let world = WorldState::test_world();
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
    fn raycast_misses_sky() {
        let world = WorldState::test_world();
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
        let mut world = WorldState::test_world();
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
        let mut world = WorldState::test_world();
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
        assert!(place_block(&mut world, &hit2, BlockType::Brick));
        assert_ne!(world.root, old_root, "Root should change after placement");
    }

    #[test]
    fn zoom_controls_edit_depth() {
        let world = WorldState::test_world();
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
}
