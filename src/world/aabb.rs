//! World-space AABBs for ray-march hits. Two variants handle the
//! two coordinate systems: `hit_aabb` walks world-space (used from
//! outside any frame), `hit_aabb_in_frame_local` expresses the AABB
//! relative to the current render frame so UI can draw it without
//! f32 precision loss at deep layers.
//!
//! Cubed-sphere hits take a different geometric path than Cartesian
//! ones: once the hit path crosses a `CubedSphereBody` ancestor, the
//! remaining slots are `(u_slot, v_slot, r_slot)` in face-normalized
//! coordinates — the AABB then comes from bulging the cell's eight
//! corners into world space via `cubesphere_local::face_space_to_body_point`.

use super::anchor::{Path, WORLD_SIZE};
use super::cubesphere::{block_corners, Face};
use super::cubesphere_local::{self, find_body_ancestor_in_path};
use super::raycast::HitInfo;
use super::tree::{slot_coords, NodeLibrary};

/// True if the hit path descends through a `CubedSphereBody`
/// ancestor (so the hit is inside a sphere). Used to pick between
/// sphere-aware AABB construction and plain cartesian cell AABB.
pub fn hit_path_crosses_body(library: &NodeLibrary, hit: &HitInfo) -> bool {
    find_body_ancestor_in_path(library, &hit.path).is_some()
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
    (min, [min[0] + extent, min[1] + extent, min[2] + extent])
}

pub fn hit_aabb_body_local(library: &NodeLibrary, hit: &HitInfo) -> ([f32; 3], [f32; 3]) {
    let Some((body_index, inner_r, outer_r)) =
        find_body_ancestor_in_path(library, &hit.path)
    else {
        return hit_aabb_in_frame_local(hit, &Path::root());
    };

    let Some(&(_, face_slot)) = hit.path.get(body_index + 1) else {
        return ([0.0; 3], [WORLD_SIZE; 3]);
    };
    let Some(face_index) = (0..6).find(|&f| super::cubesphere::FACE_SLOTS[f] == face_slot) else {
        return ([0.0; 3], [WORLD_SIZE; 3]);
    };
    let face = Face::from_index(face_index as u8);

    let (iu, iv, ir, depth) = face_cell_indices(&hit.path[body_index + 2..]);
    let cells = if depth == 0 { 1.0 } else { 3.0_f32.powi(depth as i32) };
    let un0 = iu as f32 / cells;
    let vn0 = iv as f32 / cells;
    let rn0 = ir as f32 / cells;
    let du = 1.0 / cells;
    let dv = 1.0 / cells;
    let dr = 1.0 / cells;

    let corners = [
        cubesphere_local::face_space_to_body_point(face, un0,      vn0,      rn0,      inner_r, outer_r, WORLD_SIZE),
        cubesphere_local::face_space_to_body_point(face, un0 + du, vn0,      rn0,      inner_r, outer_r, WORLD_SIZE),
        cubesphere_local::face_space_to_body_point(face, un0,      vn0 + dv, rn0,      inner_r, outer_r, WORLD_SIZE),
        cubesphere_local::face_space_to_body_point(face, un0 + du, vn0 + dv, rn0,      inner_r, outer_r, WORLD_SIZE),
        cubesphere_local::face_space_to_body_point(face, un0,      vn0,      rn0 + dr, inner_r, outer_r, WORLD_SIZE),
        cubesphere_local::face_space_to_body_point(face, un0 + du, vn0,      rn0 + dr, inner_r, outer_r, WORLD_SIZE),
        cubesphere_local::face_space_to_body_point(face, un0,      vn0 + dv, rn0 + dr, inner_r, outer_r, WORLD_SIZE),
        cubesphere_local::face_space_to_body_point(face, un0 + du, vn0 + dv, rn0 + dr, inner_r, outer_r, WORLD_SIZE),
    ];
    bounding_box(&corners)
}

pub fn hit_aabb(library: &NodeLibrary, hit: &HitInfo) -> ([f32; 3], [f32; 3]) {
    let mut origin = [0.0f32; 3];
    let mut cell_size = 1.0f32;

    if let Some((body_index, inner_r, outer_r)) =
        find_body_ancestor_in_path(library, &hit.path)
    {
        // Accumulate origin + cell_size by walking the Cartesian
        // portion of the path up to (but not into) the body.
        for &(_, slot) in hit.path.iter().take(body_index) {
            let (x, y, z) = slot_coords(slot);
            origin = [
                origin[0] + x as f32 * cell_size,
                origin[1] + y as f32 * cell_size,
                origin[2] + z as f32 * cell_size,
            ];
            cell_size /= 3.0;
        }
        // The body's own cell lives at `hit.path[body_index].slot`
        // inside the current node. `cell_size` already describes
        // the child's width; origin moves to the body cell's corner.
        let (x, y, z) = slot_coords(hit.path[body_index].1);
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

        let Some(&(_, face_slot)) = hit.path.get(body_index + 1) else {
            return (body_origin, [
                body_origin[0] + body_size,
                body_origin[1] + body_size,
                body_origin[2] + body_size,
            ]);
        };
        let Some(face_index) = (0..6).find(|&f| super::cubesphere::FACE_SLOTS[f] == face_slot) else {
            return (body_origin, [
                body_origin[0] + body_size,
                body_origin[1] + body_size,
                body_origin[2] + body_size,
            ]);
        };
        let face = Face::from_index(face_index as u8);

        let (iu, iv, ir, depth) = face_cell_indices(&hit.path[body_index + 2..]);
        let cells = if depth == 0 { 1.0 } else { 3.0_f32.powi(depth as i32) };
        let u_lo = (iu as f32 / cells) * 2.0 - 1.0;
        let v_lo = (iv as f32 / cells) * 2.0 - 1.0;
        let r_lo_n = ir as f32 / cells;
        let du = 2.0 / cells;
        let dv = 2.0 / cells;
        let drn = 1.0 / cells;
        let r_world_lo = inner_r * body_size + r_lo_n * (outer_r - inner_r) * body_size;
        let dr_world = drn * (outer_r - inner_r) * body_size;
        let corners = block_corners(body_center, face, u_lo, v_lo, r_world_lo, du, dv, dr_world);
        return bounding_box(&corners);
    }

    // Purely Cartesian hit.
    for &(_, slot) in &hit.path {
        let (x, y, z) = slot_coords(slot);
        origin = [
            origin[0] + x as f32 * cell_size,
            origin[1] + y as f32 * cell_size,
            origin[2] + z as f32 * cell_size,
        ];
        cell_size /= 3.0;
    }
    // Undo the last division — the last path entry IS the hit cell.
    cell_size *= 3.0;
    (origin, [origin[0] + cell_size, origin[1] + cell_size, origin[2] + cell_size])
}

fn hit_path_slots(hit: &HitInfo) -> Path {
    let mut path = Path::root();
    for &(_, slot) in &hit.path {
        path.push(slot as u8);
    }
    path
}

/// Accumulate face-subtree slot indices (u, v, r) through the given
/// path slice. Each slot contributes one base-3 digit per axis.
fn face_cell_indices(face_path: &[(super::tree::NodeId, usize)]) -> (u32, u32, u32, u32) {
    let mut iu = 0u32;
    let mut iv = 0u32;
    let mut ir = 0u32;
    let mut depth = 0u32;
    for &(_, slot) in face_path {
        let (us, vs, rs) = slot_coords(slot);
        iu = iu * 3 + us as u32;
        iv = iv * 3 + vs as u32;
        ir = ir * 3 + rs as u32;
        depth += 1;
    }
    (iu, iv, ir, depth)
}

fn bounding_box(corners: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = corners[0];
    let mut max = corners[0];
    for c in &corners[1..] {
        for k in 0..3 {
            if c[k] < min[k] { min[k] = c[k]; }
            if c[k] > max[k] { max[k] = c[k]; }
        }
    }
    (min, max)
}
