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

    let b = face_cell_bounds(&hit.path[body_index + 2..]);
    let un0 = b.u_lo;
    let vn0 = b.v_lo;
    let rn0 = b.r_lo;
    let du = b.size;
    let dv = b.size;
    let dr = b.size;

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

        let b = face_cell_bounds(&hit.path[body_index + 2..]);
        let u_lo = b.u_lo * 2.0 - 1.0;
        let v_lo = b.v_lo * 2.0 - 1.0;
        let r_lo_n = b.r_lo;
        let du = 2.0 * b.size;
        let dv = 2.0 * b.size;
        let drn = b.size;
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

/// A face-subtree cell, expressed directly in normalized `[0, 1]³`
/// face coordinates as `(u_lo, v_lo, r_lo, size)` — no base-3 integer
/// indices. Computed via Kahan-compensated f32 accumulation so depth
/// is not bounded by any integer type's range (`3^40 ≈ 1.2e19`
/// already exceeds u64; f32 integer accumulation loses precision by
/// depth ~24). The accumulator mirrors the shader's
/// `walk_face_subtree` in `face_walk.wgsl`, so CPU + GPU produce the
/// same bounds at any depth up to `MAX_DEPTH = 63`.
struct FaceCellBounds {
    u_lo: f32,
    v_lo: f32,
    r_lo: f32,
    /// Normalized cell width = `3^(-depth)` in face coords.
    size: f32,
    depth: u32,
}

fn face_cell_bounds(face_path: &[(super::tree::NodeId, usize)]) -> FaceCellBounds {
    // Kahan-compensated running sums per axis. Without compensation,
    // `u_sum += step_size * us` loses the added term to f32 rounding
    // once `step_size` falls below ~`u_sum * eps` — which happens
    // around depth 24 and completely swamps the contribution at
    // deeper layers.
    let mut u_sum: f32 = 0.0; let mut u_comp: f32 = 0.0;
    let mut v_sum: f32 = 0.0; let mut v_comp: f32 = 0.0;
    let mut r_sum: f32 = 0.0; let mut r_comp: f32 = 0.0;
    let mut size: f32 = 1.0;
    let mut depth: u32 = 0;
    for &(_, slot) in face_path {
        let (us, vs, rs) = slot_coords(slot);
        let step_size = size * (1.0 / 3.0);
        kahan_add(&mut u_sum, &mut u_comp, step_size * us as f32);
        kahan_add(&mut v_sum, &mut v_comp, step_size * vs as f32);
        kahan_add(&mut r_sum, &mut r_comp, step_size * rs as f32);
        size = step_size;
        depth += 1;
    }
    FaceCellBounds {
        u_lo: u_sum + u_comp,
        v_lo: v_sum + v_comp,
        r_lo: r_sum + r_comp,
        size,
        depth,
    }
}

#[inline]
fn kahan_add(sum: &mut f32, comp: &mut f32, addend: f32) {
    let y = addend - *comp;
    let t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
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
