//! World-space AABBs for ray-march hits. Two variants handle the
//! two coordinate systems: `hit_aabb` walks world-space, while
//! `hit_aabb_in_frame_local` expresses the AABB relative to the
//! current render frame for UI precision at deep layers.
//!
//! Sphere hits take a different path: once the hit crosses a
//! `CubedSphereBody` ancestor, the remaining slots are `(u, v, r)`
//! in face-local coords and the AABB comes from projecting the
//! cell's eight corners into world space via
//! `cubesphere::face_space_to_body_point`.

use super::anchor::{Path, WORLD_SIZE};
use super::cubesphere::{self, face_space_to_body_point, find_body_ancestor_in_path, Face, FACE_SLOTS};
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

/// Body-local AABB for a sphere hit: the eight corners of the
/// terminal face-subtree cell, expressed in the body cell's local
/// frame so the highlight box aligns pixel-precisely with the
/// rendered cell regardless of render-frame pop depth.
///
/// Prefers `hit.sphere_cell` (recorded at the shader's exact LOD
/// terminal) over `hit.path` reconstruction — the path can be
/// shorter than the visible cell when the walker terminated early
/// on a uniform subtree, which would produce a coarser AABB than
/// what's visible on screen.
pub fn hit_aabb_body_local(library: &NodeLibrary, hit: &HitInfo) -> ([f32; 3], [f32; 3]) {
    if let Some(cell) = hit.sphere_cell {
        // Body-march path: absolute face-normalized corners.
        // Safe at the shallow depths body march renders cleanly.
        let face = Face::from_index(cell.face as u8);
        let u0 = cell.u_lo;
        let v0 = cell.v_lo;
        let r0 = cell.r_lo;
        let du = cell.size;
        let corners = [
            face_space_to_body_point(face, u0,      v0,      r0,      cell.inner_r, cell.outer_r, WORLD_SIZE),
            face_space_to_body_point(face, u0 + du, v0,      r0,      cell.inner_r, cell.outer_r, WORLD_SIZE),
            face_space_to_body_point(face, u0,      v0 + du, r0,      cell.inner_r, cell.outer_r, WORLD_SIZE),
            face_space_to_body_point(face, u0 + du, v0 + du, r0,      cell.inner_r, cell.outer_r, WORLD_SIZE),
            face_space_to_body_point(face, u0,      v0,      r0 + du, cell.inner_r, cell.outer_r, WORLD_SIZE),
            face_space_to_body_point(face, u0 + du, v0,      r0 + du, cell.inner_r, cell.outer_r, WORLD_SIZE),
            face_space_to_body_point(face, u0,      v0 + du, r0 + du, cell.inner_r, cell.outer_r, WORLD_SIZE),
            face_space_to_body_point(face, u0 + du, v0 + du, r0 + du, cell.inner_r, cell.outer_r, WORLD_SIZE),
        ];
        return bounding_box(&corners);
    }

    // Fallback: path-derived AABB (legacy). Only reached when the
    // hit was produced by a code path that didn't populate
    // sphere_cell, which shouldn't happen after the unified raycast
    // rewrite.
    let Some((body_index, inner_r, outer_r)) =
        find_body_ancestor_in_path(library, &hit.path)
    else {
        return hit_aabb_in_frame_local(hit, &Path::root());
    };

    let Some(&(_, face_slot)) = hit.path.get(body_index + 1) else {
        return ([0.0; 3], [WORLD_SIZE; 3]);
    };
    let Some(face_index) = (0..6).find(|&f| FACE_SLOTS[f] == face_slot) else {
        return ([0.0; 3], [WORLD_SIZE; 3]);
    };
    let face = Face::from_index(face_index as u8);

    let (iu, iv, ir, depth) = face_cell_indices(&hit.path[body_index + 2..]);
    let cells = if depth == 0 { 1.0 } else { 3.0_f32.powi(depth as i32) };
    let un0 = iu as f32 / cells;
    let vn0 = iv as f32 / cells;
    let rn0 = ir as f32 / cells;
    let du = 1.0 / cells;

    let corners = [
        face_space_to_body_point(face, un0,      vn0,      rn0,      inner_r, outer_r, WORLD_SIZE),
        face_space_to_body_point(face, un0 + du, vn0,      rn0,      inner_r, outer_r, WORLD_SIZE),
        face_space_to_body_point(face, un0,      vn0 + du, rn0,      inner_r, outer_r, WORLD_SIZE),
        face_space_to_body_point(face, un0 + du, vn0 + du, rn0,      inner_r, outer_r, WORLD_SIZE),
        face_space_to_body_point(face, un0,      vn0,      rn0 + du, inner_r, outer_r, WORLD_SIZE),
        face_space_to_body_point(face, un0 + du, vn0,      rn0 + du, inner_r, outer_r, WORLD_SIZE),
        face_space_to_body_point(face, un0,      vn0 + du, rn0 + du, inner_r, outer_r, WORLD_SIZE),
        face_space_to_body_point(face, un0 + du, vn0 + du, rn0 + du, inner_r, outer_r, WORLD_SIZE),
    ];
    bounding_box(&corners)
}

pub fn hit_aabb(library: &NodeLibrary, hit: &HitInfo) -> ([f32; 3], [f32; 3]) {
    // Sphere hit: compute world-space box via body-local path + face
    // coords. We reuse `hit_aabb_body_local` and translate by the
    // body cell's world-space origin (walk the Cartesian prefix of
    // the hit path).
    if let Some((body_index, _inner_r, _outer_r)) =
        find_body_ancestor_in_path(library, &hit.path)
    {
        // Accumulate the body cell's origin in world coords by
        // walking the Cartesian prefix (everything before the body).
        let mut origin = [0.0f32; 3];
        let mut cell_size = 1.0f32;
        for &(_, slot) in hit.path.iter().take(body_index) {
            let (x, y, z) = slot_coords(slot);
            origin = [
                origin[0] + x as f32 * cell_size,
                origin[1] + y as f32 * cell_size,
                origin[2] + z as f32 * cell_size,
            ];
            cell_size /= 3.0;
        }
        let (x, y, z) = slot_coords(hit.path[body_index].1);
        let body_origin = [
            origin[0] + x as f32 * cell_size,
            origin[1] + y as f32 * cell_size,
            origin[2] + z as f32 * cell_size,
        ];
        let body_size = cell_size;
        // `hit_aabb_body_local` returns the AABB in body-local coords
        // where the body spans `[0, WORLD_SIZE)^3`; rescale and offset
        // into world coords.
        let scale = body_size / WORLD_SIZE;
        let (bl_min, bl_max) = hit_aabb_body_local(library, hit);
        return (
            [
                body_origin[0] + bl_min[0] * scale,
                body_origin[1] + bl_min[1] * scale,
                body_origin[2] + bl_min[2] * scale,
            ],
            [
                body_origin[0] + bl_max[0] * scale,
                body_origin[1] + bl_max[1] * scale,
                body_origin[2] + bl_max[2] * scale,
            ],
        );
    }

    // Purely Cartesian hit.
    let mut origin = [0.0f32; 3];
    let mut cell_size = 1.0f32;
    for &(_, slot) in &hit.path {
        let (x, y, z) = slot_coords(slot);
        origin = [
            origin[0] + x as f32 * cell_size,
            origin[1] + y as f32 * cell_size,
            origin[2] + z as f32 * cell_size,
        ];
        cell_size /= 3.0;
    }
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

/// Accumulate face-subtree slot indices through the given path slice.
/// Each slot contributes one base-3 digit per (u, v, r) axis.
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

// Silence unused import when sphere-specific helpers are only used
// via module-qualified paths in tests.
#[allow(dead_code)]
fn _cubesphere_used() -> Face { cubesphere::Face::PosX }
