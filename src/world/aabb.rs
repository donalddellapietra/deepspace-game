//! World-space AABBs for ray-march hits.
//!
//! With the unified Cartesian walker, every hit is a Cartesian cell —
//! the cubed-sphere body is just a Cartesian node whose children
//! happen to be 6 face subtrees + an interior filler. Face subtree
//! slots are `(u, v, r)` semantically but numerically identical to
//! `(x, y, z)`, so the AABB computation is the same base-3 slot walk
//! for any hit.
//!
//! Two variants handle the two coordinate systems: `hit_aabb` walks
//! world-space (used from outside any frame), `hit_aabb_in_frame_local`
//! expresses the AABB relative to the current render frame so UI can
//! draw it without f32 precision loss at deep layers.

use super::anchor::{Path, WORLD_SIZE};
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

pub fn hit_aabb(_library: &NodeLibrary, hit: &HitInfo) -> ([f32; 3], [f32; 3]) {
    // World-space AABB. Cartesian walk from world root — every cell
    // is a base-3 child regardless of whether its ancestor is a
    // plain Cartesian node, a CubedSphereBody, or a CubedSphereFace.
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
