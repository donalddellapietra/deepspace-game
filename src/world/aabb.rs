//! World-space AABBs for ray-march hits. Two variants handle the
//! two coordinate systems: `hit_aabb` walks world-space (used from
//! outside any frame), `hit_aabb_in_frame_local` expresses the AABB
//! relative to the current render frame so UI can draw it without
//! f32 precision loss at deep layers.

use super::anchor::{Path, WORLD_SIZE};
use super::raycast::HitInfo;
use super::tree::{slot_coords, NodeLibrary};

pub fn hit_aabb_in_frame_local(hit: &HitInfo, frame_path: &Path) -> ([f32; 3], [f32; 3]) {
    if hit.uv_sphere_cell.is_some() {
        let (world_min, world_max) = hit_aabb_uv_sphere(hit);
        let (frame_min_world, frame_size_world) = path_world_bounds(frame_path);
        let scale = WORLD_SIZE / frame_size_world;
        let min = [
            (world_min[0] - frame_min_world[0]) * scale,
            (world_min[1] - frame_min_world[1]) * scale,
            (world_min[2] - frame_min_world[2]) * scale,
        ];
        let max = [
            (world_max[0] - frame_min_world[0]) * scale,
            (world_max[1] - frame_min_world[1]) * scale,
            (world_max[2] - frame_min_world[2]) * scale,
        ];
        return (min, max);
    }

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
    if hit.uv_sphere_cell.is_some() {
        return hit_aabb_uv_sphere(hit);
    }

    let mut origin = [0.0f32; 3];
    let mut cell_size = 1.0f32;

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

fn path_world_bounds(path: &Path) -> ([f32; 3], f32) {
    let mut origin = [0.0f32; 3];
    let mut cell_size = WORLD_SIZE;
    for depth in 0..path.depth() as usize {
        let (sx, sy, sz) = slot_coords(path.slot(depth) as usize);
        let child_size = cell_size / 3.0;
        origin[0] += sx as f32 * child_size;
        origin[1] += sy as f32 * child_size;
        origin[2] += sz as f32 * child_size;
        cell_size = child_size;
    }
    (origin, cell_size)
}

fn hit_aabb_uv_sphere(hit: &HitInfo) -> ([f32; 3], [f32; 3]) {
    let cell = hit
        .uv_sphere_cell
        .expect("uv sphere aabb requires uv_sphere_cell metadata");
    let mut body_path = Path::root();
    for &(_, slot) in hit.path.iter().take(cell.body_path_len) {
        body_path.push(slot as u8);
    }
    let (body_world_min, body_world_size) = path_world_bounds(&body_path);
    let pow3 = 3_i64.pow(cell.ratio_depth as u32) as f64;
    let phi_lo = cell.ratio_phi as f64 / pow3;
    let theta_lo = cell.ratio_theta as f64 / pow3;
    let r_lo = cell.ratio_r as f64 / pow3;
    let size = 1.0 / pow3;
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for dp in [0.0, size] {
        for dt in [0.0, size] {
            for dr in [0.0, size] {
                let p = crate::world::uvsphere::uv_space_to_body_point_f64(
                    phi_lo + dp,
                    theta_lo + dt,
                    r_lo + dr,
                    cell.inner_r as f64,
                    cell.outer_r as f64,
                    cell.theta_cap as f64,
                    1.0,
                );
                let world = [
                    body_world_min[0] as f64 + p[0] * body_world_size as f64,
                    body_world_min[1] as f64 + p[1] * body_world_size as f64,
                    body_world_min[2] as f64 + p[2] * body_world_size as f64,
                ];
                for axis in 0..3 {
                    min[axis] = min[axis].min(world[axis] as f32);
                    max[axis] = max[axis].max(world[axis] as f32);
                }
            }
        }
    }
    (min, max)
}
