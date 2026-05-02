//! CPU mirror of the shader's `march_wrapped_planet`. Renders the
//! `WrappedPlane` frame as a sphere of rotated cartesian tangent
//! cubes — no spherical traversal primitive, no compounding
//! precision loss. Per radial layer cy (outermost → innermost):
//! one ray-vs-sphere-shell quadratic, project the entry to
//! (cell_x, cell_z), walk the slab tree to the anchor, transform
//! the ray into the cell's tangent-cube frame, dispatch
//! `cpu_raycast` (pure cartesian DDA) inside the cube. First hit
//! wins.

use super::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeLibrary};

/// Cast a ray through a `WrappedPlane`-rooted slab rendered as a
/// sphere of rotated cartesian tangent cubes.
///
/// `frame_path` is the world-root → slab-root path; it's prepended
/// to the returned `HitInfo`'s path so callers (place / break) get
/// a fully-qualified world-tree path matching the existing flat
/// raycast's output shape.
///
/// `cam_local` and `ray_dir` are in the slab-root frame's local
/// `[0, 2)³` coords.
pub fn cpu_raycast_wrapped_planet(
    library: &NodeLibrary,
    world_root: NodeId,
    frame_path: &[u8],
    cam_local: [f32; 3],
    ray_dir: [f32; 3],
    dims: [u32; 3],
    slab_depth: u8,
    lat_max: f32,
    max_depth: u32,
) -> Option<HitInfo> {
    let cs_center = [1.0_f32, 1.5, 1.5];
    let body_size = 2.0_f32;
    let r_sphere = body_size / (2.0 * std::f32::consts::PI);

    let dir_len = (ray_dir[0] * ray_dir[0]
        + ray_dir[1] * ray_dir[1]
        + ray_dir[2] * ray_dir[2])
        .sqrt()
        .max(1e-6);
    let inv_norm = 1.0 / dir_len;
    let dir = [
        ray_dir[0] * inv_norm,
        ray_dir[1] * inv_norm,
        ray_dir[2] * inv_norm,
    ];

    let oc = [
        cam_local[0] - cs_center[0],
        cam_local[1] - cs_center[1],
        cam_local[2] - cs_center[2],
    ];
    let b = oc[0] * dir[0] + oc[1] * dir[1] + oc[2] * dir[2];
    let oc_dot_oc = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2];
    let c = oc_dot_oc - r_sphere * r_sphere;
    let disc_outer = b * b - c;
    if disc_outer <= 0.0 {
        return None;
    }
    let sq_outer = disc_outer.sqrt();
    let t_exit_sphere = -b + sq_outer;
    if t_exit_sphere <= 0.0 {
        return None;
    }

    let pi = std::f32::consts::PI;
    let lon_step = 2.0 * pi / dims[0] as f32;
    let lat_step = 2.0 * lat_max / dims[2] as f32;
    let shell_thickness = r_sphere * 0.25;
    let r_inner = r_sphere - shell_thickness;
    let r_step = shell_thickness / dims[1] as f32;

    // Walk frame_path from world_root to the slab root once.
    let mut frame_chain: Vec<(NodeId, usize)> = Vec::with_capacity(frame_path.len());
    let mut cur = world_root;
    for &slot in frame_path.iter() {
        let node = library.get(cur)?;
        frame_chain.push((cur, slot as usize));
        match node.children[slot as usize] {
            Child::Node(child) => cur = child,
            _ => return None,
        }
    }
    let slab_root = cur;

    for cy in (0..dims[1] as i32).rev() {
        let r_layer = r_inner + (cy as f32 + 1.0) * r_step;
        let cr = oc_dot_oc - r_layer * r_layer;
        let disc_layer = b * b - cr;
        if disc_layer < 0.0 {
            continue;
        }
        let sq_layer = disc_layer.sqrt();
        let t_layer = -b - sq_layer;
        if t_layer < 0.0 || t_layer > t_exit_sphere {
            continue;
        }

        let pos = [
            cam_local[0] + dir[0] * t_layer,
            cam_local[1] + dir[1] * t_layer,
            cam_local[2] + dir[2] * t_layer,
        ];
        let n = [
            (pos[0] - cs_center[0]) / r_layer,
            (pos[1] - cs_center[1]) / r_layer,
            (pos[2] - cs_center[2]) / r_layer,
        ];
        let lat = n[1].clamp(-1.0, 1.0).asin();
        if lat.abs() > lat_max {
            continue;
        }
        let lon = n[2].atan2(n[0]);
        let u = (lon + pi) / (2.0 * pi);
        let v = (lat + lat_max) / (2.0 * lat_max);
        let cell_x = ((u * dims[0] as f32).floor() as i32).clamp(0, dims[0] as i32 - 1);
        let cell_z = ((v * dims[2] as f32).floor() as i32).clamp(0, dims[2] as i32 - 1);

        // Walk slab tree to (cell_x, cy, cell_z) at slab_depth granularity.
        let mut path = frame_chain.clone();
        let mut idx = slab_root;
        let mut cells_per_slot: i32 = 1;
        for _ in 1..slab_depth {
            cells_per_slot *= 2;
        }
        let mut anchor: Option<NodeId> = None;
        let mut cell_terminated_uniform: Option<u16> = None;
        let mut empty_at_layer = false;
        for level in 0..slab_depth {
            let sx = (cell_x / cells_per_slot).rem_euclid(2);
            let sy = (cy / cells_per_slot).rem_euclid(2);
            let sz = (cell_z / cells_per_slot).rem_euclid(2);
            let slot = slot_index(sx as usize, sy as usize, sz as usize);
            path.push((idx, slot));
            let node = match library.get(idx) {
                Some(n) => n,
                None => {
                    empty_at_layer = true;
                    break;
                }
            };
            match node.children[slot] {
                Child::Empty | Child::EntityRef(_) => {
                    empty_at_layer = true;
                    break;
                }
                Child::Block(bt) => {
                    cell_terminated_uniform = Some(bt);
                    break;
                }
                Child::Node(child) => {
                    if level + 1 < slab_depth {
                        idx = child;
                    } else {
                        anchor = Some(child);
                    }
                }
            }
            cells_per_slot /= 2;
        }
        if empty_at_layer {
            continue;
        }

        // Build per-cell tangent cube basis and transform ray.
        let lat_c = -lat_max + (cell_z as f32 + 0.5) * lat_step;
        let lon_c = -pi + (cell_x as f32 + 0.5) * lon_step;
        let r_c = r_inner + (cy as f32 + 0.5) * r_step;
        let (sl, cl) = lat_c.sin_cos();
        let (so, co) = lon_c.sin_cos();
        let normal_w = [cl * co, sl, cl * so];
        let east_w = [-so, 0.0, co];
        let north_w = [-sl * co, cl, -sl * so];
        let cube_origin = [
            cs_center[0] + r_c * normal_w[0],
            cs_center[1] + r_c * normal_w[1],
            cs_center[2] + r_c * normal_w[2],
        ];
        let east_arc = r_sphere * cl.abs() * lon_step;
        let north_arc = r_sphere * lat_step;
        let cube_side = east_arc.max(north_arc).max(r_step);
        let scale = 2.0 / cube_side;
        let d_origin = [
            cam_local[0] - cube_origin[0],
            cam_local[1] - cube_origin[1],
            cam_local[2] - cube_origin[2],
        ];
        let local_origin = [
            (east_w[0] * d_origin[0] + east_w[1] * d_origin[1] + east_w[2] * d_origin[2]) * scale + 1.0,
            (normal_w[0] * d_origin[0] + normal_w[1] * d_origin[1] + normal_w[2] * d_origin[2]) * scale + 1.0,
            (north_w[0] * d_origin[0] + north_w[1] * d_origin[1] + north_w[2] * d_origin[2]) * scale + 1.0,
        ];
        let local_dir = [
            (east_w[0] * dir[0] + east_w[1] * dir[1] + east_w[2] * dir[2]) * scale,
            (normal_w[0] * dir[0] + normal_w[1] * dir[1] + normal_w[2] * dir[2]) * scale,
            (north_w[0] * dir[0] + north_w[1] * dir[1] + north_w[2] * dir[2]) * scale,
        ];

        if let Some(anchor_idx) = anchor {
            // Non-uniform anchor: dispatch cartesian DDA inside cube.
            let absolute_slab_depth = frame_path.len() as u32 + slab_depth as u32;
            let cube_max_depth = max_depth.saturating_sub(absolute_slab_depth).max(1);
            if let Some(sub_hit) = super::cpu_raycast(
                library, anchor_idx, local_origin, local_dir, cube_max_depth,
            ) {
                for &(parent, slot) in &sub_hit.path {
                    path.push((parent, slot));
                }
                return Some(HitInfo {
                    path,
                    face: sub_hit.face,
                    t: sub_hit.t * inv_norm,
                    place_path: None,
                });
            }
            // Cube missed; try next radial layer.
            continue;
        }

        if cell_terminated_uniform.is_some() {
            // Path terminated at a uniform Block — the cell is one
            // material at slab granularity. Path already points at
            // the slab leaf. Compute t at cube AABB entry for the
            // returned hit distance.
            let inv_local_dir = [
                if local_dir[0].abs() > 1e-8 { 1.0 / local_dir[0] } else { 1e10 },
                if local_dir[1].abs() > 1e-8 { 1.0 / local_dir[1] } else { 1e10 },
                if local_dir[2].abs() > 1e-8 { 1.0 / local_dir[2] } else { 1e10 },
            ];
            let t1 = [
                (0.0 - local_origin[0]) * inv_local_dir[0],
                (0.0 - local_origin[1]) * inv_local_dir[1],
                (0.0 - local_origin[2]) * inv_local_dir[2],
            ];
            let t2 = [
                (3.0 - local_origin[0]) * inv_local_dir[0],
                (3.0 - local_origin[1]) * inv_local_dir[1],
                (3.0 - local_origin[2]) * inv_local_dir[2],
            ];
            let t_enter = t1[0].min(t2[0])
                .max(t1[1].min(t2[1]))
                .max(t1[2].min(t2[2]));
            let t_exit = t1[0].max(t2[0])
                .min(t1[1].max(t2[1]))
                .min(t1[2].max(t2[2]));
            if t_enter < t_exit && t_exit > 0.0 {
                let t_local = t_enter.max(0.0);
                return Some(HitInfo {
                    path,
                    face: 2,
                    t: t_local * inv_norm,
                    place_path: None,
                });
            }
            continue;
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::bootstrap::wrapped_planet_world;

    #[test]
    fn east_equator_hits_middle_cell() {
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 1);
        let frame_path = vec![13u8, 13u8];
        // Ray east of sphere centre going west — hits +X equator point.
        // Expected: lon=0, lat=0 → cell_x=13, cell_z=7, cell_y=1 (grass).
        // Slab leaf slot at (13%3, 1, 7%3) = (1, 1, 1) = slot 13.
        // Path layout: frame(2) + slab(3) + cube descent (>=1) ≥ 6.
        let cam_local = [3.0, 1.5, 1.5];
        let ray_dir = [-1.0, 0.0, 0.0];
        let hit = cpu_raycast_wrapped_planet(
            &world.library, world.root, &frame_path,
            cam_local, ray_dir,
            [27, 2, 14], 3, 1.26, 5,
        ).expect("ray to +X equator must hit");
        assert!(hit.path.len() >= 6, "path = frame(2) + slab(3) + cube ≥ 6, got {}", hit.path.len());
        // Slab leaf level = path[4]: must point at slab cell (13, 1, 7) =
        // slot (1, 1, 1) = 13. The cube descent slots after are local
        // to the rotated tangent cube and depend on rotation precision.
        assert_eq!(hit.path[4].1, slot_index(1, 1, 1),
            "slab leaf must select cell (13, 1, 7)");
    }

    #[test]
    fn pole_is_banned() {
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 1);
        let frame_path = vec![13u8, 13u8];
        let cam_local = [1.5, 3.0, 1.5];
        let ray_dir = [0.0, -1.0, 0.0];
        let hit = cpu_raycast_wrapped_planet(
            &world.library, world.root, &frame_path,
            cam_local, ray_dir,
            [27, 2, 14], 3, 1.26, 5,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn miss_when_ray_misses_sphere() {
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 1);
        let frame_path = vec![13u8, 13u8];
        let cam_local = [2.0, 2.0, 2.0];
        let ray_dir = [0.0, 1.0, 0.0];
        let hit = cpu_raycast_wrapped_planet(
            &world.library, world.root, &frame_path,
            cam_local, ray_dir,
            [27, 2, 14], 3, 1.26, 5,
        );
        assert!(hit.is_none());
    }
}
