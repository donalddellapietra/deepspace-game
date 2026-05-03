//! CPU mirror of the shader's `march_spherical_wrapped_plane`.
//! Renders the SphericalWrappedPlane frame as a sphere of rotated
//! tangent cubes — same algorithm as the shader: ray-vs-sphere,
//! step through the shell sampling cells via `(lon_idx, lat_idx, r_idx)`,
//! at each unique cell read the stored rotation + cell_offset from
//! the leaf TB and apply the unified TB transform to recurse into
//! `cpu_raycast` (Cartesian DDA on cell content).
//!
//! Used by `frame_aware_raycast` for break / place inside a
//! SphericalWrappedPlane frame. Mirrors the shader 1:1 so the
//! cell the crosshair targets is the same cell the shader shades.

use super::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeLibrary};

#[allow(clippy::too_many_arguments)]
pub fn cpu_raycast_spherical_wrapped_planet(
    library: &NodeLibrary,
    world_root: NodeId,
    frame_path: &[u8],
    cam_local: [f32; 3],
    ray_dir: [f32; 3],
    dims: [u32; 3],
    slab_depth: u8,
    body_radius_wp: f32,
    lat_max: f32,
    max_depth: u32,
) -> Option<HitInfo> {
    let body_origin = [0.0_f32, 0.0, 0.0];
    let body_size = 3.0_f32;
    let cs_center = [
        body_origin[0] + body_size * 0.5,
        body_origin[1] + body_size * 0.5,
        body_origin[2] + body_size * 0.5,
    ];
    let wp_to_render = body_size / 3.0;
    let r_outer_render = body_radius_wp * wp_to_render;

    // Cell extent.
    let mut subgrid: f32 = 1.0;
    for _ in 0..slab_depth {
        subgrid *= 3.0;
    }
    let cell_size_wp = 3.0 / subgrid;
    let cell_size_render = cell_size_wp * wp_to_render;
    let r_inner_render = r_outer_render - dims[1] as f32 * cell_size_render;

    // Normalise ray.
    let dir_len = (ray_dir[0] * ray_dir[0]
        + ray_dir[1] * ray_dir[1]
        + ray_dir[2] * ray_dir[2])
        .sqrt()
        .max(1e-6);
    let inv_norm = 1.0 / dir_len;
    let dir = [ray_dir[0] * inv_norm, ray_dir[1] * inv_norm, ray_dir[2] * inv_norm];

    // Outer sphere intersection.
    let oc = [
        cam_local[0] - cs_center[0],
        cam_local[1] - cs_center[1],
        cam_local[2] - cs_center[2],
    ];
    let b = oc[0] * dir[0] + oc[1] * dir[1] + oc[2] * dir[2];
    let oc_dot_oc = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2];
    let c = oc_dot_oc - r_outer_render * r_outer_render;
    let disc_outer = b * b - c;
    if disc_outer <= 0.0 {
        return None;
    }
    let sq_outer = disc_outer.sqrt();
    let t_far = -b + sq_outer;
    if t_far <= 0.0 {
        return None;
    }
    let t_near = -b - sq_outer;
    let t_start = t_near.max(0.0);
    let t_end = t_far;

    // Walk frame_path from world_root to the SphericalWP node.
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
    let body_node = cur;

    let pi = std::f32::consts::PI;
    let n_steps: u32 = 96;
    let dt = (t_end - t_start) / n_steps as f32;
    let mut last_cx: i32 = -999;
    let mut last_cy: i32 = -999;
    let mut last_cz: i32 = -999;

    for i in 0..n_steps {
        let t = t_start + (i as f32 + 0.5) * dt;
        let pos = [
            cam_local[0] + dir[0] * t,
            cam_local[1] + dir[1] * t,
            cam_local[2] + dir[2] * t,
        ];
        let oc_p = [
            pos[0] - cs_center[0],
            pos[1] - cs_center[1],
            pos[2] - cs_center[2],
        ];
        let r_p = (oc_p[0] * oc_p[0] + oc_p[1] * oc_p[1] + oc_p[2] * oc_p[2]).sqrt();
        if r_p < r_inner_render || r_p > r_outer_render {
            continue;
        }
        let n = [oc_p[0] / r_p, oc_p[1] / r_p, oc_p[2] / r_p];
        let lat = n[1].clamp(-1.0, 1.0).asin();
        if lat.abs() > lat_max {
            continue;
        }
        let lon = n[2].atan2(n[0]);
        let u = (lon + pi) / (2.0 * pi);
        let v = (lat + lat_max) / (2.0 * lat_max);
        let cx = ((u * dims[0] as f32).floor() as i32).clamp(0, dims[0] as i32 - 1);
        let cz = ((v * dims[2] as f32).floor() as i32).clamp(0, dims[2] as i32 - 1);
        let cy = (((r_p - r_inner_render) / cell_size_render).floor() as i32)
            .clamp(0, dims[1] as i32 - 1);

        if cx == last_cx && cy == last_cy && cz == last_cz {
            continue;
        }
        last_cx = cx;
        last_cy = cy;
        last_cz = cz;

        // Walk slab tree to (cx, cy, cz).
        let mut path = frame_chain.clone();
        let mut idx = body_node;
        let mut cells_per_slot: i32 = 1;
        for _ in 1..slab_depth {
            cells_per_slot *= 3;
        }
        let mut anchor: Option<NodeId> = None;
        let mut empty = false;
        for level in 0..slab_depth {
            let sx = (cx / cells_per_slot).rem_euclid(3);
            let sy = (cy / cells_per_slot).rem_euclid(3);
            let sz = (cz / cells_per_slot).rem_euclid(3);
            let slot = slot_index(sx as usize, sy as usize, sz as usize);
            path.push((idx, slot));
            let node = match library.get(idx) {
                Some(n) => n,
                None => {
                    empty = true;
                    break;
                }
            };
            match node.children[slot] {
                Child::Empty | Child::EntityRef(_) | Child::Block(_) => {
                    empty = true;
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
            cells_per_slot /= 3;
        }
        if empty {
            continue;
        }
        let anchor = anchor?;

        // Read cell's stored rotation + cell_offset from the TB.
        let cell_node = library.get(anchor)?;
        let (rotation, cell_offset_arr) = match cell_node.kind {
            crate::world::tree::NodeKind::TangentBlock { rotation, cell_offset } => {
                (rotation, cell_offset)
            }
            _ => continue,
        };

        // Cell content centre in render frame: cell's natural slot
        // centre + cell_offset displacement (matches shader).
        let cell_natural_lower_wp = [
            cx as f32 * cell_size_wp,
            cy as f32 * cell_size_wp,
            cz as f32 * cell_size_wp,
        ];
        let cell_content_centre = [
            body_origin[0]
                + (cell_natural_lower_wp[0] + 0.5 * cell_size_wp) * wp_to_render
                + cell_offset_arr[0] * cell_size_render,
            body_origin[1]
                + (cell_natural_lower_wp[1] + 0.5 * cell_size_wp) * wp_to_render
                + cell_offset_arr[1] * cell_size_render,
            body_origin[2]
                + (cell_natural_lower_wp[2] + 0.5 * cell_size_wp) * wp_to_render
                + cell_offset_arr[2] * cell_size_render,
        ];

        // AABB factor: rotated cube's AABB extent ratio.
        let extent_x = rotation[0][0].abs() + rotation[1][0].abs() + rotation[2][0].abs();
        let extent_y = rotation[0][1].abs() + rotation[1][1].abs() + rotation[2][1].abs();
        let extent_z = rotation[0][2].abs() + rotation[1][2].abs() + rotation[2][2].abs();
        let aabb_factor = extent_x.max(extent_y).max(extent_z).max(1.0);
        let aabb_size_render = cell_size_render * aabb_factor;

        let cell_actual_lower = [
            cell_content_centre[0] - 0.5 * aabb_size_render,
            cell_content_centre[1] - 0.5 * aabb_size_render,
            cell_content_centre[2] - 0.5 * aabb_size_render,
        ];

        // Transform ray into cell-storage frame: subtract cell origin,
        // scale, then apply R^T about pivot 1.5 with NO inscribed-cube
        // shrink (mirror of shader: displaced cells store tb_scale=1).
        // `TbBoundary::from_kind` would apply the shrink, which the
        // shader DOESN'T do for displaced cells — so we apply the
        // rotation manually here.
        let scale = 3.0 / aabb_size_render;
        let lp_origin = [
            (cam_local[0] - cell_actual_lower[0]) * scale,
            (cam_local[1] - cell_actual_lower[1]) * scale,
            (cam_local[2] - cell_actual_lower[2]) * scale,
        ];
        let lp_dir = [dir[0] * scale, dir[1] * scale, dir[2] * scale];
        // R^T · (lp − 1.5) + 1.5  (column-major: r[col][row]).
        let centred = [lp_origin[0] - 1.5, lp_origin[1] - 1.5, lp_origin[2] - 1.5];
        let local_origin = [
            rotation[0][0] * centred[0] + rotation[0][1] * centred[1] + rotation[0][2] * centred[2] + 1.5,
            rotation[1][0] * centred[0] + rotation[1][1] * centred[1] + rotation[1][2] * centred[2] + 1.5,
            rotation[2][0] * centred[0] + rotation[2][1] * centred[1] + rotation[2][2] * centred[2] + 1.5,
        ];
        let local_dir = [
            rotation[0][0] * lp_dir[0] + rotation[0][1] * lp_dir[1] + rotation[0][2] * lp_dir[2],
            rotation[1][0] * lp_dir[0] + rotation[1][1] * lp_dir[1] + rotation[1][2] * lp_dir[2],
            rotation[2][0] * lp_dir[0] + rotation[2][1] * lp_dir[1] + rotation[2][2] * lp_dir[2],
        ];

        // Recurse into the cell's content via cpu_raycast.
        let sub_max_depth = max_depth;
        if let Some(sub_hit) = super::cartesian::cpu_raycast_inner(
            library,
            anchor,
            local_origin,
            local_dir,
            sub_max_depth,
        ) {
            // Sphere-cell mask: cells overlap in render frame, so the
            // hit could be content belonging to a neighbour cell.
            // Reject if the hit's spherical (cx, cy, cz) doesn't match
            // the cell we tested. Mirrors the shader.
            let hit_world = [
                cam_local[0] + dir[0] * sub_hit.t,
                cam_local[1] + dir[1] * sub_hit.t,
                cam_local[2] + dir[2] * sub_hit.t,
            ];
            let oc_h = [
                hit_world[0] - cs_center[0],
                hit_world[1] - cs_center[1],
                hit_world[2] - cs_center[2],
            ];
            let r_h = (oc_h[0] * oc_h[0] + oc_h[1] * oc_h[1] + oc_h[2] * oc_h[2]).sqrt();
            if r_h <= 0.0 {
                continue;
            }
            let n_h = [oc_h[0] / r_h, oc_h[1] / r_h, oc_h[2] / r_h];
            let lat_h = n_h[1].clamp(-1.0, 1.0).asin();
            let lon_h = n_h[2].atan2(n_h[0]);
            let u_h = (lon_h + pi) / (2.0 * pi);
            let v_h = (lat_h + lat_max) / (2.0 * lat_max);
            let cx_h =
                ((u_h * dims[0] as f32).floor() as i32).clamp(0, dims[0] as i32 - 1);
            let cz_h =
                ((v_h * dims[2] as f32).floor() as i32).clamp(0, dims[2] as i32 - 1);
            let cy_h = (((r_h - r_inner_render) / cell_size_render).floor() as i32)
                .clamp(0, dims[1] as i32 - 1);
            if cx_h != cx || cy_h != cy || cz_h != cz {
                continue;
            }

            // Build a fully-qualified path from world_root.
            let mut combined_path = path;
            combined_path.extend(sub_hit.path);
            return Some(HitInfo {
                path: combined_path,
                face: sub_hit.face,
                t: sub_hit.t * inv_norm,
                place_path: None,
            });
        }
    }
    None
}
