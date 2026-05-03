//! CPU UV-sphere raycast for `WrappedPlane` storage.
//!
//! This path treats slots below a `WrappedPlane` as `(lon, r, lat)`
//! parameter-space cells. It is intentionally separate from
//! `wrapped_planet.rs`, which renders/raycasts the current tangent-cube
//! sphere approximation.

use std::f32::consts::PI;

use super::HitInfo;
use crate::world::sphere::range::{
    sphere_basis, sphere_radius, sphere_shell_bounds, SphereRange, SPHERE_BODY_SIZE,
};
use crate::world::tree::{slot_index, Child, NodeId, NodeLibrary, REPRESENTATIVE_EMPTY};

pub fn cpu_raycast_sphere_uv(
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
    let center = [1.5_f32, 1.5, 1.5];
    let r_sphere = sphere_radius(SPHERE_BODY_SIZE);
    let (r_inner, _r_outer) = sphere_shell_bounds(SPHERE_BODY_SIZE);
    let shell_thickness = r_sphere - r_inner;

    let dir_len = dot(ray_dir, ray_dir).sqrt().max(1e-6);
    let inv_norm = 1.0 / dir_len;
    let dir = [
        ray_dir[0] * inv_norm,
        ray_dir[1] * inv_norm,
        ray_dir[2] * inv_norm,
    ];

    let oc = sub(cam_local, center);
    let b = dot(oc, dir);
    let oc_len2 = dot(oc, oc);
    let c = oc_len2 - r_sphere * r_sphere;
    let disc = b * b - c;
    if disc <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t_exit = -b + sq;
    if t_exit <= 0.0 {
        return None;
    }

    let mut frame_chain = Vec::with_capacity(frame_path.len());
    let mut slab_root = world_root;
    for &slot in frame_path {
        let node = library.get(slab_root)?;
        frame_chain.push((slab_root, slot as usize));
        match node.children[slot as usize] {
            Child::Node(child) => slab_root = child,
            Child::Empty | Child::Block(_) | Child::EntityRef(_) => return None,
        }
    }

    let lon_step = 2.0 * PI / dims[0] as f32;
    let lat_step = 2.0 * lat_max / dims[2] as f32;
    let r_step = shell_thickness / dims[1] as f32;

    for cy in (0..dims[1] as i32).rev() {
        let r_hi = r_inner + (cy as f32 + 1.0) * r_step;
        let cr = oc_len2 - r_hi * r_hi;
        let disc_layer = b * b - cr;
        if disc_layer < 0.0 {
            continue;
        }
        let t_layer = -b - disc_layer.sqrt();
        if t_layer < 0.0 || t_layer > t_exit {
            continue;
        }

        let pos = add(cam_local, mul(dir, t_layer));
        let n = mul(sub(pos, center), 1.0 / r_hi);
        let lat = n[1].clamp(-1.0, 1.0).asin();
        if lat.abs() > lat_max {
            continue;
        }
        let lon = n[2].atan2(n[0]);
        let cell_x = (((lon + PI) / (2.0 * PI)) * dims[0] as f32)
            .floor()
            .clamp(0.0, dims[0] as f32 - 1.0) as i32;
        let cell_z = (((lat + lat_max) / (2.0 * lat_max)) * dims[2] as f32)
            .floor()
            .clamp(0.0, dims[2] as f32 - 1.0) as i32;

        let mut path = frame_chain.clone();
        let mut idx = slab_root;
        let mut cells_per_slot = 1i32;
        for _ in 1..slab_depth {
            cells_per_slot *= 3;
        }

        let mut anchor = None;
        let mut terminal_block = false;
        let mut empty = false;
        for level in 0..slab_depth {
            let sx = (cell_x / cells_per_slot).rem_euclid(3);
            let sy = (cy / cells_per_slot).rem_euclid(3);
            let sz = (cell_z / cells_per_slot).rem_euclid(3);
            let slot = slot_index(sx as usize, sy as usize, sz as usize);
            path.push((idx, slot));
            let Some(node) = library.get(idx) else {
                empty = true;
                break;
            };
            match node.children[slot] {
                Child::Empty | Child::EntityRef(_) => {
                    empty = true;
                    break;
                }
                Child::Block(_) => {
                    terminal_block = true;
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

        let t = t_layer * inv_norm;
        if terminal_block {
            return Some(HitInfo {
                path,
                face: 2,
                t,
                place_path: None,
            });
        }

        let Some(anchor_idx) = anchor else {
            continue;
        };
        let absolute_slab_depth = frame_path.len() as u32 + slab_depth as u32;
        let extra_depth = max_depth.saturating_sub(absolute_slab_depth);
        if extra_depth == 0 {
            return Some(HitInfo {
                path,
                face: 2,
                t,
                place_path: None,
            });
        }

        let lon_lo = -PI + cell_x as f32 * lon_step;
        let lat_lo = -lat_max + cell_z as f32 * lat_step;
        let r_lo = r_inner + cy as f32 * r_step;
        let r_sample = (r_hi - r_step * 1e-4).max(r_lo);
        if let Some(hit) = descend_parameter_subtree(
            library,
            anchor_idx,
            path,
            lon,
            lat,
            r_sample,
            lon_lo,
            lat_lo,
            r_lo,
            lon_step,
            lat_step,
            r_step,
            extra_depth,
            t,
        ) {
            return Some(hit);
        }
    }

    None
}

pub fn cpu_raycast_sphere_uv_subframe(
    library: &NodeLibrary,
    world_root: NodeId,
    wp_path: &[u8],
    range: SphereRange,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    lat_max: f32,
    max_depth: u32,
) -> Option<HitInfo> {
    let (r_inner, r_outer) = sphere_shell_bounds(SPHERE_BODY_SIZE);
    let shell_thickness = r_outer - r_inner;

    let dir_len = dot(ray_dir, ray_dir).sqrt().max(1e-6);
    let inv_norm = 1.0 / dir_len;
    let dir = [
        ray_dir[0] * inv_norm,
        ray_dir[1] * inv_norm,
        ray_dir[2] * inv_norm,
    ];

    let center = [0.0_f32, 0.0, -range.r_center()];
    let oc = sub(ray_origin, center);
    let b = dot(oc, dir);
    let oc_len2 = dot(oc, oc);
    let c = oc_len2 - r_outer * r_outer;
    let disc = b * b - c;
    if disc <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t_exit = -b + sq;
    if t_exit <= 0.0 {
        return None;
    }

    let mut frame_chain = Vec::with_capacity(wp_path.len());
    let mut slab_root = world_root;
    for &slot in wp_path {
        let node = library.get(slab_root)?;
        frame_chain.push((slab_root, slot as usize));
        match node.children[slot as usize] {
            Child::Node(child) => slab_root = child,
            Child::Empty | Child::Block(_) | Child::EntityRef(_) => return None,
        }
    }

    let dims = range.dims;
    let slab_depth = range.slab_depth;
    let lon_step = 2.0 * PI / dims[0] as f32;
    let lat_step = 2.0 * lat_max / dims[2] as f32;
    let r_step = shell_thickness / dims[1] as f32;

    for cy in (0..dims[1] as i32).rev() {
        let r_hi = r_inner + (cy as f32 + 1.0) * r_step;
        let cr = oc_len2 - r_hi * r_hi;
        let disc_layer = b * b - cr;
        if disc_layer < 0.0 {
            continue;
        }
        let t_layer = -b - disc_layer.sqrt();
        if t_layer < 0.0 || t_layer > t_exit {
            continue;
        }

        let pos = add(ray_origin, mul(dir, t_layer));
        let v_local = sub(pos, center);
        let r = dot(v_local, v_local).sqrt().max(1e-9);
        let sphere_vec = subframe_vec_to_sphere(v_local, range);
        let lat = (sphere_vec[1] / r).clamp(-1.0, 1.0).asin();
        if lat.abs() > lat_max {
            continue;
        }
        let lon = sphere_vec[2].atan2(sphere_vec[0]);
        let cell_x = (((lon + PI) / (2.0 * PI)) * dims[0] as f32)
            .floor()
            .clamp(0.0, dims[0] as f32 - 1.0) as i32;
        let cell_z = (((lat + lat_max) / (2.0 * lat_max)) * dims[2] as f32)
            .floor()
            .clamp(0.0, dims[2] as f32 - 1.0) as i32;

        let Some(hit) = hit_slab_cell(
            library,
            slab_root,
            frame_chain.clone(),
            wp_path.len() as u32,
            slab_depth,
            cell_x,
            cy,
            cell_z,
            lon,
            lat,
            r,
            lon_step,
            lat_step,
            r_step,
            r_inner,
            lat_max,
            max_depth,
            t_layer * inv_norm,
        ) else {
            continue;
        };
        return Some(hit);
    }

    None
}

#[allow(clippy::too_many_arguments)]
fn hit_slab_cell(
    library: &NodeLibrary,
    slab_root: NodeId,
    mut path: Vec<(NodeId, usize)>,
    wp_depth: u32,
    slab_depth: u8,
    cell_x: i32,
    cy: i32,
    cell_z: i32,
    lon: f32,
    lat: f32,
    r: f32,
    lon_step: f32,
    lat_step: f32,
    r_step: f32,
    r_inner: f32,
    lat_max: f32,
    max_depth: u32,
    t: f32,
) -> Option<HitInfo> {
    let mut idx = slab_root;
    let mut cells_per_slot = 1i32;
    for _ in 1..slab_depth {
        cells_per_slot *= 3;
    }

    let mut anchor = None;
    let mut terminal_block = false;
    for level in 0..slab_depth {
        let sx = (cell_x / cells_per_slot).rem_euclid(3);
        let sy = (cy / cells_per_slot).rem_euclid(3);
        let sz = (cell_z / cells_per_slot).rem_euclid(3);
        let slot = slot_index(sx as usize, sy as usize, sz as usize);
        path.push((idx, slot));
        let node = library.get(idx)?;
        match node.children[slot] {
            Child::Empty | Child::EntityRef(_) => return None,
            Child::Block(_) => {
                terminal_block = true;
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

    if terminal_block {
        return Some(HitInfo {
            path,
            face: 2,
            t,
            place_path: None,
        });
    }

    let anchor_idx = anchor?;
    let absolute_slab_depth = wp_depth + slab_depth as u32;
    let extra_depth = max_depth.saturating_sub(absolute_slab_depth);
    if extra_depth == 0 {
        return Some(HitInfo {
            path,
            face: 2,
            t,
            place_path: None,
        });
    }

    let lon_lo = -PI + cell_x as f32 * lon_step;
    let lat_lo = -lat_max + cell_z as f32 * lat_step;
    let r_lo = r_inner + cy as f32 * r_step;
    descend_parameter_subtree(
        library,
        anchor_idx,
        path,
        lon,
        lat,
        r,
        lon_lo,
        lat_lo,
        r_lo,
        lon_step,
        lat_step,
        r_step,
        extra_depth,
        t,
    )
}

fn subframe_vec_to_sphere(v: [f32; 3], range: SphereRange) -> [f32; 3] {
    let (east, north, radial) = sphere_basis(range.lon_center(), range.lat_center());
    [
        east[0] * v[0] + north[0] * v[1] + radial[0] * v[2],
        east[1] * v[0] + north[1] * v[1] + radial[1] * v[2],
        east[2] * v[0] + north[2] * v[1] + radial[2] * v[2],
    ]
}

#[allow(clippy::too_many_arguments)]
fn descend_parameter_subtree(
    library: &NodeLibrary,
    mut node_id: NodeId,
    mut path: Vec<(NodeId, usize)>,
    lon: f32,
    lat: f32,
    r: f32,
    mut lon_lo: f32,
    mut lat_lo: f32,
    mut r_lo: f32,
    mut lon_step: f32,
    mut lat_step: f32,
    mut r_step: f32,
    extra_depth: u32,
    t: f32,
) -> Option<HitInfo> {
    for level in 0..extra_depth {
        let node = library.get(node_id)?;
        if node.representative_block == REPRESENTATIVE_EMPTY {
            return None;
        }

        let sx = ((lon - lon_lo) / (lon_step / 3.0)).floor().clamp(0.0, 2.0) as usize;
        let sy = ((r - r_lo) / (r_step / 3.0)).floor().clamp(0.0, 2.0) as usize;
        let sz = ((lat - lat_lo) / (lat_step / 3.0)).floor().clamp(0.0, 2.0) as usize;
        let slot = slot_index(sx, sy, sz);
        path.push((node_id, slot));

        match node.children[slot] {
            Child::Empty | Child::EntityRef(_) => return None,
            Child::Block(_) => {
                return Some(HitInfo {
                    path,
                    face: 2,
                    t,
                    place_path: None,
                });
            }
            Child::Node(child) => {
                if level + 1 >= extra_depth {
                    return Some(HitInfo {
                        path,
                        face: 2,
                        t,
                        place_path: None,
                    });
                }
                lon_step /= 3.0;
                lat_step /= 3.0;
                r_step /= 3.0;
                lon_lo += sx as f32 * lon_step;
                r_lo += sy as f32 * r_step;
                lat_lo += sz as f32 * lat_step;
                node_id = child;
            }
        }
    }

    None
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn mul(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::bootstrap::{wrapped_planet_spawn, wrapped_planet_world};
    use crate::world::edit::break_block;
    use crate::world::sphere::{camera_in_sphere_subframe, sphere_range_for_path, DEFAULT_SPHERE_LAT_MAX};
    use crate::world::tree::slot_index;

    #[test]
    fn east_equator_hits_uv_middle_cell() {
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 1);
        let frame_path = vec![13u8, 13u8];
        let hit = cpu_raycast_sphere_uv(
            &world.library,
            world.root,
            &frame_path,
            [3.0, 1.5, 1.5],
            [-1.0, 0.0, 0.0],
            [27, 2, 14],
            3,
            1.26,
            5,
        )
        .expect("ray to +X equator must hit");

        assert!(
            hit.path.len() >= 5,
            "path should include frame + slab path: {:?}",
            hit.path
        );
        assert_eq!(
            hit.path[4].1,
            slot_index(1, 1, 1),
            "slab leaf must select UV cell (13, 1, 7)"
        );
    }

    #[test]
    fn pole_is_banned() {
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 1);
        let frame_path = vec![13u8, 13u8];
        let hit = cpu_raycast_sphere_uv(
            &world.library,
            world.root,
            &frame_path,
            [1.5, 3.0, 1.5],
            [0.0, -1.0, 0.0],
            [27, 2, 14],
            3,
            1.26,
            5,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn subframe_raycast_breaks_spawn_surface() {
        let mut world = wrapped_planet_world(2, [27, 2, 14], 3, 1);
        let mut wp_path = crate::world::anchor::Path::root();
        wp_path.push(13);
        wp_path.push(13);
        let range = sphere_range_for_path(
            &world.library,
            world.root,
            &wp_path,
            DEFAULT_SPHERE_LAT_MAX,
        )
        .expect("wrapped planet root has sphere range");
        let cam = wrapped_planet_spawn(2, [27, 2, 14], 3);
        let sub = camera_in_sphere_subframe(
            &cam,
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            &wp_path,
            range,
        );
        let hit = cpu_raycast_sphere_uv_subframe(
            &world.library,
            world.root,
            wp_path.as_slice(),
            range,
            sub.origin,
            sub.forward,
            DEFAULT_SPHERE_LAT_MAX,
            7,
        )
        .expect("spawn camera should hit the UV sphere surface");

        assert!(
            hit.path.len() > wp_path.depth() as usize + range.slab_depth as usize,
            "hit must descend below the slab cell for subblock edits: {:?}",
            hit.path
        );
        let old_root = world.root;
        assert!(break_block(&mut world, &hit), "sphere UV hit should be editable");
        assert_ne!(world.root, old_root, "subblock edit should replace the world root");
    }
}
