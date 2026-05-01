//! UV-sphere shell CPU raycast.

use std::f32::consts::{PI, TAU};

use super::{HitInfo, UvSphereHitCell};
use crate::world::sdf;
use crate::world::tree::{
    slot_index, Child, NodeId, NodeLibrary, EMPTY_NODE, REPRESENTATIVE_EMPTY, UNIFORM_EMPTY,
    UNIFORM_MIXED,
};
use crate::world::uvsphere::body_point_to_uv_space;

const EMPTY_CELL: u16 = REPRESENTATIVE_EMPTY;
const MAX_ITERATIONS: u32 = 4096;

#[derive(Debug, Clone)]
struct UvWalk {
    block: u16,
    phi_lo: f32,
    theta_lo: f32,
    r_lo: f32,
    size: f32,
    ratio_phi: i64,
    ratio_theta: i64,
    ratio_r: i64,
    ratio_depth: u8,
    path: Vec<(NodeId, usize)>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Boundary {
    PhiLo,
    PhiHi,
    ThetaLo,
    ThetaHi,
    RLo,
    RHi,
}

#[allow(clippy::too_many_arguments)]
pub fn uv_raycast(
    library: &NodeLibrary,
    body_root_id: NodeId,
    body_origin: [f32; 3],
    body_size: f32,
    inner_r: f32,
    outer_r: f32,
    theta_cap: f32,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    body_path_prefix: &[(NodeId, usize)],
    max_uv_depth: u32,
) -> Option<HitInfo> {
    let ray_origin_body = [
        ray_origin[0] - body_origin[0],
        ray_origin[1] - body_origin[1],
        ray_origin[2] - body_origin[2],
    ];
    let (t_enter, t_exit) = ray_sphere_segment(ray_origin_body, ray_dir, outer_r * body_size, body_size)?;
    let mut cursor_t = (t_enter.max(0.0) + 1e-4 * body_size).min(t_exit);

    for _ in 0..MAX_ITERATIONS {
        if cursor_t > t_exit {
            return None;
        }
        let point_body = [
            ray_origin_body[0] + ray_dir[0] * cursor_t,
            ray_origin_body[1] + ray_dir[1] * cursor_t,
            ray_origin_body[2] + ray_dir[2] * cursor_t,
        ];
        let uv = body_point_to_uv_space(point_body, inner_r, outer_r, theta_cap, body_size)?;
        let walk = walk_uv_subtree(
            library,
            body_root_id,
            uv.phi_n,
            uv.theta_n,
            uv.r_n,
            max_uv_depth.max(1),
        );
        if walk.block != EMPTY_CELL {
            let mut path = body_path_prefix.to_vec();
            path.extend(walk.path.iter().copied());
            let place_path = next_boundary(
                ray_origin_body,
                ray_dir,
                body_size,
                inner_r,
                outer_r,
                theta_cap,
                &walk,
                cursor_t,
                t_exit,
            )
            .map(|(_, boundary)| {
                build_place_path(
                    library,
                    body_root_id,
                    body_path_prefix,
                    walk.ratio_phi,
                    walk.ratio_theta,
                    walk.ratio_r,
                    walk.ratio_depth,
                    boundary,
                )
            });
            let normal = sphere_normal(point_body, body_size);
            return Some(HitInfo {
                path,
                face: dominant_axis_face(normal),
                t: cursor_t,
                place_path,
                uv_sphere_cell: Some(UvSphereHitCell {
                    phi_lo: walk.phi_lo,
                    theta_lo: walk.theta_lo,
                    r_lo: walk.r_lo,
                    size: walk.size,
                    inner_r,
                    outer_r,
                    theta_cap,
                    body_path_len: body_path_prefix.len(),
                    ratio_phi: walk.ratio_phi,
                    ratio_theta: walk.ratio_theta,
                    ratio_r: walk.ratio_r,
                    ratio_depth: walk.ratio_depth,
                }),
            });
        }

        let Some((next_t, _)) = next_boundary(
            ray_origin_body,
            ray_dir,
            body_size,
            inner_r,
            outer_r,
            theta_cap,
            &walk,
            cursor_t,
            t_exit,
        ) else {
            return None;
        };
        let eps = (1e-4 * body_size / 3.0_f32.powi(walk.ratio_depth as i32)).max(1e-6);
        cursor_t = next_t + eps;
    }

    None
}

fn walk_uv_subtree(
    library: &NodeLibrary,
    root_id: NodeId,
    phi_n: f32,
    theta_n: f32,
    r_n: f32,
    max_depth: u32,
) -> UvWalk {
    let limit = max_depth.max(1);
    let phi_abs = phi_n.clamp(0.0, 0.9999999);
    let theta_abs = theta_n.clamp(0.0, 0.9999999);
    let r_abs = r_n.clamp(0.0, 0.9999999);

    let mut node_id = root_id;
    let mut phi_lo = 0.0f32;
    let mut theta_lo = 0.0f32;
    let mut r_lo = 0.0f32;
    let mut size = 1.0f32;
    let mut ratio_phi: i64 = 0;
    let mut ratio_theta: i64 = 0;
    let mut ratio_r: i64 = 0;
    let mut ratio_depth: u8 = 0;
    let mut path: Vec<(NodeId, usize)> = Vec::with_capacity(limit as usize);

    #[inline]
    fn slot_at(abs_c: f32, lo: f32, child_size: f32) -> usize {
        (((abs_c - lo) / child_size).floor()).clamp(0.0, 2.0) as usize
    }

    for d in 1..=limit {
        let Some(node) = library.get(node_id) else {
            return UvWalk {
                block: EMPTY_CELL,
                phi_lo,
                theta_lo,
                r_lo,
                size,
                ratio_phi,
                ratio_theta,
                ratio_r,
                ratio_depth,
                path,
            };
        };
        let child_size = size / 3.0;
        let ps = slot_at(phi_abs, phi_lo, child_size);
        let ts = slot_at(theta_abs, theta_lo, child_size);
        let rs = slot_at(r_abs, r_lo, child_size);
        let slot = slot_index(ps, ts, rs);
        let child_ratio_phi = ratio_phi * 3 + ps as i64;
        let child_ratio_theta = ratio_theta * 3 + ts as i64;
        let child_ratio_r = ratio_r * 3 + rs as i64;
        let child_ratio_depth = ratio_depth + 1;
        let child_phi_lo = child_ratio_phi as f32 * child_size;
        let child_theta_lo = child_ratio_theta as f32 * child_size;
        let child_r_lo = child_ratio_r as f32 * child_size;
        path.push((node_id, slot));

        match node.children[slot] {
            Child::Empty | Child::EntityRef(_) => {
                return UvWalk {
                    block: EMPTY_CELL,
                    phi_lo: child_phi_lo,
                    theta_lo: child_theta_lo,
                    r_lo: child_r_lo,
                    size: child_size,
                    ratio_phi: child_ratio_phi,
                    ratio_theta: child_ratio_theta,
                    ratio_r: child_ratio_r,
                    ratio_depth: child_ratio_depth,
                    path,
                };
            }
            Child::Block(bt) => {
                return UvWalk {
                    block: bt,
                    phi_lo: child_phi_lo,
                    theta_lo: child_theta_lo,
                    r_lo: child_r_lo,
                    size: child_size,
                    ratio_phi: child_ratio_phi,
                    ratio_theta: child_ratio_theta,
                    ratio_r: child_ratio_r,
                    ratio_depth: child_ratio_depth,
                    path,
                };
            }
            Child::Node(nid) => {
                if let Some(child) = library.get(nid) {
                    match child.uniform_type {
                        UNIFORM_EMPTY => {
                            return UvWalk {
                                block: EMPTY_CELL,
                                phi_lo: child_phi_lo,
                                theta_lo: child_theta_lo,
                                r_lo: child_r_lo,
                                size: child_size,
                                ratio_phi: child_ratio_phi,
                                ratio_theta: child_ratio_theta,
                                ratio_r: child_ratio_r,
                                ratio_depth: child_ratio_depth,
                                path,
                            };
                        }
                        UNIFORM_MIXED => {}
                        block => {
                            return UvWalk {
                                block,
                                phi_lo: child_phi_lo,
                                theta_lo: child_theta_lo,
                                r_lo: child_r_lo,
                                size: child_size,
                                ratio_phi: child_ratio_phi,
                                ratio_theta: child_ratio_theta,
                                ratio_r: child_ratio_r,
                                ratio_depth: child_ratio_depth,
                                path,
                            };
                        }
                    }
                }
                if d == limit {
                    let bt = if let Some(child) = library.get(nid) {
                        match child.uniform_type {
                            UNIFORM_EMPTY => EMPTY_CELL,
                            UNIFORM_MIXED => {
                                let rep = child.representative_block;
                                if rep == REPRESENTATIVE_EMPTY { EMPTY_CELL } else { rep }
                            }
                            b => b,
                        }
                    } else {
                        EMPTY_CELL
                    };
                    return UvWalk {
                        block: bt,
                        phi_lo: child_phi_lo,
                        theta_lo: child_theta_lo,
                        r_lo: child_r_lo,
                        size: child_size,
                        ratio_phi: child_ratio_phi,
                        ratio_theta: child_ratio_theta,
                        ratio_r: child_ratio_r,
                        ratio_depth: child_ratio_depth,
                        path,
                    };
                }
                phi_lo = child_phi_lo;
                theta_lo = child_theta_lo;
                r_lo = child_r_lo;
                size = child_size;
                ratio_phi = child_ratio_phi;
                ratio_theta = child_ratio_theta;
                ratio_r = child_ratio_r;
                ratio_depth = child_ratio_depth;
                node_id = nid;
            }
        }
    }

    UvWalk {
        block: EMPTY_CELL,
        phi_lo,
        theta_lo,
        r_lo,
        size,
        ratio_phi,
        ratio_theta,
        ratio_r,
        ratio_depth,
        path,
    }
}

#[allow(clippy::too_many_arguments)]
fn next_boundary(
    ray_origin_body: [f32; 3],
    ray_dir: [f32; 3],
    body_size: f32,
    inner_r: f32,
    outer_r: f32,
    theta_cap: f32,
    walk: &UvWalk,
    min_t: f32,
    max_t: f32,
) -> Option<(f32, Boundary)> {
    let phi_lo = walk.phi_lo * TAU;
    let phi_hi = (walk.phi_lo + walk.size) * TAU;
    let theta_span = (PI - 2.0 * theta_cap).max(1e-6);
    let theta_lo = theta_cap + walk.theta_lo * theta_span;
    let theta_hi = theta_cap + (walk.theta_lo + walk.size) * theta_span;
    let r_lo = (inner_r + walk.r_lo * (outer_r - inner_r)) * body_size;
    let r_hi = (inner_r + (walk.r_lo + walk.size) * (outer_r - inner_r)) * body_size;

    let mut best: Option<(f32, Boundary)> = None;
    let mut consider = |t: Option<f32>, b: Boundary| {
        let Some(t) = t else { return };
        if !t.is_finite() || t <= min_t + 1e-6 || t > max_t + 1e-6 {
            return;
        }
        if best.map(|(bt, _)| t < bt).unwrap_or(true) {
            best = Some((t, b));
        }
    };

    consider(ray_plane_intersection(ray_origin_body, ray_dir, plane_normal(phi_lo), body_size), Boundary::PhiLo);
    consider(ray_plane_intersection(ray_origin_body, ray_dir, plane_normal(phi_hi), body_size), Boundary::PhiHi);
    consider(ray_theta_intersection(ray_origin_body, ray_dir, theta_lo, body_size), Boundary::ThetaLo);
    consider(ray_theta_intersection(ray_origin_body, ray_dir, theta_hi, body_size), Boundary::ThetaHi);
    if r_lo > 1e-6 {
        consider(ray_sphere_intersection(ray_origin_body, ray_dir, r_lo, body_size), Boundary::RLo);
    }
    consider(ray_sphere_intersection(ray_origin_body, ray_dir, r_hi, body_size), Boundary::RHi);
    best.filter(|(t, boundary)| {
        let p = [
            ray_origin_body[0] + ray_dir[0] * *t,
            ray_origin_body[1] + ray_dir[1] * *t,
            ray_origin_body[2] + ray_dir[2] * *t,
        ];
        let Some(uv) = body_point_to_uv_space(p, inner_r, outer_r, theta_cap, body_size) else {
            return false;
        };
        let eps = walk.size * 1e-3 + 1e-6;
        let phi_delta = (uv.phi_n - walk.phi_lo)
            .abs()
            .min((uv.phi_n + 1.0 - walk.phi_lo).abs());
        match boundary {
            Boundary::PhiLo | Boundary::PhiHi => {
                uv.theta_n >= walk.theta_lo - eps
                    && uv.theta_n <= walk.theta_lo + walk.size + eps
                    && uv.r_n >= walk.r_lo - eps
                    && uv.r_n <= walk.r_lo + walk.size + eps
            }
            Boundary::ThetaLo | Boundary::ThetaHi => {
                phi_delta <= walk.size + eps
                    && uv.r_n >= walk.r_lo - eps
                    && uv.r_n <= walk.r_lo + walk.size + eps
            }
            Boundary::RLo | Boundary::RHi => {
                phi_delta <= walk.size + eps
                    && uv.theta_n >= walk.theta_lo - eps
                    && uv.theta_n <= walk.theta_lo + walk.size + eps
            }
        }
    })
}

fn build_place_path(
    library: &NodeLibrary,
    body_root_id: NodeId,
    body_path_prefix: &[(NodeId, usize)],
    ratio_phi: i64,
    ratio_theta: i64,
    ratio_r: i64,
    depth: u8,
    boundary: Boundary,
) -> Vec<(NodeId, usize)> {
    let cells = 3_i64.pow(depth as u32);
    let mut phi = ratio_phi;
    let mut theta = ratio_theta;
    let mut r = ratio_r;
    match boundary {
        Boundary::PhiLo => phi -= 1,
        Boundary::PhiHi => phi += 1,
        Boundary::ThetaLo => theta -= 1,
        Boundary::ThetaHi => theta += 1,
        Boundary::RLo => r -= 1,
        Boundary::RHi => r += 1,
    }
    phi = phi.rem_euclid(cells);
    theta = theta.clamp(0, cells - 1);
    r = r.clamp(0, cells - 1);

    let mut path = body_path_prefix.to_vec();
    let mut node_id = body_root_id;
    for level in 0..depth {
        let divisor = 3_i64.pow((depth - level - 1) as u32);
        let ps = ((phi / divisor) % 3) as usize;
        let ts = ((theta / divisor) % 3) as usize;
        let rs = ((r / divisor) % 3) as usize;
        let slot = slot_index(ps, ts, rs);
        path.push((node_id, slot));
        node_id = if node_id == EMPTY_NODE {
            EMPTY_NODE
        } else if let Some(node) = library.get(node_id) {
            match node.children[slot] {
                Child::Node(child_id) => child_id,
                _ => EMPTY_NODE,
            }
        } else {
            EMPTY_NODE
        };
    }
    path
}

fn sphere_normal(point_body: [f32; 3], body_size: f32) -> [f32; 3] {
    let center = [body_size * 0.5; 3];
    sdf::normalize(sdf::sub(point_body, center))
}

fn dominant_axis_face(normal: [f32; 3]) -> u32 {
    let ax = normal[0].abs();
    let ay = normal[1].abs();
    let az = normal[2].abs();
    if ax >= ay && ax >= az {
        if normal[0] >= 0.0 { 0 } else { 1 }
    } else if ay >= az {
        if normal[1] >= 0.0 { 2 } else { 3 }
    } else if normal[2] >= 0.0 {
        4
    } else {
        5
    }
}

fn plane_normal(phi: f32) -> [f32; 3] {
    [phi.sin(), 0.0, -phi.cos()]
}

fn ray_plane_intersection(
    ray_origin_body: [f32; 3],
    ray_dir: [f32; 3],
    normal: [f32; 3],
    body_size: f32,
) -> Option<f32> {
    let center = [body_size * 0.5; 3];
    let rel = sdf::sub(ray_origin_body, center);
    let denom = sdf::dot(normal, ray_dir);
    if denom.abs() <= 1e-8 {
        return None;
    }
    Some(-sdf::dot(normal, rel) / denom)
}

fn ray_theta_intersection(
    ray_origin_body: [f32; 3],
    ray_dir: [f32; 3],
    theta: f32,
    body_size: f32,
) -> Option<f32> {
    if (theta - PI * 0.5).abs() <= 1e-6 {
        let center_y = body_size * 0.5;
        if ray_dir[1].abs() <= 1e-8 {
            return None;
        }
        return Some((center_y - ray_origin_body[1]) / ray_dir[1]);
    }
    let center = [body_size * 0.5; 3];
    let p = sdf::sub(ray_origin_body, center);
    let cos2 = theta.cos().powi(2);
    let sin2 = theta.sin().powi(2);
    let a = (ray_dir[0] * ray_dir[0] + ray_dir[2] * ray_dir[2]) * cos2
        - ray_dir[1] * ray_dir[1] * sin2;
    let b = 2.0 * ((p[0] * ray_dir[0] + p[2] * ray_dir[2]) * cos2 - p[1] * ray_dir[1] * sin2);
    let c = (p[0] * p[0] + p[2] * p[2]) * cos2 - p[1] * p[1] * sin2;
    if a.abs() <= 1e-8 {
        return None;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let expected_sign = theta.cos().signum();
    let mut best: Option<f32> = None;
    for t in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
        if !t.is_finite() {
            continue;
        }
        if expected_sign != 0.0 {
            let y = p[1] + ray_dir[1] * t;
            if y.signum() != expected_sign {
                continue;
            }
        }
        if best.map(|bt| t < bt).unwrap_or(true) {
            best = Some(t);
        }
    }
    best
}

fn ray_sphere_intersection(
    ray_origin_body: [f32; 3],
    ray_dir: [f32; 3],
    radius: f32,
    body_size: f32,
) -> Option<f32> {
    let center = [body_size * 0.5; 3];
    let oc = sdf::sub(ray_origin_body, center);
    let a = sdf::dot(ray_dir, ray_dir);
    if a <= 1e-12 {
        return None;
    }
    let b = 2.0 * sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let denom = 0.5 / a;
    let t0 = (-b - sq) * denom;
    let t1 = (-b + sq) * denom;
    [t0, t1]
        .into_iter()
        .filter(|t| t.is_finite())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
}

fn ray_sphere_segment(
    ray_origin_body: [f32; 3],
    ray_dir: [f32; 3],
    radius: f32,
    body_size: f32,
) -> Option<(f32, f32)> {
    let center = [body_size * 0.5; 3];
    let oc = sdf::sub(ray_origin_body, center);
    let a = sdf::dot(ray_dir, ray_dir);
    if a <= 1e-12 {
        return None;
    }
    let b = 2.0 * sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let denom = 0.5 / a;
    Some(((-b - sq) * denom, (-b + sq) * denom))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::raycast::MAX_UV_DEPTH;
    use crate::world::uvsphere;

    #[test]
    fn uv_raycast_hits_centerline_surface_from_above() {
        let mut lib = NodeLibrary::default();
        let setup = uvsphere::demo_planet();
        let body = uvsphere::insert_uv_sphere_body(
            &mut lib,
            setup.inner_r,
            setup.outer_r,
            setup.theta_cap,
            setup.depth,
            &setup.sdf,
        );
        lib.ref_inc(body);

        let ray_origin = [1.5, 2.95, 1.5];
        let ray_dir = [0.0, -1.0, 0.0];
        let (t_enter, t_exit) = ray_sphere_segment(ray_origin, ray_dir, setup.outer_r * 3.0, 3.0)
            .expect("ray should cross outer sphere");
        let cursor_t = (t_enter.max(0.0) + 1e-4 * 3.0).min(t_exit);
        let point_body = [
            ray_origin[0] + ray_dir[0] * cursor_t,
            ray_origin[1] + ray_dir[1] * cursor_t,
            ray_origin[2] + ray_dir[2] * cursor_t,
        ];
        let uv = body_point_to_uv_space(
            point_body,
            setup.inner_r,
            setup.outer_r,
            setup.theta_cap,
            3.0,
        )
        .expect("entry point should map to uv");
        let walk = walk_uv_subtree(&lib, body, uv.phi_n, uv.theta_n, uv.r_n, MAX_UV_DEPTH);
        assert_eq!(walk.block, EMPTY_CELL, "outer shell entry should start in empty space");
        assert_eq!(walk.r_lo, 2.0 / 3.0, "outer shell should be the third radial root band");
        assert_eq!(walk.size, 1.0 / 3.0, "outer shell root band should keep its full radial span");
        assert_eq!(
            next_boundary(
                ray_origin,
                ray_dir,
                3.0,
                setup.inner_r,
                setup.outer_r,
                setup.theta_cap,
                &walk,
                cursor_t,
                t_exit,
            )
            .map(|(_, b)| b),
            Some(Boundary::RLo),
            "the next boundary from the empty outer shell should be the inner radial surface",
        );

        let hit = uv_raycast(
            &lib,
            body,
            [0.0, 0.0, 0.0],
            3.0,
            setup.inner_r,
            setup.outer_r,
            setup.theta_cap,
            ray_origin,
            ray_dir,
            &[],
            MAX_UV_DEPTH,
        );

        assert!(hit.is_some(), "centerline ray from above should hit the UV sphere");
    }

    #[test]
    fn uv_raycast_hits_off_center_surface_from_above() {
        let mut lib = NodeLibrary::default();
        let setup = uvsphere::demo_planet();
        let body = uvsphere::insert_uv_sphere_body(
            &mut lib,
            setup.inner_r,
            setup.outer_r,
            setup.theta_cap,
            setup.depth,
            &setup.sdf,
        );
        lib.ref_inc(body);

        let hit = uv_raycast(
            &lib,
            body,
            [0.0, 0.0, 0.0],
            3.0,
            setup.inner_r,
            setup.outer_r,
            setup.theta_cap,
            [1.8, 2.95, 1.5],
            [0.0, -1.0, 0.0],
            &[],
            MAX_UV_DEPTH,
        );

        assert!(hit.is_some(), "off-center ray from above should still hit the UV sphere");
    }
}
