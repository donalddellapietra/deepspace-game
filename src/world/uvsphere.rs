//! UV-sphere geometry + worldgen helpers.
//!
//! A `NodeKind::UvSphereBody` node owns one recursive `(phi, theta, r)`
//! shell. The body's 27 children are the 3×3×3 root cells of that
//! shell; deeper descendants stay `NodeKind::Cartesian` and inherit
//! the same slot interpretation through the dedicated UV-sphere
//! marcher.

use std::f32::consts::{PI, TAU};

use super::sdf::{self, Planet, Vec3};
use super::tree::{
    empty_children, slot_index, Child, NodeId, NodeKind, NodeLibrary,
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct UvPoint {
    pub phi_n: f32,
    pub theta_n: f32,
    pub r_n: f32,
}

#[inline]
fn theta_span(theta_cap: f32) -> f32 {
    (PI - 2.0 * theta_cap).max(1e-6)
}

#[inline]
pub fn uv_space_to_body_point(
    phi_n: f32,
    theta_n: f32,
    r_n: f32,
    inner_r: f32,
    outer_r: f32,
    theta_cap: f32,
    body_size: f32,
) -> Vec3 {
    let center = [body_size * 0.5; 3];
    let radius = (inner_r + r_n * (outer_r - inner_r)) * body_size;
    let phi = phi_n * TAU;
    let theta = theta_cap + theta_n * theta_span(theta_cap);
    let sin_theta = theta.sin();
    let dir = [
        sin_theta * phi.cos(),
        theta.cos(),
        sin_theta * phi.sin(),
    ];
    sdf::add(center, sdf::scale(dir, radius))
}

pub fn uv_space_to_body_point_f64(
    phi_n: f64,
    theta_n: f64,
    r_n: f64,
    inner_r: f64,
    outer_r: f64,
    theta_cap: f64,
    body_size: f64,
) -> [f64; 3] {
    let center = [body_size * 0.5; 3];
    let radius = (inner_r + r_n * (outer_r - inner_r)) * body_size;
    let phi = phi_n * std::f64::consts::TAU;
    let theta = theta_cap + theta_n * (std::f64::consts::PI - 2.0 * theta_cap).max(1e-12);
    let sin_theta = theta.sin();
    let dir = [
        sin_theta * phi.cos(),
        theta.cos(),
        sin_theta * phi.sin(),
    ];
    [
        center[0] + dir[0] * radius,
        center[1] + dir[1] * radius,
        center[2] + dir[2] * radius,
    ]
}

pub fn body_point_to_uv_space(
    point_body: Vec3,
    inner_r: f32,
    outer_r: f32,
    theta_cap: f32,
    body_size: f32,
) -> Option<UvPoint> {
    let center = [body_size * 0.5; 3];
    let offset = sdf::sub(point_body, center);
    let r = sdf::length(offset);
    if r <= 1e-12 {
        return Some(UvPoint {
            phi_n: 0.0,
            theta_n: 0.5,
            r_n: 0.0,
        });
    }
    let mut phi = offset[2].atan2(offset[0]);
    if phi < 0.0 {
        phi += TAU;
    }
    let theta = (offset[1] / r).clamp(-1.0, 1.0).acos();
    let span = theta_span(theta_cap);
    Some(UvPoint {
        phi_n: (phi / TAU).clamp(0.0, 0.9999999),
        theta_n: ((theta - theta_cap) / span).clamp(0.0, 0.9999999),
        r_n: ((r / body_size - inner_r) / (outer_r - inner_r).max(1e-6)).clamp(0.0, 0.9999999),
    })
}

/// Ray–outer-sphere entry time, in body-frame units. `None` if miss.
pub fn ray_outer_sphere_hit(
    ray_origin_body: Vec3,
    ray_dir: Vec3,
    outer_r: f32,
    body_size: f32,
) -> Option<f32> {
    let center = [body_size * 0.5; 3];
    let outer = outer_r * body_size;
    let oc = sdf::sub(ray_origin_body, center);
    let a = sdf::dot(ray_dir, ray_dir);
    if a <= 1e-12 {
        return None;
    }
    let b = 2.0 * sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - outer * outer;
    let disc = b * b - 4.0 * a * c;
    if disc <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let denom = 0.5 / a;
    let t_enter = ((-b - sq) * denom).max(0.0);
    let t_exit = (-b + sq) * denom;
    let t = if t_enter > 0.0 { t_enter } else { t_exit };
    if t > 0.0 { Some(t) } else { None }
}

/// Scan a hit path for the first `UvSphereBody` ancestor. Returns
/// `(path_index, inner_r, outer_r, theta_cap)` where `path_index` is
/// the entry whose child is the body node.
pub fn find_body_ancestor_in_path(
    library: &NodeLibrary,
    hit_path: &[(NodeId, usize)],
) -> Option<(usize, f32, f32, f32)> {
    for (index, &(node_id, slot)) in hit_path.iter().enumerate() {
        let Some(node) = library.get(node_id) else { continue };
        let Child::Node(child_id) = node.children[slot] else { continue };
        let Some(child) = library.get(child_id) else { continue };
        if let NodeKind::UvSphereBody {
            inner_r, outer_r, theta_cap,
        } = child.kind
        {
            return Some((index, inner_r, outer_r, theta_cap));
        }
    }
    None
}

#[derive(Clone, Debug)]
pub struct PlanetSetup {
    pub inner_r: f32,
    pub outer_r: f32,
    pub theta_cap: f32,
    pub depth: u32,
    pub sdf: Planet,
}

pub fn demo_planet() -> PlanetSetup {
    let center: Vec3 = [0.5, 0.5, 0.5];
    let outer_r = 0.45_f32;
    PlanetSetup {
        inner_r: 0.0,
        outer_r,
        theta_cap: 0.0,
        depth: 18,
        sdf: Planet {
            center,
            radius: 0.30,
            noise_scale: 0.0,
            noise_freq: 1.0,
            noise_seed: 0,
            gravity: 9.8,
            influence_radius: outer_r * 2.0,
            surface_block: crate::world::palette::block::GRASS,
            core_block: crate::world::palette::block::STONE,
        },
    }
}

const SDF_DETAIL_LEVELS: u32 = 4;

pub fn insert_uv_sphere_body(
    lib: &mut NodeLibrary,
    inner_r: f32,
    outer_r: f32,
    theta_cap: f32,
    depth: u32,
    sdf: &Planet,
) -> NodeId {
    debug_assert!(0.0 <= inner_r && inner_r < outer_r && outer_r <= 0.5);
    let mut body_children = empty_children();
    let dphi = 1.0 / 3.0;
    let dtheta = 1.0 / 3.0;
    let dr = 1.0 / 3.0;
    for rs in 0..3 {
        for ts in 0..3 {
            for ps in 0..3 {
                body_children[slot_index(ps, ts, rs)] = build_uv_subtree(
                    lib,
                    inner_r,
                    outer_r,
                    theta_cap,
                    ps as f32 * dphi,
                    (ps + 1) as f32 * dphi,
                    ts as f32 * dtheta,
                    (ts + 1) as f32 * dtheta,
                    rs as f32 * dr,
                    (rs + 1) as f32 * dr,
                    depth.saturating_sub(1),
                    depth.saturating_sub(1).min(SDF_DETAIL_LEVELS),
                    sdf,
                );
            }
        }
    }
    lib.insert_with_kind(
        body_children,
        NodeKind::UvSphereBody {
            inner_r,
            outer_r,
            theta_cap,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn build_uv_subtree(
    lib: &mut NodeLibrary,
    inner_r: f32,
    outer_r: f32,
    theta_cap: f32,
    phi_lo: f32,
    phi_hi: f32,
    theta_lo_n: f32,
    theta_hi_n: f32,
    r_lo_n: f32,
    r_hi_n: f32,
    depth: u32,
    sdf_budget: u32,
    sdf: &Planet,
) -> Child {
    let body_size = 1.0f32;
    if can_use_radial_profile(sdf, body_size) {
        return build_radial_uv_subtree(
            lib,
            inner_r,
            outer_r,
            r_lo_n,
            r_hi_n,
            depth,
            sdf,
            body_size,
        );
    }
    let phi_c = 0.5 * (phi_lo + phi_hi);
    let theta_c = 0.5 * (theta_lo_n + theta_hi_n);
    let r_c = 0.5 * (r_lo_n + r_hi_n);
    let p_center = uv_space_to_body_point(
        phi_c,
        theta_c,
        r_c,
        inner_r,
        outer_r,
        theta_cap,
        body_size,
    );
    let d_center = sdf.distance(p_center);
    let shell = outer_r - inner_r;
    let theta_span_radians = theta_span(theta_cap);
    let radial_half = 0.5 * (r_hi_n - r_lo_n) * shell;
    let angular_half = 0.5
        * outer_r
        * ((phi_hi - phi_lo) * TAU)
            .max((theta_hi_n - theta_lo_n) * theta_span_radians);
    let cell_rad = (angular_half * angular_half + radial_half * radial_half).sqrt();

    if d_center > cell_rad {
        if depth == 0 {
            return Child::Empty;
        }
        return Child::Node(uniform_empty_chain(lib, depth));
    }
    if d_center < -cell_rad {
        let b = sdf.block_at(p_center);
        return lib.build_uniform_subtree(b, depth);
    }
    if depth == 0 {
        return if d_center < 0.0 {
            Child::Block(sdf.block_at(p_center))
        } else {
            Child::Empty
        };
    }
    if sdf_budget == 0 {
        return if d_center < 0.0 {
            lib.build_uniform_subtree(sdf.block_at(p_center), depth)
        } else {
            Child::Node(uniform_empty_chain(lib, depth))
        };
    }

    let mut children = empty_children();
    let dphi = (phi_hi - phi_lo) / 3.0;
    let dtheta = (theta_hi_n - theta_lo_n) / 3.0;
    let dr = (r_hi_n - r_lo_n) / 3.0;
    for rs in 0..3 {
        for ts in 0..3 {
            for ps in 0..3 {
                children[slot_index(ps, ts, rs)] = build_uv_subtree(
                    lib,
                    inner_r,
                    outer_r,
                    theta_cap,
                    phi_lo + dphi * ps as f32,
                    phi_lo + dphi * (ps + 1) as f32,
                    theta_lo_n + dtheta * ts as f32,
                    theta_lo_n + dtheta * (ts + 1) as f32,
                    r_lo_n + dr * rs as f32,
                    r_lo_n + dr * (rs + 1) as f32,
                    depth - 1,
                    sdf_budget - 1,
                    sdf,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

fn can_use_radial_profile(sdf: &Planet, body_size: f32) -> bool {
    sdf.noise_scale == 0.0
        && sdf.center == [body_size * 0.5; 3]
}

fn build_radial_uv_subtree(
    lib: &mut NodeLibrary,
    inner_r: f32,
    outer_r: f32,
    r_lo_n: f32,
    r_hi_n: f32,
    depth: u32,
    sdf: &Planet,
    body_size: f32,
) -> Child {
    let shell = outer_r - inner_r;
    let r_lo = (inner_r + r_lo_n * shell) * body_size;
    let r_hi = (inner_r + r_hi_n * shell) * body_size;
    if r_lo >= sdf.radius {
        if depth == 0 {
            return Child::Empty;
        }
        return Child::Node(uniform_empty_chain(lib, depth));
    }
    if r_hi <= sdf.radius {
        return lib.build_uniform_subtree(sdf.block_at(sdf.center), depth);
    }
    if depth == 0 {
        let r_c = 0.5 * (r_lo + r_hi);
        return if r_c < sdf.radius {
            Child::Block(sdf.block_at(sdf.center))
        } else {
            Child::Empty
        };
    }

    let mut radial_children = [Child::Empty; 3];
    let dr = (r_hi_n - r_lo_n) / 3.0;
    for rs in 0..3 {
        radial_children[rs] = build_radial_uv_subtree(
            lib,
            inner_r,
            outer_r,
            r_lo_n + dr * rs as f32,
            r_lo_n + dr * (rs + 1) as f32,
            depth - 1,
            sdf,
            body_size,
        );
    }

    let mut children = empty_children();
    for rs in 0..3 {
        for ts in 0..3 {
            for ps in 0..3 {
                children[slot_index(ps, ts, rs)] = radial_children[rs];
            }
        }
    }
    Child::Node(lib.insert(children))
}

fn uniform_empty_chain(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert([Child::Node(id); 27]);
    }
    id
}

pub fn install_at_root_center(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    setup: &PlanetSetup,
) -> (NodeId, crate::world::anchor::Path) {
    let body_id = insert_uv_sphere_body(
        lib,
        setup.inner_r,
        setup.outer_r,
        setup.theta_cap,
        setup.depth,
        &setup.sdf,
    );
    let host_slot = slot_index(1, 1, 1) as u8;
    let root_node = lib.get(world_root).expect("world root exists");
    let mut children = root_node.children;
    children[host_slot as usize] = Child::Node(body_id);
    let new_root = lib.insert(children);
    let mut body_path = crate::world::anchor::Path::root();
    body_path.push(host_slot);
    (new_root, body_path)
}
