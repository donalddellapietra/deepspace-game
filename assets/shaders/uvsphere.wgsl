// UV-sphere body marcher.
//
// MVP rendering: ray-outer-sphere intersection + lat/long visualization
// so you can SEE the spherical parameterization is correct. Each of
// the body's 27 (φ-tier, θ-tier, r-tier) root-level cells gets a
// distinct color, so you can verify the 3³ subdivision aligns with
// the (φ, θ) coordinates as expected.
//
// The actual UV DDA voxel walker lands in the next commit and replaces
// this body; the entry point + dispatch path stays the same.
//
// Body params: read from `node_kinds[uniforms.root_index].param_*`,
// which `GpuNodeKind::from_node_kind` packs from the
// `NodeKind::UvSphereBody { inner_r, outer_r, theta_cap }` variant.
// Radii are in body-local `[0, 1)` units; the render frame spans
// `[0, 3)³` so we scale by `body_size = 3.0`.

#include "bindings.wgsl"

/// Top-level marcher for `ROOT_KIND_UV_SPHERE_BODY`. Camera and ray
/// are in the body's local `[0, 3)³` render frame.
fn march_uv_sphere(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 3.0;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    let body = node_kinds[uniforms.root_index];
    let inner_r_local = bitcast<f32>(body.param_a);
    let outer_r_local = bitcast<f32>(body.param_b);
    let theta_cap     = bitcast<f32>(body.param_c);
    let body_size = 3.0;
    let center = vec3<f32>(body_size * 0.5);
    let outer_r = outer_r_local * body_size;

    // Ray–outer-sphere. Solve A·t² + 2B·t + C = 0 with the full
    // quadratic form (ray_dir is NOT unit length — jittered_ray_dir
    // returns camera.forward + screen offsets, magnitude ≈ 1.0..1.6
    // depending on screen position).
    let oc = ray_origin - center;
    let aa = dot(ray_dir, ray_dir);
    let bb = dot(oc, ray_dir);
    let cc = dot(oc, oc) - outer_r * outer_r;
    let disc = bb * bb - aa * cc;
    if disc <= 0.0 {
        return result;
    }
    let sq = sqrt(disc);
    let inv_a = 1.0 / aa;
    let t_enter = (-bb - sq) * inv_a;
    let t_exit = (-bb + sq) * inv_a;
    var t: f32;
    if t_enter > 0.0001 {
        t = t_enter;
    } else if t_exit > 0.0001 {
        t = t_exit;
    } else {
        return result;
    }

    let hit_pt = ray_origin + ray_dir * t;
    let off = hit_pt - center;
    let r_hit = length(off);
    if r_hit < 1e-6 {
        return result;
    }

    // (φ, θ) at hit. φ ∈ (-π, π] from atan2(z, x); θ from off.y / r.
    let theta = asin(clamp(off.y / r_hit, -1.0, 1.0));
    let phi = atan2(off.z, off.x);
    let two_pi = 6.2831853;

    // Cap rejection — outside `|θ| < θ_cap` the ray hits the polar
    // disk, which we don't voxelize. For MVP these pixels fall
    // through to the sky; a cap impostor is a follow-up.
    if abs(theta) > theta_cap {
        return result;
    }

    // Lat/long visualization: paint each 3³ root-level (φ, θ) cell a
    // distinct color so the spherical grid is visible. r-tier is
    // always 2 (outermost) for the outer shell, so we skip it.
    let phi_norm = (phi + 3.14159265) / two_pi;          // [0, 1)
    let theta_norm = (theta / theta_cap + 1.0) * 0.5;    // [0, 1]
    let phi_tier = clamp(floor(phi_norm * 3.0), 0.0, 2.0);
    let theta_tier = clamp(floor(theta_norm * 3.0), 0.0, 2.0);

    // Hue per (phi_tier, theta_tier): 9 distinct hues. Mild grid
    // lines highlight cell boundaries.
    let phi_frac = fract(phi_norm * 3.0);
    let theta_frac = fract(theta_norm * 3.0);
    let edge_phi = min(phi_frac, 1.0 - phi_frac);
    let edge_theta = min(theta_frac, 1.0 - theta_frac);
    let edge = min(edge_phi, edge_theta);
    let line = 1.0 - smoothstep(0.0, 0.04, edge);

    let base = vec3<f32>(
        (phi_tier + 0.5) / 3.0,
        (theta_tier + 0.5) / 3.0,
        0.5 + 0.2 * sin(phi * 3.0),
    );
    let color = mix(base, vec3<f32>(0.05, 0.05, 0.05), line * 0.6);

    // Outward radial normal.
    let normal = off / r_hit;

    result.hit = true;
    result.t = t;
    result.normal = normal;
    result.color = color;
    result.cell_min = center - vec3<f32>(outer_r);
    result.cell_size = 2.0 * outer_r;
    return result;
}
