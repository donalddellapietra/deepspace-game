#include "bindings.wgsl"
#include "ray_prim.wgsl"
#include "march_sphere_pack.wgsl"
#include "march_sphere_hit.wgsl"
#include "march_sphere_anchor.wgsl"

// Phase 3 REVISED — Step A.0+A.1: UV-sphere render of the
// WrappedPlane frame. Replaces the flat slab DDA when
// `uniforms.planet_render.x == 1.0`. The slab data is unchanged;
// what changes is the visual — instead of marching through a flat
// 2:1 rectangle, we ray-intersect an implied sphere of radius
// R = body_size / (2π) (= the slab's wrap circumference, since
// dims_x fully fills the X axis at slab_depth = log_3(dims_x)) and
// return a hit if the ray strikes the surface. Poles past `lat_max`
// (planet_render.y) are banned — the ray returns no-hit so the outer
// loop can pop / skip and the pixel reads as sky.
//
// Step A.0: geometry primitive only.
// Step A.1: map (lon, lat) → slab (cell_x, cell_z) at GRASS row
// (cell_y = dims_y - 1), look up cell, color by block_type.
// Parity-checkerboard tint added so even uniform-grass cells show
// the lat/lon → cell grid alignment unambiguously.
fn sphere_uv_in_cell(
    body_idx: u32,
    body_origin: vec3<f32>,
    body_size: f32,
    ray_origin: vec3<f32>,
    ray_dir_in: vec3<f32>,
    lat_max: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    // Quadratic discriminant assumes unit ray_dir; the marcher
    // passes camera.forward + right·ndc + up·ndc which isn't unit.
    // Renormalise for the intersect — `t` returned to the caller is
    // in the same parameterisation (scaled by 1 / |ray_dir_in|).
    let ray_dir = normalize(ray_dir_in);

    // Sphere center: middle of the body cell.
    // Radius: body_size / (2π) — the slab's circumference is
    // exactly body_size (since dims_x fills X), so R = C / (2π).
    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let r_sphere = body_size / (2.0 * 3.14159265);

    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - r_sphere * r_sphere;
    let disc = b * b - c;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let hit_pos = ray_origin + ray_dir * t_enter;
    let n = (hit_pos - cs_center) / r_sphere;

    // A.3 — Full cell-by-cell DDA in spherical (lon, lat, r) cell
    // coords at slab granularity. At each step, sample the cell; if
    // empty, compute t to ALL 6 cell-boundary surfaces (2 meridians,
    // 2 parallels, 2 spheres) and step to the smallest crossing.
    //
    // Slab Y maps to radial: cy = dims_y - 1 at r_outer, cy = 0 at
    // r_inner. Slab X (longitude) wraps the full 2π; cells span
    // lon ∈ [-π, π]. Slab Z (latitude) is bounded to ±lat_max
    // (poles banned outside this band).
    //
    // On a non-empty slab cell we mirror march_cartesian's recipe:
    //  * tag=1 (uniform-flatten Block) → render Block at slab scale.
    //  * tag=2 (non-uniform Node) → LOD-gate. If sub-cells project
    //    above LOD_PIXEL_THRESHOLD, dispatch sphere_descend_anchor to
    //    walk the anchor's subtree at finer (lon, lat, r) granularity.
    //    Otherwise splat the representative_block at slab scale.
    let dims_x = i32(uniforms.slab_dims.x);
    let dims_y = i32(uniforms.slab_dims.y);
    let dims_z = i32(uniforms.slab_dims.z);
    let slab_depth = uniforms.slab_dims.w;
    let pi = 3.14159265;
    let shell_thickness = r_sphere * 0.25;
    let r_outer = r_sphere;
    let r_inner = r_sphere - shell_thickness;
    let inv_norm = 1.0 / max(length(ray_dir_in), 1e-6);
    let lon_step = 2.0 * pi / f32(dims_x);
    let lat_step = 2.0 * lat_max / f32(dims_z);
    let r_step = shell_thickness / f32(dims_y);

    var t = max(t_enter, 0.0) + 1e-5;
    var iters: u32 = 0u;
    loop {
        if t > t_exit { break; }
        if iters > 256u { break; }
        iters = iters + 1u;

        // Current cell (cell_x, cell_y, cell_z) at the ray's t.
        let pos = ray_origin + ray_dir * t;
        let off = pos - cs_center;
        let r = length(off);
        if r < r_inner - 1e-3 || r > r_outer + 1e-3 { break; }
        let n_step = off / r;
        let lat_p = asin(clamp(n_step.y, -1.0, 1.0));
        if abs(lat_p) > lat_max { break; }
        let lon_p = atan2(n_step.z, n_step.x);

        let u_p = (lon_p + pi) / (2.0 * pi);
        let v_p = (lat_p + lat_max) / (2.0 * lat_max);
        let r_frac = (r - r_inner) / shell_thickness;
        let cell_x = clamp(i32(floor(u_p * f32(dims_x))), 0, dims_x - 1);
        let cell_z = clamp(i32(floor(v_p * f32(dims_z))), 0, dims_z - 1);
        let cell_y = clamp(i32(floor(r_frac * f32(dims_y))), 0, dims_y - 1);

        // Slab cell bounds in (lon, lat, r).
        let lon_lo = -pi + f32(cell_x) * lon_step;
        let lon_hi = lon_lo + lon_step;
        let lat_lo = -lat_max + f32(cell_z) * lat_step;
        let lat_hi = lat_lo + lat_step;
        let r_lo = r_inner + f32(cell_y) * r_step;
        let r_hi = r_lo + r_step;

        let sample = sample_slab_cell(body_idx, slab_depth, cell_x, cell_y, cell_z);
        if sample.block_type != 0xFFFEu {
            // tag=2 (non-uniform anchor) ⇒ recursive descent into the
            // anchor's subtree. NO camera-distance LOD gate: in sphere
            // mode the render frame is locked at WrappedPlane (the
            // camera doesn't deepen its frame as anchor_depth grows
            // the way Cartesian does), so a "cell_size/ray_dist <
            // threshold" check would collapse deep edits to the
            // representative — exactly the bug we're fixing. Descend
            // until tag=1 (uniform-flatten Block leaf) or stack max.
            // Tree structure ALONE drives descent — same principle as
            // Cartesian (the LOD gate there only works because frame
            // shifting keeps cells at "normal voxel size" relative to
            // the camera).
            if sample.tag == 2u {
                // Cartesian-local DDA inside the anchor.
                // Compute the slab cell's curved-boundary exit-t once,
                // BEFORE the descent — pass it in so the descent can
                // clamp any hit that's past the curved cell boundary
                // (the flat-box approximation can let sub-cells stick
                // out past the curved cell at corners; rejecting those
                // hits eliminates the "thin sliver" overhangs at slab
                // boundaries).
                var t_curved_exit = t_exit + 1.0;
                let tn_lon_lo = ray_meridian_t(oc, ray_dir, lon_lo, t);
                if tn_lon_lo > 0.0 && tn_lon_lo < t_curved_exit { t_curved_exit = tn_lon_lo; }
                let tn_lon_hi = ray_meridian_t(oc, ray_dir, lon_hi, t);
                if tn_lon_hi > 0.0 && tn_lon_hi < t_curved_exit { t_curved_exit = tn_lon_hi; }
                let tn_lat_lo = ray_parallel_t(oc, ray_dir, lat_lo, t);
                if tn_lat_lo > 0.0 && tn_lat_lo < t_curved_exit { t_curved_exit = tn_lat_lo; }
                let tn_lat_hi = ray_parallel_t(oc, ray_dir, lat_hi, t);
                if tn_lat_hi > 0.0 && tn_lat_hi < t_curved_exit { t_curved_exit = tn_lat_hi; }
                let tn_r_lo = ray_sphere_after(ray_origin, ray_dir, cs_center, r_lo, t);
                if tn_r_lo > 0.0 && tn_r_lo < t_curved_exit { t_curved_exit = tn_r_lo; }
                let tn_r_hi = ray_sphere_after(ray_origin, ray_dir, cs_center, r_hi, t);
                if tn_r_hi > 0.0 && tn_r_hi < t_curved_exit { t_curved_exit = tn_r_hi; }

                let cell_lon_center = lon_lo + 0.5 * lon_step;
                let cell_lat_center = lat_lo + 0.5 * lat_step;
                let cell_r_center   = r_lo   + 0.5 * r_step;
                let sub = sphere_descend_anchor(
                    sample.child_idx,
                    ray_origin, ray_dir,
                    cs_center, inv_norm,
                    cell_lon_center, cell_lat_center, cell_r_center,
                    lon_step, lat_step, r_step,
                    t,
                    t_curved_exit,
                );
                if sub.hit { return sub; }
                // Descent returned no-hit — either the chord found
                // only empty sub-cells, or the hit was beyond the
                // curved slab boundary. Advance past the slab cell.
                if t_curved_exit >= t_exit { break; }
                t = t_curved_exit + max(r_step * 1e-4, 1e-6);
                continue;
            }
            // tag=1 (uniform-flatten): the entire anchor subtree is
            // one Block — render at slab cell scale, no descent
            // needed (uniform Cartesian subtrees pack-flatten so the
            // shader sees one Block at any zoom).
            //
            // Slab cell is large (~0.077 in lon-radians); subtractions
            // here have plenty of f32 precision. Pass `in_cell` in
            // SLAB-TREE SLOT ORDER (lon, r, lat) — same convention as
            // the descent — so adjacent cells render with consistent
            // bevels and don't seam at slab boundaries.
            let in_cell_slab = vec3<f32>(
                clamp((lon_p - lon_lo) / lon_step, 0.0, 1.0),
                clamp((r     - r_lo)   / r_step,   0.0, 1.0),
                clamp((lat_p - lat_lo) / lat_step, 0.0, 1.0),
            );
            return make_sphere_hit(
                pos, n_step, t, inv_norm, sample.block_type,
                r, lat_p,
                in_cell_slab,
                lon_step, r_step, lat_step,
            );
        }

        // Empty cell — advance to the nearest cell-face crossing.
        var t_next = t_exit + 1.0;
        let t_lon_lo = ray_meridian_t(oc, ray_dir, lon_lo, t);
        if t_lon_lo > 0.0 && t_lon_lo < t_next { t_next = t_lon_lo; }
        let t_lon_hi = ray_meridian_t(oc, ray_dir, lon_hi, t);
        if t_lon_hi > 0.0 && t_lon_hi < t_next { t_next = t_lon_hi; }
        let t_lat_lo = ray_parallel_t(oc, ray_dir, lat_lo, t);
        if t_lat_lo > 0.0 && t_lat_lo < t_next { t_next = t_lat_lo; }
        let t_lat_hi = ray_parallel_t(oc, ray_dir, lat_hi, t);
        if t_lat_hi > 0.0 && t_lat_hi < t_next { t_next = t_lat_hi; }
        let t_r_lo = ray_sphere_after(ray_origin, ray_dir, cs_center, r_lo, t);
        if t_r_lo > 0.0 && t_r_lo < t_next { t_next = t_r_lo; }
        let t_r_hi = ray_sphere_after(ray_origin, ray_dir, cs_center, r_hi, t);
        if t_r_hi > 0.0 && t_r_hi < t_next { t_next = t_r_hi; }

        if t_next >= t_exit { break; }
        t = t_next + max(r_step * 1e-4, 1e-6);
    }

    return result;
}
