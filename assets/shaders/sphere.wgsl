#include "bindings.wgsl"
#include "tree.wgsl"
#include "face_math.wgsl"
#include "ray_prim.wgsl"
#include "face_walk.wgsl"

// Unified sphere march. ONE function — called from
// `march_cartesian`'s CubedSphereBody dispatch and from `march()`
// when the render root itself is a body cell. Both callers pass the
// body cell's origin and size in the CURRENT RENDER FRAME's
// coordinate system. Nothing in this file hardcodes a body-absolute
// constant; `body_origin` and `body_size` are the only geometric
// anchors, and both are guaranteed O(1) in render-frame units
// (frame rescaling keeps them bounded across any zoom depth).
//
// The render frame never roots at a face subtree. When the camera
// has zoomed deep into a face, `with_render_margin` keeps the
// render root at the containing body cell and the logical anchor
// path drives edit/highlight — the shader doesn't need a
// face-rooted entry point.

fn sphere_depth_tint(rn: f32) -> f32 {
    return 0.55 + 0.45 * clamp(rn, 0.0, 1.0);
}

// Per-level bevel contribution. Draws a dark ~1px band at the cell
// edges; returns 1.0 when the cell is too small on screen for a
// visible band (so deep-sub-pixel grid lines don't darken everything).
fn bevel_level(
    un: f32, vn: f32,
    u_lo: f32, v_lo: f32, size: f32,
    cell_px: f32,
) -> f32 {
    if cell_px < 2.0 {
        return 1.0;
    }
    let cell_u = clamp((un - u_lo) / size, 0.0, 1.0);
    let cell_v = clamp((vn - v_lo) / size, 0.0, 1.0);
    let face_edge = min(
        min(cell_u, 1.0 - cell_u),
        min(cell_v, 1.0 - cell_v),
    );
    let band_end = clamp(1.0 / cell_px, 0.0, 0.25);
    let bevel = smoothstep(0.0, band_end, face_edge);
    return 0.78 + 0.22 * bevel;
}

// Multi-level bevel overlay. Stacks bevel contributions from the hit
// cell + its ancestors (3× wider each step) + its sub-cells (3×
// finer each step), skipping any whose on-screen width is
// sub-pixel. Gives a visible voxel grid at every resolvable scale.
fn sphere_bevel_stack(
    un: f32, vn: f32,
    u_lo: f32, v_lo: f32, size: f32,
    reference_scale: f32,
    ray_dist: f32,
    pixel_density: f32,
) -> f32 {
    let safe_dist = max(ray_dist, 1e-6);
    let base_px = size * reference_scale / safe_dist * pixel_density;

    var b: f32 = 1.0;
    b = b * bevel_level(un, vn, u_lo, v_lo, size, base_px);

    // Ancestors: each 3× wider.
    let UP: u32 = 4u;
    var up_u = u_lo;
    var up_v = v_lo;
    var up_s = size;
    var up_px = base_px;
    for (var i: u32 = 0u; i < UP; i = i + 1u) {
        up_s = up_s * 3.0;
        up_u = floor(up_u / up_s) * up_s;
        up_v = floor(up_v / up_s) * up_s;
        up_px = up_px * 3.0;
        b = b * bevel_level(un, vn, up_u, up_v, up_s, up_px);
    }

    // Descendants: each 3× finer, until sub-pixel.
    let DN: u32 = 3u;
    var dn_u = u_lo;
    var dn_v = v_lo;
    var dn_s = size;
    var dn_px = base_px;
    for (var i: u32 = 0u; i < DN; i = i + 1u) {
        let child_s = dn_s * (1.0 / 3.0);
        let child_px = dn_px * (1.0 / 3.0);
        if child_px < 2.0 { break; }
        let u_frac = clamp((un - dn_u) / dn_s, 0.0, 0.9999999);
        let v_frac = clamp((vn - dn_v) / dn_s, 0.0, 0.9999999);
        let u_idx = floor(u_frac * 3.0);
        let v_idx = floor(v_frac * 3.0);
        dn_u = dn_u + u_idx * child_s;
        dn_v = dn_v + v_idx * child_s;
        dn_s = child_s;
        dn_px = child_px;
        b = b * bevel_level(un, vn, dn_u, dn_v, dn_s, dn_px);
    }
    return b;
}

// Per-ray LOD depth cap. A face cell at depth `d` has radial extent
// `shell * (1/3)^(d-1)` in render-frame units; pick `d` so that
// extent projects to at least `LOD_PIXEL_THRESHOLD` pixels at the
// current ray distance. Matches Cartesian's Nyquist gate, which is
// what keeps zoom-invariant rendering working.
fn face_lod_cap(ray_dist: f32, shell_size: f32) -> u32 {
    let pixel_density = uniforms.screen_height
        / (2.0 * tan(camera.fov * 0.5));
    let safe_dist = max(ray_dist, 1e-6);
    let ratio = shell_size * pixel_density
        / (safe_dist * max(LOD_PIXEL_THRESHOLD, 1e-6));
    if ratio <= 1.0 { return 1u; }
    let log3_ratio = log2(ratio) * (1.0 / 1.5849625);
    let d_f = 1.0 + log3_ratio;
    return u32(clamp(d_f, 1.0, f32(MAX_FACE_DEPTH)));
}

// One curved-UVR DDA step through the sphere shell. All geometry
// expressed in the render frame's local coordinates via the
// caller-supplied body cell (`body_origin`, `body_size`).
//
// - The shell is an annulus: inner radius `cs_inner = inner_r *
//   body_size`, outer radius `cs_outer = outer_r * body_size`, both
//   in render-frame units.
// - `oc = ray_origin - cs_center` is the ray's position relative to
//   the body center, also in render-frame units. Bounded by
//   `body_size` in magnitude — no precision leak.
// - The DDA walks curved cells by picking the minimum t to the next
//   u/v radial plane crossing or r spherical shell crossing. All
//   plane normals and sphere centers are computed from `body_origin`
//   / `body_size`, never from a global body-absolute constant.
fn march_sphere_body(
    body_node_idx: u32,
    body_origin: vec3<f32>,
    body_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let cs_outer = outer_r_local * body_size;
    let cs_inner = inner_r_local * body_size;
    let shell = cs_outer - cs_inner;
    if shell <= 0.0 { return result; }

    // Ray-outer-sphere entry. Standard quadratic; `oc` is bounded by
    // `body_size`, so none of the intermediate values overflow.
    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let eps_init = max(shell * 1e-5, 1e-7);
    var t = t_enter + eps_init;
    var steps: u32 = 0u;
    var last_face_axis: u32 = 6u;
    let pixel_density = uniforms.screen_height
        / (2.0 * tan(camera.fov * 0.5));

    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let local = oc + ray_dir * t;
        let r = length(local);
        if r >= cs_outer || r < cs_inner { break; }

        // Pick dominant face from the radial unit direction.
        let n = local / r;
        let face = pick_face(n);
        let n_axis = face_normal(face);
        let u_axis = face_u_axis(face);
        let v_axis = face_v_axis(face);

        // Cube-UV coordinates on the face. axis_dot guards against
        // grazing angles where the projection is ill-conditioned.
        let axis_dot = dot(n, n_axis);
        if axis_dot <= 1e-6 { break; }
        let cube_u = dot(n, u_axis) / axis_dot;
        let cube_v = dot(n, v_axis) / axis_dot;
        let u_ea = cube_to_ea(cube_u);
        let v_ea = cube_to_ea(cube_v);

        let un = clamp((u_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let vn = clamp((v_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let rn = clamp((r - cs_inner) / shell, 0.0, 0.9999999);

        // Walk the face subtree with a per-ray LOD cap.
        let walk_depth = face_lod_cap(t, shell);
        let walk = walk_face_subtree(body_node_idx, face, un, vn, rn, walk_depth);
        let block_id = walk.block;

        if block_id != 0u {
            var hit_normal: vec3<f32>;
            switch last_face_axis {
                case 0u: { hit_normal = -u_axis; }
                case 1u: { hit_normal =  u_axis; }
                case 2u: { hit_normal = -v_axis; }
                case 3u: { hit_normal =  v_axis; }
                case 4u: { hit_normal = -n; }
                case 5u: { hit_normal =  n; }
                default: { hit_normal =  n; }
            }
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let axis_tint = abs(hit_normal.y) * 1.0
                          + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let bevel = sphere_bevel_stack(
                un, vn,
                walk.u_lo, walk.v_lo, walk.size,
                shell, t, pixel_density,
            );
            let depth_tint = sphere_depth_tint(rn);
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            result.color = palette[block_id].rgb
                         * (ambient + diffuse * 0.78)
                         * axis_tint * bevel * depth_tint;
            return result;
        }

        // Empty cell: advance to next cell boundary. Candidates are
        // the four u/v radial planes and two r spherical shells that
        // bound the current walker cell. Every plane normal / sphere
        // center is local to this body cell (passed in by caller);
        // nothing references a global body-absolute constant.
        let u_lo_ea = walk.u_lo * 2.0 - 1.0;
        let u_hi_ea = (walk.u_lo + walk.size) * 2.0 - 1.0;
        let n_u_lo = u_axis - ea_to_cube(u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(u_hi_ea) * n_axis;

        let v_lo_ea = walk.v_lo * 2.0 - 1.0;
        let v_hi_ea = (walk.v_lo + walk.size) * 2.0 - 1.0;
        let n_v_lo = v_axis - ea_to_cube(v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(v_hi_ea) * n_axis;

        let r_lo = cs_inner + walk.r_lo * shell;
        let r_hi = cs_inner + (walk.r_lo + walk.size) * shell;

        var t_next = t_exit + 1.0;
        var winning_axis: u32 = 6u;
        let zero3 = vec3<f32>(0.0);
        let c_u_lo = ray_plane_t(oc, ray_dir, zero3, n_u_lo);
        if c_u_lo > t && c_u_lo < t_next { t_next = c_u_lo; winning_axis = 0u; }
        let c_u_hi = ray_plane_t(oc, ray_dir, zero3, n_u_hi);
        if c_u_hi > t && c_u_hi < t_next { t_next = c_u_hi; winning_axis = 1u; }
        let c_v_lo = ray_plane_t(oc, ray_dir, zero3, n_v_lo);
        if c_v_lo > t && c_v_lo < t_next { t_next = c_v_lo; winning_axis = 2u; }
        let c_v_hi = ray_plane_t(oc, ray_dir, zero3, n_v_hi);
        if c_v_hi > t && c_v_hi < t_next { t_next = c_v_hi; winning_axis = 3u; }
        let c_r_lo = ray_sphere_after(oc, ray_dir, zero3, r_lo, t);
        if c_r_lo > t && c_r_lo < t_next { t_next = c_r_lo; winning_axis = 4u; }
        let c_r_hi = ray_sphere_after(oc, ray_dir, zero3, r_hi, t);
        if c_r_hi > t && c_r_hi < t_next { t_next = c_r_hi; winning_axis = 5u; }

        if t_next >= t_exit { break; }
        last_face_axis = winning_axis;
        // SDF-min-cell reach floor: advance by at least the current
        // cell's radial extent to guarantee we leave the walker cell,
        // preventing stepper stall at deep zoom.
        let t_ulp = max(abs(t) * 1.2e-7, 1e-30);
        let cell_eps = max(shell * walk.size * 1e-3, t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    return result;
}
