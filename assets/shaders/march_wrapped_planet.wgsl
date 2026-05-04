
// Per-cell tangent-cube basis. A slab cell at index (cell_x, cy, cell_z)
// is rendered as one rotated cartesian cube placed tangent to the
// implied sphere at the cell's (lon_c, lat_c, r_c) center. Basis
// (east, normal, north) and origin/side are derived analytically
// from the integer cell index — no spherical traversal primitive,
// no compounding precision loss.
struct TangentCubeFrame {
    east_w: vec3<f32>,
    normal_w: vec3<f32>,
    north_w: vec3<f32>,
    origin_w: vec3<f32>,
    side: f32,
}

fn tangent_cube_frame_for_cell(
    cell_x: i32, cy: i32, cell_z: i32,
    cs_center: vec3<f32>, r_sphere: f32, lat_max: f32,
    lon_step: f32, lat_step: f32, r_step: f32, r_inner: f32,
) -> TangentCubeFrame {
    let pi = 3.14159265;
    let lat_c = -lat_max + (f32(cell_z) + 0.5) * lat_step;
    let lon_c = -pi + (f32(cell_x) + 0.5) * lon_step;
    let r_c = r_inner + (f32(cy) + 0.5) * r_step;
    let sl = sin(lat_c);
    let cl = cos(lat_c);
    let so = sin(lon_c);
    let co = cos(lon_c);
    var f: TangentCubeFrame;
    f.normal_w = vec3<f32>(cl * co, sl, cl * so);
    f.east_w = vec3<f32>(-so, 0.0, co);
    f.north_w = vec3<f32>(-sl * co, cl, -sl * so);
    f.origin_w = cs_center + r_c * f.normal_w;
    let east_arc = r_sphere * abs(cl) * lon_step;
    let north_arc = r_sphere * lat_step;
    f.side = max(max(east_arc, north_arc), r_step);
    return f;
}

// Render the WrappedPlane as a sphere of rotated cartesian tangent
// cubes. Each visible slab cell is one rotated cube; rendering of
// each cube uses the precision-stable cartesian DDA in
// `march_in_tangent_cube`. There is no spherical primitive: the
// outer logic is a constant-time per-radial-layer entry-seeding
// pass that finds *which* cube the ray enters, and dispatches
// march_in_tangent_cube once per occupied layer. Iterations are
// independent — no per-step compounding precision loss.
//
// Adjacent rotated cubes don't tile perfectly on a sphere, so this
// path leaves small slivers of sky between cubes at extreme zoom.
// That's a deliberate trade — the high-zoom-as-flat-wrapped-plane
// path will eventually take over before the slivers become visible.
fn march_wrapped_planet(
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

    // Renormalise the ray for the entry quadratic. `t` returned to
    // the caller is in the original parameterisation, so multiply
    // by `inv_norm = 1 / |ray_dir_in|`.
    let ray_dir = normalize(ray_dir_in);
    let inv_norm = 1.0 / max(length(ray_dir_in), 1e-6);

    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let r_sphere = body_size / (2.0 * 3.14159265);

    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let oc_dot_oc = dot(oc, oc);
    let c = oc_dot_oc - r_sphere * r_sphere;
    let disc_outer = b * b - c;
    if disc_outer <= 0.0 { return result; }
    let sq_outer = sqrt(disc_outer);
    let t_exit_sphere = -b + sq_outer;
    if t_exit_sphere <= 0.0 { return result; }

    let dims_x = i32(uniforms.slab_dims.x);
    let dims_y = i32(uniforms.slab_dims.y);
    let dims_z = i32(uniforms.slab_dims.z);
    let slab_depth = uniforms.slab_dims.w;
    let pi = 3.14159265;
    let lon_step = 2.0 * pi / f32(dims_x);
    let lat_step = 2.0 * lat_max / f32(dims_z);
    let shell_thickness = r_sphere * 0.25;
    let r_inner = r_sphere - shell_thickness;
    let r_step = shell_thickness / f32(dims_y);

    var cy = dims_y - 1;
    loop {
        if cy < 0 { break; }

        let r_layer = r_inner + (f32(cy) + 1.0) * r_step;
        let cr = oc_dot_oc - r_layer * r_layer;
        let disc_layer = b * b - cr;
        if disc_layer < 0.0 { cy = cy - 1; continue; }
        let sq_layer = sqrt(disc_layer);
        let t_layer = -b - sq_layer;
        if t_layer < 0.0 || t_layer > t_exit_sphere {
            cy = cy - 1; continue;
        }

        let pos = ray_origin + ray_dir * t_layer;
        let n = (pos - cs_center) / r_layer;
        let lat = asin(clamp(n.y, -1.0, 1.0));
        if abs(lat) > lat_max { cy = cy - 1; continue; }
        let lon = atan2(n.z, n.x);
        let u = (lon + pi) / (2.0 * pi);
        let v = (lat + lat_max) / (2.0 * lat_max);
        let cell_x = clamp(i32(floor(u * f32(dims_x))), 0, dims_x - 1);
        let cell_z = clamp(i32(floor(v * f32(dims_z))), 0, dims_z - 1);

        let sample = sample_slab_cell(body_idx, slab_depth, cell_x, cy, cell_z);
        if sample.block_type == 0xFFFEu {
            cy = cy - 1; continue;
        }

        let cube = tangent_cube_frame_for_cell(
            cell_x, cy, cell_z,
            cs_center, r_sphere, lat_max,
            lon_step, lat_step, r_step, r_inner,
        );
        let scale = 3.0 / cube.side;
        let d_origin = ray_origin - cube.origin_w;
        let local_origin = vec3<f32>(
            dot(cube.east_w, d_origin) * scale + 1.5,
            dot(cube.normal_w, d_origin) * scale + 1.5,
            dot(cube.north_w, d_origin) * scale + 1.5,
        );
        let local_dir = vec3<f32>(
            dot(cube.east_w, ray_dir) * scale,
            dot(cube.normal_w, ray_dir) * scale,
            dot(cube.north_w, ray_dir) * scale,
        );

        if sample.tag == 2u {
            let sub = march_in_tangent_cube(sample.child_idx, local_origin, local_dir);
            if sub.hit {
                let local_hit = local_origin + local_dir * sub.t;
                let local_in_cell = clamp(
                    (local_hit - sub.cell_min) / sub.cell_size,
                    vec3<f32>(0.0), vec3<f32>(1.0),
                );
                let local_bevel = cube_face_bevel(local_in_cell, sub.normal);
                var out: HitResult;
                out.hit = true;
                out.t = sub.t * inv_norm;
                out.color = sub.color * (0.7 + 0.3 * local_bevel);
                out.normal = cube.east_w * sub.normal.x
                           + cube.normal_w * sub.normal.y
                           + cube.north_w * sub.normal.z;
                out.frame_level = 0u;
                out.frame_scale = 1.0;
                let hit_world = ray_origin + ray_dir * sub.t;
                out.cell_min = hit_world - vec3<f32>(0.5);
                out.cell_size = 1.0;
                return out;
            }
        } else if sample.tag == 1u {
            // Pack-time uniform-flatten: a Cartesian ancestor of this
            // slab cell collapsed to one Block. Render as a uniform-
            // coloured tangent cube (no subtree to descend).
            let inv_local = vec3<f32>(
                select(1e10, 1.0 / local_dir.x, abs(local_dir.x) > 1e-8),
                select(1e10, 1.0 / local_dir.y, abs(local_dir.y) > 1e-8),
                select(1e10, 1.0 / local_dir.z, abs(local_dir.z) > 1e-8),
            );
            let cube_box = ray_box(local_origin, inv_local, vec3<f32>(0.0), vec3<f32>(3.0));
            if cube_box.t_enter < cube_box.t_exit && cube_box.t_exit > 0.0 {
                let t_local = max(cube_box.t_enter, 0.0);
                let entry_local = local_origin + local_dir * t_local;
                let dx_lo = abs(entry_local.x - 0.0);
                let dx_hi = abs(entry_local.x - 3.0);
                let dy_lo = abs(entry_local.y - 0.0);
                let dy_hi = abs(entry_local.y - 3.0);
                let dz_lo = abs(entry_local.z - 0.0);
                let dz_hi = abs(entry_local.z - 3.0);
                var best = dx_lo;
                var local_normal = vec3<f32>(-1.0, 0.0, 0.0);
                if dx_hi < best { best = dx_hi; local_normal = vec3<f32>(1.0, 0.0, 0.0); }
                if dy_lo < best { best = dy_lo; local_normal = vec3<f32>(0.0, -1.0, 0.0); }
                if dy_hi < best { best = dy_hi; local_normal = vec3<f32>(0.0, 1.0, 0.0); }
                if dz_lo < best { best = dz_lo; local_normal = vec3<f32>(0.0, 0.0, -1.0); }
                if dz_hi < best { best = dz_hi; local_normal = vec3<f32>(0.0, 0.0, 1.0); }
                let local_in_cell = clamp(entry_local / 3.0, vec3<f32>(0.0), vec3<f32>(1.0));
                let local_bevel = cube_face_bevel(local_in_cell, local_normal);
                var out: HitResult;
                out.hit = true;
                out.t = t_local * inv_norm;
                out.color = palette[sample.block_type].rgb * (0.7 + 0.3 * local_bevel);
                out.normal = cube.east_w * local_normal.x
                           + cube.normal_w * local_normal.y
                           + cube.north_w * local_normal.z;
                out.frame_level = 0u;
                out.frame_scale = 1.0;
                let hit_world = ray_origin + ray_dir * t_local;
                out.cell_min = hit_world - vec3<f32>(0.5);
                out.cell_size = 1.0;
                return out;
            }
        }
        cy = cy - 1;
    }

    return result;
}
