// Sphere sub-frame DDA — sphere render scoped to a sub-frame inside
// a `WrappedPlane` subtree. The active frame's "node" is a
// Cartesian Node (a sub-cell of the WP at some depth past
// slab_depth); the camera is projected into the sub-frame's local
// rotated+translated coords by the renderer
// (`world::sphere_geom::camera_in_sphere_subframe`), so the ray
// arrives here with bounded magnitudes. Sphere math then operates
// at precision proportional to the sub-frame extent — layer-
// agnostic.
//
// Frame conventions (set by the renderer per-frame in uniforms):
//   * Origin    = sub-frame center.
//   * +x axis   = lon-tangent at sub-frame center.
//   * +y axis   = lat-tangent at sub-frame center.
//   * +z axis   = radial direction (out from sphere center).
//   * Sphere center is at `(0, 0, -r_c)` with `r_c =
//     uniforms.subframe_r.z`; sphere radius is the WP's intrinsic
//     `body_size / (2π)` (body_size = 3 in standard architecture).
//   * Sub-frame's (lat, lon, r) range = `uniforms.subframe_lat_lon`
//     + `uniforms.subframe_r.xy`.
//
// Step 6: stub returns a sentinel hit on sphere-intersect so the
// dispatch path can be verified.
// Step 7+: full DDA with sub-frame-aware boundary primitives
// (meridian / parallel / sphere intersections in sub-frame local
// coords, derived from the absolute geometry by basis projection).

fn sphere_uv_in_subframe(
    sub_node_idx: u32,
    ray_origin: vec3<f32>, ray_dir_in: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let ray_dir = normalize(ray_dir_in);
    let inv_norm = 1.0 / max(length(ray_dir_in), 1e-6);

    // Sphere center in sub-frame local coords: along -z by r_c.
    let r_c = uniforms.subframe_r.z;
    let cs_center = vec3<f32>(0.0, 0.0, -r_c);
    // Sphere radius: body_size / (2π) where body_size = 3 (WP local frame size).
    let r_sphere = 3.0 / (2.0 * 3.14159265);

    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - r_sphere * r_sphere;
    let disc = b * b - c;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    // Stub: paint hit magenta so we can VISUALLY confirm the dispatch
    // path fires. Step 7 replaces this with the full sub-frame DDA.
    let pos = ray_origin + ray_dir * t_enter;
    let off = pos - cs_center;
    let n_step = off / max(length(off), 1e-9);
    result.hit = true;
    result.t = t_enter * inv_norm;
    result.color = vec3<f32>(1.0, 0.0, 1.0);
    result.normal = n_step;
    result.cell_min = pos - vec3<f32>(0.5);
    result.cell_size = 1.0;
    return result;
}
