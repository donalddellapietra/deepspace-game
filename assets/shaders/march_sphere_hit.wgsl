#include "bindings.wgsl"

// Closest-face UV bevel for a (lon, lat, r) cell.
//
// `in_cell` is the ray-hit position as a fractional offset within the
// current cell, on each of the 3 axes (lon, lat, r), in [0, 1]^3.
// Tracking this fraction through descent via the precision-stable
// multiply-by-3 trick (parent_frac * 3 - cell_idx) is what keeps the
// bevel sharp at deep depths — using `(lon_p - lon_lo_c)` directly
// would catastrophic-cancel two near-equal f32s and the bevel
// fractions degrade to noise once cell width drops below f32's
// ~1e-7 absolute precision (around descent depth 10).
//
// Face-arc weights still need physical sizes: we multiply in_cell by
// the cell's step in absolute (lon-radians, lat-radians, r-units),
// scaled by `r * cos(lat)` to get arc length on the sphere. Step
// scalars retain full f32 relative precision through descent (each
// push divides by 3, no subtractions of near-equals).
fn make_sphere_hit(
    pos: vec3<f32>, n_step: vec3<f32>, t_param: f32, inv_norm: f32,
    block_type: u32,
    r: f32, lat_p: f32,
    in_cell: vec3<f32>,
    lon_step_c: f32, lat_step_c: f32, r_step_c: f32,
) -> HitResult {
    var result: HitResult;
    let cos_lat = max(cos(lat_p), 1e-3);
    let arc_lon_lo = r * cos_lat * lon_step_c * in_cell.x;
    let arc_lon_hi = r * cos_lat * lon_step_c * (1.0 - in_cell.x);
    let arc_lat_lo = r * lat_step_c * in_cell.y;
    let arc_lat_hi = r * lat_step_c * (1.0 - in_cell.y);
    let arc_r_lo  = r_step_c * in_cell.z;
    let arc_r_hi  = r_step_c * (1.0 - in_cell.z);
    var best = arc_lon_lo;
    var axis: u32 = 0u;
    if arc_lon_hi < best { best = arc_lon_hi; axis = 0u; }
    if arc_lat_lo < best { best = arc_lat_lo; axis = 1u; }
    if arc_lat_hi < best { best = arc_lat_hi; axis = 1u; }
    if arc_r_lo  < best { best = arc_r_lo;  axis = 2u; }
    if arc_r_hi  < best { best = arc_r_hi;  axis = 2u; }

    var u_in_face: f32;
    var v_in_face: f32;
    if axis == 0u {
        u_in_face = in_cell.y;
        v_in_face = in_cell.z;
    } else if axis == 1u {
        u_in_face = in_cell.x;
        v_in_face = in_cell.z;
    } else {
        u_in_face = in_cell.x;
        v_in_face = in_cell.y;
    }
    let face_edge = min(
        min(u_in_face, 1.0 - u_in_face),
        min(v_in_face, 1.0 - v_in_face),
    );
    let shape = smoothstep(0.02, 0.14, face_edge);
    let bevel_strength = 0.7 + 0.3 * shape;

    result.hit = true;
    result.t = t_param * inv_norm;
    result.color = palette[block_type].rgb * bevel_strength;
    result.normal = n_step;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    // Neutralize main.wgsl::cube_face_bevel — see slab DDA hit comment.
    result.cell_min = pos - vec3<f32>(0.5);
    result.cell_size = 1.0;
    return result;
}
