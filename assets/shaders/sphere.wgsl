#include "bindings.wgsl"

// Shading helpers shared between the ray marcher and the fragment
// shader. The cubed-sphere DDA (`sphere_in_cell`, `march_face_root`,
// `sphere_in_face_window`) and the face-walker (`walk_face_subtree`,
// `sample_face_node`) were removed — the unified Cartesian walker in
// `march.wgsl` now handles the body cell as a plain 27-ary node, and
// the body's silhouette comes from an analytical ray–sphere test in
// the Cartesian walker's `kind == 1u` branch.

fn face_uv_for_normal(local: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let an = abs(normal);
    if an.x >= an.y && an.x >= an.z {
        return local.yz;
    }
    if an.y >= an.z {
        return local.xz;
    }
    return local.xy;
}

fn cube_face_bevel(local: vec3<f32>, normal: vec3<f32>) -> f32 {
    let uv = face_uv_for_normal(local, normal);
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return smoothstep(0.02, 0.14, edge);
}
