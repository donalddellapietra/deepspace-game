// `march_uv_subcell`: dormant.
//
// The sub-cell render path is currently disabled — `compute_render_frame`
// caps UV descent at `MAX_UV_FRAME_DEPTH = 0`, so no `UvSubCell` frames
// reach the renderer. The body-root marcher (`march_uv_sphere`) now
// uses absolute-bound stepping (`uv_next_boundary`) which is
// numerically stable at any descent depth, so the precision-driven
// motivation for sub-cell rendering is gone.
//
// This entry point stays so the dispatch in `march.wgsl` keeps
// compiling against the `ROOT_KIND_UV_SUB_CELL` discriminant; if a
// sub-cell frame ever reaches the GPU (e.g., a future LOD scheme
// raises the cap), it just delegates to the body-root marcher. A
// proper sub-cell implementation would re-derive descent bounds
// against the frame's `(phi_min, theta_min, r_min, dphi, dth, dr)`
// origin, with ribbon-pop on frame exit; that's the work for a
// future iteration.

fn march_uv_subcell(
    frame_node_idx: u32,
    body_node_idx: u32,
    body_inner_r: f32, body_outer_r: f32, body_theta_cap: f32,
    phi_min: f32, theta_min: f32, r_min: f32,
    frame_dphi: f32, frame_dth: f32, frame_dr: f32,
    ray_origin: vec3<f32>, ray_dir: vec3<f32>,
) -> HitResult {
    return march_uv_sphere(body_node_idx, ray_origin, ray_dir);
}
