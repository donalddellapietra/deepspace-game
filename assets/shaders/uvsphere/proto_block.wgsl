// Tangent-frame cartesian dispatch.
//
// Used by `march_uv_sphere` whenever its descent encounters a
// `NODE_KIND_CARTESIAN_TANGENT` child. The cell's UV bounds
// `(φ_lo..φ_hi, θ_lo..θ_hi, r_lo..r_hi)` define an oriented bounding
// box (OBB) in body-frame world coords:
//   - centre at body_center + r̂_c · r_c, where (r_c, θ_c, φ_c) is
//     the cell's centre in spherical coords;
//   - basis = (φ̂, θ̂, r̂) at the centre;
//   - half-extents in WORLD units: arc lengths along φ̂, θ̂ and the
//     plain radial Δr.
//
// The ray is transformed into the OBB's local [0, 3]³ frame and
// handed to `march_entity_subtree`, the world's standard cartesian
// DDA. On hit, the face normal is rotated back into world frame
// before returning. On miss (entire subtree empty along the ray),
// returns `result.hit = false` so the caller can fall through to the
// UV march and render whatever body content lies behind.
//
// No hardcoded cell positions or bounds anywhere — the OBB is
// derived purely from what `uv_descend_cell` reports.

fn uv_dispatch_cartesian_tangent(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    body_center: vec3<f32>,
    tangent_node_idx: u32,
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    // Cell centre in spherical coords, basis there.
    let phi_c = 0.5 * (phi_lo + phi_hi);
    let theta_c = 0.5 * (theta_lo + theta_hi);
    let r_c = 0.5 * (r_lo + r_hi);
    let cos_p = cos(phi_c);
    let sin_p = sin(phi_c);
    let cos_t = cos(theta_c);
    let sin_t = sin(theta_c);
    let r_hat     = vec3<f32>( cos_t * cos_p,  sin_t,  cos_t * sin_p);
    let theta_hat = vec3<f32>(-sin_t * cos_p,  cos_t, -sin_t * sin_p);
    let phi_hat   = vec3<f32>(-sin_p,           0.0,    cos_p);
    let center = body_center + r_hat * r_c;

    // Half-extents in WORLD units. Tangential arc-length scaling
    // matches `cell_obb` in `src/world/raycast/proto_obb.rs`.
    let h_phi = 0.5 * (phi_hi - phi_lo) * r_c * cos_t;
    let h_th  = 0.5 * (theta_hi - theta_lo) * r_c;
    let h_r   = 0.5 * (r_hi - r_lo);

    // Transform ray into OBB-local cell-grid coords. Each axis is
    // scaled so the OBB's full extent maps to [0, 3] (one slab per
    // sub-cell). Linear transform → world-ray `t` is identical to
    // OBB-local `t`; only positions and directions are rescaled.
    let to_origin = ray_origin - center;
    let proj_origin = vec3<f32>(
        dot(to_origin, phi_hat),
        dot(to_origin, theta_hat),
        dot(to_origin, r_hat),
    );
    let proj_dir = vec3<f32>(
        dot(ray_dir, phi_hat),
        dot(ray_dir, theta_hat),
        dot(ray_dir, r_hat),
    );
    let extents = vec3<f32>(
        max(h_phi, 1e-12),
        max(h_th,  1e-12),
        max(h_r,   1e-12),
    );
    let local_origin = proj_origin / extents * 1.5 + vec3<f32>(1.5);
    let local_dir = proj_dir / extents * 1.5;

    // Hand off to the world's standard cartesian DDA. Returns hit
    // information in OBB-local coords (palette colour, face normal
    // along OBB axes).
    let sub = march_entity_subtree(tangent_node_idx, local_origin, local_dir);
    if !sub.hit {
        return result;
    }

    // Rotate the OBB-local face normal back into world frame.
    let world_normal = normalize(
        sub.normal.x * phi_hat +
        sub.normal.y * theta_hat +
        sub.normal.z * r_hat
    );

    // `t` from the cartesian DDA is in OBB-local ray units; the
    // linear transform preserves the t parameter, so it equals the
    // world ray t.
    result.hit = true;
    result.t = sub.t;
    result.normal = world_normal;
    result.color = sub.color;
    result.cell_min = ray_origin + ray_dir * sub.t - vec3<f32>(0.5);
    result.cell_size = sub.cell_size;
    return result;
}
