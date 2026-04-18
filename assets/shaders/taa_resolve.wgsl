// TAAU resolve: reproject + neighborhood-clamp + blend.
//
// Runs at FULL resolution, reads CURRENT frame at half-resolution.
// Each full-res pixel does:
//
//   1. Nearest-load the current-frame half-res pixel that covers it.
//   2. Reconstruct the world-space hit position from `current_t`.
//   3. Project that world position through the PREVIOUS camera to
//      find where it was on screen last frame (`prev_uv`).
//   4. Bilinear-sample the history texture at `prev_uv`.
//   5. Compute a 3×3 min/max box of the current frame's neighborhood
//      in the half-res texture. Clamp the history color into that
//      box — this is the ghost-killer: a history sample that doesn't
//      resemble any of the current frame's plausible colors at this
//      location gets pulled back in-range.
//   6. Blend: `mix(current, clamped_history, 1 - blend_weight)`.
//   7. Write the result to BOTH the swapchain AND the new history
//      texture (MRT; same value, different format).
//
// When `history_valid == 0u` (warmup / frame-root change), steps 2-6
// are skipped and the current color is written verbatim.
//
// Miss pixels (current_t ≈ 1e20) still reproject correctly: the huge
// t collapses `rel = hit_pos - prev_cam_pos` to a pure direction,
// which is exactly what we want for sky tracking under camera rotation.

struct TaaUniforms {
    // Current frame's camera, in the SAME coord system as last frame
    // (caller ensures this via frame-root invalidation).
    cam_pos: vec3<f32>,
    _pad0: f32,
    cam_forward: vec3<f32>,
    _pad1: f32,
    cam_right: vec3<f32>,
    _pad2: f32,
    cam_up: vec3<f32>,
    cam_fov: f32,
    // Previous frame's camera.
    prev_cam_pos: vec3<f32>,
    _pad3: f32,
    prev_cam_forward: vec3<f32>,
    _pad4: f32,
    prev_cam_right: vec3<f32>,
    _pad5: f32,
    prev_cam_up: vec3<f32>,
    prev_cam_fov: f32,
    // Half-res and full-res dimensions.
    scaled_size: vec2<f32>,
    full_size: vec2<f32>,
    // Blend weight: fraction of current frame mixed in. 0.1 = 90%
    // history / 10% new — standard TAA setting. Lower values smooth
    // more at the cost of lag.
    blend_weight: f32,
    // 0 = history is garbage (seed from current); 1 = usable.
    history_valid: u32,
    _pad6: vec2<f32>,
}

@group(0) @binding(0) var current_color_tex: texture_2d<f32>;
@group(0) @binding(1) var current_t_tex: texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var lin_sampler: sampler;
@group(0) @binding(4) var<uniform> taa: TaaUniforms;

struct ResolveVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_resolve(@builtin(vertex_index) idx: u32) -> ResolveVertexOut {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: ResolveVertexOut;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

struct ResolveOut {
    @location(0) swapchain: vec4<f32>,
    @location(1) history: vec4<f32>,
}

@fragment
fn fs_resolve(in: ResolveVertexOut) -> ResolveOut {
    // Half-res pixel index covering this full-res pixel.
    let scaled_sz_i = vec2<i32>(taa.scaled_size);
    let px_scaled = vec2<i32>(floor(in.uv * taa.scaled_size));
    let px_scaled_clamped = clamp(px_scaled, vec2<i32>(0), scaled_sz_i - vec2<i32>(1));

    let current_color = textureLoad(current_color_tex, px_scaled_clamped, 0).rgb;
    let current_t = textureLoad(current_t_tex, px_scaled_clamped, 0).r;

    var final_color = current_color;

    if taa.history_valid != 0u {
        // Reconstruct this pixel's ray direction under the current camera.
        // Using the full-res uv gives us the sub-pixel center of the
        // half-res sample we just read; accurate to within ~half a full-
        // res pixel, which is below TAA's clamp tolerance.
        let aspect = taa.full_size.x / taa.full_size.y;
        let half_fov_tan = tan(taa.cam_fov * 0.5);
        let ndc_x = (in.uv.x - 0.5) * 2.0 * aspect * half_fov_tan;
        let ndc_y = (0.5 - in.uv.y) * 2.0 * half_fov_tan;
        let ray_dir = taa.cam_forward + taa.cam_right * ndc_x + taa.cam_up * ndc_y;
        let hit_pos = taa.cam_pos + ray_dir * current_t;

        // Project through the previous camera. For a miss (t=1e20)
        // the `rel` vector is dominated by ray_dir * t, so this
        // naturally reduces to direction-only reprojection of the sky.
        let rel = hit_pos - taa.prev_cam_pos;
        let z_prev = dot(rel, taa.prev_cam_forward);

        if z_prev > 1e-4 {
            let prev_half_fov_tan = tan(taa.prev_cam_fov * 0.5);
            let prev_ndc_x = dot(rel, taa.prev_cam_right) / (z_prev * aspect * prev_half_fov_tan);
            let prev_ndc_y = dot(rel, taa.prev_cam_up) / (z_prev * prev_half_fov_tan);
            let prev_uv = vec2<f32>(prev_ndc_x * 0.5 + 0.5, 0.5 - prev_ndc_y * 0.5);

            if prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0 {
                let history = textureSample(history_tex, lin_sampler, prev_uv).rgb;

                // 3×3 neighborhood clamp in RGB. Each sample is a
                // half-res pixel; loads are cheap (nine texture fetches
                // into the already-bound current_color_tex) and this
                // is the single most important quality lever.
                var cur_min = current_color;
                var cur_max = current_color;
                for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                        if dx == 0 && dy == 0 { continue; }
                        let p = clamp(
                            px_scaled_clamped + vec2<i32>(dx, dy),
                            vec2<i32>(0),
                            scaled_sz_i - vec2<i32>(1),
                        );
                        let s = textureLoad(current_color_tex, p, 0).rgb;
                        cur_min = min(cur_min, s);
                        cur_max = max(cur_max, s);
                    }
                }
                let clamped_history = clamp(history, cur_min, cur_max);
                final_color = mix(clamped_history, current_color, taa.blend_weight);
            }
        }
    }

    var out: ResolveOut;
    out.swapchain = vec4<f32>(final_color, 1.0);
    out.history = vec4<f32>(final_color, 1.0);
    return out;
}
