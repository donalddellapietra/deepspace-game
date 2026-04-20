// Ray march shader for base-3 recursive voxel tree.

#include "bindings.wgsl"
#include "ray_prim.wgsl"
#include "sphere.wgsl"
#include "march.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

/// Compute the ray direction for `uv` under the current camera,
/// with the per-frame sub-pixel jitter baked in. Used by both
/// fragment entry points so the ray-march runs on the jittered
/// sample regardless of whether TAAU is the active resolve path.
/// Jitter is zero when TAAU is off, so the non-TAA path behaves
/// exactly like before.
fn jittered_ray_dir(uv: vec2<f32>) -> vec3<f32> {
    let aspect = uniforms.screen_width / uniforms.screen_height;
    let half_fov_tan = tan(camera.fov * 0.5);
    // Jitter is specified in output-texel units; convert to NDC.
    let jitter_ndc_x = camera.jitter_x_px / uniforms.screen_width * 2.0 * aspect * half_fov_tan;
    let jitter_ndc_y = camera.jitter_y_px / uniforms.screen_height * 2.0 * half_fov_tan;
    let ndc = vec2<f32>(
        (uv.x - 0.5) * 2.0 * aspect * half_fov_tan + jitter_ndc_x,
        (0.5 - uv.y) * 2.0 * half_fov_tan + jitter_ndc_y,
    );
    return camera.forward + camera.right * ndc.x + camera.up * ndc.y;
}

/// Shared pixel-shading kernel. Returns both the (gamma-corrected)
/// RGB color that fs_main writes to the swapchain and the raw
/// HitResult so callers that need the hit t (fs_main_taa) can pass
/// it through without re-running the march.
///
/// Color is the same value `fs_main` used to return — including the
/// manual `pow(1/2.2)` gamma correction it applies before write. TAAU
/// uses this same value so visual output stays consistent with the
/// non-TAAU path; the gamma-space clamp + blend in the resolve shader
/// is a small precision hit but not a visible one.
fn shade_pixel(uv: vec2<f32>) -> vec4<f32> {
    let ray_dir = jittered_ray_dir(uv);
    let result = march(camera.pos, ray_dir);

    var color: vec3<f32>;
    if result.hit {
        let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
        let diffuse = max(dot(result.normal, sun_dir), 0.0);
        let ambient = 0.3;
        let hit_pos = camera.pos + ray_dir * result.t;
        let local = clamp((hit_pos - result.cell_min) / result.cell_size, vec3<f32>(0.0), vec3<f32>(1.0));
        let bevel = cube_face_bevel(local, result.normal);
        let lit = result.color * (ambient + diffuse * 0.7) * (0.7 + 0.3 * bevel);
        color = pow(lit, vec3<f32>(1.0 / 2.2));
    } else {
        let sky_t = ray_dir.y * 0.5 + 0.5;
        color = mix(vec3<f32>(0.7, 0.8, 0.95), vec3<f32>(0.3, 0.5, 0.85), sky_t);
    }

    if uniforms.highlight_active != 0u {
        let h_min = uniforms.highlight_min.xyz;
        let h_max = uniforms.highlight_max.xyz;
        let h_size = h_max - h_min;
        if result.hit {
            let hit_pos = camera.pos + ray_dir * result.t;
            let pad_local = max_component(h_size) * 0.03;
            let inside = all(hit_pos >= (h_min - vec3<f32>(pad_local))) &&
                         all(hit_pos <= (h_max + vec3<f32>(pad_local)));
            if inside {
                let local_h = clamp((hit_pos - h_min) / max(h_size, vec3<f32>(1e-6)), vec3<f32>(0.0), vec3<f32>(1.0));
                let edge = min(
                    min(min(local_h.x, 1.0 - local_h.x), min(local_h.y, 1.0 - local_h.y)),
                    min(local_h.z, 1.0 - local_h.z)
                );
                let glow = 1.0 - smoothstep(0.02, 0.12, edge);
                color = mix(color, vec3<f32>(1.0, 0.92, 0.18), glow * 0.85);
            }
        }
    }

    // The crosshair reticle lives in the HTML overlay — see
    // `ui/src/components/Crosshair.tsx` and
    // `src/app/edit_actions/highlight.rs`. Rendering it in the shader
    // would bake it into the ray-march framebuffer, which either
    // sits at half-res under TAAU (blurring the 1-pixel strokes) or
    // aliases against the jitter sequence (pixel crawl on slow
    // motion). DOM overlay is always at physical resolution and
    // composites cleanly on top; it's the SOTA separation.

    // Emit per-ray stats to the shader_stats buffer. Gated behind
    // the `ENABLE_STATS` override so the off-state has zero runtime
    // cost.
    if ENABLE_STATS {
        atomicAdd(&shader_stats.ray_count, 1u);
        if result.hit {
            atomicAdd(&shader_stats.hit_count, 1u);
        } else {
            atomicAdd(&shader_stats.miss_count, 1u);
        }
        if ray_steps >= 2048u {
            atomicAdd(&shader_stats.max_iter_count, 1u);
        }
        atomicAdd(&shader_stats.sum_steps_div4, (ray_steps + 3u) >> 2u);
        atomicMax(&shader_stats.max_steps, ray_steps);
        if result.hit {
            atomicAdd(&shader_stats.sum_steps_hits_div4, (ray_steps + 3u) >> 2u);
        }
        atomicAdd(&shader_stats.sum_steps_oob_div4, (ray_steps_oob + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_steps_empty_div4, (ray_steps_empty + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_steps_node_descend_div4, (ray_steps_node_descend + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_steps_lod_terminal_div4, (ray_steps_lod_terminal + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_steps_would_cull_div4, (ray_steps_would_cull + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_loads_tree_div4, (ray_loads_tree + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_loads_offsets_div4, (ray_loads_offsets + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_loads_kinds_div4, (ray_loads_kinds + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_loads_ribbon_div4, (ray_loads_ribbon + 3u) >> 2u);
    }

    // Stash t in the alpha channel so fs_main_taa can route it into
    // a second render attachment without re-running the march.
    return vec4<f32>(color, result.t);
}

/// Sky color for a miss ray. Extracted so the mask-cull path in
/// fs_main can return it without running the full march.
fn sky_color(ray_dir: vec3<f32>) -> vec3<f32> {
    let sky_t = ray_dir.y * 0.5 + 0.5;
    return mix(vec3<f32>(0.7, 0.8, 0.95), vec3<f32>(0.3, 0.5, 0.85), sky_t);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Beam-prepass cull. The coarse pass (fs_coarse_mask) marks
    // tiles that might hit content (1.0) vs definitely sky (0.0).
    // Sample a 5-tap neighborhood (center + 4 cardinal) so tiles
    // adjacent to a hit stay in the full-march path — this is the
    // on-the-fly conservative dilation that keeps silhouettes from
    // eroding.
    //
    // When every tap reads 0, no ray near this tile hit anything.
    // Early-out with the sky color and skip the register-heavy
    // `march()` entirely. Saves the dominant cost for the ~88 % of
    // rays that miss on Jerusalem nucleus.
    // Fast-path: check the center tap first. If it's 1.0 (either
    // because P1 is disabled and the mask is cleared to 1, OR
    // because the coarse pass marked this tile as a hit), we
    // already know we're marching — skip the 4 neighbor loads and
    // fall through to shade_pixel.
    //
    // Only when the center is 0 (definite miss per the coarse pass)
    // do we sample the 4 cardinal neighbors for conservative
    // dilation. In that case 4 more texture loads are still much
    // cheaper than a full march(), so it's a worthwhile check.
    //
    // Net effect: disabled-P1 pays 1 texture load per pixel (cheap);
    // enabled-P1 with hit-tiles pays 1 load per pixel; enabled-P1
    // with miss-tiles pays 5 loads per pixel but skips the march.
    let tile = vec2<i32>(in.position.xy / f32(BEAM_TILE_SIZE));
    let m00 = textureLoad(coarse_mask, tile, 0).r;
    if m00 < 0.5 {
        let m10 = textureLoad(coarse_mask, tile + vec2<i32>(1, 0), 0).r;
        let mn0 = textureLoad(coarse_mask, tile + vec2<i32>(-1, 0), 0).r;
        let m01 = textureLoad(coarse_mask, tile + vec2<i32>(0, 1), 0).r;
        let m0n = textureLoad(coarse_mask, tile + vec2<i32>(0, -1), 0).r;
        let any_neighbor = max(max(m10, mn0), max(m01, m0n));
        if any_neighbor < 0.5 {
            let ray_dir = jittered_ray_dir(in.uv);
            return vec4<f32>(sky_color(ray_dir), 1.0);
        }
    }

    let shaded = shade_pixel(in.uv);
    return vec4<f32>(shaded.rgb, 1.0);
}

/// Coarse beam-prepass fragment. Casts **4 rays at the tile corners**
/// (not center) and outputs 1.0 if ANY hits, 0.0 otherwise. Writes
/// to an R8Unorm render target at 1/BEAM_TILE_SIZE per axis.
///
/// Corner sampling vs center sampling:
///
///   Center: tile's fate decided by one ray at the tile's midpoint.
///   Sub-tile features between tile centers are invisible to the
///   cull, producing temporal shimmer as small camera motion flips
///   which feature the center ray happens to cross.
///
///   Corners: 4 rays bound the tile's screen footprint. Any feature
///   larger than the corner-to-corner sampling spacing (≈ half-tile
///   width) is caught consistently across frames. Shimmer drops to
///   the floor of sub-half-tile features.
///
/// Short-circuits: if any corner ray hits, return immediately without
/// running the remaining 3 marches. Hit-tiles cost ~1 march; sky
/// tiles cost 4 marches (still bounded by 1/64 coarse ray density,
/// so total coarse cost scales to roughly 4× the center-only
/// version in the worst case).
///
/// Does not sample `coarse_mask` — the pipeline binds a 1×1 dummy
/// there since the mask texture is the render target.
@fragment
fn fs_coarse_mask(in: VertexOutput) -> @location(0) f32 {
    // UV span of half the tile's screen footprint. Tile center is at
    // `in.uv`; corners are at ± half_tile_uv on each axis.
    let half_tile_uv = 0.5 * vec2<f32>(
        f32(BEAM_TILE_SIZE) / uniforms.screen_width,
        f32(BEAM_TILE_SIZE) / uniforms.screen_height,
    );

    // DIAGNOSTIC: 2-ray (opposite corners). If this is ~1/2 the
    // 4-ray cost, register pressure in the coarse shader scales
    // linearly with march count (bad). If it's only slightly
    // better than 4-ray, march is amortized over divergent warps.
    if march(camera.pos, jittered_ray_dir(in.uv + vec2<f32>(-half_tile_uv.x, -half_tile_uv.y))).hit { return 1.0; }
    if march(camera.pos, jittered_ray_dir(in.uv + vec2<f32>( half_tile_uv.x,  half_tile_uv.y))).hit { return 1.0; }
    return 0.0;
}

/// Raster-entity companion entry point. Writes color plus
/// `@builtin(frag_depth)` so the subsequent entity raster pass
/// z-tests against the ray-march's terrain hits. The depth is
/// `(view_proj * world_hit).z / .w` — the same projection the
/// raster pipeline uses — so the two passes share a depth buffer
/// pixel-accurately.
///
/// Sky (no hit) writes 1.0 = far plane, so entity pixels over sky
/// always draw.
struct DepthFragOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main_depth(in: VertexOutput) -> DepthFragOut {
    let ray_dir = jittered_ray_dir(in.uv);
    let shaded = shade_pixel(in.uv);
    let t = shaded.a;
    var depth: f32;
    if t > 0.0 {
        let hit = camera.pos + ray_dir * t;
        let clip = camera.view_proj * vec4<f32>(hit, 1.0);
        depth = clamp(clip.z / clip.w, 0.0, 1.0);
    } else {
        depth = 1.0;
    }
    var out: DepthFragOut;
    out.color = vec4<f32>(shaded.rgb, 1.0);
    out.depth = depth;
    return out;
}

/// TAAU entry point. Writes color to `@location(0)` and hit t to
/// `@location(1)` (R32Float), both at the half-res render target.
/// The resolve pass reconstructs world-space hit positions from
/// `(camera.pos, ray_dir, t)` to reproject history across frames.
struct TaaFragOut {
    @location(0) color: vec4<f32>,
    @location(1) t: f32,
}

@fragment
fn fs_main_taa(in: VertexOutput) -> TaaFragOut {
    let shaded = shade_pixel(in.uv);
    var out: TaaFragOut;
    out.color = vec4<f32>(shaded.rgb, 1.0);
    out.t = shaded.a;
    return out;
}
