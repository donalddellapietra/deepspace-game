// Post-processing shader: bloom threshold + ACES tonemapping + color grading.
//
// Reads from the HDR scene texture produced by the ray march pass.
// Applies bloom extraction, tone mapping, and final color grading.

struct PostUniforms {
    screen_width: f32,
    screen_height: f32,
    bloom_threshold: f32,
    bloom_intensity: f32,
    exposure: f32,
    contrast: f32,
    saturation: f32,
    vignette_strength: f32,
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> post: PostUniforms;

// ─── vertex (full-screen triangle) ──────────────────────────────────

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

// ─── ACES tonemapping ───────────────────────────────────────────────

// Attempt a simplified ACES fit (Stephen Hill's approximation).
// Input and output are both linear sRGB.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    // sRGB → AP1 (approximate)
    let ap1 = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777),
    );
    // AP1 → sRGB (approximate)
    let ap1_inv = mat3x3<f32>(
        vec3<f32>( 1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108,  1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605,  1.07602),
    );

    var color = ap1 * x;

    // RRT + ODT fit
    let a = color * (color + 0.0245786) - 0.000090537;
    let b = color * (0.983729 * color + 0.4329510) + 0.238081;
    color = a / b;

    color = ap1_inv * color;
    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

// ─── bloom extraction ───────────────────────────────────────────────

// 13-tap tent filter (Jimenez 2014) for efficient blur.
// Samples in a cross+diagonal pattern for good quality at low cost.
fn bloom_downsample(uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    let a = textureSample(hdr_texture, hdr_sampler, uv + vec2<f32>(-1.0, -1.0) * texel).rgb;
    let b = textureSample(hdr_texture, hdr_sampler, uv + vec2<f32>( 0.0, -1.0) * texel).rgb;
    let c = textureSample(hdr_texture, hdr_sampler, uv + vec2<f32>( 1.0, -1.0) * texel).rgb;
    let d = textureSample(hdr_texture, hdr_sampler, uv + vec2<f32>(-1.0,  0.0) * texel).rgb;
    let e = textureSample(hdr_texture, hdr_sampler, uv).rgb;
    let f = textureSample(hdr_texture, hdr_sampler, uv + vec2<f32>( 1.0,  0.0) * texel).rgb;
    let g = textureSample(hdr_texture, hdr_sampler, uv + vec2<f32>(-1.0,  1.0) * texel).rgb;
    let h = textureSample(hdr_texture, hdr_sampler, uv + vec2<f32>( 0.0,  1.0) * texel).rgb;
    let i = textureSample(hdr_texture, hdr_sampler, uv + vec2<f32>( 1.0,  1.0) * texel).rgb;

    var result = e * 0.25;
    result += (b + d + f + h) * 0.125;
    result += (a + c + g + i) * 0.0625;
    return result;
}

// Soft threshold: smoothly ramps contribution above the threshold.
fn bloom_threshold(color: vec3<f32>) -> vec3<f32> {
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let soft = clamp(luma - post.bloom_threshold + 0.5, 0.0, 1.0);
    let contribution = soft * soft * (1.0 / (2.0 * 0.5));  // soft knee
    let scale = max(contribution, luma - post.bloom_threshold) / max(luma, 0.0001);
    return color * scale;
}

// ─── color grading ──────────────────────────────────────────────────

fn apply_saturation(color: vec3<f32>, sat: f32) -> vec3<f32> {
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(vec3<f32>(luma), color, sat);
}

fn apply_contrast(color: vec3<f32>, contrast: f32) -> vec3<f32> {
    return (color - 0.5) * contrast + 0.5;
}

fn apply_vignette(color: vec3<f32>, uv: vec2<f32>, strength: f32) -> vec3<f32> {
    let dist = length(uv * 2.0 - 1.0);
    let vig = 1.0 - smoothstep(0.4, 1.4, dist) * strength;
    return color * vig;
}

// Linear → sRGB gamma curve.
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = pow(color, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

// ─── fragment ───────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(1.0 / post.screen_width, 1.0 / post.screen_height);

    // Sample HDR scene
    var hdr = textureSample(hdr_texture, hdr_sampler, in.uv).rgb;

    // Bloom: approximate with a wide-radius blur of bright pixels.
    // In a production pipeline this would be multi-pass; here we do
    // a single 13-tap filter at 2x texel spacing for a subtle glow.
    let bloom_raw = bloom_downsample(in.uv, texel * 2.0);
    let bloom = bloom_threshold(bloom_raw);
    hdr += bloom * post.bloom_intensity;

    // Exposure
    hdr *= post.exposure;

    // ACES tonemapping (HDR → LDR)
    var ldr = aces_tonemap(hdr);

    // Color grading
    ldr = apply_saturation(ldr, post.saturation);
    ldr = apply_contrast(ldr, post.contrast);
    ldr = apply_vignette(ldr, in.uv, post.vignette_strength);

    // Linear → sRGB
    ldr = linear_to_srgb(ldr);

    return vec4<f32>(clamp(ldr, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
