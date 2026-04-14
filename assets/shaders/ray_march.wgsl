// Ray march shader for base-3 recursive voxel tree.
//
// Iterative stack-based traversal (WGSL forbids recursion).
// Each node has 27 children (3x3x3). Each child is:
//   tag=0: Empty (air)
//   tag=1: Block (solid, block_type indexes into palette)
//   tag=2: Node (descend into child node at node_index)

struct Camera {
    pos: vec3<f32>,
    _pad0: f32,
    forward: vec3<f32>,
    _pad1: f32,
    right: vec3<f32>,
    _pad2: f32,
    up: vec3<f32>,
    fov: f32,
}

struct Palette {
    colors: array<vec4<f32>, 256>,
}

struct Uniforms {
    root_index: u32,
    node_count: u32,
    screen_width: f32,
    screen_height: f32,
    max_depth: u32,
    highlight_active: u32,
    _pad0: u32,
    _pad1: u32,
    highlight_min: vec4<f32>,
    highlight_max: vec4<f32>,
}

struct Environment {
    sun_dir: vec3<f32>,
    time: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    fog_density: f32,
    fog_start: f32,
    water_block_type: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> tree: array<u32>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> palette: Palette;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<uniform> env: Environment;

// -------------- Tree access helpers --------------

// Each GpuChild is 2 u32s (8 bytes). A node at buffer index
// `node_idx` occupies tree[(node_idx*27 + slot) * 2 .. +2].
fn child_packed(node_idx: u32, slot: u32) -> u32 {
    return tree[(node_idx * 27u + slot) * 2u];
}

fn child_node_index(node_idx: u32, slot: u32) -> u32 {
    return tree[(node_idx * 27u + slot) * 2u + 1u];
}

fn child_tag(packed: u32) -> u32 {
    return packed & 0xFFu;
}

fn child_block_type(packed: u32) -> u32 {
    return (packed >> 8u) & 0xFFu;
}

fn slot_from_xyz(x: i32, y: i32, z: i32) -> u32 {
    return u32(z * 9 + y * 3 + x);
}

// -------------- Ray-AABB intersection --------------

struct BoxHit {
    t_enter: f32,
    t_exit: f32,
}

fn ray_box(origin: vec3<f32>, inv_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> BoxHit {
    let t1 = (box_min - origin) * inv_dir;
    let t2 = (box_max - origin) * inv_dir;
    let t_lo = min(t1, t2);
    let t_hi = max(t1, t2);
    return BoxHit(
        max(max(t_lo.x, t_lo.y), t_lo.z),
        min(min(t_hi.x, t_hi.y), t_hi.z),
    );
}

// -------------- Stack-based iterative ray march --------------

// Maximum tree depth we'll traverse. 16 supports trees up to ~14 levels
// deep with room for DDA stepping across siblings at ancestor levels.
const MAX_STACK_DEPTH: u32 = 16u;

// Stack frame: tracks where we are in the DDA at each tree level.
// WGSL has no structs-in-arrays of variable size, so we use
// parallel arrays for each field.
struct MarchState {
    // Per-level DDA state (fixed-size arrays)
    node_idx: array<u32, 16>,
    cell: array<vec3<i32>, 16>,
    side_dist: array<vec3<f32>, 16>,
    // Shared across all levels
    origin: vec3<f32>,       // original ray origin (world space)
    inv_dir: vec3<f32>,
    step: vec3<i32>,
    delta_dist: vec3<f32>,
    depth: u32,
}

struct HitResult {
    hit: bool,
    color: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    block_type: u32,
}

fn march(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.block_type = 0u;

    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );

    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );

    let delta_dist = abs(inv_dir);

    // State arrays for each depth level.
    var s_node_idx: array<u32, 16>;
    var s_cell: array<vec3<i32>, 16>;
    var s_side_dist: array<vec3<f32>, 16>;
    // The world-space origin of the node at each level (its min corner).
    var s_node_origin: array<vec3<f32>, 16>;
    // The scale of one cell at each level.
    var s_cell_size: array<f32, 16>;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    // Initialize root level.
    s_node_idx[0] = uniforms.root_index;
    s_node_origin[0] = vec3<f32>(0.0);
    s_cell_size[0] = 1.0; // root node: each cell is 1.0 wide, node spans [0, 3)

    // Intersect ray with root node [0, 3)
    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }

    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = ray_origin + ray_dir * t_start;

    // Initial cell in root
    s_cell[0] = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );

    // Initial side_dist for root level
    let cell_f = vec3<f32>(s_cell[0]);
    s_side_dist[0] = vec3<f32>(
        select(
            (cell_f.x - entry_pos.x) * inv_dir.x,
            (cell_f.x + 1.0 - entry_pos.x) * inv_dir.x,
            ray_dir.x >= 0.0
        ),
        select(
            (cell_f.y - entry_pos.y) * inv_dir.y,
            (cell_f.y + 1.0 - entry_pos.y) * inv_dir.y,
            ray_dir.y >= 0.0
        ),
        select(
            (cell_f.z - entry_pos.z) * inv_dir.z,
            (cell_f.z + 1.0 - entry_pos.z) * inv_dir.z,
            ray_dir.z >= 0.0
        ),
    );

    // Main traversal loop.
    // Each iteration either:
    //   - Hits a block → return
    //   - Descends into a child node (push)
    //   - Steps DDA to next cell
    //   - Exits current node (pop)
    var iterations = 0u;
    let max_iterations = 256u;

    loop {
        if iterations >= max_iterations {
            break;
        }
        iterations += 1u;

        let cell = s_cell[depth];

        // Out of bounds check — this cell has left the 3x3x3 grid.
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            // Pop up one level.
            if depth == 0u {
                break; // Exited root, done.
            }
            depth -= 1u;

            // Advance the DDA at the parent level (we finished this child).
            if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                s_cell[depth].x += step.x;
                s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if s_side_dist[depth].y < s_side_dist[depth].z {
                s_cell[depth].y += step.y;
                s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                s_cell[depth].z += step.z;
                s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
            continue;
        }

        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        let packed = child_packed(s_node_idx[depth], slot);
        let tag = child_tag(packed);

        if tag == 0u {
            // Empty — advance DDA.
            if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                s_cell[depth].x += step.x;
                s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if s_side_dist[depth].y < s_side_dist[depth].z {
                s_cell[depth].y += step.y;
                s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                s_cell[depth].z += step.z;
                s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
        } else if tag == 1u {
            // Block hit!
            let cell_min_h = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
            let cell_max_h = cell_min_h + vec3<f32>(s_cell_size[depth]);
            let cell_box_h = ray_box(ray_origin, inv_dir, cell_min_h, cell_max_h);
            result.hit = true;
            result.t = max(cell_box_h.t_enter, 0.0);
            result.block_type = child_block_type(packed);
            result.color = palette.colors[result.block_type].rgb;
            result.normal = normal;
            return result;
        } else if tag == 2u {
            // Node — check if we should descend or treat as solid.

            // Hard depth limits.
            let at_max = depth + 1u >= uniforms.max_depth || depth + 1u >= MAX_STACK_DEPTH;

            // Screen-space LOD: compute how many pixels this cell covers.
            // If sub-pixel, no point descending — shade as dominant color.
            let cell_world_size = s_cell_size[depth];
            // Approximate ray distance to this cell from side_dist.
            let min_side = min(s_side_dist[depth].x, min(s_side_dist[depth].y, s_side_dist[depth].z));
            let ray_dist = max(min_side, 0.001);
            let lod_pixels = cell_world_size / ray_dist * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            let at_lod = lod_pixels < 1.0;

            if at_max || at_lod {
                // LOD cutoff: use the per-child representative block type.
                // bt=255 means all-empty subtree — treat as air, advance DDA.
                let bt = child_block_type(packed);
                if bt == 255u {
                    // All-empty subtree — skip like Empty.
                    if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                        s_cell[depth].x += step.x;
                        s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                        normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                    } else if s_side_dist[depth].y < s_side_dist[depth].z {
                        s_cell[depth].y += step.y;
                        s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                        normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                    } else {
                        s_cell[depth].z += step.z;
                        s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                        normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                    }
                } else {
                    // Solid representative — render as block.
                    let cell_min_l = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                    let cell_max_l = cell_min_l + vec3<f32>(s_cell_size[depth]);
                    let cell_box_l = ray_box(ray_origin, inv_dir, cell_min_l, cell_max_l);
                    result.hit = true;
                    result.t = max(cell_box_l.t_enter, 0.0);
                    result.block_type = bt;
                    result.color = palette.colors[bt].rgb;
                    result.normal = normal;
                    return result;
                }
            }

            let child_idx = child_node_index(s_node_idx[depth], slot);

            // Compute the world-space min corner of this child cell.
            let parent_origin = s_node_origin[depth];
            let parent_cell_size = s_cell_size[depth];
            let child_origin = parent_origin + vec3<f32>(cell) * parent_cell_size;
            let child_cell_size = parent_cell_size / 3.0;

            // Intersect ray with this child's AABB.
            let child_max = child_origin + vec3<f32>(parent_cell_size);
            let child_hit = ray_box(ray_origin, inv_dir, child_origin, child_max);
            let ct_start = max(child_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
            let child_entry = ray_origin + ray_dir * ct_start;

            // Convert entry point to child-local cell coordinates.
            let local_entry = (child_entry - child_origin) / child_cell_size;

            // Push onto stack.
            depth += 1u;
            s_node_idx[depth] = child_idx;
            s_node_origin[depth] = child_origin;
            s_cell_size[depth] = child_cell_size;
            s_cell[depth] = vec3<i32>(
                clamp(i32(floor(local_entry.x)), 0, 2),
                clamp(i32(floor(local_entry.y)), 0, 2),
                clamp(i32(floor(local_entry.z)), 0, 2),
            );

            // Initialize side_dist for this level.
            let lc = vec3<f32>(s_cell[depth]);
            s_side_dist[depth] = vec3<f32>(
                select(
                    (child_origin.x + lc.x * child_cell_size - ray_origin.x) * inv_dir.x,
                    (child_origin.x + (lc.x + 1.0) * child_cell_size - ray_origin.x) * inv_dir.x,
                    ray_dir.x >= 0.0
                ),
                select(
                    (child_origin.y + lc.y * child_cell_size - ray_origin.y) * inv_dir.y,
                    (child_origin.y + (lc.y + 1.0) * child_cell_size - ray_origin.y) * inv_dir.y,
                    ray_dir.y >= 0.0
                ),
                select(
                    (child_origin.z + lc.z * child_cell_size - ray_origin.z) * inv_dir.z,
                    (child_origin.z + (lc.z + 1.0) * child_cell_size - ray_origin.z) * inv_dir.z,
                    ray_dir.z >= 0.0
                ),
            );
        }
    }

    return result;
}

// -------------- Vertex / Fragment shaders --------------

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Full-screen triangle from 3 vertices.
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

// ─── Atmospheric scattering ─────────────────────────────────────────
//
// Simplified Rayleigh + Mie scattering adapted from Kappa/RRe36.
// Models light scattering through an atmosphere sphere around a planet.

const PI: f32 = 3.14159265;
const PLANET_RADIUS: f32 = 6371e3;
const ATMOS_RADIUS: f32 = 6471e3;
const RAYLEIGH_SCALE_HEIGHT: f32 = 8500.0;
const MIE_SCALE_HEIGHT: f32 = 1200.0;
const RAYLEIGH_COEFF: vec3<f32> = vec3<f32>(5.8e-6, 13.5e-6, 33.1e-6);
const MIE_COEFF: f32 = 21e-6;
const MIE_G: f32 = 0.76;

// Ray-sphere intersection: returns (near, far) distances.
fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let det = b * b - c;
    if det < 0.0 { return vec2<f32>(-1.0); }
    let d = sqrt(det);
    return vec2<f32>(-b - d, -b + d);
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let gg = g * g;
    let num = (1.0 - gg);
    let denom = 4.0 * PI * pow(1.0 + gg - 2.0 * g * cos_theta, 1.5);
    return num / max(denom, 1e-8);
}

fn atmosphere(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let cam_pos = vec3<f32>(0.0, PLANET_RADIUS + 100.0, 0.0);

    let atmos_dist = ray_sphere(cam_pos, ray_dir, ATMOS_RADIUS);
    if atmos_dist.y < 0.0 { return vec3<f32>(0.0); }

    let planet_dist = ray_sphere(cam_pos, ray_dir, PLANET_RADIUS);
    let ray_end = select(atmos_dist.y, max(planet_dist.x, 0.0), planet_dist.x > 0.0);
    let ray_start = max(atmos_dist.x, 0.0);

    let steps = 8u;
    let step_size = (ray_end - ray_start) / f32(steps);

    var rayleigh_sum = vec3<f32>(0.0);
    var mie_sum = vec3<f32>(0.0);
    var optical_depth_r = 0.0;
    var optical_depth_m = 0.0;

    let cos_theta = dot(ray_dir, sun_dir);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, MIE_G);

    for (var i = 0u; i < steps; i++) {
        let t = ray_start + (f32(i) + 0.5) * step_size;
        let pos = cam_pos + ray_dir * t;
        let altitude = length(pos) - PLANET_RADIUS;

        let density_r = exp(-altitude / RAYLEIGH_SCALE_HEIGHT) * step_size;
        let density_m = exp(-altitude / MIE_SCALE_HEIGHT) * step_size;

        optical_depth_r += density_r;
        optical_depth_m += density_m;

        // Light optical depth to sun (simplified: 4 steps)
        let sun_dist = ray_sphere(pos, sun_dir, ATMOS_RADIUS);
        let sun_step = sun_dist.y / 4.0;
        var sun_od_r = 0.0;
        var sun_od_m = 0.0;
        for (var j = 0u; j < 4u; j++) {
            let sp = pos + sun_dir * (f32(j) + 0.5) * sun_step;
            let sa = length(sp) - PLANET_RADIUS;
            sun_od_r += exp(-sa / RAYLEIGH_SCALE_HEIGHT) * sun_step;
            sun_od_m += exp(-sa / MIE_SCALE_HEIGHT) * sun_step;
        }

        let attenuation = exp(
            -(RAYLEIGH_COEFF * (optical_depth_r + sun_od_r)
              + MIE_COEFF * (optical_depth_m + sun_od_m))
        );

        rayleigh_sum += density_r * attenuation;
        mie_sum += density_m * attenuation;
    }

    let sun_intensity = env.sun_intensity * 22.0;
    return sun_intensity * (rayleigh_sum * RAYLEIGH_COEFF * phase_r
                          + mie_sum * MIE_COEFF * phase_m);
}

// ─── PBR lighting ───────────────────────────────────────────────────

// Schlick Fresnel approximation.
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// GGX/Trowbridge-Reitz normal distribution.
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-8);
}

// Smith's geometry function (Schlick-GGX).
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;
    let g1 = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g2 = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return g1 * g2;
}

fn pbr_lighting(
    albedo: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    sun_dir: vec3<f32>,
    roughness: f32,
    metallic: f32,
    f0_base: f32,
) -> vec3<f32> {
    let half_dir = normalize(view_dir + sun_dir);
    let n_dot_l = max(dot(normal, sun_dir), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 0.001);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 0.0);

    // Fresnel: metallic surfaces use albedo as F0
    let f0 = mix(vec3<f32>(f0_base), albedo, metallic);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);

    // Specular BRDF
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let specular = (d * g * fresnel) / max(4.0 * n_dot_v * n_dot_l, 0.001);

    // Diffuse (energy-conserving Lambert)
    let k_d = (1.0 - fresnel) * (1.0 - metallic);
    let diffuse = k_d * albedo / PI;

    let sun_color = env.sun_color * env.sun_intensity;
    let direct = (diffuse + specular) * sun_color * n_dot_l;

    // Ambient: sky-colored fill light from above, ground bounce from below
    let sky_ambient = vec3<f32>(0.4, 0.5, 0.7) * 0.15;
    let ground_ambient = vec3<f32>(0.3, 0.25, 0.2) * 0.05;
    let up_factor = normal.y * 0.5 + 0.5;
    let ambient = albedo * mix(ground_ambient, sky_ambient, up_factor);

    return direct + ambient;
}

// ─── Water ──────────────────────────────────────────────────────────

// Gerstner wave: physically-based ocean wave.
fn gerstner(pos: vec2<f32>, t: f32, amp: f32, wlen: f32, dir: vec2<f32>, steep: f32) -> f32 {
    let k = 6.283185 / wlen;
    let w = sqrt(9.81 * k);
    let phase = w * t - k * dot(dir, pos);
    return pow(sin(phase) * 0.5 + 0.5, steep) * amp;
}

fn water_height(pos: vec3<f32>) -> f32 {
    let p = pos.xz + pos.y / PI;
    let t = env.time * 0.76;

    var wave = 0.0;
    var amp = 0.06;
    var steep = 0.51;
    var wlen = 2.8;
    var dir = normalize(vec2<f32>(0.4, 0.8));

    // Rotation per octave
    let ca = cos(2.6);
    let sa = sin(2.6);

    for (var i = 0u; i < 4u; i++) {
        wave -= gerstner(p, t, amp, wlen, dir, steep);
        amp *= 0.55;
        wlen *= 0.63;
        steep = mix(steep, sqrt(steep), sqrt(clamp(abs(wave), 0.0, 1.0)));
        dir = vec2<f32>(dir.x * ca - dir.y * sa, dir.x * sa + dir.y * ca);
    }
    return wave;
}

fn water_normal(pos: vec3<f32>) -> vec3<f32> {
    let e = 0.02;
    let h0 = water_height(pos);
    let hx = water_height(pos + vec3<f32>(e, 0.0, 0.0));
    let hz = water_height(pos + vec3<f32>(0.0, 0.0, e));
    return normalize(vec3<f32>(-(hx - h0) / e, 1.0, -(hz - h0) / e));
}

// ─── Fog ────────────────────────────────────────────────────────────

fn apply_fog(color: vec3<f32>, dist: f32, ray_dir: vec3<f32>, sky_color: vec3<f32>) -> vec3<f32> {
    let fog_amount = 1.0 - exp(-max(dist - env.fog_start, 0.0) * env.fog_density);
    // Sun-tinted fog: brighter when looking toward the sun
    let sun_factor = max(dot(ray_dir, env.sun_dir), 0.0);
    let fog_color = mix(sky_color, env.sun_color * 1.2, pow(sun_factor, 8.0) * 0.3);
    return mix(color, fog_color, clamp(fog_amount, 0.0, 1.0));
}

// ─── Fragment shader ────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = uniforms.screen_width / uniforms.screen_height;
    let half_fov_tan = tan(camera.fov * 0.5);

    let ndc = vec2<f32>(
        (in.uv.x - 0.5) * 2.0 * aspect * half_fov_tan,
        (0.5 - in.uv.y) * 2.0 * half_fov_tan,
    );

    let ray_dir = normalize(
        camera.forward + camera.right * ndc.x + camera.up * ndc.y
    );
    let view_dir = -ray_dir;

    let result = march(camera.pos, ray_dir);
    let sun_dir = normalize(env.sun_dir);

    // Atmospheric sky for miss rays and fog blending
    let sky_color = atmosphere(ray_dir, sun_dir);
    // Add sun disc
    let sun_dot = dot(ray_dir, sun_dir);
    let sun_disc = smoothstep(0.9997, 0.9999, sun_dot) * env.sun_intensity * 50.0;
    let sky_with_sun = sky_color + env.sun_color * sun_disc;

    var color: vec3<f32>;

    if result.hit {
        let hit_pos = camera.pos + ray_dir * result.t;

        // Check if this is a water block
        let is_water = result.block_type == env.water_block_type && env.water_block_type > 0u;

        if is_water {
            // Animated water normal
            let w_normal = water_normal(hit_pos);
            let blended_n = normalize(mix(result.normal, w_normal, 0.7));

            // Fresnel reflection
            let n_dot_v = max(dot(blended_n, view_dir), 0.0);
            let fresnel = fresnel_schlick(n_dot_v, 0.02);

            // Reflected sky color
            let reflect_dir = reflect(ray_dir, blended_n);
            let reflected = atmosphere(reflect_dir, sun_dir);

            // Water absorption: deeper = bluer/darker
            let water_color = vec3<f32>(0.05, 0.15, 0.3);
            let depth_factor = 1.0 / max(n_dot_v, 0.1);
            let absorbed = water_color * exp(-vec3<f32>(0.4, 0.1, 0.05) * depth_factor);

            // Specular highlight on water surface
            let spec = pbr_lighting(
                vec3<f32>(0.0), blended_n, view_dir, sun_dir,
                0.05, 0.0, 0.02,
            );

            // Blend reflection and refraction
            color = mix(absorbed, reflected, fresnel) + spec;

            // Subsurface scattering through wave crests
            let sss = pow(clamp(-blended_n.y * 0.5 + 0.5, 0.0, 1.0), 6.0) * 0.15;
            color += vec3<f32>(0.1, 0.2, 0.15) * sss;
        } else {
            // Standard PBR for solid blocks
            let roughness = 0.85;  // voxels are rough
            let metallic = 0.0;
            color = pbr_lighting(
                result.color, result.normal, view_dir, sun_dir,
                roughness, metallic, 0.04,
            );

            // Simple ambient occlusion from face normals:
            // corners and edges between faces are darker.
            let ao = 0.7 + 0.3 * abs(dot(result.normal, normalize(vec3<f32>(1.0))));
            color *= ao;
        }

        // Distance fog
        color = apply_fog(color, result.t, ray_dir, sky_color);
    } else {
        color = sky_with_sun;
    }

    // Block highlight outline (same as before).
    if uniforms.highlight_active != 0u {
        let h_min = uniforms.highlight_min.xyz;
        let h_max = uniforms.highlight_max.xyz;
        let h_size = h_max - h_min;

        let h_inv_dir = vec3<f32>(
            select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
            select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
            select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
        );

        let hb = ray_box(camera.pos, h_inv_dir, h_min, h_max);

        if hb.t_enter < hb.t_exit && hb.t_exit > 0.0 {
            let t = max(hb.t_enter, 0.0);
            if t <= result.t + h_size.x * 0.01 {
                let hit_pos = camera.pos + ray_dir * t;
                let from_min = hit_pos - h_min;
                let from_max = h_max - hit_pos;
                let pixel_world = max(t, 0.001) * 2.0 * tan(camera.fov * 0.5) / uniforms.screen_height;
                let ew = max(pixel_world * 1.5, h_size.x * 0.003);
                let near_x = from_min.x < ew || from_max.x < ew;
                let near_y = from_min.y < ew || from_max.y < ew;
                let near_z = from_min.z < ew || from_max.z < ew;
                let edge_count = u32(near_x) + u32(near_y) + u32(near_z);
                if edge_count >= 2u {
                    color = mix(color, vec3<f32>(0.1, 0.1, 0.1), 0.7);
                }
            }
        }
    }

    // Crosshair
    let pixel = vec2<f32>(in.uv.x * uniforms.screen_width, in.uv.y * uniforms.screen_height);
    let center = vec2<f32>(uniforms.screen_width * 0.5, uniforms.screen_height * 0.5);
    let d = abs(pixel - center);
    let cross_size = 12.0;
    let cross_thickness = 1.5;
    let gap = 3.0;
    let is_crosshair = (d.x < cross_thickness && d.y >= gap && d.y < cross_size)
                    || (d.y < cross_thickness && d.x >= gap && d.x < cross_size);
    if is_crosshair {
        color = vec3<f32>(1.0) - clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // Output HDR linear color (tonemapping happens in postprocess pass)
    return vec4<f32>(max(color, vec3<f32>(0.0)), 1.0);
}
