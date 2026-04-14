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
    sun_pos: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> tree: array<u32>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> palette: Palette;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

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
    // Emission intensity of the surface (stored in the palette's
    // alpha channel). 0 = diffuse-only; >0 = the block glows with
    // its own light on top of any reflected sunlight. Averages
    // naturally through LOD because `representative_block` is the
    // dominant non-empty block in a subtree, and emission is looked
    // up per palette index — a coarse-LOD cell that representatively
    // holds STAR_SURFACE inherits the star's emission.
    emission: f32,
}

fn march(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.emission = 0.0;

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
            let bt_h = child_block_type(packed);
            result.color = palette.colors[bt_h].rgb;
            result.emission = palette.colors[bt_h].a;
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
                    // Solid representative — render as block. The
                    // representative block's emission is inherited
                    // through LOD: if a subtree's dominant block is
                    // emissive (e.g. STAR_SURFACE), the coarse cell
                    // glows too, preserving the star's appearance
                    // at any zoom level.
                    let cell_min_l = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                    let cell_max_l = cell_min_l + vec3<f32>(s_cell_size[depth]);
                    let cell_box_l = ray_box(ray_origin, inv_dir, cell_min_l, cell_max_l);
                    result.hit = true;
                    result.t = max(cell_box_l.t_enter, 0.0);
                    result.color = palette.colors[bt].rgb;
                    result.emission = palette.colors[bt].a;
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

    let result = march(camera.pos, ray_dir);

    var color: vec3<f32>;
    if result.hit {
        let hit_pos = camera.pos + ray_dir * result.t;

        if result.emission > 0.0 {
            // Emissive surface: render it as its own light source,
            // unaffected by shadows or the sun's direction. Emission
            // multiplies the base color so a star reads hot and
            // bright even through LOD averaging (which gets here via
            // `representative_block` and this same emission alpha).
            let emit = result.color * result.emission;
            color = pow(emit, vec3<f32>(1.0 / 2.2));
        } else {
            // Per-point sun direction: aim at the star's position
            // uploaded in `sun_pos`. Each planet gets a correct
            // terminator from its own position relative to the star.
            let to_sun = uniforms.sun_pos.xyz - hit_pos;
            let sun_dist = length(to_sun);
            let sun_dir = to_sun / max(sun_dist, 1e-6);
            let n_dot_l = dot(result.normal, sun_dir);
            let diffuse = max(n_dot_l, 0.0);

            // Shadow ray: march from just above the surface toward
            // the star. A hit that is NOT itself emissive before the
            // star means the point is in cast shadow (terrain or
            // another planet occluding). We allow emissive hits
            // (the star surface itself) to count as reaching the
            // light source.
            var shadow_factor: f32 = 1.0;
            if diffuse > 0.0 {
                let shadow_origin = hit_pos + result.normal * 0.0005;
                let shadow_hit = march(shadow_origin, sun_dir);
                let blocked = shadow_hit.hit
                    && shadow_hit.t < sun_dist
                    && shadow_hit.emission <= 0.0;
                if blocked { shadow_factor = 0.0; }
            }

            // Constant ambient fill so shadowed faces remain legible.
            // A proper renderer would sample a sky-radiance LUT; a
            // small constant is fine for a diffuse-only look.
            let ambient = 0.10;
            let lit = result.color * (ambient + diffuse * 1.1 * shadow_factor);
            color = pow(lit, vec3<f32>(1.0 / 2.2));
        }
    } else {
        // Deep-space sky: dark background + a bright sun disk & halo
        // around the star's apparent position. Lets the player orient
        // toward the light source at any zoom.
        let sun_vec = normalize(uniforms.sun_pos.xyz - camera.pos);
        let alignment = dot(ray_dir, sun_vec);
        let disk = smoothstep(0.9995, 0.9999, alignment);
        let halo = smoothstep(0.985, 0.9995, alignment) * 0.35;
        let sky_bg = vec3<f32>(0.015, 0.02, 0.035);
        color = sky_bg
            + vec3<f32>(1.35, 1.22, 0.95) * disk
            + vec3<f32>(1.0, 0.8, 0.55) * halo;
    }

    // Block highlight outline: Minecraft-style wireframe cube.
    // Draws all 12 edges with screen-space constant-pixel width.
    // Occluded by geometry via result.t comparison.
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

            // Only draw if the outline is in front of (or at) geometry.
            if t <= result.t + h_size.x * 0.01 {
                let hit_pos = camera.pos + ray_dir * t;
                let from_min = hit_pos - h_min;
                let from_max = h_max - hit_pos;

                // Screen-space edge width: ~1.5 pixels regardless of distance.
                let pixel_world = max(t, 0.001) * 2.0 * tan(camera.fov * 0.5) / uniforms.screen_height;
                let ew = max(pixel_world * 1.5, h_size.x * 0.003);

                // An edge exists where at least 2 of the 3 axes are near a face boundary.
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

    // Crosshair: thin cross at screen center.
    let pixel = vec2<f32>(in.uv.x * uniforms.screen_width, in.uv.y * uniforms.screen_height);
    let center = vec2<f32>(uniforms.screen_width * 0.5, uniforms.screen_height * 0.5);
    let d = abs(pixel - center);
    let cross_size = 12.0;
    let cross_thickness = 1.5;
    let gap = 3.0;
    let is_crosshair = (d.x < cross_thickness && d.y >= gap && d.y < cross_size)
                    || (d.y < cross_thickness && d.x >= gap && d.x < cross_size);
    if is_crosshair {
        // Invert color for visibility against any background.
        color = vec3<f32>(1.0) - color;
    }

    return vec4<f32>(color, 1.0);
}
