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
    colors: array<vec4<f32>, 16>,
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

// Maximum tree depth we'll traverse. 8 levels = 3^8 = 6561 voxels
// per axis at the finest level. Plenty for visual detail.
const MAX_STACK_DEPTH: u32 = 8u;

// Stack frame: tracks where we are in the DDA at each tree level.
// WGSL has no structs-in-arrays of variable size, so we use
// parallel arrays for each field.
struct MarchState {
    // Per-level DDA state (fixed-size arrays)
    node_idx: array<u32, 8>,
    cell: array<vec3<i32>, 8>,
    side_dist: array<vec3<f32>, 8>,
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
}

fn march(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;

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
    var s_node_idx: array<u32, 8>;
    var s_cell: array<vec3<i32>, 8>;
    var s_side_dist: array<vec3<f32>, 8>;
    // The world-space origin of the node at each level (its min corner).
    var s_node_origin: array<vec3<f32>, 8>;
    // The scale of one cell at each level.
    var s_cell_size: array<f32, 8>;

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
            result.hit = true;
            result.color = palette.colors[child_block_type(packed)].rgb;
            result.normal = normal;
            // Approximate hit t from the cell's AABB entry.
            let cell_min_t = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
            let cell_max_t = cell_min_t + vec3<f32>(s_cell_size[depth]);
            let hbox = ray_box(ray_origin, inv_dir, cell_min_t, cell_max_t);
            result.t = max(hbox.t_enter, 0.0);
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
                // Treat as solid using the node's dominant color.
                result.hit = true;
                let bt = child_block_type(packed);
                if bt < 10u {
                    result.color = palette.colors[bt].rgb;
                } else {
                    result.color = vec3<f32>(0.5);
                }
                result.normal = normal;
                let cell_min_l = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                let cell_max_l = cell_min_l + vec3<f32>(s_cell_size[depth]);
                let hbox_l = ray_box(ray_origin, inv_dir, cell_min_l, cell_max_l);
                result.t = max(hbox_l.t_enter, 0.0);
                return result;
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
        let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
        let diffuse = max(dot(result.normal, sun_dir), 0.0);
        let ambient = 0.3;
        let lit = result.color * (ambient + diffuse * 0.7);
        color = pow(lit, vec3<f32>(1.0 / 2.2));
    } else {
        let sky_t = ray_dir.y * 0.5 + 0.5;
        color = mix(vec3<f32>(0.7, 0.8, 0.95), vec3<f32>(0.3, 0.5, 0.85), sky_t);
    }

    // Block highlight: wireframe cube outline matching Bevy gizmos.cube() style.
    // Expanded 2% like the original, white color.
    if uniforms.highlight_active != 0u {
        let h_min = uniforms.highlight_min.xyz;
        let h_max = uniforms.highlight_max.xyz;
        let box_size = h_max - h_min;
        let expand = box_size * 0.01; // 1% each side = 2% total, matching Bevy's 1.02x
        let e_min = h_min - expand;
        let e_max = h_max + expand;

        // Per-slab intersection to find the exact entry face.
        let t1 = (e_min - camera.pos) / ray_dir;
        let t2 = (e_max - camera.pos) / ray_dir;
        let t_lo = min(t1, t2);
        let t_hi = max(t1, t2);
        let t_enter = max(max(t_lo.x, t_lo.y), t_lo.z);
        let t_exit  = min(min(t_hi.x, t_hi.y), t_hi.z);

        // Occlusion: only draw outline if it's not fully behind other geometry.
        // Use the highlight box size as epsilon since the shader may march
        // deeper than the CPU raycast (visual_depth > edit_depth).
        let occ_eps = length(box_size);
        let max_t = select(result.t + occ_eps, 1e10, !result.hit);
        if t_enter < t_exit && t_exit > 0.0 && t_enter <= max_t {
            // Use t_enter if in front of camera, else we're inside the box.
            let t = select(t_enter, 0.001, t_enter < 0.0);
            let hit_pos = camera.pos + ray_dir * t;

            // Normalize to [0,1] within expanded box.
            let local = (hit_pos - e_min) / (e_max - e_min);
            let d_lo = local;
            let d_hi = vec3<f32>(1.0) - local;
            let d_face = min(d_lo, d_hi);

            // Which axis entered the box? The one with the largest t_lo.
            // This correctly identifies the face even from any angle.
            var face_axis = 0u;
            if t_lo.y > t_lo.x && t_lo.y > t_lo.z {
                face_axis = 1u;
            } else if t_lo.z > t_lo.x {
                face_axis = 2u;
            }

            // If camera is inside box, use the closest face instead.
            if t_enter < 0.0 {
                let min_d = min(d_face.x, min(d_face.y, d_face.z));
                if d_face.x <= min_d + 0.001 {
                    face_axis = 0u;
                } else if d_face.y <= min_d + 0.001 {
                    face_axis = 1u;
                } else {
                    face_axis = 2u;
                }
            }

            // Edge width in normalized [0,1] coords. ~2% = thin wireframe.
            let ew = 0.02;

            // On the entry face, check if the other two axes are near an edge.
            var on_edge = false;
            if face_axis == 0u {
                on_edge = d_lo.y < ew || d_hi.y < ew || d_lo.z < ew || d_hi.z < ew;
            } else if face_axis == 1u {
                on_edge = d_lo.x < ew || d_hi.x < ew || d_lo.z < ew || d_hi.z < ew;
            } else {
                on_edge = d_lo.x < ew || d_hi.x < ew || d_lo.y < ew || d_hi.y < ew;
            }
            if on_edge {
                color = mix(color, vec3<f32>(1.0), 0.85);
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
