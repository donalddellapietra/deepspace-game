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
    // Cubed-sphere planet: xyz = world-space center,
    //                      w   = outer radius (0 disables).
    cs_planet: vec4<f32>,
    // x = inner radius. y, z reserved. w = highlight-active flag.
    cs_params: vec4<f32>,
    // Highlighted cell: (face, i, j, k) as f32.
    cs_highlight: vec4<f32>,
    // GPU buffer indices of the 6 face subtree roots, packed into
    // two vec4<u32>. Order: PosX, NegX, PosY, NegY, PosZ, NegZ.
    cs_face_roots_a: vec4<u32>,
    cs_face_roots_b: vec4<u32>,
}

/// Per-node kind descriptor, parallel to the tree buffer. Indexed by
/// the node's buffer index. Tag: 0=Cartesian, 1=CubedSphereBody,
/// 2=CubedSphereFace.
struct GpuNodeKind {
    tag: u32,
    inner_r: f32,
    outer_r: f32,
    face: u32,
}

@group(0) @binding(0) var<storage, read> tree: array<u32>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> palette: Palette;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> kinds: array<GpuNodeKind>;

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

// ────────────── Cubed-sphere helpers ──────────────

const PI_F: f32 = 3.1415926535;
const FRAC_PI_4: f32 = 0.785398163;

/// Cube-face coordinate → equal-angle coordinate. Inverse of the
/// `tan(u · π/4)` warp applied in Rust's `face_uv_to_dir`.
fn cube_to_ea(c: f32) -> f32 {
    return atan(c) * (4.0 / PI_F);
}

/// Equal-angle coordinate → cube-face coordinate (for boundary
/// plane math, which needs the raw ratio `k` for `u·x = 1` etc.).
fn ea_to_cube(e: f32) -> f32 {
    return tan(e * FRAC_PI_4);
}

/// Look up the `cs_face_roots` packed uniform by face index 0..6.
fn face_root(face: u32) -> u32 {
    if face < 4u { return uniforms.cs_face_roots_a[face]; }
    return uniforms.cs_face_roots_b[face - 4u];
}

/// Outward-pointing normal of the given cube face.
fn face_normal(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>( 1.0,  0.0,  0.0); } // PosX
        case 1u: { return vec3<f32>(-1.0,  0.0,  0.0); } // NegX
        case 2u: { return vec3<f32>( 0.0,  1.0,  0.0); } // PosY
        case 3u: { return vec3<f32>( 0.0, -1.0,  0.0); } // NegY
        case 4u: { return vec3<f32>( 0.0,  0.0,  1.0); } // PosZ
        default: { return vec3<f32>( 0.0,  0.0, -1.0); } // NegZ
    }
}

/// The face's u-tangent in world space. Matches Rust Face::tangents().
fn face_u_axis(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>( 0.0,  0.0, -1.0); } // PosX
        case 1u: { return vec3<f32>( 0.0,  0.0,  1.0); } // NegX
        case 2u: { return vec3<f32>( 1.0,  0.0,  0.0); } // PosY
        case 3u: { return vec3<f32>( 1.0,  0.0,  0.0); } // NegY
        case 4u: { return vec3<f32>( 1.0,  0.0,  0.0); } // PosZ
        default: { return vec3<f32>(-1.0,  0.0,  0.0); } // NegZ
    }
}

fn face_v_axis(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        case 1u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        case 2u: { return vec3<f32>( 0.0,  0.0, -1.0); }
        case 3u: { return vec3<f32>( 0.0,  0.0,  1.0); }
        case 4u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        default: { return vec3<f32>( 0.0,  1.0,  0.0); }
    }
}

/// Pick the dominant cube face for a unit direction.
fn pick_face(n: vec3<f32>) -> u32 {
    let ax = abs(n.x); let ay = abs(n.y); let az = abs(n.z);
    if ax >= ay && ax >= az {
        if n.x > 0.0 { return 0u; } else { return 1u; }
    } else if ay >= az {
        if n.y > 0.0 { return 2u; } else { return 3u; }
    } else {
        if n.z > 0.0 { return 4u; } else { return 5u; }
    }
}

/// Forward ray-plane intersection. Returns t ≥ 0 or -1 if no hit.
/// Plane passes through `through` with normal `plane_n`.
fn ray_plane_t(origin: vec3<f32>, dir: vec3<f32>,
               through: vec3<f32>, plane_n: vec3<f32>) -> f32 {
    let denom = dot(dir, plane_n);
    if abs(denom) < 1e-12 { return -1.0; }
    let t = -dot(origin - through, plane_n) / denom;
    return t;
}

/// First t strictly greater than `after` at which the ray hits the
/// sphere of given radius around `center`. -1 if no such hit.
fn ray_sphere_after(origin: vec3<f32>, dir: vec3<f32>,
                    center: vec3<f32>, radius: f32, after: f32) -> f32 {
    let oc = origin - center;
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 { return -1.0; }
    let sq = sqrt(disc);
    let t0 = -b - sq;
    let t1 = -b + sq;
    if t0 > after { return t0; }
    if t1 > after { return t1; }
    return -1.0;
}

/// Consider candidate `c` as a potential next-t: if `c > cur` and
/// `c < best`, return `c`; otherwise return `best`. Helper for the
/// "minimum positive exit t" reduction.
fn min_after(best: f32, cand: f32, cur: f32) -> f32 {
    if cand > cur && cand < best { return cand; }
    return best;
}

/// Walk a face subtree iteratively. Inputs are normalized
/// `(un, vn, rn) ∈ [0, 1]³`. Returns `(block_id, depth)`:
///   - `block_id`: 0 = empty, >0 = palette index of first terminal.
///   - `depth`: number of descents taken before the terminal was
///     found. A uniform empty region flattened at pack-time by
///     `pack_tree_lod_multi` returns `depth = 1`; a finest-level
///     hit returns `depth = subtree_depth`.
///
/// The DDA caller uses `depth` to compute exit-cell bounds at the
/// appropriate coarseness, so a ray through a huge empty chunk
/// crosses it in ONE step instead of `3^(subtree_depth-depth)`
/// steps — recovering the Cartesian octree's skip-empty speedup.
fn sample_face_tree(root_idx: u32, un_in: f32, vn_in: f32, rn_in: f32) -> vec2<u32> {
    var node = root_idx;
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);
    for (var d: u32 = 1u; d <= 22u; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let slot = rs * 9u + vs * 3u + us;
        let packed = child_packed(node, slot);
        let tag = child_tag(packed);
        if tag == 0u { return vec2<u32>(0u, d); }
        if tag == 1u { return vec2<u32>(child_block_type(packed), d); }
        // tag == 2u: descend.
        node = child_node_index(node, slot);
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
    }
    return vec2<u32>(0u, 12u);
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
}

fn march(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;

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
            result.color = palette.colors[child_block_type(packed)].rgb;
            result.normal = normal;
            return result;
        } else if tag == 2u {
            // Node — inspect its kind. A CubedSphereBody child is
            // rendered by the sphere DDA below (driven by the
            // cs_planet uniform until shader-side NodeKind dispatch
            // lands in a later commit); here we just step past its
            // cell as if it were Empty, so the Cartesian march
            // doesn't try to interpret face-subtree children as
            // axis-aligned voxels.
            let ci = child_node_index(s_node_idx[depth], slot);
            let kind_tag = kinds[ci].tag;
            // DEBUG: hard-skip ALL Node children. If the user still
            // sees stone, the renderer is hitting Block(STONE) tag=1
            // children directly (not through the Node descent path).
            if true || kind_tag == 1u || kind_tag == 2u {
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

    // Render root may be the tree root (Cartesian — normal march),
    // a body cell (camera ≥ K below body — no Cartesian content,
    // sphere DDA handles everything), or inside a face subtree
    // (deep zoom — same: sphere DDA only). Skip the Cartesian march
    // for body/face render roots so it doesn't try to interpret
    // face data as voxels.
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    let root_kind_tag = kinds[uniforms.root_index].tag;
    if root_kind_tag == 0u {
        result = march(camera.pos, ray_dir);
    }

    // Spherical planet: true 3D DDA through the face subtrees.
    //
    // Each iteration finds the (face, iu, iv, ir) cell the ray is
    // currently in at the finest subtree depth, samples that cell's
    // block via the subtree walker, and either hits or steps to the
    // exit boundary of that cell via analytic ray-plane (u/v) or
    // ray-sphere (r) intersection. No fixed step, no aliasing —
    // every pixel settles on the same cell regardless of depth or
    // camera distance.
    var cs_hit = false;
    var cs_t = 1e20;
    var cs_color = vec3<f32>(0.0);
    var cs_normal = vec3<f32>(0.0, 1.0, 0.0);

    let cs_outer = uniforms.cs_planet.w;
    if cs_outer > 0.0 {
        let cs_center = uniforms.cs_planet.xyz;
        let cs_inner = uniforms.cs_params.x;
        let shell = cs_outer - cs_inner;

        let oc = camera.pos - cs_center;
        let b = dot(oc, ray_dir);
        let c_outer = dot(oc, oc) - cs_outer * cs_outer;
        let disc = b * b - c_outer;
        if disc > 0.0 {
            let sq = sqrt(disc);
            let t_enter = max(-b - sq, 0.0);
            let t_exit = -b + sq;
            // Small epsilon in world units, anchored to shell
            // thickness. Advances t past a boundary so the next
            // iteration samples the neighboring cell instead of
            // looping on the same boundary plane.
            let eps = max(shell * 1e-5, 1e-7);
            var t = t_enter + eps;
            var steps = 0u;
            loop {
                if t >= t_exit || steps > 512u { break; }
                steps = steps + 1u;

                let p = camera.pos + ray_dir * t;
                let local = p - cs_center;
                let r = length(local);
                // The outer sphere intersect keeps us inside the
                // shell in practice, but guard both boundaries.
                if r >= cs_outer || r < cs_inner { break; }

                let n = local / r;
                let face = pick_face(n);
                let n_axis = face_normal(face);
                let u_axis = face_u_axis(face);
                let v_axis = face_v_axis(face);
                let axis_dot = dot(n, n_axis);
                let cube_u = dot(n, u_axis) / axis_dot;
                let cube_v = dot(n, v_axis) / axis_dot;
                let u_ea = cube_to_ea(cube_u);
                let v_ea = cube_to_ea(cube_v);

                let un = clamp((u_ea + 1.0) * 0.5, 0.0, 0.9999999);
                let vn = clamp((v_ea + 1.0) * 0.5, 0.0, 0.9999999);
                let rn = clamp((r - cs_inner) / shell, 0.0, 0.9999999);

                let walk = sample_face_tree(face_root(face), un, vn, rn);
                let block_id = walk.x;
                let term_depth = walk.y;
                if block_id != 0u {
                    cs_hit = true;
                    cs_t = t;
                    cs_normal = n;
                    let cell_color = palette.colors[block_id].rgb;
                    let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
                    let diffuse = max(dot(n, sun_dir), 0.0);
                    let ambient = 0.25;
                    var surface = cell_color * (ambient + diffuse * 0.75);

                    // Cursor highlight: the selected cell can live at
                    // any subtree depth, not just the finest. At
                    // `highlight_depth = 1` the wireframe encloses a
                    // 3³-cell chunk; at `highlight_depth = subtree_depth`
                    // it encloses one finest cell.
                    if uniforms.cs_params.w > 0.5 {
                        let hl_depth = max(uniforms.cs_params.z, 1.0);
                        let hl_cells = pow(3.0, hl_depth);
                        let hl_face = u32(uniforms.cs_highlight.x);
                        let iu_h = floor(un * hl_cells);
                        let iv_h = floor(vn * hl_cells);
                        let ir_h = floor(rn * hl_cells);
                        if face == hl_face
                            && iu_h == uniforms.cs_highlight.y
                            && iv_h == uniforms.cs_highlight.z
                            && ir_h == uniforms.cs_highlight.w
                        {
                            // Edge-distance in the highlight cell's
                            // local (0,1) space. Lines near any of
                            // the six bulged faces of the cell.
                            let cell_u = un * hl_cells - iu_h;
                            let cell_v = vn * hl_cells - iv_h;
                            let cell_r = rn * hl_cells - ir_h;
                            let edge = min(
                                min(min(cell_u, 1.0 - cell_u),
                                    min(cell_v, 1.0 - cell_v)),
                                min(cell_r, 1.0 - cell_r));
                            // Line width: ~3% of cell, roughly one
                            // visible "rim" at any zoom.
                            if edge < 0.05 {
                                surface = vec3<f32>(1.0, 0.9, 0.2);
                            }
                        }
                    }

                    cs_color = surface;
                    break;
                }

                // Empty cell — step to its exit. Cell bounds are at
                // the SUBTREE WALKER'S termination depth, not at a
                // fixed finest depth. This is the octree skip-empty
                // speedup: if the walker bottomed out at depth 1
                // because the whole 1/27 chunk is empty, we jump
                // 1/3 of the face per step instead of 1/3^depth.
                let cells_d = pow(3.0, f32(term_depth));
                let iu = floor(un * cells_d);
                let iv = floor(vn * cells_d);
                let ir = floor(rn * cells_d);

                // u-boundaries as world planes through center.
                let u_lo_ea = (iu       / cells_d) * 2.0 - 1.0;
                let u_hi_ea = ((iu+1.0) / cells_d) * 2.0 - 1.0;
                let k_u_lo = ea_to_cube(u_lo_ea);
                let k_u_hi = ea_to_cube(u_hi_ea);
                let n_u_lo = u_axis - k_u_lo * n_axis;
                let n_u_hi = u_axis - k_u_hi * n_axis;

                let v_lo_ea = (iv       / cells_d) * 2.0 - 1.0;
                let v_hi_ea = ((iv+1.0) / cells_d) * 2.0 - 1.0;
                let k_v_lo = ea_to_cube(v_lo_ea);
                let k_v_hi = ea_to_cube(v_hi_ea);
                let n_v_lo = v_axis - k_v_lo * n_axis;
                let n_v_hi = v_axis - k_v_hi * n_axis;

                // r-boundaries as spheres around center.
                let r_lo = cs_inner + (ir       / cells_d) * shell;
                let r_hi = cs_inner + ((ir+1.0) / cells_d) * shell;

                var t_next = t_exit + 1.0;
                t_next = min_after(t_next,
                    ray_plane_t(camera.pos, ray_dir, cs_center, n_u_lo), t);
                t_next = min_after(t_next,
                    ray_plane_t(camera.pos, ray_dir, cs_center, n_u_hi), t);
                t_next = min_after(t_next,
                    ray_plane_t(camera.pos, ray_dir, cs_center, n_v_lo), t);
                t_next = min_after(t_next,
                    ray_plane_t(camera.pos, ray_dir, cs_center, n_v_hi), t);
                t_next = min_after(t_next,
                    ray_sphere_after(camera.pos, ray_dir, cs_center, r_lo, t), t);
                t_next = min_after(t_next,
                    ray_sphere_after(camera.pos, ray_dir, cs_center, r_hi, t), t);

                if t_next >= t_exit { break; }
                // Advance just past the boundary so the next sample
                // lands inside the neighboring cell.
                t = t_next + eps;
            }
        }
    }

    var color: vec3<f32>;
    let tree_closer = result.hit && result.t <= cs_t;
    if tree_closer {
        let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
        let diffuse = max(dot(result.normal, sun_dir), 0.0);
        let ambient = 0.3;
        let lit = result.color * (ambient + diffuse * 0.7);
        color = pow(lit, vec3<f32>(1.0 / 2.2));
    } else if cs_hit {
        color = pow(cs_color, vec3<f32>(1.0 / 2.2));
    } else {
        let sky_t = ray_dir.y * 0.5 + 0.5;
        color = mix(vec3<f32>(0.7, 0.8, 0.95), vec3<f32>(0.3, 0.5, 0.85), sky_t);
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
