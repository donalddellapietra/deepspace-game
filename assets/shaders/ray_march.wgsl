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

/// Face index → body-node slot. Matches the insertion in
/// `spherical_worldgen::build`. Order: +X, −X, +Y, −Y, +Z, −Z.
/// slot_index(x,y,z) = z*9 + y*3 + x:
///   +X = (2,1,1) = 14   −X = (0,1,1) = 12
///   +Y = (1,2,1) = 16   −Y = (1,0,1) = 10
///   +Z = (1,1,2) = 22   −Z = (1,1,0) =  4
const FACE_CENTER_SLOTS: array<u32, 6> = array<u32, 6>(14u, 12u, 16u, 10u, 22u, 4u);

/// Buffer index of the face subtree at `face` inside a body node.
fn face_root_in_body(body_node_idx: u32, face: u32) -> u32 {
    return child_node_index(body_node_idx, FACE_CENTER_SLOTS[face]);
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

/// Sphere DDA inside a single `CubedSphereBody` cell.
///
/// `body_origin` / `body_extent` are the cell's lower-corner and
/// edge length in the render frame; `inner_r_local` / `outer_r_local`
/// are the body's radii in body-cell local coords (`[0, 1)`). The
/// sphere is centered at `body_origin + body_extent * 0.5`.
///
/// Returns `hit=true` + block color if the ray hits a solid face cell;
/// `hit=false` otherwise — the caller then advances the parent march
/// past this body cell.
fn march_sphere_body(
    body_node_idx: u32,
    body_origin: vec3<f32>,
    body_extent: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;

    let cs_center = body_origin + vec3<f32>(body_extent) * 0.5;
    let cs_outer = outer_r_local * body_extent;
    let cs_inner = inner_r_local * body_extent;
    let shell = cs_outer - cs_inner;

    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return result; }

    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    let eps = max(shell * 1e-5, 1e-7);
    var t = t_enter + eps;
    var steps = 0u;

    loop {
        if t >= t_exit || steps > 512u { break; }
        steps = steps + 1u;

        let p = ray_origin + ray_dir * t;
        let local = p - cs_center;
        let r = length(local);
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

        let face_root_idx = face_root_in_body(body_node_idx, face);
        let walk = sample_face_tree(face_root_idx, un, vn, rn);
        let block_id = walk.x;
        let term_depth = walk.y;
        if block_id != 0u {
            result.hit = true;
            result.t = t;
            result.normal = n;
            let cell_color = palette.colors[block_id].rgb;
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(n, sun_dir), 0.0);
            let ambient = 0.25;
            result.color = cell_color * (ambient + diffuse * 0.75);
            return result;
        }

        // Empty cell — step to exit via analytic boundary math
        // (same as the old fs_main sphere DDA).
        let cells_d = pow(3.0, f32(term_depth));
        let iu = floor(un * cells_d);
        let iv = floor(vn * cells_d);
        let ir = floor(rn * cells_d);

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

        let r_lo = cs_inner + (ir       / cells_d) * shell;
        let r_hi = cs_inner + ((ir+1.0) / cells_d) * shell;

        var t_next = t_exit + 1.0;
        t_next = min_after(t_next, ray_plane_t(ray_origin, ray_dir, cs_center, n_u_lo), t);
        t_next = min_after(t_next, ray_plane_t(ray_origin, ray_dir, cs_center, n_u_hi), t);
        t_next = min_after(t_next, ray_plane_t(ray_origin, ray_dir, cs_center, n_v_lo), t);
        t_next = min_after(t_next, ray_plane_t(ray_origin, ray_dir, cs_center, n_v_hi), t);
        t_next = min_after(t_next, ray_sphere_after(ray_origin, ray_dir, cs_center, r_lo, t), t);
        t_next = min_after(t_next, ray_sphere_after(ray_origin, ray_dir, cs_center, r_hi, t), t);

        if t_next >= t_exit { break; }
        t = t_next + eps;
    }
    return result;
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
            // Node — inspect its kind. CubedSphereBody children get
            // rendered by the sphere DDA; this replaces the old
            // top-of-fs_main cs_planet-uniform path.
            let ci = child_node_index(s_node_idx[depth], slot);
            let kind_tag = kinds[ci].tag;
            if kind_tag == 1u {
                let body_origin = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                let body_extent = s_cell_size[depth];
                let body_hit = march_sphere_body(
                    ci,
                    body_origin,
                    body_extent,
                    kinds[ci].inner_r,
                    kinds[ci].outer_r,
                    ray_origin,
                    ray_dir,
                );
                if body_hit.hit {
                    result.hit = true;
                    result.t = body_hit.t;
                    result.color = body_hit.color;
                    result.normal = body_hit.normal;
                    return result;
                }
                // Missed the sphere — step past this body cell as if
                // Empty. (The body occupies exactly one parent cell;
                // exiting its AABB IS exiting the parent cell.)
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

    let result = march(camera.pos, ray_dir);

    // Sphere rendering lives inside `march()` now — the old
    // uniform-driven cs_planet path in fs_main is gone.

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
