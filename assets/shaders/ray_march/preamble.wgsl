// Ray march shader for base-3 recursive voxel tree.
//
// One unified tree walker. When it descends into a Node child whose
// NodeKind is CubedSphereBody, it switches to the cubed-sphere DDA
// running in that body cell's local frame — no parallel uniforms,
// no separate face_root buffers, no absolute world coords. The body's
// `inner_r` / `outer_r` come from the per-node `node_kinds` buffer.

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
    /// 0 = Cartesian, 1 = body root, 2 = face-space root.
    root_kind: u32,
    /// Number of ancestor ribbon entries available. When the ray
    /// exits the frame's [0, 3)³ bubble at depth 0, the shader
    /// pops upward, walking ribbon[0]..ribbon[ribbon_count-1].
    /// 0 = no ancestors (frame is at world root).
    ribbon_count: u32,
    highlight_min: vec4<f32>,
    highlight_max: vec4<f32>,
    /// xy = (inner_r, outer_r) in body cell's local [0, 1) frame.
    /// Used when root_kind == 1 or 2.
    root_radii: vec4<f32>,
    /// x = face id, y = how many generic UVR pops remain before the
    /// next pop crosses from face root to body.
    root_face_meta: vec4<u32>,
    /// Current face-frame cell bounds inside the full face:
    /// (u_lo, v_lo, r_lo, size) in normalized [0, 1]^3.
    root_face_bounds: vec4<f32>,
    /// Reserved for sphere/body metadata.
    root_face_pop_pos: vec4<f32>,
    /// x = shell_count.
    cartesian_shell_meta: vec4<u32>,
    /// Packed as [ribbon_level_0, depth_limit_0, ribbon_level_1, depth_limit_1].
    cartesian_shell_pairs: array<vec4<u32>, 4>,
}

const ROOT_KIND_CARTESIAN: u32 = 0u;
const ROOT_KIND_BODY: u32 = 1u;
const ROOT_KIND_FACE: u32 = 2u;
const MAX_CARTESIAN_SHELLS: u32 = 8u;

/// One entry in the ancestor ribbon. `node_idx` is the buffer
/// index of the ancestor's node; `slot` is the slot in that
/// ancestor that contained the level we're popping FROM.
struct RibbonEntry {
    node_idx: u32,
    slot: u32,
}

struct NodeKindGpu {
    kind: u32,        // 0=Cartesian, 1=CubedSphereBody, 2=CubedSphereFace
    face: u32,
    inner_r: f32,
    outer_r: f32,
}

@group(0) @binding(0) var<storage, read> tree: array<u32>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> palette: Palette;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> node_kinds: array<NodeKindGpu>;
@group(0) @binding(5) var<storage, read> ribbon: array<RibbonEntry>;

// -------------- Tree access helpers --------------

fn child_packed(node_idx: u32, slot: u32) -> u32 {
    return tree[(node_idx * 27u + slot) * 2u];
}
fn child_node_index(node_idx: u32, slot: u32) -> u32 {
    return tree[(node_idx * 27u + slot) * 2u + 1u];
}
fn child_tag(packed: u32) -> u32 { return packed & 0xFFu; }
fn child_block_type(packed: u32) -> u32 { return (packed >> 8u) & 0xFFu; }

fn slot_from_xyz(x: i32, y: i32, z: i32) -> u32 {
    return u32(z * 9 + y * 3 + x);
}

// -------------- Cubed-sphere geometry helpers --------------

const PI_F: f32 = 3.1415926535;
const FRAC_PI_4: f32 = 0.785398163;

fn cube_to_ea(c: f32) -> f32 { return atan(c) * (4.0 / PI_F); }
fn ea_to_cube(e: f32) -> f32 { return tan(e * FRAC_PI_4); }

// Slot in a CubedSphereBody node's 27-grid that holds each face's
// subtree. Matches Rust `FACE_SLOTS`.
fn face_slot(face: u32) -> u32 {
    switch face {
        case 0u: { return 14u; } // PosX = (2, 1, 1)
        case 1u: { return 12u; } // NegX = (0, 1, 1)
        case 2u: { return 16u; } // PosY = (1, 2, 1)
        case 3u: { return 10u; } // NegY = (1, 0, 1)
        case 4u: { return 22u; } // PosZ = (1, 1, 2)
        default: { return 4u;  } // NegZ = (1, 1, 0)
    }
}

fn face_normal(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        case 1u: { return vec3<f32>(-1.0,  0.0,  0.0); }
        case 2u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        case 3u: { return vec3<f32>( 0.0, -1.0,  0.0); }
        case 4u: { return vec3<f32>( 0.0,  0.0,  1.0); }
        default: { return vec3<f32>( 0.0,  0.0, -1.0); }
    }
}

fn face_u_axis(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>( 0.0,  0.0, -1.0); }
        case 1u: { return vec3<f32>( 0.0,  0.0,  1.0); }
        case 2u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        case 3u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        case 4u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        default: { return vec3<f32>(-1.0,  0.0,  0.0); }
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

fn face_uv_to_dir(face: u32, u: f32, v: f32) -> vec3<f32> {
    let cube_u = ea_to_cube(u);
    let cube_v = ea_to_cube(v);
    let n = face_normal(face);
    let u_axis = face_u_axis(face);
    let v_axis = face_v_axis(face);
    return normalize(n + cube_u * u_axis + cube_v * v_axis);
}

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

fn ray_plane_t(origin: vec3<f32>, dir: vec3<f32>,
               through: vec3<f32>, plane_n: vec3<f32>) -> f32 {
    let denom = dot(dir, plane_n);
    if abs(denom) < 1e-12 { return -1.0; }
    return -dot(origin - through, plane_n) / denom;
}

// Numerical-Recipes stable ray-sphere intersection.
fn ray_sphere_after(origin: vec3<f32>, dir: vec3<f32>,
                    center: vec3<f32>, radius: f32, after: f32) -> f32 {
    let oc = origin - center;
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 { return -1.0; }
    let sq = sqrt(disc);
    let s = select(-1.0, 1.0, b >= 0.0);
    let q = -b - s * sq;
    if abs(q) < 1e-30 { return -1.0; }
    let t0 = q;
    let t1 = c / q;
    let t_lo = min(t0, t1);
    let t_hi = max(t0, t1);
    if t_lo > after { return t_lo; }
    if t_hi > after { return t_hi; }
    return -1.0;
}

// Result of walking a face subtree: which terminal (block_id),
// what depth it sits at, AND the cell's bounds in face-normalized
// `(un, vn, rn) ∈ [0, 1]³` coords. Bounds come from incremental
// Kahan-compensated accumulation during descent — they don't suffer
// the precision wall that `cells_d = pow(3, depth)` quantization
// hits past depth 14.
struct FaceWalkResult {
    block: u32,
    depth: u32,
    u_lo: f32,  // cell's lo bound in normalized face EA u
    v_lo: f32,
    r_lo: f32,
    size: f32,  // cell width = 3^-depth (same on all axes)
}

// Walk a face subtree from the body node's face-center child slot,
// returning the terminal AND the cell's normalized bounds. Bounds
// are accumulated via Kahan compensation so cumulative error stays
// at ~1 ULP regardless of depth (vs. ~depth ULPs naive).
//
// Loop bound stays at 22 to match the demo planet's face subtree
// depth (20 + 2 levels of overhead for body + face root). The bound
// disappears entirely with the unified-driver refactor.
const MAX_FACE_DEPTH: u32 = 63u;
const MAX_STACK_DEPTH: u32 = 64u;

fn walk_face_subtree(body_node_idx: u32, face: u32,
                     un_in: f32, vn_in: f32, rn_in: f32,
                     depth_limit: u32) -> FaceWalkResult {
    var result: FaceWalkResult;
    result.u_lo = 0.0;
    result.v_lo = 0.0;
    result.r_lo = 0.0;
    result.size = 1.0;
    result.depth = 1u;

    let fs = face_slot(face);
    let face_packed = child_packed(body_node_idx, fs);
    let face_tag = child_tag(face_packed);
    if face_tag == 0u {
        result.block = 0u;
        return result;
    }
    if face_tag == 1u {
        result.block = child_block_type(face_packed);
        return result;
    }
    var node = child_node_index(body_node_idx, fs);
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);

    // Kahan-compensated boundary accumulators per axis.
    var u_sum: f32 = 0.0; var u_comp: f32 = 0.0;
    var v_sum: f32 = 0.0; var v_comp: f32 = 0.0;
    var r_sum: f32 = 0.0; var r_comp: f32 = 0.0;
    var size: f32 = 1.0;

    let limit = min(depth_limit, MAX_FACE_DEPTH);
    if limit <= 1u {
        let bt = child_block_type(face_packed);
        result.block = select(0u, bt, bt != 255u);
        return result;
    }
    for (var d: u32 = 2u; d <= limit; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let slot = rs * 9u + vs * 3u + us;
        let packed = child_packed(node, slot);
        let tag = child_tag(packed);

        // Boundary update: this step's child within the parent
        // contributes (size/3) * slot to the lo-bound, and shrinks
        // size by 3. Done with Kahan compensation.
        let step_size = size * (1.0 / 3.0);
        let u_add = step_size * f32(us);
        let v_add = step_size * f32(vs);
        let r_add = step_size * f32(rs);

        let yu = u_add - u_comp;
        let tu = u_sum + yu;
        u_comp = (tu - u_sum) - yu;
        u_sum = tu;

        let yv = v_add - v_comp;
        let tv = v_sum + yv;
        v_comp = (tv - v_sum) - yv;
        v_sum = tv;

        let yr = r_add - r_comp;
        let tr = r_sum + yr;
        r_comp = (tr - r_sum) - yr;
        r_sum = tr;

        size = step_size;

        if tag == 0u || tag == 1u {
            result.block = select(0u, child_block_type(packed), tag == 1u);
            result.depth = d;
            result.u_lo = u_sum + u_comp;
            result.v_lo = v_sum + v_comp;
            result.r_lo = r_sum + r_comp;
            result.size = size;
            return result;
        }
        if d >= limit {
            let bt = child_block_type(packed);
            result.block = select(0u, bt, bt != 255u);
            result.depth = d;
            result.u_lo = u_sum + u_comp;
            result.v_lo = v_sum + v_comp;
            result.r_lo = r_sum + r_comp;
            result.size = size;
            return result;
        }
        node = child_node_index(node, slot);
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
    }

    // Hit max depth without terminal: report deepest LOD bounds.
    result.block = 0u;
    result.depth = limit;
    result.u_lo = u_sum + u_comp;
    result.v_lo = v_sum + v_comp;
    result.r_lo = r_sum + r_comp;
    result.size = size;
    return result;
}

const FACE_ROOT_LOD_THRESHOLD_PIXELS: f32 = 4.0;

fn sample_face_node(node_idx: u32,
                    un_in: f32, vn_in: f32, rn_in: f32,
                    depth_limit: u32) -> vec2<u32> {
    var node = node_idx;
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);
    let limit = min(depth_limit, MAX_FACE_DEPTH);
    for (var d: u32 = 1u; d <= limit; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let slot = rs * 9u + vs * 3u + us;
        let packed = child_packed(node, slot);
        let tag = child_tag(packed);
        if tag == 0u {
            return vec2<u32>(0u, d);
        }
        if tag == 1u {
            return vec2<u32>(child_block_type(packed), d);
        }
        node = child_node_index(node, slot);
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
    }
    return vec2<u32>(0u, limit);
}

fn walk_face_node(node_idx: u32,
                  un_in: f32, vn_in: f32, rn_in: f32,
                  ray_t: f32,
                  lod_scale: f32) -> FaceWalkResult {
    var result: FaceWalkResult;
    result.block = 0u;
    result.depth = 0u;
    result.u_lo = 0.0;
    result.v_lo = 0.0;
    result.r_lo = 0.0;
    result.size = 1.0;

    var node = node_idx;
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);

    var u_lo: f32 = 0.0;
    var v_lo: f32 = 0.0;
    var r_lo: f32 = 0.0;
    var size: f32 = 1.0;

    for (var d: u32 = 1u; d <= MAX_FACE_DEPTH; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let slot = rs * 9u + vs * 3u + us;
        let packed = child_packed(node, slot);
        let tag = child_tag(packed);

        let step_size = size * (1.0 / 3.0);
        u_lo = u_lo + step_size * f32(us);
        v_lo = v_lo + step_size * f32(vs);
        r_lo = r_lo + step_size * f32(rs);
        size = step_size;

        if tag == 0u || tag == 1u {
            result.block = select(0u, child_block_type(packed), tag == 1u);
            result.depth = d;
            result.u_lo = u_lo;
            result.v_lo = v_lo;
            result.r_lo = r_lo;
            result.size = size;
            return result;
        }

        let cell_size_local = 3.0 * size;
        let ray_dist = max(ray_t, 0.001);
        let lod_pixels = cell_size_local / ray_dist * lod_scale;
        let at_lod = lod_pixels < FACE_ROOT_LOD_THRESHOLD_PIXELS;
        let at_max = d >= uniforms.max_depth;
        if at_lod || at_max {
            let bt = child_block_type(packed);
            result.block = select(0u, bt, bt != 255u);
            result.depth = d;
            result.u_lo = u_lo;
            result.v_lo = v_lo;
            result.r_lo = r_lo;
            result.size = size;
            return result;
        }

        node = child_node_index(node, slot);
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
    }

    result.depth = MAX_FACE_DEPTH;
    result.u_lo = u_lo;
    result.v_lo = v_lo;
    result.r_lo = r_lo;
    result.size = size;
    return result;
}

fn face_point_to_body_with_bounds(point: vec3<f32>, bounds: vec4<f32>) -> vec3<f32> {
    let face = uniforms.root_face_meta.x;
    let un = bounds.x + (point.x / 3.0) * bounds.w;
    let vn = bounds.y + (point.y / 3.0) * bounds.w;
    let rn = bounds.z + (point.z / 3.0) * bounds.w;
    let dir = face_uv_to_dir(face, un * 2.0 - 1.0, vn * 2.0 - 1.0);
    let radius_local = uniforms.root_radii.x + rn * (uniforms.root_radii.y - uniforms.root_radii.x);
    let body_local = vec3<f32>(0.5) + dir * radius_local;
    return body_local * 3.0;
}

fn root_face_point_to_body(point: vec3<f32>) -> vec3<f32> {
    return face_point_to_body_with_bounds(point, uniforms.root_face_bounds);
}

fn face_root_point_to_body(point: vec3<f32>) -> vec3<f32> {
    return face_point_to_body_with_bounds(point, vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

fn face_dir_to_body(origin: vec3<f32>, dir: vec3<f32>, bounds: vec4<f32>) -> vec3<f32> {
    let eps = max(bounds.w * 1e-3, 1e-5);
    let p0 = face_point_to_body_with_bounds(origin, bounds);
    let p1 = face_point_to_body_with_bounds(origin + dir * eps, bounds);
    let d = p1 - p0;
    if dot(d, d) < 1e-12 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(d);
}

fn face_local_normal_to_body(point: vec3<f32>, normal: vec3<f32>, bounds: vec4<f32>) -> vec3<f32> {
    let p = face_point_to_body_with_bounds(point, bounds);
    let dir = normalize(p - vec3<f32>(1.5));
    let face = uniforms.root_face_meta.x;
    let n_axis = face_normal(face);
    let u_axis = face_u_axis(face);
    let v_axis = face_v_axis(face);
    if abs(normal.z) > 0.5 {
        return normalize(dir * sign(normal.z));
    }
    if abs(normal.y) > 0.5 {
        return normalize(v_axis * sign(normal.y));
    }
    let axis_dot = max(dot(dir, n_axis), 1e-5);
    let u_world = normalize(u_axis - n_axis * (dot(dir, u_axis) / axis_dot));
    return normalize(u_world * sign(normal.x));
}

fn face_box_to_body_bounds(hmin: vec3<f32>, hmax: vec3<f32>, bounds: vec4<f32>) -> mat2x3<f32> {
    var mn = vec3<f32>(1e20);
    var mx = vec3<f32>(-1e20);
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let corner = vec3<f32>(
            select(hmin.x, hmax.x, (i & 1u) != 0u),
            select(hmin.y, hmax.y, (i & 2u) != 0u),
            select(hmin.z, hmax.z, (i & 4u) != 0u),
        );
        let p = face_point_to_body_with_bounds(corner, bounds);
        mn = min(mn, p);
        mx = max(mx, p);
    }
    return mat2x3<f32>(mn, mx);
}
