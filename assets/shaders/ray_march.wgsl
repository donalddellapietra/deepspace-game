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
    root_face_pop_pos: vec4<f32>,
}

const ROOT_KIND_CARTESIAN: u32 = 0u;
const ROOT_KIND_BODY: u32 = 1u;
const ROOT_KIND_FACE: u32 = 2u;

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

// Shell architecture: each march_cartesian call uses a bounded depth
// budget. The ribbon provides outer shells for context at coarser
// scales. This prevents pathological traversal in heavily deduplicated
// trees where depth_limit=6+ causes the DDA to exhaust iterations.
const SHELL_BUDGET: u32 = 3u;

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

// -------------- Ray-AABB --------------

struct BoxHit { t_enter: f32, t_exit: f32, }

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

fn pow3_u(exp: u32) -> f32 {
    var scale = 1.0;
    for (var i: u32 = 0u; i < exp; i = i + 1u) {
        scale = scale * 3.0;
    }
    return scale;
}

struct HitResult {
    hit: bool,
    color: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    /// Which ancestor-pop level the hit happened in. 0 = original
    /// camera frame; >0 = popped that many times into ancestors.
    /// `t` is in this frame's units, not the camera's.
    frame_level: u32,
    highlight_min: vec3<f32>,
    highlight_max: vec3<f32>,
    frame_scale: f32,
    cell_min: vec3<f32>,
    cell_size: f32,
}

fn face_uv_for_normal(local: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let an = abs(normal);
    if an.x >= an.y && an.x >= an.z {
        return local.yz;
    }
    if an.y >= an.z {
        return local.xz;
    }
    return local.xy;
}

fn cube_face_bevel(local: vec3<f32>, normal: vec3<f32>) -> f32 {
    let uv = face_uv_for_normal(local, normal);
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return smoothstep(0.02, 0.14, edge);
}

fn sphere_cell_shape(cell_u: f32, cell_v: f32, cell_r: f32) -> f32 {
    let face_edge = min(
        min(cell_u, 1.0 - cell_u),
        min(cell_v, 1.0 - cell_v),
    );
    let bevel = smoothstep(0.02, 0.14, face_edge);
    _ = cell_r;
    return 0.78 + 0.22 * bevel;
}

fn max_component(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

fn march_face_root(
    root_node_idx: u32,
    ray_origin_body: vec3<f32>,
    ray_dir: vec3<f32>,
    bounds: vec4<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.highlight_min = vec3<f32>(0.0);
    result.highlight_max = vec3<f32>(0.0);
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let face = uniforms.root_face_meta.x;
    let cs_center = vec3<f32>(1.5);
    let cs_outer = uniforms.root_radii.y * 3.0;
    let cs_inner = uniforms.root_radii.x * 3.0;
    let shell = cs_outer - cs_inner;
    let oc = ray_origin_body - cs_center;
    let b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let eps_init = max(shell * 1e-5, 1e-7);
    var t = t_enter + eps_init;
    var steps = 0u;
    var last_face_id: u32 = 6u;
    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;

        let local = oc + ray_dir * t;
        let r = length(local);
        if r >= cs_outer || r < cs_inner { break; }
        let n = local / r;
        let hit_face = pick_face(n);
        if hit_face != face { break; }
        let n_axis = face_normal(face);
        let u_axis = face_u_axis(face);
        let v_axis = face_v_axis(face);
        let axis_dot = dot(n, n_axis);
        let cube_u = dot(n, u_axis) / axis_dot;
        let cube_v = dot(n, v_axis) / axis_dot;
        let u_ea = cube_to_ea(cube_u);
        let v_ea = cube_to_ea(cube_v);
        let un_abs = clamp((u_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let vn_abs = clamp((v_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let rn_abs = clamp((r - cs_inner) / shell, 0.0, 0.9999999);
        if un_abs < bounds.x || un_abs >= bounds.x + bounds.w ||
           vn_abs < bounds.y || vn_abs >= bounds.y + bounds.w ||
           rn_abs < bounds.z || rn_abs >= bounds.z + bounds.w {
            break;
        }

        let un_local = (un_abs - bounds.x) / bounds.w;
        let vn_local = (vn_abs - bounds.y) / bounds.w;
        let rn_local = (rn_abs - bounds.z) / bounds.w;
        let walk = sample_face_node(
            root_node_idx,
            un_local,
            vn_local,
            rn_local,
            uniforms.max_depth,
        );
        let block_id = walk.x;
        let term_depth = walk.y;
        let cells_d = pow3_u(term_depth);
        let iu = floor(un_local * cells_d);
        let iv = floor(vn_local * cells_d);
        let ir = floor(rn_local * cells_d);

        if block_id != 0u {
            var hit_normal: vec3<f32>;
            switch last_face_id {
                case 0u: { hit_normal = -u_axis; }
                case 1u: { hit_normal =  u_axis; }
                case 2u: { hit_normal = -v_axis; }
                case 3u: { hit_normal =  v_axis; }
                case 4u: { hit_normal = -n; }
                case 5u: { hit_normal =  n; }
                default: { hit_normal =  n; }
            }
            result.hit = true;
            result.t = t;
            let cell_u = un_local * cells_d - iu;
            let cell_v = vn_local * cells_d - iv;
            let cell_r = rn_local * cells_d - ir;
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let axis_tint = abs(hit_normal.y) * 1.0
                          + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let block_shape = sphere_cell_shape(cell_u, cell_v, cell_r);
            result.color = palette.colors[block_id].rgb * (ambient + diffuse * 0.78) * axis_tint * block_shape;
            result.normal = hit_normal;
            return result;
        }

        let cell_lo = 1.0 / cells_d;
        let u_lo = iu / cells_d;
        let v_lo = iv / cells_d;
        let r_lo_local = ir / cells_d;
        let u_lo_ea = (bounds.x + u_lo * bounds.w) * 2.0 - 1.0;
        let u_hi_ea = (bounds.x + (u_lo + cell_lo) * bounds.w) * 2.0 - 1.0;
        let n_u_lo = u_axis - ea_to_cube(u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(u_hi_ea) * n_axis;

        let v_lo_ea = (bounds.y + v_lo * bounds.w) * 2.0 - 1.0;
        let v_hi_ea = (bounds.y + (v_lo + cell_lo) * bounds.w) * 2.0 - 1.0;
        let n_v_lo = v_axis - ea_to_cube(v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(v_hi_ea) * n_axis;

        let r_lo = cs_inner + (bounds.z + r_lo_local * bounds.w) * shell;
        let r_hi = cs_inner + (bounds.z + (r_lo_local + cell_lo) * bounds.w) * shell;

        var t_next = t_exit + 1.0;
        var winning_face: u32 = 6u;
        let zero3 = vec3<f32>(0.0);
        let cand_u_lo = ray_plane_t(oc, ray_dir, zero3, n_u_lo);
        if cand_u_lo > t && cand_u_lo < t_next { t_next = cand_u_lo; winning_face = 0u; }
        let cand_u_hi = ray_plane_t(oc, ray_dir, zero3, n_u_hi);
        if cand_u_hi > t && cand_u_hi < t_next { t_next = cand_u_hi; winning_face = 1u; }
        let cand_v_lo = ray_plane_t(oc, ray_dir, zero3, n_v_lo);
        if cand_v_lo > t && cand_v_lo < t_next { t_next = cand_v_lo; winning_face = 2u; }
        let cand_v_hi = ray_plane_t(oc, ray_dir, zero3, n_v_hi);
        if cand_v_hi > t && cand_v_hi < t_next { t_next = cand_v_hi; winning_face = 3u; }
        let cand_r_lo = ray_sphere_after(oc, ray_dir, zero3, r_lo, t);
        if cand_r_lo > t && cand_r_lo < t_next { t_next = cand_r_lo; winning_face = 4u; }
        let cand_r_hi = ray_sphere_after(oc, ray_dir, zero3, r_hi, t);
        if cand_r_hi > t && cand_r_hi < t_next { t_next = cand_r_hi; winning_face = 5u; }

        if t_next >= t_exit { break; }
        last_face_id = winning_face;
        let t_ulp = max(abs(t) * 1.2e-7, 1e-30);
        let cell_eps = max(shell * cell_lo * bounds.w * 1e-3, t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    return result;
}

// Sphere DDA running inside one CubedSphereBody cell. The body cell
// is given in the render-frame's coords (origin + size); radii are
// scaled into the same frame. Returns hit/miss; on miss the caller
// continues the Cartesian DDA past the body cell.
fn sphere_in_cell(
    body_node_idx: u32,
    body_cell_origin: vec3<f32>,
    body_cell_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.highlight_min = vec3<f32>(0.0);
    result.highlight_max = vec3<f32>(0.0);
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let cs_center = body_cell_origin + vec3<f32>(body_cell_size * 0.5);
    let cs_outer = outer_r_local * body_cell_size;
    let cs_inner = inner_r_local * body_cell_size;
    let shell = cs_outer - cs_inner;

    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let eps_init = max(shell * 1e-5, 1e-7);
    var t = t_enter + eps_init;
    var steps = 0u;
    var last_face_id: u32 = 6u;
    var deepest_term_depth: u32 = 1u;
    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;

        let local = oc + ray_dir * t;
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

        let walk = walk_face_subtree(body_node_idx, face, un, vn, rn, uniforms.max_depth);
        let block_id = walk.block;
        let term_depth = walk.depth;

        if block_id != 0u {
            var hit_normal: vec3<f32>;
            switch last_face_id {
                case 0u: { hit_normal = -u_axis; }
                case 1u: { hit_normal =  u_axis; }
                case 2u: { hit_normal = -v_axis; }
                case 3u: { hit_normal =  v_axis; }
                case 4u: { hit_normal = -n; }
                case 5u: { hit_normal =  n; }
                default: { hit_normal =  n; }
            }
            let cell_color = palette.colors[block_id].rgb;
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let axis_tint = abs(hit_normal.y) * 1.0
                          + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let cell_u = clamp((un - walk.u_lo) / walk.size, 0.0, 1.0);
            let cell_v = clamp((vn - walk.v_lo) / walk.size, 0.0, 1.0);
            let cell_r = clamp((rn - walk.r_lo) / walk.size, 0.0, 1.0);
            let block_shape = sphere_cell_shape(cell_u, cell_v, cell_r);
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            result.color = cell_color * (ambient + diffuse * 0.78) * axis_tint * block_shape;
            return result;
        }

        deepest_term_depth = max(deepest_term_depth, term_depth);
        // Cell bounds come from the walker's Kahan-compensated
        // accumulation. No more `floor(un * 3^depth)` quantization,
        // so depths past 14 stay precision-correct.
        let u_lo_ea = walk.u_lo * 2.0 - 1.0;
        let u_hi_ea = (walk.u_lo + walk.size) * 2.0 - 1.0;
        let n_u_lo = u_axis - ea_to_cube(u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(u_hi_ea) * n_axis;

        let v_lo_ea = walk.v_lo * 2.0 - 1.0;
        let v_hi_ea = (walk.v_lo + walk.size) * 2.0 - 1.0;
        let n_v_lo = v_axis - ea_to_cube(v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(v_hi_ea) * n_axis;

        let r_lo = cs_inner + walk.r_lo * shell;
        let r_hi = cs_inner + (walk.r_lo + walk.size) * shell;

        var t_next = t_exit + 1.0;
        var winning_face: u32 = 6u;
        let zero3 = vec3<f32>(0.0);
        let cand_u_lo = ray_plane_t(oc, ray_dir, zero3, n_u_lo);
        if cand_u_lo > t && cand_u_lo < t_next { t_next = cand_u_lo; winning_face = 0u; }
        let cand_u_hi = ray_plane_t(oc, ray_dir, zero3, n_u_hi);
        if cand_u_hi > t && cand_u_hi < t_next { t_next = cand_u_hi; winning_face = 1u; }
        let cand_v_lo = ray_plane_t(oc, ray_dir, zero3, n_v_lo);
        if cand_v_lo > t && cand_v_lo < t_next { t_next = cand_v_lo; winning_face = 2u; }
        let cand_v_hi = ray_plane_t(oc, ray_dir, zero3, n_v_hi);
        if cand_v_hi > t && cand_v_hi < t_next { t_next = cand_v_hi; winning_face = 3u; }
        let cand_r_lo = ray_sphere_after(oc, ray_dir, zero3, r_lo, t);
        if cand_r_lo > t && cand_r_lo < t_next { t_next = cand_r_lo; winning_face = 4u; }
        let cand_r_hi = ray_sphere_after(oc, ray_dir, zero3, r_hi, t);
        if cand_r_hi > t && cand_r_hi < t_next { t_next = cand_r_hi; winning_face = 5u; }

        if t_next >= t_exit { break; }
        last_face_id = winning_face;
        let t_ulp = max(abs(t) * 1.2e-7, 1e-30);
        let cell_eps = max(shell * walk.size * 1e-3, t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    return result;
}

fn sphere_in_face_window(
    body_node_idx: u32,
    face: u32,
    face_u_min: f32,
    face_v_min: f32,
    face_r_min: f32,
    face_size: f32,
    face_depth: u32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin_body: vec3<f32>,
    ray_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.highlight_min = vec3<f32>(0.0);
    result.highlight_max = vec3<f32>(0.0);
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let body_origin = vec3<f32>(0.0);
    let body_size = 3.0;
    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let cs_outer = outer_r_local * body_size;
    let cs_inner = inner_r_local * body_size;
    let shell = cs_outer - cs_inner;

    let oc = ray_origin_body - cs_center;
    let b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let eps_init = max(shell * 1e-5, 1e-7);
    var t = t_enter + eps_init;
    var steps = 0u;
    var last_face_id: u32 = 6u;
    let depth_limit = min(MAX_FACE_DEPTH, face_depth + uniforms.max_depth);
    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;

        let local = oc + ray_dir * t;
        let r = length(local);
        if r >= cs_outer || r < cs_inner { break; }

        let n = local / r;
        let hit_face = pick_face(n);
        if hit_face != face { break; }
        let n_axis = face_normal(face);
        let u_axis = face_u_axis(face);
        let v_axis = face_v_axis(face);
        let axis_dot = dot(n, n_axis);
        let cube_u = dot(n, u_axis) / axis_dot;
        let cube_v = dot(n, v_axis) / axis_dot;
        let u_ea = cube_to_ea(cube_u);
        let v_ea = cube_to_ea(cube_v);

        let un_abs = clamp((u_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let vn_abs = clamp((v_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let rn_abs = clamp((r - cs_inner) / shell, 0.0, 0.9999999);
        if un_abs < face_u_min || un_abs >= face_u_min + face_size ||
           vn_abs < face_v_min || vn_abs >= face_v_min + face_size ||
           rn_abs < face_r_min || rn_abs >= face_r_min + face_size {
            break;
        }

        let walk = walk_face_subtree(body_node_idx, face, un_abs, vn_abs, rn_abs, depth_limit);
        let block_id = walk.block;
        if block_id != 0u {
            var hit_normal: vec3<f32>;
            switch last_face_id {
                case 0u: { hit_normal = -u_axis; }
                case 1u: { hit_normal =  u_axis; }
                case 2u: { hit_normal = -v_axis; }
                case 3u: { hit_normal =  v_axis; }
                case 4u: { hit_normal = -n; }
                case 5u: { hit_normal =  n; }
                default: { hit_normal =  n; }
            }
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            let cell_u = clamp((un_abs - walk.u_lo) / walk.size, 0.0, 1.0);
            let cell_v = clamp((vn_abs - walk.v_lo) / walk.size, 0.0, 1.0);
            let cell_r = clamp((rn_abs - walk.r_lo) / walk.size, 0.0, 1.0);
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let axis_tint = abs(hit_normal.y) * 1.0
                          + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let block_shape = sphere_cell_shape(cell_u, cell_v, cell_r);
            result.color = palette.colors[block_id].rgb * (ambient + diffuse * 0.78) * axis_tint * block_shape;
            return result;
        }

        let u_lo_ea = walk.u_lo * 2.0 - 1.0;
        let u_hi_ea = (walk.u_lo + walk.size) * 2.0 - 1.0;
        let n_u_lo = u_axis - ea_to_cube(u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(u_hi_ea) * n_axis;

        let v_lo_ea = walk.v_lo * 2.0 - 1.0;
        let v_hi_ea = (walk.v_lo + walk.size) * 2.0 - 1.0;
        let n_v_lo = v_axis - ea_to_cube(v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(v_hi_ea) * n_axis;

        let r_lo = cs_inner + walk.r_lo * shell;
        let r_hi = cs_inner + (walk.r_lo + walk.size) * shell;

        var t_next = t_exit + 1.0;
        var winning_face: u32 = 6u;
        let zero3 = vec3<f32>(0.0);
        let cand_u_lo = ray_plane_t(oc, ray_dir, zero3, n_u_lo);
        if cand_u_lo > t && cand_u_lo < t_next { t_next = cand_u_lo; winning_face = 0u; }
        let cand_u_hi = ray_plane_t(oc, ray_dir, zero3, n_u_hi);
        if cand_u_hi > t && cand_u_hi < t_next { t_next = cand_u_hi; winning_face = 1u; }
        let cand_v_lo = ray_plane_t(oc, ray_dir, zero3, n_v_lo);
        if cand_v_lo > t && cand_v_lo < t_next { t_next = cand_v_lo; winning_face = 2u; }
        let cand_v_hi = ray_plane_t(oc, ray_dir, zero3, n_v_hi);
        if cand_v_hi > t && cand_v_hi < t_next { t_next = cand_v_hi; winning_face = 3u; }
        let cand_r_lo = ray_sphere_after(oc, ray_dir, zero3, r_lo, t);
        if cand_r_lo > t && cand_r_lo < t_next { t_next = cand_r_lo; winning_face = 4u; }
        let cand_r_hi = ray_sphere_after(oc, ray_dir, zero3, r_hi, t);
        if cand_r_hi > t && cand_r_hi < t_next { t_next = cand_r_hi; winning_face = 5u; }

        if t_next >= t_exit { break; }
        last_face_id = winning_face;
        let t_ulp = max(abs(t) * 1.2e-7, 1e-30);
        let cell_eps = max(shell * walk.size * 1e-3, t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    return result;
}

// -------------- Stack-based Cartesian tree DDA --------------

/// Cartesian DDA in a single frame rooted at `root_node_idx`. The
/// frame's cell spans `[0, 3)³` in `ray_origin/ray_dir` coords.
/// Returns hit on cell terminal; on miss (ray exits the frame),
/// returns hit=false so the caller can pop to the ancestor ribbon.
fn march_cartesian(
    root_node_idx: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>,
    depth_limit: u32, skip_node_idx: u32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.highlight_min = vec3<f32>(0.0);
    result.highlight_max = vec3<f32>(0.0);
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );
    // After ribbon pops, ray_dir magnitude shrinks (÷3 per pop).
    // LOD pixel calculations need world-space distances, so scale
    // side_dist by ray_metric to get actual distance.
    let ray_metric = max(length(ray_dir), 1e-6);
    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );
    let delta_dist = abs(inv_dir);

    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_cell: array<vec3<i32>, MAX_STACK_DEPTH>;
    var s_side_dist: array<vec3<f32>, MAX_STACK_DEPTH>;
    var s_node_origin: array<vec3<f32>, MAX_STACK_DEPTH>;
    var s_cell_size: array<f32, MAX_STACK_DEPTH>;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    s_node_idx[0] = root_node_idx;
    s_node_origin[0] = vec3<f32>(0.0);
    s_cell_size[0] = 1.0;

    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }

    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = ray_origin + ray_dir * t_start;

    s_cell[0] = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );
    let cell_f = vec3<f32>(s_cell[0]);
    s_side_dist[0] = vec3<f32>(
        select((cell_f.x - entry_pos.x) * inv_dir.x,
               (cell_f.x + 1.0 - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
        select((cell_f.y - entry_pos.y) * inv_dir.y,
               (cell_f.y + 1.0 - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
        select((cell_f.z - entry_pos.z) * inv_dir.z,
               (cell_f.z + 1.0 - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
    );

    var iterations = 0u;
    let max_iterations = 2048u;

    loop {
        if iterations >= max_iterations { break; }
        iterations += 1u;

        let cell = s_cell[depth];

        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth -= 1u;

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
            // Empty — DDA advance.
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
            let cell_min_h = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
            let cell_max_h = cell_min_h + vec3<f32>(s_cell_size[depth]);
            let cell_box_h = ray_box(ray_origin, inv_dir, cell_min_h, cell_max_h);
            result.hit = true;
            result.t = max(cell_box_h.t_enter, 0.0);
            result.color = palette.colors[child_block_type(packed)].rgb;
            result.normal = normal;
            result.cell_min = cell_min_h;
            result.cell_size = s_cell_size[depth];
            return result;
        } else {
            // tag == 2u: Node child. Look up its kind.
            let child_idx = child_node_index(s_node_idx[depth], slot);
            let kind = node_kinds[child_idx].kind;

            if kind == 1u {
                // CubedSphereBody: dispatch sphere DDA in this body's cell.
                let body_origin = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                let body_size = s_cell_size[depth];
                let inner_r = node_kinds[child_idx].inner_r;
                let outer_r = node_kinds[child_idx].outer_r;
                let sph = sphere_in_cell(
                    child_idx, body_origin, body_size,
                    inner_r, outer_r, ray_origin, ray_dir,
                );
                if sph.hit {
                    return sph;
                }
                // Sphere missed — advance Cartesian DDA past this cell.
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
            if false {
                // Real path (re-enable after diagnostic confirms dispatch):
                let body_origin = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                let body_size = s_cell_size[depth];
                let inner_r = node_kinds[child_idx].inner_r;
                let outer_r = node_kinds[child_idx].outer_r;
                let sph = sphere_in_cell(
                    child_idx, body_origin, body_size,
                    inner_r, outer_r, ray_origin, ray_dir,
                );
                if sph.hit {
                    return sph;
                }
                // Sphere missed — advance Cartesian DDA past this cell.
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

            // Shell skip: when re-entering a parent shell after a
            // ribbon pop, skip the child node we already traversed
            // in the inner shell. This prevents redundant descent
            // into the same subtree at each ribbon level.
            if depth == 0u && child_idx == skip_node_idx {
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

            // Cartesian Node: depth/LOD check, then descend.
            let at_max = depth + 1u >= depth_limit || depth + 1u >= MAX_STACK_DEPTH;
            let child_cell_size = s_cell_size[depth] / 3.0;
            let cell_world_size = child_cell_size;
            let min_side = min(s_side_dist[depth].x, min(s_side_dist[depth].y, s_side_dist[depth].z));
            let ray_dist = max(min_side * ray_metric, 0.001);
            let lod_pixels = cell_world_size / ray_dist * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            let at_lod = lod_pixels < 1.0;

            if at_max || at_lod {
                let bt = child_block_type(packed);
                if bt == 255u {
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
                    let cell_min_l = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                    let cell_max_l = cell_min_l + vec3<f32>(s_cell_size[depth]);
                    let cell_box_l = ray_box(ray_origin, inv_dir, cell_min_l, cell_max_l);
                    result.hit = true;
                    result.t = max(cell_box_l.t_enter, 0.0);
                    result.color = palette.colors[bt].rgb;
                    result.normal = normal;
                    result.cell_min = cell_min_l;
                    result.cell_size = s_cell_size[depth];
                    return result;
                }
            } else {
                let parent_origin = s_node_origin[depth];
                let parent_cell_size = s_cell_size[depth];
                let child_origin = parent_origin + vec3<f32>(cell) * parent_cell_size;

                let child_max = child_origin + vec3<f32>(parent_cell_size);
                let child_hit = ray_box(ray_origin, inv_dir, child_origin, child_max);
                let ct_start = max(child_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
                let child_entry = ray_origin + ray_dir * ct_start;
                let local_entry = (child_entry - child_origin) / child_cell_size;

                depth += 1u;
                s_node_idx[depth] = child_idx;
                s_node_origin[depth] = child_origin;
                s_cell_size[depth] = child_cell_size;
                s_cell[depth] = vec3<i32>(
                    clamp(i32(floor(local_entry.x)), 0, 2),
                    clamp(i32(floor(local_entry.y)), 0, 2),
                    clamp(i32(floor(local_entry.z)), 0, 2),
                );
                let lc = vec3<f32>(s_cell[depth]);
                s_side_dist[depth] = vec3<f32>(
                    select((child_origin.x + lc.x * child_cell_size - ray_origin.x) * inv_dir.x,
                           (child_origin.x + (lc.x + 1.0) * child_cell_size - ray_origin.x) * inv_dir.x, ray_dir.x >= 0.0),
                    select((child_origin.y + lc.y * child_cell_size - ray_origin.y) * inv_dir.y,
                           (child_origin.y + (lc.y + 1.0) * child_cell_size - ray_origin.y) * inv_dir.y, ray_dir.y >= 0.0),
                    select((child_origin.z + lc.z * child_cell_size - ray_origin.z) * inv_dir.z,
                           (child_origin.z + (lc.z + 1.0) * child_cell_size - ray_origin.z) * inv_dir.z, ray_dir.z >= 0.0),
                );
            }
        }
    }

    return result;
}

// -------------- Frame dispatch + ancestor pop --------------

/// Top-level march. Dispatches the current frame's DDA on its
/// NodeKind (Cartesian or sphere body), then on miss pops to the
/// next ancestor in the ribbon and continues. When ribbon is
/// exhausted, returns sky (hit=false).
///
/// Each pop transforms the ray into the parent's frame coords:
/// `parent_pos = slot_xyz + frame_pos / 3`, `parent_dir = frame_dir / 3`.
/// The parent's frame cell still spans `[0, 3)³` in its own
/// coords, so the inner DDA is unchanged — only the ray is
/// rescaled and the buffer node_idx swapped.
fn march(world_ray_origin: vec3<f32>, world_ray_dir: vec3<f32>) -> HitResult {
    var ray_origin = world_ray_origin;
    var ray_dir = world_ray_dir;
    var current_idx = uniforms.root_index;
    var current_kind = uniforms.root_kind;
    var inner_r = uniforms.root_radii.x;
    var outer_r = uniforms.root_radii.y;
    var cur_face_bounds = uniforms.root_face_bounds;
    var ribbon_level: u32 = 0u;
    var cur_hmin = uniforms.highlight_min.xyz;
    var cur_hmax = uniforms.highlight_max.xyz;
    var cur_scale: f32 = 1.0;

    // skip_node_idx: after a ribbon pop, the inner shell's root node
    // index is passed here so march_cartesian skips re-entering that
    // subtree (it was already traversed by the inner shell).
    var skip_node_idx: u32 = 0xFFFFFFFFu;

    var hops: u32 = 0u;
    loop {
        if hops > 80u { break; }
        hops = hops + 1u;

        var r: HitResult;
        if current_kind == ROOT_KIND_BODY {
            let body_origin = vec3<f32>(0.0);
            let body_size = 3.0;
            r = sphere_in_cell(
                current_idx, body_origin, body_size,
                inner_r, outer_r, ray_origin, ray_dir,
            );
        } else if current_kind == ROOT_KIND_FACE {
            r = march_face_root(current_idx, ray_origin, ray_dir, cur_face_bounds);
        } else {
            r = march_cartesian(current_idx, ray_origin, ray_dir, SHELL_BUDGET, skip_node_idx);
        }
        if r.hit {
            r.frame_level = ribbon_level;
            r.highlight_min = cur_hmin;
            r.highlight_max = cur_hmax;
            r.frame_scale = cur_scale;
            // Transform cell_min/cell_size from the popped frame back
            // to the camera frame so the fragment shader's bevel/grid
            // computation uses consistent coordinates.
            if cur_scale < 1.0 {
                let hit_popped = ray_origin + ray_dir * r.t;
                let cell_local = clamp(
                    (hit_popped - r.cell_min) / r.cell_size,
                    vec3<f32>(0.0), vec3<f32>(1.0),
                );
                let hit_camera = world_ray_origin + world_ray_dir * r.t;
                r.cell_size = r.cell_size / cur_scale;
                r.cell_min = hit_camera - cell_local * r.cell_size;
            }
            return r;
        }

        // Ray exited the current frame. Try popping to ancestor.
        if ribbon_level >= uniforms.ribbon_count {
            break;
        }
        if current_kind == ROOT_KIND_FACE {
            let body_pop_level = uniforms.root_face_meta.y;
            if ribbon_level < body_pop_level {
                let entry = ribbon[ribbon_level];
                let s = entry.slot;
                let sx = i32(s % 3u);
                let sy = i32((s / 3u) % 3u);
                let sz = i32(s / 9u);
                let slot_off = vec3<f32>(f32(sx), f32(sy), f32(sz));
                let old_size = cur_face_bounds.w;
                cur_face_bounds = vec4<f32>(
                    cur_face_bounds.x - slot_off.x * old_size,
                    cur_face_bounds.y - slot_off.y * old_size,
                    cur_face_bounds.z - slot_off.z * old_size,
                    old_size * 3.0,
                );
                cur_scale = cur_scale * (1.0 / 3.0);
                current_idx = entry.node_idx;
                ribbon_level = ribbon_level + 1u;
                continue;
            }
            if body_pop_level >= uniforms.ribbon_count {
                break;
            }
            let body_entry = ribbon[body_pop_level];
            current_idx = body_entry.node_idx;
            current_kind = ROOT_KIND_BODY;
            inner_r = node_kinds[current_idx].inner_r;
            outer_r = node_kinds[current_idx].outer_r;
            ribbon_level = body_pop_level + 1u;
        } else {
            // Multi-level ribbon pop: pop up to SHELL_BUDGET entries
            // at once before the next march_cartesian call. This cuts
            // the number of march calls from render_depth+1 to
            // ceil(render_depth/SHELL_BUDGET)+1.
            var pops: u32 = 0u;
            loop {
                if pops >= SHELL_BUDGET { break; }
                if ribbon_level >= uniforms.ribbon_count { break; }

                let entry = ribbon[ribbon_level];
                let s = entry.slot;
                let sx = i32(s % 3u);
                let sy = i32((s / 3u) % 3u);
                let sz = i32(s / 9u);
                let slot_off = vec3<f32>(f32(sx), f32(sy), f32(sz));
                skip_node_idx = current_idx;
                ray_origin = slot_off + ray_origin / 3.0;
                ray_dir = ray_dir / 3.0;
                if uniforms.highlight_active != 0u {
                    cur_hmin = slot_off + cur_hmin / 3.0;
                    cur_hmax = slot_off + cur_hmax / 3.0;
                }
                cur_scale = cur_scale * (1.0 / 3.0);
                current_idx = entry.node_idx;
                ribbon_level = ribbon_level + 1u;
                pops = pops + 1u;

                let k = node_kinds[current_idx].kind;
                if k == 1u {
                    current_kind = ROOT_KIND_BODY;
                    inner_r = node_kinds[current_idx].inner_r;
                    outer_r = node_kinds[current_idx].outer_r;
                    break;
                }
                current_kind = ROOT_KIND_CARTESIAN;
            }
        }
    }

    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.highlight_min = cur_hmin;
    result.highlight_max = cur_hmax;
    result.frame_scale = cur_scale;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;
    return result;
}

// -------------- Vertex / Fragment shaders --------------

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = uniforms.screen_width / uniforms.screen_height;
    let half_fov_tan = tan(camera.fov * 0.5);
    let ndc = vec2<f32>(
        (in.uv.x - 0.5) * 2.0 * aspect * half_fov_tan,
        (0.5 - in.uv.y) * 2.0 * half_fov_tan,
    );
    let ray_dir = camera.forward + camera.right * ndc.x + camera.up * ndc.y;

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
        let h_min = select(uniforms.highlight_min.xyz, result.highlight_min, result.hit);
        let h_max = select(uniforms.highlight_max.xyz, result.highlight_max, result.hit);
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
        let pad = max(h_size.x * 0.02, 0.002);
        let box_min = h_min - vec3<f32>(pad);
        let box_max = h_max + vec3<f32>(pad);
        let h_inv_dir = vec3<f32>(
            select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
            select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
            select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
        );
        let hb = ray_box(camera.pos, h_inv_dir, box_min, box_max);
        if hb.t_enter < hb.t_exit && hb.t_exit > 0.0 {
            let t = max(hb.t_enter, 0.0);
            if t <= result.t + h_size.x * 0.05 {
                let hit_pos = camera.pos + ray_dir * t;
                let from_min = hit_pos - box_min;
                let from_max = box_max - hit_pos;
                let pixel_world = max(t, 0.001) * 2.0 * tan(camera.fov * 0.5) / uniforms.screen_height;
                let ew = max(pixel_world * 2.25, h_size.x * 0.02);
                let near_x = from_min.x < ew || from_max.x < ew;
                let near_y = from_min.y < ew || from_max.y < ew;
                let near_z = from_min.z < ew || from_max.z < ew;
                let edge_count = u32(near_x) + u32(near_y) + u32(near_z);
                if edge_count >= 2u {
                    color = mix(color, vec3<f32>(1.0, 0.92, 0.18), 0.92);
                }
            }
        }
    }

    let pixel = vec2<f32>(in.uv.x * uniforms.screen_width, in.uv.y * uniforms.screen_height);
    let center = vec2<f32>(uniforms.screen_width * 0.5, uniforms.screen_height * 0.5);
    let d = abs(pixel - center);
    let cross_size = 12.0;
    let cross_thickness = 1.5;
    let gap = 3.0;
    let is_crosshair = (d.x < cross_thickness && d.y >= gap && d.y < cross_size)
                    || (d.y < cross_thickness && d.x >= gap && d.x < cross_size);
    if is_crosshair {
        let cross_color = select(
            vec3<f32>(0.95, 0.95, 0.98),
            vec3<f32>(1.0, 0.92, 0.18),
            result.hit,
        );
        color = mix(color, cross_color, 0.95);
    }

    return vec4<f32>(color, 1.0);
}
