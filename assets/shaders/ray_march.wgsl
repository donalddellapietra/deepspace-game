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

// Mirrors Rust `GpuRibbonFrame`: 8 vec4-sized slots = 128 bytes.
struct RibbonFrame {
    root_index: u32,
    sphere_active: u32,
    world_scale: f32,
    face: u32,
    camera_local: vec4<f32>,
    frame_face_node_idx: u32,
    frame_un_size: f32,
    frame_alpha_n_u: f32,
    frame_alpha_n_v: f32,
    camera_un_remainder: f32,
    camera_vn_remainder: f32,
    camera_rn_remainder: f32,
    frame_alpha_r: f32,
    frame_n_u_lo_ref: vec4<f32>,
    frame_n_v_lo_ref: vec4<f32>,
    frame_r_lo_world: f32,
    sphere_inner_r_world: f32,
    sphere_outer_r_world: f32,
    sphere_shell_world: f32,
    face_n_axis: vec4<f32>,
}

const MAX_RIBBON_FRAMES: u32 = 8u;

// Face-subtree walker max-depth cap. Set to MAX_DEPTH-equivalent
// so the walker can resolve any cell the tree could possibly hold
// (the tree's `MAX_DEPTH` is 63; clamp here matches that). Walker
// terminates early on Block/Empty terminals, so per-pixel cost is
// content-bounded rather than 63-bounded — only deep mixed
// subtrees (= user edits) traverse the full depth, and there
// usually only along a few rays.
const WALKER_MAX_DEPTH: u32 = 63u;

// Layout-equivalent to Rust `GpuPlanet`. WGSL would round
// `vec3<u32>` to 16 bytes; using individual u32 padding fields
// keeps the struct exactly 48 bytes to match the Rust definition.
struct Planet {
    enabled: u32,
    body_node_index: u32,
    inner_r_world: f32,
    outer_r_world: f32,
    oc_world: vec4<f32>,
    _reserved: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Uniforms {
    node_count: u32,
    screen_width: f32,
    screen_height: f32,
    max_depth: u32,
    highlight_active: u32,
    ribbon_count: u32,
    _pad0: u32,
    _pad1: u32,
    highlight_min: vec4<f32>,
    highlight_max: vec4<f32>,
    // Camera in the highlight AABB's local frame. AABB + camera
    // share this frame so the ray-box test is layer-local f32,
    // precision bounded by WORLD_SIZE regardless of anchor depth.
    highlight_camera: vec4<f32>,
    planet: Planet,
    ribbon: array<RibbonFrame, 8>,
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

// Walk a face subtree from the body node's face-center child slot,
// using normalized `(un, vn, rn) ∈ [0, 1]³` coords.
//
// Returns the terminal cell's block + depth AND its precise extent
// in normalized face coords `(u_lo, v_lo, r_lo, size)` — accumulated
// incrementally during descent so f32 precision stays bounded by
// ~7 digits regardless of depth. The sphere DDA uses these values
// directly for cell-boundary math, sidestepping the
// `iu = floor(un * pow(3, depth))` precision wall that capped
// rendering at depth ~12 in the previous implementation.
struct WalkResult {
    block_id: u32,
    term_depth: u32,
    u_lo: f32,
    v_lo: f32,
    r_lo: f32,
    cell_size: f32,
}

fn walk_face_subtree(body_node_idx: u32, face: u32,
                     un_in: f32, vn_in: f32, rn_in: f32) -> WalkResult {
    let fs = face_slot(face);
    let face_packed = child_packed(body_node_idx, fs);
    let face_tag = child_tag(face_packed);
    var result: WalkResult;
    if (face_tag == 0u || face_tag == 1u) {
        result.block_id = select(0u, child_block_type(face_packed), face_tag == 1u);
        result.term_depth = 1u;
        result.u_lo = 0.0;
        result.v_lo = 0.0;
        result.r_lo = 0.0;
        result.cell_size = 1.0;
        return result;
    }
    var node = child_node_index(body_node_idx, fs);
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);
    var u_lo: f32 = 0.0;
    var v_lo: f32 = 0.0;
    var r_lo: f32 = 0.0;
    var size: f32 = 1.0;
    for (var d: u32 = 2u; d <= WALKER_MAX_DEPTH; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let next_size = size / 3.0;
        let next_u_lo = u_lo + f32(us) * next_size;
        let next_v_lo = v_lo + f32(vs) * next_size;
        let next_r_lo = r_lo + f32(rs) * next_size;
        let slot = rs * 9u + vs * 3u + us;
        let packed = child_packed(node, slot);
        let tag = child_tag(packed);
        if tag == 0u {
            result.block_id = 0u; result.term_depth = d;
            result.u_lo = next_u_lo; result.v_lo = next_v_lo; result.r_lo = next_r_lo;
            result.cell_size = next_size;
            return result;
        }
        if tag == 1u {
            result.block_id = child_block_type(packed); result.term_depth = d;
            result.u_lo = next_u_lo; result.v_lo = next_v_lo; result.r_lo = next_r_lo;
            result.cell_size = next_size;
            return result;
        }
        node = child_node_index(node, slot);
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
        u_lo = next_u_lo;
        v_lo = next_v_lo;
        r_lo = next_r_lo;
        size = next_size;
    }
    result.block_id = 0u; result.term_depth = 22u;
    result.u_lo = u_lo; result.v_lo = v_lo; result.r_lo = r_lo;
    result.cell_size = size;
    return result;
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

// Stack depth for the Cartesian DDA. Sized to match
// `WALKER_MAX_DEPTH` so a single ribbon frame can descend the full
// tree without spilling. WGSL requires fixed-size arrays so this is
// a compile-time constant; runtime cost is per-pixel state size,
// not per-pixel descent — descent terminates early on terminals.
const MAX_STACK_DEPTH: u32 = 32u;

struct HitResult {
    hit: bool,
    color: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
}

// Sphere DDA running in sphere-relative world coords. `oc` is
// `camera_world - sphere_center_world`, CPU-computed via
// `WorldPos::offset_from` so it's path-anchored and bounded by the
// body cell size — magnitudes stay in [-shell, +shell] regardless
// of where the camera is anchored in the tree.
//
// Returns a HitResult with `t` in world units. Caller composites
// with ribbon-frame marches by comparing world-scale t.
//
// No max-depth cap: the walker returns each terminal cell's exact
// `(u_lo, v_lo, r_lo, size)` via additive accumulation during
// descent, so cell-boundary math is precision-stable at any depth
// (no `cells_d = pow(3, depth)` to overflow f32 integer-exact).
fn sphere_in_cell(
    body_node_idx: u32,
    oc_world: vec3<f32>,
    ray_dir: vec3<f32>,
    inner_r_world: f32,
    outer_r_world: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;

    let cs_outer = outer_r_world;
    let cs_inner = inner_r_world;
    let shell = cs_outer - cs_inner;

    let oc = oc_world;
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
        if t >= t_exit || steps > 16384u { break; }
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

        let walk = walk_face_subtree(body_node_idx, face, un, vn, rn);
        let block_id = walk.block_id;

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
            let ambient = 0.25;
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            result.color = cell_color * (ambient + diffuse * 0.75);
            return result;
        }

        // Cell extent comes from the walker, computed via additive
        // accumulation during descent. Each extent value is bounded
        // in [0, 1] regardless of `term_depth`, so f32 precision in
        // the boundary-plane coefficients stays at ~7 digits even
        // for face-subtree depths in the high teens or twenties.
        // No more `cells_d = pow(3, depth)` precision wall.
        let u_lo_n = walk.u_lo;
        let v_lo_n = walk.v_lo;
        let r_lo_n = walk.r_lo;
        let cell_n = walk.cell_size;

        let u_lo_ea = u_lo_n * 2.0 - 1.0;
        let u_hi_ea = (u_lo_n + cell_n) * 2.0 - 1.0;
        let n_u_lo = u_axis - ea_to_cube(u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(u_hi_ea) * n_axis;

        let v_lo_ea = v_lo_n * 2.0 - 1.0;
        let v_hi_ea = (v_lo_n + cell_n) * 2.0 - 1.0;
        let n_v_lo = v_axis - ea_to_cube(v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(v_hi_ea) * n_axis;

        let r_lo = cs_inner + r_lo_n * shell;
        let r_hi = cs_inner + (r_lo_n + cell_n) * shell;

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
        let cell_eps = max(shell * cell_n * 1e-3, 1e-7);
        t = t_next + cell_eps;
    }

    return result;
}

// Walker that descends a face subtree starting from an ARBITRARY
// node (typically deep inside the face subtree, not necessarily the
// face_root). Same accumulation as `walk_face_subtree`; the caller
// provides starting (un, vn, rn) which are interpreted as the
// FRAME-LOCAL coords (in [0, 1] of the frame's subregion).
fn walk_face_subtree_from(
    start_node_idx: u32,
    un_in: f32, vn_in: f32, rn_in: f32,
) -> WalkResult {
    var node = start_node_idx;
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);
    var u_lo: f32 = 0.0;
    var v_lo: f32 = 0.0;
    var r_lo: f32 = 0.0;
    var size: f32 = 1.0;
    var result: WalkResult;
    for (var d: u32 = 1u; d <= WALKER_MAX_DEPTH; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let next_size = size / 3.0;
        let next_u_lo = u_lo + f32(us) * next_size;
        let next_v_lo = v_lo + f32(vs) * next_size;
        let next_r_lo = r_lo + f32(rs) * next_size;
        let slot = rs * 9u + vs * 3u + us;
        let packed = child_packed(node, slot);
        let tag = child_tag(packed);
        if tag == 0u {
            result.block_id = 0u; result.term_depth = d;
            result.u_lo = next_u_lo; result.v_lo = next_v_lo; result.r_lo = next_r_lo;
            result.cell_size = next_size;
            return result;
        }
        if tag == 1u {
            result.block_id = child_block_type(packed); result.term_depth = d;
            result.u_lo = next_u_lo; result.v_lo = next_v_lo; result.r_lo = next_r_lo;
            result.cell_size = next_size;
            return result;
        }
        node = child_node_index(node, slot);
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
        u_lo = next_u_lo;
        v_lo = next_v_lo;
        r_lo = next_r_lo;
        size = next_size;
    }
    result.block_id = 0u; result.term_depth = 22u;
    result.u_lo = u_lo; result.v_lo = v_lo; result.r_lo = r_lo;
    result.cell_size = size;
    return result;
}

// Per-frame sphere DDA. Operates entirely in the frame's local face
// coords [0, 1] for descent, with reference-plus-delta plane normals
// for ray-cell intersection. Linearizes the cube-sphere mapping at
// the camera; valid for samples within the frame's world extent
// (~ shell / 3^frame_face_subtree_depth). Returns a HitResult with
// `t` in WORLD units so it composites with other ribbon frames'
// hits and the global sphere pass.
fn sphere_in_frame(frame: RibbonFrame, ray_dir: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    if frame.sphere_active == 0u { return result; }

    let oc = uniforms.planet.oc_world.xyz;  // camera relative to body center, world units
    let cs_outer = frame.sphere_outer_r_world;
    let cs_inner = frame.sphere_inner_r_world;
    let shell = frame.sphere_shell_world;

    // Outer sphere intersection (numerical-recipes form for stability).
    let b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    // Camera projection onto the frame's face. We use the FRAME's
    // face axes, not pick_face — the frame has a pre-determined
    // face-subtree branch. If camera is on a different face the
    // dot products are still well-defined; the walker just rejects
    // samples outside the frame's [0, 1] region.
    let n_axis = frame.face_n_axis.xyz;
    let u_axis_w = face_u_axis(frame.face);
    let v_axis_w = face_v_axis(frame.face);

    // Sample's (un, vn, rn) parameterized via linearization at the
    // camera. Validity: errors grow as `t^2 * curvature`. Within a
    // ribbon frame's world extent (~ shell / 3^K for K = frame's
    // face-subtree depth), this stays << one frame-local cell.
    let oc_n = dot(oc, n_axis);
    if abs(oc_n) < 1e-10 { return result; }
    let cube_u_cam = dot(oc, u_axis_w) / oc_n;
    let cube_v_cam = dot(oc, v_axis_w) / oc_n;
    let dcube_u_dt = (dot(ray_dir, u_axis_w) - cube_u_cam * dot(ray_dir, n_axis)) / oc_n;
    let dcube_v_dt = (dot(ray_dir, v_axis_w) - cube_v_cam * dot(ray_dir, n_axis)) / oc_n;
    let inv_one_plus_uu = 1.0 / (1.0 + cube_u_cam * cube_u_cam);
    let inv_one_plus_vv = 1.0 / (1.0 + cube_v_cam * cube_v_cam);
    let dun_dt = 0.5 * (4.0 / PI_F) * inv_one_plus_uu * dcube_u_dt;
    let dvn_dt = 0.5 * (4.0 / PI_F) * inv_one_plus_vv * dcube_v_dt;
    let r_camera = sqrt(dot(oc, oc));
    let drn_dt = (dot(oc, ray_dir) / r_camera) / shell;

    let inv_size = 1.0 / max(frame.frame_un_size, 1e-30);
    let alpha_un_remainder = dun_dt * inv_size;
    let alpha_vn_remainder = dvn_dt * inv_size;
    let alpha_rn_remainder = drn_dt * inv_size;

    let frame_n_u_lo = frame.frame_n_u_lo_ref.xyz;
    let frame_n_v_lo = frame.frame_n_v_lo_ref.xyz;
    let alpha_n_u = frame.frame_alpha_n_u;
    let alpha_n_v = frame.frame_alpha_n_v;
    let alpha_r = frame.frame_alpha_r;

    let eps_init = max(shell * 1e-5, 1e-7);
    var t = t_enter + eps_init;
    var steps = 0u;
    var last_face_id: u32 = 6u;

    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;

        // Sample's frame-local remainders via linearization.
        let dt = t;
        let sample_un = frame.camera_un_remainder + alpha_un_remainder * dt;
        let sample_vn = frame.camera_vn_remainder + alpha_vn_remainder * dt;
        let sample_rn = frame.camera_rn_remainder + alpha_rn_remainder * dt;

        // Validity: sample must lie within the frame's [0, 1]
        // subregion AND the sphere shell.
        if sample_un < 0.0 || sample_un >= 1.0 { break; }
        if sample_vn < 0.0 || sample_vn >= 1.0 { break; }
        if sample_rn < 0.0 || sample_rn >= 1.0 { break; }

        // Walk face subtree from the frame's root, with frame-local
        // (un, vn, rn) in [0, 1].
        let walk = walk_face_subtree_from(
            frame.frame_face_node_idx,
            sample_un, sample_vn, sample_rn,
        );

        if walk.block_id != 0u {
            // Hit. Normal: use last crossed face axis (cube approx).
            var hit_normal: vec3<f32>;
            switch last_face_id {
                case 0u: { hit_normal = -u_axis_w; }
                case 1u: { hit_normal =  u_axis_w; }
                case 2u: { hit_normal = -v_axis_w; }
                case 3u: { hit_normal =  v_axis_w; }
                case 4u: { hit_normal = -n_axis; }
                case 5u: { hit_normal =  n_axis; }
                default: { hit_normal =  n_axis; }
            }
            let cell_color = palette.colors[walk.block_id].rgb;
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let ambient = 0.25;
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            result.color = cell_color * (ambient + diffuse * 0.75);
            return result;
        }

        // Cell boundary planes via reference-plus-delta. The cell's
        // u_lo in the frame's [0, 1] is `walk.u_lo`. World plane
        // normal at that boundary:
        //   n_u_at_walk_u_lo = frame_n_u_lo_ref - walk.u_lo * alpha_n_u * n_axis
        // Both terms have bounded f32 magnitudes; the product
        // `walk.u_lo * alpha_n_u` is small (alpha is ~frame_un_size *
        // sec^2, frame_un_size << 1 for deep frames). Subtraction
        // gives a precision-stable plane normal.
        let n_u_lo = frame_n_u_lo - walk.u_lo * alpha_n_u * n_axis;
        let n_u_hi = frame_n_u_lo - (walk.u_lo + walk.cell_size) * alpha_n_u * n_axis;
        let n_v_lo = frame_n_v_lo - walk.v_lo * alpha_n_v * n_axis;
        let n_v_hi = frame_n_v_lo - (walk.v_lo + walk.cell_size) * alpha_n_v * n_axis;

        let r_lo = frame.frame_r_lo_world + walk.r_lo * alpha_r;
        let r_hi = frame.frame_r_lo_world + (walk.r_lo + walk.cell_size) * alpha_r;

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
        // Advance step: scaled to current cell's WORLD size to
        // prevent overstepping at deep cells. Floor at 1e-7 keeps t
        // moving when the cell-scaled value falls below f32 ULP
        // (deep deep cells; ray will skip but won't infinite-loop).
        let cell_world_size = max(alpha_r * walk.cell_size, 1e-30);
        let cell_eps = max(cell_world_size * 1e-3, 1e-7);
        t = t_next + cell_eps;
    }

    return result;
}

// -------------- Stack-based Cartesian tree DDA --------------

fn march(frame_root_index: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;

    // Sphere bodies are rendered by a separate pass in fs_main using
    // path-anchored `oc_world` for precision. March() skips sphere
    // content entirely — if the frame's root is itself a sphere
    // body or face-subtree node, return no-hit.
    let root_kind_entry = node_kinds[frame_root_index];
    if (root_kind_entry.kind != 0u) {
        return result;
    }

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

    // Sized to MAX_STACK_DEPTH so a Cartesian descent can reach
    // any tree depth (not capped at 16 levels like the prior code).
    var s_node_idx: array<u32, 32>;
    var s_cell: array<vec3<i32>, 32>;
    var s_side_dist: array<vec3<f32>, 32>;
    var s_node_origin: array<vec3<f32>, 32>;
    var s_cell_size: array<f32, 32>;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    s_node_idx[0] = frame_root_index;
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
    // Hard ceiling for the Cartesian DDA's per-pixel iteration
    // count — purely a runaway-loop guard, not a semantic depth
    // cap. Most rays terminate well within this.
    let max_iterations = 16384u;

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
            return result;
        } else {
            // tag == 2u: Node child. Look up its kind.
            let child_idx = child_node_index(s_node_idx[depth], slot);
            let kind = node_kinds[child_idx].kind;

            // Sphere body or face-subtree child: skip. The sphere
            // pass (outside this march) handles all body content via
            // path-anchored coords so the ribbon's Cartesian march
            // doesn't paint cubic cells over the sphere's bulged
            // voxels. Advance DDA past this cell and continue.
            if kind != 0u {
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
            let at_max = depth + 1u >= uniforms.max_depth || depth + 1u >= MAX_STACK_DEPTH;
            let cell_world_size = s_cell_size[depth];
            let min_side = min(s_side_dist[depth].x, min(s_side_dist[depth].y, s_side_dist[depth].z));
            let ray_dist = max(min_side, 0.001);
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
                    return result;
                }
            } else {
                let parent_origin = s_node_origin[depth];
                let parent_cell_size = s_cell_size[depth];
                let child_origin = parent_origin + vec3<f32>(cell) * parent_cell_size;
                let child_cell_size = parent_cell_size / 3.0;

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
    let ray_dir = normalize(camera.forward + camera.right * ndc.x + camera.up * ndc.y);

    // Ribbon compositor: march each frame in its local coords and
    // keep the hit with smallest world-scale t. Frames come deepest-
    // first from the CPU. Ribbon marches skip sphere content — that
    // comes from the dedicated sphere pass below.
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    var best_t_world: f32 = 1e20;

    // Track whether ANY ribbon frame's per-frame sphere DDA produced
    // a hit. If so, the global sphere pass below is skipped — the
    // per-frame DDA's hit is at higher precision (camera-anchored
    // linearization within the frame's bounded world extent), and
    // letting the global pass also render would z-fight against it.
    var sphere_hit_from_frame = false;

    for (var i: u32 = 0u; i < uniforms.ribbon_count; i = i + 1u) {
        let frame = uniforms.ribbon[i];
        if (frame.sphere_active != 0u) {
            // Per-frame sphere DDA in frame-local coords. CPU only
            // sets sphere_active=1 for frames deep enough that
            // camera-anchored linearization stays accurate
            // (face_subtree_depth ≥ MIN_FRAME_FACE_DEPTH on the
            // CPU side); for shallow frames, the global sphere
            // pass below renders.
            let r = sphere_in_frame(frame, ray_dir);
            if (r.hit && r.t < best_t_world) {
                result = r;
                best_t_world = r.t;
                sphere_hit_from_frame = true;
            }
        } else {
            let r = march(frame.root_index, frame.camera_local.xyz, ray_dir);
            if (r.hit) {
                let t_world = r.t * frame.world_scale;
                if (t_world < best_t_world) {
                    result = r;
                    best_t_world = t_world;
                }
            }
        }
    }

    // Sphere pass: single global ray-sphere + face-subtree walk in
    // path-anchored sphere-relative coords. Precision stays bounded
    // at any camera anchor depth because `oc_world` is CPU-computed
    // via path arithmetic (not by subtracting two large world-scale
    // f32 vectors).
    // Skip the global sphere pass when ANY per-frame sphere DDA
    // already produced a hit. The per-frame hit is at higher
    // precision; running the global pass too would z-fight with
    // it (both rendering nearly the same cell at slightly
    // different positions). For frames where per-frame DOESN'T
    // apply (shallow body-interior frames, or no body-interior
    // frames at all e.g. camera flying above the planet), the
    // global pass handles sphere rendering on its own.
    if (uniforms.planet.enabled != 0u && !sphere_hit_from_frame) {
        let sphere_r = sphere_in_cell(
            uniforms.planet.body_node_index,
            uniforms.planet.oc_world.xyz,
            ray_dir,
            uniforms.planet.inner_r_world,
            uniforms.planet.outer_r_world,
        );
        if (sphere_r.hit && sphere_r.t < best_t_world) {
            result = sphere_r;
            best_t_world = sphere_r.t;
        }
    }

    // Convert best hit's t to world units for the highlight depth
    // test below (which uses camera.pos world coords).
    result.t = best_t_world;

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

    if uniforms.highlight_active != 0u {
        let h_min = uniforms.highlight_min.xyz;
        let h_max = uniforms.highlight_max.xyz;
        let h_size = h_max - h_min;
        let h_inv_dir = vec3<f32>(
            select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
            select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
            select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
        );
        let h_cam = uniforms.highlight_camera.xyz;
        let hb = ray_box(h_cam, h_inv_dir, h_min, h_max);
        if hb.t_enter < hb.t_exit && hb.t_exit > 0.0 {
            let t = max(hb.t_enter, 0.0);
            if t <= result.t + h_size.x * 0.01 {
                let hit_pos = h_cam + ray_dir * t;
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

    let pixel = vec2<f32>(in.uv.x * uniforms.screen_width, in.uv.y * uniforms.screen_height);
    let center = vec2<f32>(uniforms.screen_width * 0.5, uniforms.screen_height * 0.5);
    let d = abs(pixel - center);
    let cross_size = 12.0;
    let cross_thickness = 1.5;
    let gap = 3.0;
    let is_crosshair = (d.x < cross_thickness && d.y >= gap && d.y < cross_size)
                    || (d.y < cross_thickness && d.x >= gap && d.x < cross_size);
    if is_crosshair {
        color = vec3<f32>(1.0) - color;
    }

    return vec4<f32>(color, 1.0);
}
