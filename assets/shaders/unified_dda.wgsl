#include "bindings.wgsl"
#include "tree.wgsl"
#include "ray_prim.wgsl"

// The ONE cell-traversal DDA primitive.
//
// Walks a single render frame rooted at `node_idx`, whose cells span
// `[0, 3)³` in `ray_origin/ray_dir` coords. Handles descent, DDA
// advance, neighbor-step, bubble-up, and per-cell NodeKind dispatch:
//
//   * Cartesian  — slot-pick DDA (`floor(residual)` picks the child
//                  slot, DDA advances to the next cell boundary).
//   * CubedSphereBody — ray-sphere-outer intersect at body entry
//                       selects the hit face, rotates `ray_dir` by
//                       the face basis, and descends into the face
//                       subtree with face-normalized residual.
//   * CubedSphereFace — face subtree root: descendant cells are
//                       Cartesian-kind with UVR axis semantics; the
//                       Cartesian arm's slot-pick handles them
//                       unchanged. Shell-exit (`r` axis OOB at face
//                       depth 0) routes inner→core subtree or
//                       outer→body exit; UV-axis exits terminate as
//                       miss for Stage 2 (Stage 3 adds seam
//                       rotation).
//
// Precision rules (per docs/principles/no-absolute-coordinates.md):
//   * `ray_origin` / `ray_dir` arrive in FRAME-LOCAL `[0, 3)³`.
//     Ribbon-pop handles the per-level ÷3 rescale before calling.
//   * The descent stack is slot-path-shaped (integer `s_cell`
//     packed cell coords) — no absolute world coordinates.
//     `cur_node_origin` is a reversible incremental offset within
//     the current frame, not an absolute world anchor.
//   * Face-subtree walking happens in a rotated sub-frame via
//     `march_face_subtree`: the face's orthonormal basis rotates
//     `ray_dir` into face-local `(u, v, r)` coords and the
//     face-normalized entry point becomes `[0, 3)³`. All per-cell
//     arithmetic inside the face stays O(1) magnitude.
//   * `MAX_STACK_DEPTH` (see bindings.wgsl) bounds descent; LOD
//     (Nyquist pixel) prunes below tree depth.
//
// Arguments:
//   * `node_idx`       BFS index of the frame-root node.
//   * `ray_origin/dir` frame-local ray.
//   * `skip_slot`      slot (0..27) of the child at the CURRENT
//                      frame's root that a ribbon-pop just came out
//                      of; skipped at depth 0 so the DDA doesn't
//                      re-traverse the inner-shell subtree. Pass
//                      `0xFFFFFFFFu` on first-call / no-skip.
//   * `max_depth_cap`  LOD-independent hard cap on frame descent.

// ───────────────────────────── Face basis helpers (port of Rust Face)

struct FaceBasis {
    u_axis: vec3<f32>,
    v_axis: vec3<f32>,
    n_axis: vec3<f32>,
}

fn face_basis(face: u32) -> FaceBasis {
    // Mirrors src/world/cubesphere/mod.rs::Face::tangents + normal.
    // Rows of R_face = (u_axis, v_axis, n_axis), i.e.
    //   face_vec = R_face · body_vec  when body_vec has coords
    //   (bx, by, bz) in body-XYZ, R_face has rows u/v/n.
    var b: FaceBasis;
    switch face {
        case 0u: { // PosX
            b.u_axis = vec3<f32>( 0.0, 0.0, -1.0);
            b.v_axis = vec3<f32>( 0.0, 1.0,  0.0);
            b.n_axis = vec3<f32>( 1.0, 0.0,  0.0);
        }
        case 1u: { // NegX
            b.u_axis = vec3<f32>( 0.0, 0.0,  1.0);
            b.v_axis = vec3<f32>( 0.0, 1.0,  0.0);
            b.n_axis = vec3<f32>(-1.0, 0.0,  0.0);
        }
        case 2u: { // PosY
            b.u_axis = vec3<f32>( 1.0, 0.0,  0.0);
            b.v_axis = vec3<f32>( 0.0, 0.0, -1.0);
            b.n_axis = vec3<f32>( 0.0, 1.0,  0.0);
        }
        case 3u: { // NegY
            b.u_axis = vec3<f32>( 1.0, 0.0,  0.0);
            b.v_axis = vec3<f32>( 0.0, 0.0,  1.0);
            b.n_axis = vec3<f32>( 0.0,-1.0,  0.0);
        }
        case 4u: { // PosZ
            b.u_axis = vec3<f32>( 1.0, 0.0,  0.0);
            b.v_axis = vec3<f32>( 0.0, 1.0,  0.0);
            b.n_axis = vec3<f32>( 0.0, 0.0,  1.0);
        }
        case 5u, default: { // NegZ (default is unreachable with valid face)
            b.u_axis = vec3<f32>(-1.0, 0.0,  0.0);
            b.v_axis = vec3<f32>( 0.0, 1.0,  0.0);
            b.n_axis = vec3<f32>( 0.0, 0.0, -1.0);
        }
    }
    return b;
}

// body_point_to_face_space port. `point_body` is in body-local
// `[0, body_size)³` with body_size=3; inner_body / outer_body are
// the shell radii scaled to body_size units (inner_r * 3, outer_r *
// 3 for the CPU's cell-local `[0, 1)` convention).
struct FacePoint {
    face: u32,
    un: f32,
    vn: f32,
    rn: f32,
    r: f32, // radial distance from body center (body-local units)
}

fn body_point_to_face_space(
    point_body: vec3<f32>, inner_body: f32, outer_body: f32,
) -> FacePoint {
    var fp: FacePoint;
    let d = point_body - vec3<f32>(1.5); // body_size=3 → center at 1.5
    let r2 = dot(d, d);
    // Degenerate center: all points become PosX with default coords.
    // Callers that hit this case get a sensible fallback.
    if r2 < 1e-12 {
        fp.face = 0u;
        fp.un = 0.5;
        fp.vn = 0.5;
        fp.rn = 0.0;
        fp.r = 0.0;
        return fp;
    }
    let r = sqrt(r2);
    let n = d / r;
    let ax = abs(n.x);
    let ay = abs(n.y);
    let az = abs(n.z);
    var face: u32 = 0u;
    var cube_u: f32 = 0.0;
    var cube_v: f32 = 0.0;
    if ax >= ay && ax >= az {
        if n.x > 0.0 {
            face = 0u; // PosX
            cube_u = -n.z / ax;
            cube_v =  n.y / ax;
        } else {
            face = 1u; // NegX
            cube_u =  n.z / ax;
            cube_v =  n.y / ax;
        }
    } else if ay >= az {
        if n.y > 0.0 {
            face = 2u; // PosY
            cube_u =  n.x / ay;
            cube_v = -n.z / ay;
        } else {
            face = 3u; // NegY
            cube_u =  n.x / ay;
            cube_v =  n.z / ay;
        }
    } else {
        if n.z > 0.0 {
            face = 4u; // PosZ
            cube_u =  n.x / az;
            cube_v =  n.y / az;
        } else {
            face = 5u; // NegZ
            cube_u = -n.x / az;
            cube_v =  n.y / az;
        }
    }
    let u_ea = atan(cube_u) * (4.0 / 3.14159265);
    let v_ea = atan(cube_v) * (4.0 / 3.14159265);
    fp.face = face;
    fp.un = clamp(0.5 * (u_ea + 1.0), 0.0, 1.0);
    fp.vn = clamp(0.5 * (v_ea + 1.0), 0.0, 1.0);
    fp.rn = clamp((r - inner_body) / (outer_body - inner_body), 0.0, 1.0);
    fp.r = r;
    return fp;
}

// Port of `face_space_to_body_point` in `src/world/cubesphere`.
// Given face-normalized `(un, vn, rn) ∈ [0,1]³` and shell radii in
// body-size units (body_size = 3), return the body-local point.
// Used when reconstructing the exit body-space position at a face
// subtree's seam boundary so the neighbor face's entry point can be
// computed geometrically.
fn face_space_to_body_point(
    face: u32, un: f32, vn: f32, rn: f32,
    inner_body: f32, outer_body: f32,
) -> vec3<f32> {
    let u_ea = 2.0 * un - 1.0;
    let v_ea = 2.0 * vn - 1.0;
    // ea_to_cube(x) = tan(x · π/4).
    let cu = tan(u_ea * 0.78539816);
    let cv = tan(v_ea * 0.78539816);
    let basis = face_basis(face);
    // body direction = normalize(n + cu * u + cv * v). The basis
    // vectors are unit and mutually orthogonal, so the un-normalized
    // length is sqrt(1 + cu² + cv²).
    let px = basis.n_axis.x + cu * basis.u_axis.x + cv * basis.v_axis.x;
    let py = basis.n_axis.y + cu * basis.u_axis.y + cv * basis.v_axis.y;
    let pz = basis.n_axis.z + cu * basis.u_axis.z + cv * basis.v_axis.z;
    let plen = sqrt(px * px + py * py + pz * pz);
    let inv = 1.0 / max(plen, 1e-12);
    let dir = vec3<f32>(px * inv, py * inv, pz * inv);
    let r = inner_body + (outer_body - inner_body) * rn;
    return vec3<f32>(1.5) + dir * r;
}

// Project a body-local point onto a SPECIFIED face's normalized
// `(un, vn, rn)` coords. Unlike `body_point_to_face_space` — which
// picks the face by dominant axis — this helper is told which face
// to use and is safe for seam-crossing handoff: points on a shared
// cube edge (two faces of equal dominant axis) are projected onto
// the caller's explicit choice.
//
// Geometry: normalize the direction from the body center, then
// project onto the chosen face's plane (component along the face's
// normal axis). The in-plane components along the face's `(u, v)`
// tangents give the cube coords; `cube_to_ea` warps them to the
// equal-angle face coord, scaled into `[0, 1]`. `rn` is the radial
// position on the shell and is invariant across seams.
fn body_point_to_face_forced(
    point_body: vec3<f32>, inner_body: f32, outer_body: f32, face: u32,
) -> FacePoint {
    var fp: FacePoint;
    let d = point_body - vec3<f32>(1.5);
    let r2 = dot(d, d);
    if r2 < 1e-12 {
        fp.face = face;
        fp.un = 0.5;
        fp.vn = 0.5;
        fp.rn = 0.0;
        fp.r = 0.0;
        return fp;
    }
    let r = sqrt(r2);
    let n = d / r;
    let basis = face_basis(face);
    // `basis.n_axis` is the outward face normal; project n onto it.
    // `n_comp` is |cos(angle)| — for a point near the face, this is
    // close to 1. On the shared cube edge it's ~0.707.
    let n_comp = dot(basis.n_axis, n);
    // Cube coords: component along face's u/v, divided by n_comp to
    // undo the oblique-projection foreshortening.
    let inv_n = 1.0 / max(abs(n_comp), 1e-6);
    let cube_u = dot(basis.u_axis, n) * inv_n * sign_or_one(n_comp);
    let cube_v = dot(basis.v_axis, n) * inv_n * sign_or_one(n_comp);
    let u_ea = atan(cube_u) * (4.0 / 3.14159265);
    let v_ea = atan(cube_v) * (4.0 / 3.14159265);
    fp.face = face;
    fp.un = clamp(0.5 * (u_ea + 1.0), 0.0, 1.0);
    fp.vn = clamp(0.5 * (v_ea + 1.0), 0.0, 1.0);
    fp.rn = clamp((r - inner_body) / (outer_body - inner_body), 0.0, 1.0);
    fp.r = r;
    return fp;
}

fn sign_or_one(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// ────────────────────────────────── Face subtree walker
//
// Cartesian DDA over a face subtree's descendants. Called from
// unified_dda's CubedSphereBody arm after the ray has been
// transformed into face-local coords:
//   * `face_origin` is the face-normalized entry scaled to `[0, 3)³`
//     (i.e. `(un, vn, rn) * 3`).
//   * `face_dir` is the body-local ray direction rotated by the
//     face's orthonormal basis.
//
// Semantics match `march_entity_subtree`: returns a HitResult with
// `hit=true` on terminal cell (block, LOD splat) and `hit=false` on
// exit. The caller classifies the exit (UV-miss = Stage 2 terminate;
// inner-shell → core subtree; outer-shell → body exit) by inspecting
// the face-local position the walker was at when it bubbled out of
// its root. Since WGSL can't return multiple values cleanly, we pack
// the exit axis/direction into unused HitResult fields:
//   * `cell_min.xyz` = final face-local position (exit point).
//   * `cell_size`    = axis code of OOB axis: 0=r-, 1=r+, 2=u-,
//                      3=u+, 4=v-, 5=v+, -1 if exhausted.
//   * `normal`       = last-computed normal (face-local).
//
// On hit, `t` is the face-local t from `face_origin`. Caller adds the
// body-entry `t_enter` (already in ribbon units) to produce the
// reported ribbon-frame t; this is first-order correct at the face
// center and slightly off-scale toward the face edges (the
// linearization error the spec documents as O(cell_size²)).
fn march_face_subtree(
    root_node_idx: u32,
    face_origin: vec3<f32>,
    face_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = face_origin;
    result.cell_size = -1.0;

    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / face_dir.x, abs(face_dir.x) > 1e-8),
        select(1e10, 1.0 / face_dir.y, abs(face_dir.y) > 1e-8),
        select(1e10, 1.0 / face_dir.z, abs(face_dir.z) > 1e-8),
    );
    let ray_metric = max(length(face_dir), 1e-6);
    let step = vec3<i32>(
        select(-1, 1, face_dir.x >= 0.0),
        select(-1, 1, face_dir.y >= 0.0),
        select(-1, 1, face_dir.z >= 0.0),
    );
    let delta_dist = abs(inv_dir);

    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_cell: array<u32, MAX_STACK_DEPTH>;
    var cur_cell_size: f32 = 1.0;
    var cur_node_origin: vec3<f32> = vec3<f32>(0.0);
    var cur_side_dist: vec3<f32>;
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;
    s_node_idx[0] = root_node_idx;

    let root_header_off = node_offsets[root_node_idx];
    var cur_occupancy: u32 = tree[root_header_off];
    var cur_first_child: u32 = tree[root_header_off + 1u];

    // Entry pos = face_origin — caller already placed it inside
    // `[0, 3)³` (at the outer-shell entry the walker starts at
    // `rn = 1.0 → face_origin.z = 3.0`, so nudge inward by 1e-4 to
    // land inside the top cell cleanly).
    let eps = 1e-4;
    let entry_pos = vec3<f32>(
        clamp(face_origin.x, eps, 3.0 - eps),
        clamp(face_origin.y, eps, 3.0 - eps),
        clamp(face_origin.z, eps, 3.0 - eps),
    );
    let entry_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );
    s_cell[0] = pack_cell(entry_cell);
    let cell_f = vec3<f32>(entry_cell);
    cur_side_dist = vec3<f32>(
        select((cell_f.x - entry_pos.x) * inv_dir.x,
               (cell_f.x + 1.0 - entry_pos.x) * inv_dir.x, face_dir.x >= 0.0),
        select((cell_f.y - entry_pos.y) * inv_dir.y,
               (cell_f.y + 1.0 - entry_pos.y) * inv_dir.y, face_dir.y >= 0.0),
        select((cell_f.z - entry_pos.z) * inv_dir.z,
               (cell_f.z + 1.0 - entry_pos.z) * inv_dir.z, face_dir.z >= 0.0),
    );

    var iterations = 0u;
    let max_iterations = 1024u;
    loop {
        if iterations >= max_iterations { break; }
        iterations += 1u;

        let cell = unpack_cell(s_cell[depth]);

        // Bubble-up: cell OOB in the current node's `[0, 3)³`.
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u {
                // Root bubble-out: classify which axis exited and
                // compute the exit point at the face-local boundary
                // plane the ray crossed. We intersect the ray with
                // the plane defined by the OOB axis (x=0/3, y=0/3,
                // or z=0/3) — this is precise regardless of how many
                // axis steps the DDA took to get here. Face-local
                // axes (u, v, r) map to (x, y, z) in face_dir's frame.
                var axis_code: f32 = -1.0;
                var t_exit: f32 = 0.0;
                if cell.z < 0 {
                    axis_code = 0.0;      // r- (inner shell)
                    t_exit = (0.0 - entry_pos.z) * inv_dir.z;
                } else if cell.z > 2 {
                    axis_code = 1.0;      // r+ (outer shell)
                    t_exit = (3.0 - entry_pos.z) * inv_dir.z;
                } else if cell.x < 0 {
                    axis_code = 2.0;      // u-
                    t_exit = (0.0 - entry_pos.x) * inv_dir.x;
                } else if cell.x > 2 {
                    axis_code = 3.0;      // u+
                    t_exit = (3.0 - entry_pos.x) * inv_dir.x;
                } else if cell.y < 0 {
                    axis_code = 4.0;      // v-
                    t_exit = (0.0 - entry_pos.y) * inv_dir.y;
                } else if cell.y > 2 {
                    axis_code = 5.0;      // v+
                    t_exit = (3.0 - entry_pos.y) * inv_dir.y;
                }
                let exit_pos = entry_pos + face_dir * max(t_exit, 0.0);
                result.hit = false;
                result.cell_min = exit_pos;
                result.cell_size = axis_code;
                return result;
            }
            depth -= 1u;
            cur_cell_size = cur_cell_size * 3.0;
            let parent_cell = unpack_cell(s_cell[depth]);
            cur_node_origin = cur_node_origin - vec3<f32>(parent_cell) * cur_cell_size;
            let lc_pop = vec3<f32>(parent_cell);
            cur_side_dist = vec3<f32>(
                select((cur_node_origin.x + lc_pop.x * cur_cell_size - entry_pos.x) * inv_dir.x,
                       (cur_node_origin.x + (lc_pop.x + 1.0) * cur_cell_size - entry_pos.x) * inv_dir.x, face_dir.x >= 0.0),
                select((cur_node_origin.y + lc_pop.y * cur_cell_size - entry_pos.y) * inv_dir.y,
                       (cur_node_origin.y + (lc_pop.y + 1.0) * cur_cell_size - entry_pos.y) * inv_dir.y, face_dir.y >= 0.0),
                select((cur_node_origin.z + lc_pop.z * cur_cell_size - entry_pos.z) * inv_dir.z,
                       (cur_node_origin.z + (lc_pop.z + 1.0) * cur_cell_size - entry_pos.z) * inv_dir.z, face_dir.z >= 0.0),
            );
            let parent_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];
            let m_oob = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(parent_cell + vec3<i32>(m_oob) * step);
            cur_side_dist += m_oob * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_oob;
            continue;
        }

        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        let slot_bit = 1u << slot;
        if ((cur_occupancy & slot_bit) == 0u) {
            let m_empty = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_empty) * step);
            cur_side_dist += m_empty * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_empty;
            continue;
        }
        let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;

        if tag == 1u {
            // Block hit. Report cell box in face-local coords; caller
            // translates back to ribbon-frame for the final HitResult.
            let cell_min_h = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
            let cell_box_h = ray_box(
                face_origin, inv_dir,
                cell_min_h, cell_min_h + vec3<f32>(cur_cell_size),
            );
            result.hit = true;
            result.t = max(cell_box_h.t_enter, 0.0);
            result.color = palette[(packed >> 8u) & 0xFFFFu].rgb;
            result.normal = normal;
            result.cell_min = cell_min_h;
            result.cell_size = cur_cell_size;
            return result;
        }
        if tag != 2u {
            // Stage 2: entities inside face subtrees aren't supported.
            let m_skip = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_skip) * step);
            cur_side_dist += m_skip * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_skip;
            continue;
        }
        let child_idx = tree[child_base + 1u];
        let child_bt = child_block_type(packed);
        if child_bt == 0xFFFEu {
            let m_rep = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_rep) * step);
            cur_side_dist += m_rep * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_rep;
            continue;
        }

        let at_max = depth + 1u >= MAX_STACK_DEPTH;
        let child_cell_size = cur_cell_size / 3.0;
        let min_side = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
        let ray_dist = max(min_side * ray_metric, 0.001);
        let lod_pixels = child_cell_size / ray_dist
            * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
        let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;
        if at_max || at_lod {
            let bt = child_bt;
            if bt == 0xFFFEu || bt == 0xFFFDu {
                let m_lodt = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_lodt) * step);
                cur_side_dist += m_lodt * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_lodt;
            } else {
                let cell_min_l = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
                let cell_box_l = ray_box(
                    face_origin, inv_dir,
                    cell_min_l, cell_min_l + vec3<f32>(cur_cell_size),
                );
                result.hit = true;
                result.t = max(cell_box_l.t_enter, 0.0);
                result.color = palette[bt].rgb;
                result.normal = normal;
                result.cell_min = cell_min_l;
                result.cell_size = cur_cell_size;
                return result;
            }
        } else {
            let child_origin = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
            let node_hit = ray_box(
                face_origin, inv_dir,
                child_origin,
                child_origin + vec3<f32>(3.0) * child_cell_size,
            );
            let ct_start = max(node_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
            let child_entry = face_origin + face_dir * ct_start;
            let local_entry = (child_entry - child_origin) / child_cell_size;
            depth += 1u;
            s_node_idx[depth] = child_idx;
            cur_node_origin = child_origin;
            cur_cell_size = child_cell_size;
            let child_header_off = node_offsets[child_idx];
            cur_occupancy = tree[child_header_off];
            cur_first_child = tree[child_header_off + 1u];
            let child_cell_i = vec3<i32>(
                clamp(i32(floor(local_entry.x)), 0, 2),
                clamp(i32(floor(local_entry.y)), 0, 2),
                clamp(i32(floor(local_entry.z)), 0, 2),
            );
            s_cell[depth] = pack_cell(child_cell_i);
            let lc = vec3<f32>(child_cell_i);
            cur_side_dist = vec3<f32>(
                select((child_origin.x + lc.x * child_cell_size - entry_pos.x) * inv_dir.x,
                       (child_origin.x + (lc.x + 1.0) * child_cell_size - entry_pos.x) * inv_dir.x, face_dir.x >= 0.0),
                select((child_origin.y + lc.y * child_cell_size - entry_pos.y) * inv_dir.y,
                       (child_origin.y + (lc.y + 1.0) * child_cell_size - entry_pos.y) * inv_dir.y, face_dir.y >= 0.0),
                select((child_origin.z + lc.z * child_cell_size - entry_pos.z) * inv_dir.z,
                       (child_origin.z + (lc.z + 1.0) * child_cell_size - entry_pos.z) * inv_dir.z, face_dir.z >= 0.0),
            );
        }
    }
    return result;
}

// ────────────────────────────────── Unified DDA — top-level

fn unified_dda(
    node_idx: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    skip_slot: u32,
    max_depth_cap: u32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );
    let ray_metric = max(length(ray_dir), 1e-6);
    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );
    let delta_dist = abs(inv_dir);

    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_cell: array<u32, MAX_STACK_DEPTH>;

    var cur_cell_size: f32 = 1.0;
    var cur_node_origin: vec3<f32> = vec3<f32>(0.0);
    var cur_side_dist: vec3<f32>;
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    s_node_idx[0] = node_idx;

    let root_header_off = node_offsets[node_idx];
    var cur_occupancy: u32 = tree[root_header_off];
    var cur_first_child: u32 = tree[root_header_off + 1u];
    if ENABLE_STATS {
        ray_loads_offsets = ray_loads_offsets + 1u;
        ray_loads_tree = ray_loads_tree + 2u;
    }

    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }
    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = ray_origin + ray_dir * t_start;

    let root_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );
    s_cell[0] = pack_cell(root_cell);
    let cell_f = vec3<f32>(root_cell);
    cur_side_dist = vec3<f32>(
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
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let cell = unpack_cell(s_cell[depth]);

        // ============================================================
        // OOB / bubble-up.
        // ============================================================
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth -= 1u;
            cur_cell_size = cur_cell_size * 3.0;
            let parent_cell = unpack_cell(s_cell[depth]);
            cur_node_origin = cur_node_origin - vec3<f32>(parent_cell) * cur_cell_size;
            let lc_pop = vec3<f32>(parent_cell);
            cur_side_dist = vec3<f32>(
                select((cur_node_origin.x + lc_pop.x * cur_cell_size - entry_pos.x) * inv_dir.x,
                       (cur_node_origin.x + (lc_pop.x + 1.0) * cur_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                select((cur_node_origin.y + lc_pop.y * cur_cell_size - entry_pos.y) * inv_dir.y,
                       (cur_node_origin.y + (lc_pop.y + 1.0) * cur_cell_size - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
                select((cur_node_origin.z + lc_pop.z * cur_cell_size - entry_pos.z) * inv_dir.z,
                       (cur_node_origin.z + (lc_pop.z + 1.0) * cur_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
            );
            if ENABLE_STATS { ray_steps_oob = ray_steps_oob + 1u; }

            let parent_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];
            if ENABLE_STATS {
                ray_loads_offsets = ray_loads_offsets + 1u;
                ray_loads_tree = ray_loads_tree + 2u;
            }

            let m_oob = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(parent_cell + vec3<i32>(m_oob) * step);
            cur_side_dist += m_oob * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_oob;
            continue;
        }

        // ============================================================
        // PER-CELL NODEKIND DISPATCH — one read per iteration,
        // cached in `kind` below. Cartesian is by far the dominant
        // path (every fractal world + Cartesian substrate of sphere
        // worlds) so we test the sphere variants first and fall
        // through to Cartesian.
        // ============================================================
        let kind = node_kinds[s_node_idx[depth]].kind;

        if kind == NODE_KIND_CUBED_SPHERE_BODY {
            // ---- CubedSphereBody arm --------------------------------
            //
            // We've descended INTO the body node. Its 27 children are
            // 6 face-roots at FACE_SLOTS + 1 core at CORE_SLOT + 20
            // empties; we bypass the Cartesian slot-pick entirely and
            // do a ray-sphere-outer intersect to pick the ONE face
            // the ray enters through. All math stays in ribbon-frame
            // coords — the body cell occupies
            //   origin `cur_node_origin`, size `3 * cur_cell_size`
            // so the body center is `cur_node_origin + 1.5 *
            // cur_cell_size` and the outer shell radius is
            // `outer_r * 3 * cur_cell_size`. radii are stored in the
            // cell-local `[0, 1)` convention per the worldgen spec.
            //
            // Stage 3: when the face walker exits at a UV boundary we
            // look up the adjacent face in `seam_table`, rotate the
            // body-space ray into the neighbor's frame, and restart
            // the walker on the neighbor face subtree. This continues
            // up to `MAX_SEAM_TRANSITIONS` times to bound pathological
            // grazing rays.
            let kind_data = node_kinds[s_node_idx[depth]];
            let body_center = cur_node_origin + vec3<f32>(1.5 * cur_cell_size);
            let outer_rib = kind_data.outer_r * 3.0 * cur_cell_size;
            let inner_rib = kind_data.inner_r * 3.0 * cur_cell_size;
            let inner_body = inner_rib / cur_cell_size; // in body-size=3 units
            let outer_body = outer_rib / cur_cell_size;

            // Stable Numerical-Recipes ray-sphere (camera can be
            // inside or outside the shell).
            let oc = ray_origin - body_center;
            let b = dot(oc, ray_dir);
            let c_outer = dot(oc, oc) - outer_rib * outer_rib;
            let disc_outer = b * b - c_outer;
            if disc_outer < 0.0 {
                // Ray misses outer shell entirely. Treat body cell as
                // empty: advance through it as if the cell held
                // nothing. The ray-box cell-exit handles that.
                let m_miss = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_miss) * step);
                cur_side_dist += m_miss * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_miss;
                continue;
            }
            let sq_outer = sqrt(disc_outer);
            let t0 = -b - sq_outer;
            let t1 = -b + sq_outer;
            var t_enter_outer = t0;
            if t_enter_outer < 0.0 { t_enter_outer = t1; }
            if t_enter_outer < 0.0 {
                let m_miss = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_miss) * step);
                cur_side_dist += m_miss * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_miss;
                continue;
            }
            var t_enter = t_enter_outer;
            if t0 < 0.0 {
                t_enter = 0.0;
            }

            let entry_rib = ray_origin + ray_dir * t_enter;
            let entry_body = (entry_rib - cur_node_origin) / cur_cell_size; // in [0, 3)

            // Entry face picked by dominant axis.
            var fp = body_point_to_face_space(entry_body, inner_body, outer_body);
            var cur_face: u32 = fp.face;
            var cur_un: f32 = fp.un;
            var cur_vn: f32 = fp.vn;
            var cur_rn: f32 = fp.rn;

            // Loop across seam crossings. Each iteration calls
            // `march_face_subtree` on the current face; on UV-exit we
            // hop to the neighbor face via `seam_table` and continue.
            var seam_hit = false;
            var seam_result: HitResult;
            seam_result.hit = false;
            // Track whether we exited the body cell entirely (r+ or
            // shell miss), so the outer DDA can advance past it.
            var body_cell_exit = false;
            // Whether we should descend into the core subtree after
            // this iteration (inner shell).
            var do_core_dispatch = false;

            var seam_count: u32 = 0u;
            loop {
                if seam_count >= MAX_SEAM_TRANSITIONS {
                    // Pathological grazing ray: bail out as if the
                    // ray passed through without hitting anything.
                    body_cell_exit = true;
                    break;
                }
                seam_count += 1u;

                let face_slot = FACE_SLOTS[cur_face];
                let face_bit = 1u << face_slot;
                if (cur_occupancy & face_bit) == 0u {
                    // No face subtree present (degenerate body).
                    body_cell_exit = true;
                    break;
                }
                let face_rank = countOneBits(cur_occupancy & (face_bit - 1u));
                let face_child_base = cur_first_child + face_rank * 2u;
                let face_packed = tree[face_child_base];
                let face_tag = face_packed & 0xFFu;
                if face_tag != 2u {
                    body_cell_exit = true;
                    break;
                }
                let face_root_idx = tree[face_child_base + 1u];

                // Rotate ray_dir into current face's local basis.
                let basis_cur = face_basis(cur_face);
                let face_dir = vec3<f32>(
                    dot(basis_cur.u_axis, ray_dir),
                    dot(basis_cur.v_axis, ray_dir),
                    dot(basis_cur.n_axis, ray_dir),
                );
                let face_origin_local = vec3<f32>(cur_un * 3.0, cur_vn * 3.0, cur_rn * 3.0);

                let sub = march_face_subtree(face_root_idx, face_origin_local, face_dir);
                if sub.hit {
                    // Hit inside this face subtree.
                    seam_result.hit = true;
                    seam_result.t = t_enter + sub.t * cur_cell_size;
                    seam_result.color = sub.color;
                    let face_n = sub.normal;
                    let body_n = basis_cur.u_axis * face_n.x
                               + basis_cur.v_axis * face_n.y
                               + basis_cur.n_axis * face_n.z;
                    seam_result.normal = body_n;
                    seam_result.cell_min = entry_rib;
                    seam_result.cell_size = cur_cell_size;
                    seam_result.frame_level = 0u;
                    seam_result.frame_scale = 1.0;
                    seam_hit = true;
                    break;
                }

                let axis_code = sub.cell_size;
                if axis_code == 0.0 {
                    // Inner shell exit: break out to core-dispatch
                    // code below.
                    do_core_dispatch = true;
                    break;
                }
                if axis_code == 1.0 || axis_code < 0.0 {
                    // Outer-shell exit or walker exhaustion: the ray
                    // has passed through the body cell entirely.
                    body_cell_exit = true;
                    break;
                }

                // Axis codes 2..=5 encode a UV boundary exit:
                //   2 = u- (residual.x < 0)   → edge 0
                //   3 = u+ (residual.x ≥ 3)   → edge 1
                //   4 = v- (residual.y < 0)   → edge 2
                //   5 = v+ (residual.y ≥ 3)   → edge 3
                // The walker already picked a single exit axis; the
                // cube-corner tie-break (two axes OOB simultaneously)
                // is implicit in the walker's priority order
                // (r first, then x, then y). For grazing rays that
                // hit a corner the walker's pick is deterministic
                // and both candidate seams ultimately connect so
                // either choice is fine geometrically.
                let exit_pos_face = sub.cell_min; // (u, v, r) * 3
                var exit_edge: u32;
                if axis_code == 2.0 { exit_edge = 0u; }
                else if axis_code == 3.0 { exit_edge = 1u; }
                else if axis_code == 4.0 { exit_edge = 2u; }
                else { exit_edge = 3u; } // 5.0 = v+

                // Look up neighbor face from the precomputed seam
                // table. 4 edges per face; skip the R_seam multiply
                // for rd_local since `ray_dir` is already in
                // body-local coords — the next iteration re-rotates
                // via the neighbor's basis directly (equivalent by
                // construction: R_nbr · ray_dir = R_seam · R_cur · ray_dir).
                let seam_idx = cur_face * 4u + exit_edge;
                let seam = seam_table[seam_idx];
                let neighbor_face = seam.face_pad.x;

                // Reconstruct the exit body-space point from the
                // face-local exit (un, vn, rn) and re-project onto
                // the neighbor face. Both steps are O(1) and stay
                // in body-local [0, 3)³ coords — no absolute world
                // coordinates enter.
                let exit_un = clamp(exit_pos_face.x * (1.0 / 3.0), 0.0, 1.0);
                let exit_vn = clamp(exit_pos_face.y * (1.0 / 3.0), 0.0, 1.0);
                let exit_rn = clamp(exit_pos_face.z * (1.0 / 3.0), 0.0, 1.0);
                let exit_body = face_space_to_body_point(
                    cur_face, exit_un, exit_vn, exit_rn,
                    inner_body, outer_body,
                );
                let fp_nbr = body_point_to_face_forced(
                    exit_body, inner_body, outer_body, neighbor_face,
                );

                cur_face = neighbor_face;
                cur_un = fp_nbr.un;
                cur_vn = fp_nbr.vn;
                cur_rn = fp_nbr.rn;
                // Loop back and walk the neighbor face subtree.
            }

            if seam_hit {
                return seam_result;
            }
            if do_core_dispatch {
                // Inner shell → core subtree (CORE_SLOT child).
                let core_bit = 1u << CORE_SLOT;
                if (cur_occupancy & core_bit) == 0u {
                    // No core child: treat as miss, advance cell.
                    let m_ef = min_axis_mask(cur_side_dist);
                    s_cell[depth] = pack_cell(cell + vec3<i32>(m_ef) * step);
                    cur_side_dist += m_ef * delta_dist * cur_cell_size;
                    normal = -vec3<f32>(step) * m_ef;
                    continue;
                }
                let core_rank = countOneBits(cur_occupancy & (core_bit - 1u));
                let core_child_base = cur_first_child + core_rank * 2u;
                let core_packed = tree[core_child_base];
                let core_tag = core_packed & 0xFFu;
                if core_tag == 1u {
                    // Inner shell hits solid core directly.
                    result.hit = true;
                    let c_inner = dot(oc, oc) - inner_rib * inner_rib;
                    let disc_inner = b * b - c_inner;
                    var t_hit = t_enter;
                    if disc_inner >= 0.0 {
                        let sq_inner = sqrt(disc_inner);
                        let ti0 = -b - sq_inner;
                        let ti1 = -b + sq_inner;
                        t_hit = ti0;
                        if t_hit < 0.0 { t_hit = ti1; }
                        if t_hit < 0.0 { t_hit = t_enter; }
                    }
                    result.t = max(t_hit, 0.0);
                    result.color = palette[(core_packed >> 8u) & 0xFFFFu].rgb;
                    result.normal = -normalize(ray_dir);
                    result.cell_min = ray_origin + ray_dir * t_hit;
                    result.cell_size = inner_rib;
                    return result;
                }
                // Non-block core: fall through to body-cell exit.
            }

            // body_cell_exit fallthrough: advance past the body cell.
            let m_ef = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_ef) * step);
            cur_side_dist += m_ef * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_ef;
            continue;
        }

        // ============================================================
        // Cartesian + CubedSphereFace share the slot-pick arm.
        //
        // A CubedSphereFace node is the root of a face subtree whose
        // descendant cells are Cartesian-kind (UVR axis semantics
        // handled by face-basis rotation at body-entry). When we
        // arrive at one HERE (rather than via the CubedSphereBody
        // arm's `march_face_subtree` call), it means ribbon-pop
        // landed us inside a face subtree from an ancestor frame —
        // we proceed with the same Cartesian slot-pick.
        // ============================================================
        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        let slot_bit = 1u << slot;
        if ((cur_occupancy & slot_bit) == 0u) {
            if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
            let m_empty = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_empty) * step);
            cur_side_dist += m_empty * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_empty;
            continue;
        }

        let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 1u; }

        if tag == 1u {
            let cell_min_h = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
            let cell_max_h = cell_min_h + vec3<f32>(cur_cell_size);
            let cell_box_h = ray_box(ray_origin, inv_dir, cell_min_h, cell_max_h);
            result.hit = true;
            result.t = max(cell_box_h.t_enter, 0.0);
            result.color = palette[(packed >> 8u) & 0xFFFFu].rgb;
            result.normal = normal;
            result.cell_min = cell_min_h;
            result.cell_size = cur_cell_size;
            return result;
        } else if ENABLE_ENTITIES && tag == 3u {
            let entity_idx = tree[child_base + 1u];
            let entity = entities[entity_idx];
            let ebb = ray_box(ray_origin, inv_dir, entity.bbox_min, entity.bbox_max);
            if ebb.t_enter >= ebb.t_exit || ebb.t_exit < 0.0 {
                let m_bb = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_bb) * step);
                cur_side_dist += m_bb * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_bb;
                continue;
            }

            let bbox_size = entity.bbox_max - entity.bbox_min;
            let ray_dist_e = max(ebb.t_enter * ray_metric, 0.001);
            let lod_pixels_e = bbox_size.x / ray_dist_e
                * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            let at_max_e = depth + 1u >= MAX_STACK_DEPTH;
            let at_lod_e = lod_pixels_e < LOD_PIXEL_THRESHOLD;
            if at_max_e || at_lod_e {
                let rep = entity.representative_block;
                if rep < 0xFFFDu {
                    result.hit = true;
                    result.t = max(ebb.t_enter, 0.0);
                    result.color = palette[rep].rgb;
                    result.normal = -normalize(ray_dir);
                    result.cell_min = entity.bbox_min;
                    result.cell_size = bbox_size.x;
                    return result;
                }
                let m_lod_e = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_lod_e) * step);
                cur_side_dist += m_lod_e * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_lod_e;
                continue;
            }

            let scale3 = vec3<f32>(3.0) / bbox_size;
            let local_origin = (ray_origin - entity.bbox_min) * scale3;
            let local_dir = ray_dir * scale3;
            let sub = march_entity_subtree(entity.subtree_bfs, local_origin, local_dir);
            if sub.hit {
                let size_over_3 = bbox_size * (1.0 / 3.0);
                result.hit = true;
                result.t = sub.t / scale3.x;
                result.color = sub.color;
                result.normal = sub.normal;
                result.cell_min = entity.bbox_min + sub.cell_min * size_over_3;
                result.cell_size = sub.cell_size * size_over_3.x;
                return result;
            }
            let m_ent_miss = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_ent_miss) * step);
            cur_side_dist += m_ent_miss * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_ent_miss;
            continue;
        } else {
            let child_idx = tree[child_base + 1u];
            if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 1u; }

            let cell_slot = u32(cell.x) + u32(cell.y) * 3u + u32(cell.z) * 9u;
            if depth == 0u && cell_slot == skip_slot {
                let m_skip = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_skip) * step);
                cur_side_dist += m_skip * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_skip;
                continue;
            }

            let child_bt = child_block_type(packed);
            if child_bt == 0xFFFEu {
                if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
                let m_rep = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_rep) * step);
                cur_side_dist += m_rep * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_rep;
                continue;
            }

            let at_max = depth + 1u > max_depth_cap || depth + 1u >= MAX_STACK_DEPTH;
            let child_cell_size = cur_cell_size / 3.0;
            let cell_world_size = child_cell_size;
            let min_side = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
            let ray_dist = max(min_side * ray_metric, 0.001);
            let lod_pixels = cell_world_size / ray_dist * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;

            if at_max || at_lod {
                if ENABLE_STATS { ray_steps_lod_terminal = ray_steps_lod_terminal + 1u; }
                let bt = child_block_type(packed);
                if bt == 0xFFFEu {
                    let m_lodt = min_axis_mask(cur_side_dist);
                    s_cell[depth] = pack_cell(cell + vec3<i32>(m_lodt) * step);
                    cur_side_dist += m_lodt * delta_dist * cur_cell_size;
                    normal = -vec3<f32>(step) * m_lodt;
                } else {
                    let cell_min_l = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
                    let cell_max_l = cell_min_l + vec3<f32>(cur_cell_size);
                    let cell_box_l = ray_box(ray_origin, inv_dir, cell_min_l, cell_max_l);
                    result.hit = true;
                    result.t = max(cell_box_l.t_enter, 0.0);
                    result.color = palette[bt].rgb;
                    result.normal = normal;
                    result.cell_min = cell_min_l;
                    result.cell_size = cur_cell_size;
                    return result;
                }
            } else {
                let child_origin = cur_node_origin + vec3<f32>(cell) * cur_cell_size;

                let aabb_bits = aabbs[child_idx] & 0xFFFu;
                let has_aabb = aabb_bits != 0u;
                let amin = select(
                    vec3<f32>(0.0),
                    vec3<f32>(
                        f32(aabb_bits & 3u),
                        f32((aabb_bits >> 2u) & 3u),
                        f32((aabb_bits >> 4u) & 3u),
                    ),
                    has_aabb,
                );
                let amax = select(
                    vec3<f32>(3.0),
                    vec3<f32>(
                        f32(((aabb_bits >> 6u) & 3u) + 1u),
                        f32(((aabb_bits >> 8u) & 3u) + 1u),
                        f32(((aabb_bits >> 10u) & 3u) + 1u),
                    ),
                    has_aabb,
                );
                let aabb_min_world = child_origin + amin * child_cell_size;
                let aabb_max_world = child_origin + amax * child_cell_size;
                let aabb_hit = ray_box(ray_origin, inv_dir, aabb_min_world, aabb_max_world);
                if aabb_hit.t_exit <= aabb_hit.t_enter || aabb_hit.t_exit < 0.0 {
                    let m_aabb = min_axis_mask(cur_side_dist);
                    s_cell[depth] = pack_cell(cell + vec3<i32>(m_aabb) * step);
                    cur_side_dist += m_aabb * delta_dist * cur_cell_size;
                    normal = -vec3<f32>(step) * m_aabb;
                    if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
                    continue;
                }

                if ENABLE_STATS { ray_steps_node_descend = ray_steps_node_descend + 1u; }
                let node_hit = ray_box(
                    ray_origin, inv_dir,
                    child_origin,
                    child_origin + vec3<f32>(3.0) * child_cell_size,
                );
                let ct_start = max(node_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
                let child_entry = ray_origin + ray_dir * ct_start;
                let local_entry = (child_entry - child_origin) / child_cell_size;

                if ENABLE_STATS {
                    let preview_header_off = node_offsets[child_idx];
                    let preview_occ = tree[preview_header_off];
                    let preview_entry_cell = vec3<i32>(
                        i32(floor(local_entry.x)),
                        i32(floor(local_entry.y)),
                        i32(floor(local_entry.z)),
                    );
                    let pm = path_mask_conservative(preview_entry_cell, step);
                    if (preview_occ & pm) == 0u {
                        ray_steps_would_cull = ray_steps_would_cull + 1u;
                    }
                }

                depth += 1u;
                s_node_idx[depth] = child_idx;
                cur_node_origin = child_origin;
                cur_cell_size = child_cell_size;
                let child_header_off = node_offsets[child_idx];
                cur_occupancy = tree[child_header_off];
                cur_first_child = tree[child_header_off + 1u];
                if ENABLE_STATS {
                    ray_loads_offsets = ray_loads_offsets + 1u;
                    ray_loads_tree = ray_loads_tree + 2u;
                }
                let new_cell = vec3<i32>(
                    clamp(i32(floor(local_entry.x)), 0, 2),
                    clamp(i32(floor(local_entry.y)), 0, 2),
                    clamp(i32(floor(local_entry.z)), 0, 2),
                );
                s_cell[depth] = pack_cell(new_cell);
                let lc = vec3<f32>(new_cell);
                cur_side_dist = vec3<f32>(
                    select((child_origin.x + lc.x * child_cell_size - entry_pos.x) * inv_dir.x,
                           (child_origin.x + (lc.x + 1.0) * child_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                    select((child_origin.y + lc.y * child_cell_size - entry_pos.y) * inv_dir.y,
                           (child_origin.y + (lc.y + 1.0) * child_cell_size - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
                    select((child_origin.z + lc.z * child_cell_size - entry_pos.z) * inv_dir.z,
                           (child_origin.z + (lc.z + 1.0) * child_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
                );
            }
        }
    }

    return result;
}
