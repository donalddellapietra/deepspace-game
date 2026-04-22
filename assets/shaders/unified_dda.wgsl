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
//   * Face-subtree walking happens in body-local ribbon-frame coords
//     via `march_face_subtree_curved`: at each cell step, cell
//     boundaries are the REAL curved equal-angle u/v planes and
//     r-spheres on the shell (not the fake axis-aligned boxes a
//     rotated-face-space walk would produce). Boundary-surface ray
//     intersects stay O(1) magnitude regardless of face-subtree depth.
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

fn sign_or_one(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// ────────────────────────────────── Curved-cell face walker
//
// Body-local curved DDA over a cubed-sphere face subtree. The ray
// stays in body-local (ribbon-frame) coords throughout — cell
// boundaries are the actual curved equal-angle surfaces on the
// sphere shell, not pretend axis-aligned planes in face-local.
//
// Geometry of a face-subtree cell at face F with face-normalized
// bounds `[u_lo, u_hi] × [v_lo, v_hi] × [r_lo, r_hi]` (each ∈ [0,1]):
//   * u = u_const plane: set of directions `normalize(n_F +
//     cube_u(u_const) · u_axis_F + s · v_axis_F)` for scalar s.
//     These directions form a 2D plane through body center with
//     normal = normalize(cross(v_axis_F, n_F + cube_u · u_axis_F)).
//   * v = v_const plane: symmetric.
//   * r = r_const shell: sphere centered at body center with
//     radius = inner_rib + (outer_rib - inner_rib) · r_const.
//
// This replaces the broken Stage 3 approach that rotated the ray
// into face-local coords and walked axis-aligned cells — which
// treated the face's orthonormal basis as globally valid and so
// rendered the whole sphere as a warped cube.
//
// Function signature packs the walker's result into HitResult:
//   * On hit: `hit=true`, `t` is ribbon-frame param, `cell_min` is
//     the body-local hit point.
//   * On exit: `hit=false`, `cell_min` is the body-local exit point,
//     `cell_size` is an exit code:
//       0 = r- (inner shell, dispatch core)
//       1 = r+ (outer shell, body exit)
//      -1 = exhausted (treat as body exit)
//     (UV seam exits are handled internally — the walker keeps
//      walking on the neighbor face until one of the above cases.)
//
// The walker computes u/v plane normals fresh per iteration because
// they depend on the current cell's u_const/v_const edges; the
// small cross() cost is negligible vs the tree loads.
fn march_face_subtree_curved(
    body_occupancy: u32,
    body_first_child: u32,
    start_root_node_idx: u32,
    start_face: u32,
    body_center_rib: vec3<f32>,
    outer_rib: f32,
    inner_rib: f32,
    ray_origin_rib: vec3<f32>,
    ray_dir_rib: vec3<f32>,
    t_enter_body: f32,
    ray_metric_rib: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = -1.0;

    // Body-centered ray: oc = origin - body_center. All t's below
    // are in ribbon-frame units relative to the original ray_origin.
    let oc = ray_origin_rib - body_center_rib;
    let rd = ray_dir_rib;

    // Tiny slice of the ray in front of us (we just landed at t_enter
    // on the outer shell, so start just inside).
    let eps_t = 1e-5 * outer_rib;
    var t_cur: f32 = t_enter_body + eps_t;


    // Current face (0..5) — may change on seam crossings.
    var cur_face: u32 = start_face;

    // Face subtree stack. `s_node_idx[d]` = node at depth d. The
    // face subtree root is at depth 0. `s_slot[d]` holds (u_slot,
    // v_slot, r_slot) of the CHILD chosen at depth d (packed 2 bits
    // each) for the range [0, depth). Current-cell slot at depth
    // `depth` is held in (cur_us, cur_vs, cur_rs) below. When we
    // descend, we push (cur_us, cur_vs, cur_rs) into s_slot[depth-1]
    // (wait — actually into s_slot[old_depth]) and depth+=1.
    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_slot_u: array<i32, MAX_STACK_DEPTH>;
    var s_slot_v: array<i32, MAX_STACK_DEPTH>;
    var s_slot_r: array<i32, MAX_STACK_DEPTH>;
    s_node_idx[0] = start_root_node_idx;

    var depth: u32 = 0u;
    var cur_u_lo: f32 = 0.0;
    var cur_v_lo: f32 = 0.0;
    var cur_r_lo: f32 = 0.0;
    var cur_cell_ext: f32 = 1.0 / 3.0;
    var cur_us: i32 = 0;
    var cur_vs: i32 = 0;
    var cur_rs: i32 = 0;

    // Compute entry (un, vn, rn) on current face, derive initial slot.
    // Wrapping this in a closure-like block to re-run on seam cross.
    var cur_header_off = node_offsets[start_root_node_idx];
    var cur_occupancy: u32 = tree[cur_header_off];
    var cur_first_child: u32 = tree[cur_header_off + 1u];

    // Track the surface normal of the LAST boundary the ray crossed.
    // At any cell-block hit, this is the normal of the face we're
    // viewing — the boundary that separated the previous empty cell
    // from this solid cell. Stored in body-local (ribbon-frame) coords
    // to match the Cartesian walker's convention.
    //
    // Initialised below to the entry cell's CELL-CENTER radial (see
    // Stage 4 faceted-shading comment near the step-advance block).
    var hit_normal: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

    // Compute (un, vn, rn) of the ray's current body-centered position
    // onto `cur_face`. Used to pick the initial slot at depth=0 after
    // entering a face (either fresh from body-enter, or after a seam
    // crossing from a neighbor face).
    //
    // After setting cur_face, call `reinit_on_face()` to seed cur_us /
    // cur_vs / cur_rs / cur_u_lo / cur_v_lo / cur_r_lo at depth 0.
    //
    // WGSL has no nested fns, so inline this logic below.

    // Seed initial face slot from the entry point.
    let p0 = oc + rd * t_cur;
    let r2_0 = dot(p0, p0);
    var r0 = sqrt(max(r2_0, 1e-30));
    var rn0 = clamp((r0 - inner_rib) / max(outer_rib - inner_rib, 1e-10), 0.0, 1.0);
    // (un, vn) on cur_face via forced projection.
    let basis0 = face_basis(cur_face);
    let n_comp0 = dot(basis0.n_axis, p0);
    let inv_nc0 = 1.0 / max(abs(n_comp0), 1e-6);
    let cu0 = dot(basis0.u_axis, p0) * inv_nc0 * sign_or_one(n_comp0);
    let cv0 = dot(basis0.v_axis, p0) * inv_nc0 * sign_or_one(n_comp0);
    let un0 = clamp(0.5 * (atan(cu0) * (4.0 / 3.14159265) + 1.0), 0.0, 1.0);
    let vn0 = clamp(0.5 * (atan(cv0) * (4.0 / 3.14159265) + 1.0), 0.0, 1.0);
    cur_us = clamp(i32(floor(un0 * 3.0)), 0, 2);
    cur_vs = clamp(i32(floor(vn0 * 3.0)), 0, 2);
    cur_rs = clamp(i32(floor(rn0 * 3.0)), 0, 2);
    cur_u_lo = f32(cur_us) / 3.0;
    cur_v_lo = f32(cur_vs) / 3.0;
    cur_r_lo = f32(cur_rs) / 3.0;
    cur_cell_ext = 1.0 / 3.0;

    // Seed hit_normal to the ENTRY cell's centre radial on cur_face.
    // Ray just crossed the outer r-shell; the first cell's r-face is
    // what we're "viewing" if that cell turns out to be solid.
    {
        let u_mid_init = cur_u_lo + 0.5 * cur_cell_ext;
        let v_mid_init = cur_v_lo + 0.5 * cur_cell_ext;
        let cu_init = tan((2.0 * u_mid_init - 1.0) * 0.78539816);
        let cv_init = tan((2.0 * v_mid_init - 1.0) * 0.78539816);
        var raw_init = basis0.n_axis
            + cu_init * basis0.u_axis
            + cv_init * basis0.v_axis;
        let rln = max(length(raw_init), 1e-6);
        var nu_init = raw_init / rln;
        if dot(nu_init, rd) > 0.0 { nu_init = -nu_init; }
        hit_normal = nu_init;
    }

    var iterations: u32 = 0u;
    let max_iterations: u32 = 2048u;
    var seam_count: u32 = 0u;
    let max_seams: u32 = MAX_SEAM_TRANSITIONS;

    loop {
        if iterations >= max_iterations { break; }
        iterations += 1u;

        // Current cell bounds (face-normalized):
        let u_lo = cur_u_lo;
        let u_hi = cur_u_lo + cur_cell_ext;
        let v_lo = cur_v_lo;
        let v_hi = cur_v_lo + cur_cell_ext;
        let r_lo = cur_r_lo;
        let r_hi = cur_r_lo + cur_cell_ext;

        // ---- Per-cell Nyquist LOD gate --------------------------------
        //
        // If the CURRENT cell projects below the pixel threshold, any
        // further descent is sub-pixel; treat the cell as a terminal
        // splat. When the cell's slot resolves to a real block (tag=1)
        // or a Node with a representative block, return a hit using
        // the stored boundary-crossing normal. When it's empty, fall
        // through to the ordinary DDA advance so the ray keeps
        // stepping through small empty cells until it clears the
        // face subtree (r-shell or seam exit).
        {
            let cell_rib_size_lod = cur_cell_ext * outer_rib;
            let ray_dist_lod = max(t_cur * ray_metric_rib, 0.001);
            let cell_pixels = cell_rib_size_lod / ray_dist_lod
                * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            if cell_pixels < LOD_PIXEL_THRESHOLD {
                let slot_lod = u32(cur_rs * 9 + cur_vs * 3 + cur_us);
                let slot_bit_lod = 1u << slot_lod;
                var lod_bt: u32 = 0xFFFEu;
                if (cur_occupancy & slot_bit_lod) != 0u {
                    let rank_lod = countOneBits(cur_occupancy & (slot_bit_lod - 1u));
                    let child_base_lod = cur_first_child + rank_lod * 2u;
                    let packed_lod = tree[child_base_lod];
                    let tag_lod = packed_lod & 0xFFu;
                    if tag_lod == 1u {
                        lod_bt = (packed_lod >> 8u) & 0xFFFFu;
                    } else if tag_lod == 2u {
                        lod_bt = child_block_type(packed_lod);
                    }
                }
                if lod_bt < 0xFFFDu {
                    result.hit = true;
                    result.t = t_cur;
                    result.color = palette[lod_bt].rgb;
                    result.normal = hit_normal;
                    let hp_lod = oc + rd * t_cur;
                    result.cell_min = hp_lod + body_center_rib;
                    result.cell_size = cur_cell_ext * outer_rib;
                    return result;
                }
                // Empty sub-pixel cell: fall through to ordinary DDA
                // advance below. The advance still uses this cell's
                // bounds to step to the next neighbor, same as a
                // normal empty cell. The LOD gate skipped only the
                // descent into children we wouldn't have visited.
            }
        }

        // ---- Compute exit surfaces for THIS cell and pick the
        //      smallest positive t strictly > t_cur.
        //
        // u-boundary planes: normal = cross(v_axis, n + cube_u ·
        //   u_axis). Plane passes through body center.
        //   ray-plane: t = -dot(oc, n) / dot(rd, n).
        let basis = face_basis(cur_face);
        let cu_lo = tan((2.0 * u_lo - 1.0) * 0.78539816);
        let cu_hi = tan((2.0 * u_hi - 1.0) * 0.78539816);
        let cv_lo = tan((2.0 * v_lo - 1.0) * 0.78539816);
        let cv_hi = tan((2.0 * v_hi - 1.0) * 0.78539816);
        let n_plane_u_lo = cross(basis.v_axis, basis.n_axis + cu_lo * basis.u_axis);
        let n_plane_u_hi = cross(basis.v_axis, basis.n_axis + cu_hi * basis.u_axis);
        let n_plane_v_lo = cross(basis.n_axis + cv_lo * basis.v_axis, basis.u_axis);
        let n_plane_v_hi = cross(basis.n_axis + cv_hi * basis.v_axis, basis.u_axis);
        let R_lo = inner_rib + (outer_rib - inner_rib) * r_lo;
        let R_hi = inner_rib + (outer_rib - inner_rib) * r_hi;

        let t_eps = t_cur + 1e-6 * outer_rib;

        // ray-plane intersects. Return -1 if parallel/behind.
        let t_u_lo_raw = ray_plane_t(oc, rd, vec3<f32>(0.0), n_plane_u_lo);
        let t_u_hi_raw = ray_plane_t(oc, rd, vec3<f32>(0.0), n_plane_u_hi);
        let t_v_lo_raw = ray_plane_t(oc, rd, vec3<f32>(0.0), n_plane_v_lo);
        let t_v_hi_raw = ray_plane_t(oc, rd, vec3<f32>(0.0), n_plane_v_hi);
        // ray-sphere. Use our stable after-t helper.
        let t_r_lo_raw = ray_sphere_after(oc, rd, vec3<f32>(0.0), R_lo, t_eps);
        let t_r_hi_raw = ray_sphere_after(oc, rd, vec3<f32>(0.0), R_hi, t_eps);

        let BIG: f32 = 1e30;
        let t_u_lo = select(BIG, t_u_lo_raw, t_u_lo_raw > t_eps);
        let t_u_hi = select(BIG, t_u_hi_raw, t_u_hi_raw > t_eps);
        let t_v_lo = select(BIG, t_v_lo_raw, t_v_lo_raw > t_eps);
        let t_v_hi = select(BIG, t_v_hi_raw, t_v_hi_raw > t_eps);
        let t_r_lo = select(BIG, t_r_lo_raw, t_r_lo_raw > t_eps);
        let t_r_hi = select(BIG, t_r_hi_raw, t_r_hi_raw > t_eps);

        // Pick minimum. Record which face (0=u-, 1=u+, 2=v-, 3=v+,
        // 4=r-, 5=r+) was crossed.
        var t_next: f32 = t_u_lo;
        var exit_axis: u32 = 0u;
        if t_u_hi < t_next { t_next = t_u_hi; exit_axis = 1u; }
        if t_v_lo < t_next { t_next = t_v_lo; exit_axis = 2u; }
        if t_v_hi < t_next { t_next = t_v_hi; exit_axis = 3u; }
        if t_r_lo < t_next { t_next = t_r_lo; exit_axis = 4u; }
        if t_r_hi < t_next { t_next = t_r_hi; exit_axis = 5u; }

        if t_next >= BIG {
            // Shouldn't happen; ray is trapped. Bail as body-exit.
            result.hit = false;
            result.cell_min = oc + rd * t_cur + body_center_rib;
            result.cell_size = 1.0;
            return result;
        }

        // ---- Look up the current cell's child and decide action.
        let slot = u32(cur_rs * 9 + cur_vs * 3 + cur_us);
        let slot_bit = 1u << slot;
        if (cur_occupancy & slot_bit) != 0u {
            let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
            let child_base = cur_first_child + rank * 2u;
            let packed = tree[child_base];
            let tag = packed & 0xFFu;

            if tag == 1u {
                // Block hit at this cell. Use t at cell entry (t_cur)
                // for the ribbon-frame t. Stage 4: the hit normal is
                // the normal of the boundary surface the ray crossed
                // LAST before entering this cell (tracked in
                // `hit_normal`). For the first cell (no prior crossing),
                // `hit_normal` was seeded to the outer-shell radial at
                // entry. This yields faceted cube-cell shading on the
                // sphere — each terminal solid cell shows up to 3
                // distinct normal directions (u/v/r-face) just like
                // Cartesian voxel blocks have ±x/±y/±z faces.
                result.hit = true;
                result.t = t_cur;
                result.color = palette[(packed >> 8u) & 0xFFFFu].rgb;
                result.normal = hit_normal;
                let hit_pos = oc + rd * t_cur;
                result.cell_min = hit_pos + body_center_rib;
                result.cell_size = cur_cell_ext * outer_rib; // approx
                return result;
            }

            if tag == 2u {
                let child_idx = tree[child_base + 1u];
                let child_bt = child_block_type(packed);
                // LOD termination check. Child cell extent in ribbon:
                // roughly cur_cell_ext/3 × outer_rib (diagonal size).
                let child_cell_ext_fn = cur_cell_ext / 3.0;
                let child_cell_rib = child_cell_ext_fn * outer_rib;
                let ray_dist = max(t_cur * ray_metric_rib, 0.001);
                let lod_pixels = child_cell_rib / ray_dist
                    * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
                let at_max = depth + 1u >= MAX_STACK_DEPTH;
                let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;

                if !(at_max || at_lod) && child_bt != 0xFFFEu {
                    // Descend.
                    s_slot_u[depth] = cur_us;
                    s_slot_v[depth] = cur_vs;
                    s_slot_r[depth] = cur_rs;
                    depth += 1u;
                    s_node_idx[depth] = child_idx;
                    cur_header_off = node_offsets[child_idx];
                    cur_occupancy = tree[cur_header_off];
                    cur_first_child = tree[cur_header_off + 1u];
                    // Shrink cell extent; new cell is the child cell
                    // within the current one, at slot determined by
                    // the current ray position projected onto face.
                    cur_cell_ext = child_cell_ext_fn;
                    // Parent cell bounds become the base for children.
                    let parent_u_lo = u_lo;
                    let parent_v_lo = v_lo;
                    let parent_r_lo = r_lo;
                    // Current ray position in face-normalized coords.
                    let p_here = oc + rd * t_cur;
                    let rh = max(length(p_here), 1e-6);
                    let b = face_basis(cur_face);
                    let nc = dot(b.n_axis, p_here);
                    let inv_nc = 1.0 / max(abs(nc), 1e-6);
                    let cu = dot(b.u_axis, p_here) * inv_nc * sign_or_one(nc);
                    let cv = dot(b.v_axis, p_here) * inv_nc * sign_or_one(nc);
                    let un = clamp(0.5 * (atan(cu) * (4.0 / 3.14159265) + 1.0), 0.0, 1.0);
                    let vn = clamp(0.5 * (atan(cv) * (4.0 / 3.14159265) + 1.0), 0.0, 1.0);
                    let rn = clamp((rh - inner_rib) / max(outer_rib - inner_rib, 1e-10), 0.0, 1.0);
                    // Child slot: fractional position within parent.
                    let fu = clamp((un - parent_u_lo) / (cur_cell_ext * 3.0), 0.0, 1.0 - 1e-6);
                    let fv = clamp((vn - parent_v_lo) / (cur_cell_ext * 3.0), 0.0, 1.0 - 1e-6);
                    let fr = clamp((rn - parent_r_lo) / (cur_cell_ext * 3.0), 0.0, 1.0 - 1e-6);
                    cur_us = clamp(i32(floor(fu * 3.0)), 0, 2);
                    cur_vs = clamp(i32(floor(fv * 3.0)), 0, 2);
                    cur_rs = clamp(i32(floor(fr * 3.0)), 0, 2);
                    cur_u_lo = parent_u_lo + f32(cur_us) * cur_cell_ext;
                    cur_v_lo = parent_v_lo + f32(cur_vs) * cur_cell_ext;
                    cur_r_lo = parent_r_lo + f32(cur_rs) * cur_cell_ext;
                    // Do NOT advance t_cur; the ray is still at the
                    // same body position. Re-loop to process new cell.
                    continue;
                } else if !(at_max || at_lod) && child_bt == 0xFFFEu {
                    // Representative-empty: treat as empty, fall through.
                } else {
                    // LOD terminal hit.
                    let bt = child_bt;
                    if bt == 0xFFFEu || bt == 0xFFFDu {
                        // Empty / no-representative: fall through as empty.
                    } else {
                        result.hit = true;
                        result.t = t_cur;
                        result.color = palette[bt].rgb;
                        // Faceted normal from the last boundary the ray
                        // crossed (same convention as the tag==1 hit above).
                        result.normal = hit_normal;
                        let hit_pos = oc + rd * t_cur;
                        result.cell_min = hit_pos + body_center_rib;
                        result.cell_size = cur_cell_ext * outer_rib;
                        return result;
                    }
                }
            }
            // tag 0 or other: fall through as empty.
        }
        // Empty or fall-through: advance to exit surface of this cell
        // and step to the neighbor slot in the appropriate direction.
        //
        // Record the normal of the surface we just crossed; this will
        // be the hit normal if the NEXT cell is solid. Sign-picked to
        // oppose `rd` so the normal always points back toward the ray
        // origin (standard shading convention). Normal is in body-
        // local coords (ribbon-frame after body-center translation,
        // same as the Cartesian walker's normal).
        //
        // Stage 4 faceted voxel shading: r-shell normal uses the
        // CELL-CENTER radial direction (one fixed direction per cell)
        // instead of the pointwise radial, giving each cell a flat
        // r-face facet. u- and v-plane normals are naturally flat
        // within a cell (they only depend on the cell's u_const /
        // v_const edges). At deep depths the cell-center radial
        // converges to the pointwise radial, so this stays continuous
        // with infinite descent.
        {
            var raw_n: vec3<f32>;
            if exit_axis == 0u {
                raw_n = n_plane_u_lo;
            } else if exit_axis == 1u {
                raw_n = n_plane_u_hi;
            } else if exit_axis == 2u {
                raw_n = n_plane_v_lo;
            } else if exit_axis == 3u {
                raw_n = n_plane_v_hi;
            } else {
                // r-shell (inner or outer): radial at the CURRENT
                // cell's center on `cur_face`. One direction per cell
                // → flat facet. `u_mid`/`v_mid` are face-normalized
                // coords of the cell's centre; map them to cube
                // tangent-plane coords via the standard atan → tan
                // inverse.
                let u_mid = u_lo + 0.5 * cur_cell_ext;
                let v_mid = v_lo + 0.5 * cur_cell_ext;
                let cu_mid = tan((2.0 * u_mid - 1.0) * 0.78539816);
                let cv_mid = tan((2.0 * v_mid - 1.0) * 0.78539816);
                raw_n = basis.n_axis
                    + cu_mid * basis.u_axis
                    + cv_mid * basis.v_axis;
            }
            // Normalize and sign-flip to oppose rd.
            let rn_len = max(length(raw_n), 1e-6);
            var n_unit = raw_n / rn_len;
            if dot(n_unit, rd) > 0.0 { n_unit = -n_unit; }
            hit_normal = n_unit;
        }
        t_cur = t_next;

        // Update slot based on which axis exited.
        // 0=u-, 1=u+, 2=v-, 3=v+, 4=r-, 5=r+
        if exit_axis == 0u {
            cur_us -= 1;
            cur_u_lo -= cur_cell_ext;
        } else if exit_axis == 1u {
            cur_us += 1;
            cur_u_lo += cur_cell_ext;
        } else if exit_axis == 2u {
            cur_vs -= 1;
            cur_v_lo -= cur_cell_ext;
        } else if exit_axis == 3u {
            cur_vs += 1;
            cur_v_lo += cur_cell_ext;
        } else if exit_axis == 4u {
            cur_rs -= 1;
            cur_r_lo -= cur_cell_ext;
        } else {
            cur_rs += 1;
            cur_r_lo += cur_cell_ext;
        }

        // Check for OOB — bubble up.
        loop {
            if cur_us >= 0 && cur_us <= 2 && cur_vs >= 0 && cur_vs <= 2 && cur_rs >= 0 && cur_rs <= 2 {
                break;
            }
            // This level's slot went OOB. At face-subtree depth 0,
            // this means we exited the face subtree entirely. If it's
            // an r exit, return to caller with shell code. If it's a
            // u/v exit, do seam crossing here: pick neighbor face
            // from the ray's body-centered exit position.
            if depth == 0u {
                let is_r_exit = (cur_rs < 0) || (cur_rs > 2);
                if is_r_exit {
                    result.hit = false;
                    result.cell_min = oc + rd * t_cur + body_center_rib;
                    if cur_rs < 0 { result.cell_size = 0.0; } // inner
                    else { result.cell_size = 1.0; }          // outer
                    return result;
                }
                // UV exit = face-seam cross. Determine neighbor face
                // from the exit EDGE of the current face: the neighbor
                // is the face whose normal is the direction we were
                // stepping in (± u or ± v of current face).
                seam_count += 1u;
                if seam_count >= max_seams {
                    result.hit = false;
                    result.cell_min = oc + rd * t_cur + body_center_rib;
                    result.cell_size = 1.0; // treat as outer-exit
                    return result;
                }
                let p_exit = oc + rd * t_cur;
                let rex = max(length(p_exit), 1e-6);
                // Determine exit edge from which of (cur_us, cur_vs)
                // went OOB. cur_rs must be in bounds here (r-exit was
                // handled by is_r_exit above).
                let basis_exit = face_basis(cur_face);
                var target_normal: vec3<f32>;
                if cur_us < 0      { target_normal = -basis_exit.u_axis; }
                else if cur_us > 2 { target_normal =  basis_exit.u_axis; }
                else if cur_vs < 0 { target_normal = -basis_exit.v_axis; }
                else               { target_normal =  basis_exit.v_axis; }
                // Find the face index whose normal matches target.
                // We know target_normal is ±e_i for some i (since
                // u_axis, v_axis are unit cube axes).
                var new_face: u32 = cur_face;
                if target_normal.x >  0.5 { new_face = 0u; } // PosX
                else if target_normal.x < -0.5 { new_face = 1u; } // NegX
                else if target_normal.y >  0.5 { new_face = 2u; } // PosY
                else if target_normal.y < -0.5 { new_face = 3u; } // NegY
                else if target_normal.z >  0.5 { new_face = 4u; } // PosZ
                else                           { new_face = 5u; } // NegZ
                // Sanity: new_face must be different from cur_face.
                if new_face == cur_face {
                    result.hit = false;
                    result.cell_min = p_exit + body_center_rib;
                    result.cell_size = 1.0;
                    return result;
                }
                cur_face = new_face;
                // Look up the neighbor face's subtree root in the
                // body node's children.
                let face_slot = FACE_SLOTS[cur_face];
                let face_bit = 1u << face_slot;
                if (body_occupancy & face_bit) == 0u {
                    // No neighbor face subtree: treat as body-exit.
                    result.hit = false;
                    result.cell_min = p_exit + body_center_rib;
                    result.cell_size = 1.0;
                    return result;
                }
                let face_rank = countOneBits(body_occupancy & (face_bit - 1u));
                let face_child_base = body_first_child + face_rank * 2u;
                let face_packed = tree[face_child_base];
                let face_tag = face_packed & 0xFFu;
                if face_tag != 2u {
                    result.hit = false;
                    result.cell_min = p_exit + body_center_rib;
                    result.cell_size = 1.0;
                    return result;
                }
                let new_root_idx = tree[face_child_base + 1u];
                // Reset subtree stack to new face root.
                depth = 0u;
                s_node_idx[0] = new_root_idx;
                cur_header_off = node_offsets[new_root_idx];
                cur_occupancy = tree[cur_header_off];
                cur_first_child = tree[cur_header_off + 1u];

                // Reinit slot on new face at depth 0.
                let rnex = clamp((rex - inner_rib) / max(outer_rib - inner_rib, 1e-10), 0.0, 1.0);
                let bn = face_basis(cur_face);
                let nc = dot(bn.n_axis, p_exit);
                let inv_nc = 1.0 / max(abs(nc), 1e-6);
                let cu = dot(bn.u_axis, p_exit) * inv_nc * sign_or_one(nc);
                let cv = dot(bn.v_axis, p_exit) * inv_nc * sign_or_one(nc);
                let un_ = clamp(0.5 * (atan(cu) * (4.0 / 3.14159265) + 1.0), 0.0, 1.0);
                let vn_ = clamp(0.5 * (atan(cv) * (4.0 / 3.14159265) + 1.0), 0.0, 1.0);
                cur_us = clamp(i32(floor(un_ * 3.0)), 0, 2);
                cur_vs = clamp(i32(floor(vn_ * 3.0)), 0, 2);
                cur_rs = clamp(i32(floor(rnex * 3.0)), 0, 2);
                cur_u_lo = f32(cur_us) / 3.0;
                cur_v_lo = f32(cur_vs) / 3.0;
                cur_r_lo = f32(cur_rs) / 3.0;
                cur_cell_ext = 1.0 / 3.0;
                // Break inner bubble-up loop; re-enter main loop with
                // fresh face state.
                break;
            }
            // Bubble up one level. Recompute parent cell bounds.
            depth -= 1u;
            cur_cell_ext *= 3.0;
            // Parent slot at this (now current) depth is what we
            // descended from: s_slot_*[depth].
            let pu = s_slot_u[depth];
            let pv = s_slot_v[depth];
            let pr = s_slot_r[depth];
            // Parent's u_lo = current_u_lo reconciled: the parent cell
            // started at parent_lo = cur_u_lo - pu * cur_cell_ext_child.
            // After multiplying cur_cell_ext by 3 above, the parent's
            // cell_ext is cur_cell_ext (3×), and its lo = cur_u_lo -
            // (whatever offset we added from the child slot + any
            // neighbor steps). Simpler: parent_u_lo = parent_u_slot /
            // 3^(depth+1) accumulated. But we already track cur_u_lo
            // incrementally; when descending we stored the child's
            // u_lo. To invert we need the parent's u_lo. We can
            // derive parent_u_lo by subtracting the child slot's
            // contribution — but we've stepped in between. Redo: pop
            // s_slot_u[depth] and reset cur_u_lo = pu * parent_ext +
            // cur_u_lo - the offset we added. Actually easier to
            // recompute from full stack:
            var u_lo_acc: f32 = 0.0;
            var v_lo_acc: f32 = 0.0;
            var r_lo_acc: f32 = 0.0;
            var ext_acc: f32 = 1.0 / 3.0;
            for (var d: u32 = 0u; d < depth; d = d + 1u) {
                u_lo_acc += f32(s_slot_u[d]) * ext_acc;
                v_lo_acc += f32(s_slot_v[d]) * ext_acc;
                r_lo_acc += f32(s_slot_r[d]) * ext_acc;
                ext_acc /= 3.0;
            }
            // cur_us at this (parent) depth = pu + the step we took.
            // But we came from child; we need to figure out which
            // neighbor cell we should be in at this depth. We know
            // the stepped exit axis at the CHILD level — but after
            // the child's slot became OOB, the parent's slot should
            // step by 1 on that same axis.
            //
            // Concretely: if cur_us went to -1 at child level, then
            // at parent we are at slot (pu - 1, pv, pr) — we crossed
            // the parent's u_lo boundary. If cur_us went to 3, we
            // are at (pu + 1, pv, pr). Similarly for v, r.
            //
            // Determine which axis overflowed at child (only one):
            if cur_us < 0 { cur_us = pu - 1; cur_vs = pv; cur_rs = pr; }
            else if cur_us > 2 { cur_us = pu + 1; cur_vs = pv; cur_rs = pr; }
            else if cur_vs < 0 { cur_vs = pv - 1; cur_us = pu; cur_rs = pr; }
            else if cur_vs > 2 { cur_vs = pv + 1; cur_us = pu; cur_rs = pr; }
            else if cur_rs < 0 { cur_rs = pr - 1; cur_us = pu; cur_vs = pv; }
            else { cur_rs = pr + 1; cur_us = pu; cur_vs = pv; }
            cur_u_lo = u_lo_acc + f32(cur_us) * ext_acc;
            cur_v_lo = v_lo_acc + f32(cur_vs) * ext_acc;
            cur_r_lo = r_lo_acc + f32(cur_rs) * ext_acc;
            // Refresh node header.
            cur_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[cur_header_off];
            cur_first_child = tree[cur_header_off + 1u];
            // Loop back to re-check OOB on this (new, parent) level.
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
            // Stage 3b: `march_face_subtree_curved` walks the face
            // subtree in body-local coords with true curved-cell
            // boundary surfaces (two u-planes, two v-planes, two
            // r-spheres per cell). UV-boundary exits at face-subtree
            // depth 0 are seams; the walker hops internally to the
            // neighbor face subtree via body-local re-projection.
            // Shell exits (r-) trigger core-subtree dispatch below;
            // (r+) exits return to this dispatcher as body-exit.
            let kind_data = node_kinds[s_node_idx[depth]];
            let body_center = cur_node_origin + vec3<f32>(1.5 * cur_cell_size);
            let outer_rib = kind_data.outer_r * 3.0 * cur_cell_size;
            let inner_rib = kind_data.inner_r * 3.0 * cur_cell_size;
            let inner_body = inner_rib / cur_cell_size; // in body-size=3 units
            let outer_body = outer_rib / cur_cell_size;

            // Stable Numerical-Recipes ray-sphere (camera can be
            // inside or outside the shell). ray_dir may be non-unit
            // in unified_dda (comes through ribbon-pop rescales and
            // raw jittered camera rays), so use the full quadratic
            // form: (dir·dir) t² + 2b t + c = 0 with b = oc·dir,
            // c = oc·oc - r². Discriminant = b² - (dir·dir)·c.
            let oc = ray_origin - body_center;
            let dd = dot(ray_dir, ray_dir);
            let inv_dd = 1.0 / max(dd, 1e-20);
            let b = dot(oc, ray_dir);
            let c_outer = dot(oc, oc) - outer_rib * outer_rib;
            let disc_outer = b * b - dd * c_outer;
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
            // For unit dir, t = -b ± sq; for non-unit dir, divide by
            // dd. Use the stable form: t0 = (-b - sq) / dd.
            let t0 = (-b - sq_outer) * inv_dd;
            let t1 = (-b + sq_outer) * inv_dd;
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

            // Look up the face subtree root from the body's children.
            var start_root_idx: u32 = 0u;
            var face_ok: bool = false;
            {
                let face_slot = FACE_SLOTS[cur_face];
                let face_bit = 1u << face_slot;
                if (cur_occupancy & face_bit) != 0u {
                    let face_rank = countOneBits(cur_occupancy & (face_bit - 1u));
                    let face_child_base = cur_first_child + face_rank * 2u;
                    let face_packed = tree[face_child_base];
                    if (face_packed & 0xFFu) == 2u {
                        start_root_idx = tree[face_child_base + 1u];
                        face_ok = true;
                    }
                }
            }

            var body_cell_exit = false;
            var do_core_dispatch = false;
            var seam_hit = false;
            var seam_result: HitResult;
            seam_result.hit = false;

            if !face_ok {
                body_cell_exit = true;
            } else {
                let sub = march_face_subtree_curved(
                    cur_occupancy, cur_first_child,
                    start_root_idx, cur_face,
                    body_center,
                    outer_rib, inner_rib,
                    ray_origin, ray_dir,
                    t_enter,
                    ray_metric,
                );
                if sub.hit {
                    seam_result = sub;
                    seam_hit = true;
                } else {
                    if sub.cell_size == 0.0 {
                        do_core_dispatch = true;
                    } else {
                        body_cell_exit = true;
                    }
                }
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
                    let disc_inner = b * b - dd * c_inner;
                    var t_hit = t_enter;
                    if disc_inner >= 0.0 {
                        let sq_inner = sqrt(disc_inner);
                        let ti0 = (-b - sq_inner) * inv_dd;
                        let ti1 = (-b + sq_inner) * inv_dd;
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
        // A CubedSphereFace node is the root of a face subtree. Under
        // normal operation the CubedSphereBody arm calls
        // `march_face_subtree_curved` which traverses the face subtree
        // in body-local coords and never returns to the top-level
        // Cartesian DDA until a hit or body-exit. We only reach this
        // arm with a face kind when ribbon-pop lands the frame INSIDE
        // a face subtree from an ancestor frame; in that case we fall
        // through to the Cartesian slot-pick (which renders the face
        // subtree as axis-aligned cells — a known rendering
        // approximation for ribbon-pop'd face frames).
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
