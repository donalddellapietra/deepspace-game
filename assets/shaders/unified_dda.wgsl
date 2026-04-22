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

// ────────────────────────────── Slot-path + residual face walker
//
// Port of the CPU walker in `src/world/cubesphere/walker.rs`. The
// state model is precision-correct at face-subtree depth 30+: NO
// absolute f32 quantity scales as `1/3^N` anywhere. See
// `docs/principles/no-absolute-coordinates.md`.
//
// State carried per-iteration (mirrors FaceWalker in walker.rs):
//   * s_slot_u/v/r[0..depth]   — integer slot chain. THE position.
//   * s_node_idx[0..depth+1]   — node-index parallel stack.
//   * cur_us/vs/rs             — child-of-current-cell slot
//                                (explicit integer tracking; avoids
//                                the floor-vs-slot race at integer
//                                boundaries).
//   * residual_o               — ray origin in the CURRENT cell's
//                                local `[0, 3)³` frame. Rescaled by
//                                ×3 on every descent (minus the child
//                                slot). O(1) magnitude at any depth.
//   * rd_local                 — ray direction in the current cell's
//                                local frame. Multiplied by 3 per
//                                descent. f32-safe through depth
//                                ~38; only its RATIO is used in
//                                boundary tests.
//   * u_c/v_c/r_c              — face-normalized cell LOWER CORNER.
//                                Tracked for SHADING NORMAL
//                                evaluation ONLY (tan() on the
//                                cell-center UV in the face basis).
//                                NEVER used in any boundary test.
//
// Cell boundaries in the residual frame are the integer values
// `cur_us, cur_us+1, ...` — exact, no f32 ULP loss. Ray-axis-plane
// t-values `(boundary - residual_i) / rd_local_i` compute at O(1)
// precision regardless of depth.
//
// Linearization: inside a face subtree, cells are treated as flat
// parallelograms in face-normalized coords (silhouette error
// O(cell_size²) body units; sub-pixel at face-subtree depth ≥ 3).
// The outer shell is still the real curved sphere via Stage 2's
// ray-sphere entry; this walker only sees the ALREADY-ENTERED ray.
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
// Compile-time guarantee: grep the function body for
// `1.0 / pow(3.0,` or `cur_u_lo` — they are ABSENT. Cell extent in
// face-normalized units does not appear as a stored f32 state
// variable nor as an inline expression in boundary tests; only for
// smooth shading-normal evaluation (auxiliary `cur_cell_ext`, whose
// bit-loss at deep depth is insensitive per the architecture doc).
// Cell boundaries are tested in the residual frame where they are
// trivially the integer slot lo/hi. See
// `docs/architecture/sphere-unified-dda.md` §Stage 3d.
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

    // Body-centered ray in ribbon frame. Used to reconstruct hit
    // positions for the LOD gate and for the seam-cross reprojection.
    let oc = ray_origin_rib - body_center_rib;
    let rd = ray_dir_rib;

    // Tiny slice of the ray in front of us (we just landed at t_enter
    // on the outer shell, so start just inside).
    let eps_t = 1e-5 * outer_rib;
    var t_cur: f32 = t_enter_body + eps_t;

    var cur_face: u32 = start_face;

    // ── Face-subtree walk stack (slot-path + residual) ────────
    var s_node_idx: array<u32, MAX_FACE_STACK_DEPTH>;
    var s_slot_u: array<i32, MAX_FACE_STACK_DEPTH>;
    var s_slot_v: array<i32, MAX_FACE_STACK_DEPTH>;
    var s_slot_r: array<i32, MAX_FACE_STACK_DEPTH>;
    s_node_idx[0] = start_root_node_idx;

    var depth: u32 = 0u;
    var cur_us: i32 = 0;
    var cur_vs: i32 = 0;
    var cur_rs: i32 = 0;

    // Face-normalized cell LOWER CORNER (shading-normal source only).
    // At depth 0, the initial child cell's lower corner is
    // (cur_us/3, cur_vs/3, cur_rs/3). Stays in [0, 1] throughout.
    var u_c: f32 = 0.0;
    var v_c: f32 = 0.0;
    var r_c: f32 = 0.0;

    // Residual ray origin in current cell's `[0, 3)³` local frame.
    // Starts as (un, vn, rn) * 3 of the ray's body-frame entry
    // projected onto `cur_face`.
    var residual_o: vec3<f32> = vec3<f32>(0.0);
    // Ray direction in current cell's local frame (×3 per descent).
    // Seeded by a finite-difference Jacobian of
    // `body_point_to_face_space` at the entry point — same closed-form
    // as `initial_rd_local` in `src/world/cubesphere/walker.rs`.
    var rd_local: vec3<f32> = vec3<f32>(0.0);
    // Body-frame size of one residual unit in ribbon units. At depth
    // d the residual spans `[0, 3)` face-normalized → `[0, 3·outer_rib
    // / 3^(d+1))` body-units → one residual unit = `outer_rib /
    // 3^(d+1)` body-units. Tracked compounded (÷3 on descent, ×3 on
    // ascent) instead of `1.0 / pow(3.0, depth+1)` to keep ALL
    // precision-critical arithmetic free of absolute-3^N state.
    // Used by the LOD gate and hit-cell-size report — not in boundary
    // tests (those live purely in residual coords).
    var residual_to_rib: f32 = outer_rib * (1.0 / 3.0);

    var cur_header_off = node_offsets[start_root_node_idx];
    var cur_occupancy: u32 = tree[cur_header_off];
    var cur_first_child: u32 = tree[cur_header_off + 1u];

    // Surface normal of the LAST boundary the ray crossed — becomes
    // the hit normal if the NEXT cell is solid. Initialised below to
    // the entry cell's center radial on `cur_face`.
    var hit_normal: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

    // ── Seed state from the body-frame entry point ──
    // Ray position at t_cur (in ribbon frame, body-centered).
    let p_entry = oc + rd * t_cur;
    let r_entry = sqrt(max(dot(p_entry, p_entry), 1e-30));
    let rn_entry = clamp(
        (r_entry - inner_rib) / max(outer_rib - inner_rib, 1e-10),
        0.0, 1.0,
    );
    let basis0 = face_basis(cur_face);
    {
        let n_comp0 = dot(basis0.n_axis, p_entry);
        let inv_nc0 = 1.0 / max(abs(n_comp0), 1e-6);
        let cu0 = dot(basis0.u_axis, p_entry) * inv_nc0 * sign_or_one(n_comp0);
        let cv0 = dot(basis0.v_axis, p_entry) * inv_nc0 * sign_or_one(n_comp0);
        let un0 = clamp(0.5 * (atan(cu0) * (4.0 / 3.14159265) + 1.0), 0.0, 1.0);
        let vn0 = clamp(0.5 * (atan(cv0) * (4.0 / 3.14159265) + 1.0), 0.0, 1.0);

        // Residual in face-subtree root's [0, 3)³ frame.
        residual_o = vec3<f32>(
            clamp(un0 * 3.0, 0.0, 3.0 - 1.0e-5),
            clamp(vn0 * 3.0, 0.0, 3.0 - 1.0e-5),
            clamp(rn_entry * 3.0, 0.0, 3.0 - 1.0e-5),
        );
        // Initial slot = floor(residual), clamped to [0, 2].
        cur_us = clamp(i32(floor(residual_o.x)), 0, 2);
        cur_vs = clamp(i32(floor(residual_o.y)), 0, 2);
        cur_rs = clamp(i32(floor(residual_o.z)), 0, 2);
        // u_c / v_c / r_c = current NODE's lower corner. At depth 0 the
        // node is the face-subtree root, which has lower corner 0 in
        // face-normalized coords.
        u_c = 0.0;
        v_c = 0.0;
        r_c = 0.0;
    }

    // Seed rd_local via finite-difference Jacobian of
    // `body_point_to_face_space` at p_entry. Matches
    // `initial_rd_local` in walker.rs.
    {
        let h_step: f32 = 1.0e-3;
        let len_rd = max(length(rd), 1e-20);
        let u_dir = rd / len_rd;
        let p_plus = p_entry + h_step * u_dir;
        // Project p_plus into face space on the same face we entered.
        // body_point_to_face_space expects a body-local point (offset
        // from the half-body-size); we pass (p_plus + body_center)
        // translated consistently — but the face-space math only uses
        // the centered vector, so operating on p_plus directly vs.
        // cur_face basis gives the same (un, vn, rn).
        let r_plus = sqrt(max(dot(p_plus, p_plus), 1e-30));
        let rn_plus = (r_plus - inner_rib) / max(outer_rib - inner_rib, 1e-10);
        // Project onto cur_face (ignore face re-pick — this is a
        // 1e-3-body-unit step, tiny vs face size).
        let nc_plus = dot(basis0.n_axis, p_plus);
        let inv_nc_plus = 1.0 / max(abs(nc_plus), 1e-6);
        let cu_plus = dot(basis0.u_axis, p_plus) * inv_nc_plus * sign_or_one(nc_plus);
        let cv_plus = dot(basis0.v_axis, p_plus) * inv_nc_plus * sign_or_one(nc_plus);
        let un_plus = 0.5 * (atan(cu_plus) * (4.0 / 3.14159265) + 1.0);
        let vn_plus = 0.5 * (atan(cv_plus) * (4.0 / 3.14159265) + 1.0);

        let un_entry = residual_o.x * (1.0 / 3.0);
        let vn_entry = residual_o.y * (1.0 / 3.0);
        let rn_entry2 = residual_o.z * (1.0 / 3.0);

        let delta_un = un_plus - un_entry;
        let delta_vn = vn_plus - vn_entry;
        let delta_rn = rn_plus - rn_entry2;

        // rd_local in face-normalized [0, 3) coords per unit body-t.
        // Scale by len_rd / h_step so t parameter stays in body-t.
        let scale_init = len_rd / h_step;
        rd_local = vec3<f32>(
            delta_un * 3.0 * scale_init,
            delta_vn * 3.0 * scale_init,
            delta_rn * 3.0 * scale_init,
        );
    }

    // Seed hit_normal to the ENTRY cell's center radial on cur_face.
    // The ray just crossed the outer r-shell; the first cell's r-face
    // is what we're "viewing" if that cell turns out to be solid.
    // `u_c / v_c` are the face-subtree ROOT's lower corner (0,0); the
    // entry cell is at cur_us/vs within that root, so its center is
    // `f32(cur_us + 0.5) × 1/3`.
    {
        let u_mid_init = u_c + (f32(cur_us) + 0.5) * (1.0 / 3.0);
        let v_mid_init = v_c + (f32(cur_vs) + 0.5) * (1.0 / 3.0);
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

        // ── Boundary computation in the RESIDUAL frame ──
        //
        // The current cell occupies the integer sub-box
        // `[cur_us..cur_us+1] × [cur_vs..cur_vs+1] × [cur_rs..cur_rs+1]`
        // within the residual `[0, 3)³` frame. Cell boundaries are
        // EXACT integers; ray-plane t is `(boundary - residual_i) /
        // rd_local_i`, numerically stable at ALL depths.
        let cell_lo = vec3<f32>(f32(cur_us), f32(cur_vs), f32(cur_rs));
        let cell_hi = cell_lo + vec3<f32>(1.0);

        // ---- Per-cell Nyquist LOD gate --------------------------------
        //
        // Cell's body-frame extent in ribbon units: one residual unit
        // at current depth corresponds to `residual_to_rib` body
        // units (tracked compounded, no absolute-3^N arithmetic). We
        // compare projected pixel size against the Nyquist floor.
        let cell_rib_size_lod = residual_to_rib;
        {
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
                    result.cell_size = cell_rib_size_lod;
                    return result;
                }
                // Empty sub-pixel cell: fall through to ordinary DDA
                // advance.
            }
        }

        // ---- Boundary t computation in residual frame.
        //
        // For each axis: boundary_pos = rd_local_i > 0 ? cell_hi_i :
        // cell_lo_i; t_axis = (boundary_pos - residual_o_i) /
        // rd_local_i. Pick minimum positive.
        //
        // Use a tiny eps_t guard on the RESULT to avoid re-selecting
        // the boundary we just arrived on after an advance.
        let inv_rd = vec3<f32>(
            select(1e30, 1.0 / rd_local.x, abs(rd_local.x) > 1.0e-30),
            select(1e30, 1.0 / rd_local.y, abs(rd_local.y) > 1.0e-30),
            select(1e30, 1.0 / rd_local.z, abs(rd_local.z) > 1.0e-30),
        );
        let boundary_pos = vec3<f32>(
            select(cell_lo.x, cell_hi.x, rd_local.x > 0.0),
            select(cell_lo.y, cell_hi.y, rd_local.y > 0.0),
            select(cell_lo.z, cell_hi.z, rd_local.z > 0.0),
        );
        let t_axis = (boundary_pos - residual_o) * inv_rd;
        // A "parallel to axis" rd_local component yields t_axis ≈
        // inv_rd * (boundary - residual) = 1e30 * small-finite; mask
        // such components to BIG.
        let BIG: f32 = 1.0e30;
        let t_u = select(BIG, t_axis.x, abs(rd_local.x) > 1.0e-30 && t_axis.x >= 0.0);
        let t_v = select(BIG, t_axis.y, abs(rd_local.y) > 1.0e-30 && t_axis.y >= 0.0);
        let t_r = select(BIG, t_axis.z, abs(rd_local.z) > 1.0e-30 && t_axis.z >= 0.0);

        var t_local_step: f32 = t_u;
        var exit_axis: u32 = 0u;
        if t_v < t_local_step { t_local_step = t_v; exit_axis = 1u; }
        if t_r < t_local_step { t_local_step = t_r; exit_axis = 2u; }

        if t_local_step >= BIG {
            // Ray parallel to all three axes — degenerate; bail as
            // body-exit.
            result.hit = false;
            result.cell_min = oc + rd * t_cur + body_center_rib;
            result.cell_size = 1.0;
            return result;
        }

        // Exit direction: +1 if rd_local along exit axis > 0, else -1.
        var exit_positive: bool = false;
        if exit_axis == 0u { exit_positive = rd_local.x > 0.0; }
        else if exit_axis == 1u { exit_positive = rd_local.y > 0.0; }
        else { exit_positive = rd_local.z > 0.0; }

        // ---- Per-cell tree lookup (same as before).
        let slot = u32(cur_rs * 9 + cur_vs * 3 + cur_us);
        let slot_bit = 1u << slot;
        if (cur_occupancy & slot_bit) != 0u {
            let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
            let child_base = cur_first_child + rank * 2u;
            let packed = tree[child_base];
            let tag = packed & 0xFFu;

            if tag == 1u {
                result.hit = true;
                result.t = t_cur;
                result.color = palette[(packed >> 8u) & 0xFFFFu].rgb;
                result.normal = hit_normal;
                let hit_pos = oc + rd * t_cur;
                result.cell_min = hit_pos + body_center_rib;
                result.cell_size = cell_rib_size_lod;
                return result;
            }

            if tag == 2u {
                let child_idx = tree[child_base + 1u];
                let child_bt = child_block_type(packed);
                // Child's projected size for LOD check.
                let child_cell_rib = residual_to_rib * (1.0 / 3.0);
                let ray_dist = max(t_cur * ray_metric_rib, 0.001);
                let lod_pixels = child_cell_rib / ray_dist
                    * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
                let at_max = depth + 1u >= MAX_FACE_STACK_DEPTH;
                let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;

                if !(at_max || at_lod) && child_bt != 0xFFFEu {
                    // ── Descend. Mirror walker.rs::descend ──
                    // Push current slot + node, bump depth, refresh
                    // the node header cache.
                    s_slot_u[depth] = cur_us;
                    s_slot_v[depth] = cur_vs;
                    s_slot_r[depth] = cur_rs;
                    depth += 1u;
                    s_node_idx[depth] = child_idx;
                    cur_header_off = node_offsets[child_idx];
                    cur_occupancy = tree[cur_header_off];
                    cur_first_child = tree[cur_header_off + 1u];

                    // Rescale residual into child cell's [0, 3)³.
                    residual_o = (residual_o - cell_lo) * 3.0;
                    rd_local = rd_local * 3.0;
                    residual_to_rib = residual_to_rib * (1.0 / 3.0);

                    // New slot within the child cell.
                    cur_us = clamp(i32(floor(residual_o.x)), 0, 2);
                    cur_vs = clamp(i32(floor(residual_o.y)), 0, 2);
                    cur_rs = clamp(i32(floor(residual_o.z)), 0, 2);

                    // Update cell LOWER CORNER tracker from the full
                    // integer slot stack. `ext_acc` stays in f32 but
                    // the accumulation multiplies small integers by
                    // a shrinking power of 1/3 — the result stays
                    // bounded in [0, 1]. At deep depth its low bits
                    // fall below f32 ULP; harmless because u_c/v_c/
                    // r_c are used only for the smooth tan() shading-
                    // normal evaluation, never in boundary tests.
                    var u_acc: f32 = 0.0;
                    var v_acc: f32 = 0.0;
                    var r_acc: f32 = 0.0;
                    var ext_acc: f32 = 1.0 / 3.0;
                    for (var d: u32 = 0u; d < depth; d = d + 1u) {
                        u_acc += f32(s_slot_u[d]) * ext_acc;
                        v_acc += f32(s_slot_v[d]) * ext_acc;
                        r_acc += f32(s_slot_r[d]) * ext_acc;
                        ext_acc = ext_acc * (1.0 / 3.0);
                    }
                    u_c = u_acc;
                    v_c = v_acc;
                    r_c = r_acc;
                    continue;
                } else if !(at_max || at_lod) && child_bt == 0xFFFEu {
                    // Representative-empty: treat as empty.
                } else {
                    // LOD terminal splat.
                    let bt = child_bt;
                    if !(bt == 0xFFFEu || bt == 0xFFFDu) {
                        result.hit = true;
                        result.t = t_cur;
                        result.color = palette[bt].rgb;
                        result.normal = hit_normal;
                        let hit_pos = oc + rd * t_cur;
                        result.cell_min = hit_pos + body_center_rib;
                        result.cell_size = cell_rib_size_lod;
                        return result;
                    }
                }
            }
            // tag 0 or fall-through: advance.
        }

        // ── Advance to cell exit ──
        //
        // Update hit_normal to the boundary we just crossed. The
        // shading-normal evaluation uses the Jacobian basis at
        // (u_c, v_c, r_c) — the CURRENT cell's lower corner — so
        // normals stay well-conditioned at any depth.
        //
        // Stage 4 faceted voxel shading: r-shell normal is evaluated
        // at the CELL-CENTER UV (one direction per cell → flat
        // facet). u- and v-plane normals use the corresponding
        // `tan()` on the exit-side u or v boundary — one direction
        // per cell edge.
        {
            // Derive the local boundary normal in body-frame via the
            // face basis at (u_c, v_c, r_c). We need:
            //   * u-face: normal = cross(v_axis, n + cube_u(u_edge) ·
            //     u_axis) — this is the gradient of the equal-angle
            //     u = const plane on the cube face.
            //   * v-face: symmetric.
            //   * r-face: radial at the cell-center UV.
            //
            // Cell extent in face-normalized coords is
            // `residual_to_rib / outer_rib` — NOT computed as
            // `1.0 / pow(3.0, depth+1)`. The compounded ÷3 on
            // descent keeps it free of absolute-3^N arithmetic.
            let cur_cell_ext = residual_to_rib / max(outer_rib, 1e-10);
            // u_c / v_c / r_c track the CURRENT NODE'S lower corner;
            // the ray's active cell is the cur_slot child within that
            // node (cell extent = cur_cell_ext, lower corner at
            // `u_c + cur_slot × cur_cell_ext`).
            let cell_lo_u = u_c + f32(cur_us) * cur_cell_ext;
            let cell_lo_v = v_c + f32(cur_vs) * cur_cell_ext;
            let u_center = cell_lo_u + cur_cell_ext * 0.5;
            let v_center = cell_lo_v + cur_cell_ext * 0.5;
            let u_edge = select(cell_lo_u, cell_lo_u + cur_cell_ext, exit_positive);
            let v_edge = select(cell_lo_v, cell_lo_v + cur_cell_ext, exit_positive);
            let basis = face_basis(cur_face);
            var raw_n: vec3<f32>;
            if exit_axis == 0u {
                // u-plane at `u_edge`.
                let cu_edge = tan((2.0 * u_edge - 1.0) * 0.78539816);
                raw_n = cross(basis.v_axis, basis.n_axis + cu_edge * basis.u_axis);
            } else if exit_axis == 1u {
                // v-plane at `v_edge`.
                let cv_edge = tan((2.0 * v_edge - 1.0) * 0.78539816);
                raw_n = cross(basis.n_axis + cv_edge * basis.v_axis, basis.u_axis);
            } else {
                // r-shell: radial at cell-center UV.
                let cu_mid = tan((2.0 * u_center - 1.0) * 0.78539816);
                let cv_mid = tan((2.0 * v_center - 1.0) * 0.78539816);
                raw_n = basis.n_axis + cu_mid * basis.u_axis + cv_mid * basis.v_axis;
            }
            let rn_len = max(length(raw_n), 1e-6);
            var n_unit = raw_n / rn_len;
            if dot(n_unit, rd) > 0.0 { n_unit = -n_unit; }
            hit_normal = n_unit;
        }

        // Advance residual to the boundary (snap the exit-axis
        // component exactly to the boundary integer to avoid
        // floor-vs-slot drift).
        residual_o = residual_o + rd_local * t_local_step;
        if exit_axis == 0u {
            residual_o.x = select(cell_lo.x, cell_hi.x, exit_positive);
            cur_us = cur_us + select(-1, 1, exit_positive);
        } else if exit_axis == 1u {
            residual_o.y = select(cell_lo.y, cell_hi.y, exit_positive);
            cur_vs = cur_vs + select(-1, 1, exit_positive);
        } else {
            residual_o.z = select(cell_lo.z, cell_hi.z, exit_positive);
            cur_rs = cur_rs + select(-1, 1, exit_positive);
        }
        // Local-t equals body-t by construction (see comment near
        // rd_local seeding: rd_local was scaled to give face-normalized
        // deltas per unit body-t, and subsequent ×3 descents preserve
        // that parameterization — see walker.rs::descend for the
        // algebra). Advance ribbon-frame t_cur by t_local_step.
        t_cur = t_cur + t_local_step;

        // ── Bubble up on OOB ──
        loop {
            if cur_us >= 0 && cur_us <= 2
                && cur_vs >= 0 && cur_vs <= 2
                && cur_rs >= 0 && cur_rs <= 2 {
                break;
            }
            if depth == 0u {
                let is_r_exit = (cur_rs < 0) || (cur_rs > 2);
                if is_r_exit {
                    result.hit = false;
                    result.cell_min = oc + rd * t_cur + body_center_rib;
                    if cur_rs < 0 { result.cell_size = 0.0; } else { result.cell_size = 1.0; }
                    return result;
                }
                // UV seam cross.
                seam_count += 1u;
                if seam_count >= max_seams {
                    result.hit = false;
                    result.cell_min = oc + rd * t_cur + body_center_rib;
                    result.cell_size = 1.0;
                    return result;
                }
                let p_exit = oc + rd * t_cur;
                let rex = max(length(p_exit), 1e-6);
                let basis_exit = face_basis(cur_face);
                var target_normal: vec3<f32>;
                if cur_us < 0      { target_normal = -basis_exit.u_axis; }
                else if cur_us > 2 { target_normal =  basis_exit.u_axis; }
                else if cur_vs < 0 { target_normal = -basis_exit.v_axis; }
                else               { target_normal =  basis_exit.v_axis; }
                var new_face: u32 = cur_face;
                if target_normal.x >  0.5 { new_face = 0u; }
                else if target_normal.x < -0.5 { new_face = 1u; }
                else if target_normal.y >  0.5 { new_face = 2u; }
                else if target_normal.y < -0.5 { new_face = 3u; }
                else if target_normal.z >  0.5 { new_face = 4u; }
                else                           { new_face = 5u; }
                if new_face == cur_face {
                    result.hit = false;
                    result.cell_min = p_exit + body_center_rib;
                    result.cell_size = 1.0;
                    return result;
                }
                cur_face = new_face;
                let face_slot = FACE_SLOTS[cur_face];
                let face_bit = 1u << face_slot;
                if (body_occupancy & face_bit) == 0u {
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
                // Reset walker to the new face's subtree root.
                depth = 0u;
                s_node_idx[0] = new_root_idx;
                cur_header_off = node_offsets[new_root_idx];
                cur_occupancy = tree[cur_header_off];
                cur_first_child = tree[cur_header_off + 1u];
                residual_to_rib = outer_rib * (1.0 / 3.0);

                // Re-project the exit point onto the new face to seed
                // the residual. This stops linearization drift from
                // compounding across seams.
                let rnex = clamp(
                    (rex - inner_rib) / max(outer_rib - inner_rib, 1e-10),
                    0.0, 1.0,
                );
                let bn = face_basis(cur_face);
                let nc = dot(bn.n_axis, p_exit);
                let inv_nc = 1.0 / max(abs(nc), 1e-6);
                let cu = dot(bn.u_axis, p_exit) * inv_nc * sign_or_one(nc);
                let cv = dot(bn.v_axis, p_exit) * inv_nc * sign_or_one(nc);
                let un_ = clamp(0.5 * (atan(cu) * (4.0 / 3.14159265) + 1.0), 0.0, 1.0);
                let vn_ = clamp(0.5 * (atan(cv) * (4.0 / 3.14159265) + 1.0), 0.0, 1.0);

                residual_o = vec3<f32>(
                    clamp(un_ * 3.0, 0.0, 3.0 - 1.0e-5),
                    clamp(vn_ * 3.0, 0.0, 3.0 - 1.0e-5),
                    clamp(rnex * 3.0, 0.0, 3.0 - 1.0e-5),
                );
                cur_us = clamp(i32(floor(residual_o.x)), 0, 2);
                cur_vs = clamp(i32(floor(residual_o.y)), 0, 2);
                cur_rs = clamp(i32(floor(residual_o.z)), 0, 2);
                // At depth 0 on new face, node is the face-subtree root
                // with lower corner 0.
                u_c = 0.0;
                v_c = 0.0;
                r_c = 0.0;

                // Re-seed rd_local via finite-difference on the new
                // face (same algebra as initial seed). This stops
                // seam drift from accumulating.
                {
                    let h_step: f32 = 1.0e-3;
                    let len_rd = max(length(rd), 1e-20);
                    let u_dir = rd / len_rd;
                    let p_plus = p_exit + h_step * u_dir;
                    let r_plus = sqrt(max(dot(p_plus, p_plus), 1e-30));
                    let rn_plus = (r_plus - inner_rib) / max(outer_rib - inner_rib, 1e-10);
                    let nc_plus = dot(bn.n_axis, p_plus);
                    let inv_nc_plus = 1.0 / max(abs(nc_plus), 1e-6);
                    let cu_plus = dot(bn.u_axis, p_plus) * inv_nc_plus * sign_or_one(nc_plus);
                    let cv_plus = dot(bn.v_axis, p_plus) * inv_nc_plus * sign_or_one(nc_plus);
                    let un_plus = 0.5 * (atan(cu_plus) * (4.0 / 3.14159265) + 1.0);
                    let vn_plus = 0.5 * (atan(cv_plus) * (4.0 / 3.14159265) + 1.0);

                    let un_entry = residual_o.x * (1.0 / 3.0);
                    let vn_entry = residual_o.y * (1.0 / 3.0);
                    let rn_entry2 = residual_o.z * (1.0 / 3.0);
                    let scale_s = len_rd / h_step;
                    rd_local = vec3<f32>(
                        (un_plus - un_entry) * 3.0 * scale_s,
                        (vn_plus - vn_entry) * 3.0 * scale_s,
                        (rn_plus - rn_entry2) * 3.0 * scale_s,
                    );
                }
                break;
            }

            // ── Ascend one level. Mirror walker.rs::ascend ──
            let child_us = s_slot_u[depth - 1u];
            let child_vs = s_slot_v[depth - 1u];
            let child_rs = s_slot_r[depth - 1u];
            depth -= 1u;

            // Undo the descent rescale: residual / 3 + slot, rd_local / 3.
            rd_local = rd_local * (1.0 / 3.0);
            residual_o = residual_o * (1.0 / 3.0)
                + vec3<f32>(f32(child_us), f32(child_vs), f32(child_rs));
            residual_to_rib = residual_to_rib * 3.0;

            // Refresh node header cache.
            cur_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[cur_header_off];
            cur_first_child = tree[cur_header_off + 1u];

            // Parent's cur_slot: whatever slot in the parent now
            // contains residual_o after the child's OOB bubbled up.
            if cur_us < 0 { cur_us = child_us - 1; cur_vs = child_vs; cur_rs = child_rs; }
            else if cur_us > 2 { cur_us = child_us + 1; cur_vs = child_vs; cur_rs = child_rs; }
            else if cur_vs < 0 { cur_vs = child_vs - 1; cur_us = child_us; cur_rs = child_rs; }
            else if cur_vs > 2 { cur_vs = child_vs + 1; cur_us = child_us; cur_rs = child_rs; }
            else if cur_rs < 0 { cur_rs = child_rs - 1; cur_us = child_us; cur_vs = child_vs; }
            else { cur_rs = child_rs + 1; cur_us = child_us; cur_vs = child_vs; }

            // Re-derive u_c/v_c/r_c from the (now-popped) slot stack.
            var u_acc: f32 = 0.0;
            var v_acc: f32 = 0.0;
            var r_acc: f32 = 0.0;
            var ext_acc: f32 = 1.0 / 3.0;
            for (var d: u32 = 0u; d < depth; d = d + 1u) {
                u_acc += f32(s_slot_u[d]) * ext_acc;
                v_acc += f32(s_slot_v[d]) * ext_acc;
                r_acc += f32(s_slot_r[d]) * ext_acc;
                ext_acc = ext_acc * (1.0 / 3.0);
            }
            u_c = u_acc;
            v_c = v_acc;
            r_c = r_acc;
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
