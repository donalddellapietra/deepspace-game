#include "bindings.wgsl"
#include "tree.wgsl"
#include "face_math.wgsl"
#include "ray_prim.wgsl"

// Cubed-sphere voxel walker.
//
// Renders a `NodeKind::CubedSphereBody` cell as a smooth sphere whose
// surface voxels live in 6 u/v/r-indexed face subtrees. The walker is:
//
//   1. PRECISION-SAFE — ray is transformed into body-local coords once
//      upfront (body spans `[0, 3)³`, center at `(1.5)³`). All sphere
//      math (ray-sphere, ray-plane, length, normalize) runs with
//      numbers bounded by the body's local extent, so precision never
//      degrades with the caller's frame scale. No `oc`-magnitude-
//      catastrophic-cancellation, no ULP wall at deep anchor.
//
//   2. CROSS-FACE SEAMLESS — each inner-loop iteration re-derives the
//      face from the ray's current position via `pick_face(n)`. A ray
//      that crosses a cube edge automatically transitions into the
//      neighbor face's subtree on the next iteration. No per-face
//      walkers, no edge-handoff table.
//
//   3. PER-RAY LOD — descent depth is capped by the same pixel-density
//      Nyquist gate the Cartesian walker uses, so the same physical
//      view resolves to the same terminal cells regardless of anchor
//      depth.
//
//   4. HIT_PATH POPULATED — on hit, `result.hit_path` carries
//      `[face_slot, u/v/r descent slots]` so the fragment shader's
//      path-prefix highlight match works at any anchor.

// ─────────────────────────────────────────────────────────── shading

fn face_uv_for_normal(local: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let an = abs(normal);
    if an.x >= an.y && an.x >= an.z { return local.yz; }
    if an.y >= an.z { return local.xz; }
    return local.xy;
}

fn cube_face_bevel(local: vec3<f32>, normal: vec3<f32>) -> f32 {
    let uv = face_uv_for_normal(local, normal);
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return smoothstep(0.02, 0.14, edge);
}

fn sphere_shade(block_id: u32, hit_normal: vec3<f32>) -> vec3<f32> {
    let cell_color = palette.colors[block_id].rgb;
    let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
    let diffuse = max(dot(hit_normal, sun_dir), 0.0);
    let axis_tint = abs(hit_normal.y) * 1.0
                  + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
    let ambient = 0.22;
    return cell_color * (ambient + diffuse * 0.78) * axis_tint;
}

// ──────────────────────────────────────────────────────────────── LOD

// Per-ray face-subtree depth cap. Returns the deepest depth at which
// a cell's physical extent (= `shell * (1/3)^(d-1)` in body-local
// units) still projects to at least `LOD_PIXEL_THRESHOLD` pixels on
// screen. `t_local` is the ray's body-local t at the current cell.
fn face_lod_depth_cap(t_local: f32, shell: f32) -> u32 {
    let pixel_density = uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
    let safe_t = max(t_local, 1e-6);
    let ratio = shell * pixel_density
        / (safe_t * max(LOD_PIXEL_THRESHOLD, 1e-6));
    if ratio <= 1.0 { return 1u; }
    // log3(x) = log2(x) / log2(3)
    let log3_ratio = log2(ratio) * (1.0 / 1.5849625);
    let d_f = 1.0 + log3_ratio;
    return u32(clamp(d_f, 1.0, f32(MAX_FACE_DEPTH)));
}

// ─────────────────────────────────────── face-subtree descent (inline)

// Result of walking the body's face-slot subtree for a point at
// (face, un, vn, rn). `block == 0` means empty (no hit); bounds are
// always filled in (used for next-cell boundary computation).
struct FaceDescent {
    block: u32,
    depth: u32,     // 0 = body's face slot itself is the terminal
    u_lo: f32,      // cell's lo bound in normalized face coords [0, 1]
    v_lo: f32,
    r_lo: f32,
    size: f32,      // cell side length = 3^-depth
    // Slot sequence from face root down to terminal. Index i = slot
    // at descent depth (i + 1). Only the first `depth` entries are
    // valid. Used to populate `hit_path` on a HIT.
    slots: array<u32, 32>,
}

// Walk the body's face-slot subtree toward (un, vn, rn). Returns the
// terminal cell reached: either a Block (hit), a uniform Empty, or a
// LOD-terminal Node at `depth_limit`.
fn descend_face(
    body_node_idx: u32,
    face: u32,
    un_in: f32, vn_in: f32, rn_in: f32,
    depth_limit: u32,
) -> FaceDescent {
    var result: FaceDescent;
    result.block = 0u;
    result.depth = 0u;
    result.u_lo = 0.0;
    result.v_lo = 0.0;
    result.r_lo = 0.0;
    result.size = 1.0;

    // Body's face-slot child.
    let fs = face_slot(face);
    let body_header_off = node_offsets[body_node_idx];
    let body_occupancy = tree[body_header_off];
    let body_first_child = tree[body_header_off + 1u];
    let body_bit = 1u << fs;
    if (body_occupancy & body_bit) == 0u {
        // Face slot empty — no voxels on this face. Cell covers the
        // full face. Advance t past the face in caller.
        return result;
    }
    let body_rank = countOneBits(body_occupancy & (body_bit - 1u));
    let body_child_base = body_first_child + body_rank * 2u;
    let face_packed = tree[body_child_base];
    let face_tag = face_packed & 0xFFu;
    if face_tag == 1u {
        // Whole face is a uniform Block.
        result.block = (face_packed >> 8u) & 0xFFu;
        return result;
    }

    // Descend face subtree.
    var node = tree[body_child_base + 1u];
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);

    var u_sum: f32 = 0.0;
    var v_sum: f32 = 0.0;
    var r_sum: f32 = 0.0;
    var cell_size: f32 = 1.0;

    let limit = min(depth_limit, MAX_FACE_DEPTH);
    if limit < 1u {
        // Can't descend — return face root's representative.
        let bt = (face_packed >> 8u) & 0xFFu;
        result.block = select(0u, bt, bt != 255u);
        return result;
    }

    for (var d: u32 = 1u; d <= limit; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let slot = rs * 9u + vs * 3u + us;
        result.slots[d - 1u] = slot;

        let header_off = node_offsets[node];
        let occ = tree[header_off];
        let fc = tree[header_off + 1u];
        let bit = 1u << slot;
        let occupied = (occ & bit) != 0u;
        let rank = countOneBits(occ & (bit - 1u));
        let child_base = fc + rank * 2u;
        let packed = select(0u, tree[child_base], occupied);
        let tag = packed & 0xFFu;

        // Accumulate cell bounds BEFORE potentially returning so the
        // bounds represent the child we're about to enter (or
        // terminate in).
        let step_size = cell_size * (1.0 / 3.0);
        u_sum = u_sum + step_size * f32(us);
        v_sum = v_sum + step_size * f32(vs);
        r_sum = r_sum + step_size * f32(rs);
        cell_size = step_size;

        if tag == 0u {
            // Empty cell.
            result.block = 0u;
            result.depth = d;
            result.u_lo = u_sum;
            result.v_lo = v_sum;
            result.r_lo = r_sum;
            result.size = cell_size;
            return result;
        }
        if tag == 1u {
            // Block hit.
            result.block = (packed >> 8u) & 0xFFu;
            result.depth = d;
            result.u_lo = u_sum;
            result.v_lo = v_sum;
            result.r_lo = r_sum;
            result.size = cell_size;
            return result;
        }
        // tag == 2u: Node child. Descend unless we've hit the limit.
        if d >= limit {
            // LOD-terminal: return the child's representative block.
            let bt = (packed >> 8u) & 0xFFu;
            result.block = select(0u, bt, bt != 255u);
            result.depth = d;
            result.u_lo = u_sum;
            result.v_lo = v_sum;
            result.r_lo = r_sum;
            result.size = cell_size;
            return result;
        }
        node = tree[child_base + 1u];
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
    }

    // Unreachable: loop always returns. Defensive default.
    result.block = 0u;
    result.depth = limit;
    result.u_lo = u_sum;
    result.v_lo = v_sum;
    result.r_lo = r_sum;
    result.size = cell_size;
    return result;
}

// ──────────────────────────────────────────────────────── main walker

fn sphere_in_cell(
    body_node_idx: u32,
    body_cell_origin: vec3<f32>,   // in render frame
    body_cell_size: f32,             // in render frame
    inner_r_local: f32,              // [0, 1] in body-cell-local units
    outer_r_local: f32,
    ray_origin: vec3<f32>,           // in render frame
    ray_dir: vec3<f32>,              // in render frame (unit vector)
    walker_max_depth: u32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    // Transform ray into body-local coords (body spans [0, 3)³, center
    // at (1.5, 1.5, 1.5) exactly). Direction is a unit vector so it
    // doesn't scale; only origin translates + scales. The t parameter
    // in body-local units maps to render-frame units via:
    //   t_render = t_local / scale   (scale = 3.0 / body_cell_size)
    //
    // All subsequent sphere math (ray-sphere, ray-plane, length,
    // normalize) runs with numbers bounded by the body's local extent
    // (<= 3.0). No ULP wall at deep anchor.
    let scale = 3.0 / body_cell_size;
    let bl_origin = (ray_origin - body_cell_origin) * scale;
    let bl_dir = ray_dir;
    let inv_scale = 1.0 / scale;  // = body_cell_size / 3.0

    let center = vec3<f32>(1.5);
    let cs_outer = outer_r_local * 3.0;
    let cs_inner = inner_r_local * 3.0;
    let shell = cs_outer - cs_inner;

    // Ray-sphere intersect against outer shell (unit direction → a = 1).
    let oc = bl_origin - center;
    let b = dot(oc, bl_dir);
    let c_quad = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_quad;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter_local = max(-b - sq, 0.0);
    let t_exit_local = -b + sq;
    if t_exit_local <= 0.0 { return result; }

    let eps_init = max(shell * 1e-5, 1e-7);
    var t_local = t_enter_local + eps_init;
    var steps: u32 = 0u;
    // 0..5 = last-crossed boundary axis (for hit normal); 6 = sphere entry.
    var last_axis: u32 = 6u;

    loop {
        if t_local >= t_exit_local || steps > 4096u { break; }
        steps = steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let local = oc + bl_dir * t_local;
        let r = length(local);
        if r >= cs_outer || r < cs_inner { break; }

        // Derive current face + cubed-sphere coords from ray position.
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

        // Per-ray LOD gate: walker depth capped by pixel-density.
        let lod_cap = face_lod_depth_cap(t_local, shell);
        let walk_limit = min(min(walker_max_depth, MAX_FACE_DEPTH), lod_cap);
        let walk = descend_face(body_node_idx, face, un, vn, rn, walk_limit);

        if walk.block != 0u {
            // HIT.
            var hit_normal: vec3<f32>;
            switch last_axis {
                case 0u: { hit_normal = -u_axis; }
                case 1u: { hit_normal =  u_axis; }
                case 2u: { hit_normal = -v_axis; }
                case 3u: { hit_normal =  v_axis; }
                case 4u: { hit_normal = -n; }
                case 5u: { hit_normal =  n; }
                default: { hit_normal =  n; }
            }
            result.hit = true;
            result.t = t_local * inv_scale;  // back to render-frame t
            result.normal = hit_normal;
            result.color = sphere_shade(walk.block, hit_normal);
            // Cell bounds in render frame. We use the hit position +/-
            // the walker cell's body-local size for a stable box that
            // main.wgsl's bevel calc can clamp against.
            let hit_pos_render = ray_origin + ray_dir * result.t;
            let cell_local_size = shell * walk.size;
            let cell_render_size = cell_local_size * inv_scale;
            result.cell_min = hit_pos_render - vec3<f32>(cell_render_size * 0.5);
            result.cell_size = cell_render_size;
            // hit_path: [face_slot, walker's descent slots]
            pack_slot_into_path(&result.hit_path, 0u, face_slot(face));
            for (var i: u32 = 0u; i < walk.depth; i = i + 1u) {
                pack_slot_into_path(&result.hit_path, i + 1u, walk.slots[i]);
            }
            result.hit_path_depth = walk.depth + 1u;
            return result;
        }

        // Empty cell — advance to next boundary crossing of the
        // walker's cell. Iso-u and iso-v planes pass through the body
        // center (origin in body-local oc-frame); r-shells are spheres
        // around the same center.
        let cell_lo = walk.size;  // cell side length in normalized coords
        let u_lo_norm = walk.u_lo;
        let u_hi_norm = walk.u_lo + cell_lo;
        let v_lo_norm = walk.v_lo;
        let v_hi_norm = walk.v_lo + cell_lo;
        let r_lo_norm = walk.r_lo;
        let r_hi_norm = walk.r_lo + cell_lo;

        let u_lo_ea = u_lo_norm * 2.0 - 1.0;
        let u_hi_ea = u_hi_norm * 2.0 - 1.0;
        let n_u_lo = u_axis - ea_to_cube(u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(u_hi_ea) * n_axis;

        let v_lo_ea = v_lo_norm * 2.0 - 1.0;
        let v_hi_ea = v_hi_norm * 2.0 - 1.0;
        let n_v_lo = v_axis - ea_to_cube(v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(v_hi_ea) * n_axis;

        let r_lo = cs_inner + r_lo_norm * shell;
        let r_hi = cs_inner + r_hi_norm * shell;

        var t_next = t_exit_local + 1.0;
        var next_axis: u32 = 6u;
        let zero3 = vec3<f32>(0.0);
        let cand_u_lo = ray_plane_t(oc, bl_dir, zero3, n_u_lo);
        if cand_u_lo > t_local && cand_u_lo < t_next {
            t_next = cand_u_lo; next_axis = 0u;
        }
        let cand_u_hi = ray_plane_t(oc, bl_dir, zero3, n_u_hi);
        if cand_u_hi > t_local && cand_u_hi < t_next {
            t_next = cand_u_hi; next_axis = 1u;
        }
        let cand_v_lo = ray_plane_t(oc, bl_dir, zero3, n_v_lo);
        if cand_v_lo > t_local && cand_v_lo < t_next {
            t_next = cand_v_lo; next_axis = 2u;
        }
        let cand_v_hi = ray_plane_t(oc, bl_dir, zero3, n_v_hi);
        if cand_v_hi > t_local && cand_v_hi < t_next {
            t_next = cand_v_hi; next_axis = 3u;
        }
        let cand_r_lo = ray_sphere_after(oc, bl_dir, zero3, r_lo, t_local);
        if cand_r_lo > t_local && cand_r_lo < t_next {
            t_next = cand_r_lo; next_axis = 4u;
        }
        let cand_r_hi = ray_sphere_after(oc, bl_dir, zero3, r_hi, t_local);
        if cand_r_hi > t_local && cand_r_hi < t_next {
            t_next = cand_r_hi; next_axis = 5u;
        }

        if t_next >= t_exit_local { break; }
        last_axis = next_axis;
        // Boundary-crossing epsilon: scaled by cell size so tiny cells
        // at the deepest LOD don't overshoot, but finite enough that
        // the next pick_face doesn't land exactly on the boundary.
        let cell_eps = max(shell * cell_lo * 1e-3, 1e-6);
        t_local = t_next + cell_eps;
    }

    return result;
}
