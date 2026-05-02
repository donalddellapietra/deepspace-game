#include "bindings.wgsl"
#include "ray_prim.wgsl"
#include "march_helpers.wgsl"

// Per-frame stack ceiling. Past this we splat representative_block.
// 24 covers the default cell_subtree_depth=20 with headroom; sphere
// mode is opt-in (--planet-render-sphere) so the stack arrays only
// cost when sphere render is active.
const SPHERE_DESCENT_DEPTH: u32 = 24u;

// ─────────────────────────────────────────────────────────────────────
// Cartesian-local DDA with frame-aware re-zero on push.
//
// At slab-cell entry, an orthonormal frame is built at the slab
// cell's center (e_lon, e_r, e_lat tangents to the sphere). The ray
// is transformed into that frame so the slab cell occupies `[0, 3)³`.
//
// On EACH push, we DON'T let `cur_node_origin` drift away from zero.
// Instead we re-zero: the cell we're descending into becomes the new
// `[0, 3)³` frame. The ray transforms accordingly:
//
//   new_O = (old_O - cell) * 3
//   new_D =  old_D       * 3
//
// (D-scaled, t-preserved.) After re-zero, `cur_node_origin = vec3(0)`
// and `cur_cell_size = 1` always — the DDA's standard machinery sees
// O(1) coordinates at every depth, not 1/3^N values that would crush
// against the f32 ULP floor.
//
// The cell index path (s_cell stack) makes pop reversible:
//
//   on pop: old_O = new_O / 3 + popped_cell
//           old_D = new_D / 3
//
// Sphere curvature within an anchor sub-cell of width ε on a planet
// of radius r introduces only `O(ε²/r)` deviation from the local-flat
// approximation — for ε ≤ slab_step ≈ 0.04 and r ≈ 0.5 that's at most
// 1.6e-3 of cell extent at the slab cell, much smaller in sub-cells.
// The descent is thus geometrically safe while keeping coords O(1).
//
// Bevel is computed in local coords from the hit's in-cell fractions
// and the entry-face normal; the local axis-aligned normal is rotated
// back to the world basis (e_lon, e_r, e_lat) for shading.
// ─────────────────────────────────────────────────────────────────────

fn sphere_descend_anchor(
    anchor_idx: u32,
    ray_origin: vec3<f32>, ray_dir: vec3<f32>,
    cs_center: vec3<f32>, inv_norm: f32,
    cell_lon_center: f32, cell_lat_center: f32, cell_r_center: f32,
    cell_lon_step: f32, cell_lat_step: f32, cell_r_step: f32,
    t_in: f32,
    t_slab_exit: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    // ── Slab cell's orthonormal world frame ──────────────────────────
    let cos_lat = cos(cell_lat_center);
    let sin_lat = sin(cell_lat_center);
    let cos_lon = cos(cell_lon_center);
    let sin_lon = sin(cell_lon_center);
    let e_r   = vec3<f32>(cos_lat * cos_lon, sin_lat,        cos_lat * sin_lon);
    let e_lon = vec3<f32>(-sin_lon,           0.0,            cos_lon);
    let e_lat = vec3<f32>(-sin_lat * cos_lon, cos_lat,       -sin_lat * sin_lon);
    // Slab-axis convention (matches slab tree slot layout
    // dims=[lon, r, lat]): local x → e_lon, y → e_r, z → e_lat.
    let ext_x = cell_r_center * cos_lat * cell_lon_step;
    let ext_y = cell_r_step;
    let ext_z = cell_r_center * cell_lat_step;
    let scale_x = 3.0 / max(ext_x, 1e-30);
    let scale_y = 3.0 / max(ext_y, 1e-30);
    let scale_z = 3.0 / max(ext_z, 1e-30);
    // Slab cell corner (where local = 0,0,0) in world.
    let cell_corner = cs_center + cell_r_center * e_r
        - 0.5 * ext_x * e_lon
        - 0.5 * ext_y * e_r
        - 0.5 * ext_z * e_lat;

    // Transform ray into slab cell's local frame. Slab cell = [0,3)³.
    let dv = ray_origin - cell_corner;
    var cur_O = vec3<f32>(
        dot(dv,      e_lon) * scale_x,
        dot(dv,      e_r)   * scale_y,
        dot(dv,      e_lat) * scale_z,
    );
    var cur_D = vec3<f32>(
        dot(ray_dir, e_lon) * scale_x,
        dot(ray_dir, e_r)   * scale_y,
        dot(ray_dir, e_lat) * scale_z,
    );

    // ── DDA setup. cur_node_origin = 0, cur_cell_size = 1 (always).
    // After every push we re-zero, so these scalars are constants and
    // we never accumulate ULP drift.
    var cur_inv_dir = vec3<f32>(
        select(1e30, 1.0 / cur_D.x, abs(cur_D.x) > 1e-12),
        select(1e30, 1.0 / cur_D.y, abs(cur_D.y) > 1e-12),
        select(1e30, 1.0 / cur_D.z, abs(cur_D.z) > 1e-12),
    );
    var cur_step = vec3<i32>(
        select(-1, 1, cur_D.x >= 0.0),
        select(-1, 1, cur_D.y >= 0.0),
        select(-1, 1, cur_D.z >= 0.0),
    );
    var cur_delta_dist = abs(cur_inv_dir);

    // Initial cell at slab entry (t = t_in is the slab DDA's current
    // t; ray is geometrically inside the curved slab cell at this t).
    var t = t_in;
    var entry_pos = cur_O + cur_D * t;
    var cur_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );

    // Side-dist: per-axis t at which the ray will next cross a face
    // of the current cell. With cur_cell_size = 1 always (re-zero),
    // delta_dist increment per cell is just `cur_delta_dist`.
    var cf = vec3<f32>(cur_cell);
    var cur_side_dist = vec3<f32>(
        select((cf.x       - cur_O.x) * cur_inv_dir.x,
               (cf.x + 1.0 - cur_O.x) * cur_inv_dir.x, cur_D.x >= 0.0),
        select((cf.y       - cur_O.y) * cur_inv_dir.y,
               (cf.y + 1.0 - cur_O.y) * cur_inv_dir.y, cur_D.y >= 0.0),
        select((cf.z       - cur_O.z) * cur_inv_dir.z,
               (cf.z + 1.0 - cur_O.z) * cur_inv_dir.z, cur_D.z >= 0.0),
    );

    // Entry normal: which face of slab [0, 3)³ did the ray cross?
    // Compute via per-axis ray-slab math; max of per-axis entry-t
    // identifies the face. (This is needed for the bevel UV when the
    // very first cell already has a tag=1 hit — no advance step has
    // run, so the "default" `normal` would otherwise be zero.)
    let t1_slab = (vec3<f32>(0.0) - cur_O) * cur_inv_dir;
    let t2_slab = (vec3<f32>(3.0) - cur_O) * cur_inv_dir;
    let t_lo_slab = min(t1_slab, t2_slab);
    let entry_t_slab = max(t_lo_slab.x, max(t_lo_slab.y, t_lo_slab.z));
    var normal: vec3<f32>;
    if t_lo_slab.x >= entry_t_slab - 1e-9 {
        normal = vec3<f32>(-f32(cur_step.x), 0.0, 0.0);
    } else if t_lo_slab.y >= entry_t_slab - 1e-9 {
        normal = vec3<f32>(0.0, -f32(cur_step.y), 0.0);
    } else {
        normal = vec3<f32>(0.0, 0.0, -f32(cur_step.z));
    }

    // ── Stack: node IDs and cell-index paths (for pop). Cell indices
    // are stored at the moment of push so pop can un-re-zero exactly.
    var s_node_idx: array<u32, SPHERE_DESCENT_DEPTH>;
    var s_cell:     array<u32, SPHERE_DESCENT_DEPTH>;
    var depth: u32 = 0u;
    s_node_idx[0] = anchor_idx;

    let root_header_off = node_offsets[anchor_idx];
    var cur_occupancy: u32 = tree[root_header_off];
    var cur_first_child: u32 = tree[root_header_off + 1u];

    var iters: u32 = 0u;
    let max_iters: u32 = 4096u;
    var did_hit: bool = false;
    var hit_t: f32 = 0.0;
    var hit_block: u32 = 0u;
    var hit_in_cell: vec3<f32> = vec3<f32>(0.5);

    loop {
        if iters >= max_iters { break; }
        iters = iters + 1u;

        // OOB → pop one frame, undo the re-zero on the ray.
        if cur_cell.x < 0 || cur_cell.x > 2 || cur_cell.y < 0 || cur_cell.y > 2 || cur_cell.z < 0 || cur_cell.z > 2 {
            if depth == 0u { break; }
            // Pop. Reverse the push transform:
            //   old_O = new_O / 3 + popped_cell
            //   old_D = new_D / 3
            depth = depth - 1u;
            let popped = unpack_cell(s_cell[depth]);
            cur_O = cur_O * (1.0 / 3.0) + vec3<f32>(popped);
            cur_D = cur_D * (1.0 / 3.0);
            cur_inv_dir = cur_inv_dir * 3.0;
            cur_delta_dist = cur_delta_dist * 3.0;

            // Determine which face we crossed (in OLD-frame coords) and
            // advance the parent's cell on that axis.
            var step_xyz = vec3<i32>(0);
            if cur_cell.x < 0 { step_xyz.x = -1; }
            if cur_cell.x > 2 { step_xyz.x = 1; }
            if cur_cell.y < 0 { step_xyz.y = -1; }
            if cur_cell.y > 2 { step_xyz.y = 1; }
            if cur_cell.z < 0 { step_xyz.z = -1; }
            if cur_cell.z > 2 { step_xyz.z = 1; }
            cur_cell = popped + step_xyz;

            // Restore parent's occupancy.
            let parent_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];

            // Recompute side_dist for the new (parent-frame) cell.
            let pcf = vec3<f32>(cur_cell);
            cur_side_dist = vec3<f32>(
                select((pcf.x       - cur_O.x) * cur_inv_dir.x,
                       (pcf.x + 1.0 - cur_O.x) * cur_inv_dir.x, cur_D.x >= 0.0),
                select((pcf.y       - cur_O.y) * cur_inv_dir.y,
                       (pcf.y + 1.0 - cur_O.y) * cur_inv_dir.y, cur_D.y >= 0.0),
                select((pcf.z       - cur_O.z) * cur_inv_dir.z,
                       (pcf.z + 1.0 - cur_O.z) * cur_inv_dir.z, cur_D.z >= 0.0),
            );
            normal = -vec3<f32>(step_xyz);
            continue;
        }

        let slot = u32(cur_cell.x + cur_cell.y * 3 + cur_cell.z * 9);
        let bit = 1u << slot;

        if (cur_occupancy & bit) == 0u {
            // Empty slot — DDA-step along smallest side_dist.
            let m = min_axis_mask(cur_side_dist);
            cur_cell = cur_cell + vec3<i32>(m) * cur_step;
            cur_side_dist = cur_side_dist + m * cur_delta_dist;
            normal = -vec3<f32>(cur_step) * m;
            continue;
        }

        let rank = countOneBits(cur_occupancy & (bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        let block_type = (packed >> 8u) & 0xFFFFu;

        if tag == 1u {
            // Block leaf — hit in current cell.
            let cell_min_l = vec3<f32>(cur_cell);
            let cell_max_l = cell_min_l + vec3<f32>(1.0);
            let bx = ray_box(cur_O, cur_inv_dir, cell_min_l, cell_max_l);
            hit_t = max(bx.t_enter, 0.0);
            // Clamp against curved slab cell exit. Flat sub-cells can
            // stick past the curved slab boundary (~0.4% in lon-arc at
            // off-equator lats); rejecting hits past `t_slab_exit`
            // prevents thin-sliver overhangs at slab cell edges.
            if hit_t > t_slab_exit { return result; }
            let pos_at_hit = cur_O + cur_D * hit_t;
            hit_in_cell = clamp(pos_at_hit - cell_min_l, vec3<f32>(0.0), vec3<f32>(1.0));
            hit_block = block_type;
            did_hit = true;
            break;
        }
        if tag != 2u {
            // Entity / unknown — advance.
            let m = min_axis_mask(cur_side_dist);
            cur_cell = cur_cell + vec3<i32>(m) * cur_step;
            cur_side_dist = cur_side_dist + m * cur_delta_dist;
            normal = -vec3<f32>(cur_step) * m;
            continue;
        }
        // tag == 2u — non-uniform Node.
        if block_type == 0xFFFEu {
            // Subtree empty — skip cell.
            let m = min_axis_mask(cur_side_dist);
            cur_cell = cur_cell + vec3<i32>(m) * cur_step;
            cur_side_dist = cur_side_dist + m * cur_delta_dist;
            normal = -vec3<f32>(cur_step) * m;
            continue;
        }
        let child_idx = tree[child_base + 1u];
        if depth + 1u >= SPHERE_DESCENT_DEPTH {
            // Stack ceiling — splat representative.
            let cell_min_l = vec3<f32>(cur_cell);
            let cell_max_l = cell_min_l + vec3<f32>(1.0);
            let bx = ray_box(cur_O, cur_inv_dir, cell_min_l, cell_max_l);
            hit_t = max(bx.t_enter, 0.0);
            if hit_t > t_slab_exit { return result; }
            let pos_at_hit = cur_O + cur_D * hit_t;
            hit_in_cell = clamp(pos_at_hit - cell_min_l, vec3<f32>(0.0), vec3<f32>(1.0));
            hit_block = block_type;
            did_hit = true;
            break;
        }

        // ── Push with re-zero ──
        // Save the cell we descended into, transform ray so the cell
        // becomes the new [0, 3)³ frame.
        s_cell[depth] = pack_cell(cur_cell);
        let cell_f = vec3<f32>(cur_cell);
        cur_O = (cur_O - cell_f) * 3.0;
        cur_D = cur_D * 3.0;
        cur_inv_dir = cur_inv_dir * (1.0 / 3.0);
        cur_delta_dist = cur_delta_dist * (1.0 / 3.0);

        depth = depth + 1u;
        s_node_idx[depth] = child_idx;
        let child_header_off = node_offsets[child_idx];
        cur_occupancy = tree[child_header_off];
        cur_first_child = tree[child_header_off + 1u];

        // Compute new sub-cell at depth+1 from the ray's current pos
        // (= same world position; in new frame coords).
        let new_pos = cur_O + cur_D * t;
        cur_cell = vec3<i32>(
            clamp(i32(floor(new_pos.x)), 0, 2),
            clamp(i32(floor(new_pos.y)), 0, 2),
            clamp(i32(floor(new_pos.z)), 0, 2),
        );
        let ncf = vec3<f32>(cur_cell);
        cur_side_dist = vec3<f32>(
            select((ncf.x       - cur_O.x) * cur_inv_dir.x,
                   (ncf.x + 1.0 - cur_O.x) * cur_inv_dir.x, cur_D.x >= 0.0),
            select((ncf.y       - cur_O.y) * cur_inv_dir.y,
                   (ncf.y + 1.0 - cur_O.y) * cur_inv_dir.y, cur_D.y >= 0.0),
            select((ncf.z       - cur_O.z) * cur_inv_dir.z,
                   (ncf.z + 1.0 - cur_O.z) * cur_inv_dir.z, cur_D.z >= 0.0),
        );

        // Recompute entry normal in the new frame: which face of the
        // new [0, 3)³ does the ray cross? Same per-axis ray-box maths
        // as the initial slab entry. The previous "normal preserved
        // across push" comment was wrong — for a hit on a freshly
        // pushed cell with no advance step, the bevel needs the
        // child's entry face, not the slab's.
        let t1_new = (vec3<f32>(0.0) - cur_O) * cur_inv_dir;
        let t2_new = (vec3<f32>(3.0) - cur_O) * cur_inv_dir;
        let t_lo_new = min(t1_new, t2_new);
        let entry_t_new = max(t_lo_new.x, max(t_lo_new.y, t_lo_new.z));
        if t_lo_new.x >= entry_t_new - 1e-9 {
            normal = vec3<f32>(-f32(cur_step.x), 0.0, 0.0);
        } else if t_lo_new.y >= entry_t_new - 1e-9 {
            normal = vec3<f32>(0.0, -f32(cur_step.y), 0.0);
        } else {
            normal = vec3<f32>(0.0, 0.0, -f32(cur_step.z));
        }
    }

    if !did_hit { return result; }

    // ── Bevel from local in-cell fractions ───────────────────────────
    var u: f32;
    var v: f32;
    if abs(normal.x) > 0.5 {
        u = hit_in_cell.y;
        v = hit_in_cell.z;
    } else if abs(normal.y) > 0.5 {
        u = hit_in_cell.x;
        v = hit_in_cell.z;
    } else {
        u = hit_in_cell.x;
        v = hit_in_cell.y;
    }
    let face_edge = min(min(u, 1.0 - u), min(v, 1.0 - v));
    let shape = smoothstep(0.02, 0.14, face_edge);
    let bevel_strength = 0.7 + 0.3 * shape;

    // Local axis-aligned normal → world basis (e_lon = local x,
    // e_r = local y, e_lat = local z, per slab-axis convention above).
    let n_world = normal.x * e_lon + normal.y * e_r + normal.z * e_lat;

    // ── Camera-frame t ───────────────────────────────────────────────
    // The ray entered descent at t_in (slab DDA's current t). Inside
    // the descent, t was preserved across pushes (D was scaled, but
    // pos(t) = O + D·t holds at every depth with the same t value).
    // So hit_t (the local t at which we hit the cell) IS the world t
    // at hit, in the same parameterisation as `ray_dir` input.
    // inv_norm scales to camera-frame normalised t for the result.
    let world_hit = ray_origin + ray_dir * hit_t;

    var out: HitResult;
    out.hit = true;
    out.t = hit_t * inv_norm;
    out.color = palette[hit_block].rgb * bevel_strength;
    out.normal = n_world;
    out.frame_level = 0u;
    out.frame_scale = 1.0;
    // Neutralise main.wgsl::shade_pixel's cube_face_bevel — we already
    // applied the bevel using local axis-aligned coords above.
    out.cell_min = world_hit - vec3<f32>(0.5);
    out.cell_size = 1.0;
    return out;
}
