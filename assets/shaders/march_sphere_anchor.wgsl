#include "bindings.wgsl"
#include "ray_prim.wgsl"
#include "march_helpers.wgsl"
#include "df64.wgsl"

// Per-frame stack ceiling. With df64 precision in the descent state
// we no longer hit a precision floor at depth ~10-12; bumping to 28
// covers `cell_subtree_depth=24` with headroom. Sphere mode is opt-in
// (--planet-render-sphere) so the stack arrays only cost when active.
const SPHERE_DESCENT_DEPTH: u32 = 28u;

// ─────────────────────────────────────────────────────────────────────
// Cartesian-local DDA with df64 state and frame-aware re-zero.
//
// At slab-cell entry, an orthonormal frame is built at the slab cell's
// center (e_lon, e_r, e_lat tangents to the sphere). The ray is
// transformed into that frame so the slab cell occupies [0, 3)³.
//
// On each push we re-zero: the cell we're descending into becomes the
// new [0, 3)³ frame:
//
//   new_O = (old_O - cell) * 3
//   new_D =  old_D       * 3
//
// (D-scaled, t-preserved.) After re-zero, `cur_node_origin = vec3(0)`
// and `cur_cell_size = 1` always.
//
// At depth K, |cur_D| has scaled by 3^K and |cur_inv_dir| by 3^-K;
// at K=20 the f32 ULP near typical t values is comparable to per-cell
// side_dist increments. Plain f32 mixes adjacent cells' boundaries
// and the descent picks the wrong axis or wrong cell. df64 (~46-bit
// mantissa) preserves cell-correctness through depth 24.
//
// The `t` variable is preserved across pushes and stored in df; the
// catastrophic-cancellation risk in `cur_O + cur_D * t` (both terms
// O(3^K), result O(1)) only resolves correctly with df.
// ─────────────────────────────────────────────────────────────────────

fn dfv3_make_side_dist(cur_O: DFv3, inv_dir: DFv3, cur_D_hi: vec3<f32>, cell: vec3<i32>) -> DFv3 {
    let cf = vec3<f32>(cell);
    let face_x: f32 = select(cf.x, cf.x + 1.0, cur_D_hi.x >= 0.0);
    let face_y: f32 = select(cf.y, cf.y + 1.0, cur_D_hi.y >= 0.0);
    let face_z: f32 = select(cf.z, cf.z + 1.0, cur_D_hi.z >= 0.0);
    let dx = df_mul(df_sub(df_from_f32(face_x), cur_O.x), inv_dir.x);
    let dy = df_mul(df_sub(df_from_f32(face_y), cur_O.y), inv_dir.y);
    let dz = df_mul(df_sub(df_from_f32(face_z), cur_O.z), inv_dir.z);
    return DFv3(dx, dy, dz);
}

fn df_inv_safe(d: DF) -> DF {
    if abs(d.hi) > 1e-12 { return df_inv(d); }
    return df_from_f32(1e30);
}

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
    let ext_x = cell_r_center * cos_lat * cell_lon_step;
    let ext_y = cell_r_step;
    let ext_z = cell_r_center * cell_lat_step;
    let scale_x = 3.0 / max(ext_x, 1e-30);
    let scale_y = 3.0 / max(ext_y, 1e-30);
    let scale_z = 3.0 / max(ext_z, 1e-30);
    let cell_corner = cs_center + cell_r_center * e_r
        - 0.5 * ext_x * e_lon
        - 0.5 * ext_y * e_r
        - 0.5 * ext_z * e_lat;

    // f32 ray transformed into slab cell's local frame, then widened
    // to df. Initial state has lo = 0; precision accumulates as the
    // descent re-zeros and rescales.
    let dv = ray_origin - cell_corner;
    let cur_O_hi = vec3<f32>(
        dot(dv,      e_lon) * scale_x,
        dot(dv,      e_r)   * scale_y,
        dot(dv,      e_lat) * scale_z,
    );
    let cur_D_hi = vec3<f32>(
        dot(ray_dir, e_lon) * scale_x,
        dot(ray_dir, e_r)   * scale_y,
        dot(ray_dir, e_lat) * scale_z,
    );
    var cur_O: DFv3 = dfv3_from_f32(cur_O_hi);
    var cur_D: DFv3 = dfv3_from_f32(cur_D_hi);

    var cur_inv_dir: DFv3 = DFv3(
        df_inv_safe(cur_D.x),
        df_inv_safe(cur_D.y),
        df_inv_safe(cur_D.z),
    );
    var cur_step = vec3<i32>(
        select(-1, 1, cur_D.x.hi >= 0.0),
        select(-1, 1, cur_D.y.hi >= 0.0),
        select(-1, 1, cur_D.z.hi >= 0.0),
    );
    var cur_delta_dist: DFv3 = dfv3_abs(cur_inv_dir);

    // Initial cell at slab entry — t = t_in (slab DDA's current t,
    // ray geometrically inside the curved slab cell at this t).
    var t: DF = df_from_f32(t_in);
    var entry_pos: DFv3 = dfv3_add(cur_O, dfv3_mul(cur_D, DFv3(t, t, t)));
    let entry_pos_f = dfv3_to_f32(entry_pos);
    var cur_cell = vec3<i32>(
        clamp(i32(floor(entry_pos_f.x)), 0, 2),
        clamp(i32(floor(entry_pos_f.y)), 0, 2),
        clamp(i32(floor(entry_pos_f.z)), 0, 2),
    );

    var cur_side_dist: DFv3 = dfv3_make_side_dist(cur_O, cur_inv_dir, cur_D_hi, cur_cell);

    // Entry normal: which face of slab [0, 3)³ did the ray cross?
    // Compute via per-axis ray-slab math; max of per-axis entry-t
    // identifies the face. Comparisons are df-aware (sub-ULP precision
    // matters at deep depth where cur_inv_dir has shrunk ×3^-K).
    var t_lo_x: DF = df_mul(df_sub(df_from_f32(0.0), cur_O.x), cur_inv_dir.x);
    var t_hi_x: DF = df_mul(df_sub(df_from_f32(3.0), cur_O.x), cur_inv_dir.x);
    if df_lt(t_hi_x, t_lo_x) { let tmp = t_lo_x; t_lo_x = t_hi_x; t_hi_x = tmp; }
    var t_lo_y: DF = df_mul(df_sub(df_from_f32(0.0), cur_O.y), cur_inv_dir.y);
    var t_hi_y: DF = df_mul(df_sub(df_from_f32(3.0), cur_O.y), cur_inv_dir.y);
    if df_lt(t_hi_y, t_lo_y) { let tmp = t_lo_y; t_lo_y = t_hi_y; t_hi_y = tmp; }
    var t_lo_z: DF = df_mul(df_sub(df_from_f32(0.0), cur_O.z), cur_inv_dir.z);
    var t_hi_z: DF = df_mul(df_sub(df_from_f32(3.0), cur_O.z), cur_inv_dir.z);
    if df_lt(t_hi_z, t_lo_z) { let tmp = t_lo_z; t_lo_z = t_hi_z; t_hi_z = tmp; }

    var normal: vec3<f32>;
    if !df_lt(t_lo_x, t_lo_y) && !df_lt(t_lo_x, t_lo_z) {
        normal = vec3<f32>(-f32(cur_step.x), 0.0, 0.0);
    } else if !df_lt(t_lo_y, t_lo_z) {
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

        // OOB → pop one frame, undo the re-zero.
        if cur_cell.x < 0 || cur_cell.x > 2 || cur_cell.y < 0 || cur_cell.y > 2 || cur_cell.z < 0 || cur_cell.z > 2 {
            if depth == 0u { break; }
            depth = depth - 1u;
            let popped = unpack_cell(s_cell[depth]);
            // Pop: old_O = new_O / 3 + popped, old_D = new_D / 3.
            cur_O = dfv3_add(dfv3_div3(cur_O), dfv3_from_f32(vec3<f32>(popped)));
            cur_D = dfv3_div3(cur_D);
            cur_inv_dir = dfv3_times3(cur_inv_dir);
            cur_delta_dist = dfv3_times3(cur_delta_dist);

            var step_xyz = vec3<i32>(0);
            if cur_cell.x < 0 { step_xyz.x = -1; }
            if cur_cell.x > 2 { step_xyz.x = 1; }
            if cur_cell.y < 0 { step_xyz.y = -1; }
            if cur_cell.y > 2 { step_xyz.y = 1; }
            if cur_cell.z < 0 { step_xyz.z = -1; }
            if cur_cell.z > 2 { step_xyz.z = 1; }
            cur_cell = popped + step_xyz;

            let parent_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];

            let cur_D_hi_now = dfv3_hi(cur_D);
            cur_side_dist = dfv3_make_side_dist(cur_O, cur_inv_dir, cur_D_hi_now, cur_cell);
            normal = -vec3<f32>(step_xyz);
            continue;
        }

        let slot = u32(cur_cell.x + cur_cell.y * 3 + cur_cell.z * 9);
        let bit = 1u << slot;

        if (cur_occupancy & bit) == 0u {
            // Empty slot — DDA-step along smallest side_dist.
            let m = df_min_axis_mask(cur_side_dist);
            t = dfv3_dot_mask(cur_side_dist, m);
            cur_cell = cur_cell + vec3<i32>(m) * cur_step;
            // cur_side_dist[axis] += cur_delta_dist[axis] (axis = m).
            if m.x > 0.5 {
                cur_side_dist.x = df_add(cur_side_dist.x, cur_delta_dist.x);
            } else if m.y > 0.5 {
                cur_side_dist.y = df_add(cur_side_dist.y, cur_delta_dist.y);
            } else {
                cur_side_dist.z = df_add(cur_side_dist.z, cur_delta_dist.z);
            }
            normal = -vec3<f32>(cur_step) * m;
            continue;
        }

        let rank = countOneBits(cur_occupancy & (bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        let block_type = (packed >> 8u) & 0xFFFFu;

        if tag == 1u {
            // Block leaf — ray entered this cell at `t` (preserved
            // across DDA advances and pushes). Skip the ray_box redo.
            var hit_t_df: DF = t;
            if df_lt(hit_t_df, df_from_f32(0.0)) { hit_t_df = df_from_f32(0.0); }
            // Curved-slab boundary clamp: f32 t_slab_exit is the
            // outermost meridian/parallel/sphere intersection past `t`,
            // and we reject hits beyond it to avoid flat sub-cells
            // sticking past the curved slab edge.
            if df_to_f32(hit_t_df) > t_slab_exit { return result; }
            let pos_at_hit_df: DFv3 = dfv3_add(cur_O, dfv3_mul(cur_D, DFv3(hit_t_df, hit_t_df, hit_t_df)));
            let pos_at_hit = dfv3_to_f32(pos_at_hit_df);
            let cell_min_l = vec3<f32>(cur_cell);
            hit_in_cell = clamp(pos_at_hit - cell_min_l, vec3<f32>(0.0), vec3<f32>(1.0));
            hit_block = block_type;
            hit_t = df_to_f32(hit_t_df);
            did_hit = true;
            break;
        }
        if tag != 2u {
            // Entity / unknown — advance.
            let m = df_min_axis_mask(cur_side_dist);
            t = dfv3_dot_mask(cur_side_dist, m);
            cur_cell = cur_cell + vec3<i32>(m) * cur_step;
            if m.x > 0.5 { cur_side_dist.x = df_add(cur_side_dist.x, cur_delta_dist.x); }
            else if m.y > 0.5 { cur_side_dist.y = df_add(cur_side_dist.y, cur_delta_dist.y); }
            else { cur_side_dist.z = df_add(cur_side_dist.z, cur_delta_dist.z); }
            normal = -vec3<f32>(cur_step) * m;
            continue;
        }
        // tag == 2u — non-uniform Node.
        if block_type == 0xFFFEu {
            // Subtree empty — skip cell.
            let m = df_min_axis_mask(cur_side_dist);
            t = dfv3_dot_mask(cur_side_dist, m);
            cur_cell = cur_cell + vec3<i32>(m) * cur_step;
            if m.x > 0.5 { cur_side_dist.x = df_add(cur_side_dist.x, cur_delta_dist.x); }
            else if m.y > 0.5 { cur_side_dist.y = df_add(cur_side_dist.y, cur_delta_dist.y); }
            else { cur_side_dist.z = df_add(cur_side_dist.z, cur_delta_dist.z); }
            normal = -vec3<f32>(cur_step) * m;
            continue;
        }
        let child_idx = tree[child_base + 1u];
        if depth + 1u >= SPHERE_DESCENT_DEPTH {
            // Stack ceiling — splat representative.
            var hit_t_df: DF = t;
            if df_lt(hit_t_df, df_from_f32(0.0)) { hit_t_df = df_from_f32(0.0); }
            if df_to_f32(hit_t_df) > t_slab_exit { return result; }
            let pos_at_hit_df: DFv3 = dfv3_add(cur_O, dfv3_mul(cur_D, DFv3(hit_t_df, hit_t_df, hit_t_df)));
            let pos_at_hit = dfv3_to_f32(pos_at_hit_df);
            let cell_min_l = vec3<f32>(cur_cell);
            hit_in_cell = clamp(pos_at_hit - cell_min_l, vec3<f32>(0.0), vec3<f32>(1.0));
            hit_block = block_type;
            hit_t = df_to_f32(hit_t_df);
            did_hit = true;
            break;
        }

        // ── Push with re-zero ──
        s_cell[depth] = pack_cell(cur_cell);
        let cell_f = dfv3_from_f32(vec3<f32>(cur_cell));
        cur_O = dfv3_times3(dfv3_sub(cur_O, cell_f));
        cur_D = dfv3_times3(cur_D);
        cur_inv_dir = dfv3_div3(cur_inv_dir);
        cur_delta_dist = dfv3_div3(cur_delta_dist);

        depth = depth + 1u;
        s_node_idx[depth] = child_idx;
        let child_header_off = node_offsets[child_idx];
        cur_occupancy = tree[child_header_off];
        cur_first_child = tree[child_header_off + 1u];

        // new_pos = cur_O + cur_D * t — precision-critical: both
        // terms are O(3^K), result is O(1) after cancellation.
        let new_pos_df: DFv3 = dfv3_add(cur_O, dfv3_mul(cur_D, DFv3(t, t, t)));
        let new_pos = dfv3_to_f32(new_pos_df);
        cur_cell = vec3<i32>(
            clamp(i32(floor(new_pos.x)), 0, 2),
            clamp(i32(floor(new_pos.y)), 0, 2),
            clamp(i32(floor(new_pos.z)), 0, 2),
        );
        let cur_D_hi_now = dfv3_hi(cur_D);
        cur_side_dist = dfv3_make_side_dist(cur_O, cur_inv_dir, cur_D_hi_now, cur_cell);

        // Recompute entry normal in the new frame: which face of the
        // new [0, 3)³ does the ray cross?
        var n_lo_x: DF = df_mul(df_sub(df_from_f32(0.0), cur_O.x), cur_inv_dir.x);
        var n_hi_x: DF = df_mul(df_sub(df_from_f32(3.0), cur_O.x), cur_inv_dir.x);
        if df_lt(n_hi_x, n_lo_x) { let tmp = n_lo_x; n_lo_x = n_hi_x; n_hi_x = tmp; }
        var n_lo_y: DF = df_mul(df_sub(df_from_f32(0.0), cur_O.y), cur_inv_dir.y);
        var n_hi_y: DF = df_mul(df_sub(df_from_f32(3.0), cur_O.y), cur_inv_dir.y);
        if df_lt(n_hi_y, n_lo_y) { let tmp = n_lo_y; n_lo_y = n_hi_y; n_hi_y = tmp; }
        var n_lo_z: DF = df_mul(df_sub(df_from_f32(0.0), cur_O.z), cur_inv_dir.z);
        var n_hi_z: DF = df_mul(df_sub(df_from_f32(3.0), cur_O.z), cur_inv_dir.z);
        if df_lt(n_hi_z, n_lo_z) { let tmp = n_lo_z; n_lo_z = n_hi_z; n_hi_z = tmp; }
        if !df_lt(n_lo_x, n_lo_y) && !df_lt(n_lo_x, n_lo_z) {
            normal = vec3<f32>(-f32(cur_step.x), 0.0, 0.0);
        } else if !df_lt(n_lo_y, n_lo_z) {
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

    let n_world = normal.x * e_lon + normal.y * e_r + normal.z * e_lat;
    let world_hit = ray_origin + ray_dir * hit_t;

    var out: HitResult;
    out.hit = true;
    out.t = hit_t * inv_norm;
    out.color = palette[hit_block].rgb * bevel_strength;
    out.normal = n_world;
    out.frame_level = 0u;
    out.frame_scale = 1.0;
    out.cell_min = world_hit - vec3<f32>(0.5);
    out.cell_size = 1.0;
    return out;
}
