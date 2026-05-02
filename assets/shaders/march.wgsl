#include "bindings.wgsl"
#include "tree.wgsl"
#include "ray_prim.wgsl"

// Cell-packing helpers. Cell coords at each depth range -1..=3
// (legal 0..=2 plus ±1 over-step to trigger pop). Pack +1-shifted
// into 3 bits per axis = 9 bits per u32. Shrinks the per-thread
// `s_cell` stack from 96 B (vec3<i32>×8) to 32 B (u32×8), targeting
// the Fragment Occupancy register cliff measured at 9.7% on
// Jerusalem nucleus (rule: <25% = register pressure).
fn pack_cell(c: vec3<i32>) -> u32 {
    let ux = u32(c.x + 1) & 7u;
    let uy = u32(c.y + 1) & 7u;
    let uz = u32(c.z + 1) & 7u;
    return ux | (uy << 3u) | (uz << 6u);
}

fn unpack_cell(p: u32) -> vec3<i32> {
    return vec3<i32>(
        i32(p & 7u) - 1,
        i32((p >> 3u) & 7u) - 1,
        i32((p >> 6u) & 7u) - 1,
    );
}

// Conservative 27-bit "path mask" — the tensor product of per-axis
// 3-bit masks of cells reachable from `entry_cell` moving in `step`
// direction. Over-approximates the actual ray path (any axis-wise
// reachable cell triple, not only the specific 3D path the ray
// traces). Safe for occupancy-intersection culling: if the full
// superset misses all occupied slots, the actual path certainly
// does. Used for instrumentation only right now — does not affect
// traversal.
fn path_mask_conservative(entry_cell: vec3<i32>, step: vec3<i32>) -> u32 {
    let ec = vec3<u32>(
        u32(clamp(entry_cell.x, 0, 2)),
        u32(clamp(entry_cell.y, 0, 2)),
        u32(clamp(entry_cell.z, 0, 2)),
    );
    // Per-axis 3-bit mask. step > 0: bits [ec..2]; step < 0: bits
    // [0..ec]. step is always ±1 in march_cartesian (non-zero).
    let mx: u32 = select((1u << (ec.x + 1u)) - 1u, (7u << ec.x) & 7u, step.x > 0);
    let my: u32 = select((1u << (ec.y + 1u)) - 1u, (7u << ec.y) & 7u, step.y > 0);
    let mz: u32 = select((1u << (ec.z + 1u)) - 1u, (7u << ec.z) & 7u, step.z > 0);
    // Smear each 3-bit axis mask into its 27-bit "axis active"
    // pattern. x repeats stride-3 (bits 0,3,6,...); y expands to a
    // 9-bit xy-plane then repeats stride-9; z gates whole 9-bit
    // planes. Closed-form — no loops, no lookups.
    let x_active: u32 = mx * 0x01249249u;
    let y_9: u32 = ((my & 1u) * 0x007u)
                 | (((my >> 1u) & 1u) * 0x038u)
                 | (((my >> 2u) & 1u) * 0x1C0u);
    let y_active: u32 = y_9 * 0x00040201u;
    let z_active: u32 = ((mz & 1u) * 0x000001FFu)
                     | (((mz >> 1u) & 1u) * 0x0003FE00u)
                     | (((mz >> 2u) & 1u) * 0x07FC0000u);
    return x_active & y_active & z_active;
}

// Entity subtree walker. Cartesian DDA walking a standalone voxel
// subtree — no sphere/face/ribbon dispatch, no AABB side-buffer,
// no beam-prepass coupling. Called from `march_cartesian`'s tag==3
// branch after the ray has been transformed into the entity's
// `[0, 3)³` local frame. WGSL's no-recursion rule forces this to
// be a separate function rather than a re-entrant call to
// `march_cartesian`.
//
// On hit, returned `HitResult.t / cell_min / cell_size` are in
// entity-local units; the caller scales back to world via the
// entity's bbox size.

/// Map an axis-aligned face normal (single ±1 component) to a face id
/// matching the cube-face convention used by the CPU debug printer:
/// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z, 7=unknown / non-axis.
fn normal_to_face(n: vec3<f32>) -> u32 {
    if n.x >  0.5 { return 0u; }
    if n.x < -0.5 { return 1u; }
    if n.y >  0.5 { return 2u; }
    if n.y < -0.5 { return 3u; }
    if n.z >  0.5 { return 4u; }
    if n.z < -0.5 { return 5u; }
    return 7u;
}

/// Walker-probe writer. Called from `march_cartesian`'s hit / miss
/// return points. Gated on `uniforms.probe_pixel.z != 0u` AND the
/// current pixel matching `uniforms.probe_pixel.xy`. Non-atomic
/// stores are safe because at most one fragment invocation in the
/// grid passes both gates.
fn write_walker_probe(
    hit_flag: u32,
    steps: u32,
    final_depth: u32,
    cell: vec3<i32>,
    cur_node_origin: vec3<f32>,
    cur_cell_size: f32,
    hit_t: f32,
    normal: vec3<f32>,
    content_flag: u32,
) {
    if uniforms.probe_pixel.z == 0u { return; }
    if current_pixel.x != uniforms.probe_pixel.x { return; }
    if current_pixel.y != uniforms.probe_pixel.y { return; }
    walker_probe.hit_flag = hit_flag;
    walker_probe.ray_steps = steps;
    walker_probe.final_depth = final_depth;
    let cx = u32(clamp(cell.x + 1, 0, 7)) & 7u;
    let cy = u32(clamp(cell.y + 1, 0, 7)) & 7u;
    let cz = u32(clamp(cell.z + 1, 0, 7)) & 7u;
    walker_probe.terminal_cell = cx | (cy << 2u) | (cz << 4u);
    walker_probe.cur_node_origin_x_bits = bitcast<u32>(cur_node_origin.x);
    walker_probe.cur_node_origin_y_bits = bitcast<u32>(cur_node_origin.y);
    walker_probe.cur_node_origin_z_bits = bitcast<u32>(cur_node_origin.z);
    walker_probe.cur_cell_size_bits = bitcast<u32>(cur_cell_size);
    walker_probe.hit_t_bits = bitcast<u32>(hit_t);
    walker_probe.hit_face = normal_to_face(normal);
    walker_probe.content_flag = content_flag;
    walker_probe._reserved11 = 0u;
}

// Precision-stable cartesian DDA in nested local frames.
//
// Why nested local frames: with absolute coords in [0, 3]³, cells at
// depth N have size 3^-N. Past N≈14 the cell size hits the f32 ULP
// wall — `ray_origin + ray_dir * t` can no longer resolve sub-cell
// positions, the DDA picks wrong children, the visual smears.
//
// In this rewrite, every level's DDA runs at `cell_size = 1` in its
// own local [0, 3]³ frame. Ray dir scales by 3 at each push (`*= 3`),
// by 1/3 at each pop. Positions stay in O(1) magnitude at all depths
// — no precision degradation.
//
// Critical invariant: the ray's t parameter is identical in every
// frame. If `cur_origin = (root_origin - cell_chain) * 3^N` and
// `cur_dir = root_dir * 3^N`, then for any t the local position
// satisfies `cur_origin + cur_dir * t = (root_pos(t) - cell_chain)
// * 3^N` — same t indexes the same world point in every frame. So
// hit-time `t` is returned UNSCALED; only positions need the
// per-frame transform on push/pop.
//
// `cur_cell_size_root = 3^-depth` is tracked separately so
// `result.cell_size` and the LOD pixel test report values in the
// input ray's frame (matching what callers like
// `uv_dispatch_cartesian_tangent` and the world ribbon-pop expect).
fn march_entity_subtree(
    root_node_idx: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );

    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_cell: array<u32, MAX_STACK_DEPTH>;

    // Per-level local ray. At depth 0 these match the input ray.
    // Push: origin = (origin - cell_offset) * 3; dir *= 3.
    // Pop:  origin = origin / 3 + parent_cell_offset; dir /= 3.
    var cur_origin: vec3<f32> = ray_origin;
    var cur_dir: vec3<f32> = ray_dir;
    var cur_inv_dir: vec3<f32> = vec3<f32>(
        select(1e10, 1.0 / cur_dir.x, abs(cur_dir.x) > 1e-8),
        select(1e10, 1.0 / cur_dir.y, abs(cur_dir.y) > 1e-8),
        select(1e10, 1.0 / cur_dir.z, abs(cur_dir.z) > 1e-8),
    );
    var cur_delta_dist: vec3<f32> = abs(cur_inv_dir);
    // Cell-size IN THE ROOT FRAME at the current depth (3^-depth).
    // For `result.cell_size` and the LOD pixel-size test in the
    // caller's frame.
    var cur_cell_size_root: f32 = 1.0;

    var cur_side_dist: vec3<f32>;
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;
    s_node_idx[0] = root_node_idx;

    let root_header_off = node_offsets[root_node_idx];
    var cur_occupancy: u32 = tree[root_header_off];
    var cur_first_child: u32 = tree[root_header_off + 1u];

    let root_hit = ray_box(cur_origin, cur_inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }
    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = cur_origin + cur_dir * t_start;
    let entry_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );
    s_cell[0] = pack_cell(entry_cell);
    let cell_f = vec3<f32>(entry_cell);
    cur_side_dist = vec3<f32>(
        select((cell_f.x - entry_pos.x) * cur_inv_dir.x,
               (cell_f.x + 1.0 - entry_pos.x) * cur_inv_dir.x, cur_dir.x >= 0.0),
        select((cell_f.y - entry_pos.y) * cur_inv_dir.y,
               (cell_f.y + 1.0 - entry_pos.y) * cur_inv_dir.y, cur_dir.y >= 0.0),
        select((cell_f.z - entry_pos.z) * cur_inv_dir.z,
               (cell_f.z + 1.0 - entry_pos.z) * cur_inv_dir.z, cur_dir.z >= 0.0),
    );
    // `cur_side_dist[axis]` is "extra t-from-entry to next axis face"
    // in the CURRENT level's local frame. Per cell crossing it grows
    // by `cur_delta_dist[axis]`.

    var iterations = 0u;
    let max_iterations = 2048u;
    loop {
        if iterations >= max_iterations { break; }
        iterations += 1u;
        let cell = unpack_cell(s_cell[depth]);
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            // POP: from child to parent. Inverse of push transform.
            depth -= 1u;
            let popped_cell = unpack_cell(s_cell[depth]);
            let popped_f = vec3<f32>(popped_cell);
            // Ray transform: child_origin → parent_origin. Inverse
            // of push.
            cur_origin = cur_origin / 3.0 + popped_f;
            cur_dir = cur_dir / 3.0;
            cur_inv_dir = cur_inv_dir * 3.0;
            cur_delta_dist = cur_delta_dist * 3.0;
            cur_cell_size_root = cur_cell_size_root * 3.0;
            // Recompute side_dist for parent frame at the cell we
            // popped INTO (= cell we descended FROM in the parent).
            let entry_pos_p = cur_origin + cur_dir * 0.0; // ray-origin position is the reference
            // Side_dist[axis] = t-from-entry to next face along axis,
            // in the PARENT's local frame. We're at popped_cell in
            // parent. Using ray-origin (t=0) as reference is fine
            // because side_dist is recomputed each iteration as the
            // DDA only uses min(cur_side_dist) for stepping.
            //
            // Actually we need side_dist relative to the ray's
            // current position. Since the ray is at the boundary
            // (it just exited child at this t), recompute:
            cur_side_dist = vec3<f32>(
                select((popped_f.x - entry_pos_p.x) * cur_inv_dir.x,
                       (popped_f.x + 1.0 - entry_pos_p.x) * cur_inv_dir.x, cur_dir.x >= 0.0),
                select((popped_f.y - entry_pos_p.y) * cur_inv_dir.y,
                       (popped_f.y + 1.0 - entry_pos_p.y) * cur_inv_dir.y, cur_dir.y >= 0.0),
                select((popped_f.z - entry_pos_p.z) * cur_inv_dir.z,
                       (popped_f.z + 1.0 - entry_pos_p.z) * cur_inv_dir.z, cur_dir.z >= 0.0),
            );
            let parent_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];
            let m_oob = min_axis_mask(cur_side_dist);
            let advanced = popped_cell + vec3<i32>(m_oob) * step;
            s_cell[depth] = pack_cell(advanced);
            cur_side_dist += m_oob * cur_delta_dist;
            normal = -vec3<f32>(step) * m_oob;
            continue;
        }

        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        let slot_bit = 1u << slot;
        if ((cur_occupancy & slot_bit) == 0u) {
            let m_empty = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_empty) * step);
            cur_side_dist += m_empty * cur_delta_dist;
            normal = -vec3<f32>(step) * m_empty;
            continue;
        }
        let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;

        if tag == 1u {
            // Block leaf hit. Compute hit-cell box in CURRENT local
            // frame; ray-box gives the t at which the ray entered
            // this cell. Local-t equals input-frame t (invariant),
            // so return as-is.
            let cell_min_local = vec3<f32>(cell);
            let cell_max_local = cell_min_local + vec3<f32>(1.0);
            let cell_box = ray_box(cur_origin, cur_inv_dir, cell_min_local, cell_max_local);
            result.hit = true;
            result.t = max(cell_box.t_enter, 0.0);
            result.color = palette[(packed >> 8u) & 0xFFFFu].rgb;
            result.normal = normal;
            result.cell_min = ray_origin + ray_dir * result.t - vec3<f32>(0.5) * cur_cell_size_root;
            result.cell_size = cur_cell_size_root;
            return result;
        }
        // tag == 2u: Cartesian Node descent. tag==3 (EntityRef) is
        // not supported inside entity subtrees.
        if tag != 2u {
            let m_skip = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_skip) * step);
            cur_side_dist += m_skip * cur_delta_dist;
            normal = -vec3<f32>(step) * m_skip;
            continue;
        }
        let child_idx = tree[child_base + 1u];
        let child_bt = (packed >> 8u) & 0xFFFFu;
        if child_bt == 0xFFFEu {  // REPRESENTATIVE_EMPTY
            let m_rep = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_rep) * step);
            cur_side_dist += m_rep * cur_delta_dist;
            normal = -vec3<f32>(step) * m_rep;
            continue;
        }

        let at_max = depth + 1u >= MAX_STACK_DEPTH;
        let child_cell_size_root = cur_cell_size_root / 3.0;
        // LOD: terminate descent when child cells would be sub-pixel
        // on screen. `min_side` is the local-frame t to the next
        // axis face (= world-t since they're invariant); times the
        // input ray dir's length gives world-distance.
        let min_side = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
        let ray_dist = max(min_side * max(length(ray_dir), 1e-6), 0.001);
        let lod_pixels = child_cell_size_root / ray_dist
            * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
        let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;
        if at_max || at_lod {
            let bt = child_bt;
            if bt == 0xFFFEu || bt == 0xFFFDu {
                let m_lodt = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_lodt) * step);
                cur_side_dist += m_lodt * cur_delta_dist;
                normal = -vec3<f32>(step) * m_lodt;
            } else {
                let cell_min_local = vec3<f32>(cell);
                let cell_max_local = cell_min_local + vec3<f32>(1.0);
                let cell_box = ray_box(cur_origin, cur_inv_dir, cell_min_local, cell_max_local);
                result.hit = true;
                result.t = max(cell_box.t_enter, 0.0);
                result.color = palette[bt].rgb;
                result.normal = normal;
                result.cell_min = ray_origin + ray_dir * result.t - vec3<f32>(0.5) * cur_cell_size_root;
                result.cell_size = cur_cell_size_root;
                return result;
            }
        } else {
            // PUSH: descend into child cell. Transform ray into
            // child's [0, 3]³ local frame.
            let cell_f_push = vec3<f32>(cell);
            cur_origin = (cur_origin - cell_f_push) * 3.0;
            cur_dir = cur_dir * 3.0;
            cur_inv_dir = cur_inv_dir / 3.0;
            cur_delta_dist = cur_delta_dist / 3.0;
            cur_cell_size_root = child_cell_size_root;
            depth += 1u;
            s_node_idx[depth] = child_idx;
            let child_header_off = node_offsets[child_idx];
            cur_occupancy = tree[child_header_off];
            cur_first_child = tree[child_header_off + 1u];
            // Re-derive entry cell + side_dist in CHILD's local frame.
            let entry_pos_c = cur_origin + cur_dir * (max(root_hit.t_enter, 0.0) + 0.0001);
            // Note: clamp on entry_pos_c — at boundary, floor() may
            // step out by 1 ULP. Clamp to [0, 2].
            let child_cell_i = vec3<i32>(
                clamp(i32(floor(entry_pos_c.x)), 0, 2),
                clamp(i32(floor(entry_pos_c.y)), 0, 2),
                clamp(i32(floor(entry_pos_c.z)), 0, 2),
            );
            s_cell[depth] = pack_cell(child_cell_i);
            let lc = vec3<f32>(child_cell_i);
            cur_side_dist = vec3<f32>(
                select((lc.x - entry_pos_c.x) * cur_inv_dir.x,
                       (lc.x + 1.0 - entry_pos_c.x) * cur_inv_dir.x, cur_dir.x >= 0.0),
                select((lc.y - entry_pos_c.y) * cur_inv_dir.y,
                       (lc.y + 1.0 - entry_pos_c.y) * cur_inv_dir.y, cur_dir.y >= 0.0),
                select((lc.z - entry_pos_c.z) * cur_inv_dir.z,
                       (lc.z + 1.0 - entry_pos_c.z) * cur_inv_dir.z, cur_dir.z >= 0.0),
            );
        }
    }
    return result;
}

// Cartesian DDA in a single frame rooted at `root_node_idx`. The
// frame's cell spans `[0, 3)³` in `ray_origin/ray_dir` coords.
// Returns hit on cell terminal; on miss (ray exits the frame),
// returns hit=false so the caller can pop to the ancestor ribbon.
fn march_cartesian(
    root_node_idx: u32, ray_origin_in: vec3<f32>, ray_dir: vec3<f32>,
    depth_limit: u32, skip_slot: u32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    // Mutable alias of the input ray origin so the X-wrap branch
    // can translate it. WGSL function parameters are immutable.
    var ray_origin: vec3<f32> = ray_origin_in;

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
    // Packed per-depth cell coords, 32 B total (vs 96 B for the
    // pre-pack vec3<i32>×8). See pack_cell / unpack_cell above.
    var s_cell: array<u32, MAX_STACK_DEPTH>;

    // Current-depth cell size. Pure function of `depth` (1/3^depth), so
    // a scalar mutated on push (÷3) / pop (×3) is exactly equivalent
    // to a per-depth array. Saves ~20 B of per-thread state.
    var cur_cell_size: f32 = 1.0;

    // Current-depth node origin (world coords of the current frame's
    // [0,0,0] corner). Updated incrementally:
    //   descend: += vec3<f32>(s_cell[depth]) * parent_cell_size
    //   pop:     -= vec3<f32>(s_cell[new_depth]) * new_parent_cell_size
    // Reversible because s_cell[parent_depth] is preserved while we're
    // descended into the child — the DDA only advances s_cell[depth]
    // at the CURRENT depth. On pop we subtract the same contribution
    // we added on descend. Saves ~60 B of per-thread state.
    var cur_node_origin: vec3<f32> = vec3<f32>(0.0);

    // Current-depth "side distance" — t-parameter from `entry_pos`
    // (root box-entry point) to the next axis-aligned plane crossing
    // of the current cell, per axis. The DDA picks the minimum axis
    // each step and advances by `delta_dist * cur_cell_size`.
    //
    // Reference point is `entry_pos` (not `ray_origin`) so the root
    // init at depth 0 behaves identically to the baseline stacked
    // version. On pop we recompute from scratch in ~6 FMAs — pops are
    // infrequent compared to per-cell advances, so the recompute cost
    // amortizes to ~free. Saves ~60 B of per-thread state.
    var cur_side_dist: vec3<f32>;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    s_node_idx[0] = root_node_idx;

    // Interleaved-layout header for the CURRENT depth, cached in
    // scalar registers. Written on entry, refreshed on descend AND
    // on pop. Inner-loop slot checks at the same node never re-read
    // tree[] headers — the compiler keeps occupancy / first_child in
    // registers across the inner loop. With the interleaved layout,
    // the header's two u32s live in the same 64-byte cache line as
    // the first child entry, so even the initial load pair is a
    // single-line fetch on Apple Silicon's L1.
    let root_header_off = node_offsets[root_node_idx];
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
    // Mutable: refreshed by the X-wrap branch when ray_origin
    // translates on a wrap (the side_dist recomputes use entry_pos
    // as the reference point, so wrap must update both).
    var entry_pos: vec3<f32> = ray_origin + ray_dir * t_start;

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

            // Popped to a shallower depth. Reload cur_occupancy /
            // cur_first_child for the new depth's node. Pops are
            // rare compared to per-slot checks, so this is cheap.
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

        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        // Use the cached current-depth header. No storage-buffer
        // load per iteration — the compiler keeps cur_occupancy /
        // cur_first_child in registers across the inner loop.
        let slot_bit = 1u << slot;
        if ((cur_occupancy & slot_bit) == 0u) {
            // Empty — DDA advance. No access to the compact array.
            if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
            let m_empty = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_empty) * step);
            cur_side_dist += m_empty * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_empty;
            continue;
        }

        // Non-empty: compute rank via popcount, then load the
        // interleaved child entry (2 u32s: packed tag/block_type/pad
        // and BFS node_index). `cur_first_child` is already in
        // tree[] u32 units; the entry sits at `first_child + rank*2`
        // in the SAME buffer as the header we just read, typically
        // on the same cache line.
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
            // Walker probe (tag==1 hit).
            write_walker_probe(
                1u, iterations, depth, cell,
                cur_node_origin, cur_cell_size,
                result.t, normal, 1u,
            );
            return result;
        } else if ENABLE_ENTITIES && tag == 3u {
            // tag=3 — EntityRef. Guarded by the compile-time
            // `ENABLE_ENTITIES` override: fractal / sphere preset
            // worlds never produce tag=3 children, so the shader
            // compiler DCEs this branch + the call into
            // `march_entity_subtree` entirely. Measured on
            // Jerusalem nucleus 2560x1440: ENABLE_ENTITIES=false
            // recovers ~2 ms/frame (~6%) vs leaving the branch
            // runtime-present.
            //
            // The cell is a per-frame scene overlay; the entity's
            // actual bbox is the (sub-cell)
            // box from `entities[idx]`. Ray-box cull against that
            // bbox first so sub-cell motion is cheap — no tree
            // rebuild needed to reflect the new position.
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
                // LOD-terminal: splat the representative block
                // (u16) if it's a real palette entry. 0xFFFD
                // (entity sentinel) and 0xFFFE (empty sentinel)
                // both mean "don't splat" — advance DDA.
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

            // Transform ray into the entity subtree's [0, 3)³
            // local frame and descend. WGSL's no-recursion rule
            // forces a separate `march_entity_subtree` walker.
            let scale3 = vec3<f32>(3.0) / bbox_size;
            let local_origin = (ray_origin - entity.bbox_min) * scale3;
            let local_dir = ray_dir * scale3;
            let sub = march_entity_subtree(entity.subtree_bfs, local_origin, local_dir);
            if sub.hit {
                let size_over_3 = bbox_size * (1.0 / 3.0);
                result.hit = true;
                // Entity is cubic, scale3.x == y == z; pick any axis.
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
            // tag == 2u: Node child. Load node_index from the
            // second u32 of the compact entry we already located.
            let child_idx = tree[child_base + 1u];
            if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 1u; }

            // Shell skip: when re-entering a parent shell after a
            // ribbon pop, skip the SLOT we already traversed in the
            // inner shell. Uses slot index (not node_idx) so it works
            // correctly in deduplicated trees where siblings share the
            // same packed node. Checked BEFORE the kind lookup so a
            // ribbon-pop landing on a sphere-body slot doesn't
            // re-dispatch the sphere DDA we already traversed.
            let cell_slot = u32(cell.x) + u32(cell.y) * 3u + u32(cell.z) * 9u;
            if depth == 0u && cell_slot == skip_slot {
                let m_skip = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_skip) * step);
                cur_side_dist += m_skip * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_skip;
                continue;
            }

            // Empty-representative fast path: when the packed
            // child's representative_block is 255, the subtree has
            // no non-empty content (either uniform-empty deeper in
            // the tree, or a sub-pixel LOD'd collection of empty
            // cells). Descending into it will just bottom out at
            // LOD-terminal with bt==255 and advance one cell anyway
            // — same visual result, wasted levels of traversal.
            // Skip straight to DDA advance, matching the tag=0
            // branch above.
            //
            // This is the dominant cost source when zoomed-in over
            // empty space: rays hit tag=2 cells whose subtrees are
            // effectively empty, descend N levels to LOD-terminal,
            // advance one cell, repeat — 2-3 iterations where one
            // would do.
            let child_bt = child_block_type(packed);
            if child_bt == 0xFFFEu {
                if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
                let m_rep = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_rep) * step);
                cur_side_dist += m_rep * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_rep;
                continue;
            }

            // Cartesian Node: depth/LOD check, then descend.
            // depth_limit = MAX_STACK_DEPTH — LOD controls the
            // effective depth, not an artificial per-shell budget.
            let at_max = depth + 1u > depth_limit || depth + 1u >= MAX_STACK_DEPTH;
            let child_cell_size = cur_cell_size / 3.0;
            let cell_world_size = child_cell_size;
            let min_side = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
            let ray_dist = max(min_side * ray_metric, 0.001);
            let lod_pixels = cell_world_size / ray_dist * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            // Distance-based LOD: stop descending when the child
            // cell would be smaller than LOD_PIXEL_THRESHOLD pixels
            // on screen. Tunable override (default 1.0 = strict
            // Nyquist; higher values descend less). This is
            // invariant under zoom: the same physical content
            // produces the same lod_pixels regardless of what
            // `anchor_depth` the frame is rooted at.
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

                // Content-AABB cull. The 12-bit AABB used to live in
                // `packed` bits 16-27; it moved to a parallel
                // `aabbs[child_idx]` storage buffer when block_type
                // widened to u16 (bits 16-23 now carry block_type_hi).
                // If the ray misses the AABB → skip entire descent,
                // advance parent DDA; otherwise descend into the
                // child and let DDA find the content normally.
                //
                // History note: we used to also use `aabb_hit.t_enter`
                // to TRIM the DDA entry (jump past empty leading
                // cells), but that produced a 3×3-tile grid of
                // floor-voxel gaps on Sponza at close range — mid-
                // node `local_entry` + `new_cell` clamp + DDA init
                // around `entry_pos` drifted off-by-one at leaf
                // boundaries where sibling nodes meet. DDA entry now
                // uses the NODE box (same as pre-AABB code); the
                // cull still captures ~all the perf win.
                //
                // aabb_bits == 0 is a degenerate case (empty subtree
                // edge cases during pack); treat it as the full
                // [0, 3)^3 so behavior matches the pre-AABB code.
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

                // UV-sphere mid-descent dispatch. When the child is
                // tagged `NODE_KIND_UV_SPHERE_BODY`, transform the
                // ray into the child's [0, 3)³ local frame and hand
                // it to `march_uv_sphere`. On UV hit, return; on UV
                // miss, continue the parent's Cartesian DDA past
                // this child slot.
                let kind_id = node_kinds[child_idx].kind;
                if kind_id == NODE_KIND_UV_SPHERE_BODY {
                    let inv_size = 1.0 / child_cell_size;
                    let uv_ray_origin = (ray_origin - child_origin) * inv_size;
                    let uv_ray_dir = ray_dir * inv_size;
                    let uv_result = march_uv_sphere(child_idx, uv_ray_origin, uv_ray_dir);
                    if uv_result.hit {
                        result.hit = true;
                        result.t = uv_result.t;
                        result.color = uv_result.color;
                        result.normal = uv_result.normal;
                        result.cell_min = child_origin
                            + uv_result.cell_min * (child_cell_size / 3.0);
                        result.cell_size = uv_result.cell_size * (child_cell_size / 3.0);
                        return result;
                    }
                    // UV miss → step the parent DDA past this child.
                    let m_uv_miss = min_axis_mask(cur_side_dist);
                    s_cell[depth] = pack_cell(cell + vec3<i32>(m_uv_miss) * step);
                    cur_side_dist += m_uv_miss * delta_dist * cur_cell_size;
                    normal = -vec3<f32>(step) * m_uv_miss;
                    continue;
                }

                // Use the NODE box for the DDA entry trim, not the
                // content AABB. The AABB is correct for the CULL test
                // above (rays missing the AABB skip the descent
                // entirely), but using `aabb_hit.t_enter` for
                // `ct_start` was causing a 3×3-tile grid of visual
                // gaps on Sponza's floor at close range. Root cause:
                // with the tight AABB `t_enter` lands mid-node, so
                // `local_entry` plus `new_cell` clamp + DDA init
                // around `entry_pos` produced an off-by-one cell
                // that DDA couldn't recover from at leaf boundaries
                // where sibling nodes meet. Trimming via the NODE's
                // t_enter puts `local_entry` on the node boundary —
                // the exact position the pre-AABB code used — so DDA
                // init stays byte-identical to the correct baseline.
                // We lose the "skip leading empty cells inside AABB"
                // micro-optimization, but the CULL win (whole-
                // descent skip on miss) stays, which is >90% of the
                // perf benefit anyway.
                let node_hit = ray_box(
                    ray_origin, inv_dir,
                    child_origin,
                    child_origin + vec3<f32>(3.0) * child_cell_size,
                );
                let ct_start = max(node_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
                let child_entry = ray_origin + ray_dir * ct_start;
                let local_entry = (child_entry - child_origin) / child_cell_size;

                // Instrumentation: count of descents the path-mask
                // cull would catch if enabled. An earlier experiment
                // promoted this to a real cull: it reduced avg_steps
                // 16% and avg_loads 10%, but delivered ZERO wall-
                // clock improvement on Apple Silicon because the GPU
                // was already memory-hiding the "wasted" descents.
                // Reverted to instrumentation-only; the counter stays
                // as a diagnostic for future perf investigations.
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
                // Load the new current-depth header into scalar
                // registers so the inner loop never re-reads tree[]
                // headers. `node_offsets` is the only per-descent
                // cross-buffer hop; subsequent per-cell accesses
                // stay in registers.
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
                // Use `entry_pos` as the reference (matching the root
                // init). In the baseline stacked version the root used
                // entry_pos and descent used ray_origin — that meant
                // side_dist values at different depths were offset by
                // t_start from each other. With a single scalar we
                // pick one reference; entry_pos keeps root behavior
                // byte-exact and shifts descent values by -t_start
                // (typically <0.001 when the camera is inside the
                // root box, negligible for the LOD-pixel check).
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

// Top-level march. Dispatches the current frame's Cartesian DDA,
// then on miss pops to the next ancestor in the ribbon and
// continues. When ribbon is exhausted, returns sky (hit=false).
//
// Each pop transforms the ray into the parent's frame coords:
// `parent_pos = slot_xyz + frame_pos / 3`, `parent_dir = frame_dir / 3`.
// The parent's frame cell still spans `[0, 3)³` in its own
// coords, so the inner DDA is unchanged — only the ray is
// rescaled and the buffer node_idx swapped.
fn march(world_ray_origin: vec3<f32>, world_ray_dir: vec3<f32>) -> HitResult {
    // UV-sphere dispatch (camera-inside-body case): when the render
    // frame IS the body, hand the ray straight to `march_uv_sphere`.
    // The camera-outside-body case is handled by mid-descent
    // dispatch from inside `march_cartesian`.
    if uniforms.root_kind == ROOT_KIND_UV_SPHERE_BODY {
        return march_uv_sphere(uniforms.root_index, world_ray_origin, world_ray_dir);
    }

    var ray_origin = world_ray_origin;
    var ray_dir = world_ray_dir;
    var current_idx = uniforms.root_index;
    var ribbon_level: u32 = 0u;
    var cur_scale: f32 = 1.0;

    // skip_slot: after a ribbon pop, the slot index (in the parent)
    // of the child we just left. march_cartesian skips this slot at
    // depth 0 to avoid re-entering the subtree already traversed by
    // the inner shell. Uses slot (not node_idx) for dedup correctness.
    var skip_slot: u32 = 0xFFFFFFFFu;

    var hops: u32 = 0u;
    loop {
        if hops > 80u { break; }
        hops = hops + 1u;

        // Cartesian frame: no depth cap beyond the hardware stack
        // ceiling. `LOD_PIXEL_THRESHOLD` (Nyquist) is the sole
        // visual LOD gate — rays stop descending when cells fall
        // below the pixel floor.
        var r: HitResult = march_cartesian(
            current_idx, ray_origin, ray_dir, MAX_STACK_DEPTH, skip_slot,
        );
        if r.hit {
            r.frame_level = ribbon_level;
            r.frame_scale = cur_scale;
            // r.t is FRAME-LOCAL t (ray_dir is kept at camera-frame
            // magnitude across pops, so each frame's inner DDA computes
            // a local t, bounded O(1)). Convert to camera-frame t for
            // the caller and for cell_min/cell_size anchoring.
            //   t_camera = t_frame / cur_scale   (cur_scale = 1/3^N)
            if cur_scale < 1.0 {
                let hit_popped = ray_origin + ray_dir * r.t;
                let cell_local = clamp(
                    (hit_popped - r.cell_min) / r.cell_size,
                    vec3<f32>(0.0), vec3<f32>(1.0),
                );
                r.t = r.t / cur_scale;
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
        // Single-level ribbon pop with empty-shell fast-exit.
        //
        // Pop exactly one ancestor entry, transform the ray into
        // the ancestor's [0,3)³ frame, then fall through to the
        // outer loop which re-enters march_cartesian. When the
        // ribbon entry's `siblings_all_empty` flag is set, every
        // slot of the ancestor other than the one we popped out
        // of is tag=0 — so the DDA would only traverse empty
        // cells. Skip it: ray_box to the shell exit, advance
        // ray_origin, and let the outer loop pop again.
        let entry = ribbon[ribbon_level];
        if ENABLE_STATS { ray_loads_ribbon = ray_loads_ribbon + 1u; }
        let s = entry.slot_bits & RIBBON_SLOT_MASK;
        let sx = i32(s % 3u);
        let sy = i32((s / 3u) % 3u);
        let sz = i32(s / 9u);
        let slot_off = vec3<f32>(f32(sx), f32(sy), f32(sz));
        skip_slot = s;
        // Ray pop: rescale origin into parent's [0,3)³, keep
        // ray_dir at camera-frame magnitude. The old scheme
        // divided ray_dir by 3 on every pop, which kept `t`
        // invariant across frames but caused ray_dir to
        // underflow after ~18 pops (3^-18 ≈ 6e-9). With
        // ray_dir preserved, each frame's DDA runs with O(1)
        // precision; t inside march_cartesian is frame-local.
        // Camera-frame t is recovered on hit return as
        // t_cam = t_frame / cur_scale.
        ray_origin = slot_off + ray_origin / 3.0;
        cur_scale = cur_scale * (1.0 / 3.0);
        current_idx = entry.node_idx;
        ribbon_level = ribbon_level + 1u;

        // Empty-shell fast exit: if every sibling is empty, skip
        // this shell's DDA and advance the ray to the shell's
        // exit boundary. Next outer iteration will pop again.
        let siblings_all_empty =
            (entry.slot_bits & RIBBON_SIBLINGS_ALL_EMPTY) != 0u;
        if siblings_all_empty {
            let inv_dir_shell = vec3<f32>(
                select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
                select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
                select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
            );
            let shell_hit = ray_box(
                ray_origin, inv_dir_shell,
                vec3<f32>(0.0), vec3<f32>(3.0),
            );
            if shell_hit.t_exit > 0.0 {
                // Advance past the shell boundary so the
                // next pop lands us OUTSIDE this shell's
                // [0,3)³ in grandparent coords.
                ray_origin = ray_origin + ray_dir * (shell_hit.t_exit + 0.001);
                if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
            }
        }
    }

    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = cur_scale;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;
    return result;
}
