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

// Conservative 8-bit "path mask" — the tensor product of per-axis
// 3-bit masks of cells reachable from `entry_cell` moving in `step`
// direction. Over-approximates the actual ray path (any axis-wise
// reachable cell triple, not only the specific 3D path the ray
// traces). Safe for occupancy-intersection culling: if the full
// superset misses all occupied slots, the actual path certainly
// does. Used for instrumentation only right now — does not affect
// traversal.
fn path_mask_conservative(entry_cell: vec3<i32>, step: vec3<i32>) -> u32 {
    let ec = vec3<u32>(
        u32(clamp(entry_cell.x, 0, 1)),
        u32(clamp(entry_cell.y, 0, 1)),
        u32(clamp(entry_cell.z, 0, 1)),
    );
    // Per-axis 3-bit mask. step > 0: bits [ec..2]; step < 0: bits
    // [0..ec]. step is always ±1 in march_cartesian (non-zero).
    let mx: u32 = select((1u << (ec.x + 1u)) - 1u, (7u << ec.x) & 7u, step.x > 0);
    let my: u32 = select((1u << (ec.y + 1u)) - 1u, (7u << ec.y) & 7u, step.y > 0);
    let mz: u32 = select((1u << (ec.z + 1u)) - 1u, (7u << ec.z) & 7u, step.z > 0);
    // Smear each 3-bit axis mask into its 8-bit "axis active"
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
// `[0, 2)³` local frame. WGSL's no-recursion rule forces this to
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
    curvature_offset: f32,
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
    walker_probe.curvature_offset_bits = bitcast<u32>(curvature_offset);
}

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
    s_node_idx[0] = root_node_idx;

    let root_header_off = node_offsets[root_node_idx];
    var cur_occupancy: u32 = tree[root_header_off];
    var cur_first_child: u32 = tree[root_header_off + 1u];

    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(2.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }
    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = ray_origin + ray_dir * t_start;
    let entry_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 1),
        clamp(i32(floor(entry_pos.y)), 0, 1),
        clamp(i32(floor(entry_pos.z)), 0, 1),
    );
    s_cell[0] = pack_cell(entry_cell);
    let cell_f = vec3<f32>(entry_cell);
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
        let cell = unpack_cell(s_cell[depth]);
        if cell.x < 0 || cell.x > 1 || cell.y < 0 || cell.y > 1 || cell.z < 0 || cell.z > 1 {
            if depth == 0u { break; }
            depth -= 1u;
            cur_cell_size = cur_cell_size * 2.0;
            let popped = unpack_cell(s_cell[depth]);
            cur_node_origin = cur_node_origin - vec3<f32>(popped) * cur_cell_size;
            let lc_pop = vec3<f32>(popped);
            cur_side_dist = vec3<f32>(
                select((cur_node_origin.x + lc_pop.x * cur_cell_size - entry_pos.x) * inv_dir.x,
                       (cur_node_origin.x + (lc_pop.x + 1.0) * cur_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                select((cur_node_origin.y + lc_pop.y * cur_cell_size - entry_pos.y) * inv_dir.y,
                       (cur_node_origin.y + (lc_pop.y + 1.0) * cur_cell_size - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
                select((cur_node_origin.z + lc_pop.z * cur_cell_size - entry_pos.z) * inv_dir.z,
                       (cur_node_origin.z + (lc_pop.z + 1.0) * cur_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
            );
            let parent_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];
            let m_oob = min_axis_mask(cur_side_dist);
            let advanced = popped + vec3<i32>(m_oob) * step;
            s_cell[depth] = pack_cell(advanced);
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
        }
        // tag == 2u: Cartesian Node descent. tag==3 (EntityRef) is
        // treated as miss inside entity subtrees — no entities-
        // inside-entities for now.
        if tag != 2u {
            let m_skip = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_skip) * step);
            cur_side_dist += m_skip * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_skip;
            continue;
        }
        let child_idx = tree[child_base + 1u];
        let child_bt = (packed >> 8u) & 0xFFFFu;
        if child_bt == 0xFFFEu {  // REPRESENTATIVE_EMPTY (matches gpu::pack u16 sentinel)
            let m_rep = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_rep) * step);
            cur_side_dist += m_rep * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_rep;
            continue;
        }

        let at_max = depth + 1u >= MAX_STACK_DEPTH;
        let child_cell_size = cur_cell_size / 2.0;
        let min_side = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
        let ray_dist = max(min_side * ray_metric, 0.001);
        let lod_pixels = child_cell_size / ray_dist
            * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
        let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;
        if at_max || at_lod {
            let bt = child_bt;
            if bt == 0xFFFEu || bt == 0xFFFDu {  // REPRESENTATIVE_EMPTY / ENTITY_REPRESENTATIVE
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
            let ct_start = max(root_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
            let child_entry = ray_origin + ray_dir * ct_start;
            let local_entry = (child_entry - child_origin) / child_cell_size;
            depth += 1u;
            s_node_idx[depth] = child_idx;
            cur_node_origin = child_origin;
            cur_cell_size = child_cell_size;
            let child_header_off = node_offsets[child_idx];
            cur_occupancy = tree[child_header_off];
            cur_first_child = tree[child_header_off + 1u];
            let child_cell_i = vec3<i32>(
                clamp(i32(floor(local_entry.x)), 0, 1),
                clamp(i32(floor(local_entry.y)), 0, 1),
                clamp(i32(floor(local_entry.z)), 0, 1),
            );
            s_cell[depth] = pack_cell(child_cell_i);
            let lc = vec3<f32>(child_cell_i);
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
    return result;
}

// Cartesian DDA in a single frame rooted at `root_node_idx`. The
// frame's cell spans `[0, 2)³` in `ray_origin/ray_dir` coords.
// Returns hit on cell terminal; on miss (ray exits the frame),
// returns hit=false so the caller can pop to the ancestor ribbon.
//
// X-wrap branch (Phase 2): when `uniforms.root_kind ==
// ROOT_KIND_WRAPPED_PLANE` and the ray exits the root cell
// purely on the X axis at depth==0, the marcher translates
// `ray_origin.x` by ±wrap_shift (in slab-root local units) and
// re-enters the same root from the opposite face instead of
// returning a miss. Y / Z OOB and depth>0 OOB still ribbon-pop.
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

    // Current-depth cell size. Pure function of `depth` (1/2^depth), so
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

    // Phase 3 Step 3.0 — per-depth Y-bend stack. `s_y_drop[d]` is the
    // parabolic-drop value applied at descent into depth d:
    //   drop = ct_at_descent² · uniforms.curvature.x
    // Used at side_dist init / OOB pop so the DDA's Y crossings stay
    // consistent with the bent `local_entry.y` selection at descent.
    // 0.0 at depth 0 (root has no descent-time bend); set on every
    // deeper descend; read on pop. Within a single cell the DDA walks
    // straight (linear approximation) — fine for small cells; the
    // bend re-applies whenever we re-descend.
    var s_y_drop: array<f32, MAX_STACK_DEPTH>;
    s_y_drop[0] = 0.0;

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

    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(2.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }

    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    // Mutable: refreshed by the X-wrap branch when ray_origin
    // translates on a wrap (the side_dist recomputes use entry_pos
    // as the reference point, so wrap must update both).
    var entry_pos: vec3<f32> = ray_origin + ray_dir * t_start;

    let root_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 1),
        clamp(i32(floor(entry_pos.y)), 0, 1),
        clamp(i32(floor(entry_pos.z)), 0, 1),
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

        if cell.x < 0 || cell.x > 1 || cell.y < 0 || cell.y > 1 || cell.z < 0 || cell.z > 1 {
            // Phase 2 X-wrap: at depth==0 inside a WrappedPlane root
            // frame, an X-only OOB wraps the ray instead of returning
            // miss. Y / Z OOB and depth>0 OOB take the existing
            // ribbon-pop / break path. The wrap shift is in slab-root
            // local units (`[0, 2)` frame), so f32 magnitudes stay
            // bounded — same precision discipline as the descent /
            // ribbon-pop math.
            //
            // Gating: depth==0 AND root_kind==WRAPPED_PLANE AND only
            // the X axis is OOB. Y or Z OOB simultaneously means the
            // ray crossed a corner of the slab and should pop normally
            // (the slab's Y / Z faces are not wrapped). When the
            // camera is OUTSIDE a slab the render frame is Cartesian
            // (root_kind != WRAPPED_PLANE), so this branch can't fire
            // in that case — wrap stays scoped to inside-slab views.
            let x_oob = cell.x < 0 || cell.x > 1;
            let yz_in = cell.y >= 0 && cell.y <= 2
                     && cell.z >= 0 && cell.z <= 2;
            if depth == 0u
                && uniforms.root_kind == ROOT_KIND_WRAPPED_PLANE
                && x_oob && yz_in
            {
                // Slab-root local cell size at the slab leaf level:
                // the WrappedPlane node spans `[0, 2)` and contains
                // `2^slab_depth` cells per axis. The wrap shift is
                // `dims_x * cell_size_at_slab_depth`. With Phase 2's
                // invariant `dims_x == 2^slab_depth`, this evaluates
                // to exactly 3.0 — i.e., the full WrappedPlane node
                // width. East OOB → shift west; west OOB → shift east.
                let dims_x = uniforms.slab_dims.x;
                let slab_depth_u = uniforms.slab_dims.w;
                let cell_size_slab = 2.0 / pow(2.0, f32(slab_depth_u));
                let wrap_shift = f32(dims_x) * cell_size_slab;
                let east_oob = cell.x > 1;
                let sign = select(1.0, -1.0, east_oob);
                ray_origin.x = ray_origin.x + sign * wrap_shift;

                // Re-enter the SAME root node from the opposite face.
                // cur_node_origin / cur_cell_size / s_node_idx[0] /
                // cur_occupancy / cur_first_child all stay unchanged
                // — only the entry point changes.
                let new_root_hit = ray_box(
                    ray_origin, inv_dir,
                    vec3<f32>(0.0), vec3<f32>(2.0),
                );
                if new_root_hit.t_enter >= new_root_hit.t_exit
                    || new_root_hit.t_exit < 0.0
                {
                    // Geometrically impossible after a valid wrap
                    // (the new origin is off-axis but ray_dir.x is
                    // unchanged, so the slab box is still hit). Bail
                    // defensively if it ever happens.
                    break;
                }
                let new_t_start = max(new_root_hit.t_enter, 0.0) + 0.001;
                entry_pos = ray_origin + ray_dir * new_t_start;
                let new_root_cell = vec3<i32>(
                    clamp(i32(floor(entry_pos.x)), 0, 1),
                    clamp(i32(floor(entry_pos.y)), 0, 1),
                    clamp(i32(floor(entry_pos.z)), 0, 1),
                );
                s_cell[0] = pack_cell(new_root_cell);
                let cf_w = vec3<f32>(new_root_cell);
                cur_side_dist = vec3<f32>(
                    select((cf_w.x - entry_pos.x) * inv_dir.x,
                           (cf_w.x + 1.0 - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                    select((cf_w.y - entry_pos.y) * inv_dir.y,
                           (cf_w.y + 1.0 - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
                    select((cf_w.z - entry_pos.z) * inv_dir.z,
                           (cf_w.z + 1.0 - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
                );
                if ENABLE_STATS { ray_steps_oob = ray_steps_oob + 1u; }
                continue;
            }
            if depth == 0u { break; }
            depth -= 1u;
            cur_cell_size = cur_cell_size * 2.0;
            let parent_cell = unpack_cell(s_cell[depth]);
            cur_node_origin = cur_node_origin - vec3<f32>(parent_cell) * cur_cell_size;
            let lc_pop = vec3<f32>(parent_cell);
            // Y-axis side_dist references the parent's stored bend
            // (popped depth = `depth` after the decrement above).
            // Drop is 0.0 at depth 0, so popping to root recovers the
            // un-bent crossings byte-for-byte.
            let bent_entry_y_p = entry_pos.y - s_y_drop[depth];
            cur_side_dist = vec3<f32>(
                select((cur_node_origin.x + lc_pop.x * cur_cell_size - entry_pos.x) * inv_dir.x,
                       (cur_node_origin.x + (lc_pop.x + 1.0) * cur_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                select((cur_node_origin.y + lc_pop.y * cur_cell_size - bent_entry_y_p) * inv_dir.y,
                       (cur_node_origin.y + (lc_pop.y + 1.0) * cur_cell_size - bent_entry_y_p) * inv_dir.y, ray_dir.y >= 0.0),
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
            // Walker probe (tag==1 hit). Curvature offset is the
            // parabolic-drop value the bend math WOULD apply at this
            // hit's t — `result.t² · A`. Computed but NOT yet applied
            // to the cell selection (Step 3.0 pre-bend phase: we use
            // the probe to verify the math before changing what the
            // marcher does). Once the probe values match the CPU
            // reference at known camera positions, Step 3.0 applies
            // the bend — until then, the marcher is bit-identical
            // to the flat path.
            let horiz_dir_sq_h = ray_dir.x * ray_dir.x + ray_dir.z * ray_dir.z;
            let curvature_offset = result.t * result.t * horiz_dir_sq_h * uniforms.curvature.x;
            write_walker_probe(
                1u, iterations, depth, cell,
                cur_node_origin, cur_cell_size,
                result.t, normal, 1u, curvature_offset,
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

            // Transform ray into the entity subtree's [0, 2)³
            // local frame and descend. WGSL's no-recursion rule
            // forces a separate `march_entity_subtree` walker.
            let scale3 = vec3<f32>(2.0) / bbox_size;
            let local_origin = (ray_origin - entity.bbox_min) * scale3;
            let local_dir = ray_dir * scale3;
            let sub = march_entity_subtree(entity.subtree_bfs, local_origin, local_dir);
            if sub.hit {
                let size_over_3 = bbox_size * (1.0 / 2.0);
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
            let cell_slot = u32(cell.x) + u32(cell.y) * 2u + u32(cell.z) * 4u;
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

            // TangentBlock dispatch — frame-local rotation around the
            // cube's geometric centre (1.0, 1.0, 1.0). NO world-space
            // coordinates: the ray is re-expressed in the child's
            // [0, 2)³ via (ray_origin - child_origin) / cur_cell_size,
            // then rotated by the stored R^T around the cube centre.
            // On hit, normal is rotated back via R · local_normal.
            if node_kinds[child_idx].kind == NODE_KIND_TANGENT_BLOCK {
                let child_origin_tb = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
                // Scale maps the slot's parent extent (size cur_cell_size)
                // into the child's [0, 2)³ local frame: 3 / cur_cell_size.
                let scale = 2.0 / cur_cell_size;
                let local_pre_origin = (ray_origin - child_origin_tb) * scale;
                let local_pre_dir = ray_dir * scale;
                // Stored rotation R has columns rc0/rc1/rc2.
                // (R^T · v).i = dot(rc_i, v).
                let rc0 = node_kinds[child_idx].rot_col0.xyz;
                let rc1 = node_kinds[child_idx].rot_col1.xyz;
                let rc2 = node_kinds[child_idx].rot_col2.xyz;
                let centered = local_pre_origin - vec3<f32>(1.0);
                let rotated = vec3<f32>(
                    dot(rc0, centered),
                    dot(rc1, centered),
                    dot(rc2, centered),
                );
                let local_origin = rotated + vec3<f32>(1.0);
                let local_dir = vec3<f32>(
                    dot(rc0, local_pre_dir),
                    dot(rc1, local_pre_dir),
                    dot(rc2, local_pre_dir),
                );
                let sub = march_in_tangent_cube(child_idx, local_origin, local_dir);
                if sub.hit {
                    let local_hit = local_origin + local_dir * sub.t;
                    let local_in_cell = clamp(
                        (local_hit - sub.cell_min) / sub.cell_size,
                        vec3<f32>(0.0), vec3<f32>(1.0),
                    );
                    let local_bevel = cube_face_bevel(local_in_cell, sub.normal);
                    var out: HitResult;
                    out.hit = true;
                    // The scale factor applies to both origin and dir,
                    // so the parameter t is preserved across the
                    // transform — sub.t is the world ray parameter.
                    out.t = sub.t;
                    out.color = sub.color * (0.7 + 0.3 * local_bevel);
                    // Rotate normal back to outer frame: world = R · local.
                    // (R · v) = rc0·v.x + rc1·v.y + rc2·v.z.
                    out.normal = rc0 * sub.normal.x
                               + rc1 * sub.normal.y
                               + rc2 * sub.normal.z;
                    out.frame_level = 0u;
                    out.frame_scale = 1.0;
                    let hit_world = ray_origin + ray_dir * sub.t;
                    out.cell_min = hit_world - vec3<f32>(0.5);
                    out.cell_size = 1.0;
                    return out;
                }
                // Cube missed — advance DDA past this slot.
                let m_tb = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_tb) * step);
                cur_side_dist += m_tb * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_tb;
                continue;
            }

            // Cartesian Node: depth/LOD check, then descend.
            // depth_limit = MAX_STACK_DEPTH — LOD controls the
            // effective depth, not an artificial per-shell budget.
            let at_max = depth + 1u > depth_limit || depth + 1u >= MAX_STACK_DEPTH;
            let child_cell_size = cur_cell_size / 2.0;
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
                // [0, 2)^3 so behavior matches the pre-AABB code.
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
                    vec3<f32>(2.0),
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
                    child_origin + vec3<f32>(2.0) * child_cell_size,
                );
                let ct_start = max(node_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
                let child_entry = ray_origin + ray_dir * ct_start;
                // Phase 3 Step 3.0 — parabolic Y-bend. The drop is
                // proportional to HORIZONTAL distance squared, not
                // total ray distance — a vertical ray (looking
                // straight down) has zero horizontal travel and
                // therefore zero bend, which is the correct
                // geometry for "ray over planet center → no
                // curvature offset". A horizontal ray at altitude
                // gets the full bend.
                //
                //   horiz_dist(t) = t · sqrt(ray_dir.x² + ray_dir.z²)
                //   drop          = horiz_dist² · A
                //                 = t² · (ray_dir.x² + ray_dir.z²) · A
                //
                // Recorded in `s_y_drop[depth+1]` so the side_dist
                // init below references a bent-Y entry_pos. Cell
                // selection and Y-plane crossings stay consistent at
                // this depth. On pop, we recompute side_dist using
                // the parent's stored drop.
                let horiz_dir_sq = ray_dir.x * ray_dir.x + ray_dir.z * ray_dir.z;
                let raw_drop = ct_start * ct_start * horiz_dir_sq * uniforms.curvature.x;
                // Phase 3 Step 3.0 Option 1 — clamp drop to 2R.
                // Beyond a drop of 2R (= the planet's diameter), the
                // bent ray has gone "behind" the planet — further
                // drop is meaningless and only causes the ray to
                // re-hit the slab via X-wrap from the other side
                // ("ghost slab" artifact at high altitude). Capping
                // the drop pulls those rays out of wrap range:
                // the bent_y stays at -2R below the surface, far
                // enough below the slab's vertical extent that the
                // OOB pop fires (slab Y is bounded, no Y-wrap) and
                // the ray escapes to sky.
                //
                // R for the slab: with dims_x = 2^slab_depth (the
                // fully-fills-X invariant), slab X extent in frame
                // units = 3.0, so R_frame = 3.0 / (2π) ≈ 0.477.
                // Hardcoded for now; if curvature is meaningful only
                // when uniforms.slab_dims.w > 0, we compute it as
                // (3.0 * cell_size_at_slab) / (2π) = 3 / (2π) for
                // the canonical slab. Plain world has no implied
                // radius — the caller passes A=0 to disable.
                let r_frame = 3.0 / (2.0 * 3.14159265);
                let curvature_drop = min(raw_drop, 2.0 * r_frame);
                let bent_child_entry_y = child_entry.y - curvature_drop;
                let local_entry = vec3<f32>(
                    (child_entry.x - child_origin.x) / child_cell_size,
                    (bent_child_entry_y - child_origin.y) / child_cell_size,
                    (child_entry.z - child_origin.z) / child_cell_size,
                );

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
                s_y_drop[depth] = curvature_drop;
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
                    clamp(i32(floor(local_entry.x)), 0, 1),
                    clamp(i32(floor(local_entry.y)), 0, 1),
                    clamp(i32(floor(local_entry.z)), 0, 1),
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
                // Y-axis side_dist references the BENT entry-y for
                // this depth (= entry_pos.y - s_y_drop[depth]). This
                // keeps Y crossings consistent with the bent
                // local_entry.y above. X / Z use the un-bent
                // entry_pos because the bend is Y-only.
                let bent_entry_y_d = entry_pos.y - s_y_drop[depth];
                cur_side_dist = vec3<f32>(
                    select((child_origin.x + lc.x * child_cell_size - entry_pos.x) * inv_dir.x,
                           (child_origin.x + (lc.x + 1.0) * child_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                    select((child_origin.y + lc.y * child_cell_size - bent_entry_y_d) * inv_dir.y,
                           (child_origin.y + (lc.y + 1.0) * child_cell_size - bent_entry_y_d) * inv_dir.y, ray_dir.y >= 0.0),
                    select((child_origin.z + lc.z * child_cell_size - entry_pos.z) * inv_dir.z,
                           (child_origin.z + (lc.z + 1.0) * child_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
                );
            }
        }
    }

    return result;
}

// Result of a slab-cell tree walk. `tag` mirrors pack-format:
//   0 = empty / no child at this slot
//   1 = uniform-flatten Block (`block_type` is the leaf material)
//   2 = non-uniform Node (`block_type` is the representative_block;
//       `child_idx` is the BFS index of the anchor's subtree, ready
//       for sub-cell DDA descent).
struct SlabSample {
    block_type: u32,
    tag: u32,
    child_idx: u32,
};

// Phase 3 REVISED Step A.1 — sample the slab tree at (cx, cy, cz).
//
// Walks `slab_depth` levels of 8-children Cartesian descent. At
// each level: integer-divide the cell coords by `2^(remaining_levels)`
// to get the slot index, look up the slab tree's child entry there.
// At tag=1 (uniform-flatten Block leaf) → return that material.
// At tag=2 (non-uniform Node) at the LAST level → return its
// representative_block AND child_idx so the caller can decide whether
// to LOD-splat or descend into the subtree.
// On tag=2 mid-walk → descend one level. tag=0 / unknown → empty.
fn sample_slab_cell(
    slab_root_idx: u32,
    slab_depth: u32,
    cx: i32, cy: i32, cz: i32,
) -> SlabSample {
    var out: SlabSample;
    out.block_type = 0xFFFEu;
    out.tag = 0u;
    out.child_idx = 0u;
    var idx = slab_root_idx;
    var cells_per_slot: i32 = 1;
    for (var k: u32 = 1u; k < slab_depth; k = k + 1u) {
        cells_per_slot = cells_per_slot * 2;
    }
    for (var level: u32 = 0u; level < slab_depth; level = level + 1u) {
        let sx = (cx / cells_per_slot) % 2;
        let sy = (cy / cells_per_slot) % 2;
        let sz = (cz / cells_per_slot) % 2;
        let slot = u32(sx + sy * 2 + sz * 4);

        let header_off = node_offsets[idx];
        let occ = tree[header_off];
        let bit = 1u << slot;
        if (occ & bit) == 0u { return out; }
        let first_child = tree[header_off + 1u];
        let rank = countOneBits(occ & (bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        let block_type = (packed >> 8u) & 0xFFFFu;

        if tag == 1u {
            out.block_type = block_type;
            out.tag = 1u;
            return out;
        }
        if level == slab_depth - 1u {
            out.block_type = block_type;
            out.tag = tag;
            if tag == 2u {
                out.child_idx = tree[child_base + 1u];
            }
            return out;
        }
        if tag == 2u {
            idx = tree[child_base + 1u];
        } else {
            return out;
        }
        cells_per_slot = cells_per_slot / 2;
    }
    return out;
}

// Tangent-cube DDA. Used by `march_wrapped_planet` after the ray has
// been transformed into a slab cell's local `[0, 2)³` frame.
//
// Mirrors `march_cartesian`'s core stack-based DDA but with a
// deeper stack to accommodate `cell_subtree_depth` without LOD-
// pixel termination (the ray's WrappedPlane-local distance to
// in-cube leaves is sub-pixel from a far camera, so a Nyquist-
// bounded stack would collapse deep edits to the representative
// — exactly the bug we're fixing). Stack 24 covers the wrapped-
// planet's default `cell_subtree_depth = 20` plus margin; deeper
// edits past 24 levels splat the rep as a last-resort terminal.
//
// Simplified vs `march_cartesian`: no entity dispatch, no X-wrap,
// no Y-curvature, no walker probe, no LOD termination. Pure tree-
// structure descent.
const TANGENT_STACK_DEPTH: u32 = 24u;

fn march_in_tangent_cube(
    root_node_idx: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>,
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
    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );
    let delta_dist = abs(inv_dir);

    var s_node_idx: array<u32, TANGENT_STACK_DEPTH>;
    var s_cell: array<u32, TANGENT_STACK_DEPTH>;
    var cur_cell_size: f32 = 1.0;
    var cur_node_origin: vec3<f32> = vec3<f32>(0.0);
    var cur_side_dist: vec3<f32>;
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;
    s_node_idx[0] = root_node_idx;

    let root_header_off = node_offsets[root_node_idx];
    var cur_occupancy: u32 = tree[root_header_off];
    var cur_first_child: u32 = tree[root_header_off + 1u];

    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(2.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }

    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = ray_origin + ray_dir * t_start;
    let root_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 1),
        clamp(i32(floor(entry_pos.y)), 0, 1),
        clamp(i32(floor(entry_pos.z)), 0, 1),
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

    var iterations: u32 = 0u;
    loop {
        if iterations >= 2048u { break; }
        iterations = iterations + 1u;

        let cell = unpack_cell(s_cell[depth]);

        if cell.x < 0 || cell.x > 1 || cell.y < 0 || cell.y > 1 || cell.z < 0 || cell.z > 1 {
            if depth == 0u { break; }
            depth = depth - 1u;
            cur_cell_size = cur_cell_size * 2.0;
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

            let parent_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];

            let m_oob = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(parent_cell + vec3<i32>(m_oob) * step);
            cur_side_dist = cur_side_dist + m_oob * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_oob;
            continue;
        }

        let slot = u32(cell.x + cell.y * 2 + cell.z * 4);
        let slot_bit = 1u << slot;
        if (cur_occupancy & slot_bit) == 0u {
            let m_empty = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_empty) * step);
            cur_side_dist = cur_side_dist + m_empty * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_empty;
            continue;
        }

        let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;

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
        }

        if tag != 2u {
            // Unknown tag (EntityRef etc.) — treat as empty for the
            // tangent walk; entities don't live inside tangent cubes.
            let m_other = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_other) * step);
            cur_side_dist = cur_side_dist + m_other * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_other;
            continue;
        }

        // tag == 2u: Node child. Skip whole cell when subtree empty.
        let child_bt = (packed >> 8u) & 0xFFFFu;
        if child_bt == 0xFFFEu {
            let m_rep = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_rep) * step);
            cur_side_dist = cur_side_dist + m_rep * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_rep;
            continue;
        }

        let child_idx = tree[child_base + 1u];

        // Stack ceiling — only fires when cell_subtree_depth exceeds
        // TANGENT_STACK_DEPTH. Splat representative as a last-resort
        // terminal so the cell is still visible (just at lower
        // resolution than the user edited at).
        let at_max = depth + 1u >= TANGENT_STACK_DEPTH;
        if at_max {
            let cell_min_h = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
            let cell_max_h = cell_min_h + vec3<f32>(cur_cell_size);
            let cell_box_h = ray_box(ray_origin, inv_dir, cell_min_h, cell_max_h);
            result.hit = true;
            result.t = max(cell_box_h.t_enter, 0.0);
            result.color = palette[child_bt].rgb;
            result.normal = normal;
            result.cell_min = cell_min_h;
            result.cell_size = cur_cell_size;
            return result;
        }

        // Descend into the Node child. Use NODE box (not the AABB)
        // for entry trim — same reasoning as march_cartesian.
        let child_origin = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
        let child_cell_size = cur_cell_size / 2.0;
        let node_hit = ray_box(
            ray_origin, inv_dir,
            child_origin,
            child_origin + vec3<f32>(2.0) * child_cell_size,
        );
        let ct_start = max(node_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
        let child_entry = ray_origin + ray_dir * ct_start;
        let local_entry = vec3<f32>(
            (child_entry.x - child_origin.x) / child_cell_size,
            (child_entry.y - child_origin.y) / child_cell_size,
            (child_entry.z - child_origin.z) / child_cell_size,
        );

        depth = depth + 1u;
        s_node_idx[depth] = child_idx;
        cur_node_origin = child_origin;
        cur_cell_size = child_cell_size;
        let child_header_off = node_offsets[child_idx];
        cur_occupancy = tree[child_header_off];
        cur_first_child = tree[child_header_off + 1u];

        let new_cell = vec3<i32>(
            clamp(i32(floor(local_entry.x)), 0, 1),
            clamp(i32(floor(local_entry.y)), 0, 1),
            clamp(i32(floor(local_entry.z)), 0, 1),
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

    return result;
}

// Per-cell tangent-cube basis. A slab cell at index (cell_x, cy, cell_z)
// is rendered as one rotated cartesian cube placed tangent to the
// implied sphere at the cell's (lon_c, lat_c, r_c) center. Basis
// (east, normal, north) and origin/side are derived analytically
// from the integer cell index — no spherical traversal primitive,
// no compounding precision loss.
struct TangentCubeFrame {
    east_w: vec3<f32>,
    normal_w: vec3<f32>,
    north_w: vec3<f32>,
    origin_w: vec3<f32>,
    side: f32,
}

fn tangent_cube_frame_for_cell(
    cell_x: i32, cy: i32, cell_z: i32,
    cs_center: vec3<f32>, r_sphere: f32, lat_max: f32,
    lon_step: f32, lat_step: f32, r_step: f32, r_inner: f32,
) -> TangentCubeFrame {
    let pi = 3.14159265;
    let lat_c = -lat_max + (f32(cell_z) + 0.5) * lat_step;
    let lon_c = -pi + (f32(cell_x) + 0.5) * lon_step;
    let r_c = r_inner + (f32(cy) + 0.5) * r_step;
    let sl = sin(lat_c);
    let cl = cos(lat_c);
    let so = sin(lon_c);
    let co = cos(lon_c);
    var f: TangentCubeFrame;
    f.normal_w = vec3<f32>(cl * co, sl, cl * so);
    f.east_w = vec3<f32>(-so, 0.0, co);
    f.north_w = vec3<f32>(-sl * co, cl, -sl * so);
    f.origin_w = cs_center + r_c * f.normal_w;
    let east_arc = r_sphere * abs(cl) * lon_step;
    let north_arc = r_sphere * lat_step;
    f.side = max(max(east_arc, north_arc), r_step);
    return f;
}

// Render the WrappedPlane as a sphere of rotated cartesian tangent
// cubes. Each visible slab cell is one rotated cube; rendering of
// each cube uses the precision-stable cartesian DDA in
// `march_in_tangent_cube`. There is no spherical primitive: the
// outer logic is a constant-time per-radial-layer entry-seeding
// pass that finds *which* cube the ray enters, and dispatches
// march_in_tangent_cube once per occupied layer. Iterations are
// independent — no per-step compounding precision loss.
//
// Adjacent rotated cubes don't tile perfectly on a sphere, so this
// path leaves small slivers of sky between cubes at extreme zoom.
// That's a deliberate trade — the high-zoom-as-flat-wrapped-plane
// path will eventually take over before the slivers become visible.
fn march_wrapped_planet(
    body_idx: u32,
    body_origin: vec3<f32>,
    body_size: f32,
    ray_origin: vec3<f32>,
    ray_dir_in: vec3<f32>,
    lat_max: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    // Renormalise the ray for the entry quadratic. `t` returned to
    // the caller is in the original parameterisation, so multiply
    // by `inv_norm = 1 / |ray_dir_in|`.
    let ray_dir = normalize(ray_dir_in);
    let inv_norm = 1.0 / max(length(ray_dir_in), 1e-6);

    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let r_sphere = body_size / (2.0 * 3.14159265);

    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let oc_dot_oc = dot(oc, oc);
    let c = oc_dot_oc - r_sphere * r_sphere;
    let disc_outer = b * b - c;
    if disc_outer <= 0.0 { return result; }
    let sq_outer = sqrt(disc_outer);
    let t_exit_sphere = -b + sq_outer;
    if t_exit_sphere <= 0.0 { return result; }

    let dims_x = i32(uniforms.slab_dims.x);
    let dims_y = i32(uniforms.slab_dims.y);
    let dims_z = i32(uniforms.slab_dims.z);
    let slab_depth = uniforms.slab_dims.w;
    let pi = 3.14159265;
    let lon_step = 2.0 * pi / f32(dims_x);
    let lat_step = 2.0 * lat_max / f32(dims_z);
    let shell_thickness = r_sphere * 0.25;
    let r_inner = r_sphere - shell_thickness;
    let r_step = shell_thickness / f32(dims_y);

    var cy = dims_y - 1;
    loop {
        if cy < 0 { break; }

        let r_layer = r_inner + (f32(cy) + 1.0) * r_step;
        let cr = oc_dot_oc - r_layer * r_layer;
        let disc_layer = b * b - cr;
        if disc_layer < 0.0 { cy = cy - 1; continue; }
        let sq_layer = sqrt(disc_layer);
        let t_layer = -b - sq_layer;
        if t_layer < 0.0 || t_layer > t_exit_sphere {
            cy = cy - 1; continue;
        }

        let pos = ray_origin + ray_dir * t_layer;
        let n = (pos - cs_center) / r_layer;
        let lat = asin(clamp(n.y, -1.0, 1.0));
        if abs(lat) > lat_max { cy = cy - 1; continue; }
        let lon = atan2(n.z, n.x);
        let u = (lon + pi) / (2.0 * pi);
        let v = (lat + lat_max) / (2.0 * lat_max);
        let cell_x = clamp(i32(floor(u * f32(dims_x))), 0, dims_x - 1);
        let cell_z = clamp(i32(floor(v * f32(dims_z))), 0, dims_z - 1);

        let sample = sample_slab_cell(body_idx, slab_depth, cell_x, cy, cell_z);
        if sample.block_type == 0xFFFEu {
            cy = cy - 1; continue;
        }

        let cube = tangent_cube_frame_for_cell(
            cell_x, cy, cell_z,
            cs_center, r_sphere, lat_max,
            lon_step, lat_step, r_step, r_inner,
        );
        let scale = 2.0 / cube.side;
        let d_origin = ray_origin - cube.origin_w;
        let local_origin = vec3<f32>(
            dot(cube.east_w, d_origin) * scale + 1.0,
            dot(cube.normal_w, d_origin) * scale + 1.0,
            dot(cube.north_w, d_origin) * scale + 1.0,
        );
        let local_dir = vec3<f32>(
            dot(cube.east_w, ray_dir) * scale,
            dot(cube.normal_w, ray_dir) * scale,
            dot(cube.north_w, ray_dir) * scale,
        );

        if sample.tag == 2u {
            let sub = march_in_tangent_cube(sample.child_idx, local_origin, local_dir);
            if sub.hit {
                let local_hit = local_origin + local_dir * sub.t;
                let local_in_cell = clamp(
                    (local_hit - sub.cell_min) / sub.cell_size,
                    vec3<f32>(0.0), vec3<f32>(1.0),
                );
                let local_bevel = cube_face_bevel(local_in_cell, sub.normal);
                var out: HitResult;
                out.hit = true;
                out.t = sub.t * inv_norm;
                out.color = sub.color * (0.7 + 0.3 * local_bevel);
                out.normal = cube.east_w * sub.normal.x
                           + cube.normal_w * sub.normal.y
                           + cube.north_w * sub.normal.z;
                out.frame_level = 0u;
                out.frame_scale = 1.0;
                let hit_world = ray_origin + ray_dir * sub.t;
                out.cell_min = hit_world - vec3<f32>(0.5);
                out.cell_size = 1.0;
                return out;
            }
        } else if sample.tag == 1u {
            // Pack-time uniform-flatten: a Cartesian ancestor of this
            // slab cell collapsed to one Block. Render as a uniform-
            // coloured tangent cube (no subtree to descend).
            let inv_local = vec3<f32>(
                select(1e10, 1.0 / local_dir.x, abs(local_dir.x) > 1e-8),
                select(1e10, 1.0 / local_dir.y, abs(local_dir.y) > 1e-8),
                select(1e10, 1.0 / local_dir.z, abs(local_dir.z) > 1e-8),
            );
            let cube_box = ray_box(local_origin, inv_local, vec3<f32>(0.0), vec3<f32>(2.0));
            if cube_box.t_enter < cube_box.t_exit && cube_box.t_exit > 0.0 {
                let t_local = max(cube_box.t_enter, 0.0);
                let entry_local = local_origin + local_dir * t_local;
                let dx_lo = abs(entry_local.x - 0.0);
                let dx_hi = abs(entry_local.x - 2.0);
                let dy_lo = abs(entry_local.y - 0.0);
                let dy_hi = abs(entry_local.y - 2.0);
                let dz_lo = abs(entry_local.z - 0.0);
                let dz_hi = abs(entry_local.z - 2.0);
                var best = dx_lo;
                var local_normal = vec3<f32>(-1.0, 0.0, 0.0);
                if dx_hi < best { best = dx_hi; local_normal = vec3<f32>(1.0, 0.0, 0.0); }
                if dy_lo < best { best = dy_lo; local_normal = vec3<f32>(0.0, -1.0, 0.0); }
                if dy_hi < best { best = dy_hi; local_normal = vec3<f32>(0.0, 1.0, 0.0); }
                if dz_lo < best { best = dz_lo; local_normal = vec3<f32>(0.0, 0.0, -1.0); }
                if dz_hi < best { best = dz_hi; local_normal = vec3<f32>(0.0, 0.0, 1.0); }
                let local_in_cell = clamp(entry_local / 2.0, vec3<f32>(0.0), vec3<f32>(1.0));
                let local_bevel = cube_face_bevel(local_in_cell, local_normal);
                var out: HitResult;
                out.hit = true;
                out.t = t_local * inv_norm;
                out.color = palette[sample.block_type].rgb * (0.7 + 0.3 * local_bevel);
                out.normal = cube.east_w * local_normal.x
                           + cube.normal_w * local_normal.y
                           + cube.north_w * local_normal.z;
                out.frame_level = 0u;
                out.frame_scale = 1.0;
                let hit_world = ray_origin + ray_dir * t_local;
                out.cell_min = hit_world - vec3<f32>(0.5);
                out.cell_size = 1.0;
                return out;
            }
        }
        cy = cy - 1;
    }

    return result;
}

// Top-level march. Dispatches the current frame's Cartesian DDA,
// then on miss pops to the next ancestor in the ribbon and
// continues. When ribbon is exhausted, returns sky (hit=false).
//
// Each pop transforms the ray into the parent's frame coords:
// `parent_pos = slot_xyz + frame_pos / 3`, `parent_dir = frame_dir / 3`.
// The parent's frame cell still spans `[0, 2)³` in its own
// coords, so the inner DDA is unchanged — only the ray is
// rescaled and the buffer node_idx swapped.
fn march(world_ray_origin: vec3<f32>, world_ray_dir: vec3<f32>) -> HitResult {
    var ray_origin = world_ray_origin;
    var ray_dir = world_ray_dir;
    var current_idx = uniforms.root_index;
    var ribbon_level: u32 = 0u;
    var cur_scale: f32 = 1.0;

    // TangentBlock frame root: the camera position and basis are in
    // the TB's unrotated local frame, but the shader needs to see
    // content from the SAME rotated perspective as rays entering
    // from outside (which apply R^T at the TB boundary). Apply R^T
    // to ray_origin (centered at [1.0,1.0,1.0]) so the Cartesian
    // DDA traces through the TB in the rotated view. ray_dir is
    // already R^T-rotated by the CPU's frame_path_rotation basis.
    // On ribbon pop, R reverses this (my ribbon TB fix).
    let frame_root_kind = node_kinds[current_idx].kind;
    if frame_root_kind == NODE_KIND_TANGENT_BLOCK {
        let rc0 = node_kinds[current_idx].rot_col0.xyz;
        let rc1 = node_kinds[current_idx].rot_col1.xyz;
        let rc2 = node_kinds[current_idx].rot_col2.xyz;
        let centered = ray_origin - vec3<f32>(1.0);
        ray_origin = vec3<f32>(1.0) + vec3<f32>(
            dot(rc0, centered),
            dot(rc1, centered),
            dot(rc2, centered),
        );
    }

    // skip_slot: after a ribbon pop, the slot index (in the parent)
    // of the child we just left. march_cartesian skips this slot at
    // depth 0 to avoid re-entering the subtree already traversed by
    // the inner shell. Uses slot (not node_idx) for dedup correctness.
    var skip_slot: u32 = 0xFFFFFFFFu;

    var hops: u32 = 0u;
    loop {
        if hops > 80u { break; }
        hops = hops + 1u;

        // Frame dispatch on NodeKind. WrappedPlane (kind == 1) always
        // renders as a sphere of rotated tangent cubes via
        // `march_wrapped_planet` — no spherical primitive in the
        // traversal, just per-cell rotation. All other kinds fall
        // through to the cartesian DDA.
        var r: HitResult;
        let cur_kind = node_kinds[current_idx].kind;
        if cur_kind == 1u {
            r = march_wrapped_planet(
                current_idx, vec3<f32>(0.0), 2.0,
                ray_origin, ray_dir,
                uniforms.planet_render.y,
            );
        } else {
            // Cartesian frame: no depth cap beyond the hardware stack
            // ceiling. `LOD_PIXEL_THRESHOLD` (Nyquist) is the sole
            // visual LOD gate — rays stop descending when cells fall
            // below the pixel floor.
            r = march_cartesian(
                current_idx, ray_origin, ray_dir, MAX_STACK_DEPTH, skip_slot,
            );
        }
        if r.hit {
            r.frame_level = ribbon_level;
            r.frame_scale = cur_scale;
            // r.t is FRAME-LOCAL t (ray_dir is kept at camera-frame
            // magnitude across pops, so each frame's inner DDA computes
            // a local t, bounded O(1)). Convert to camera-frame t for
            // the caller and for cell_min/cell_size anchoring.
            //   t_camera = t_frame / cur_scale   (cur_scale = 1/2^N)
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
        let sx = i32(s % 2u);
        let sy = i32((s / 2u) % 2u);
        let sz = i32(s / 4u);
        let slot_off = vec3<f32>(f32(sx), f32(sy), f32(sz));
        skip_slot = s;
        // Ray pop: rescale origin into parent's [0,3)³.
        // TangentBlock children need rotation R applied on pop
        // (R maps child-local to parent frame).
        let child_kind = node_kinds[entry.child_bfs].kind;
        if child_kind == NODE_KIND_TANGENT_BLOCK {
            let rc0 = node_kinds[entry.child_bfs].rot_col0.xyz;
            let rc1 = node_kinds[entry.child_bfs].rot_col1.xyz;
            let rc2 = node_kinds[entry.child_bfs].rot_col2.xyz;
            let scaled = ray_origin / 2.0;
            let centered = scaled - vec3<f32>(0.5);
            ray_origin = slot_off + vec3<f32>(0.5)
                       + rc0 * centered.x + rc1 * centered.y + rc2 * centered.z;
            ray_dir = rc0 * ray_dir.x + rc1 * ray_dir.y + rc2 * ray_dir.z;
        } else {
            ray_origin = slot_off + ray_origin / 2.0;
        }
        cur_scale = cur_scale * (1.0 / 2.0);
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
                vec3<f32>(0.0), vec3<f32>(2.0),
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
