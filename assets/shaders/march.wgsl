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

    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }
    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = ray_origin + ray_dir * t_start;
    let entry_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
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
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth -= 1u;
            cur_cell_size = cur_cell_size * 3.0;
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
        let child_cell_size = cur_cell_size / 3.0;
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
                clamp(i32(floor(local_entry.x)), 0, 2),
                clamp(i32(floor(local_entry.y)), 0, 2),
                clamp(i32(floor(local_entry.z)), 0, 2),
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
// frame's cell spans `[0, 3)³` in `ray_origin/ray_dir` coords.
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
            // Phase 2 X-wrap: at depth==0 inside a WrappedPlane root
            // frame, an X-only OOB wraps the ray instead of returning
            // miss. Y / Z OOB and depth>0 OOB take the existing
            // ribbon-pop / break path. The wrap shift is in slab-root
            // local units (`[0, 3)` frame), so f32 magnitudes stay
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
            let x_oob = cell.x < 0 || cell.x > 2;
            let yz_in = cell.y >= 0 && cell.y <= 2
                     && cell.z >= 0 && cell.z <= 2;
            if depth == 0u
                && uniforms.root_kind == ROOT_KIND_WRAPPED_PLANE
                && x_oob && yz_in
            {
                // Slab-root local cell size at the slab leaf level:
                // the WrappedPlane node spans `[0, 3)` and contains
                // `3^slab_depth` cells per axis. The wrap shift is
                // `dims_x * cell_size_at_slab_depth`. With Phase 2's
                // invariant `dims_x == 3^slab_depth`, this evaluates
                // to exactly 3.0 — i.e., the full WrappedPlane node
                // width. East OOB → shift west; west OOB → shift east.
                let dims_x = uniforms.slab_dims.x;
                let slab_depth_u = uniforms.slab_dims.w;
                let cell_size_slab = 3.0 / pow(3.0, f32(slab_depth_u));
                let wrap_shift = f32(dims_x) * cell_size_slab;
                let east_oob = cell.x > 2;
                let sign = select(1.0, -1.0, east_oob);
                ray_origin.x = ray_origin.x + sign * wrap_shift;

                // Re-enter the SAME root node from the opposite face.
                // cur_node_origin / cur_cell_size / s_node_idx[0] /
                // cur_occupancy / cur_first_child all stay unchanged
                // — only the entry point changes.
                let new_root_hit = ray_box(
                    ray_origin, inv_dir,
                    vec3<f32>(0.0), vec3<f32>(3.0),
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
                    clamp(i32(floor(entry_pos.x)), 0, 2),
                    clamp(i32(floor(entry_pos.y)), 0, 2),
                    clamp(i32(floor(entry_pos.z)), 0, 2),
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
            cur_cell_size = cur_cell_size * 3.0;
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
                // R for the slab: with dims_x = 3^slab_depth (the
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

// Hardware ceiling for the sphere anchor descent stack. Sized
// independently from `MAX_STACK_DEPTH` (the Cartesian frame stack,
// which is tuned at 8 for register pressure on Apple Silicon and
// works because Cartesian's frame-aware system pops between
// frames + LOD_PIXEL_THRESHOLD prunes descent). Sphere mode has
// neither — descent into a non-uniform anchor is one continuous
// stack — so this needs to cover any reasonable
// `cell_subtree_depth` (default 20) the user might edit into.
//
// Set to 24 to leave headroom above the 20-level default. Above
// this the descent splats representative (same shape as Cartesian's
// `at_max`). When edits exceed this, raise this constant; sphere
// mode is opt-in (--planet-render-sphere) so the extra register
// pressure only applies when sphere render is active.
const SPHERE_DESCENT_DEPTH: u32 = 24u;

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
// Walks `slab_depth` levels of 27-children Cartesian descent. At
// each level: integer-divide the cell coords by `3^(remaining_levels)`
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
        cells_per_slot = cells_per_slot * 3;
    }
    for (var level: u32 = 0u; level < slab_depth; level = level + 1u) {
        let sx = (cx / cells_per_slot) % 3;
        let sy = (cy / cells_per_slot) % 3;
        let sz = (cz / cells_per_slot) % 3;
        let slot = u32(sx + sy * 3 + sz * 9);

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
        cells_per_slot = cells_per_slot / 3;
    }
    return out;
}

// Closest-face UV bevel for a (lon, lat, r) cell.
//
// `in_cell` is the ray-hit position as a fractional offset within the
// current cell, on each of the 3 axes (lon, lat, r), in [0, 1]^3.
// Tracking this fraction through descent via the precision-stable
// multiply-by-3 trick (parent_frac * 3 - cell_idx) is what keeps the
// bevel sharp at deep depths — using `(lon_p - lon_lo_c)` directly
// would catastrophic-cancel two near-equal f32s and the bevel
// fractions degrade to noise once cell width drops below f32's
// ~1e-7 absolute precision (around descent depth 10).
//
// Face-arc weights still need physical sizes: we multiply in_cell by
// the cell's step in absolute (lon-radians, lat-radians, r-units),
// scaled by `r * cos(lat)` to get arc length on the sphere. Step
// scalars retain full f32 relative precision through descent (each
// push divides by 3, no subtractions of near-equals).
fn make_sphere_hit(
    pos: vec3<f32>, n_step: vec3<f32>, t_param: f32, inv_norm: f32,
    block_type: u32,
    r: f32, lat_p: f32,
    in_cell: vec3<f32>,
    lon_step_c: f32, lat_step_c: f32, r_step_c: f32,
) -> HitResult {
    var result: HitResult;
    let cos_lat = max(cos(lat_p), 1e-3);
    let arc_lon_lo = r * cos_lat * lon_step_c * in_cell.x;
    let arc_lon_hi = r * cos_lat * lon_step_c * (1.0 - in_cell.x);
    let arc_lat_lo = r * lat_step_c * in_cell.y;
    let arc_lat_hi = r * lat_step_c * (1.0 - in_cell.y);
    let arc_r_lo  = r_step_c * in_cell.z;
    let arc_r_hi  = r_step_c * (1.0 - in_cell.z);
    var best = arc_lon_lo;
    var axis: u32 = 0u;
    if arc_lon_hi < best { best = arc_lon_hi; axis = 0u; }
    if arc_lat_lo < best { best = arc_lat_lo; axis = 1u; }
    if arc_lat_hi < best { best = arc_lat_hi; axis = 1u; }
    if arc_r_lo  < best { best = arc_r_lo;  axis = 2u; }
    if arc_r_hi  < best { best = arc_r_hi;  axis = 2u; }

    var u_in_face: f32;
    var v_in_face: f32;
    if axis == 0u {
        u_in_face = in_cell.y;
        v_in_face = in_cell.z;
    } else if axis == 1u {
        u_in_face = in_cell.x;
        v_in_face = in_cell.z;
    } else {
        u_in_face = in_cell.x;
        v_in_face = in_cell.y;
    }
    let face_edge = min(
        min(u_in_face, 1.0 - u_in_face),
        min(v_in_face, 1.0 - v_in_face),
    );
    let shape = smoothstep(0.02, 0.14, face_edge);
    let bevel_strength = 0.7 + 0.3 * shape;

    result.hit = true;
    result.t = t_param * inv_norm;
    result.color = palette[block_type].rgb * bevel_strength;
    result.normal = n_step;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    // Neutralize main.wgsl::cube_face_bevel — see slab DDA hit comment.
    result.cell_min = pos - vec3<f32>(0.5);
    result.cell_size = 1.0;
    return result;
}

// Stack-based DDA inside an anchor block at sub-cell granularity.
// Mirrors `march_cartesian` but in (lon, lat, r) space using the
// sphere-cell boundary primitives (ray_meridian_t, ray_parallel_t,
// ray_sphere_after).
//
// Recipe — tree-structure descent (no camera-distance LOD):
// * 27-children descent — each level divides (lon, lat, r) cell sizes
//   by 3.
// * tag=1 → hit the leaf Block at this sub-cell.
// * tag=2 + non-empty representative → push child frame, continue
//   DDA at finer cells. (Pack format auto-flattens uniform Cartesian
//   subtrees to tag=1, so descent only enters genuinely non-uniform
//   subtrees — i.e. the path the user actually edited.)
// * tag=2 + empty representative → skip whole cell.
// * empty slot / unknown → advance ray to the cell's nearest face.
// * stack at MAX_STACK_DEPTH → splat representative (hardware ceiling).
//
// On exit (ray leaves the slab cell or stack underflows), returns
// hit=false so the caller's main slab DDA can continue past this
// slab cell.
fn sphere_descend_anchor(
    anchor_idx: u32,
    ray_origin: vec3<f32>, ray_dir: vec3<f32>,
    oc: vec3<f32>, cs_center: vec3<f32>, inv_norm: f32,
    t_in: f32, t_exit: f32,
    slab_lon_lo: f32, slab_lon_step: f32,
    slab_lat_lo: f32, slab_lat_step: f32,
    slab_r_lo:   f32, slab_r_step:   f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    var s_node_idx: array<u32, SPHERE_DESCENT_DEPTH>;
    var s_cell: array<u32, SPHERE_DESCENT_DEPTH>;
    var depth: u32 = 0u;
    s_node_idx[0] = anchor_idx;

    // ── State tracked through descent (Approach A) ───────────────────
    //
    // `cur_*_center` is the absolute (lon, lat, r) coord of the CURRENT
    // CELL's center. Maintained per-cell via integer-multiple updates:
    //   on push:    center += (new_cell_idx - 1) * new_step
    //   on pop:     center -= (popped_idx - 1) * old_step ; then *=3
    //   on advance: center += step_dir * cur_step (one of 6 faces)
    // Every update is an exact-step shift — never an absolute add of a
    // tiny fraction to a large base, so it doesn't accumulate ULP drift
    // the way `cur_*_org += cell.x * step` did.
    //
    // Cell bounds are derived as `center ± half_step`: half_step is
    // exact (step is divided by 3 each push, bit-exact in f32) and the
    // ± is symmetric, so the meridian/parallel/sphere primitives see
    // face positions without the cumulative drift the old `cur_*_org`
    // pattern produced.
    //
    // `frame_in_frac` is the ray's [0,1]³ position inside the CURRENT
    // FRAME's 3×3×3 grid. On push it's `parent_frac * 3 - cell_idx`
    // (bit-exact); on pop `(popped + child_frac) / 3`. At in-frame
    // advance we use FACE-CROSSING DETECTION (which `t_xxx == t_next`)
    // to step the cell index by ±1 along that axis — pure integer
    // math, immune to f32 cell-width-vs-coord-magnitude issues that
    // plagued `floor((lon_p - cur_lon_org) / step)`. The ray's
    // crossed-axis position at the new t is exactly the face's
    // boundary, so the crossed component of `frame_in_frac` snaps
    // to a precise integer-aligned fraction; non-crossed components
    // are left at their pre-advance value (the ray's other axes
    // didn't move enough within one cell to matter for bevel).
    // ─────────────────────────────────────────────────────────────────

    var cur_lon_step = slab_lon_step / 3.0;
    var cur_lat_step = slab_lat_step / 3.0;
    var cur_r_step   = slab_r_step   / 3.0;

    var t = t_in;

    // Initial cell + frame_in_frac at depth 0. Slab cell is large
    // (~0.077 rad in lon) so absolute math here has plenty of f32
    // precision (~5 decimal digits in the [0, 1] result).
    let pos0 = ray_origin + ray_dir * t;
    let off0 = pos0 - cs_center;
    let r0 = max(length(off0), 1e-9);
    let n0 = off0 / r0;
    let lat0 = asin(clamp(n0.y, -1.0, 1.0));
    let lon0 = atan2(n0.z, n0.x);
    var frame_in_frac = vec3<f32>(
        (lon0 - slab_lon_lo) / slab_lon_step,
        (r0   - slab_r_lo)   / slab_r_step,
        (lat0 - slab_lat_lo) / slab_lat_step,
    );
    let cell0 = clamp(vec3<i32>(
        i32(floor(frame_in_frac.x * 3.0)),
        i32(floor(frame_in_frac.y * 3.0)),
        i32(floor(frame_in_frac.z * 3.0)),
    ), vec3<i32>(0), vec3<i32>(2));
    s_cell[0] = pack_cell(cell0);

    // Cell center: absolute, but only updated via exact integer-step
    // shifts after this initial computation. No drift accumulation.
    var cur_lon_center = slab_lon_lo + (f32(cell0.x) + 0.5) * cur_lon_step;
    var cur_r_center   = slab_r_lo   + (f32(cell0.y) + 0.5) * cur_r_step;
    var cur_lat_center = slab_lat_lo + (f32(cell0.z) + 0.5) * cur_lat_step;

    var iters: u32 = 0u;
    loop {
        if iters > 256u { break; }
        iters = iters + 1u;
        if t > t_exit { break; }

        let cell = unpack_cell(s_cell[depth]);
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            // OOB at depth d: ray exited depth-d's frame. The exit axis
            // tells us which face of the parent's cell (depth d-1) we
            // crossed; we pop and advance the parent's cell by ±1 on
            // that axis.
            if depth == 0u { break; }
            let oob_x = select(0, select(1, -1, cell.x < 0), cell.x < 0 || cell.x > 2);
            let oob_y = select(0, select(1, -1, cell.y < 0), cell.y < 0 || cell.y > 2);
            let oob_z = select(0, select(1, -1, cell.z < 0), cell.z < 0 || cell.z > 2);
            let old_step_lon = cur_lon_step;
            let old_step_lat = cur_lat_step;
            let old_step_r   = cur_r_step;
            depth = depth - 1u;
            cur_lon_step = old_step_lon * 3.0;
            cur_lat_step = old_step_lat * 3.0;
            cur_r_step   = old_step_r   * 3.0;
            let popped = unpack_cell(s_cell[depth]);
            // popped is the cell at depth-d's parent that contained
            // the (depth+1) frame we just exited.
            // Its center, after popping, is shifted by `(popped - 1) *
            // old_step` from the (now-current) cur_*_center. Subtract
            // that to get parent's "popped cell" center, then add the
            // OOB step at parent step to advance to popped's neighbour.
            cur_lon_center = cur_lon_center
                - (f32(popped.x) - 1.0) * old_step_lon
                + f32(oob_x) * cur_lon_step;
            cur_r_center = cur_r_center
                - (f32(popped.y) - 1.0) * old_step_r
                + f32(oob_y) * cur_r_step;
            cur_lat_center = cur_lat_center
                - (f32(popped.z) - 1.0) * old_step_lat
                + f32(oob_z) * cur_lat_step;
            // Advance parent's cell index by the OOB direction.
            let new_cell_p = vec3<i32>(
                popped.x + oob_x,
                popped.y + oob_y,
                popped.z + oob_z,
            );
            s_cell[depth] = pack_cell(new_cell_p);
            // frame_in_frac on pop: `(popped + child_frac) / 3`. Then
            // on the advanced axis(es), snap to precise face fraction.
            frame_in_frac = (vec3<f32>(popped) + frame_in_frac) / 3.0;
            if oob_x < 0 { frame_in_frac.x = f32(new_cell_p.x + 1) / 3.0; }
            if oob_x > 0 { frame_in_frac.x = f32(new_cell_p.x) / 3.0; }
            if oob_y < 0 { frame_in_frac.y = f32(new_cell_p.y + 1) / 3.0; }
            if oob_y > 0 { frame_in_frac.y = f32(new_cell_p.y) / 3.0; }
            if oob_z < 0 { frame_in_frac.z = f32(new_cell_p.z + 1) / 3.0; }
            if oob_z > 0 { frame_in_frac.z = f32(new_cell_p.z) / 3.0; }
            continue;
        }

        let cell_f = vec3<f32>(cell);
        // Precision-stable in-cell fraction. Used for bevel.
        let in_cell = clamp(frame_in_frac * 3.0 - cell_f, vec3<f32>(0.0), vec3<f32>(1.0));

        // Cell bounds via center ± half_step. No accumulating drift.
        let half_lon = cur_lon_step * 0.5;
        let half_lat = cur_lat_step * 0.5;
        let half_r   = cur_r_step * 0.5;
        let lon_lo = cur_lon_center - half_lon;
        let lon_hi = cur_lon_center + half_lon;
        let r_lo   = cur_r_center   - half_r;
        let r_hi   = cur_r_center   + half_r;
        let lat_lo = cur_lat_center - half_lat;
        let lat_hi = cur_lat_center + half_lat;

        let slot = u32(cell.x + cell.y * 3 + cell.z * 9);
        let header_off = node_offsets[s_node_idx[depth]];
        let occ = tree[header_off];
        let bit = 1u << slot;

        // t to all 6 cell faces. t_next = smallest > current t.
        var t_next = t_exit + 1.0;
        let t_lon_lo = ray_meridian_t(oc, ray_dir, lon_lo, t);
        if t_lon_lo > 0.0 && t_lon_lo < t_next { t_next = t_lon_lo; }
        let t_lon_hi = ray_meridian_t(oc, ray_dir, lon_hi, t);
        if t_lon_hi > 0.0 && t_lon_hi < t_next { t_next = t_lon_hi; }
        let t_lat_lo = ray_parallel_t(oc, ray_dir, lat_lo, t);
        if t_lat_lo > 0.0 && t_lat_lo < t_next { t_next = t_lat_lo; }
        let t_lat_hi = ray_parallel_t(oc, ray_dir, lat_hi, t);
        if t_lat_hi > 0.0 && t_lat_hi < t_next { t_next = t_lat_hi; }
        let t_r_lo = ray_sphere_after(ray_origin, ray_dir, cs_center, r_lo, t);
        if t_r_lo > 0.0 && t_r_lo < t_next { t_next = t_r_lo; }
        let t_r_hi = ray_sphere_after(ray_origin, ray_dir, cs_center, r_hi, t);
        if t_r_hi > 0.0 && t_r_hi < t_next { t_next = t_r_hi; }

        // Helper: advance cell + center + frame_in_frac via face-
        // crossing detection. Used by every "skip this cell" branch.
        // Inlined as a closure-like block since WGSL has no closures.
        // (Each branch repeats the same 30 lines with a `continue`.)

        if (occ & bit) == 0u {
            // Empty slot. Step cell index ±1 on the crossed axis
            // (precision-stable integer math).
            var step_x: i32 = 0;
            var step_y: i32 = 0;
            var step_z: i32 = 0;
            if t_next == t_lon_lo { step_x = -1; }
            else if t_next == t_lon_hi { step_x = 1; }
            else if t_next == t_r_lo { step_y = -1; }
            else if t_next == t_r_hi { step_y = 1; }
            else if t_next == t_lat_lo { step_z = -1; }
            else if t_next == t_lat_hi { step_z = 1; }
            let new_cell = vec3<i32>(cell.x + step_x, cell.y + step_y, cell.z + step_z);
            s_cell[depth] = pack_cell(new_cell);
            cur_lon_center = cur_lon_center + f32(step_x) * cur_lon_step;
            cur_r_center   = cur_r_center   + f32(step_y) * cur_r_step;
            cur_lat_center = cur_lat_center + f32(step_z) * cur_lat_step;
            // Snap crossed axis frame_in_frac to the new cell's near face.
            if step_x < 0 { frame_in_frac.x = f32(new_cell.x + 1) / 3.0; }
            if step_x > 0 { frame_in_frac.x = f32(new_cell.x)     / 3.0; }
            if step_y < 0 { frame_in_frac.y = f32(new_cell.y + 1) / 3.0; }
            if step_y > 0 { frame_in_frac.y = f32(new_cell.y)     / 3.0; }
            if step_z < 0 { frame_in_frac.z = f32(new_cell.z + 1) / 3.0; }
            if step_z > 0 { frame_in_frac.z = f32(new_cell.z)     / 3.0; }
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            continue;
        }

        let first_child = tree[header_off + 1u];
        let rank = countOneBits(occ & (bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        let block_type = (packed >> 8u) & 0xFFFFu;

        if tag == 1u {
            // Block leaf. Hit at this sub-cell.
            //
            // Axis-convention swizzle: descent's frame_in_frac is in
            // (lon, r, lat) order to match the slab-tree slot layout
            // (cell.y = r, cell.z = lat). make_sphere_hit takes
            // (lon, lat, r) — swap y↔z.
            let pos_h = ray_origin + ray_dir * t;
            let off_h = pos_h - cs_center;
            let r_h = max(length(off_h), 1e-9);
            let n_h = off_h / r_h;
            let lat_h = asin(clamp(n_h.y, -1.0, 1.0));
            return make_sphere_hit(
                pos_h, n_h, t, inv_norm, block_type,
                r_h, lat_h,
                vec3<f32>(in_cell.x, in_cell.z, in_cell.y),
                cur_lon_step, cur_lat_step, cur_r_step,
            );
        }

        if tag != 2u {
            // EntityRef etc. — treat as empty for sphere subtree DDA.
            // Same advance shape as the empty-slot branch.
            var step_x: i32 = 0;
            var step_y: i32 = 0;
            var step_z: i32 = 0;
            if t_next == t_lon_lo { step_x = -1; }
            else if t_next == t_lon_hi { step_x = 1; }
            else if t_next == t_r_lo { step_y = -1; }
            else if t_next == t_r_hi { step_y = 1; }
            else if t_next == t_lat_lo { step_z = -1; }
            else if t_next == t_lat_hi { step_z = 1; }
            let new_cell = vec3<i32>(cell.x + step_x, cell.y + step_y, cell.z + step_z);
            s_cell[depth] = pack_cell(new_cell);
            cur_lon_center = cur_lon_center + f32(step_x) * cur_lon_step;
            cur_r_center   = cur_r_center   + f32(step_y) * cur_r_step;
            cur_lat_center = cur_lat_center + f32(step_z) * cur_lat_step;
            if step_x < 0 { frame_in_frac.x = f32(new_cell.x + 1) / 3.0; }
            if step_x > 0 { frame_in_frac.x = f32(new_cell.x)     / 3.0; }
            if step_y < 0 { frame_in_frac.y = f32(new_cell.y + 1) / 3.0; }
            if step_y > 0 { frame_in_frac.y = f32(new_cell.y)     / 3.0; }
            if step_z < 0 { frame_in_frac.z = f32(new_cell.z + 1) / 3.0; }
            if step_z > 0 { frame_in_frac.z = f32(new_cell.z)     / 3.0; }
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            continue;
        }

        // tag == 2u: non-uniform Node.
        if block_type == 0xFFFEu {
            // Subtree empty per `representative_block` — skip whole
            // cell. Same advance shape.
            var step_x: i32 = 0;
            var step_y: i32 = 0;
            var step_z: i32 = 0;
            if t_next == t_lon_lo { step_x = -1; }
            else if t_next == t_lon_hi { step_x = 1; }
            else if t_next == t_r_lo { step_y = -1; }
            else if t_next == t_r_hi { step_y = 1; }
            else if t_next == t_lat_lo { step_z = -1; }
            else if t_next == t_lat_hi { step_z = 1; }
            let new_cell = vec3<i32>(cell.x + step_x, cell.y + step_y, cell.z + step_z);
            s_cell[depth] = pack_cell(new_cell);
            cur_lon_center = cur_lon_center + f32(step_x) * cur_lon_step;
            cur_r_center   = cur_r_center   + f32(step_y) * cur_r_step;
            cur_lat_center = cur_lat_center + f32(step_z) * cur_lat_step;
            if step_x < 0 { frame_in_frac.x = f32(new_cell.x + 1) / 3.0; }
            if step_x > 0 { frame_in_frac.x = f32(new_cell.x)     / 3.0; }
            if step_y < 0 { frame_in_frac.y = f32(new_cell.y + 1) / 3.0; }
            if step_y > 0 { frame_in_frac.y = f32(new_cell.y)     / 3.0; }
            if step_z < 0 { frame_in_frac.z = f32(new_cell.z + 1) / 3.0; }
            if step_z > 0 { frame_in_frac.z = f32(new_cell.z)     / 3.0; }
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            continue;
        }

        // Stack ceiling — same as Cartesian's MAX_STACK_DEPTH gate.
        let at_max = depth + 1u >= SPHERE_DESCENT_DEPTH;
        if at_max {
            let pos_h = ray_origin + ray_dir * t;
            let off_h = pos_h - cs_center;
            let r_h = max(length(off_h), 1e-9);
            let n_h = off_h / r_h;
            let lat_h = asin(clamp(n_h.y, -1.0, 1.0));
            return make_sphere_hit(
                pos_h, n_h, t, inv_norm, block_type,
                r_h, lat_h,
                vec3<f32>(in_cell.x, in_cell.z, in_cell.y),
                cur_lon_step, cur_lat_step, cur_r_step,
            );
        }

        // Push child frame.
        let child_idx = tree[child_base + 1u];
        depth = depth + 1u;
        s_node_idx[depth] = child_idx;
        cur_lon_step = cur_lon_step / 3.0;
        cur_lat_step = cur_lat_step / 3.0;
        cur_r_step   = cur_r_step   / 3.0;

        // Precision-stable: child's in-frame frac = parent's in_cell.
        frame_in_frac = in_cell;

        // Pick entry sub-cell at depth d+1 from the (now-precise) frac.
        let cell_c = clamp(vec3<i32>(
            i32(floor(frame_in_frac.x * 3.0)),
            i32(floor(frame_in_frac.y * 3.0)),
            i32(floor(frame_in_frac.z * 3.0)),
        ), vec3<i32>(0), vec3<i32>(2));
        s_cell[depth] = pack_cell(cell_c);

        // New cell center: parent (the cell we just pushed into) was
        // at cur_*_center. Shift to new sub-cell:
        //   new_center = parent_center + (cell_c - 1) * new_step
        cur_lon_center = cur_lon_center + (f32(cell_c.x) - 1.0) * cur_lon_step;
        cur_r_center   = cur_r_center   + (f32(cell_c.y) - 1.0) * cur_r_step;
        cur_lat_center = cur_lat_center + (f32(cell_c.z) - 1.0) * cur_lat_step;
    }

    return result;
}

// Phase 3 REVISED — Step A.0+A.1: UV-sphere render of the
// WrappedPlane frame. Replaces the flat slab DDA when
// `uniforms.planet_render.x == 1.0`. The slab data is unchanged;
// what changes is the visual — instead of marching through a flat
// 2:1 rectangle, we ray-intersect an implied sphere of radius
// R = body_size / (2π) (= the slab's wrap circumference, since
// dims_x fully fills the X axis at slab_depth = log_3(dims_x)) and
// return a hit if the ray strikes the surface. Poles past `lat_max`
// (planet_render.y) are banned — the ray returns no-hit so the outer
// loop can pop / skip and the pixel reads as sky.
//
// Step A.0: geometry primitive only.
// Step A.1: map (lon, lat) → slab (cell_x, cell_z) at GRASS row
// (cell_y = dims_y - 1), look up cell, color by block_type.
// Parity-checkerboard tint added so even uniform-grass cells show
// the lat/lon → cell grid alignment unambiguously.
fn sphere_uv_in_cell(
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

    // Quadratic discriminant assumes unit ray_dir; the marcher
    // passes camera.forward + right·ndc + up·ndc which isn't unit.
    // Renormalise for the intersect — `t` returned to the caller is
    // in the same parameterisation (scaled by 1 / |ray_dir_in|).
    let ray_dir = normalize(ray_dir_in);

    // Sphere center: middle of the body cell.
    // Radius: body_size / (2π) — the slab's circumference is
    // exactly body_size (since dims_x fills X), so R = C / (2π).
    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let r_sphere = body_size / (2.0 * 3.14159265);

    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - r_sphere * r_sphere;
    let disc = b * b - c;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let hit_pos = ray_origin + ray_dir * t_enter;
    let n = (hit_pos - cs_center) / r_sphere;

    // A.3 — Full cell-by-cell DDA in spherical (lon, lat, r) cell
    // coords at slab granularity. At each step, sample the cell; if
    // empty, compute t to ALL 6 cell-boundary surfaces (2 meridians,
    // 2 parallels, 2 spheres) and step to the smallest crossing.
    //
    // Slab Y maps to radial: cy = dims_y - 1 at r_outer, cy = 0 at
    // r_inner. Slab X (longitude) wraps the full 2π; cells span
    // lon ∈ [-π, π]. Slab Z (latitude) is bounded to ±lat_max
    // (poles banned outside this band).
    //
    // On a non-empty slab cell we mirror march_cartesian's recipe:
    //  * tag=1 (uniform-flatten Block) → render Block at slab scale.
    //  * tag=2 (non-uniform Node) → LOD-gate. If sub-cells project
    //    above LOD_PIXEL_THRESHOLD, dispatch sphere_descend_anchor to
    //    walk the anchor's subtree at finer (lon, lat, r) granularity.
    //    Otherwise splat the representative_block at slab scale.
    let dims_x = i32(uniforms.slab_dims.x);
    let dims_y = i32(uniforms.slab_dims.y);
    let dims_z = i32(uniforms.slab_dims.z);
    let slab_depth = uniforms.slab_dims.w;
    let pi = 3.14159265;
    let shell_thickness = r_sphere * 0.25;
    let r_outer = r_sphere;
    let r_inner = r_sphere - shell_thickness;
    let inv_norm = 1.0 / max(length(ray_dir_in), 1e-6);
    let lon_step = 2.0 * pi / f32(dims_x);
    let lat_step = 2.0 * lat_max / f32(dims_z);
    let r_step = shell_thickness / f32(dims_y);

    var t = max(t_enter, 0.0) + 1e-5;
    var iters: u32 = 0u;
    loop {
        if t > t_exit { break; }
        if iters > 256u { break; }
        iters = iters + 1u;

        // Current cell (cell_x, cell_y, cell_z) at the ray's t.
        let pos = ray_origin + ray_dir * t;
        let off = pos - cs_center;
        let r = length(off);
        if r < r_inner || r > r_outer + 1e-3 { break; }
        let n_step = off / r;
        let lat_p = asin(clamp(n_step.y, -1.0, 1.0));
        if abs(lat_p) > lat_max { break; }
        let lon_p = atan2(n_step.z, n_step.x);

        let u_p = (lon_p + pi) / (2.0 * pi);
        let v_p = (lat_p + lat_max) / (2.0 * lat_max);
        let r_frac = (r - r_inner) / shell_thickness;
        let cell_x = clamp(i32(floor(u_p * f32(dims_x))), 0, dims_x - 1);
        let cell_z = clamp(i32(floor(v_p * f32(dims_z))), 0, dims_z - 1);
        let cell_y = clamp(i32(floor(r_frac * f32(dims_y))), 0, dims_y - 1);

        // Slab cell bounds in (lon, lat, r).
        let lon_lo = -pi + f32(cell_x) * lon_step;
        let lon_hi = lon_lo + lon_step;
        let lat_lo = -lat_max + f32(cell_z) * lat_step;
        let lat_hi = lat_lo + lat_step;
        let r_lo = r_inner + f32(cell_y) * r_step;
        let r_hi = r_lo + r_step;

        let sample = sample_slab_cell(body_idx, slab_depth, cell_x, cell_y, cell_z);
        if sample.block_type != 0xFFFEu {
            // tag=2 (non-uniform anchor) ⇒ recursive descent into the
            // anchor's subtree. NO camera-distance LOD gate: in sphere
            // mode the render frame is locked at WrappedPlane (the
            // camera doesn't deepen its frame as anchor_depth grows
            // the way Cartesian does), so a "cell_size/ray_dist <
            // threshold" check would collapse deep edits to the
            // representative — exactly the bug we're fixing. Descend
            // until tag=1 (uniform-flatten Block leaf) or stack max.
            // Tree structure ALONE drives descent — same principle as
            // Cartesian (the LOD gate there only works because frame
            // shifting keeps cells at "normal voxel size" relative to
            // the camera).
            if sample.tag == 2u {
                let sub = sphere_descend_anchor(
                    sample.child_idx,
                    ray_origin, ray_dir,
                    oc, cs_center, inv_norm,
                    t, t_exit,
                    lon_lo, lon_step, lat_lo, lat_step, r_lo, r_step,
                );
                if sub.hit { return sub; }
                // Anchor descent exited without a hit — every sub-cell
                // along the ray's chord was empty. Advance past the
                // slab cell.
                var t_n = t_exit + 1.0;
                let tn_lon_lo = ray_meridian_t(oc, ray_dir, lon_lo, t);
                if tn_lon_lo > 0.0 && tn_lon_lo < t_n { t_n = tn_lon_lo; }
                let tn_lon_hi = ray_meridian_t(oc, ray_dir, lon_hi, t);
                if tn_lon_hi > 0.0 && tn_lon_hi < t_n { t_n = tn_lon_hi; }
                let tn_lat_lo = ray_parallel_t(oc, ray_dir, lat_lo, t);
                if tn_lat_lo > 0.0 && tn_lat_lo < t_n { t_n = tn_lat_lo; }
                let tn_lat_hi = ray_parallel_t(oc, ray_dir, lat_hi, t);
                if tn_lat_hi > 0.0 && tn_lat_hi < t_n { t_n = tn_lat_hi; }
                let tn_r_lo = ray_sphere_after(ray_origin, ray_dir, cs_center, r_lo, t);
                if tn_r_lo > 0.0 && tn_r_lo < t_n { t_n = tn_r_lo; }
                let tn_r_hi = ray_sphere_after(ray_origin, ray_dir, cs_center, r_hi, t);
                if tn_r_hi > 0.0 && tn_r_hi < t_n { t_n = tn_r_hi; }
                if t_n >= t_exit { break; }
                t = t_n + max(r_step * 1e-4, 1e-6);
                continue;
            }
            // tag=1 (uniform-flatten): the entire anchor subtree is
            // one Block — render at slab cell scale, no descent
            // needed (uniform Cartesian subtrees pack-flatten so the
            // shader sees one Block at any zoom).
            //
            // Slab cell is large (~0.077 in lon-radians); subtractions
            // here have plenty of f32 precision. Pass in_cell directly
            // so make_sphere_hit's bevel uses precision-stable
            // arc = step * frac math.
            let in_cell_slab = vec3<f32>(
                clamp((lon_p - lon_lo) / lon_step, 0.0, 1.0),
                clamp((lat_p - lat_lo) / lat_step, 0.0, 1.0),
                clamp((r     - r_lo)   / r_step,   0.0, 1.0),
            );
            return make_sphere_hit(
                pos, n_step, t, inv_norm, sample.block_type,
                r, lat_p,
                in_cell_slab,
                lon_step, lat_step, r_step,
            );
        }

        // Empty cell — advance to the nearest cell-face crossing.
        var t_next = t_exit + 1.0;
        let t_lon_lo = ray_meridian_t(oc, ray_dir, lon_lo, t);
        if t_lon_lo > 0.0 && t_lon_lo < t_next { t_next = t_lon_lo; }
        let t_lon_hi = ray_meridian_t(oc, ray_dir, lon_hi, t);
        if t_lon_hi > 0.0 && t_lon_hi < t_next { t_next = t_lon_hi; }
        let t_lat_lo = ray_parallel_t(oc, ray_dir, lat_lo, t);
        if t_lat_lo > 0.0 && t_lat_lo < t_next { t_next = t_lat_lo; }
        let t_lat_hi = ray_parallel_t(oc, ray_dir, lat_hi, t);
        if t_lat_hi > 0.0 && t_lat_hi < t_next { t_next = t_lat_hi; }
        let t_r_lo = ray_sphere_after(ray_origin, ray_dir, cs_center, r_lo, t);
        if t_r_lo > 0.0 && t_r_lo < t_next { t_next = t_r_lo; }
        let t_r_hi = ray_sphere_after(ray_origin, ray_dir, cs_center, r_hi, t);
        if t_r_hi > 0.0 && t_r_hi < t_next { t_next = t_r_hi; }

        if t_next >= t_exit { break; }
        t = t_next + max(r_step * 1e-4, 1e-6);
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

        // Phase 3 REVISED — frame dispatch on NodeKind. When the
        // current frame root is a `WrappedPlane` AND sphere-render
        // mode is enabled, replace the flat slab DDA with the
        // analytical UV-sphere path. Otherwise (Cartesian frame, or
        // sphere-render disabled) use the regular Cartesian DDA.
        // `kind == 1u` matches `ROOT_KIND_WRAPPED_PLANE` /
        // `GpuNodeKind::WrappedPlane` (`from_node_kind`).
        var r: HitResult;
        let cur_kind = node_kinds[current_idx].kind;
        if cur_kind == 1u && uniforms.planet_render.x > 0.5 {
            r = sphere_uv_in_cell(
                current_idx, vec3<f32>(0.0), 3.0,
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
