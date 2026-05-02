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
    root_node_idx: u32, ray_origin_in: vec3<f32>, ray_dir_in: vec3<f32>,
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
    // ray_dir / inv_dir / step / delta_dist are mutated on the
    // TangentBlock rotation push (Path B: ray scaled + rotated so the
    // rotated cube spans [0, 3)³ in local coords, matching the
    // unrotated DDA's cell-at-integer convention exactly). They stay
    // constant for non-rotated descents — `var` lets us swap them
    // in place on push and restore on pop without duplicating the
    // DDA loop.
    var ray_dir: vec3<f32> = ray_dir_in;
    var inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );
    // After ribbon pops, ray_dir magnitude shrinks (÷3 per pop).
    // LOD pixel calculations need world-space distances, so scale
    // side_dist by ray_metric to get actual distance.
    let ray_metric = max(length(ray_dir), 1e-6);
    var step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );
    var delta_dist = abs(inv_dir);

    // TangentBlock rotation state. `rot_active` flips true when the
    // DDA crosses into a `NodeKind::TangentBlock` child. The ray is
    // scaled + rotated so the rotated cube becomes [0, 3)³ in local
    // coords — inside, the DDA runs byte-identical to the unrotated
    // path (cells at integer positions, cur_cell_size = 1.0). On hit,
    // cell_min / normal are mapped back to world via the saved cube
    // transform. V1 supports a single rotation in the descent stack;
    // nested TangentBlocks beyond the first descend without applying
    // further rotation.
    var rot_active: bool = false;
    var rot_pushed_at_depth: u32 = 0u;
    var saved_ray_origin: vec3<f32>;
    var saved_ray_dir: vec3<f32>;
    var saved_inv_dir: vec3<f32>;
    var saved_step: vec3<i32>;
    var saved_delta_dist: vec3<f32>;
    var saved_node_origin: vec3<f32>;
    var saved_cell_size: f32;
    // Saved at push for hit-time conversion: the rotated cube's
    // origin (= child_origin in WORLD coords) and edge length (=
    // parent_cell_size in world units). cell_min_world =
    //   saved_cube_origin + M * cell_min_local * (saved_cube_size / 3)
    // where M's columns are uniforms.tangent_rotation_col0/1/2.
    var saved_cube_origin: vec3<f32>;
    var saved_cube_size: f32;

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
            // TangentBlock rotation pop. When OOB at depth ==
            // rot_pushed_at_depth + 1, we're popping out of the
            // rotated subtree — restore the saved parent-frame ray
            // vars + cur_node_origin / cur_cell_size directly. The
            // standard pop math below assumes axis-aligned ÷3 scaling
            // and would corrupt the rotated frame's local origin.
            if rot_active && depth == rot_pushed_at_depth + 1u {
                depth -= 1u;
                ray_origin = saved_ray_origin;
                ray_dir = saved_ray_dir;
                inv_dir = saved_inv_dir;
                step = saved_step;
                delta_dist = saved_delta_dist;
                cur_node_origin = saved_node_origin;
                cur_cell_size = saved_cell_size;
                rot_active = false;
                let parent_header_off = node_offsets[s_node_idx[depth]];
                cur_occupancy = tree[parent_header_off];
                cur_first_child = tree[parent_header_off + 1u];
                if ENABLE_STATS {
                    ray_loads_offsets = ray_loads_offsets + 1u;
                    ray_loads_tree = ray_loads_tree + 2u;
                    ray_steps_oob = ray_steps_oob + 1u;
                }
                let parent_cell_r = unpack_cell(s_cell[depth]);
                let lc_pop_r = vec3<f32>(parent_cell_r);
                let bent_entry_y_r = entry_pos.y - s_y_drop[depth];
                cur_side_dist = vec3<f32>(
                    select((cur_node_origin.x + lc_pop_r.x * cur_cell_size - entry_pos.x) * inv_dir.x,
                           (cur_node_origin.x + (lc_pop_r.x + 1.0) * cur_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                    select((cur_node_origin.y + lc_pop_r.y * cur_cell_size - bent_entry_y_r) * inv_dir.y,
                           (cur_node_origin.y + (lc_pop_r.y + 1.0) * cur_cell_size - bent_entry_y_r) * inv_dir.y, ray_dir.y >= 0.0),
                    select((cur_node_origin.z + lc_pop_r.z * cur_cell_size - entry_pos.z) * inv_dir.z,
                           (cur_node_origin.z + (lc_pop_r.z + 1.0) * cur_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
                );
                let m_oob_r = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(parent_cell_r + vec3<i32>(m_oob_r) * step);
                cur_side_dist += m_oob_r * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_oob_r;
                continue;
            }
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
            // Inside a rotated subtree, cell_min and normal are in
            // the scaled-rotated local frame (cube spans [0, 3)³,
            // cur_cell_size = 1/3^k where k is depth past the push).
            // Map back to world via M (column form) and the saved
            // cube origin / size: world_from_local(p) = cube_origin
            // + M·p · (cube_size / 3). cell_size scales by the same
            // (cube_size / 3) factor; t is preserved by the ray
            // scaling on push.
            if rot_active {
                let c0 = uniforms.tangent_rotation_col0.xyz;
                let c1 = uniforms.tangent_rotation_col1.xyz;
                let c2 = uniforms.tangent_rotation_col2.xyz;
                let world_per_local = saved_cube_size / 3.0;
                let n_l = normal;
                result.normal = c0 * n_l.x + c1 * n_l.y + c2 * n_l.z;
                let m_min = c0 * cell_min_h.x + c1 * cell_min_h.y + c2 * cell_min_h.z;
                result.cell_min = saved_cube_origin + m_min * world_per_local;
                result.cell_size = cur_cell_size * world_per_local;
            } else {
                result.normal = normal;
                result.cell_min = cell_min_h;
                result.cell_size = cur_cell_size;
            }
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
                    if rot_active {
                        let c0 = uniforms.tangent_rotation_col0.xyz;
                        let c1 = uniforms.tangent_rotation_col1.xyz;
                        let c2 = uniforms.tangent_rotation_col2.xyz;
                        let world_per_local = saved_cube_size / 3.0;
                        let n_l = normal;
                        result.normal = c0 * n_l.x + c1 * n_l.y + c2 * n_l.z;
                        let m_min = c0 * cell_min_l.x + c1 * cell_min_l.y + c2 * cell_min_l.z;
                        result.cell_min = saved_cube_origin + m_min * world_per_local;
                        result.cell_size = cur_cell_size * world_per_local;
                    } else {
                        result.normal = normal;
                        result.cell_min = cell_min_l;
                        result.cell_size = cur_cell_size;
                    }
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

                // TangentBlock rotation push (Path B: scale ray so the
                // rotated cube spans [0,3)³ in local coords). Detect
                // descent into a `NodeKind::TangentBlock` child (and
                // not already inside a rotated subtree — V1 supports a
                // single rotation in the descent stack).
                let is_tangent = !rot_active
                              && child_idx < arrayLength(&node_kinds)
                              && node_kinds[child_idx].kind == NODE_KIND_TANGENT_BLOCK;
                if is_tangent {
                    if ENABLE_STATS { ray_steps_node_descend = ray_steps_node_descend + 1u; }
                    // Save state for restoration on pop / hit-time
                    // conversion of cell_min and normal back to world.
                    saved_ray_origin = ray_origin;
                    saved_ray_dir = ray_dir;
                    saved_inv_dir = inv_dir;
                    saved_step = step;
                    saved_delta_dist = delta_dist;
                    saved_node_origin = cur_node_origin;
                    saved_cell_size = cur_cell_size;
                    saved_cube_origin = child_origin;       // world
                    saved_cube_size = cur_cell_size;         // world
                    rot_active = true;
                    rot_pushed_at_depth = depth;

                    // Rotation columns: M * v_rotated = v_parent.
                    // World → rotated uses Mᵀ (= dot products against
                    // the columns). Both ray_origin and ray_dir are
                    // scaled by `scale = 3 / cube_size_world` so the
                    // rotated cube becomes [0, 3)³ in local coords.
                    // Scaling both keeps the t-parameter invariant
                    // (verified: pos_local(t) = scale·Mᵀ·(pos_world(t)
                    // − cube_origin) is linear in t with the same
                    // coefficient on `t`).
                    let c0 = uniforms.tangent_rotation_col0.xyz;
                    let c1 = uniforms.tangent_rotation_col1.xyz;
                    let c2 = uniforms.tangent_rotation_col2.xyz;
                    let scale = 3.0 / saved_cube_size;
                    let dx = ray_origin - saved_cube_origin;
                    ray_origin = vec3<f32>(dot(c0, dx), dot(c1, dx), dot(c2, dx)) * scale;
                    ray_dir = vec3<f32>(
                        dot(c0, ray_dir), dot(c1, ray_dir), dot(c2, ray_dir),
                    ) * scale;
                    inv_dir = vec3<f32>(
                        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
                        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
                        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
                    );
                    step = vec3<i32>(
                        select(-1, 1, ray_dir.x >= 0.0),
                        select(-1, 1, ray_dir.y >= 0.0),
                        select(-1, 1, ray_dir.z >= 0.0),
                    );
                    delta_dist = abs(inv_dir);

                    // Recompute entry_pos in scaled-rotated frame —
                    // ray vs the rotated cube's [0,3)³ box.
                    let cube_hit = ray_box(ray_origin, inv_dir,
                                           vec3<f32>(0.0), vec3<f32>(3.0));
                    if cube_hit.t_enter >= cube_hit.t_exit || cube_hit.t_exit < 0.0 {
                        // Geometrically the parent AABB cull already
                        // confirmed the world AABB is hit; if the
                        // rotated cube fails the local box test it's
                        // a precision edge (very oblique grazing ray).
                        // Restore + advance the parent DDA.
                        ray_origin = saved_ray_origin;
                        ray_dir = saved_ray_dir;
                        inv_dir = saved_inv_dir;
                        step = saved_step;
                        delta_dist = saved_delta_dist;
                        rot_active = false;
                        let m_miss = min_axis_mask(cur_side_dist);
                        s_cell[depth] = pack_cell(cell + vec3<i32>(m_miss) * step);
                        cur_side_dist += m_miss * delta_dist * cur_cell_size;
                        normal = -vec3<f32>(step) * m_miss;
                        continue;
                    }
                    let cube_t_start = max(cube_hit.t_enter, 0.0) + 0.001;
                    entry_pos = ray_origin + ray_dir * cube_t_start;

                    depth += 1u;
                    s_node_idx[depth] = child_idx;
                    s_y_drop[depth] = 0.0;
                    cur_node_origin = vec3<f32>(0.0);
                    cur_cell_size = 1.0;
                    let cho = node_offsets[child_idx];
                    cur_occupancy = tree[cho];
                    cur_first_child = tree[cho + 1u];
                    if ENABLE_STATS {
                        ray_loads_offsets = ray_loads_offsets + 1u;
                        ray_loads_tree = ray_loads_tree + 2u;
                    }
                    let nc = vec3<i32>(
                        clamp(i32(floor(entry_pos.x)), 0, 2),
                        clamp(i32(floor(entry_pos.y)), 0, 2),
                        clamp(i32(floor(entry_pos.z)), 0, 2),
                    );
                    s_cell[depth] = pack_cell(nc);
                    let lc = vec3<f32>(nc);
                    cur_side_dist = vec3<f32>(
                        select((lc.x       - entry_pos.x) * inv_dir.x,
                               (lc.x + 1.0 - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                        select((lc.y       - entry_pos.y) * inv_dir.y,
                               (lc.y + 1.0 - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
                        select((lc.z       - entry_pos.z) * inv_dir.z,
                               (lc.z + 1.0 - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
                    );
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

// Closest-face UV bevel for a (lon, lat, r) cell. Same recipe as the
// existing slab-cell hit code, extracted as a helper so the slab DDA
// and the sub-cell anchor DDA share one source of truth.
//
// `r, lat_p, lon_p` are the ray's spherical coords at hit. The cell's
// six faces are at the supplied lo/hi bounds; closest-face is chosen
// by world-space arc length (lon arcs scale by r·cos(lat), lat arcs
// by r, r-shells in radial units). The other two axes' in-cell
// fractional positions form the 2D UV used to bevel the face edge.
fn make_sphere_hit(
    pos: vec3<f32>, n_step: vec3<f32>, t_param: f32, inv_norm: f32,
    block_type: u32,
    r: f32, lat_p: f32, lon_p: f32,
    lon_lo_c: f32, lon_hi_c: f32,
    lat_lo_c: f32, lat_hi_c: f32,
    r_lo_c: f32, r_hi_c: f32,
    lon_step_c: f32, lat_step_c: f32, r_step_c: f32,
) -> HitResult {
    var result: HitResult;
    let cos_lat = max(cos(lat_p), 1e-3);
    let arc_lon_lo = r * cos_lat * abs(lon_p - lon_lo_c);
    let arc_lon_hi = r * cos_lat * abs(lon_p - lon_hi_c);
    let arc_lat_lo = r * abs(lat_p - lat_lo_c);
    let arc_lat_hi = r * abs(lat_p - lat_hi_c);
    let arc_r_lo  = abs(r - r_lo_c);
    let arc_r_hi  = abs(r - r_hi_c);
    var best = arc_lon_lo;
    var axis: u32 = 0u;
    if arc_lon_hi < best { best = arc_lon_hi; axis = 0u; }
    if arc_lat_lo < best { best = arc_lat_lo; axis = 1u; }
    if arc_lat_hi < best { best = arc_lat_hi; axis = 1u; }
    if arc_r_lo  < best { best = arc_r_lo;  axis = 2u; }
    if arc_r_hi  < best { best = arc_r_hi;  axis = 2u; }

    let lon_in_cell = clamp((lon_p - lon_lo_c) / lon_step_c, 0.0, 1.0);
    let lat_in_cell = clamp((lat_p - lat_lo_c) / lat_step_c, 0.0, 1.0);
    let r_in_cell   = clamp((r     - r_lo_c)   / r_step_c,   0.0, 1.0);
    var u_in_face: f32;
    var v_in_face: f32;
    if axis == 0u {
        u_in_face = lat_in_cell;
        v_in_face = r_in_cell;
    } else if axis == 1u {
        u_in_face = lon_in_cell;
        v_in_face = r_in_cell;
    } else {
        u_in_face = lon_in_cell;
        v_in_face = lat_in_cell;
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

// Tangent-cube DDA. Used by `sphere_descend_anchor`'s TangentBlock
// branch after the ray has been transformed into the cube's local
// `[0, 3)³` frame.
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

    var iterations: u32 = 0u;
    loop {
        if iterations >= 2048u { break; }
        iterations = iterations + 1u;

        let cell = unpack_cell(s_cell[depth]);

        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth = depth - 1u;
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

            let parent_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];

            let m_oob = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(parent_cell + vec3<i32>(m_oob) * step);
            cur_side_dist = cur_side_dist + m_oob * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_oob;
            continue;
        }

        let slot = u32(cell.x + cell.y * 3 + cell.z * 9);
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
        let child_cell_size = cur_cell_size / 3.0;
        let node_hit = ray_box(
            ray_origin, inv_dir,
            child_origin,
            child_origin + vec3<f32>(3.0) * child_cell_size,
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
    r_sphere: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    // TangentBlock dispatch: when the slab cell anchor is itself a
    // TangentBlock, transform the world ray into the cell's local
    // tangent-cube frame and hand off to the precision-stable
    // Cartesian DDA. Sphere descent NEVER traverses below this
    // point — every voxel beneath the slab cell sees pure Cartesian.
    if anchor_idx < arrayLength(&node_kinds) && node_kinds[anchor_idx].kind == NODE_KIND_TANGENT_BLOCK {
        // Tangent point: cell center on the sphere.
        let lat_c = slab_lat_lo + slab_lat_step * 0.5;
        let lon_c = slab_lon_lo + slab_lon_step * 0.5;
        let r_c   = slab_r_lo   + slab_r_step   * 0.5;

        let sl = sin(lat_c);
        let cl = cos(lat_c);
        let so = sin(lon_c);
        let co = cos(lon_c);
        // Sphere-surface basis at (lat_c, lon_c). Right-handed:
        // east × normal == north.
        let normal_w = vec3<f32>(cl * co, sl, cl * so);
        let east_w   = vec3<f32>(-so,    0.0, co);
        let north_w  = vec3<f32>(-sl * co, cl, -sl * so);

        // Cube origin = cell center; cube side = the cell's largest
        // arc / radial extent, so the cube fully contains the cell.
        let cube_origin = cs_center + r_c * normal_w;
        let east_arc  = r_sphere * abs(cl) * slab_lon_step;
        let north_arc = r_sphere * slab_lat_step;
        let cube_side = max(max(east_arc, north_arc), slab_r_step);
        let scale = 3.0 / cube_side;

        // World → local cube frame: translate to cube origin, rotate
        // by R^T (rows are basis vectors), scale into [0, 3).
        let d_origin = ray_origin - cube_origin;
        let local_origin = vec3<f32>(
            dot(east_w,   d_origin) * scale + 1.5,
            dot(normal_w, d_origin) * scale + 1.5,
            dot(north_w,  d_origin) * scale + 1.5,
        );
        let local_dir = vec3<f32>(
            dot(east_w,   ray_dir) * scale,
            dot(normal_w, ray_dir) * scale,
            dot(north_w,  ray_dir) * scale,
        );

        // Use the dedicated tangent walker: deeper stack
        // (TANGENT_STACK_DEPTH = 24) so deep edits don't collapse
        // to the representative_block at MAX_STACK_DEPTH = 8.
        // Tree-structure-driven descent — no LOD pixel cap, since
        // the render frame is locked at WrappedPlane and the LOD
        // calc would always read sub-pixel for in-cube leaves.
        let sub = march_in_tangent_cube(anchor_idx, local_origin, local_dir);
        if !sub.hit { return result; }

        // Bevel must be computed in the LOCAL cube frame: the world
        // cube is rotated, so main.wgsl's `(hit_world - cell_min) /
        // cell_size` formula can't recover [0,1] local coords from
        // axis-aligned cell_min/size. Compute the local-within-cell
        // here, run the same `cube_face_bevel` Cartesian uses, and
        // bake it into result.color. Then neutralise cell_min/size
        // so main.wgsl's secondary bevel multiplier is 1.0.
        let local_hit = local_origin + local_dir * sub.t;
        let local_in_cell = clamp((local_hit - sub.cell_min) / sub.cell_size, vec3<f32>(0.0), vec3<f32>(1.0));
        let local_bevel = cube_face_bevel(local_in_cell, sub.normal);
        // local_t == world_t (the dir scale is absorbed in the
        // parameterisation). Rotate normal back to world for diffuse.
        var out: HitResult;
        out.hit = true;
        out.t = sub.t * inv_norm;
        out.color = sub.color * (0.7 + 0.3 * local_bevel);
        out.normal = east_w * sub.normal.x + normal_w * sub.normal.y + north_w * sub.normal.z;
        out.frame_level = 0u;
        out.frame_scale = 1.0;
        let hit_world = ray_origin + ray_dir * sub.t;
        out.cell_min = hit_world - vec3<f32>(0.5);
        out.cell_size = 1.0;
        return out;
    }

    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_cell: array<u32, MAX_STACK_DEPTH>;
    var depth: u32 = 0u;
    s_node_idx[0] = anchor_idx;

    // Frame 0 = anchor block. Each axis is 1/3 of the slab cell.
    var cur_lon_org = slab_lon_lo;
    var cur_lat_org = slab_lat_lo;
    var cur_r_org   = slab_r_lo;
    var cur_lon_step = slab_lon_step / 3.0;
    var cur_lat_step = slab_lat_step / 3.0;
    var cur_r_step   = slab_r_step   / 3.0;

    var t = t_in;

    // Initial sub-cell from the ray's t.
    {
        let pos0 = ray_origin + ray_dir * t;
        let off0 = pos0 - cs_center;
        let r0 = max(length(off0), 1e-9);
        let n0 = off0 / r0;
        let lat0 = asin(clamp(n0.y, -1.0, 1.0));
        let lon0 = atan2(n0.z, n0.x);
        let cx0 = clamp(i32(floor((lon0 - cur_lon_org) / cur_lon_step)), 0, 2);
        let cy0 = clamp(i32(floor((r0   - cur_r_org)   / cur_r_step)),   0, 2);
        let cz0 = clamp(i32(floor((lat0 - cur_lat_org) / cur_lat_step)), 0, 2);
        s_cell[0] = pack_cell(vec3<i32>(cx0, cy0, cz0));
    }

    var iters: u32 = 0u;
    loop {
        if iters > 256u { break; }
        iters = iters + 1u;
        if t > t_exit { break; }

        let cell = unpack_cell(s_cell[depth]);
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            // OOB: pop one frame, or exit if at the anchor's outer face.
            if depth == 0u { break; }
            depth = depth - 1u;
            cur_lon_step = cur_lon_step * 3.0;
            cur_lat_step = cur_lat_step * 3.0;
            cur_r_step   = cur_r_step   * 3.0;
            let popped = unpack_cell(s_cell[depth]);
            cur_lon_org = cur_lon_org - f32(popped.x) * cur_lon_step;
            cur_r_org   = cur_r_org   - f32(popped.y) * cur_r_step;
            cur_lat_org = cur_lat_org - f32(popped.z) * cur_lat_step;
            // After the pop, recompute which cell we're in at the
            // parent frame using the ray's current t. This may match
            // `popped` (we exited an axis only), or its neighbour
            // (the axis stepped over a parent boundary already).
            let pos_p = ray_origin + ray_dir * t;
            let off_p = pos_p - cs_center;
            let r_p = max(length(off_p), 1e-9);
            let n_p = off_p / r_p;
            let lat_p = asin(clamp(n_p.y, -1.0, 1.0));
            let lon_p = atan2(n_p.z, n_p.x);
            let cx_p = i32(floor((lon_p - cur_lon_org) / cur_lon_step));
            let cy_p = i32(floor((r_p   - cur_r_org)   / cur_r_step));
            let cz_p = i32(floor((lat_p - cur_lat_org) / cur_lat_step));
            s_cell[depth] = pack_cell(vec3<i32>(cx_p, cy_p, cz_p));
            continue;
        }

        // Cell bounds for boundary crossings + occupancy lookup.
        let lon_lo = cur_lon_org + f32(cell.x) * cur_lon_step;
        let lon_hi = lon_lo + cur_lon_step;
        let r_lo   = cur_r_org   + f32(cell.y) * cur_r_step;
        let r_hi   = r_lo + cur_r_step;
        let lat_lo = cur_lat_org + f32(cell.z) * cur_lat_step;
        let lat_hi = lat_lo + cur_lat_step;

        let slot = u32(cell.x + cell.y * 3 + cell.z * 9);
        let header_off = node_offsets[s_node_idx[depth]];
        let occ = tree[header_off];
        let bit = 1u << slot;

        // t_next: smallest cell-face crossing > t. Used by every
        // "advance past this cell" branch below.
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

        if (occ & bit) == 0u {
            // Empty slot. Advance to the next cell within this frame.
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            let pos_a = ray_origin + ray_dir * t;
            let off_a = pos_a - cs_center;
            let r_a = max(length(off_a), 1e-9);
            let n_a = off_a / r_a;
            let lat_a = asin(clamp(n_a.y, -1.0, 1.0));
            let lon_a = atan2(n_a.z, n_a.x);
            let cx_a = i32(floor((lon_a - cur_lon_org) / cur_lon_step));
            let cy_a = i32(floor((r_a   - cur_r_org)   / cur_r_step));
            let cz_a = i32(floor((lat_a - cur_lat_org) / cur_lat_step));
            s_cell[depth] = pack_cell(vec3<i32>(cx_a, cy_a, cz_a));
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
            let pos_h = ray_origin + ray_dir * t;
            let off_h = pos_h - cs_center;
            let r_h = max(length(off_h), 1e-9);
            let n_h = off_h / r_h;
            let lat_h = asin(clamp(n_h.y, -1.0, 1.0));
            let lon_h = atan2(n_h.z, n_h.x);
            return make_sphere_hit(
                pos_h, n_h, t, inv_norm, block_type,
                r_h, lat_h, lon_h,
                lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
                cur_lon_step, cur_lat_step, cur_r_step,
            );
        }

        if tag != 2u {
            // EntityRef etc. — treat as empty for sphere subtree DDA
            // (no entities-inside-anchor for now). Advance.
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            let pos_e = ray_origin + ray_dir * t;
            let off_e = pos_e - cs_center;
            let r_e = max(length(off_e), 1e-9);
            let n_e = off_e / r_e;
            let lat_e = asin(clamp(n_e.y, -1.0, 1.0));
            let lon_e = atan2(n_e.z, n_e.x);
            let cx_e = i32(floor((lon_e - cur_lon_org) / cur_lon_step));
            let cy_e = i32(floor((r_e   - cur_r_org)   / cur_r_step));
            let cz_e = i32(floor((lat_e - cur_lat_org) / cur_lat_step));
            s_cell[depth] = pack_cell(vec3<i32>(cx_e, cy_e, cz_e));
            continue;
        }

        // tag == 2u: non-uniform Node. LOD-gate descent.
        if block_type == 0xFFFEu {
            // Subtree empty per `representative_block` — skip whole cell.
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            let pos_r = ray_origin + ray_dir * t;
            let off_r = pos_r - cs_center;
            let r_r = max(length(off_r), 1e-9);
            let n_r = off_r / r_r;
            let lat_r = asin(clamp(n_r.y, -1.0, 1.0));
            let lon_r = atan2(n_r.z, n_r.x);
            let cx_r = i32(floor((lon_r - cur_lon_org) / cur_lon_step));
            let cy_r = i32(floor((r_r   - cur_r_org)   / cur_r_step));
            let cz_r = i32(floor((lat_r - cur_lat_org) / cur_lat_step));
            s_cell[depth] = pack_cell(vec3<i32>(cx_r, cy_r, cz_r));
            continue;
        }

        // Stack ceiling = hardware limit. No camera-distance LOD here:
        // sphere mode locks the render frame at WrappedPlane, so the
        // Cartesian "lod_pixels < threshold" check would collapse deep
        // edits to the representative_block — exactly the bug we're
        // fixing. Tree structure (uniform-flatten = tag=1, terminate)
        // alone drives descent.
        let at_max = depth + 1u >= MAX_STACK_DEPTH;
        if at_max {
            let pos_h = ray_origin + ray_dir * t;
            let off_h = pos_h - cs_center;
            let r_h = max(length(off_h), 1e-9);
            let n_h = off_h / r_h;
            let lat_h = asin(clamp(n_h.y, -1.0, 1.0));
            let lon_h = atan2(n_h.z, n_h.x);
            return make_sphere_hit(
                pos_h, n_h, t, inv_norm, block_type,
                r_h, lat_h, lon_h,
                lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
                cur_lon_step, cur_lat_step, cur_r_step,
            );
        }

        // Push child frame.
        let child_idx = tree[child_base + 1u];
        depth = depth + 1u;
        s_node_idx[depth] = child_idx;
        cur_lon_org = lon_lo;
        cur_lat_org = lat_lo;
        cur_r_org   = r_lo;
        cur_lon_step = cur_lon_step / 3.0;
        cur_lat_step = cur_lat_step / 3.0;
        cur_r_step   = cur_r_step   / 3.0;

        // Pick the entry sub-cell from the ray's current t.
        let pos_c = ray_origin + ray_dir * t;
        let off_c = pos_c - cs_center;
        let r_c = max(length(off_c), 1e-9);
        let n_c = off_c / r_c;
        let lat_c = asin(clamp(n_c.y, -1.0, 1.0));
        let lon_c = atan2(n_c.z, n_c.x);
        let cx_c = clamp(i32(floor((lon_c - cur_lon_org) / cur_lon_step)), 0, 2);
        let cy_c = clamp(i32(floor((r_c   - cur_r_org)   / cur_r_step)),   0, 2);
        let cz_c = clamp(i32(floor((lat_c - cur_lat_org) / cur_lat_step)), 0, 2);
        s_cell[depth] = pack_cell(vec3<i32>(cx_c, cy_c, cz_c));
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
                    r_sphere,
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
            return make_sphere_hit(
                pos, n_step, t, inv_norm, sample.block_type,
                r, lat_p, lon_p,
                lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
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
