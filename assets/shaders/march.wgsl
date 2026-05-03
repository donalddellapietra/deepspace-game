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
    result.local_in_cell = vec3<f32>(0.0);

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
            let hit_local = ray_origin + ray_dir * result.t;
            result.local_in_cell = clamp(
                (hit_local - cell_min_h) / cur_cell_size,
                vec3<f32>(0.0), vec3<f32>(1.0),
            );
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
                let hit_local = ray_origin + ray_dir * result.t;
                result.local_in_cell = clamp(
                    (hit_local - cell_min_l) / cur_cell_size,
                    vec3<f32>(0.0), vec3<f32>(1.0),
                );
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
    result.local_in_cell = vec3<f32>(0.0);

    // Local-frame DDA. Each push transforms the ray into the child's
    // [0, 3)³ local frame (origin = (origin - vec3(cell)) * 3, dir *=3).
    // After every push, the local cell size is 1.0, so absolute
    // magnitudes stay bounded regardless of depth — `MAX_STACK_DEPTH`
    // can grow far past 8 without losing precision in side_dist
    // accumulation (the bug that capped the old absolute-frame DDA).
    //
    // The parameter `t` is preserved by the (origin - vec3(cell)) * 3
    // and dir * 3 transforms (both factor out: scale * (orig + t*dir -
    // cell_offset) = new_orig + t * new_dir), so `result.t` returned
    // to the caller is in the OUTER ray's units, no conversion needed.
    //
    // For OUTER-frame outputs (cell_min, cell_size), we track:
    //   cur_scale       — running 3^-depth (outer cell size at current depth)
    //   cur_outer_origin — running outer-frame position of current frame's [0,0,0]
    //                      = sum over pushed cells_i of vec3(cells_i) * 3^-i.
    //                      Bounded by [0, 3); used only for cell_min output, not DDA.
    var ray_origin: vec3<f32> = ray_origin_in;
    var ray_dir: vec3<f32> = ray_dir_in;

    // Outer-frame ray (immutable across pushes). Used at depth==0 for
    // X-wrap, and as the reference for `ray_metric` (world-distance
    // LOD scale). After each ribbon pop the caller divides ray_dir by
    // 3, so `ray_metric` correctly tracks world units.
    let outer_ray_dir = ray_dir_in;
    let outer_ray_origin_in = ray_origin_in;

    // `ray_metric` is computed from the OUTER ray_dir (the one passed
    // by `march()`). After ribbon pops, ray_dir magnitude shrinks
    // (÷3 per pop). LOD pixel calculations need world-space distances,
    // and `t` is preserved by the local push transform, so multiplying
    // the local side_dist by `ray_metric` yields a world-space distance.
    let ray_metric = max(length(outer_ray_dir), 1e-6);

    // `step` is fixed for the function lifetime: ray_dir is multiplied
    // by 3 (positive) on push, so signs of components never change.
    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );

    // Local-frame inv_dir / delta_dist. Recomputed on every push and
    // pop as a single division (no compounding multiply chain that
    // would accumulate quantization error at deep depth).
    var inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );
    var delta_dist = abs(inv_dir);

    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    // Packed per-depth cell coords (parent's cell when descended,
    // current DDA cell at top of stack). 3 bits per axis = 9 bits/u32.
    var s_cell: array<u32, MAX_STACK_DEPTH>;

    // NEW: per-push saved state (parent's local-frame ray_origin,
    // ray_dir, side_dist, restored exactly on pop). Stored in PARENT-
    // LOCAL coords — bounded magnitudes regardless of depth, so
    // growing `MAX_STACK_DEPTH` doesn't hurt precision.
    //
    // We save ray_dir explicitly (rather than recomputing as
    // ray_dir / 3.0 on pop) because f32 division by 3.0 is NOT exact;
    // a push-pop round trip drifts ray_dir by ~1 ULP each cycle, which
    // compounds over many iterations.
    var s_origin: array<vec3<f32>, MAX_STACK_DEPTH>;
    var s_dir: array<vec3<f32>, MAX_STACK_DEPTH>;
    var s_side_dist: array<vec3<f32>, MAX_STACK_DEPTH>;

    // Outer-frame conversion factors. Always cur_scale = 1.0,
    // cur_outer_origin = (0,0,0) at depth 0; updated incrementally on
    // push/pop. cur_outer_origin is a base-3 expansion (sum of
    // vec3(cell_i) * 3^-i) bounded in [0, 3) — used ONLY for output
    // cell_min / cell_size on hit. Precision drift in this scalar
    // affects shading face position by < 1 ULP; never affects DDA.
    var cur_scale: f32 = 1.0;
    var cur_outer_origin: vec3<f32> = vec3<f32>(0.0);

    // Phase 3 Step 3.0 — per-depth Y-bend stack. Stored in OUTER
    // distance units (matches old semantics). Converted to local-Y
    // by dividing by `cur_scale_at_d`. 0.0 at depth 0; A=0 (default
    // for non-WrappedPlane frames) makes this entire path a no-op.
    var s_y_drop: array<f32, MAX_STACK_DEPTH>;
    s_y_drop[0] = 0.0;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    s_node_idx[0] = root_node_idx;

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
    // `entry_pos` is a position firmly INSIDE the [0,3)³ root box —
    // used ONLY for initial cell selection (floor() into 0..2). Never
    // assigned to ray_origin: the LOCAL ray_origin must remain the
    // push-transformed image of the ORIGINAL input ray_origin so that
    // ray_box-derived hit `t` is in the caller's parameterization.
    //
    // Side_dist invariant: side_dist[i] = t (from local ray_origin) at
    // which the ray crosses the next i-axis cell boundary in the step
    // direction. Standard DDA formula `(boundary - ray_origin) *
    // inv_dir`. Works for ray_origin outside the [0,3) box too —
    // side_dist values stay positive (ray heading into box) and the
    // axis ordering matches what entry_pos-based init would give
    // (the per-axis offsets are uniform = t_start).
    var entry_pos: vec3<f32> = ray_origin + ray_dir * t_start;

    let root_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );
    s_cell[0] = pack_cell(root_cell);
    let cell_f0 = vec3<f32>(root_cell);
    var cur_side_dist: vec3<f32> = vec3<f32>(
        select((cell_f0.x - ray_origin.x) * inv_dir.x,
               (cell_f0.x + 1.0 - ray_origin.x) * inv_dir.x, ray_dir.x >= 0.0),
        select((cell_f0.y - ray_origin.y) * inv_dir.y,
               (cell_f0.y + 1.0 - ray_origin.y) * inv_dir.y, ray_dir.y >= 0.0),
        select((cell_f0.z - ray_origin.z) * inv_dir.z,
               (cell_f0.z + 1.0 - ray_origin.z) * inv_dir.z, ray_dir.z >= 0.0),
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
            // miss. At depth==0, local frame == outer frame
            // (cur_scale = 1.0, cur_outer_origin = (0,0,0)), so the
            // wrap math operates directly on the local ray_origin/dir.
            let x_oob = cell.x < 0 || cell.x > 2;
            let yz_in = cell.y >= 0 && cell.y <= 2
                     && cell.z >= 0 && cell.z <= 2;
            if depth == 0u
                && uniforms.root_kind == ROOT_KIND_WRAPPED_PLANE
                && x_oob && yz_in
            {
                let dims_x = uniforms.slab_dims.x;
                let slab_depth_u = uniforms.slab_dims.w;
                let cell_size_slab = 3.0 / pow(3.0, f32(slab_depth_u));
                let wrap_shift = f32(dims_x) * cell_size_slab;
                let east_oob = cell.x > 2;
                let sign = select(1.0, -1.0, east_oob);
                ray_origin.x = ray_origin.x + sign * wrap_shift;

                let new_root_hit = ray_box(
                    ray_origin, inv_dir,
                    vec3<f32>(0.0), vec3<f32>(3.0),
                );
                if new_root_hit.t_enter >= new_root_hit.t_exit
                    || new_root_hit.t_exit < 0.0
                {
                    break;
                }
                let new_t_start = max(new_root_hit.t_enter, 0.0) + 0.001;
                // X-wrap: ray_origin was translated above (Phase 2
                // semantics — wrap_shift). Don't overwrite ray_origin
                // with entry_pos: the post-wrap ray_origin IS the
                // effective ray origin for `t` arithmetic in this
                // continuation of the march. Use entry_pos for cell
                // selection only, side_dist init from ray_origin.
                entry_pos = ray_origin + ray_dir * new_t_start;
                let new_root_cell = vec3<i32>(
                    clamp(i32(floor(entry_pos.x)), 0, 2),
                    clamp(i32(floor(entry_pos.y)), 0, 2),
                    clamp(i32(floor(entry_pos.z)), 0, 2),
                );
                s_cell[0] = pack_cell(new_root_cell);
                let cf_w = vec3<f32>(new_root_cell);
                cur_side_dist = vec3<f32>(
                    select((cf_w.x - ray_origin.x) * inv_dir.x,
                           (cf_w.x + 1.0 - ray_origin.x) * inv_dir.x, ray_dir.x >= 0.0),
                    select((cf_w.y - ray_origin.y) * inv_dir.y,
                           (cf_w.y + 1.0 - ray_origin.y) * inv_dir.y, ray_dir.y >= 0.0),
                    select((cf_w.z - ray_origin.z) * inv_dir.z,
                           (cf_w.z + 1.0 - ray_origin.z) * inv_dir.z, ray_dir.z >= 0.0),
                );
                if ENABLE_STATS { ray_steps_oob = ray_steps_oob + 1u; }
                continue;
            }
            if depth == 0u { break; }
            // Pop: restore parent local-frame state from stack. We
            // saved origin/dir/side_dist exactly at push time; the
            // child's DDA never modifies the parent slots, so a pure
            // restore is bit-exact (no f32 drift from `/ 3.0` inverse).
            depth -= 1u;
            let parent_cell = unpack_cell(s_cell[depth]);
            ray_origin = s_origin[depth];
            ray_dir = s_dir[depth];
            cur_side_dist = s_side_dist[depth];
            cur_scale = cur_scale * 3.0;
            cur_outer_origin = cur_outer_origin - vec3<f32>(parent_cell) * cur_scale;
            inv_dir = vec3<f32>(
                select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
                select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
                select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
            );
            delta_dist = abs(inv_dir);
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
            cur_side_dist += m_oob * delta_dist;
            normal = -vec3<f32>(step) * m_oob;
            continue;
        }

        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        let slot_bit = 1u << slot;
        if ((cur_occupancy & slot_bit) == 0u) {
            if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
            let m_empty = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_empty) * step);
            cur_side_dist += m_empty * delta_dist;
            normal = -vec3<f32>(step) * m_empty;
            continue;
        }

        let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 1u; }

        if tag == 1u {
            // Block hit. Compute local_in_cell directly from the
            // local-frame ray + t. NO absolute coords — the cell
            // occupies [vec3(cell), vec3(cell)+1] in current local
            // frame regardless of depth.
            let local_cell_min = vec3<f32>(cell);
            let local_cell_max = local_cell_min + vec3<f32>(1.0);
            let cell_box_h = ray_box(ray_origin, inv_dir, local_cell_min, local_cell_max);
            result.hit = true;
            result.t = max(cell_box_h.t_enter, 0.0);
            result.color = palette[(packed >> 8u) & 0xFFFFu].rgb;
            result.normal = normal;
            let hit_local = ray_origin + ray_dir * result.t;
            result.local_in_cell = clamp(hit_local - local_cell_min, vec3<f32>(0.0), vec3<f32>(1.0));
            // Walker probe (curvature_offset from outer ray_dir).
            let horiz_dir_sq_h = outer_ray_dir.x * outer_ray_dir.x
                + outer_ray_dir.z * outer_ray_dir.z;
            let curvature_offset = result.t * result.t * horiz_dir_sq_h * uniforms.curvature.x;
            write_walker_probe(
                1u, iterations, depth, cell,
                vec3<f32>(0.0), cur_scale,
                result.t, normal, 1u, curvature_offset,
            );
            return result;
        } else if ENABLE_ENTITIES && tag == 3u {
            // tag=3 — EntityRef. Entity bbox is in OUTER world coords,
            // but our `ray_origin/ray_dir` are in current local frame.
            // Transform the entity bbox into local frame via cur_scale
            // / cur_outer_origin, then proceed as before.
            let entity_idx = tree[child_base + 1u];
            let entity = entities[entity_idx];
            // outer_world = cur_outer_origin + local * cur_scale
            // → local = (outer - cur_outer_origin) / cur_scale
            let inv_scale = 1.0 / cur_scale;
            let bbox_min_local = (entity.bbox_min - cur_outer_origin) * inv_scale;
            let bbox_max_local = (entity.bbox_max - cur_outer_origin) * inv_scale;
            let ebb = ray_box(ray_origin, inv_dir, bbox_min_local, bbox_max_local);
            if ebb.t_enter >= ebb.t_exit || ebb.t_exit < 0.0 {
                let m_bb = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_bb) * step);
                cur_side_dist += m_bb * delta_dist;
                normal = -vec3<f32>(step) * m_bb;
                continue;
            }

            let bbox_size_outer = entity.bbox_max - entity.bbox_min;
            let ray_dist_e = max(ebb.t_enter * ray_metric, 0.001);
            let lod_pixels_e = bbox_size_outer.x / ray_dist_e
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
                    // local_in_cell within entity bbox (in current local frame)
                    let hit_local = ray_origin + ray_dir * result.t;
                    let bbox_size_local = bbox_max_local - bbox_min_local;
                    result.local_in_cell = clamp(
                        (hit_local - bbox_min_local) / max(bbox_size_local, vec3<f32>(1e-8)),
                        vec3<f32>(0.0), vec3<f32>(1.0),
                    );
                    return result;
                }
                let m_lod_e = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_lod_e) * step);
                cur_side_dist += m_lod_e * delta_dist;
                normal = -vec3<f32>(step) * m_lod_e;
                continue;
            }

            // Entity subtree dispatch. The entity walker expects the
            // ray in entity-local [0,3)³ coords (the entity's bbox
            // mapped onto [0,3)³). We transform from CURRENT-LOCAL
            // coords; the scale chain handles the precision for us.
            //   entity_local = (current_local - bbox_min_local) * 3 / bbox_size_local
            // bbox_size_local = bbox_size_outer / cur_scale — but this
            // cancels out: 3 / bbox_size_local = 3 * cur_scale / bbox_size_outer.
            // Equivalently: entity_local = (outer_world - bbox_min_outer) * 3 / bbox_size_outer.
            // Compute via current local quantities:
            let scale3_local = vec3<f32>(3.0) / (bbox_max_local - bbox_min_local);
            let ent_origin = (ray_origin - bbox_min_local) * scale3_local;
            let ent_dir = ray_dir * scale3_local;
            let sub = march_entity_subtree(entity.subtree_bfs, ent_origin, ent_dir);
            if sub.hit {
                result.hit = true;
                let scale3_outer_x = 3.0 / bbox_size_outer.x;
                result.t = sub.t / scale3_outer_x;
                result.color = sub.color;
                result.normal = sub.normal;
                // sub.local_in_cell is already in [0,1]³ within the
                // sub-march's hit cell — pass through unchanged.
                result.local_in_cell = sub.local_in_cell;
                return result;
            }
            let m_ent_miss = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_ent_miss) * step);
            cur_side_dist += m_ent_miss * delta_dist;
            normal = -vec3<f32>(step) * m_ent_miss;
            continue;
        } else {
            let child_idx = tree[child_base + 1u];
            if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 1u; }

            let cell_slot = u32(cell.x) + u32(cell.y) * 3u + u32(cell.z) * 9u;
            if depth == 0u && cell_slot == skip_slot {
                let m_skip = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_skip) * step);
                cur_side_dist += m_skip * delta_dist;
                normal = -vec3<f32>(step) * m_skip;
                continue;
            }

            let child_bt = child_block_type(packed);
            if child_bt == 0xFFFEu {
                if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
                let m_rep = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_rep) * step);
                cur_side_dist += m_rep * delta_dist;
                normal = -vec3<f32>(step) * m_rep;
                continue;
            }

            // TangentBlock dispatch — frame-local rotation around the
            // cube's geometric centre. With the new scheme our ray is
            // already in current local frame (cell extent = 1.0,
            // child's [0,3)³ frame is reached via (origin - vec3(cell))
            // * 3 — exactly the push transform we already use). So the
            // dispatch math reduces to a clean local transform.
            if node_kinds[child_idx].kind == NODE_KIND_TANGENT_BLOCK {
                let local_pre_origin = (ray_origin - vec3<f32>(cell)) * 3.0;
                let local_pre_dir = ray_dir * 3.0;
                let rc0 = node_kinds[child_idx].rot_col0.xyz;
                let rc1 = node_kinds[child_idx].rot_col1.xyz;
                let rc2 = node_kinds[child_idx].rot_col2.xyz;
                let local_origin = local_pre_origin;
                let local_dir = vec3<f32>(
                    dot(rc0, local_pre_dir),
                    dot(rc1, local_pre_dir),
                    dot(rc2, local_pre_dir),
                );
                let sub = march_in_tangent_cube(child_idx, local_origin, local_dir);
                if sub.hit {
                    // sub.local_in_cell is in [0,1]³ within the
                    // sub-march's hit cell. Bevel uses it directly.
                    let local_bevel = cube_face_bevel(sub.local_in_cell, sub.normal);
                    var out: HitResult;
                    out.hit = true;
                    out.t = sub.t;
                    out.color = sub.color * (0.7 + 0.3 * local_bevel);
                    out.normal = rc0 * sub.normal.x
                               + rc1 * sub.normal.y
                               + rc2 * sub.normal.z;
                    out.frame_level = 0u;
                    out.frame_scale = 1.0;
                    out.local_in_cell = sub.local_in_cell;
                    return out;
                }
                let m_tb = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_tb) * step);
                cur_side_dist += m_tb * delta_dist;
                normal = -vec3<f32>(step) * m_tb;
                continue;
            }

            // Cartesian Node: depth/LOD check, then descend.
            //
            // LOD termination uses the OUTER (world) cell size of the
            // prospective child — that's `cur_scale / 3`. `min_side` is
            // a local-frame side_dist (parameter t until next plane
            // crossing). Since `t` is preserved by the local push
            // transform, `min_side * ray_metric` is a world-space
            // distance regardless of how deep we've descended. So the
            // LOD math is invariant under our depth.
            let at_max = depth + 1u > depth_limit || depth + 1u >= MAX_STACK_DEPTH;
            let cell_world_size = cur_scale / 3.0;
            let min_side = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
            let ray_dist = max(min_side * ray_metric, 0.001);
            let lod_pixels = cell_world_size / ray_dist
                * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;

            if at_max || at_lod {
                if ENABLE_STATS { ray_steps_lod_terminal = ray_steps_lod_terminal + 1u; }
                let bt = child_block_type(packed);
                if bt == 0xFFFEu {
                    let m_lodt = min_axis_mask(cur_side_dist);
                    s_cell[depth] = pack_cell(cell + vec3<i32>(m_lodt) * step);
                    cur_side_dist += m_lodt * delta_dist;
                    normal = -vec3<f32>(step) * m_lodt;
                } else {
                    let local_cell_min = vec3<f32>(cell);
                    let local_cell_max = local_cell_min + vec3<f32>(1.0);
                    let cell_box_l = ray_box(ray_origin, inv_dir, local_cell_min, local_cell_max);
                    result.hit = true;
                    result.t = max(cell_box_l.t_enter, 0.0);
                    result.color = palette[bt].rgb;
                    result.normal = normal;
                    let hit_local = ray_origin + ray_dir * result.t;
                    result.local_in_cell = clamp(hit_local - local_cell_min, vec3<f32>(0.0), vec3<f32>(1.0));
                    return result;
                }
            } else {
                // AABB cull. amin/amax are integers in 0..3 in the
                // CHILD's local coords. Mapped into PARENT-local coords
                // (current frame), the child-cell occupies [vec3(cell),
                // vec3(cell)+1], and amin/amax become sub-cell positions
                // at vec3(cell) + amin/3 .. vec3(cell) + amax/3.
                let aabb_bits = aabbs[child_idx] & 0xFFFu;
                if aabb_bits == 0u {
                    let m_empty = min_axis_mask(cur_side_dist);
                    s_cell[depth] = pack_cell(cell + vec3<i32>(m_empty) * step);
                    cur_side_dist += m_empty * delta_dist;
                    normal = -vec3<f32>(step) * m_empty;
                    if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
                    continue;
                }
                let amin = vec3<f32>(
                    f32(aabb_bits & 3u),
                    f32((aabb_bits >> 2u) & 3u),
                    f32((aabb_bits >> 4u) & 3u),
                );
                let amax = vec3<f32>(
                    f32(((aabb_bits >> 6u) & 3u) + 1u),
                    f32(((aabb_bits >> 8u) & 3u) + 1u),
                    f32(((aabb_bits >> 10u) & 3u) + 1u),
                );
                let inv3 = 1.0 / 3.0;
                let aabb_min_local = vec3<f32>(cell) + amin * inv3;
                let aabb_max_local = vec3<f32>(cell) + amax * inv3;
                let aabb_hit = ray_box(ray_origin, inv_dir, aabb_min_local, aabb_max_local);
                if aabb_hit.t_exit <= aabb_hit.t_enter || aabb_hit.t_exit < 0.0 {
                    let m_aabb = min_axis_mask(cur_side_dist);
                    s_cell[depth] = pack_cell(cell + vec3<i32>(m_aabb) * step);
                    cur_side_dist += m_aabb * delta_dist;
                    normal = -vec3<f32>(step) * m_aabb;
                    if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
                    continue;
                }

                if ENABLE_STATS { ray_steps_node_descend = ray_steps_node_descend + 1u; }

                // Descend. Use the NODE box [vec3(cell), vec3(cell)+1]
                // for entry trim — same logic as the outer-frame code
                // (avoid off-by-one when ray exactly grazes a sibling
                // boundary).
                let local_cell_min_d = vec3<f32>(cell);
                let local_cell_max_d = local_cell_min_d + vec3<f32>(1.0);
                let node_hit = ray_box(ray_origin, inv_dir, local_cell_min_d, local_cell_max_d);
                // 0.0001 in local frame == 0.0001 * cur_scale outer.
                let ct_start = max(node_hit.t_enter, 0.0) + 0.0001;
                let child_entry_local = ray_origin + ray_dir * ct_start;

                // Phase 3 Step 3.0 — Y-bend. Computed in OUTER-distance
                // units (matches old semantics). horiz_dir_sq is from
                // OUTER ray_dir; ct_start in local maps to ct_start *
                // cur_scale in outer t (since t is preserved, and at
                // depth d, "outer t" of a local-t value... actually t
                // IS preserved; ct_start is already in outer t units).
                //
                // Wait — t is preserved across local↔outer transform,
                // so ct_start_local IS ct_start_outer. The drop value
                // computation matches the old code semantics directly.
                let horiz_dir_sq = outer_ray_dir.x * outer_ray_dir.x
                    + outer_ray_dir.z * outer_ray_dir.z;
                let raw_drop = ct_start * ct_start * horiz_dir_sq * uniforms.curvature.x;
                let r_frame = 3.0 / (2.0 * 3.14159265);
                let curvature_drop = min(raw_drop, 2.0 * r_frame);
                // The drop is in OUTER Y units. To apply in current
                // local frame, divide by cur_scale (outer→local Y
                // conversion at this depth).
                let curvature_drop_local = curvature_drop / cur_scale;
                let bent_child_entry_y_local = child_entry_local.y - curvature_drop_local;

                if ENABLE_STATS {
                    let preview_header_off = node_offsets[child_idx];
                    let preview_occ = tree[preview_header_off];
                    // Local entry into PARENT cell already gives child
                    // sub-cell at granularity 1/3 — preview cell coords
                    // for path-mask are floor(local_entry_in_child).
                    let local_entry_in_child = vec3<f32>(
                        (child_entry_local.x - f32(cell.x)) * 3.0,
                        (bent_child_entry_y_local - f32(cell.y)) * 3.0,
                        (child_entry_local.z - f32(cell.z)) * 3.0,
                    );
                    let preview_entry_cell = vec3<i32>(
                        i32(floor(local_entry_in_child.x)),
                        i32(floor(local_entry_in_child.y)),
                        i32(floor(local_entry_in_child.z)),
                    );
                    let pm = path_mask_conservative(preview_entry_cell, step);
                    if (preview_occ & pm) == 0u {
                        ray_steps_would_cull = ray_steps_would_cull + 1u;
                    }
                }

                // Save parent state on stack BEFORE the push transform.
                s_origin[depth] = ray_origin;
                s_dir[depth] = ray_dir;
                s_side_dist[depth] = cur_side_dist;

                // Push transform into child's [0,3)³ local frame.
                // CRITICAL: transform the ACTUAL ray_origin (which
                // corresponds, via the push chain, to the original
                // input ray_origin) — NOT child_entry_local. Otherwise
                // `result.t` would be measured from the wrong reference
                // point and would silently drift by ct_start per push.
                ray_origin = (ray_origin - vec3<f32>(cell)) * 3.0;
                ray_dir = ray_dir * 3.0;
                cur_outer_origin = cur_outer_origin + vec3<f32>(cell) * cur_scale;
                cur_scale = cur_scale * (1.0 / 3.0);
                inv_dir = vec3<f32>(
                    select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
                    select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
                    select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
                );
                delta_dist = abs(inv_dir);

                depth += 1u;
                s_node_idx[depth] = child_idx;
                s_y_drop[depth] = curvature_drop;
                let child_header_off = node_offsets[child_idx];
                cur_occupancy = tree[child_header_off];
                cur_first_child = tree[child_header_off + 1u];
                if ENABLE_STATS {
                    ray_loads_offsets = ray_loads_offsets + 1u;
                    ray_loads_tree = ray_loads_tree + 2u;
                }
                // Initial cell selection in CHILD frame: use the
                // bent-Y child entry (in CHILD-LOCAL coords). t is
                // preserved across push, so child entry in CHILD frame
                // = ray_origin (post-push) + ray_dir (post-push) *
                // ct_start. Equivalent to: transform parent's
                // child_entry_local to child frame.
                let child_entry_in_child = vec3<f32>(
                    ray_origin.x + ray_dir.x * ct_start,
                    ray_origin.y + ray_dir.y * ct_start - curvature_drop_local * 3.0,
                    ray_origin.z + ray_dir.z * ct_start,
                );
                let new_cell = vec3<i32>(
                    clamp(i32(floor(child_entry_in_child.x)), 0, 2),
                    clamp(i32(floor(child_entry_in_child.y)), 0, 2),
                    clamp(i32(floor(child_entry_in_child.z)), 0, 2),
                );
                s_cell[depth] = pack_cell(new_cell);
                let lc = vec3<f32>(new_cell);
                // side_dist init from ray_origin (NOT child entry):
                // standard DDA `(boundary - ray_origin) * inv_dir`.
                // Side_dist may include the ct_start offset if
                // ray_origin is upstream of the cell, but axis ordering
                // is preserved (offsets uniform across axes). DDA
                // converges to the correct cell at correct t.
                //
                // Y-bend (Phase 3 curvature): reference the bent ray
                // origin so Y crossings stay consistent with the bent
                // cell selection above. Bent amount in CHILD-LOCAL Y
                // units = drop_outer / cur_scale_child = drop_outer /
                // (cur_scale_parent / 3) = curvature_drop_local * 3.
                // For A=0 (default), curvature_drop_local = 0 so this
                // is a no-op and the path is bit-identical to flat.
                let bent_origin_y_child = ray_origin.y - curvature_drop_local * 3.0;
                cur_side_dist = vec3<f32>(
                    select((lc.x - ray_origin.x) * inv_dir.x,
                           (lc.x + 1.0 - ray_origin.x) * inv_dir.x, ray_dir.x >= 0.0),
                    select((lc.y - bent_origin_y_child) * inv_dir.y,
                           (lc.y + 1.0 - bent_origin_y_child) * inv_dir.y, ray_dir.y >= 0.0),
                    select((lc.z - ray_origin.z) * inv_dir.z,
                           (lc.z + 1.0 - ray_origin.z) * inv_dir.z, ray_dir.z >= 0.0),
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

// Tangent-cube DDA. Used by `march_wrapped_planet` after the ray has
// been transformed into a slab cell's local `[0, 3)³` frame.
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
    result.local_in_cell = vec3<f32>(0.0);

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
            let hit_local = ray_origin + ray_dir * result.t;
            result.local_in_cell = clamp(
                (hit_local - cell_min_h) / cur_cell_size,
                vec3<f32>(0.0), vec3<f32>(1.0),
            );
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
            let hit_local = ray_origin + ray_dir * result.t;
            result.local_in_cell = clamp(
                (hit_local - cell_min_h) / cur_cell_size,
                vec3<f32>(0.0), vec3<f32>(1.0),
            );
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
    result.local_in_cell = vec3<f32>(0.0);

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
        let scale = 3.0 / cube.side;
        let d_origin = ray_origin - cube.origin_w;
        let local_origin = vec3<f32>(
            dot(cube.east_w, d_origin) * scale + 1.5,
            dot(cube.normal_w, d_origin) * scale + 1.5,
            dot(cube.north_w, d_origin) * scale + 1.5,
        );
        let local_dir = vec3<f32>(
            dot(cube.east_w, ray_dir) * scale,
            dot(cube.normal_w, ray_dir) * scale,
            dot(cube.north_w, ray_dir) * scale,
        );

        if sample.tag == 2u {
            let sub = march_in_tangent_cube(sample.child_idx, local_origin, local_dir);
            if sub.hit {
                let local_bevel = cube_face_bevel(sub.local_in_cell, sub.normal);
                var out: HitResult;
                out.hit = true;
                out.t = sub.t * inv_norm;
                out.color = sub.color * (0.7 + 0.3 * local_bevel);
                out.normal = cube.east_w * sub.normal.x
                           + cube.normal_w * sub.normal.y
                           + cube.north_w * sub.normal.z;
                out.frame_level = 0u;
                out.frame_scale = 1.0;
                out.local_in_cell = sub.local_in_cell;
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
            let cube_box = ray_box(local_origin, inv_local, vec3<f32>(0.0), vec3<f32>(3.0));
            if cube_box.t_enter < cube_box.t_exit && cube_box.t_exit > 0.0 {
                let t_local = max(cube_box.t_enter, 0.0);
                let entry_local = local_origin + local_dir * t_local;
                let dx_lo = abs(entry_local.x - 0.0);
                let dx_hi = abs(entry_local.x - 3.0);
                let dy_lo = abs(entry_local.y - 0.0);
                let dy_hi = abs(entry_local.y - 3.0);
                let dz_lo = abs(entry_local.z - 0.0);
                let dz_hi = abs(entry_local.z - 3.0);
                var best = dx_lo;
                var local_normal = vec3<f32>(-1.0, 0.0, 0.0);
                if dx_hi < best { best = dx_hi; local_normal = vec3<f32>(1.0, 0.0, 0.0); }
                if dy_lo < best { best = dy_lo; local_normal = vec3<f32>(0.0, -1.0, 0.0); }
                if dy_hi < best { best = dy_hi; local_normal = vec3<f32>(0.0, 1.0, 0.0); }
                if dz_lo < best { best = dz_lo; local_normal = vec3<f32>(0.0, 0.0, -1.0); }
                if dz_hi < best { best = dz_hi; local_normal = vec3<f32>(0.0, 0.0, 1.0); }
                let local_in_cell = clamp(entry_local / 3.0, vec3<f32>(0.0), vec3<f32>(1.0));
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
                out.local_in_cell = local_in_cell;
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

        // Frame dispatch on NodeKind. WrappedPlane (kind == 1) always
        // renders as a sphere of rotated tangent cubes via
        // `march_wrapped_planet` — no spherical primitive in the
        // traversal, just per-cell rotation. All other kinds fall
        // through to the cartesian DDA.
        var r: HitResult;
        let cur_kind = node_kinds[current_idx].kind;
        if cur_kind == 1u {
            r = march_wrapped_planet(
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
        // Ray pop: rescale origin into parent's [0,3)³.
        // TangentBlock children need rotation R applied on pop
        // (R maps child-local to parent frame).
        let child_kind = node_kinds[entry.child_bfs].kind;
        if child_kind == NODE_KIND_TANGENT_BLOCK {
            let rc0 = node_kinds[entry.child_bfs].rot_col0.xyz;
            let rc1 = node_kinds[entry.child_bfs].rot_col1.xyz;
            let rc2 = node_kinds[entry.child_bfs].rot_col2.xyz;
            // Direction-only: position pops Cartesian (no rotation),
            // direction rotated by R (forward) to undo the R^T that
            // was applied on entry.
            ray_origin = slot_off + ray_origin / 3.0;
            ray_dir = rc0 * ray_dir.x + rc1 * ray_dir.y + rc2 * ray_dir.z;
        } else {
            ray_origin = slot_off + ray_origin / 3.0;
        }
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
    result.local_in_cell = vec3<f32>(0.0);
    return result;
}
