#include "bindings.wgsl"
#include "tree.wgsl"
#include "ray_prim.wgsl"
#include "sphere.wgsl"

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

// Cartesian DDA in a single frame rooted at `root_node_idx`. The
// frame's cell spans `[0, 3)³` in `ray_origin/ray_dir` coords.
// Returns hit on cell terminal; on miss (ray exits the frame),
// returns hit=false so the caller can pop to the ancestor ribbon.
fn march_cartesian(
    root_node_idx: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>,
    depth_limit: u32, skip_slot: u32,
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
            result.color = palette.colors[(packed >> 8u) & 0xFFu].rgb;
            result.normal = normal;
            result.cell_min = cell_min_h;
            result.cell_size = cur_cell_size;
            return result;
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

            let kind = node_kinds[child_idx].kind;
            if ENABLE_STATS { ray_loads_kinds = ray_loads_kinds + 1u; }

            if kind == 1u {
                // CubedSphereBody: dispatch sphere DDA in this body's cell.
                let body_origin = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
                let body_size = cur_cell_size;
                let inner_r = node_kinds[child_idx].inner_r;
                let outer_r = node_kinds[child_idx].outer_r;
                if ENABLE_STATS { ray_loads_kinds = ray_loads_kinds + 2u; }
                let sph = sphere_in_cell(
                    child_idx, body_origin, body_size,
                    inner_r, outer_r, ray_origin, ray_dir,
                );
                if sph.hit {
                    return sph;
                }
                // Sphere missed — advance Cartesian DDA past this cell.
                let m_sph = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_sph) * step);
                cur_side_dist += m_sph * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_sph;
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
            if child_bt == 255u {
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
                if bt == 255u {
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
                    result.color = palette.colors[bt].rgb;
                    result.normal = normal;
                    result.cell_min = cell_min_l;
                    result.cell_size = cur_cell_size;
                    return result;
                }
            } else {
                let child_origin = cur_node_origin + vec3<f32>(cell) * cur_cell_size;

                // Content AABB culling + DDA init, unified. The
                // packed child entry carries the child node's content
                // AABB (in child-local slot coords, 0..=3) in bits
                // 16-27 — computed at pack time by `content_aabb()`.
                //
                // Two wins in one ray-box:
                //   1. If the ray misses the AABB, skip the entire
                //      descent (including child DDA iterations).
                //   2. If the ray hits, use `aabb_hit.t_enter` as the
                //      DDA entry t instead of a separate ray-box
                //      against the full child. This skips any leading
                //      empty cells in the child between the node
                //      boundary and the content AABB.
                //
                // aabb_bits == 0 is a degenerate case (should only
                // hit empty-subtree edge cases during pack); treat it
                // as the full node [0, 3)^3 so behavior matches the
                // pre-AABB code.
                let aabb_bits = (packed >> 16u) & 0xFFFu;
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
                    // Content AABB missed. Advance parent DDA.
                    let m_aabb = min_axis_mask(cur_side_dist);
                    s_cell[depth] = pack_cell(cell + vec3<i32>(m_aabb) * step);
                    cur_side_dist += m_aabb * delta_dist * cur_cell_size;
                    normal = -vec3<f32>(step) * m_aabb;
                    if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
                    continue;
                }

                if ENABLE_STATS { ray_steps_node_descend = ray_steps_node_descend + 1u; }
                // Reuse aabb_hit.t_enter as the child-DDA entry t.
                // Leading empty cells between child boundary and AABB
                // are skipped for free.
                let ct_start = max(aabb_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
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

/// Read one packed 8-bit cell from the acceleration grid. Returns
/// `{occupied: bit 7, df: bits 0-6}` as a single u32 (caller masks).
/// Index is assumed in-range; callers guard before calling.
fn grid_load(gx: i32, gy: i32, gz: i32) -> u32 {
    let idx = u32(gz) * GRID_DIM * GRID_DIM + u32(gy) * GRID_DIM + u32(gx);
    let word = grid[idx >> 2u];
    return (word >> ((idx & 3u) * 8u)) & 0xFFu;
}

struct GridAdvance {
    /// Parametric t at which the ray enters the first occupied grid
    /// cell (or >= 1e20 on definitive sky miss).
    t_enter: f32,
    /// Parametric t at which the ray exits that occupied grid cell.
    /// Caller passes `t_exit - t_enter` as `max_t` to the bounded
    /// tree walk, then (on no-hit) resumes grid DDA from `t_exit`.
    t_exit: f32,
}

/// Grid-accelerated traversal step over the root Cartesian frame.
///
/// Starts at parametric `start_t` along the ray and walks the 81³
/// acceleration grid until it reaches an occupied grid cell. Empty
/// runs collapse via Chebyshev DF jumps. Returns the t-range of the
/// occupied cell so the caller can bound a tree walk to just that
/// cell, then resume grid DDA from `t_exit` on miss.
///
/// On definitive sky (ray exits the grid without ever finding an
/// occupied cell), sets `t_enter = 1e20`.
fn grid_advance(ray_origin: vec3<f32>, ray_dir: vec3<f32>, start_t: f32) -> GridAdvance {
    var out: GridAdvance;
    out.t_enter = 1e20;
    out.t_exit = 1e20;

    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );
    let box_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if box_hit.t_exit <= 0.0 || box_hit.t_enter >= box_hit.t_exit {
        return out;
    }
    var t = max(max(box_hit.t_enter, 0.0), start_t);
    let t_box_exit = box_hit.t_exit;
    if t >= t_box_exit { return out; }

    let step_x = select(-1, 1, ray_dir.x >= 0.0);
    let step_y = select(-1, 1, ray_dir.y >= 0.0);
    let step_z = select(-1, 1, ray_dir.z >= 0.0);
    let delta_dist = abs(inv_dir) * GRID_CELL_SIZE;

    let pos0 = ray_origin + ray_dir * t;
    var gx = clamp(i32(floor(pos0.x / GRID_CELL_SIZE)), 0, i32(GRID_DIM) - 1);
    var gy = clamp(i32(floor(pos0.y / GRID_CELL_SIZE)), 0, i32(GRID_DIM) - 1);
    var gz = clamp(i32(floor(pos0.z / GRID_CELL_SIZE)), 0, i32(GRID_DIM) - 1);

    // Side distances: absolute t at which the ray crosses each axis'
    // next cell boundary.
    var next_x = f32(gx + select(0, 1, step_x > 0)) * GRID_CELL_SIZE;
    var next_y = f32(gy + select(0, 1, step_y > 0)) * GRID_CELL_SIZE;
    var next_z = f32(gz + select(0, 1, step_z > 0)) * GRID_CELL_SIZE;
    var side_dist = vec3<f32>(
        (next_x - ray_origin.x) * inv_dir.x,
        (next_y - ray_origin.y) * inv_dir.y,
        (next_z - ray_origin.z) * inv_dir.z,
    );

    var iter = 0u;
    loop {
        if iter >= 512u { break; }
        iter = iter + 1u;

        if gx < 0 || gx >= i32(GRID_DIM)
            || gy < 0 || gy >= i32(GRID_DIM)
            || gz < 0 || gz >= i32(GRID_DIM)
            || t >= t_box_exit {
            return out;
        }

        let cell = grid_load(gx, gy, gz);
        let occupied = (cell & 0x80u) != 0u;
        if occupied {
            // The ray enters this occupied cell at `t`. It exits
            // the cell at the minimum of the three axis boundary
            // crossings (`side_dist`) — that's a parametric t, the
            // earliest axis plane leave. Return both so the caller
            // can bound a tree walk to this cell's extent.
            out.t_enter = t;
            out.t_exit = min(side_dist.x, min(side_dist.y, side_dist.z));
            return out;
        }
        let df = cell & 0x7Fu;
        if df > 1u {
            // Chebyshev DF guarantees no occupied cell within `df`
            // cells along any axis direction. Advance `df - 1`
            // cells along the ray — the -1 leaves one cell of
            // slack so we always step INTO a new cell, preserving
            // the invariant that side_dist is recomputed for the
            // cell we're about to check.
            let jump = f32(df - 1u) * GRID_CELL_SIZE;
            t = t + jump;
            if t >= t_box_exit { return out; }
            let pos = ray_origin + ray_dir * t;
            gx = clamp(i32(floor(pos.x / GRID_CELL_SIZE)), 0, i32(GRID_DIM) - 1);
            gy = clamp(i32(floor(pos.y / GRID_CELL_SIZE)), 0, i32(GRID_DIM) - 1);
            gz = clamp(i32(floor(pos.z / GRID_CELL_SIZE)), 0, i32(GRID_DIM) - 1);
            next_x = f32(gx + select(0, 1, step_x > 0)) * GRID_CELL_SIZE;
            next_y = f32(gy + select(0, 1, step_y > 0)) * GRID_CELL_SIZE;
            next_z = f32(gz + select(0, 1, step_z > 0)) * GRID_CELL_SIZE;
            side_dist = vec3<f32>(
                (next_x - ray_origin.x) * inv_dir.x,
                (next_y - ray_origin.y) * inv_dir.y,
                (next_z - ray_origin.z) * inv_dir.z,
            );
            continue;
        }
        // df <= 1: step exactly one cell via standard DDA.
        let m = min_axis_mask(side_dist);
        if m.x > 0.5 {
            t = side_dist.x;
            gx = gx + step_x;
            side_dist.x = side_dist.x + delta_dist.x;
        } else if m.y > 0.5 {
            t = side_dist.y;
            gy = gy + step_y;
            side_dist.y = side_dist.y + delta_dist.y;
        } else {
            t = side_dist.z;
            gz = gz + step_z;
            side_dist.z = side_dist.z + delta_dist.z;
        }
    }
    // Iteration cap: signal miss. Caller will fall through to sky.
    return out;
}

// Top-level march. Dispatches the current frame's DDA on its
// NodeKind (Cartesian or sphere body), then on miss pops to the
// next ancestor in the ribbon and continues. When ribbon is
// exhausted, returns sky (hit=false).
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
    var current_kind = uniforms.root_kind;
    var inner_r = uniforms.root_radii.x;
    var outer_r = uniforms.root_radii.y;
    var cur_face_bounds = uniforms.root_face_bounds;
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

        var r: HitResult;
        if current_kind == ROOT_KIND_BODY {
            let body_origin = vec3<f32>(0.0);
            let body_size = 3.0;
            r = sphere_in_cell(
                current_idx, body_origin, body_size,
                inner_r, outer_r, ray_origin, ray_dir,
            );
        } else if current_kind == ROOT_KIND_FACE {
            r = march_face_root(current_idx, ray_origin, ray_dir, cur_face_bounds);
        } else {
            // Cartesian frame. At ribbon_level == 0 (root frame),
            // we optionally use the acceleration grid as a pre-
            // filter. The grid only helps when the ray begins
            // OUTSIDE any occupied grid cell — then it can jump
            // across long empty runs via Chebyshev DF before handing
            // off to the tree walker, or skip the tree walk entirely
            // when the ray exits the grid without touching content.
            //
            // When the camera sits INSIDE an occupied grid cell (e.g.
            // Jerusalem nucleus at (1.5, 1.5, 1.5)), the grid returns
            // t_enter = 0 on the first probe and the tree walker
            // runs unbounded from `ray_origin` — same as the no-grid
            // path. Detect and short-circuit that case so we don't
            // pay the grid's own iteration cost for zero benefit.
            //
            // Ancestor frames after a ribbon pop use the plain
            // `march_cartesian` path unchanged — the grid is baked
            // for the current root only.
            r = march_cartesian(current_idx, ray_origin, ray_dir, MAX_STACK_DEPTH, skip_slot);
        }
        if r.hit {
            r.frame_level = ribbon_level;
            r.frame_scale = cur_scale;
            // Transform cell_min/cell_size from the popped frame back
            // to the camera frame so the fragment shader's bevel/grid
            // computation uses consistent coordinates.
            if cur_scale < 1.0 {
                let hit_popped = ray_origin + ray_dir * r.t;
                let cell_local = clamp(
                    (hit_popped - r.cell_min) / r.cell_size,
                    vec3<f32>(0.0), vec3<f32>(1.0),
                );
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
        if current_kind == ROOT_KIND_FACE {
            let body_pop_level = uniforms.root_face_meta.y;
            if ribbon_level < body_pop_level {
                let entry = ribbon[ribbon_level];
                if ENABLE_STATS { ray_loads_ribbon = ray_loads_ribbon + 1u; }
                let s = entry.slot_bits & RIBBON_SLOT_MASK;
                let sx = i32(s % 3u);
                let sy = i32((s / 3u) % 3u);
                let sz = i32(s / 9u);
                let slot_off = vec3<f32>(f32(sx), f32(sy), f32(sz));
                let old_size = cur_face_bounds.w;
                cur_face_bounds = vec4<f32>(
                    cur_face_bounds.x - slot_off.x * old_size,
                    cur_face_bounds.y - slot_off.y * old_size,
                    cur_face_bounds.z - slot_off.z * old_size,
                    old_size * 3.0,
                );
                cur_scale = cur_scale * (1.0 / 3.0);
                current_idx = entry.node_idx;
                ribbon_level = ribbon_level + 1u;
                continue;
            }
            if body_pop_level >= uniforms.ribbon_count {
                break;
            }
            let body_entry = ribbon[body_pop_level];
            current_idx = body_entry.node_idx;
            current_kind = ROOT_KIND_BODY;
            inner_r = node_kinds[current_idx].inner_r;
            outer_r = node_kinds[current_idx].outer_r;
            if ENABLE_STATS {
                ray_loads_ribbon = ray_loads_ribbon + 1u;
                ray_loads_kinds = ray_loads_kinds + 2u;
            }
            ribbon_level = body_pop_level + 1u;
        } else {
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
            //
            // This is the "zoomed-in inside empty sky" fast path.
            // Without it, each empty ancestor shell costs ~3–5 empty
            // DDA iterations, compounding linearly with ribbon depth
            // (10+ shells in the regressed workload).
            if ribbon_level < uniforms.ribbon_count {
                let entry = ribbon[ribbon_level];
                if ENABLE_STATS { ray_loads_ribbon = ray_loads_ribbon + 1u; }
                let s = entry.slot_bits & RIBBON_SLOT_MASK;
                let sx = i32(s % 3u);
                let sy = i32((s / 3u) % 3u);
                let sz = i32(s / 9u);
                let slot_off = vec3<f32>(f32(sx), f32(sy), f32(sz));
                skip_slot = s;
                ray_origin = slot_off + ray_origin / 3.0;
                ray_dir = ray_dir / 3.0;
                cur_scale = cur_scale * (1.0 / 3.0);
                current_idx = entry.node_idx;
                ribbon_level = ribbon_level + 1u;

                let k = node_kinds[current_idx].kind;
                if ENABLE_STATS { ray_loads_kinds = ray_loads_kinds + 1u; }
                if k == 1u {
                    current_kind = ROOT_KIND_BODY;
                    inner_r = node_kinds[current_idx].inner_r;
                    outer_r = node_kinds[current_idx].outer_r;
                    if ENABLE_STATS { ray_loads_kinds = ray_loads_kinds + 2u; }
                } else {
                    current_kind = ROOT_KIND_CARTESIAN;
                    // Empty-shell fast exit: if every sibling is
                    // empty, skip this shell's DDA and advance the
                    // ray to the shell's exit boundary. Next outer
                    // iteration will pop again.
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
