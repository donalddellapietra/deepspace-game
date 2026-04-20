#include "bindings.wgsl"
#include "tree.wgsl"
#include "ray_prim.wgsl"
#include "sphere.wgsl"

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
    var s_cell: array<vec3<i32>, MAX_STACK_DEPTH>;

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

    // Active-body state for LOD-terminal sphere clipping. Set when
    // the walker descends into a `kind == 1u` (CubedSphereBody) child
    // after the ray-sphere pre-clip succeeds; cleared when the walker
    // pops back above the body's depth. Used at LOD-terminal hits
    // inside the body to reject cells whose centers lie outside the
    // body's outer sphere — without this, face-subtree LOD cubes
    // visibly stick out past the sphere's silhouette at mid zoom.
    var active_body_depth: i32 = -1;
    var active_body_center: vec3<f32> = vec3<f32>(0.0);
    var active_body_radius: f32 = 0.0;

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

    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }

    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = ray_origin + ray_dir * t_start;

    s_cell[0] = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );
    let cell_f = vec3<f32>(s_cell[0]);
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

        let cell = s_cell[depth];

        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth -= 1u;
            // Clear the active-body state when popping above the
            // body's depth — we've exited the body cell, its sphere
            // no longer constrains LOD-terminal hits.
            if i32(depth) < active_body_depth {
                active_body_depth = -1;
            }
            // Restore parent-depth scalars. cur_cell_size ×3 undoes
            // the descend divide; cur_node_origin subtracts the exact
            // vec we added on descend (s_cell[parent_depth] was
            // preserved while we were inside the child, so this is
            // byte-exact — no accumulated floating-point error).
            cur_cell_size = cur_cell_size * 3.0;
            cur_node_origin = cur_node_origin - vec3<f32>(s_cell[depth]) * cur_cell_size;
            // Recompute cur_side_dist from scratch at the parent
            // depth. Same formula as the descend-site init — same
            // entry_pos reference. ~6 FMAs per pop, amortized to
            // ~nothing vs. the per-thread register savings.
            let lc_pop = vec3<f32>(s_cell[depth]);
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

            let m_oob = min_axis_mask(cur_side_dist);
            s_cell[depth] += vec3<i32>(m_oob) * step;
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
            s_cell[depth] += vec3<i32>(m_empty) * step;
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
            // Walker path (slot sequence from this frame's root down
            // to the hit cell). main.wgsl prepends render_path to get
            // the full world-root-relative path for highlight match.
            result.hit_path_depth = depth + 1u;
            for (var d: u32 = 0u; d <= depth; d = d + 1u) {
                let c = s_cell[d];
                let sl = u32(c.x) + u32(c.y) * 3u + u32(c.z) * 9u;
                pack_slot_into_path(&result.hit_path, d, sl);
            }
            return result;
        } else {
            // tag == 2u: Node child. Load node_index from the
            // second u32 of the compact entry we already located.
            let child_idx = tree[child_base + 1u];

            // Shell skip: when re-entering a parent shell after a
            // ribbon pop, skip the SLOT we already traversed in the
            // inner shell. Uses slot index (not node_idx) so it works
            // correctly in deduplicated trees where siblings share the
            // same packed node. Checked BEFORE the kind lookup so a
            // ribbon-pop landing on a sphere-body slot doesn't
            // re-dispatch the sphere DDA we already traversed.
            let cell_slot = u32(s_cell[depth].x) + u32(s_cell[depth].y) * 3u + u32(s_cell[depth].z) * 9u;
            if depth == 0u && cell_slot == skip_slot {
                let m_skip = min_axis_mask(cur_side_dist);
                s_cell[depth] += vec3<i32>(m_skip) * step;
                cur_side_dist += m_skip * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_skip;
                continue;
            }

            // Sphere-SDF pre-clip for CubedSphereBody children.
            //
            // The body cell contains 6 face subtrees on face-center
            // slots, a uniform interior filler in the center slot, and
            // 20 empty corner slots. Cartesian DDA inside the body is
            // what walks the voxel terrain, but the outer silhouette
            // needs to be the smooth sphere — the voxel content at the
            // outer shell is faceted cubic cells.
            //
            // Solution: analytic ray-sphere test in the local frame
            // (center = body cell center, radius = outer_r *
            // cell_size). Rays that miss the outer sphere skip the
            // body cell entirely — no silhouette leak. Rays that hit
            // descend into the body as a normal 27-ary Cartesian node;
            // worldgen guarantees voxels only exist inside the sphere,
            // so voxel hits sample the actual terrain.
            //
            // Precision: center and radius live in the current frame's
            // local coords (cur_node_origin + cell * cur_cell_size),
            // bounded by the frame's [0, 3)³ box. No body-frame 1.5
            // absolute math, no ULP wall at deep anchor.
            let kind = node_kinds[child_idx].kind;
            if kind == 1u {
                let outer_r = node_kinds[child_idx].outer_r;
                let body_origin_sph = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
                let body_size_sph = cur_cell_size;
                let cs_center = body_origin_sph + vec3<f32>(body_size_sph * 0.5);
                let cs_outer = outer_r * body_size_sph;
                let oc = ray_origin - cs_center;
                let b = dot(oc, ray_dir);
                let c = dot(oc, oc) - cs_outer * cs_outer;
                let disc = b * b - c;
                let sph_outside = disc <= 0.0 || (-b + sqrt(max(disc, 0.0))) <= 0.0;
                if sph_outside {
                    // Ray misses the outer sphere — advance DDA past
                    // the body cell, leaving a round silhouette edge.
                    let m_sph = min_axis_mask(cur_side_dist);
                    s_cell[depth] += vec3<i32>(m_sph) * step;
                    cur_side_dist += m_sph * delta_dist * cur_cell_size;
                    normal = -vec3<f32>(step) * m_sph;
                    continue;
                }
                // Hit — record the body's sphere so LOD-terminal
                // hits inside the body can be clipped against it.
                // Depth is set to the CHILD's depth (depth + 1);
                // cleared on pop above that level.
                active_body_depth = i32(depth + 1u);
                active_body_center = cs_center;
                active_body_radius = cs_outer;
                // Fall through to Node descent.
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
                s_cell[depth] += vec3<i32>(m_rep) * step;
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
                // LOD-terminal sphere clip: if we're inside an active
                // body and this LOD cell's center lies outside the
                // body's outer sphere, reject the hit and advance DDA.
                // Without this, face-subtree LOD cubes (1/27 of the
                // body) stick out past the sphere's silhouette at
                // mid-zoom views (the face subtree's body sub-box
                // extends into cube corners outside the sphere).
                //
                // Cell center in the walker's current frame:
                //   cur_node_origin + cell * cur_cell_size + cur_cell_size/2
                let cell_center_w = cur_node_origin
                    + vec3<f32>(cell) * cur_cell_size
                    + vec3<f32>(cur_cell_size * 0.5);
                let dv = cell_center_w - active_body_center;
                let outside_sphere = active_body_depth >= 0
                    && dot(dv, dv) > active_body_radius * active_body_radius;
                if bt == 255u || outside_sphere {
                    let m_lodt = min_axis_mask(cur_side_dist);
                    s_cell[depth] += vec3<i32>(m_lodt) * step;
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
                    result.hit_path_depth = depth + 1u;
                    for (var d: u32 = 0u; d <= depth; d = d + 1u) {
                        let c = s_cell[d];
                        let sl = u32(c.x) + u32(c.y) * 3u + u32(c.z) * 9u;
                        pack_slot_into_path(&result.hit_path, d, sl);
                    }
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
                    s_cell[depth] += vec3<i32>(m_aabb) * step;
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
                s_cell[depth] = vec3<i32>(
                    clamp(i32(floor(local_entry.x)), 0, 2),
                    clamp(i32(floor(local_entry.y)), 0, 2),
                    clamp(i32(floor(local_entry.z)), 0, 2),
                );
                let lc = vec3<f32>(s_cell[depth]);
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

// Top-level march. The root frame is always walked as a Cartesian
// node — if its NodeKind is CubedSphereBody, the sphere silhouette
// is handled internally by `march_cartesian`'s `kind == 1u` branch
// (ray-sphere pre-clip). Face subtrees are plain 27-ary Cartesian
// nodes as far as the walker is concerned, so there's no separate
// dispatch for them. When the ray exits the current frame, we pop
// one ancestor via the ribbon and retry.
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

    // Per-frame sphere-clip hint. When the render root is itself a
    // body cell (Sphere frame at face_depth == 0), `march_cartesian`'s
    // `kind == 1u` pre-clip never fires — the walker starts INSIDE
    // the body. We set this flag per-frame (cleared on ribbon pops
    // once the frame above IS Cartesian) so the inner DDA can skip
    // the body entirely when the ray misses the outer sphere.
    var frame_is_body_root: bool = node_kinds[current_idx].kind == 1u;

    // skip_slot: after a ribbon pop, the slot index (in the parent)
    // of the child we just left. march_cartesian skips this slot at
    // depth 0 to avoid re-entering the subtree already traversed by
    // the inner shell. Uses slot (not node_idx) for dedup correctness.
    var skip_slot: u32 = 0xFFFFFFFFu;

    var hops: u32 = 0u;
    loop {
        if hops > 80u { break; }
        hops = hops + 1u;

        // Ribbon-level LOD budget: the ancestor pop count is
        // the tree's native distance metric. Inside our anchor
        // cell (ribbon_level=0) we allow `BASE_DETAIL_DEPTH`
        // levels of descent; each additional shell (ribbon pop)
        // drops the budget by one, bottoming out at 1.
        let detail_budget = select(
            1u,
            BASE_DETAIL_DEPTH - ribbon_level,
            ribbon_level < BASE_DETAIL_DEPTH,
        );
        let cart_depth_limit = min(detail_budget, MAX_STACK_DEPTH);
        var r: HitResult;
        // If the current frame IS a body cell, do a ray-sphere test
        // against its `[0, 3)³` outer sphere first. Miss → skip the
        // inner DDA and go straight to ribbon pop (silhouette stays
        // round). Hit → descend normally.
        var skip_march: bool = false;
        if frame_is_body_root {
            let outer_r_root = node_kinds[current_idx].outer_r;
            let cs_center_root = vec3<f32>(1.5);
            let cs_outer_root = outer_r_root * 3.0;
            let oc_root = ray_origin - cs_center_root;
            let b_root = dot(oc_root, ray_dir);
            let c_root = dot(oc_root, oc_root) - cs_outer_root * cs_outer_root;
            let disc_root = b_root * b_root - c_root;
            skip_march = disc_root <= 0.0
                || (-b_root + sqrt(max(disc_root, 0.0))) <= 0.0;
        }
        if skip_march {
            r.hit = false;
            r.t = 1e20;
            r.frame_level = 0u;
            r.frame_scale = 1.0;
            r.cell_min = vec3<f32>(0.0);
            r.cell_size = 1.0;
        } else {
            r = march_cartesian(current_idx, ray_origin, ray_dir, cart_depth_limit, skip_slot);
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
        // Single-level ribbon pop with empty-shell fast-exit.
        //
        // Pop exactly one ancestor entry, transform the ray into
        // the ancestor's [0,3)³ frame, then fall through to the
        // outer loop which re-enters march_cartesian. Body nodes
        // along the ribbon are treated as Cartesian here — the
        // sphere-SDF pre-clip in march_cartesian's kind==1u branch
        // handles the silhouette when the walker descends into them.
        let entry = ribbon[ribbon_level];
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
        // Refresh the body-root flag for the popped frame. Most pops
        // land on a Cartesian ancestor; occasionally they land on a
        // body-as-ancestor (if a body contains another body, or if
        // the render_path passed through a body into a face subtree).
        frame_is_body_root = node_kinds[current_idx].kind == 1u;

        // Empty-shell fast exit: if every sibling is empty, skip
        // this shell's DDA and advance the ray to the shell's exit
        // boundary. Next outer iteration will pop again.
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
