// Inner DDA: march within a single frame rooted at `root_node_idx`.
//
// Per-depth state is split across three storage classes, chosen by
// how the state changes with depth:
//
//   register scalars — `cur_cell_size`, `cur_node_origin`, `cur_side_dist`:
//     mutated on every descend/pop but uniquely derivable from the
//     current depth alone. Maintaining a single scalar and updating
//     it on transition is equivalent to a per-depth array but uses
//     one register slot per field instead of MAX_STACK_DEPTH slots.
//     Matches the occupancy-stack-slim branch's trick: these three
//     fields are recoverable on pop (cell_size ×3, node_origin
//     subtracts the descend contribution, side_dist recomputed in
//     ~6 FMAs from entry_pos).
//
//   workgroup memory (this file's `s_*_tg` arrays) — `s_cell`, `s_node_idx`:
//     `s_cell` advances per-iteration in the CURRENT depth's slot
//     while staying pinned at parent depths (so parents can recover
//     their own state on pop). `s_node_idx` is read on OOB pop to
//     refresh the tree header. Neither can be replaced by a scalar
//     plus math — they're genuinely stateful per depth.
//
// Workgroup budget: MAX_STACK_DEPTH * TG_STRIDE slots × (16 B cell +
//  4 B node_idx) = 5 * 64 * 20 = 6400 B / workgroup. Well under the
// 32 KB pool so multiple workgroups share a core.
//
// Addressing: depth-major stride by TG_STRIDE (= @workgroup_size.x *
// @workgroup_size.y = 64) so adjacent threads in a SIMD group hit
// consecutive banks. Slot for thread `lid` at depth `d`:
//   index = d * TG_STRIDE + lid

#include "tree.wgsl"
#include "ray_prim.wgsl"
#include "sphere.wgsl"

const TG_STRIDE: u32 = 64u;  // matches @workgroup_size(8, 8, 1)
const TG_SLOTS: u32 = TG_STRIDE * MAX_STACK_DEPTH;

var<workgroup> s_node_idx_tg: array<u32,       TG_SLOTS>;
var<workgroup> s_cell_tg:     array<vec3<i32>, TG_SLOTS>;

fn march_cartesian(
    root_node_idx: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>,
    depth_limit: u32, skip_slot: u32, lid: u32,
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
    // After ribbon pops, ray_dir magnitude shrinks (÷3 per pop). LOD
    // pixel calculations need world-space distances, so scale
    // side_dist by ray_metric to get actual distance.
    let ray_metric = max(length(ray_dir), 1e-6);
    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );
    let delta_dist = abs(inv_dir);

    // Scalar stack fields — see module docstring. Initialised for
    // depth=0 here; updated on descend/pop below.
    var cur_cell_size: f32 = 1.0;
    var cur_node_origin: vec3<f32> = vec3<f32>(0.0);
    var cur_side_dist: vec3<f32>;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    s_node_idx_tg[0u * TG_STRIDE + lid] = root_node_idx;

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
    s_cell_tg[0u * TG_STRIDE + lid] = entry_cell;
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
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let depth_base = depth * TG_STRIDE + lid;
        let cell = s_cell_tg[depth_base];

        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth -= 1u;
            // Restore parent-depth scalars. cur_cell_size ×3 undoes
            // the descend divide; cur_node_origin subtracts the exact
            // vec we added on descend (s_cell[parent_depth] was
            // preserved while we were inside the child, so this is
            // byte-exact — no accumulated floating-point error).
            cur_cell_size = cur_cell_size * 3.0;
            let parent_base = depth * TG_STRIDE + lid;
            let parent_cell = s_cell_tg[parent_base];
            cur_node_origin = cur_node_origin - vec3<f32>(parent_cell) * cur_cell_size;
            // Recompute cur_side_dist from scratch at the parent
            // depth. Same formula as the descend-site init, same
            // entry_pos reference. ~6 FMAs per pop, amortized to
            // ~free vs. the per-thread register savings.
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

            let parent_header_off = node_offsets[s_node_idx_tg[parent_base]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];

            var new_cell = parent_cell;
            if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                new_cell.x += step.x;
                cur_side_dist.x += delta_dist.x * cur_cell_size;
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if cur_side_dist.y < cur_side_dist.z {
                new_cell.y += step.y;
                cur_side_dist.y += delta_dist.y * cur_cell_size;
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                new_cell.z += step.z;
                cur_side_dist.z += delta_dist.z * cur_cell_size;
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
            s_cell_tg[parent_base] = new_cell;
            continue;
        }

        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        let slot_bit = 1u << slot;
        if ((cur_occupancy & slot_bit) == 0u) {
            if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
            var new_cell = cell;
            if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                new_cell.x += step.x;
                cur_side_dist.x += delta_dist.x * cur_cell_size;
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if cur_side_dist.y < cur_side_dist.z {
                new_cell.y += step.y;
                cur_side_dist.y += delta_dist.y * cur_cell_size;
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                new_cell.z += step.z;
                cur_side_dist.z += delta_dist.z * cur_cell_size;
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
            s_cell_tg[depth_base] = new_cell;
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
            result.color = palette.colors[(packed >> 8u) & 0xFFu].rgb;
            result.normal = normal;
            result.cell_min = cell_min_h;
            result.cell_size = cur_cell_size;
            return result;
        } else {
            let child_idx = tree[child_base + 1u];

            let cell_slot = u32(cell.x) + u32(cell.y) * 3u + u32(cell.z) * 9u;
            if depth == 0u && cell_slot == skip_slot {
                var new_cell = cell;
                if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                    new_cell.x += step.x;
                    cur_side_dist.x += delta_dist.x * cur_cell_size;
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if cur_side_dist.y < cur_side_dist.z {
                    new_cell.y += step.y;
                    cur_side_dist.y += delta_dist.y * cur_cell_size;
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    new_cell.z += step.z;
                    cur_side_dist.z += delta_dist.z * cur_cell_size;
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                s_cell_tg[depth_base] = new_cell;
                continue;
            }

            let kind = node_kinds[child_idx].kind;

            if kind == 1u {
                let body_origin = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
                let body_size = cur_cell_size;
                let inner_r = node_kinds[child_idx].inner_r;
                let outer_r = node_kinds[child_idx].outer_r;
                let sph = sphere_in_cell(
                    child_idx, body_origin, body_size,
                    inner_r, outer_r, ray_origin, ray_dir,
                );
                if sph.hit {
                    return sph;
                }
                var new_cell = cell;
                if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                    new_cell.x += step.x;
                    cur_side_dist.x += delta_dist.x * cur_cell_size;
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if cur_side_dist.y < cur_side_dist.z {
                    new_cell.y += step.y;
                    cur_side_dist.y += delta_dist.y * cur_cell_size;
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    new_cell.z += step.z;
                    cur_side_dist.z += delta_dist.z * cur_cell_size;
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                s_cell_tg[depth_base] = new_cell;
                continue;
            }
            let child_bt = child_block_type(packed);
            if child_bt == 255u {
                if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
                var new_cell = cell;
                if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                    new_cell.x += step.x;
                    cur_side_dist.x += delta_dist.x * cur_cell_size;
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if cur_side_dist.y < cur_side_dist.z {
                    new_cell.y += step.y;
                    cur_side_dist.y += delta_dist.y * cur_cell_size;
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    new_cell.z += step.z;
                    cur_side_dist.z += delta_dist.z * cur_cell_size;
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                s_cell_tg[depth_base] = new_cell;
                continue;
            }

            let at_max = depth + 1u > depth_limit || depth + 1u >= MAX_STACK_DEPTH;
            let child_cell_size = cur_cell_size / 3.0;
            let cell_world_size = child_cell_size;
            let min_side = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
            let ray_dist = max(min_side * ray_metric, 0.001);
            let lod_pixels = cell_world_size / ray_dist * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;

            if at_max || at_lod {
                if ENABLE_STATS { ray_steps_lod_terminal = ray_steps_lod_terminal + 1u; }
                let bt = child_block_type(packed);
                if bt == 255u {
                    var new_cell = cell;
                    if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                        new_cell.x += step.x;
                        cur_side_dist.x += delta_dist.x * cur_cell_size;
                        normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                    } else if cur_side_dist.y < cur_side_dist.z {
                        new_cell.y += step.y;
                        cur_side_dist.y += delta_dist.y * cur_cell_size;
                        normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                    } else {
                        new_cell.z += step.z;
                        cur_side_dist.z += delta_dist.z * cur_cell_size;
                        normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                    }
                    s_cell_tg[depth_base] = new_cell;
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
                if ENABLE_STATS { ray_steps_node_descend = ray_steps_node_descend + 1u; }
                let parent_origin = cur_node_origin;
                let parent_cell_size = cur_cell_size;
                let child_origin = parent_origin + vec3<f32>(cell) * parent_cell_size;

                let child_max = child_origin + vec3<f32>(parent_cell_size);
                let child_hit = ray_box(ray_origin, inv_dir, child_origin, child_max);
                let ct_start = max(child_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
                let child_entry = ray_origin + ray_dir * ct_start;
                let local_entry = (child_entry - child_origin) / child_cell_size;

                depth += 1u;
                let new_base = depth * TG_STRIDE + lid;
                s_node_idx_tg[new_base] = child_idx;
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
                s_cell_tg[new_base] = new_cell;
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
