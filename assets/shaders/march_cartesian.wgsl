// Inner DDA: march within a single frame rooted at `root_node_idx`.
//
// Per-depth state is split across three storage classes, chosen by
// how the state changes with depth:
//
//   register scalars — `cur_cell_size`, `cur_node_origin`,
//                       `cur_side_dist`, `cur_cell`, `cur_node_idx`:
//     shadow the current depth's stack slot in a thread-private
//     register. The inner DDA loop reads and mutates them every
//     iteration for the cost of a register access — no workgroup-
//     memory round-trip, no address math, no load latency.
//
//   workgroup memory (`s_cell_tg`, `s_node_idx_tg`) — PARENT depths only:
//     written once when we descend (save the parent's final state so
//     a later pop can recover it), read once when we pop (restore
//     what the parent was doing). The current depth's state stays in
//     registers; TG memory is used as the "depth stack" storage.
//     This shrinks the per-iteration TG traffic from 2 ops/iter to 0.
//
// `cur_cell` at the current depth is read and mutated every DDA
// iteration, but the WGSL compiler can't keep an array slot
// (`s_cell_tg[depth_base]`) in a register across iterations because
// workgroup memory has shared-aliasing semantics (another thread
// could theoretically write there). Shadowing it in a local `var`
// tells the compiler "this is thread-private" and it lives in a
// register. Same trick as the three scalar fields above, applied to
// the remaining stack arrays.
//
// Workgroup budget: MAX_STACK_DEPTH * TG_STRIDE slots × (16 B cell +
// 4 B node_idx) = 5 * 64 * 20 = 6400 B / workgroup. Unchanged from
// the previous commit — the arrays are still sized for full-depth
// save/restore, just accessed less often.
//
// Addressing: depth-major stride by TG_STRIDE (= @workgroup_size.x *
// @workgroup_size.y = 64) so adjacent threads hit consecutive banks.
//   index = depth * TG_STRIDE + lid

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

    // All the "stack-slim" scalars (initial values for depth=0).
    var cur_cell_size: f32 = 1.0;
    var cur_node_origin: vec3<f32> = vec3<f32>(0.0);
    var cur_side_dist: vec3<f32>;
    // Register-scalar shadows for s_cell / s_node_idx at the current
    // depth. Mutated every iteration; TG backing store is only
    // touched on depth transitions.
    var cur_cell: vec3<i32>;
    var cur_node_idx: u32 = root_node_idx;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    let root_header_off = node_offsets[root_node_idx];
    var cur_occupancy: u32 = tree[root_header_off];
    var cur_first_child: u32 = tree[root_header_off + 1u];

    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }

    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = ray_origin + ray_dir * t_start;

    cur_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );
    let cell_f = vec3<f32>(cur_cell);
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

        if cur_cell.x < 0 || cur_cell.x > 2 || cur_cell.y < 0 || cur_cell.y > 2 || cur_cell.z < 0 || cur_cell.z > 2 {
            if depth == 0u { break; }
            depth -= 1u;
            // Restore parent-depth scalars. cur_cell_size ×3 undoes
            // the descend divide; cur_node_origin subtracts the exact
            // vec we added on descend (saved parent cell lives in TG
            // because we wrote it there on descend). Byte-exact — no
            // accumulated floating-point error.
            cur_cell_size = cur_cell_size * 3.0;
            let parent_base = depth * TG_STRIDE + lid;
            // Reload parent's saved cell + node into registers.
            cur_cell = s_cell_tg[parent_base];
            cur_node_idx = s_node_idx_tg[parent_base];
            cur_node_origin = cur_node_origin - vec3<f32>(cur_cell) * cur_cell_size;
            // Recompute cur_side_dist from scratch at the parent
            // depth. Same formula as the descend-site init — same
            // entry_pos reference. ~6 FMAs per pop, amortized to
            // ~free vs. the per-thread register savings.
            let lc_pop = vec3<f32>(cur_cell);
            cur_side_dist = vec3<f32>(
                select((cur_node_origin.x + lc_pop.x * cur_cell_size - entry_pos.x) * inv_dir.x,
                       (cur_node_origin.x + (lc_pop.x + 1.0) * cur_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                select((cur_node_origin.y + lc_pop.y * cur_cell_size - entry_pos.y) * inv_dir.y,
                       (cur_node_origin.y + (lc_pop.y + 1.0) * cur_cell_size - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
                select((cur_node_origin.z + lc_pop.z * cur_cell_size - entry_pos.z) * inv_dir.z,
                       (cur_node_origin.z + (lc_pop.z + 1.0) * cur_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
            );
            if ENABLE_STATS { ray_steps_oob = ray_steps_oob + 1u; }

            let parent_header_off = node_offsets[cur_node_idx];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];

            if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                cur_cell.x += step.x;
                cur_side_dist.x += delta_dist.x * cur_cell_size;
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if cur_side_dist.y < cur_side_dist.z {
                cur_cell.y += step.y;
                cur_side_dist.y += delta_dist.y * cur_cell_size;
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                cur_cell.z += step.z;
                cur_side_dist.z += delta_dist.z * cur_cell_size;
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
            continue;
        }

        let slot = slot_from_xyz(cur_cell.x, cur_cell.y, cur_cell.z);
        let slot_bit = 1u << slot;
        if ((cur_occupancy & slot_bit) == 0u) {
            if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
            if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                cur_cell.x += step.x;
                cur_side_dist.x += delta_dist.x * cur_cell_size;
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if cur_side_dist.y < cur_side_dist.z {
                cur_cell.y += step.y;
                cur_side_dist.y += delta_dist.y * cur_cell_size;
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                cur_cell.z += step.z;
                cur_side_dist.z += delta_dist.z * cur_cell_size;
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
            continue;
        }

        let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;

        if tag == 1u {
            let cell_min_h = cur_node_origin + vec3<f32>(cur_cell) * cur_cell_size;
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

            let cell_slot = u32(cur_cell.x) + u32(cur_cell.y) * 3u + u32(cur_cell.z) * 9u;
            if depth == 0u && cell_slot == skip_slot {
                if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                    cur_cell.x += step.x;
                    cur_side_dist.x += delta_dist.x * cur_cell_size;
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if cur_side_dist.y < cur_side_dist.z {
                    cur_cell.y += step.y;
                    cur_side_dist.y += delta_dist.y * cur_cell_size;
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    cur_cell.z += step.z;
                    cur_side_dist.z += delta_dist.z * cur_cell_size;
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                continue;
            }

            let kind = node_kinds[child_idx].kind;

            if kind == 1u {
                let body_origin = cur_node_origin + vec3<f32>(cur_cell) * cur_cell_size;
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
                if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                    cur_cell.x += step.x;
                    cur_side_dist.x += delta_dist.x * cur_cell_size;
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if cur_side_dist.y < cur_side_dist.z {
                    cur_cell.y += step.y;
                    cur_side_dist.y += delta_dist.y * cur_cell_size;
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    cur_cell.z += step.z;
                    cur_side_dist.z += delta_dist.z * cur_cell_size;
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                continue;
            }
            let child_bt = child_block_type(packed);
            if child_bt == 255u {
                if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
                if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                    cur_cell.x += step.x;
                    cur_side_dist.x += delta_dist.x * cur_cell_size;
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if cur_side_dist.y < cur_side_dist.z {
                    cur_cell.y += step.y;
                    cur_side_dist.y += delta_dist.y * cur_cell_size;
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    cur_cell.z += step.z;
                    cur_side_dist.z += delta_dist.z * cur_cell_size;
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
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
                    if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                        cur_cell.x += step.x;
                        cur_side_dist.x += delta_dist.x * cur_cell_size;
                        normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                    } else if cur_side_dist.y < cur_side_dist.z {
                        cur_cell.y += step.y;
                        cur_side_dist.y += delta_dist.y * cur_cell_size;
                        normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                    } else {
                        cur_cell.z += step.z;
                        cur_side_dist.z += delta_dist.z * cur_cell_size;
                        normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                    }
                } else {
                    let cell_min_l = cur_node_origin + vec3<f32>(cur_cell) * cur_cell_size;
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
                // Save parent's cell + node to TG before we descend.
                // These are the values a later pop will load back
                // into cur_cell / cur_node_idx to resume the parent.
                let parent_base = depth * TG_STRIDE + lid;
                s_cell_tg[parent_base] = cur_cell;
                s_node_idx_tg[parent_base] = cur_node_idx;

                let parent_origin = cur_node_origin;
                let parent_cell_size = cur_cell_size;
                let child_origin = parent_origin + vec3<f32>(cur_cell) * parent_cell_size;

                let child_max = child_origin + vec3<f32>(parent_cell_size);
                let child_hit = ray_box(ray_origin, inv_dir, child_origin, child_max);
                let ct_start = max(child_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
                let child_entry = ray_origin + ray_dir * ct_start;
                let local_entry = (child_entry - child_origin) / child_cell_size;

                depth += 1u;
                cur_node_idx = child_idx;
                cur_node_origin = child_origin;
                cur_cell_size = child_cell_size;
                let child_header_off = node_offsets[child_idx];
                cur_occupancy = tree[child_header_off];
                cur_first_child = tree[child_header_off + 1u];
                cur_cell = vec3<i32>(
                    clamp(i32(floor(local_entry.x)), 0, 2),
                    clamp(i32(floor(local_entry.y)), 0, 2),
                    clamp(i32(floor(local_entry.z)), 0, 2),
                );
                let lc = vec3<f32>(cur_cell);
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
