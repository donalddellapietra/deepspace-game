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

/// Flat DDA through a single brick node. A brick packs all 27 cells
/// of a Cartesian node's 3×3×3 grid as block-type bytes in 7 u32s,
/// right after the 2-u32 header. This function does NOT recurse —
/// every cell is terminal (block, or block_type=255 meaning empty).
/// Returns `hit=true` with the terminal cell, or `hit=false` when
/// the ray exits the brick without hitting anything (caller then
/// advances its parent DDA past this brick's cell as if it were an
/// empty LOD-terminal).
///
/// Caller must provide the initial `cur_cell` and `cur_side_dist`
/// already set up for the brick entry point — same math the tree
/// descent path uses to initialize its child's stack. Reusing the
/// caller's computed values avoids duplicating the ray_box +
/// local_entry divides that are already the only expensive part of
/// the brick-entry critical path.
///
/// `brick_data_off` points at the first of the 7 packed-data u32s
/// (= brick_header_off + 2). `brick_origin` / `brick_cell_size` are
/// the world-frame placement of the brick — brick_cell_size is the
/// size of EACH of the 3×3×3 cells inside, i.e. parent_cell_size/3
/// from the caller's descent frame.
fn march_brick(
    brick_data_off: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    brick_origin: vec3<f32>,
    brick_cell_size: f32,
    inv_dir: vec3<f32>,
    step: vec3<i32>,
    delta_dist: vec3<f32>,
    initial_cell: vec3<i32>,
    initial_side_dist: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = brick_cell_size;

    var cur_cell = initial_cell;
    var cur_side_dist = initial_side_dist;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var iterations = 0u;
    let max_brick_iter = 12u;

    if ENABLE_STATS { atomicAdd(&shader_stats.brick_entries, 1u); }

    loop {
        if iterations >= max_brick_iter {
            if ENABLE_STATS { atomicAdd(&shader_stats.brick_no_hits, 1u); }
            break;
        }
        iterations += 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        if cur_cell.x < 0 || cur_cell.x > 2
        || cur_cell.y < 0 || cur_cell.y > 2
        || cur_cell.z < 0 || cur_cell.z > 2 {
            if ENABLE_STATS { atomicAdd(&shader_stats.brick_no_hits, 1u); }
            return result;
        }

        let slot = u32(cur_cell.x) + u32(cur_cell.y) * 3u + u32(cur_cell.z) * 9u;
        let word = tree[brick_data_off + (slot >> 2u)];
        let bt = (word >> ((slot & 3u) * 8u)) & 0xFFu;

        if bt == BRICK_EMPTY_BT {
            if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
            if cur_side_dist.x < cur_side_dist.y && cur_side_dist.x < cur_side_dist.z {
                cur_cell.x += step.x;
                cur_side_dist.x += delta_dist.x * brick_cell_size;
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if cur_side_dist.y < cur_side_dist.z {
                cur_cell.y += step.y;
                cur_side_dist.y += delta_dist.y * brick_cell_size;
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                cur_cell.z += step.z;
                cur_side_dist.z += delta_dist.z * brick_cell_size;
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
            continue;
        }

        if ENABLE_STATS { ray_steps_lod_terminal = ray_steps_lod_terminal + 1u; }
        let cell_min = brick_origin + vec3<f32>(cur_cell) * brick_cell_size;
        let cell_max = cell_min + vec3<f32>(brick_cell_size);
        let cell_box = ray_box(ray_origin, inv_dir, cell_min, cell_max);
        result.hit = true;
        result.t = max(cell_box.t_enter, 0.0);
        if ENABLE_STATS {
            if iterations == 1u {
                atomicAdd(&shader_stats.brick_first_cell_hits, 1u);
            } else {
                atomicAdd(&shader_stats.brick_advance_hits, 1u);
            }
        }
        result.color = palette.colors[bt].rgb;
        result.normal = normal;
        result.cell_min = cell_min;
        result.cell_size = brick_cell_size;
        return result;
    }

    return result;
}

fn march_cartesian(
    root_node_idx: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>,
    skip_slot: u32, lid: u32,
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

    // ── REGISTER PRESSURE TEST: +256 B of live scalar state ──
    var w0 = vec4<f32>(ray_dir, 0.0);
    var w1 = vec4<f32>(ray_dir.yzx, 1.0);
    var w2 = vec4<f32>(ray_dir.zxy, 2.0);
    var w3 = vec4<f32>(ray_dir * 2.0, 3.0);
    var w4 = vec4<f32>(ray_dir * 3.0, 4.0);
    var w5 = vec4<f32>(ray_dir * 4.0, 5.0);
    var w6 = vec4<f32>(ray_dir * 5.0, 6.0);
    var w7 = vec4<f32>(ray_dir * 6.0, 7.0);
    var w8 = vec4<f32>(ray_dir * 7.0, 8.0);
    var w9 = vec4<f32>(ray_dir * 8.0, 9.0);
    var w10 = vec4<f32>(ray_dir * 9.0, 10.0);
    var w11 = vec4<f32>(ray_dir * 10.0, 11.0);
    var w12 = vec4<f32>(ray_dir * 11.0, 12.0);
    var w13 = vec4<f32>(ray_dir * 12.0, 13.0);
    var w14 = vec4<f32>(ray_dir * 13.0, 14.0);
    var w15 = vec4<f32>(ray_dir * 14.0, 15.0);

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

    // Keep waste live — guard never fires but keeps state resident.
    if w0.x + w1.x + w2.x + w3.x + w4.x + w5.x + w6.x + w7.x
     + w8.x + w9.x + w10.x + w11.x + w12.x + w13.x + w14.x + w15.x > 1e30 {
        return result;
    }

    let root_header_off = node_offsets[root_node_idx];
    var cur_occupancy: u32 = tree[root_header_off];

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

    // If the frame root itself is brick-packed, short-circuit the
    // sparse-descent loop. The root's entry-setup (cur_cell,
    // cur_side_dist) above is exactly what march_brick needs since
    // the root brick is at origin (0,0,0) with cell size 1 — the
    // same coordinate system as the root frame.
    if (cur_occupancy & BRICK_FLAG_BIT) != 0u {
        return march_brick(
            root_header_off + 2u,
            ray_origin, ray_dir,
            vec3<f32>(0.0), 1.0,
            inv_dir, step, delta_dist,
            cur_cell, cur_side_dist,
        );
    }

    var cur_first_child: u32 = tree[root_header_off + 1u];

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

            // Stack-safety cap: shader can hold MAX_STACK_DEPTH levels
            // of parent state in TG memory, so depth cannot exceed
            // MAX_STACK_DEPTH - 1. Normally the packer force-collapses
            // at DEFAULT_LOD_LEAF_DEPTH = MAX_STACK_DEPTH - 1, so this
            // only fires inside preserve-regions where deeper tag=2
            // subtrees survive the pack-time collapse.
            let at_stack_cap = depth + 1u >= MAX_STACK_DEPTH;
            let child_cell_size = cur_cell_size / 3.0;
            // Per-pixel LOD: terminate if the child cell projects to
            // less than LOD_PIXEL_THRESHOLD pixels on screen.
            let min_side = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
            let ray_dist = max(min_side * ray_metric, 0.001);
            let lod_pixels = child_cell_size / ray_dist * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;

            if at_stack_cap || at_lod {
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

                let parent_origin = cur_node_origin;
                let parent_cell_size = cur_cell_size;
                let child_origin = parent_origin + vec3<f32>(cur_cell) * parent_cell_size;

                // Shared entry-setup for both brick and tree descent.
                // Computing once (rather than duplicating inside
                // march_brick) is the critical bit for brick perf —
                // the 3 divides in local_entry and the vec3 side_dist
                // init add up to ~50 ALU which dominates short brick
                // walks otherwise.
                let child_max = child_origin + vec3<f32>(parent_cell_size);
                let child_hit = ray_box(ray_origin, inv_dir, child_origin, child_max);
                let ct_start = max(child_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
                let child_entry = ray_origin + ray_dir * ct_start;
                let local_entry = (child_entry - child_origin) / child_cell_size;
                let new_cell = vec3<i32>(
                    clamp(i32(floor(local_entry.x)), 0, 2),
                    clamp(i32(floor(local_entry.y)), 0, 2),
                    clamp(i32(floor(local_entry.z)), 0, 2),
                );
                let lc = vec3<f32>(new_cell);
                // Brick path references ray_origin; tree path uses
                // entry_pos (matches the root init so side_dist values
                // at different depths share a reference). Two
                // initializations because the DDA reference for a
                // brick is self-contained — no relationship to the
                // outer entry_pos.
                let new_side_dist_tree = vec3<f32>(
                    select((child_origin.x + lc.x * child_cell_size - entry_pos.x) * inv_dir.x,
                           (child_origin.x + (lc.x + 1.0) * child_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                    select((child_origin.y + lc.y * child_cell_size - entry_pos.y) * inv_dir.y,
                           (child_origin.y + (lc.y + 1.0) * child_cell_size - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
                    select((child_origin.z + lc.z * child_cell_size - entry_pos.z) * inv_dir.z,
                           (child_origin.z + (lc.z + 1.0) * child_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
                );

                // Peek at the child's header to decide between brick
                // (flat inline DDA, no depth change) and tree (push
                // parent state + recurse).
                let child_header_off = node_offsets[child_idx];
                let child_occ_peek = tree[child_header_off];

                if (child_occ_peek & BRICK_FLAG_BIT) != 0u {
                    // Brick child: flat DDA inside its 3×3×3, no
                    // stack push. Hit → return; miss → advance parent
                    // DDA past this cell same as an empty LOD-terminal
                    // and continue at current depth.
                    //
                    // Reuse `new_side_dist_tree` for the brick's
                    // initial side_dist: the DDA's axis-choice logic
                    // (`min(side_dist.x, side_dist.y, side_dist.z)`)
                    // is invariant under constant offset of the
                    // reference point, and `side_dist += delta *
                    // cell_size` per iteration preserves that offset
                    // as well. Using entry_pos as the brick's
                    // reference (like the tree) costs nothing and
                    // shares the vec3-select init — about 18 ALU per
                    // brick entry saved.
                    let brick_result = march_brick(
                        child_header_off + 2u,
                        ray_origin, ray_dir,
                        child_origin, child_cell_size,
                        inv_dir, step, delta_dist,
                        new_cell, new_side_dist_tree,
                    );
                    if brick_result.hit {
                        return brick_result;
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

                // Non-brick child: normal descent. Save parent's cell
                // + node to TG before pushing the stack.
                let parent_base = depth * TG_STRIDE + lid;
                s_cell_tg[parent_base] = cur_cell;
                s_node_idx_tg[parent_base] = cur_node_idx;

                depth += 1u;
                cur_node_idx = child_idx;
                cur_node_origin = child_origin;
                cur_cell_size = child_cell_size;
                cur_occupancy = child_occ_peek;
                cur_first_child = tree[child_header_off + 1u];
                cur_cell = new_cell;
                cur_side_dist = new_side_dist_tree;
            }
        }
    }

    return result;
}
