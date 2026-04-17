// Inner DDA: march within a single frame rooted at `root_node_idx`.
//
// Per-depth state is split across three storage classes, chosen by
// how the state changes with depth:
//
//   register scalars — `cur_cell_size`, `cur_node_origin`,
//                       `cur_side_dist`, `cur_cell`, `cur_node_idx`:
//     shadow the current depth's stack slot in a thread-private
//     register. The inner DDA loop reads and mutates them every
//     iteration for the cost of a register access.
//
//   workgroup memory (`s_cell_tg`, `s_node_idx_tg`) — PARENT depths only:
//     written once when we descend (save the parent's final state so
//     a later pop can recover it), read once when we pop.
//
// `cur_cell` at the current depth is read and mutated every DDA
// iteration; shadowing it in a thread-local `var` keeps it in a
// register instead of hitting workgroup-shared memory every step.
//
// Addressing: depth-major stride by TG_STRIDE (= @workgroup_size.x *
// @workgroup_size.y = 64) so adjacent threads hit consecutive banks.
//
// Bricks: `march_brick` handles first-class `NodeKind::Brick`
// leaves. The brick header's side code (bits 28-29) tells us whether
// the brick is a 3³, 9³, or 27³ dense grid; march_brick walks the
// grid with a flat byte read per cell — no popcount, no rank, no
// tag dispatch. The caller (this `march_cartesian`) detects the
// brick by peeking the child's header word, decodes the side, and
// sets up the initial cell + side_dist at the brick's own cell
// resolution (parent_cell_size / side).

#include "tree.wgsl"
#include "ray_prim.wgsl"
#include "sphere.wgsl"

const TG_STRIDE: u32 = 64u;  // matches @workgroup_size(8, 8, 1)
const TG_SLOTS: u32 = TG_STRIDE * MAX_STACK_DEPTH;

var<workgroup> s_node_idx_tg: array<u32,       TG_SLOTS>;
var<workgroup> s_cell_tg:     array<vec3<i32>, TG_SLOTS>;

// ───────── brick header decoding ─────────
// Bits 28-29 of a brick's first u32 carry the side code:
//   0 → side = 3, 1 → side = 9, 2 → side = 27.
fn brick_side_from_header(header: u32) -> u32 {
    let code = (header >> 28u) & 3u;
    return select(select(27u, 9u, code == 1u), 3u, code == 0u);
}

// ───────── march_brick ─────────
// Flat DDA across a brick's `side³` cells. Returns `hit=true` with
// the first non-empty cell along the ray, or `hit=false` when the
// ray exits the brick without hitting any solid cell.
//
// Cell indexing: `slot = x + y*side + z*side²`, with the per-slot
// block-type byte stored at `cells_u32[slot/4] >> ((slot%4)*8)`.
// Empty cells carry `BRICK_EMPTY_BT` (255).
//
// The caller pre-computes the initial `cur_cell` and `cur_side_dist`
// at the brick's own cell resolution. max-iter cap scales with side:
// a ray traverses at most ~3·side cells along a brick's diagonal.
fn march_brick(
    brick_data_off: u32,
    side: u32,
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
    // Ray can traverse up to 3·side cells along a brick diagonal;
    // give a healthy margin for float-precision drift.
    let max_brick_iter = 4u * side;
    let side_i = i32(side);
    let side_sq = side * side;

    if ENABLE_STATS { atomicAdd(&shader_stats.brick_entries, 1u); }

    loop {
        if iterations >= max_brick_iter {
            if ENABLE_STATS { atomicAdd(&shader_stats.brick_no_hits, 1u); }
            break;
        }
        iterations += 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        if cur_cell.x < 0 || cur_cell.x >= side_i
        || cur_cell.y < 0 || cur_cell.y >= side_i
        || cur_cell.z < 0 || cur_cell.z >= side_i {
            if ENABLE_STATS { atomicAdd(&shader_stats.brick_no_hits, 1u); }
            return result;
        }

        let slot = u32(cur_cell.x) + u32(cur_cell.y) * side + u32(cur_cell.z) * side_sq;
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

// ───────── march_cartesian ─────────

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
    // After ribbon pops, ray_dir magnitude shrinks (÷3 per pop); LOD
    // pixel calculations need world-space distances, so scale
    // side_dist by ray_metric.
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

    // Stack-slim scalars for the current depth.
    var cur_cell_size: f32 = 1.0;
    var cur_node_origin: vec3<f32> = vec3<f32>(0.0);
    var cur_side_dist: vec3<f32>;
    var cur_cell: vec3<i32>;
    var cur_node_idx: u32 = root_node_idx;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    // Keep the waste state live (guard never fires).
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

    // ── root-brick shortcut ─────────────
    // If the frame root is itself a brick, go straight to march_brick.
    // The root's brick cell_size depends on its side: side-3 has
    // cells of size 1 (same as Cartesian at depth 0), side-9 has
    // cells of size 1/3, side-27 has cells of size 1/9.
    if (cur_occupancy & BRICK_FLAG_BIT) != 0u {
        let root_brick_side = brick_side_from_header(cur_occupancy);
        let root_brick_cell_size = 3.0 / f32(root_brick_side);
        let root_side_i = i32(root_brick_side);
        let entry_local = entry_pos;
        let init_cell = vec3<i32>(
            clamp(i32(floor(entry_local.x / root_brick_cell_size)), 0, root_side_i - 1),
            clamp(i32(floor(entry_local.y / root_brick_cell_size)), 0, root_side_i - 1),
            clamp(i32(floor(entry_local.z / root_brick_cell_size)), 0, root_side_i - 1),
        );
        let lc = vec3<f32>(init_cell);
        let init_side_dist = vec3<f32>(
            select((lc.x * root_brick_cell_size - entry_local.x) * inv_dir.x,
                   ((lc.x + 1.0) * root_brick_cell_size - entry_local.x) * inv_dir.x, ray_dir.x >= 0.0),
            select((lc.y * root_brick_cell_size - entry_local.y) * inv_dir.y,
                   ((lc.y + 1.0) * root_brick_cell_size - entry_local.y) * inv_dir.y, ray_dir.y >= 0.0),
            select((lc.z * root_brick_cell_size - entry_local.z) * inv_dir.z,
                   ((lc.z + 1.0) * root_brick_cell_size - entry_local.z) * inv_dir.z, ray_dir.z >= 0.0),
        );
        return march_brick(
            root_header_off + 2u,
            root_brick_side,
            ray_origin, ray_dir,
            vec3<f32>(0.0), root_brick_cell_size,
            inv_dir, step, delta_dist,
            init_cell, init_side_dist,
        );
    }

    var cur_first_child: u32 = tree[root_header_off + 1u];

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

        // Out-of-bounds → pop to parent depth.
        if cur_cell.x < 0 || cur_cell.x > 2 || cur_cell.y < 0 || cur_cell.y > 2 || cur_cell.z < 0 || cur_cell.z > 2 {
            if depth == 0u { break; }
            depth -= 1u;
            cur_cell_size = cur_cell_size * 3.0;
            let parent_base = depth * TG_STRIDE + lid;
            cur_cell = s_cell_tg[parent_base];
            cur_node_idx = s_node_idx_tg[parent_base];
            cur_node_origin = cur_node_origin - vec3<f32>(cur_cell) * cur_cell_size;
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
        }

        // tag == 2: pointer to a child node.
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

        // ── sphere body child ──
        if kind == 1u {
            let body_origin = cur_node_origin + vec3<f32>(cur_cell) * cur_cell_size;
            let body_size = cur_cell_size;
            let inner_r = node_kinds[child_idx].inner_r;
            let outer_r = node_kinds[child_idx].outer_r;
            let sph = sphere_in_cell(
                child_idx, body_origin, body_size,
                inner_r, outer_r, ray_origin, ray_dir,
            );
            if sph.hit { return sph; }
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

        // ── stored rep_block sentinel ──
        // If the stored rep_block is empty (255), the shader treats
        // this slot as air — advance past without descending.
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

        // ── LOD + stack-cap terminal ──
        let child_cell_size = cur_cell_size / 3.0;
        let at_stack_cap = depth + 1u >= MAX_STACK_DEPTH;
        let min_side = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
        let ray_dist = max(min_side * ray_metric, 0.001);
        let lod_pixels = child_cell_size / ray_dist * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
        let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;

        if at_stack_cap || at_lod {
            if ENABLE_STATS { ray_steps_lod_terminal = ray_steps_lod_terminal + 1u; }
            let cell_min_l = cur_node_origin + vec3<f32>(cur_cell) * cur_cell_size;
            let cell_max_l = cell_min_l + vec3<f32>(cur_cell_size);
            let cell_box_l = ray_box(ray_origin, inv_dir, cell_min_l, cell_max_l);
            result.hit = true;
            result.t = max(cell_box_l.t_enter, 0.0);
            result.color = palette.colors[child_bt].rgb;
            result.normal = normal;
            result.cell_min = cell_min_l;
            result.cell_size = cur_cell_size;
            return result;
        }

        // ── descent ──
        if ENABLE_STATS { ray_steps_node_descend = ray_steps_node_descend + 1u; }

        let parent_origin = cur_node_origin;
        let parent_cell_size = cur_cell_size;
        let child_origin = parent_origin + vec3<f32>(cur_cell) * parent_cell_size;

        // Peek child header to decide brick vs sparse descent.
        let child_header_off = node_offsets[child_idx];
        let child_occ_peek = tree[child_header_off];

        if (child_occ_peek & BRICK_FLAG_BIT) != 0u {
            // ── brick child ──
            // Compute the brick's own cell size from its side.
            let brick_side = brick_side_from_header(child_occ_peek);
            let brick_cell_size = parent_cell_size / f32(brick_side);
            let brick_side_i = i32(brick_side);

            // Resolve the ray's entry point into the brick. Use a
            // FRESH ray_box against the brick's actual volume and
            // snap just inside to land in the first cell. Epsilon is
            // relative to brick_cell_size so it's proportional to
            // the smallest feature we care about.
            let child_max = child_origin + vec3<f32>(parent_cell_size);
            let child_hit = ray_box(ray_origin, inv_dir, child_origin, child_max);
            let ct_start = max(child_hit.t_enter, 0.0) + 0.0001 * brick_cell_size;
            let child_entry = ray_origin + ray_dir * ct_start;

            // Compute brick-local entry cell. `local_brick` in
            // [0, side). Clamp because ray_box precision near a
            // corner can put local_brick just outside the valid
            // range.
            let local_brick = (child_entry - child_origin) / brick_cell_size;
            let new_cell_brick = vec3<i32>(
                clamp(i32(floor(local_brick.x)), 0, brick_side_i - 1),
                clamp(i32(floor(local_brick.y)), 0, brick_side_i - 1),
                clamp(i32(floor(local_brick.z)), 0, brick_side_i - 1),
            );
            let lc = vec3<f32>(new_cell_brick);
            // side_dist references entry_pos (root-frame) to match
            // the sparse-descent convention. DDA stepping only cares
            // about invariant offsets; any consistent reference works.
            let new_side_dist_brick = vec3<f32>(
                select((child_origin.x + lc.x * brick_cell_size - entry_pos.x) * inv_dir.x,
                       (child_origin.x + (lc.x + 1.0) * brick_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                select((child_origin.y + lc.y * brick_cell_size - entry_pos.y) * inv_dir.y,
                       (child_origin.y + (lc.y + 1.0) * brick_cell_size - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
                select((child_origin.z + lc.z * brick_cell_size - entry_pos.z) * inv_dir.z,
                       (child_origin.z + (lc.z + 1.0) * brick_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
            );

            let brick_result = march_brick(
                child_header_off + 2u,
                brick_side,
                ray_origin, ray_dir,
                child_origin, brick_cell_size,
                inv_dir, step, delta_dist,
                new_cell_brick, new_side_dist_brick,
            );
            if brick_result.hit {
                return brick_result;
            }
            // Brick miss → advance parent DDA past this cell.
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

        // ── sparse-node descent ──
        // Child is a recursive Cartesian/Sphere/Face node: push parent
        // state to TG, drop into the child's frame.
        let child_hit = ray_box(ray_origin, inv_dir, child_origin, child_origin + vec3<f32>(parent_cell_size));
        let ct_start = max(child_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
        let child_entry = ray_origin + ray_dir * ct_start;
        let local_entry = (child_entry - child_origin) / child_cell_size;
        let new_cell = vec3<i32>(
            clamp(i32(floor(local_entry.x)), 0, 2),
            clamp(i32(floor(local_entry.y)), 0, 2),
            clamp(i32(floor(local_entry.z)), 0, 2),
        );
        let lc = vec3<f32>(new_cell);
        let new_side_dist = vec3<f32>(
            select((child_origin.x + lc.x * child_cell_size - entry_pos.x) * inv_dir.x,
                   (child_origin.x + (lc.x + 1.0) * child_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
            select((child_origin.y + lc.y * child_cell_size - entry_pos.y) * inv_dir.y,
                   (child_origin.y + (lc.y + 1.0) * child_cell_size - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
            select((child_origin.z + lc.z * child_cell_size - entry_pos.z) * inv_dir.z,
                   (child_origin.z + (lc.z + 1.0) * child_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
        );

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
        cur_side_dist = new_side_dist;
    }

    return result;
}
