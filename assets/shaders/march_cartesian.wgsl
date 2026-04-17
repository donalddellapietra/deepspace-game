// Inner DDA: march within a single frame rooted at `root_node_idx`.
// Per-depth stack state lives in workgroup (threadgroup) memory,
// hoisted out of the register file to raise Apple Silicon occupancy.
//
// Layout: depth-major stride by `TG_STRIDE` (= @workgroup_size.x *
// @workgroup_size.y = 64) so adjacent threads in a SIMD group read
// consecutive banks. Slot for thread `lid` at stack depth `depth`:
//   index = depth * TG_STRIDE + lid
//
// Budget: MAX_STACK_DEPTH * TG_STRIDE = 5 * 64 = 320 slots.
//   s_node_idx_tg:    320 × u32       =  1280 B
//   s_cell_tg:        320 × vec3<i32> =  5120 B (16-byte aligned)
//   s_side_dist_tg:   320 × vec3<f32> =  5120 B
//   s_node_origin_tg: 320 × vec3<f32> =  5120 B
//   s_cell_size_tg:   320 × f32       =  1280 B
//                                    = 17920 B / workgroup

#include "tree.wgsl"
#include "ray_prim.wgsl"
#include "sphere.wgsl"

const TG_STRIDE: u32 = 64u;  // matches @workgroup_size(8, 8, 1)
const TG_SLOTS: u32 = TG_STRIDE * MAX_STACK_DEPTH;

var<workgroup> s_node_idx_tg:    array<u32,       TG_SLOTS>;
var<workgroup> s_cell_tg:        array<vec3<i32>, TG_SLOTS>;
var<workgroup> s_side_dist_tg:   array<vec3<f32>, TG_SLOTS>;
var<workgroup> s_node_origin_tg: array<vec3<f32>, TG_SLOTS>;
var<workgroup> s_cell_size_tg:   array<f32,       TG_SLOTS>;

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
    let ray_metric = max(length(ray_dir), 1e-6);
    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );
    let delta_dist = abs(inv_dir);

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    s_node_idx_tg[0u * TG_STRIDE + lid] = root_node_idx;
    s_node_origin_tg[0u * TG_STRIDE + lid] = vec3<f32>(0.0);
    s_cell_size_tg[0u * TG_STRIDE + lid] = 1.0;

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
    s_side_dist_tg[0u * TG_STRIDE + lid] = vec3<f32>(
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
        let side_dist = s_side_dist_tg[depth_base];
        let cell_size = s_cell_size_tg[depth_base];

        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth -= 1u;
            if ENABLE_STATS { ray_steps_oob = ray_steps_oob + 1u; }

            let new_base = depth * TG_STRIDE + lid;
            let parent_header_off = node_offsets[s_node_idx_tg[new_base]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];

            var sd = s_side_dist_tg[new_base];
            let cs = s_cell_size_tg[new_base];
            var cl = s_cell_tg[new_base];
            if sd.x < sd.y && sd.x < sd.z {
                cl.x += step.x;
                sd.x += delta_dist.x * cs;
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if sd.y < sd.z {
                cl.y += step.y;
                sd.y += delta_dist.y * cs;
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                cl.z += step.z;
                sd.z += delta_dist.z * cs;
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
            s_cell_tg[new_base] = cl;
            s_side_dist_tg[new_base] = sd;
            continue;
        }

        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        let slot_bit = 1u << slot;
        if ((cur_occupancy & slot_bit) == 0u) {
            if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
            var cl = cell;
            var sd = side_dist;
            if sd.x < sd.y && sd.x < sd.z {
                cl.x += step.x;
                sd.x += delta_dist.x * cell_size;
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if sd.y < sd.z {
                cl.y += step.y;
                sd.y += delta_dist.y * cell_size;
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                cl.z += step.z;
                sd.z += delta_dist.z * cell_size;
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
            s_cell_tg[depth_base] = cl;
            s_side_dist_tg[depth_base] = sd;
            continue;
        }

        let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;

        if tag == 1u {
            let node_origin = s_node_origin_tg[depth_base];
            let cell_min_h = node_origin + vec3<f32>(cell) * cell_size;
            let cell_max_h = cell_min_h + vec3<f32>(cell_size);
            let cell_box_h = ray_box(ray_origin, inv_dir, cell_min_h, cell_max_h);
            result.hit = true;
            result.t = max(cell_box_h.t_enter, 0.0);
            result.color = palette.colors[(packed >> 8u) & 0xFFu].rgb;
            result.normal = normal;
            result.cell_min = cell_min_h;
            result.cell_size = cell_size;
            return result;
        } else {
            let child_idx = tree[child_base + 1u];

            let cell_slot = u32(cell.x) + u32(cell.y) * 3u + u32(cell.z) * 9u;
            if depth == 0u && cell_slot == skip_slot {
                var cl = cell;
                var sd = side_dist;
                if sd.x < sd.y && sd.x < sd.z {
                    cl.x += step.x;
                    sd.x += delta_dist.x * cell_size;
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if sd.y < sd.z {
                    cl.y += step.y;
                    sd.y += delta_dist.y * cell_size;
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    cl.z += step.z;
                    sd.z += delta_dist.z * cell_size;
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                s_cell_tg[depth_base] = cl;
                s_side_dist_tg[depth_base] = sd;
                continue;
            }

            let kind = node_kinds[child_idx].kind;

            if kind == 1u {
                let node_origin = s_node_origin_tg[depth_base];
                let body_origin = node_origin + vec3<f32>(cell) * cell_size;
                let body_size = cell_size;
                let inner_r = node_kinds[child_idx].inner_r;
                let outer_r = node_kinds[child_idx].outer_r;
                let sph = sphere_in_cell(
                    child_idx, body_origin, body_size,
                    inner_r, outer_r, ray_origin, ray_dir,
                );
                if sph.hit {
                    return sph;
                }
                var cl = cell;
                var sd = side_dist;
                if sd.x < sd.y && sd.x < sd.z {
                    cl.x += step.x;
                    sd.x += delta_dist.x * cell_size;
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if sd.y < sd.z {
                    cl.y += step.y;
                    sd.y += delta_dist.y * cell_size;
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    cl.z += step.z;
                    sd.z += delta_dist.z * cell_size;
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                s_cell_tg[depth_base] = cl;
                s_side_dist_tg[depth_base] = sd;
                continue;
            }
            let child_bt = child_block_type(packed);
            if child_bt == 255u {
                if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
                var cl = cell;
                var sd = side_dist;
                if sd.x < sd.y && sd.x < sd.z {
                    cl.x += step.x;
                    sd.x += delta_dist.x * cell_size;
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if sd.y < sd.z {
                    cl.y += step.y;
                    sd.y += delta_dist.y * cell_size;
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    cl.z += step.z;
                    sd.z += delta_dist.z * cell_size;
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                s_cell_tg[depth_base] = cl;
                s_side_dist_tg[depth_base] = sd;
                continue;
            }

            let at_max = depth + 1u > depth_limit || depth + 1u >= MAX_STACK_DEPTH;
            let child_cell_size = cell_size / 3.0;
            let cell_world_size = child_cell_size;
            let min_side = min(side_dist.x, min(side_dist.y, side_dist.z));
            let ray_dist = max(min_side * ray_metric, 0.001);
            let lod_pixels = cell_world_size / ray_dist * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;

            if at_max || at_lod {
                if ENABLE_STATS { ray_steps_lod_terminal = ray_steps_lod_terminal + 1u; }
                let bt = child_block_type(packed);
                if bt == 255u {
                    var cl = cell;
                    var sd = side_dist;
                    if sd.x < sd.y && sd.x < sd.z {
                        cl.x += step.x;
                        sd.x += delta_dist.x * cell_size;
                        normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                    } else if sd.y < sd.z {
                        cl.y += step.y;
                        sd.y += delta_dist.y * cell_size;
                        normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                    } else {
                        cl.z += step.z;
                        sd.z += delta_dist.z * cell_size;
                        normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                    }
                    s_cell_tg[depth_base] = cl;
                    s_side_dist_tg[depth_base] = sd;
                } else {
                    let node_origin = s_node_origin_tg[depth_base];
                    let cell_min_l = node_origin + vec3<f32>(cell) * cell_size;
                    let cell_max_l = cell_min_l + vec3<f32>(cell_size);
                    let cell_box_l = ray_box(ray_origin, inv_dir, cell_min_l, cell_max_l);
                    result.hit = true;
                    result.t = max(cell_box_l.t_enter, 0.0);
                    result.color = palette.colors[bt].rgb;
                    result.normal = normal;
                    result.cell_min = cell_min_l;
                    result.cell_size = cell_size;
                    return result;
                }
            } else {
                if ENABLE_STATS { ray_steps_node_descend = ray_steps_node_descend + 1u; }
                let parent_origin = s_node_origin_tg[depth_base];
                let parent_cell_size = cell_size;
                let child_origin = parent_origin + vec3<f32>(cell) * parent_cell_size;

                let child_max = child_origin + vec3<f32>(parent_cell_size);
                let child_hit = ray_box(ray_origin, inv_dir, child_origin, child_max);
                let ct_start = max(child_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
                let child_entry = ray_origin + ray_dir * ct_start;
                let local_entry = (child_entry - child_origin) / child_cell_size;

                depth += 1u;
                let new_base = depth * TG_STRIDE + lid;
                s_node_idx_tg[new_base] = child_idx;
                s_node_origin_tg[new_base] = child_origin;
                s_cell_size_tg[new_base] = child_cell_size;
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
                s_side_dist_tg[new_base] = vec3<f32>(
                    select((child_origin.x + lc.x * child_cell_size - ray_origin.x) * inv_dir.x,
                           (child_origin.x + (lc.x + 1.0) * child_cell_size - ray_origin.x) * inv_dir.x, ray_dir.x >= 0.0),
                    select((child_origin.y + lc.y * child_cell_size - ray_origin.y) * inv_dir.y,
                           (child_origin.y + (lc.y + 1.0) * child_cell_size - ray_origin.y) * inv_dir.y, ray_dir.y >= 0.0),
                    select((child_origin.z + lc.z * child_cell_size - ray_origin.z) * inv_dir.z,
                           (child_origin.z + (lc.z + 1.0) * child_cell_size - ray_origin.z) * inv_dir.z, ray_dir.z >= 0.0),
                );
            }
        }
    }

    return result;
}
