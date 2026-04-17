// Outer `march()` loop: dispatches the current frame's DDA on its
// NodeKind (Cartesian or sphere body), then on miss pops to the next
// ancestor in the ribbon and continues. When the ribbon is exhausted,
// returns sky (hit=false).
//
// Each pop transforms the ray into the parent's frame coords:
// `parent_pos = slot_xyz + frame_pos / 3`, `parent_dir = frame_dir / 3`.
// The parent's frame cell still spans `[0, 3)³` in its own coords, so
// the inner DDA is unchanged — only the ray is rescaled and the buffer
// node_idx swapped.
//
// `lid` is the compute local-invocation index, forwarded to
// `march_cartesian` for its workgroup-memory stack addressing.

#include "bindings.wgsl"
#include "ray_prim.wgsl"
#include "sphere.wgsl"

fn march(world_ray_origin: vec3<f32>, world_ray_dir: vec3<f32>, lid: u32) -> HitResult {
    var ray_origin = world_ray_origin;
    var ray_dir = world_ray_dir;
    var current_idx = uniforms.root_index;
    var current_kind = uniforms.root_kind;
    var inner_r = uniforms.root_radii.x;
    var outer_r = uniforms.root_radii.y;
    var cur_face_bounds = uniforms.root_face_bounds;
    var ribbon_level: u32 = 0u;
    var cur_scale: f32 = 1.0;

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
            let detail_budget = select(
                1u,
                BASE_DETAIL_DEPTH - ribbon_level,
                ribbon_level < BASE_DETAIL_DEPTH,
            );
            let cart_depth_limit = min(detail_budget, MAX_STACK_DEPTH);
            r = march_cartesian(current_idx, ray_origin, ray_dir, cart_depth_limit, skip_slot, lid);
        }
        if r.hit {
            r.frame_level = ribbon_level;
            r.frame_scale = cur_scale;
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

        if ribbon_level >= uniforms.ribbon_count {
            break;
        }
        if current_kind == ROOT_KIND_FACE {
            let body_pop_level = uniforms.root_face_meta.y;
            if ribbon_level < body_pop_level {
                let entry = ribbon[ribbon_level];
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
            ribbon_level = body_pop_level + 1u;
        } else {
            if ribbon_level < uniforms.ribbon_count {
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

                let k = node_kinds[current_idx].kind;
                if k == 1u {
                    current_kind = ROOT_KIND_BODY;
                    inner_r = node_kinds[current_idx].inner_r;
                    outer_r = node_kinds[current_idx].outer_r;
                } else {
                    current_kind = ROOT_KIND_CARTESIAN;
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
