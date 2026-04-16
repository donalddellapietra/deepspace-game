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
    var s_side_dist: array<vec3<f32>, MAX_STACK_DEPTH>;
    var s_node_origin: array<vec3<f32>, MAX_STACK_DEPTH>;
    var s_cell_size: array<f32, MAX_STACK_DEPTH>;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    s_node_idx[0] = root_node_idx;
    s_node_origin[0] = vec3<f32>(0.0);
    s_cell_size[0] = 1.0;

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
    s_side_dist[0] = vec3<f32>(
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
            if ENABLE_STATS { ray_steps_oob = ray_steps_oob + 1u; }

            if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                s_cell[depth].x += step.x;
                s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if s_side_dist[depth].y < s_side_dist[depth].z {
                s_cell[depth].y += step.y;
                s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                s_cell[depth].z += step.z;
                s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
            continue;
        }

        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        let packed = child_packed(s_node_idx[depth], slot);
        let tag = child_tag(packed);

        if tag == 0u {
            // Empty — DDA advance.
            if ENABLE_STATS { ray_steps_empty = ray_steps_empty + 1u; }
            if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                s_cell[depth].x += step.x;
                s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if s_side_dist[depth].y < s_side_dist[depth].z {
                s_cell[depth].y += step.y;
                s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                s_cell[depth].z += step.z;
                s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
        } else if tag == 1u {
            let cell_min_h = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
            let cell_max_h = cell_min_h + vec3<f32>(s_cell_size[depth]);
            let cell_box_h = ray_box(ray_origin, inv_dir, cell_min_h, cell_max_h);
            result.hit = true;
            result.t = max(cell_box_h.t_enter, 0.0);
            result.color = palette.colors[child_block_type(packed)].rgb;
            result.normal = normal;
            result.cell_min = cell_min_h;
            result.cell_size = s_cell_size[depth];
            return result;
        } else {
            // tag == 2u: Node child. Look up its kind.
            let child_idx = child_node_index(s_node_idx[depth], slot);
            let kind = node_kinds[child_idx].kind;

            if kind == 1u {
                // CubedSphereBody: dispatch sphere DDA in this body's cell.
                let body_origin = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                let body_size = s_cell_size[depth];
                let inner_r = node_kinds[child_idx].inner_r;
                let outer_r = node_kinds[child_idx].outer_r;
                let sph = sphere_in_cell(
                    child_idx, body_origin, body_size,
                    inner_r, outer_r, ray_origin, ray_dir,
                );
                if sph.hit {
                    return sph;
                }
                // Sphere missed — advance Cartesian DDA past this cell.
                if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                    s_cell[depth].x += step.x;
                    s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if s_side_dist[depth].y < s_side_dist[depth].z {
                    s_cell[depth].y += step.y;
                    s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    s_cell[depth].z += step.z;
                    s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                continue;
            }
            if false {
                // Real path (re-enable after diagnostic confirms dispatch):
                let body_origin = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                let body_size = s_cell_size[depth];
                let inner_r = node_kinds[child_idx].inner_r;
                let outer_r = node_kinds[child_idx].outer_r;
                let sph = sphere_in_cell(
                    child_idx, body_origin, body_size,
                    inner_r, outer_r, ray_origin, ray_dir,
                );
                if sph.hit {
                    return sph;
                }
                // Sphere missed — advance Cartesian DDA past this cell.
                if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                    s_cell[depth].x += step.x;
                    s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if s_side_dist[depth].y < s_side_dist[depth].z {
                    s_cell[depth].y += step.y;
                    s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    s_cell[depth].z += step.z;
                    s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                continue;
            }

            // Shell skip: when re-entering a parent shell after a
            // ribbon pop, skip the SLOT we already traversed in the
            // inner shell. Uses slot index (not node_idx) so it works
            // correctly in deduplicated trees where siblings share the
            // same packed node.
            let cell_slot = u32(s_cell[depth].x) + u32(s_cell[depth].y) * 3u + u32(s_cell[depth].z) * 9u;
            if depth == 0u && cell_slot == skip_slot {
                if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                    s_cell[depth].x += step.x;
                    s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if s_side_dist[depth].y < s_side_dist[depth].z {
                    s_cell[depth].y += step.y;
                    s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    s_cell[depth].z += step.z;
                    s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                continue;
            }

            // Cartesian Node: depth/LOD check, then descend.
            // depth_limit = MAX_STACK_DEPTH — LOD controls the
            // effective depth, not an artificial per-shell budget.
            let at_max = depth + 1u > depth_limit || depth + 1u >= MAX_STACK_DEPTH;
            let child_cell_size = s_cell_size[depth] / 3.0;
            let cell_world_size = child_cell_size;
            let min_side = min(s_side_dist[depth].x, min(s_side_dist[depth].y, s_side_dist[depth].z));
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
                    if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                        s_cell[depth].x += step.x;
                        s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                        normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                    } else if s_side_dist[depth].y < s_side_dist[depth].z {
                        s_cell[depth].y += step.y;
                        s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                        normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                    } else {
                        s_cell[depth].z += step.z;
                        s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                        normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                    }
                } else {
                    let cell_min_l = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                    let cell_max_l = cell_min_l + vec3<f32>(s_cell_size[depth]);
                    let cell_box_l = ray_box(ray_origin, inv_dir, cell_min_l, cell_max_l);
                    result.hit = true;
                    result.t = max(cell_box_l.t_enter, 0.0);
                    result.color = palette.colors[bt].rgb;
                    result.normal = normal;
                    result.cell_min = cell_min_l;
                    result.cell_size = s_cell_size[depth];
                    return result;
                }
            } else {
                if ENABLE_STATS { ray_steps_node_descend = ray_steps_node_descend + 1u; }
                let parent_origin = s_node_origin[depth];
                let parent_cell_size = s_cell_size[depth];
                let child_origin = parent_origin + vec3<f32>(cell) * parent_cell_size;

                let child_max = child_origin + vec3<f32>(parent_cell_size);
                let child_hit = ray_box(ray_origin, inv_dir, child_origin, child_max);
                let ct_start = max(child_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
                let child_entry = ray_origin + ray_dir * ct_start;
                let local_entry = (child_entry - child_origin) / child_cell_size;

                depth += 1u;
                s_node_idx[depth] = child_idx;
                s_node_origin[depth] = child_origin;
                s_cell_size[depth] = child_cell_size;
                s_cell[depth] = vec3<i32>(
                    clamp(i32(floor(local_entry.x)), 0, 2),
                    clamp(i32(floor(local_entry.y)), 0, 2),
                    clamp(i32(floor(local_entry.z)), 0, 2),
                );
                let lc = vec3<f32>(s_cell[depth]);
                s_side_dist[depth] = vec3<f32>(
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
            // Ribbon-level LOD budget: the ancestor pop count is
            // the tree's native distance metric. Inside our anchor
            // cell (ribbon_level=0) we allow `BASE_DETAIL_DEPTH`
            // levels of descent; each additional shell (ribbon pop)
            // drops the budget by one, bottoming out at 1. This
            // gives cubic LOD shells that are invariant under zoom
            // — zooming out grows the ribbon by one outer shell at
            // budget=1 and leaves everything else unchanged.
            // Nyquist (LOD_PIXEL_THRESHOLD) still acts as an inner
            // floor so we don't descend into sub-pixel detail.
            let detail_budget = select(
                1u,
                BASE_DETAIL_DEPTH - ribbon_level,
                ribbon_level < BASE_DETAIL_DEPTH,
            );
            let cart_depth_limit = min(detail_budget, MAX_STACK_DEPTH);
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
