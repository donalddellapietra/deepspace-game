// Entity ray-march pass — hash-grid DDA with pixel-threshold LOD.
//
// Called from the fragment shader AFTER the world `march(...)` call.
// The caller composes the two hits with min(t); the world ray-march
// (ribbon pops, sphere/face dispatch, anchor coords) is entirely
// untouched.
//
// ## Spatial index
//
// A `BIN_GRID_RES³` hash grid over the frame's [0, WORLD_SIZE)³,
// built CPU-side each frame. See `src/world/entity_bins.rs`.
//
//   entity_bin_offsets[BIN_GRID_RES³ + 1]   prefix sums
//   entity_bin_entries[total_insertions]   entity indices, grouped
//
// ## LOD gates
//
// Per-entity pixel projection drives two optimizations:
//
// 1. **Sub-pixel skip** — when the entity's on-screen bbox
//    projects to less than `LOD_PIXEL_THRESHOLD` pixels, skip the
//    whole subtree and splat `representative_block` as a single
//    color. Avoids paying the ray-transform + subtree-entry
//    overhead on a descent that's going to LOD-terminate at the
//    same color anyway.
//
// 2. **Bounded descent** — for entities above the pixel threshold,
//    pass `march_cartesian` a `depth_limit` equal to
//    `floor(log3(entity_pixels / threshold))`. At `D` levels, a
//    cell occupies `size / 3^D` world units ≈ 1 pixel on screen;
//    further descent would be sub-pixel and the in-shader
//    pixel-LOD check would terminate immediately. Pre-computing
//    the ceiling shaves the final DDA iteration + LOD check per
//    entity.
//
// ## Contract
//
// `BIN_GRID_RES` MUST match `BIN_GRID_RES` in
// `src/world/entity_bins.rs`.

const BIN_GRID_RES: u32 = 32u;
const BIN_GRID_RES_I: i32 = 32;
const BIN_GRID_RES_F: f32 = 32.0;
const BIN_SIZE: f32 = 3.0 / BIN_GRID_RES_F;

// 1 / log2(3). `log3(x) = log2(x) * INV_LOG2_3`.
const INV_LOG2_3: f32 = 0.63092975357145735;

fn march_entities(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitResult {
    var best: HitResult;
    best.hit = false;
    best.t = 1e20;
    best.frame_level = 0u;
    best.frame_scale = 1.0;
    best.cell_min = vec3<f32>(0.0);
    best.cell_size = 1.0;

    if uniforms.entity_count == 0u { return best; }

    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );

    // Clip ray to grid bounds [0, 3)³. Skip entire pass on miss.
    let grid_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if grid_hit.t_enter >= grid_hit.t_exit || grid_hit.t_exit < 0.0 {
        return best;
    }

    let t_start = max(grid_hit.t_enter, 0.0) + 0.0001;
    let entry = ray_origin + ray_dir * t_start;

    // Pixel-projection helper: entity_size_world × focal_px / ray_dist
    // gives the bbox's on-screen pixel count (small-angle approx).
    let focal_px = uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));

    // Initial bin cell under the ray.
    var cell = vec3<i32>(
        clamp(i32(floor(entry.x / BIN_SIZE)), 0, BIN_GRID_RES_I - 1),
        clamp(i32(floor(entry.y / BIN_SIZE)), 0, BIN_GRID_RES_I - 1),
        clamp(i32(floor(entry.z / BIN_SIZE)), 0, BIN_GRID_RES_I - 1),
    );
    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );
    let delta_dist = abs(inv_dir) * BIN_SIZE;

    // Initial side_dist: t to the next axis crossing for each axis,
    // measured from `ray_origin`. DDA invariant:
    // `min(side_dist) = t at exit of current bin`.
    let cell_f = vec3<f32>(cell) * BIN_SIZE;
    var side_dist = vec3<f32>(
        select(
            (cell_f.x - ray_origin.x) * inv_dir.x,
            (cell_f.x + BIN_SIZE - ray_origin.x) * inv_dir.x,
            ray_dir.x >= 0.0,
        ),
        select(
            (cell_f.y - ray_origin.y) * inv_dir.y,
            (cell_f.y + BIN_SIZE - ray_origin.y) * inv_dir.y,
            ray_dir.y >= 0.0,
        ),
        select(
            (cell_f.z - ray_origin.z) * inv_dir.z,
            (cell_f.z + BIN_SIZE - ray_origin.z) * inv_dir.z,
            ray_dir.z >= 0.0,
        ),
    );

    let max_iter: u32 = BIN_GRID_RES * 3u + 4u;

    for (var iter: u32 = 0u; iter < max_iter; iter = iter + 1u) {
        if cell.x < 0 || cell.x >= BIN_GRID_RES_I
            || cell.y < 0 || cell.y >= BIN_GRID_RES_I
            || cell.z < 0 || cell.z >= BIN_GRID_RES_I {
            break;
        }

        if ENABLE_STATS { entity_bin_visits = entity_bin_visits + 1u; }

        let bin_id = u32(cell.x)
            + u32(cell.y) * BIN_GRID_RES
            + u32(cell.z) * BIN_GRID_RES * BIN_GRID_RES;
        let entry_start = entity_bin_offsets[bin_id];
        let entry_end = entity_bin_offsets[bin_id + 1u];

        for (var i: u32 = entry_start; i < entry_end; i = i + 1u) {
            let e_idx = entity_bin_entries[i];
            let e = entities[e_idx];

            if ENABLE_STATS { entity_aabb_tests = entity_aabb_tests + 1u; }
            let bb = ray_box(ray_origin, inv_dir, e.bbox_min, e.bbox_max);
            if bb.t_enter >= bb.t_exit || bb.t_exit < 0.0 { continue; }
            if bb.t_enter >= best.t { continue; }
            if ENABLE_STATS { entity_aabb_hits = entity_aabb_hits + 1u; }

            // On-screen projection of this entity at its closest
            // point along the ray. Entity is cubic (size.x == y == z).
            let size = e.bbox_max - e.bbox_min;
            let ray_dist = max(bb.t_enter, 0.001);
            let entity_pixels = size.x / ray_dist * focal_px;

            // Cheap win #2: sub-pixel entity — skip subtree and
            // splat the pre-computed representative color.
            if entity_pixels < LOD_PIXEL_THRESHOLD {
                if ENABLE_STATS { entity_subpixel_skips = entity_subpixel_skips + 1u; }
                let t_hit = max(bb.t_enter, 0.0);
                let rep_bt = e.representative_block;
                if rep_bt < 255u {
                    best.hit = true;
                    best.t = t_hit;
                    best.color = palette.colors[rep_bt].rgb;
                    best.normal = -normalize(ray_dir);
                    best.cell_min = e.bbox_min;
                    best.cell_size = size.x;
                }
                continue;
            }

            // Cheap win #1: per-entity depth budget.
            let log3_px = log2(max(entity_pixels / LOD_PIXEL_THRESHOLD, 1.0))
                * INV_LOG2_3;
            let depth_limit = u32(clamp(
                floor(log3_px) + 1.0,
                1.0,
                f32(MAX_STACK_DEPTH),
            ));

            let scale3 = vec3<f32>(3.0) / size;
            let local_origin = (ray_origin - e.bbox_min) * scale3;
            let local_dir = ray_dir * scale3;

            if ENABLE_STATS { entity_subtree_marches = entity_subtree_marches + 1u; }
            let local_hit = march_cartesian(
                e.subtree_bfs, local_origin, local_dir,
                depth_limit, 0xFFFFFFFFu,
            );
            if !local_hit.hit { continue; }
            if ENABLE_STATS { entity_subtree_hits = entity_subtree_hits + 1u; }
            if local_hit.t >= best.t { continue; }

            let size_over_3 = size * (1.0 / 3.0);
            best = local_hit;
            best.cell_min = e.bbox_min + local_hit.cell_min * size_over_3;
            best.cell_size = local_hit.cell_size * size_over_3.x;
        }

        let min_side = min(side_dist.x, min(side_dist.y, side_dist.z));
        if best.hit && best.t <= min_side {
            break;
        }

        let mask = min_axis_mask(side_dist);
        cell = cell + vec3<i32>(mask) * step;
        side_dist = side_dist + mask * delta_dist;
    }

    return best;
}
