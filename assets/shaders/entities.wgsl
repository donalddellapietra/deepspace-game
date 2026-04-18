// Entity ray-march pass — hash-grid DDA.
//
// Called from the fragment shader AFTER the world `march(...)` call.
// The caller composes the two hits with min(t); the world ray-march
// (ribbon pops, sphere/face dispatch, anchor coords) is entirely
// untouched.
//
// ## Spatial index
//
// The CPU builds a uniform `BIN_GRID_RES³` grid over the render
// frame's [0, WORLD_SIZE)³ each frame, registering each entity in
// every bin its bbox overlaps. Two storage buffers carry the grid:
//
//   entity_bin_offsets[BIN_GRID_RES³ + 1]
//     prefix sum: entities in bin i live at entries[i..i+1].
//   entity_bin_entries[total_insertions]
//     flat entity indices grouped by bin.
//
// ## Algorithm
//
// 1. Ray-box against the grid bounds. Skip if miss.
// 2. DDA through the grid one bin at a time.
// 3. For each bin, iterate its entity list, ray-box + subtree march
//    (the same `march_cartesian` the world walker uses).
// 4. After each bin, early-out when the current best hit is closer
//    than the next bin's entry t — no remaining bin can yield a
//    closer hit along this ray.
//
// Duplicate testing: entities spanning multiple bins appear in
// every bin their bbox touches. A ray may test the same entity
// twice; min(t) composition keeps it correct, just redundant.
//
// ## Shader/CPU contract
//
// `BIN_GRID_RES` here MUST match `BIN_GRID_RES` in
// `src/world/entity_bins.rs`. Hard-coded on both sides so the
// shader can use a compile-time constant for the DDA bounds.

const BIN_GRID_RES: u32 = 32u;
const BIN_GRID_RES_I: i32 = 32;
const BIN_GRID_RES_F: f32 = 32.0;
const BIN_SIZE: f32 = 3.0 / BIN_GRID_RES_F;

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
    // measured from `ray_origin` (NOT from the grid entry). Matches
    // the DDA invariant `min(side_dist) = t at exit of current bin`.
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

    // DDA ceiling: the ray can visit at most 3*res bins along the
    // full diagonal; extra slack guards against numerical edge
    // cases at bin boundaries.
    let max_iter: u32 = BIN_GRID_RES * 3u + 4u;

    for (var iter: u32 = 0u; iter < max_iter; iter = iter + 1u) {
        if cell.x < 0 || cell.x >= BIN_GRID_RES_I
            || cell.y < 0 || cell.y >= BIN_GRID_RES_I
            || cell.z < 0 || cell.z >= BIN_GRID_RES_I {
            break;
        }

        let bin_id = u32(cell.x)
            + u32(cell.y) * BIN_GRID_RES
            + u32(cell.z) * BIN_GRID_RES * BIN_GRID_RES;
        let entry_start = entity_bin_offsets[bin_id];
        let entry_end = entity_bin_offsets[bin_id + 1u];

        for (var i: u32 = entry_start; i < entry_end; i = i + 1u) {
            let e_idx = entity_bin_entries[i];
            let e = entities[e_idx];

            let bb = ray_box(ray_origin, inv_dir, e.bbox_min, e.bbox_max);
            if bb.t_enter >= bb.t_exit || bb.t_exit < 0.0 { continue; }
            if bb.t_enter >= best.t { continue; }

            // Ray into entity-local [0, 3)³. Uniform scale means
            // local_t == world_t — no conversion needed.
            let size = e.bbox_max - e.bbox_min;
            let scale3 = vec3<f32>(3.0) / size;
            let local_origin = (ray_origin - e.bbox_min) * scale3;
            let local_dir = ray_dir * scale3;

            let local_hit = march_cartesian(
                e.subtree_bfs, local_origin, local_dir,
                MAX_STACK_DEPTH, 0xFFFFFFFFu,
            );
            if !local_hit.hit { continue; }
            if local_hit.t >= best.t { continue; }

            let size_over_3 = size * (1.0 / 3.0);
            best = local_hit;
            best.cell_min = e.bbox_min + local_hit.cell_min * size_over_3;
            best.cell_size = local_hit.cell_size * size_over_3.x;
        }

        // Early termination: `min(side_dist)` is the t at which the
        // ray exits the current bin. Any closer hit must already
        // live in the current or an earlier bin, so we can stop.
        let min_side = min(side_dist.x, min(side_dist.y, side_dist.z));
        if best.hit && best.t <= min_side {
            break;
        }

        // DDA advance: step one unit on the axis with smallest
        // side_dist, bump that axis's side_dist by delta_dist.
        let mask = min_axis_mask(side_dist);
        cell = cell + vec3<i32>(mask) * step;
        side_dist = side_dist + mask * delta_dist;
    }

    return best;
}
