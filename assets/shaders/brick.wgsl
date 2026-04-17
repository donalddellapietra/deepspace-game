#include "bindings.wgsl"
#include "ray_prim.wgsl"

// Flat 27³ DDA inside a single brick.
//
// A brick is a dense voxel grid that replaces a 3-level Cartesian
// subtree (3³ × 3³ × 3³ = 19683 cells). Cell value 0 = empty, otherwise
// the value is a palette index that the caller renders directly.
//
// The brick lives in the tree at world bounds
// `[brick_world_min, brick_world_min + brick_world_size)³`. Each cell
// is `brick_world_size / 27` in world units, indexed by
// `(z * 729 + y * 27 + x)` flat-row-major.
//
// Storage is packed 4 cells per u32 (little-endian). Reading a cell:
//
//     let i = z * 729 + y * 27 + x;
//     let word = brick_data[brick_data_offset + (i >> 2u)];
//     let cell_value = (word >> ((i & 3u) * 8u)) & 0xFFu;
//
// Returns `hit=false` if the ray exits the brick without hitting a
// non-empty cell — caller (Cartesian DDA) then advances past the
// brick's parent cell as if it had been empty.

fn brick_cell_value(brick_data_offset: u32, x: i32, y: i32, z: i32) -> u32 {
    let i = u32(z) * 729u + u32(y) * 27u + u32(x);
    let word = brick_data[brick_data_offset + (i >> 2u)];
    return (word >> ((i & 3u) * 8u)) & 0xFFu;
}

fn march_brick(
    brick_data_offset: u32,
    brick_world_min: vec3<f32>,
    brick_world_size: f32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    parent_normal: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let cell_world_size = brick_world_size / BRICK_DIM_F;
    let inv_cell_size = BRICK_DIM_F / brick_world_size;

    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );

    let brick_world_max = brick_world_min + vec3<f32>(brick_world_size);
    let entry = ray_box(ray_origin, inv_dir, brick_world_min, brick_world_max);
    if entry.t_enter >= entry.t_exit || entry.t_exit < 0.0 {
        return result;
    }

    let t_start = max(entry.t_enter, 0.0) + 0.0001 * cell_world_size;
    let entry_pos = ray_origin + ray_dir * t_start;

    // Map entry point to cell coords in [0, BRICK_DIM)³. Local =
    // (entry_pos - brick_world_min) / cell_world_size, then floored
    // and clamped.
    let local_entry = (entry_pos - brick_world_min) * inv_cell_size;
    var cell = vec3<i32>(
        clamp(i32(floor(local_entry.x)), 0, i32(BRICK_DIM) - 1),
        clamp(i32(floor(local_entry.y)), 0, i32(BRICK_DIM) - 1),
        clamp(i32(floor(local_entry.z)), 0, i32(BRICK_DIM) - 1),
    );

    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );
    let delta_dist = abs(inv_dir) * cell_world_size;

    // side_dist[axis] = world-distance t from ray_origin to next
    // axis-aligned grid plane crossing at this cell. Same DDA recurrence
    // as the outer march, but in flat brick-cell space.
    let cell_f = vec3<f32>(cell);
    let cell_min_world = brick_world_min + cell_f * cell_world_size;
    let cell_max_world = cell_min_world + vec3<f32>(cell_world_size);
    var side_dist = vec3<f32>(
        select((cell_min_world.x - ray_origin.x) * inv_dir.x,
               (cell_max_world.x - ray_origin.x) * inv_dir.x, ray_dir.x >= 0.0),
        select((cell_min_world.y - ray_origin.y) * inv_dir.y,
               (cell_max_world.y - ray_origin.y) * inv_dir.y, ray_dir.y >= 0.0),
        select((cell_min_world.z - ray_origin.z) * inv_dir.z,
               (cell_max_world.z - ray_origin.z) * inv_dir.z, ray_dir.z >= 0.0),
    );

    var normal = parent_normal;

    // Bound: a worst-case axis-aligned ray crosses 3 * 27 = 81 cells.
    // 96 = small slack for off-axis rays. Anything past that is a bug.
    var iter: u32 = 0u;
    loop {
        if iter >= 96u { break; }
        iter = iter + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        if cell.x < 0 || cell.x >= i32(BRICK_DIM)
            || cell.y < 0 || cell.y >= i32(BRICK_DIM)
            || cell.z < 0 || cell.z >= i32(BRICK_DIM) {
            return result; // miss → caller advances parent DDA
        }

        let v = brick_cell_value(brick_data_offset, cell.x, cell.y, cell.z);
        if v != 0u {
            let cmin = brick_world_min + vec3<f32>(cell) * cell_world_size;
            let cmax = cmin + vec3<f32>(cell_world_size);
            let chit = ray_box(ray_origin, inv_dir, cmin, cmax);
            result.hit = true;
            result.t = max(chit.t_enter, 0.0);
            result.color = palette.colors[v].rgb;
            result.normal = normal;
            result.cell_min = cmin;
            result.cell_size = cell_world_size;
            return result;
        }

        let m = min_axis_mask(side_dist);
        cell += vec3<i32>(m) * step;
        side_dist += m * delta_dist;
        normal = -vec3<f32>(step) * m;
    }
    return result;
}
