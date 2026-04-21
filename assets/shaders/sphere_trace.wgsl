// Curved-space sphere-tracer. GPU mirror of src/world/sphere_trace.rs.
//
// When uniforms.root_kind == ROOT_KIND_REMAP_SPHERE, march() calls
// sremap_march() instead of march_cartesian(). The tree is a plain
// Cartesian tree whose [-1, 1]^3 content renders as the unit ball in
// world space via the Nowell cube→sphere map F.
//
// Algorithm per ray:
//   1. Intersect world-space ray with unit ball.
//   2. Sphere-trace in world space. At each step, invert F (Newton)
//      to get cube coord, point-query the tree for occupancy, and
//      advance by σ_min(J(c)) · safe_cube_distance.
//   3. On first empty→filled transition, refine and report hit.

#include "bindings.wgsl"
#include "tree.wgsl"
#include "sphere_remap.wgsl"

// Tree point-query result: occupancy + palette index + safe distance
// + the AABB of the containing cell (needed by the hit path so
// downstream bevel math can compute the per-pixel cube_local).
struct SremapQuery {
    filled: u32,         // 0 = empty, 1 = filled
    block_type: u32,     // palette index
    safe: f32,           // L∞ distance to nearest cell boundary in cube space
    cell_size: f32,      // cube-space size of the cell containing `c`
    cell_min: vec3<f32>, // cube-space min corner of that cell
}

fn sremap_tree_query(root_node_idx: u32, c: vec3<f32>, max_depth: u32) -> SremapQuery {
    var out: SremapQuery;
    var cell_min = vec3<f32>(-1.0);
    var cell_size = 2.0;
    var current_idx = root_node_idx;
    for (var depth: u32 = 0u; depth < 64u; depth = depth + 1u) {
        if (depth >= max_depth) {
            // LOD terminal: treat as filled (conservative).
            out.filled = 1u;
            out.block_type = 0u;
            out.safe = max(sremap_safe_cube_l_inf(c, cell_min, cell_size), 1e-5);
            out.cell_min = cell_min;
            out.cell_size = cell_size;
            return out;
        }
        let cs = cell_size / 3.0;
        let sx = clamp(i32(floor((c.x - cell_min.x) / cs)), 0, 2);
        let sy = clamp(i32(floor((c.y - cell_min.y) / cs)), 0, 2);
        let sz = clamp(i32(floor((c.z - cell_min.z) / cs)), 0, 2);
        let slot = slot_from_xyz(sx, sy, sz);
        let packed = child_packed(current_idx, slot);
        let tag = child_tag(packed);
        cell_min = cell_min + vec3<f32>(f32(sx), f32(sy), f32(sz)) * cs;
        cell_size = cs;
        if (tag == 0u) {
            out.filled = 0u;
            out.block_type = 0u;
            out.safe = max(sremap_safe_cube_l_inf(c, cell_min, cell_size), 1e-5);
            out.cell_min = cell_min;
            out.cell_size = cell_size;
            return out;
        }
        if (tag == 1u) {
            out.filled = 1u;
            out.block_type = child_block_type(packed);
            out.safe = max(sremap_safe_cube_l_inf(c, cell_min, cell_size), 1e-5);
            out.cell_min = cell_min;
            out.cell_size = cell_size;
            return out;
        }
        // tag == 2u: descend
        current_idx = child_node_index(current_idx, slot);
    }
    // Step budget exhausted — treat as empty so the march keeps going
    // rather than painting an artifact.
    out.filled = 0u;
    out.block_type = 0u;
    out.safe = 1e-3;
    out.cell_min = cell_min;
    out.cell_size = cell_size;
    return out;
}

// The sphere body sits at the center of the frame's [0, 3)^3 cube.
// Radius 0.6 gives enough room at the edges to place the camera
// outside the body without bumping the frame boundary. When the
// architecture is ready for sphere bodies as sub-cells, these will
// come from the uniform buffer.
const SREMAP_BALL_CENTER: vec3<f32> = vec3<f32>(1.5, 1.5, 1.5);
const SREMAP_BALL_RADIUS: f32 = 0.6;

// The main sphere trace. The sphere body is a ball of radius
// SREMAP_BALL_RADIUS centered at SREMAP_BALL_CENTER in the input
// frame. Walks a frame-space straight ray through it; backing tree
// lives in cube coords [-1, 1]^3 mapped through F. Returns a
// HitResult in frame-space units, compatible with the rest of the
// renderer pipeline.
fn sremap_march(root_node_idx: u32, frame_ray_origin: vec3<f32>, frame_ray_dir: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.color = vec3<f32>(0.5);
    result.normal = vec3<f32>(0.0, 1.0, 0.0);
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;
    result.frame_level = 0u;
    result.frame_scale = 1.0;

    // Transform frame-space ray into unit-ball-local space (ball
    // becomes radius 1 at origin). Frame-space t corresponds to
    // frame_ray_dir magnitude units; we convert everything to a
    // unit-direction inside the local space so arc lengths line up.
    let origin_local = (frame_ray_origin - SREMAP_BALL_CENTER) / SREMAP_BALL_RADIUS;
    let dmag_frame = length(frame_ray_dir);
    if (dmag_frame < 1e-6) { return result; }
    let dir_frame_unit = frame_ray_dir / dmag_frame;
    // In local space, 1 unit of arc = 1 frame unit / radius. The
    // direction vector's magnitude there is `1/radius` per 1 frame
    // arc; a unit-length local direction `d` advances by `radius`
    // frame units per unit of local arc length `s`. We march in `s`
    // and convert to frame t = s / (1/radius) · (1/dmag_frame)
    //                        = s * radius / dmag_frame.
    let d = dir_frame_unit; // direction is scale-invariant (unit)

    let rb = sremap_ray_unit_ball(origin_local, d);
    if (rb.z < 0.5) {
        return result; // miss
    }
    let t_enter = rb.x;
    let t_exit = rb.y;
    if (t_exit < 0.0) {
        return result;
    }

    let entry_offset = 1e-3;
    let min_local_step = 1e-4;
    let sigma_floor = 0.02;
    let newton_iters = 4u;
    let max_steps = 256u;
    let max_depth = uniforms.max_depth;

    var s = max(t_enter, 0.0) + entry_offset;
    let w_entry = origin_local + d * s;
    var c_warm = w_entry;

    var prev_s = s;
    var prev_filled = false;

    for (var step: u32 = 0u; step < max_steps; step = step + 1u) {
        if (s > t_exit + min_local_step) {
            return result; // exited ball
        }
        let w_local = origin_local + d * s;
        let c = sremap_inverse(w_local, c_warm, newton_iters);
        c_warm = c;

        let q = sremap_tree_query(root_node_idx, c, max_depth);
        let filled = q.filled == 1u;

        if (filled && !prev_filled) {
            // Crossed empty→filled: refine via linear interp on s.
            let s_hit = select(0.5 * (prev_s + s), s, step == 0u);
            let w_hit_local = origin_local + d * s_hit;
            let w_hit_frame = w_hit_local * SREMAP_BALL_RADIUS + SREMAP_BALL_CENTER;
            // Re-invert and re-query at the refined hit position so
            // the cell AABB below matches the pixel we shade.
            let c_hit = sremap_inverse(w_hit_local, c_warm, newton_iters);
            let qh = sremap_tree_query(root_node_idx, c_hit, max_depth);
            // Cube-space local inside the hit cell, in [0, 1]^3.
            let cube_local = clamp(
                (c_hit - qh.cell_min) / qh.cell_size,
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(1.0, 1.0, 1.0),
            );

            result.hit = true;
            result.t = s_hit * SREMAP_BALL_RADIUS / dmag_frame;
            // Radial normal is correct for outer-shell hits. Interior
            // voxel walls would want the cube-face normal warped by
            // J⁻ᵀ — defer until someone's actually carving interior
            // surfaces; right now this tracer only returns the first
            // empty→filled transition from outside.
            let nlen = max(length(w_hit_local), 1e-6);
            result.normal = w_hit_local / nlen;
            // Pack `cube_local` through the HitResult's cell_min /
            // cell_size fields: shade_pixel does
            //   local = (hit_pos - cell_min) / cell_size
            // With cell_size = 1 and cell_min = hit_pos - cube_local,
            // that evaluates back to cube_local at this pixel's hit.
            result.cell_min = w_hit_frame - cube_local;
            result.cell_size = 1.0;
            result.color = palette[qh.block_type].rgb;
            return result;
        }

        let sigma_raw = sremap_sigma_min(c);
        let sigma = max(sigma_raw, sigma_floor);
        let step_size = max(q.safe * sigma, min_local_step);
        prev_s = s;
        prev_filled = filled;
        s = s + step_size;
    }

    return result;
}
