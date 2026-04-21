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

// Per-step tree point-query. Returns:
//   .x = is_filled (0.0 or 1.0)
//   .y = safe cube-space distance (L∞ to cell boundary)
fn sremap_tree_query(root_node_idx: u32, c: vec3<f32>, max_depth: u32) -> vec2<f32> {
    var cell_min = vec3<f32>(-1.0);
    var cell_size = 2.0;
    var current_idx = root_node_idx;
    for (var depth: u32 = 0u; depth < 64u; depth = depth + 1u) {
        if (depth >= max_depth) {
            // LOD terminal: treat as filled (conservative — renders
            // subtree as its representative block; refined later).
            return vec2<f32>(1.0, max(sremap_safe_cube_l_inf(c, cell_min, cell_size), 1e-5));
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
            return vec2<f32>(0.0, max(sremap_safe_cube_l_inf(c, cell_min, cell_size), 1e-5));
        }
        if (tag == 1u) {
            return vec2<f32>(1.0, max(sremap_safe_cube_l_inf(c, cell_min, cell_size), 1e-5));
        }
        // tag == 2u: descend
        current_idx = child_node_index(current_idx, slot);
    }
    return vec2<f32>(0.0, 1e-3);
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
        let filled = q.x > 0.5;
        let safe = q.y;

        if (filled && !prev_filled) {
            // Crossed empty→filled: refine via linear interp on s.
            let s_hit = select(0.5 * (prev_s + s), s, step == 0u);
            let w_hit_local = origin_local + d * s_hit;
            // Back to frame space for the normal and hit position.
            let w_hit_frame = w_hit_local * SREMAP_BALL_RADIUS + SREMAP_BALL_CENTER;
            // Convert local arc length to frame-space t:
            //   t_frame = s * radius / dmag_frame
            result.hit = true;
            result.t = s_hit * SREMAP_BALL_RADIUS / dmag_frame;
            let nlen = max(length(w_hit_local), 1e-6);
            result.normal = w_hit_local / nlen;
            result.cell_min = w_hit_frame - vec3<f32>(0.01);
            result.cell_size = 0.02;
            result.color = vec3<f32>(0.7);
            return result;
        }

        let sigma_raw = sremap_sigma_min(c);
        let sigma = max(sigma_raw, sigma_floor);
        let step_size = max(safe * sigma, min_local_step);
        prev_s = s;
        prev_filled = filled;
        s = s + step_size;
    }

    return result;
}
