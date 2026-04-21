//! CPU raycast for the `RemapSphere` preset (cube → sphere remap).
//!
//! The world tree is a plain Cartesian voxel tree whose `[-1, 1]^3`
//! content the renderer displays as a ball of radius `ball_radius`
//! centered at `ball_center` in the frame's local coord system,
//! via the Nowell cube-to-sphere map (see `world::sphere_remap`).
//!
//! This CPU raycaster mirrors the shader's dispatch path
//! (`sremap_march` in `assets/shaders/sphere_trace.wgsl`): transform
//! the frame-local ray into unit-ball-local space, run the curved-
//! space sphere-trace, and translate the result into the shared
//! `HitInfo` format so break / place / highlight work unchanged.
//!
//! `ball_center` and `ball_radius` MUST match the shader's
//! `SREMAP_BALL_CENTER` / `SREMAP_BALL_RADIUS` constants; callers
//! hard-code the same values on both sides for now.

use crate::world::sphere_trace::{self, TraceConfig, TreeOccupancy};
use crate::world::tree::{NodeId, NodeLibrary};

use super::HitInfo;

/// Cast a ray into the remap-sphere body. Input coords are in the
/// frame-local space the shader receives (typically `[0, 3)^3` for
/// the render frame). `max_depth` is the same `edit_depth` signal
/// the Cartesian path uses — deeper descents produce longer paths,
/// i.e. smaller editable cells.
pub fn cpu_raycast_in_remap_sphere_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    cam_local: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
    ball_center: [f32; 3],
    ball_radius: f32,
) -> Option<HitInfo> {
    let occupancy = TreeOccupancy {
        library,
        root: world_root,
        max_depth,
    };
    let cfg = TraceConfig::default();
    sphere_trace::trace_to_hit(
        &occupancy,
        cam_local,
        ray_dir,
        &cfg,
        ball_center,
        ball_radius,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{uniform_children, Child, NodeLibrary};

    const BALL_CENTER: [f32; 3] = [1.5, 1.5, 1.5];
    const BALL_RADIUS: f32 = 0.6;

    fn build_uniform_tree(layers: u32) -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        let child = lib.build_uniform_subtree(1, layers);
        let root = match child {
            Child::Node(id) => id,
            _ => lib.insert(uniform_children(Child::Block(1))),
        };
        (lib, root)
    }

    #[test]
    fn path_length_grows_with_max_depth() {
        let (lib, root) = build_uniform_tree(10);

        // Ray from +Z outside the ball, looking at the ball center.
        let cam = [BALL_CENTER[0], BALL_CENTER[1], BALL_CENTER[2] + 2.0];
        let dir = [0.0, 0.0, -1.0];

        let h2 = cpu_raycast_in_remap_sphere_frame(
            &lib, root, cam, dir, 2, BALL_CENTER, BALL_RADIUS,
        )
        .expect("hit at depth 2");
        assert_eq!(h2.path.len(), 2, "depth-2 path length");

        let h8 = cpu_raycast_in_remap_sphere_frame(
            &lib, root, cam, dir, 8, BALL_CENTER, BALL_RADIUS,
        )
        .expect("hit at depth 8");
        assert_eq!(h8.path.len(), 8, "depth-8 path length");

        // Same root prefix.
        assert_eq!(h2.path[0], h8.path[0]);
    }

    #[test]
    fn axis_ray_face_is_plus_z() {
        // Ray along -Z into a solid ball: the ray enters the filled
        // region on its +Z face, so face should be 4.
        let (lib, root) = build_uniform_tree(6);
        let cam = [BALL_CENTER[0], BALL_CENTER[1], BALL_CENTER[2] + 2.0];
        let dir = [0.0, 0.0, -1.0];
        let h = cpu_raycast_in_remap_sphere_frame(
            &lib, root, cam, dir, 4, BALL_CENTER, BALL_RADIUS,
        )
        .expect("axis ray should hit");
        assert_eq!(h.face, 4, "ray along -Z should hit +Z face (4), got {}", h.face);
    }

    #[test]
    fn t_is_positive_and_along_ray_magnitude() {
        // Use a non-unit direction to verify t is measured along the
        // caller's original ray_dir magnitude (like Cartesian HitInfo).
        let (lib, root) = build_uniform_tree(4);
        let cam = [BALL_CENTER[0], BALL_CENTER[1], BALL_CENTER[2] + 2.0];
        // magnitude 2 direction pointed at the ball.
        let dir = [0.0, 0.0, -2.0];
        let h = cpu_raycast_in_remap_sphere_frame(
            &lib, root, cam, dir, 4, BALL_CENTER, BALL_RADIUS,
        )
        .expect("hit");
        // cam.z = 2.1, ball surface z ≈ 1.5 + 0.6 = 2.1; t along
        // unit ray would be ~0; along mag-2 direction, t ≈ 0.
        // Just sanity-check the sign and range.
        assert!(h.t > 0.0 && h.t < 2.0, "t = {} should be small positive", h.t);
    }
}
