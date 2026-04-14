//! Yaw/pitch camera with locally-oriented frame.
//!
//! The camera's location is a [`Position`] (path + sub-slot offset).
//! Call sites that want XYZ coordinates in a specific ancestor's
//! frame go through
//! [`Position::pos_in_ancestor_frame`]; the camera itself exposes no
//! absolute-XYZ accessors — that's the whole point of the refactor.

use crate::world::gpu::GpuCamera;
use crate::world::position::Position;
use crate::world::sdf;

/// Yaw/pitch camera in a locally-oriented frame.
///
/// `smoothed_up` tracks the local "up" direction (away from the
/// dominant planet's center, or world +Y in empty space), smoothed
/// over time so the horizon doesn't jerk when gravity switches.
/// `yaw` rotates around `smoothed_up`; `pitch` tilts the look vector
/// out of the tangent plane toward `smoothed_up`.
pub struct Camera {
    pub position: Position,
    pub smoothed_up: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    /// Construct at an explicit `Position`. Use
    /// `Position::from_world_pos_in_tree` to build the position from
    /// an XYZ spawn point — that walks the real tree, dispatching
    /// on `NodeKind` so the path is semantically valid even where it
    /// crosses body / face subtrees.
    pub fn at_position(
        position: Position,
        smoothed_up: [f32; 3],
        yaw: f32,
        pitch: f32,
    ) -> Self {
        Self { position, smoothed_up, yaw, pitch }
    }

    /// Lerp `smoothed_up` toward `target_up` at rate `k` per dt.
    pub fn update_up(&mut self, target_up: [f32; 3], dt: f32) {
        let k = (dt * 4.0).min(1.0);
        let blended = [
            self.smoothed_up[0] + (target_up[0] - self.smoothed_up[0]) * k,
            self.smoothed_up[1] + (target_up[1] - self.smoothed_up[1]) * k,
            self.smoothed_up[2] + (target_up[2] - self.smoothed_up[2]) * k,
        ];
        self.smoothed_up = sdf::normalize(blended);
    }

    /// (forward, right, up) in world space, built from smoothed_up +
    /// yaw + pitch. Returned basis is orthonormal — `up` is
    /// perpendicular to `forward`.
    ///
    /// Yaw convention: positive yaw = turn LEFT (counterclockwise
    /// around up). This matches the original world-Y camera so
    /// mouse-look feels consistent on a planet.
    pub fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let ref_up = self.smoothed_up;
        let (t_right, t_fwd) = sdf::tangent_basis(ref_up);
        let (sy, cy) = self.yaw.sin_cos();
        let horiz_fwd = sdf::sub(sdf::scale(t_fwd, cy), sdf::scale(t_right, sy));
        let horiz_right = sdf::add(sdf::scale(t_right, cy), sdf::scale(t_fwd, sy));
        let (sp, cp) = self.pitch.sin_cos();
        let fwd = sdf::normalize(sdf::add(
            sdf::scale(horiz_fwd, cp),
            sdf::scale(ref_up, sp),
        ));
        let up = [
            horiz_right[1] * fwd[2] - horiz_right[2] * fwd[1],
            horiz_right[2] * fwd[0] - horiz_right[0] * fwd[2],
            horiz_right[0] * fwd[1] - horiz_right[1] * fwd[0],
        ];
        (fwd, horiz_right, sdf::normalize(up))
    }

    pub fn forward(&self) -> [f32; 3] { self.basis().0 }

    /// Build the GPU camera block. `pos_in_frame` is the camera's
    /// XYZ in whatever frame the shader is rendering in (usually the
    /// render-root's `[0, 3)³` cell). Caller computes it with
    /// `Position::pos_in_ancestor_frame_in_tree` so crossings
    /// through body/face subtrees reconstruct correct XYZ.
    pub fn gpu_camera(&self, fov: f32, pos_in_frame: [f32; 3]) -> GpuCamera {
        let (fwd, r, up) = self.basis();
        GpuCamera {
            pos: pos_in_frame,
            _pad0: 0.0,
            forward: fwd,
            _pad1: 0.0,
            right: r,
            _pad2: 0.0,
            up,
            fov,
        }
    }
}
