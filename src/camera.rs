//! Yaw/pitch camera with locally-oriented frame.
//!
//! Step 3 of the path-based refactor: the camera's location is now a
//! [`Position`] (path + sub-slot offset). [`Camera::world_pos`] and
//! [`Camera::set_world_pos`] are **temporary shims** that let callers
//! still expressed in absolute XYZ keep working. Steps 4–6 migrate
//! those callers; step 6 deletes the shims.

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
    /// Construct at an absolute XYZ location, anchored at `depth` in
    /// the tree. Temporary — step 4 callers build `Position` directly.
    pub fn at_world_pos(
        pos: [f32; 3],
        depth: u8,
        smoothed_up: [f32; 3],
        yaw: f32,
        pitch: f32,
    ) -> Self {
        Self {
            position: Position::from_world_pos(pos, depth),
            smoothed_up,
            yaw,
            pitch,
        }
    }

    /// Absolute XYZ coordinates in the root cell's `[0, 3)³` frame.
    /// Shim for XYZ-only call sites; step 6 removes callers.
    #[inline]
    pub fn world_pos(&self) -> [f32; 3] {
        self.position.world_pos()
    }

    /// Overwrite the camera's location from absolute XYZ, preserving
    /// the current anchoring depth. Shim; step 4 replaces all call
    /// sites with path-based updates.
    pub fn set_world_pos(&mut self, pos: [f32; 3]) {
        let depth = self.position.depth;
        self.position = Position::from_world_pos(pos, depth);
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

    pub fn gpu_camera(&self, fov: f32) -> GpuCamera {
        let (fwd, r, up) = self.basis();
        GpuCamera {
            pos: self.world_pos(),
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
