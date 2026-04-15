//! Yaw/pitch camera with locally-oriented frame.
//!
//! The camera's position is a `WorldPos` — a symbolic path-anchor
//! plus an offset held in `[0, 1)³` of that cell — NOT a raw XYZ.
//! Rendering, editing, and physics derive world-space coordinates
//! on demand via `world_pos_f32()`.

use crate::world::anchor::WorldPos;
use crate::world::gpu::GpuCamera;
use crate::world::sdf;

/// Yaw/pitch camera in a locally-oriented frame.
///
/// `smoothed_up` tracks the local "up" direction (away from the
/// dominant planet's center, or world +Y in empty space), smoothed
/// over time so the horizon doesn't jerk when gravity switches.
/// `yaw` rotates around `smoothed_up`; `pitch` tilts the look vector
/// out of the tangent plane toward `smoothed_up`.
pub struct Camera {
    pub position: WorldPos,
    pub smoothed_up: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    /// World-space XYZ derived from the `WorldPos`. Used by legacy
    /// XYZ-native call sites (renderer, cubed-sphere raycast, LOD
    /// distance culling). Canonical storage stays in `position`.
    #[inline]
    pub fn world_pos_f32(&self) -> [f32; 3] {
        self.position.to_world_xyz()
    }

    /// F64 world position — precise at deep anchor where f32 loses
    /// sub-cell resolution. Used by CPU raycast / edit actions.
    #[inline]
    pub fn world_pos_f64(&self) -> [f64; 3] {
        self.position.to_world_xyz_f64()
    }

    /// World-space size of the anchor's cell. Flight speed,
    /// crosshair reach, and physics integration all scale with this
    /// so movement "feels" the same at every zoom level.
    #[inline]
    pub fn cell_size(&self) -> f32 {
        self.position.cell_size()
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
        self.gpu_camera_at(self.world_pos_f32(), fov)
    }

    /// Build a `GpuCamera` with an explicit position. Used by the
    /// render-frame pipeline: position is the camera's frame-local
    /// coordinate (f32-safe at any anchor depth), not world XYZ.
    pub fn gpu_camera_at(&self, pos: [f32; 3], fov: f32) -> GpuCamera {
        let (fwd, r, up) = self.basis();
        GpuCamera {
            pos,
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
