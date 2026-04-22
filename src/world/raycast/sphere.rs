//! Legacy CPU sphere raycast was retired here. The body-march, the
//! face-window plumbing, and the face-subtree walker all moved to
//! `crate::world::raycast::unified::unified_raycast`, which uses a
//! residual + slot-path + per-cell Jacobian primitive that's
//! precision-bounded at every face-subtree depth (the legacy DDA
//! collapsed at depth ~20 due to f32 cell-wall plane normals
//! fusing).
//!
//! The only thing left in this module is `LodParams` — kept as the
//! public knob the dispatcher and `App::frame_aware_raycast` still
//! pass through, even though the unified primitive's bounded
//! descent doesn't need a screen-LOD cap.

/// Per-ray screen-LOD parameters. Originally consumed by the legacy
/// face-subtree walker to terminate at the screen-Nyquist threshold;
/// the unified primitive stops naturally on cell boundaries instead,
/// so these fields are retained only for API compatibility with the
/// `cpu_raycast_in_frame` caller chain.
#[derive(Copy, Clone, Debug)]
pub struct LodParams {
    pub pixel_density: f32,
    pub lod_threshold: f32,
}

impl LodParams {
    /// Equivalent of "always go to max depth" — used when the caller
    /// wants edit-target depth rather than screen-LOD depth.
    pub fn fixed_max() -> Self {
        Self { pixel_density: 1e30, lod_threshold: 1.0 }
    }
}
