//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! All edits feed off the GPU cursor probe — one shader-side
//! `march()` algorithm produces the hit, a thin CPU layer
//! (`probe_hit`) materialises `(NodeId, slot)` paths for the edit
//! module. No parallel CPU raycast.

mod break_place;
mod highlight;
mod probe_hit;
pub(crate) mod upload;
mod zoom;

pub(super) const MAX_LOCAL_VISUAL_DEPTH: u32 = 12;
pub(super) const MAX_FOCUSED_FRAME_CAMERA_EXTENT: f32 = 8.0;
pub(super) const FRAME_VISUAL_MIN_PIXELS: f32 = 1.0;
pub(super) const FRAME_FOCUS_MIN_PIXELS: f32 = 1.0;
