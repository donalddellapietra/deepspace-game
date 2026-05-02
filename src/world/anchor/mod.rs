//! Path-anchored world coordinates.
//!
//! See `docs/experimental-architecture/anchor-refactor-decisions.md`.
//!
//! Every position is `(anchor: Path, offset: [f32; 3])` where the
//! offset is kept in `[0, 1)³` of the anchor cell's local frame.
//! f32 never accumulates across cells: as motion overflows a cell,
//! the anchor advances. Zoom changes the anchor's depth.
//!
//! Step 1 of the anchor refactor introduces these types alongside
//! the legacy `[f32; 3]` coordinates without changing runtime
//! behavior. Later steps migrate the camera, renderer, and editing
//! paths onto `WorldPos`.

mod path;
mod world_pos;

pub use path::{Path, Transition};
pub use world_pos::WorldPos;

/// Local frame convention: every node's children span
/// `[0, WORLD_SIZE)³` because there are 2 children per axis.
/// This is a frame-local coordinate constant, not an absolute
/// world-scale measurement. See `docs/no-absolute-coordinates.md`.
pub const WORLD_SIZE: f32 = 2.0;

#[cfg(test)]
mod tests;
