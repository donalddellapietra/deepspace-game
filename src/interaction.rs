//! Shared movement and physics for all entities.
//!
//! Every constant — walk speed, sprint speed, jump impulse — is in
//! **interaction-layer cells per second**, so behaviour is identical
//! at every zoom level.  The caller supplies `cell_size` (world-space
//! edge length of one cell at the current interaction depth); the
//! module converts once and hands the world-space delta to
//! [`collision::move_and_collide`] for gravity and AABB resolution.

use crate::world::collision::{self, PlayerPhysics};
use crate::world::tree::NodeLibrary;

// ── Constants (cells / sec) ─────────────────────────────────────

const WALK_SPEED: f32 = 5.0;
const SPRINT_SPEED: f32 = 10.0;
const JUMP_IMPULSE: f32 = 8.0;

// ── Input ───────────────────────────────────────────────────────

/// Per-frame movement intent, decoupled from key bindings.
///
/// Build one of these from whatever input source you have (keyboard,
/// AI controller, network) and pass it to [`tick`].
pub struct MoveInput {
    /// +1 forward, −1 backward.
    pub forward: f32,
    /// +1 right, −1 left.
    pub strafe: f32,
    /// `true` on the frame the entity wants to jump.
    pub jump: bool,
    /// `true` while sprint is held.
    pub sprint: bool,
}

// ── Tick ────────────────────────────────────────────────────────

/// Advance one frame of movement + physics for an entity.
///
/// * `pos`        — feet position in world-space `[0, 3)`.
/// * `physics`    — mutable velocity / on-ground state.
/// * `input`      — this frame's movement intent.
/// * `yaw`        — look direction (radians) for camera-relative movement.
/// * `dt`         — frame delta in seconds (clamped externally).
/// * `cell_size`  — world-space size of one interaction-layer cell.
/// * `library`    — node library for collision queries.
/// * `root`       — root node id.
/// * `max_depth`  — tree depth for collision probing.
pub fn tick(
    pos: &mut [f32; 3],
    physics: &mut PlayerPhysics,
    input: &MoveInput,
    yaw: f32,
    dt: f32,
    cell_size: f32,
    library: &NodeLibrary,
    root: u64,
    max_depth: u32,
) {
    // ── Jump ────────────────────────────────────────────────
    // velocity_y lives in world-units/sec (collision applies gravity
    // as `GRAVITY * cell_size * dt`).  Scale the impulse by cell_size
    // so jump height is always ~1.6 cells regardless of zoom.
    if input.jump && physics.on_ground {
        physics.velocity_y = JUMP_IMPULSE * cell_size;
        physics.on_ground = false;
    }

    // ── Horizontal delta ────────────────────────────────────
    let speed = if input.sprint { SPRINT_SPEED } else { WALK_SPEED };
    let (sy, cy) = yaw.sin_cos();
    let fwd_x = -sy;
    let fwd_z = -cy;
    let right_x = cy;
    let right_z = -sy;

    let raw_x = fwd_x * input.forward + right_x * input.strafe;
    let raw_z = fwd_z * input.forward + right_z * input.strafe;
    let len = (raw_x * raw_x + raw_z * raw_z).sqrt();

    let (dx, dz) = if len > 1e-4 {
        let s = speed * cell_size * dt / len;
        (raw_x * s, raw_z * s)
    } else {
        (0.0, 0.0)
    };

    // ── Collision + gravity ─────────────────────────────────
    collision::move_and_collide(
        pos, physics, [dx, dz], dt, cell_size, library, root, max_depth,
    );
}
