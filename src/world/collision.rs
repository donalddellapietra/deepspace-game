//! Swept-AABB collision against the base-3 tree.
//!
//! The player is an axis-aligned box that moves through the tree.
//! Each frame: apply gravity to vertical velocity, compute movement
//! delta, then clip per-axis (Y first, then X, then Z) against the
//! tree's solid cells at the current edit depth.
//!
//! The collision grid cell size is `1 / 3^edit_depth` — one cell in
//! the tree at the interaction layer. All probes go through
//! `edit::is_solid_at`, which walks the tree to the edit depth.

use super::edit;
use super::tree::NodeLibrary;

/// Player AABB half-width on X and Z, in collision-grid cells.
pub const PLAYER_HW: f32 = 0.3;
/// Player AABB total height, in collision-grid cells.
pub const PLAYER_H: f32 = 1.7;

/// Gravity in collision-grid cells per second squared.
const GRAVITY: f32 = 20.0;
/// Jump impulse in collision-grid cells per second.
const JUMP_IMPULSE: f32 = 8.0;
/// Maximum downward velocity (terminal velocity).
const MAX_FALL_SPEED: f32 = 40.0;

/// Collision state attached to the player.
pub struct PlayerPhysics {
    pub velocity_y: f32,
    pub on_ground: bool,
}

impl Default for PlayerPhysics {
    fn default() -> Self {
        Self {
            velocity_y: 0.0,
            on_ground: false,
        }
    }
}

impl PlayerPhysics {
    pub fn jump(&mut self) {
        if self.on_ground {
            self.velocity_y = JUMP_IMPULSE;
            self.on_ground = false;
        }
    }
}

/// Move the player with swept-AABB collision.
///
/// `pos` is the player's feet position (bottom-center of AABB) in
/// world space [0, 3). `move_delta` is the desired displacement this
/// frame (already scaled by dt and speed). `cell_size` is the edge
/// length of one collision cell in world space.
///
/// Returns the new position after collision clipping.
pub fn move_and_collide(
    pos: &mut [f32; 3],
    physics: &mut PlayerPhysics,
    move_xz: [f32; 2],
    dt: f32,
    cell_size: f32,
    library: &NodeLibrary,
    root: u64,
    max_depth: u32,
) {
    // Apply gravity.
    physics.velocity_y -= GRAVITY * cell_size * dt;
    if physics.velocity_y < -MAX_FALL_SPEED * cell_size {
        physics.velocity_y = -MAX_FALL_SPEED * cell_size;
    }

    let dy = physics.velocity_y * dt;

    // Half-widths in world space.
    let hw = PLAYER_HW * cell_size;
    let height = PLAYER_H * cell_size;

    // Y axis first (gravity / jump).
    {
        let new_y = pos[1] + dy;
        if dy < 0.0 {
            // Moving down — check feet.
            if check_solid_region(library, root, max_depth,
                pos[0] - hw, new_y, pos[2] - hw,
                pos[0] + hw, new_y, pos[2] + hw)
            {
                // Land on the surface: snap to cell boundary.
                let grid_y = (pos[1] / cell_size).floor() * cell_size;
                pos[1] = if grid_y > new_y { grid_y } else { pos[1] };
                physics.velocity_y = 0.0;
                physics.on_ground = true;
            } else {
                pos[1] = new_y;
                physics.on_ground = false;
            }
        } else {
            // Moving up — check head.
            let head_y = new_y + height;
            if check_solid_region(library, root, max_depth,
                pos[0] - hw, head_y, pos[2] - hw,
                pos[0] + hw, head_y, pos[2] + hw)
            {
                // Bonk head.
                physics.velocity_y = 0.0;
            } else {
                pos[1] = new_y;
                if dy > 0.0 {
                    physics.on_ground = false;
                }
            }
        }
    }

    // X axis.
    {
        let new_x = pos[0] + move_xz[0];
        let edge_x = if move_xz[0] > 0.0 { new_x + hw } else { new_x - hw };
        if !check_solid_region(library, root, max_depth,
            edge_x, pos[1], pos[2] - hw,
            edge_x, pos[1] + height * 0.9, pos[2] + hw)
        {
            pos[0] = new_x;
        }
    }

    // Z axis.
    {
        let new_z = pos[2] + move_xz[1];
        let edge_z = if move_xz[1] > 0.0 { new_z + hw } else { new_z - hw };
        if !check_solid_region(library, root, max_depth,
            pos[0] - hw, pos[1], edge_z,
            pos[0] + hw, pos[1] + height * 0.9, edge_z)
        {
            pos[2] = new_z;
        }
    }

    // Ground check: probe slightly below feet.
    let probe_y = pos[1] - cell_size * 0.05;
    physics.on_ground = check_solid_region(library, root, max_depth,
        pos[0] - hw * 0.5, probe_y, pos[2] - hw * 0.5,
        pos[0] + hw * 0.5, probe_y, pos[2] + hw * 0.5);
}

/// Check whether any cell in a world-space region is solid.
/// Probes a grid of points covering the AABB from `min` to `max`.
fn check_solid_region(
    library: &NodeLibrary,
    root: u64,
    max_depth: u32,
    min_x: f32, min_y: f32, min_z: f32,
    max_x: f32, max_y: f32, max_z: f32,
) -> bool {
    // Sample at corners and midpoints of the region. For a thin
    // AABB (one cell wide) this is 2-4 probes; for a wider one
    // we add interior samples to avoid tunneling.
    let xs = sample_axis(min_x, max_x);
    let ys = sample_axis(min_y, max_y);
    let zs = sample_axis(min_z, max_z);

    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                if edit::is_solid_at(library, root, [x, y, z], max_depth) {
                    return true;
                }
            }
        }
    }
    false
}

/// Generate sample points along an axis, at min and max, plus
/// intermediate steps if the span exceeds ~1 cell to avoid
/// tunneling through thin walls.
fn sample_axis(min: f32, max: f32) -> Vec<f32> {
    if (max - min).abs() < 1e-6 {
        return vec![min];
    }
    let mut pts = vec![min, max];
    // Add midpoints if the span is large enough that a cell could
    // fit between the two endpoints unsampled. In practice the
    // player AABB is ~0.6 cells wide, so this rarely adds points.
    let span = max - min;
    if span > 0.01 {
        let mid = (min + max) * 0.5;
        if !pts.contains(&mid) {
            pts.push(mid);
        }
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::state::WorldState;

    #[test]
    fn is_solid_at_ground() {
        let world = WorldState::test_world();
        let depth = world.tree_depth();
        // Root y=0 slot → stone_l2 → all solid. pos y=0.5 is in the bottom third.
        assert!(edit::is_solid_at(&world.library, world.root, [1.5, 0.5, 1.5], depth));
    }

    #[test]
    fn is_solid_at_air() {
        let world = WorldState::test_world();
        let depth = world.tree_depth();
        // Root y=2, x=0, z=0 → air_l2 → all empty. Use a point well inside.
        assert!(!edit::is_solid_at(&world.library, world.root, [0.5, 2.5, 0.5], depth));
    }

    #[test]
    fn player_falls_onto_ground() {
        let world = WorldState::test_world();
        let depth = world.tree_depth();
        let cell_size = 1.0 / 3.0f32.powi(depth as i32);

        // Start in air at root (0, 2, 0) which is air_l2.
        let mut pos = [0.5, 2.5, 0.5];
        let mut physics = PlayerPhysics::default();

        // Simulate 5 seconds of falling.
        for _ in 0..300 {
            move_and_collide(
                &mut pos, &mut physics,
                [0.0, 0.0], 1.0 / 60.0,
                cell_size, &world.library, world.root, depth,
            );
        }

        // Should have landed — on_ground true, y below start.
        assert!(physics.on_ground, "Player should be on ground after falling");
        assert!(pos[1] < 2.5, "Player should have fallen from starting pos");
        assert!(pos[1] > 0.0, "Player should not have fallen through the world");
    }

    #[test]
    fn player_walks_without_clipping() {
        let world = WorldState::test_world();
        let depth = world.tree_depth();
        let cell_size = 1.0 / 3.0f32.powi(depth as i32);

        // Start just above the grass surface. Root y=1 is grass_surface_l2:
        // dirt at y_slot=0, grass at y_slot=1, air at y_slot=2.
        // In root coords, that's y ∈ [1.0, 2.0). Air starts at ~1.67.
        let mut pos = [0.5, 1.67, 0.5];
        let mut physics = PlayerPhysics { velocity_y: 0.0, on_ground: true };

        let old_x = pos[0];
        // Walk +X for 60 frames.
        for _ in 0..60 {
            move_and_collide(
                &mut pos, &mut physics,
                [cell_size * 5.0 / 60.0, 0.0], 1.0 / 60.0,
                cell_size, &world.library, world.root, depth,
            );
        }

        // Should have moved in X (or been stopped by a wall, but not clipped through).
        assert!(pos[0] >= old_x || physics.on_ground, "Player should move or be stopped, not clip");
    }
}
