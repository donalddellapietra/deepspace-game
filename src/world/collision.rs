//! Swept-AABB collision with radial gravity for planet-layer play.
//!
//! The player is an axis-aligned box whose local frame is oriented
//! so that `local_up` points away from the nearest planet's center.
//! Movement intent (WASD) is interpreted in that local frame, then
//! converted to a world-space delta and applied axis-by-axis against
//! the tree's solid cells (which remain world-axis-aligned).
//!
//! Behavior falls back to legacy world-Y gravity when the player is
//! outside every planet's influence sphere — enabling flight through
//! space between planets.

use super::edit;
use super::sdf::{self, Vec3};
use super::state::WorldState;
use super::tree::NodeLibrary;

/// Player AABB half-width on tangent axes, in collision-grid cells.
pub const PLAYER_HW: f32 = 0.3;
/// Player AABB height along local_up, in collision-grid cells.
pub const PLAYER_H: f32 = 1.7;

/// Jump impulse magnitude, world units / second. Scaled by
/// the dominant planet's surface gravity so the arc height is
/// consistent across planets of different mass.
const JUMP_IMPULSE_SCALE: f32 = 0.6;
/// Fall-speed cap (world units / second). Absolute world-space cap —
/// not cell-scaled, since gravity is in world units too.
const MAX_FALL_SPEED: f32 = 25.0;
/// Default Y gravity when outside any planet's influence. We set
/// this to zero so space is free-flight (space itself should never
/// pull you anywhere — if you want gravity, be on a planet).
const FALLBACK_GRAVITY: f32 = 0.0;

/// Collision state attached to the player. Velocity is 3D so radial
/// gravity can pull along any axis.
pub struct PlayerPhysics {
    pub velocity: Vec3,
    pub on_ground: bool,
}

impl Default for PlayerPhysics {
    fn default() -> Self {
        Self { velocity: [0.0, 0.0, 0.0], on_ground: false }
    }
}

impl PlayerPhysics {
    /// Apply a jump impulse along `local_up`. `gravity_mag` is the
    /// magnitude of the gravity vector at the player's current
    /// position; the impulse scales with sqrt(gravity) so jump
    /// height is roughly constant across planets.
    pub fn jump(&mut self, local_up: Vec3, gravity_mag: f32) {
        if self.on_ground {
            let impulse = JUMP_IMPULSE_SCALE * gravity_mag.max(1.0).sqrt();
            self.velocity = sdf::add(self.velocity, sdf::scale(local_up, impulse));
            self.on_ground = false;
        }
    }
}

/// Integrate physics for one frame and clip against the tree.
///
/// `pos` is the player's feet position in world space [0, 3).
/// `move_tangent` is [right, forward] intent in the local tangent
/// plane (each in [-1, 1], magnitude ≤ √2).
/// `cell_size` is the edge length of one collision cell in world
/// space (= 1 / 3^edit_depth).
pub fn move_and_collide(
    pos: &mut Vec3,
    physics: &mut PlayerPhysics,
    world: &WorldState,
    move_tangent: [f32; 2],
    walk_speed: f32,
    dt: f32,
    cell_size: f32,
    max_depth: u32,
) {
    let (local_up, gravity_vec, on_planet) = gravity_frame(world, *pos, cell_size);

    // 1. Apply gravity to velocity.
    physics.velocity = sdf::add(physics.velocity, sdf::scale(gravity_vec, dt));
    // Cap downward-along-local-up speed (absolute world units).
    let along_up = sdf::dot(physics.velocity, local_up);
    if along_up < -MAX_FALL_SPEED {
        let excess = along_up + MAX_FALL_SPEED;
        physics.velocity = sdf::sub(physics.velocity, sdf::scale(local_up, excess));
    }

    // 2. Build local tangent basis (right, forward) perpendicular to local_up.
    let (tangent_right, tangent_fwd) = tangent_basis(local_up);
    let speed = walk_speed;
    let move_world = sdf::add(
        sdf::scale(tangent_right, move_tangent[0] * speed),
        sdf::scale(tangent_fwd, move_tangent[1] * speed),
    );

    // 3. Per-axis sweep-clip. World-axis order (Y first so the
    //    player settles on solid floors; then X, then Z). Gravity
    //    and tangential movement are combined into a single world
    //    delta per frame.
    let delta = sdf::add(sdf::scale(physics.velocity, dt), sdf::scale(move_world, dt));
    let hw = PLAYER_HW * cell_size;
    let height = PLAYER_H * cell_size;

    // Sweep Y.
    //
    // We binary-search for the contact point rather than snapping to
    // a cell-aligned grid. At deep zooms the cell size is well below
    // f32 precision, so `floor(pos / cs) * cs` is a no-op and the
    // player hovers at the pre-step position instead of settling on
    // the surface. A 20-iteration bisection gives ~1e-6 precision in
    // world units, which is fine for any cell size we'd probe at.
    {
        let new_y = pos[1] + delta[1];
        if delta[1] < 0.0 {
            if check_solid_region(&world.library, world.root, max_depth,
                pos[0] - hw, new_y, pos[2] - hw,
                pos[0] + hw, new_y, pos[2] + hw)
            {
                let mut lo = new_y;   // known inside/contacting solid
                let mut hi = pos[1];  // assumed clear
                for _ in 0..20 {
                    let mid = (lo + hi) * 0.5;
                    if check_solid_region(&world.library, world.root, max_depth,
                        pos[0] - hw, mid, pos[2] - hw,
                        pos[0] + hw, mid, pos[2] + hw)
                    {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                pos[1] = hi;
                // Ground contact zeros accumulated velocity. Without
                // this, off-axis gravity (e.g. radial gravity when the
                // player walks away from a planet's pole) keeps
                // building X/Z velocity forever, pulling the player
                // "downhill" across frames even on flat walking.
                physics.velocity = [0.0, 0.0, 0.0];
            } else {
                pos[1] = new_y;
            }
        } else if delta[1] > 0.0 {
            let head_y = new_y + height;
            if check_solid_region(&world.library, world.root, max_depth,
                pos[0] - hw, head_y, pos[2] - hw,
                pos[0] + hw, head_y, pos[2] + hw)
            {
                physics.velocity[1] = 0.0;
            } else {
                pos[1] = new_y;
            }
        }
    }

    // Sweep X.
    {
        let new_x = pos[0] + delta[0];
        let edge_x = if delta[0] > 0.0 { new_x + hw } else { new_x - hw };
        if !check_solid_region(&world.library, world.root, max_depth,
            edge_x, pos[1], pos[2] - hw,
            edge_x, pos[1] + height * 0.9, pos[2] + hw)
        {
            pos[0] = new_x;
        } else {
            physics.velocity[0] = 0.0;
        }
    }

    // Sweep Z.
    {
        let new_z = pos[2] + delta[2];
        let edge_z = if delta[2] > 0.0 { new_z + hw } else { new_z - hw };
        if !check_solid_region(&world.library, world.root, max_depth,
            pos[0] - hw, pos[1], edge_z,
            pos[0] + hw, pos[1] + height * 0.9, edge_z)
        {
            pos[2] = new_z;
        } else {
            physics.velocity[2] = 0.0;
        }
    }

    // 4. Ground probe: a short ray in -local_up from feet.
    //
    // Offset must be in *world* units, not cell-scaled. At deep
    // tree depths (cs ~ 1e-10) any cell-scaled offset collapses
    // below f32 precision — probe_y == pos[1] and the probe
    // always reads the air cell above, so on_ground stays false
    // and jumps never fire. A fixed world-unit offset survives
    // f32 truncation near pos magnitudes of order 1.
    const GROUND_PROBE_OFFSET: f32 = 1e-5;
    let probe_hw = hw.max(GROUND_PROBE_OFFSET);
    let probe = sdf::add(*pos, sdf::scale(local_up, -GROUND_PROBE_OFFSET));
    physics.on_ground = check_solid_region(&world.library, world.root, max_depth,
        probe[0] - probe_hw * 0.5, probe[1], probe[2] - probe_hw * 0.5,
        probe[0] + probe_hw * 0.5, probe[1], probe[2] + probe_hw * 0.5);

    // When not on a planet, damp tangential velocity so space drifting
    // doesn't accumulate infinitely. On-planet: no damping (friction is
    // handled implicitly by ground contact zeroing per-axis vel above).
    if !on_planet {
        let damp = (1.0 - 0.5 * dt).clamp(0.0, 1.0);
        physics.velocity = sdf::scale(physics.velocity, damp);
    }
}

/// Resolve the local gravity frame at `pos`. Returns
/// `(local_up, gravity_vec, on_planet)` where:
///   - `local_up`  : unit vector, direction "up" for the player.
///   - `gravity_vec`: acceleration (world units / s²).
///   - `on_planet` : true if any planet's influence covers `pos`.
pub fn gravity_frame(world: &WorldState, pos: Vec3, _cell_size: f32) -> (Vec3, Vec3, bool) {
    if let Some(p) = world.dominant_planet(pos) {
        return (p.up_at(pos), p.gravity_at(pos), true);
    }
    ([0.0, 1.0, 0.0], [0.0, -FALLBACK_GRAVITY, 0.0], false)
}

/// Orthonormal basis in the plane perpendicular to `up`. Picks a
/// stable "forward" by projecting world -Z onto that plane (falls
/// back to world +X if degenerate near the poles).
pub fn tangent_basis(up: Vec3) -> (Vec3, Vec3) {
    let ref_fwd = [0.0, 0.0, -1.0];
    let dot = sdf::dot(ref_fwd, up);
    let fwd_unnorm = sdf::sub(ref_fwd, sdf::scale(up, dot));
    let fwd = if sdf::length(fwd_unnorm) < 0.01 {
        // Near pole — use world +X instead.
        let alt = [1.0, 0.0, 0.0];
        let d2 = sdf::dot(alt, up);
        sdf::normalize(sdf::sub(alt, sdf::scale(up, d2)))
    } else {
        sdf::normalize(fwd_unnorm)
    };
    // right = fwd × up
    let right = [
        fwd[1] * up[2] - fwd[2] * up[1],
        fwd[2] * up[0] - fwd[0] * up[2],
        fwd[0] * up[1] - fwd[1] * up[0],
    ];
    (sdf::normalize(right), fwd)
}

/// Check whether any cell in a world-space region is solid. Probes
/// corners plus a midpoint to avoid tunneling through thin walls.
fn check_solid_region(
    library: &NodeLibrary,
    root: u64,
    max_depth: u32,
    min_x: f32, min_y: f32, min_z: f32,
    max_x: f32, max_y: f32, max_z: f32,
) -> bool {
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

fn sample_axis(min: f32, max: f32) -> Vec<f32> {
    if (max - min).abs() < 1e-6 { return vec![min]; }
    let mut pts = vec![min, max];
    if (max - min) > 0.01 {
        let mid = (min + max) * 0.5;
        if !pts.contains(&mid) { pts.push(mid); }
    }
    pts
}

// ───────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::worldgen;

    fn cell_size_for(depth: u32) -> f32 { 1.0 / 3.0f32.powi(depth as i32) }

    #[test]
    fn tangent_basis_is_orthonormal() {
        let up = sdf::normalize([0.3, 1.0, -0.4]);
        let (r, f) = tangent_basis(up);
        assert!((sdf::length(r) - 1.0).abs() < 1e-4);
        assert!((sdf::length(f) - 1.0).abs() < 1e-4);
        assert!(sdf::dot(r, up).abs() < 1e-4);
        assert!(sdf::dot(f, up).abs() < 1e-4);
        assert!(sdf::dot(r, f).abs() < 1e-4);
    }

    #[test]
    fn gravity_points_toward_planet_center() {
        let world = worldgen::generate_world();
        let p = &world.planets[0];
        // Stand a little above surface on +X.
        let pos = [p.center[0] + p.radius + 0.01, p.center[1], p.center[2]];
        let (up, g, on) = gravity_frame(&world, pos, 0.001);
        assert!(on, "should be on a planet");
        // up points away from center (in +X).
        assert!(up[0] > 0.9);
        // gravity pulls toward center (in -X).
        assert!(g[0] < 0.0);
    }

    #[test]
    fn gravity_zero_in_void() {
        let world = worldgen::generate_world();
        // Far from every planet we configured: no gravity (free flight).
        let (_, g, on) = gravity_frame(&world, [0.05, 0.05, 0.05], 0.001);
        assert!(!on);
        assert_eq!(g, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn player_falls_toward_planet() {
        let world = worldgen::generate_world();
        let depth = world.tree_depth();
        let cs = cell_size_for(depth);
        let p = &world.planets[0];
        // Start 0.05 above surface on +Y so gravity is purely -Y.
        let mut pos = [p.center[0], p.center[1] + p.radius + 0.05, p.center[2]];
        let start_y = pos[1];
        let mut phys = PlayerPhysics::default();
        for _ in 0..120 {
            move_and_collide(&mut pos, &mut phys, &world, [0.0, 0.0], 0.1, 1.0/60.0, cs, depth);
        }
        assert!(pos[1] < start_y, "player should have fallen: start {}, now {}", start_y, pos[1]);
    }

    #[test]
    fn player_can_jump_from_ground() {
        let world = worldgen::generate_world();
        let depth = world.tree_depth();
        let cs = cell_size_for(depth);
        let p = &world.planets[0];
        let mut pos = [p.center[0], p.center[1] + p.radius + p.noise_scale * 3.0, p.center[2]];
        let mut phys = PlayerPhysics::default();
        // Land.
        for _ in 0..60 {
            move_and_collide(&mut pos, &mut phys, &world, [0.0, 0.0], 0.1, 1.0/60.0, cs, depth);
        }
        assert!(phys.on_ground, "should be on ground after settling: pos {:?}", pos);
        let landed_y = pos[1];
        // Jump.
        let up = p.up_at(pos);
        let g = sdf::length(p.gravity_at(pos));
        phys.jump(up, g);
        assert!(!phys.on_ground, "on_ground should clear on jump");
        // Simulate a few frames and check the player actually rose.
        let mut peak = landed_y;
        for _ in 0..30 {
            move_and_collide(&mut pos, &mut phys, &world, [0.0, 0.0], 0.1, 1.0/60.0, cs, depth);
            if pos[1] > peak { peak = pos[1]; }
        }
        assert!(peak > landed_y + 0.001,
            "player should have jumped higher: landed {}, peak {}", landed_y, peak);
    }

    #[test]
    fn player_walks_in_all_four_cardinal_directions() {
        let world = worldgen::generate_world();
        let depth = world.tree_depth();
        let cs = cell_size_for(depth);
        let p = &world.planets[0];
        let spawn = [p.center[0], p.center[1] + p.radius + p.noise_scale * 3.0, p.center[2]];

        for (label, input, axis, sign) in [
            ("+X (D)", [1.0, 0.0], 0, 1.0f32),
            ("-X (A)", [-1.0, 0.0], 0, -1.0),
            ("+tF (W)", [0.0, 1.0], 2, -1.0), // t_fwd = -Z
            ("-tF (S)", [0.0, -1.0], 2, 1.0),
        ] {
            let mut pos = spawn;
            let mut phys = PlayerPhysics::default();
            for _ in 0..10 {
                move_and_collide(&mut pos, &mut phys, &world, [0.0, 0.0], 0.1, 1.0/60.0, cs, depth);
            }
            let start = pos[axis];
            for _ in 0..30 {
                move_and_collide(&mut pos, &mut phys, &world, input, 0.1, 1.0/60.0, cs, depth);
            }
            let delta = pos[axis] - start;
            assert!(delta * sign > 0.0005,
                "direction {} failed: axis {} delta {} (sign {})",
                label, axis, delta, sign);
        }
    }

    #[test]
    fn player_walks_on_planet_surface() {
        // Player stands on +Y side of planet and walks in +X (via
        // tangent input). After a few frames pos[0] should advance.
        let world = worldgen::generate_world();
        let depth = world.tree_depth();
        let cs = cell_size_for(depth);
        let p = &world.planets[0];
        let mut pos = [p.center[0], p.center[1] + p.radius + p.noise_scale * 3.0, p.center[2]];
        let start_x = pos[0];
        let mut phys = PlayerPhysics::default();
        // Let gravity land first (10 frames).
        for _ in 0..10 {
            move_and_collide(&mut pos, &mut phys, &world, [0.0, 0.0], 0.1, 1.0/60.0, cs, depth);
        }
        let landed_x = pos[0];
        // Then walk in +X (tangent right) for 60 frames.
        for _ in 0..60 {
            move_and_collide(&mut pos, &mut phys, &world, [1.0, 0.0], 0.1, 1.0/60.0, cs, depth);
        }
        assert!(pos[0] > landed_x + 0.001,
            "player should have walked in +X: landed {}, now {}", landed_x, pos[0]);
        assert!((start_x - landed_x).abs() < 0.001,
            "player shouldn't have moved in X before walking: {} → {}", start_x, landed_x);
    }

    #[test]
    fn player_falls_sideways_on_equator() {
        // Stand on the +X side of the planet; gravity pulls in -X.
        let world = worldgen::generate_world();
        let depth = world.tree_depth();
        let cs = cell_size_for(depth);
        let p = &world.planets[0];
        let mut pos = [p.center[0] + p.radius + 0.05, p.center[1], p.center[2]];
        let start_x = pos[0];
        let mut phys = PlayerPhysics::default();
        for _ in 0..120 {
            move_and_collide(&mut pos, &mut phys, &world, [0.0, 0.0], 0.1, 1.0/60.0, cs, depth);
        }
        assert!(pos[0] < start_x,
            "player should fall in -X toward planet center: start {}, now {}",
            start_x, pos[0]);
    }
}
