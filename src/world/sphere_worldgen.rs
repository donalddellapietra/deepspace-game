//! Voxelized-ball worldgen for `NodeKind::SphereBody` worlds.
//!
//! Produces a world whose root is a `SphereBody` cube filled with a
//! stone ball inscribed in the cube's body-frame `[-1, 1]³`. The
//! shader bends shading normals through the Bergamo cube→sphere
//! Jacobian so the ball renders as a smoothly-lit sphere, even though
//! the underlying data is a cube of voxels.
//!
//! The voxelizer recurses down to `layers` levels, dedupping at every
//! step. Cells fully inside the inscribed sphere become uniform-stone
//! subtrees; cells fully outside are empty; cells straddling the
//! sphere surface are descended into until they resolve or bottom out
//! at `layers`. At the leaf depth, a cell's center decides its fate.
//!
//! Depth trade-off: surface voxel count is `O(3^(2·layers))`. At
//! `layers=8` this is ≈13 k unique straddling cells — totally fine.
//! At `layers=40` it would be `10²⁴`; defer that to streaming
//! worldgen (out of MVP scope).

use super::anchor::{Path, WorldPos, WORLD_SIZE};
use super::palette::block;
use super::state::WorldState;
use super::tree::{
    empty_children, slot_index, Child, NodeId, NodeKind, NodeLibrary, BRANCH, CENTER_SLOT,
};

/// Default voxelization depth. Keeps the surface node count modest
/// (~10k straddling cells) so the initial pack is fast and the
/// shader's LOD-terminal path handles any deeper rendering via the
/// representative-block splat.
pub const DEFAULT_SPHERE_LAYERS: u8 = 8;

/// Build a world whose root is `NodeKind::SphereBody` filled with a
/// solid stone ball.
pub fn bootstrap_sphere_body_world(layers: u8) -> crate::world::bootstrap::WorldBootstrap {
    let world = build_sphere_body_world(layers);
    crate::world::bootstrap::WorldBootstrap {
        default_spawn_pos: default_spawn(layers),
        // Camera at body-frame (+0.8, −0.3, +0.8), looking toward the
        // body origin. With the engine's basis convention (yaw=0 →
        // fwd=(0,0,-1), positive yaw LEFT around +y, positive pitch
        // UP), looking toward origin from (+0.8, −0.3, +0.8) gives
        // direction = normalize(-0.8, +0.3, -0.8). horiz xz direction
        // = (-1/√2, -1/√2) ⇒ yaw = π/4; sin(pitch) = 0.3 / √1.37 ⇒
        // pitch ≈ asin(0.256).
        default_spawn_yaw: std::f32::consts::FRAC_PI_4,
        default_spawn_pitch: (0.3_f32 / (0.64_f32 + 0.09 + 0.64).sqrt()).asin(),
        plain_layers: layers,
        color_registry: crate::world::palette::ColorRegistry::new(),
        world,
    }
}

fn default_spawn(layers: u8) -> WorldPos {
    // World root IS the SphereBody spanning [0, WORLD_SIZE=3)³. The
    // planet is a body-frame radius-`BALL_RADIUS` ball centered at
    // body (0, 0, 0), which in shader coords is a sphere of radius
    // `BALL_RADIUS · 1.5` centered at (1.5, 1.5, 1.5). The camera
    // must stay INSIDE the cube (so a SphereBody ancestor exists
    // → shader sphere-flag on) but OUTSIDE the ball (otherwise we're
    // inside stone).
    //
    // Spawn well outside the planet but still in the cube, offset
    // from the sun axis so the terminator is visible: body-frame
    // (+0.8, −0.3, +0.8) is distance ≈ 1.165 from origin — well
    // outside the radius-0.5 ball, well inside the cube, and the
    // visible hemisphere catches both lit (+y) and shadowed (−y)
    // regions.
    let anchor_depth = layers.min(crate::world::tree::MAX_DEPTH as u8);
    WorldPos::from_frame_local(
        &Path::root(),
        [2.7, 1.05, 2.7],
        anchor_depth,
    )
}

fn build_sphere_body_world(layers: u8) -> WorldState {
    assert!(layers > 0, "sphere world must have at least one layer");
    let mut lib = NodeLibrary::default();

    // A reusable uniform-stone subtree of each depth we might need.
    let mut uniform_stone: Vec<NodeId> = Vec::with_capacity(layers as usize + 1);
    let mut cur = lib.insert(
        crate::world::tree::uniform_children(Child::Block(block::STONE)),
    );
    uniform_stone.push(cur);
    for _ in 1..layers {
        cur = lib.insert(crate::world::tree::uniform_children(Child::Node(cur)));
        uniform_stone.push(cur);
    }

    // Build the SphereBody cube — body-frame [-1, 1]³ voxelized as a
    // stone ball of radius 1 (the inscribed sphere). World root IS
    // the SphereBody: the camera spawns in a cube-corner pocket
    // outside the inscribed ball but inside the cube, so every
    // visible hit has a SphereBody ancestor and the shader sphere-
    // flag stays active.
    let root_children = build_ball_children(
        &mut lib,
        &uniform_stone,
        layers,
        [0.0, 0.0, 0.0],
        1.0,
    );
    let root = lib.insert_with_kind(root_children, NodeKind::SphereBody);
    lib.ref_inc(root);

    let world = WorldState { root, library: lib };
    eprintln!(
        "Sphere body world: layers={}, library_entries={}, depth={}",
        layers,
        world.library.len(),
        world.tree_depth(),
    );
    world
}

/// Body-frame ball radius. Smaller than 1 so the SphereBody cube has
/// empty space around the planet — the camera can stand in that
/// space (still under a SphereBody ancestor, so the shader sphere-
/// flag stays on) and see the whole ball silhouette from a distance.
const BALL_RADIUS: f32 = 0.5;

fn build_ball_children(
    lib: &mut NodeLibrary,
    uniform_stone: &[NodeId],
    depth_remaining: u8,
    center: [f32; 3],
    half: f32,
) -> crate::world::tree::Children {
    let mut children = empty_children();
    let child_half = half / BRANCH as f32;
    for sz in 0..BRANCH {
        for sy in 0..BRANCH {
            for sx in 0..BRANCH {
                let slot = slot_index(sx, sy, sz);
                let cc = [
                    center[0] + (sx as f32 - 1.0) * 2.0 * child_half,
                    center[1] + (sy as f32 - 1.0) * 2.0 * child_half,
                    center[2] + (sz as f32 - 1.0) * 2.0 * child_half,
                ];
                children[slot] = build_ball_child(
                    lib,
                    uniform_stone,
                    depth_remaining - 1,
                    cc,
                    child_half,
                );
            }
        }
    }
    children
}

fn build_ball_child(
    lib: &mut NodeLibrary,
    uniform_stone: &[NodeId],
    depth_remaining: u8,
    center: [f32; 3],
    half: f32,
) -> Child {
    let dist = (center[0] * center[0] + center[1] * center[1] + center[2] * center[2]).sqrt();
    // A cube cell's furthest corner from its center is `half * √3`.
    // If `dist + half√3 ≤ R`, the entire cell is inside the ball;
    // if `dist − half√3 ≥ R`, entirely outside.
    let corner = half * 3.0f32.sqrt();
    if dist + corner <= BALL_RADIUS {
        // Fully inside. Emit a uniform-stone subtree of depth equal
        // to `depth_remaining`. depth_remaining==0 means the cell is
        // a leaf Block; else an existing uniform-stone node.
        return if depth_remaining == 0 {
            Child::Block(block::STONE)
        } else {
            Child::Node(uniform_stone[depth_remaining as usize - 1])
        };
    }
    if dist >= BALL_RADIUS + corner {
        return Child::Empty;
    }
    if depth_remaining == 0 {
        // Leaf: decide by the cell's center. Fine for MVP; a surface-
        // aware leaf would Bresenham-subsample but the Bergamo normal
        // remap makes leaf-level decisions invisible past ~depth 6.
        return if dist < BALL_RADIUS {
            Child::Block(block::STONE)
        } else {
            Child::Empty
        };
    }
    // Mixed cell: recurse.
    let children = build_ball_children(
        lib,
        uniform_stone,
        depth_remaining,
        center,
        half,
    );
    Child::Node(lib.insert(children))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_root_is_sphere_body() {
        let bs = bootstrap_sphere_body_world(1);
        let root = bs.world.library.get(bs.world.root).expect("root");
        assert_eq!(root.kind, NodeKind::SphereBody);
    }

    #[test]
    fn sphere_body_has_voxels_inside_inscribed_sphere() {
        let bs = bootstrap_sphere_body_world(2);
        let root = bs.world.library.get(bs.world.root).unwrap();
        let mut non_empty = 0;
        for c in root.children.iter() {
            if !matches!(c, Child::Empty) { non_empty += 1; }
        }
        // The ball must produce SOME filled content inside the cube.
        assert!(non_empty > 0, "sphere body produced zero filled cells");
    }

    #[test]
    fn default_layers_world_builds() {
        let bs = bootstrap_sphere_body_world(DEFAULT_SPHERE_LAYERS);
        let root = bs.world.library.get(bs.world.root).unwrap();
        assert_eq!(root.kind, NodeKind::SphereBody);
    }
}
