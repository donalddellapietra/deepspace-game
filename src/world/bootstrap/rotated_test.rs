//! Single-rotated-block test world.
//!
//! Total tree depth ≥ 30. Layout:
//!
//!   root → embedding chain (E levels of slot-13-only cartesian) →
//!   "scene" 3×3×3 cartesian → slot (1,1,1) is a NodeKind::TangentBlock
//!   wrapping a self-similar 3-color patterned subtree (K levels deep).
//!
//! E + 1 + 1 + K ≥ 30. Defaults: E = 15, K = 14 → tree depth 31.
//!
//! Why an embedding chain: camera anchor descent only proceeds while
//! the path lands on real `Child::Node`s. With the rotated block
//! directly under root, a camera spawned in any non-(1,1,1) slot
//! at depth 0 would land on `Empty` immediately and `compute_render_frame`
//! would stop at depth 0 — capping the walker at MAX_STACK_DEPTH = 8
//! absolute levels. The embedding chain places real Cartesian nodes
//! at every level so the anchor descends naturally with zoom and the
//! active frame reaches the rotated cell's neighborhood.
//!
//! Why ≥ 30 levels: at depth 18+, 3^-d falls past f32 absolute-coord
//! precision. Anything that renders correctly must be using frame-
//! local descent. A 12-deep test could be cheated with absolute coords.
//!
//! Self-similar interior: every level inside the rotated subtree
//! references the SAME `patterned_node` of one shallower depth, so
//! at any zoom the leaf frontier exposes the 3-color signature for
//! visual rotation verification. Library size is `O(depth)`.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, BRANCH, Child, NodeId, NodeKind, NodeLibrary, CENTER_SLOT,
};

/// Levels of slot-13-only cartesian above the scene node. Picks
/// the camera's anchor descent through neutral cells — see module
/// docs for why this is needed.
pub const ROTATED_TEST_EMBEDDING_DEPTH: u32 = 15;

/// Levels of patterned cartesian inside the TangentBlock. Total
/// tree depth = embedding + scene (1) + tangent_block (1) + interior
/// = ROTATED_TEST_EMBEDDING_DEPTH + 2 + ROTATED_TEST_INTERIOR_DEPTH.
pub const ROTATED_TEST_INTERIOR_DEPTH: u32 = 14;

/// Build a self-similar 3-color patterned subtree of the given depth.
///
/// At depth 1 the 27 children are `Block` leaves colored by
/// `(x+y+z) % 3`. At depth K > 1 all 27 children reference the
/// depth-(K-1) patterned subtree. Result: at any zoom level the
/// rotated cube exposes the 3-color stripe signature one level
/// finer — every level has visible non-uniform content for rotation
/// verification.
fn patterned_node(library: &mut NodeLibrary, depth: u32) -> NodeId {
    debug_assert!(depth >= 1);
    if depth == 1 {
        let mut children = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let color = match (x + y + z) % 3 {
                        0 => block::BRICK,
                        1 => block::STONE,
                        _ => block::WOOD,
                    };
                    children[slot_index(x, y, z)] = Child::Block(color);
                }
            }
        }
        library.insert(children)
    } else {
        let sub = patterned_node(library, depth - 1);
        let mut children = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    children[slot_index(x, y, z)] = Child::Node(sub);
                }
            }
        }
        library.insert(children)
    }
}

pub(crate) fn bootstrap_rotated_test_world() -> WorldBootstrap {
    let mut library = NodeLibrary::default();

    // Patterned interior (K levels deep), wrapped in a TangentBlock
    // so the renderer recognizes the rotation boundary.
    let interior_root = patterned_node(&mut library, ROTATED_TEST_INTERIOR_DEPTH);
    let mut tb_children = empty_children();
    for z in 0..BRANCH {
        for y in 0..BRANCH {
            for x in 0..BRANCH {
                tb_children[slot_index(x, y, z)] = Child::Node(interior_root);
            }
        }
    }
    let rotated_id = library.insert_with_kind(tb_children, NodeKind::TangentBlock);

    // "Scene" node: 3×3×3 cartesian. Center slot (1,1,1) is the
    // rotated block; flanking slots (0,1,1) and (2,1,1) are stone
    // cubes so visual tests have axis-aligned reference geometry.
    let mut scene_children = empty_children();
    scene_children[slot_index(0, 1, 1)] = Child::Block(block::STONE);
    scene_children[slot_index(1, 1, 1)] = Child::Node(rotated_id);
    scene_children[slot_index(2, 1, 1)] = Child::Block(block::STONE);
    let scene = library.insert(scene_children);

    // Wrap in `ROTATED_TEST_EMBEDDING_DEPTH` cartesian layers, each
    // placing the inner subtree at the centre slot (13). Outer slots
    // are empty so the scene sits in otherwise-empty space — but the
    // chain itself is real Cartesian nodes, so the camera anchor can
    // descend through it from any starting position.
    let mut current = Child::Node(scene);
    for _ in 0..ROTATED_TEST_EMBEDDING_DEPTH {
        let mut children = empty_children();
        children[CENTER_SLOT] = current;
        current = Child::Node(library.insert(children));
    }
    let root = match current {
        Child::Node(id) => id,
        _ => unreachable!("embedding wraps Child::Node, can't bottom out as Block/Empty"),
    };
    library.ref_inc(root);

    let world = WorldState { root, library };
    let tree_depth = world.tree_depth() as u8;

    // Spawn at the WORLD CENTER inside the embedding chain, then
    // deepen to a moderate anchor depth. The center sits in slot
    // (1,1,1) at every embedding level, so the anchor descends
    // through real Cartesian nodes naturally — the active frame
    // reaches the scene neighborhood and the walker has its full
    // stack budget for fine detail.
    //
    // Y offset 2.0 puts the camera above the scene's middle row
    // (y=1 of the scene's 3×3×3) so the rotated cube + flanking
    // stones are framed against air. Z offset 2.5 places the
    // camera just outside the scene cube on the +Z side, looking
    // back through it.
    let center = 1.5f32;
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [center, center + 0.5, center + 1.0],
        ROTATED_TEST_EMBEDDING_DEPTH as u8 + 2,
    )
    .deepened_to(ROTATED_TEST_EMBEDDING_DEPTH as u8 + 6);

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -0.35,
        plain_layers: tree_depth,
        color_registry: ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_depth_is_at_least_30() {
        let boot = bootstrap_rotated_test_world();
        let depth = boot.world.tree_depth();
        // 30 is the architectural correctness bar — at depth 18+,
        // f32 absolute coords lose precision, so anything that
        // renders correctly here must be using frame-local descent.
        assert!(
            depth >= 30,
            "tree depth must be >= 30 to defeat absolute-coord cheats, got {}",
            depth
        );
    }

    /// The embedding chain must be a real `Cartesian` chain that
    /// `compute_render_frame` can descend through from any starting
    /// point near the world center. Tests slot 13 at every embedding
    /// level resolves to the next embedding node (or scene at the
    /// bottom).
    #[test]
    fn embedding_chain_descends_through_center() {
        let boot = bootstrap_rotated_test_world();
        let mut node_id = boot.world.root;
        for level in 0..ROTATED_TEST_EMBEDDING_DEPTH {
            let node = boot.world.library.get(node_id).expect("embedding node");
            assert_eq!(node.kind, NodeKind::Cartesian);
            match node.children[CENTER_SLOT] {
                Child::Node(child_id) => node_id = child_id,
                other => panic!("embedding level {} center slot was {:?}", level, other),
            }
        }
        // After unwrapping the chain we should be at the scene node.
        let scene = boot.world.library.get(node_id).expect("scene");
        assert_eq!(scene.kind, NodeKind::Cartesian);
    }

    #[test]
    fn rotated_node_has_kind_tangent_block() {
        let boot = bootstrap_rotated_test_world();
        // Walk down the embedding chain to the scene.
        let mut node_id = boot.world.root;
        for _ in 0..ROTATED_TEST_EMBEDDING_DEPTH {
            node_id = match boot.world.library.get(node_id).unwrap().children[CENTER_SLOT] {
                Child::Node(id) => id,
                _ => panic!("embedding broken"),
            };
        }
        // The scene's center slot is the TangentBlock.
        let scene = boot.world.library.get(node_id).unwrap();
        let rotated_id = match scene.children[slot_index(1, 1, 1)] {
            Child::Node(id) => id,
            other => panic!("expected center slot to be a Node, got {:?}", other),
        };
        let rotated = boot.world.library.get(rotated_id).unwrap();
        assert_eq!(rotated.kind, NodeKind::TangentBlock);
    }

    /// Library size must remain O(depth) — content-addressed dedup
    /// must collapse identical patterned references and the embedding
    /// chain's slot-13-only nodes.
    #[test]
    fn library_is_compact() {
        let boot = bootstrap_rotated_test_world();
        let lib_count = boot.world.library.len();
        assert!(
            lib_count <= 96,
            "library should be O(depth), got {} entries",
            lib_count
        );
    }

    #[test]
    fn stone_controls_flank_rotated_cell() {
        let boot = bootstrap_rotated_test_world();
        let mut node_id = boot.world.root;
        for _ in 0..ROTATED_TEST_EMBEDDING_DEPTH {
            node_id = match boot.world.library.get(node_id).unwrap().children[CENTER_SLOT] {
                Child::Node(id) => id,
                _ => panic!("embedding broken"),
            };
        }
        let scene = boot.world.library.get(node_id).unwrap();
        for slot in [slot_index(0, 1, 1), slot_index(2, 1, 1)] {
            match scene.children[slot] {
                Child::Block(b) => assert_eq!(b, block::STONE),
                other => panic!("expected stone control at slot {}, got {:?}", slot, other),
            }
        }
    }
}
