//! Sphere-ring primitive: a `NodeKind::UvRing` root with `N` cells
//! arranged around a circle in the ring node's local `[0, 3)³`.
//!
//! Cell positions on a circle CANNOT be expressed as fixed
//! `[0, 3)³` slot coordinates in a Cartesian root — there is no
//! grid-aligned mapping that places `N` cubes on a circle without
//! gaps or overlap. The `UvRing` topology adapter is therefore the
//! only place that knows about ring positioning; storage is a flat
//! `[N, 1, 1]` slab and the shader / CPU raycast apply the ring
//! placement at march time, in a single cell-local transform per
//! ray per cell.
//!
//! What this preset validates:
//!
//! 1. **Cell-local ring topology.** The ring is implicitly in the
//!    node's `[0, 3)³`; no `body_origin` / `body_size` parameters
//!    leak into the marcher. Mirrors how
//!    [`dodecahedron_test`] never references global coordinates —
//!    every transform is relative to *some* node's local frame.
//!
//! 2. **Storage / render layout split.** Cells are stored in a flat
//!    `[N, 1, 1]` slab (indexed via `sample_slab_cell`); they are
//!    rendered at ring positions. The displacement is applied
//!    exactly once per cell per ray, by the topology adapter —
//!    the same place on CPU (`cpu_raycast_uv_ring`) and GPU
//!    (`march_uv_ring`).
//!
//! 3. **TB-style descent through the slab.** Once the ring topology
//!    transforms the ray into a cell's `[0, 3)³`, the cell content
//!    is a standard Cartesian subtree — the existing tangent-cube
//!    DDA handles it without any UvRing-specific code.

use super::WorldBootstrap;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, uniform_children, Child, NodeKind, NodeLibrary,
};

/// Number of cells around the ring. Twenty-seven keeps the slab
/// storage at `[27, 1, 1]` with `slab_depth = 3`, cleanly tiling
/// `3³ = 27` storage slots — the same slab geometry as
/// [`wrapped_planet`]'s default longitude axis. Cells are visually
/// dense enough to read as a smooth ring at typical camera framings.
const RING_CELLS: u32 = 27;

/// Storage slab depth: `3` levels of Cartesian descent, since
/// `3^3 = 27 = RING_CELLS`.
const RING_SLAB_DEPTH: u8 = 3;

/// Depth of each per-cell content subtree. Matches
/// [`dodecahedron_test::CUBE_SUBTREE_DEPTH`] so the ring exercises
/// the same f32-precision regime as the dodecahedron preset under
/// the per-cell topology transform.
const CELL_SUBTREE_DEPTH: u8 = 30;

/// Build a uniform-block recursive Cartesian subtree of `depth`
/// levels. Every level dedups against the same library entry —
/// all `RING_CELLS` cells share one content subtree.
fn build_uniform_cartesian_subtree(
    library: &mut NodeLibrary,
    block_id: u16,
    depth: u8,
) -> Child {
    if depth == 0 {
        return Child::Block(block_id);
    }
    let inner = build_uniform_cartesian_subtree(library, block_id, depth - 1);
    Child::Node(library.insert(uniform_children(inner)))
}

/// Build the `[N, 1, 1]` slab tree by recursive ternary subdivision
/// along the X axis. The resulting node is a Cartesian-shaped tree
/// of depth `slab_depth`; the outermost node carries the `UvRing`
/// kind so the renderer dispatches `march_uv_ring` at it.
fn build_ring_slab(
    library: &mut NodeLibrary,
    leaves: Vec<Child>,
    root_kind: NodeKind,
) -> Child {
    let mut layer = leaves;
    let mut size = layer.len();
    while size > 1 {
        debug_assert_eq!(size % 3, 0);
        let next_size = size / 3;
        let mut next: Vec<Child> = Vec::with_capacity(next_size);
        for x in 0..next_size {
            let mut children = empty_children();
            for dx in 0..3 {
                children[slot_index(dx, 0, 0)] = layer[x * 3 + dx];
            }
            let kind = if next_size == 1 { root_kind } else { NodeKind::Cartesian };
            let all_empty = children.iter().all(|c| c.is_empty());
            next.push(if all_empty {
                Child::Empty
            } else {
                Child::Node(library.insert_with_kind(children, kind))
            });
        }
        layer = next;
        size = next_size;
    }
    layer[0]
}

pub(super) fn sphere_ring_test_world() -> WorldState {
    let mut library = NodeLibrary::default();

    // Single deduped content subtree, shared by every ring cell.
    let cell_content =
        build_uniform_cartesian_subtree(&mut library, block::STONE, CELL_SUBTREE_DEPTH);

    let leaves: Vec<Child> = (0..RING_CELLS).map(|_| cell_content).collect();

    let root_child = build_ring_slab(
        &mut library,
        leaves,
        NodeKind::UvRing {
            dims: [RING_CELLS, 1, 1],
            slab_depth: RING_SLAB_DEPTH,
        },
    );
    let root = match root_child {
        Child::Node(id) => id,
        _ => library.insert(empty_children()),
    };
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "sphere_ring_test world: tree_depth={}, library_entries={}, ring_cells={}",
        world.tree_depth(),
        world.library.len(),
        RING_CELLS,
    );
    world
}

/// Camera spawn: top-front of the root cube `[0, 3)³`, framed so
/// the ring (centred at `(1.5, 1.5, 1.5)`, radius 1.0 in the XZ
/// plane) reads as a 3D oval rather than a degenerate flat
/// projection. Slot `(1, 2, 2)` puts the camera at world position
/// `(1.5, 2.5, 2.5)` — directly above the +Z arc of the ring,
/// looking back down toward the centre via `yaw = +π/2` (toward
/// -Z) plus a moderate pitch.
pub(super) fn sphere_ring_test_spawn() -> WorldPos {
    WorldPos::uniform_column(slot_index(1, 2, 2) as u8, 1, [0.5, 0.5, 0.5])
}

pub(super) fn bootstrap_sphere_ring_test_world() -> WorldBootstrap {
    let world = sphere_ring_test_world();
    let spawn_pos = sphere_ring_test_spawn();
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        // Yaw = -π/2 looks toward -Z (down the +Z arc of the ring
        // toward the -Z arc); pitch -π/4 tilts the gaze toward the
        // ring centre at (1.5, 1.5, 1.5).
        default_spawn_yaw: -std::f32::consts::FRAC_PI_2,
        default_spawn_pitch: -std::f32::consts::FRAC_PI_4,
        plain_layers: 0,
        color_registry: crate::world::palette::ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_is_uv_ring_kind() {
        let world = sphere_ring_test_world();
        let root_node = world.library.get(world.root).expect("root exists");
        match root_node.kind {
            NodeKind::UvRing { dims, slab_depth } => {
                assert_eq!(dims, [RING_CELLS, 1, 1]);
                assert_eq!(slab_depth, RING_SLAB_DEPTH);
            }
            other => panic!("expected UvRing root, got {other:?}"),
        }
    }

    #[test]
    fn tree_has_expected_depth() {
        let world = sphere_ring_test_world();
        // Slab depth + cell content depth — the ring storage depth
        // contributes its full RING_SLAB_DEPTH levels above the
        // shared cell subtree.
        assert_eq!(
            world.tree_depth(),
            RING_SLAB_DEPTH as u32 + CELL_SUBTREE_DEPTH as u32,
        );
    }

    /// Every cell along the ring's storage X axis resolves to a
    /// non-empty Node — proving the slab bootstrap populated all
    /// `RING_CELLS` slots and that the deduped content is reachable
    /// at every cell.
    #[test]
    fn every_cell_in_ring_storage_is_populated() {
        let world = sphere_ring_test_world();
        for cell_x in 0..RING_CELLS {
            let mut node_id = world.root;
            let mut cells_per_slot: u32 = 1;
            for _ in 1..RING_SLAB_DEPTH {
                cells_per_slot *= 3;
            }
            for _ in 0..RING_SLAB_DEPTH {
                let sx = ((cell_x / cells_per_slot) % 3) as usize;
                let slot = slot_index(sx, 0, 0);
                let node = world.library.get(node_id).expect("node exists");
                match node.children[slot] {
                    Child::Node(child) => {
                        node_id = child;
                    }
                    other => panic!(
                        "ring cell {cell_x} missing at slot {slot}: {other:?}",
                    ),
                }
                cells_per_slot = (cells_per_slot / 3).max(1);
            }
        }
    }

    /// Ring content is shared (deduped) across all cells. With
    /// `RING_CELLS = 27` cells all pointing at one uniform-stone
    /// subtree, the library should contain only the slab descent
    /// nodes plus the one shared content chain — far fewer than 27
    /// distinct copies.
    #[test]
    fn ring_content_is_deduped() {
        let world = sphere_ring_test_world();
        let total = world.library.len();
        // Slab descent: 1 root + 3 mid + 9 leaf-parents = 13 nodes.
        // Content chain: CELL_SUBTREE_DEPTH = 30 nodes.
        // Plus a small slack for any other library entries created
        // by `insert_with_kind` deduping under a kind-specific hash.
        let upper_bound = (RING_SLAB_DEPTH as usize) * 3
            + (CELL_SUBTREE_DEPTH as usize)
            + 16;
        assert!(
            total < upper_bound,
            "library should stay deduped: got {total} entries, expected < {upper_bound}",
        );
    }
}
