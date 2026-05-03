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
/// all `RING_CELLS` cells share this one Cartesian content chain
/// underneath their distinct tangent-block heads.
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

/// Build a `TangentBlock`-headed cell subtree of total depth
/// `CELL_SUBTREE_DEPTH`. The outermost node carries `rotation` so
/// the existing TB primitive applies the ring tangent basis at
/// cube descent — exactly the same pattern as
/// [`dodecahedron_test::build_tangent_block_subtree`]. The TB head
/// also prevents the GPU packer from collapsing the uniform-stone
/// subtree into a single Block (which would make per-cell edits
/// land on the whole cell instead of individual voxels).
fn build_tangent_block_cell(
    library: &mut NodeLibrary,
    block_id: u16,
    depth: u8,
    rotation: [[f32; 3]; 3],
) -> Child {
    assert!(depth >= 1, "TangentBlock subtree depth must be >= 1");
    let inner = build_uniform_cartesian_subtree(library, block_id, depth - 1);
    Child::Node(library.insert_with_kind(
        uniform_children(inner),
        NodeKind::TangentBlock { rotation },
    ))
}

/// Ring tangent basis at angle `theta` (radians, CCW around +Y).
/// Column-major `[tangent | radial | up]`: storage `+X` → tangent,
/// storage `+Y` → radial, storage `+Z` → up. Same convention as
/// [`dodecahedron_test::rotation_align_y_to`].
fn ring_rotation(theta: f32) -> [[f32; 3]; 3] {
    let (st, ct) = theta.sin_cos();
    [
        [-st, 0.0, ct],
        [ct, 0.0, st],
        [0.0, 1.0, 0.0],
    ]
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

    // Per-cell TangentBlock heads carrying the ring tangent basis
    // rotation. The Cartesian content chain underneath dedups across
    // all cells; only the TB heads themselves are distinct (one
    // library entry per cell, since each rotation hashes to a
    // different bit pattern).
    let two_pi = std::f32::consts::TAU;
    let pi = std::f32::consts::PI;
    let leaves: Vec<Child> = (0..RING_CELLS)
        .map(|cell_x| {
            let theta = -pi + (cell_x as f32 + 0.5) * (two_pi / RING_CELLS as f32);
            let rotation = ring_rotation(theta);
            build_tangent_block_cell(
                &mut library, block::STONE, CELL_SUBTREE_DEPTH, rotation,
            )
        })
        .collect();

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

    /// Zooming in on a UvRing world advances the anchor by exactly
    /// `slab_depth` slots (the storage path for the cell nearest
    /// the camera in ring topology). Without `zoom_in_uv_ring`, the
    /// standard 3³ slot pick lands on slots like `(1, 1, 2)` which
    /// have no children in the `[27, 1, 1]` slab — render frame
    /// stays at root regardless of zoom level (the bug from the
    /// debug capture).
    #[test]
    fn zoom_in_descends_into_uv_ring_cell() {
        use crate::world::anchor::WorldPos;

        let world = sphere_ring_test_world();
        let mut pos = WorldPos::new(
            crate::world::anchor::Path::root(),
            // Camera at world (1.5, 1.5, 2.5) — the +Z arc of the
            // ring. Offset = world / 3 because anchor depth is 0.
            [0.5, 0.5, 5.0 / 6.0],
        );
        pos.zoom_in_in_world(&world.library, world.root);

        assert_eq!(
            pos.anchor.depth(),
            RING_SLAB_DEPTH,
            "zoom_in on UvRing root should advance anchor by exactly slab_depth ({}); got {}",
            RING_SLAB_DEPTH,
            pos.anchor.depth(),
        );

        for k in 0..RING_SLAB_DEPTH as usize {
            let slot = pos.anchor.slot(k);
            let (_, sy, sz) = crate::world::tree::slot_coords(slot as usize);
            assert_eq!(sy, 0, "slab slot at level {k} has sy != 0: {slot}");
            assert_eq!(sz, 0, "slab slot at level {k} has sz != 0: {slot}");
        }

        // Walk the slab path; final child must be a TangentBlock head.
        let mut node_id = world.root;
        for k in 0..RING_SLAB_DEPTH as usize {
            let slot = pos.anchor.slot(k) as usize;
            let n = world.library.get(node_id).expect("node along slab path");
            match n.children[slot] {
                Child::Node(child) => {
                    if k + 1 == RING_SLAB_DEPTH as usize {
                        let cell = world.library.get(child).expect("cell head");
                        assert!(
                            cell.kind.is_tangent_block(),
                            "cell head not a TB: {:?}",
                            cell.kind,
                        );
                    } else {
                        node_id = child;
                    }
                }
                other => panic!("expected Node at slab level {k}: {other:?}"),
            }
        }
    }

    /// Each ring cell carries a distinct TangentBlock rotation.
    /// With `RING_CELLS = 27` distinct rotations, the library
    /// contains 27 TB head entries; the Cartesian content chain
    /// underneath dedups across all cells.
    #[test]
    fn ring_cells_are_tangent_blocks_with_distinct_rotations() {
        let world = sphere_ring_test_world();
        // Walk the slab path for each cell_x and confirm the leaf
        // child is a Node whose kind is TangentBlock.
        let mut tb_ids = std::collections::HashSet::new();
        for cell_x in 0..RING_CELLS {
            let mut node_id = world.root;
            let mut cells_per_slot: u32 = 1;
            for _ in 1..RING_SLAB_DEPTH {
                cells_per_slot *= 3;
            }
            for level in 0..RING_SLAB_DEPTH {
                let sx = ((cell_x / cells_per_slot) % 3) as usize;
                let slot = slot_index(sx, 0, 0);
                let node = world.library.get(node_id).expect("node");
                match node.children[slot] {
                    Child::Node(child) => {
                        if level + 1 == RING_SLAB_DEPTH {
                            let n = world.library.get(child).expect("cell head");
                            assert!(
                                n.kind.is_tangent_block(),
                                "cell {cell_x} head is not a TangentBlock: {:?}",
                                n.kind,
                            );
                            tb_ids.insert(child);
                        } else {
                            node_id = child;
                        }
                    }
                    other => panic!("cell {cell_x} missing at slot {slot}: {other:?}"),
                }
                cells_per_slot = (cells_per_slot / 3).max(1);
            }
        }
        assert_eq!(
            tb_ids.len(),
            RING_CELLS as usize,
            "expected {RING_CELLS} distinct TB heads, got {}",
            tb_ids.len(),
        );
    }
}
