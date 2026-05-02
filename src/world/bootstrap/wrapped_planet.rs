//! `WrappedPlane` planet bootstrap (Phase 1: hardcoded slab).

use super::WorldBootstrap;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, uniform_children, BRANCH, Child, NodeKind, NodeLibrary,
    MAX_DEPTH,
};

/// Default slab dims (cells along x, y, z) for `--wrapped-planet`.
///
/// Axis convention in this worldgen (NOTE: differs from the
/// architecture doc — see comment block below):
/// - `dims[0]` = X = longitude. Wrap-axis. Must equal `3^slab_depth`
///   for Phase 2 wrap correctness (slab must fully fill the
///   WrappedPlane node along X so the shader's `depth==0 && OOB-on-X`
///   trigger lands on the slab footprint edge). With `slab_depth = 3`
///   that's `dims[0] = 27`.
/// - `dims[1]` = Y = vertical (gravity-aligned). Grass at the top
///   row, stone at the bottom row, dirt between. This is the
///   "dig-down depth" the player experiences. The architecture doc
///   calls this `Z` ("shallow Z"); we use `Y` because Y is the
///   game's gravity axis.
/// - `dims[2]` = Z = latitude (the OTHER horizontal). Bounded; ends
///   are non-buildable polar strips (Phase 4 handling pending).
///
/// X:Z aspect targets ≈ 2:1 so cells are roughly square when wrapped
/// onto the sphere (longitude spans 360°, latitude 180°). Exact 2:1
/// is impossible while `dims[0] = 3^N` (powers of 3 are always odd);
/// the closest integer split of 27 is 14 (≈ 1.93:1). True exact 2:1
/// requires moving the wrap trigger from the WrappedPlane node edge
/// down to the slab footprint edge — a deeper change deferred until
/// after the curvature math is settled.
pub const DEFAULT_WRAPPED_PLANET_SLAB_DIMS: [u32; 3] = [27, 2, 14];
/// Default depth descended below the `WrappedPlane` node to reach
/// the slab cell anchors. `slab_depth = 3` ⇒ subgrid is 27³, which
/// matches `dims[0]`. Higher values widen the WrappedPlane so the
/// slab no longer fully fills X, breaking the wrap geometry; lower
/// values shrink the subgrid below `dims[0]`.
pub const DEFAULT_WRAPPED_PLANET_SLAB_DEPTH: u8 = 3;
/// Default tree depth at which the `WrappedPlane` node is installed.
/// Slab cell anchors live at `embedding_depth + slab_depth`; each
/// anchor has a recursive subtree of `cell_subtree_depth` more levels
/// underneath. Total tree depth = embedding + slab + cell_subtree.
pub const DEFAULT_WRAPPED_PLANET_EMBEDDING_DEPTH: u8 = 2;
/// Default depth of the recursive subtree under each slab cell.
/// Per `[Recursive architecture]`, slab cells are NOT terminal blocks
/// — they're `Child::Node(uniform_subtree)` so each cell is a real
/// anchor with content beneath it. `20` gives layer-5 anchor blocks
/// (= `EMBEDDING + SLAB`) each with a 20-deep subtree, summing to a
/// 25-layer tree by default.
pub const DEFAULT_WRAPPED_PLANET_CELL_SUBTREE_DEPTH: u8 = 20;

/// Build a `NodeKind::WrappedPlane`-rooted slab embedded in an
/// otherwise empty world.
///
/// The world tree has the shape:
///
/// ```text
/// root (Cartesian, depth 0)
///   slot 13 (centre) -> Cartesian (depth 1, all-empty except slot 13)
///   ...                                          // embedding_depth-1 layers
///   -> WrappedPlane { dims, slab_depth }         (depth = embedding_depth)
///        -> Cartesian descendants (slab_depth levels)
///             -> Block(GRASS|DIRT|STONE) | Empty (depth = embedding_depth+slab_depth)
/// ```
///
/// The slab footprint inside the WrappedPlane's `3^slab_depth` per
/// axis subgrid is `dims[0] × dims[1] × dims[2]` cells. Cells outside
/// that footprint are `Child::Empty` (sparse occupancy = absent).
///
/// Surface profile within the slab footprint mirrors `--plain-world`:
/// Y=top → grass; Y=middle → dirt; Y=bottom → stone. With dims.y=10
/// and slab_dims.y=2 thickness the canonical layout is grass row at
/// y=1 (top), dirt rows below it; we keep the y=0 row (the lowest)
/// as stone for a recognisable cross-section.
///
/// **Phase 1** does not enable wrap or curvature — the resulting
/// rectangle just renders as a small flat patch of terrain.
pub fn wrapped_planet_world(
    embedding_depth: u8,
    slab_dims: [u32; 3],
    slab_depth: u8,
    cell_subtree_depth: u8,
) -> WorldState {
    assert!(embedding_depth > 0, "embedding_depth must be >= 1");
    assert!(slab_depth > 0, "slab_depth must be >= 1");
    assert!(
        cell_subtree_depth >= 1,
        "wrapped_planet requires cell_subtree_depth >= 1 (TangentBlock wraps the per-cell chain)",
    );
    let total_depth = (embedding_depth as usize)
        .saturating_add(slab_depth as usize)
        .saturating_add(cell_subtree_depth as usize);
    assert!(
        total_depth <= MAX_DEPTH,
        "embedding+slab+cell_subtree ({}) exceeds MAX_DEPTH ({})",
        total_depth,
        MAX_DEPTH,
    );
    // Subgrid extent: 3^slab_depth cells per axis. dims must fit.
    let mut subgrid: u32 = 1;
    for _ in 0..slab_depth {
        subgrid = subgrid.checked_mul(BRANCH as u32).expect("slab_depth too large");
    }
    assert!(
        slab_dims[0] <= subgrid && slab_dims[1] <= subgrid && slab_dims[2] <= subgrid,
        "slab_dims {:?} do not fit in subgrid {} (slab_depth={}); increase slab_depth",
        slab_dims, subgrid, slab_depth,
    );

    let mut library = NodeLibrary::default();

    // Per-material uniform anchor subtrees. Each is a chain of
    // Cartesian nodes `cell_subtree_depth` levels deep, all 27
    // children pointing to the same material. Content-addressed
    // dedup means the entire chain is exactly `cell_subtree_depth`
    // library entries per material — irrespective of the slab
    // population count. This is what makes each slab cell a
    // proper anchor block per `[Recursive architecture]`: cells
    // are `Child::Node(...)`, not `Child::Block(...)`, so the
    // recursive subdivision goes all the way down.
    //
    // The OUTERMOST node of each cell's chain (= the slab cell
    // anchor) is `NodeKind::TangentBlock`. The shader detects this
    // kind, transforms the ray into the cell's local tangent cube
    // frame, and dispatches `march_in_tangent_cube` from there — so
    // every voxel below the slab cell sees the precision-stable
    // Cartesian path. The chain BELOW the TangentBlock is regular
    // Cartesian and uniform-flattens away in the GPU pack.
    fn build_uniform_anchor(library: &mut NodeLibrary, block: u16, depth: u8) -> Child {
        if depth == 0 {
            return Child::Block(block);
        }
        let inner = build_uniform_anchor(library, block, depth - 1);
        Child::Node(library.insert(uniform_children(inner)))
    }
    let make_anchor = |library: &mut NodeLibrary, block: u16| -> Child {
        let inner = build_uniform_anchor(library, block, cell_subtree_depth - 1);
        Child::Node(library.insert_with_kind(
            uniform_children(inner),
            NodeKind::TangentBlock { rotation: crate::world::tree::IDENTITY_ROTATION },
        ))
    };
    let stone_anchor = make_anchor(&mut library, block::STONE);
    let dirt_anchor  = make_anchor(&mut library, block::DIRT);
    let grass_anchor = make_anchor(&mut library, block::GRASS);

    // Build the slab subtree as a flat `subgrid^3` Cartesian volume,
    // sparsely populated to the slab_dims footprint. The simplest
    // exact construction is to walk the leaf-cell grid and at each
    // (x, y, z) pick the per-material anchor (or Empty), then
    // bottom-up assemble 3×3×3 nodes layer by layer.
    //
    // Leaf-cell selection rule for the slab footprint:
    //   x ∈ [0, dims.x)  AND  y ∈ [0, dims.y)  AND  z ∈ [0, dims.z)
    //     - bottom row (y == 0): stone_anchor
    //     - middle rows (1..dims.y-1): dirt_anchor
    //     - top row (y == dims.y-1): grass_anchor
    //   else: Empty
    let leaf_at = |x: u32, y: u32, z: u32| -> Child {
        if x < slab_dims[0] && y < slab_dims[1] && z < slab_dims[2] {
            if y == 0 {
                stone_anchor
            } else if y + 1 == slab_dims[1] {
                grass_anchor
            } else {
                dirt_anchor
            }
        } else {
            Child::Empty
        }
    };

    // Layer 0: leaves laid out at subgrid^3.
    let n0 = subgrid as usize;
    let mut layer: Vec<Vec<Vec<Child>>> = (0..n0)
        .map(|z| {
            (0..n0)
                .map(|y| (0..n0).map(|x| leaf_at(x as u32, y as u32, z as u32)).collect())
                .collect()
        })
        .collect();
    // (This builds the 3D array at full subgrid resolution. For the
    // canonical [20,10,2]@slab_depth=3 case that's 27³ = 19683
    // entries — cheap.)

    // Successively group every 3×3×3 block into a single Cartesian
    // node. After `slab_depth` rounds, layer is a 1×1×1 array
    // containing the WrappedPlane root's children-pre-image: actually
    // we want to stop the Cartesian-grouping ONE round before that
    // last grouping, then explicitly insert the WrappedPlane node
    // for that final 3×3×3 → 1 step so the kind tag attaches.
    let mut size = n0;
    for _round in 0..(slab_depth as usize - 1) {
        let new_size = size / 3;
        let mut next: Vec<Vec<Vec<Child>>> = (0..new_size)
            .map(|_| (0..new_size).map(|_| vec![Child::Empty; new_size]).collect())
            .collect();
        for nz in 0..new_size {
            for ny in 0..new_size {
                for nx in 0..new_size {
                    let mut children = empty_children();
                    let mut all_empty = true;
                    for cz in 0..BRANCH {
                        for cy in 0..BRANCH {
                            for cx in 0..BRANCH {
                                let x = nx * BRANCH + cx;
                                let y = ny * BRANCH + cy;
                                let z = nz * BRANCH + cz;
                                let c = layer[z][y][x];
                                if !c.is_empty() {
                                    all_empty = false;
                                }
                                children[slot_index(cx, cy, cz)] = c;
                            }
                        }
                    }
                    next[nz][ny][nx] = if all_empty {
                        // Pure-empty Cartesian subtree gets dropped
                        // by uniform-flatten anyway, but constructing
                        // an explicit Empty here saves a library entry.
                        Child::Empty
                    } else {
                        Child::Node(library.insert_with_kind(children, NodeKind::Cartesian))
                    };
                }
            }
        }
        layer = next;
        size = new_size;
    }
    debug_assert_eq!(size, BRANCH as usize, "expected one final 3³ grouping for the WrappedPlane root");

    // Final pass: assemble the 27 layer-cells into one
    // `NodeKind::WrappedPlane` node. This is the slab root.
    let mut slab_children = empty_children();
    for cz in 0..BRANCH {
        for cy in 0..BRANCH {
            for cx in 0..BRANCH {
                slab_children[slot_index(cx, cy, cz)] = layer[cz][cy][cx];
            }
        }
    }
    let wrapped_plane_root = library.insert_with_kind(
        slab_children,
        NodeKind::WrappedPlane { dims: slab_dims, slab_depth },
    );

    // Embed: wrap in `embedding_depth` Cartesian layers, each
    // placing the inner subtree at the centre slot (13). Outer slots
    // are Empty so the slab sits in otherwise empty space.
    let mut current = Child::Node(wrapped_plane_root);
    for _ in 0..(embedding_depth as usize) {
        let mut children = empty_children();
        children[slot_index(1, 1, 1)] = current;
        current = Child::Node(library.insert_with_kind(children, NodeKind::Cartesian));
    }
    let root = match current {
        Child::Node(id) => id,
        // A Block / Empty here would mean embedding_depth = 0 with
        // a degenerate slab — guarded above with the assert.
        _ => unreachable!("wrapped-planet embedding produced a non-Node root"),
    };
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "wrapped_planet world: embedding_depth={}, slab_dims={:?}, slab_depth={}, cell_subtree_depth={}, library_entries={}, tree_depth={}",
        embedding_depth, slab_dims, slab_depth, cell_subtree_depth,
        world.library.len(), world.tree_depth(),
    );
    world
}

/// Spawn position for the wrapped planet preset: above the slab top
/// surface, looking straight down at it.
///
/// **Anchor depth = `embedding_depth + slab_depth`** (slab leaf
/// level). The deeper anchor is REQUIRED for X-wrap to fire on
/// player movement: `step_neighbor_in_world` only wraps when an
/// overflowing slot's parent is the WrappedPlane node, which lives
/// at tree depth `embedding_depth`. The slot whose parent is the
/// WrappedPlane is `slots[embedding_depth]` — the slot at tree
/// depth `embedding_depth + 1`. So the path must be at least
/// `embedding_depth + 1` slots long for the chain of bubble-up
/// overflows from offset.x crossing 1.0 to ever reach the
/// WrappedPlane's children grid. Anchoring at the leaf level
/// (`embedding_depth + slab_depth`) gives the natural representation
/// where each `add_local` step crosses a leaf cell, and the wrap
/// fires after `dims[0]` such steps.
///
/// Initial offset is computed in the WrappedPlane cell's `[0, 1)`
/// frame (where the slab occupies `[0, dims.i / 3^slab_depth)`),
/// then `deepened_to` pushes the anchor to leaf depth via slot
/// arithmetic on the offset (precision-stable, no f32 accumulation).
///
/// Camera offset (in WrappedPlane cell `[0, 1)` coords):
/// - cam_x = slab_x_centre = (dims.x / 3^slab_depth) / 2
/// - cam_y = slab_top + air_gap = (dims.y / 3^slab_depth) + clearance
/// - cam_z = slab_z_centre = (dims.z / 3^slab_depth) / 2
pub fn wrapped_planet_spawn(
    embedding_depth: u8,
    slab_dims: [u32; 3],
    slab_depth: u8,
    library: &crate::world::tree::NodeLibrary,
    world_root: crate::world::tree::NodeId,
) -> WorldPos {
    let subgrid = (BRANCH as u32).pow(slab_depth as u32) as f32;
    let frac_x = slab_dims[0] as f32 / subgrid;
    let frac_y = slab_dims[1] as f32 / subgrid;
    let frac_z = slab_dims[2] as f32 / subgrid;
    let cam_x = (frac_x * 0.5).clamp(0.001, 0.999);
    let clearance = (frac_x * 0.7).max(0.05);
    let cam_y = (frac_y + clearance).clamp(0.001, 0.95);
    let cam_z = (frac_z * 0.5).clamp(0.001, 0.999);
    // Construct at the WrappedPlane cell, then deepen to slab leaf
    // depth so movement-time `add_local → renormalize_world` sees
    // a path long enough that X-overflow at the leaf bubbles all
    // the way up to the WrappedPlane's children grid (`slots[
    // embedding_depth]`), where the wrap-aware step actually fires.
    WorldPos::uniform_column(
        slot_index(1, 1, 1) as u8,
        embedding_depth,
        [cam_x, cam_y, cam_z],
    )
    .deepened_to(embedding_depth + slab_depth, library, world_root)
}

pub(super) fn bootstrap_wrapped_planet_world(
    embedding_depth: u8,
    slab_dims: [u32; 3],
    slab_depth: u8,
    cell_subtree_depth: u8,
) -> WorldBootstrap {
    let world = wrapped_planet_world(
        embedding_depth, slab_dims, slab_depth, cell_subtree_depth,
    );
    // Camera spawns AT the slab cell anchor depth (just above the
    // GRASS surface). The cell's subtree below — `cell_subtree_depth`
    // levels of recursive material — is invisible from this anchor;
    // zooming IN by N levels reveals the next N layers of the
    // subtree, all the way down to the leaf Block at total tree depth.
    let spawn_pos = wrapped_planet_spawn(
        embedding_depth, slab_dims, slab_depth, &world.library, world.root,
    );
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -std::f32::consts::FRAC_PI_2,
        // `plain_layers = 0` opts out of the spawn-depth surface-
        // tracking path in `App::with_test_config` — that path
        // assumes a `--plain-world` ground plane and would otherwise
        // override our wrapped-planet spawn with `plain_surface_spawn`.
        plain_layers: 0,
        color_registry: crate::world::palette::ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::Child;

    /// Worldgen regression: the wrapped planet must produce a tree
    /// containing a `NodeKind::WrappedPlane` node carrying the
    /// requested dims. Walk down the centre-slot embedding chain and
    /// assert.
    #[test]
    fn wrapped_planet_produces_wrapped_plane_node() {
        let embedding_depth: u8 = 8;
        let slab_dims = [27u32, 10, 2];
        let slab_depth: u8 = 3;
        let world = wrapped_planet_world(embedding_depth, slab_dims, slab_depth, 1);
        let mut node_id = world.root;
        for _ in 0..embedding_depth {
            let node = world.library.get(node_id).expect("node exists");
            assert_eq!(node.kind, NodeKind::Cartesian, "embedding layer must be Cartesian");
            match node.children[slot_index(1, 1, 1)] {
                Child::Node(child) => node_id = child,
                other => panic!("expected Child::Node at embedding centre, got {:?}", other),
            }
        }
        let slab_root = world.library.get(node_id).expect("slab root exists");
        match slab_root.kind {
            NodeKind::WrappedPlane { dims, slab_depth: sd } => {
                assert_eq!(dims, slab_dims);
                assert_eq!(sd, slab_depth);
            }
            other => panic!("expected WrappedPlane at depth {embedding_depth}, got {:?}", other),
        }
    }

    /// Total tree depth = embedding_depth + slab_depth + cell_subtree_depth.
    #[test]
    fn wrapped_planet_total_tree_depth() {
        let world = wrapped_planet_world(8, [27, 10, 2], 3, 1);
        assert_eq!(world.tree_depth(), 8 + 3 + 1);
        let world = wrapped_planet_world(8, [27, 10, 2], 3, 5);
        assert_eq!(world.tree_depth(), 8 + 3 + 5);
    }

    /// Every populated slab cell anchor must be `NodeKind::TangentBlock`
    /// — the slab is rendered as a sphere of rotated tangent cubes, so
    /// each cell carries the kind metadata that drives the per-cell
    /// rotation in `march_wrapped_planet`.
    #[test]
    fn slab_anchor_is_tangent_block() {
        let embedding_depth: u8 = 2;
        let slab_dims = [27u32, 2, 14];
        let slab_depth: u8 = 3;
        let cell_subtree_depth: u8 = 5;

        // Path to a populated slab cell: embedding centres (1,1,1),
        // then slab-depth steps into x=1, y=0 (stone row), z=1.
        let mut populated_path = Vec::new();
        for _ in 0..embedding_depth {
            populated_path.push(slot_index(1, 1, 1));
        }
        for _ in 0..slab_depth {
            populated_path.push(slot_index(1, 0, 1));
        }
        let walk = |w: &WorldState, path: &[usize]| -> Option<NodeKind> {
            let mut node_id = w.root;
            for &slot in path {
                let node = w.library.get(node_id)?;
                match node.children[slot] {
                    Child::Node(child) => node_id = child,
                    _ => return None,
                }
            }
            w.library.get(node_id).map(|n| n.kind)
        };

        let world = wrapped_planet_world(
            embedding_depth, slab_dims, slab_depth, cell_subtree_depth,
        );
        let kind = walk(&world, &populated_path).expect("populated cell exists");
        assert!(kind.is_tangent_block(), "expected TangentBlock, got {kind:?}");
    }
}
