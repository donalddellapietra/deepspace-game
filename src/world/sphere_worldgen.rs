//! SDF-driven worldgen for a cubed-sphere body.
//!
//! Stage 0b of the unified-DDA rewrite — CPU-only. This module owns
//! the tree construction for a single spherical planet:
//!
//! - [`PlanetSetup`] — declarative description (radii, depth, SDF).
//! - [`demo_planet`] — the starter / demo preset.
//! - [`insert_spherical_body`] — build the body subtree from an SDF
//!   and insert it into a [`NodeLibrary`], tagging the root with
//!   `NodeKind::CubedSphereBody` and the 6 face-center slots with
//!   `NodeKind::CubedSphereFace`.
//! - [`install_at_root_center`] — place a body at slot 13 (the depth-1
//!   centre cell) of a given world-root node.
//!
//! The body lives entirely inside the voxel tree. There is no
//! separate `SphericalPlanet` handle, no parallel raycaster, no
//! extra GPU buffer — the tree is the single source of truth. The
//! NodeKind tags on the body root and its 6 face-center slots tell
//! future shader / DDA code how to interpret child axes when the
//! unified DDA walks through; for Stage 0b everything is still
//! rendered as Cartesian (see `GpuNodeKind::from_node_kind`).
//!
//! Radii (`inner_r`, `outer_r`) are expressed in the **containing
//! cell's local `[0, 1)` frame** per the spec §1d; they scale with
//! wherever the body cell lives in the tree.

use super::anchor::Path;
use super::cubesphere::{Face, CORE_SLOT, FACE_SLOTS};
use super::palette::block;
use super::sdf::{Planet, Vec3};
use super::tree::{
    empty_children, slot_index, uniform_children, Child, NodeId, NodeKind, NodeLibrary,
};

// ────────────────────────────────────────────────── detail budget

/// Max levels the SDF recursion is allowed to descend into a body
/// subtree before committing to uniform solid-or-empty based on the
/// cell's centre sample. Below this, the remaining tree depth is
/// wrapped in a dedup'd uniform filler.
///
/// Cartesian-indexed worldgen visits 9× more straddle cells per
/// added level (a 2D surface through the 3D grid), and runs across
/// all 27 body slots rather than just 6 face-center slots — so
/// every added level costs ~40× the work of the previous one.
/// 4 keeps worldgen tractable (~seconds for a single planet) while
/// leaving the LOD-terminal sphere-clip in the shader to smooth the
/// silhouette past the voxel resolution limit.
pub const SDF_DETAIL_LEVELS: u32 = 4;

// ────────────────────────────────────────────────── PlanetSetup

/// Declarative planet description. Radii are in **the body cell's
/// local `[0, 1)` frame** (per spec §1d). The body cell's actual
/// world-space size is determined by where in the tree it lives.
#[derive(Clone, Debug)]
pub struct PlanetSetup {
    /// Radii in body cell local `[0, 1)`.
    pub inner_r: f32,
    pub outer_r: f32,
    /// Face subtree depth.
    pub depth: u32,
    /// SDF in body cell local frame (centre = `(0.5, 0.5, 0.5)`).
    pub sdf: Planet,
}

/// The demo / starter planet. Body cell-local: `outer_r <= 0.5` so
/// the sphere fits cleanly in one cell of its parent.
pub fn demo_planet() -> PlanetSetup {
    let center: Vec3 = [0.5, 0.5, 0.5];
    let inner_r = 0.12_f32;
    let outer_r = 0.45_f32;
    PlanetSetup {
        inner_r,
        outer_r,
        depth: 28,
        sdf: Planet {
            center,
            radius: 0.30,
            noise_scale: 0.015,
            noise_freq: 8.0,
            noise_seed: 2024,
            gravity: 9.8,
            influence_radius: outer_r * 2.0,
            surface_block: block::GRASS,
            core_block: block::STONE,
        },
    }
}

// ────────────────────────────────────────────── uniform chains

/// Build a `depth`-level all-empty subtree via content-addressed
/// dedup and return its `NodeId`. A single chain is materialised
/// (O(depth) nodes) because the library dedupes identical
/// siblings; any number of `uniform_empty_chain` callers at the
/// same depth will share the same `NodeId`.
pub fn uniform_empty_chain(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut current = lib.insert(empty_children());
    for _ in 1..depth.max(1) {
        current = lib.insert(uniform_children(Child::Node(current)));
    }
    current
}

// ────────────────────────────────────────────────── body insertion

/// Build the body as a Cartesian-indexed 27-ary tree of SDF-carved
/// content, insert it into `lib`, and return its `NodeId`. The body
/// node itself carries `NodeKind::CubedSphereBody { inner_r, outer_r }`
/// so the renderer's ray-sphere pre-clip can fire on ray-body
/// dispatch; the 6 face-center slots additionally carry
/// `NodeKind::CubedSphereFace { face }` so a later camera-basis
/// pipeline can rotate "up" to the face's radial axis when zooming
/// into a face subtree.
///
/// All other body slots (edges, corners, interior) are plain
/// Cartesian subtrees. The sphere surface may wrap into edge and
/// corner slots when the sphere's radius is larger than the
/// face-center slot's reach — the Cartesian-indexed worldgen makes
/// every slot check its own position against the SDF, so gaps
/// between face subtrees on the sphere surface are populated.
///
/// `inner_r` and `outer_r` are in the containing-cell's local
/// `[0, 1)³` frame (per the spec's §1d). `depth` is the maximum
/// recursion depth inside the body.
///
/// The SDF (`sdf`) is sampled in the containing cell's local frame
/// — its `center` should be `(0.5, 0.5, 0.5)` and its `radius` and
/// `noise_scale` should be in cell-local units.
pub fn insert_spherical_body(
    lib: &mut NodeLibrary,
    inner_r: f32,
    outer_r: f32,
    depth: u32,
    sdf: &Planet,
) -> NodeId {
    debug_assert!(
        0.0 < inner_r && inner_r < outer_r && outer_r <= 0.5,
        "radii must satisfy 0 < inner_r < outer_r <= 0.5 (cell-local)",
    );

    let sdf_budget = depth.min(SDF_DETAIL_LEVELS);

    // Precompute uniform subtrees for every depth 1..=depth for
    // every block type the SDF's `block_at` can return (surface,
    // DIRT, core) plus the all-empty subtree. The SDF-carving
    // recursion hits commit-to-uniform millions of times at the
    // budget boundary; turning each of those into an O(1) table
    // lookup instead of an O(depth) `lib.build_uniform_subtree`
    // call is what keeps worldgen tractable at high SDF budgets
    // (~100x speedup at SDF_DETAIL_LEVELS=6).
    //
    // `uniform_empty[d]` (d >= 1) = NodeId of a `d`-level all-empty
    // subtree. `uniform_block[b][d]` = Child for a `d`-level uniform
    // subtree of block `b` (Block at d=0, Node at d >= 1). Index 0
    // of uniform_empty is a placeholder (callers handle d=0 with
    // `Child::Empty` directly).
    let depth_plus_one = depth as usize + 1;
    let mut uniform_empty: Vec<NodeId> = Vec::with_capacity(depth_plus_one);
    let mut uniform_block: std::collections::HashMap<u16, Vec<Child>> =
        std::collections::HashMap::new();
    let block_kinds: [u16; 3] = [sdf.surface_block, block::DIRT, sdf.core_block];
    {
        let empty_leaf = lib.insert(empty_children());
        uniform_empty.push(empty_leaf);
        uniform_empty.push(empty_leaf);
        for _ in 2..=depth {
            let prev_e = *uniform_empty.last().unwrap();
            let next_e = lib.insert(uniform_children(Child::Node(prev_e)));
            uniform_empty.push(next_e);
        }
        for b in block_kinds.iter().copied() {
            if uniform_block.contains_key(&b) {
                continue;
            }
            let mut v: Vec<Child> = Vec::with_capacity(depth_plus_one);
            v.push(Child::Block(b));
            let leaf = lib.insert(uniform_children(Child::Block(b)));
            v.push(Child::Node(leaf));
            for _ in 2..=depth {
                let prev = *v.last().unwrap();
                let id = lib.insert(uniform_children(prev));
                v.push(Child::Node(id));
            }
            uniform_block.insert(b, v);
        }
    }

    // Build one Cartesian subtree per body-cell slot. Each slot
    // occupies a `1/3 x 1/3 x 1/3` sub-box of the body's local
    // `[0, 1)³` frame, and that sub-box is SDF-carved directly.
    //
    // Face-centre slots (6 of them) additionally get the face tag
    // on their top node, so the camera pipeline recognises them as
    // "inside a planet face" at `face_depth >= 1`.
    let mut body_children = empty_children();
    for zs in 0..3 {
        for ys in 0..3 {
            for xs in 0..3 {
                let slot = slot_index(xs, ys, zs);
                let sub_min: Vec3 = [xs as f32 / 3.0, ys as f32 / 3.0, zs as f32 / 3.0];
                let sub_max: Vec3 = [
                    (xs + 1) as f32 / 3.0,
                    (ys + 1) as f32 / 3.0,
                    (zs + 1) as f32 / 3.0,
                ];
                let child = build_face_subtree(
                    lib,
                    sub_min,
                    sub_max,
                    depth,
                    sdf_budget,
                    sdf,
                    &uniform_empty,
                    &uniform_block,
                );
                let face_idx = FACE_SLOTS.iter().position(|&s| s == slot);
                body_children[slot] = match face_idx {
                    Some(fi) => tag_with_face(lib, child, Face::from_index(fi as u8)),
                    None => child,
                };
            }
        }
    }

    lib.insert_with_kind(body_children, NodeKind::CubedSphereBody { inner_r, outer_r })
}

/// Recursive builder for a Cartesian-sub-box SDF-carved subtree.
/// Every recursion level samples the SDF at the sub-box's centre
/// in the body's local `[0, 1)³` frame; cells entirely outside the
/// planet SDF collapse to empty, cells entirely inside collapse to
/// a uniform block subtree, and straddlers recurse until
/// `sdf_budget` is exhausted.
///
/// Named `build_face_subtree` because every slot the body visits —
/// face-centre or otherwise — feeds this routine to build its
/// local Cartesian subtree; face-centre slots additionally get
/// re-tagged by [`tag_with_face`] after this returns.
#[allow(clippy::too_many_arguments)]
fn build_face_subtree(
    lib: &mut NodeLibrary,
    sub_min: Vec3,
    sub_max: Vec3,
    depth: u32,
    sdf_budget: u32,
    sdf: &Planet,
    uniform_empty: &[NodeId],
    uniform_block: &std::collections::HashMap<u16, Vec<Child>>,
) -> Child {
    let center: Vec3 = [
        0.5 * (sub_min[0] + sub_max[0]),
        0.5 * (sub_min[1] + sub_max[1]),
        0.5 * (sub_min[2] + sub_max[2]),
    ];
    let d_center = sdf.distance(center);

    // Cell bounding-sphere radius = half the diagonal of the sub-box.
    let hx = 0.5 * (sub_max[0] - sub_min[0]);
    let hy = 0.5 * (sub_max[1] - sub_min[1]);
    let hz = 0.5 * (sub_max[2] - sub_min[2]);
    let cell_rad = (hx * hx + hy * hy + hz * hz).sqrt();

    if d_center > cell_rad {
        if depth == 0 {
            return Child::Empty;
        }
        return Child::Node(uniform_empty[depth as usize]);
    }
    if d_center < -cell_rad {
        let b = sdf.block_at(center);
        if depth == 0 {
            return Child::Block(b);
        }
        return uniform_block
            .get(&b)
            .and_then(|v| v.get(depth as usize).copied())
            .unwrap_or_else(|| lib.build_uniform_subtree(b, depth));
    }

    if depth == 0 {
        return if d_center < 0.0 {
            Child::Block(sdf.block_at(center))
        } else {
            Child::Empty
        };
    }
    if sdf_budget == 0 {
        return if d_center < 0.0 {
            let b = sdf.block_at(center);
            uniform_block
                .get(&b)
                .and_then(|v| v.get(depth as usize).copied())
                .unwrap_or_else(|| lib.build_uniform_subtree(b, depth))
        } else {
            Child::Node(uniform_empty[depth as usize])
        };
    }

    let mut children = empty_children();
    let tx = (sub_max[0] - sub_min[0]) / 3.0;
    let ty = (sub_max[1] - sub_min[1]) / 3.0;
    let tz = (sub_max[2] - sub_min[2]) / 3.0;
    for zs in 0..3 {
        for ys in 0..3 {
            for xs in 0..3 {
                let cmin: Vec3 = [
                    sub_min[0] + xs as f32 * tx,
                    sub_min[1] + ys as f32 * ty,
                    sub_min[2] + zs as f32 * tz,
                ];
                let cmax: Vec3 = [cmin[0] + tx, cmin[1] + ty, cmin[2] + tz];
                children[slot_index(xs, ys, zs)] = build_face_subtree(
                    lib,
                    cmin,
                    cmax,
                    depth - 1,
                    sdf_budget - 1,
                    sdf,
                    uniform_empty,
                    uniform_block,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

/// Wrap a face-centre slot's subtree with `NodeKind::CubedSphereFace`
/// so the camera pipeline can detect when the render frame is inside
/// a specific face and rotate the camera basis to the face's local
/// `(u, v, r)` axes. The child slot's content is unchanged.
fn tag_with_face(lib: &mut NodeLibrary, child: Child, face: Face) -> Child {
    match child {
        Child::Node(id) => {
            let n = lib.get(id).expect("face root just inserted");
            let children = n.children;
            Child::Node(lib.insert_with_kind(children, NodeKind::CubedSphereFace { face }))
        }
        Child::Empty => Child::Node(lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereFace { face },
        )),
        Child::Block(b) => Child::Node(lib.insert_with_kind(
            uniform_children(Child::Block(b)),
            NodeKind::CubedSphereFace { face },
        )),
        // EntityRef is never produced by the SDF builder; surface
        // as Cartesian-tagged passthrough for completeness.
        Child::EntityRef(_) => child,
    }
}

// ────────────────────────────────────────── install_at_root_center

/// Build a body node and place it at `host_slots` starting from
/// `world_root`. Returns the new world-root id and the full path
/// from root to the body.
pub fn insert_into_tree(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    host_slots: &[u8],
    setup: &PlanetSetup,
) -> (NodeId, Path) {
    assert!(!host_slots.is_empty(), "host_slots must point at a child");

    let body_id = insert_spherical_body(
        lib,
        setup.inner_r,
        setup.outer_r,
        setup.depth,
        &setup.sdf,
    );

    let new_root = install_body(lib, world_root, host_slots, body_id);
    let mut body_path = Path::root();
    for &s in host_slots {
        body_path.push(s);
    }
    (new_root, body_path)
}

/// Walk down `slots`, expanding any uniform terminals on the path
/// into Node children, then install `new_node` at the leaf and
/// rebuild parents on the way up. Returns the new world-root id.
fn install_body(
    lib: &mut NodeLibrary,
    root: NodeId,
    slots: &[u8],
    new_node: NodeId,
) -> NodeId {
    fn rebuild(
        lib: &mut NodeLibrary,
        current: NodeId,
        slots: &[u8],
        level: usize,
        new_node: NodeId,
    ) -> NodeId {
        let target = slots[level] as usize;
        let node = lib.get(current).expect("install path must exist in library");
        let mut new_children = node.children;
        if level + 1 == slots.len() {
            new_children[target] = Child::Node(new_node);
        } else {
            let next_id = match node.children[target] {
                Child::Node(nid) => rebuild(lib, nid, slots, level + 1, new_node),
                Child::Empty | Child::Block(_) | Child::EntityRef(_) => {
                    let expanded = lib.insert(empty_children());
                    rebuild(lib, expanded, slots, level + 1, new_node)
                }
            };
            new_children[target] = Child::Node(next_id);
        }
        lib.insert(new_children)
    }
    rebuild(lib, root, slots, 0, new_node)
}

/// Convenience — install at slot 13 (depth-1 centre cell of the root).
pub fn install_at_root_center(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    setup: &PlanetSetup,
) -> (NodeId, Path) {
    insert_into_tree(lib, world_root, &[CORE_SLOT as u8], setup)
}

// ─────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_planet_builds_body_with_all_six_faces() {
        let mut lib = NodeLibrary::default();
        let setup = demo_planet();
        let body = insert_spherical_body(
            &mut lib,
            setup.inner_r,
            setup.outer_r,
            6,
            &setup.sdf,
        );
        let body_node = lib.get(body).unwrap();
        assert!(matches!(body_node.kind, NodeKind::CubedSphereBody { .. }));
        for face in Face::ALL {
            let slot = FACE_SLOTS[face as usize];
            match body_node.children[slot] {
                Child::Node(id) => {
                    let n = lib.get(id).unwrap();
                    assert!(
                        matches!(n.kind, NodeKind::CubedSphereFace { face: f } if f == face),
                        "slot {slot} kind {:?} != CubedSphereFace {{ face: {face:?} }}",
                        n.kind,
                    );
                }
                other => panic!("face slot {slot} not a Node, got {other:?}"),
            }
        }
        // Core slot is stone (or a uniform-stone subtree).
        match body_node.children[CORE_SLOT] {
            Child::Node(_) | Child::Block(_) => {}
            other => panic!("core slot empty / unexpected: {other:?}"),
        }
    }
}
