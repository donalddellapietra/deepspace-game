//! Self-similar voxel fractals adapted for a base-3 recursive tree.
//!
//! Each fractal here is expressed as "which of the 27 sub-cells of a
//! 3×3×3 node are filled at every level of recursion, and with what
//! block". Since the tree is content-addressed (see `super::tree`),
//! every level dedups to a single library node — a depth-`d` fractal
//! is O(d) storage, no matter how wide its silhouette becomes.
//!
//! # Coloring — bringing PySpace's orbit traps into a voxel world
//!
//! The PySpace reference (`external/PySpace`) colors fractals with
//! *orbit traps*: a running min/max/sum of the folded point's position
//! tracked through the iteration, then mapped to RGB at the hit. The
//! visual effect is that different *structural roles* (corners, edges,
//! face centers) get different hues — the geometry's self-similarity
//! shows through the color.
//!
//! We can't fold at render time, but we can bake the equivalent into
//! the voxel palette: assign **different blocks to different slot
//! roles**. A Menger corner cell gets one color, an edge cell gets
//! another. Each level of self-similarity inherits the same role
//! mapping, so the colored structure repeats at every zoom.
//!
//! Per-fractal palette choices are made by the `bootstrap_*_world`
//! functions and match the aesthetic of PySpace's original scenes
//! (cream Sierpinski, cool-blue Menger, prismatic dust, ...).
//!
//! # Trinary adaptation of PySpace's fractal zoo
//!
//! PySpace targets a ray-marched SDF renderer with `FoldScale(2)`
//! (binary self-similarity). Our tree subdivides at scale 3 — some
//! fractals are natural fits (Menger is *defined* on a ternary
//! subdivision), others need recasting into "pick a subset of 27
//! cells per level":
//!
//! | PySpace fractal          | Trinary adaptation                  |
//! |--------------------------|-------------------------------------|
//! | `menger` / `mausoleum`   | [`menger`] — 20/27 cells (native)   |
//! | `sierpinski_tetrahedron` | [`sierpinski_tet`] — 4 cube corners |
//! | —                        | [`cantor_dust`] — 8 cube corners    |
//! | (inverse of Menger)      | [`jerusalem_cross`] — 7 axis cells  |
//! | Sierpinski pyramid       | [`sierpinski_pyramid`] — 5 cells    |
//!
//! SDF-only PySpace scenes (mandelbox, tree_planet, snow_stadium,
//! butterweed_hills) are not ported — they depend on a continuous
//! distance estimator and have no faithful discrete-subdivision
//! representation.

pub mod cantor_dust;
pub mod edge_scaffold;
pub mod hollow_cube;
pub mod jerusalem_cross;
pub mod mausoleum;
pub mod menger;
pub mod sierpinski_pyramid;
pub mod sierpinski_tet;

use super::tree::{empty_children, slot_index, Child, NodeId, NodeLibrary};

/// One filled sub-cell in a fractal's 3×3×3 pattern.
///
/// `(x, y, z)` are the slot coordinates in `0..3`. `block` is the
/// palette index written at the deepest level; at all higher levels
/// the slot holds `Child::Node(<next level>)` so colored cells only
/// appear at the leaves (which is what makes the fractal "look" like
/// its palette).
pub(super) type Slot = (u8, u8, u8, u16);

/// Build a self-similar fractal of `depth` levels.
///
/// At every level, the cells listed in `slots` point to the next
/// level's node and all other cells are empty. At the deepest level
/// the slot's `block` index becomes a `Child::Block` leaf. Content-
/// addressed dedup compresses the whole tree to `depth` unique nodes.
pub(super) fn self_similar_fractal(
    lib: &mut NodeLibrary,
    depth: u8,
    slots: &[Slot],
) -> NodeId {
    assert!(depth >= 1, "fractal depth must be >= 1");
    for &(x, y, z, _) in slots {
        debug_assert!(x < 3 && y < 3 && z < 3, "slot out of range: ({x},{y},{z})");
    }

    // Deepest level: each slot is a Block leaf coloured by its role.
    let mut children = empty_children();
    for &(x, y, z, b) in slots {
        children[slot_index(x as usize, y as usize, z as usize)] = Child::Block(b);
    }
    let mut current = lib.insert(children);

    // Higher levels: each slot points to the prior level. Dedup
    // ensures only one new library entry per level.
    for _ in 1..depth {
        let mut children = empty_children();
        for &(x, y, z, _) in slots {
            children[slot_index(x as usize, y as usize, z as usize)] = Child::Node(current);
        }
        current = lib.insert(children);
    }
    current
}
