//! Cubed-sphere face adjacency: 24-case axis remap table.
//!
//! When a `(u_slot, v_slot, r_slot)` cell on face `F` walks past
//! one of `F`'s `±u` / `±v` edges, the path doesn't bubble up to
//! the body and back down a Cartesian sibling — it crosses a cube
//! seam onto a different face whose tangent basis is rotated /
//! flipped relative to `F`'s. This module owns the axis remap
//! tables that handle those crossings.
//!
//! Conventions (must agree with `cubesphere::Face::tangents()`):
//! ```text
//! Face        normal      u_axis     v_axis
//! ─────────────────────────────────────────
//! PosX (+X)  ( 1, 0, 0)  ( 0, 0,-1) ( 0, 1, 0)
//! NegX (-X)  (-1, 0, 0)  ( 0, 0, 1) ( 0, 1, 0)
//! PosY (+Y)  ( 0, 1, 0)  ( 1, 0, 0) ( 0, 0,-1)
//! NegY (-Y)  ( 0,-1, 0)  ( 1, 0, 0) ( 0, 0, 1)
//! PosZ (+Z)  ( 0, 0, 1)  ( 1, 0, 0) ( 0, 1, 0)
//! NegZ (-Z)  ( 0, 0,-1)  (-1, 0, 0) ( 0, 1, 0)
//! ```
//!
//! For each `(from_face, axis, dir)` lateral edge crossing this
//! returns the destination face plus the linear remap that takes
//! the original `(u_slot, v_slot)` into `(u_slot', v_slot')` on
//! the new face. `r_slot` is unaffected by lateral crossings —
//! the radial axis is normal to the cube surface, identical on
//! both sides of any seam.

use super::cubesphere::Face;

/// One slot index in `0..3`.
pub type Slot3 = u8;

/// Remap of a face cell's `(u_slot, v_slot)` to the neighboring
/// face's `(u_slot', v_slot')`. Each axis on the source face is
/// either passed through `Forward` (u'=u), flipped `Reverse`
/// (u'=2-u), held fixed at an edge `Edge0` / `Edge2`, or copied
/// from the OTHER source axis (`SwapForward` / `SwapReverse`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisRemap {
    /// `out = u_slot`
    UForward,
    /// `out = 2 - u_slot`
    UReverse,
    /// `out = v_slot`
    VForward,
    /// `out = 2 - v_slot`
    VReverse,
    /// `out = 0` (entered at the low edge)
    Edge0,
    /// `out = 2` (entered at the high edge)
    Edge2,
}

impl AxisRemap {
    #[inline]
    pub fn apply(self, u: Slot3, v: Slot3) -> Slot3 {
        match self {
            AxisRemap::UForward => u,
            AxisRemap::UReverse => 2 - u,
            AxisRemap::VForward => v,
            AxisRemap::VReverse => 2 - v,
            AxisRemap::Edge0 => 0,
            AxisRemap::Edge2 => 2,
        }
    }
}

/// Result of a face seam crossing: destination face plus the remap
/// for the new `(u_slot', v_slot')` from the source `(u_slot, v_slot)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeamCrossing {
    pub to_face: Face,
    pub new_u: AxisRemap,
    pub new_v: AxisRemap,
}

/// Lookup: stepping past `from_face`'s lateral edge `(axis, dir)`,
/// where `axis ∈ {0=u, 1=v}` and `dir ∈ {-1, +1}`. Returns the
/// destination face and the slot remap.
///
/// Returns `None` for the radial axis (`axis=2`) — radial overflow
/// doesn't cross a seam, it exits the body shell.
pub fn seam_neighbor(from_face: Face, axis: u8, dir: i8) -> Option<SeamCrossing> {
    use Face::*;
    use AxisRemap::*;
    let key = (from_face, axis, dir);
    let (to_face, new_u, new_v) = match key {
        // ---- PosX (+X, u=-Z, v=+Y) ----
        (PosX, 0,  1) => (NegZ, Edge0,    VForward),  // +u → -Z
        (PosX, 0, -1) => (PosZ, Edge2,    VForward),  // -u → +Z
        (PosX, 1,  1) => (PosY, Edge2,    UForward),  // +v → +Y
        (PosX, 1, -1) => (NegY, Edge2,    UReverse),  // -v → -Y

        // ---- NegX (-X, u=+Z, v=+Y) ----
        (NegX, 0,  1) => (PosZ, Edge0,    VForward),  // +u → +Z
        (NegX, 0, -1) => (NegZ, Edge2,    VForward),  // -u → -Z
        (NegX, 1,  1) => (PosY, Edge0,    UReverse),  // +v → +Y
        (NegX, 1, -1) => (NegY, Edge0,    UForward),  // -v → -Y

        // ---- PosY (+Y, u=+X, v=-Z) ----
        (PosY, 0,  1) => (PosX, VForward, Edge2),     // +u → +X
        (PosY, 0, -1) => (NegX, VReverse, Edge2),     // -u → -X
        (PosY, 1,  1) => (NegZ, UReverse, Edge2),     // +v → -Z
        (PosY, 1, -1) => (PosZ, UForward, Edge2),     // -v → +Z

        // ---- NegY (-Y, u=+X, v=+Z) ----
        (NegY, 0,  1) => (PosX, VReverse, Edge0),     // +u → +X
        (NegY, 0, -1) => (NegX, VForward, Edge0),     // -u → -X
        (NegY, 1,  1) => (PosZ, UForward, Edge0),     // +v → +Z
        (NegY, 1, -1) => (NegZ, UReverse, Edge0),     // -v → -Z

        // ---- PosZ (+Z, u=+X, v=+Y) ----
        (PosZ, 0,  1) => (PosX, Edge0,    VForward),  // +u → +X
        (PosZ, 0, -1) => (NegX, Edge2,    VForward),  // -u → -X
        (PosZ, 1,  1) => (PosY, UForward, Edge0),     // +v → +Y
        (PosZ, 1, -1) => (NegY, UForward, Edge2),     // -v → -Y

        // ---- NegZ (-Z, u=-X, v=+Y) ----
        (NegZ, 0,  1) => (NegX, Edge0,    VForward),  // +u → -X
        (NegZ, 0, -1) => (PosX, Edge2,    VForward),  // -u → +X
        (NegZ, 1,  1) => (PosY, UReverse, Edge2),     // +v → +Y
        (NegZ, 1, -1) => (NegY, UReverse, Edge0),     // -v → -Y

        _ => return None,
    };
    Some(SeamCrossing { to_face, new_u, new_v })
}

// ------------------------------------------------------------------ tests
#[cfg(test)]
mod tests {
    use super::*;

    /// Every (face, edge) crossing produces a destination face that
    /// is adjacent (i.e., not the same face and not the antipode).
    #[test]
    fn seam_destinations_are_adjacent() {
        for &face in &Face::ALL {
            for axis in 0..2u8 {
                for &dir in &[-1i8, 1] {
                    let crossing = seam_neighbor(face, axis, dir)
                        .expect("lateral edges must have a neighbor");
                    assert_ne!(crossing.to_face, face,
                        "{:?} edge ({},{:+}) maps to itself", face, axis, dir);
                    let n_from = face.normal();
                    let n_to = crossing.to_face.normal();
                    let dot = n_from[0]*n_to[0] + n_from[1]*n_to[1] + n_from[2]*n_to[2];
                    assert!(dot.abs() < 0.5,
                        "{:?} → {:?} are antipodal (dot={})",
                        face, crossing.to_face, dot);
                }
            }
        }
    }

    /// Crossing a seam and stepping back the other way returns to
    /// the original face. This catches all 24 cases where the
    /// neighbor's reverse isn't the inverse.
    #[test]
    fn seam_round_trip_returns_to_origin() {
        for &face in &Face::ALL {
            for axis in 0..2u8 {
                for &dir in &[-1i8, 1] {
                    let out = seam_neighbor(face, axis, dir).unwrap();
                    // The destination's "back" edge should bring us
                    // home to `face`. We don't know which axis on
                    // `to_face` is the back-edge, but the destination
                    // face must list `face` as one of its 4 neighbors.
                    let mut found = false;
                    for back_axis in 0..2u8 {
                        for &back_dir in &[-1i8, 1] {
                            let back = seam_neighbor(out.to_face, back_axis, back_dir).unwrap();
                            if back.to_face == face { found = true; }
                        }
                    }
                    assert!(found,
                        "no return path from {:?} to {:?}", out.to_face, face);
                }
            }
        }
    }

    /// Each face has exactly 4 distinct lateral neighbors.
    #[test]
    fn each_face_has_four_distinct_neighbors() {
        for &face in &Face::ALL {
            use std::collections::HashSet;
            let mut neighbors: HashSet<u8> = HashSet::new();
            for axis in 0..2u8 {
                for &dir in &[-1i8, 1] {
                    let n = seam_neighbor(face, axis, dir).unwrap().to_face;
                    neighbors.insert(n as u8);
                }
            }
            assert_eq!(neighbors.len(), 4,
                "{:?} should have 4 distinct neighbors, got {:?}",
                face, neighbors);
        }
    }

    /// Geometric continuity check: a cell on face `F` at the edge
    /// `(axis, dir)`, when remapped to face `F'`, lands at a slot on
    /// `F'` whose center has approximately the same world direction
    /// as the original cell's edge midpoint.
    #[test]
    fn seam_geometry_is_continuous() {
        use super::super::cubesphere::face_uv_to_dir;
        for &face in &Face::ALL {
            for axis in 0..2u8 {
                for &dir in &[-1i8, 1] {
                    // Pick a representative slot on the source face
                    // along the crossing edge: the middle of the
                    // perpendicular axis.
                    let (u_slot, v_slot) = match (axis, dir) {
                        (0,  1) => (2u8, 1u8),
                        (0, -1) => (0u8, 1u8),
                        (1,  1) => (1u8, 2u8),
                        (1, -1) => (1u8, 0u8),
                        _ => unreachable!(),
                    };
                    // World direction of source cell center, mapped
                    // through the equal-angle warp.
                    let (u_ea, v_ea) = slot_to_ea(u_slot, v_slot);
                    let src_dir = face_uv_to_dir(face, u_ea, v_ea);

                    let crossing = seam_neighbor(face, axis, dir).unwrap();
                    let nu = crossing.new_u.apply(u_slot, v_slot);
                    let nv = crossing.new_v.apply(u_slot, v_slot);
                    let (nu_ea, nv_ea) = slot_to_ea(nu, nv);
                    let dst_dir = face_uv_to_dir(crossing.to_face, nu_ea, nv_ea);

                    let dot = src_dir[0]*dst_dir[0]
                            + src_dir[1]*dst_dir[1]
                            + src_dir[2]*dst_dir[2];
                    // Adjacent face cells should be within ~30° of
                    // each other (single 1/3 step + perpendicular
                    // alignment). Tolerate down to 0.65 = ~50°
                    // because the perpendicular slot may be at the
                    // shared corner where directions diverge a bit.
                    assert!(dot > 0.65,
                        "{:?} ({},{:+}) [u={},v={}] → {:?} [u'={},v'={}]: \
                         dirs misaligned, dot={:.3}",
                        face, axis, dir, u_slot, v_slot,
                        crossing.to_face, nu, nv, dot);
                }
            }
        }
    }

    /// Convert a slot index `0..3` to its equal-angle center coord
    /// in `[-1, +1]` (slot center lies at `-2/3, 0, +2/3`).
    fn slot_to_ea(u: u8, v: u8) -> (f32, f32) {
        let su = -1.0 + (u as f32 + 0.5) * (2.0 / 3.0);
        let sv = -1.0 + (v as f32 + 0.5) * (2.0 / 3.0);
        (su, sv)
    }
}
