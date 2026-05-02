//! Sphere geometry helpers — derivation of (lat, lon, r) bounds for
//! a sub-frame inside a `WrappedPlane` subtree.
//!
//! The `WrappedPlane` node represents a planet body whose local
//! `[0, 3)³` frame contains an implied sphere of radius
//! `body_size / (2π)` centered at `(1.5, 1.5, 1.5)`. The slab data
//! fills the sphere as a `(lon, lat, r)` cell grid; sub-cells below
//! the slab continue the same partition recursively (each Cartesian
//! Node split is interpreted as a 3-way split of the parent's
//! `(lon, lat, r)` range, with axis convention:
//!
//!   slot.x → lon   slot.y → r   slot.z → lat
//!
//! For any path under a `WrappedPlane` root, the `(lat, lon, r)`
//! range is fully determined by the sequence of slot picks. This
//! module computes that range; sphere sub-frame rendering uses it
//! to scope the DDA so the math operates on bounded magnitudes —
//! layer-agnostic precision.
//!
//! Step 1 of the sphere sub-frame architecture (see design sketch).
//! Pure function over `(library, world_root, path)` — no `App`
//! state — so it's directly unit-testable.

use std::f32::consts::PI;

use crate::world::anchor::Path;
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};

/// Shell thickness as a fraction of the sphere radius. Mirrors the
/// shader's `let shell_thickness = r_sphere * 0.25;`.
pub const SHELL_THICKNESS_FRAC: f32 = 0.25;

/// Sphere sub-frame range in `WrappedPlane`-local sphere coords.
/// Lat / lon in radians; r in local-frame units (where the
/// `WrappedPlane`'s own frame is `[0, body_size)³`, body_size = 3.0
/// in the standard architecture).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSubFrameRange {
    pub lat_lo: f32,
    pub lat_hi: f32,
    pub lon_lo: f32,
    pub lon_hi: f32,
    pub r_lo: f32,
    pub r_hi: f32,
    /// `WrappedPlane` ancestor's slab dims (cells per axis).
    pub wp_dims: [u32; 3],
    /// `WrappedPlane` ancestor's slab depth (tree levels for slab
    /// data).
    pub wp_slab_depth: u8,
    /// Position in the input path where the `WrappedPlane` node was
    /// reached (= number of slots from the world root to the WP).
    /// Slot picks past this index refine the sphere range.
    pub wp_path_depth: u8,
}

impl SphereSubFrameRange {
    pub fn lon_extent(&self) -> f32 {
        self.lon_hi - self.lon_lo
    }
    pub fn lat_extent(&self) -> f32 {
        self.lat_hi - self.lat_lo
    }
    pub fn r_extent(&self) -> f32 {
        self.r_hi - self.r_lo
    }
    pub fn lon_center(&self) -> f32 {
        (self.lon_lo + self.lon_hi) * 0.5
    }
    pub fn lat_center(&self) -> f32 {
        (self.lat_lo + self.lat_hi) * 0.5
    }
    pub fn r_center(&self) -> f32 {
        (self.r_lo + self.r_hi) * 0.5
    }
}

/// Compute the sphere sub-frame range for a path inside a
/// `WrappedPlane` subtree. Returns `None` if the path doesn't enter
/// any `WrappedPlane` node along its descent (= path is purely in
/// Cartesian regions).
///
/// `body_size` is the `WrappedPlane`'s local frame size (always
/// `WORLD_SIZE = 3.0` in the standard architecture — the WP node
/// occupies its parent's full slot, and its own children grid spans
/// `[0, 3)³`). `lat_max` is the polar-ban latitude (passed from the
/// renderer's configuration; default `1.26 ≈ 72°`).
///
/// The returned range can extend BEYOND the materialized tree —
/// e.g., if the path stops at a `Block` child, the range computed
/// up to that point is returned, and slots beyond that point in the
/// path are ignored. This mirrors how `compute_render_frame`
/// truncates at terminal children.
pub fn subframe_range(
    library: &NodeLibrary,
    world_root: NodeId,
    path: &Path,
    body_size: f32,
    lat_max: f32,
) -> Option<SphereSubFrameRange> {
    let r_sphere = body_size / (2.0 * PI);
    let shell_thickness = r_sphere * SHELL_THICKNESS_FRAC;
    let r_outer = r_sphere;
    let r_inner = r_sphere - shell_thickness;

    // Phase 1: walk the library along `path` to LOCATE the WP node.
    // We only need the library to find which path index is the WP;
    // beyond that, range refinement is a pure function of slot
    // arithmetic and doesn't require the tree to be materialized
    // (uniform-flatten anchors, pre-pack uniform Nodes, etc. all
    // work the same).
    let mut wp_dims: Option<[u32; 3]> = None;
    let mut wp_slab_depth: u8 = 0;
    let mut wp_path_depth: u8 = 0;

    if let Some(root_node) = library.get(world_root) {
        if let NodeKind::WrappedPlane { dims, slab_depth } = root_node.kind {
            wp_dims = Some(dims);
            wp_slab_depth = slab_depth;
            wp_path_depth = 0;
        }
    }
    if wp_dims.is_none() {
        let mut node = world_root;
        for (k, &slot) in path.as_slice().iter().enumerate() {
            let n = match library.get(node) {
                Some(n) => n,
                None => break,
            };
            match n.children[slot as usize] {
                Child::Node(child) => {
                    node = child;
                    if let Some(child_node) = library.get(child) {
                        if let NodeKind::WrappedPlane { dims, slab_depth } = child_node.kind {
                            wp_dims = Some(dims);
                            wp_slab_depth = slab_depth;
                            wp_path_depth = (k + 1) as u8;
                            break;
                        }
                    }
                }
                _ => return None, // path exited the tree without hitting any WP
            }
        }
    }

    let dims = wp_dims?;

    // Phase 2: refine sphere range for every slot AFTER the WP root.
    // Pure slot arithmetic; runs regardless of whether the tree has
    // a Node at each step (uniform-flattened anchors, deeper edits,
    // etc.).
    let mut lat_lo = -lat_max;
    let mut lat_hi = lat_max;
    let mut lon_lo = -PI;
    let mut lon_hi = PI;
    let mut r_lo = r_inner;
    let mut r_hi = r_outer;

    for &slot in &path.as_slice()[wp_path_depth as usize..] {
        let (sx, sy, sz) = slot_coords(slot as usize);
        let lon_step = (lon_hi - lon_lo) / 3.0;
        let lat_step = (lat_hi - lat_lo) / 3.0;
        let r_step = (r_hi - r_lo) / 3.0;
        let new_lon_lo = lon_lo + sx as f32 * lon_step;
        let new_lat_lo = lat_lo + sz as f32 * lat_step;
        let new_r_lo = r_lo + sy as f32 * r_step;
        lon_lo = new_lon_lo;
        lon_hi = new_lon_lo + lon_step;
        lat_lo = new_lat_lo;
        lat_hi = new_lat_lo + lat_step;
        r_lo = new_r_lo;
        r_hi = new_r_lo + r_step;
    }

    Some(SphereSubFrameRange {
        lat_lo,
        lat_hi,
        lon_lo,
        lon_hi,
        r_lo,
        r_hi,
        wp_dims: dims,
        wp_slab_depth,
        wp_path_depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::bootstrap::wrapped_planet_world;
    use crate::world::tree::{empty_children, slot_index, NodeLibrary};

    const BODY_SIZE: f32 = 3.0;
    const LAT_MAX: f32 = 1.26;

    fn cartesian_only_world() -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(empty_children());
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(leaf);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        (lib, root)
    }

    #[test]
    fn returns_none_for_path_with_no_wrapped_plane() {
        let (lib, root) = cartesian_only_world();
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8);
        let r = subframe_range(&lib, root, &path, BODY_SIZE, LAT_MAX);
        assert!(r.is_none(), "purely-Cartesian path must yield None");
    }

    #[test]
    fn returns_full_sphere_at_wp_root() {
        // Standard wrapped-planet world: WP at depth 2.
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 0);
        // Path stopping AT the WP node (after 2 embedding slots).
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8);
        path.push(slot_index(1, 1, 1) as u8);
        let r = subframe_range(&lib_ref(&world), world.root, &path, BODY_SIZE, LAT_MAX)
            .expect("path lands on WP");
        assert_eq!(r.wp_dims, [27, 2, 14]);
        assert_eq!(r.wp_slab_depth, 3);
        assert_eq!(r.wp_path_depth, 2);
        // Range covers the full sphere band.
        assert!((r.lon_lo - (-PI)).abs() < 1e-6);
        assert!((r.lon_hi - PI).abs() < 1e-6);
        assert!((r.lat_lo - (-LAT_MAX)).abs() < 1e-6);
        assert!((r.lat_hi - LAT_MAX).abs() < 1e-6);
        let r_sphere = BODY_SIZE / (2.0 * PI);
        let shell = r_sphere * SHELL_THICKNESS_FRAC;
        assert!((r.r_lo - (r_sphere - shell)).abs() < 1e-6);
        assert!((r.r_hi - r_sphere).abs() < 1e-6);
    }

    #[test]
    fn refines_range_one_third_per_slot_below_wp() {
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 3);
        // Path: 2 embedding slots → WP, then slot 13 (= center,
        // (1, 1, 1)) inside the WP.
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8); // 13
        path.push(slot_index(1, 1, 1) as u8); // 13 → WP
        path.push(slot_index(1, 1, 1) as u8); // 13 → middle of WP
        let r = subframe_range(&lib_ref(&world), world.root, &path, BODY_SIZE, LAT_MAX)
            .expect("path enters WP");
        // Center sub-cell of WP: each axis is the middle 1/3.
        let third_lon = (2.0 * PI) / 3.0;
        let third_lat = (2.0 * LAT_MAX) / 3.0;
        let r_sphere = BODY_SIZE / (2.0 * PI);
        let shell = r_sphere * SHELL_THICKNESS_FRAC;
        let third_r = shell / 3.0;
        assert!((r.lon_lo - (-PI + third_lon)).abs() < 1e-6,
                "expected lon_lo = -π + 2π/3, got {}", r.lon_lo);
        assert!((r.lon_hi - (-PI + 2.0 * third_lon)).abs() < 1e-6);
        assert!((r.lat_lo - (-LAT_MAX + third_lat)).abs() < 1e-6);
        assert!((r.lat_hi - (-LAT_MAX + 2.0 * third_lat)).abs() < 1e-6);
        assert!((r.r_lo - ((r_sphere - shell) + third_r)).abs() < 1e-6);
        assert!((r.r_hi - ((r_sphere - shell) + 2.0 * third_r)).abs() < 1e-6);
    }

    #[test]
    fn slot_axis_mapping_matches_sphere_convention() {
        // Verify slot.x → lon, slot.y → r, slot.z → lat.
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 3);
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8);
        path.push(slot_index(1, 1, 1) as u8);
        // Slot (2, 0, 0) — east-most lon, bottom r, south-most lat.
        path.push(slot_index(2, 0, 0) as u8);
        let r = subframe_range(&lib_ref(&world), world.root, &path, BODY_SIZE, LAT_MAX)
            .expect("path enters WP");
        let third_lon = (2.0 * PI) / 3.0;
        let third_lat = (2.0 * LAT_MAX) / 3.0;
        let r_sphere = BODY_SIZE / (2.0 * PI);
        let shell = r_sphere * SHELL_THICKNESS_FRAC;
        let third_r = shell / 3.0;
        // sx = 2 → eastern lon third.
        assert!((r.lon_lo - (-PI + 2.0 * third_lon)).abs() < 1e-6);
        // sy = 0 → bottom r third.
        assert!((r.r_lo - (r_sphere - shell)).abs() < 1e-6);
        assert!((r.r_hi - ((r_sphere - shell) + third_r)).abs() < 1e-6);
        // sz = 0 → southern lat third.
        assert!((r.lat_lo - (-LAT_MAX)).abs() < 1e-6);
        assert!((r.lat_hi - (-LAT_MAX + third_lat)).abs() < 1e-6);
    }

    #[test]
    fn deep_path_extents_shrink_by_three_to_the_n() {
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 6);
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8);
        path.push(slot_index(1, 1, 1) as u8);
        // 5 sub-cells below the WP, all centered.
        for _ in 0..5 {
            path.push(slot_index(1, 1, 1) as u8);
        }
        let r = subframe_range(&lib_ref(&world), world.root, &path, BODY_SIZE, LAT_MAX)
            .expect("path enters WP");
        let expected_lon_extent = (2.0 * PI) / 3.0_f32.powi(5);
        let expected_lat_extent = (2.0 * LAT_MAX) / 3.0_f32.powi(5);
        let r_sphere = BODY_SIZE / (2.0 * PI);
        let shell = r_sphere * SHELL_THICKNESS_FRAC;
        let expected_r_extent = shell / 3.0_f32.powi(5);
        assert!((r.lon_extent() - expected_lon_extent).abs() < 1e-6,
                "lon extent at depth 5: expected {expected_lon_extent}, got {}", r.lon_extent());
        assert!((r.lat_extent() - expected_lat_extent).abs() < 1e-6);
        assert!((r.r_extent() - expected_r_extent).abs() < 1e-6);
    }

    fn lib_ref(world: &crate::world::state::WorldState) -> &NodeLibrary {
        &world.library
    }
}
