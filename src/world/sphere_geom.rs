//! Sphere geometry helpers — derivation of (lat, lon, r) bounds for
//! a sub-frame inside a `WrappedPlane` subtree.
//!
//! The `WrappedPlane` node represents a planet body whose local
//! `[0, 2)³` frame contains an implied sphere of radius
//! `body_size / (2π)` centered at `(1.0, 1.0, 1.0)`. The slab data
//! fills the sphere as a `(lon, lat, r)` cell grid; sub-cells below
//! the slab continue the same partition recursively (each Cartesian
//! Node split is interpreted as a 2-way split of the parent's
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

use crate::world::anchor::{Path, WorldPos};
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};

/// Shell thickness as a fraction of the sphere radius. Mirrors the
/// shader's `let shell_thickness = r_sphere * 0.25;`.
pub const SHELL_THICKNESS_FRAC: f32 = 0.25;

/// Default polar-ban latitude (radians) — mirrors `--planet-render-sphere`'s
/// `planet_render.y` default. Sub-frame range computations use this when
/// the renderer's runtime value isn't available.
pub const DEFAULT_SPHERE_LAT_MAX: f32 = 1.26;

/// Sphere sub-frame range in `WrappedPlane`-local sphere coords.
/// Lat / lon in radians; r in local-frame units (where the
/// `WrappedPlane`'s own frame is `[0, body_size)³`, body_size = 2.0
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
/// `WORLD_SIZE = 2.0` in the standard architecture — the WP node
/// occupies its parent's full slot, and its own children grid spans
/// `[0, 2)³`). `lat_max` is the polar-ban latitude (passed from the
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
        let half_lon = (lon_hi - lon_lo) / 2.0;
        let half_lat = (lat_hi - lat_lo) / 2.0;
        let half_r = (r_hi - r_lo) / 2.0;
        let new_lon_lo = lon_lo + sx as f32 * half_lon;
        let new_lat_lo = lat_lo + sz as f32 * half_lat;
        let new_r_lo = r_lo + sy as f32 * half_r;
        lon_lo = new_lon_lo;
        lon_hi = new_lon_lo + half_lon;
        lat_lo = new_lat_lo;
        lat_hi = new_lat_lo + half_lat;
        r_lo = new_r_lo;
        r_hi = new_r_lo + half_r;
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

/// Camera position + basis projected into a sphere sub-frame's
/// local rotated+translated coordinate system. The sub-frame's
/// origin is at the sub-frame center (= sphere body center
/// translated by `r_c · radial`), with axes:
///   * +x = lon-tangent at sub-frame center
///   * +y = lat-tangent at sub-frame center
///   * +z = radial direction at sub-frame center
///
/// All three basis vectors are unit-length in the WP's local frame
/// (= unit-length in world up to the WP's scale factor — frames are
/// pure scale+translation, no rotation, so direction vectors are
/// preserved). Camera position magnitude is bounded by
/// `camera_distance_to_sub_frame_center` rather than by the sphere
/// body's WP-local extent — small for cameras anchored INSIDE the
/// sub-frame, which is the case the precision discipline exists to
/// solve.
pub struct SphereSubFrameLocal {
    pub origin: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    /// Sub-frame center radial distance from the sphere body center.
    /// Useful for the shader to compute the sphere's position in
    /// sub-frame coords (= `(0, 0, -r_c)`).
    pub r_c: f32,
}

/// Project a camera (position + basis) into a sphere sub-frame's
/// local coords.
///
/// `wp_path` is the path from the world root to the `WrappedPlane`
/// node — typically `range.wp_path_depth` slots from the active
/// frame's `render_path`. `range` is the sub-frame's geometry.
/// `body_size` is the WP's local frame size (2.0 in the standard
/// architecture).
pub fn camera_in_sphere_subframe(
    cam_pos: &WorldPos,
    cam_forward: [f32; 3],
    cam_right: [f32; 3],
    cam_up: [f32; 3],
    wp_path: &Path,
    range: &SphereSubFrameRange,
    body_size: f32,
) -> SphereSubFrameLocal {
    let lon_c = range.lon_center();
    let lat_c = range.lat_center();
    let r_c = range.r_center();
    let cl = lat_c.cos();
    let sl = lat_c.sin();
    let co = lon_c.cos();
    let so = lon_c.sin();
    let radial = [cl * co, sl, cl * so];
    let lon_tan = [-so, 0.0, co];
    let lat_tan = [-sl * co, cl, -sl * so];

    // Sub-frame center in WP-local coords.
    let half = body_size * 0.5;
    let sub_center_wp = [
        half + r_c * radial[0],
        half + r_c * radial[1],
        half + r_c * radial[2],
    ];

    // Camera in WP local.
    let cam_wp = cam_pos.in_frame(wp_path);
    let cam_offset = [
        cam_wp[0] - sub_center_wp[0],
        cam_wp[1] - sub_center_wp[1],
        cam_wp[2] - sub_center_wp[2],
    ];

    let dot3 = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    SphereSubFrameLocal {
        origin: [
            dot3(cam_offset, lon_tan),
            dot3(cam_offset, lat_tan),
            dot3(cam_offset, radial),
        ],
        forward: [
            dot3(cam_forward, lon_tan),
            dot3(cam_forward, lat_tan),
            dot3(cam_forward, radial),
        ],
        right: [
            dot3(cam_right, lon_tan),
            dot3(cam_right, lat_tan),
            dot3(cam_right, radial),
        ],
        up: [
            dot3(cam_up, lon_tan),
            dot3(cam_up, lat_tan),
            dot3(cam_up, radial),
        ],
        r_c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::bootstrap::wrapped_planet_world;
    use crate::world::tree::{empty_children, slot_index, NodeLibrary};

    const BODY_SIZE: f32 = 2.0;
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
        let world = wrapped_planet_world(2, [8, 2, 4], 3, 1);
        // Path stopping AT the WP node (after 2 embedding slots).
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8);
        path.push(slot_index(1, 1, 1) as u8);
        let r = subframe_range(&lib_ref(&world), world.root, &path, BODY_SIZE, LAT_MAX)
            .expect("path lands on WP");
        assert_eq!(r.wp_dims, [8, 2, 4]);
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
    fn refines_range_one_half_per_slot_below_wp() {
        let world = wrapped_planet_world(2, [8, 2, 4], 3, 3);
        // Path: 2 embedding slots → WP, then slot 7 (= (1,1,1))
        // inside the WP.
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8);
        path.push(slot_index(1, 1, 1) as u8);
        path.push(slot_index(1, 1, 1) as u8); // middle of WP
        let r = subframe_range(&lib_ref(&world), world.root, &path, BODY_SIZE, LAT_MAX)
            .expect("path enters WP");
        // Center sub-cell of WP: each axis is the upper 1/2.
        let half_lon = (2.0 * PI) / 2.0;
        let half_lat = (2.0 * LAT_MAX) / 2.0;
        let r_sphere = BODY_SIZE / (2.0 * PI);
        let shell = r_sphere * SHELL_THICKNESS_FRAC;
        let half_r = shell / 2.0;
        assert!((r.lon_lo - (-PI + half_lon)).abs() < 1e-6,
                "expected lon_lo = -π + π, got {}", r.lon_lo);
        assert!((r.lon_hi - (-PI + 2.0 * half_lon)).abs() < 1e-6);
        assert!((r.lat_lo - (-LAT_MAX + half_lat)).abs() < 1e-6);
        assert!((r.lat_hi - (-LAT_MAX + 2.0 * half_lat)).abs() < 1e-6);
        assert!((r.r_lo - ((r_sphere - shell) + half_r)).abs() < 1e-6);
        assert!((r.r_hi - ((r_sphere - shell) + 2.0 * half_r)).abs() < 1e-6);
    }

    #[test]
    fn slot_axis_mapping_matches_sphere_convention() {
        // Verify slot.x → lon, slot.y → r, slot.z → lat.
        let world = wrapped_planet_world(2, [8, 2, 4], 3, 3);
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8);
        path.push(slot_index(1, 1, 1) as u8);
        // Slot (1, 0, 0) — upper-half lon, bottom r, bottom lat.
        path.push(slot_index(1, 0, 0) as u8);
        let r = subframe_range(&lib_ref(&world), world.root, &path, BODY_SIZE, LAT_MAX)
            .expect("path enters WP");
        let half_lon = (2.0 * PI) / 2.0;
        let half_lat = (2.0 * LAT_MAX) / 2.0;
        let r_sphere = BODY_SIZE / (2.0 * PI);
        let shell = r_sphere * SHELL_THICKNESS_FRAC;
        let half_r = shell / 2.0;
        // sx = 1 → upper lon half.
        assert!((r.lon_lo - (-PI + 1.0 * half_lon)).abs() < 1e-6);
        // sy = 0 → bottom r half.
        assert!((r.r_lo - (r_sphere - shell)).abs() < 1e-6);
        assert!((r.r_hi - ((r_sphere - shell) + half_r)).abs() < 1e-6);
        // sz = 0 → bottom lat half.
        assert!((r.lat_lo - (-LAT_MAX)).abs() < 1e-6);
        assert!((r.lat_hi - (-LAT_MAX + half_lat)).abs() < 1e-6);
    }

    #[test]
    fn deep_path_extents_shrink_by_two_to_the_n() {
        let world = wrapped_planet_world(2, [8, 2, 4], 3, 6);
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8);
        path.push(slot_index(1, 1, 1) as u8);
        // 5 sub-cells below the WP, all centered.
        for _ in 0..5 {
            path.push(slot_index(1, 1, 1) as u8);
        }
        let r = subframe_range(&lib_ref(&world), world.root, &path, BODY_SIZE, LAT_MAX)
            .expect("path enters WP");
        let expected_lon_extent = (2.0 * PI) / 2.0_f32.powi(5);
        let expected_lat_extent = (2.0 * LAT_MAX) / 2.0_f32.powi(5);
        let r_sphere = BODY_SIZE / (2.0 * PI);
        let shell = r_sphere * SHELL_THICKNESS_FRAC;
        let expected_r_extent = shell / 2.0_f32.powi(5);
        assert!((r.lon_extent() - expected_lon_extent).abs() < 1e-6,
                "lon extent at depth 5: expected {expected_lon_extent}, got {}", r.lon_extent());
        assert!((r.lat_extent() - expected_lat_extent).abs() < 1e-6);
        assert!((r.r_extent() - expected_r_extent).abs() < 1e-6);
    }

    fn lib_ref(world: &crate::world::state::WorldState) -> &NodeLibrary {
        &world.library
    }

    // -------------------- camera_in_sphere_subframe --------------------

    /// Camera at the sub-frame center → origin = (0, 0, 0).
    /// Layer-agnostic precision relies on this: the sub-frame center
    /// is the origin of the local rotated+translated frame.
    #[test]
    fn camera_at_subframe_center_projects_to_origin() {
        let world = wrapped_planet_world(2, [8, 2, 4], 3, 3);
        let mut wp_path = Path::root();
        wp_path.push(slot_index(1, 1, 1) as u8);
        wp_path.push(slot_index(1, 1, 1) as u8);
        // Sub-frame at WP center cell (1, 1, 1) → range center is
        // sphere center direction at (lat=0, lon=0, r=middle).
        let mut anchor = wp_path;
        anchor.push(slot_index(1, 1, 1) as u8);
        let range = subframe_range(&lib_ref(&world), world.root, &anchor, BODY_SIZE, LAT_MAX)
            .unwrap();
        // Camera position: sub-frame center in WP local.
        let lon_c = range.lon_center();
        let lat_c = range.lat_center();
        let r_c = range.r_center();
        let radial = [
            lat_c.cos() * lon_c.cos(),
            lat_c.sin(),
            lat_c.cos() * lon_c.sin(),
        ];
        let half = BODY_SIZE * 0.5;
        let center_wp = [
            half + r_c * radial[0],
            half + r_c * radial[1],
            half + r_c * radial[2],
        ];
        let cam_pos = WorldPos::from_frame_local(&wp_path, center_wp, wp_path.depth());
        let local = camera_in_sphere_subframe(
            &cam_pos,
            [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            &wp_path, &range, BODY_SIZE,
        );
        assert!(local.origin[0].abs() < 1e-3, "origin.x ≈ 0, got {}", local.origin[0]);
        assert!(local.origin[1].abs() < 1e-3, "origin.y ≈ 0, got {}", local.origin[1]);
        assert!(local.origin[2].abs() < 1e-3, "origin.z ≈ 0, got {}", local.origin[2]);
    }

    /// Camera offset from sub-frame center along the lon-tangent
    /// direction in WP-local → projects to a pure +x in sub-frame
    /// local (other axes ≈ 0).
    #[test]
    fn camera_offset_along_lon_tangent_projects_to_x_axis() {
        let world = wrapped_planet_world(2, [8, 2, 4], 3, 3);
        let mut wp_path = Path::root();
        wp_path.push(slot_index(1, 1, 1) as u8);
        wp_path.push(slot_index(1, 1, 1) as u8);
        // Use middle sub-cell (lat = 0, lon ≈ 0) so the basis lines
        // up nicely with WP-local axes.
        let mut anchor = wp_path;
        anchor.push(slot_index(1, 1, 1) as u8);
        let range = subframe_range(&lib_ref(&world), world.root, &anchor, BODY_SIZE, LAT_MAX)
            .unwrap();
        let lon_c = range.lon_center();
        let lat_c = range.lat_center();
        let r_c = range.r_center();
        let radial = [
            lat_c.cos() * lon_c.cos(),
            lat_c.sin(),
            lat_c.cos() * lon_c.sin(),
        ];
        let lon_tan = [-lon_c.sin(), 0.0, lon_c.cos()];
        let half = BODY_SIZE * 0.5;
        let center_wp = [
            half + r_c * radial[0],
            half + r_c * radial[1],
            half + r_c * radial[2],
        ];
        // Camera offset by 0.01 (small) in the lon-tangent direction.
        let offset = 0.01_f32;
        let cam_wp = [
            center_wp[0] + offset * lon_tan[0],
            center_wp[1] + offset * lon_tan[1],
            center_wp[2] + offset * lon_tan[2],
        ];
        let cam_pos = WorldPos::from_frame_local(&wp_path, cam_wp, wp_path.depth());
        let local = camera_in_sphere_subframe(
            &cam_pos,
            [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            &wp_path, &range, BODY_SIZE,
        );
        assert!((local.origin[0] - offset).abs() < 1e-4,
                "x ≈ offset, got {}", local.origin[0]);
        assert!(local.origin[1].abs() < 1e-4,
                "y ≈ 0, got {}", local.origin[1]);
        assert!(local.origin[2].abs() < 1e-4,
                "z ≈ 0, got {}", local.origin[2]);
    }

    /// Camera basis is preserved as unit-length under projection
    /// (the basis vectors are pure rotations of world-axis-aligned
    /// inputs).
    #[test]
    fn camera_basis_remains_unit_length() {
        let world = wrapped_planet_world(2, [8, 2, 4], 3, 3);
        let mut wp_path = Path::root();
        wp_path.push(slot_index(1, 1, 1) as u8);
        wp_path.push(slot_index(1, 1, 1) as u8);
        let mut anchor = wp_path;
        anchor.push(slot_index(0, 1, 1) as u8); // off-center sub-frame
        let range = subframe_range(&lib_ref(&world), world.root, &anchor, BODY_SIZE, LAT_MAX)
            .unwrap();
        let cam_pos = WorldPos::from_frame_local(&wp_path, [1.0, 1.0, 1.0], wp_path.depth());
        let local = camera_in_sphere_subframe(
            &cam_pos,
            [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
            &wp_path, &range, BODY_SIZE,
        );
        let mag = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((mag(local.forward) - 1.0).abs() < 1e-4);
        assert!((mag(local.right) - 1.0).abs() < 1e-4);
        assert!((mag(local.up) - 1.0).abs() < 1e-4);
    }
}
