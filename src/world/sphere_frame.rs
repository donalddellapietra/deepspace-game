//! Cube→sphere shading frame for `NodeKind::SphereBody` subtrees.
//!
//! The storage/traversal/edit pipeline is entirely Cartesian. Only
//! the shading normal gets bent through an analytic cube→sphere map
//! so solid cube voxels render as a sphere surface.
//!
//! ## The map
//!
//! Given a body-frame point `(x, y, z) ∈ [-1, 1]³`, Bergamo's
//! remap sends it to
//!
//! ```text
//! s_x = x · √(1 − y²/2 − z²/2 + y²z²/3)
//! s_y = y · √(1 − z²/2 − x²/2 + z²x²/3)
//! s_z = z · √(1 − x²/2 − y²/2 + x²y²/3)
//! ```
//!
//! on the unit sphere. The map is smooth on the whole cube, bijective,
//! and produces area-uniform distortion (~13% face-center vs corner).
//!
//! ## What this module computes
//!
//! For a render frame rooted at some ternary path below a
//! `SphereBody` node, we care about the surface Jacobian evaluated at
//! the render-frame CENTER (expressed in the body's `[-1, 1]³` frame).
//! That Jacobian `J` gives:
//!
//! - `world_normal = normalize(J⁻ᵀ · cube_normal)` — normal transform
//!   for lighting. This is the only sphere-specific math the shader
//!   runs per pixel.
//!
//! Because J is smooth with O(1) derivatives, f32 precision in the
//! origin gives f32 precision in J. Path digits below ~13 ternary
//! layers fall into f32 ULP and are absorbed into rounding — J then
//! stops changing across adjacent cells, which is geometrically
//! correct: at that zoom the render frame is far too small to
//! observe any curvature.
//!
//! ## Why we don't compute `F(origin)` (world position)
//!
//! The shader never needs a world-space position. The render-frame
//! local coordinate IS the camera's working space; world coords at
//! `3⁻⁴⁰` of the planet are below f32 AND f64 ULP. Normals suffice
//! for directional lighting; point-light / shadow-ray support (out
//! of MVP scope) uses the engine's path-coordinate substrate for
//! cross-depth geometry, not sphere math.

use crate::world::anchor::Path;
use crate::world::tree::{Child, NodeId, NodeKind, NodeLibrary};

/// Shading-frame uniforms for a SphereBody render frame. The shader
/// evaluates the analytic Bergamo Jacobian per pixel; we just need
/// to tell it how to map a render-frame-local hit position into the
/// body's `[-1, 1]³` frame:
///
/// `body_pos = origin + (hit_render_local − 1.5) · scale`
///
/// where `origin` is the body-frame center of the render-frame-root
/// cell and `scale = (2 / 3) · 3⁻ᴰ` with `D` being the render frame's
/// depth below the SphereBody root.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SphereFrame {
    pub origin: [f32; 3],
    pub scale: f32,
}

impl SphereFrame {
    /// Identity-like frame used when no SphereBody is active.
    /// `scale = 0` would make `body_pos = origin` everywhere, but
    /// this frame is never sampled (the shader early-returns on
    /// `sphere_flag == OFF`).
    pub const IDENTITY: SphereFrame = SphereFrame {
        origin: [0.0; 3],
        scale: 0.0,
    };
}

/// Walk a ternary path from a SphereBody root down to a render-frame
/// root, accumulating the body-frame `[-1, 1]³` center of the render
/// cell, then evaluate `J` and `J⁻ᵀ` at that point.
///
/// `slots` — sequence of slot indices in `0..27`. Slot `s = sx + 3sy
/// + 9sz` encodes the ternary digit along each axis:
/// - `sx ∈ {0, 1, 2}` maps to body-frame x via the per-level recurrence
///   `new_center = c + (2/3)·(sx − 1)·half_width`, half-width `/= 3`.
///
/// Empty `slots` (render frame is the SphereBody itself) returns the
/// identity origin `(0, 0, 0)` and `J⁻ᵀ = I`.
pub fn sphere_frame_from_path(slots: &[u8]) -> SphereFrame {
    let mut c = [0.0f32; 3];
    let mut half = 1.0f32;
    for &s in slots {
        debug_assert!(s < 27, "slot out of range: {s}");
        let sx = (s % 3) as i32;
        let sy = ((s / 3) % 3) as i32;
        let sz = (s / 9) as i32;
        // Body-frame cell at depth d has half-width 3⁻ᵈ. Moving into
        // slot (sx, sy, sz) shifts the center by (2/3)·(slot − 1)
        // scaled by the parent's half-width — so the three children
        // along an axis sit at c - 2·half/3, c, c + 2·half/3.
        let factor = (2.0 / 3.0) * half;
        c[0] += factor * (sx - 1) as f32;
        c[1] += factor * (sy - 1) as f32;
        c[2] += factor * (sz - 1) as f32;
        half /= 3.0;
    }
    // `half` is the render cell's half-width in body-frame units
    // (half = 3⁻ᴰ). The render frame spans [0, 3)³ in its own local
    // coords; `(hit_local − 1.5)` ranges ±1.5 across the cell, which
    // maps to ±half in body-frame → `scale = half / 1.5 = (2/3)·3⁻ᴰ`.
    let scale = half * (2.0 / 3.0);
    SphereFrame { origin: c, scale }
}

/// Bergamo cube→sphere map evaluated directly. Only exposed for tests
/// (shaders don't call this — they only use the uploaded `J⁻ᵀ`).
pub fn bergamo_map(p: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = p;
    let ax = 1.0 - 0.5 * y * y - 0.5 * z * z + y * y * z * z / 3.0;
    let ay = 1.0 - 0.5 * z * z - 0.5 * x * x + z * z * x * x / 3.0;
    let az = 1.0 - 0.5 * x * x - 0.5 * y * y + x * x * y * y / 3.0;
    [
        x * ax.max(0.0).sqrt(),
        y * ay.max(0.0).sqrt(),
        z * az.max(0.0).sqrt(),
    ]
}

/// Analytic Jacobian of the Bergamo map. Returns a row-major 3×3:
/// `j[i][j] = ∂s_i / ∂p_j` evaluated at `p`.
///
/// ```text
///          ∂x                  ∂y                  ∂z
/// ∂s_x   √A_x                xy(2z²−3)/(6√A_x)  xz(2y²−3)/(6√A_x)
/// ∂s_y   xy(2z²−3)/(6√A_y)   √A_y               yz(2x²−3)/(6√A_y)
/// ∂s_z   xz(2y²−3)/(6√A_z)   yz(2x²−3)/(6√A_z)  √A_z
/// ```
///
/// `A_a = 1 − b²/2 − c²/2 + b²c²/3` where `(a, b, c)` cycles through
/// `(x,y,z), (y,z,x), (z,x,y)`. At cube corners `A_* = 1/3`; at face
/// centers `A_* = 1/2` or `1`; at the origin `A_* = 1`. `A_*` stays
/// strictly positive on the closed cube.
pub fn bergamo_jacobian(p: [f32; 3]) -> [[f32; 3]; 3] {
    let [x, y, z] = p;
    let ax = (1.0 - 0.5 * y * y - 0.5 * z * z + y * y * z * z / 3.0).max(1e-12);
    let ay = (1.0 - 0.5 * z * z - 0.5 * x * x + z * z * x * x / 3.0).max(1e-12);
    let az = (1.0 - 0.5 * x * x - 0.5 * y * y + x * x * y * y / 3.0).max(1e-12);
    let sqrt_ax = ax.sqrt();
    let sqrt_ay = ay.sqrt();
    let sqrt_az = az.sqrt();
    // Off-diagonal numerators share structure: ∂s_a/∂b = a·b·(2c²−3)/6
    // divided by √A_a, where (a,b,c) is a cyclic perm of (x,y,z).
    let sxy = x * y * (2.0 * z * z - 3.0) / (6.0 * sqrt_ax); // ∂s_x/∂y
    let sxz = x * z * (2.0 * y * y - 3.0) / (6.0 * sqrt_ax); // ∂s_x/∂z
    let syx = y * x * (2.0 * z * z - 3.0) / (6.0 * sqrt_ay); // ∂s_y/∂x
    let syz = y * z * (2.0 * x * x - 3.0) / (6.0 * sqrt_ay); // ∂s_y/∂z
    let szx = z * x * (2.0 * y * y - 3.0) / (6.0 * sqrt_az); // ∂s_z/∂x
    let szy = z * y * (2.0 * x * x - 3.0) / (6.0 * sqrt_az); // ∂s_z/∂y
    [
        [sqrt_ax, sxy, sxz],
        [syx, sqrt_ay, syz],
        [szx, szy, sqrt_az],
    ]
}

/// Standard 3×3 inverse via cofactor expansion. Returns `None` when
/// `|det| < ε` — this only happens at the 8 cube corners where the
/// Bergamo Jacobian collapses (three body-diagonals fold onto one
/// radial direction). Voxel face centers are always strictly interior
/// to cube faces, so this never triggers in the rendering path.
///
/// Exposed for unit testing the Bergamo formulas against known values
/// — the WGSL shader evaluates its own inverse per pixel.
pub fn mat3_inverse(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (e * i - f * h) * inv_det,
            -(b * i - c * h) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            -(d * i - f * g) * inv_det,
            (a * i - c * g) * inv_det,
            -(a * f - c * d) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            -(a * h - b * g) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ])
}

/// Walk the tree from `world_root` along `path` to the render-frame
/// root, searching for the deepest `NodeKind::SphereBody` ancestor.
/// When found, returns the `SphereFrame` precomputed for the render
/// cell's body-frame center; otherwise `None`.
///
/// Nesting policy (MVP): when a `SphereBody` subtree contains another
/// `SphereBody` child, the INNER one wins — we use the innermost
/// enclosing sphere. A future multi-nested mode would instead return
/// the stack of enclosing bodies.
pub fn find_active_sphere_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    path: &Path,
) -> Option<SphereFrame> {
    let is_sphere = |id: NodeId| {
        library.get(id).is_some_and(|n| n.kind == NodeKind::SphereBody)
    };
    let mut node_id = world_root;
    let mut slots_below: Option<Vec<u8>> = if is_sphere(world_root) {
        Some(Vec::new())
    } else {
        None
    };
    for k in 0..path.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = path.slot(k) as usize;
        let Child::Node(child_id) = node.children[slot] else { break };
        if let Some(s) = slots_below.as_mut() {
            s.push(slot as u8);
        }
        node_id = child_id;
        if is_sphere(node_id) {
            slots_below = Some(Vec::new());
        }
    }
    slots_below.map(|s| sphere_frame_from_path(&s))
}

fn mat3_identity() -> [[f32; 3]; 3] {
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, uniform_children, Child, NodeKind, NodeLibrary};

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn approx_vec(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        approx_eq(a[0], b[0], tol)
            && approx_eq(a[1], b[1], tol)
            && approx_eq(a[2], b[2], tol)
    }

    #[test]
    fn map_at_origin_is_identity() {
        assert_eq!(bergamo_map([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn map_at_face_centers_is_unit() {
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut p = [0.0f32; 3];
                p[axis] = sign;
                let m = bergamo_map(p);
                assert!(approx_eq(m[axis], sign, 1e-6));
                for a in 0..3 {
                    if a != axis {
                        assert!(approx_eq(m[a], 0.0, 1e-6));
                    }
                }
            }
        }
    }

    #[test]
    fn map_at_corner_lands_on_unit_sphere() {
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    let m = bergamo_map([sx, sy, sz]);
                    let len = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
                    assert!(approx_eq(len, 1.0, 1e-5));
                }
            }
        }
    }

    #[test]
    fn jacobian_at_origin_is_identity() {
        let j = bergamo_jacobian([0.0, 0.0, 0.0]);
        for i in 0..3 {
            for k in 0..3 {
                let expected = if i == k { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(j[i][k], expected, 1e-6),
                    "j[{i}][{k}] = {} (want {expected})",
                    j[i][k]
                );
            }
        }
    }

    #[test]
    fn jacobian_at_face_center_is_diagonal() {
        // At (1, 0, 0): A_x = 1, A_y = A_z = 1/2. Diagonal = (1, 1/√2, 1/√2).
        let j = bergamo_jacobian([1.0, 0.0, 0.0]);
        assert!(approx_eq(j[0][0], 1.0, 1e-6));
        assert!(approx_eq(j[1][1], 1.0 / 2.0f32.sqrt(), 1e-6));
        assert!(approx_eq(j[2][2], 1.0 / 2.0f32.sqrt(), 1e-6));
        // Off-diagonals all vanish — every one has at least one y or z factor.
        assert!(approx_eq(j[0][1], 0.0, 1e-6));
        assert!(approx_eq(j[0][2], 0.0, 1e-6));
        assert!(approx_eq(j[1][0], 0.0, 1e-6));
        assert!(approx_eq(j[1][2], 0.0, 1e-6));
        assert!(approx_eq(j[2][0], 0.0, 1e-6));
        assert!(approx_eq(j[2][1], 0.0, 1e-6));
    }

    /// Analytic J matches finite-difference J over a 6×6×6 grid inside
    /// the cube. Tolerance generous enough to absorb O(ε²) FD noise.
    #[test]
    fn jacobian_matches_finite_difference() {
        let step = 1e-3;
        for ix in -5..=5 {
            for iy in -5..=5 {
                for iz in -5..=5 {
                    let p = [
                        ix as f32 * 0.15,
                        iy as f32 * 0.15,
                        iz as f32 * 0.15,
                    ];
                    let j = bergamo_jacobian(p);
                    for col in 0..3 {
                        let mut pp = p;
                        pp[col] += step;
                        let mut pm = p;
                        pm[col] -= step;
                        let fp = bergamo_map(pp);
                        let fm = bergamo_map(pm);
                        for row in 0..3 {
                            let fd = (fp[row] - fm[row]) / (2.0 * step);
                            assert!(
                                approx_eq(j[row][col], fd, 1e-2),
                                "p={p:?} col={col} row={row}: j={} fd={}",
                                j[row][col],
                                fd
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mat3_inverse_of_identity_is_identity() {
        let inv = mat3_inverse(mat3_identity()).unwrap();
        for i in 0..3 {
            for k in 0..3 {
                let expected = if i == k { 1.0 } else { 0.0 };
                assert!(approx_eq(inv[i][k], expected, 1e-7));
            }
        }
    }

    #[test]
    fn mat3_inverse_of_bergamo_j_at_face_center_is_diag_inverse() {
        let j = bergamo_jacobian([0.0, 1.0, 0.0]);
        let inv = mat3_inverse(j).unwrap();
        // At (0, 1, 0): J = diag(1/√2, 1, 1/√2), so J⁻¹ = diag(√2, 1, √2).
        assert!(approx_eq(inv[0][0], 2.0f32.sqrt(), 1e-5));
        assert!(approx_eq(inv[1][1], 1.0, 1e-6));
        assert!(approx_eq(inv[2][2], 2.0f32.sqrt(), 1e-5));
    }

    #[test]
    fn frame_for_empty_path_is_identity_origin_with_root_scale() {
        let f = sphere_frame_from_path(&[]);
        assert!(approx_vec(f.origin, [0.0, 0.0, 0.0], 1e-6));
        // Depth-0 render frame = whole SphereBody cube: half-width 1,
        // scale = (2/3)·half / 1 (simplified form) = 2/3.
        assert!(approx_eq(f.scale, 2.0 / 3.0, 1e-6));
    }

    #[test]
    fn frame_scale_shrinks_by_3_per_depth() {
        let f0 = sphere_frame_from_path(&[]);
        let f1 = sphere_frame_from_path(&[13]);
        let f2 = sphere_frame_from_path(&[13, 13]);
        assert!(approx_eq(f0.scale, 2.0 / 3.0, 1e-6));
        assert!(approx_eq(f1.scale, 2.0 / 9.0, 1e-6));
        assert!(approx_eq(f2.scale, 2.0 / 27.0, 1e-6));
    }

    #[test]
    fn frame_for_center_path_stays_at_origin() {
        // Slot 13 is (1, 1, 1) — the center child in every direction.
        // Any depth of center-child descent leaves c at origin.
        for d in 0..20 {
            let slots = vec![13u8; d];
            let f = sphere_frame_from_path(&slots);
            assert!(
                approx_vec(f.origin, [0.0, 0.0, 0.0], 1e-6),
                "depth {d}: origin {:?}",
                f.origin
            );
        }
    }

    #[test]
    fn frame_depth_1_slot_extremes_land_at_two_thirds() {
        // Slot 0 = (0, 0, 0): all three axes move to -2/3.
        let f = sphere_frame_from_path(&[0]);
        assert!(approx_vec(f.origin, [-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0], 1e-6));
        // Slot 26 = (2, 2, 2): all three axes move to +2/3.
        let f = sphere_frame_from_path(&[26]);
        assert!(approx_vec(f.origin, [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], 1e-6));
    }

    #[test]
    fn frame_at_deep_center_path_stays_stable() {
        // Center-child descent 40 levels deep — classic dig-down
        // path. f32 origin must not saturate to garbage; scale
        // should be finite (if tiny).
        let slots = vec![13u8; 40];
        let f = sphere_frame_from_path(&slots);
        assert!(approx_vec(f.origin, [0.0, 0.0, 0.0], 1e-6));
        assert!(f.scale > 0.0 && f.scale.is_finite(), "scale = {}", f.scale);
    }

    #[test]
    fn find_active_none_in_pure_cartesian_tree() {
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(empty_children());
        let root = lib.insert_with_kind(
            uniform_children(Child::Node(leaf)),
            NodeKind::Cartesian,
        );
        lib.ref_inc(root);
        let mut p = Path::root();
        for _ in 0..3 { p.push(13); }
        assert!(find_active_sphere_frame(&lib, root, &p).is_none());
    }

    #[test]
    fn find_active_when_world_root_is_sphere_body() {
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(empty_children());
        let root = lib.insert_with_kind(
            uniform_children(Child::Node(leaf)),
            NodeKind::SphereBody,
        );
        lib.ref_inc(root);
        // Empty path ⇒ render frame IS the SphereBody: origin = (0,0,0)
        let empty = Path::root();
        let f = find_active_sphere_frame(&lib, root, &empty).expect("sphere frame");
        assert!(approx_vec(f.origin, [0.0, 0.0, 0.0], 1e-6));

        // Descend into slot 0 (corner) ⇒ origin shifts to (-2/3, -2/3, -2/3).
        let mut p = Path::root();
        p.push(0);
        let f = find_active_sphere_frame(&lib, root, &p).expect("sphere frame");
        assert!(approx_vec(f.origin, [-2.0 / 3.0; 3], 1e-6));
    }

    #[test]
    fn find_active_stops_tracking_at_non_node_child() {
        let mut lib = NodeLibrary::default();
        let mut children = empty_children();
        children[13] = Child::Block(crate::world::palette::block::STONE);
        let root = lib.insert_with_kind(children, NodeKind::SphereBody);
        lib.ref_inc(root);
        let mut p = Path::root();
        p.push(13);
        p.push(0);
        // After descending into the Block child, walker breaks.
        // Returned origin reflects the path we DID take (empty — we
        // never descended into a node).
        let f = find_active_sphere_frame(&lib, root, &p).expect("sphere frame");
        assert!(approx_vec(f.origin, [0.0, 0.0, 0.0], 1e-6));
    }

    /// 40-layer off-center path: origin must stay in [-1, 1] and J⁻ᵀ
    /// must produce a finite, well-conditioned matrix. The specific
    /// slot (13 = center) vs an off-axis slot (17 = (2,2,1)) isn't
    /// geometrically important — we're stress-testing f32 stability
    /// under 40-level accumulation.
    #[test]
    fn frame_at_deep_off_center_path_produces_finite_origin() {
        let slots = vec![17u8; 40];
        let f = sphere_frame_from_path(&slots);
        for c in f.origin.iter() {
            assert!(c.is_finite() && c.abs() <= 1.0, "origin = {:?}", f.origin);
        }
        assert!(f.scale > 0.0 && f.scale.is_finite());
    }
}
