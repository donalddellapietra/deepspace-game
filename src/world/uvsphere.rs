//! UV-sphere geometry + worldgen.
//!
//! A UV-sphere body lives inside the Cartesian tree as a single
//! `NodeKind::UvSphereBody { inner_r, outer_r, theta_cap }` node.
//! Its 27 children are interpreted as `(φ-tier, θ-tier, r-tier)`
//! cells in spherical-coordinate parameter space rather than xyz.
//! Descendants stay `NodeKind::Cartesian` — the slot-index
//! convention re-interpretation is contagious all the way down the
//! body subtree, but the storage is identical to a Cartesian tree.
//!
//! Parameter space:
//! - `φ ∈ [0, 2π)`     longitude. Wraps modulo 2π.
//! - `θ ∈ [-θ_cap, +θ_cap]`  latitude. Bounded; the polar caps
//!   `|θ| > θ_cap` are not voxelized — the user opted to "replace
//!   the poles with caps", so the body's surface is a spherical
//!   zone (band) plus optional cap impostors.
//! - `r ∈ [inner_r, outer_r]`  radial shell.
//!
//! World position from `(φ, θ, r)`, with body center `c`:
//!   `x = c.x + r·cos(θ)·cos(φ)`
//!   `y = c.y + r·sin(θ)`
//!   `z = c.z + r·cos(θ)·sin(φ)`
//!
//! Inner_r / outer_r / center are in the body cell's local
//! `[0, 1)` frame; `θ_cap` is in radians, `0 < θ_cap ≤ π/2`.
//!
//! The local spherical frame `{∂/∂φ, ∂/∂θ, ∂/∂r}` is orthogonal
//! at every point. That's why UV-sphere voxels have right angles
//! everywhere — including at face/edge boundaries that the
//! cubed-sphere couldn't do without diamond shear.

use super::sdf::{self, Planet, Vec3};
use super::tree::{
    empty_children, slot_index, uniform_children, Child, NodeId, NodeKind, NodeLibrary,
};

// ─────────────────────────────────────────── coord conversions

/// `(φ, θ, r)` in body-local units (φ rad, θ rad, r in body cell
/// `[0, 1)` frame).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct UvPoint {
    pub phi: f32,
    pub theta: f32,
    pub r: f32,
}

/// Convert body-local xyz (offset from body cell origin, in `[0, 1)³`)
/// to `(φ, θ, r)`. `body_size` is the body cell's size in the
/// caller's frame; `point_body` is in that frame, measured from the
/// body cell's origin (so the body center is at `body_size * 0.5`).
///
/// Returns `None` if the point coincides with the body center.
pub fn body_point_to_uv(point_body: Vec3, body_size: f32) -> Option<UvPoint> {
    let center = [body_size * 0.5; 3];
    let off = sdf::sub(point_body, center);
    let r = sdf::length(off);
    if r <= 1e-12 {
        return None;
    }
    // φ from x/z plane; θ from y / r.
    let phi = off[2].atan2(off[0]);
    let phi = if phi < 0.0 { phi + std::f32::consts::TAU } else { phi };
    let theta = (off[1] / r).clamp(-1.0, 1.0).asin();
    Some(UvPoint { phi, theta, r })
}

/// Inverse of `body_point_to_uv`: `(φ, θ, r)` → body-local xyz.
pub fn uv_to_body_point(uv: UvPoint, body_size: f32) -> Vec3 {
    let center = [body_size * 0.5; 3];
    let cos_t = uv.theta.cos();
    let sin_t = uv.theta.sin();
    let cos_p = uv.phi.cos();
    let sin_p = uv.phi.sin();
    [
        center[0] + uv.r * cos_t * cos_p,
        center[1] + uv.r * sin_t,
        center[2] + uv.r * cos_t * sin_p,
    ]
}

/// Slot tiers `(φ_tier, θ_tier, r_tier) ∈ {0,1,2}³` for a UV cell
/// inside a parent with bounds `(φ_lo..φ_hi, θ_lo..θ_hi, r_lo..r_hi)`.
/// φ wraps; θ and r clamp.
#[inline]
pub fn slot_for_uv(
    uv: UvPoint,
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
) -> (usize, usize, usize) {
    let dphi = (phi_hi - phi_lo).max(1e-12);
    let dth = (theta_hi - theta_lo).max(1e-12);
    let dr = (r_hi - r_lo).max(1e-12);
    let p_norm = ((uv.phi - phi_lo) / dphi).rem_euclid(1.0);
    let t_norm = ((uv.theta - theta_lo) / dth).clamp(0.0, 1.0 - 1e-7);
    let r_norm = ((uv.r - r_lo) / dr).clamp(0.0, 1.0 - 1e-7);
    let pt = (p_norm * 3.0).floor().clamp(0.0, 2.0) as usize;
    let tt = (t_norm * 3.0).floor().clamp(0.0, 2.0) as usize;
    let rt = (r_norm * 3.0).floor().clamp(0.0, 2.0) as usize;
    (pt, tt, rt)
}

// ─────────────────────────────────────────── ray–body primitives

/// Ray-sphere entry time, in body-frame units. Both shells (inner /
/// outer) use this primitive. `origin` and `dir` in body-local frame
/// where body center is at `[body_size/2; 3]`. Returns `None` if the
/// ray misses the sphere of `radius_local · body_size`.
pub fn ray_sphere_hit(
    origin: Vec3,
    dir: Vec3,
    radius_local: f32,
    body_size: f32,
) -> Option<f32> {
    let center = [body_size * 0.5; 3];
    let radius = radius_local * body_size;
    let oc = sdf::sub(origin, center);
    let b = sdf::dot(oc, dir);
    let c = sdf::dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t_enter = -b - sq;
    let t_exit = -b + sq;
    let t = if t_enter > 0.0 {
        t_enter
    } else if t_exit > 0.0 {
        t_exit
    } else {
        return None;
    };
    Some(t)
}

/// Ray–cone (axis = body Y, apex at body center, half-angle
/// `π/2 - |θ|` away from axis). Returns the smallest `t > t_min` on
/// the cone where `sign(y) == sign(theta)`. `theta` in radians;
/// θ = 0 degenerates to the equatorial plane (use a half-plane test
/// instead). `θ = ±π/2` is the pole — also degenerate.
///
/// Cone equation around Y axis: `x² + z² = (tan(α))² · y²`, where
/// `α = π/2 - |θ|` is the half-angle from the axis. Equivalently:
/// `cos²(θ) · y² = sin²(θ) · (x² + z²)` after rearranging — but we
/// use the form below to keep f32 stable for θ near 0.
///
/// We restrict to the half-cone matching `sign(theta)`:
/// y > 0 for θ > 0, y < 0 for θ < 0.
pub fn ray_cone_hit(
    origin: Vec3,
    dir: Vec3,
    theta: f32,
    body_size: f32,
    t_min: f32,
) -> Option<f32> {
    let center = [body_size * 0.5; 3];
    let oc = sdf::sub(origin, center);
    // Half-cone axis-aligned cone: cos²θ · y² = sin²θ · (x² + z²),
    // restricted to sign(y) == sign(θ).
    let s = theta.sin();
    let c = theta.cos();
    let s2 = s * s;
    let c2 = c * c;
    // Substitute oc + dir·t and solve quadratic A·t² + 2·B·t + C = 0.
    // f(p) = c²·p_y² - s²·(p_x² + p_z²)
    // f(p)·sign(s) >= 0 picks the right half-cone (when s != 0).
    let aa = c2 * dir[1] * dir[1] - s2 * (dir[0] * dir[0] + dir[2] * dir[2]);
    let bb = c2 * oc[1] * dir[1] - s2 * (oc[0] * dir[0] + oc[2] * dir[2]);
    let cc = c2 * oc[1] * oc[1] - s2 * (oc[0] * oc[0] + oc[2] * oc[2]);
    if aa.abs() < 1e-12 {
        // Linear: 2·B·t + C = 0.
        if bb.abs() < 1e-12 {
            return None;
        }
        let t = -cc / (2.0 * bb);
        if t > t_min && half_cone_ok(oc, dir, t, theta) {
            return Some(t);
        }
        return None;
    }
    let disc = bb * bb - aa * cc;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t1 = (-bb - sq) / aa;
    let t2 = (-bb + sq) / aa;
    let mut best: Option<f32> = None;
    for &t in &[t1, t2] {
        if t > t_min && half_cone_ok(oc, dir, t, theta) {
            best = Some(match best {
                Some(b) => b.min(t),
                None => t,
            });
        }
    }
    best
}

#[inline]
fn half_cone_ok(oc: Vec3, dir: Vec3, t: f32, theta: f32) -> bool {
    // Hit must lie on the half-cone matching sign(θ). For θ = 0 the
    // surface is the y=0 plane and any y is "on the right side";
    // callers should special-case θ = 0 with a half-plane test.
    let y = oc[1] + dir[1] * t;
    if theta > 0.0 {
        y >= -1e-7
    } else if theta < 0.0 {
        y <= 1e-7
    } else {
        true
    }
}

/// Ray–half-plane test for a constant-φ surface. `phi` in radians.
/// The plane passes through the body Y axis and has outward normal
/// `(-sin(φ), 0, cos(φ))` (so the half-plane is the `+x'` direction
/// in the rotated body frame).
///
/// Returns the smallest `t > t_min` where the ray intersects the
/// plane on the correct side (radial outward from the Y axis).
pub fn ray_phi_plane_hit(
    origin: Vec3,
    dir: Vec3,
    phi: f32,
    body_size: f32,
    t_min: f32,
) -> Option<f32> {
    let center = [body_size * 0.5; 3];
    let oc = sdf::sub(origin, center);
    let s = phi.sin();
    let c = phi.cos();
    // Plane equation: x·(-sin φ) + z·cos φ = 0, i.e. -s·x + c·z = 0.
    // Substitute oc + dir·t: -s·(oc.x + dir.x·t) + c·(oc.z + dir.z·t) = 0
    let denom = -s * dir[0] + c * dir[2];
    if denom.abs() < 1e-12 {
        return None;
    }
    let num = s * oc[0] - c * oc[2];
    let t = num / denom;
    if t <= t_min {
        return None;
    }
    // Check the hit is on the correct half (radial-out from Y axis,
    // not the antipodal half-plane). After substitution the in-plane
    // coord is x' = c·x + s·z; we want x' >= 0.
    let xp = c * (oc[0] + dir[0] * t) + s * (oc[2] + dir[2] * t);
    if xp < -1e-6 {
        return None;
    }
    Some(t)
}

// ─────────────────────────────────────────── path helpers

/// Walk a hit path looking for the first `UvSphereBody` ancestor.
/// Returns `(path_index, inner_r, outer_r, theta_cap)` where
/// `path_index` is the entry whose child is the body node.
pub fn find_body_ancestor_in_path(
    library: &NodeLibrary,
    hit_path: &[(NodeId, usize)],
) -> Option<(usize, f32, f32, f32)> {
    for (index, &(node_id, slot)) in hit_path.iter().enumerate() {
        let Some(node) = library.get(node_id) else { continue };
        let Child::Node(child_id) = node.children[slot] else { continue };
        let Some(child) = library.get(child_id) else { continue };
        if let NodeKind::UvSphereBody {
            inner_r,
            outer_r,
            theta_cap,
        } = child.kind
        {
            return Some((index, inner_r, outer_r, theta_cap));
        }
    }
    None
}

// ─────────────────────────────────────────── worldgen

/// Demo planet wrapped as a `UvSphereBody`. Inner / outer radii are
/// in the body cell's local `[0, 1)` frame.
#[derive(Clone, Debug)]
pub struct UvSphereSetup {
    pub inner_r: f32,
    pub outer_r: f32,
    /// Latitude cap in radians: voxelization extends only over
    /// `θ ∈ [-θ_cap, +θ_cap]`. Outside that, polar caps are not
    /// stored — rays through the cap region exit the body's voxel
    /// shell and pass to the cap impostor (or sky for MVP).
    pub theta_cap: f32,
    /// Tree depth of the body subtree (number of levels of 3³
    /// recursion below the body root).
    pub depth: u32,
    pub sdf: Planet,
}

pub fn demo_uv_sphere() -> UvSphereSetup {
    // Smaller-than-cell body so the camera (which lives inside the
    // body cell at the [0, 3)³ render frame) can still frame the
    // whole planet without it filling the entire view.
    let inner_r = 0.05_f32;
    let outer_r = 0.20_f32;
    let theta_cap = 80.0_f32.to_radians(); // ~6:1 worst aspect at the cap
    // Mirror plain_world's "non-ternary-rational band radii" trick.
    // The body's r range is `[inner_r, outer_r]` = [0.05, 0.20], span
    // 0.15. Pick band thicknesses so the grass-dirt and dirt-stone
    // boundaries land at fractions of this range whose lowest-terms
    // denominators have factors other than 3 — guaranteeing every
    // recursion level near the surface contains MIXED grass/dirt/
    // stone cells. Pack can't flatten mixed cells, so breaks reveal
    // a real 3³ sub-cell grid at any depth, no "warm-up break"
    // required.
    //
    //  surface  at r = 0.15  (frac 2/3 of body r range — ternary-
    //                          rational, but harmless: cells above
    //                          are uniform empty, cells below are
    //                          inside the body)
    //  grass→dirt at r = 0.1425  (frac 37/60 — denom 60 = 2²·3·5)
    //  dirt→stone at r = 0.105   (frac 11/30 — denom 30 = 2·3·5)
    let grass_thickness = 0.0075_f32;  //  5% of r range
    let dirt_thickness = 0.0375_f32;   // 25% of r range
    UvSphereSetup {
        inner_r,
        outer_r,
        theta_cap,
        // 20-layer body. The smooth-ball SDF + non-ternary-rational
        // band radii produce a thin shell of mixed content at every
        // tree level; dedup keeps the library small. Edit depth is
        // driven by `app.edit_depth()` (= anchor depth = zoom level);
        // break_block scales the cell size from "whole body slice"
        // at zoom 1 down to a leaf voxel at zoom 20.
        depth: 20,
        sdf: Planet {
            center: [0.5, 0.5, 0.5],
            radius: 0.15,
            // Smooth ball — no noise. Layered grass / dirt / stone
            // bands give the surface its variety instead.
            noise_scale: 0.0,
            noise_freq: 1.0,
            noise_seed: 0,
            gravity: 9.8,
            influence_radius: outer_r * 2.0,
            surface_block: crate::world::palette::block::GRASS,
            core_block: crate::world::palette::block::STONE,
            grass_thickness,
            dirt_thickness,
        },
    }
}

/// What fills a UV cell when its r-range lies entirely inside one
/// radial band. `None` means the cell straddles a band boundary and
/// must recurse — the analogue of `plain_world::fill_for_range`.
/// Smooth-ball SDF only — noise displacement isn't represented in
/// this band check.
#[derive(Copy, Clone)]
enum UvFill {
    Empty,
    Block(u16),
}

fn fill_for_uv_cell(
    sdf: &Planet,
    inner_r: f32,
    outer_r: f32,
    rn_lo: f32,
    rn_hi: f32,
) -> Option<UvFill> {
    let r_lo = inner_r + rn_lo * (outer_r - inner_r);
    let r_hi = inner_r + rn_hi * (outer_r - inner_r);
    let surface = sdf.radius;
    let grass_band_inner = surface - sdf.grass_thickness;
    let dirt_band_inner = grass_band_inner - sdf.dirt_thickness;

    if r_lo >= surface {
        return Some(UvFill::Empty);
    }
    if r_hi <= dirt_band_inner {
        return Some(UvFill::Block(sdf.core_block));
    }
    if r_lo >= grass_band_inner && r_hi <= surface {
        return Some(UvFill::Block(sdf.surface_block));
    }
    if r_lo >= dirt_band_inner && r_hi <= grass_band_inner {
        return Some(UvFill::Block(crate::world::palette::block::DIRT));
    }
    None
}

/// Build a UV-sphere body node. Caller is responsible for placing it
/// inside a parent (e.g., world root's center slot) and managing the
/// refcount.
///
/// Recursion is along r only — for a smooth-ball SDF the (φ, θ)
/// coordinates don't affect the band-fill decision, so all 9
/// (φ-tier, θ-tier) combinations within a given r-tier share the
/// same subtree. That keeps worldgen O(depth) instead of O(9^depth).
/// Body-tree depth at which UV-shaped cells transition to
/// `CartesianTangent` cells. Above this depth (body-depths 1..N-1) the
/// cells stay UV — visible curvature from outside the sphere comes
/// from these outer levels. AT this depth (body-depth N), each cell
/// becomes a `CartesianTangent` Node whose children are plain
/// cartesian voxels in the cell's tangent frame `(φ̂, θ̂, r̂)`. Below
/// this depth (body-depths N+1..20) all subdivision is cartesian.
///
/// The hybrid lets the macroscopic sphere render as UV (preserving
/// curvature where it's visible) while deep zoom — where one UV
/// cell occupies many pixels and the cube-vs-curve difference is
/// sub-pixel — runs through the precision-stable cartesian DDA.
pub const CARTESIAN_TANGENT_BODY_DEPTH: u32 = 7;

pub fn insert_uv_sphere_body(
    lib: &mut NodeLibrary,
    inner_r: f32,
    outer_r: f32,
    theta_cap: f32,
    depth: u32,
    sdf: &Planet,
) -> NodeId {
    debug_assert!(0.0 < inner_r && inner_r < outer_r && outer_r <= 0.5);
    debug_assert!(0.0 < theta_cap && theta_cap <= std::f32::consts::FRAC_PI_2);

    let mut children = empty_children();
    let r_lo_n = 0.0_f32;
    let r_hi_n = 1.0_f32;
    let drn = (r_hi_n - r_lo_n) / 3.0;
    for rt in 0..3 {
        let rn_lo = r_lo_n + drn * rt as f32;
        let rn_hi = rn_lo + drn;
        // Body root's children are body-depth 1.
        let r_child = build_uv_r_subtree(
            lib, inner_r, outer_r, rn_lo, rn_hi,
            depth.saturating_sub(1), 1, sdf,
        );
        for tt in 0..3 {
            for pt in 0..3 {
                children[slot_index(pt, tt, rt)] = r_child;
            }
        }
    }
    lib.insert_with_kind(
        children,
        NodeKind::UvSphereBody { inner_r, outer_r, theta_cap },
    )
}

/// Recursive build along r only. Mirrors `plain_world`'s
/// `build_plain_subtree` shape: cells fully inside one radial band
/// commit to a uniform subtree; straddling cells subdivide all the
/// way to leaf depth. With non-ternary-rational band radii, the
/// straddle never resolves, so every level near the surface keeps
/// mixed children — pack can't flatten them, breaks reveal a real
/// 3³ sub-cell grid.
///
/// The (φ, θ) dimensions inside each level are SHARED — all 9
/// (φ-tier, θ-tier) cells point at the same r-subtree. With the
/// content-addressed library this happens for free at insert time
/// (the duplicate-children node dedups), but we don't even build
/// the duplicates here — `insert_uv_sphere_body` writes the same
/// `Child` into all 9 slot positions of each r-tier directly.
fn build_uv_r_subtree(
    lib: &mut NodeLibrary,
    inner_r: f32, outer_r: f32,
    rn_lo: f32, rn_hi: f32,
    depth: u32,
    body_depth: u32,
    sdf: &Planet,
) -> Child {
    // At body-depth `CARTESIAN_TANGENT_BODY_DEPTH` we transition from
    // UV-march territory to cartesian-tangent dispatch. The Node we
    // build at this exact depth is marked `CartesianTangent`; that
    // tells the GPU/CPU UV walkers to stop descending and hand the
    // cell off to a cartesian DDA in its tangent frame. Above the
    // boundary (body_depth < N) → plain Cartesian (UV march descends
    // through them). Below (body_depth > N) → plain Cartesian (the
    // CartesianTangent ancestor's cartesian DDA already handles
    // them as pure xyz cells).
    let kind = if body_depth == CARTESIAN_TANGENT_BODY_DEPTH {
        NodeKind::CartesianTangent
    } else {
        NodeKind::Cartesian
    };
    let wrap_uniform_block = |lib: &mut NodeLibrary, b: u16| -> Child {
        // For uniform-Block fills AT the tangent boundary depth, we
        // can't use `build_uniform_subtree` directly — that would
        // return either a `Cartesian` Node or a bare Block. We need
        // a `CartesianTangent` Node containing a uniform subtree, so
        // the dispatch fires at this cell.
        if depth == 0 {
            return Child::Block(b);
        }
        if body_depth == CARTESIAN_TANGENT_BODY_DEPTH {
            let inner = lib.build_uniform_subtree(b, depth - 1);
            let mut children = empty_children();
            for slot in 0..27 {
                children[slot] = inner;
            }
            Child::Node(lib.insert_with_kind(children, NodeKind::CartesianTangent))
        } else {
            lib.build_uniform_subtree(b, depth)
        }
    };

    if let Some(fill) = fill_for_uv_cell(sdf, inner_r, outer_r, rn_lo, rn_hi) {
        return match fill {
            UvFill::Empty => {
                if depth == 0 {
                    Child::Empty
                } else {
                    Child::Node(uniform_empty_chain(lib, depth))
                }
            }
            UvFill::Block(b) => wrap_uniform_block(lib, b),
        };
    }

    if depth == 0 {
        // Cell still straddles a band at leaf depth — pick the block
        // type from the cell center's r.
        let r_c = inner_r + 0.5 * (rn_lo + rn_hi) * (outer_r - inner_r);
        return if r_c >= sdf.radius {
            Child::Empty
        } else {
            // Synthesize a center point on the equator. block_at
            // depends only on |p - center|, so the (φ, θ) doesn't
            // matter — any equatorial point at the right r works.
            let p_center = [0.5 + r_c, 0.5, 0.5];
            Child::Block(sdf.block_at(p_center))
        };
    }

    let mut children = empty_children();
    let drc = (rn_hi - rn_lo) / 3.0;
    for rt in 0..3 {
        let sub_lo = rn_lo + drc * rt as f32;
        let sub_hi = sub_lo + drc;
        let sub = build_uv_r_subtree(
            lib, inner_r, outer_r, sub_lo, sub_hi,
            depth - 1, body_depth + 1, sdf,
        );
        for tt in 0..3 {
            for pt in 0..3 {
                children[slot_index(pt, tt, rt)] = sub;
            }
        }
    }
    Child::Node(lib.insert_with_kind(children, kind))
}

fn uniform_empty_chain(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert(uniform_children(Child::Node(id)));
    }
    id
}

/// Install a UV-sphere body at the world root's center slot. Returns
/// the new world root and the body's path from the new root.
pub fn install_at_root_center(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    setup: &UvSphereSetup,
) -> (NodeId, crate::world::anchor::Path) {
    let body_id = insert_uv_sphere_body(
        lib,
        setup.inner_r,
        setup.outer_r,
        setup.theta_cap,
        setup.depth,
        &setup.sdf,
    );
    let host_slot = slot_index(1, 1, 1) as u8;
    let root_node = lib.get(world_root).expect("world root exists");
    let mut children = root_node.children;
    children[host_slot as usize] = Child::Node(body_id);
    let new_root = lib.insert(children);
    let mut body_path = crate::world::anchor::Path::root();
    body_path.push(host_slot);
    (new_root, body_path)
}

// ─────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uv_round_trip_equator() {
        for &phi in &[0.1_f32, 1.0, 3.0, 5.5] {
            let uv = UvPoint { phi, theta: 0.0, r: 0.3 };
            let body = uv_to_body_point(uv, 1.0);
            let back = body_point_to_uv(body, 1.0).unwrap();
            assert!((back.phi - phi).abs() < 1e-4, "phi {phi} → {}", back.phi);
            assert!(back.theta.abs() < 1e-4);
            assert!((back.r - 0.3).abs() < 1e-4);
        }
    }

    #[test]
    fn uv_round_trip_off_equator() {
        for &theta in &[-1.0_f32, -0.3, 0.3, 1.0] {
            let uv = UvPoint { phi: 1.5, theta, r: 0.4 };
            let body = uv_to_body_point(uv, 1.0);
            let back = body_point_to_uv(body, 1.0).unwrap();
            assert!((back.phi - 1.5).abs() < 1e-4);
            assert!((back.theta - theta).abs() < 1e-4, "θ {theta} → {}", back.theta);
            assert!((back.r - 0.4).abs() < 1e-4);
        }
    }

    #[test]
    fn ray_sphere_hits_outer_shell() {
        // From outside the body cell, ray pointed straight at the
        // body center hits the outer shell.
        let origin = [0.5, 0.5, -1.0];
        let dir = [0.0, 0.0, 1.0];
        let t = ray_sphere_hit(origin, dir, 0.45, 1.0).unwrap();
        // World hit at z = -1 + t·1; expected at body center - outer_r,
        // i.e. z = 0.5 - 0.45 = 0.05; so t = 1.05.
        assert!((t - 1.05).abs() < 1e-4, "t = {t}");
    }

    #[test]
    fn ray_sphere_misses() {
        let origin = [-1.0, 2.0, -1.0];
        let dir = [0.0, 0.0, 1.0];
        assert!(ray_sphere_hit(origin, dir, 0.45, 1.0).is_none());
    }

    #[test]
    fn ray_phi_plane_hits_in_front() {
        // Body at center [0.5; 3]. Ray from south face heading north.
        // φ = π/2 plane is the +z half. Ray should hit on the +z side.
        let origin = [0.5, 0.5, -0.2];
        let dir = [0.0, 0.0, 1.0];
        let t = ray_phi_plane_hit(origin, dir, std::f32::consts::FRAC_PI_2, 1.0, 0.0)
            .expect("hit");
        // Expected: hit when z = 0.5 (passing through Y axis).
        assert!((t - 0.7).abs() < 1e-4, "t = {t}");
    }

    #[test]
    fn ray_cone_hits_upper_half() {
        // Ray going up from below body equator — must hit the upper-θ
        // cone.
        let origin = [0.5, 0.4, 0.0];
        let dir = sdf::normalize([0.0, 1.0, 1.0]);
        let theta = std::f32::consts::FRAC_PI_4; // 45°
        let t = ray_cone_hit(origin, dir, theta, 1.0, 0.0).expect("hit");
        let hit = sdf::add(origin, sdf::scale(dir, t));
        let off = sdf::sub(hit, [0.5, 0.5, 0.5]);
        // Cone surface check: cos²(θ)·y² ≈ sin²(θ)·(x²+z²).
        let lhs = theta.cos().powi(2) * off[1] * off[1];
        let rhs = theta.sin().powi(2) * (off[0] * off[0] + off[2] * off[2]);
        assert!((lhs - rhs).abs() < 1e-3, "cone surface lhs={lhs} rhs={rhs}");
        // Upper half: y > body center y.
        assert!(off[1] >= -1e-6);
    }

    #[test]
    fn slot_for_uv_buckets_correctly() {
        let phi_lo = 0.0;
        let phi_hi = std::f32::consts::TAU;
        let theta_cap = 1.4;
        let r_lo = 0.0;
        let r_hi = 1.0;
        // Just past φ_lo; θ slightly south; r near outer.
        let uv = UvPoint {
            phi: 0.1,
            theta: -0.5,
            r: 0.95,
        };
        let (pt, tt, rt) = slot_for_uv(uv, phi_lo, phi_hi, -theta_cap, theta_cap, r_lo, r_hi);
        assert_eq!(pt, 0);
        // -0.5 normalized = (−0.5 - −1.4) / 2.8 ≈ 0.32 → tier 0.
        assert_eq!(tt, 0);
        assert_eq!(rt, 2);
    }

    #[test]
    fn insert_uv_sphere_body_creates_node_with_kind() {
        use crate::world::palette::block;
        let mut lib = NodeLibrary::default();
        let sdf = Planet {
            center: [0.5; 3], radius: 0.30,
            noise_scale: 0.0, noise_freq: 1.0, noise_seed: 0,
            gravity: 0.0, influence_radius: 1.0,
            surface_block: block::GRASS, core_block: block::STONE,
            grass_thickness: 0.0, dirt_thickness: 0.0,
        };
        let body = insert_uv_sphere_body(&mut lib, 0.12, 0.45, 1.4, 6, &sdf);
        let node = lib.get(body).unwrap();
        match node.kind {
            NodeKind::UvSphereBody { inner_r, outer_r, theta_cap } => {
                assert!((inner_r - 0.12).abs() < 1e-6);
                assert!((outer_r - 0.45).abs() < 1e-6);
                assert!((theta_cap - 1.4).abs() < 1e-6);
            }
            other => panic!("expected UvSphereBody, got {:?}", other),
        }
    }
}
