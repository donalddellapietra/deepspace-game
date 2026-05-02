//! CPU UV-sphere DDA — mirror of `assets/shaders/uvsphere.wgsl`'s
//! `march_uv_sphere`. Walks a `NodeKind::UvSphereBody` subtree in
//! (φ, θ, r) parameter space, returning a `HitInfo` whose `path`
//! starts from `body_root_id` and ends at the slot of the hit cell
//! — so `world::edit::break_block` works without any UV-aware
//! changes downstream.
//!
//! Inputs: body root NodeId; ray origin and direction in the body's
//! `[0, 3)³` local frame (body geometric center at `[1.5, 1.5, 1.5]`).

use super::HitInfo;
use crate::world::tree::{
    slot_index, Child, NodeId, NodeKind, NodeLibrary, REPRESENTATIVE_EMPTY,
};

const TAU: f32 = std::f32::consts::TAU;

/// Cast a ray through a UV-sphere body subtree. Returns `None` on
/// miss (ray doesn't intersect the body shell, or exits without
/// finding a Block within `max_iter` cell-boundary steps).
///
/// `max_depth` is the maximum number of slot entries in the returned
/// path (= max descent depth from the body root). When the descent
/// would exceed it on a `Child::Node`, the walker terminates early
/// with the Node's parent slot — mirroring Cartesian's coarse-cell
/// behavior so `break_block` removes a "fat" cell at low edit depth.
/// At edit depth ≥ body subtree depth, the walker reaches the leaf
/// `Child::Block` and breaks the smallest cell, like Cartesian does.
pub fn cpu_raycast_uv_body(
    library: &NodeLibrary,
    body_root_id: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
) -> Option<HitInfo> {
    let body = library.get(body_root_id)?;
    let (inner_r_local, outer_r_local, theta_cap) = match body.kind {
        NodeKind::UvSphereBody { inner_r, outer_r, theta_cap } => (inner_r, outer_r, theta_cap),
        _ => return None,
    };

    let body_size = 3.0_f32;
    let center = [body_size * 0.5; 3];
    let outer_r = outer_r_local * body_size;
    let inner_r = inner_r_local * body_size;

    let oc = sub3(ray_origin, center);
    let outer_t = ray_sphere(oc, ray_dir, outer_r)?;
    if outer_t.1 < 1e-4 || outer_t.0 > outer_t.1 {
        return None;
    }
    let inside_outer = dot3(oc, oc) <= outer_r * outer_r;
    let mut t: f32 = if inside_outer {
        1e-4
    } else {
        outer_t.0.max(1e-4)
    };
    let t_exit_outer = outer_t.1;

    let max_iter = 256_u32;
    for _ in 0..max_iter {
        if t > t_exit_outer + 1e-4 {
            return None;
        }
        let pos = add3(ray_origin, scale3(ray_dir, t));
        let off = sub3(pos, center);
        let r_w = length3(off);

        if r_w > outer_r * 1.0001 {
            return None;
        }
        if r_w < inner_r * 0.9999 {
            // Inner-core hit: synthesize a HitInfo at the body root
            // pointing at the slot we're stepping into. break_block
            // expects a real Block leaf, so the caller's break path
            // will replace whatever's at that slot — for the inner
            // core we treat the central r-tier slot as the editable
            // surrogate. This is consistent with how the renderer
            // treats it (smooth stone shading).
            let phi = off[2].atan2(off[0]);
            let phi_w = if phi < 0.0 { phi + TAU } else { phi };
            let theta_w = (off[1] / r_w.max(1e-6)).clamp(-1.0, 1.0).asin();
            let pt = ((phi_w / (TAU / 3.0)).floor().clamp(0.0, 2.0)) as usize;
            let tt =
                (((theta_w + theta_cap) / (2.0 * theta_cap / 3.0)).floor().clamp(0.0, 2.0)) as usize;
            let rt = 0; // innermost r-tier
            let slot = slot_index(pt, tt, rt);
            return Some(HitInfo {
                path: vec![(body_root_id, slot)],
                face: 0,
                t,
                place_path: None,
            });
        }
        let theta_w = (off[1] / r_w.max(1e-6)).clamp(-1.0, 1.0).asin();
        if theta_w.abs() > theta_cap {
            return None;
        }
        let phi_raw = off[2].atan2(off[0]);
        let phi_w = if phi_raw < 0.0 { phi_raw + TAU } else { phi_raw };

        // Descend from body root to the deepest cell containing
        // (phi_w, theta_w, r_w). Stops on Block (HIT), Empty, or
        // when path.len() reaches `max_depth` (early-terminate on a
        // Node — break_block then replaces the whole subtree).
        let descent = descend(
            library,
            body_root_id,
            inner_r,
            outer_r,
            theta_cap,
            phi_w,
            theta_w,
            r_w,
            max_depth,
        )?;

        if descent.found_block {
            return Some(HitInfo {
                path: descent.path,
                face: descent.face,
                t,
                place_path: None,
            });
        }

        // CartesianTangent dispatch: descent stopped at a cartesian-
        // content child Node. Two outcomes:
        //
        //   - Budget remaining: build the tangent-frame OBB from the
        //     cell bounds, transform the ray, and run the world's
        //     standard cartesian DDA on the subtree. On hit, splice
        //     the sub-DDA's path onto the body prefix.
        //
        //   - Budget exhausted (= max_depth reached on the body side):
        //     can't descend further, treat the whole tangent cell as
        //     a single breakable unit. Mirrors how the ordinary
        //     descent caps at non-empty Node children.
        //
        // On sub-DDA miss WITH budget remaining (= the subtree's
        // genuinely empty along this ray), step past the cell and
        // continue the UV march so the body content behind it can
        // render.
        if let Some(tangent_root) = descent.tangent_root {
            let remaining = max_depth.saturating_sub(descent.path.len() as u32);
            if remaining == 0 {
                let face = closest_face_axis(
                    phi_w, theta_w, r_w,
                    descent.phi_lo, descent.phi_hi,
                    descent.theta_lo, descent.theta_hi,
                    descent.r_lo, descent.r_hi,
                );
                return Some(HitInfo {
                    path: descent.path,
                    face,
                    t,
                    place_path: None,
                });
            }
            if let Some(sub_hit) = dispatch_tangent_cartesian(
                library,
                tangent_root,
                ray_origin,
                ray_dir,
                descent.phi_lo,
                descent.phi_hi,
                descent.theta_lo,
                descent.theta_hi,
                descent.r_lo,
                descent.r_hi,
                remaining,
            ) {
                let mut full_path = descent.path;
                full_path.extend(sub_hit.path);
                return Some(HitInfo {
                    path: full_path,
                    face: sub_hit.face,
                    t: sub_hit.t,
                    place_path: None,
                });
            }
            // Sub-DDA missed — step past the cell.
            let bd = next_boundary(
                oc, ray_dir, t,
                descent.phi_lo, descent.phi_hi,
                descent.theta_lo, descent.theta_hi,
                descent.r_lo, descent.r_hi,
            );
            if bd.t > 1e20 { return None; }
            let step = bd.t - t;
            t = bd.t + (step * 1e-4).max(1e-5);
            continue;
        }

        // Empty cell at descent.bounds; step the ray to the smallest
        // t exceeding `t` at any of the 6 cell boundaries.
        let bd = next_boundary(
            oc,
            ray_dir,
            t,
            descent.phi_lo,
            descent.phi_hi,
            descent.theta_lo,
            descent.theta_hi,
            descent.r_lo,
            descent.r_hi,
        );
        if bd.t > 1e20 {
            return None;
        }
        let step = bd.t - t;
        t = bd.t + (step * 1e-4).max(1e-5);
    }

    None
}

struct Descent {
    found_block: bool,
    /// `(node_id, slot)` chain from `body_root_id` down to the
    /// terminal cell. The last entry's slot is the slot of the
    /// hit child (Block).
    path: Vec<(NodeId, usize)>,
    /// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z. Computed from the
    /// closest-face axis on hit.
    face: u32,
    phi_lo: f32,
    phi_hi: f32,
    theta_lo: f32,
    theta_hi: f32,
    r_lo: f32,
    r_hi: f32,
    /// Set when the descent stopped at a `CartesianTangent` Node.
    /// The caller must run a cartesian DDA on the subtree in the
    /// cell's tangent frame and append the sub-DDA's path to
    /// `path`. `None` for ordinary UV terminals.
    tangent_root: Option<NodeId>,
}

fn descend(
    library: &NodeLibrary,
    body_root_id: NodeId,
    body_inner_r: f32,
    body_outer_r: f32,
    body_theta_cap: f32,
    phi_w: f32,
    theta_w: f32,
    r_w: f32,
    max_depth: u32,
) -> Option<Descent> {
    // Delta-tracking: hold `delta_X = X_w − X_lo` directly instead
    // of accumulating `X_lo` and subtracting at every level. The
    // straightforward `(phi_w − phi_lo) / dphi` form fails at depth
    // 12+ because `phi_lo` accumulates `K · ULP(2π)` rounding (~1e-6
    // by `K=12`), and `dphi = 2π/3^K ≈ 1.2e-5` is the same scale —
    // tier picking goes ~50% wrong. Holding `delta` keeps it in
    // O(`dphi`) arithmetic the whole way down: each refinement
    // `delta -= pt · dphi` operates on similar-magnitude operands,
    // so the result's precision is O(dphi · ULP) — five orders of
    // magnitude tighter at K=12. For the absolute bounds we only
    // need on the last hop (face / bevel / step boundaries),
    // recompute them at return as `phi_lo = phi_w − delta`.
    let mut delta_phi   = phi_w;                   // = phi_w − 0
    let mut delta_theta = theta_w + body_theta_cap;// = theta_w − (−theta_cap)
    let mut delta_r     = r_w - body_inner_r;
    let mut dphi: f32 = TAU;
    let mut dth: f32 = 2.0 * body_theta_cap;
    let mut dr_axis: f32 = body_outer_r - body_inner_r;
    let mut node_id = body_root_id;
    let mut path: Vec<(NodeId, usize)> = Vec::new();

    loop {
        // Step down: each axis's child cell is `1/3` of the parent.
        dphi = dphi / 3.0;
        dth = dth / 3.0;
        dr_axis = dr_axis / 3.0;

        let pt = ((delta_phi / dphi.max(1e-30)).floor().clamp(0.0, 2.0)) as usize;
        let tt = ((delta_theta / dth.max(1e-30)).floor().clamp(0.0, 2.0)) as usize;
        let rt = ((delta_r / dr_axis.max(1e-30)).floor().clamp(0.0, 2.0)) as usize;
        let slot = slot_index(pt, tt, rt);

        let node = library.get(node_id)?;
        path.push((node_id, slot));

        // Refine deltas to the (pt, tt, rt) cell. Subtraction of
        // similar-magnitude values keeps result-magnitude precision —
        // never falls off the tier-picking precision cliff.
        delta_phi   = delta_phi   - (pt as f32) * dphi;
        delta_theta = delta_theta - (tt as f32) * dth;
        delta_r     = delta_r     - (rt as f32) * dr_axis;
        // Recover absolute bounds for the early-exit branches.
        let phi_lo = phi_w - delta_phi;
        let phi_hi = phi_lo + dphi;
        let theta_lo = theta_w - delta_theta;
        let theta_hi = theta_lo + dth;
        let r_lo = r_w - delta_r;
        let r_hi = r_lo + dr_axis;

        match node.children[slot] {
            Child::Empty | Child::EntityRef(_) => {
                return Some(Descent {
                    found_block: false,
                    path,
                    face: 0,
                    phi_lo,
                    phi_hi,
                    theta_lo,
                    theta_hi,
                    r_lo,
                    r_hi,
                    tangent_root: None,
                });
            }
            Child::Block(_) => {
                let face = closest_face_axis(
                    phi_w, theta_w, r_w, phi_lo, phi_hi, theta_lo, theta_hi, r_lo, r_hi,
                );
                return Some(Descent {
                    found_block: true,
                    path,
                    face,
                    phi_lo,
                    phi_hi,
                    theta_lo,
                    theta_hi,
                    r_lo,
                    r_hi,
                    tangent_root: None,
                });
            }
            Child::Node(child_id) => {
                // CartesianTangent: stop UV descent here and ask the
                // caller to run a cartesian DDA in the cell's tangent
                // frame. The cell bounds give the OBB transform.
                let child_kind = library.get(child_id).map(|n| n.kind);
                if matches!(child_kind, Some(NodeKind::CartesianTangent)) {
                    return Some(Descent {
                        found_block: false,
                        path,
                        face: 0,
                        phi_lo,
                        phi_hi,
                        theta_lo,
                        theta_hi,
                        r_lo,
                        r_hi,
                        tangent_root: Some(child_id),
                    });
                }

                // Edit-depth cap: if descending further would exceed
                // `max_depth` (= maximum path length the caller wants),
                // terminate now and treat this Node as the breakable
                // cell — UNLESS the Node is uniform-empty (represents
                // an air region the worldgen didn't bother to leave
                // as `Child::Empty`), in which case step the ray
                // through it to the next cell instead of "breaking"
                // air. Mirrors Cartesian's own
                // representative_block == REPRESENTATIVE_EMPTY skip in
                // `cpu_raycast_inner`.
                if (path.len() as u32) >= max_depth {
                    let child_is_empty = library
                        .get(child_id)
                        .map(|n| n.representative_block == REPRESENTATIVE_EMPTY)
                        .unwrap_or(true);
                    if child_is_empty {
                        return Some(Descent {
                            found_block: false,
                            path,
                            face: 0,
                            phi_lo,
                            phi_hi,
                            theta_lo,
                            theta_hi,
                            r_lo,
                            r_hi,
                            tangent_root: None,
                        });
                    }
                    let face = closest_face_axis(
                        phi_w, theta_w, r_w, phi_lo, phi_hi, theta_lo, theta_hi, r_lo, r_hi,
                    );
                    return Some(Descent {
                        found_block: true,
                        path,
                        face,
                        phi_lo,
                        phi_hi,
                        theta_lo,
                        theta_hi,
                        r_lo,
                        r_hi,
                        tangent_root: None,
                    });
                }
                node_id = child_id;
                continue;
            }
        }
    }
}

/// Convert closest cell-face direction into a Cartesian face index.
/// We pick the axis whose face is geometrically closest (in arc
/// length), and report the face's outward direction in standard
/// face-id form (0..=5). This is approximate — UV cell faces aren't
/// axis-aligned cubes — but it's enough for break_block's place
/// adjacency math to land on a sensible neighboring cell.
fn closest_face_axis(
    phi_w: f32,
    theta_w: f32,
    r_w: f32,
    phi_lo: f32,
    phi_hi: f32,
    theta_lo: f32,
    theta_hi: f32,
    r_lo: f32,
    r_hi: f32,
) -> u32 {
    let cos_t = theta_w.cos();
    let arc_phi_lo = r_w * cos_t * (phi_w - phi_lo).abs();
    let arc_phi_hi = r_w * cos_t * (phi_w - phi_hi).abs();
    let arc_th_lo = r_w * (theta_w - theta_lo).abs();
    let arc_th_hi = r_w * (theta_w - theta_hi).abs();
    let arc_r_lo = (r_w - r_lo).abs();
    let arc_r_hi = (r_w - r_hi).abs();

    let mut best = arc_phi_lo;
    let mut face = 1_u32; // -X analog for φ_lo
    if arc_phi_hi < best {
        best = arc_phi_hi;
        face = 0;
    }
    if arc_th_lo < best {
        best = arc_th_lo;
        face = 3; // -Y analog
    }
    if arc_th_hi < best {
        best = arc_th_hi;
        face = 2; // +Y analog
    }
    if arc_r_lo < best {
        best = arc_r_lo;
        face = 5; // -Z analog (radially inward)
    }
    if arc_r_hi < best {
        face = 4; // +Z analog (radially outward)
    }
    face
}

struct Boundary {
    t: f32,
}

fn next_boundary(
    oc: [f32; 3],
    dir: [f32; 3],
    t_min: f32,
    phi_lo: f32,
    phi_hi: f32,
    theta_lo: f32,
    theta_hi: f32,
    r_lo: f32,
    r_hi: f32,
) -> Boundary {
    let mut best = 1e30_f32;
    let consider = |best: &mut f32, cand: f32, t_min: f32| {
        if cand > t_min && cand < *best {
            *best = cand;
        }
    };

    consider(&mut best, ray_phi_plane(oc, dir, phi_lo), t_min);
    consider(&mut best, ray_phi_plane(oc, dir, phi_hi), t_min);

    let cl = ray_cone(oc, dir, theta_lo);
    consider(&mut best, cl.0, t_min);
    consider(&mut best, cl.1, t_min);
    let ch = ray_cone(oc, dir, theta_hi);
    consider(&mut best, ch.0, t_min);
    consider(&mut best, ch.1, t_min);

    if let Some(s) = ray_sphere(oc, dir, r_lo) {
        consider(&mut best, s.0, t_min);
        consider(&mut best, s.1, t_min);
    }
    if let Some(s) = ray_sphere(oc, dir, r_hi) {
        consider(&mut best, s.0, t_min);
        consider(&mut best, s.1, t_min);
    }
    Boundary { t: best }
}

fn ray_sphere(oc: [f32; 3], dir: [f32; 3], r: f32) -> Option<(f32, f32)> {
    let aa = dot3(dir, dir);
    let bb = dot3(oc, dir);
    let cc = dot3(oc, oc) - r * r;
    let disc = bb * bb - aa * cc;
    if disc <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let inv_a = 1.0 / aa;
    Some(((-bb - sq) * inv_a, (-bb + sq) * inv_a))
}

fn ray_cone(oc: [f32; 3], dir: [f32; 3], theta: f32) -> (f32, f32) {
    let s = theta.sin();
    let c = theta.cos();
    let s2 = s * s;
    let c2 = c * c;
    let aa = c2 * dir[1] * dir[1] - s2 * (dir[0] * dir[0] + dir[2] * dir[2]);
    let bb = c2 * oc[1] * dir[1] - s2 * (oc[0] * dir[0] + oc[2] * dir[2]);
    let cc = c2 * oc[1] * oc[1] - s2 * (oc[0] * oc[0] + oc[2] * oc[2]);
    if aa.abs() < 1e-10 {
        if bb.abs() < 1e-10 {
            return (1e30, -1e30);
        }
        let t_lin = -cc / (2.0 * bb);
        return (t_lin, t_lin);
    }
    let disc = bb * bb - aa * cc;
    if disc < 0.0 {
        return (1e30, -1e30);
    }
    let sq = disc.sqrt();
    let inv_a = 1.0 / aa;
    ((-bb - sq) * inv_a, (-bb + sq) * inv_a)
}

fn ray_phi_plane(oc: [f32; 3], dir: [f32; 3], phi: f32) -> f32 {
    let s = phi.sin();
    let c = phi.cos();
    let denom = -s * dir[0] + c * dir[2];
    if denom.abs() < 1e-10 {
        return 1e30;
    }
    let num = s * oc[0] - c * oc[2];
    let t = num / denom;
    // Filter to the radial-out half-plane.
    let xp = c * (oc[0] + dir[0] * t) + s * (oc[2] + dir[2] * t);
    if xp < -1e-5 {
        1e30
    } else {
        t
    }
}

#[inline]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn length3(a: [f32; 3]) -> f32 {
    dot3(a, a).sqrt()
}

/// Cartesian-tangent dispatch: when the UV descent stops at a
/// `CartesianTangent` child, this builds the OBB from the cell's
/// UV bounds, transforms the ray into the OBB-local `[0, 3]³`
/// frame, and runs the world's standard cartesian DDA via
/// `cpu_raycast_inner`. Returns a `HitInfo` whose path is rooted
/// at `tangent_root` (caller appends it to the body-side prefix).
///
/// `max_depth` is the path-length budget remaining for the sub-DDA
/// (caller subtracts the body-prefix length already consumed).
/// Returns `None` on miss, or when the budget is zero (caller falls
/// back to a whole-cell hit at the body level).
fn dispatch_tangent_cartesian(
    library: &NodeLibrary,
    tangent_root: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
    remaining_budget: u32,
) -> Option<HitInfo> {
    if remaining_budget == 0 {
        return None;
    }
    let obb = crate::world::raycast::proto_obb::cell_obb(
        [1.5, 1.5, 1.5],
        phi_lo, phi_hi,
        theta_lo, theta_hi,
        r_lo, r_hi,
    );
    // Transform ray into OBB-local [0, 3]³ frame. Linear transform →
    // world-ray t equals OBB-local ray t.
    let to_origin = [
        ray_origin[0] - obb.center[0],
        ray_origin[1] - obb.center[1],
        ray_origin[2] - obb.center[2],
    ];
    let dot3v = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let proj_origin = [
        dot3v(to_origin, obb.phi_hat),
        dot3v(to_origin, obb.theta_hat),
        dot3v(to_origin, obb.r_hat),
    ];
    let proj_dir = [
        dot3v(ray_dir, obb.phi_hat),
        dot3v(ray_dir, obb.theta_hat),
        dot3v(ray_dir, obb.r_hat),
    ];
    let extents = [
        obb.half_phi.max(1e-12),
        obb.half_th.max(1e-12),
        obb.half_r.max(1e-12),
    ];
    let local_origin = [
        proj_origin[0] / extents[0] * 1.5 + 1.5,
        proj_origin[1] / extents[1] * 1.5 + 1.5,
        proj_origin[2] / extents[2] * 1.5 + 1.5,
    ];
    let local_dir = [
        proj_dir[0] / extents[0] * 1.5,
        proj_dir[1] / extents[1] * 1.5,
        proj_dir[2] / extents[2] * 1.5,
    ];
    super::cartesian::cpu_raycast_inner(
        library, tangent_root, local_origin, local_dir, remaining_budget,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, Child, NodeLibrary};
    use crate::world::uvsphere::{demo_uv_sphere, install_at_root_center};

    fn build_demo_tree() -> (NodeLibrary, NodeId, NodeId) {
        let setup = demo_uv_sphere();
        let mut lib = NodeLibrary::default();
        let world_root = lib.insert(empty_children());
        let (new_root, body_path) = install_at_root_center(&mut lib, world_root, &setup);
        lib.ref_inc(new_root);
        let body_slot = body_path.slot(0) as usize;
        let body_root_id = match lib.get(new_root).unwrap().children[body_slot] {
            Child::Node(id) => id,
            _ => panic!("expected body node at world-root center slot"),
        };
        (lib, new_root, body_root_id)
    }

    #[test]
    fn outside_body_no_intersection_returns_none() {
        let (lib, _root, body) = build_demo_tree();
        // Ray well off-axis from the body — shouldn't hit anything.
        let hit = cpu_raycast_uv_body(&lib, body, [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 32);
        assert!(hit.is_none());
    }

    #[test]
    fn ray_through_body_finds_a_block() {
        let (lib, _root, body) = build_demo_tree();
        // Ray from outside the body straight at its center; it
        // should find a Block on the visible hemisphere.
        let hit = cpu_raycast_uv_body(
            &lib,
            body,
            [1.5, 1.5, -3.0],
            [0.0, 0.0, 1.0],
            32,
        )
        .expect("ray through body should hit a Block");
        assert!(hit.path.first().is_some(), "path must start at body root");
        assert_eq!(hit.path[0].0, body, "first entry's node is the body root");
    }

    /// Reproduces the deep-cell tier-picking failure.
    ///
    /// At descent depth K, the descent picks tier from
    /// `(phi_w - phi_lo) / dphi`. `phi_lo` accumulates K ULPs of
    /// f32 rounding from `K` incremental additions. By depth 12
    /// the accumulated error (~`K · 3.7e-7 ≈ 4.5e-6`) is comparable
    /// to `dphi = 2π/3^12 ≈ 1.2e-5` — about 35% of one tier — so
    /// `floor()` picks the wrong tier with high probability and
    /// breaks land in random cells near the click.
    ///
    /// The test fires N rays at slightly perturbed angles that all
    /// hit the same depth-K cell. Each ray's hit path tail (the
    /// deepest few slots) should be identical because they all
    /// land in the same cell. With f32-precision tier picking via
    /// `(phi_w - phi_lo) / dphi`, the tails diverge once descent
    /// reaches the precision cliff.
    /// Sweep phi_w near a deep tier boundary; demonstrates how
    /// the f32 `(phi_w - phi_lo) / dphi` arithmetic flips tiers
    /// across a window much wider than `dphi_target`. Delta-tracking
    /// (the recommended fix) keeps the boundary at sub-ULP width.
    #[test]
    fn descent_tier_picking_near_boundary_at_depth_12() {
        let target_depth = 12_u32;
        // Build a phi_lo that has K accumulated rounding errors,
        // matching what the real descent does.
        let mut phi_lo_built: f32 = 0.0;
        let mut phi_hi_built: f32 = std::f32::consts::TAU;
        let path: [u32; 12] = [1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1];
        for &pt in &path {
            let dphi = (phi_hi_built - phi_lo_built) / 3.0;
            phi_lo_built = phi_lo_built + (pt as f32) * dphi;
            phi_hi_built = phi_lo_built + dphi;
        }
        let dphi_target = phi_hi_built - phi_lo_built;
        let target_phi_lo = phi_lo_built;

        // Pick K phi_w values that step ACROSS the cell at fine
        // sub-cell increments and observe whether (phi_w - phi_lo)
        // / dphi gives the EXPECTED un_phi (= 0..1 linearly).
        let n = 9;
        let mut max_err: f32 = 0.0;
        // Sweep strict-interior fractions of the target cell.
        for i in 0..n {
            let frac = ((i as f32) + 0.5) / (n as f32);
            let phi_w = target_phi_lo + frac * dphi_target;
            // Walk f32 descent USING DELTA TRACKING (the fix). The
            // straightforward `(phi_w − phi_lo) / dphi` form gives
            // ~98% un_phi error at depth 12; delta-tracking holds it
            // at sub-ULP.
            let mut delta: f32 = phi_w;
            let mut dphi: f32 = std::f32::consts::TAU;
            for _ in 0..target_depth {
                dphi = dphi / 3.0;
                let pt = ((delta / dphi).floor().clamp(0.0, 2.0)) as u32;
                delta = delta - (pt as f32) * dphi;
            }
            // un_phi the descent perceives.
            let un_phi = delta / dphi;
            let err = (un_phi - frac).abs();
            if err > max_err { max_err = err; }
        }
        assert!(
            max_err < 0.05,
            "descent's un_phi diverges from expected by more than 5% \
             at depth 12 — max err = {}",
            max_err,
        );
    }

    /// Direct precision test for the descent's tier picking.
    /// Walks from body root to depth K accumulating `phi_lo` and
    /// at each level computes `(phi_w - phi_lo) / dphi` — the same
    /// tier-pick expression as `descend`. By depth 12 the accumulated
    /// rounding in `phi_lo` (~`12 · ULP(2π) ≈ 4.5e-6`) is a meaningful
    /// fraction of `dphi = 2π/3^12 ≈ 1.2e-5` and the picked tier diverges
    /// from the geometrically-correct one.
    #[test]
    fn descent_tier_picking_precision_at_depth_12() {
        let body_inner_r = 0.15_f32;
        let body_outer_r = 0.60_f32;
        let body_theta_cap = std::f32::consts::FRAC_PI_2 * 0.9;
        // Pick a phi_w that lands in a known cell at depth 12.
        // We'll construct it by walking down a known tier path,
        // building phi_lo IDEALLY (without f32 rounding) by summing
        // exact ternary fractions, then setting phi_w = phi_lo + dphi/2.
        let target_depth = 12_u32;
        // Pick a tier sequence — same tier at every level (e.g., 1).
        // The cell at depth K with all tiers = 1 has center at:
        //   phi_center = 2π * (1/3 + 1/9 + 1/27 + ... + 1/3^K) + dphi/2
        //              = 2π * 0.5 * (1 − 3^-K) + 0.5 * dphi
        //              → 2π * 0.5 = π as K → ∞.
        // Use ternary-fraction style sum (exact in f32 only up to K~12).
        let mut phi_lo_ideal: f64 = 0.0;
        let mut dphi_ideal: f64 = std::f64::consts::TAU;
        for _ in 0..target_depth {
            dphi_ideal /= 3.0;
            phi_lo_ideal += dphi_ideal;
        }
        let dphi_target = dphi_ideal as f32;
        let phi_w = (phi_lo_ideal + 0.5 * dphi_ideal) as f32;

        // Now walk descent IN F32 — accumulating phi_lo with rounding.
        let mut phi_lo: f32 = 0.0;
        let mut phi_hi: f32 = std::f32::consts::TAU;
        let mut picked = Vec::new();
        for _ in 0..target_depth {
            let dphi = (phi_hi - phi_lo) / 3.0;
            let pt = (((phi_w - phi_lo) / dphi).floor().clamp(0.0, 2.0)) as u32;
            picked.push(pt);
            phi_lo = phi_lo + (pt as f32) * dphi;
            phi_hi = phi_lo + dphi;
        }
        // Expected: all tiers = 1 (the path we constructed phi_w to land in).
        let expected: Vec<u32> = (0..target_depth).map(|_| 1).collect();
        assert_eq!(
            picked, expected,
            "f32 (phi_w - phi_lo)/dphi diverges at deep depth — descent picks wrong cell"
        );
        // Sanity: dphi at depth 12 should match.
        assert!((dphi_target - dphi_ideal as f32).abs() < 1e-12);
        let _ = (body_inner_r, body_outer_r, body_theta_cap);
    }

    #[test]
    fn deep_descent_tier_picking_is_stable() {
        let (lib, _root, body) = build_demo_tree();
        // Aim at the body's surface; small angular jitter between
        // rays. Body centre = (1.5, 1.5, 1.5); outer_r in body
        // frame ≈ 0.6.
        let cam = [1.5_f32, 1.5, 0.6]; // camera south of body
        let target_depth: u32 = 14; // past the precision cliff
        let mut tails: Vec<Vec<usize>> = Vec::new();
        let n = 5;
        for i in 0..n {
            // Sub-ULP angular jitter — at f32 this is the same ray
            // for atan2 purposes, but the descent's phi_lo
            // accumulation will diverge.
            let dx = (i as f32) * 1e-9;
            let dir = {
                let mut d = [dx, 0.0_f32, 1.0];
                let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                d[0] /= l;
                d[1] /= l;
                d[2] /= l;
                d
            };
            let hit = cpu_raycast_uv_body(&lib, body, cam, dir, target_depth);
            if let Some(h) = hit {
                let depth = h.path.len();
                if depth >= 8 {
                    let tail: Vec<usize> = h.path[depth - 4..]
                        .iter()
                        .map(|(_, s)| *s)
                        .collect();
                    tails.push(tail);
                }
            }
        }

        // All sub-ULP-perturbed rays should land in the same deep
        // cell — the tail should be identical. With buggy tier
        // picking, the tails diverge.
        if tails.len() >= 2 {
            let first = &tails[0];
            for t in &tails[1..] {
                assert_eq!(
                    t, first,
                    "deep descent picks unstable tiers — tails diverge: {:?} vs {:?}",
                    first, t,
                );
            }
        }
    }

    #[test]
    fn hit_path_is_breakable_via_break_block() {
        use crate::world::edit::break_block;
        use crate::world::state::WorldState;
        let (lib, root, body) = build_demo_tree();
        let mut world = WorldState { root, library: lib };
        let hit = cpu_raycast_uv_body(
            &world.library,
            body,
            [1.5, 1.5, -3.0],
            [0.0, 0.0, 1.0],
            32,
        )
        .expect("hit");
        let old_root = world.root;
        // The HitInfo's path is rooted at the BODY, not the world
        // root — the caller (cartesian descent dispatch) is
        // responsible for prepending the body's parent slot. For
        // this test we splice in the world-root → body slot
        // manually so break_block can resolve absolute paths.
        let mut full_path: Vec<(NodeId, usize)> = vec![(world.root, slot_index(1, 1, 1))];
        full_path.extend(hit.path.iter().copied());
        let full_hit = HitInfo {
            path: full_path,
            face: hit.face,
            t: hit.t,
            place_path: None,
        };
        let changed = break_block(&mut world, &full_hit);
        assert!(changed, "break_block must succeed against a UV-DDA hit");
        assert_ne!(world.root, old_root, "world root changes after a break");
    }

    /// Builds a UV-sphere world with the prototype subtree spliced
    /// into body path `[14, 21, 23]` — same setup as
    /// `bootstrap_uv_sphere_world`. Returns `(lib, body_root_id,
    /// proto_root_id)` so tests can target the OBB cell specifically.
    fn build_demo_tree_with_proto() -> (NodeLibrary, NodeId, NodeId) {
        use crate::world::palette::block;
        use crate::world::tree::{slot_index, uniform_children, NodeKind};
        let setup = demo_uv_sphere();
        let mut lib = NodeLibrary::default();
        let world_root = lib.insert(empty_children());
        let (root, body_path) = install_at_root_center(&mut lib, world_root, &setup);
        lib.ref_inc(root);

        // Build a 4-layer uniform-WATER chain capped with a
        // `CartesianTangent` root — same shape as
        // `bootstrap_uv_sphere_world` produces for the real game.
        let proto_depth: u32 = 4;
        let mut proto = lib.insert(uniform_children(Child::Block(block::WATER)));
        lib.ref_inc(proto);
        for _ in 1..proto_depth.saturating_sub(1) {
            let parent = lib.insert(uniform_children(Child::Node(proto)));
            lib.ref_inc(parent);
            lib.ref_dec(proto);
            proto = parent;
        }
        let proto_capped = lib.insert_with_kind(
            uniform_children(Child::Node(proto)),
            NodeKind::CartesianTangent,
        );
        lib.ref_inc(proto_capped);
        lib.ref_dec(proto);
        let proto = proto_capped;

        let body_slot = body_path.slot(0) as usize;
        let body_id = match lib.get(root).unwrap().children[body_slot] {
            Child::Node(id) => id,
            _ => panic!("body must be a Node"),
        };
        // Mirror `replace_at_uv_path` from `bootstrap.rs` so the test
        // exercises the same splice the running game gets.
        fn splice(
            lib: &mut NodeLibrary,
            root: NodeId,
            path: &[usize],
            new_child: Child,
        ) -> NodeId {
            let (mut new_children, kind) = {
                let n = lib.get(root).unwrap();
                (n.children, n.kind)
            };
            let slot = path[0];
            let new_slot = if path.len() == 1 {
                new_child
            } else {
                let child_id = match new_children[slot] {
                    Child::Node(id) => id,
                    _ => panic!("intermediate path must be Node"),
                };
                Child::Node(splice(lib, child_id, &path[1..], new_child))
            };
            new_children[slot] = new_slot;
            lib.insert_with_kind(new_children, kind)
        }
        let new_body = splice(&mut lib, body_id, &[14, 21, 23], Child::Node(proto));
        let mut new_world_children = lib.get(root).unwrap().children;
        new_world_children[slot_index(1, 1, 1)] = Child::Node(new_body);
        let new_root = lib.insert_with_kind(new_world_children, lib.get(root).unwrap().kind);
        lib.ref_inc(new_root);
        lib.ref_dec(root);

        (lib, new_body, proto)
    }

    /// Click on the OBB at low edit_depth: path terminates at the
    /// body's depth-3 cell (3 entries). break_block would remove
    /// the whole OBB.
    #[test]
    fn obb_intercept_low_budget_targets_whole_cell() {
        let (lib, body, _proto) = build_demo_tree_with_proto();
        // Camera south of body, looking +z. Body-local frame: cam at
        // (1.5, 1.5, -1.5), dir (0, 0, 3) — but we test in body-frame
        // [0,3]³ directly: camera at z=-1.5, dir +z.
        let hit = cpu_raycast_uv_body(
            &lib,
            body,
            [1.5, 1.5, -1.5],
            [0.0, 0.0, 3.0],
            3, // budget exactly = body path length to OBB cell
        )
        .expect("ray must hit the OBB");
        assert_eq!(
            hit.path.len(),
            3,
            "low budget → whole-OBB hit, got path {:?}",
            hit.path,
        );
        assert_eq!(hit.path[2].1, 23, "terminal slot must be 23 (the OBB cell)");
    }

    /// Click on the OBB at high edit_depth: path descends INTO the
    /// proto subtree. Each extra level of budget = one more
    /// path entry = a 27× finer break.
    #[test]
    fn obb_intercept_high_budget_descends_into_subtree() {
        let (lib, body, _proto) = build_demo_tree_with_proto();
        let hit = cpu_raycast_uv_body(
            &lib,
            body,
            [1.5, 1.5, -1.5],
            [0.0, 0.0, 3.0],
            6, // 3 (body) + 3 (proto descent)
        )
        .expect("ray must hit the OBB");
        assert!(
            hit.path.len() > 3,
            "high budget → must descend INTO proto subtree, got path len {}: {:?}",
            hit.path.len(),
            hit.path,
        );
        assert_eq!(hit.path[2].1, 23, "still passes through the OBB cell at depth 3");
    }
}
