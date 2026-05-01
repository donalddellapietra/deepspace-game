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
use crate::world::tree::{slot_index, Child, NodeId, NodeKind, NodeLibrary};

const TAU: f32 = std::f32::consts::TAU;

/// Cast a ray through a UV-sphere body subtree. Returns `None` on
/// miss (ray doesn't intersect the body shell, or exits without
/// finding a Block within `max_iter` cell-boundary steps).
///
/// `max_descent_depth` caps the descent inside any one DDA step; the
/// natural cap is the body subtree's leaf depth. The shader uses 63
/// (= tree's MAX_DEPTH); CPU mirrors that.
pub fn cpu_raycast_uv_body(
    library: &NodeLibrary,
    body_root_id: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    max_descent_depth: u32,
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
        // (phi_w, theta_w, r_w). Stops on Block (HIT) or Empty.
        let descent = descend(
            library,
            body_root_id,
            inner_r,
            outer_r,
            theta_cap,
            phi_w,
            theta_w,
            r_w,
            max_descent_depth,
        )?;

        if descent.found_block {
            return Some(HitInfo {
                path: descent.path,
                face: descent.face,
                t,
                place_path: None,
            });
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
    let mut phi_lo = 0.0_f32;
    let mut phi_hi = TAU;
    let mut theta_lo = -body_theta_cap;
    let mut theta_hi = body_theta_cap;
    let mut r_lo = body_inner_r;
    let mut r_hi = body_outer_r;
    let mut node_id = body_root_id;
    let mut path: Vec<(NodeId, usize)> = Vec::new();

    for _depth in 0..max_depth {
        let dphi = (phi_hi - phi_lo) / 3.0;
        let dth = (theta_hi - theta_lo) / 3.0;
        let dr = (r_hi - r_lo) / 3.0;

        let pt = (((phi_w - phi_lo) / dphi.max(1e-12)).floor().clamp(0.0, 2.0)) as usize;
        let tt = (((theta_w - theta_lo) / dth.max(1e-12)).floor().clamp(0.0, 2.0)) as usize;
        let rt = (((r_w - r_lo) / dr.max(1e-12)).floor().clamp(0.0, 2.0)) as usize;
        let slot = slot_index(pt, tt, rt);

        let node = library.get(node_id)?;
        path.push((node_id, slot));

        // Refine bounds to the (pt, tt, rt) cell.
        phi_lo = phi_lo + pt as f32 * dphi;
        phi_hi = phi_lo + dphi;
        theta_lo = theta_lo + tt as f32 * dth;
        theta_hi = theta_lo + dth;
        r_lo = r_lo + rt as f32 * dr;
        r_hi = r_lo + dr;

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
                });
            }
            Child::Node(child_id) => {
                node_id = child_id;
                continue;
            }
        }
    }
    // Hit the descent cap without resolving — treat as empty for
    // safety. The DDA will step to the next cell boundary; if that
    // continues for the full iter budget, the caller returns None.
    Some(Descent {
        found_block: false,
        path,
        face: 0,
        phi_lo,
        phi_hi,
        theta_lo,
        theta_hi,
        r_lo,
        r_hi,
    })
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
}
