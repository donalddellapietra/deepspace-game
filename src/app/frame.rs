//! Render-frame helpers: walking the camera path to find the
//! active frame, transforming positions/AABBs into that frame.
//!
//! The "render frame" is the GPU's view of the world. The shader
//! starts ray marching at a frame root, with the camera expressed
//! in that frame's coordinates. Cartesian frames are linear
//! `[0, 3)³`; UV-sphere sub-cells carry their own `(φ, θ, r)`
//! origin + extents so the shader can keep cell-local math at any
//! zoom depth without leaning on absolute body-frame f32 precision.
//!
//! All functions here are **pure** — no `App` state — for direct
//! unit testing.

use crate::world::anchor::{Path, WORLD_SIZE};
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};

const TWO_PI: f32 = std::f32::consts::TAU;

/// Hard cap on UV-tier descent depth inside a `UvSphereBody`.
///
/// Currently `0`: the body-root marcher (`march_uv_sphere`) now uses
/// a stack-based DDA that propagates `un_*` cell-locally between
/// iterations — error stays at `1` cell-local ULP regardless of
/// descent depth. The sub-cell architecture's only job was hiding
/// the precision cliff of the previous per-iteration recompute;
/// with that fixed, sub-cell rendering is redundant and would
/// re-introduce coverage gaps (a frame at body-tree depth K only
/// covers `1 / 3^K` of the body's `(φ, θ, r)` range).
///
/// The `ActiveFrameKind::UvSubCell` variant and shader-side
/// `march_uv_subcell` stay in the codebase as dormant scaffolding
/// for any future LOD scheme that wants frame-local precision plus
/// a ribbon-pop mechanism.
const MAX_UV_FRAME_DEPTH: u8 = 0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    /// Render frame is rooted at a `NodeKind::UvSphereBody` node.
    /// The shader dispatches the UV-sphere DDA at depth==0; body
    /// params are read shader-side from
    /// `node_kinds[uniforms.root_index]`.
    UvSphereBody {
        inner_r: f32,
        outer_r: f32,
        theta_cap: f32,
    },
    /// Render frame is a UV sub-cell INSIDE a `UvSphereBody`. The
    /// frame node itself is a Cartesian-style 27-children node (the
    /// sub-cells produced by the body's recursive UV worldgen), but
    /// the cell's geometry is curved — children at this frame are
    /// addressed by `(pt, tt, rt)` UV tiers, not `(sx, sy, sz)`
    /// cartesian sub-cells.
    ///
    /// The frame carries the cell's range in the body's spherical
    /// coords so the shader can rebuild the local Jacobian without
    /// re-walking the full body tree from the root each iteration —
    /// every cell-local quantity stays at f32 ULPs of `[0, 1]` cell
    /// fractions regardless of how deep the frame is.
    UvSubCell {
        /// BFS index of the enclosing `UvSphereBody` node. Edit-side
        /// raycasts dispatch the body-rooted UV walker against this
        /// node — they don't know how to start mid-descent yet.
        body_node_id: NodeId,
        /// Depth at which the body node sits in the active frame's
        /// `render_path`. `render_path[..body_path_depth]` is the
        /// cartesian prefix to the body; `render_path[body_path_depth..]`
        /// is the UV-tier descent inside it.
        body_path_depth: u8,
        body_inner_r: f32,
        body_outer_r: f32,
        body_theta_cap: f32,
        phi_min: f32,
        theta_min: f32,
        r_min: f32,
        dphi: f32,
        dth: f32,
        dr: f32,
    },
}

impl ActiveFrame {
    /// Path used by Cartesian-frame helpers (`WorldPos::in_frame`,
    /// camera-local raycast). For Cartesian and body-root frames the
    /// active frame IS cartesian, so return `render_path` directly.
    /// For a UV sub-cell, the render path's tail is UV-tier slots that
    /// don't correspond to cartesian sub-cells — the cartesian-safe
    /// frame is the body's path (the prefix before any UV descent).
    pub fn cartesian_path(&self) -> Path {
        match self.kind {
            ActiveFrameKind::Cartesian | ActiveFrameKind::UvSphereBody { .. } => {
                self.render_path
            }
            ActiveFrameKind::UvSubCell { body_path_depth, .. } => {
                self.render_path.with_truncated(body_path_depth)
            }
        }
    }

    /// Node id at the cartesian-safe frame level. Same idea as
    /// `cartesian_path` — for UV sub-cells, return the enclosing
    /// body node, not the sub-cell node.
    pub fn cartesian_node_id(&self) -> NodeId {
        match self.kind {
            ActiveFrameKind::Cartesian | ActiveFrameKind::UvSphereBody { .. } => {
                self.node_id
            }
            ActiveFrameKind::UvSubCell { body_node_id, .. } => body_node_id,
        }
    }

    /// `NodeKind` at the cartesian-safe frame level.
    pub fn cartesian_kind(&self) -> ActiveFrameKind {
        match self.kind {
            ActiveFrameKind::Cartesian => ActiveFrameKind::Cartesian,
            ActiveFrameKind::UvSphereBody { .. } => self.kind,
            ActiveFrameKind::UvSubCell {
                body_inner_r, body_outer_r, body_theta_cap, ..
            } => ActiveFrameKind::UvSphereBody {
                inner_r: body_inner_r,
                outer_r: body_outer_r,
                theta_cap: body_theta_cap,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveFrame {
    /// Path used by the linear ribbon / camera-local transforms.
    pub render_path: Path,
    /// Logical interaction/render layer path. Identical to
    /// `render_path` for the purely-Cartesian architecture.
    pub logical_path: Path,
    pub node_id: NodeId,
    pub kind: ActiveFrameKind,
}

/// Build a `Path` from the slot prefix the GPU ribbon walker
/// actually reached. This is the renderer's effective frame.
pub fn frame_from_slots(slots: &[u8]) -> Path {
    let mut frame = Path::root();
    for &slot in slots {
        frame.push(slot);
    }
    frame
}

/// Resolve the active frame.
///
/// Descends from `world_root` along `camera_anchor` for at most
/// `desired_depth` slot steps. Inside a UV-sphere body, slot picking
/// switches from cartesian (anchor-driven) to UV-tier (geometric):
/// the camera's body-frame position determines `(pt, tt, rt)` per
/// descent, since the anchor's slots beyond the body level are
/// cartesian-deepened and don't correspond to the body's
/// `(φ, θ, r)`-tiered child layout.
pub fn compute_render_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    camera_anchor: &Path,
    desired_depth: u8,
) -> ActiveFrame {
    let mut target = *camera_anchor;
    target.truncate(desired_depth);
    let mut node_id = world_root;
    let mut reached = Path::root();
    let mut kind = match library.get(world_root).map(|n| n.kind) {
        Some(NodeKind::UvSphereBody { inner_r, outer_r, theta_cap }) => {
            ActiveFrameKind::UvSphereBody { inner_r, outer_r, theta_cap }
        }
        _ => ActiveFrameKind::Cartesian,
    };

    // Track the body node + depth at which we entered the UV body.
    // Used as the start index for accumulating the camera's body-frame
    // position out of the anchor's cartesian-deepened slots inside
    // the body, and (later) as the cartesian-safe fallback frame for
    // edit-side raycasts that don't know how to descend into a UV
    // sub-cell yet.
    let mut body_path_depth: Option<u8> = match kind {
        ActiveFrameKind::UvSphereBody { .. } => Some(0),
        _ => None,
    };
    let mut body_node_id: Option<NodeId> = match kind {
        ActiveFrameKind::UvSphereBody { .. } => Some(node_id),
        _ => None,
    };

    let target_depth = target.depth() as usize;
    let mut k: usize = 0;
    while k < target_depth {
        match kind {
            ActiveFrameKind::Cartesian => {
                let Some(node) = library.get(node_id) else { break };
                let slot = target.slot(k) as usize;
                match node.children[slot] {
                    Child::Node(child_id) => {
                        reached.push(slot as u8);
                        node_id = child_id;
                        if let Some(child_node) = library.get(child_id) {
                            if let NodeKind::UvSphereBody { inner_r, outer_r, theta_cap } = child_node.kind {
                                kind = ActiveFrameKind::UvSphereBody {
                                    inner_r, outer_r, theta_cap,
                                };
                                body_path_depth = Some(reached.depth());
                                body_node_id = Some(child_id);
                            }
                        }
                        k += 1;
                    }
                    Child::Block(_) | Child::Empty | Child::EntityRef(_) => break,
                }
            }
            ActiveFrameKind::UvSphereBody { inner_r, outer_r, theta_cap } => {
                // Stop UV descent at MAX_UV_FRAME_DEPTH to keep
                // frame_dphi above f32 ULPs of body-frame phi —
                // see the constant's docstring for why.
                let body_d = body_path_depth.unwrap_or(reached.depth());
                let uv_descent_so_far = reached.depth().saturating_sub(body_d);
                if uv_descent_so_far >= MAX_UV_FRAME_DEPTH {
                    break;
                }
                // Descend into the body. UV ranges span the whole
                // body at this point; pick the tier the camera lies
                // in based on its body-frame spherical coords.
                let body_pos = body_frame_pos_from_anchor(camera_anchor, body_d as usize);
                let phi_min0 = 0.0;
                let theta_min0 = -theta_cap;
                let r_min0 = inner_r;
                let dphi0 = TWO_PI;
                let dth0 = 2.0 * theta_cap;
                let dr0 = outer_r - inner_r;
                let Some(slot) = uv_slot_geometric(
                    body_pos,
                    phi_min0, theta_min0, r_min0,
                    dphi0, dth0, dr0,
                ) else { break };
                let Some(node) = library.get(node_id) else { break };
                match node.children[slot as usize] {
                    Child::Node(child_id) => {
                        let (pt, tt, rt) = unpack_uv_slot(slot);
                        let new_dphi = dphi0 / 3.0;
                        let new_dth = dth0 / 3.0;
                        let new_dr = dr0 / 3.0;
                        let body_id = body_node_id.unwrap_or(node_id);
                        let body_d = body_path_depth.unwrap_or(reached.depth());
                        reached.push(slot);
                        node_id = child_id;
                        kind = ActiveFrameKind::UvSubCell {
                            body_node_id: body_id,
                            body_path_depth: body_d,
                            body_inner_r: inner_r,
                            body_outer_r: outer_r,
                            body_theta_cap: theta_cap,
                            phi_min: phi_min0 + (pt as f32) * new_dphi,
                            theta_min: theta_min0 + (tt as f32) * new_dth,
                            r_min: r_min0 + (rt as f32) * new_dr,
                            dphi: new_dphi,
                            dth: new_dth,
                            dr: new_dr,
                        };
                        k += 1;
                    }
                    Child::Block(_) | Child::Empty | Child::EntityRef(_) => break,
                }
            }
            ActiveFrameKind::UvSubCell {
                body_node_id: bid, body_path_depth: bd,
                body_inner_r, body_outer_r, body_theta_cap,
                phi_min, theta_min, r_min, dphi, dth, dr,
            } => {
                // Stop UV descent at MAX_UV_FRAME_DEPTH (see const).
                let uv_descent_so_far = reached.depth().saturating_sub(bd);
                if uv_descent_so_far >= MAX_UV_FRAME_DEPTH {
                    break;
                }
                let body_d = body_path_depth.unwrap_or(reached.depth());
                let body_pos = body_frame_pos_from_anchor(camera_anchor, body_d as usize);
                let Some(slot) = uv_slot_geometric(
                    body_pos,
                    phi_min, theta_min, r_min,
                    dphi, dth, dr,
                ) else { break };
                let Some(node) = library.get(node_id) else { break };
                match node.children[slot as usize] {
                    Child::Node(child_id) => {
                        let (pt, tt, rt) = unpack_uv_slot(slot);
                        let new_dphi = dphi / 3.0;
                        let new_dth = dth / 3.0;
                        let new_dr = dr / 3.0;
                        reached.push(slot);
                        node_id = child_id;
                        kind = ActiveFrameKind::UvSubCell {
                            body_node_id: bid,
                            body_path_depth: bd,
                            body_inner_r,
                            body_outer_r,
                            body_theta_cap,
                            phi_min: phi_min + (pt as f32) * new_dphi,
                            theta_min: theta_min + (tt as f32) * new_dth,
                            r_min: r_min + (rt as f32) * new_dr,
                            dphi: new_dphi,
                            dth: new_dth,
                            dr: new_dr,
                        };
                        k += 1;
                    }
                    Child::Block(_) | Child::Empty | Child::EntityRef(_) => break,
                }
            }
        }
    }
    ActiveFrame {
        render_path: reached,
        logical_path: reached,
        node_id,
        kind,
    }
}

pub fn with_render_margin(
    library: &NodeLibrary,
    world_root: NodeId,
    logical_path: &Path,
    render_margin: u8,
) -> ActiveFrame {
    let logical = compute_render_frame(library, world_root, logical_path, logical_path.depth());
    // Shell architecture: the render frame IS the innermost
    // shell root. The shader pops outward via the ribbon for
    // coarser context. Each shell has a bounded depth budget.
    let min_render_depth = logical.logical_path.depth();
    let render_depth = logical
        .logical_path
        .depth()
        .saturating_sub(render_margin)
        .max(min_render_depth);
    if render_depth == logical.logical_path.depth() {
        return logical;
    }

    let mut render_path = logical.logical_path;
    render_path.truncate(render_depth);
    let render = compute_render_frame(library, world_root, &render_path, render_depth);
    ActiveFrame {
        render_path: render.render_path,
        logical_path: logical.logical_path,
        node_id: render.node_id,
        kind: render.kind,
    }
}

// ----------------------------------------------------------- helpers

/// Cartesian slot accumulation of `anchor[body_path_depth..]` into
/// a body-frame `[0, 3)³` position. Inside a UV body, the camera's
/// `WorldPos` anchor is still cartesian-deepened (the WorldPos
/// machinery has no UV-tier semantics yet), so the slots past the
/// body level encode the camera's body-frame position to f32
/// precision — exactly what we need to pick the UV tier the camera
/// lies in geometrically.
///
/// Without an offset hint we centre the deepest cell. Tier picking
/// is robust to this within a single cell; only on exact boundaries
/// does the half-cell shift matter, and at deep anchor those
/// boundaries are sub-ULP.
fn body_frame_pos_from_anchor(anchor: &Path, body_path_depth: usize) -> [f32; 3] {
    let mut pos = [0.0f32; 3];
    let mut size = WORLD_SIZE;
    let depth = anchor.depth() as usize;
    let mut k = body_path_depth;
    while k < depth {
        let (sx, sy, sz) = slot_coords(anchor.slot(k) as usize);
        let child = size / 3.0;
        pos[0] += sx as f32 * child;
        pos[1] += sy as f32 * child;
        pos[2] += sz as f32 * child;
        size = child;
        k += 1;
    }
    pos[0] += 0.5 * size;
    pos[1] += 0.5 * size;
    pos[2] += 0.5 * size;
    pos
}

/// Pick the UV tier (slot 0..27) for a body-frame point within a
/// UV cell spanning
/// `(phi_min, phi_min+dphi) × (theta_min, theta_min+dth) × (r_min, r_min+dr)`.
/// The body's cartesian centre is `(1.5, 1.5, 1.5)` in the body's
/// `[0, 3)³` frame.
///
/// Returns `None` for degenerate inputs (point at the body centre,
/// non-finite coords).
fn uv_slot_geometric(
    body_frame_pos: [f32; 3],
    phi_min: f32, theta_min: f32, r_min: f32,
    dphi: f32, dth: f32, dr: f32,
) -> Option<u8> {
    let cx = body_frame_pos[0] - 1.5;
    let cy = body_frame_pos[1] - 1.5;
    let cz = body_frame_pos[2] - 1.5;
    let r = (cx * cx + cy * cy + cz * cz).sqrt();
    if !r.is_finite() || r < 1e-6 {
        return None;
    }
    let theta = (cy / r).clamp(-1.0, 1.0).asin();
    let mut phi = cz.atan2(cx);
    if phi < 0.0 { phi += TWO_PI; }
    let new_dphi = dphi / 3.0;
    let new_dth = dth / 3.0;
    let new_dr = dr / 3.0;
    let pt = (((phi - phi_min) / new_dphi).floor() as i32).clamp(0, 2) as u8;
    let tt = (((theta - theta_min) / new_dth).floor() as i32).clamp(0, 2) as u8;
    let rt = (((r - r_min) / new_dr).floor() as i32).clamp(0, 2) as u8;
    Some(pt + tt * 3 + rt * 9)
}

#[inline]
fn unpack_uv_slot(slot: u8) -> (u8, u8, u8) {
    let pt = slot % 3;
    let tt = (slot / 3) % 3;
    let rt = slot / 9;
    (pt, tt, rt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, uniform_children, NodeLibrary};

    fn cartesian_chain(depth: u8) -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..depth {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        lib.ref_inc(node);
        (lib, node)
    }

    // --------- compute_render_frame ---------

    #[test]
    fn render_frame_root_when_desired_depth_zero() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..3 { anchor.push(13); }
        let frame = compute_render_frame(&lib, root, &anchor, 0);
        assert_eq!(frame.render_path.depth(), 0);
        assert_eq!(frame.node_id, root);
    }

    #[test]
    fn render_frame_descends_through_cartesian() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..4 { anchor.push(13); }
        let frame = compute_render_frame(&lib, root, &anchor, 3);
        assert_eq!(frame.render_path.depth(), 3);
    }

    #[test]
    fn render_frame_truncates_when_camera_anchor_shallow() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        anchor.push(13);
        let frame = compute_render_frame(&lib, root, &anchor, 5);
        assert!(frame.render_path.depth() <= 1);
    }

    #[test]
    fn render_frame_stops_when_path_misses_node() {
        // Build root with a Block at slot 5 (not a Node).
        let mut lib = NodeLibrary::default();
        let mut root_children = empty_children();
        root_children[5] = Child::Block(crate::world::palette::block::STONE);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        let mut anchor = Path::root();
        anchor.push(5);
        anchor.push(0);
        let frame = compute_render_frame(&lib, root, &anchor, 2);
        assert_eq!(frame.render_path.depth(), 0, "Block child terminates descent");
    }

    #[test]
    fn frame_from_slots_builds_exact_prefix() {
        let slots = [13u8, 16u8, 4u8];
        let p = frame_from_slots(&slots);
        assert_eq!(p.depth(), slots.len() as u8);
        assert_eq!(p.as_slice(), &slots);
    }

    // UV sub-cell descent is dormant (`MAX_UV_FRAME_DEPTH = 0`) —
    // the body-root marcher's stack-based DDA covers all zoom depths
    // with cell-local precision, so the sub-cell variant has no
    // active rendering path. Tests for it land if/when ribbon-pop
    // scaffolding wakes the variant back up.
}
