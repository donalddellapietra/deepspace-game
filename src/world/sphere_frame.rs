//! Detect the SphereBody ancestor of the current render frame.
//!
//! When the camera's render frame is at or below a `NodeKind::SphereBody`
//! node, the shader uses the analytic cube→sphere rendering path at
//! `march` entry — same visual as external views of the planet. This
//! module answers: is there a SphereBody ancestor? If yes, where does
//! its inscribed sphere sit in render-frame-local `[0, WORLD_SIZE)³`
//! coords? The renderer uploads that as a uniform so the shader's
//! short-circuit works.

use crate::world::anchor::{Path, WORLD_SIZE};
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};

/// Inscribed-sphere descriptor for a SphereBody ancestor, expressed
/// in the current render frame's local `[0, WORLD_SIZE)³` coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SphereBodyInfo {
    pub center: [f32; 3],
    pub radius: f32,
}

/// Walk from `world_root` along `render_path` checking for the
/// deepest SphereBody ancestor. When found, compute the sphere body's
/// inscribed-sphere center/radius in render-frame-local coords.
///
/// Nesting policy: if multiple SphereBody nodes appear in the path,
/// the innermost one wins — we render the closest enclosing planet.
pub fn find_active_sphere_body(
    library: &NodeLibrary,
    world_root: NodeId,
    render_path: &[u8],
) -> Option<SphereBodyInfo> {
    // `state` tracks the current cell's bounds in the active SphereBody's
    // own body-shader frame `[0, WORLD_SIZE)³`:
    //   origin = lower corner of the current cell
    //   size   = edge length of the current cell (< WORLD_SIZE once below
    //            the SphereBody node).
    // `None` means we're not inside any SphereBody ancestor yet.
    let mut node_id = world_root;
    let mut state: Option<([f32; 3], f32)> = None;

    let is_sphere = |nid: NodeId, lib: &NodeLibrary| {
        lib.get(nid).is_some_and(|n| n.kind == NodeKind::SphereBody)
    };

    if is_sphere(world_root, library) {
        state = Some(([0.0; 3], WORLD_SIZE));
    }

    for &slot in render_path {
        let Some(node) = library.get(node_id) else { break };
        let Child::Node(child_id) = node.children[slot as usize] else { break };

        if let Some((origin, size)) = state {
            let (sx, sy, sz) = slot_coords(slot as usize);
            let child_size = size / 3.0;
            state = Some((
                [
                    origin[0] + sx as f32 * child_size,
                    origin[1] + sy as f32 * child_size,
                    origin[2] + sz as f32 * child_size,
                ],
                child_size,
            ));
        }

        node_id = child_id;

        if is_sphere(node_id, library) {
            // Entered a fresh SphereBody — reset to its top.
            state = Some(([0.0; 3], WORLD_SIZE));
        }
    }

    state.map(|(origin, cell_size)| {
        // Render frame's `[0, WORLD_SIZE)³` maps to body-shader
        // `[origin, origin + cell_size)³`. Scale from body-shader to
        // render-local = WORLD_SIZE / cell_size.
        let scale = WORLD_SIZE / cell_size;
        let sb_center_bs = WORLD_SIZE * 0.5;
        let sb_radius_bs = WORLD_SIZE * 0.5;
        SphereBodyInfo {
            center: [
                (sb_center_bs - origin[0]) * scale,
                (sb_center_bs - origin[1]) * scale,
                (sb_center_bs - origin[2]) * scale,
            ],
            radius: sb_radius_bs * scale,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, uniform_children, Child, NodeLibrary, CENTER_SLOT};

    fn make_sphere_world() -> (NodeLibrary, NodeId) {
        // Mirror the MVP worldgen: Cartesian wrapper with SphereBody at
        // center slot, empty elsewhere.
        let mut lib = NodeLibrary::default();
        let stone = lib.insert(uniform_children(Child::Block(
            crate::world::palette::block::STONE,
        )));
        let sphere =
            lib.insert_with_kind(uniform_children(Child::Node(stone)), NodeKind::SphereBody);
        let mut root_children = empty_children();
        root_children[CENTER_SLOT] = Child::Node(sphere);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        (lib, root)
    }

    #[test]
    fn none_when_render_frame_is_wrapper() {
        let (lib, root) = make_sphere_world();
        assert!(find_active_sphere_body(&lib, root, &[]).is_none());
    }

    #[test]
    fn active_when_render_frame_is_sphere_body_itself() {
        let (lib, root) = make_sphere_world();
        let info = find_active_sphere_body(&lib, root, &[CENTER_SLOT as u8]).unwrap();
        // Render frame = SphereBody node itself. Sphere center = middle of
        // render frame, radius = half the render frame.
        assert_eq!(info.center, [WORLD_SIZE * 0.5; 3]);
        assert_eq!(info.radius, WORLD_SIZE * 0.5);
    }

    #[test]
    fn scales_when_render_frame_is_deeper_cell() {
        // Build a world where the SphereBody has a Cartesian interior
        // we can descend into (mimics a future carved sphere — for now
        // the sphere_worldgen uses uniform-stone nodes which pack
        // flattens, so we can't actually descend below it in practice).
        // This test just exercises the math.
        let mut lib = NodeLibrary::default();
        let stone = lib.insert(uniform_children(Child::Block(
            crate::world::palette::block::STONE,
        )));
        // Nested Cartesian inside SphereBody so find_active walks further.
        let mut inner_children = empty_children();
        inner_children[CENTER_SLOT] = Child::Node(stone);
        let inner = lib.insert(inner_children);
        let sphere =
            lib.insert_with_kind(uniform_children(Child::Node(inner)), NodeKind::SphereBody);
        let mut root_children = empty_children();
        root_children[CENTER_SLOT] = Child::Node(sphere);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        // render_path = [CENTER_SLOT, CENTER_SLOT]: wrapper → sphere →
        // inner (center child of sphere). Render frame = inner cell.
        // Inner cell in sphere body's body-shader frame: [1, 2)³, size 1.
        let path = [CENTER_SLOT as u8, CENTER_SLOT as u8];
        let info = find_active_sphere_body(&lib, root, &path).unwrap();
        // Scale from body-shader to render-local = WORLD_SIZE / 1 = 3.
        // Sphere body center in body-shader = 1.5. Cell origin = (1,1,1).
        // Center in render-local = (1.5 - 1) * 3 = 1.5. Radius 1.5 * 3 = 4.5.
        assert!((info.center[0] - 1.5).abs() < 1e-5);
        assert!((info.radius - 4.5).abs() < 1e-5);
    }
}
