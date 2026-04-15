//! Event-loop application struct and top-level frame driver.

use std::sync::Arc;

use winit::window::Window;

use crate::camera::Camera;
use crate::game_state::{GameUiState, SavedMeshes};
use crate::input::Keys;
use crate::player;
use crate::renderer::Renderer;
use crate::world::anchor::{Path, WorldPos, WORLD_SIZE};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{Child, NodeId, NodeKind, MAX_DEPTH};

/// Levels shallower than the camera's anchor at which the render
/// frame sits. The frame walks down the camera's path until either
/// (a) it reaches `anchor_depth - RENDER_FRAME_K`, or (b) it would
/// cross into a non-Cartesian node — whichever happens first. The
/// non-Cartesian stop is required because the shader's main DDA
/// only knows how to march through Cartesian children at the frame
/// root (sphere body / face-cell roots are next-session work).
///
/// `RENDER_FRAME_MAX_DEPTH` was the historical pin at root (0) used
/// to validate the sphere dispatch; with the precision rewrite in
/// place it's now `MAX_DEPTH` so the walker can descend freely
/// through Cartesian zones.
pub const RENDER_FRAME_K: u8 = 3;
pub const RENDER_FRAME_MAX_DEPTH: u8 = MAX_DEPTH as u8;

pub mod cursor;
pub mod edit_actions;
pub mod event_loop;
pub mod input_handlers;
#[cfg(not(target_arch = "wasm32"))]
pub mod overlay_integration;

pub struct App {
    pub(super) window: Option<Arc<Window>>,
    pub(super) renderer: Option<Renderer>,
    pub(super) camera: Camera,
    pub(super) world: WorldState,
    pub(super) frozen: bool,
    pub(super) cursor_locked: bool,
    pub(super) keys: Keys,
    pub(super) last_frame: std::time::Instant,
    pub(super) tree_depth: u32,
    pub(super) palette: ColorRegistry,
    pub(super) saved_meshes: SavedMeshes,
    pub(super) save_mode: bool,
    pub(super) ui: GameUiState,
    pub(super) debug_overlay_visible: bool,
    pub(super) fps_smooth: f64,
    /// Path from `world.root` to the planet's body node. Used for
    /// spawn-position derivation and as a hint for future debug
    /// teleport / cursor logic; rendering reads the body via the
    /// normal tree walk + `NodeKind` dispatch.
    #[allow(dead_code)]
    pub(super) planet_path: Path,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) webview: Option<wry::WebView>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) frames_waited: u32,
}

impl App {
    pub fn new() -> Self {
        // Build a Cartesian world tree, then insert the planet body
        // into its central depth-1 cell. After install, the planet
        // is a `NodeKind::CubedSphereBody` node living inside the
        // tree — there's no parallel `cs_planet` handle.
        let mut world = crate::world::worldgen::generate_world();
        let setup = crate::world::spherical_worldgen::demo_planet();
        let (new_root, planet_path) =
            crate::world::spherical_worldgen::install_at_root_center(
                &mut world.library, world.root, &setup,
            );
        world.swap_root(new_root);
        let tree_depth = world.tree_depth();
        eprintln!(
            "Planet inserted at path {:?}; library has {} nodes, depth {}",
            planet_path.as_slice(), world.library.len(), tree_depth,
        );

        // Spawn just above the planet's body cell. Body cell at
        // depth 1 slot 13 spans world `[1, 2)³`. Sphere outer_r
        // local = `setup.outer_r`, world = `outer_r * 1.0` (cell
        // size is 1 world unit at depth 1). Top of sphere at world
        // y = 1.5 + outer_r. Spawn slightly above.
        let body_top_y = 1.5 + setup.outer_r;
        let spawn_xyz = [1.5, (body_top_y + 0.05).min(WORLD_SIZE - 0.001), 1.5];
        debug_assert!(spawn_xyz.iter().all(|&v| (0.0..WORLD_SIZE).contains(&v)));

        let anchor_depth = ((tree_depth as i32 - 6 + 1).max(1) as u8).min(60);
        let position = WorldPos::from_world_xyz(spawn_xyz, anchor_depth);

        Self {
            window: None,
            renderer: None,
            camera: Camera {
                position,
                smoothed_up: [0.0, 1.0, 0.0],
                yaw: 0.0,
                // Steep pitch so the planet (below camera) fills
                // the lower half of the view at spawn.
                pitch: -1.2,
            },
            world,
            frozen: false,
            cursor_locked: false,
            keys: Keys::default(),
            last_frame: std::time::Instant::now(),
            tree_depth,
            palette: ColorRegistry::new(),
            saved_meshes: SavedMeshes::default(),
            save_mode: false,
            ui: GameUiState::new(),
            debug_overlay_visible: false,
            fps_smooth: 0.0,
            planet_path,
            #[cfg(not(target_arch = "wasm32"))]
            webview: None,
            #[cfg(not(target_arch = "wasm32"))]
            frames_waited: 0,
        }
    }

    #[inline]
    pub(super) fn anchor_depth(&self) -> u32 {
        self.camera.position.anchor.depth() as u32
    }

    #[inline]
    pub(super) fn zoom_level(&self) -> i32 {
        (self.tree_depth as i32) - (self.anchor_depth() as i32) + 1
    }

    /// Walk the world tree to find the render-frame's NodeId.
    ///
    /// The frame can be **Cartesian** (shader runs the XYZ DDA
    /// from it) or **CubedSphereBody** (shader dispatches into
    /// sphere DDA at start-of-march, body fills the `[0, 3)³`
    /// frame). `CubedSphereFace` frames are out of scope this
    /// pass — the walker stops before entering a face cell.
    ///
    /// Cartesian descent is now safe because the GPU pack includes
    /// an ancestor ribbon back to the absolute world root, and
    /// the shader pops upward when rays exit the frame's [0, 3)³
    /// bubble. So content outside the frame stays visible: the
    /// planet at root.children[13] still renders even when the
    /// camera frame is deep in an empty Cartesian sibling subtree.
    pub(super) fn render_frame(&self) -> (Path, NodeId) {
        let desired_depth = (self.anchor_depth().saturating_sub(RENDER_FRAME_K as u32) as u8)
            .min(RENDER_FRAME_MAX_DEPTH);
        compute_render_frame(
            &self.world.library, self.world.root,
            &self.camera.position.anchor, desired_depth,
        )
    }

    /// `NodeKind` of the current render-frame root. Renderer uses
    /// this to set the `root_kind` uniform so the shader knows
    /// whether to enter the standard Cartesian DDA or dispatch
    /// directly into the sphere DDA at start-of-march.
    pub(super) fn render_frame_kind(&self) -> NodeKind {
        let (_, node_id) = self.render_frame();
        self.world.library.get(node_id)
            .map(|n| n.kind)
            .unwrap_or(NodeKind::Cartesian)
    }

    pub(super) fn update(&mut self, dt: f32) {
        player::update(&mut self.camera, dt);

        let (frame, _) = self.render_frame();
        let cam_local = self.camera.position.in_frame(&frame);
        if let Some(renderer) = &self.renderer {
            renderer.update_camera(&self.camera.gpu_camera_at(cam_local, 1.2));
        }
    }

    pub(super) fn step_chunk(&mut self, axis: usize, direction: i32) {
        if self.frozen { return; }
        self.camera.position.anchor.step_neighbor_cartesian(axis, direction);
        self.camera.position.offset = [0.5, 0.5, 0.5];
    }

    pub fn debug_teleport(&mut self, slots: &[u8], offset: [f32; 3]) {
        let mut anchor = Path::root();
        for &s in slots.iter().take(crate::world::tree::MAX_DEPTH) {
            anchor.push(s);
        }
        self.camera.position = WorldPos::new(anchor, offset);
        self.apply_zoom();
    }

    pub(super) fn log_location(&self) {
        let p = &self.camera.position;
        log::info!(
            "camera anchor depth={} slots={:?} offset={:?} world={:?}",
            p.anchor.depth(),
            p.anchor.as_slice(),
            p.offset,
            p.to_world_xyz(),
        );
    }
}

/// World-space origin and size of the cell at `path`. Origin is
/// the world XYZ of the cell's `(0, 0, 0)` corner; size is its
/// side length in world units. Used to transform between world
/// coords and frame-local shader coords.
pub fn frame_origin_size_world(path: &Path) -> ([f32; 3], f32) {
    let mut origin = [0.0f32; 3];
    let mut size = WORLD_SIZE;
    for k in 0..path.depth() as usize {
        let (sx, sy, sz) = crate::world::tree::slot_coords(path.slot(k) as usize);
        let child = size / 3.0;
        origin[0] += sx as f32 * child;
        origin[1] += sy as f32 * child;
        origin[2] += sz as f32 * child;
        size = child;
    }
    (origin, size)
}

/// Transform an axis-aligned bounding box from world coordinates
/// into the shader-frame coords for `frame_path`. Frame's cell
/// maps to the shader's `[0, 3)³`, so the scale factor is
/// `WORLD_SIZE / frame_cell_size_world`.
pub fn aabb_world_to_frame(
    frame_path: &Path,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let (frame_origin, frame_size) = frame_origin_size_world(frame_path);
    let scale = WORLD_SIZE / frame_size;
    (
        [
            (aabb_min[0] - frame_origin[0]) * scale,
            (aabb_min[1] - frame_origin[1]) * scale,
            (aabb_min[2] - frame_origin[2]) * scale,
        ],
        [
            (aabb_max[0] - frame_origin[0]) * scale,
            (aabb_max[1] - frame_origin[1]) * scale,
            (aabb_max[2] - frame_origin[2]) * scale,
        ],
    )
}

/// Pure render-frame walker: descends the camera's path, accepting
/// Cartesian or CubedSphereBody nodes and stopping at face cells.
/// Extracted from `App::render_frame` for testability.
pub fn compute_render_frame(
    library: &crate::world::tree::NodeLibrary,
    world_root: NodeId,
    camera_anchor: &Path,
    desired_depth: u8,
) -> (Path, NodeId) {
    let mut frame = *camera_anchor;
    frame.truncate(desired_depth);
    let mut node_id = world_root;
    let mut reached = 0u8;
    for k in 0..frame.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = frame.slot(k) as usize;
        match node.children[slot] {
            Child::Node(child_id) => {
                let Some(child) = library.get(child_id) else { break };
                match child.kind {
                    NodeKind::Cartesian | NodeKind::CubedSphereBody { .. } => {
                        node_id = child_id;
                        reached = (k as u8) + 1;
                    }
                    NodeKind::CubedSphereFace { .. } => break,
                }
            }
            Child::Block(_) | Child::Empty => break,
        }
    }
    frame.truncate(reached);
    (frame, node_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, slot_index, uniform_children, Child, NodeKind};

    fn cartesian_chain(depth: u8) -> (crate::world::tree::NodeLibrary, NodeId) {
        let mut lib = crate::world::tree::NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..depth {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        lib.ref_inc(node);
        (lib, node)
    }

    #[test]
    fn render_frame_root_when_desired_depth_zero() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..3 {
            anchor.push(13);
        }
        let (frame, node_id) = compute_render_frame(&lib, root, &anchor, 0);
        assert_eq!(frame.depth(), 0);
        assert_eq!(node_id, root);
    }

    #[test]
    fn render_frame_descends_through_cartesian() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..4 {
            anchor.push(13);
        }
        let (frame, _node_id) = compute_render_frame(&lib, root, &anchor, 3);
        assert_eq!(frame.depth(), 3,
            "should descend 3 levels through Cartesian");
    }

    #[test]
    fn render_frame_stops_at_face_cell() {
        // Build: root (Cartesian) → body (CubedSphereBody) → face
        // subtree (CubedSphereFace). Frame should stop at body.
        let mut lib = crate::world::tree::NodeLibrary::default();
        let face = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereFace {
                face: crate::world::cubesphere::Face::PosX,
            },
        );
        let mut body_children = empty_children();
        body_children[14] = Child::Node(face);  // +X face slot
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        // Camera anchor descends body → face: slots [13, 14].
        let mut anchor = Path::root();
        anchor.push(13);
        anchor.push(14);

        let (frame, node_id) = compute_render_frame(&lib, root, &anchor, 2);
        assert_eq!(frame.depth(), 1, "stopped before face cell");
        assert_eq!(node_id, body, "frame at body");
    }

    #[test]
    fn render_frame_descends_into_body() {
        // Build: root → body (CubedSphereBody, no face children).
        let mut lib = crate::world::tree::NodeLibrary::default();
        let body = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(13);
        let (frame, node_id) = compute_render_frame(&lib, root, &anchor, 1);
        assert_eq!(frame.depth(), 1);
        assert_eq!(node_id, body);
    }

    #[test]
    fn render_frame_truncates_when_camera_anchor_shallow() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        anchor.push(13); // depth 1
        // desired depth = 5 but camera anchor is only depth 1.
        let (frame, _node_id) = compute_render_frame(&lib, root, &anchor, 5);
        // Frame can be at most camera anchor depth (after truncate).
        assert!(frame.depth() <= 1);
    }

    // --------- frame_origin_size_world ---------

    #[test]
    fn frame_origin_size_root() {
        let p = Path::root();
        let (origin, size) = frame_origin_size_world(&p);
        assert_eq!(origin, [0.0, 0.0, 0.0]);
        assert!((size - WORLD_SIZE).abs() < 1e-6);
    }

    #[test]
    fn frame_origin_size_center_slot() {
        let mut p = Path::root();
        p.push(slot_index(1, 1, 1) as u8);  // body's typical slot
        let (origin, size) = frame_origin_size_world(&p);
        // World [0, 3): center cell occupies [1, 2)³.
        assert!((origin[0] - 1.0).abs() < 1e-6);
        assert!((origin[1] - 1.0).abs() < 1e-6);
        assert!((origin[2] - 1.0).abs() < 1e-6);
        assert!((size - 1.0).abs() < 1e-6);
    }

    #[test]
    fn frame_origin_size_corner_slot() {
        let mut p = Path::root();
        p.push(slot_index(0, 0, 0) as u8);
        let (origin, size) = frame_origin_size_world(&p);
        assert_eq!(origin, [0.0, 0.0, 0.0]);
        assert!((size - 1.0).abs() < 1e-6);
    }

    #[test]
    fn frame_origin_size_two_levels_deep() {
        let mut p = Path::root();
        p.push(slot_index(2, 2, 2) as u8);  // [2, 3)
        p.push(slot_index(1, 1, 1) as u8);  // [2 + 1/3, 2 + 2/3)
        let (_origin, size) = frame_origin_size_world(&p);
        assert!((size - (1.0 / 3.0)).abs() < 1e-6);
    }

    // --------- aabb_world_to_frame ---------

    #[test]
    fn aabb_root_frame_is_identity() {
        let p = Path::root();
        let (mn, mx) = aabb_world_to_frame(&p, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
        assert!((mn[0] - 0.5).abs() < 1e-6);
        assert!((mx[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn aabb_body_frame_scales_3x() {
        let mut p = Path::root();
        p.push(slot_index(1, 1, 1) as u8);  // body
        // Body occupies world [1, 2). AABB at world [1.4, 1.6) maps
        // to body-local [(1.4-1)*3, (1.6-1)*3) = [1.2, 1.8).
        let (mn, mx) = aabb_world_to_frame(&p, [1.4, 1.4, 1.4], [1.6, 1.6, 1.6]);
        assert!((mn[0] - 1.2).abs() < 1e-5);
        assert!((mx[0] - 1.8).abs() < 1e-5);
    }

    #[test]
    fn aabb_corners_round_trip() {
        // Inverse transform: frame_local + frame_origin = world.
        // Verify by reconstructing.
        let mut p = Path::root();
        p.push(slot_index(2, 2, 2) as u8);
        let world_min = [2.5, 2.5, 2.5];
        let world_max = [2.7, 2.7, 2.7];
        let (mn, mx) = aabb_world_to_frame(&p, world_min, world_max);
        let (frame_origin, frame_size) = frame_origin_size_world(&p);
        let back_min = [
            mn[0] * frame_size / WORLD_SIZE + frame_origin[0],
            mn[1] * frame_size / WORLD_SIZE + frame_origin[1],
            mn[2] * frame_size / WORLD_SIZE + frame_origin[2],
        ];
        let back_max = [
            mx[0] * frame_size / WORLD_SIZE + frame_origin[0],
            mx[1] * frame_size / WORLD_SIZE + frame_origin[1],
            mx[2] * frame_size / WORLD_SIZE + frame_origin[2],
        ];
        for i in 0..3 {
            assert!((back_min[i] - world_min[i]).abs() < 1e-5);
            assert!((back_max[i] - world_max[i]).abs() < 1e-5);
        }
    }
}
