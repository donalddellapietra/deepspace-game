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
use crate::world::tree::{Child, NodeId};

/// Stride between consecutive ribbon-frame depths. Every K levels
/// along the camera's anchor path gets its own render frame, so
/// nested ribbons cover the camera's vicinity with sufficiently
/// fine scale while leaving breadcrumb ancestors for far-field
/// silhouette / planet rendering.
pub const RIBBON_K: u8 = 3;

/// Maximum number of simultaneous ribbon frames sent to the shader.
/// Has to match the fixed-size array in `GpuUniforms.ribbon` and
/// the shader's `ribbon: array<RibbonFrame, MAX_RIBBON_FRAMES>`.
pub const MAX_RIBBON_FRAMES: usize = 8;

/// One frame in the camera's ribbon — a node in the tree whose cell
/// is expressed in bounded `[0, WORLD_SIZE)^3` local coordinates.
/// The shader marches each frame independently and composites by
/// smallest world-scale t so frames at different scales coexist
/// without seams.
#[derive(Clone, Copy, Debug)]
pub struct RibbonFrame {
    pub path: Path,
    pub node_id: NodeId,
    /// Camera position expressed in this frame's `[0, WORLD_SIZE)^3`
    /// local coordinates via `WorldPos::in_frame`. Path-exact, so
    /// precision stays bounded at any anchor depth.
    pub camera_local: [f32; 3],
    /// `(1/3)^frame_depth` — multiplier from this frame's
    /// local t-units to world t-units. Deeper frames have smaller
    /// world_scale; the ribbon compositor picks the hit with the
    /// smallest `t * world_scale`.
    pub world_scale: f32,
}

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

        // Spawn just above the planet body's +Y face. Path-anchored:
        // body center is `WorldPos::new(planet_path, [0.5, 0.5, 0.5])`
        // and the spawn is `setup.outer_r + small_margin` above it
        // along world +Y, expressed in the body-cell-local frame.
        // Avoids hardcoded `[1.5, ..., 1.5]` so the spawn follows
        // the planet wherever its path lives.
        let body_cell_size_world =
            crate::world::anchor::WORLD_SIZE / 3.0_f32.powi(planet_path.depth() as i32);
        let above_offset_world = setup.outer_r * body_cell_size_world + 0.05 * body_cell_size_world;
        let body_center =
            WorldPos::new(planet_path, [0.5, 0.5, 0.5]).to_world_xyz();
        let spawn_xyz = [
            body_center[0],
            (body_center[1] + above_offset_world).min(WORLD_SIZE - 0.001),
            body_center[2],
        ];
        debug_assert!(spawn_xyz.iter().all(|&v| (0.0..WORLD_SIZE).contains(&v)));

        // Spawn anchor depth: 6 levels shallower than the tree's
        // depth so the user starts at a moderate zoom (sees the
        // planet at near-full screen). The `-6` is a UX-tuning
        // constant, not a precision/system cap.
        const SPAWN_ZOOM_OFFSET: i32 = 6;
        let anchor_depth = ((tree_depth as i32 - SPAWN_ZOOM_OFFSET + 1)
            .max(1) as u8)
            .min(crate::world::tree::MAX_DEPTH as u8);
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

    /// Build the camera's ribbon: a chain of frames from the root
    /// down toward the camera's anchor, stepping `RIBBON_K` levels
    /// per frame. The deepest frame is emitted first (for early-exit
    /// efficiency in the shader's compositor).
    ///
    /// Every frame's path is a prefix of the camera's anchor, so each
    /// frame's cell geometrically contains the camera. Frames share
    /// content via the content-addressed library, so the GPU buffer
    /// pack doesn't duplicate nodes — each frame just records the
    /// buffer index of its root node.
    ///
    /// Truncates the chain at the first path slot where the tree
    /// walk from `world.root` hits a terminal — beyond that point
    /// there's no NodeId to pack.
    pub(super) fn render_ribbon(&self) -> Vec<RibbonFrame> {
        build_ribbon(&self.world.library, self.world.root, &self.camera.position)
    }
}

/// Pure ribbon-construction function, testable without an `App`.
/// Produces the ribbon for a camera at `camera` in the tree rooted
/// at `world_root`. Returns frames deepest-first, with frames whose
/// path crosses into a sphere body / face subtree filtered out.
/// Always returns at least one frame (the root) so the sky + any
/// Cartesian content renders even if no deep frame qualifies.
pub fn build_ribbon(
    library: &crate::world::tree::NodeLibrary,
    world_root: NodeId,
    camera: &WorldPos,
) -> Vec<RibbonFrame> {
    use crate::world::tree::NodeKind;

    let anchor = camera.anchor;
    let max_reachable_depth = {
        let mut node_id = world_root;
        let mut reached = 0u8;
        for k in 0..anchor.depth() as usize {
            let Some(node) = library.get(node_id) else { break };
            let slot = anchor.slot(k) as usize;
            match node.children[slot] {
                Child::Node(cid) => {
                    node_id = cid;
                    reached = (k as u8) + 1;
                }
                Child::Block(_) | Child::Empty => break,
            }
        }
        reached
    };

    let mut frames: Vec<RibbonFrame> = Vec::new();
    let mut emitted_depths: Vec<u8> = Vec::new();
    let mut d = 0u8;
    loop {
        let frame_depth = d.min(max_reachable_depth);
        if !emitted_depths.contains(&frame_depth) {
            emitted_depths.push(frame_depth);
        }
        if d >= max_reachable_depth { break; }
        d = d.saturating_add(RIBBON_K);
    }

    // Order deepest first — the shader early-exits when a nearer
    // hit has been found, and deep frames tend to contain the
    // foreground.
    emitted_depths.sort();
    emitted_depths.reverse();

    // Cap to MAX_RIBBON_FRAMES by dropping shallowest (keeping root).
    while emitted_depths.len() > MAX_RIBBON_FRAMES {
        let penultimate = emitted_depths.len() - 2;
        emitted_depths.remove(penultimate);
    }

    for &depth in &emitted_depths {
        let mut path = anchor;
        path.truncate(depth);

        // Walk world_root to resolve the node at this path AND
        // detect whether the walk crosses a non-Cartesian node.
        // Frames whose path passes through a sphere body / face
        // subtree are SKIPPED — the global sphere pass renders that
        // content. Per-frame sphere DDA was tried in 343f464 but
        // reverted (linearization-at-camera produced z-fighting +
        // seams against the global pass).
        let mut node_id = world_root;
        let mut inside_sphere = false;
        for k in 0..path.depth() as usize {
            let Some(node) = library.get(node_id) else { break };
            if !matches!(node.kind, NodeKind::Cartesian) {
                inside_sphere = true;
                break;
            }
            let slot = path.slot(k) as usize;
            match node.children[slot] {
                Child::Node(cid) => { node_id = cid; }
                _ => break,
            }
        }
        if !inside_sphere {
            if let Some(node) = library.get(node_id) {
                if !matches!(node.kind, NodeKind::Cartesian) {
                    inside_sphere = true;
                }
            }
        }
        if inside_sphere {
            continue;
        }

        let camera_local = camera.in_frame(&path);
        let world_scale = (1.0_f32 / 3.0_f32).powi(depth as i32);

        frames.push(RibbonFrame {
            path,
            node_id,
            camera_local,
            world_scale,
        });
    }

    // Always keep at least the root frame so the Cartesian space-
    // tree (sky + any Cartesian content) gets rendered.
    if frames.is_empty() {
        let camera_local = camera.in_frame(&Path::root());
        frames.push(RibbonFrame {
            path: Path::root(),
            node_id: world_root,
            camera_local,
            world_scale: 1.0,
        });
    }

    frames
}

impl App {
    pub(super) fn update(&mut self, dt: f32) {
        player::update(&mut self.camera, dt);

        let world_pos = self.camera.world_pos_f32();
        if let Some(renderer) = &self.renderer {
            renderer.update_camera(&self.camera.gpu_camera_at(world_pos, 1.2));
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

#[cfg(test)]
mod ribbon_tests {
    use super::*;
    use crate::world::cubesphere::insert_spherical_body;
    use crate::world::sdf::Planet;
    use crate::world::palette::block;
    use crate::world::spherical_worldgen;
    use crate::world::tree::{empty_children, slot_index, Child as TChild, NodeLibrary};
    use crate::world::worldgen;

    /// Build the demo world with a planet at root slot 13. Mirrors
    /// `App::new`'s setup so ribbon tests exercise realistic data.
    fn demo_world() -> (NodeLibrary, NodeId, Path) {
        let mut world = worldgen::generate_world();
        let setup = spherical_worldgen::demo_planet();
        let (new_root, planet_path) =
            spherical_worldgen::install_at_root_center(
                &mut world.library, world.root, &setup,
            );
        world.swap_root(new_root);
        (world.library, world.root, planet_path)
    }

    #[test]
    fn ribbon_emits_at_least_one_frame_for_root_camera() {
        let mut lib = NodeLibrary::default();
        let root = lib.insert(empty_children());
        lib.ref_inc(root);
        let camera = WorldPos::new(Path::root(), [0.5; 3]);
        let frames = build_ribbon(&lib, root, &camera);
        assert_eq!(frames.len(), 1, "root-only camera should produce 1 frame");
        assert_eq!(frames[0].path.depth(), 0);
        assert_eq!(frames[0].world_scale, 1.0);
    }

    #[test]
    fn ribbon_deepest_first_ordering() {
        // Camera at anchor depth 9 in a world whose tree is deep.
        let (lib, root, _planet) = demo_world();
        // Force anchor through slot 22 (above body) — Cartesian path,
        // many frames possible.
        let mut anchor = Path::root();
        anchor.push(22);
        anchor.push(13);
        anchor.push(13);
        anchor.push(13);
        anchor.push(13);
        anchor.push(13);
        anchor.push(13);
        anchor.push(13);
        anchor.push(13);
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let frames = build_ribbon(&lib, root, &camera);
        assert!(!frames.is_empty(), "ribbon must not be empty");
        // Deepest-first ordering: first frame's depth >= last frame's.
        for w in frames.windows(2) {
            assert!(
                w[0].path.depth() >= w[1].path.depth(),
                "ribbon must be ordered deepest-first; got {} then {}",
                w[0].path.depth(), w[1].path.depth(),
            );
        }
    }

    #[test]
    fn ribbon_filters_frames_inside_planet_body() {
        // Frames whose path passes through the body are FILTERED.
        // The global sphere pass renders that content; if the
        // ribbon's Cartesian march also walked it (inside body =
        // CubedSphereFace + Cartesian descendants), we'd get cubic
        // cells overlapping the sphere's bulged voxels. Per-frame
        // sphere DDA was the alternate plan but it produced
        // z-fighting + seams from linearization-at-camera errors;
        // reverted.
        let (lib, root, planet_path) = demo_world();
        let mut anchor = planet_path; // [13]
        anchor.push(crate::world::cubesphere::FACE_SLOTS[2] as u8);
        for _ in 0..6 {
            anchor.push(13);
        }
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let frames = build_ribbon(&lib, root, &camera);
        assert!(!frames.is_empty(), "must always have at least root frame");
        for f in &frames {
            assert_eq!(
                f.path.depth(), 0,
                "frame at depth {} crosses sphere content; should be filtered",
                f.path.depth(),
            );
        }
    }

    #[test]
    fn ribbon_camera_local_is_path_anchored_precise() {
        // For a frame at depth F that's a prefix of camera.anchor,
        // camera_local should be a precise pure-path computation
        // — equivalent to in_frame(frame_path) — bounded in
        // [0, WORLD_SIZE)^3 regardless of depth.
        let (lib, root, _planet) = demo_world();
        let mut anchor = Path::root();
        for _ in 0..15 {
            anchor.push(22);
        }
        let camera = WorldPos::new(anchor, [0.25, 0.5, 0.75]);
        let frames = build_ribbon(&lib, root, &camera);
        // Allow tiny f32 overshoot at the upper bound — composing 15
        // levels of slot offsets accumulates ~1 ULP per level.
        let upper = WORLD_SIZE * (1.0 + 1e-5);
        for f in &frames {
            for &c in &f.camera_local {
                assert!(
                    c.is_finite() && c >= -1e-5 && c <= upper,
                    "frame depth={} camera_local component out of bounds: {}",
                    f.path.depth(), c,
                );
            }
        }
    }

    #[test]
    fn ribbon_world_scale_matches_frame_depth() {
        // world_scale for a frame at depth F must equal (1/3)^F
        // exactly — that's the conversion factor from frame-local
        // t to world t for ribbon compositing.
        let (lib, root, _planet) = demo_world();
        let mut anchor = Path::root();
        for _ in 0..10 {
            anchor.push(22);
        }
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let frames = build_ribbon(&lib, root, &camera);
        for f in &frames {
            let expected = (1.0_f32 / 3.0).powi(f.path.depth() as i32);
            assert!(
                (f.world_scale - expected).abs() < 1e-7,
                "frame depth={} world_scale={} expected {}",
                f.path.depth(), f.world_scale, expected,
            );
        }
    }

    #[test]
    fn ribbon_empty_anchor_produces_only_root() {
        let (lib, root, _planet) = demo_world();
        let camera = WorldPos::new(Path::root(), [0.5; 3]);
        let frames = build_ribbon(&lib, root, &camera);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].path.depth(), 0);
    }

    #[test]
    fn ribbon_count_capped_at_max_ribbon_frames() {
        let (lib, root, _planet) = demo_world();
        let mut anchor = Path::root();
        for _ in 0..30 {
            anchor.push(22);
        }
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let frames = build_ribbon(&lib, root, &camera);
        assert!(
            frames.len() <= MAX_RIBBON_FRAMES,
            "ribbon must respect MAX_RIBBON_FRAMES cap; got {}",
            frames.len(),
        );
        // Root frame must always survive the cap.
        assert!(
            frames.iter().any(|f| f.path.depth() == 0),
            "root frame must always be in the ribbon",
        );
    }

    #[test]
    fn ribbon_truncates_at_tree_terminal() {
        // Build a tiny world whose root has a 2-deep chain on slot 0
        // and Empty everywhere else. Anchor at slot 0 down to depth
        // 10. The walk should truncate at the chain's end (depth 2).
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(empty_children());
        let mut chain_l1_children = empty_children();
        chain_l1_children[0] = TChild::Node(leaf);
        let chain_l1 = lib.insert(chain_l1_children);
        let mut root_children = empty_children();
        root_children[0] = TChild::Node(chain_l1);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        for _ in 0..10 {
            anchor.push(0);
        }
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let frames = build_ribbon(&lib, root, &camera);
        // No frame should exceed depth 2 since walk truncates there.
        for f in &frames {
            assert!(
                f.path.depth() <= 2,
                "frame depth {} exceeds tree's 2-deep terminal",
                f.path.depth(),
            );
        }
    }

    #[test]
    fn ribbon_frame_node_id_matches_walk_from_root() {
        let (lib, root, _planet) = demo_world();
        let mut anchor = Path::root();
        for _ in 0..6 {
            anchor.push(22);
        }
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let frames = build_ribbon(&lib, root, &camera);
        for f in &frames {
            // Independently walk world_root to the frame's path
            // and verify the resulting NodeId matches.
            let mut nid = root;
            let mut walk_ok = true;
            for k in 0..f.path.depth() as usize {
                if let Some(node) = lib.get(nid) {
                    let s = f.path.slot(k) as usize;
                    if let TChild::Node(c) = node.children[s] {
                        nid = c;
                    } else {
                        walk_ok = false;
                        break;
                    }
                }
            }
            assert!(walk_ok, "frame's path must be walkable in world tree");
            assert_eq!(
                f.node_id, nid,
                "frame depth={} node_id mismatch with independent walk",
                f.path.depth(),
            );
        }
    }

    #[test]
    fn planet_oc_world_path_anchored_bounded_at_deep_anchor() {
        // Camera deep under the body's PosY face, very deep anchor.
        // `offset_from(camera, body_center)` must stay bounded by
        // body cell size (~1.0 world unit), not blow up at deep
        // anchor depth.
        let (_lib, _root, planet_path) = demo_world();
        let mut anchor = planet_path; // [13]
        anchor.push(crate::world::cubesphere::FACE_SLOTS[2] as u8);
        for _ in 0..18 {
            anchor.push(13);
        }
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let body_center = WorldPos::new(planet_path, [0.5, 0.5, 0.5]);
        let oc = camera.offset_from(&body_center);
        for &c in &oc {
            assert!(c.is_finite(), "oc component must be finite at deep anchor");
            // Body cell at depth 1 is 1.0 world wide. oc should be
            // inside the body's volume (magnitude < 1.0).
            assert!(
                c.abs() < 1.0,
                "oc component {} exceeds body cell radius — precision likely broken",
                c,
            );
        }
    }

    #[test]
    fn build_ribbon_is_deterministic() {
        // Same inputs must produce the same ribbon; render_ribbon
        // is called every frame, any non-determinism would cause
        // visual flicker.
        let (lib, root, _planet) = demo_world();
        let mut anchor = Path::root();
        for _ in 0..7 {
            anchor.push(22);
        }
        let camera = WorldPos::new(anchor, [0.3, 0.4, 0.5]);
        let a = build_ribbon(&lib, root, &camera);
        let b = build_ribbon(&lib, root, &camera);
        assert_eq!(a.len(), b.len());
        for (ai, bi) in a.iter().zip(b.iter()) {
            assert_eq!(ai.path.depth(), bi.path.depth());
            assert_eq!(ai.path.as_slice(), bi.path.as_slice());
            assert_eq!(ai.node_id, bi.node_id);
            assert_eq!(ai.camera_local, bi.camera_local);
            assert_eq!(ai.world_scale, bi.world_scale);
        }
    }

    #[test]
    fn _suppress_unused_warnings() {
        // Keep imports referenced even if some sub-tests don't use them.
        let _ = (
            block::STONE,
            Planet {
                center: [0.5; 3], radius: 0.3, noise_scale: 0.0,
                noise_freq: 1.0, noise_seed: 0, gravity: 0.0,
                influence_radius: 1.0,
                surface_block: block::GRASS, core_block: block::STONE,
            },
            insert_spherical_body as fn(&mut NodeLibrary, f32, f32, u32, &Planet) -> NodeId,
            slot_index(1, 1, 1),
        );
    }
}
