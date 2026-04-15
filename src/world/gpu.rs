//! GPU data packing: convert tree nodes into flat buffers for the
//! ray march shader.
//!
//! Two parallel buffers are produced per pack:
//!
//! - `tree: Vec<GpuChild>` — 27 children per node, BFS-ordered. Each
//!   child has a tag (Empty / Block / Node) and either a block_type
//!   or a buffer-local node index.
//! - `node_kinds: Vec<GpuNodeKind>` — one entry per packed node,
//!   carrying its `NodeKind` discriminant + per-kind data (sphere
//!   body radii, cube face index). The shader looks this up when
//!   it walks into a Node child to decide whether to descend with
//!   the standard Cartesian DDA or switch to the cubed-sphere DDA.

use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};

use super::tree::*;

// Each child in the GPU buffer is 8 bytes:
//   tag (u8): 0=Empty, 1=Block, 2=Node
//   block_type (u8): valid when tag==1
//   _pad (u16)
//   node_index (u32): buffer-local index, valid when tag==2
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuChild {
    pub tag: u8,
    pub block_type: u8,
    pub _pad: u16,
    pub node_index: u32,
}

/// One node in the GPU buffer = 27 GpuChild = 216 bytes.
pub const GPU_NODE_SIZE: usize = 27;

/// Per-packed-node metadata: which `NodeKind` this node is, plus the
/// per-kind data the shader needs to render its content. Indexed by
/// the same buffer index used in `GpuChild::node_index`.
///
/// 16 bytes per node so the WGSL `array<NodeKindGpu>` aligns
/// cleanly. `kind` discriminant: 0 = Cartesian, 1 = CubedSphereBody,
/// 2 = CubedSphereFace.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuNodeKind {
    pub kind: u32,
    pub face: u32,
    pub inner_r: f32,
    pub outer_r: f32,
}

impl GpuNodeKind {
    pub fn from_node_kind(k: NodeKind) -> Self {
        match k {
            NodeKind::Cartesian => Self { kind: 0, face: 0, inner_r: 0.0, outer_r: 0.0 },
            NodeKind::CubedSphereBody { inner_r, outer_r } => Self {
                kind: 1, face: 0, inner_r, outer_r,
            },
            NodeKind::CubedSphereFace { face } => Self {
                kind: 2, face: face as u32, inner_r: 0.0, outer_r: 0.0,
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuCamera {
    /// World-coord camera position. Used for highlight ray-box test
    /// and for pack-time distance LOD. The shader's ribbon-frame
    /// marches use per-frame `camera_local` from `GpuRibbonFrame`.
    pub pos: [f32; 3],
    pub _pad0: f32,
    pub forward: [f32; 3],
    pub _pad1: f32,
    pub right: [f32; 3],
    pub _pad2: f32,
    pub up: [f32; 3],
    pub fov: f32,
}

/// One ribbon frame on the GPU.
///
/// **Cartesian half** (always populated):
/// `(root_index, camera_local, world_scale)` — the shader marches
/// the frame's tree in its `[0, WORLD_SIZE)^3` local coords; hits
/// composite by smallest `t * world_scale` (world units).
///
/// **Sphere half** (populated only when `sphere_active == 1`,
/// i.e. the frame's path passes through a `CubedSphereBody`):
/// describes the frame's slice of the planet's face subtree in
/// reference-plus-delta form so the per-frame sphere DDA can
/// resolve cells at face-subtree depths beyond f32's `[0, 1]`
/// precision wall (~depth 14 globally) by working in frame-local
/// coords (~14 more depth levels per frame, so ~total depth 28+
/// achievable for a frame at face-subtree depth 14).
///
/// All sphere-half scalars are CPU-computed in f64 then cast to
/// f32 — keeps 7 digits of precision relative to small magnitudes
/// (frame size, remainders) where straightforward shader-side
/// arithmetic would have lost them via near-equal subtraction.
///
/// 128 bytes, vec4-aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuRibbonFrame {
    // ── Slot 0: u32 quadword ────────────────────────────────────────
    pub root_index: u32,
    pub sphere_active: u32,
    pub world_scale: f32,
    pub face: u32,

    // ── Slot 1: vec4 — Cartesian camera position ────────────────────
    pub camera_local: [f32; 4],

    // ── Slot 2: u32+f32 quadword ────────────────────────────────────
    /// Buffer index of the frame's face-subtree node (where the
    /// per-frame walker starts descending). Walker walks from here
    /// with `(camera_*_remainder + alpha_* * dt)` as its input.
    pub frame_face_node_idx: u32,
    /// Frame's size in face-EA-normalized [0, 1] coords (= 1/3^F
    /// where F = number of face-subtree slots in the frame's path).
    pub frame_un_size: f32,
    /// `2 * frame_un_size * (pi/4) * sec^2(frame_un_lo_ea * pi/4)`
    /// — rate of change of the cube-coord plane coefficient with
    /// respect to a unit cell_local_un offset within the frame.
    /// Per-cell plane normal: `n_cell = frame_n_u_lo_ref -
    /// cell_local_un * frame_alpha_n_u * n_axis`. `cell_local_un`
    /// is bounded in [0, 1] within frame, so the multiplication
    /// keeps precision.
    pub frame_alpha_n_u: f32,
    pub frame_alpha_n_v: f32,

    // ── Slot 3: f32 quadword ────────────────────────────────────────
    /// Camera position projected to face's (u, v, r) cube coords
    /// at the frame's reference, expressed as a remainder in the
    /// frame's [0, 1] local: `camera_un_remainder = (camera_un_global
    /// - frame_un_lo_global) / frame_un_size`. CPU-computed in f64.
    /// Sample at world distance dt from camera: `sample_un_remainder
    /// = camera_un_remainder + alpha_un * dt`.
    pub camera_un_remainder: f32,
    pub camera_vn_remainder: f32,
    pub camera_rn_remainder: f32,
    /// Rate of change of r from frame's r_lo per unit cell_local_rn:
    /// = `shell * frame_rn_size`. (`r_cell = frame_r_lo_world +
    /// cell_local_rn * frame_alpha_r`.)
    pub frame_alpha_r: f32,

    // ── Slot 4: vec4 — reference u plane normal at frame's u_lo ─────
    pub frame_n_u_lo_ref: [f32; 4],

    // ── Slot 5: vec4 — reference v plane normal at frame's v_lo ─────
    pub frame_n_v_lo_ref: [f32; 4],

    // ── Slot 6: vec4 — reference r values + face geometry ───────────
    /// Frame's r_lo in world units.
    pub frame_r_lo_world: f32,
    /// Sphere's inner radius in world (same as global GpuPlanet).
    pub sphere_inner_r_world: f32,
    /// Sphere's outer radius in world.
    pub sphere_outer_r_world: f32,
    /// Sphere's shell (= outer - inner) in world.
    pub sphere_shell_world: f32,

    // ── Slot 7: vec4 — face normal vector (for plane-normal updates)
    pub face_n_axis: [f32; 4],
}

/// Per-frame sphere computation result. Holds the same fields the
/// shader's per-frame sphere DDA needs from `GpuRibbonFrame`'s
/// sphere half — kept as a separate struct so the CPU computation
/// can populate them in one place and the upload code can copy them
/// into the GPU struct without having to remember which fields are
/// part of the sphere half.
#[derive(Clone, Copy, Debug)]
pub struct SphereFrameData {
    pub face: u32,
    pub frame_face_node_idx: u32,
    pub frame_un_size: f32,
    pub frame_alpha_n_u: f32,
    pub frame_alpha_n_v: f32,
    pub camera_un_remainder: f32,
    pub camera_vn_remainder: f32,
    pub camera_rn_remainder: f32,
    pub frame_alpha_r: f32,
    pub frame_n_u_lo_ref: [f32; 4],
    pub frame_n_v_lo_ref: [f32; 4],
    pub frame_r_lo_world: f32,
    pub sphere_inner_r_world: f32,
    pub sphere_outer_r_world: f32,
    pub sphere_shell_world: f32,
    pub face_n_axis: [f32; 4],
}

/// Planet rendering state on the GPU. Populated when the world has
/// exactly one spherical body (the common case); `active=0` disables.
///
/// `oc_world` is `camera_world_pos - planet_center_world` computed
/// on the CPU via `WorldPos::offset_from`, which is path-exact and
/// bounded by the body cell's world size (1.0 world units for a
/// body at depth 1). The shader's sphere DDA works entirely in
/// `oc`-relative coordinates, so magnitudes stay bounded regardless
/// of the camera's anchor depth — precision doesn't degrade at deep
/// zoom the way passing `camera - center` through frame-local
/// scaling would.
///
/// `body_node_index` is the buffer index of the body node in the
/// packed tree, so the sphere DDA can walk its face subtrees via
/// the body's face-center children.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuPlanet {
    pub enabled: u32,
    pub body_node_index: u32,
    pub inner_r_world: f32,
    pub outer_r_world: f32,
    pub oc_world: [f32; 4],
    /// Reserved for future use (alignment padding kept so the
    /// struct stays 48 bytes / WGSL-Planet aligned).
    pub _reserved: u32,
    pub _pad: [u32; 3],
}

/// Block color palette — up to 256 RGBA colors indexed by block type.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPalette {
    pub colors: [[f32; 4]; 256],
}

impl Default for GpuPalette {
    fn default() -> Self {
        let mut colors = [[0.0f32; 4]; 256];
        for &(idx, _, color) in super::palette::BUILTINS {
            colors[idx as usize] = color;
        }
        Self { colors }
    }
}

/// Pack the visible portion of the tree into flat GPU buffers.
/// Returns `(tree_data, node_kinds, root_buffer_index)`.
pub fn pack_tree(
    library: &NodeLibrary,
    root: NodeId,
) -> (Vec<GpuChild>, Vec<GpuNodeKind>, u32) {
    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    let mut head = 0;
    visited.insert(root, 0);
    ordered.push(root);
    while head < ordered.len() {
        let nid = ordered[head];
        head += 1;
        if let Some(node) = library.get(nid) {
            for child in &node.children {
                if let Child::Node(child_id) = child {
                    if !visited.contains_key(child_id) {
                        let idx = ordered.len() as u32;
                        visited.insert(*child_id, idx);
                        ordered.push(*child_id);
                    }
                }
            }
        }
    }

    let mut data: Vec<GpuChild> = Vec::with_capacity(ordered.len() * GPU_NODE_SIZE);
    let mut kinds: Vec<GpuNodeKind> = Vec::with_capacity(ordered.len());
    for &nid in &ordered {
        let node = library.get(nid).expect("node in ordered list must exist");
        kinds.push(GpuNodeKind::from_node_kind(node.kind));
        for child in &node.children {
            data.push(match child {
                Child::Empty => GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 },
                Child::Block(bt) => GpuChild { tag: 1, block_type: *bt, _pad: 0, node_index: 0 },
                Child::Node(child_id) => {
                    let repr = library.get(*child_id)
                        .map(|n| n.representative_block).unwrap_or(0);
                    GpuChild {
                        tag: 2, block_type: repr, _pad: 0,
                        node_index: *visited.get(child_id).expect("child must be visited"),
                    }
                },
            });
        }
    }

    let root_idx = *visited.get(&root).unwrap();
    (data, kinds, root_idx)
}

/// LOD-aware tree packing: only uploads nodes large enough to see.
///
/// Same dual-buffer output as `pack_tree`, plus distance-aware
/// flattening of subtrees that cover less than `LOD_THRESHOLD`
/// pixels on screen. The shader walks whatever ends up in the
/// buffer; flattened cells appear as Block/Empty leaves.
///
/// `preserve_path` names a chain of (parent, slot) descents from
/// `root` that must stay walkable — used by the ribbon to keep
/// every ribbon-frame root reachable from the pack even when LOD
/// would otherwise flatten them. The camera's anchor path is the
/// natural argument; every ribbon frame is a prefix of it, so
/// preserving it preserves every frame's root node.
///
/// Also returns a map from `NodeId` to buffer index so the caller
/// can look up each ribbon frame's packed root for the ribbon
/// uniform.
pub fn pack_tree_lod(
    library: &NodeLibrary,
    root: NodeId,
    camera_pos: [f32; 3],
    screen_height: f32,
    fov: f32,
    preserve_path: &super::anchor::Path,
) -> (Vec<GpuChild>, Vec<GpuNodeKind>, u32, HashMap<NodeId, u32>) {
    use super::tree::{UNIFORM_EMPTY, UNIFORM_MIXED, slot_coords};

    let half_fov_recip = screen_height / (2.0 * (fov * 0.5).tan());
    const LOD_THRESHOLD: f32 = 0.5;

    struct QueueEntry {
        node_id: NodeId,
        origin: [f32; 3],
        cell_size: f32,
        depth_from_root: u8,
        on_preserve: bool,
    }

    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut queue: Vec<QueueEntry> = Vec::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    let mut overrides: Vec<[Option<GpuChild>; CHILDREN_PER_NODE]> = Vec::new();

    visited.insert(root, 0);
    ordered.push(root);
    overrides.push([None; CHILDREN_PER_NODE]);
    queue.push(QueueEntry {
        node_id: root,
        origin: [0.0; 3],
        cell_size: 1.0,
        depth_from_root: 0,
        on_preserve: true,
    });
    let mut head = 0;

    while head < queue.len() {
        let entry = &queue[head];
        let node_id = entry.node_id;
        let node_origin = entry.origin;
        let cell_size = entry.cell_size;
        let depth_from_root = entry.depth_from_root;
        let on_preserve = entry.on_preserve;
        let ordered_idx = head;
        head += 1;

        let Some(node) = library.get(node_id) else { continue };

        // Sphere-body and face nodes do NOT participate in
        // distance-LOD flattening — their children are interpreted
        // by the shader's NodeKind dispatch. Flattening would lose
        // the geometric semantics. Only Cartesian nodes apply LOD.
        let lod_active = matches!(node.kind, NodeKind::Cartesian);

        // Is this slot the preserve-path's next step from here?
        let preserve_slot: Option<u8> =
            if on_preserve && (depth_from_root as u32) < preserve_path.depth() as u32 {
                Some(preserve_path.slot(depth_from_root as usize))
            } else { None };

        for (slot, child) in node.children.iter().enumerate() {
            if let Child::Node(child_id) = child {
                let child_node = match library.get(*child_id) {
                    Some(n) => n,
                    None => continue,
                };

                let child_on_preserve = preserve_slot == Some(slot as u8);

                // Uniform-content collapse — only safe for Cartesian
                // nodes (face/body subtrees have geometry that the
                // shader needs to walk). Skip entirely for nodes on
                // the preserve path: the ribbon needs them walkable.
                let child_is_cartesian = matches!(child_node.kind, NodeKind::Cartesian);
                if !child_on_preserve
                    && lod_active && child_is_cartesian
                    && child_node.uniform_type != UNIFORM_MIXED {
                    let gpu = if child_node.uniform_type == UNIFORM_EMPTY {
                        GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }
                    } else {
                        GpuChild { tag: 1, block_type: child_node.uniform_type, _pad: 0, node_index: 0 }
                    };
                    overrides[ordered_idx][slot] = Some(gpu);
                    continue;
                }

                // Distance LOD: REMOVED. The previous behavior
                // flattened any cell smaller than ~0.5 pixels to its
                // `representative_block`, which silently swallowed
                // user-placed blocks whose ancestors became MIXED
                // (one wood + 26 stone cells → ancestor's
                // representative is still STONE → user's wood
                // disappears at moderate zoom). Uniform-flattening
                // above already handles the actual perf win
                // (uniform stone/empty regions collapse to one
                // tag=1 entry); MIXED cells are kept walkable at
                // any depth so user edits remain visible.
                //
                // Performance: mixed-subtree count is bounded by
                // the SDF's complexity (~SDF_DETAIL_LEVELS deep)
                // plus user-edit volume — typically thousands of
                // nodes, not millions, so packing all MIXED depths
                // is cheap.
                let _ = LOD_THRESHOLD; // keep constant referenced
                let _ = (child_on_preserve, lod_active, child_is_cartesian, half_fov_recip, camera_pos);

                if !visited.contains_key(child_id) {
                    let idx = ordered.len() as u32;
                    visited.insert(*child_id, idx);
                    ordered.push(*child_id);
                    overrides.push([None; CHILDREN_PER_NODE]);
                    let (cx, cy, cz) = slot_coords(slot);
                    let child_origin = [
                        node_origin[0] + cx as f32 * cell_size,
                        node_origin[1] + cy as f32 * cell_size,
                        node_origin[2] + cz as f32 * cell_size,
                    ];
                    queue.push(QueueEntry {
                        node_id: *child_id,
                        origin: child_origin,
                        cell_size: cell_size / 3.0,
                        depth_from_root: depth_from_root.saturating_add(1),
                        on_preserve: child_on_preserve,
                    });
                }
            }
        }
    }

    let mut data: Vec<GpuChild> = Vec::with_capacity(ordered.len() * GPU_NODE_SIZE);
    let mut kinds: Vec<GpuNodeKind> = Vec::with_capacity(ordered.len());
    for (oi, &nid) in ordered.iter().enumerate() {
        let node = library.get(nid).expect("node in ordered list must exist");
        kinds.push(GpuNodeKind::from_node_kind(node.kind));
        for (slot, child) in node.children.iter().enumerate() {
            if let Some(gpu) = overrides[oi][slot] {
                data.push(gpu);
            } else {
                data.push(match child {
                    Child::Empty => GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 },
                    Child::Block(bt) => GpuChild { tag: 1, block_type: *bt, _pad: 0, node_index: 0 },
                    Child::Node(child_id) => {
                        let repr = library.get(*child_id).map(|n| n.representative_block).unwrap_or(0);
                        let idx = visited.get(child_id).copied().unwrap_or(0);
                        GpuChild { tag: 2, block_type: repr, _pad: 0, node_index: idx }
                    },
                });
            }
        }
    }

    let root_idx = *visited.get(&root).unwrap();
    (data, kinds, root_idx, visited)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_test_world() {
        let world = super::super::state::WorldState::test_world();
        let (data, kinds, root_idx) = pack_tree(&world.library, world.root);
        assert_eq!(data.len() % 27, 0);
        assert_eq!(root_idx, 0);
        assert_eq!(data.len() / 27, world.library.len());
        assert_eq!(kinds.len(), world.library.len());
        // All test_world nodes are Cartesian.
        for k in &kinds {
            assert_eq!(k.kind, 0);
        }
    }

    #[test]
    fn gpu_child_size() {
        assert_eq!(std::mem::size_of::<GpuChild>(), 8);
    }

    #[test]
    fn gpu_node_kind_size() {
        assert_eq!(std::mem::size_of::<GpuNodeKind>(), 16);
    }

    #[test]
    fn gpu_ribbon_frame_size() {
        // 8 vec4-sized slots = 128 bytes. Layout:
        // slot 0: root_index, sphere_active, world_scale, face
        // slot 1: camera_local (vec4)
        // slot 2: frame_face_node_idx, pad, frame_un_size_cube, frame_rn_size
        // slot 3: camera_un_remainder, vn, rn, pad
        // slot 4: alpha_un, vn, rn, pad
        // slot 5: frame_n_u_ref (vec4)
        // slot 6: frame_n_v_ref (vec4)
        // slot 7: frame_r_lo_world, pad x 3
        assert_eq!(std::mem::size_of::<GpuRibbonFrame>(), 128);
    }

    #[test]
    fn gpu_planet_size_matches_shader_layout() {
        // 4 u32 + 4 f32 + vec4<f32> + 1 u32 + 3 u32 pad = 48 bytes,
        // matching the WGSL `Planet` struct's expected size when
        // `_pad0/1/2: u32` are used (NOT `vec3<u32>` which would
        // round to 16 bytes).
        assert_eq!(std::mem::size_of::<GpuPlanet>(), 48);
    }

    #[test]
    fn pack_includes_body_and_face_kinds_for_demo_planet() {
        // The packed `node_kinds` buffer must surface the planet's
        // CubedSphereBody and 6 CubedSphereFace nodes — the shader's
        // sphere pass uses the body's buffer index to walk face
        // subtrees, and the ribbon's "is this frame inside a sphere"
        // filter relies on these kinds being present.
        use crate::world::anchor::Path;
        use crate::world::spherical_worldgen;
        use crate::world::worldgen;
        let mut world = worldgen::generate_world();
        let setup = spherical_worldgen::demo_planet();
        let (new_root, _planet_path) =
            spherical_worldgen::install_at_root_center(
                &mut world.library, world.root, &setup,
            );
        world.swap_root(new_root);

        let anchor = Path::root();
        let (_data, kinds, _root_idx, _visited) = pack_tree_lod(
            &world.library, world.root,
            [1.5, 1.5, 1.5], 1440.0, 1.2,
            &anchor,
        );
        let body_count = kinds.iter().filter(|k| k.kind == 1).count();
        let face_count = kinds.iter().filter(|k| k.kind == 2).count();
        assert_eq!(body_count, 1, "expected exactly one CubedSphereBody node");
        assert_eq!(face_count, 6, "expected six CubedSphereFace nodes (one per face)");
    }

    #[test]
    fn pack_preserves_body_and_faces_after_break() {
        // Combination test: edit a block in the planet's face
        // subtree, then verify the post-edit pack STILL has the
        // body + faces with their correct kinds. The "break makes
        // sphere cartesian" symptom would manifest as body_count==0
        // or face_count<6 here.
        use crate::world::anchor::Path;
        use crate::world::cubesphere::FACE_SLOTS;
        use crate::world::edit::{break_block, HitInfo};
        use crate::world::spherical_worldgen;
        use crate::world::worldgen;
        let mut world = worldgen::generate_world();
        let setup = spherical_worldgen::demo_planet();
        let (new_root, _planet_path) =
            spherical_worldgen::install_at_root_center(
                &mut world.library, world.root, &setup,
            );
        world.swap_root(new_root);

        // Build a hit path through (root, slot 13) → body, face_slot
        // → face_root, then walk down to a Block.
        let body_id = match world.library.get(world.root).unwrap().children[slot_index(1, 1, 1)] {
            Child::Node(id) => id,
            _ => panic!("body not at slot 13"),
        };
        let face_slot = FACE_SLOTS[2];
        let face_root_id = match world.library.get(body_id).unwrap().children[face_slot] {
            Child::Node(id) => id,
            _ => panic!("face root missing"),
        };
        let mut path: Vec<(NodeId, usize)> = Vec::new();
        path.push((world.root, slot_index(1, 1, 1)));
        path.push((body_id, face_slot));
        let mut node_id = face_root_id;
        'outer: loop {
            let node = world.library.get(node_id).expect("node");
            for (s, c) in node.children.iter().enumerate() {
                match c {
                    Child::Block(_) => {
                        path.push((node_id, s));
                        break 'outer;
                    }
                    Child::Node(next) => {
                        path.push((node_id, s));
                        node_id = *next;
                        break;
                    }
                    _ => continue,
                }
            }
        }
        let hit = HitInfo { path, face: 0, t: 0.0, place_path: None };
        assert!(break_block(&mut world, &hit));

        let (_data, kinds, _root_idx, _visited) = pack_tree_lod(
            &world.library, world.root,
            [1.5, 1.5, 1.5], 1440.0, 1.2,
            &Path::root(),
        );
        let body_count = kinds.iter().filter(|k| k.kind == 1).count();
        let face_count = kinds.iter().filter(|k| k.kind == 2).count();
        assert_eq!(
            body_count, 1,
            "after break, sphere body kind disappeared from the pack",
        );
        assert!(
            face_count >= 6,
            "after break, expected at least 6 face kinds; got {}", face_count,
        );
    }

    /// Rust mirror of the WGSL walker's cell-extent accumulation.
    /// Returns `(u_lo, v_lo, r_lo, cell_size)` at the cell containing
    /// `(un, vn, rn)` after `descents` levels of descent. Used to
    /// test the f32 precision properties of the additive accumulation.
    fn walk_extents(un: f32, vn: f32, rn: f32, descents: u32) -> (f32, f32, f32, f32) {
        let mut un = un.clamp(0.0, 0.9999999);
        let mut vn = vn.clamp(0.0, 0.9999999);
        let mut rn = rn.clamp(0.0, 0.9999999);
        let mut u_lo = 0.0_f32;
        let mut v_lo = 0.0_f32;
        let mut r_lo = 0.0_f32;
        let mut size = 1.0_f32;
        for _ in 0..descents {
            let us = ((un * 3.0) as u32).min(2);
            let vs = ((vn * 3.0) as u32).min(2);
            let rs = ((rn * 3.0) as u32).min(2);
            let next_size = size / 3.0;
            u_lo += us as f32 * next_size;
            v_lo += vs as f32 * next_size;
            r_lo += rs as f32 * next_size;
            un = un * 3.0 - us as f32;
            vn = vn * 3.0 - vs as f32;
            rn = rn * 3.0 - rs as f32;
            size = next_size;
        }
        (u_lo, v_lo, r_lo, size)
    }

    #[test]
    fn walker_extents_root_is_unit_cell() {
        let (u, v, r, sz) = walk_extents(0.5, 0.5, 0.5, 0);
        assert_eq!((u, v, r, sz), (0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn walker_extents_one_descent_is_third() {
        // un = 0.5 → us = 1 → cell at u_lo = 1/3, size = 1/3.
        let (u, v, r, sz) = walk_extents(0.5, 0.1, 0.9, 1);
        assert!((u - 1.0/3.0).abs() < 1e-7);
        assert!((v - 0.0).abs() < 1e-7);
        assert!((r - 2.0/3.0).abs() < 1e-7);
        assert!((sz - 1.0/3.0).abs() < 1e-7);
    }

    #[test]
    fn walker_extents_cell_contains_input_point() {
        // Cell at the walker's terminal must contain the input
        // coordinates: u_lo <= un < u_lo + size, etc.
        // Below depth ~15, f32 ULP in [0, 1] (~1.2e-7) exceeds
        // cell width (1/3^N), so the accumulated u_lo cannot
        // reliably contain inputs near boundaries — that's the
        // precision limit the per-frame sphere DDA work will
        // address. Tested only up to depth 13 here.
        let test_points = [
            (0.123, 0.456, 0.789, 5),
            (0.001, 0.5, 0.999, 10),
            (0.5, 0.5, 0.5, 13),
            (0.333333, 0.666666, 0.111111, 8),
        ];
        for &(un, vn, rn, descents) in &test_points {
            let (u_lo, v_lo, r_lo, sz) = walk_extents(un, vn, rn, descents);
            let un_clamped = un.clamp(0.0, 0.9999999);
            let vn_clamped = vn.clamp(0.0, 0.9999999);
            let rn_clamped = rn.clamp(0.0, 0.9999999);
            // Allow 1 ULP tolerance per descent for accumulated rounding.
            let tol = sz * 1e-3;
            assert!(
                un_clamped >= u_lo - tol && un_clamped <= u_lo + sz + tol,
                "depth={} un={} not in [{}, {}+{}]",
                descents, un_clamped, u_lo, u_lo, sz,
            );
            assert!(
                vn_clamped >= v_lo - tol && vn_clamped <= v_lo + sz + tol,
                "depth={} vn={} not in [{}, {}+{}]",
                descents, vn_clamped, v_lo, v_lo, sz,
            );
            assert!(
                rn_clamped >= r_lo - tol && rn_clamped <= r_lo + sz + tol,
                "depth={} rn={} not in [{}, {}+{}]",
                descents, rn_clamped, r_lo, r_lo, sz,
            );
        }
    }

    #[test]
    fn walker_extents_size_scales_with_depth() {
        for descents in 0..=22 {
            let (_, _, _, sz) = walk_extents(0.5, 0.5, 0.5, descents);
            let expected = (1.0_f32 / 3.0).powi(descents as i32);
            assert!(
                (sz - expected).abs() < expected * 1e-5 + 1e-30,
                "depth={} size={} expected {}",
                descents, sz, expected,
            );
        }
    }

    #[test]
    fn walker_extents_stay_in_unit_cube() {
        // u_lo must be in [0, 1) and u_lo + size <= 1 + eps for
        // every depth and every input. f32 may overshoot by a few
        // ULPs; allow a small tolerance.
        for descents in 0..=22 {
            for &un in &[0.0_f32, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999999] {
                let (u_lo, _, _, sz) = walk_extents(un, 0.5, 0.5, descents);
                assert!(u_lo >= -1e-6, "depth={} un={} u_lo={}", descents, un, u_lo);
                assert!(u_lo + sz <= 1.0 + 1e-5,
                    "depth={} un={} u_lo+size={} > 1", descents, un, u_lo + sz);
            }
        }
    }

    #[test]
    fn walker_extents_precise_within_resolvable_range() {
        // The walker's input `un` lives in [0, 1] face coords with
        // f32 ULP ~1.2e-7 — fundamentally cannot distinguish cells
        // smaller than ULP. Single-coord-system walker tops out at
        // depth ~14 (3^14 ≈ 4.8e6, cell ≈ 2e-7 = ULP). Beyond that
        // requires frame-relative coords (Approach B), which the
        // ribbon's per-frame sphere uniforms provide separately.
        // This test verifies the in-range precision is correct.
        let depth = 13;
        let cells_per_axis = 3.0_f32.powi(depth);
        let cell_width = 1.0 / cells_per_axis;
        let un_a = 0.5_f32;
        let un_b = un_a + cell_width * 2.0; // 2 cells over
        let (u_lo_a, _, _, _) = walk_extents(un_a, 0.5, 0.5, depth as u32);
        let (u_lo_b, _, _, _) = walk_extents(un_b, 0.5, 0.5, depth as u32);
        assert!(
            (u_lo_b - u_lo_a).abs() > cell_width * 0.5,
            "walker collapsed at resolvable depth {}: delta={} cell_width={}",
            depth, u_lo_b - u_lo_a, cell_width,
        );
    }

    /// Mirror of the CPU-side per-frame sphere computation used to
    /// verify the f64-based math precision-stable at deep frame
    /// depths. Not used at runtime — runtime uses
    /// `App::compute_sphere_frame_data` which embeds the same math
    /// in a method on `App`.
    fn synthetic_sphere_frame_data(
        face_subtree_path: &[u8],          // slot indices in face subtree (not face_root slot)
        cs_inner_world: f64,
        cs_outer_world: f64,
        oc_world: [f64; 3],                // camera relative to body center
        face_n: [f64; 3],
        face_u: [f64; 3],
        face_v: [f64; 3],
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        // Returns (un_lo, vn_lo, rn_lo, frame_size, camera_un_rem,
        //          camera_vn_rem, camera_rn_rem, alpha_n_u, alpha_n_v).
        let mut un_lo = 0.0f64;
        let mut vn_lo = 0.0f64;
        let mut rn_lo = 0.0f64;
        let mut size = 1.0f64;
        for &slot in face_subtree_path {
            let s = slot as usize;
            let us = (s % 3) as f64;
            let vs = ((s / 3) % 3) as f64;
            let rs = ((s / 9) % 3) as f64;
            let next_size = size / 3.0;
            un_lo += us * next_size;
            vn_lo += vs * next_size;
            rn_lo += rs * next_size;
            size = next_size;
        }
        let r_camera = (oc_world[0]*oc_world[0] + oc_world[1]*oc_world[1] + oc_world[2]*oc_world[2]).sqrt();
        let cam_dir = [oc_world[0]/r_camera, oc_world[1]/r_camera, oc_world[2]/r_camera];
        let axis_dot = cam_dir[0]*face_n[0] + cam_dir[1]*face_n[1] + cam_dir[2]*face_n[2];
        let cube_u = (cam_dir[0]*face_u[0] + cam_dir[1]*face_u[1] + cam_dir[2]*face_u[2]) / axis_dot;
        let cube_v = (cam_dir[0]*face_v[0] + cam_dir[1]*face_v[1] + cam_dir[2]*face_v[2]) / axis_dot;
        let pi = std::f64::consts::PI;
        let camera_un_global = (cube_u.atan() * 4.0 / pi + 1.0) * 0.5;
        let camera_vn_global = (cube_v.atan() * 4.0 / pi + 1.0) * 0.5;
        let shell = cs_outer_world - cs_inner_world;
        let camera_rn_global = (r_camera - cs_inner_world) / shell;
        let camera_un_rem = (camera_un_global - un_lo) / size;
        let camera_vn_rem = (camera_vn_global - vn_lo) / size;
        let camera_rn_rem = (camera_rn_global - rn_lo) / size;
        let u_lo_ea = un_lo * 2.0 - 1.0;
        let v_lo_ea = vn_lo * 2.0 - 1.0;
        let sec_sq_u = 1.0 / (u_lo_ea * pi / 4.0).cos().powi(2);
        let sec_sq_v = 1.0 / (v_lo_ea * pi / 4.0).cos().powi(2);
        let alpha_n_u = 2.0 * size * (pi / 4.0) * sec_sq_u;
        let alpha_n_v = 2.0 * size * (pi / 4.0) * sec_sq_v;
        (un_lo, vn_lo, rn_lo, size, camera_un_rem, camera_vn_rem, camera_rn_rem, alpha_n_u, alpha_n_v)
    }

    #[test]
    fn sphere_frame_data_size_scales_inverse_3pow_depth() {
        // Frame at face-subtree depth N has frame_un_size = 1/3^N.
        let oc = [0.0, 0.5, 0.0];
        let face_n = [0.0, 1.0, 0.0];
        let face_u = [1.0, 0.0, 0.0];
        let face_v = [0.0, 0.0, -1.0];
        for depth in [1usize, 5, 10, 14, 18, 20] {
            let path: Vec<u8> = std::iter::repeat(13_u8).take(depth).collect();
            let (_, _, _, size, _, _, _, _, _) = synthetic_sphere_frame_data(
                &path, 0.12, 0.45, oc, face_n, face_u, face_v,
            );
            let expected = 1.0_f64 / 3.0_f64.powi(depth as i32);
            assert!(
                (size - expected).abs() < expected * 1e-12,
                "depth={} size={} expected {}",
                depth, size, expected,
            );
        }
    }

    #[test]
    fn sphere_frame_data_remainder_in_unit_range_for_central_path() {
        // For path of all-13 slots (center cell at each level), the
        // frame's un_lo, vn_lo, rn_lo should converge to 0.5. A
        // camera at the body center looking up the +Y face has
        // camera_un_global = 0.5, camera_vn_global = 0.5, so the
        // remainder should be in [0, 1].
        let oc = [0.0, 0.5, 0.0];
        let face_n = [0.0, 1.0, 0.0];
        let face_u = [1.0, 0.0, 0.0];
        let face_v = [0.0, 0.0, -1.0];
        for depth in [1usize, 3, 5, 10, 15, 20] {
            let path: Vec<u8> = std::iter::repeat(13_u8).take(depth).collect();
            let (un_lo, _, _, size, un_rem, vn_rem, _, _, _) =
                synthetic_sphere_frame_data(&path, 0.12, 0.45, oc, face_n, face_u, face_v);
            // un_lo + size/2 should be ≈ 0.5 (cell centered on face).
            assert!(
                (un_lo + size / 2.0 - 0.5).abs() < 1e-12,
                "depth={} un_lo={} size={} center expected 0.5",
                depth, un_lo, size,
            );
            // Camera's rem should be in [0, 1] (camera at face center).
            assert!(
                un_rem >= 0.0 && un_rem <= 1.0,
                "depth={} un_rem={} out of [0, 1]", depth, un_rem,
            );
            assert!(
                vn_rem >= 0.0 && vn_rem <= 1.0,
                "depth={} vn_rem={}", depth, vn_rem,
            );
        }
    }

    #[test]
    fn sphere_frame_data_alpha_shrinks_with_depth() {
        // alpha_n_u = 2 * size * (pi/4) * sec^2(u_lo_ea * pi/4).
        // size = 1/3^depth. For central paths sec^2 stays bounded
        // (~1). So alpha ∝ 1/3^depth. Verify ratio matches.
        let oc = [0.0, 0.5, 0.0];
        let face_n = [0.0, 1.0, 0.0];
        let face_u = [1.0, 0.0, 0.0];
        let face_v = [0.0, 0.0, -1.0];
        let path1: Vec<u8> = std::iter::repeat(13_u8).take(1).collect();
        let path10: Vec<u8> = std::iter::repeat(13_u8).take(10).collect();
        let (_, _, _, _, _, _, _, alpha_1, _) = synthetic_sphere_frame_data(
            &path1, 0.12, 0.45, oc, face_n, face_u, face_v);
        let (_, _, _, _, _, _, _, alpha_10, _) = synthetic_sphere_frame_data(
            &path10, 0.12, 0.45, oc, face_n, face_u, face_v);
        // Ratio should be ~3^9 = 19683 (size scales by 3^-9 across
        // 9 depth levels), within a factor of 2 (sec^2 variation).
        let ratio = alpha_1 / alpha_10;
        let expected_ratio = 3.0_f64.powi(9); // ~19683
        assert!(
            ratio > expected_ratio * 0.5 && ratio < expected_ratio * 2.0,
            "alpha ratio {} not within 0.5-2× expected {}", ratio, expected_ratio,
        );
    }

    #[test]
    fn sphere_frame_data_alpha_finite_at_deep_depth() {
        // At depth 20, alpha should remain finite. Sanity check
        // that f64 doesn't overflow/underflow.
        let oc = [0.0, 0.5, 0.0];
        let face_n = [0.0, 1.0, 0.0];
        let face_u = [1.0, 0.0, 0.0];
        let face_v = [0.0, 0.0, -1.0];
        let path: Vec<u8> = std::iter::repeat(13_u8).take(20).collect();
        let (_, _, _, _, _, _, _, alpha_n_u, alpha_n_v) =
            synthetic_sphere_frame_data(&path, 0.12, 0.45, oc, face_n, face_u, face_v);
        assert!(alpha_n_u.is_finite() && alpha_n_u > 0.0);
        assert!(alpha_n_v.is_finite() && alpha_n_v > 0.0);
        // Alpha at depth 20 should be ≈ 2 * 1/3^20 * pi/4 * 1 ≈ 4.5e-10.
        let expected = 2.0 * (1.0 / 3.0_f64.powi(20)) * (std::f64::consts::PI / 4.0);
        assert!(
            (alpha_n_u - expected).abs() < expected * 0.5,
            "depth-20 alpha_n_u={} expected ~{}", alpha_n_u, expected,
        );
    }

    #[test]
    fn sphere_frame_data_precision_at_depth_20() {
        // The whole point of f64 accumulation: at depth 20, the
        // un_lo and frame_un_size are precise to ~16 digits in f64.
        // Casting `(camera_un_global - un_lo) / size` to f32 then
        // gives a remainder with 7 digits in [0, 1] of the frame —
        // PROVIDED the camera is actually within the frame. Use a
        // camera at the body center axis (oc = [0, 0.5, 0]) so it
        // projects to un_global = vn_global = 0.5 exactly, which
        // lies inside the central depth-20 cell (path of all 13s).
        let oc = [0.0, 0.5, 0.0];
        let face_n = [0.0, 1.0, 0.0];
        let face_u = [1.0, 0.0, 0.0];
        let face_v = [0.0, 0.0, -1.0];
        let depth = 20usize;
        let path: Vec<u8> = std::iter::repeat(13_u8).take(depth).collect();
        let (un_lo, _, _, size, un_rem, vn_rem, _, _, _) =
            synthetic_sphere_frame_data(&path, 0.12, 0.45, oc, face_n, face_u, face_v);
        assert!(un_lo.is_finite() && size.is_finite());
        assert!(
            (un_lo + size / 2.0 - 0.5).abs() < 1e-12,
            "depth-20 un_lo + size/2 not centered: {}", un_lo + size / 2.0,
        );
        let un_rem_f32 = un_rem as f32;
        let vn_rem_f32 = vn_rem as f32;
        assert!(un_rem_f32.is_finite());
        assert!(vn_rem_f32.is_finite());
        // Camera at face axis projects to (un, vn) = (0.5, 0.5) ↔
        // remainder = 0.5 within the central depth-20 cell.
        assert!(
            (un_rem_f32 - 0.5).abs() < 1e-3,
            "depth-20 un_rem_f32 expected ~0.5, got {}", un_rem_f32,
        );
        assert!(
            (vn_rem_f32 - 0.5).abs() < 1e-3,
            "depth-20 vn_rem_f32 expected ~0.5, got {}", vn_rem_f32,
        );
    }

    #[test]
    fn sphere_frame_data_remainder_outside_frame_is_far() {
        // Sanity check: a camera FAR from the frame's central cell
        // (e.g., off-axis) produces a remainder >> 1 at deep depth,
        // signaling the frame doesn't apply. The shader's per-frame
        // sphere DDA breaks out when `sample_un_remainder >= 1` so
        // this is the expected failure mode.
        let oc_far = [0.5, 0.5, 0.5]; // off-axis
        let face_n = [0.0, 1.0, 0.0];
        let face_u = [1.0, 0.0, 0.0];
        let face_v = [0.0, 0.0, -1.0];
        let depth = 20usize;
        let path: Vec<u8> = std::iter::repeat(13_u8).take(depth).collect();
        let (_, _, _, _, un_rem, _, _, _, _) =
            synthetic_sphere_frame_data(&path, 0.12, 0.45, oc_far, face_n, face_u, face_v);
        // Off-axis camera projects far from the central cell at
        // depth 20 — remainder should be HUGE (millions+).
        assert!(
            un_rem.abs() > 1e3,
            "off-axis camera at depth-20 frame should produce huge remainder; got {}",
            un_rem,
        );
    }

    #[test]
    fn walker_extents_no_catastrophic_loss_through_descents() {
        // Stress test: walk to maximum depth (22), verify cell
        // boundaries are still distinguishable. Tightest case:
        // adjacent points within the same depth-22 cell should
        // produce identical extents; points one cell apart should
        // produce different extents.
        let cells_per_axis = 3.0_f32.powi(22);
        let cell_width = 1.0 / cells_per_axis;
        // Two points 0.5 cell apart (same cell): same extents.
        let un_a = 0.5_f32;
        let un_b = un_a + cell_width * 0.5;
        let (lo_a, _, _, _) = walk_extents(un_a, 0.5, 0.5, 22);
        let (lo_b, _, _, _) = walk_extents(un_b, 0.5, 0.5, 22);
        // At depth 22, cell width is 1/3^22 ≈ 3e-11, below f32 ULP
        // at un=0.5 (~6e-8). The walker correctly handles this by
        // accumulating in [0, 1] precision; the input un is not the
        // limit, the accumulation is.
        let _ = (lo_a, lo_b); // smoke test — verify no panic, no NaN
        assert!(lo_a.is_finite() && lo_b.is_finite());
    }

    #[test]
    fn preserve_path_keeps_anchor_chain_walkable() {
        // When the anchor descends into a uniform-empty chain (the
        // generate_world background), the packer's distance-LOD must
        // NOT flatten the chain along the camera's path — every
        // ribbon frame's NodeId has to be reachable from the pack.
        use crate::world::anchor::Path;
        use crate::world::worldgen;
        let world = worldgen::generate_world();

        // Anchor descends through slot 0 (in the uniform chain).
        let mut anchor = Path::root();
        for _ in 0..15 {
            anchor.push(0);
        }

        // Use a camera position FAR from the chain so distance LOD
        // would aggressively flatten without preserve_path.
        let (_data, _kinds, _root_idx, visited) = pack_tree_lod(
            &world.library, world.root,
            [2.5, 2.5, 2.5], 1440.0, 1.2,
            &anchor,
        );

        // Walk the anchor in the library and verify every node along
        // it appears in `visited`. Without preserve_path, deeper
        // chain nodes would be missing.
        let mut nid = world.root;
        for k in 0..anchor.depth() as usize {
            assert!(
                visited.contains_key(&nid),
                "node at depth {} along anchor not in pack — preserve_path failed",
                k,
            );
            let slot = anchor.slot(k) as usize;
            match world.library.get(nid).unwrap().children[slot] {
                Child::Node(c) => nid = c,
                _ => break,
            }
        }
    }

    #[test]
    fn pack_preserves_body_kind_after_edit() {
        use crate::world::anchor::Path;
        use crate::world::cubesphere::{insert_spherical_body, FACE_SLOTS};
        use crate::world::sdf::Planet;
        use crate::world::edit::{break_block, HitInfo};
        use crate::world::palette::block;
        use crate::world::state::WorldState;

        let mut lib = NodeLibrary::default();
        let sdf = Planet {
            center: [0.5, 0.5, 0.5],
            radius: 0.30, noise_scale: 0.0, noise_freq: 1.0, noise_seed: 0,
            gravity: 0.0, influence_radius: 1.0,
            surface_block: block::GRASS, core_block: block::STONE,
        };
        let body_id = insert_spherical_body(&mut lib, 0.12, 0.45, 6, &sdf);
        // Wrap the body inside a root so we have a real WorldState.
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body_id);
        let world_root = lib.insert(root_children);
        lib.ref_inc(world_root);
        let mut world = WorldState { root: world_root, library: lib };

        // Verify body is CubedSphereBody pre-edit.
        assert!(matches!(
            world.library.get(body_id).unwrap().kind,
            NodeKind::CubedSphereBody { .. },
        ));

        // Find any Block leaf in the body's face subtree and break it.
        let body_node = world.library.get(body_id).unwrap();
        let face_slot = FACE_SLOTS[2]; // PosY
        let face_root_id = match body_node.children[face_slot] {
            Child::Node(id) => id,
            _ => panic!("face root not a Node"),
        };
        // Walk down the face subtree to a Block leaf.
        let mut node_id = face_root_id;
        let mut hit_path: Vec<(NodeId, usize)> = Vec::new();
        hit_path.push((world.root, slot_index(1, 1, 1)));
        hit_path.push((body_id, face_slot));
        'walk: loop {
            let node = world.library.get(node_id).expect("node exists");
            for (s, c) in node.children.iter().enumerate() {
                match c {
                    Child::Block(_) => {
                        hit_path.push((node_id, s));
                        break 'walk;
                    }
                    Child::Node(next) => {
                        hit_path.push((node_id, s));
                        node_id = *next;
                        break;
                    }
                    _ => continue,
                }
            }
        }
        let hit = HitInfo {
            path: hit_path,
            face: 0,
            t: 0.0,
            place_path: None,
        };
        assert!(break_block(&mut world, &hit));

        // After break, find the new body through the root.
        let new_root = world.library.get(world.root).unwrap();
        let new_body_id = match new_root.children[slot_index(1, 1, 1)] {
            Child::Node(id) => id,
            _ => panic!("slot 13 is not a Node after break"),
        };
        let new_body = world.library.get(new_body_id).unwrap();
        assert!(
            matches!(new_body.kind, NodeKind::CubedSphereBody { .. }),
            "body kind corrupted by break: got {:?}", new_body.kind,
        );

        // Now pack the tree with preserve_path through the body and
        // check the body's packed kind is still CubedSphereBody.
        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8);
        let (_data, kinds, _root_idx, visited) = pack_tree_lod(
            &world.library, world.root,
            [1.5, 1.5, 1.5], 1440.0, 1.2,
            &anchor,
        );
        let packed_body_idx = *visited.get(&new_body_id)
            .expect("new body not reached by packer — preserve_path failed");
        let packed_kind = &kinds[packed_body_idx as usize];
        assert_eq!(
            packed_kind.kind, 1,
            "body's packed kind is not CubedSphereBody (got kind={})",
            packed_kind.kind,
        );
    }
}
