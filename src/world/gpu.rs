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

/// One ribbon frame on the GPU — (frame-root buffer index, camera
/// position in frame-local coords, world scale). The shader marches
/// each frame independently in its `[0, WORLD_SIZE)^3` local system
/// and composites by smallest `t * world_scale`.
///
/// 32 bytes, 16-byte aligned so WGSL `array<RibbonFrame, N>` packs
/// without padding surprises.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuRibbonFrame {
    pub root_index: u32,
    pub _pad0: u32,
    pub world_scale: f32,
    pub _pad1: u32,
    pub camera_local: [f32; 4],
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
    /// Max face-subtree depth the sphere DDA uses for cell-boundary
    /// math. Per-sample walker still descends to true block depth for
    /// correct content lookup, but the DDA's advancement step size
    /// is floored at this depth to keep boundary math inside f32
    /// precision. Deeper edits still exist in the tree; they render
    /// as part of the capped-depth cell until their screen size
    /// demands finer rendering (handled by the ribbon's deeper
    /// frames rendering Cartesian tree around them).
    pub max_term_depth: u32,
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

                if !child_on_preserve && lod_active && child_is_cartesian {
                    let (cx, cy, cz) = slot_coords(slot);
                    let child_center = [
                        node_origin[0] + (cx as f32 + 0.5) * cell_size,
                        node_origin[1] + (cy as f32 + 0.5) * cell_size,
                        node_origin[2] + (cz as f32 + 0.5) * cell_size,
                    ];
                    let dx = child_center[0] - camera_pos[0];
                    let dy = child_center[1] - camera_pos[1];
                    let dz = child_center[2] - camera_pos[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);
                    let screen_pixels = cell_size / dist * half_fov_recip;
                    if screen_pixels < LOD_THRESHOLD {
                        let gpu = if child_node.representative_block < 255 {
                            GpuChild { tag: 1, block_type: child_node.representative_block, _pad: 0, node_index: 0 }
                        } else {
                            GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }
                        };
                        overrides[ordered_idx][slot] = Some(gpu);
                        continue;
                    }
                }

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
        assert_eq!(std::mem::size_of::<GpuRibbonFrame>(), 32);
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
