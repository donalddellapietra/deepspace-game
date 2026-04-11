//! Uniform-layer tree-walk renderer (Phase 5 v1).
//!
//! Every frame, walk the content-addressed tree from the root down to
//! `CameraZoom.layer` and emit one Bevy entity per surviving node at
//! that layer. Entities are reused across frames via `RenderState`,
//! and meshes are baked lazily into a `NodeId`-keyed cache. See
//! `docs/architecture/rendering.md` for the full design.
//!
//! Key decisions for v1:
//!
//! * One Bevy unit equals one leaf voxel. A leaf entity has scale
//!   `1.0` and its baked mesh is `bake_volume(NODE_VOXELS_PER_AXIS)`.
//!   A layer-K node has scale `5 ^ (MAX_LAYER - K)`.
//! * The tree is rooted in Bevy space at `ROOT_ORIGIN`, chosen so the
//!   leaf at the all-zero path sits centred directly below the world
//!   origin with its top face at `y = 0`. The camera spawns at
//!   `y ≈ 3`, so it always starts above the grass.
//! * v1 skips frustum culling. Instead it emits every node whose
//!   centre is within `RADIUS_LEAF_UNITS` of the camera. Proper
//!   frustum culling is deferred to Phase 5.1.

use bevy::ecs::hierarchy::ChildOf;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::model::{mesher::bake_volume, BakedSubMesh};

use super::state::WorldState;
use super::tree::{
    block_from_voxel, slot_coords, voxel_idx, NodeId, CHILDREN_PER_NODE,
    EMPTY_NODE, MAX_LAYER, NODE_VOXELS_PER_AXIS,
};

// --------------------------------------------------------------- camera zoom

#[derive(Resource)]
pub struct CameraZoom {
    /// Which tree layer the camera renders. Clamped to
    /// `MIN_ZOOM..=MAX_ZOOM`.
    pub layer: u8,
}

pub const MIN_ZOOM: u8 = 2;
pub const MAX_ZOOM: u8 = MAX_LAYER;

impl Default for CameraZoom {
    fn default() -> Self {
        // Start at the leaf layer: that's where one Bevy unit equals one
        // voxel and the world is readable without any up-level scaling.
        Self { layer: MAX_LAYER }
    }
}

impl CameraZoom {
    pub fn zoom_in(&mut self) -> bool {
        if self.layer < MAX_ZOOM {
            self.layer += 1;
            true
        } else {
            false
        }
    }
    pub fn zoom_out(&mut self) -> bool {
        if self.layer > MIN_ZOOM {
            self.layer -= 1;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------- constants

/// Origin of the root node in Bevy space.
///
/// `y` is chosen so that the top face of the `GROUND_Y_VOXELS`-deep
/// grass surface lines up with Bevy `y = 0`. The root-local y range
/// `(0, GROUND_Y_VOXELS)` maps to Bevy `(-GROUND_Y_VOXELS, 0)`, and
/// the all-zero-path leaf sits at the very bottom of the grass
/// region.
///
/// `x` and `z` are **integer offsets**. It is important that they
/// are integer: the raycast in `src/interaction/mod.rs` steps through
/// Bevy integer cells `[c, c+1]`, and the highlight gizmo centres a
/// unit cube at `c + 0.5`. If `ROOT_ORIGIN.{x,z}` had a fractional
/// part, voxels would sit on a non-integer lattice and the outline
/// would drift half a voxel away from the block it names.
///
/// `-13` puts the all-zero-path leaf at Bevy `(-13..12)` on each of
/// x and z, so Bevy `(0, ·, 0)` is near the leaf's centre (voxel 13
/// out of 25). The player spawns at `(0, 2, 0)`, a couple of voxels
/// above the grass surface, and falls onto it.
pub const ROOT_ORIGIN: Vec3 = Vec3::new(
    -13.0,
    -(super::state::GROUND_Y_VOXELS as f32),
    -13.0,
);

/// How far a rendered node's centre may be from the camera (in leaf
/// units, i.e. Bevy units since `1 unit = 1 leaf`) and still be
/// emitted. Used as the v1 replacement for frustum culling.
const RADIUS_LEAF_UNITS: f32 = 400.0;

// ----------------------------------------------------------------- state

/// A map from tree path (of length `depth`) to the Bevy entity
/// currently rendering that path. `NodeId` tracks what the entity's
/// mesh is a function of, so edits naturally invalidate it.
#[derive(Resource, Default)]
pub struct RenderState {
    /// Cached per-`NodeId` baked sub-meshes. An entry survives across
    /// frames and zoom layers — the mesh is a function of the node's
    /// voxel grid, not of which layer it was emitted at. v1 never
    /// evicts these.
    meshes: HashMap<NodeId, Vec<BakedSubMesh>>,
    /// Live entities, keyed by "path prefix" (a `SmallPath`).
    entities: HashMap<SmallPath, (Entity, NodeId)>,
    /// Zoom layer the `entities` set was built for. If it changes,
    /// everything gets despawned and rebuilt.
    last_zoom_layer: u8,
    /// Whether we have done at least one render pass.
    initialised: bool,
}

/// A compact identifier for a node's position in the tree during a
/// single-frame walk: `depth` significant slot indices from the root.
/// Used as the `RenderState.entities` key, so reuse survives across
/// frames as long as the camera keeps looking at the same spot.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct SmallPath {
    depth: u8,
    slots: [u8; MAX_LAYER as usize],
}

impl SmallPath {
    fn empty() -> Self {
        Self {
            depth: 0,
            slots: [0; MAX_LAYER as usize],
        }
    }

    fn push(&self, slot: u8) -> Self {
        let mut out = *self;
        out.slots[out.depth as usize] = slot;
        out.depth += 1;
        out
    }
}

// ------------------------------------------------------------- math helpers

/// Scale multiplier for a node at `layer`. A layer-`MAX_LAYER` (leaf)
/// node has scale `1.0`; a layer-`MAX_LAYER - 1` node has scale `5.0`;
/// the root (layer 0) has scale `5^MAX_LAYER`. Values get huge at the
/// root but stay finite in f32 — only ~`6e8` at layer 0.
#[inline]
fn scale_for_layer(layer: u8) -> f32 {
    let up = (MAX_LAYER - layer) as i32;
    // 5^up. Up to 5^12 ≈ 2.4e8, well within f32.
    let mut acc: f32 = 1.0;
    for _ in 0..up {
        acc *= 5.0;
    }
    acc
}

/// Bevy-space extent (per axis) of a node at `layer`.
#[inline]
fn extent_for_layer(layer: u8) -> f32 {
    scale_for_layer(layer) * (NODE_VOXELS_PER_AXIS as f32)
}

/// Bevy-space edge length of ONE cell inside a layer-`layer` node's
/// `25³` grid. At the leaf layer this is `1.0` (a leaf cell is a
/// unit cube in Bevy space); at `layer = MAX_LAYER - 1` it's `5.0`;
/// at the root it's `5^MAX_LAYER`.
///
/// The editor and the highlight gizmo use this to size the
/// per-view-layer outline cube.
#[inline]
pub fn cell_size_at_layer(layer: u8) -> f32 {
    scale_for_layer(layer)
}

// ----------------------------------------------------------- mesh caching

fn get_or_bake_mesh<'a>(
    render_state: &'a mut RenderState,
    world: &WorldState,
    node_id: NodeId,
    meshes: &mut Assets<Mesh>,
) -> &'a [BakedSubMesh] {
    if !render_state.meshes.contains_key(&node_id) {
        let node = world
            .library
            .get(node_id)
            .expect("render: node missing from library");
        let voxels = node.voxels.clone();
        let baked = bake_volume(
            NODE_VOXELS_PER_AXIS as i32,
            move |x, y, z| {
                if x < 0
                    || y < 0
                    || z < 0
                    || x >= NODE_VOXELS_PER_AXIS as i32
                    || y >= NODE_VOXELS_PER_AXIS as i32
                    || z >= NODE_VOXELS_PER_AXIS as i32
                {
                    return None;
                }
                block_from_voxel(
                    voxels[voxel_idx(x as usize, y as usize, z as usize)],
                )
            },
            meshes,
        );
        render_state.meshes.insert(node_id, baked);
    }
    render_state
        .meshes
        .get(&node_id)
        .expect("just inserted")
        .as_slice()
}

// ------------------------------------------------------------- tree walk

/// One "visit" the tree walk wants the reconciler to spawn/update.
struct Visit {
    path: SmallPath,
    node_id: NodeId,
    origin: Vec3,
}

fn walk(
    world: &WorldState,
    target_layer: u8,
    camera_pos: Vec3,
    out: &mut Vec<Visit>,
) {
    if world.root == EMPTY_NODE {
        return;
    }
    let mut stack: Vec<(NodeId, SmallPath, Vec3, u8)> = Vec::with_capacity(256);
    stack.push((world.root, SmallPath::empty(), ROOT_ORIGIN, 0));

    let radius_sq = RADIUS_LEAF_UNITS * RADIUS_LEAF_UNITS;

    while let Some((node_id, path, origin, depth)) = stack.pop() {
        // Compute this node's AABB in Bevy space.
        let extent = extent_for_layer(depth);
        let aabb_min = origin;
        let aabb_max = origin + Vec3::splat(extent);

        // Per-axis "distance from camera to AABB": 0 if inside,
        // otherwise the gap to the nearest face. L2 norm of the three
        // gaps is the minimum distance from the camera point to the
        // AABB. Much more accurate than a centre-based sphere test
        // when the AABB is very large and contains the camera.
        let dx = (aabb_min.x - camera_pos.x)
            .max(0.0)
            .max(camera_pos.x - aabb_max.x);
        let dy = (aabb_min.y - camera_pos.y)
            .max(0.0)
            .max(camera_pos.y - aabb_max.y);
        let dz = (aabb_min.z - camera_pos.z)
            .max(0.0)
            .max(camera_pos.z - aabb_max.z);
        let min_dist_sq = dx * dx + dy * dy + dz * dz;
        if min_dist_sq > radius_sq {
            continue;
        }

        // Reached the target layer → emit.
        if depth == target_layer {
            out.push(Visit {
                path,
                node_id,
                origin,
            });
            continue;
        }

        // Descend into children. If this node is already a leaf
        // (no children) we can't go deeper — emit it at its actual
        // layer instead. This shouldn't happen in a fully-materialised
        // grassland world but is safe.
        let Some(node) = world.library.get(node_id) else { continue };
        let Some(children) = node.children.as_ref() else {
            out.push(Visit {
                path,
                node_id,
                origin,
            });
            continue;
        };

        let child_extent = extent_for_layer(depth + 1);
        for slot in 0..CHILDREN_PER_NODE {
            let child_id = children[slot];
            if child_id == EMPTY_NODE {
                continue;
            }
            let (sx, sy, sz) = slot_coords(slot);
            let child_origin = origin
                + Vec3::new(
                    sx as f32 * child_extent,
                    sy as f32 * child_extent,
                    sz as f32 * child_extent,
                );
            let child_path = path.push(slot as u8);
            stack.push((child_id, child_path, child_origin, depth + 1));
        }
    }
}

// ----------------------------------------------------------------- system

/// Bevy system: walk the tree, reconcile `RenderState` entities.
pub fn render_world(
    mut commands: Commands,
    world: Res<WorldState>,
    zoom: Res<CameraZoom>,
    camera_q: Query<&Transform, With<Camera3d>>,
    materials: Option<Res<BlockMaterials>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut render_state: ResMut<RenderState>,
) {
    let Some(materials) = materials else {
        return;
    };
    let Ok(camera_tf) = camera_q.single() else {
        return;
    };
    let camera_pos = camera_tf.translation;

    let target_layer = zoom.layer.clamp(0, MAX_LAYER);

    // If zoom changed, or we're on our first pass, drop everything.
    if !render_state.initialised || render_state.last_zoom_layer != target_layer
    {
        for (_, (entity, _)) in render_state.entities.drain() {
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.despawn();
            }
        }
        render_state.last_zoom_layer = target_layer;
        render_state.initialised = true;
    }

    // Walk the tree → collect visible positions.
    let mut visits: Vec<Visit> = Vec::with_capacity(256);
    walk(&world, target_layer, camera_pos, &mut visits);

    let node_scale = scale_for_layer(target_layer);

    // Reconcile: what's alive now, what changed, what's gone.
    let mut alive: HashMap<SmallPath, (Entity, NodeId)> =
        HashMap::with_capacity(visits.len());

    for visit in visits {
        let new_node_id = visit.node_id;
        let existing = render_state.entities.remove(&visit.path);

        match existing {
            Some((entity, existing_id)) if existing_id == new_node_id => {
                // Reuse. Update translation (cheap; handles float
                // drift if it ever creeps in).
                if let Ok(mut ec) = commands.get_entity(entity) {
                    ec.insert(
                        Transform::from_translation(visit.origin)
                            .with_scale(Vec3::splat(node_scale)),
                    );
                }
                alive.insert(visit.path, (entity, existing_id));
            }
            other => {
                if let Some((old_entity, _)) = other {
                    if let Ok(mut ec) = commands.get_entity(old_entity) {
                        ec.despawn();
                    }
                }
                // Spawn a new root entity for this path, parent one
                // child per sub-mesh.
                let baked = get_or_bake_mesh(
                    &mut render_state,
                    &world,
                    new_node_id,
                    &mut meshes,
                )
                .to_vec();

                let parent = commands
                    .spawn((
                        Transform::from_translation(visit.origin)
                            .with_scale(Vec3::splat(node_scale)),
                        Visibility::Visible,
                    ))
                    .id();

                for sub in &baked {
                    commands.spawn((
                        Mesh3d(sub.mesh.clone()),
                        MeshMaterial3d(materials.get(sub.block_type)),
                        Transform::default(),
                        Visibility::Inherited,
                        ChildOf(parent),
                    ));
                }

                alive.insert(visit.path, (parent, new_node_id));
            }
        }
    }

    // Everything left in the old map was NOT visited this frame →
    // despawn.
    for (_, (entity, _)) in render_state.entities.drain() {
        if let Ok(mut ec) = commands.get_entity(entity) {
            ec.despawn();
        }
    }
    render_state.entities = alive;
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_at_leaf_is_one() {
        assert_eq!(scale_for_layer(MAX_LAYER), 1.0);
    }

    #[test]
    fn scale_at_layer_above_leaf_is_five() {
        assert_eq!(scale_for_layer(MAX_LAYER - 1), 5.0);
    }

    #[test]
    fn extent_at_leaf_is_node_voxels() {
        assert_eq!(extent_for_layer(MAX_LAYER), NODE_VOXELS_PER_AXIS as f32);
    }

    #[test]
    fn small_path_push() {
        let p = SmallPath::empty().push(7).push(12);
        assert_eq!(p.depth, 2);
        assert_eq!(p.slots[0], 7);
        assert_eq!(p.slots[1], 12);
    }

    #[test]
    fn walk_grassland_at_leaves_emits_at_least_one_visit() {
        let world = WorldState::new_grassland();
        let mut visits = Vec::new();
        walk(&world, MAX_LAYER, Vec3::ZERO, &mut visits);
        assert!(
            !visits.is_empty(),
            "grassland walk at leaf layer should emit at least one visit"
        );
    }

    #[test]
    fn walk_radius_limits_emit_count() {
        let world = WorldState::new_grassland();
        let mut visits = Vec::new();
        walk(&world, MAX_LAYER, Vec3::ZERO, &mut visits);
        // Radius ~400 at scale 1.0 → bounded number of leaves. Hard
        // ceiling chosen generously; this test is a guard rail against
        // accidentally unbounded walks.
        assert!(
            visits.len() < 200_000,
            "walk emitted {} visits; expected far fewer",
            visits.len()
        );
    }
}
