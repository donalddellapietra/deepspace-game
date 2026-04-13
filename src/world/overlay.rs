//! Overlay subtree rendering for NPCs and other entities.
//!
//! Overlay entries are composited alongside the world tree walk in
//! `render_world`. Each entry describes an entity (NPC, vehicle, etc.)
//! whose body parts are stored as leaf nodes in the shared
//! `NodeLibrary`. The renderer bakes and reconciles overlay meshes
//! using the same mesh cache pipeline as terrain.

use bevy::ecs::hierarchy::ChildOf;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::block::Palette;
use crate::model::mesher::bake_volume;
use crate::model::BakedSubMesh;

use super::state::WorldState;
use super::tree::{voxel_idx, NodeId, EMPTY_VOXEL, NODE_VOXELS_PER_AXIS};

// ------------------------------------------------------ types

/// One body part (or composite) to render as an overlay.
pub struct OverlayPart {
    pub node_id: NodeId,
    /// Offset from entity root in local voxel space (rest + anim).
    pub offset: Vec3,
    /// Part rotation (animation).
    pub rotation: Quat,
    /// Pivot point in voxel-local coords for rotation centre.
    pub pivot: Vec3,
}

/// One entity to composite during rendering.
pub struct OverlayInstance {
    /// Stable Bevy entity ID of the NPC root.
    pub id: Entity,
    pub parts: Vec<OverlayPart>,
    /// Anchor-relative Bevy position.
    pub bevy_pos: Vec3,
    /// Facing rotation.
    pub rotation: Quat,
    /// Uniform scale (maps voxel space to view-cell space).
    pub scale: f32,
}

/// Populated each frame by `collect_overlays`, consumed by the
/// overlay reconcile pass in `render_world`.
#[derive(Resource, Default)]
pub struct OverlayList {
    pub entries: Vec<OverlayInstance>,
}

// ------------------------------------------------------ state

/// Tracks live overlay entities for frame-to-frame reconciliation.
/// Keyed by `(entity ID, part index)`.
#[derive(Default)]
pub struct OverlayState {
    entities: HashMap<(Entity, usize), (Entity, NodeId, Vec3)>,
    /// Leaf-node meshes baked for overlay parts. Keyed by NodeId.
    /// Shared with the world mesh cache conceptually, but stored
    /// separately to avoid coupling with BakedNode.
    meshes: HashMap<NodeId, Vec<BakedSubMesh>>,
}

// ------------------------------------------------------ reconcile

/// Bake + reconcile overlay entities. Called at the end of
/// `render_world` after the world reconcile.
pub fn reconcile_overlays(
    commands: &mut Commands,
    world: &WorldState,
    palette: &Palette,
    meshes: &mut Assets<Mesh>,
    overlay_list: &OverlayList,
    state: &mut OverlayState,
) {
    // Pre-bake any new overlay leaf nodes.
    for entry in &overlay_list.entries {
        for part in &entry.parts {
            if !state.meshes.contains_key(&part.node_id) {
                let node = world
                    .library
                    .get(part.node_id)
                    .expect("overlay: node missing from library");
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
                        let v = voxels[voxel_idx(x as usize, y as usize, z as usize)];
                        if v == EMPTY_VOXEL { None } else { Some(v) }
                    },
                    meshes,
                );
                state.meshes.insert(part.node_id, baked);
            }
        }
    }

    // Reconcile: reuse entities where NodeId matches, despawn+respawn otherwise.
    let mut alive: HashMap<(Entity, usize), (Entity, NodeId, Vec3)> =
        HashMap::with_capacity(
            overlay_list.entries.iter().map(|e| e.parts.len()).sum(),
        );

    for entry in &overlay_list.entries {
        for (part_idx, part) in entry.parts.iter().enumerate() {
            let part_origin = entry.bevy_pos
                + entry.rotation * (entry.scale * part.offset);
            let part_rotation = entry.rotation * part.rotation;

            let key = (entry.id, part_idx);
            let existing = state.entities.remove(&key);

            match existing {
                Some((ent, prev_id, _)) if prev_id == part.node_id => {
                    // Same node — reuse entity, update transform.
                    if let Ok(mut ec) = commands.get_entity(ent) {
                        ec.insert(
                            Transform::from_translation(part_origin)
                                .with_scale(Vec3::splat(entry.scale))
                                .with_rotation(part_rotation),
                        );
                    }
                    alive.insert(key, (ent, part.node_id, part_origin));
                }
                other => {
                    if let Some((old_ent, _, _)) = other {
                        if let Ok(mut ec) = commands.get_entity(old_ent) {
                            ec.despawn();
                        }
                    }

                    let baked = state
                        .meshes
                        .get(&part.node_id)
                        .cloned()
                        .unwrap_or_default();

                    let parent = commands
                        .spawn((
                            Transform::from_translation(part_origin)
                                .with_scale(Vec3::splat(entry.scale))
                                .with_rotation(part_rotation),
                            Visibility::Visible,
                        ))
                        .id();

                    for sub in &baked {
                        let Some(mat) = palette.material(sub.voxel) else {
                            continue;
                        };
                        commands.spawn((
                            Mesh3d(sub.mesh.clone()),
                            MeshMaterial3d(mat),
                            Transform::from_translation(-part.pivot),
                            Visibility::Inherited,
                            ChildOf(parent),
                        ));
                    }

                    alive.insert(key, (parent, part.node_id, part_origin));
                }
            }
        }
    }

    // Despawn overlay entities not visited this frame.
    for (_, (entity, _, _)) in state.entities.drain() {
        if let Ok(mut ec) = commands.get_entity(entity) {
            ec.despawn();
        }
    }
    state.entities = alive;
}

/// Clear all overlay entities (called on zoom change or force rebuild).
pub fn clear_overlay_entities(commands: &mut Commands, state: &mut OverlayState) {
    for (_, (entity, _, _)) in state.entities.drain() {
        if let Ok(mut ec) = commands.get_entity(entity) {
            ec.despawn();
        }
    }
}
