//! Uniform-layer tree-walk renderer.
//!
//! Every frame, walk the content-addressed tree from the root down to
//! `CameraZoom.layer + 2` (the target layer) and emit one Bevy entity
//! per surviving node. Meshes are resolved via `MeshStore` which
//! checks prebaked data first, then falls back to cold baking.
//!
//! See `docs/architecture/rendering.md` for the full design.

use bevy::ecs::hierarchy::ChildOf;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::block::Palette;
use crate::model::BakedSubMesh;

use super::mesh_cache::MeshStore;
use super::state::WorldState;
use super::tree::{NodeId, EMPTY_NODE, MAX_LAYER};
use super::view::{cell_size_at_layer, target_layer_for, WorldAnchor};
use super::walk::{walk, SmallPath, Visit, WalkFrame};

// ------------------------------------------------------- markers

/// Marker on the parent entity of a rendered node.
#[derive(Component)]
pub struct WorldRenderedNode(pub NodeId);

/// Marker on each child sub-mesh entity, remembering its voxel index.
#[derive(Component)]
pub struct SubMeshBlock(pub u8);

// --------------------------------------------------------------- camera zoom

#[derive(Resource)]
pub struct CameraZoom {
    pub layer: u8,
}

pub const MIN_ZOOM: u8 = 2;
pub const MAX_ZOOM: u8 = MAX_LAYER;

impl Default for CameraZoom {
    fn default() -> Self {
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

pub const RADIUS_VIEW_CELLS: f32 = 32.0;

// ----------------------------------------------------------------- state

#[derive(Resource, Default)]
pub struct RenderTimings {
    pub render_total_us: u64,
    pub walk_us: u64,
    pub bake_us: u64,
    pub reconcile_us: u64,
    pub visit_count: usize,
    pub group_count: usize,
    pub collision_us: u64,
    pub cold_bakes: usize,
    pub unbaked: usize,
}

#[derive(Resource, Default)]
pub struct RenderState {
    /// Mesh store: handles prebaked loading, cold baking, and caching.
    pub mesh_store: MeshStore,
    /// Live entities, keyed by tree path.
    entities: HashMap<SmallPath, (Entity, NodeId, Vec3)>,
    /// Zoom layer the entities were built for.
    last_zoom_layer: u8,
    initialised: bool,
    pub force_rebuild: bool,
    pub overlay: super::overlay::OverlayState,
    walk_stack: Vec<WalkFrame>,
    visits: Vec<Visit>,
}

// ----------------------------------------------------------------- system

/// Bevy system: walk the tree, reconcile `RenderState` entities.
pub fn render_world(
    mut commands: Commands,
    world: Res<WorldState>,
    zoom: Res<CameraZoom>,
    anchor: Res<WorldAnchor>,
    camera_q: Query<&Transform, With<Camera3d>>,
    palette: Option<Res<Palette>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut render_state: ResMut<RenderState>,
    mut timings: ResMut<RenderTimings>,
    overlay_list: Res<super::overlay::OverlayList>,
) {
    let Some(palette) = palette else { return };
    let Ok(camera_tf) = camera_q.single() else { return };
    let camera_pos = camera_tf.translation;
    let render_total_start = bevy::platform::time::Instant::now();

    // Load prebaked meshes on first frame.
    render_state.mesh_store.ensure_loaded();

    let target_layer = target_layer_for(zoom.layer);
    let emit_layer = target_layer.saturating_sub(1);
    let radius_bevy = RADIUS_VIEW_CELLS * cell_size_at_layer(zoom.layer);

    // On zoom change or first pass, drop entities but keep mesh cache.
    if !render_state.initialised
        || render_state.last_zoom_layer != zoom.layer
        || render_state.force_rebuild
    {
        render_state.force_rebuild = false;
        for (_, (entity, _, _)) in render_state.entities.drain() {
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.despawn();
            }
        }
        super::overlay::clear_overlay_entities(&mut commands, &mut render_state.overlay);
        render_state.mesh_store.clear_paths();
        render_state.last_zoom_layer = zoom.layer;
        render_state.initialised = true;
    }

    // Receive completed async mesh loads from the I/O thread.
    render_state.mesh_store.receive_async_meshes(&mut meshes);

    // Walk the tree → collect visible nodes.
    timings.bake_us = 0;
    let walk_start = bevy::platform::time::Instant::now();
    let mut walk_stack = std::mem::take(&mut render_state.walk_stack);
    let mut visits = std::mem::take(&mut render_state.visits);
    walk(
        &world, emit_layer, target_layer,
        camera_pos, radius_bevy, &anchor,
        &mut walk_stack, &mut visits,
    );
    timings.walk_us = walk_start.elapsed().as_micros() as u64;
    timings.visit_count = visits.len();

    // Bake pass: ensure every visited node has a mesh, or request it
    // async. Nodes pending async load return false — the reconcile
    // step will use parent-fallback for them.
    const MAX_COLD_BAKES: usize = 16;
    {
        let bake_start = bevy::platform::time::Instant::now();
        let mut cold_bakes = 0usize;
        for v in visits.iter() {
            let node = world.library.get(v.node_id)
                .expect("render: node missing from library");

            // Distance-based priority: closer nodes load first.
            let dist = v.origin.distance(camera_pos);
            let priority = (dist * 10.0) as u32;

            render_state.mesh_store.ensure_baked(
                &world, v.node_id, &v.path, &mut meshes,
                &mut cold_bakes, MAX_COLD_BAKES, priority,
            );

            if node.children.is_some() {
                render_state.mesh_store.set_path_node(v.path, v.node_id);
            }
        }
        timings.bake_us = bake_start.elapsed().as_micros() as u64;
        timings.cold_bakes = cold_bakes;
        timings.unbaked = visits.iter()
            .filter(|v| !render_state.mesh_store.is_baked(v.node_id))
            .count();
    }

    // Reconcile: spawn/update/despawn entities.
    let reconcile_start = bevy::platform::time::Instant::now();
    let mut alive: HashMap<SmallPath, (Entity, NodeId, Vec3)> =
        HashMap::with_capacity(visits.len());

    for visit in visits.drain(..) {
        let new_node_id = visit.node_id;
        let existing = render_state.entities.remove(&visit.path);

        match existing {
            Some((entity, existing_id, last_origin))
                if existing_id == new_node_id =>
            {
                if last_origin != visit.origin {
                    if let Ok(mut ec) = commands.get_entity(entity) {
                        ec.insert(
                            Transform::from_translation(visit.origin)
                                .with_scale(Vec3::splat(visit.scale)),
                        );
                    }
                }
                alive.insert(visit.path, (entity, existing_id, visit.origin));
            }
            other => {
                if let Some((old_entity, _, _)) = other {
                    if let Ok(mut ec) = commands.get_entity(old_entity) {
                        ec.despawn();
                    }
                }

                let baked = render_state.mesh_store.get_merged(new_node_id)
                    .unwrap_or(&[]);

                let parent = commands
                    .spawn((
                        WorldRenderedNode(new_node_id),
                        Transform::from_translation(visit.origin)
                            .with_scale(Vec3::splat(visit.scale)),
                        Visibility::Visible,
                    ))
                    .id();

                for sub in baked {
                    let Some(mat) = palette.material(sub.voxel) else {
                        continue;
                    };
                    commands.spawn((
                        Mesh3d(sub.mesh.clone()),
                        MeshMaterial3d(mat),
                        SubMeshBlock(sub.voxel),
                        Transform::default(),
                        Visibility::Inherited,
                        ChildOf(parent),
                    ));
                }

                alive.insert(visit.path, (parent, new_node_id, visit.origin));
            }
        }
    }

    for (_, (entity, _, _)) in render_state.entities.drain() {
        if let Ok(mut ec) = commands.get_entity(entity) {
            ec.despawn();
        }
    }
    render_state.entities = alive;

    // Overlay reconcile (NPCs).
    super::overlay::reconcile_overlays(
        &mut commands, &world, &palette, &mut meshes,
        &overlay_list, &mut render_state.overlay,
    );

    timings.reconcile_us = reconcile_start.elapsed().as_micros() as u64;

    // Prefetch: async-request meshes for adjacent zoom layers so the
    // next zoom transition has data ready. Non-blocking, no frame cost
    // beyond the walk itself.
    {
        let prefetch_layers: [Option<u8>; 2] = [
            if zoom.layer > MIN_ZOOM { Some(zoom.layer - 1) } else { None },
            if zoom.layer < MAX_ZOOM { Some(zoom.layer + 1) } else { None },
        ];
        for pf_layer in prefetch_layers.into_iter().flatten() {
            let pf_target = target_layer_for(pf_layer);
            let pf_emit = pf_target.saturating_sub(1);
            let pf_radius = RADIUS_VIEW_CELLS * cell_size_at_layer(pf_layer);
            walk(
                &world, pf_emit, pf_target,
                camera_pos, pf_radius, &anchor,
                &mut walk_stack, &mut visits,
            );
            for pv in visits.iter() {
                let dist = pv.origin.distance(camera_pos);
                render_state.mesh_store.prefetch(pv.node_id, (dist * 10.0) as u32);
            }
        }
    }

    timings.render_total_us = render_total_start.elapsed().as_micros() as u64;

    render_state.walk_stack = walk_stack;
    render_state.visits = visits;
}
