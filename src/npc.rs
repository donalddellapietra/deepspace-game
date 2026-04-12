//! Spawn animated NPC characters from glTF assets.
//!
//! Press `N` to spawn a fox NPC near the player. The NPC uses a
//! `WorldPosition` (same as the player) so it lives on the voxel
//! grid, obeys gravity and collision, and gets anchor-relative
//! `Transform` derivation for free via `derive_transforms`.
//!
//! NPCs randomly walk around at the leaf layer and despawn when the
//! camera zooms out past their visible scale.

use std::f32::consts::TAU;
use std::time::Duration;

use bevy::prelude::*;
use bevy::scene::SceneInstanceReady;

use crate::camera::FpsCam;
use crate::player::Player;
use crate::world::collision;
use crate::world::position::Position;
use crate::world::tree::MAX_LAYER;
use crate::world::view::{
    bevy_from_position, cell_size_at_layer, position_from_bevy, position_to_leaf_coord,
    WorldAnchor,
};
use crate::world::{CameraZoom, WorldPosition, WorldState};

const FOX_PATH: &str = "characters/Fox.glb";

/// Approximate height of the Fox model in its own coordinate space.
const FOX_MODEL_HEIGHT: f32 = 80.0;

/// NPCs are visible at this layer and below (finer). At coarser
/// layers they despawn (fold into statistical representation).
const NPC_MAX_VISIBLE_LAYER: u8 = MAX_LAYER;

/// NPC walk speed in cells per second (same unit as player).
const NPC_WALK_SPEED_CELLS: f32 = 3.0;
/// Gravity in cells per second² (same as player).
const NPC_GRAVITY_CELLS: f32 = 20.0;

pub struct NpcPlugin;

impl Plugin for NpcPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, preload_npc_assets)
            .add_systems(
                Update,
                (
                    spawn_npc_on_keypress,
                    npc_ai,
                    npc_physics,
                    npc_zoom_visibility,
                ),
            );
    }
}

/// Marker for NPC root entities.
#[derive(Component)]
pub struct Npc;

/// NPC velocity (same idea as player's Velocity).
#[derive(Component)]
struct NpcVelocity(Vec3);

/// Simple random-walk AI state.
#[derive(Component)]
struct NpcAi {
    /// Current heading in radians.
    heading: f32,
    /// Time remaining before picking a new heading.
    change_timer: Timer,
}

/// Loaded glTF handle.
#[derive(Resource)]
struct NpcGltf(Handle<Gltf>);

/// Animation graph built once the glTF is ready.
#[derive(Resource, Clone)]
struct NpcAnimations {
    graph_handle: Handle<AnimationGraph>,
    walk: AnimationNodeIndex,
}

/// Per-entity reference to the animation to play once the scene loads.
#[derive(Component, Clone)]
struct NpcAnimationToPlay {
    graph_handle: Handle<AnimationGraph>,
    index: AnimationNodeIndex,
}

fn preload_npc_assets(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(NpcGltf(asset_server.load(FOX_PATH)));
}

fn spawn_npc_on_keypress(
    mut commands: Commands,
    keyboard: Res<ButtonInput<KeyCode>>,
    gltf_handle: Option<Res<NpcGltf>>,
    asset_server: Res<AssetServer>,
    gltfs: Res<Assets<Gltf>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    animations_res: Option<Res<NpcAnimations>>,
    player_q: Query<&WorldPosition, With<Player>>,
    anchor: Res<WorldAnchor>,
    zoom: Res<CameraZoom>,
    cam_q: Query<&FpsCam>,
) {
    if !keyboard.just_pressed(KeyCode::KeyN) {
        return;
    }

    let Some(gltf_handle) = gltf_handle else { return };
    if !asset_server.is_loaded_with_dependencies(&gltf_handle.0) {
        info!("NPC asset still loading...");
        return;
    }

    let gltf = gltfs
        .get(&gltf_handle.0)
        .expect("loaded glTF should exist");

    // Build animation graph on first spawn.
    let animations = if let Some(res) = animations_res {
        res.clone()
    } else {
        let (graph, indices) =
            AnimationGraph::from_clips([gltf.named_animations["Walk"].clone()]);
        let graph_handle = graphs.add(graph);
        let anims = NpcAnimations {
            graph_handle,
            walk: indices[0],
        };
        commands.insert_resource(anims.clone());
        anims
    };

    // Place the NPC a few cells in front of the player.
    let Ok(player_pos) = player_q.single() else { return };
    let cell = cell_size_at_layer(zoom.layer);
    let forward = if let Ok(cam) = cam_q.single() {
        Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos())
    } else {
        Vec3::NEG_Z
    };

    // Compute the NPC's Bevy position, then convert back to a
    // WorldPosition through the proper leaf-coord path. This
    // handles all coordinate wrapping correctly.
    let player_bevy = bevy_from_position(&player_pos.0, &anchor);
    let npc_bevy = player_bevy + forward * 5.0 * cell + Vec3::Y * 1.0;
    let Some(npc_pos) = position_from_bevy(npc_bevy, &anchor) else {
        warn!("NPC spawn position outside world");
        return;
    };

    let spawn_bevy = bevy_from_position(&npc_pos, &anchor);
    let scale = npc_scale(zoom.layer);

    let scene_handle = gltf
        .default_scene
        .clone()
        .expect("Fox.glb has a default scene");

    // Random initial heading.
    let heading = rand_heading();

    commands
        .spawn((
            Npc,
            WorldPosition(npc_pos),
            NpcVelocity(Vec3::ZERO),
            NpcAi {
                heading,
                change_timer: Timer::from_seconds(2.0 + rand_f32() * 3.0, TimerMode::Once),
            },
            NpcAnimationToPlay {
                graph_handle: animations.graph_handle.clone(),
                index: animations.walk,
            },
            SceneRoot(scene_handle),
            Transform::from_translation(spawn_bevy)
                .with_scale(Vec3::splat(scale))
                .with_rotation(Quat::from_rotation_y(heading)),
        ))
        .observe(play_npc_animation);

    info!("Spawned NPC at {spawn_bevy:?} (layer {})", zoom.layer);
}

fn play_npc_animation(
    ready: On<SceneInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    anim_q: Query<&NpcAnimationToPlay>,
    mut players: Query<(Entity, &mut AnimationPlayer)>,
) {
    let target = ready.event().entity;
    let Ok(anim) = anim_q.get(target) else { return };

    for child in children.iter_descendants(target) {
        if let Ok((entity, mut player)) = players.get_mut(child) {
            let mut transitions = AnimationTransitions::new();
            transitions
                .play(&mut player, anim.index, Duration::ZERO)
                .repeat();
            commands
                .entity(entity)
                .insert(AnimationGraphHandle(anim.graph_handle.clone()))
                .insert(transitions);
        }
    }
}

/// Simple random-walk: pick a heading, walk that direction for a few
/// seconds, pick a new one. NPCs live at MAX_LAYER.
fn npc_ai(time: Res<Time>, mut q: Query<(&mut NpcAi, &mut NpcVelocity, &mut Transform), With<Npc>>) {
    let dt = time.delta_secs();
    for (mut ai, mut vel, mut tf) in &mut q {
        ai.change_timer.tick(time.delta());

        if ai.change_timer.just_finished() {
            ai.heading = rand_heading();
            ai.change_timer = Timer::from_seconds(2.0 + rand_f32() * 3.0, TimerMode::Once);
        }

        let speed = NPC_WALK_SPEED_CELLS; // At leaf layer, cell = 1.0.
        // Movement direction from heading (same convention as camera).
        let dir_x = -ai.heading.sin();
        let dir_z = -ai.heading.cos();
        vel.0.x = dir_x * speed;
        vel.0.z = dir_z * speed;

        // Face the movement direction. The fox model faces +Z in its
        // local space, so we rotate by PI + heading to align it with
        // the (-sin, -cos) movement vector.
        tf.rotation = Quat::from_rotation_y(ai.heading + std::f32::consts::PI);

        let _ = dt; // used indirectly via timer tick
    }
}

/// Apply gravity and collision to NPCs, then let `derive_transforms`
/// handle the Bevy Transform (it runs later in the player plugin).
fn npc_physics(
    time: Res<Time>,
    world: Res<WorldState>,
    zoom: Res<CameraZoom>,
    mut q: Query<(&mut WorldPosition, &mut NpcVelocity), With<Npc>>,
) {
    let dt = time.delta_secs();
    let cell = cell_size_at_layer(zoom.layer);

    for (mut wpos, mut vel) in &mut q {
        // Gravity (in cells/s², scaled to leaves).
        vel.0.y -= NPC_GRAVITY_CELLS * cell * dt;

        let h_delta = bevy::math::Vec2::new(vel.0.x * cell * dt, vel.0.z * cell * dt);

        collision::move_and_collide(
            &mut wpos.0,
            &mut vel.0,
            h_delta,
            dt,
            &world,
            zoom.layer,
        );
    }
}

/// Rescale NPCs with zoom. They stay visible (just smaller) when 1
/// layer out. Despawn when 2+ layers out from their native layer.
fn npc_zoom_visibility(
    mut commands: Commands,
    zoom: Res<CameraZoom>,
    mut q: Query<(Entity, &mut Visibility, &mut Transform), With<Npc>>,
) {
    if !zoom.is_changed() {
        return;
    }

    let scale = npc_scale(zoom.layer);

    for (entity, mut vis, mut tf) in &mut q {
        let layers_out = NPC_MAX_VISIBLE_LAYER.saturating_sub(zoom.layer);
        info!(
            "NPC zoom: layer={}, layers_out={layers_out}, scale={scale}",
            zoom.layer
        );
        if layers_out >= 2 {
            // 2+ layers out — despawn entirely.
            commands.entity(entity).despawn();
        } else {
            // At native layer or 1 layer out — still visible, just
            // proportionally smaller.
            *vis = Visibility::Inherited;
            tf.scale = Vec3::splat(scale);
        }
    }
}

// -------------------------------------------------- helpers

fn npc_scale(view_layer: u8) -> f32 {
    let cell = cell_size_at_layer(view_layer);
    (1.5 * cell) / FOX_MODEL_HEIGHT
}

/// Simple deterministic-ish random float in [0, 1).
fn rand_f32() -> f32 {
    // Use system time nanoseconds as cheap entropy.
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (t % 10000) as f32 / 10000.0
}

/// Random heading in [0, TAU).
fn rand_heading() -> f32 {
    rand_f32() * TAU
}
