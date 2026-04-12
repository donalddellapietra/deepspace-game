//! Voxel-based NPC system.
//!
//! NPCs are small voxel models composed of articulated body parts
//! (head, torso, arms, legs). Each part is a small voxel grid that
//! gets greedy-meshed exactly like terrain, rendered with the same
//! materials, and is editable with the same block tools.
//!
//! Animation is rigid-body keyframe interpolation: each pose defines
//! per-part transforms (position + rotation), and the system lerps
//! between keyframes. No re-meshing needed per frame.
//!
//! Press `N` to spawn a voxel NPC near the player.

use std::f32::consts::{PI, TAU};
use std::collections::HashMap;

use bevy::prelude::*;

use crate::block::Palette;
use crate::camera::FpsCam;
use crate::model::mesher::bake_volume;
use crate::model::BakedSubMesh;
use crate::player::Player;
use crate::world::collision;
use crate::world::tree::{EMPTY_VOXEL, Voxel};
use crate::world::view::{
    bevy_from_position, cell_size_at_layer, position_from_bevy,
    WorldAnchor,
};
use crate::world::{CameraZoom, WorldPosition, WorldState};

// ================================================================ data

/// A single body part: a small voxel grid with a pivot point.
#[derive(Clone)]
pub struct VoxelPart {
    pub name: String,
    pub size: [u8; 3],
    pub voxels: Vec<Voxel>,
    /// Pivot point in voxel-local coords (rotation center).
    pub pivot: Vec3,
    /// Default offset from the NPC root (rest pose position).
    pub rest_offset: Vec3,
}

impl VoxelPart {
}

/// One keyframe: per-part position offset and rotation.
#[derive(Clone)]
pub struct NpcPose {
    /// Keyed by part name → (position offset from rest, rotation).
    pub parts: HashMap<String, (Vec3, Quat)>,
}

/// A named animation clip.
#[derive(Clone)]
pub struct NpcAnimation {
    pub keyframes: Vec<NpcPose>,
    pub frame_duration: f32,
    pub looping: bool,
}

/// Shared blueprint for an NPC type. All instances reference this.
#[derive(Clone, Resource)]
pub struct NpcBlueprint {
    pub name: String,
    pub parts: Vec<VoxelPart>,
    pub animations: HashMap<String, NpcAnimation>,
}

// ============================================= baked mesh cache

/// Cached meshes for each part of a blueprint, keyed by part index.
#[derive(Resource, Default)]
struct NpcMeshCache {
    /// blueprint_name → vec of per-part sub-meshes
    meshes: HashMap<String, Vec<Vec<BakedSubMesh>>>,
}

fn bake_part(
    part: &VoxelPart,
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    let sx = part.size[0] as i32;
    let sy = part.size[1] as i32;
    let sz = part.size[2] as i32;
    let voxels = part.voxels.clone();
    let size_x = sx;
    let size_y = sy;

    // The mesher expects a cubic `size` parameter, but we can use the
    // max dimension and let out-of-bounds lookups return None.
    let max_dim = sx.max(sy).max(sz);

    bake_volume(
        max_dim,
        move |x, y, z| {
            if x < 0 || y < 0 || z < 0 || x >= sx || y >= sy || z >= sz {
                return None;
            }
            let idx = (z * size_y + y) * size_x + x;
            let v = voxels[idx as usize];
            if v == EMPTY_VOXEL { None } else { Some(v) }
        },
        meshes,
    )
}

// ====================================================== components

/// Marker for NPC root entities.
#[derive(Component)]
pub struct Npc;

/// NPC velocity (same idea as player's Velocity).
#[derive(Component)]
struct NpcVelocity(Vec3);

/// Simple random-walk AI state.
#[derive(Component)]
struct NpcAi {
    heading: f32,
    change_timer: Timer,
}

/// Which animation is playing and where we are in it.
#[derive(Component)]
struct NpcAnimState {
    current_anim: String,
    time: f32,
}

/// Links a child entity to its part index in the blueprint.
#[derive(Component)]
struct NpcPartIndex(usize);

// ========================================================= constants

const NPC_WALK_SPEED_CELLS: f32 = 3.0;
const NPC_GRAVITY_CELLS: f32 = 20.0;

// ============================================================ plugin

pub struct NpcPlugin;

impl Plugin for NpcPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NpcMeshCache>()
            .add_systems(Startup, init_default_blueprint)
            .add_systems(
                Update,
                (
                    spawn_npc_on_keypress,
                    npc_ai,
                    npc_animate,
                    npc_physics,
                    npc_zoom_visibility,
                ),
            );
    }
}

// ======================================== default humanoid blueprint

fn init_default_blueprint(mut commands: Commands) {
    commands.insert_resource(build_humanoid_blueprint());
}

/// Build a default humanoid NPC from procedural voxel art.
/// Body proportions (in voxels): ~5 wide, ~12 tall, ~3 deep.
fn build_humanoid_blueprint() -> NpcBlueprint {
    let skin: Voxel = 8;  // Metal (light grey — skin-ish)
    let shirt: Voxel = 6; // Sand (warm tone for shirt)
    let pants: Voxel = 7; // Water (blue for pants)
    let hair: Voxel = 4;  // Wood (brown hair)
    let shoe: Voxel = 1;  // Stone (dark shoes)

    // ---- Head (4×4×4) ----
    let head = {
        let (sx, sy, sz) = (4u8, 4, 4);
        let mut v = vec![EMPTY_VOXEL; (sx as usize) * (sy as usize) * (sz as usize)];
        let idx = |x: usize, y: usize, z: usize| (z * sy as usize + y) * sx as usize + x;
        // Fill solid head
        for z in 0..sz as usize {
            for y in 0..sy as usize {
                for x in 0..sx as usize {
                    v[idx(x, y, z)] = skin;
                }
            }
        }
        // Hair on top row
        for z in 0..sz as usize {
            for x in 0..sx as usize {
                v[idx(x, 3, z)] = hair;
            }
        }
        VoxelPart {
            name: "head".into(),
            size: [sx, sy, sz],
            voxels: v,
            pivot: Vec3::new(2.0, 0.0, 2.0),
            rest_offset: Vec3::new(-2.0, 10.0, -2.0),
        }
    };

    // ---- Torso (4×5×3) ----
    let torso = {
        let (sx, sy, sz) = (4u8, 5, 3);
        let mut v = vec![EMPTY_VOXEL; (sx as usize) * (sy as usize) * (sz as usize)];
        let idx = |x: usize, y: usize, z: usize| (z * sy as usize + y) * sx as usize + x;
        for z in 0..sz as usize {
            for y in 0..sy as usize {
                for x in 0..sx as usize {
                    v[idx(x, y, z)] = shirt;
                }
            }
        }
        VoxelPart {
            name: "torso".into(),
            size: [sx, sy, sz],
            voxels: v,
            pivot: Vec3::new(2.0, 2.5, 1.5),
            rest_offset: Vec3::new(-2.0, 5.0, -1.5),
        }
    };

    // ---- Left Arm (2×5×2) ----
    let arm_l = {
        let (sx, sy, sz) = (2u8, 5, 2);
        let mut v = vec![EMPTY_VOXEL; (sx as usize) * (sy as usize) * (sz as usize)];
        let idx = |x: usize, y: usize, z: usize| (z * sy as usize + y) * sx as usize + x;
        for z in 0..sz as usize {
            for y in 0..sy as usize {
                for x in 0..sx as usize {
                    v[idx(x, y, z)] = if y >= 3 { skin } else { shirt };
                }
            }
        }
        VoxelPart {
            name: "arm_l".into(),
            size: [sx, sy, sz],
            voxels: v,
            pivot: Vec3::new(1.0, 5.0, 1.0),
            rest_offset: Vec3::new(-4.0, 5.0, -1.0),
        }
    };

    // ---- Right Arm (2×5×2) ----
    let arm_r = {
        let (sx, sy, sz) = (2u8, 5, 2);
        let mut v = vec![EMPTY_VOXEL; (sx as usize) * (sy as usize) * (sz as usize)];
        let idx = |x: usize, y: usize, z: usize| (z * sy as usize + y) * sx as usize + x;
        for z in 0..sz as usize {
            for y in 0..sy as usize {
                for x in 0..sx as usize {
                    v[idx(x, y, z)] = if y >= 3 { skin } else { shirt };
                }
            }
        }
        VoxelPart {
            name: "arm_r".into(),
            size: [sx, sy, sz],
            voxels: v,
            pivot: Vec3::new(1.0, 5.0, 1.0),
            rest_offset: Vec3::new(2.0, 5.0, -1.0),
        }
    };

    // ---- Left Leg (2×5×2) ----
    let leg_l = {
        let (sx, sy, sz) = (2u8, 5, 2);
        let mut v = vec![EMPTY_VOXEL; (sx as usize) * (sy as usize) * (sz as usize)];
        let idx = |x: usize, y: usize, z: usize| (z * sy as usize + y) * sx as usize + x;
        for z in 0..sz as usize {
            for y in 0..sy as usize {
                for x in 0..sx as usize {
                    v[idx(x, y, z)] = if y < 1 { shoe } else { pants };
                }
            }
        }
        VoxelPart {
            name: "leg_l".into(),
            size: [sx, sy, sz],
            voxels: v,
            pivot: Vec3::new(1.0, 5.0, 1.0),
            rest_offset: Vec3::new(-2.0, 0.0, -1.0),
        }
    };

    // ---- Right Leg (2×5×2) ----
    let leg_r = {
        let (sx, sy, sz) = (2u8, 5, 2);
        let mut v = vec![EMPTY_VOXEL; (sx as usize) * (sy as usize) * (sz as usize)];
        let idx = |x: usize, y: usize, z: usize| (z * sy as usize + y) * sx as usize + x;
        for z in 0..sz as usize {
            for y in 0..sy as usize {
                for x in 0..sx as usize {
                    v[idx(x, y, z)] = if y < 1 { shoe } else { pants };
                }
            }
        }
        VoxelPart {
            name: "leg_r".into(),
            size: [sx, sy, sz],
            voxels: v,
            pivot: Vec3::new(1.0, 5.0, 1.0),
            rest_offset: Vec3::new(0.0, 0.0, -1.0),
        }
    };

    // ---- Animations ----
    let mut animations = HashMap::new();

    // Idle: subtle head bob
    let idle = NpcAnimation {
        keyframes: vec![
            NpcPose { parts: HashMap::new() }, // rest
            NpcPose {
                parts: HashMap::from([
                    ("head".into(), (Vec3::new(0.0, 0.15, 0.0), Quat::IDENTITY)),
                ]),
            },
        ],
        frame_duration: 0.8,
        looping: true,
    };
    animations.insert("idle".into(), idle);

    // Walk cycle: 4 keyframes — arms and legs swing in opposition
    let swing = 0.45; // radians
    let walk = NpcAnimation {
        keyframes: vec![
            // Frame 0: left leg forward, right arm forward
            NpcPose {
                parts: HashMap::from([
                    ("leg_l".into(), (Vec3::ZERO, Quat::from_rotation_x(swing))),
                    ("leg_r".into(), (Vec3::ZERO, Quat::from_rotation_x(-swing))),
                    ("arm_l".into(), (Vec3::ZERO, Quat::from_rotation_x(-swing))),
                    ("arm_r".into(), (Vec3::ZERO, Quat::from_rotation_x(swing))),
                ]),
            },
            // Frame 1: neutral
            NpcPose { parts: HashMap::new() },
            // Frame 2: right leg forward, left arm forward
            NpcPose {
                parts: HashMap::from([
                    ("leg_l".into(), (Vec3::ZERO, Quat::from_rotation_x(-swing))),
                    ("leg_r".into(), (Vec3::ZERO, Quat::from_rotation_x(swing))),
                    ("arm_l".into(), (Vec3::ZERO, Quat::from_rotation_x(swing))),
                    ("arm_r".into(), (Vec3::ZERO, Quat::from_rotation_x(-swing))),
                ]),
            },
            // Frame 3: neutral
            NpcPose { parts: HashMap::new() },
        ],
        frame_duration: 0.2,
        looping: true,
    };
    animations.insert("walk".into(), walk);

    NpcBlueprint {
        name: "humanoid".into(),
        parts: vec![head, torso, arm_l, arm_r, leg_l, leg_r],
        animations,
    }
}

// =========================================================== spawning

fn spawn_npc_on_keypress(
    mut commands: Commands,
    keyboard: Res<ButtonInput<KeyCode>>,
    blueprint: Option<Res<NpcBlueprint>>,
    palette: Option<Res<Palette>>,
    mut mesh_cache: ResMut<NpcMeshCache>,
    mut meshes: ResMut<Assets<Mesh>>,
    player_q: Query<&WorldPosition, With<Player>>,
    anchor: Res<WorldAnchor>,
    zoom: Res<CameraZoom>,
    cam_q: Query<&FpsCam>,
) {
    if !keyboard.just_pressed(KeyCode::KeyN) {
        return;
    }

    let Some(blueprint) = blueprint else { return };
    let Some(palette) = palette else { return };
    let Ok(player_pos) = player_q.single() else { return };

    let cell = cell_size_at_layer(zoom.layer);
    let forward = if let Ok(cam) = cam_q.single() {
        Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos())
    } else {
        Vec3::NEG_Z
    };

    let player_bevy = bevy_from_position(&player_pos.0, &anchor);
    let npc_bevy = player_bevy + forward * 5.0 * cell + Vec3::Y * 1.0;
    let Some(npc_pos) = position_from_bevy(npc_bevy, &anchor) else {
        warn!("NPC spawn position outside world");
        return;
    };

    let spawn_bevy = bevy_from_position(&npc_pos, &anchor);
    let scale = npc_scale(zoom.layer);
    let heading = rand_heading();

    // Ensure meshes are baked for this blueprint
    if !mesh_cache.meshes.contains_key(&blueprint.name) {
        let part_meshes: Vec<Vec<BakedSubMesh>> = blueprint
            .parts
            .iter()
            .map(|part| bake_part(part, &mut meshes))
            .collect();
        mesh_cache.meshes.insert(blueprint.name.clone(), part_meshes);
    }

    let cached = mesh_cache.meshes.get(&blueprint.name).unwrap();

    // Spawn root entity
    let root = commands
        .spawn((
            Npc,
            WorldPosition(npc_pos),
            NpcVelocity(Vec3::ZERO),
            NpcAi {
                heading,
                change_timer: Timer::from_seconds(
                    2.0 + rand_f32() * 3.0,
                    TimerMode::Once,
                ),
            },
            NpcAnimState {
                current_anim: "walk".into(),
                time: 0.0,
            },
            Transform::from_translation(spawn_bevy)
                .with_scale(Vec3::splat(scale))
                .with_rotation(Quat::from_rotation_y(heading + PI)),
            Visibility::Visible,
        ))
        .id();

    // Spawn child entities for each body part
    for (part_idx, part) in blueprint.parts.iter().enumerate() {
        let part_entity = commands
            .spawn((
                NpcPartIndex(part_idx),
                Transform::from_translation(part.rest_offset),
                Visibility::Inherited,
                bevy::ecs::hierarchy::ChildOf(root),
            ))
            .id();

        // Attach sub-mesh children for each voxel type in this part
        if let Some(sub_meshes) = cached.get(part_idx) {
            for sub in sub_meshes {
                if let Some(mat) = palette.material(sub.voxel) {
                    commands.spawn((
                        Mesh3d(sub.mesh.clone()),
                        MeshMaterial3d(mat),
                        Transform::from_translation(-part.pivot),
                        Visibility::Inherited,
                        bevy::ecs::hierarchy::ChildOf(part_entity),
                    ));
                }
            }
        }
    }

    info!("Spawned voxel NPC at {spawn_bevy:?} (layer {})", zoom.layer);
}

// =========================================================== animation

fn npc_animate(
    time: Res<Time>,
    blueprint: Option<Res<NpcBlueprint>>,
    mut npc_q: Query<(&mut NpcAnimState, &Children), With<Npc>>,
    mut part_q: Query<(&NpcPartIndex, &mut Transform)>,
) {
    let Some(blueprint) = blueprint else { return };
    let dt = time.delta_secs();

    for (mut anim_state, children) in &mut npc_q {
        anim_state.time += dt;

        let Some(anim) = blueprint.animations.get(&anim_state.current_anim) else {
            continue;
        };
        if anim.keyframes.is_empty() {
            continue;
        }

        let total_duration = anim.keyframes.len() as f32 * anim.frame_duration;
        let t = if anim.looping {
            anim_state.time % total_duration
        } else {
            anim_state.time.min(total_duration - 0.001)
        };

        // Find current and next keyframe
        let frame_f = t / anim.frame_duration;
        let frame_a = (frame_f as usize) % anim.keyframes.len();
        let frame_b = (frame_a + 1) % anim.keyframes.len();
        let blend = frame_f.fract();

        let pose_a = &anim.keyframes[frame_a];
        let pose_b = &anim.keyframes[frame_b];

        // Apply to each part child
        for child in children.iter() {
            let Ok((part_idx, mut tf)) = part_q.get_mut(child) else {
                continue;
            };
            let part = &blueprint.parts[part_idx.0];

            let (pos_a, rot_a) = pose_a
                .parts
                .get(&part.name)
                .cloned()
                .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
            let (pos_b, rot_b) = pose_b
                .parts
                .get(&part.name)
                .cloned()
                .unwrap_or((Vec3::ZERO, Quat::IDENTITY));

            let pos = pos_a.lerp(pos_b, blend);
            let rot = rot_a.slerp(rot_b, blend);

            // Final transform: rest offset + animated offset, rotated around pivot
            tf.translation = part.rest_offset + pos;
            tf.rotation = rot;
        }
    }
}

// ================================================================= AI

fn npc_ai(
    time: Res<Time>,
    mut q: Query<(&mut NpcAi, &mut NpcVelocity, &mut Transform, &mut NpcAnimState), With<Npc>>,
) {
    for (mut ai, mut vel, mut tf, mut anim) in &mut q {
        ai.change_timer.tick(time.delta());

        if ai.change_timer.just_finished() {
            ai.heading = rand_heading();
            ai.change_timer =
                Timer::from_seconds(2.0 + rand_f32() * 3.0, TimerMode::Once);
        }

        let speed = NPC_WALK_SPEED_CELLS;
        let dir_x = -ai.heading.sin();
        let dir_z = -ai.heading.cos();
        vel.0.x = dir_x * speed;
        vel.0.z = dir_z * speed;

        // Face movement direction
        tf.rotation = Quat::from_rotation_y(ai.heading + PI);

        // Keep walk animation active
        anim.current_anim = "walk".into();
    }
}

// ============================================================= physics

fn npc_physics(
    time: Res<Time>,
    world: Res<WorldState>,
    zoom: Res<CameraZoom>,
    mut q: Query<(&mut WorldPosition, &mut NpcVelocity), With<Npc>>,
) {
    let dt = time.delta_secs();
    let cell = cell_size_at_layer(zoom.layer);

    for (mut wpos, mut vel) in &mut q {
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

// ====================================================== zoom visibility

fn npc_zoom_visibility(
    mut commands: Commands,
    zoom: Res<CameraZoom>,
    mut q: Query<(Entity, &mut Transform), With<Npc>>,
) {
    if !zoom.is_changed() {
        return;
    }

    let scale = npc_scale(zoom.layer);

    for (entity, mut tf) in &mut q {
        let layers_out = crate::world::tree::MAX_LAYER.saturating_sub(zoom.layer);
        if layers_out >= 2 {
            commands.entity(entity).despawn();
        } else {
            tf.scale = Vec3::splat(scale);
        }
    }
}

// ============================================================= helpers

fn npc_scale(view_layer: u8) -> f32 {
    // Each NPC is ~14 voxels tall. Scale so it's about 1.5 cells tall.
    let cell = cell_size_at_layer(view_layer);
    (1.5 * cell) / 14.0
}

fn rand_f32() -> f32 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (t % 10000) as f32 / 10000.0
}

fn rand_heading() -> f32 {
    rand_f32() * TAU
}
