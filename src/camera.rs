use std::f32::consts::FRAC_PI_2;

use bevy::{
    input::mouse::AccumulatedMouseMotion,
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};

use crate::inventory::InventoryState;
use crate::player::{Player, PLAYER_HEIGHT};
use crate::world::render::cell_size_at_layer;
use crate::world::CameraZoom;

const SENSITIVITY: f32 = 0.003;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CursorLocked(false))
            .add_systems(Startup, (spawn_camera, spawn_crosshair))
            .add_systems(Update, (manage_cursor, first_person_camera).chain());
    }
}

#[derive(Component)]
pub struct FpsCam {
    pub yaw: f32,
    pub pitch: f32,
}

/// Whether the cursor is currently locked. Block interaction only fires when true.
#[derive(Resource)]
pub struct CursorLocked(pub bool);

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        // Pitch 0 = look forward. The follow system in
        // `first_person_camera` snaps this entity onto the player on
        // frame 1, so the spawn translation is just a placeholder —
        // we use `Vec3::ZERO` instead of any hardcoded global Bevy
        // coordinate to make it obvious that this isn't a position
        // we picked.
        FpsCam { yaw: 0.0, pitch: 0.0 },
        Transform::default(),
    ));
}

fn spawn_crosshair(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(50.0), top: Val::Percent(50.0),
            width: Val::Px(4.0), height: Val::Px(4.0),
            margin: UiRect { left: Val::Px(-2.0), top: Val::Px(-2.0), ..default() },
            ..default()
        },
        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
    ));
}

/// Grab cursor on left click (only if not already grabbed). Release on Escape.
/// The grab-click does NOT count as a block interaction.
fn manage_cursor(
    mouse: Res<ButtonInput<MouseButton>>,
    key: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    mut cursor_options: Single<&mut CursorOptions>,
    mut locked: ResMut<CursorLocked>,
) {
    // Never grab cursor while inventory is open
    if inv.open { return }
    if mouse.just_pressed(MouseButton::Left) && !locked.0 {
        cursor_options.visible = false;
        cursor_options.grab_mode = CursorGrabMode::Locked;
        locked.0 = true;
        return; // consume this click — don't also break a block
    }
    if key.just_pressed(KeyCode::Escape) {
        cursor_options.visible = true;
        cursor_options.grab_mode = CursorGrabMode::None;
        locked.0 = false;
    }
}

fn first_person_camera(
    motion: Res<AccumulatedMouseMotion>,
    locked: Res<CursorLocked>,
    zoom: Res<CameraZoom>,
    player_q: Query<&Transform, (With<Player>, Without<FpsCam>)>,
    mut cam_q: Query<(&mut Transform, &mut FpsCam), Without<Player>>,
) {
    let Ok(player_tf) = player_q.single() else { return };
    let Ok((mut cam_tf, mut cam)) = cam_q.single_mut() else { return };

    // Only rotate when cursor is locked
    if locked.0 {
        cam.yaw -= motion.delta.x * SENSITIVITY;
        cam.pitch = (cam.pitch + motion.delta.y * SENSITIVITY)
            .clamp(-FRAC_PI_2 + 0.05, FRAC_PI_2 - 0.05);
    }

    // Player.y = feet. Camera sits at the eye, but the eye height
    // scales with the view layer's cell size: at view L the player
    // operates "at one cell" of body width / height, and one cell is
    // `cell_size_at_layer(L)` Bevy units. The fixed-FOV camera at a
    // higher eye then sees the same number of cells regardless of L
    // — pressing Q to zoom out lifts the camera so the world looks
    // proportionally smaller.
    let cell = cell_size_at_layer(zoom.layer);
    cam_tf.translation = player_tf.translation + Vec3::Y * (PLAYER_HEIGHT * cell);
    cam_tf.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, -cam.pitch, 0.0);
}
