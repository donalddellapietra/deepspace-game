use std::f32::consts::FRAC_PI_2;

use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};

use crate::player::Player;

const SENSITIVITY: f32 = 0.003;
const ZOOM_SPEED: f32 = 1.5;
const MIN_DISTANCE: f32 = 4.0;
const MAX_DISTANCE: f32 = 40.0;
const MIN_PITCH: f32 = 0.05;
const MAX_PITCH: f32 = FRAC_PI_2 - 0.05;
const TARGET_HEIGHT_OFFSET: f32 = 1.8;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, (grab_cursor, orbit_camera).chain());
    }
}

#[derive(Component)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        OrbitCamera {
            yaw: 0.0,
            pitch: 0.4,
            distance: 15.0,
        },
        Transform::from_xyz(0.0, 20.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn grab_cursor(
    mouse: Res<ButtonInput<MouseButton>>,
    key: Res<ButtonInput<KeyCode>>,
    mut cursor_options: Single<&mut CursorOptions>,
) {
    if mouse.just_pressed(MouseButton::Left) {
        cursor_options.visible = false;
        cursor_options.grab_mode = CursorGrabMode::Locked;
    }
    if key.just_pressed(KeyCode::Escape) {
        cursor_options.visible = true;
        cursor_options.grab_mode = CursorGrabMode::None;
    }
}

fn orbit_camera(
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    player_query: Query<&Transform, (With<Player>, Without<OrbitCamera>)>,
    mut camera_query: Query<(&mut Transform, &mut OrbitCamera), Without<Player>>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let Ok((mut cam_tf, mut orbit)) = camera_query.single_mut() else {
        return;
    };

    // Rotate orbit with mouse (don't multiply by delta_time -- already per-frame)
    let delta = mouse_motion.delta;
    orbit.yaw -= delta.x * SENSITIVITY;
    orbit.pitch = (orbit.pitch - delta.y * SENSITIVITY).clamp(MIN_PITCH, MAX_PITCH);

    // Zoom with scroll wheel
    orbit.distance =
        (orbit.distance - mouse_scroll.delta.y * ZOOM_SPEED).clamp(MIN_DISTANCE, MAX_DISTANCE);

    // Place camera on a sphere around the player's head
    let target = player_tf.translation + Vec3::Y * TARGET_HEIGHT_OFFSET;
    let offset = Vec3::new(
        orbit.distance * orbit.pitch.cos() * orbit.yaw.sin(),
        orbit.distance * orbit.pitch.sin(),
        orbit.distance * orbit.pitch.cos() * orbit.yaw.cos(),
    );

    cam_tf.translation = target + offset;
    *cam_tf = cam_tf.looking_at(target, Vec3::Y);
}
