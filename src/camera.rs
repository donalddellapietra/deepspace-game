use std::f32::consts::FRAC_PI_2;

use bevy::{
    input::mouse::AccumulatedMouseMotion,
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};

use crate::player::Player;

const SENSITIVITY: f32 = 0.003;
const EYE_HEIGHT: f32 = 1.7;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (spawn_camera, spawn_crosshair))
            .add_systems(Update, (grab_cursor, first_person_camera).chain());
    }
}

#[derive(Component)]
pub struct FpsCam {
    pub yaw: f32,
    pub pitch: f32,
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        FpsCam {
            yaw: 0.0,
            pitch: 0.0,
        },
        Transform::from_xyz(0.0, 10.0, 0.0),
    ));
}

fn spawn_crosshair(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(50.0),
            top: Val::Percent(50.0),
            width: Val::Px(4.0),
            height: Val::Px(4.0),
            margin: UiRect {
                left: Val::Px(-2.0),
                top: Val::Px(-2.0),
                ..default()
            },
            ..default()
        },
        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
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

fn first_person_camera(
    mouse_motion: Res<AccumulatedMouseMotion>,
    player_query: Query<&Transform, (With<Player>, Without<FpsCam>)>,
    mut camera_query: Query<(&mut Transform, &mut FpsCam), Without<Player>>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let Ok((mut cam_tf, mut cam)) = camera_query.single_mut() else {
        return;
    };

    let delta = mouse_motion.delta;
    cam.yaw -= delta.x * SENSITIVITY;
    cam.pitch = (cam.pitch + delta.y * SENSITIVITY).clamp(-FRAC_PI_2 + 0.05, FRAC_PI_2 - 0.05);

    cam_tf.translation = player_tf.translation + Vec3::Y * EYE_HEIGHT;
    cam_tf.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, -cam.pitch, 0.0);
}
