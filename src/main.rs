mod camera;
mod player;
mod world;

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Deep Space".into(),
                    ..default()
                }),
                ..default()
            }),
        )
        .add_plugins((
            world::WorldPlugin,
            player::PlayerPlugin,
            camera::CameraPlugin,
        ))
        .add_systems(Startup, setup_environment)
        .run();
}

fn setup_environment(mut commands: Commands) {
    // Sun -- angled directional light with shadows
    commands.spawn((
        DirectionalLight {
            illuminance: 15_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            0.4,
            0.0,
        )),
    ));
}
