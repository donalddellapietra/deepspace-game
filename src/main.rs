mod block;
mod camera;
mod editor;
mod interaction;
mod layer;
mod model;
mod player;
mod ui;
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
        .insert_resource(ClearColor(Color::srgb(0.5, 0.7, 0.9)))
        .add_plugins((
            block::BlockPlugin,
            model::ModelPlugin,
            layer::LayerPlugin,
            world::WorldPlugin,
            editor::EditorPlugin,
            interaction::InteractionPlugin,
            player::PlayerPlugin,
            camera::CameraPlugin,
            ui::UiPlugin,
        ))
        .add_systems(Startup, setup_environment)
        .run();
}

fn setup_environment(mut commands: Commands) {
    // Ambient light so shadowed faces aren't pitch black
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.9, 0.95, 1.0),
        brightness: 400.0,
        ..default()
    });

    // Sun
    commands.spawn((
        DirectionalLight {
            illuminance: 20_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -0.7,
            0.4,
            0.0,
        )),
    ));
}
