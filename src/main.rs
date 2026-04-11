mod block;
mod camera;
mod diagnostics;
mod editor;
mod interaction;
mod inventory;
mod model;
mod player;
mod ui;
mod world;

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { title: "Deep Space".into(), ..default() }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.5, 0.7, 0.9)))
        .add_plugins((
            block::BlockPlugin,
            world::WorldPlugin,
            editor::EditorPlugin,
            interaction::InteractionPlugin,
            inventory::InventoryPlugin,
            player::PlayerPlugin,
            camera::CameraPlugin,
            ui::UiPlugin,
            diagnostics::DiagnosticsPlugin,
        ))
        .add_systems(Startup, setup_environment)
        .run();
}

fn setup_environment(mut commands: Commands) {
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.9, 0.95, 1.0),
        brightness: 400.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight { illuminance: 20_000.0, shadows_enabled: true, ..default() },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.7, 0.4, 0.0)),
    ));
}
