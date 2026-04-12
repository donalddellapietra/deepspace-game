mod block;
mod camera;
mod diagnostics;
mod editor;
mod import;
mod interaction;
mod inventory;
mod model;
mod overlay;
mod player;
mod ui;
mod world;

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

use block::BslMaterial;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Deep Space".into(),
                fit_canvas_to_parent: true,
                prevent_default_event_handling: true,
                // Transparent window allows the wry WebView overlay to
                // composite over the Metal rendering on native builds.
                transparent: true,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.5, 0.7, 0.9)))
        .add_plugins(MaterialPlugin::<BslMaterial>::default())
        .add_plugins((
            block::BlockPlugin,
            world::WorldPlugin,
            editor::EditorPlugin,
            interaction::InteractionPlugin,
            inventory::InventoryPlugin,
            player::PlayerPlugin,
            camera::CameraPlugin,
            ui::UiPlugin,
            overlay::OverlayPlugin,
            diagnostics::DiagnosticsPlugin,
        ))
        .add_systems(Startup, setup_environment);

    #[cfg(feature = "debug_import")]
    app.add_systems(Startup, debug_stamp_monument);

    app.run();
}

#[cfg(feature = "debug_import")]
fn debug_stamp_monument(mut world_state: ResMut<world::WorldState>) {
    const VOX_BYTES: &[u8] = include_bytes!("../assets/vox/monu1.vox");
    let model = import::vox::load_first_model_bytes(VOX_BYTES).expect("failed to parse .vox");
    info!(
        "Loaded monument: {}×{}×{} ({} non-empty voxels)",
        model.size_x,
        model.size_y,
        model.size_z,
        model.data.iter().filter(|&&v| v != 0).count(),
    );

    let mut anchor = player::spawn_position();
    anchor.step_voxels(2, 10);
    let stamped = import::stamp::stamp_model(&mut world_state, &anchor, &model);
    info!("Stamped monument: {stamped} leaves modified");
}

fn setup_environment(mut commands: Commands) {
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.9, 0.95, 1.0),
        brightness: 800.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            illuminance: 20_000.0,
            shadows_enabled: true,
            shadow_depth_bias: 0.3,
            shadow_normal_bias: 2.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.7, 0.4, 0.0)),
    ));
}
