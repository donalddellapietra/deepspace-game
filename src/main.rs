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

use bevy::light::{CascadeShadowConfig, CascadeShadowConfigBuilder};
use bevy::prelude::*;

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
        // Atmosphere renders the sky; ClearColor only peeks through
        // sub-pixel gaps between mesh faces, so match it to the
        // dominant terrain color (grass) to hide seams.
        .insert_resource(ClearColor(Color::srgb(0.3, 0.6, 0.2)))
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
        .add_systems(Startup, setup_environment)
        .add_systems(Update, update_shadow_cascades);

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
        DirectionalLight { illuminance: 20_000.0, shadows_enabled: true, ..default() },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.7, 0.4, 0.0)),
    ));
}

/// Keep shadow cascade bounds in sync with the zoom layer so shadows
/// cover the full view distance at every scale.
fn update_shadow_cascades(
    zoom: Res<world::render::CameraZoom>,
    mut lights: Query<&mut CascadeShadowConfig, With<DirectionalLight>>,
) {
    if !zoom.is_changed() {
        return;
    }
    let cell = world::view::cell_size_at_layer(zoom.layer);
    let radius = world::render::RADIUS_VIEW_CELLS * cell;
    for mut config in &mut lights {
        // Scale cascade bounds with zoom so the shadow map texel
        // density stays consistent at every layer.
        *config = CascadeShadowConfigBuilder {
            first_cascade_far_bound: 10.0 * cell,
            maximum_distance: radius,
            overlap_proportion: 0.4,
            ..default()
        }
        .build();
    }
}
