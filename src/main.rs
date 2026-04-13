mod block;
mod camera;
mod diagnostics;
mod editor;
mod import;
mod interaction;
mod inventory;
mod model;
mod npc;
mod overlay;
mod player;
mod ui;
mod world;

use bevy::light::{CascadeShadowConfig, CascadeShadowConfigBuilder};
use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

use block::{BslMaterial, PaletteMaterial};

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
            npc::NpcPlugin,
            diagnostics::DiagnosticsPlugin,
        ))
        .add_systems(Startup, setup_environment)
        .add_systems(Update, update_shadow_cascades);

    #[cfg(feature = "debug_import")]
    app.add_systems(PostStartup, debug_stamp_monument);

    app.run();
}

#[cfg(feature = "debug_import")]
fn debug_stamp_monument(
    mut world_state: ResMut<world::WorldState>,
    mut palette: ResMut<block::Palette>,
    mut mat_assets: ResMut<Assets<PaletteMaterial>>,
) {
    const VOX_BYTES: &[u8] = include_bytes!("../assets/vox/monu1.vox");
    let model = import::vox::load_first_model_bytes(
        VOX_BYTES,
        &mut palette,
        &mut mat_assets,
    )
    .expect("failed to parse .vox");
    info!(
        "Loaded monument: {}×{}×{} ({} non-empty voxels, palette now {} entries)",
        model.size_x,
        model.size_y,
        model.size_z,
        model.data.iter().filter(|&&v| v != 0).count(),
        palette.len(),
    );

    let mut anchor = player::spawn_position();
    anchor.step_voxels(2, 10);
    let stamped = import::stamp::stamp_model(&mut world_state, &anchor, &model);
    info!("Stamped monument: {stamped} leaves modified");
}

fn setup_environment(mut commands: Commands) {
    // Cool blue-tinted ambient pushes unlit/shadowed surfaces toward
    // blue, contrasting with the warm directional light — this is the
    // core of BSL's "warm sun / cool shadow" look. Brightness lowered
    // from 800 to fit the HDR pipeline (AcesFitted tone-maps it).
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.7, 0.8, 1.0),
        brightness: 400.0,
        ..default()
    });

    // Warm golden sun — BSL tints its directional light toward amber.
    // Illuminance lowered from 20k to avoid blowing out highlights in
    // the HDR pipeline and to eliminate the shadow acne that occurred
    // with AcesFitted at the old intensity.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.95, 0.85),
            illuminance: 8_000.0,
            shadows_enabled: true,
            shadow_depth_bias: 0.3,
            shadow_normal_bias: 2.0,
            ..default()
        },
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
    // Cap shadow distance: at zoomed-out layers the shadow pass would
    // render the entire view volume (radius can be 20k+ Bevy units at
    // layer 8), destroying GPU perf. Shadows beyond ~2000 units are
    // invisible anyway — the atmosphere haze hides them.
    let shadow_max = radius.min(2000.0 * cell.min(5.0));
    for mut config in &mut lights {
        *config = CascadeShadowConfigBuilder {
            first_cascade_far_bound: 10.0 * cell,
            maximum_distance: shadow_max,
            overlap_proportion: 0.4,
            ..default()
        }
        .build();
    }
}
