use bevy::diagnostic::{
    DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin,
};
use bevy::prelude::*;

use crate::player::Player;
use crate::world::state::GROUND_Y_VOXELS;
use crate::world::view::{position_to_leaf_coord, target_layer_for};
use crate::world::render::RenderTimings;
use crate::world::{CameraZoom, WorldPosition, WorldState};

pub struct DiagnosticsPlugin;

impl Plugin for DiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            EntityCountDiagnosticsPlugin::default(),
        ))
        .insert_resource(DiagTimer(Timer::from_seconds(2.0, TimerMode::Repeating)))
        .add_systems(Startup, spawn_debug_overlay)
        .add_systems(Update, (log_diag, toggle_debug_overlay, update_debug_overlay));
    }
}

#[derive(Resource)]
struct DiagTimer(Timer);

#[derive(Component)]
struct DebugOverlay;

#[derive(Component)]
struct DebugOverlayText;

fn spawn_debug_overlay(mut commands: Commands) {
    commands
        .spawn((
            DebugOverlay,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(16.0),
                right: Val::Px(16.0),
                padding: UiRect::all(Val::Px(10.0)),
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        ))
        .with_children(|p| {
            p.spawn((
                DebugOverlayText,
                Text::new(""),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::srgb(0.85, 1.0, 0.85)),
            ));
        });
}

fn toggle_debug_overlay(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut q: Query<&mut Node, With<DebugOverlay>>,
) {
    if !keyboard.just_pressed(KeyCode::F3) {
        return;
    }
    let Ok(mut node) = q.single_mut() else { return };
    node.display = match node.display {
        Display::None => Display::Flex,
        _ => Display::None,
    };
}

fn update_debug_overlay(
    diagnostics: Res<DiagnosticsStore>,
    zoom: Res<CameraZoom>,
    state: Res<WorldState>,
    timings: Res<RenderTimings>,
    player_q: Query<&WorldPosition, With<Player>>,
    overlay_q: Query<&Node, With<DebugOverlay>>,
    mut text_q: Query<&mut Text, With<DebugOverlayText>>,
) {
    let Ok(overlay) = overlay_q.single() else { return };
    if overlay.display == Display::None {
        return;
    }
    let Ok(mut text) = text_q.single_mut() else { return };

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let frame_time_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let entity_count = diagnostics
        .get(&EntityCountDiagnosticsPlugin::ENTITY_COUNT)
        .and_then(|d| d.value())
        .unwrap_or(0.0);

    // Player position in the world's absolute leaf-coord frame.
    // `WorldPosition` is the source of truth; the Bevy `Transform`
    // is just its projection through the floating anchor and would
    // always show sub-voxel drift (`[0, 1)`), which isn't useful.
    let leaf = player_q
        .single()
        .map(|wp| position_to_leaf_coord(&wp.0))
        .unwrap_or([0, 0, 0]);
    // `height` is signed leaf y relative to the grass top face:
    // `0` = feet on the ground, `+n` = n leaves above, `-n` = n
    // leaves underground.
    let height = leaf[1] - GROUND_Y_VOXELS;
    let target = target_layer_for(zoom.layer);

    *text = Text::new(format!(
        "DEBUG [F3]\n\
         fps           {:>11.1}\n\
         frame time    {:>8.2} ms\n\
         entities      {:>11}\n\
         world x       {:>11}\n\
         world y       {:>11}\n\
         world z       {:>11}\n\
         height        {:>11}\n\
         view layer    {:>11}\n\
         target layer  {:>11}\n\
         library       {:>11}\n\
         visits        {:>11}\n\
         groups        {:>11}\n\
         walk          {:>8} us\n\
         reconcile     {:>8} us\n\
         collision     {:>8} us\n\
         triangles     {:>11}",
        fps,
        frame_time_ms,
        entity_count as u64,
        leaf[0],
        leaf[1],
        leaf[2],
        height,
        zoom.layer,
        target,
        state.library.len(),
        timings.visit_count,
        timings.group_count,
        timings.walk_us,
        timings.reconcile_us,
        timings.collision_us,
        timings.triangles,
    ));
}

fn log_diag(
    time: Res<Time>,
    mut timer: ResMut<DiagTimer>,
    state: Res<WorldState>,
    zoom: Res<CameraZoom>,
    player_q: Query<&WorldPosition, With<Player>>,
) {
    timer.0.tick(time.delta());
    if !timer.0.just_finished() {
        return;
    }
    let leaf = player_q
        .single()
        .map(|wp| position_to_leaf_coord(&wp.0))
        .unwrap_or([0, 0, 0]);
    let height = leaf[1] - GROUND_Y_VOXELS;
    info!(
        "leaf=({},{},{}) height={} zoom_layer={} library_entries={}",
        leaf[0],
        leaf[1],
        leaf[2],
        height,
        zoom.layer,
        state.library.len(),
    );
}
