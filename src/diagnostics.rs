use bevy::diagnostic::{
    DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin,
};
use bevy::prelude::*;

use crate::player::Player;
use crate::world::{CameraZoom, WorldState};

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
    player_q: Query<&Transform, With<Player>>,
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
    let pos = player_q
        .single()
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);

    *text = Text::new(format!(
        "DEBUG [F3]\n\
         fps           {:>7.1}\n\
         frame time    {:>6.2} ms\n\
         entities      {:>7}\n\
         pos           {:>6.1} {:>6.1} {:>6.1}\n\
         zoom layer    {:>7}\n\
         library       {:>7}",
        fps,
        frame_time_ms,
        entity_count as u64,
        pos.x,
        pos.y,
        pos.z,
        zoom.layer,
        state.library.len(),
    ));
}

fn log_diag(
    time: Res<Time>,
    mut timer: ResMut<DiagTimer>,
    state: Res<WorldState>,
    zoom: Res<CameraZoom>,
    player_q: Query<&Transform, With<Player>>,
) {
    timer.0.tick(time.delta());
    if !timer.0.just_finished() {
        return;
    }
    let pos = player_q
        .single()
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);
    info!(
        "pos=({:.1},{:.1},{:.1}) zoom_layer={} library_entries={}",
        pos.x,
        pos.y,
        pos.z,
        zoom.layer,
        state.library.len(),
    );
}
