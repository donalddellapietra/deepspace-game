use bevy::prelude::*;

use crate::player::Player;
use crate::world::{CameraZoom, WorldState};

pub struct DiagnosticsPlugin;

impl Plugin for DiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DiagTimer(Timer::from_seconds(2.0, TimerMode::Repeating)))
            .add_systems(Update, log_diag);
    }
}

#[derive(Resource)]
struct DiagTimer(Timer);

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
