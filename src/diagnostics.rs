use bevy::prelude::*;

use crate::interaction::TargetedBlock;
use crate::player::Player;
use crate::world::WorldState;

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
    time: Res<Time>, mut timer: ResMut<DiagTimer>,
    state: Res<WorldState>,
    targeted: Res<TargetedBlock>, player_q: Query<&Transform, With<Player>>,
) {
    timer.0.tick(time.delta());
    if !timer.0.just_finished() { return }
    let pos = player_q.single().map(|t| t.translation).unwrap_or(Vec3::ZERO);
    let tgt = targeted.hit.map(|h| format!("({},{},{})", h.x, h.y, h.z)).unwrap_or("none".into());
    info!("pos=({:.1},{:.1},{:.1}) depth={} chunks={} target={}",
        pos.x, pos.y, pos.z, state.depth(), state.world.chunks.len(), tgt);
}
