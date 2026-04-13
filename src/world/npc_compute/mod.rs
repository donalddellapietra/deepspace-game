//! NPC GPU data resources.
//!
//! Provides NpcGpuData resource used by the spawn bridge.
//! Compute shader pipeline will be added when ready to connect
//! to the instanced draw calls.

mod data;

pub use data::{GpuNpcState, NpcGpuData};

use bevy::prelude::*;

pub struct NpcComputePlugin;

impl Plugin for NpcComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NpcGpuData>();
    }
}
