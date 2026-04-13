//! GPU compute pipeline for NPC simulation.
//!
//! Dispatches a compute shader that updates all NPC state (AI, physics,
//! animation) in parallel on the GPU. The CPU only uploads initial state
//! and per-frame uniforms — zero per-NPC CPU work during simulation.
//!
//! Architecture:
//! - `NpcGpuData` (main world): CPU-side mirror, used for spawning
//! - `NpcGpuState` (GPU): 48-byte struct per NPC in a storage buffer
//! - `NpcComputeUniforms` (GPU): per-frame globals (dt, frame, gravity)
//! - Render graph node dispatches compute before camera rendering

mod data;
mod pipeline;
mod node;

pub use data::{GpuNpcState, NpcComputeUniforms, NpcGpuData};

use std::borrow::Cow;
use std::sync::OnceLock;

use bevy::prelude::*;
use bevy::render::{
    extract_resource::ExtractResourcePlugin,
    render_graph::{self, RenderGraph, RenderLabel},
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy::shader::Shader;

use self::node::NpcComputeNode;
use self::pipeline::init_npc_compute_pipeline;

/// Global handle for the embedded compute shader.
static NPC_COMPUTE_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();

pub fn compute_shader_handle() -> Handle<Shader> {
    NPC_COMPUTE_SHADER.get().cloned().expect("NPC compute shader not loaded")
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct NpcComputeLabel;

pub struct NpcComputePlugin;

impl Plugin for NpcComputePlugin {
    fn build(&self, app: &mut App) {
        // Embed compute shader at compile time.
        let handle = {
            let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
            shaders.add(Shader::from_wgsl(
                include_str!("../../../assets/shaders/npc_compute.wgsl"),
                "npc_compute.wgsl",
            ))
        };
        NPC_COMPUTE_SHADER.set(handle).ok();

        app.init_resource::<NpcGpuData>();
        // TODO: Enable compute render systems when shader is wired.
        // For now, just register the resource for the main world spawn bridge.
    }
}
