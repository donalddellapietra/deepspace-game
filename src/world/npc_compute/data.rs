//! GPU-side NPC data types and CPU-side mirror resource.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::ShaderType;
use bytemuck::{Pod, Zeroable};

/// GPU-side NPC state. 48 bytes, 16-byte aligned.
/// Matches the WGSL `NpcState` struct in npc_compute.wgsl.
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
#[repr(C)]
pub struct GpuNpcState {
    pub position: [f32; 3],
    pub heading: f32,
    pub velocity: [f32; 3],
    pub ai_timer: f32,
    pub anim_time: f32,
    pub speed: f32,
    pub seed: u32,
    /// Bit 0 = alive.
    pub flags: u32,
}

/// Per-frame uniform data for the compute shader.
/// Matches the WGSL `Uniforms` struct.
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
#[repr(C)]
pub struct NpcComputeUniforms {
    pub delta_time: f32,
    pub frame: u32,
    pub npc_count: u32,
    pub gravity: f32,
    pub world_min_xz: [f32; 2],
    pub world_size_xz: [f32; 2],
}

/// CPU-side resource holding NPC data for GPU upload.
/// Extracted to the render world each frame.
#[derive(Resource, Clone, ExtractResource)]
pub struct NpcGpuData {
    /// NPC states — uploaded to the GPU storage buffer when dirty.
    pub states: Vec<GpuNpcState>,
    /// Whether the buffer needs re-upload (new NPCs spawned).
    pub dirty: bool,
    /// Per-frame uniforms.
    pub uniforms: NpcComputeUniforms,
}

impl Default for NpcGpuData {
    fn default() -> Self {
        Self {
            states: Vec::new(),
            dirty: false,
            uniforms: NpcComputeUniforms {
                delta_time: 0.0,
                frame: 0,
                npc_count: 0,
                gravity: 20.0,
                world_min_xz: [0.0; 2],
                world_size_xz: [1.0; 2],
            },
        }
    }
}
