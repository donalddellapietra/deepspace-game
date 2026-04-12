use bevy::pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

/// BSL-inspired material: extends StandardMaterial with custom lighting
/// parameters adapted from BSL Minecraft shaders.
pub type BslMaterial = ExtendedMaterial<StandardMaterial, BslExtension>;

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct BslExtension {
    #[uniform(100)]
    pub params: BslParams,
}

#[derive(Clone, Debug, ShaderType)]
pub struct BslParams {
    /// Ambient light color (rgb) and intensity (a).
    pub ambient_color: Vec4,
    /// Subsurface scattering strength (0.0 = opaque, ~0.5 = translucent).
    pub subsurface_strength: f32,
    /// Vertex-baked AO influence (1.0 = full effect).
    pub ao_strength: f32,
    pub _padding: Vec2,
}

impl Default for BslParams {
    fn default() -> Self {
        Self {
            ambient_color: Vec4::new(0.9, 0.95, 1.0, 0.3),
            subsurface_strength: 0.0,
            ao_strength: 1.0,
            _padding: Vec2::ZERO,
        }
    }
}

impl Default for BslExtension {
    fn default() -> Self {
        Self {
            params: BslParams::default(),
        }
    }
}

impl MaterialExtension for BslExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/bsl_voxel.wgsl".into()
    }
}
