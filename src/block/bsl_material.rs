use std::sync::OnceLock;

use bevy::pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::{Shader, ShaderRef};

/// BSL-inspired material: extends StandardMaterial with custom lighting
/// parameters adapted from BSL Minecraft shaders.
pub type BslMaterial = ExtendedMaterial<StandardMaterial, BslExtension>;

/// Global handle so `fragment_shader()` (which is static) can return it.
static BSL_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();

pub(crate) fn load_bsl_shader(app: &mut App) {
    let handle = {
        let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
        shaders.add(Shader::from_wgsl(
            include_str!("../../assets/shaders/bsl_voxel.wgsl"),
            "bsl_voxel.wgsl",
        ))
    };
    BSL_SHADER.set(handle).ok();
}

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
            // Cool blue ambient — shadows tint toward this, contrasting
            // with the warm golden directional light (BSL signature).
            ambient_color: Vec4::new(0.8, 0.85, 1.0, 0.3),
            subsurface_strength: 0.0,
            ao_strength: 0.6,
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
        BSL_SHADER
            .get()
            .cloned()
            .map(ShaderRef::Handle)
            .unwrap_or_else(|| "shaders/bsl_voxel.wgsl".into())
    }
}
