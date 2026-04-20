#![allow(unused)]

use crate::layout::ext::{LayoutBindingType, LayoutEntry};

/// A sampler that can be used to sample a texture.
///
/// Example WGSL syntax:
/// ```rust,ignore
/// @group(0) @binding(0)
/// var s: sampler;
/// ```
///
/// Example GLSL syntax:
/// ```cpp,ignore
/// layout(binding = 0)
/// uniform sampler s;
/// ```
///
/// Corresponds to [WebGPU `GPUSamplerBindingLayout`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpusamplerbindinglayout).
pub fn sampler() -> SamplerBase {
    SamplerBase {}
}

pub struct SamplerBase {}

impl SamplerBase {
    /// The sampling result is produced based on more than a single color sample from a texture,
    /// e.g. when bilinear interpolation is enabled.
    pub fn filtering(self) -> LayoutEntry<Sampler> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }
    }
    /// The sampling result is produced based on a single color sample from a texture.
    pub fn non_filtering(self) -> LayoutEntry<Sampler> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }
    }
    /// Use as a comparison sampler instead of a normal sampler.
    /// For more info take a look at the analogous functionality in OpenGL: <https://www.khronos.org/opengl/wiki/Sampler_Object#Comparison_mode>.
    pub fn comparison(self) -> LayoutEntry<Sampler> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }
    }
}

pub struct Sampler(wgpu::SamplerBindingType);

impl LayoutBindingType for Sampler {
    fn into_base(&self) -> wgpu::BindingType {
        wgpu::BindingType::Sampler(self.0)
    }
}
