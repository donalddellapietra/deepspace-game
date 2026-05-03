#![allow(unused)]

use std::num::NonZero;

use crate::layout::ext::{LayoutBindingType, LayoutEntry};

pub struct UniformBuffer {
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZero<u64>>,
}

/// A buffer for uniform values.
///
/// Example WGSL syntax:
/// ```rust,ignore
/// struct Globals {
///     a_uniform: vec2<f32>,
///     another_uniform: vec2<f32>,
/// }
/// @group(0) @binding(0)
/// var<uniform> globals: Globals;
/// ```
///
/// Example GLSL syntax:
/// ```cpp,ignore
/// layout(std140, binding = 0)
/// uniform Globals {
///     vec2 aUniform;
///     vec2 anotherUniform;
/// };
/// ```
pub fn uniform_buffer() -> LayoutEntry<UniformBuffer> {
    LayoutEntry {
        visibility: wgpu::ShaderStages::NONE,
        ty: UniformBuffer {
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl LayoutBindingType for UniformBuffer {
    fn into_base(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: self.has_dynamic_offset,
            min_binding_size: self.min_binding_size,
        }
    }
}

impl LayoutEntry<UniformBuffer> {
    /// Indicates that the binding has a dynamic offset.
    ///
    /// One offset must be passed to [`RenderPass::set_bind_group`][RPsbg]
    /// for each dynamic binding in increasing order of binding number.
    ///
    pub fn dynamic_offset(mut self) -> Self {
        self.ty.has_dynamic_offset = true;
        self
    }
    /// The minimum size for a [`BufferBinding`] matching this entry, in bytes.
    ///
    /// - When calling [`create_bind_group`], the resource at this bind point
    ///   must be a [`BindingResource::Buffer`] whose effective size is at
    ///   least `size`.
    ///
    /// - When calling [`create_render_pipeline`] or [`create_compute_pipeline`],
    ///   `size` must be at least the [minimum buffer binding size] for the
    ///   shader module global at this bind point: large enough to hold the
    ///   global's value, along with one element of a trailing runtime-sized
    ///   array, if present.
    ///
    /// If this is omitted, resolving to `None`:
    ///
    /// - Each draw or dispatch command checks that the buffer range at this
    ///   bind point satisfies the [minimum buffer binding size].
    ///
    pub fn min_binding_size(mut self, value: NonZero<u64>) -> Self {
        self.ty.min_binding_size = Some(value);
        self
    }
}
