#![allow(unused)]

use std::num::NonZero;

use crate::layout::ext::{LayoutBindingType, LayoutEntry};

/// A storage buffer.
///
/// Example WGSL syntax:
/// ```rust,ignore
/// @group(0) @binding(0)
/// var<storage, read_write> my_element: array<vec4<f32>>;
/// ```
///
/// Example GLSL syntax:
/// ```cpp,ignore
/// layout (set=0, binding=0) buffer myStorageBuffer {
///     vec4 myElement[];
/// };
/// ```
pub fn storage_buffer() -> StorageBufferBase {
    StorageBufferBase {}
}

pub struct StorageBufferBase {}

impl StorageBufferBase {
    /// The buffer can only be read in the shader,
    /// and it:
    /// - may or may not be annotated with `read` (WGSL).
    /// - must be annotated with `readonly` (GLSL).
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var<storage, read> my_element: array<vec4<f32>>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout (set=0, binding=0) readonly buffer myStorageBuffer {
    ///     vec4 myElement[];
    /// };
    /// ```
    pub fn read_only(self) -> LayoutEntry<StorageBuffer> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: StorageBuffer {
                read_only: true,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
    /// The buffer can be read or written to in the shader
    pub fn read_write(self) -> LayoutEntry<StorageBuffer> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: StorageBuffer {
                read_only: false,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}

pub struct StorageBuffer {
    read_only: bool,
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZero<u64>>,
}

impl LayoutBindingType for StorageBuffer {
    fn into_base(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage {
                read_only: self.read_only,
            },
            has_dynamic_offset: self.has_dynamic_offset,
            min_binding_size: self.min_binding_size,
        }
    }
}

impl LayoutEntry<StorageBuffer> {
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
