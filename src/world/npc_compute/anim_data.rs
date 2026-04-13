//! Baked animation data for GPU upload.
//!
//! Converts the blueprint's HashMap-based keyframe data into flat
//! arrays suitable for GPU storage buffers.

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::npc::TreeBlueprint;

/// Maximum parts per NPC (must match MAX_PARTS in npc.rs).
pub const GPU_MAX_PARTS: usize = 8;
/// Maximum keyframes per animation.
pub const GPU_MAX_KEYFRAMES: usize = 8;

/// Per-part static data: rest offset, pivot, color index.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuPartInfo {
    pub rest_offset: [f32; 3],
    pub _pad0: f32,
    pub pivot: [f32; 3],
    pub _pad1: f32,
}

/// One keyframe entry for one part: offset + rotation.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuKeyframe {
    pub offset: [f32; 3],
    pub _pad: f32,
    pub rotation: [f32; 4], // quaternion xyzw
}

/// All animation data baked for GPU upload.
/// Layout: parts[MAX_PARTS] then keyframes[MAX_KEYFRAMES][MAX_PARTS].
pub struct GpuAnimData {
    pub parts: [GpuPartInfo; GPU_MAX_PARTS],
    pub keyframes: [[GpuKeyframe; GPU_MAX_PARTS]; GPU_MAX_KEYFRAMES],
    pub num_parts: u32,
    pub num_keyframes: u32,
    pub frame_duration: f32,
    pub total_duration: f32,
}

/// Bake a TreeBlueprint's walk animation into GPU-ready data.
pub fn bake_anim_data(tree_bp: &TreeBlueprint) -> GpuAnimData {
    let mut parts = [GpuPartInfo {
        rest_offset: [0.0; 3],
        _pad0: 0.0,
        pivot: [0.0; 3],
        _pad1: 0.0,
    }; GPU_MAX_PARTS];

    for (i, tp) in tree_bp.parts.iter().enumerate().take(GPU_MAX_PARTS) {
        parts[i] = GpuPartInfo {
            rest_offset: tp.rest_offset.to_array(),
            _pad0: 0.0,
            pivot: tp.pivot.to_array(),
            _pad1: 0.0,
        };
    }

    let num_parts = tree_bp.parts.len().min(GPU_MAX_PARTS) as u32;

    // Get the walk animation.
    let anim = tree_bp.animations.get("walk");
    let (num_keyframes, frame_duration) = match anim {
        Some(a) => (a.keyframes.len().min(GPU_MAX_KEYFRAMES), a.frame_duration),
        None => (0, 1.0),
    };

    let mut keyframes = [[GpuKeyframe {
        offset: [0.0; 3],
        _pad: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0], // identity quaternion
    }; GPU_MAX_PARTS]; GPU_MAX_KEYFRAMES];

    if let Some(anim) = anim {
        for (kf_idx, pose) in anim.keyframes.iter().enumerate().take(GPU_MAX_KEYFRAMES) {
            for (part_idx, tp) in tree_bp.parts.iter().enumerate().take(GPU_MAX_PARTS) {
                let (offset, rotation) = pose
                    .parts
                    .get(&tp.name)
                    .cloned()
                    .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
                keyframes[kf_idx][part_idx] = GpuKeyframe {
                    offset: offset.to_array(),
                    _pad: 0.0,
                    rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
                };
            }
        }
    }

    let total_duration = num_keyframes as f32 * frame_duration;

    GpuAnimData {
        parts,
        keyframes,
        num_parts,
        num_keyframes: num_keyframes as u32,
        frame_duration,
        total_duration,
    }
}
