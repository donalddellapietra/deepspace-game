// Visual debug paint modes (placeholder dispatch).

#include "bindings.wgsl"

/// Mode-selector. Reads `uniforms.debug_mode.x`. Returns 0 when
/// debug paint is disabled.
fn debug_mode_active() -> u32 {
    return uniforms.debug_mode.x;
}

/// Paint a single pixel for the requested debug mode. Returns
/// fuchsia for any non-zero `mode` until per-mode arms are wired,
/// so a stale enable reads as an obviously-unwired sentinel rather
/// than passing through as black.
fn debug_paint(_mode: u32, _result: HitResult, _ray_dir: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0, 0.0, 1.0);
}
