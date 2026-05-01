// Visual debug paint modes for the wrapped-Cartesian planet marcher.
//
// Phase 0c will populate this with the 8-mode dispatch ported from
// the abandoned `sphere_debug.wgsl`. Phase 0b ships placeholder
// implementations so `main.wgsl::shade_pixel` can call into the
// dispatch surface without having to gate on `dbg_mode != 0u` at
// the include level.

#include "bindings.wgsl"

/// Mode-selector. Reads `uniforms.debug_mode.x`. Until Phase 0c
/// wires `set_debug_mode`, the uniform stays at zero and `shade_pixel`
/// short-circuits past `debug_paint`.
fn debug_mode_active() -> u32 {
    return uniforms.debug_mode.x;
}

/// Paint a single pixel for the requested debug mode. Phase 0c will
/// populate the per-mode arms; Phase 0b returns fuchsia for any
/// non-zero `mode` so a stale enable reads as an obviously-unwired
/// sentinel rather than passing through as black.
fn debug_paint(_mode: u32, _result: HitResult, _ray_dir: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0, 0.0, 1.0);
}
