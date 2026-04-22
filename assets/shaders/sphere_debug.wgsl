// Sphere DDA debug paint helpers.
//
// Enabled by `uniforms.sphere_debug_mode.x != 0`. Each mode replaces
// `sphere_in_cell`'s normal output with a per-pixel diagnostic color
// that surfaces one specific aspect of the walker / plane-advance
// state. Modes are mutually exclusive; cycle with F6 in the live app.
//
// The diagnostic state is accumulated as a local `SphereDebug` struct
// inside `sphere_in_cell` and passed to `sphere_debug_color` at
// return. Name ↔ value mapping also lives in
// `crate::renderer::SPHERE_DEBUG_MODE_NAMES` (Rust); add a case here
// whenever a new mode lands there, otherwise the mode falls through
// to solid magenta.

struct SphereDebug {
    steps: u32,
    walker_depth: u32,
    walker_size: f32,
    walker_ratio_u: u32,
    walker_ratio_v: u32,
    walker_ratio_depth: u32,
    /// 0 = loop exited with no walker call (shouldn't happen past
    /// the first step); 1 = walker returned EMPTY and we advanced
    /// via plane DDA; 2 = walker returned a block (real content).
    result_kind: u32,
    /// Axis-crossing for the LAST plane advance (0=u_lo, 1=u_hi,
    /// 2=v_lo, 3=v_hi, 4=r_lo, 5=r_hi). 6 = no plane taken.
    winning: u32,
    /// Face-subtree root BFS idx at the point sphere_in_cell
    /// handed off to the walker. For debugging whether a place
    /// action changes the face_node_idx that the ground rays
    /// descend into.
    face_node_idx: u32,
}

fn sphere_debug_init() -> SphereDebug {
    var d: SphereDebug;
    d.steps = 0u;
    d.walker_depth = 0u;
    d.walker_size = 0.0;
    d.walker_ratio_u = 0u;
    d.walker_ratio_v = 0u;
    d.walker_ratio_depth = 0u;
    d.result_kind = 0u;
    d.winning = 6u;
    d.face_node_idx = 0u;
    return d;
}

// Branchless HSV-ish hue rainbow. f ∈ [0, 1] → red (0) → yellow (1/6)
// → green (1/3) → cyan (1/2) → blue (2/3) → magenta (5/6) → back.
fn sphere_debug_rainbow(f: f32) -> vec3<f32> {
    let t = clamp(f, 0.0, 1.0);
    let h = t * 6.0;
    let r = clamp(abs(h - 3.0) - 1.0, 0.0, 1.0);
    let g = clamp(2.0 - abs(h - 2.0), 0.0, 1.0);
    let b = clamp(2.0 - abs(h - 4.0), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

// Blue → cyan → green → yellow → red heatmap. Chosen over rainbow
// for "count" views since the perceived brightness grows with count.
fn sphere_debug_heatmap(f: f32) -> vec3<f32> {
    let t = clamp(f, 0.0, 1.0);
    let r = clamp(2.0 * t - 0.5, 0.0, 1.0);
    let g = clamp(1.5 - abs(2.0 * t - 1.0) * 2.0, 0.0, 1.0);
    let b = clamp(1.5 - 2.0 * t, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn sphere_debug_color(mode: u32, d: SphereDebug) -> vec3<f32> {
    switch mode {
        // Mode 1: DDA step count. Cap 256 for headroom on
        // pathological depths; anything redder than ~red-orange
        // is probably looping.
        case 1u: { return sphere_debug_heatmap(f32(d.steps) / 256.0); }
        // Mode 2: walker terminal depth. Denominator 16 so d=10
        // lands at ~0.6 (yellow-green) and d=15 at red.
        case 2u: { return sphere_debug_rainbow(f32(d.walker_depth) / 16.0); }
        // Mode 3: classification of the LAST walker result on the
        // ray. Green = cell had content (expected for any opaque
        // block we hit); red = empty terminal that caused a plane
        // advance; blue = loop exited before calling walker (sky
        // path or immediate miss).
        case 3u: {
            if d.result_kind == 2u { return vec3<f32>(0.15, 0.85, 0.20); }
            if d.result_kind == 1u { return vec3<f32>(0.85, 0.25, 0.15); }
            return vec3<f32>(0.20, 0.25, 0.80);
        }
        // Mode 4: winning plane axis for the LAST advance. If two
        // planes have collapsed normals and the ray deterministically
        // picks one over the other across the whole screen, expect a
        // single hue to dominate at the failure band.
        case 4u: {
            switch d.winning {
                case 0u: { return vec3<f32>(1.0, 0.3, 0.3); }  // u_lo
                case 1u: { return vec3<f32>(0.6, 0.1, 0.1); }  // u_hi
                case 2u: { return vec3<f32>(0.3, 1.0, 0.3); }  // v_lo
                case 3u: { return vec3<f32>(0.1, 0.6, 0.1); }  // v_hi
                case 4u: { return vec3<f32>(0.3, 0.5, 1.0); }  // r_lo
                case 5u: { return vec3<f32>(0.1, 0.2, 0.6); }  // r_hi
                default: { return vec3<f32>(0.25, 0.25, 0.25); }
            }
        }
        // Mode 5: log cell size, rainbow-mapped. -log2(sz) is
        // proportional to descent depth (~3 levels per octave since
        // ratio 3). Expect smooth gradient inward; a band where the
        // color jumps backwards would mean the walker bailed out
        // shallower than LOD called for.
        case 5u: {
            let sz = max(d.walker_size, 1e-25);
            return sphere_debug_rainbow(clamp(-log2(sz) / 16.0, 0.0, 1.0));
        }
        // Mode 6: ratio_u / ratio_v / ratio_depth mod 8 as RGB.
        // Adjacent cells differ by 1 in at least one ratio component,
        // so neighboring cells show different hues. Any pixel-scale
        // noise in this mode = slot-pick is jittering in sub-cell
        // precision; smooth banding = walker is stable.
        case 6u: {
            let rm = f32(d.walker_ratio_u & 7u) / 7.0;
            let gm = f32(d.walker_ratio_v & 7u) / 7.0;
            let bm = f32(d.walker_ratio_depth & 7u) / 7.0;
            return vec3<f32>(rm, gm, bm);
        }
        // Mode 7: face_node_idx paint via Knuth multiplicative hash
        // (0x9E3779B1 ~= golden ratio). Scatters adjacent BFS
        // indices to visually distinct colors so small BFS-idx deltas
        // between pre and post place become visible. If this painting
        // differs pre vs post on rays that don't go through the
        // placed cell, the tree pack or GPU upload is producing
        // effectively different structure.
        case 7u: {
            let h = d.face_node_idx * 2654435761u;
            let r = f32(h & 0xFFu) / 255.0;
            let g = f32((h >> 8u) & 0xFFu) / 255.0;
            let b = f32((h >> 16u) & 0xFFu) / 255.0;
            return vec3<f32>(r, g, b);
        }
        default: { return vec3<f32>(1.0, 0.0, 1.0); }
    }
}
