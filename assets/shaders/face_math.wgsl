// Cubed-sphere face primitives: equal-area cube ↔ EA transforms,
// per-face axis lookups, and direction↔face-uv projection.

const PI_F: f32 = 3.1415926535;
const FRAC_PI_4: f32 = 0.785398163;

fn cube_to_ea(c: f32) -> f32 { return atan(c) * (4.0 / PI_F); }
fn ea_to_cube(e: f32) -> f32 { return tan(e * FRAC_PI_4); }

// Slot in a CubedSphereBody node's 27-grid that holds each face's
// subtree. Matches Rust `FACE_SLOTS`.
fn face_slot(face: u32) -> u32 {
    switch face {
        case 0u: { return 14u; } // PosX = (2, 1, 1)
        case 1u: { return 12u; } // NegX = (0, 1, 1)
        case 2u: { return 16u; } // PosY = (1, 2, 1)
        case 3u: { return 10u; } // NegY = (1, 0, 1)
        case 4u: { return 22u; } // PosZ = (1, 1, 2)
        default: { return 4u;  } // NegZ = (1, 1, 0)
    }
}

fn face_normal(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        case 1u: { return vec3<f32>(-1.0,  0.0,  0.0); }
        case 2u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        case 3u: { return vec3<f32>( 0.0, -1.0,  0.0); }
        case 4u: { return vec3<f32>( 0.0,  0.0,  1.0); }
        default: { return vec3<f32>( 0.0,  0.0, -1.0); }
    }
}

fn face_u_axis(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>( 0.0,  0.0, -1.0); }
        case 1u: { return vec3<f32>( 0.0,  0.0,  1.0); }
        case 2u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        case 3u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        case 4u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        default: { return vec3<f32>(-1.0,  0.0,  0.0); }
    }
}

fn face_v_axis(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        case 1u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        case 2u: { return vec3<f32>( 0.0,  0.0, -1.0); }
        case 3u: { return vec3<f32>( 0.0,  0.0,  1.0); }
        case 4u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        default: { return vec3<f32>( 0.0,  1.0,  0.0); }
    }
}

fn face_uv_to_dir(face: u32, u: f32, v: f32) -> vec3<f32> {
    let cube_u = ea_to_cube(u);
    let cube_v = ea_to_cube(v);
    let n = face_normal(face);
    let u_axis = face_u_axis(face);
    let v_axis = face_v_axis(face);
    return normalize(n + cube_u * u_axis + cube_v * v_axis);
}

fn pick_face(n: vec3<f32>) -> u32 {
    let ax = abs(n.x); let ay = abs(n.y); let az = abs(n.z);
    if ax >= ay && ax >= az {
        if n.x > 0.0 { return 0u; } else { return 1u; }
    } else if ay >= az {
        if n.y > 0.0 { return 2u; } else { return 3u; }
    } else {
        if n.z > 0.0 { return 4u; } else { return 5u; }
    }
}
