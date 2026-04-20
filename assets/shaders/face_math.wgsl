// Cubed-sphere face geometry. Projection primitives that map
// between face-local (u, v) ∈ [-1, 1]² equal-area coords and unit
// direction vectors. Mirror of the CPU-side `cubesphere.rs`.

const PI_F: f32 = 3.1415926535;
const FRAC_PI_4: f32 = 0.785398163;

// Equal-area <-> cube tangent: distributes solid angle uniformly
// across the cubemap face. Cuts worst-case pixel stretch from ~2.4×
// (plain cube) to ~1.15×. `c` is the straight-line cube-plane
// coordinate; `e` is the equal-area coordinate normalized to [-1, 1].
fn cube_to_ea(c: f32) -> f32 { return atan(c) * (4.0 / PI_F); }
fn ea_to_cube(e: f32) -> f32 { return tan(e * FRAC_PI_4); }

// Body-grid slot containing each face's subtree. Mirror of Rust
// `FACE_SLOTS`. Faces live at the 6 axis-centers of the body's
// 27-child grid; the center slot (1,1,1) holds the interior filler.
fn face_slot(face: u32) -> u32 {
    switch face {
        case 0u: { return 14u; } // PosX = slot(2, 1, 1)
        case 1u: { return 12u; } // NegX = slot(0, 1, 1)
        case 2u: { return 16u; } // PosY = slot(1, 2, 1)
        case 3u: { return 10u; } // NegY = slot(1, 0, 1)
        case 4u: { return 22u; } // PosZ = slot(1, 1, 2)
        default: { return 4u;  } // NegZ = slot(1, 1, 0)
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

// Face-local (u, v) ∈ [-1, 1]² -> unit direction on the sphere.
fn face_uv_to_dir(face: u32, u: f32, v: f32) -> vec3<f32> {
    let cube_u = ea_to_cube(u);
    let cube_v = ea_to_cube(v);
    let n = face_normal(face);
    let ua = face_u_axis(face);
    let va = face_v_axis(face);
    return normalize(n + cube_u * ua + cube_v * va);
}

// Pick the dominant cube face for a unit direction. The face is the
// one whose normal has the largest dot product with `n` — equivalently,
// `abs(n)`'s largest component determines the axis and its sign
// determines PosX/NegX etc.
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
