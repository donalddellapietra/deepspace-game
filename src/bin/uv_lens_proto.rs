//! Sphere-Mercator lens prototype renderer (CPU).
//!
//! Builds the actual wrapped-planet world (no `tangent_planes`, no
//! sphere DDA, no `TangentBlock` — purely the flat wrapped-Cartesian
//! slab), then for each output pixel:
//!
//!   1. Builds a camera ray.
//!   2. Lenses it through `PlanetLens::project_ray` to get a slab-axis
//!      ray (de, dn, dh) and an anchor in slab cell coordinates.
//!   3. Reinterprets the slab-axis components as world-axis components
//!      (the slab IS world-axis-aligned, so this works directly), and
//!      hands the resulting world ray to the existing `cpu_raycast`.
//!   4. On hit, lens-rotates the slab face normal by the cell's local
//!      tangent frame (`PlanetLens::shade_normal`) — the per-block
//!      "rotate to UV-sphere tangent" the prototype is demonstrating.
//!   5. Diffuse + ambient + gamma → PNG byte.
//!
//! Output: `tmp/uv_lens_proto.png` in the current working directory.
//!
//! Usage: `cargo run --release --bin uv_lens_proto -- [width] [height]`

use deepspace_game::world::bootstrap::{
    wrapped_planet_world,
    DEFAULT_WRAPPED_PLANET_CELL_SUBTREE_DEPTH,
    DEFAULT_WRAPPED_PLANET_EMBEDDING_DEPTH,
    DEFAULT_WRAPPED_PLANET_SLAB_DEPTH,
    DEFAULT_WRAPPED_PLANET_SLAB_DIMS,
};
use deepspace_game::world::palette::BUILTINS;
use deepspace_game::world::raycast::{cpu_raycast, HitInfo};
use deepspace_game::world::sphere::uv_lens::PlanetLens;
use deepspace_game::world::state::WorldState;
use deepspace_game::world::tree::Child;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let w: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(384);
    let h: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(384);

    eprintln!("uv_lens_proto: building wrapped-planet world…");
    let world = wrapped_planet_world(
        DEFAULT_WRAPPED_PLANET_EMBEDDING_DEPTH,
        DEFAULT_WRAPPED_PLANET_SLAB_DIMS,
        DEFAULT_WRAPPED_PLANET_SLAB_DEPTH,
        DEFAULT_WRAPPED_PLANET_CELL_SUBTREE_DEPTH,
        false, // tangent_planes OFF — pure flat wrapped-Cartesian slab
    );

    // Slab embedding geometry (matches `wrapped_planet_world`):
    // - World coords are [0, 3)³.
    // - Embedding centres slot 13 (x=y=z=1) at every level for
    //   `embedding_depth` levels. After 2 levels the WrappedPlane node
    //   sits in cell `[4/3, 5/3)³` of size 1/3.
    // - Inside the WrappedPlane the slab subgrid is `3^slab_depth = 27`
    //   per axis. Cells are world-cubic with size `s = (1/3) / 27 = 1/81`.
    // - The slab footprint occupies cells `[0, dims[i])` on each axis
    //   inside the subgrid, anchored at the depth-2 cell's `[4/3,4/3,4/3)`.
    let dims = DEFAULT_WRAPPED_PLANET_SLAB_DIMS;
    let subgrid: f32 = 3f32.powi(DEFAULT_WRAPPED_PLANET_SLAB_DEPTH as i32);
    let cell_size_world: f32 = (1.0 / 3.0) / subgrid;
    let slab_world_origin = [4.0 / 3.0_f32; 3];

    // Lens: spherical Mercator over the slab. The "planet" is centred
    // on the slab's longitudinal/latitudinal centre at the slab's
    // surface row (top of `slab_y = dims[1] - 1`). The radius is
    // chosen so one full longitudinal wrap == the slab's X extent.
    let body_size = dims[0] as f32 * cell_size_world;
    let lens_center = [
        slab_world_origin[0] + dims[0] as f32 * cell_size_world * 0.5,
        slab_world_origin[1] + dims[1] as f32 * cell_size_world,
        slab_world_origin[2] + dims[2] as f32 * cell_size_world * 0.5,
    ];
    let lens = PlanetLens::from_worldgen(lens_center, body_size, dims);

    let max_depth: u32 = (DEFAULT_WRAPPED_PLANET_EMBEDDING_DEPTH as u32)
        + DEFAULT_WRAPPED_PLANET_SLAB_DEPTH as u32
        + 1;
    let sun = normalize([0.4, 0.7, 0.3]);

    eprintln!(
        "uv_lens_proto: scene R={:.4}, body_size={:.4}, cell_size={:.5}",
        lens.radius, body_size, cell_size_world,
    );

    // 3 views, each demonstrating a different aspect of the lens:
    //   orbit  — full disk; proves the silhouette is round (lens is on)
    //   horizon — low altitude looking sideways; surface curves away
    //   close  — just outside the surface; cell boundaries visible
    //            (each cell's per-tangent-frame normal gives faceted
    //             shading proving "rotate per block" is what's happening)
    render_view(
        "tmp/uv_lens_proto_orbit.png", w, h, &world, &lens,
        slab_world_origin, cell_size_world, max_depth, sun,
        // 4R out, slight overhead pitch.
        [lens_center[0], lens_center[1] + 1.5 * lens.radius, lens_center[2] + 4.0 * lens.radius],
        lens_center, 0.55,
    )?;

    render_view(
        "tmp/uv_lens_proto_horizon.png", w, h, &world, &lens,
        slab_world_origin, cell_size_world, max_depth, sun,
        // 1.5R out, off-equator, looking back at center — shows the
        // limb curve clearly because the camera is much closer.
        [
            lens_center[0] + 0.4 * lens.radius,
            lens_center[1] + 0.3 * lens.radius,
            lens_center[2] + 1.5 * lens.radius,
        ],
        lens_center, 0.7,
    )?;

    render_view(
        "tmp/uv_lens_proto_close.png", w, h, &world, &lens,
        slab_world_origin, cell_size_world, max_depth, sun,
        // 1.05R out, looking tangentially. Each visible cell renders
        // with its OWN tangent-frame-rotated normal, so adjacent cells
        // show a discrete brightness step — the per-block faceting.
        [
            lens_center[0] + 0.0,
            lens_center[1] + 0.05 * lens.radius,
            lens_center[2] + 1.05 * lens.radius,
        ],
        // Aim slightly off-center toward the horizon.
        [
            lens_center[0] - 0.5 * lens.radius,
            lens_center[1],
            lens_center[2] - 0.5 * lens.radius,
        ],
        0.75,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_view(
    path: &str,
    w: usize, h: usize,
    world: &WorldState, lens: &PlanetLens,
    slab_world_origin: [f32; 3], s: f32,
    max_depth: u32, sun: [f32; 3],
    cam: [f32; 3], look_at: [f32; 3], half_fov_rad: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let look = [look_at[0] - cam[0], look_at[1] - cam[1], look_at[2] - cam[2]];
    let forward = normalize(look);
    let world_up = [0.0_f32, 1.0, 0.0];
    // Guard against forward being nearly collinear with world_up
    // (degenerate top-down/bottom-up shots): fall back to +Z up.
    let up_ref = if forward[1].abs() > 0.99 { [0.0, 0.0, 1.0] } else { world_up };
    let right = normalize(cross(forward, up_ref));
    let up = cross(right, forward);

    let aspect = w as f32 / h as f32;
    let half_tan = half_fov_rad.tan();

    let mut pixels = vec![0u8; w * h * 4];
    let (mut hits, mut misses, mut sky_misses) = (0usize, 0usize, 0usize);

    for y in 0..h {
        for x in 0..w {
            let nx = ((x as f32 + 0.5) / w as f32 - 0.5) * 2.0 * aspect * half_tan;
            let ny = (0.5 - (y as f32 + 0.5) / h as f32) * 2.0 * half_tan;
            let ray_dir = normalize([
                forward[0] + right[0] * nx + up[0] * ny,
                forward[1] + right[1] * nx + up[1] * ny,
                forward[2] + right[2] * nx + up[2] * ny,
            ]);

            let (r, g, b) = render_pixel(
                world, lens, slab_world_origin, s,
                cam, ray_dir, max_depth, sun,
                &mut hits, &mut misses, &mut sky_misses,
            );

            let i = (y * w + x) * 4;
            pixels[i] = quantize(r);
            pixels[i + 1] = quantize(g);
            pixels[i + 2] = quantize(b);
            pixels[i + 3] = 255;
        }
    }

    std::fs::create_dir_all("tmp")?;
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;
    eprintln!(
        "uv_lens_proto: {} ({} hits, {} sky, {} lens-anchor-sky)",
        path, hits, sky_misses, misses,
    );
    Ok(())
}

fn render_pixel(
    world: &WorldState,
    lens: &PlanetLens,
    slab_world_origin: [f32; 3],
    s: f32,
    cam: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
    sun: [f32; 3],
    hits: &mut usize,
    misses: &mut usize,
    sky_misses: &mut usize,
) -> (f32, f32, f32) {
    // 1. Lens.
    let p = match lens.project_ray(cam, ray_dir) {
        Some(p) => p,
        None => {
            *sky_misses += 1;
            return sky(ray_dir);
        }
    };

    // 2. Reinterpret slab-axis (de, dn, dh) as world-axis (x, y, z).
    //    Valid because the slab is world-axis-aligned and cubic in
    //    world coords, with slab-X = world-X, slab-Y = world-Y,
    //    slab-Z = world-Z. This is the heart of the lens trick: the
    //    slab DDA marches a STRAIGHT world ray through the FLAT slab,
    //    but the ray's components were chosen by the lens to make the
    //    visited cells match what a sphere DDA would have visited.
    let world_origin = [
        slab_world_origin[0] + p.slab_anchor[0] * s,
        slab_world_origin[1] + p.slab_anchor[1] * s,
        slab_world_origin[2] + p.slab_anchor[2] * s,
    ];
    let world_dir = [p.slab_dir[0] * s, p.slab_dir[1] * s, p.slab_dir[2] * s];

    // Nudge the start INTO the slab from the surface anchor so the
    // first cell traversal is unambiguous.
    let nudge = 1e-4 * s;
    let nudged_origin = [
        world_origin[0] + world_dir[0] * nudge,
        world_origin[1] + world_dir[1] * nudge,
        world_origin[2] + world_dir[2] * nudge,
    ];

    // 3. March the slab.
    let hit = match cpu_raycast(&world.library, world.root, nudged_origin, world_dir, max_depth) {
        Some(h) => h,
        None => {
            *misses += 1;
            // Lens hit the sphere but the slab returned no content —
            // happens at the pole strips (slab Z extent < full lat).
            return sky(ray_dir);
        }
    };
    *hits += 1;

    // 4. Block colour.
    let block_rgb = leaf_color(world, &hit);

    // 5. Lens-rotate the slab face normal to its UV-sphere tangent.
    let world_hit = [
        nudged_origin[0] + world_dir[0] * hit.t,
        nudged_origin[1] + world_dir[1] * hit.t,
        nudged_origin[2] + world_dir[2] * hit.t,
    ];
    let hit_slab = [
        (world_hit[0] - slab_world_origin[0]) / s,
        (world_hit[1] - slab_world_origin[1]) / s,
        (world_hit[2] - slab_world_origin[2]) / s,
    ];
    // Snap to the slab-cell CENTER for the tangent-frame computation.
    // This is the per-block rotation: every voxel in cell (i, j, k) gets
    // ONE rotation (the cell's own tangent frame), giving faceted shading
    // — what "blocks rotated to UV sphere" means visually. Using the
    // continuous hit position would interpolate the frame across the cell
    // and hide the per-block discreteness.
    let cell_center_slab = [
        hit_slab[0].floor() + 0.5,
        hit_slab[1].floor() + 0.5,
        hit_slab[2].floor() + 0.5,
    ];
    let slab_normal = face_to_normal(hit.face);
    let world_normal = lens.shade_normal(cell_center_slab, slab_normal);

    // 6. Diffuse + ambient.
    let diffuse = (world_normal[0] * sun[0]
        + world_normal[1] * sun[1]
        + world_normal[2] * sun[2])
        .max(0.0);
    let lit = 0.3 + 0.7 * diffuse;
    (block_rgb[0] * lit, block_rgb[1] * lit, block_rgb[2] * lit)
}

fn leaf_color(world: &WorldState, hit: &HitInfo) -> [f32; 3] {
    let (node_id, slot) = *hit.path.last().unwrap();
    let Some(node) = world.library.get(node_id) else {
        return [1.0, 0.0, 1.0];
    };
    let block_id = match node.children[slot] {
        Child::Block(id) => id,
        Child::Node(child_id) => world
            .library
            .get(child_id)
            .map(|n| n.representative_block)
            .unwrap_or(0),
        _ => return [1.0, 0.0, 1.0],
    };
    for &(idx, _name, c) in BUILTINS {
        if idx == block_id {
            return [c[0], c[1], c[2]];
        }
    }
    [1.0, 0.0, 1.0]
}

fn face_to_normal(face: u32) -> [f32; 3] {
    match face {
        0 => [1.0, 0.0, 0.0],
        1 => [-1.0, 0.0, 0.0],
        2 => [0.0, 1.0, 0.0],
        3 => [0.0, -1.0, 0.0],
        4 => [0.0, 0.0, 1.0],
        _ => [0.0, 0.0, -1.0],
    }
}

fn sky(d: [f32; 3]) -> (f32, f32, f32) {
    let t = d[1] * 0.5 + 0.5;
    (
        0.7 * (1.0 - t) + 0.3 * t,
        0.8 * (1.0 - t) + 0.5 * t,
        0.95 * (1.0 - t) + 0.85 * t,
    )
}

fn quantize(c: f32) -> u8 {
    (c.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < 1e-12 { v } else { [v[0] / l, v[1] / l, v[2] / l] }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
