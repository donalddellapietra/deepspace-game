//! Cubed-sphere editing helpers: break, place, and raycast targeting
//! that respects the Cartesian tree as an occluder.

use crate::world::cubesphere::{CsRayHit, SphericalPlanet};
use crate::world::edit;
use crate::world::state::WorldState;
use crate::world::tree::Child;

/// Depth at which cubed-sphere edits and highlighting operate.
///
/// Scales with zoom around a visually-sensible base: default zoom
/// highlights a cell whose on-screen size feels like a "block," and
/// each scroll-in or scroll-out drops or adds one level of 3× finer
/// or coarser cells. Capped below `f32` integer-precision so the
/// shader's `floor(un * 3^depth) == hl_i` comparison stays exact.
/// Past planet.depth = 14 the shader can't distinguish adjacent
/// cells reliably (2^24 ≈ 1.6e7; 3^14 ≈ 4.8e6 — just under).
pub fn cs_edit_depth(cs_planet: Option<&SphericalPlanet>, zoom_level: i32) -> u32 {
    let planet_depth = cs_planet.map(|p| p.depth).unwrap_or(0);
    if planet_depth == 0 { return 0; }
    // Default base depth: cells at ~1/81 of a face edge — visible
    // block-sized on the shell, roughly cubic in all three axes.
    const BASE_DEPTH: i32 = 4;
    const BASELINE_ZOOM: i32 = 15;
    // Shader-safe upper bound on 3^depth so f32 integer compares stay exact.
    const SHADER_SAFE_MAX: i32 = 14;
    // Scroll in (zoom_level < 15) → finer; scroll out → coarser.
    let d = BASE_DEPTH + (BASELINE_ZOOM - zoom_level);
    let max_d = (planet_depth as i32).min(SHADER_SAFE_MAX);
    d.clamp(1, max_d) as u32
}

/// Raycast the planet at the current edit depth AND ensure the
/// hit is closer than any Cartesian tree block along the same
/// ray. Returns the hit or None, so both break and place can
/// share the same targeting logic.
pub fn cs_cursor_hit(
    world: &WorldState,
    cs_planet: Option<&SphericalPlanet>,
    camera_pos: [f32; 3],
    ray_dir: [f32; 3],
    zoom_level: i32,
    edit_depth: u32,
) -> Option<CsRayHit> {
    let depth = cs_edit_depth(cs_planet, zoom_level);
    if depth == 0 { return None; }
    let hit = cs_planet?.raycast(
        &world.library, camera_pos, ray_dir, depth,
    )?;
    let tree_t = edit::cpu_raycast(
        &world.library, world.root,
        camera_pos, ray_dir, edit_depth,
    ).map(|h| h.t).unwrap_or(f32::INFINITY);
    if hit.t >= tree_t { return None; }
    Some(hit)
}

/// Cubed-sphere break: clear the subtree at the targeted cell.
pub fn try_cs_break(
    world: &mut WorldState,
    cs_planet: Option<&mut SphericalPlanet>,
    camera_pos: [f32; 3],
    ray_dir: [f32; 3],
    zoom_level: i32,
    edit_depth: u32,
) -> bool {
    let hit = {
        let planet_ref: Option<&SphericalPlanet> = cs_planet.as_deref();
        match cs_cursor_hit(world, planet_ref, camera_pos, ray_dir, zoom_level, edit_depth) {
            Some(h) => h,
            None => return false,
        }
    };
    let Some(planet) = cs_planet else { return false; };
    planet.set_cell_at_depth(
        &mut world.library,
        hit.face, hit.iu, hit.iv, hit.ir, hit.depth,
        Child::Empty,
    )
}

/// Cubed-sphere place: fill `hit.prev` (the empty cell adjacent
/// to the solid hit on the ray-entry side) with `new_block`.
/// Skips if no prev exists (ray spawned inside solid) or if the
/// adjacent cell already contains the requested block.
pub fn try_cs_place(
    world: &mut WorldState,
    cs_planet: Option<&mut SphericalPlanet>,
    camera_pos: [f32; 3],
    ray_dir: [f32; 3],
    zoom_level: i32,
    edit_depth: u32,
    new_block: u8,
) -> bool {
    let hit = {
        let planet_ref: Option<&SphericalPlanet> = cs_planet.as_deref();
        match cs_cursor_hit(world, planet_ref, camera_pos, ray_dir, zoom_level, edit_depth) {
            Some(h) => h,
            None => return false,
        }
    };
    let Some((face, iu, iv, ir)) = hit.prev else { return false; };
    let Some(planet) = cs_planet else { return false; };
    planet.set_cell_at_depth(
        &mut world.library,
        face, iu, iv, ir, hit.depth,
        Child::Block(new_block),
    )
}
