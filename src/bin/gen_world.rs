//! Offline world generator. Builds a world tree, prebakes all meshes,
//! and serializes both to `assets/`. Run when terrain parameters change:
//!
//! ```sh
//! cargo run --bin gen_world
//! ```
//!
//! The main game binary loads these files at startup instead of
//! generating the world or cold-baking meshes in-process.

use std::path::PathBuf;
use std::time::Instant;

use deepspace_game::world::render::prebake_node_raw;
use deepspace_game::world::serial::{
    write_prebaked_indexed, write_world_file, PrebakedMeshes,
};
use deepspace_game::world::state::WorldState;
use deepspace_game::world::tree::EMPTY_NODE;

fn main() {
    let start = Instant::now();

    // Generate the world.
    eprintln!("generating world...");
    let world = WorldState::new_sphere();
    let lib_len = world.library.len();
    eprintln!(
        "  generated in {:.1}s, {} library entries",
        start.elapsed().as_secs_f64(),
        lib_len,
    );

    // Write world.bin.
    let world_path = PathBuf::from("assets/world.bin");
    if let Some(parent) = world_path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create assets dir");
    }
    eprintln!("writing {}...", world_path.display());
    write_world_file(&world_path, world.root, &world.library)
        .expect("failed to write world file");
    let world_size = std::fs::metadata(&world_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "  wrote {} ({:.1} KB)",
        world_path.display(),
        world_size as f64 / 1024.0,
    );

    // Prebake meshes for every node in the library.
    eprintln!("prebaking meshes...");
    let bake_start = Instant::now();
    let mut prebaked = PrebakedMeshes::new();
    let mut baked_count = 0usize;
    let mut skipped = 0usize;

    for (&node_id, _node) in world.library.iter_nodes() {
        if node_id == EMPTY_NODE {
            continue;
        }
        let raw = prebake_node_raw(&world, node_id);
        if raw.is_empty() {
            skipped += 1;
        } else {
            prebaked.insert(node_id, raw);
            baked_count += 1;
        }
    }
    eprintln!(
        "  prebaked {} nodes, skipped {} empty, in {:.1}s",
        baked_count,
        skipped,
        bake_start.elapsed().as_secs_f64(),
    );

    // Write meshes.idx + meshes.bin (indexed format for on-demand loading).
    let idx_path = PathBuf::from("assets/meshes.idx");
    let meshes_path = PathBuf::from("assets/meshes.bin");
    eprintln!("writing {} + {}...", idx_path.display(), meshes_path.display());
    write_prebaked_indexed(&idx_path, &meshes_path, &prebaked)
        .expect("failed to write prebaked meshes");
    let idx_size = std::fs::metadata(&idx_path).map(|m| m.len()).unwrap_or(0);
    let meshes_size = std::fs::metadata(&meshes_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "  wrote {} ({:.1} KB) + {} ({:.1} KB)",
        idx_path.display(), idx_size as f64 / 1024.0,
        meshes_path.display(), meshes_size as f64 / 1024.0,
    );

    eprintln!(
        "done in {:.1}s total",
        start.elapsed().as_secs_f64(),
    );
}
