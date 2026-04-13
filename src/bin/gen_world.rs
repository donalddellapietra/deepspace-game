//! Offline world generator. Builds a world tree and serializes it to
//! `assets/world.bin`. Run when terrain parameters change:
//!
//! ```sh
//! cargo run --bin gen_world
//! ```
//!
//! The main game binary loads this file at startup instead of
//! generating the world in-process.

use std::path::PathBuf;
use std::time::Instant;

// The gen_world binary lives inside the deepspace-game crate, so it
// can access all internal modules directly.
use deepspace_game::world::serial::write_world_file;
use deepspace_game::world::state::WorldState;

fn main() {
    let start = Instant::now();

    // Generate the world. Currently uses the sphere builder; swap to
    // any other builder (grassland, terrain, etc.) as needed.
    eprintln!("generating world...");
    let world = WorldState::new_sphere();
    eprintln!(
        "  generated in {:.1}s, {} library entries",
        start.elapsed().as_secs_f64(),
        world.library.len(),
    );

    // Write to assets/world.bin.
    let out_path = PathBuf::from("assets/world.bin");
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create assets dir");
    }
    eprintln!("writing {}...", out_path.display());
    write_world_file(&out_path, world.root, &world.library)
        .expect("failed to write world file");

    let file_size = std::fs::metadata(&out_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "  wrote {} ({:.1} KB) in {:.1}s total",
        out_path.display(),
        file_size as f64 / 1024.0,
        start.elapsed().as_secs_f64(),
    );
}
