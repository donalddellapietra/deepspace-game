use bevy::prelude::*;
use noise::{NoiseFn, Perlin};

/// Seeded terrain height generator using multi-octave Perlin noise.
/// Shared between chunk meshing and player grounding.
#[derive(Resource)]
pub struct TerrainGenerator {
    noise: Perlin,
}

impl TerrainGenerator {
    pub fn new(seed: u32) -> Self {
        Self {
            noise: Perlin::new(seed),
        }
    }

    /// Sample terrain height at a world-space (x, z) position.
    /// Returns Y height. Uses 4 octaves of FBM for natural terrain.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let mut height = 0.0;
        let mut amplitude = 20.0;
        let mut frequency = 0.01;

        for _ in 0..4 {
            let sample = self.noise.get([
                x as f64 * frequency as f64,
                z as f64 * frequency as f64,
            ]) as f32;
            height += sample * amplitude;
            amplitude *= 0.5;
            frequency *= 2.0;
        }

        height
    }
}
