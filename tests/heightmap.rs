//! GPU heightmap tests.
//!
//! Covers:
//! - Correctness: for known worlds (all-empty, single solid block,
//!   stepped terrain, plain-test world), the per-texel top-Y that
//!   the compute shader writes matches what the tree structure
//!   actually implies.
//! - Timing: the gen compute dispatch stays well inside budget at
//!   the heightmap sizes we ship (81², 243², 729² with
//!   `--release`).
//!
//! Tests use a headless wgpu device (no surface), so they run
//! anywhere with a working Metal / Vulkan / WebGPU driver. macOS
//! sandboxing of `cargo test` may skip device creation; in that
//! case the test returns early with a logged warning rather than
//! failing.
//!
//! The heightmap gen shader expects the same `tree` /
//! `node_offsets` buffer layout the renderer uses (see
//! `world::gpu::pack_tree`), so we reuse that packer unchanged.

#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use wgpu::util::DeviceExt;

use deepspace_game::renderer::heightmap::{
    is_no_ground, ClampUniforms, EntityHeightmapClamp, HeightmapGen, HeightmapTexture,
    HeightmapUniforms, GROUND_NONE, HEIGHTMAP_FORMAT,
};
use deepspace_game::world::anchor::WORLD_SIZE;
use deepspace_game::world::gpu::pack_tree;
use deepspace_game::world::tree::{
    empty_children, slot_index, uniform_children, Child, NodeLibrary,
};

// -----------------------------------------------------------------
// Device plumbing
// -----------------------------------------------------------------

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn create_headless_device() -> Option<Gpu> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok()?;
    let required_limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    }
    .using_resolution(adapter.limits());
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("heightmap_test"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .ok()?;
    Some(Gpu { device, queue })
}

/// Fallible wrapper so tests can skip cleanly if no device is
/// available (macOS sandboxed test runners sometimes can't create
/// Metal devices).
fn gpu_or_skip() -> Option<Gpu> {
    pollster::block_on(create_headless_device())
}

// -----------------------------------------------------------------
// Test-world builders
// -----------------------------------------------------------------

struct TestWorld {
    tree_buf: wgpu::Buffer,
    node_offsets_buf: wgpu::Buffer,
    root_bfs: u32,
    frame_depth: u32,
}

impl TestWorld {
    fn from_library(
        gpu: &Gpu,
        library: &NodeLibrary,
        root: deepspace_game::world::tree::NodeId,
    ) -> Self {
        let (tree, _node_kinds, node_offsets, _node_ids, root_bfs) = pack_tree(library, root);
        let tree_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tree"),
            size: (tree.len() as u64 * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&tree_buf, 0, bytemuck::cast_slice(&tree));
        let node_offsets_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node_offsets"),
            size: (node_offsets.len() as u64 * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &node_offsets_buf,
            0,
            bytemuck::cast_slice(&node_offsets),
        );
        Self {
            tree_buf,
            node_offsets_buf,
            root_bfs,
            frame_depth: 0,
        }
    }
}

// -----------------------------------------------------------------
// Dispatch + readback
// -----------------------------------------------------------------

struct HeightmapRun {
    /// Flat `side × side` array, row-major: `data[v * side + u]`.
    data: Vec<f32>,
    side: u32,
    /// Wall-clock duration for `queue.submit` through
    /// `on_submitted_work_done`. Approximates GPU work time; the
    /// readback copy + mapping are timed separately.
    gpu_ms: f64,
}

fn run_heightmap(
    gpu: &Gpu,
    world: &TestWorld,
    hgen: &HeightmapGen,
    delta: u32,
) -> HeightmapRun {
    let heightmap = HeightmapTexture::new(&gpu.device, &gpu.queue, delta);
    let uniforms = HeightmapUniforms::new(world.root_bfs, world.frame_depth, delta, 0.0, WORLD_SIZE);
    heightmap.write_uniforms(&gpu.queue, &uniforms);

    let bg = hgen.make_bind_group(
        &gpu.device,
        &world.tree_buf,
        &world.node_offsets_buf,
        &heightmap,
    );

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("heightmap_gen_test"),
        });
    hgen.record(&mut encoder, &bg, &heightmap);

    // Readback: copy the texture into a buffer, map, collect.
    let side = heightmap.side;
    let bytes_per_row = side * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr = bytes_per_row.div_ceil(align) * align;
    let readback_size = (padded_bpr * side) as u64;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("heightmap_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &heightmap.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(side),
            },
        },
        wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
    );

    let submit_start = Instant::now();
    gpu.queue.submit(std::iter::once(encoder.finish()));
    let _ = gpu.device.poll(wgpu::PollType::Wait);
    let gpu_ms = submit_start.elapsed().as_secs_f64() * 1000.0;

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let _ = gpu.device.poll(wgpu::PollType::Wait);
    rx.recv().expect("map channel").expect("map ok");
    let mapped = slice.get_mapped_range();

    let mut data = Vec::with_capacity((side * side) as usize);
    for row in 0..side {
        let row_start = (row * padded_bpr) as usize;
        for col in 0..side {
            let offset = row_start + (col as usize) * 4;
            let bytes = [
                mapped[offset],
                mapped[offset + 1],
                mapped[offset + 2],
                mapped[offset + 3],
            ];
            data.push(f32::from_le_bytes(bytes));
        }
    }
    drop(mapped);
    readback.unmap();

    HeightmapRun { data, side, gpu_ms }
}

// -----------------------------------------------------------------
// Correctness tests
// -----------------------------------------------------------------

#[test]
fn empty_world_every_texel_is_no_ground() {
    let Some(gpu) = gpu_or_skip() else {
        eprintln!("skipping: no gpu");
        return;
    };
    let mut lib = NodeLibrary::default();
    let root = lib.insert(empty_children());
    let world = TestWorld::from_library(&gpu, &lib, root);
    let hgen = HeightmapGen::new(&gpu.device);
    let run = run_heightmap(&gpu, &world, &hgen, 2);
    assert_eq!(run.side, 9);
    for (i, &y) in run.data.iter().enumerate() {
        assert!(
            is_no_ground(y),
            "texel {i} expected GROUND_NONE, got {y} (sentinel={GROUND_NONE})",
        );
    }
}

#[test]
fn solid_ground_at_y_zero_slot_reports_one_cell_top_y() {
    // Root: slot y=0 filled with a uniform-stone subtree, y=1 and
    // y=2 empty. Top of ground = (0 + 1) * (WORLD_SIZE / 3) = 1.0.
    let Some(gpu) = gpu_or_skip() else {
        eprintln!("skipping: no gpu");
        return;
    };
    let mut lib = NodeLibrary::default();
    let stone_leaf = lib.insert(uniform_children(Child::Block(1)));
    let mut root_children = empty_children();
    for x in 0..3 {
        for z in 0..3 {
            root_children[slot_index(x, 0, z)] = Child::Node(stone_leaf);
        }
    }
    let root = lib.insert(root_children);
    let world = TestWorld::from_library(&gpu, &lib, root);
    let hgen = HeightmapGen::new(&gpu.device);
    // delta=2 → 9×9 heightmap. Each texel resolves a (collision-
    // depth=2) cell whose top Y is 1.0 (first third of WORLD_SIZE).
    let run = run_heightmap(&gpu, &world, &hgen, 2);
    assert_eq!(run.side, 9);
    for (i, &y) in run.data.iter().enumerate() {
        assert!(
            (y - 1.0).abs() < 1e-4,
            "texel {i}: expected y=1.0, got {y}",
        );
    }
}

#[test]
fn solid_ground_at_top_y_slot_reports_world_size() {
    // Root: slot y=2 filled (above everything else).
    // Top of ground = 3.0 (full WORLD_SIZE).
    let Some(gpu) = gpu_or_skip() else { return; };
    let mut lib = NodeLibrary::default();
    let stone_leaf = lib.insert(uniform_children(Child::Block(1)));
    let mut root_children = empty_children();
    for x in 0..3 {
        for z in 0..3 {
            root_children[slot_index(x, 2, z)] = Child::Node(stone_leaf);
        }
    }
    let root = lib.insert(root_children);
    let world = TestWorld::from_library(&gpu, &lib, root);
    let hgen = HeightmapGen::new(&gpu.device);
    let run = run_heightmap(&gpu, &world, &hgen, 2);
    for (i, &y) in run.data.iter().enumerate() {
        assert!(
            (y - 3.0).abs() < 1e-4,
            "texel {i}: expected y=3.0, got {y}",
        );
    }
}

#[test]
fn stepped_terrain_reports_column_specific_heights() {
    // Build a world where the (x=0, z=0) corner has high terrain
    // (y=2 solid), the (x=2, z=2) corner has low terrain (y=0
    // solid), and the middle is bare. Heightmap should have three
    // distinct values across texels.
    let Some(gpu) = gpu_or_skip() else { return; };
    let mut lib = NodeLibrary::default();
    let stone_leaf = lib.insert(uniform_children(Child::Block(1)));
    let mut root_children = empty_children();
    root_children[slot_index(0, 2, 0)] = Child::Node(stone_leaf); // top-corner tall
    root_children[slot_index(2, 0, 2)] = Child::Node(stone_leaf); // bottom-corner short
    // mid slot (1,1,1) stays empty → texel there = GROUND_NONE.
    let root = lib.insert(root_children);
    let world = TestWorld::from_library(&gpu, &lib, root);
    let hgen = HeightmapGen::new(&gpu.device);
    let run = run_heightmap(&gpu, &world, &hgen, 1); // 3×3 heightmap
    assert_eq!(run.side, 3);
    let at = |u: u32, v: u32| run.data[(v * 3 + u) as usize];
    // Note: u,v axes of the heightmap are world (x, z). Slot (0, *, 0) → texel (0, 0).
    assert!((at(0, 0) - 3.0).abs() < 1e-4, "tall corner at (0,0) got {}", at(0, 0));
    assert!((at(2, 2) - 1.0).abs() < 1e-4, "short corner at (2,2) got {}", at(2, 2));
    assert!(is_no_ground(at(1, 1)), "mid empty got {}", at(1, 1));
}

#[test]
fn deeper_delta_resolves_sub_cell_features() {
    // Root with one stone leaf at slot (1, 0, 1) at depth 2 — that
    // stone sub-cell's top Y = (0+1) * (WORLD_SIZE/9) = 1/3. With
    // delta=2 (9×9 heightmap) the center texel should see it;
    // surrounding texels should be GROUND_NONE since the stone
    // only occupies one slot.
    let Some(gpu) = gpu_or_skip() else { return; };
    let mut lib = NodeLibrary::default();
    let mut mid_node_children = empty_children();
    mid_node_children[slot_index(1, 0, 1)] = Child::Block(1);
    let mid_node = lib.insert(mid_node_children);
    let mut root_children = empty_children();
    root_children[slot_index(1, 1, 1)] = Child::Node(mid_node);
    let root = lib.insert(root_children);
    let world = TestWorld::from_library(&gpu, &lib, root);
    let hgen = HeightmapGen::new(&gpu.device);
    let run = run_heightmap(&gpu, &world, &hgen, 2); // 9×9
    // The single stone lives at root slot (1,1,1) → mid slot (1,0,1).
    // Flattened to 9×9 texel coords: u = 1*3 + 1 = 4, v = 1*3 + 1 = 4.
    let center = run.data[(4 * 9 + 4) as usize];
    // Mid-node y_origin = 1 * (WORLD_SIZE/3) = 1.0. Stone at y=0 of
    // mid-node → top Y = 1.0 + (0+1)*(WORLD_SIZE/9) = 1.0 + 1/3.
    let expected = 1.0 + 1.0 / 3.0;
    assert!(
        (center - expected).abs() < 1e-4,
        "center texel y expected ≈ {expected}, got {center}",
    );
    // All other texels should be GROUND_NONE (nothing else solid).
    for v in 0..9u32 {
        for u in 0..9u32 {
            if u == 4 && v == 4 { continue; }
            let y = run.data[(v * 9 + u) as usize];
            assert!(
                is_no_ground(y),
                "texel ({u},{v}) expected GROUND_NONE, got {y}",
            );
        }
    }
}

#[test]
fn picks_highest_y_when_multiple_layers_solid() {
    // Stack solid at y=0 AND y=2. Top Y should be from y=2.
    let Some(gpu) = gpu_or_skip() else { return; };
    let mut lib = NodeLibrary::default();
    let stone_leaf = lib.insert(uniform_children(Child::Block(1)));
    let mut root_children = empty_children();
    for x in 0..3 {
        for z in 0..3 {
            root_children[slot_index(x, 0, z)] = Child::Node(stone_leaf);
            root_children[slot_index(x, 2, z)] = Child::Node(stone_leaf);
        }
    }
    let root = lib.insert(root_children);
    let world = TestWorld::from_library(&gpu, &lib, root);
    let hgen = HeightmapGen::new(&gpu.device);
    let run = run_heightmap(&gpu, &world, &hgen, 1);
    for &y in &run.data {
        assert!(
            (y - 3.0).abs() < 1e-4,
            "expected top Y = 3.0 (from y=2 layer), got {y}",
        );
    }
}

#[test]
fn plain_test_world_grass_surface_y_off_center() {
    // plain_test_world stacks three root layers:
    //   y_slot=0: stone/checker ground
    //   y_slot=1: grass_surface_l2 (grass at sub-slot y=1)
    //   y_slot=2: air_l2 at most XZ cells, features_l2 at root
    //             slot (x=1, z=1) (wood / brick / sand / leaf).
    //
    // For XZ texels OUTSIDE the (1,1) root-slot column (u / v not
    // in 3..=5), the top of ground is the grass surface: root
    // y_slot=1 -> grass_surface_l2 sub-y_slot=1 (grass). Top-Y =
    // 1 * (WORLD_SIZE/3) + 2 * (WORLD_SIZE/9) = 1 + 2/3.
    let Some(gpu) = gpu_or_skip() else { return; };
    let world_state = deepspace_game::world::bootstrap::plain_test_world();
    let world = TestWorld::from_library(&gpu, &world_state.library, world_state.root);
    let hgen = HeightmapGen::new(&gpu.device);
    let run = run_heightmap(&gpu, &world, &hgen, 2);

    let expected = 1.0 + 2.0 / 3.0;
    let mut checked = 0;
    for v in 0..9u32 {
        for u in 0..9u32 {
            // Skip the center 3×3 block that lands inside
            // features_l2 — separately checked below.
            if (3..=5).contains(&u) && (3..=5).contains(&v) {
                continue;
            }
            let y = run.data[(v * 9 + u) as usize];
            assert!(
                (y - expected).abs() < 1e-3,
                "grass-surface texel ({u},{v}): expected ≈ {expected}, got {y}",
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 9 * 9 - 9, "grass-surface coverage");
}

#[test]
fn plain_test_world_features_raise_ground_at_center() {
    // features_l2 puts wood at mid-column (1, *, 1) → top of the
    // wood stack is at root-y=2 sub-y=1 top = 2 + 2/3 ≈ 2.667.
    // That's the tallest feature; the center texel (u=4, v=4)
    // lands on it.
    let Some(gpu) = gpu_or_skip() else { return; };
    let world_state = deepspace_game::world::bootstrap::plain_test_world();
    let world = TestWorld::from_library(&gpu, &world_state.library, world_state.root);
    let hgen = HeightmapGen::new(&gpu.device);
    let run = run_heightmap(&gpu, &world, &hgen, 2);

    let center = run.data[(4 * 9 + 4) as usize];
    // Wood stack at features_l2 (1, 0, 1) and (1, 1, 1), then
    // leaf_l1 at (1, 2, 1) — leaf is solid, so top reaches
    // y_slot=2 of features_l2 → top-Y = 2 + 3/3 = 3.0.
    let expected = 3.0;
    assert!(
        (center - expected).abs() < 1e-3,
        "features center (u=4, v=4): expected {expected}, got {center}",
    );
}

// -----------------------------------------------------------------
// Timing tests
// -----------------------------------------------------------------

/// Build a moderately non-trivial world with ground at varying Y
/// across the XZ plane. Used for realistic timing — an all-empty
/// world is the best case for the tree walker.
fn build_heightmap_workload_world() -> deepspace_game::world::state::WorldState {
    // Re-use the existing plain test world (has a recursive grass
    // surface — gives the tree walker real work to do).
    deepspace_game::world::bootstrap::plain_test_world()
}

/// Assert that the GPU dispatch for `delta` stays below
/// `budget_ms`. Uses 3 warm-up runs + 5 measured runs; takes the
/// median of the measured ones to absorb scheduling noise.
fn assert_gen_under_budget(delta: u32, budget_ms: f64) {
    let Some(gpu) = gpu_or_skip() else {
        eprintln!("skipping timing: no gpu");
        return;
    };
    let world_state = build_heightmap_workload_world();
    let world = TestWorld::from_library(&gpu, &world_state.library, world_state.root);
    let hgen = HeightmapGen::new(&gpu.device);

    // Warm-up — pipeline compile + first-dispatch driver costs
    // dominate the first run otherwise.
    for _ in 0..3 {
        let _ = run_heightmap(&gpu, &world, &hgen, delta);
    }
    let mut samples: Vec<f64> = Vec::with_capacity(5);
    for _ in 0..5 {
        let run = run_heightmap(&gpu, &world, &hgen, delta);
        samples.push(run.gpu_ms);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    eprintln!(
        "heightmap_gen_timing delta={delta} side={} samples_ms={:?} median={:.3} budget={:.3}",
        3u32.pow(delta), samples, median, budget_ms,
    );
    assert!(
        median <= budget_ms,
        "delta={delta} (side={}) median gpu+readback = {median:.3} ms exceeds budget {budget_ms:.3} ms",
        3u32.pow(delta),
    );
}

#[test]
fn timing_81_x_81_under_4ms() {
    assert_gen_under_budget(4, 4.0);
}

#[test]
fn timing_243_x_243_under_10ms() {
    assert_gen_under_budget(5, 10.0);
}

#[test]
fn timing_729_x_729_under_30ms() {
    // 729² = 531k texels, each walking up to 6 tree levels. Even
    // on a warm cache this is measurable. Budget lives here so a
    // future regression stands out; 30 ms is 3× what current
    // Apple silicon produces, so we'll only trip on real stalls.
    assert_gen_under_budget(6, 30.0);
}

// -----------------------------------------------------------------
// Sentinel / edge cases
// -----------------------------------------------------------------

#[test]
fn ground_none_sentinel_is_distinct_from_valid_y_values() {
    // A Y value in the valid range (0, WORLD_SIZE] must never
    // compare as `is_no_ground`.
    for y in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -0.001, 1.0e5, -1.0e10] {
        assert!(
            !is_no_ground(y),
            "y={y} should NOT be classified as no-ground",
        );
    }
    assert!(is_no_ground(GROUND_NONE));
}

#[test]
fn heightmap_format_is_single_channel_float() {
    assert_eq!(HEIGHTMAP_FORMAT, wgpu::TextureFormat::R32Float);
}

// -----------------------------------------------------------------
// Entity Y clamp tests
// -----------------------------------------------------------------
//
// These exercise `EntityHeightmapClamp` in isolation: we synthesize
// a heightmap texture with known per-texel Y values (bypassing
// `HeightmapGen`) and verify that the clamp pass patches each
// instance's translate.y correctly. Testing the clamp without the
// generator is deliberate — if gen fails and clamp fails, the two
// bugs confound; keeping them decoupled means a correctness
// regression in one leaves the other's tests intact.

/// Instance layout matching `renderer::entity_raster::InstanceData`.
/// Duplicated here (rather than imported) because the raster module
/// is currently private and pulling it through would force
/// widening visibility just for tests.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, PartialEq)]
struct TestInstance {
    translate: [f32; 3],
    scale: f32,
    tint: [f32; 4],
}

/// Allocate a heightmap texture and upload `data` (row-major,
/// `side × side`) directly — no gen pass. This matches the byte
/// layout that `HeightmapTexture::new` allocates and that
/// `textureLoad` in the clamp shader expects.
fn upload_synthetic_heightmap(gpu: &Gpu, side: u32, data: &[f32]) -> HeightmapTexture {
    assert_eq!(data.len(), (side * side) as usize);
    let delta = side.trailing_zeros() / 2; // only used for bookkeeping
    let _ = delta;
    // Reuse HeightmapTexture::new to get a correctly-configured
    // texture, then overwrite the pixels via queue.write_texture.
    // `delta` here is log-of-side-base-3; compute it properly.
    let mut d: u32 = 0;
    let mut s: u32 = 1;
    while s < side {
        s *= 3;
        d += 1;
    }
    assert_eq!(s, side, "side {side} is not a power of 3");
    let hm = HeightmapTexture::new(&gpu.device, &gpu.queue, d);
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &hm.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(side * 4),
            rows_per_image: Some(side),
        },
        wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
    );
    hm
}

/// Build a storage-usable instance buffer populated with
/// `instances`. `COPY_SRC` is added so we can read back via a
/// staging buffer after the clamp dispatch.
fn upload_instances(gpu: &Gpu, instances: &[TestInstance]) -> wgpu::Buffer {
    gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_instances"),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn readback_instances(gpu: &Gpu, buffer: &wgpu::Buffer, count: usize) -> Vec<TestInstance> {
    let bytes = (count * std::mem::size_of::<TestInstance>()) as u64;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_instances_readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_instances_copy"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    gpu.queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let _ = gpu.device.poll(wgpu::PollType::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let out: Vec<TestInstance> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    out
}

/// End-to-end helper: upload heightmap + instances, dispatch
/// clamp, return the post-clamp instances. Measures the compute
/// submit → done duration for timing assertions.
fn run_clamp(
    gpu: &Gpu,
    heightmap: &HeightmapTexture,
    instances_in: &[TestInstance],
) -> (Vec<TestInstance>, f64) {
    let buffer = upload_instances(gpu, instances_in);
    let clamp = EntityHeightmapClamp::new(&gpu.device);
    let u = ClampUniforms::new(
        instances_in.len() as u32,
        heightmap.side,
        WORLD_SIZE,
    );
    let uniforms = clamp.make_uniforms_buffer(&gpu.device, &u);
    let bg = clamp.make_bind_group(&gpu.device, &buffer, &uniforms, heightmap);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("clamp_test"),
        });
    clamp.record(&mut encoder, &bg, instances_in.len() as u32);
    let submit_start = Instant::now();
    gpu.queue.submit(std::iter::once(encoder.finish()));
    let _ = gpu.device.poll(wgpu::PollType::Wait);
    let gpu_ms = submit_start.elapsed().as_secs_f64() * 1000.0;

    let out = readback_instances(gpu, &buffer, instances_in.len());
    (out, gpu_ms)
}

#[test]
fn clamp_snaps_entity_to_ground_y_at_its_texel() {
    let Some(gpu) = gpu_or_skip() else { return; };
    // 3×3 heightmap with a different Y per texel — so a wrong
    // texel lookup produces a wrong Y.
    let side: u32 = 3;
    let cell_xz = WORLD_SIZE / side as f32; // 1.0
    let mut data = vec![0.0_f32; 9];
    for v in 0..3u32 {
        for u in 0..3u32 {
            data[(v * side + u) as usize] = 0.1 * (u as f32 + 1.0) + 10.0 * (v as f32);
        }
    }
    let hm = upload_synthetic_heightmap(&gpu, side, &data);

    // Place an entity dead center of each texel.
    let mut instances = Vec::new();
    for v in 0..3u32 {
        for u in 0..3u32 {
            instances.push(TestInstance {
                translate: [
                    (u as f32 + 0.5) * cell_xz,
                    999.0, // garbage Y — must be overwritten
                    (v as f32 + 0.5) * cell_xz,
                ],
                scale: 1.0,
                tint: [1.0, 1.0, 1.0, 1.0],
            });
        }
    }

    let (out, _) = run_clamp(&gpu, &hm, &instances);
    for v in 0..3u32 {
        for u in 0..3u32 {
            let idx = (v * side + u) as usize;
            let expected = data[idx];
            let got = out[idx].translate[1];
            assert!(
                (got - expected).abs() < 1e-4,
                "texel ({u},{v}): expected y={expected}, got {got}",
            );
            // XZ should be untouched.
            assert_eq!(out[idx].translate[0], instances[idx].translate[0]);
            assert_eq!(out[idx].translate[2], instances[idx].translate[2]);
        }
    }
}

#[test]
fn clamp_leaves_y_untouched_outside_heightmap_extent() {
    let Some(gpu) = gpu_or_skip() else { return; };
    let side: u32 = 3;
    let data = vec![5.0_f32; 9]; // uniform ground
    let hm = upload_synthetic_heightmap(&gpu, side, &data);
    let instances = vec![
        TestInstance { translate: [-1.0, 42.0,  1.5], scale: 1.0, tint: [1.0; 4] },
        TestInstance { translate: [10.0, 43.0,  1.5], scale: 1.0, tint: [1.0; 4] },
        TestInstance { translate: [ 1.5, 44.0, -5.0], scale: 1.0, tint: [1.0; 4] },
        TestInstance { translate: [ 1.5, 45.0, 20.0], scale: 1.0, tint: [1.0; 4] },
    ];
    let (out, _) = run_clamp(&gpu, &hm, &instances);
    for (i, inst) in out.iter().enumerate() {
        let expected_y = instances[i].translate[1];
        assert!(
            (inst.translate[1] - expected_y).abs() < 1e-4,
            "out-of-extent entity {i}: Y changed from {expected_y} to {}",
            inst.translate[1],
        );
    }
}

#[test]
fn clamp_leaves_y_untouched_at_no_ground_sentinel() {
    let Some(gpu) = gpu_or_skip() else { return; };
    let side: u32 = 3;
    let mut data = vec![7.0_f32; 9];
    data[4] = GROUND_NONE; // center texel has no ground
    let hm = upload_synthetic_heightmap(&gpu, side, &data);
    let instances = vec![
        TestInstance {
            translate: [1.5, 100.0, 1.5], // center texel
            scale: 1.0,
            tint: [1.0; 4],
        },
        TestInstance {
            translate: [0.5, 200.0, 0.5], // texel (0,0) — has ground
            scale: 1.0,
            tint: [1.0; 4],
        },
    ];
    let (out, _) = run_clamp(&gpu, &hm, &instances);
    assert!(
        (out[0].translate[1] - 100.0).abs() < 1e-4,
        "no-ground entity Y must be untouched, got {}",
        out[0].translate[1],
    );
    assert!(
        (out[1].translate[1] - 7.0).abs() < 1e-4,
        "ground-bearing entity Y must snap, got {}",
        out[1].translate[1],
    );
}

#[test]
fn clamp_record_with_zero_entities_is_a_noop() {
    // With zero entities, `record` must NOT dispatch — otherwise
    // the shader would process garbage instances + possibly write
    // out-of-bounds. We don't bind a zero-byte buffer (wgpu
    // rejects that outright), so the test dispatches with a stub
    // 1-entity buffer but passes `entity_count = 0` to record. The
    // instance's Y must stay unchanged, proving `record` skipped
    // the dispatch.
    let Some(gpu) = gpu_or_skip() else { return; };
    let hm = upload_synthetic_heightmap(&gpu, 3, &vec![5.0; 9]);
    let stub = vec![TestInstance {
        translate: [1.5, 99.0, 1.5],
        scale: 1.0,
        tint: [1.0; 4],
    }];
    let buffer = upload_instances(&gpu, &stub);
    let clamp = EntityHeightmapClamp::new(&gpu.device);
    let u = ClampUniforms::new(0, hm.side, WORLD_SIZE);
    let uniforms = clamp.make_uniforms_buffer(&gpu.device, &u);
    let bg = clamp.make_bind_group(&gpu.device, &buffer, &uniforms, &hm);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    clamp.record(&mut encoder, &bg, 0);
    gpu.queue.submit(std::iter::once(encoder.finish()));
    let _ = gpu.device.poll(wgpu::PollType::Wait);
    let out = readback_instances(&gpu, &buffer, 1);
    assert!(
        (out[0].translate[1] - 99.0).abs() < 1e-4,
        "zero-count dispatch must leave instance untouched; got {}",
        out[0].translate[1],
    );
}

fn make_grid_instances(n: usize, side: u32) -> Vec<TestInstance> {
    // Spread n entities uniformly across the [0, WORLD_SIZE)²
    // plane. Uses a deterministic grid so the timing test's XZ
    // distribution doesn't depend on RNG state.
    let cell_xz = WORLD_SIZE / side as f32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let u = (i as u32 * 131u32) % side;
        let v = (i as u32 * 241u32) % side;
        out.push(TestInstance {
            translate: [
                (u as f32 + 0.5) * cell_xz,
                0.0,
                (v as f32 + 0.5) * cell_xz,
            ],
            scale: 1.0,
            tint: [1.0; 4],
        });
    }
    out
}

fn assert_clamp_under_budget(n: usize, budget_ms: f64) {
    let Some(gpu) = gpu_or_skip() else {
        eprintln!("skipping clamp timing: no gpu");
        return;
    };
    let side: u32 = 243;
    let data: Vec<f32> = (0..(side * side))
        .map(|i| (i % 123) as f32 * 0.01)
        .collect();
    let hm = upload_synthetic_heightmap(&gpu, side, &data);
    let clamp = EntityHeightmapClamp::new(&gpu.device);
    let instances = make_grid_instances(n, side);

    let buffer = upload_instances(&gpu, &instances);
    let u = ClampUniforms::new(n as u32, side, WORLD_SIZE);
    let uniforms = clamp.make_uniforms_buffer(&gpu.device, &u);
    let bg = clamp.make_bind_group(&gpu.device, &buffer, &uniforms, &hm);

    // Warm-up.
    for _ in 0..3 {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        clamp.record(&mut encoder, &bg, n as u32);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        let _ = gpu.device.poll(wgpu::PollType::Wait);
    }

    let mut samples: Vec<f64> = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        clamp.record(&mut encoder, &bg, n as u32);
        let submit_start = Instant::now();
        gpu.queue.submit(std::iter::once(encoder.finish()));
        let _ = gpu.device.poll(wgpu::PollType::Wait);
        samples.push(submit_start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    eprintln!(
        "clamp_timing n={n} samples_ms={:?} median={:.3} budget={:.3}",
        samples, median, budget_ms,
    );
    assert!(
        median <= budget_ms,
        "clamp({n}) median = {median:.3} ms exceeds budget {budget_ms:.3} ms",
    );
}

#[test]
fn clamp_timing_10k_under_2ms() {
    assert_clamp_under_budget(10_000, 2.0);
}

#[test]
fn clamp_timing_100k_under_5ms() {
    assert_clamp_under_budget(100_000, 5.0);
}

#[test]
fn full_pipeline_gen_then_clamp_end_to_end() {
    // Wire the whole chain: build a world with solid ground at a
    // known Y, run heightmap gen, then place entities with garbage
    // Y values and run clamp — verify each lands on the generated
    // heightmap.
    let Some(gpu) = gpu_or_skip() else { return; };

    // Ground: uniform stone floor in root slot y=0 only. Heightmap
    // will report top-Y = 1.0 for every texel.
    let mut lib = NodeLibrary::default();
    let stone_leaf = lib.insert(uniform_children(Child::Block(1)));
    let mut root_children = empty_children();
    for x in 0..3 {
        for z in 0..3 {
            root_children[slot_index(x, 0, z)] = Child::Node(stone_leaf);
        }
    }
    let root = lib.insert(root_children);
    let world = TestWorld::from_library(&gpu, &lib, root);

    // Build the heightmap.
    let hgen = HeightmapGen::new(&gpu.device);
    let delta: u32 = 2;
    let hm = HeightmapTexture::new(&gpu.device, &gpu.queue, delta);
    let hm_u = HeightmapUniforms::new(world.root_bfs, world.frame_depth, delta, 0.0, WORLD_SIZE);
    hm.write_uniforms(&gpu.queue, &hm_u);
    let hm_bg = hgen.make_bind_group(
        &gpu.device, &world.tree_buf, &world.node_offsets_buf, &hm,
    );
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("e2e") });
    hgen.record(&mut encoder, &hm_bg, &hm);
    gpu.queue.submit(std::iter::once(encoder.finish()));
    let _ = gpu.device.poll(wgpu::PollType::Wait);

    // Now run clamp with a handful of entities at varied XZ.
    let instances = vec![
        TestInstance { translate: [0.5, 100.0, 0.5], scale: 1.0, tint: [1.0; 4] },
        TestInstance { translate: [1.5, -50.0, 1.5], scale: 1.0, tint: [1.0; 4] },
        TestInstance { translate: [2.5,   0.0, 2.5], scale: 1.0, tint: [1.0; 4] },
    ];
    let (out, _) = run_clamp(&gpu, &hm, &instances);
    for (i, inst) in out.iter().enumerate() {
        assert!(
            (inst.translate[1] - 1.0).abs() < 1e-4,
            "entity {i}: expected y=1.0 (gen'd heightmap), got {}",
            inst.translate[1],
        );
    }
}
