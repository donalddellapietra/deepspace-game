//! `.vox` / `.vxs` model embedding.

use super::WorldBootstrap;
use crate::world::anchor::{Path, WorldPos};
use crate::world::state::WorldState;
use crate::world::tree::{BRANCH, Child, NodeLibrary, MAX_DEPTH};

/// Load a `.vox` file via the import pipeline and embed it as the
/// child of a tree of `total_depth` levels. The model sits at the
/// center slot of each wrapper layer so the camera can spawn in
/// air nearby and look at it from any side.
///
/// Panics if the file doesn't exist or doesn't parse — this is a
/// bootstrap function, not a content runtime path.
pub fn bootstrap_vox_model_world(
    path: &std::path::Path,
    total_depth: u8,
    interior_depth: u8,
) -> WorldBootstrap {
    use crate::import::{self, tree_builder};
    use crate::world::tree::{CENTER_SLOT, empty_children};

    let total_depth = total_depth.max(1).min(MAX_DEPTH as u8);

    let mut lib = NodeLibrary::default();
    let mut registry = crate::world::palette::ColorRegistry::new();

    // Use the sparse path for `.vxs` when `interior_depth == 0` —
    // iterates only occupied voxels instead of the padded cube. For
    // Sponza (5 M voxels in a 729³ ≈ 388 M padded cube) this is
    // ~77× less work at load time.
    let ext_is_vxs = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("vxs"))
        .unwrap_or(false);

    let (size_x, size_y, size_z, model_root_id) = if ext_is_vxs && interior_depth == 0 {
        let t = std::time::Instant::now();
        let sparse = import::vxs::load_sparse(path, &mut registry)
            .unwrap_or_else(|e| panic!("failed to load {:?}: {}", path, e));
        eprintln!(
            "vox_world: loaded (sparse) {:?} ({}x{}x{} = {} voxels) in {:.2?}",
            path,
            sparse.size_x,
            sparse.size_y,
            sparse.size_z,
            sparse.voxels.len(),
            t.elapsed(),
        );
        let t = std::time::Instant::now();
        let root = tree_builder::build_tree_sparse(&sparse, &mut lib);
        eprintln!(
            "vox_world: sparse tree build in {:.2?} ({} library nodes)",
            t.elapsed(),
            lib.len(),
        );
        (
            sparse.size_x as usize,
            sparse.size_y as usize,
            sparse.size_z as usize,
            root,
        )
    } else {
        let model = import::load(path, &mut registry)
            .unwrap_or_else(|e| panic!("failed to load {:?}: {}", path, e));
        eprintln!(
            "vox_world: loaded {:?} ({}x{}x{} = {} voxels, interior_depth={})",
            path,
            model.size_x,
            model.size_y,
            model.size_z,
            model.data.iter().filter(|&&v| v != 0).count(),
            interior_depth,
        );
        let root =
            tree_builder::build_tree_with_interior(&model, &mut lib, interior_depth);
        (model.size_x, model.size_y, model.size_z, root)
    };

    // Silhouette depth: log3(max_dim). Each voxel then contributes
    // `interior_depth` additional levels (uniform-fill subtree),
    // so the model's full tree footprint is silhouette + interior.
    let max_dim = size_x.max(size_y).max(size_z).max(1);
    let mut dim = 1usize;
    let mut silhouette_depth: u8 = 0;
    while dim < max_dim {
        dim *= BRANCH;
        silhouette_depth += 1;
    }
    let model_depth = silhouette_depth.saturating_add(interior_depth);
    eprintln!(
        "vox_world: silhouette depth={}, interior depth={}, model total={}, library nodes after build={}",
        silhouette_depth, interior_depth, model_depth, lib.len(),
    );

    // Wrap in outer air layers up to target total_depth. Each wrap
    // places the current subtree at the center slot of a 27-child
    // node whose other 26 slots are Empty.
    //
    // When `vox_copies > 1`, the FINAL wrap layer fills multiple slots
    // instead of just the center — content-addressed dedup means all
    // copies share the same NodeId, so library size doesn't grow, but
    // ray-traversal working set does. Used for scene-scaling stress tests.
    let copies = std::env::var("VOX_COPIES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1)
        .clamp(1, 27);
    let mut current = model_root_id;
    let wraps = total_depth.saturating_sub(model_depth);
    for w in 0..wraps {
        let mut children = empty_children();
        if copies > 1 && w == wraps - 1 {
            // Outermost wrap: place `copies` instances across slots.
            for slot in 0..(copies as usize) {
                children[slot] = Child::Node(current);
            }
        } else {
            children[CENTER_SLOT] = Child::Node(current);
        }
        current = lib.insert(children);
    }
    eprintln!("vox_world: placed {} copies in outermost wrap", copies);
    eprintln!(
        "vox_world: wrapped {} layers, final tree depth={}, library total={}",
        wraps, wraps + model_depth, lib.len(),
    );

    lib.ref_inc(current);
    let world = WorldState { root: current, library: lib };

    // Compute the model's world-space bounds so spawn is always at a
    // position the camera can see the model from, regardless of the
    // model's voxel dimensions or how deep the wrap put it.
    //
    // Geometry: the model is wrapped at `CENTER_SLOT=(1,1,1)` of each
    // outer layer, so it lives in cell (1,1,1)^(wraps) of the root.
    // That cell's world extent is `[wrap_origin, wrap_origin+wrap_size]^3`
    // with `wrap_size = 3 * (1/3)^wraps` and `wrap_origin = 1.5 - wrap_size/2`.
    // Inside that cell, the model's voxel grid fills
    // `[0, size_axis/padded]` of the cell, where `padded = 3^silhouette_depth`
    // (silhouette-only; interior_depth adds layers *inside* each voxel but
    // doesn't change the voxel grid's extent inside the wrap cell).
    let padded = BRANCH.pow(silhouette_depth as u32) as f32;
    let wrap_size = 3.0 * (1.0 / BRANCH as f32).powi(wraps as i32);
    let wrap_origin = 1.5 - wrap_size / 2.0;
    let extent_x = wrap_size * (size_x as f32 / padded);
    let extent_y = wrap_size * (size_y as f32 / padded);
    let extent_z = wrap_size * (size_z as f32 / padded);
    let center_x = wrap_origin + extent_x / 2.0;
    let center_y = wrap_origin + extent_y / 2.0;
    let center_z = wrap_origin + extent_z / 2.0;
    eprintln!(
        "vox_world: model world bounds x=[{:.3}..{:.3}] y=[{:.3}..{:.3}] z=[{:.3}..{:.3}], center=({:.3},{:.3},{:.3})",
        wrap_origin, wrap_origin + extent_x,
        wrap_origin, wrap_origin + extent_y,
        wrap_origin, wrap_origin + extent_z,
        center_x, center_y, center_z,
    );

    // Spawn directly above the model's x/z centroid, outside the
    // wrap cell (y > wrap_cell_max), looking straight down.
    //
    // Why this spawn layout works reliably:
    //  - Camera x,z inside the model's voxel footprint → rays going
    //    down the pitch axis hit model cells (not empty siblings).
    //  - Camera y > 2.0 places it in a root cell ABOVE the wrap cell,
    //    so rays traverse an empty cell first, then enter the model
    //    tree via the wrap — no "inside-the-model-tree at spawn"
    //    edge case.
    //  - Pitch ≈ -π/2 keeps the ray direction aligned with the
    //    thin-slab Z axis of most humanoid GLBs (soldier, fox), so
    //    you see the whole silhouette in one view.
    let cam_x = center_x.clamp(0.05, 2.95);
    let cam_y = (wrap_origin + wrap_size + 0.5).min(2.95);
    let cam_z = center_z.clamp(0.05, 2.95);
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [cam_x, cam_y, cam_z],
        2,
        &world.library,
        world.root,
    ).deepened_to(3, &world.library, world.root);
    let yaw = 0.0;
    let pitch = -1.5;   // nearly straight down

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: yaw,
        default_spawn_pitch: pitch,
        plain_layers: total_depth,
        color_registry: registry,
    }
}
