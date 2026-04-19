//! Pack-time distance-field analyzer.
//!
//! Packs a world (jerusalem, menger, ...) at a given plain_layers
//! and emits distance-field distributions at two granularities:
//!
//! **Single-node (3×3×3)**: DF measured within one packed node's
//! occupancy mask. Treats both tag=1 leaves and tag=2 child-Node
//! slots as "occupied" — i.e. any slot the shader would descend
//! into. Tells us the safe skip a ray gets within ONE inner DDA
//! frame.
//!
//! **Multi-layer (3^(N+1))**: DF measured after expanding each
//! tag=2 child's own occupancy into the grid, recursively, N
//! levels deep. This matches what a ray actually experiences:
//! descending into a tag=2 cell reveals that its own interior
//! is mostly empty (most cross cells of a Jerusalem sub-cross
//! are themselves empty arms). The ray's effective empty-skip
//! grows with expansion depth because fractal density falls with
//! resolution: Jerusalem is 7/27 ≈ 26% at one level but only
//! (7/27)^2 ≈ 6.7% at two levels.
//!
//! If multi-layer DF grows meaningfully compared to single-layer,
//! it means the shader could store a pre-expanded occupancy grid
//! per node (larger than 27 bits) and skip across runs that the
//! current single-node view misses.
//!
//! Usage:
//!   cargo run --bin df_analysis -- [preset] [plain_layers] [expand]
//!
//! Defaults: jerusalem, 20, 0. `expand 0` = single-node only;
//! `expand 2` = compute the 27×27×27 expansion histogram too.

use deepspace_game::world::bootstrap::{bootstrap_world, WorldPreset};
use deepspace_game::world::gpu::pack_tree;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let preset_name = args.get(0).map(String::as_str).unwrap_or("jerusalem");
    let layers: u8 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let expand: u32 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let preset = match preset_name {
        "jerusalem" => WorldPreset::JerusalemCross,
        "menger" => WorldPreset::Menger,
        "cantor" => WorldPreset::CantorDust,
        "sierpinski-tet" => WorldPreset::SierpinskiTet,
        "sierpinski-pyr" => WorldPreset::SierpinskiPyramid,
        "mausoleum" => WorldPreset::Mausoleum,
        "edge-scaffold" => WorldPreset::EdgeScaffold,
        "hollow-cube" => WorldPreset::HollowCube,
        "plain" => WorldPreset::PlainTest,
        other => {
            eprintln!(
                "unknown preset `{other}` — try: jerusalem, menger, cantor, \
                 sierpinski-tet, sierpinski-pyr, mausoleum, edge-scaffold, \
                 hollow-cube, plain"
            );
            std::process::exit(2);
        }
    };

    eprintln!("Bootstrapping preset={preset_name} plain_layers={layers}");
    let bootstrap = bootstrap_world(preset, Some(layers));
    eprintln!("Packing tree...");
    let (tree, _node_kinds, node_offsets, _node_ids, root_bfs_idx) =
        pack_tree(&bootstrap.world.library, bootstrap.world.root);

    let node_count = node_offsets.len();
    eprintln!(
        "Packed {} unique nodes, tree buffer = {} u32s ({} MB)",
        node_count,
        tree.len(),
        tree.len() * 4 / (1024 * 1024),
    );

    // Chebyshev DF histogram: index is DF value (0..=2), over all
    // (node, empty_cell) pairs.
    let mut cheb_hist = [0u64; 3];
    // Per-axis run histogram: [axis][run_len-1]. axis ordering:
    // 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z. run_len ∈ {1,2,3}.
    let mut axis_hist = [[0u64; 3]; 6];
    // Min-over-used-axes DF: the realistic skip bound for a ray with
    // non-axial direction. For each empty cell, pick a ray octant
    // (+x+y+z worst case = +dir for every axis); the skip is
    // min(run_+x, run_+y, run_+z). Do this for all 8 octants per
    // cell and histogram the minimum.
    let mut octant_hist = [0u64; 3];

    let mut total_empty = 0u64;
    let mut total_occupied = 0u64;
    // Popcount histogram to see how sparse typical nodes are.
    let mut popcount_hist = [0u64; 28];

    for bfs in 0..node_count {
        let header = node_offsets[bfs] as usize;
        let occ = tree[header];
        let pop = occ.count_ones() as usize;
        popcount_hist[pop] += 1;
        total_occupied += pop as u64;

        for slot in 0..27u32 {
            if (occ >> slot) & 1 != 0 {
                continue;
            }
            total_empty += 1;
            let (cx, cy, cz) = (
                (slot % 3) as i32,
                ((slot / 3) % 3) as i32,
                (slot / 9) as i32,
            );

            // Chebyshev DF: min over occupied cells of max(|dx|,|dy|,|dz|).
            // Max possible in 3×3×3 is 2 (diagonal corner to corner).
            let mut cheb: u32 = 2;
            for s in 0..27u32 {
                if (occ >> s) & 1 == 0 {
                    continue;
                }
                let (ox, oy, oz) = (
                    (s % 3) as i32,
                    ((s / 3) % 3) as i32,
                    (s / 9) as i32,
                );
                let d = (cx - ox).abs().max((cy - oy).abs()).max((cz - oz).abs()) as u32;
                if d < cheb {
                    cheb = d;
                }
            }
            cheb_hist[cheb as usize] += 1;

            // Per-axis run length: starting at this empty cell, count
            // consecutive empty cells including self along each of
            // 6 directions, capping at the node boundary.
            let dirs: [(i32, i32, i32); 6] = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ];
            let mut run_len = [0u32; 6];
            for (a, &(dx, dy, dz)) in dirs.iter().enumerate() {
                let mut x = cx;
                let mut y = cy;
                let mut z = cz;
                let mut len = 0u32;
                loop {
                    if x < 0 || x > 2 || y < 0 || y > 2 || z < 0 || z > 2 {
                        break;
                    }
                    let s = (x as u32) + 3 * (y as u32) + 9 * (z as u32);
                    if (occ >> s) & 1 != 0 {
                        break;
                    }
                    len += 1;
                    x += dx;
                    y += dy;
                    z += dz;
                }
                run_len[a] = len;
                // len ∈ {1, 2, 3} (at least 1 because this cell is
                // empty). Clamp for the histogram index.
                let idx = (len.saturating_sub(1).min(2)) as usize;
                axis_hist[a][idx] += 1;
            }

            // Per-octant "all three axes must skip together" DF: for
            // each of 8 octants, pick the run along the matching
            // axes and take the minimum. Histogram the WORST octant
            // — that's the safe skip for diagonal rays from this
            // cell.
            let mut worst_oct: u32 = 3;
            for oct in 0..8u32 {
                let sx = if oct & 1 != 0 { 0 } else { 1 };
                let sy = if oct & 2 != 0 { 2 } else { 3 };
                let sz = if oct & 4 != 0 { 4 } else { 5 };
                let o = run_len[sx].min(run_len[sy]).min(run_len[sz]);
                if o < worst_oct {
                    worst_oct = o;
                }
            }
            let idx = worst_oct.saturating_sub(1).min(2) as usize;
            octant_hist[idx] += 1;
        }
    }

    let total_cells = total_empty + total_occupied;
    println!();
    println!("=== {preset_name} (plain_layers={layers}) ===");
    println!(
        "Packed nodes: {node_count}   total cells: {total_cells}   occupied: {total_occupied} ({:.1}%)   empty: {total_empty} ({:.1}%)",
        100.0 * total_occupied as f64 / total_cells as f64,
        100.0 * total_empty as f64 / total_cells as f64,
    );

    println!();
    println!("Popcount histogram (cells-occupied per node):");
    for pc in 0..=27 {
        if popcount_hist[pc] == 0 {
            continue;
        }
        let pct = 100.0 * popcount_hist[pc] as f64 / node_count as f64;
        println!("  pop={pc:>2}: {:>10} nodes ({:>5.1}%)", popcount_hist[pc], pct);
    }

    println!();
    println!("Within-node Chebyshev DF distribution (over {total_empty} empty cells):");
    for df in 0..=2u32 {
        let pct = 100.0 * cheb_hist[df as usize] as f64 / total_empty as f64;
        println!(
            "  DF={df}: {:>10} cells ({:>5.1}%)",
            cheb_hist[df as usize], pct,
        );
    }
    let cheb_mean = (cheb_hist[0] * 0 + cheb_hist[1] * 1 + cheb_hist[2] * 2) as f64
        / total_empty as f64;
    println!("  mean Chebyshev DF: {cheb_mean:.3}");

    println!();
    println!("Per-axis run length distribution (over {total_empty} empty cells × 6 axes):");
    let axis_names = ["+x", "-x", "+y", "-y", "+z", "-z"];
    for a in 0..6 {
        let r1 = axis_hist[a][0];
        let r2 = axis_hist[a][1];
        let r3 = axis_hist[a][2];
        let total = r1 + r2 + r3;
        let mean = (r1 + r2 * 2 + r3 * 3) as f64 / total as f64;
        println!(
            "  {}: run=1:{:>8} ({:>4.1}%)   run=2:{:>8} ({:>4.1}%)   run=3:{:>8} ({:>4.1}%)   mean={mean:.2}",
            axis_names[a],
            r1, 100.0 * r1 as f64 / total as f64,
            r2, 100.0 * r2 as f64 / total as f64,
            r3, 100.0 * r3 as f64 / total as f64,
        );
    }

    println!();
    println!("Worst-octant DF distribution (safe diagonal-ray skip, {total_empty} empty cells):");
    for df in 0..=2u32 {
        let pct = 100.0 * octant_hist[df as usize] as f64 / total_empty as f64;
        println!(
            "  min_skip={}: {:>10} cells ({:>5.1}%)",
            df + 1,
            octant_hist[df as usize],
            pct,
        );
    }
    let oct_mean = (octant_hist[0] * 1 + octant_hist[1] * 2 + octant_hist[2] * 3) as f64
        / total_empty as f64;
    println!("  mean diagonal-safe skip: {oct_mean:.3} cells");

    println!();
    println!("=== INTERPRETATION ===");
    println!(
        "  A stored DF is worth the cost only if it can replace ≥1 extra DDA cell per hit."
    );
    if cheb_mean < 1.1 {
        println!(
            "  Chebyshev DF mean {cheb_mean:.2} is essentially 1 — storing it would not\n  \
             change inner-loop cost meaningfully. Stop there: within-node SDF is a dud\n  \
             for this fractal."
        );
    } else {
        println!(
            "  Chebyshev DF mean {cheb_mean:.2} > 1 — each safe skip saves ~{:.1}\n  \
             DDA iterations on average in empty regions. Worth storing.",
            cheb_mean - 1.0,
        );
    }
    let axis_max_mean = (0..6)
        .map(|a| (axis_hist[a][0] + axis_hist[a][1] * 2 + axis_hist[a][2] * 3) as f64
            / (axis_hist[a][0] + axis_hist[a][1] + axis_hist[a][2]) as f64)
        .fold(0.0_f64, f64::max);
    println!(
        "  Max per-axis run mean: {axis_max_mean:.2}. Axis-aligned rays through empty\n  \
         rows get this. Diagonal rays only get the min, captured above as\n  \
         diagonal-safe skip = {oct_mean:.2}."
    );

    // ---------------------------------------------------------------
    // Multi-layer expansion analysis.
    //
    // Build a dense boolean grid of size 3^(expand+1), populated by
    // recursively expanding each tag=2 child's occupancy mask into
    // its sub-region. This matches what a ray experiences: a tag=2
    // slot at the current level is mostly empty inside (most cells
    // of the sub-cross are themselves empty), and the ray walks
    // those sub-empty cells before hitting content. Measure DF on
    // the expanded grid to see how much empty-skip is available at
    // finer resolutions that single-node DF misses.
    //
    // This is computed ONLY for the root node. With fractal
    // self-similarity the answer generalizes.
    // ---------------------------------------------------------------

    if expand == 0 {
        return;
    }

    println!();
    println!("=== Multi-layer expansion: {expand} levels (root node) ===");

    let grid_size: usize = 3usize.pow(expand + 1);
    let total_cells = grid_size.pow(3);
    eprintln!(
        "Expanding root into {grid_size}×{grid_size}×{grid_size} = {total_cells} cells..."
    );
    let mut grid = vec![false; total_cells];
    expand_into_grid(
        &tree,
        &node_offsets,
        root_bfs_idx as usize,
        expand,
        &mut grid,
        (0, 0, 0),
        grid_size,
    );

    let occupied: usize = grid.iter().filter(|b| **b).count();
    let empty_count = total_cells - occupied;
    eprintln!(
        "  Expanded grid: {} occupied ({:.3}%), {} empty ({:.3}%)",
        occupied,
        100.0 * occupied as f64 / total_cells as f64,
        empty_count,
        100.0 * empty_count as f64 / total_cells as f64,
    );

    // Chebyshev distance transform via a multi-pass chamfer. Output
    // df_grid[i] = min Chebyshev distance from cell i to any
    // occupied cell. Capped at grid_size.
    let mut df_grid: Vec<u32> = grid
        .iter()
        .map(|b| if *b { 0 } else { u32::MAX })
        .collect();
    // Two-sweep chamfer (forward then backward) for Chebyshev:
    let idx = |x: usize, y: usize, z: usize| -> usize {
        x + grid_size * (y + grid_size * z)
    };
    // Forward pass.
    for z in 0..grid_size {
        for y in 0..grid_size {
            for x in 0..grid_size {
                if df_grid[idx(x, y, z)] == 0 {
                    continue;
                }
                let mut best = df_grid[idx(x, y, z)];
                for &(dx, dy, dz) in &[
                    (-1_i32, 0_i32, 0_i32),
                    (0, -1, 0), (0, 0, -1),
                    (-1, -1, 0), (-1, 0, -1), (0, -1, -1),
                    (-1, -1, -1),
                    (1, -1, 0), (-1, 1, 0),
                    (1, -1, -1), (-1, 1, -1), (-1, -1, 1),
                    (1, -1, -1), (-1, 1, -1),
                ] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0 || ny < 0 || nz < 0 {
                        continue;
                    }
                    if (nx as usize) >= grid_size
                        || (ny as usize) >= grid_size
                        || (nz as usize) >= grid_size
                    {
                        continue;
                    }
                    let nd = df_grid[idx(nx as usize, ny as usize, nz as usize)];
                    if nd != u32::MAX {
                        best = best.min(nd + 1);
                    }
                }
                df_grid[idx(x, y, z)] = best;
            }
        }
    }
    // Backward pass.
    for z in (0..grid_size).rev() {
        for y in (0..grid_size).rev() {
            for x in (0..grid_size).rev() {
                if df_grid[idx(x, y, z)] == 0 {
                    continue;
                }
                let mut best = df_grid[idx(x, y, z)];
                for &(dx, dy, dz) in &[
                    (1_i32, 0_i32, 0_i32),
                    (0, 1, 0), (0, 0, 1),
                    (1, 1, 0), (1, 0, 1), (0, 1, 1),
                    (1, 1, 1),
                    (-1, 1, 0), (1, -1, 0),
                    (-1, 1, 1), (1, -1, 1), (1, 1, -1),
                ] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0 || ny < 0 || nz < 0 {
                        continue;
                    }
                    if (nx as usize) >= grid_size
                        || (ny as usize) >= grid_size
                        || (nz as usize) >= grid_size
                    {
                        continue;
                    }
                    let nd = df_grid[idx(nx as usize, ny as usize, nz as usize)];
                    if nd != u32::MAX {
                        best = best.min(nd + 1);
                    }
                }
                df_grid[idx(x, y, z)] = best;
            }
        }
    }

    // Histograms of DF values over empty cells, separated by
    // whether the cell sits in a tag=0 anchor slot ("outer" — what
    // rays at anchor depth traverse BEFORE any descent) or inside a
    // tag=2 anchor slot ("inner" — only reached after descending).
    //
    // The outer histogram tells us what multi-layer DF buys a ray
    // that hasn't descended yet. The inner histogram tells us what
    // a ray inside a sub-cross sees. These are very different
    // optimisation questions.
    let max_df = df_grid.iter().copied().filter(|v| *v != u32::MAX).max().unwrap_or(0);
    let mut df_hist = vec![0u64; (max_df as usize) + 2];
    let mut df_hist_outer = vec![0u64; (max_df as usize) + 2];
    let mut df_hist_inner = vec![0u64; (max_df as usize) + 2];
    let df_hist_len = df_hist.len();
    let anchor_stride = 3usize.pow(expand); // size of one anchor slot in grid cells
    let root_header_off = node_offsets[root_bfs_idx as usize] as usize;
    let root_occ = tree[root_header_off];
    for (i, &b) in grid.iter().enumerate() {
        if b {
            continue;
        }
        let x = i % grid_size;
        let y = (i / grid_size) % grid_size;
        let z = i / (grid_size * grid_size);
        let ax = x / anchor_stride;
        let ay = y / anchor_stride;
        let az = z / anchor_stride;
        let anchor_slot = ax + 3 * ay + 9 * az;
        let anchor_is_tag2 = (root_occ >> anchor_slot) & 1 != 0;

        let d = df_grid[i];
        if d == u32::MAX {
            df_hist[df_hist_len - 1] += 1;
            if anchor_is_tag2 {
                df_hist_inner[df_hist_len - 1] += 1;
            } else {
                df_hist_outer[df_hist_len - 1] += 1;
            }
        } else {
            df_hist[d as usize] += 1;
            if anchor_is_tag2 {
                df_hist_inner[d as usize] += 1;
            } else {
                df_hist_outer[d as usize] += 1;
            }
        }
    }

    fn report_hist(
        label: &str,
        hist: &[u64],
        max_df: u32,
        grid_size: usize,
        expand: u32,
    ) -> f64 {
        let total: u64 = hist.iter().sum();
        if total == 0 {
            println!("{label}: (no cells in this class)");
            return 0.0;
        }
        println!("{label} (n={total}):");
        let mut df_sum = 0u64;
        let mut df_n = 0u64;
        for d in 0..=max_df as usize {
            if hist[d] == 0 {
                continue;
            }
            let pct = 100.0 * hist[d] as f64 / total as f64;
            println!("  DF={d:>3}: {:>10} cells ({:>5.2}%)", hist[d], pct);
            df_sum += hist[d] as u64 * d as u64;
            df_n += hist[d] as u64;
        }
        if hist[hist.len() - 1] > 0 {
            println!(
                "  DF=unreachable: {:>10} cells (no occupied in grid)",
                hist[hist.len() - 1],
            );
        }
        let df_mean = df_sum as f64 / df_n.max(1) as f64;
        println!(
            "  mean DF: {df_mean:.2} sub-cells = {:.3} parent-cells = {:.3} world units",
            df_mean / 3.0_f64.powi(expand as i32),
            df_mean / grid_size as f64,
        );
        df_mean
    }

    println!();
    println!(
        "Chebyshev DF on {grid_size}^3 expanded grid (over {empty_count} empty cells total):"
    );
    let df_mean = report_hist("  OVERALL", &df_hist, max_df, grid_size, expand);

    println!();
    println!(
        "By anchor-slot type (at sub-cell size 1/{grid_size}; anchor = top-level 3×3×3):"
    );
    let df_outer_mean = report_hist(
        "  OUTER (cells in tag=0 anchor slots — rays reach these without descending)",
        &df_hist_outer, max_df, grid_size, expand,
    );
    let df_inner_mean = report_hist(
        "  INNER (cells inside tag=2 anchor slots — only reached after descent)",
        &df_hist_inner, max_df, grid_size, expand,
    );

    // Compare single-layer DF against each class of multi-layer
    // empty. The OUTER class is the decisive one — it's what rays
    // traversing empty space at anchor depth would skip past. The
    // INNER class is only relevant if the shader also operates at
    // finer resolution inside tag=2 cells.
    let outer_in_parent = df_outer_mean / 3.0_f64.powi(expand as i32);
    let inner_in_parent = df_inner_mean / 3.0_f64.powi(expand as i32);
    let overall_in_parent = df_mean / 3.0_f64.powi(expand as i32);
    let _ = overall_in_parent;
    println!();
    println!("=== COMPARISON (parent-cell units) ===");
    println!("  Single-node   mean: {cheb_mean:.2}");
    println!("  Multi-layer OUTER mean (tag=0 anchor slots): {outer_in_parent:.2}");
    println!("  Multi-layer INNER mean (tag=2 anchor slots): {inner_in_parent:.2}");
    println!();
    if outer_in_parent > cheb_mean * 1.5 {
        println!(
            "  OUTER rays: multi-layer DF is {:.1}× single-layer. A ray in an\n  \
             empty anchor slot could skip further per iteration than the 3×3×3\n  \
             view suggests. Worth considering a per-node finer DF grid.",
            outer_in_parent / cheb_mean,
        );
    } else {
        println!(
            "  OUTER rays: multi-layer DF is only {:.1}× single-layer. No useful\n  \
             gain — the tag=0 anchor slots are BOUNDED BY tag=2 neighbours whose\n  \
             content sits ~1 parent-cell away regardless of sub-cell resolution.",
            outer_in_parent / cheb_mean,
        );
    }
}

/// Recursively write the packed-tree contents into a dense boolean
/// occupancy grid. `levels` remaining = how many more tiers of
/// substitution to perform. At level 0, the current node's tag=2
/// children are collapsed into their slot being "occupied" (any
/// subtree with content); at level >0, we descend into the child's
/// own 3×3×3 occupancy and keep going.
///
/// This is what the shader's DDA would see if it operated on a
/// 3^(levels+1) × 3^(levels+1) × 3^(levels+1) pre-expanded grid
/// rooted at one node, instead of stepping through 3×3×3 + descent
/// at every level separately.
fn expand_into_grid(
    tree: &[u32],
    node_offsets: &[u32],
    node_bfs_idx: usize,
    levels: u32,
    grid: &mut [bool],
    origin: (usize, usize, usize),
    grid_size: usize,
) {
    let header_off = node_offsets[node_bfs_idx] as usize;
    let occ = tree[header_off];
    let first_child = tree[header_off + 1] as usize;

    // Sub-cell size in the grid: how many grid cells per slot at
    // this level. At top-level recursion in a 3^(N+1) grid with
    // levels=N, each slot is 3^N cells wide.
    let sub_stride = 3usize.pow(levels);

    for slot in 0..27u32 {
        let (sx, sy, sz) = (
            (slot % 3) as usize,
            ((slot / 3) % 3) as usize,
            (slot / 9) as usize,
        );
        let cell_origin = (
            origin.0 + sx * sub_stride,
            origin.1 + sy * sub_stride,
            origin.2 + sz * sub_stride,
        );
        if (occ >> slot) & 1 == 0 {
            // Empty: leave false. No content anywhere in this sub-cube.
            continue;
        }
        // Read the packed child entry.
        let rank = (occ & ((1u32 << slot) - 1)).count_ones() as usize;
        let child_off = first_child + rank * 2;
        let packed = tree[child_off];
        let tag = packed & 0xFFu32;

        if tag == 1 {
            // Leaf: fill the whole sub-cube.
            fill_cube(grid, grid_size, cell_origin, sub_stride);
            continue;
        }
        if tag != 2 {
            // Unknown tag (sphere body / face / compact). Treat as
            // occupied sub-cube conservatively so DF doesn't over-
            // report skip past it.
            fill_cube(grid, grid_size, cell_origin, sub_stride);
            continue;
        }

        // Node child.
        if levels == 0 {
            // No more expansion budget; just mark the cell occupied.
            fill_cube(grid, grid_size, cell_origin, sub_stride);
            continue;
        }
        let child_bfs = tree[child_off + 1] as usize;
        expand_into_grid(
            tree, node_offsets, child_bfs, levels - 1, grid, cell_origin, grid_size,
        );
    }
}

fn fill_cube(
    grid: &mut [bool],
    grid_size: usize,
    origin: (usize, usize, usize),
    extent: usize,
) {
    for dz in 0..extent {
        for dy in 0..extent {
            for dx in 0..extent {
                let x = origin.0 + dx;
                let y = origin.1 + dy;
                let z = origin.2 + dz;
                grid[x + grid_size * (y + grid_size * z)] = true;
            }
        }
    }
}
