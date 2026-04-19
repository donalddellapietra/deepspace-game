//! Pack-time distance-field analyzer.
//!
//! Packs a world (jerusalem, menger, ...) at a given plain_layers,
//! walks every unique node in the deduplicated compact tree, and
//! computes two distance-field distributions over empty cells:
//!
//! 1. **Chebyshev DF within node (3×3×3)**: safe skip distance for
//!    any ray direction. `DF=2` means the ray can advance 2 cells
//!    anywhere without passing a content cell.
//!
//! 2. **Per-axis run length**: for each empty cell and each of the
//!    6 axis directions, how many consecutive empty cells (including
//!    self) before the first occupied cell or the node boundary.
//!    `run=3` means the ray can traverse the full 3-cell row in
//!    that axis with zero content hits.
//!
//! The analyzer prints a histogram of each metric so we can judge
//! whether a stored DF buys enough shader-side skips to justify
//! the +storage-per-node cost. If Chebyshev DF is "1 everywhere"
//! (which hand-analysis predicts for Jerusalem and Menger-family),
//! the per-axis run is the only variant worth storing.
//!
//! Usage:
//!   cargo run --bin df_analysis -- [preset] [plain_layers]
//!
//! Defaults: jerusalem, 20.

use deepspace_game::world::bootstrap::{bootstrap_world, WorldPreset};
use deepspace_game::world::gpu::pack_tree;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let preset_name = args.get(0).map(String::as_str).unwrap_or("jerusalem");
    let layers: u8 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

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
    let (tree, _node_kinds, node_offsets, _node_ids, _root) =
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
}
