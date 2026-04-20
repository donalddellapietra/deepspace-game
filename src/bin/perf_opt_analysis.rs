//! Pack-time analyzer for **actionable** LOD optimizations.
//!
//! The older `df_analysis` computes within-node Chebyshev DF and a
//! "worst-octant" skip metric that over-conservatively assumes every
//! ray is exactly diagonal. This tool fixes two specific omissions:
//!
//! 1. **Per-axis DF** — reports the mean run length along each of
//!    the 6 axis directions separately. A ray's dominant axis
//!    determines what skip it can actually use; worst-octant only
//!    describes pure-diagonal rays.
//!
//! 2. **Path-mask cull rate** — at every packed node, for every
//!    (entry_cell, ray_octant) configuration, computes the
//!    conservative tensor-product "reachable mask" and checks
//!    whether the ray's path through this node intersects any
//!    occupied slot. Reports the fraction of (node, entry, octant)
//!    triples where the cull would fire — i.e. the descent can be
//!    skipped entirely because no occupied cell lies in the ray's
//!    path.
//!
//! Together these measure two genuinely independent optimizations:
//! DF attacks empty-cell advances (`avg_empty`); path-mask cull
//! attacks wasted descents (`avg_descend` + `avg_oob` pair). Both
//! can apply simultaneously.
//!
//! Usage:
//!   cargo run --release --bin perf_opt_analysis -- [preset] [plain_layers]
//!
//! Defaults: jerusalem 20.

use deepspace_game::world::bootstrap::{bootstrap_world, WorldPreset};
use deepspace_game::world::gpu::pack_tree;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let preset_name = args.get(0).map(String::as_str).unwrap_or("jerusalem");
    let layers: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);

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
            eprintln!("unknown preset `{other}`");
            std::process::exit(2);
        }
    };

    eprintln!("Bootstrapping preset={preset_name} plain_layers={layers}");
    let bootstrap = bootstrap_world(preset, Some(layers));
    eprintln!("Packing tree...");
    let (tree, _node_kinds, node_offsets, _node_ids, _root_bfs_idx, _grid) =
        pack_tree(&bootstrap.world.library, bootstrap.world.root);

    let node_count = node_offsets.len();
    eprintln!(
        "Packed {} unique nodes, {} u32s ({} MB)",
        node_count,
        tree.len(),
        tree.len() * 4 / (1024 * 1024).max(1),
    );

    println!();
    println!("=== {preset_name} plain_layers={layers} ===");
    println!("Packed nodes: {node_count}");

    analyze_per_axis_df(&tree, &node_offsets);
    analyze_path_mask_cull(&tree, &node_offsets);
    analyze_combined_projection(&tree, &node_offsets);
}

/// For every empty cell in every node, measure the run length along
/// each of the 6 axis directions. Reports per-axis means. A ray
/// whose *dominant* axis is `+X` can use the `+X` run length as its
/// effective skip when in an empty cell.
fn analyze_per_axis_df(tree: &[u32], node_offsets: &[u32]) {
    // For each of 6 axes, sum of run lengths (1..=3) per empty cell.
    let mut axis_sum = [0u64; 6];
    let mut axis_count = [0u64; 6];
    // Per-axis run histogram: [axis][run-1].
    let mut axis_hist = [[0u64; 3]; 6];
    let dirs: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ];

    for &off in node_offsets {
        let occ = tree[off as usize];
        for slot in 0..27u32 {
            if (occ >> slot) & 1 != 0 {
                continue;
            }
            let (cx, cy, cz) = (slot as i32 % 3, (slot as i32 / 3) % 3, slot as i32 / 9);
            for (a, &(dx, dy, dz)) in dirs.iter().enumerate() {
                let (mut x, mut y, mut z) = (cx, cy, cz);
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
                axis_sum[a] += len as u64;
                axis_count[a] += 1;
                axis_hist[a][(len.saturating_sub(1).min(2)) as usize] += 1;
            }
        }
    }

    let names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];
    println!();
    println!("─────────── Per-axis DF (mean empty-run length) ───────────");
    println!("A ray's DOMINANT axis determines effective skip. Worst-octant");
    println!("Chebyshev (df_analysis) is min over axes; this is each one.");
    println!();
    for a in 0..6 {
        let mean = axis_sum[a] as f64 / axis_count[a].max(1) as f64;
        let [r1, r2, r3] = axis_hist[a];
        let tot = (r1 + r2 + r3).max(1);
        println!(
            "  {}: mean={mean:.2}    run=1:{:>5.1}%   run=2:{:>5.1}%   run=3:{:>5.1}%",
            names[a],
            100.0 * r1 as f64 / tot as f64,
            100.0 * r2 as f64 / tot as f64,
            100.0 * r3 as f64 / tot as f64,
        );
    }

    let max_axis = (0..6)
        .map(|a| axis_sum[a] as f64 / axis_count[a].max(1) as f64)
        .fold(0.0_f64, f64::max);
    let min_axis = (0..6)
        .map(|a| axis_sum[a] as f64 / axis_count[a].max(1) as f64)
        .fold(f64::INFINITY, f64::min);
    println!();
    println!("  Best-axis mean: {max_axis:.2}  (axis-aligned ray can skip this many cells)");
    println!("  Worst-axis mean: {min_axis:.2}");
    if max_axis > 1.3 {
        println!(
            "  → Per-axis DF of {max_axis:.1} means axis-dominant rays get ~{:.0}%",
            100.0 * (max_axis - 1.0) / max_axis,
        );
        println!("    fewer empty steps vs no-DF baseline. Storing 6 axes × 2 bits per slot");
        println!("    (27 × 12 = 324 bits/node) lets every ray use its dominant-axis value.");
    } else {
        println!(
            "  → Per-axis max only {max_axis:.2} — even axis-aligned rays gain little. Not worth storing."
        );
    }
}

/// For every packed node, enumerate all 27×8 = 216
/// (entry_cell, step_octant) configurations. For each, compute the
/// conservative path-mask (tensor product of per-axis 3-bit masks)
/// and test whether `occupancy & path_mask == 0`. Report the
/// fraction of configurations where the cull would fire.
///
/// This is independent of the DF analysis — it measures
/// wasted-descent avoidance, not empty-cell skipping.
fn analyze_path_mask_cull(tree: &[u32], node_offsets: &[u32]) {
    // (node, entry_cell, octant) triples examined.
    let mut total_descents_sim: u64 = 0;
    let mut culled: u64 = 0;
    // Cull rate by node popcount bucket.
    let mut culled_by_popcount = [0u64; 28];
    let mut total_by_popcount = [0u64; 28];

    // Precompute the 216 reachable masks (entry_cell × octant → 27-bit).
    let mut reachable: [[u32; 8]; 27] = [[0; 8]; 27];
    for ec in 0..27u32 {
        let (ex, ey, ez) = (ec % 3, (ec / 3) % 3, ec / 9);
        for oct in 0..8u32 {
            let sx = if oct & 1 == 0 { 1i32 } else { -1 };
            let sy = if oct & 2 == 0 { 1i32 } else { -1 };
            let sz = if oct & 4 == 0 { 1i32 } else { -1 };
            // Axis-reachable masks (3 bits each): cells reachable from
            // entry_cell along the axis given the step direction.
            let xm = axis_mask(ex as i32, sx);
            let ym = axis_mask(ey as i32, sy);
            let zm = axis_mask(ez as i32, sz);
            // Tensor product into 27-bit mask.
            let mut mask = 0u32;
            for x in 0..3u32 {
                if (xm >> x) & 1 == 0 {
                    continue;
                }
                for y in 0..3u32 {
                    if (ym >> y) & 1 == 0 {
                        continue;
                    }
                    for z in 0..3u32 {
                        if (zm >> z) & 1 == 0 {
                            continue;
                        }
                        mask |= 1u32 << (x + 3 * y + 9 * z);
                    }
                }
            }
            reachable[ec as usize][oct as usize] = mask;
        }
    }

    for &off in node_offsets {
        let occ = tree[off as usize];
        let pop = occ.count_ones() as usize;
        for ec in 0..27u32 {
            for oct in 0..8u32 {
                let mask = reachable[ec as usize][oct as usize];
                total_descents_sim += 1;
                total_by_popcount[pop] += 1;
                if (occ & mask) == 0 {
                    culled += 1;
                    culled_by_popcount[pop] += 1;
                }
            }
        }
    }

    let cull_rate = 100.0 * culled as f64 / total_descents_sim.max(1) as f64;
    println!();
    println!("─────────── Path-mask cull rate ───────────");
    println!("For each (node, entry_cell, ray_octant) configuration, would the");
    println!("conservative tensor-product reachable mask prove no occupied cell is");
    println!("on the ray's path through this node? If yes, the descent is skippable.");
    println!();
    println!("  Total configurations: {total_descents_sim}");
    println!("  Culled:               {culled} ({cull_rate:.1}%)");
    println!();
    println!("  Cull rate by node popcount (how occupancy density affects cull):");
    for pc in 0..=27 {
        if total_by_popcount[pc] == 0 {
            continue;
        }
        let rate = 100.0 * culled_by_popcount[pc] as f64 / total_by_popcount[pc] as f64;
        println!(
            "    pop={pc:>2}: {:>10} configs, cull rate {rate:>5.1}%",
            total_by_popcount[pc],
        );
    }

    if cull_rate > 20.0 {
        println!();
        println!(
            "  → {cull_rate:.0}% of descents would be culled. Each cull saves the descent,"
        );
        println!("    its inner empty-advances, and the eventual pop. If each culled");
        println!("    descent eliminates ~4 DDA steps, the aggregate savings is");
        println!("    ~{:.1} steps saved per average descent decision.", 4.0 * cull_rate / 100.0);
    } else if cull_rate > 5.0 {
        println!();
        println!(
            "  → {cull_rate:.0}% cull rate is modest. Worth implementing if the per-check"
        );
        println!("    cost is kept ≤ 5 shader ops (tensor product + 1 tree read + AND)."
        );
    } else {
        println!();
        println!("  → Cull rate too low to justify the per-descent check overhead.");
    }
}

fn axis_mask(entry: i32, step: i32) -> u32 {
    // cells reachable along this axis from `entry` moving in `step`
    // direction, as a 3-bit mask on bits {0, 1, 2}.
    let mut m = 0u32;
    let mut x = entry;
    while (0..3).contains(&x) {
        m |= 1 << x;
        x += step;
    }
    m
}

/// Combines per-axis DF savings + path-mask cull savings into a
/// rough projection of total DDA-step reduction for a ray. This is
/// a BACK-OF-ENVELOPE model, not a full simulation; it assumes
/// independent savings on the two cost buckets we measured at the
/// Jerusalem-nucleus case (avg_empty=61, avg_descend=29, pop=28).
fn analyze_combined_projection(tree: &[u32], node_offsets: &[u32]) {
    // Recompute best-axis mean and cull rate briefly.
    let mut axis_sum = [0u64; 6];
    let mut axis_count = [0u64; 6];
    let dirs: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ];
    for &off in node_offsets {
        let occ = tree[off as usize];
        for slot in 0..27u32 {
            if (occ >> slot) & 1 != 0 {
                continue;
            }
            let (cx, cy, cz) = (slot as i32 % 3, (slot as i32 / 3) % 3, slot as i32 / 9);
            for (a, &(dx, dy, dz)) in dirs.iter().enumerate() {
                let (mut x, mut y, mut z) = (cx, cy, cz);
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
                axis_sum[a] += len as u64;
                axis_count[a] += 1;
            }
        }
    }
    let best_axis = (0..6)
        .map(|a| axis_sum[a] as f64 / axis_count[a].max(1) as f64)
        .fold(0.0_f64, f64::max);

    // Path-mask cull rate (average over all node configurations).
    let mut total = 0u64;
    let mut culled = 0u64;
    let mut reachable: [[u32; 8]; 27] = [[0; 8]; 27];
    for ec in 0..27u32 {
        let (ex, ey, ez) = (ec % 3, (ec / 3) % 3, ec / 9);
        for oct in 0..8u32 {
            let sx = if oct & 1 == 0 { 1 } else { -1 };
            let sy = if oct & 2 == 0 { 1 } else { -1 };
            let sz = if oct & 4 == 0 { 1 } else { -1 };
            let (xm, ym, zm) = (
                axis_mask(ex as i32, sx),
                axis_mask(ey as i32, sy),
                axis_mask(ez as i32, sz),
            );
            let mut mask = 0u32;
            for x in 0..3u32 {
                if (xm >> x) & 1 == 0 {
                    continue;
                }
                for y in 0..3u32 {
                    if (ym >> y) & 1 == 0 {
                        continue;
                    }
                    for z in 0..3u32 {
                        if (zm >> z) & 1 == 0 {
                            continue;
                        }
                        mask |= 1u32 << (x + 3 * y + 9 * z);
                    }
                }
            }
            reachable[ec as usize][oct as usize] = mask;
        }
    }
    for &off in node_offsets {
        let occ = tree[off as usize];
        for ec in 0..27u32 {
            for oct in 0..8u32 {
                total += 1;
                if (occ & reachable[ec as usize][oct as usize]) == 0 {
                    culled += 1;
                }
            }
        }
    }
    let cull_rate = culled as f64 / total.max(1) as f64;

    // Reference: Jerusalem-nucleus spawn observed stats.
    const BASELINE_EMPTY: f64 = 61.0;
    const BASELINE_DESCEND: f64 = 29.0;
    const BASELINE_POP: f64 = 28.0;
    const BASELINE_TOTAL: f64 = 137.0;

    let empty_after_df = BASELINE_EMPTY * (1.0 / best_axis.max(1.0));
    let culled_descends = BASELINE_DESCEND * cull_rate;
    let descend_after = BASELINE_DESCEND - culled_descends;
    // Each culled descent also removes its eventual pop + ~2 empty advances inside.
    let pop_after = BASELINE_POP - culled_descends;
    let empty_from_culled_descends = culled_descends * 2.0;
    let empty_final = (empty_after_df - empty_from_culled_descends).max(0.0);

    let after_total = empty_final + descend_after + pop_after;

    println!();
    println!("─────────── Combined projection ───────────");
    println!("Using Jerusalem-nucleus baseline: avg_empty=61, avg_descend=29, avg_oob=28,");
    println!("total=137 steps/ray. Ignores compound effects (descent-cull reduces its");
    println!("empty-inner-work too). Treat as rough floor, not ceiling.");
    println!();
    println!("  Per-axis DF (best axis = {best_axis:.2}):");
    println!("    empty: 61 → {empty_after_df:.1} steps (axis-dominant ray skips {best_axis:.1}x)");
    println!();
    println!("  Path-mask cull (rate = {:.1}%):", cull_rate * 100.0);
    println!(
        "    descends: 29 → {descend_after:.1} ({culled_descends:.1} culled)"
    );
    println!("    pops: 28 → {pop_after:.1} (linked 1:1 with culled descents)");
    println!("    additional empty saved: ~{empty_from_culled_descends:.1} (inner-advances that don't happen)");
    println!();
    println!("  Combined projection:");
    println!(
        "    empty ({:.0}) + descend ({:.0}) + pop ({:.0}) + other ~19 = {:.0} steps/ray",
        empty_final, descend_after, pop_after,
        empty_final + descend_after + pop_after + 19.0,
    );
    let proj_total = empty_final + descend_after + pop_after + 19.0;
    println!(
        "    vs baseline 137 = {:.1}x speedup",
        BASELINE_TOTAL / proj_total.max(1.0)
    );
    println!();
    let _ = tree;
    let _ = node_offsets;
}
