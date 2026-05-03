//! Minecraft block-name → RGBA lookup.
//!
//! Values are derived from Minecraft's canonical map-color palette
//! (the colors used on cartographer maps), which gives every block a
//! representative RGB. Alpha is 255 for solids, 0 for air/void.
//! Unknown blocks fall through to the magenta sentinel so they're
//! visually obvious in-game.
//!
//! This is v1 — a small, curated LUT covering the ~40 blocks that
//! appear in most vanilla builds. Full-fidelity (all ~800 vanilla
//! blocks, biome tinting, textures averaged from a resource pack) is
//! deliberately deferred to v2.

/// Opaque RGBA. Alpha 0 means "treat as air — do not emit a voxel".
pub type Rgba = [u8; 4];

/// Sentinel for any air-like block. The caller must skip emitting.
pub const AIR: Rgba = [0, 0, 0, 0];

/// Emitted for blocks not in the LUT. Bright magenta so missing
/// mappings are obvious in the rendered output.
pub const UNKNOWN: Rgba = [255, 0, 255, 255];

/// Strip the `minecraft:` namespace if present, for the LUT lookup.
fn bare_name(full: &str) -> &str {
    full.strip_prefix("minecraft:").unwrap_or(full)
}

/// Resolve a Minecraft block id (e.g. `minecraft:oak_planks`) to
/// RGBA. Returns [`AIR`] for air/void variants and [`UNKNOWN`] when
/// no entry matches.
pub fn block_rgba(name: &str) -> Rgba {
    let b = bare_name(name);
    match b {
        // Air variants — skip entirely.
        "air" | "cave_air" | "void_air" => AIR,

        // Core terrain.
        "stone" | "cobblestone" | "stone_bricks" | "mossy_cobblestone"
        | "mossy_stone_bricks" | "cracked_stone_bricks" | "smooth_stone"
        | "andesite" | "polished_andesite" => [112, 112, 112, 255],
        "granite" | "polished_granite" => [149, 103, 85, 255],
        "diorite" | "polished_diorite" => [188, 188, 188, 255],
        "deepslate" | "cobbled_deepslate" | "polished_deepslate"
        | "deepslate_bricks" | "deepslate_tiles" => [70, 70, 76, 255],
        "bedrock" => [60, 60, 60, 255],
        "dirt" | "coarse_dirt" | "rooted_dirt" | "podzol" | "mycelium"
        | "dirt_path" => [134, 96, 67, 255],
        "grass_block" => [127, 178, 56, 255],
        "sand" | "sandstone" | "smooth_sandstone" | "chiseled_sandstone"
        | "cut_sandstone" => [247, 233, 163, 255],
        "red_sand" | "red_sandstone" | "smooth_red_sandstone"
        | "chiseled_red_sandstone" | "cut_red_sandstone" => [188, 98, 49, 255],
        "gravel" => [136, 127, 118, 255],
        "clay" => [160, 166, 181, 255],
        "snow" | "snow_block" | "powder_snow" => [255, 255, 255, 255],
        "ice" | "packed_ice" | "blue_ice" => [160, 207, 255, 255],

        // Liquids — give water slight transparency so it reads as liquid.
        "water" => [64, 64, 255, 200],
        "lava" => [255, 102, 0, 255],

        // Woods — planks/logs/leaves share a color family per species.
        "oak_log" | "oak_wood" | "stripped_oak_log" | "stripped_oak_wood"
        | "oak_planks" => [143, 119, 72, 255],
        "birch_log" | "birch_wood" | "stripped_birch_log"
        | "stripped_birch_wood" | "birch_planks" => [216, 200, 137, 255],
        "spruce_log" | "spruce_wood" | "stripped_spruce_log"
        | "stripped_spruce_wood" | "spruce_planks" => [104, 75, 42, 255],
        "jungle_log" | "jungle_wood" | "stripped_jungle_log"
        | "stripped_jungle_wood" | "jungle_planks" => [156, 122, 73, 255],
        "acacia_log" | "acacia_wood" | "stripped_acacia_log"
        | "stripped_acacia_wood" | "acacia_planks" => [169, 88, 33, 255],
        "dark_oak_log" | "dark_oak_wood" | "stripped_dark_oak_log"
        | "stripped_dark_oak_wood" | "dark_oak_planks" => [76, 50, 35, 255],

        "oak_leaves" | "jungle_leaves" | "dark_oak_leaves" => [60, 120, 40, 255],
        "birch_leaves" => [128, 167, 85, 255],
        "spruce_leaves" => [80, 110, 80, 255],
        "acacia_leaves" => [110, 130, 55, 255],

        // Masonry & metal.
        "bricks" => [155, 80, 65, 255],
        "iron_block" => [220, 220, 220, 255],
        "gold_block" => [247, 211, 61, 255],
        "diamond_block" => [92, 219, 213, 255],
        "emerald_block" => [77, 180, 94, 255],
        "redstone_block" => [175, 24, 24, 255],
        "lapis_block" => [38, 53, 146, 255],
        "netherite_block" => [68, 58, 58, 255],
        "copper_block" => [192, 122, 87, 255],
        "obsidian" => [20, 16, 30, 255],
        "glowstone" => [255, 204, 100, 255],
        "sea_lantern" => [200, 230, 240, 255],

        // Nether / end.
        "netherrack" => [125, 53, 49, 255],
        "soul_sand" | "soul_soil" => [70, 48, 35, 255],
        "end_stone" | "end_stone_bricks" => [223, 221, 163, 255],
        "purpur_block" | "purpur_pillar" => [163, 118, 163, 255],

        // Glass (with a hint of alpha to read as a window).
        "glass" => [230, 240, 255, 180],
        "tinted_glass" => [40, 40, 50, 220],

        // Wool / concrete / terracotta — 16-color Minecraft palette.
        // These share the canonical dye RGBs (used by wool, concrete,
        // and terracotta with brightness-modifier variants).
        b if has_color_prefix(b, "white") => [240, 240, 240, 255],
        b if has_color_prefix(b, "orange") => [235, 130, 45, 255],
        b if has_color_prefix(b, "magenta") => [190, 70, 180, 255],
        b if has_color_prefix(b, "light_blue") => [100, 170, 215, 255],
        b if has_color_prefix(b, "yellow") => [230, 200, 30, 255],
        b if has_color_prefix(b, "lime") => [120, 200, 30, 255],
        b if has_color_prefix(b, "pink") => [230, 140, 170, 255],
        b if has_color_prefix(b, "gray") => [70, 70, 70, 255],
        b if has_color_prefix(b, "light_gray") => [160, 160, 160, 255],
        b if has_color_prefix(b, "cyan") => [40, 130, 160, 255],
        b if has_color_prefix(b, "purple") => [130, 55, 170, 255],
        b if has_color_prefix(b, "blue") => [50, 65, 175, 255],
        b if has_color_prefix(b, "brown") => [95, 60, 35, 255],
        b if has_color_prefix(b, "green") => [65, 100, 35, 255],
        b if has_color_prefix(b, "red") => [175, 40, 40, 255],
        b if has_color_prefix(b, "black") => [20, 20, 25, 255],

        _ => UNKNOWN,
    }
}

/// True if `b` starts with the color prefix AND ends in a known
/// dye-colored suffix (wool / concrete / terracotta / stained_glass
/// etc.). Keeps the wildcard from accidentally matching e.g.
/// `red_sand` (which has its own sand mapping above).
fn has_color_prefix(b: &str, color: &str) -> bool {
    if !b.starts_with(color) { return false; }
    let rest = &b[color.len()..];
    matches!(
        rest,
        "_wool"
            | "_concrete"
            | "_concrete_powder"
            | "_terracotta"
            | "_glazed_terracotta"
            | "_stained_glass"
            | "_stained_glass_pane"
            | "_carpet"
            | "_bed"
            | "_candle"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_is_transparent() {
        assert_eq!(block_rgba("minecraft:air")[3], 0);
        assert_eq!(block_rgba("minecraft:cave_air")[3], 0);
    }

    #[test]
    fn stone_has_canonical_color() {
        assert_eq!(block_rgba("minecraft:stone"), [112, 112, 112, 255]);
    }

    #[test]
    fn grass_is_green() {
        let [r, g, b, a] = block_rgba("minecraft:grass_block");
        assert!(g > r && g > b && a == 255);
    }

    #[test]
    fn wool_colors_resolve() {
        assert_eq!(block_rgba("minecraft:red_wool"), [175, 40, 40, 255]);
        assert_eq!(block_rgba("minecraft:black_concrete"), [20, 20, 25, 255]);
    }

    #[test]
    fn red_sand_not_captured_by_red_prefix() {
        // The generic `red` color match would be wrong; the explicit
        // red_sand rule must win.
        assert_eq!(block_rgba("minecraft:red_sand"), [188, 98, 49, 255]);
    }

    #[test]
    fn unknown_falls_through_to_magenta() {
        assert_eq!(block_rgba("minecraft:totally_made_up"), UNKNOWN);
    }

    #[test]
    fn bare_names_work_without_namespace() {
        assert_eq!(block_rgba("stone"), [112, 112, 112, 255]);
    }
}
