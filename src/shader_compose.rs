//! Minimal `#include`-style WGSL composer.
//!
//! Resolves `#include "foo.wgsl"` directives against a compile-time
//! registry of `include_str!`'d shader sources (so it works on WASM
//! without filesystem access). Guards against cycles and dedupes
//! repeated includes so diamond-dependency graphs just work.
//!
//! Syntax: a line whose first non-whitespace content is
//! `#include "name.wgsl"` is replaced by the contents of that file
//! (recursively composed). Anything else on the line is ignored, so
//! trailing comments are NOT supported — keep include lines clean.

use std::collections::HashSet;

const SOURCES: &[(&str, &str)] = &[
    ("bindings.wgsl",  include_str!("../assets/shaders/bindings.wgsl")),
    ("tree.wgsl",      include_str!("../assets/shaders/tree.wgsl")),
    ("ray_prim.wgsl",  include_str!("../assets/shaders/ray_prim.wgsl")),
    ("sphere_debug.wgsl", include_str!("../assets/shaders/sphere_debug.wgsl")),
    ("sphere.wgsl",    include_str!("../assets/shaders/sphere.wgsl")),
    ("march.wgsl",     include_str!("../assets/shaders/march.wgsl")),
    ("main.wgsl",      include_str!("../assets/shaders/main.wgsl")),
    ("taa_resolve.wgsl", include_str!("../assets/shaders/taa_resolve.wgsl")),
    ("heightmap_gen.wgsl", include_str!("../assets/shaders/heightmap_gen.wgsl")),
    ("entity_heightmap_clamp.wgsl", include_str!("../assets/shaders/entity_heightmap_clamp.wgsl")),
    ("entity_raster.wgsl", include_str!("../assets/shaders/entity_raster.wgsl")),
];

fn lookup(name: &str) -> &'static str {
    SOURCES
        .iter()
        .find_map(|(n, s)| if *n == name { Some(*s) } else { None })
        .unwrap_or_else(|| panic!("shader include not registered: {name:?}"))
}

pub fn compose(entry: &str) -> String {
    let mut out = String::new();
    let mut included: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = Vec::new();
    include_recursive(entry, &mut out, &mut included, &mut stack);
    out
}

fn include_recursive(
    name: &str,
    out: &mut String,
    included: &mut HashSet<String>,
    stack: &mut Vec<String>,
) {
    if stack.iter().any(|n| n == name) {
        panic!("shader include cycle: {stack:?} -> {name:?}");
    }
    if !included.insert(name.to_string()) {
        return;
    }
    stack.push(name.to_string());

    let src = lookup(name);
    for line in src.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("#include") {
            let quoted = rest.trim();
            let inner = quoted
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .unwrap_or_else(|| {
                    panic!("malformed #include in {name}: {line:?}")
                });
            include_recursive(inner, out, included, stack);
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    stack.pop();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composes_entry_point() {
        let src = compose("main.wgsl");
        assert!(src.contains("@vertex"), "vs_main missing");
        assert!(src.contains("@fragment"), "fs_main missing");
        assert!(src.contains("fn march("), "march missing");
        assert!(src.contains("fn march_cartesian("), "march_cartesian missing");
        assert!(src.contains("struct Uniforms"), "Uniforms missing");
        assert!(!src.contains("#include"), "directive leaked into output");
    }

    #[test]
    fn dedupes_diamond_imports() {
        let src = compose("main.wgsl");
        let binding_count = src.matches("@group(0) @binding(0)").count();
        assert_eq!(binding_count, 1, "bindings.wgsl included more than once");
    }

    /// Parse the composed `main.wgsl` through naga and validate it.
    /// Catches WGSL-level errors (unknown identifiers, type
    /// mismatches, bad struct layouts) that would otherwise only
    /// surface at runtime when the shader module is created.
    #[test]
    fn naga_validates_main_shader() {
        let src = compose("main.wgsl");
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("wgsl parse error:\n{}", e.emit_to_string(&src)));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("wgsl validation error:\n{:?}", e));
        let _ = info;
    }

    /// The Rust-side `GpuUniforms` must match the WGSL-side `Uniforms`
    /// struct in both size and field order (field-by-field alignment
    /// is std140-adjacent via the explicit `[f32; 4]` / `[u32; 4]`
    /// fields). The composer-level check is rough: verify each named
    /// field appears in the WGSL source in the same order as the
    /// Rust struct. A true byte-level check would require a naga
    /// struct-layout walk — rough is good enough to catch additions
    /// that forget to mirror on one side.
    #[test]
    fn uniforms_field_order_matches_wgsl() {
        let src = compose("main.wgsl");
        let struct_body = {
            let start = src.find("struct Uniforms").expect("Uniforms struct missing");
            let tail = &src[start..];
            let brace = tail.find('{').unwrap();
            let end_rel = tail[brace..].find('}').unwrap();
            tail[brace..brace + end_rel].to_string()
        };
        let expected_order: &[&str] = &[
            "root_index",
            "node_count",
            "screen_width",
            "screen_height",
            "max_depth",
            "highlight_active",
            "root_kind",
            "ribbon_count",
            "entity_count",
            "highlight_min",
            "highlight_max",
            "root_radii",
            "root_face_meta",
            "root_face_bounds",
            "root_face_pop_pos",
            "sphere_debug_mode",
            "probe_pixel",
        ];
        let mut cursor = 0;
        for name in expected_order {
            let Some(idx) = struct_body[cursor..].find(name) else {
                panic!("WGSL Uniforms missing field `{name}` after position {cursor}");
            };
            cursor += idx + name.len();
        }
    }
}
