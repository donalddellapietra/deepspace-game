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
    ("face_math.wgsl", include_str!("../assets/shaders/face_math.wgsl")),
    ("ray_prim.wgsl",  include_str!("../assets/shaders/ray_prim.wgsl")),
    ("face_walk.wgsl", include_str!("../assets/shaders/face_walk.wgsl")),
    ("sphere.wgsl",    include_str!("../assets/shaders/sphere.wgsl")),
    ("sphere_remap.wgsl", include_str!("../assets/shaders/sphere_remap.wgsl")),
    ("sphere_trace.wgsl", include_str!("../assets/shaders/sphere_trace.wgsl")),
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
        assert!(src.contains("fn sphere_in_cell("), "sphere_in_cell missing");
        assert!(src.contains("fn sremap_march("), "sremap_march missing");
        assert!(src.contains("fn sremap_forward("), "sremap_forward missing");
        assert!(src.contains("fn sremap_inverse("), "sremap_inverse missing");
        assert!(src.contains("ROOT_KIND_REMAP_SPHERE"), "REMAP_SPHERE constant missing");
        assert!(src.contains("struct Uniforms"), "Uniforms missing");
        assert!(!src.contains("#include"), "directive leaked into output");
    }

    #[test]
    fn dedupes_diamond_imports() {
        let src = compose("main.wgsl");
        let binding_count = src.matches("@group(0) @binding(0)").count();
        assert_eq!(binding_count, 1, "bindings.wgsl included more than once");
    }

    /// Parse + validate the composed main.wgsl with naga. Catches WGSL
    /// syntax / type errors at `cargo test` time rather than at
    /// `create_shader_module` time on the GPU — crucial when adding
    /// shader code without running the renderer.
    #[test]
    fn main_wgsl_validates_with_naga() {
        let src = compose("main.wgsl");
        let module = match naga::front::wgsl::parse_str(&src) {
            Ok(m) => m,
            Err(e) => panic!("main.wgsl failed WGSL parse:\n{}", e.emit_to_string(&src)),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        if let Err(e) = validator.validate(&module) {
            panic!("main.wgsl failed naga validation: {:#?}", e);
        }
    }
}
