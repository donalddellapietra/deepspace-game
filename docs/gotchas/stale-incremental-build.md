# Gotchas

## Stale incremental build artifacts (linker error)

**Symptom**: `ld: symbol(s) not found for architecture arm64` with
references to old symbols that no longer exist in the code.

**Cause**: Rust's incremental compilation caches object files. When
code changes significantly (e.g., switching branches, removing large
chunks of code), stale `.o` files reference symbols that no longer
exist. The linker sees the old objects and fails.

**Fix**: Clear only the incremental cache, not the entire target:

```sh
rm -rf target/debug/incremental/deepspace_game-*
cargo build
```

This rebuilds in ~5 seconds instead of ~4 minutes (`cargo clean`
rebuilds all dependencies from scratch).

**Prevention**: This happens most often when switching between
branches with different code (e.g., `main` ↔ `gpu-instancing`).
If you see the linker error, clear the incremental cache first.
