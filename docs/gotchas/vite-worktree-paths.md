# Vite serves main repo files, not worktree files

**Symptom**: You edit React/CSS files in a git worktree
(`.claude/worktrees/*/ui/src/...`), but the game's WebView shows
the old version. Changes don't appear no matter how many times you
restart Vite.

**Cause**: Vite resolves module paths through the real filesystem.
Even though the worktree has its own copy of `ui/src/`, Vite's
module graph traces imports back to the main repo's `ui/` directory.
The worktree's `node_modules/.vite` cache may also reference the
main repo paths.

**Fix**: Copy changed UI files to the main repo too:

```sh
# From the worktree directory:
cp ui/src/components/MyComponent.tsx \
   /path/to/main/repo/ui/src/components/MyComponent.tsx
```

Or edit the main repo's UI files directly — the worktree only
matters for Rust source (which `cargo` builds from the worktree's
`src/`).

**Prevention**: When making UI changes in a worktree, always update
both the worktree and main repo copies of the file. Or avoid UI
changes in worktrees altogether — do them on the main branch.
