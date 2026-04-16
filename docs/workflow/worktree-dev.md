# Worktree + dev loop workflow

Concise, verified steps for spinning up an isolated branch in a worktree and launching the dev build.

## 1. Create the worktree

Branch off whatever commit you want — another worktree's HEAD, a specific hash, a tag, `main`, etc. Always pass an explicit starting commit; don't rely on "current branch."

```bash
# From any repo/worktree:
git -C <source-repo-or-worktree> worktree add \
    -b <new-branch-name> \
    /Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/<name> \
    <commit-ish>
```

Example — branch `attempt1` off commit `a66fa8b` from a sibling worktree:

```bash
git -C ~/GitHub/deepspace-game-refactor-position \
    worktree add -b attempt1 \
    /Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/attempt1 \
    a66fa8b
```

All new worktrees live under `.claude/worktrees/` in the main repo.

## 2. First-time setup in the worktree

Node modules don't carry over from the source worktree. Install them before the first dev run:

```bash
cd /Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/<name>/ui
npm install
```

One-shot, only needed the first time (or after `package.json` changes).

## 3. Launch the dev loop

`scripts/dev.sh` must run from the worktree root (not from `ui/`, not from the main repo):

```bash
cd /Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/<name> && ./scripts/dev.sh
```

One-liner to copy-paste. Combine `cd` and `./scripts/dev.sh` with `&&` so the
subshell can't accidentally leak the wrong cwd into a later command.

**Running a worktree someone else (e.g. Claude) set up**: same command —
just substitute `<name>` for the worktree directory name. No extra setup
needed if step 2 (`npm install` in `ui/`) has already been done in that
worktree. Run from your shell, not the assistant's — the native game
opens a window, so it needs your session's display.

The script:
1. Starts Vite on :5173 in the background (`cd ui && npx vite`).
2. Waits for Vite to respond.
3. Runs `cargo run` in the worktree, which launches the native game with a wry WebView overlay pointed at Vite.

It cleans up both on exit via `trap`.

## 4. Monitoring the dev output

Launch in the background and poll the log file every ~3s — do **not** rely on a grep-filtered monitor (it can stay silent through real errors):

```
Bash(run_in_background=true): cd <worktree> && ./scripts/dev.sh
Monitor: while true; do sleep 3; echo "---$(date +%H:%M:%S)---"; \
         tail -n 8 <output-file> | sed 's/\x1b\[[0-9;]*m//g'; done
```

Success signals:
- `Vite ready`
- `Finished \`dev\` profile ... target(s)`
- `Running \`target/debug/deepspace-game\``
- `Generated empty space tree: ...` (game started successfully)

Failure signals — watch for these in the tail:
- `error:` / `Error` / `panicked` (Rust)
- `ERR_MODULE_NOT_FOUND` / `UNRESOLVED_IMPORT` (missing `ui/node_modules` — go back to step 2)
- `no such file or directory: ./scripts/dev.sh` (wrong cwd — go back to step 3)

## 5. Tearing down

`trap cleanup EXIT` kills `deepspace-game` and Vite when `dev.sh` exits. To kill the dev loop yourself, stop the backgrounded Bash task (or the monitor, which cascades if the script is attached to the terminal).

To remove the worktree entirely:

```bash
git worktree remove /Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/<name>
git branch -D <branch-name>   # only if you also want to drop the branch
```
