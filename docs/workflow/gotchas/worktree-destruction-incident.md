# Worktree destruction incident (2026-04-16)

## What happened

Claude was asked to "create a new worktree off of head for this branch" for
the path `.claude/worktrees/deep-layers-asymmetry-fix`. The branch
`deep-layers-asymmetry-fix` already existed with months of work — the entire
anchor refactor, sphere rendering, harness debugging, local shell frame
rewrite, and more (commits `f85061f` through `8e607ba`, 50+ commits).

Claude:

1. Ran `git worktree remove --force` on the existing worktree.
2. Ran `git branch -D deep-layers-asymmetry-fix`, deleting the local branch.
3. Ran `git worktree add -b deep-layers-asymmetry-fix ... ray-march-engine-2`,
   creating a **new** branch based on `ray-march-engine-2` HEAD (`eb6ff6a`)
   instead of the existing deep-layers work (`8e607ba`).
4. Wrote 27 tests against the wrong codebase (the pre-refactor code that
   doesn't have any of the deep-layers changes).
5. Force-pushed the new branch to origin, overwriting the remote with the
   wrong base.

The remote was restored to `8e607ba` after the mistake was caught.

## Why it happened

Claude misinterpreted "off of head" as meaning the HEAD of the **current
branch** (`ray-march-engine-2`) rather than the HEAD of the **target branch**
(`deep-layers-asymmetry-fix`). It then took three destructive actions in
sequence without confirming any of them:

- `git worktree remove --force` — destroyed the local working copy
- `git branch -D` — deleted the local branch reference
- `git push --force` — overwrote the remote backup

Each of these is individually dangerous. Together they created a scenario
where the only surviving copy of 50+ commits was the reflog.

## Root causes

1. **Misunderstanding the instruction.** "Off of head" in context meant
   "based on the current state of deep-layers-asymmetry-fix." Claude read
   it as "based on HEAD of the current checkout (ray-march-engine-2)."

2. **No confirmation before destructive operations.** Claude's own
   instructions say to confirm before hard-to-reverse actions. It did not
   confirm before removing the worktree, deleting the branch, or
   force-pushing.

3. **Not verifying the result.** After creating the worktree, Claude saw
   `8e607ba` in `git log` (because the remote tracking branch pulled it
   in) but didn't investigate why it didn't match the expected `eb6ff6a`.
   It then ran `git reset --hard eb6ff6a`, actively destroying the
   correct state.

4. **Force-push without thinking.** The force-push was the final nail —
   it overwrote the remote backup of the branch. Claude treated it as a
   routine "sync to remote" instead of recognizing it was overwriting
   someone else's work with a completely different branch base.

## Rules to prevent this

- **Never `git worktree remove --force` an existing worktree without
  explicit user confirmation.** The worktree may contain uncommitted or
  unpushed work.
- **Never `git branch -D` a branch that has a remote tracking branch
  with divergent history.** Check `git log origin/branch..branch` first.
- **Never `git push --force` without explicit user confirmation,
  especially when the local branch was just recreated.**
- **When asked to create a worktree for an existing branch, check out
  that branch — don't delete and recreate it from a different base.**
- **If a worktree already exists at the requested path, ask what to do
  instead of destroying it.**

## What was lost

Nothing permanently — the remote was restored from reflog. But the local
worktree state (any uncommitted changes, local-only config, build cache)
was destroyed, and significant time was wasted writing tests against the
wrong codebase.

## The useless tests

The 27 "interaction tests" that were written test the `ray-march-engine-2`
code (pre-refactor editing, pre-anchor camera, pre-local-frame rendering).
They do not test the deep-layers refactored code at all. They should not
be carried forward.
