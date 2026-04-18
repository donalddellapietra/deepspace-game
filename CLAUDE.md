The command for running is scripts/dev.sh (native, with wry overlay).
For the browser/WASM build, use scripts/dev-wasm.sh — it builds the
React UI bundle (one-time) and trunk-serves at http://localhost:8080.
Requires Chrome / Edge or any WebGPU-enabled browser.

When you create a worktree, always check which is our current branch, and create off of that.

Use state of the art rust game development practice.

Never assume something worked. After running any command (build, dev server, test), always read the actual output and verify success before telling the user it's working. Don't say "it's running" based on the command being launched — wait for and check the output.

If you are on a worktree, always commit and push your changes unless they are broken. If you are on my branch, always ask me first.
