# Rewrite rule

Non-trivial changes are landed as **full file rewrites**, not stacks
of surgical edits. If the file is a monolith, split it into smaller
responsibility-owned files as part of the rewrite.

## When this applies

- Enum variant renames (every match arm gets rewritten).
- Changing a struct's fields or a function's signature (every caller
  gets rewritten).
- Adding a new responsibility to a file that already mixes several.
- Any change that would leave the file half-migrated if done Edit by
  Edit.

Narrow typo or single-bug fixes can still use `Edit`.

## Procedure

1. **Read fully.** Understand the actual responsibilities in the
   file.
2. **Decide on splits.** If the file carries multiple concerns, split
   it into smaller files with clear ownership. Don't keep a
   monolith around "for compatibility."
3. **Rewrite from scratch** around the new model. Don't port Edit by
   Edit.
4. **Sweep call-sites.** Every file that referenced the old shape
   gets rewritten too. Compile-green at the end of the sweep, not
   at each individual file.

## Why

Edit-patches through a broken or in-flux architecture compound: each
patch makes the next harder, leaves half-renamed variants, partial
migrations, dead code. Clean rewrites force confronting the real
architecture question; patches let you avoid it.

When in doubt, rewrite.
