# 03: Hints, new attempts, and export

**What to build:** A learner can request help, start a clean attempt, and take their work away without losing the history of what they checked.

**Blocked by:** 02: Preserve checked source

**Status:** resolved

- [x] Progressive hints and an explicit solution reveal can be expanded and collapsed; revealing the solution never edits answer cells.
- [x] Hint and solution reveals remain recorded as assistance events, and results distinguish assisted passes from later independent work.
- [x] New attempt creates a fresh editable notebook while preserving prior attempts and notes.
- [x] Export and reimport of one attempt or all local history preserve notebooks, derived Python source, notes, assistance, receipts, and exact checked-source hashes.

## Comments

Resolved with progressive catalog hints, a reference reveal that never mutates answer cells, assistance snapshots on receipts, Contents-backed attempt notebook copies, non-destructive import ID remapping, active attempt restoration, and single/all-attempt JSON archives with receipt source-hash validation. Verified with focused unit/type checks, browser acceptance, the full browser_gym target, and the repository Python suite.
