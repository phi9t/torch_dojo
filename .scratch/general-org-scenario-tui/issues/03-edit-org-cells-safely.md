# 03: Edit arbitrary cells without changing execution order

**What to build:** A learner can edit any discovered Python cell in the focused TUI and trust that Emacs saves the correct source block in the run copy before execution or navigation. Restart replays edited cells from the beginning with a fresh kernel.

**Blocked by:** 02: Discover and run real Org notebook cells

**Status:** resolved

- [x] Edits use a stable run-local cell identity, work for unnamed cells, and are saved before advance, navigation, restart, or quit without changing the original Org file.
- [x] The TUI marks unsaved edits immediately and marks an edited completed cell's prior result stale until restart; restart preserves code edits and clears generated results.
- [x] Source edits cannot inject Org block boundaries or switch the target block, and a structural edit outside the TUI stops execution with a clear recovery message.
- [x] A busy or timed-out request is never sent twice by rapid keys or reconnect logic; lost responses report an unknown outcome and possible owned kernel.
- [x] Tests exercise multiline Unicode source, completed-cell editing, restart, malformed and oversized edits, and cleanup after a failed cell.

## Comments

Resolved with source saves by cell ID, immediate dirty and stale markers, structure validation, and unknown-outcome retry guards. The bridge, scenario engine, Textual, and full Python suites pass in the ticket worktree.
