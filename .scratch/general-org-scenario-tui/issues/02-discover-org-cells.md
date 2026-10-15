# 02: Discover and run real Org notebook cells

**What to build:** A learner can work through any number of Python blocks in document order. The navigator and focused-cell view use each block's Org heading and prose, and all cells share the same kernel state.

**Blocked by:** 01: Run one chosen Org cell through Emacs and Jupyter

**Status:** resolved

- [x] Named and unnamed Python blocks, multiple blocks under one heading, and blocks under nested headings become distinct, stable cell steps in source order.
- [x] The navigator displays the dynamic step count, heading context, and pending/completed states; browsing an earlier result never executes it again.
- [x] A file with no Python blocks fails clearly; non-Python source blocks are left untouched and their skipped count is visible before execution.
- [x] Python blocks with execution-changing Org Babel headers fail with a heading and line number, while ordinary prose, tables, links, and display-oriented result headers do not prevent a run.
- [x] Tests prove that two or more cells share one kernel ID and that each F10 advances exactly one step.

## Comments

Org discovery now supplies the nearest prose and parsed heading path for each cell. Dynamic step labels show heading context. ERT covers two ordered cells using one client and kernel ID, and Textual tests cover dynamic counts, F10, browsing, and the focused context view.
