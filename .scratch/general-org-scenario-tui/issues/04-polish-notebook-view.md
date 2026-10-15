# 04: Make editing and text results feel like a notebook

**What to build:** The focused-cell TUI is comfortable to edit and read: source, context, output, process provenance, and controls remain clear while the learner moves through a real Org notebook.

**Blocked by:** 03: Edit arbitrary cells without changing execution order

**Status:** resolved

- [x] The editor shows heading context, Python syntax, and line numbers, and retains caret, selection, undo history, and scroll position across log and snapshot updates when source is unchanged.
- [x] Results visibly separate stdout, displayed text values, tracebacks, empty output, and unsupported MIME labels; long Unicode text and tracebacks scroll cleanly.
- [x] The quiet blue layout keeps the active cell, navigator, Emacs PID, kernel ID, and both live process log panes legible at 80×24 and gives results and logs more room at 120×36.
- [x] Keyboard focus moves predictably among navigator, editor, results, logs, and controls; resizing does not leave previously wrapped log or result lines clipped.
- [x] The current server token is redacted from logs, bridge errors, and displayed cell text; TUI tests cover the visual and editing behaviors.

## Comments

Implemented the focused Textual view with structured result sections, reflowing result and process logs, preserved editor state, and token redaction across displayed text. Tests exercise 80×24 and 120×36 layouts, Tab order, undo, editing, result types, and resize. Live terminal and GPU verification remain with the integration qualification ticket because port 8899 is an unowned service in this worktree.
