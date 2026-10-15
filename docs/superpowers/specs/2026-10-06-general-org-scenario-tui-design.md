# General Org scenario TUI

## Purpose

Make the scenario TUI feel like a focused, stepwise notebook for Python and PyTorch work. A learner chooses a repo-local Org file, edits its Python cells, runs them in document order in one persistent GPU kernel, and sees each result beside its source. The execution path remains **TUI → Emacs worker in the Emacs insula → emacs-jupyter → Jupyter kernel in the Jupyter insula**. The frontend never parses the Org file to decide what runs and never sends a Jupyter execution request.

The runner owns four lifecycle steps: **Setup, Connect, Final, Cleanup**. The selected Org file supplies the middle cells. The first version runs cells in document order; it does not add, delete, or arbitrarily rerun cells. Results are text: stdout, `text/plain`, error tracebacks, and labels for unsupported MIME types. Image and HTML rendering are outside this version.

## Supported Org files

`insula/scenarios/run_tui.sh [path/to/file.org]` accepts a readable Org file inside the Torch Dojo checkout. With no path, it uses `insula/scenarios/default.org`. The host creates an ignored run copy under `target/insula/scenario/<run-id>/run.org`; the chosen source file is never modified. Paths outside the checkout and files without a Python source block receive a clear error before a kernel is started.

Emacs uses Org's parser to find every `python` source block in document order, including unnamed blocks, multiple blocks below one heading, and blocks beneath nested headings. Each block gets a stable run-local cell ID based on its ordinal. Its label is its Org name when present, otherwise the containing heading and ordinal; nearby prose under that heading is shown as context. Headings, lists, tables, links, and non-Python blocks remain in the run copy. Non-Python source blocks do not execute; the opening screen reports how many were skipped.

This is a Python notebook profile, not full Org Babel evaluation. Common display-oriented `:results` headers may remain in the file, but the TUI shows Jupyter's text outputs consistently. A Python block with execution-changing headers such as `:var`, `:session`, `:noweb`, `:dir`, or `:eval` is rejected with its heading and line number before execution; the runner does not silently ignore those semantics. The block structure is fixed for a run. Editing changes source inside a block, not its `#+BEGIN_SRC` or `#+END_SRC` boundaries. External structural edits to the run copy require a restart/new run rather than guessing which cell to execute.

## Learner experience

Keep the quiet editorial blue layout, with a checkpoint navigator and one large active cell. The navigator shows Setup, Connect, each discovered cell, Final, and Cleanup, plus completion, pending, failure, and edited states. The cell view shows the heading path, optional block name, relevant prose, a Python editor with line numbers and syntax highlighting, and a scrollable result directly below. The Emacs and Jupyter live log panes and provenance strip stay visible. At 80×24, the cell and provenance remain legible; at 120×36, logs and results gain room. Long code, Unicode, and tracebacks scroll without damaging the layout.

`F10` executes exactly the pending step. `F9` and the navigator browse completed cells without execution. `Tab` moves between navigator, editor, result, logs, and controls. Standard editor typing and undo work within the source area; a snapshot or log update must not reset the caret, selection, undo history, or scroll when the cell source has not changed. A visible edited marker appears immediately. Before advance, navigation, restart, or quit, the TUI sends the edited source to Emacs, which writes it to the run copy. Editing a completed cell marks its displayed result stale until restart; restart preserves source edits and clears generated results.

Results distinguish stdout, displayed `text/plain` values, errors, and unsupported MIME labels. An empty output is shown explicitly. The result panel can be scrolled independently. The TUI redacts the server token from displayed process logs, bridge errors, and cell text. The Emacs PID and kernel ID come from the worker snapshot; the Jupyter pane follows only events from the current server session. A reused, quiet server may have no new log lines until Connect starts a kernel.

## Step and kernel lifecycle

Setup validates the selected rootfs, shared workspace, authenticated server, token-file mode, and `torch-dojo-gpu` kernelspec. Connect starts one owned kernel through emacs-jupyter and records its ID. Each middle step reads its current code from the Emacs Org buffer, sends one execute request through that client, records structured output in Emacs, and writes a plain-text Org result into the run copy. All cells share the same kernel state. Final reports the number of completed cells, run-copy path, and kernel ID; it performs no file-specific assertion. Cleanup shuts down the owned kernel and confirms its ID is absent from the authenticated server.

A cell error pauses advancement and shows its traceback; restart is needed to run from the beginning with edited source. A timeout or lost response has unknown outcome and must never trigger an automatic retry. Quit tries to clean up the owned kernel and reports a failed cleanup. The host stops only a server it started, and only after confirming no kernel remains. The bundled GPU scenario retains its host-visible workspace-file assertion in its verifier, separate from the generic Final step.

## Ownership and interfaces

- The host launcher validates the selected path, makes the run copy, supervises the two insulas, and supplies the Emacs worker with the run directory. It has no Org execution parser.
- The Emacs scenario engine discovers cells, owns the ordered step list and run buffer, controls kernel lifecycle, and writes results. The bridge exposes only snapshots, source edits by run-local cell ID, next, previous, restart, and quit. No RPC accepts arbitrary Emacs Lisp.
- The Textual frontend renders snapshots and structured results, manages the current editor draft and focus, and follows redacted logs. It never chooses execution order independently of the worker.

The snapshot reports dynamic steps, stable cell IDs, source, heading/context, state, structured text output, pending/viewed indices, Emacs PID, kernel ID, and run path. Source edits are bounded to the existing 1 MiB request limit and reject Org boundary directives. The current-user-only socket and mode-`0700` run directory remain private.

## Verification

1. Emacs tests cover zero, one, and many Python blocks; named and unnamed blocks; nested headings; prose and non-Python blocks; rejected execution headers; stable IDs after source edits; result insertion; restart; and owned kernel cleanup with a fake transport.
2. Frontend tests cover dynamic step counts, editing and undo retention, dirty and stale markers, one F10 per step, navigator browsing without execution, long output and traceback rendering, narrow/wide terminal layouts, and token redaction.
3. Host tests cover path validation, source-file immutability, run-copy creation, and startup failure before kernel launch. The existing bundle remains runnable without a path argument.
4. Live GPU verification runs the bundled Org file through the Emacs bridge, checks a stable kernel ID, synchronized CUDA output, a host-visible workspace file, and absent kernel after Cleanup. A second Org fixture with a different number of cells proves discovery is dynamic. Interactive terminal smoke checks editing, result display, both log panes, restart, and quit.
