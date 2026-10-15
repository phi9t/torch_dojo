# General Org Scenario TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a chosen repo-local Org file as an ordered, editable Python notebook in the existing scenario TUI, with Setup, Connect, Final, and Cleanup supplied by the runner.

**Architecture:** Emacs parses the Org run copy and owns dynamic cell IDs, source, results, step order, and the emacs-jupyter client. The bridge returns dynamic snapshots and accepts bounded source edits by cell ID. The host selects and copies a file; Textual renders and edits snapshots without parsing Org or calling Jupyter.

**Tech Stack:** Emacs Lisp/Org and ERT; Python 3 and unittest; Textual 6; existing bwrap CUDA rootfs and Jupyter Server.

**Spec:** `docs/superpowers/specs/2026-10-06-general-org-scenario-tui-design.md`

## Global Constraints

- Every execution remains TUI → Emacs insula → emacs-jupyter → Jupyter insula; the frontend never sends a Jupyter execution request.
- A run has Setup, Connect, N Python cells in document order, Final, Cleanup; N must be at least one.
- Source files remain unchanged; only ignored run copies under `target/insula/scenario/` are edited.
- Text output only: stdout, `text/plain`, traceback, and unsupported MIME labels. Keep the existing live process logs and provenance strip.
- One F10 sends at most one step; an unknown outcome is never retried automatically. Preserve kernel ownership and cleanup rules.
- The bridge keeps its command allowlist, current-user-only socket, 1 MiB request limit, and token redaction.

## Review Focus

- A nested, unnamed Python block after another block under the same heading must get a distinct stable ID and the right heading label: Task 1 test.
- An Org file with `:var`, `:session`, `:noweb`, `:dir`, or `:eval` on a Python block must fail with its line number before execution: Task 1 test.
- Source containing Org block delimiters or an external structural edit must not make the worker execute another cell: Tasks 1 and 3 tests.
- A long traceback at 80×24 must remain readable while the editor keeps its cursor and undo state across log/snapshot updates: Task 5 test and terminal smoke.
- A bad path, empty Python file, wrong token, or lost worker response must not leak a token, retry a cell, or orphan a kernel without a diagnostic: Tasks 2, 4, and 5 tests.

---

### Task 1: Emacs Org Cell Model

**Files:** Create `insula/emacs/scenario_org.el`, `insula/emacs/scenario_org_test.el`.

**Interfaces:** `torch-dojo-scenario-org-discover` returns a vector of cell alists with `id` (`cell-1`, ...), `name`, `heading`, `text`, `source`, and `line`; `torch-dojo-scenario-org-skipped-count` reports non-Python source blocks. `torch-dojo-scenario-org-source`, `torch-dojo-scenario-org-set-source`, `torch-dojo-scenario-org-write-result`, and `torch-dojo-scenario-org-clear-results` operate on those IDs in the current Org buffer. A stored structure signature detects changes to the block count, order, language, or headers before any execute/edit; source-only edits preserve IDs.

- [ ] **Step 1: Write failing ERT tests** for named and unnamed blocks, nested headings, multiple blocks per heading, prose, skipped non-Python blocks, existing `#+RESULTS:`, and distinct stable IDs after a source edit. Assert zero Python blocks and execution-changing headers fail with a heading and line number; assert an external block insertion changes the signature; assert a source with `#+END_SRC` or `#+BEGIN_SRC` is rejected without changing the buffer.
- [ ] **Step 2: Run the ERT file and confirm red.** `insula/capsule/enter.sh run --role emacs -- /usr/bin/emacs --batch -Q -l /workspace/torch_dojo/insula/emacs/scenario_org_test.el -f ert-run-tests-batch-and-exit`.
- [ ] **Step 3: Implement the Org model** with Org element parsing. Keep the run-local ID tied to block order, revalidate structure before edits/execution, and insert/clear only results associated with discovered Python blocks. Accept display-oriented `:results` headers; reject `:var`, `:session`, `:noweb`, `:dir`, and `:eval` on Python blocks.
- [ ] **Step 4: Run the ERT file and confirm green.**
- [ ] **Step 5: Commit** the Org model and tests.

### Task 2: Dynamic Scenario Engine and Bundled File

**Files:** Modify `insula/emacs/scenario.el`, `insula/emacs/scenario_test.el`, `insula/emacs/verify_scenario.el`, `insula/scenarios/default.org`.

**Interfaces:** `torch-dojo-scenario-steps` becomes buffer-local dynamic step data derived from Task 1. Step kinds are `setup`, `connect`, `cell`, `final`, `cleanup`; a cell step contains its run-local cell ID. Status is buffer-local state and does not require a `#+SCENARIO_STATUS` line in the chosen Org file. The engine stores structured output as `stdout`, `text`, `mime`, and `error`, alongside formatted Org text. The existing transport connect/execute/shutdown functions remain the only Jupyter path.

- [ ] **Step 1: Write failing ERT tests** for one and several discovered cells, a chosen file without scenario metadata, one next per step, shared client and kernel ID, generic Final summary, cleanup, restart preserving edited source, cell error pause, timeout with no resend, and wrong token/port. Assert Final does no bundled workspace-specific check.
- [ ] **Step 2: Run `scenario_test.el` ERT and confirm red.** Use the same capsule Emacs command as Task 1 with `scenario_test.el`.
- [ ] **Step 3: Implement dynamic dispatch** using Task 1 IDs; remove seven-step index and named-block assumptions. Migrate `default.org` to three middle Python cells so its run still has seven steps. Make its workspace cell check its own write, while the host verifier separately checks host visibility. Update `verify_scenario.el` to assert the new Final and Cleanup evidence.
- [ ] **Step 4: Run `scenario_test.el` ERT and confirm green.** Also run `insula/emacs/verify_scenario.sh` for the bundled GPU path.
- [ ] **Step 5: Commit** the engine, bundled file, and tests.

### Task 3: Dynamic Bridge Snapshots and Editing

**Files:** Modify `insula/emacs/scenario_bridge.el`, `insula/emacs/scenario_bridge_test.el`, `insula/scenarios/tui_client.py` only if response decoding needs adjustment.

**Interfaces:** `snapshot.steps[]` includes `kind`, `cell_id` or null, `name`, `heading`, `text`, `source` or null, `state`, `result`, and structured `output` or null. Keep `pending_index`, `viewed_index`, `emacs_pid`, `kernel_id`, and `run_file`, and add `skipped_count`. `set_source` accepts `{cell_id, source}`. Commands remain `snapshot`, `set_source`, `next`, `previous`, `restart`, `quit`.

- [ ] **Step 1: Write failing bridge ERT tests** for two differently named/unnamed cells, editing by ID, result serialization, 1 MiB and boundary-directive rejection, external structure change, malformed request, and no arbitrary-Lisp RPC. Assert a completed-cell edit leaves its prior result recorded until restart.
- [ ] **Step 2: Run `scenario_bridge_test.el` ERT and confirm red.**
- [ ] **Step 3: Implement snapshots and source edits** by delegating cell lookup and validation to Task 1, without parsing Org in the bridge. Keep asynchronous reply and unknown-outcome behavior intact.
- [ ] **Step 4: Run both `scenario_bridge_test.el` and `scenario_test.el` ERT and confirm green.**
- [ ] **Step 5: Commit** the bridge and tests.

### Task 4: File Selection, Run Copies, and Verification

**Files:** Modify `insula/emacs/scenario_host.py`, `insula/scenarios/tui_host.py`, `insula/scenarios/verify_tui.py`, `tests/test_scenario_launcher.py`, `tests/test_scenario_tui_host.py`; add `insula/scenarios/fixtures/two_cells.org`.

**Interfaces:** `scenario_host.resolve_scenario_source(root: Path, argument: str | None) -> Path` accepts the default or a readable Org path inside the checkout. `scenario_host.create_run(root: Path, source: Path | None = None) -> Path` copies it to `run.org`. `run_tui.sh [path/to/file.org]` runs interactively; `run_tui.sh --verify` remains the bundled GPU acceptance check. A path together with `--verify` gets a clear usage error until a generic verifier is defined.

- [ ] **Step 1: Write failing Python tests** for default/custom selection, path or symlink outside checkout, directory/non-Org input, missing file, unchanged source, distinct run copies, and validation before server start. Test a two-cell fixture yields six dynamic steps through a fake bridge. Update bundled verifier assertions for Setup, Connect, three cells, Final, Cleanup, GPU output, host workspace evidence, and kernel absence.
- [ ] **Step 2: Run `python -m unittest tests.test_scenario_launcher tests.test_scenario_tui_host -v` and confirm red.**
- [ ] **Step 3: Implement path and run lifecycle changes** without moving Org parsing into Python. Make bundled host workspace evidence read the expected file directly, replacing the current `workspace-checked` marker that belongs to the removed special step. Keep server ownership, possible-orphan diagnostics, and token handling.
- [ ] **Step 4: Run those Python tests and `insula/scenarios/run_tui.sh --verify`; confirm green and host-visible GPU evidence.**
- [ ] **Step 5: Commit** host, fixture, verifier, and tests.

### Task 5: Polished Dynamic Notebook View

**Files:** Modify `insula/scenarios/tui_app.py`, `insula/scenarios/tui_logs.py` if needed, `tests/test_scenario_tui.py`, `tests/test_scenario_tui_logs.py`, `README.md`.

**Interfaces:** The frontend consumes Task 3's dynamic snapshot only. Its editor sends `{cell_id, source}` before advance, navigation, restart, or quit. The selected cell shows heading path, prose, syntax-highlighted source with line numbers, and separately styled stdout, value, traceback, empty-output, and MIME-label sections. Existing Emacs/Jupyter logs and provenance remain visible.

- [ ] **Step 1: Write failing Textual tests** for arbitrary cell counts, skipped-block notice, dirty and stale-result markers, edit flushing on each navigation path, no edit sent for unchanged source, cursor/selection/undo and scroll retention on log/snapshot refresh, Tab focus order, long Unicode source and traceback, token redaction in all displayed text, one F10 while busy, and no retry after unknown outcome. Exercise 80×24 and 120×36 layouts.
- [ ] **Step 2: Run with the Textual venv and confirm red.** `target/insula/tui-venv/bin/python -m unittest tests.test_scenario_tui tests.test_scenario_tui_logs -v`.
- [ ] **Step 3: Implement the focused-cell presentation** using the existing quiet blue theme. Preserve editor widget state unless the selected cell or source truly changed. Reflow or redraw wrapped result/log text after terminal resize. Update README with file selection, supported Org profile, editing keys, output limits, and the fixed lifecycle steps.
- [ ] **Step 4: Run the Textual tests and confirm green.** Smoke-test the real TUI in a terminal with the bundled and two-cell files: edit, F10, F9, restart, logs, Final, and quit.
- [ ] **Step 5: Run final gates:** `make test`; Task 1–3 ERT files; `insula/scenarios/run_tui.sh --verify`; `insula/emacs/verify_attach.sh`; `git diff --check`. Confirm the custom file leaves its source untouched and all owned kernels are gone.
- [ ] **Step 6: Commit** frontend, documentation, and tests.
