# Standalone Scenario TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a learner edit and execute the bundled seven-step Org scenario in a full-screen TUI, with every execution passing through a persistent Emacs worker and emacs-jupyter.

**Architecture:** An Emacs worker in the Emacs insula owns `scenario.el`, the run-specific Org buffer, and the emacs-jupyter client. A host Textual app talks to that worker over a private Unix socket; a host wrapper reuses the current rootfs, Jupyter Server, run-directory, and evidence helpers.

**Tech Stack:** Emacs Lisp, newline-delimited JSON over a Unix socket, Python 3.10+, Textual, unittest, ERT, bubblewrap, Jupyter Server, emacs-jupyter.

**Spec:** `docs/superpowers/specs/2026-10-06-standalone-scenario-tui-design.md`

## Global Constraints

- The execution path is TUI façade → Emacs worker in Emacs insula → emacs-jupyter → Jupyter Server and kernel in Jupyter insula. The Python frontend has no kernel or Jupyter execution client.
- Support only `insula/scenarios/default.org` in this version. Preserve its seven checkpoints and the existing terminal Emacs interface.
- Store the copied Org file, socket, and diagnostics under ignored `target/insula/scenario/<run-id>/`. The run directory is mode `0700`, and the socket is current-user-only.
- Each `F10` runs at most one pending step. Never automatically resend a cell after a timeout or lost worker connection.
- Keep token values out of the TUI, logs, snapshots, and errors. Show the token file path and mode when setup succeeds.
- Reuse a healthy authenticated server or start one through the existing launcher; stop only a server owned by this run and only when no kernels remain. Do not rebuild the rootfs for host Textual.

## Review Focus

- A source string with non-ASCII text, embedded newlines, or Org delimiters must round-trip through JSON and remain one named Python block; pin in Task 1's source-edit test.
- An oversized or malformed request must fail without changing the Org buffer or executing a step; pin in Task 1's protocol test.
- Two rapid `F10` presses while a step is busy must result in one worker `next` command; pin in Task 2's UI test.
- A lost connection after `next` is sent must be reported as outcome unknown without a retry; pin in Task 2's client test.
- Worker death with a possible live kernel must retain a nonzero exit, report the run path and known kernel ID, and leave the server for inspection; pin in Task 3's lifecycle test.

---

### Task 1: Private Emacs scenario bridge

**Files:**
- Create: `insula/emacs/scenario_bridge.el`
- Create: `insula/emacs/scenario_bridge_test.el`
- Modify: `insula/emacs/scenario.el` only for a narrow named-source setter or state hook the bridge cannot safely express.

**Interfaces:**
- Consumes: `torch-dojo-scenario-open`, `torch-dojo-scenario-next`, `torch-dojo-scenario-previous`, `torch-dojo-scenario-restart`, `torch-dojo-scenario-quit`, `torch-dojo-scenario-source`, and the buffer-local state in `scenario.el`.
- Produces: `torch-dojo-scenario-bridge-start()` listening at `<DOJO_SCENARIO_DIR>/bridge.sock`; one JSON request per line, `{"id": <integer>, "command": <allowed-name>, "args": {}}`, and one response with matching `id`, `ok`, and `snapshot` or sanitized `error`. Commands: `snapshot`, `set_source`, `next`, `previous`, `restart`, `quit`. `set_source` args are `{"name": "context"|"cuda"|"workspace", "source": <string>}`. Bound a request to 1 MiB.
- Snapshot fields: `steps` (seven ordered objects with `name`, `state`, `source` or null, `result` or null), `pending_index` (0–7), `viewed_index` (0–6 or null), `busy`, `failed`, `status`, `kernel_id` (string or null), `run_file` (path). Use `pending`, `complete`, `failed`, and `locked` for step state. `result` comes from `torch-dojo-scenario-history` or the current failure, not by reparsing Org results.
- `next` may finish asynchronously; send its response only when the existing scenario completion/failure callback runs. Do not start another command while one is pending. For `quit`, send the response before Emacs exits.

- [ ] **Step 1: Write failing ERT tests** in `scenario_bridge_test.el` for initial snapshot, request ID echo and allowlist, one `next` with fake transport, and `set_source` persistence. Assert that a multiline Unicode source round-trips and a source containing a line that looks like `#+END_SRC` is rejected without changing the Org file. Assert malformed and >1 MiB requests cannot advance the index.
- [ ] **Step 2: Run the ERT file and confirm red.** Run `insula/capsule/enter.sh run --role emacs -- /usr/bin/emacs --batch -Q -l /workspace/torch_dojo/insula/emacs/scenario_bridge_test.el -f ert-run-tests-batch-and-exit`. Expected: missing bridge symbols/tests fail.
- [ ] **Step 3: Implement** `torch-dojo-scenario-bridge-start()`, `torch-dojo-scenario-bridge-dispatch(request)`, `torch-dojo-scenario-bridge-snapshot()`, and `torch-dojo-scenario-bridge-set-source(name source)` in `scenario_bridge.el`. Use `make-network-process` with `:family 'local`; enforce mode `0600` on the created socket and reject a run directory outside the existing scenario path. Keep one in-flight RPC for `next`/`restart`/`quit`; use a completion hook or callback from `scenario.el` to send its final snapshot. Handle partial and multiple newline-framed requests.
- [ ] **Step 4: Run the ERT file and confirm green.** Repeat Step 2's command. Expected: all bridge tests pass; `run.org` reflects only named-cell edits.
- [ ] **Step 5: Commit** the bridge and tests with `git add insula/emacs/scenario.el insula/emacs/scenario_bridge.el insula/emacs/scenario_bridge_test.el && git commit -m "feat: expose scenario through a private Emacs bridge"`.

### Task 2: Full-screen façade and editable cells

**Files:**
- Create: `insula/scenarios/tui_client.py` (socket transport and typed snapshot validation).
- Create: `insula/scenarios/tui_app.py` (Textual widgets and actions).
- Create: `tests/test_scenario_tui.py`.
- Modify: `pyproject.toml` (optional `tui` dependency group, no rootfs package change).

**Interfaces:**
- Consumes: Task 1's JSON commands and snapshot fields.
- Produces: `ScenarioClient(socket_path: Path).request(command: str, args: dict | None = None) -> dict`; `ScenarioApp(client: ScenarioClient)` with `F10` next, `F9` previous, `Tab` focus, restart and quit controls. `ScenarioClient` raises a distinct `OutcomeUnknown` if the connection fails after a `next` write.
- The screen shows seven checkpoints at left, selected step and `TextArea` at upper right, `RichLog` results below, and status/key hints. Source for each named Python step remains editable even after completion; its recorded result stays unchanged until restart. The editor draft is flushed with `set_source` before `next`, selection change, restart, or quit. The pending step stays visible after an error; completed-step browsing never sends `next`.

- [ ] **Step 1: Write failing unittest/Textual pilot tests** for the initial setup screen, editor draft flush before `next`, result display and MIME names, completed-step browsing and edits without execution or result changes, restart preserving source, quit, and two rapid `F10` presses yielding one `next`. Test socket loss after sending `next` produces `OutcomeUnknown` and no retry. Assert imports/actions never call a Jupyter API.
- [ ] **Step 2: Run the focused tests and confirm red.** Run `python -m unittest tests.test_scenario_tui -v`. Expected: missing TUI modules or failing pilot assertions.
- [ ] **Step 3: Implement** `ScenarioClient.request()` in `tui_client.py` and `ScenarioApp` in `tui_app.py`. Textual actions must run socket calls in a worker thread, disable advance while busy, and update the screen only from returned snapshots. Use function keys in the editor; only bind `n/p/r/q` when focus is outside it. Render text and errors directly, and list non-text MIME names without image rendering. Do not import Jupyter client packages.
- [ ] **Step 4: Run the focused tests and confirm green.** Install the optional host dependency in an ignored local venv if absent (`target/insula/tui-venv`), then run that venv's Python with `-m unittest tests.test_scenario_tui -v`. Expected: all pilot and socket tests pass.
- [ ] **Step 5: Commit** with `git add pyproject.toml insula/scenarios/tui_client.py insula/scenarios/tui_app.py tests/test_scenario_tui.py && git commit -m "feat: add editable scenario TUI facade"`.

### Task 3: Host lifecycle and failure ownership

**Files:**
- Create: `insula/scenarios/tui_host.py`.
- Create: `insula/scenarios/run_tui.sh`.
- Create: `tests/test_scenario_tui_host.py`.
- Modify: `insula/emacs/scenario_host.py` only to extract helpers shared with its existing entrypoints.

**Interfaces:**
- Consumes: `scenario_host.ensure_server`, `create_run`, `check_workspace_evidence`, `finish_server`, and Task 1's `torch-dojo-scenario-bridge-start()` plus Task 2's `ScenarioApp`.
- Produces: `run_tui.sh` invokes host Python, validates selected rootfs, creates the run, starts `/usr/bin/emacs --batch -Q` in `enter.sh run --role emacs`, waits for `bridge.sock` readiness, and opens the Textual app. The worker environment forwards only `DOJO_SCENARIO_DIR` from the host; the token is read inside Emacs. Retain worker stderr in the run directory without token values.

- [ ] **Step 1: Write failing lifecycle tests** for rootfs validation before server acquisition, private run and socket mode, startup failure, normal quit, reused versus owned server cleanup, host workspace evidence, and worker death after a possible `next` with nonzero status and no server stop. Assert wrong token/port failures never echo the token and leave no newly owned kernel.
- [ ] **Step 2: Run `python -m unittest tests.test_scenario_tui_host -v` and confirm red.** Expected: missing launcher module or failing ownership assertions.
- [ ] **Step 3: Implement** `main() -> int` in `tui_host.py` and `run_tui.sh`. Supervise the worker process, wait for a successful `snapshot` instead of socket existence alone, and preserve run path plus last known kernel ID on failure. Quit the worker on ordinary UI exit, then perform the existing host workspace check and call `finish_server` only after confirmed worker cleanup. Report a possible orphan and leave an owned server running if worker cleanup is uncertain.
- [ ] **Step 4: Run lifecycle and existing launcher tests.** Run `python -m unittest tests.test_scenario_tui_host tests.test_scenario_launcher -v`. Expected: all pass.
- [ ] **Step 5: Commit** with `git add insula/scenarios/tui_host.py insula/scenarios/run_tui.sh insula/emacs/scenario_host.py tests/test_scenario_tui_host.py && git commit -m "feat: supervise scenario TUI and Emacs worker"`.

### Task 4: Real GPU path, terminal smoke, and user instructions

**Files:**
- Modify: `README.md` (host install, run command, F10/F9/editor/restart/quit, token path, cleanup and inspection path).
- Create: `insula/scenarios/verify_tui.py` (scripted bridge client; never talks to Jupyter directly).
- Modify: `tests/test_scenario_tui_host.py` for verifier failure cases if needed.

**Interfaces:**
- Consumes: Tasks 1–3. `verify_tui.py` starts the same host lifecycle, sends seven explicit `next` commands and checks snapshots/host evidence; it does not send execution requests to Jupyter. The test command is `insula/scenarios/run_tui.sh --verify`.
- Produces: a repeatable GPU acceptance path and documented learner workflow.

- [ ] **Step 1: Write a failing verifier test** that asserts the script drives seven separate `next` calls, sees one stable `kernel_id`, inspects context/CUDA/workspace results, and fails for wrong token and port without showing token value. Assert cleanup leaves the owned kernel absent.
- [ ] **Step 2: Run the focused verifier test and confirm red.** Run `python -m unittest tests.test_scenario_tui_host -v`. Expected: verifier behavior unavailable.
- [ ] **Step 3: Implement** `verify_tui.py` and `run_tui.sh --verify`; document install and controls in `README.md`. Accept `device=cuda:0 sum=10.0`, `/workspace/torch_dojo` paths, host-visible `workspace.txt`, and absent owned kernel as the live success criteria.
- [ ] **Step 4: Run live and regression gates.** Run `insula/scenarios/run_tui.sh --verify`, then interactively run `insula/scenarios/run_tui.sh` in a real terminal and check edit → F10 → result, F9, restart retaining edit, and quit. Run `make test`, the bridge ERT suite, `python -m unittest tests.test_insula_plan -v`, `insula/jupyter_rootfs/verify.sh`, `insula/emacs/verify_attach.sh`, and `insula/emacs/verify_scenario.sh`. Expected: all pass; no token displayed; no leaked kernel.
- [ ] **Step 5: Commit** with `git add README.md insula/scenarios/verify_tui.py insula/scenarios/run_tui.sh tests/test_scenario_tui_host.py && git commit -m "docs: verify and explain standalone scenario TUI"`.
