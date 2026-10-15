# Emacs Jupyter Scenario TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a learner advance a real Emacs-to-Jupyter GPU scenario one checkpoint or editable Python cell at a time in terminal Doom Emacs.

**Architecture:** A host wrapper owns the run directory and optional server start; an Org buffer in the Emacs insula owns the step display and keyboard controls. `scenario.el` keeps one `emacs-jupyter` client across cells and writes results and a workspace-check marker into the shared checkout. The existing server and capsule launchers remain the execution boundary.

**Tech Stack:** Bash, Emacs Lisp/Org/ERT, `emacs-jupyter`, Jupyter Server API, Python unittest, tmux, CUDA PyTorch.

**Spec:** `docs/superpowers/specs/2026-10-06-emacs-jupyter-scenario-tui-design.md`

## Global Constraints

- Use terminal Doom Emacs in the Emacs insula and Jupyter Server in the separate Jupyter insula, both on the selected rootfs and the mutable `/workspace/torch_dojo` mount.
- Use `http://127.0.0.1:8899/`, `target/insula/shared/token` (mode `0600`), and the `torch-dojo-gpu` kernel spec. Never show the token value in the scenario UI or wrapper output.
- Keep generated Org files, workspace evidence, and run markers under ignored `target/insula/scenario/`; preserve edits in each run's Org copy.
- One `n` runs exactly one pending step; `p` only inspects; `r` restarts with a new kernel while preserving cell edits; `q` cleans up the owned kernel and exits.
- The server stays up after the cleanup step. The wrapper stops it on exit only when the wrapper started it and no other kernels remain.
- Leave `insula/jupyter_rootfs/verify.py` and the Docker entrypoint as independent workflows. This scenario demonstrates the insula connection, not a `dojo/*_eval.py` exercise.

## Review Focus

- An occupied port with no authenticated Torch Dojo server must produce a specific failure without replacing that service (Task 1 test).
- A missing, empty, or non-`0600` token file must stop before a kernel is created and must not expose a token (Task 2 test).
- Editing a completed cell must leave its recorded result unchanged until `r`; editing a pending cell must affect the code sent (Task 2 test).
- A request timeout must display **outcome unknown** and must not retry the request automatically (Task 2 test).
- A server started by the wrapper with another active kernel must be left running on exit; cleanup must touch only the scenario kernel/files (Task 3 test).

---

### Task 1: Run wrapper and editable scenario buffer

**Files:**
- Create: `insula/emacs/run_scenario.sh` (run creation, server ownership, Emacs launch)
- Create: `insula/scenarios/default.org` (named, ordered cells and checkpoint text)
- Modify: `insula/capsule/enter.sh` (forward only `DOJO_SCENARIO_DIR` into the Emacs role)
- Create: `tests/test_scenario_launcher.py`
- Create: `insula/emacs/scenario.el` (initial buffer and progression state)
- Create: `insula/emacs/scenario_test.el`

**Interfaces:**
- `run_scenario.sh` takes no arguments; it exports `DOJO_SCENARIO_DIR=/workspace/torch_dojo/target/insula/scenario/<unique-run-id>` only to the Emacs role, then runs `run.sh --load /workspace/torch_dojo/insula/emacs/scenario.el --eval '(torch-dojo-scenario-open)'`.
- Each run directory contains `run.org`, copied from `default.org`; `scenario.el` derives all run-local paths from `DOJO_SCENARIO_DIR`.
- `(torch-dojo-scenario-open)` opens `run.org`, installs buffer-local `n/p/r/q` bindings, and positions at step 1. `(torch-dojo-scenario-next)` dispatches one step and records its outcome. `scenario.el` exports these names for later tasks.
- Host wrapper prints run path and server log path, suppresses `run_server.sh`'s token-bearing URL, and preserves Emacs's nonzero exit status.

- [x] **Step 1: Write failing launcher and ERT tests.** In `test_scenario_launcher.py`, assert one unique run copy per invocation, rootfs validation before server start, reused authenticated server left running, occupied wrong port rejected without launch, and no token value in stdout/stderr. In `scenario_test.el`, assert `n` advances exactly one setup step, `p` does not advance or execute, and the buffer keeps the cursor at the pending named block. Use mocked shell commands for launcher lifecycle tests; use a fake setup provider for ERT.
- [x] **Step 2: Verify red.** Run `python -m unittest tests.test_scenario_launcher -v` and `insula/capsule/enter.sh run --role emacs -- /usr/bin/emacs --batch -Q -l insula/emacs/scenario_test.el -f ert-run-tests-batch-and-exit`; expect missing files/functions or failed assertions.
- [x] **Step 3: Implement the wrapper, Org template, and minimal state/display.** The template has ordered names `context`, `cuda`, and `workspace`; the workspace block writes to the preselected run directory. The setup checkpoint displays rootfs, mount, URL, token-file mode, and kernel-spec readiness without token contents. The wrapper waits for authenticated server readiness and distinguishes an occupied wrong port from a valid running server.
- [x] **Step 4: Verify green.** Run both Step 2 commands; expect all Task 1 tests to pass. Run `bash -n insula/emacs/run_scenario.sh` and `make test`; expect success.
- [x] **Step 5: Commit.** Stage only this task's files and commit `feat: open stepwise Jupyter scenario in Emacs`.

### Task 2: Persistent Jupyter client, cell results, and failure states

**Files:**
- Modify: `insula/emacs/scenario.el`
- Modify: `insula/emacs/scenario_test.el`
- Modify: `insula/scenarios/default.org`

**Interfaces:**
- `(torch-dojo-scenario-transport-connect on-success on-error)` authenticates using the token file and returns a scenario-owned client and kernel ID through `on-success`.
- For batch negative checks only, this transport honors the existing capsule-forwarded `DOJO_VERIFY_TOKEN` and `DOJO_VERIFY_PORT` overrides. Interactive `run_scenario.sh` clears those variables before entering the capsule; its URL stays fixed at port 8899.
- `(torch-dojo-scenario-transport-execute client code on-success on-error)` sends the buffer's current block text once; success reports stdout, text/plain, and non-text MIME names; error reports traceback; timeout reports outcome `unknown`.
- `(torch-dojo-scenario-transport-shutdown client on-done)` shuts down only that client/kernel. The transport functions are the fakeable seam for ERT; use the proven `verify_attach.el` server/client setup and `emacs-jupyter` message APIs for the real implementation.
- `(torch-dojo-scenario-next)` remains disabled while a transport callback is pending. A failed/unknown step pauses until `r` or `q`. Successful step completion inserts an Org `#+RESULTS:` block and saves `run.org`.

- [x] **Step 1: Write failing ERT tests.** Fake the transport callbacks. Assert connect creates exactly one client, three successive cell steps reuse it, a pending cell edit changes the sent source, a completed cell edit leaves its recorded result alone, stdout/text/MIME names appear under the proper block, traceback appears on error, timeout says `outcome unknown` with one send, and bad token/mode fails before connect. Assert a second `n` during a pending callback does nothing.
- [x] **Step 2: Verify red.** Run `insula/capsule/enter.sh run --role emacs -- /usr/bin/emacs --batch -Q -l insula/emacs/scenario_test.el -f ert-run-tests-batch-and-exit`; expect the new test names to fail.
- [x] **Step 3: Implement transport and result handling.** Read token only inside Emacs; use `jupyter-server` at port 8899 and one `jupyter-client` with `torch-dojo-gpu`. Read Org source from the active buffer at dispatch time. Subscribe to execution messages to gather stream, display data, execute result, and error traceback; give each request a 60-second deadline without resending it. Insert and save the result at completion, then update the status area.
- [x] **Step 4: Verify green.** Run the Step 2 ERT command and `insula/emacs/verify_attach.sh`; expect all ERT tests and the existing real attachment check to pass.
- [x] **Step 5: Commit.** Stage this task's files and commit `feat: execute scenario cells through emacs-jupyter`.

### Task 3: Workspace check, restart, and owned cleanup

**Files:**
- Modify: `insula/emacs/scenario.el`
- Modify: `insula/emacs/scenario_test.el`
- Modify: `insula/emacs/run_scenario.sh`
- Modify: `tests/test_scenario_launcher.py`
- Modify: `insula/scenarios/default.org`

**Interfaces:**
- On successful workspace-check step, `scenario.el` reads the run-specific evidence file and creates `DOJO_SCENARIO_DIR/workspace-checked`; the host wrapper checks that same file and marker after Emacs exits.
- `(torch-dojo-scenario-restart)` (`r`) shuts down the owned kernel, removes scenario-generated results/marker/evidence for this run, preserves block source edits, and returns to setup. It starts a new kernel only on the next connect step.
- `(torch-dojo-scenario-quit)` (`q`) attempts owned-kernel cleanup even after a failure, reports cleanup errors separately, and exits with a failure status when the scenario or cleanup failed. The cleanup checkpoint also confirms the owned kernel ID is absent through the authenticated server API.
- The wrapper calls existing `stop_server.sh` only if it started the server and a checked authenticated API query reports zero kernels. It never uses `stop_server.sh` when another kernel remains because that script reaps all kernels.

- [x] **Step 1: Write failing ERT and launcher tests.** Assert workspace check reads exact run evidence and writes a marker; host check fails if the marker exists but file is missing/mismatched; `r` preserves edited blocks but removes old results and uses a new client at the next connect; `q` on a failed cell still shuts down only the owned client; cleanup errors remain visible. Mock the API to assert an owned server with a foreign kernel stays up, and a server reused by the wrapper stays up regardless of kernel count.
- [x] **Step 2: Verify red.** Run the ERT and Python commands from Tasks 1–2; expect the new assertions to fail.
- [x] **Step 3: Implement workspace and lifecycle behavior.** The workspace block writes its predetermined path. Query `/api/kernels` with the token to confirm disappearance of the owned ID; retain the server after the cleanup checkpoint for log inspection. Preserve the initial failure when cleanup also fails, and remove only this run's evidence and result blocks on `r`.
- [x] **Step 4: Verify green.** Run the ERT and Python commands from Tasks 1–2 plus `make test`; expect success. Run `bash -n insula/emacs/run_scenario.sh`; expect exit 0.
- [x] **Step 5: Commit.** Stage this task's files and commit `feat: verify scenario workspace and clean owned resources`.

### Task 4: Live GPU and terminal walkthrough

**Files:**
- Create: `insula/emacs/verify_scenario.sh` (batch driver using the real `scenario.el` controls)
- Modify: `README.md` (launch, keys, run path, token path, tmux windows, expected checkpoints/results)
- Modify: `insula/emacs/scenario_test.el` or `tests/test_scenario_launcher.py` only for defects exposed by live verification

**Interfaces:**
- `verify_scenario.sh` runs the scenario in the Emacs insula, dispatches each of the seven `n` steps individually, logs step names/outcomes without the token, and exits nonzero unless CUDA output, shared workspace evidence, stable kernel ID, and kernel cleanup all pass.
- The documented interactive path is `insula/emacs/run_scenario.sh`; show a two-window tmux session with `emacs` running the scenario and `jupyter` following `target/insula/shared/server.log`.

- [x] **Step 1: Add a failing batch-verifier check.** Make the batch driver assert the expected ordered checkpoints, `torch.cuda.synchronize()` result, identical kernel ID across cells, host-visible workspace contents, and absent kernel after cleanup. Use `DOJO_VERIFY_TOKEN` and `DOJO_VERIFY_PORT` in negative invocations; each must fail before cell execution.
- [x] **Step 2: Verify red.** Run `insula/emacs/verify_scenario.sh`; expect a missing command or explicit failed assertion before completing the scenario.
- [x] **Step 3: Implement the batch driver and README walkthrough.** Reuse the same dispatch functions and Org template as the interactive path; do not build a separate API-only evaluator. Document `n/p/r/q`, setup/connect/context/CUDA/workspace/check/cleanup, results, URL, token path, and the tmux log view.
- [x] **Step 4: Verify green on the GPU host.** Run `make test`, `insula/jupyter_rootfs/verify.sh`, `insula/emacs/verify_attach.sh`, and `insula/emacs/verify_scenario.sh`; expect all to pass. Open the scenario interactively in tmux, press `n` once per checkpoint, inspect each inline result and the server-log window, then press `q`. Stop/restart an owned server and confirm token rotation and no scenario kernel remains. Record command output or a concise verification log under ignored `target/insula/scenario/`.
- [x] **Step 5: Commit.** Stage only this task's files and commit `docs: add live Emacs scenario walkthrough` (or `feat:` if batch driver code dominates).
