# Emacs Jupyter scenario TUI

## Purpose

Let a learner watch and control a complete Torch Dojo Jupyter session from terminal Doom Emacs. One `n` keypress advances one setup, connection, code-cell, check, or cleanup step. Each Python cell is visible before execution and its result appears immediately below it afterward. Jupyter Server continues to run in its separate insula and tmux window.

This first scenario demonstrates the insula connection and a CUDA kernel. It does not judge a `dojo/*_eval.py` exercise. Its cells are editable so the learner can experiment before evaluating them.

## User flow

Run `insula/emacs/run_scenario.sh` from the host checkout. The wrapper validates the selected rootfs, starts Jupyter Server with the existing `run_server.sh` if needed, records whether it owns that server, copies `insula/scenarios/default.org` to an ignored run-specific path, and opens that copy in terminal Doom Emacs inside the Emacs insula. An already-running authenticated server is reused. The wrapper never replaces or stops a server it did not start.

The Org buffer is the TUI. A small status area shows the current step, the next action, and the most recent outcome. The cursor remains on the current cell when a cell step is pending. Keys:

| Key | Action |
| --- | --- |
| `n` | Run exactly the pending step, then stop and show its result. |
| `p` | Navigate to an earlier completed step and its recorded result; do not re-execute code. |
| `r` | Shut down the scenario-owned kernel, clear scenario results while preserving cell edits, and begin again with a new kernel. |
| `q` | Clean up the scenario-owned kernel and exit this scenario Emacs process. The host wrapper then stops Jupyter only if it started that server. |

The initial scenario has these ordered steps:

1. **Setup:** show the selected rootfs, repo mount, server URL, token-file mode, and kernel-spec availability; report readiness. The token value is never displayed.
2. **Connect:** authenticate from the Emacs insula using `emacs-jupyter`, start one `torch-dojo-gpu` client/kernel, and show the kernel ID. All subsequent cells use this client.
3. **Context cell:** execute a Python block showing `sys.executable`, cwd, and home.
4. **CUDA cell:** create a CUDA tensor, synchronize, and show a numeric result and device.
5. **Workspace cell:** write a run-specific file under `/workspace/torch_dojo/target/insula/scenario/` and display its path and contents.
6. **Workspace check:** read that file through the shared repo mount and show the value. The host wrapper also checks the file after Emacs exits, so the check includes a host-side observation.
7. **Cleanup:** shut down the client/kernel and confirm that the scenario's kernel is absent from the server API.

`n` is disabled while an asynchronous request is outstanding. Completion of a step never automatically starts the next one. When viewing an earlier result, `n` still advances the pending step. Editing an already-run cell changes its text but does not change the recorded result; the learner must use `r` to rerun the sequence in a fresh kernel. The run-specific Org copy is saved after each completed step. The server stays up after the cleanup step so its log can be inspected; the wrapper stops an owned server when the learner exits with `q`, provided no other kernels are active.

## Components and ownership

- `insula/scenarios/default.org` owns the ordered, named Python source blocks and explanatory text. The wrapper opens a run-specific copy under ignored `target/insula/scenario/`, so edits and saved results do not change the tracked template. Code is read from the buffer at execution time, so edits to a pending cell are honored.
- `insula/emacs/scenario.el` owns the Emacs-facing step state, keybindings, Org navigation, result insertion, and `emacs-jupyter` client. It uses one client/kernel for the whole run and captures stdout, text results, and error tracebacks. Non-text MIME output is identified by MIME type in the terminal buffer.
- `insula/emacs/run_scenario.sh` owns host setup and server lifecycle. It reuses the existing capsule and server launchers, enters the Emacs insula, and performs host-side workspace verification after a completed workspace step. On exit it stops the server only when it started that server and no other kernels are active; otherwise it leaves the server running and reports why. It must preserve the exit status of a failed scenario.
- Existing `insula/jupyter_rootfs/verify.py` remains a noninteractive API regression check. The scenario uses the actual Emacs-to-Jupyter path.

Scenario-generated workspace files live under ignored `target/insula/scenario/`. The scenario removes only run-specific files and kernels it owns. The rootfs, Jupyter Server, and Emacs client keep their current namespace and mount boundaries.

## State and failure behavior

The state machine is `ready -> setup -> connected -> cell 1 -> cell 2 -> cell 3 -> workspace checked -> cleaned`. Each transition records its start, outcome, and displayed result. A failed cell shows the Jupyter error/traceback in its Org result and pauses the scenario. A timeout is reported as **outcome unknown**; the UI does not resend that cell because doing so could repeat a write or computation. `r` inspects and shuts down the owned kernel before starting over. `q` attempts cleanup even after a failure and reports cleanup errors separately from the original error.

If setup finds the port occupied by another service, a missing token, or an invalid rootfs, it stops with a specific message. Authentication failures stop before a kernel is created. Server and kernel identifiers may be displayed; token contents may not. The run-specific workspace path is chosen before cell execution, so the host wrapper can verify the same path without parsing terminal output.

## Verification

1. Unit tests cover state transitions: one `n` performs one action; `p` never executes; pending cell edits are used; failures and unknown outcomes pause; `r`/`q` clean only owned resources.
2. Batch Emacs tests with a fake Jupyter transport check Org result insertion for stdout, a text result, and an error traceback.
3. On the GPU host, run the scenario through its setup, connection, three cells, workspace check, and cleanup. Confirm the CUDA result, a host-visible file, one stable kernel ID across cells, and no surviving scenario kernel.
4. Repeat with a bad token and an occupied/wrong port. Confirm visible failure and no leaked kernel. Run in the two-window tmux layout to check that the learner can follow the server log while advancing cells in Emacs.

The existing `make test`, capsule-plan test, CUDA API verifier, and Emacs attachment verifier continue to pass.
