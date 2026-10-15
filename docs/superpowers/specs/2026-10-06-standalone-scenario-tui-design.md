# Standalone scenario TUI over Emacs Jupyter

## Purpose and boundary

Give a learner a separate full-screen terminal interface for stepping through the bundled Torch Dojo GPU scenario. The learner sees the current checkpoint, an editable Python cell when one is pending, and the result of each completed step without switching to an Emacs buffer. One explicit next-step action runs one checkpoint or cell and stops.

The execution path is **TUI façade → persistent Emacs worker in the Emacs insula → emacs-jupyter → Jupyter Server and kernel in the Jupyter insula**. The façade never starts a kernel, opens a Jupyter WebSocket, or sends a cell execution request directly to Jupyter. The existing terminal Doom Emacs scenario remains available as another interface to the same scenario engine. This milestone supports only `insula/scenarios/default.org`; arbitrary Org notebooks and `dojo/*_eval.py` judging are outside it.

## Learner experience

Run a host launcher such as `insula/scenarios/run_tui.sh`. It validates the selected rootfs, reuses a healthy authenticated Jupyter Server or starts one with the existing launcher, makes an ignored run-specific copy of `default.org`, starts a persistent Emacs worker in the Emacs insula, and opens the full-screen TUI on the host. The server keeps its separate insula and GPU projection. The wrapper records whether it started the server and stops only an owned server with no other kernels after the TUI exits.

The screen has a checkpoint list on the left, the current step's text and editable Python cell on the upper right, a scrollable result panel below it, and a status and key-hint line. A completed checkpoint shows its recorded result; selecting it never reruns it. The pending cell shows the current source from the run-specific Org copy. The user can edit it before execution. Edits are flushed to the Emacs worker and saved in that Org copy before advancing or exiting. The tracked template is never modified.

`F10` advances exactly the pending step from any focus, including the editor. `F9` browses backward through completed steps. `Tab` moves focus between controls and the editor. A restart control shuts down the scenario-owned kernel, clears results, preserves source edits, and returns to setup; a quit control cleans up and exits. Outside the editor, `n/p/r/q` may mirror next/previous/restart/quit; inside it those letters remain normal text input. The interface keeps the next action disabled while a request is outstanding. There is no automatic advancement.

The seven checkpoints and their evidence remain the current scenario's: setup (selected rootfs, shared mount, URL, token-file mode, kernel spec), connect (one kernel ID), context cell (`sys.executable`, cwd, home), synchronized CUDA cell (device and numeric result), workspace-writing cell, workspace readback plus host observation, and cleanup (owned kernel ID absent). The token's value is never displayed. The TUI shows stdout, text/plain results, error tracebacks, and the names of non-text MIME types. It does not attempt terminal rendering of images.

## Emacs worker and local protocol

The Emacs worker loads the existing `insula/emacs/scenario.el`, opens the run-specific `run.org`, and owns its buffer, step index, result history, `emacs-jupyter` client, and kernel. The façade sends commands to the worker; it does not duplicate the step engine or parse the Org document to decide what to execute. The worker reads source from the Org buffer when `next` runs and writes results into the same buffer. A pending-cell edit updates only that named source block; an already-run cell may be edited but its recorded result remains unchanged until restart.

Use one Unix-domain socket under the mode-`0700` run directory, visible to the host and the Emacs insula through the shared repo mount. A small newline-delimited JSON protocol accepts only `snapshot`, `set_source`, `next`, `previous`, `restart`, and `quit`. Each request has an ID; each response has the matching ID, success/error status, and a snapshot containing the ordered checkpoints, pending and viewed indices, selected step source and result, run status, and kernel ID when present. `set_source` names one of the three bundled Python blocks and enforces a bounded message size; the protocol offers no general Emacs Lisp evaluation command. Socket permissions allow the current user only. The socket and any connection diagnostics remain under ignored `target/insula/scenario/<run-id>/`.

The façade calls the worker off its UI thread and marks the action busy until a response arrives. A failed cell or authentication step shows its error and pauses. A Jupyter request timeout is **outcome unknown**: neither worker nor façade resends the cell. A lost worker connection is also shown as an unknown outcome if a step was in flight; the learner gets the run path and any known kernel ID for inspection. Restart and quit ask the worker to shut down only its owned kernel and report cleanup errors separately. The wrapper retains a nonzero exit status on failure and performs the existing host workspace-file check after a completed workspace step. If the worker dies before cleanup, the wrapper reports the possible orphan and leaves the server running for inspection.

## Ownership and implementation seams

- `scenario.el` remains the authority for step order, Org source, result insertion, and emacs-jupyter execution. A narrow bridge exposes the allowed commands and snapshots without moving execution into the Python frontend.
- A new Textual frontend owns only layout, keyboard and button actions, editor draft state, and rendering snapshots. Textual's `TextArea` supplies multiline code editing and `RichLog` supplies a scrollable result area. Install its TUI requirements as an optional host dependency; the selected CUDA rootfs does not need rebuilding for the frontend.
- The host launcher reuses `scenario_host.py` for run creation, server ownership, and workspace evidence. It starts and supervises the Emacs worker, waits for the socket to become ready, and closes the worker and owned server on exit. Existing `run_scenario.sh`, `verify_scenario.sh`, and Docker workflow retain their current interfaces.

## Verification

1. Fake-worker frontend tests exercise screen state: setup and connect, one next action per keypress, editable pending source, results beneath the selected step, browsing without execution, disabled advance while busy, reset preserving edits, and quit.
2. Emacs bridge tests with a fake Jupyter transport exercise the protocol allowlist, request IDs, source updates in the Org buffer, snapshots, result/error serialization, restart, and owned cleanup. They also assert that the frontend does not need a Jupyter client to run a step.
3. On the GPU host, drive the façade through all seven checkpoints with the real worker in the Emacs insula and server/kernel in the Jupyter insula. Confirm a stable kernel ID, rootfs Python and workspace paths, `device=cuda:0 sum=10.0`, host-visible file, inline result, and absent owned kernel after cleanup. Exercise wrong token and wrong port, and verify no token exposure or leaked kernel.
4. Smoke-test the full-screen interface in a terminal, including source editing, F10 advancement, F9 browsing, restart, and quit. Run `make test`, the capsule-plan test, CUDA API verifier, existing Emacs attachment verifier, and existing Emacs scenario verifier as regression gates.
