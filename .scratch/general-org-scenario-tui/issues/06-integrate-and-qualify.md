# 06: Qualify one scenario engine for custom and bundled Org files

**What to build:** Learners have one documented scenario flow for both the bundled GPU notebook and their own supported Org files, with the fixed seven-step assumptions removed and a complete regression run behind the result.

**Blocked by:** 04: Make editing and text results feel like a notebook; 05: Run the bundled GPU scenario on the general path

**Status:** resolved

- [x] The general path is the only scenario execution engine; the bundled verifiers derive step counts from the worker, and startup labels no longer assume seven steps.
- [x] Documentation explains the supported Org profile, file selection, ordered execution, text-only output, editing and restart behavior, lifecycle steps, and the Emacs-to-Jupyter provenance shown in the TUI.
- [x] Automated tests cover custom and bundled files, bad paths and unsupported headers, source immutability, unknown outcomes, token redaction, server ownership, and kernel cleanup.
- [x] Live GPU and terminal smoke checks exercise editing, browsing, Final, Cleanup, and both log panes; no scenario-owned kernel remains afterward.
- [x] The complete Python, Emacs, capsule-plan, and Jupyter attachment regression gates pass.

## Resolution

The existing Emacs Org engine handles both the bundled notebook and the two-cell fixture. The bundled bridge verifier reads the lifecycle from its initial snapshot and checks GPU, workspace, Final, and Cleanup evidence without a fixed step loop. The Emacs batch verifier also uses the discovered step count. The TUI starts with neutral progress labels until the snapshot arrives. The README now states the supported headers, editing boundary, stale results, and restart behavior.

Qualification on an isolated Jupyter Server at port 8906 with a separate mode-0600 token file:

- `make test`: 57 passed, 18 optional Textual tests skipped because the system Python lacks Textual. The Textual venv ran all 57 without skips.
- `scenario_org_test.el`, `scenario_test.el`, and `scenario_bridge_test.el`: 6, 15, and 6 ERT tests passed sequentially.
- `tests.test_insula_plan`: passed; `verify_attach.sh`: attached, executed, and shut down a kernel.
- Bundled TUI bridge verification and Emacs batch verification: CUDA sum 10.0, rootfs Python, host workspace file, stable kernel ID, and kernel absent after Cleanup.
- Custom two-cell Org run: six dynamic steps, CUDA state reused (`sum=6.0`), source edit executed from the run copy, original source unchanged, and owned kernel absent after Cleanup.
- Interactive 120×36 terminal run: edited source, F10, F9, restart, Final, Cleanup, quit, and both Emacs and Jupyter log panes checked. The authenticated kernel list was empty after quit.
- `git diff --check` passed. No request or stop command was sent to the server on port 8899.
