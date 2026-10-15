# 01: Run one chosen Org cell through Emacs and Jupyter

**What to build:** A learner can choose a repo-local Org file containing one Python block, open it in the scenario TUI, and step through Setup, Connect, the cell, Final, and Cleanup. The result comes from the persistent Emacs worker's emacs-jupyter client and the source file remains untouched. Keep the existing bundled scenario usable while this new path is added.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] A readable Org file inside the checkout opens as a private run copy; a missing file, non-Org file, directory, or path that resolves outside the checkout fails before a kernel starts.
- [x] Emacs discovers one Python block and exposes five runner-and-cell steps to the TUI without the host or TUI deciding what Org code to execute.
- [x] One F10 runs one step, the cell uses the Emacs-owned kernel, and its text result appears in the TUI and run copy.
- [x] Final reports the run and Cleanup confirms the owned kernel is absent; the chosen source file remains byte-for-byte unchanged.
- [x] Automated coverage and an interactive or batch demo exercise the full path while the bundled default still works.

## Comments

- Implemented selected-file runs, dynamic Emacs-owned cells and bridge snapshots, generic Final/Cleanup, and the existing bundled path. ERT exercises a five-step one-cell run with a fake kernel transport; Python and Textual tests cover selection, the bridge client, and the dynamic TUI. Live GPU verification passed on an isolated server at port 8898: one kernel ID, synchronized CUDA output (`device=cuda:0 sum=10.0`), a host-visible workspace file, and an absent owned kernel after Cleanup. That test server was stopped; the existing server on port 8899 was untouched.
