# 05: Run the bundled GPU scenario on the general path

**What to build:** The default Torch Dojo Org scenario uses the same dynamic cell engine as a chosen file, while retaining its proof that execution goes through Emacs to a GPU Jupyter kernel and shares the mutable workspace.

**Blocked by:** 03: Edit arbitrary cells without changing execution order

**Status:** resolved

- [x] Starting the TUI without a path runs the bundled Org file with runner-owned Setup, Connect, Final, and Cleanup around its Python cells.
- [x] The bundled verifier checks one stable kernel ID, rootfs Python and workspace paths, a synchronized CUDA result, and a file written by the kernel that is visible from the host checkout.
- [x] Final is generic; bundled workspace evidence lives in the bundled verifier, and Cleanup confirms the owned kernel ID is absent from the authenticated server.
- [x] The existing batch verifier and interactive terminal scenario remain usable or have documented equivalent commands, with wrong token and port failures covered.
- [x] GPU integration tests pass without sending a Jupyter execution request from the host or TUI.

## Comments

The bundled file runs as three discovered Python cells between the four runner-owned steps. The TUI verifier requires the rootfs Python path, shared workspace path, CUDA output, stable kernel ID, host-visible file, and authenticated kernel absence. The batch verifier also completed all seven steps. The Org file documents the interactive Emacs keys. Live verification used an isolated GPU Jupyter server on port 8898; wrong-port and wrong-token runs failed as expected. The isolated server and token were removed after zero kernels remained.
