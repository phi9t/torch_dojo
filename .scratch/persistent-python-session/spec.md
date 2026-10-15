# Dependable persistent Python/Torch workouts

Preserve the existing Emacs Jupyter buffer/region/REPL workflow against Torch Dojo's GPU insula. Keep expensive objects in one kernel and never replay an uncertain execution automatically.

## First slice: transport recovery

A project-owned transport must be scoped to Dojo clients; do not globally modify the installed Emacs Jupyter package. Preserve the frontend session ID across socket replacement, retain original requests and recover available buffered output/reply/idle. Send code once and reject submission when disconnected. Show pending outcomes as unknown on transport loss, and reconcile them when completion arrives. Detect a changed kernel message generation even when the REST kernel ID stays the same; invalidate old pending outcomes and require explicit acknowledgement before using the new empty workspace. A detach closes only the frontend, not the kernel. Kernel shutdown remains an explicit action.

Public seam: Dojo connect/run REPL commands, session disconnect/reconnect/status, existing jupyter-eval-buffer/region/string, and request outcome lookup. Acceptance uses actual batch Emacs in its insula against an owned GPU kernel, including offline completion, stable CUDA objects, disconnected submission rejection, and restart state loss. Batch verification does not claim interactive UI or SSH-tunnel qualification.

## Follow-up slices

1. Interactive asynchronous streaming, responsiveness, interrupt, and real tunnel failure qualification.
2. Small successful-result reuse helper: explicit invalidation, health checks, cleanup; linear workout scripts.
3. Selected-frame debugger support and bounded tensor inspection, exact submitted source snapshots, and paused reconnect.

Explore Python bindings to C++/LibTorch and Rust separately; xeus/Cling, native CUDA debugger attachment, Wasm, and Evcxr stay in research until qualified. These do not block the Python baseline.
