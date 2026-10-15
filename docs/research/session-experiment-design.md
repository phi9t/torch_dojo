# Persistent workout experiments

Status: research experiments, not a production implementation. Recorded 2026-10-07 UTC.

The Python milestone is supported by real GPU and separate Emacs-insula experiments. A small session cache also supports the desired linear workout: explicitly reevaluate the buffer, reuse setup, and change the experiment beneath it. The current Emacs transport still loses completion/output produced during a disconnect. That limitation should be addressed before claiming reliable session recovery.

## Hypotheses and acceptance criteria

| Direction | Disposable experiment | Acceptance criterion | Observation / status |
| --- | --- | --- | --- |
| Persistent GPU namespace | Make model and CUDA tensor; evaluate buffer and region; reconnect idle | Same Python objects and CUDA allocation; new result from existing model | Passed, protocol and Emacs probes |
| Execution recovery | Submit one increment and delayed output; drop TCP while busy; reconnect with same frontend identity | Increment once; correlate original reply/output/idle without another execute request | Direct protocol passed; current Emacs client loses offline completion/output |
| Unrecoverable output | Repeat busy drop but change frontend identity | Namespace survives; missing original completion remains explicitly unknown | Observed; explicit state inspection found one mutation |
| Restart distinction | Restart only the experiment's kernel through REST | Same server kernel ID but new kernel generation; previous namespace absent | Passed; Emacs also observed different PID |
| Paused-frame evaluation | Dump exact submitted source; break inside function; request locals and evaluate a bounded tensor expression | Selected frame contains correct local CUDA tensor | Passed through protocol and research-only Emacs control-channel support |
| Debugger recovery | Lose transport while paused, reconnect, request debugInfo and fresh stack | Stopped state retained; new frame inspection succeeds without code resubmission | Passed |
| Step / continue | Step and inspect fresh source line; continue original request | Correct next line, execution completes, model and input retained | Passed |
| Edited regions | Change function and insert a source line; dump exact new text | New source identity, breakpoint on new line, new calculation observed | Passed |
| Linear fixture reuse | Submit identical workout twice, then edited calculation | Setup counters one; identical model/tensor/port; edited result changes | Passed in GPU protocol and separate Emacs-insula experiments |
| Explicit cache lifecycle | Cache None; fail setup; invalidate service/model; kill service | None retained, failures visible, teardown observed, unhealthy resource rejected | Passed |
| Cache limits | Change factory without changing key; fail after a side effect; restart kernel | Old value reused, side effect repeated on explicit retry, restart clears cache | Observed, intentional boundaries |
| Native orchestration | Retain C++/Rust opaque objects beside CUDA fixture; link project library; load edited library at different path | Repeated calls preserve native state; old and new artifacts remain distinguishable | Passed; see [native experiment report](native-orchestration-experiments.md) |

Existing baseline details and raw-evidence references are in [persistent Python GPU sessions](persistent-python-gpu-sessions.md) and [Emacs runtime](emacs-session-runtime.md). The integration recommendation remains [a project-owned session adapter around Emacs Jupyter](emacs-session-integration.md), with direct Jupyter debugger requests investigated before a Dape bridge.

## Linear workout experiment

`experiments/persistent-session/once_lab.py` is an intentionally small experiment helper. Its dictionary belongs to an imported module, outside the reevaluated buffer. Ordinary import returns the existing module from `sys.modules`; explicitly reloading that helper, deleting the module, or restarting the kernel is outside its retention contract. [Python import system](https://docs.python.org/3/reference/import.html#the-module-cache).

```python
from once_lab import once, invalidate

model = once("model", model_setup)
x = once("data", data_setup)
service = once("service", start_service,
               healthy=lambda pair: pair[1].is_alive(), close=stop_service)
y = model(x) + 1
assert y.shape == (4, 8) and y.is_cuda
```

The service is a real local TCP echo server in a thread, not a mock handle. The test uses a small CUDA model and input to qualify identity and semantics; it does not measure savings for a large fixture. It submits the same workout text twice, then changes `+ 1` to `+ 2`. Setup counters stay at one and the result changes as expected. It compares Python object identity, CUDA `data_ptr`, and service endpoint. It reconnects the frontend and verifies the same cached objects, explicitly stops/replaces the service, and replaces the model.

Failures are not inserted into the cache. A failed setup can still have external side effects: the test deliberately increments a counter before throwing, then explicitly retries and observes two increments. This proves the helper provides successful-result reuse, not exactly-once external effects. It never retries automatically. A changed factory with the same key still returns the old value; edited setup requires explicit invalidation or an explicitly changed key.

Invalidation calls the registered cleanup before removing the entry. If cleanup raises, the entry stays reachable and the failure propagates. Removing an entry does not release objects still referenced by buffer globals, a debugger frame, or native code. The helper does not provide concurrent single-flight setup, cycle detection, cancellation reconciliation, automatic dependency invalidation, cross-kernel persistence, or comprehensive resource ownership. Those remain design questions if the experiment becomes production code.

The standalone Emacs cache probe uses public `jupyter-eval-buffer`, `jupyter-eval-region`, and `jupyter-connect-server-repl`. It recreates the simple model/input setup, repeats the workout, edits only the calculation, attaches a new REPL client to the same kernel, reevaluates, and explicitly invalidates the fixture. Python assertions establish the setup counts and identities. This qualifies the concise editor workflow rather than only direct protocol calls.

## Reproduction and evidence

Prerequisites: the selected GPU rootfs and existing Jupyter server at `127.0.0.1:8899`; the probes read `target/insula/shared/token` without printing it. Each creates its own kernel and deletes only that kernel on ordinary exit. Run from the repository root, one copy of each probe at a time:

```bash
bash experiments/persistent-session/run_protocol_probe.sh
bash experiments/persistent-session/run_emacs_probe.sh
bash experiments/persistent-session/run_once_probe.sh
bash experiments/persistent-session/run_emacs_once_probe.sh
bash experiments/persistent-session/run_native_probe.sh
```

Evidence:

- `target/insula/research/protocol/{results.json,messages.json,verification.log}`: nine baseline protocol scenarios, independently rerun this turn, exit 0.
- `target/insula/research/emacs/{evidence.json,run.log}`: independently rerun this turn, exit 0; documented evidence outcomes separately asserted and owned kernel deletion verified by HTTP 404.
- `target/insula/research/once/{results.json,messages.json,run.log,workout.py}`: ten version/result records; assertions run in GPU kernel, exit 0.
- `target/insula/research/emacs-once/{evidence.json,run.log}`: separate editor-driven cache experiment, exit 0; buffer repeat, edited region, public reattachment, and invalidation asserted.

The native experiment was also independently rerun; `target/insula/research/native_verification.log` records that run. Its report records exact host build commands, compiler versions, and the boundary between CPU native calls and GPU Torch execution.

GPU runtime: Python 3.12.3, Torch 2.13.0+cu132, CUDA 13.2, NVIDIA B200. Baseline kernel/server: ipykernel 6.31.0, Jupyter Server 2.17.0, debugpy 1.8.22. Emacs 29.3, pinned Emacs Jupyter commit `242fdc709ce0faa3b9ee81dcc48cfd791898e6b8`. Rootfs digest and full version observations are recorded in the baseline report. These are observed versions, not unpinned recommendations.

Two initial Emacs cache runs failed before evaluation because the new harness omitted the required buffer argument to `jupyter-eval-buffer`, then supplied nil. Inspection of the pinned implementation and working baseline established that it requires a buffer object. The harness was corrected to pass `(current-buffer)` and completed successfully. These harness failures are distinct from the genuine transport limitations recorded above.

## Remaining experiment designs, ordered by priority

These are queued qualification runs, not verified capabilities. None blocks the observed Python GPU milestone.

### Recovery adapter and bounded inspection

First exercise the production boundary before expanding languages. Preserve one frontend identity across socket replacement; keep pending execute IDs and source snapshots; disable automatic execute resend. Test an abrupt socket drop after send but before acknowledgement, and a client-process exit/restart with identity loaded from metadata. Pass only if a mutation counter remains one and recovered replies match the original request. If send outcome or completion is unavailable, surface unknown rather than retrying. Then test a real interrupted SSH tunnel, half-open socket detection, server restart, multiple frontends, and long-running jobs. Current probes use local shared networking and short executions.

Execution completion needs its matching reply and idle; debugger control requests need matching `debug_reply` while execution may stay busy. The protocol defines debugger requests on the control channel. [Jupyter messaging](https://jupyter-client.readthedocs.io/en/stable/messaging.html#debug-request).

For tensor inspection, use a large retained CUDA tensor and default metadata queries: shape, dtype, device, stride, allocation size. Require explicit user action for a bounded slice or reduction. Measure response size and latency, and verify no accidental full tensor rendering or GPU-to-host copy in the metadata path. Add watches while paused, but distinguish inspection expressions from expressions with side effects. Current experiments only inspected small bounded expressions; they establish frame correctness, not an inspection policy or performance bound.

### Native C++ dependencies: xeus-cpp versus xeus-cling

Create a tiny shared project library with a header and a stateful object. For each pinned kernel, record include/library configuration and load the dependency, retain an object, call it across cells, then edit and resubmit a function. Inspect old/new definitions and destruction. Repeat with the exact insula's LibTorch headers/libraries and a retained CUDA tensor; verify ABI, link paths, device, object identity, and kernel error after a deliberately incompatible library. Pass only with observed successful calls, controlled failures, and versioned source/library identity. The ctypes experiment does not substitute for this qualification.

### Native debugger lane: C++ / CUDA

Build a debug-symbol binary using the same project dependency; for CUDA include a kernel and known device state. Attach GDB/LLDB and cuda-gdb as applicable, set host/device breakpoints, inspect bounded locals, evaluate supported expressions, step, and continue. Drop only the frontend transport while stopped; then reattach and test whether process, stop reason, and selected frame survive. Separately test the debugger's explicit detach, since it may resume execution. Record which component owns the inferior and debugger process. Validate this independently from ipykernel/debugpy: arbitrary new compiled definitions and debugger expression evaluation have different contracts.

### Wasm / Emscripten

Pin an Emscripten toolchain and build a stateful module with an explicit dependency. In a retained Node or browser host, call the module repeatedly and confirm instance memory survives. Separately qualify side-module loading and an interpreter kernel such as a Wasm xeus build; these are different execution models. Implement the smallest external Emacs transport bridge and test frontend reconnect versus host exit/reload. Pass only if reconnection preserves the same runtime instance; host replacement must report lost state. Native shared libraries and CUDA pointers are not evidence for Wasm dependency compatibility. Toolchain absence is an environment gap, not proof of impossibility.

### Rust interactive evaluation

Pin Evcxr, add a small project crate and one external dependency, retain an object across evals, edit a function/type, and observe old-object behavior. Interrupt a long calculation and determine whether the runtime and objects remain usable. Reconnect Emacs through its Jupyter kernel and distinguish frontend loss from runtime/compiler reset. The Rust cdylib orchestration experiment qualifies Python calling Rust, not persistent Rust source evaluation.

## Decision supported by this evidence

Keep Python as the top-level linear workout driver and retain the expensive fixture in the kernel. Implement reliability at the connection/request/source-mapping boundary before choosing a new editor interface. Add explicit paused-frame evaluation alongside namespace evaluation. Keep cache invalidation and service teardown visible and small; do not turn the workout into a durable workflow system. Native interpreter kernels, native debuggers, and Wasm require their own lifecycle qualification even if they share connection metadata and inspection UI.
