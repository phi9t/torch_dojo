# Persistent Python GPU sessions: runtime research

Research date: 2026-10-06 (America/Los_Angeles).

## Result

The existing GPU Jupyter insula can preserve a live CUDA workspace through frontend disconnects. The same kernel also supports breakpoints, inspection of frame locals, frame evaluation, stepping, and reconnect while paused. These debugger results were measured with a direct protocol probe and a batch Emacs probe using research-only control-channel requests. They establish feasibility, not a finished Emacs debugger interface.

Reattaching to a kernel and recovering a frontend connection are different operations. The former can recover Python objects. The latter also needs the original frontend identity and pending-request records to recover output and execution completion. Code must never be resubmitted automatically when its outcome is uncertain.

The recommendation is to reuse Emacs Jupyter's evaluation and rendering behind an owned session adapter. Add explicit session recovery and Jupyter debugger messaging. Do not start a client rewrite or promise Dape integration until the concrete connection defects are resolved. See the [integration investigation](emacs-session-integration.md) and [Emacs runtime investigation](emacs-session-runtime.md).

## Environment and reproduction

The experiment used a locally built rootfs. Measured versions:

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| Torch | 2.13.0+cu132 |
| CUDA runtime reported by Torch | 13.2 |
| GPU | NVIDIA B200 |
| Jupyter Server | 2.17.0 |
| ipykernel | 6.31.0 |
| IPython | 9.15.0 |
| debugpy | 1.8.22 |
| websocket-client | 1.8.0 |
| Emacs | 29.3 |
| Emacs Jupyter commit | 242fdc709ce0faa3b9ee81dcc48cfd791898e6b8 |

Use the selected rootfs and an already running server at `127.0.0.1:8899`. If the server is absent, start it with `insula/jupyter_rootfs/run_server.sh`. The probes read `target/insula/shared/token` without printing it. They create and delete only their own kernels; they do not restart or stop the shared server.

```bash
bash experiments/persistent-session/run_protocol_probe.sh
bash experiments/persistent-session/run_emacs_probe.sh
```

The [protocol probe](../../experiments/persistent-session/protocol_probe.py) writes `target/insula/research/protocol/results.json` and `messages.json`. The second file contains received messages, including source and results; it is local evidence under ignored `target/`. The Emacs probe writes separate evidence under `target/insula/research/emacs/`. No additional production dependencies were installed.

## Protocol observations

The final protocol run completed with exit status 0. Each execution request was submitted once. The probe waits for both its matching execution reply and idle status, without assuming ordering across channels.

| Experiment | Observed result |
| --- | --- |
| Create persistent GPU state | Created `torch.nn.Linear(8, 8).cuda()` and an 8×8 CUDA tensor; recorded object IDs and tensor data pointer. |
| Idle disconnect/reconnect | Same model, tensor, and device allocation survived. A subsequent model forward pass worked. |
| Abrupt TCP drop during execution | Closed the underlying TCP socket without a WebSocket close frame or debugger detach. Reconnected after execution completed. |
| Same frontend identity after drop | Received `offline-output 1`, the original execution reply, and idle. The submission counter remained 1. |
| Different frontend identity | Python state survived, but the original offline output was not recovered. A later explicit inspection found the counter was 1. This inspection did not replay the original request. |
| Breakpoint in submitted code | Registered source using `dumpCell`, set a verified breakpoint at line 3, and stopped in `research_step`. |
| Locals and frame evaluation | `scopes`/`variables` included `local_tensor` and `x`. Frame evaluation returned `((8, 8), 'cuda:0', True)` for shape, device, and expected tensor contents. |
| Abrupt TCP drop while paused | Reconnection followed by `debugInfo` recovered stopped-thread and breakpoint state. A refreshed stack frame supported evaluation of `local_tensor.shape`. |
| Step/continue | Stepped from line 3 to line 4, continued to completion, and verified the original model/tensor IDs and data pointer. |
| Edited region source | Changed `x + 2` to `x + 3` and inserted a comment. `dumpCell` returned a different source path. A breakpoint at the new line 4 hit the new source and evaluation confirmed the changed tensor contents. |
| Kernel restart | REST restart retained the kernel ID but changed the message-header kernel session. `model` and `x` were absent afterward. |

An initial probe incorrectly required `kernel_info_reply.content.debugger` to be true. That optional field was absent in this environment. A successful `debugInfo` request and the subsequent debugger experiments established support directly. The corrected probe records the absent advertisement. A production capability check must account for this version behavior.

## Emacs observations

The Emacs probe ran in the separate Emacs insula. `jupyter-eval-buffer` created a CUDA model and tensor. `jupyter-eval-region` executed only a selected region with deliberately failing code before and after it. The resulting tensor had shape `(4, 8)` on CUDA.

Stopping and restarting kernel I/O preserved the model/tensor object IDs, both while idle and during execution. However, after the execution-time disconnect, the old request never reached idle in the frontend. Its retained messages contained only `status` and `execute_input`; its completion stream, execution reply, and idle message were absent. A later explicit state query found that the submitted code had completed once. The total submission counter was 2: the first region and the later disconnected execution.

The public `jupyter-connect-server-repl` command attached by kernel ID, created `*jupyter-repl[torch-dojo-emacs-research]*`, and associated the Python source buffer. Region evaluation through that REPL verified the original objects and unchanged submission counter. The resulting forward pass retained shape `(4, 8)`. The public command therefore supports live-state attachment today. It does not recover the original frontend request.

A research-only channel routing override sent debugger requests through the Emacs client's control channel. The full Emacs probe then registered exact buffer text, set a breakpoint at line 3, and asynchronously submitted the region. Its local CUDA tensor inspection returned `((4, 16), 'cuda:0', True)`. After stopping/restarting I/O while paused, `debugInfo` recovered the stopped threads and a refreshed frame supported evaluation. The probe stepped to line 4, continued, and explicitly verified the result plus original model/tensor identities. Editing the buffer changed its source path and a new line-4 breakpoint hit the edited code.

The paused execution's original request reached idle after continuation: its completion messages were produced after reconnection. This differs from the earlier execution whose completion occurred offline and was not recovered by Emacs.

Debugger responses need a different completion rule from execution. An initial Emacs helper used an idle-dependent reply wait and failed while the kernel was paused. The final helper waits for the matching `debug_reply`. The stepping experiment also demonstrated that a `next` acknowledgment can precede the actual frame transition; the probe waits for the expected stopped frame instead of treating the acknowledgment as completion. The pinned checkout does not provide a complete debugger API or UI.

An explicit restart changed the process PID and cleared the model, tensor, and submission counter. The protocol probe separately established a changed kernel message-header session despite an unchanged server kernel ID. Each run's exact IDs are recorded in its evidence file.

## What this proves, and what it does not

The protocol tests prove that the existing server/kernel stack can support both lanes in one Python process. The Emacs tests separately determine what the pinned frontend can currently deliver. A protocol success must not be reported as an implemented Emacs debugger.

Socket drops were local transport failures. We did not interrupt an actual SSH tunnel, test network half-open connections, introduce delayed packets, or test server/host failures. The drops test abrupt transport loss after the operating system reports closure; they do not establish failure-detection timing.

This was one frontend per owned kernel, with short outages and bounded output. It does not prove multi-client debugger arbitration, days-long sessions, durable output journaling, authentication renewal, or memory bounds for offline buffering. The tensors and model are small. Spark executors, distributed training workers, C++/CUDA device debugging, Rust, and Wasm were not exercised.

Frame evaluation can execute code and affect state. The shape/device inspection was bounded. Retrieving debugger variable representations can still invoke formatting and transfer/synchronize data; no performance or side-effect guarantee was measured.

## Minimum session contract

1. Store the endpoint and kernel ID separately from the frontend session ID. Temporary reconnection should retain the frontend ID used in the WebSocket query and request headers.
2. Track the kernel message-header session as its generation. A changed generation invalidates namespace expectations, debugger frames, and variable handles even when the kernel ID is unchanged.
3. Record each submission ID, exact source text, evaluation target, and observed completion. Transport errors do not prove that code failed to execute. Never resend an uncertain execution automatically.
4. Restore connection and debugger state before accepting new execution. Use `debugInfo` and fresh thread/frame queries after a paused reconnect. Do not rely on stale frame IDs.
5. Keep workspace evaluation on the shell channel and debugger operations on the control channel. The selected frame must be explicit while paused. Match debugger replies without waiting for kernel idle; step/continue acknowledgments do not prove that the execution transition finished.
6. Register each submitted source snapshot and retain its returned kernel source path plus buffer location/revision. A buffer edit must not silently move breakpoints on an older executing snapshot.
7. Report missing output separately from lost kernel state. Kernel survival does not imply recovery of a previous frontend's messages.
8. Keep user attachment separate from kernel ownership. Disconnecting an editor must not implicitly delete a kernel; explicit shutdown should be a separate action.

## Primary-source explanation

Jupyter message headers identify the kernel/client session, and parent headers correlate outputs with submissions. Kernel session changes indicate restarts. Shell and control channels have separate purposes. See the [Jupyter messaging specification](https://jupyter-client.readthedocs.io/en/stable/messaging.html).

Jupyter Server 2.17.0 buffers offline messages in memory by default. Its buffer recovery requires the same kernel/client session key. This explains the same-ID success and changed-ID output gap. The implementation uses an in-memory list and includes a note about adding a bounded queue. It is not a durable history service. See the [pinned kernel manager](https://github.com/jupyter-server/jupyter_server/blob/v2.17.0/jupyter_server/services/kernels/kernelmanager.py) and [pinned WebSocket connection](https://github.com/jupyter-server/jupyter_server/blob/v2.17.0/jupyter_server/services/kernels/connection/channels.py).

ipykernel provides debugger state recovery, submitted-code registration, workspace inspection, and frame operations through debugpy. The measured behavior fits these interfaces. See the [pinned debugger implementation](https://github.com/ipython/ipykernel/blob/v6.31.0/ipykernel/debugger.py) and [Jupyter debugger proposal](https://github.com/jupyter/enhancement-proposals/blob/master/47-jupyter-debugger-protocol/jupyter-debugger-protocol.md).

## Next decisions

Resolve the pinned Emacs client's connection identity, uncertain-send retry, and pending-request recovery before writing the production implementation spec. The complete batch Emacs breakpoint/frame/reconnect sequence is now feasible through experimental requests. Compare a small debugger interface against a Dape bridge and qualify interactive event handling, output rendering, and reconnect semantics. Neither UI option is selected as an implemented solution by this report.

Keep subsequent investigations queued: native C++ with project libraries and LibTorch; native C++/CUDA debugger attachment; Wasm lifetime/dependencies/attachment; Rust persistence/interruption. Their findings must not delay qualification of the Python session contract.
