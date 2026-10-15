# Emacs integration for persistent Python GPU sessions

Research date: 2026-10-06 (America/Los_Angeles). This report contains source inspection, not runtime verification. The companion lifecycle/debugger experiments determine whether the proposed boundary is usable in practice.

## Implemented first slice

The project now provides a scoped [Dojo session transport](../../insula/emacs/session.md). Its live eight-test regression suite recovers original offline completion/output, rejects disconnected and uncertain-send replay, gates reconnect/restart on a fresh generation probe, and qualifies public launch plus inherited lifecycle operations. Terminal Doom startup also has a smoke test. The source observations below describe the original pinned client; stock clients remain unchanged. Interactive responsiveness, real tunnels, persistent metadata after Emacs exit, cache production support, and debugger UI remain follow-up tickets in `.scratch/persistent-python-session/`.

## Recommendation

Keep Emacs Jupyter's evaluation and rich-output machinery behind a project-owned session adapter, but qualify and fix its WebSocket lifecycle before relying on it for reconnect. Add Jupyter debugger messaging to that adapter rather than launching a separate debugpy process for the same kernel. Prototype a DAP-to-Jupyter bridge if Dape UI reuse proves cheaper than a small debugger interface; Dape is not an existing drop-in Jupyter frontend.

This recommendation is an architectural inference. The [runtime report](persistent-python-gpu-sessions.md) establishes live-state attachment through the public `jupyter-connect-server-repl`, an Emacs completion/output gap after reconnect, and a complete experimental control-channel breakpoint/frame/reconnect sequence from batch Emacs. This supports reuse of the transport, but interactive debugger UI and production recovery remain unqualified.

## Versions inspected

The selected rootfs is `dbf415074cefc888ded2e20ff96b8be5a44c5aad3356395b9044bec4dfeae21a`. Its contract records image ID `sha256:dbf415074cefc888ded2e20ff96b8be5a44c5aad3356395b9044bec4dfeae21a` and base `ghcr.io/pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime@sha256:7492928e093d67276716440161f694a0e4ea27796d599d64b23cb76cdd665e71`. These are inspected build identities, not GPU-runtime results.

The [Containerfile](../../insula/rootfs/Containerfile) pins Jupyter Server 2.17.0, ipykernel 6.31.0, Jupyter Client 8.6.3, websocket-client 1.8.0, and Doom commit `8b475fcb2574286d7f4413c22cc7902d80d21ced`. Emacs is an apt `emacs-nox` dependency rather than an explicitly version-pinned build. The installed Emacs Jupyter checkout is commit `242fdc709ce0faa3b9ee81dcc48cfd791898e6b8`, determined by `git rev-parse HEAD` inside the selected rootfs. The Doom package declaration itself only says `(package! jupyter)`.

Dape source inspected on this date advertises version 0.27.1; the upstream HEAD resolved to `a5dcf2d086a91fbe6e600b6de482d8f680f4410c`. Dape is not established as an installed dependency of this rootfs. Its source requires Emacs 29.1 and jsonrpc 1.0.25. [Dape source](https://github.com/svaante/dape/blob/a5dcf2d086a91fbe6e600b6de482d8f680f4410c/dape.el)

## Existing evaluation and connection machinery

The current [scenario adapter](../../insula/emacs/scenario.el) creates a fresh `torch-dojo-gpu` kernel. Its execution wrapper gathers messages using `jupyter-messages` with a 60-second timeout before invoking result callbacks. Callback-shaped arguments do not make that wrapper an incremental streaming API. It handles `stream`, plain/rich display data, and errors, and marks `jupyter-timeout-before-idle` as unknown. It has no durable attachment record or debugger context. This describes source behavior, not successful execution.

Emacs Jupyter already has message publication/subscription, client-bound handlers, request correlation, and rich output infrastructure. Reuse these layers before rewriting language-specific rendering or completion. Request APIs can be subscribed without making the scenario's blocking wait wrapper the production API. [Pinned client source](https://github.com/emacs-jupyter/jupyter/blob/242fdc709ce0faa3b9ee81dcc48cfd791898e6b8/jupyter-client.el), [pinned message source](https://github.com/emacs-jupyter/jupyter/blob/242fdc709ce0faa3b9ee81dcc48cfd791898e6b8/jupyter-messages.el)

Three concrete lifecycle hazards were found:

- In the server kernel action handler, `disconnect` closes and recreates the WebSocket, whereas `connect` closes it and sets it to nil. This contradicts the public client operation names. Treat it as a suspected implementation defect requiring a focused runtime test.
- The send path catches `websocket-closed`, reopens the socket, and invokes the send function again. This is a transport retry, not a guarantee of safe execution delivery. The project adapter must prohibit automatic replay of execution whose delivery is uncertain.
- Close/error handling contains unfinished status/retry paths. Client `connected-p` checks that an IO slot is bound, and heartbeat checking returns true; neither should be used alone as connection-health evidence.

[Server kernel implementation](https://github.com/emacs-jupyter/jupyter/blob/242fdc709ce0faa3b9ee81dcc48cfd791898e6b8/jupyter-server-kernel.el#L209-L328), [client lifecycle implementation](https://github.com/emacs-jupyter/jupyter/blob/242fdc709ce0faa3b9ee81dcc48cfd791898e6b8/jupyter-client.el#L370-L451)

Each `jupyter-api-kernel-websocket` call allocates a fresh `jupyter-session` and uses its ID as the `session_id` URL query. Reopening a socket therefore changes the frontend identity by default. A reconnect adapter must deliberately preserve this identity when recovering offline messages. [Pinned REST implementation](https://github.com/emacs-jupyter/jupyter/blob/242fdc709ce0faa3b9ee81dcc48cfd791898e6b8/jupyter-rest-api.el#L868-L886)

Jupyter Server 2.17.0's offline buffer is in memory and keyed to a client session; it is not a durable execution journal. Kernel ID, frontend session ID, and kernel message-header session identify different things. Keep all three separate. Reattaching to a live kernel can recover its namespace even if frontend identity prevents recovery of old output. [Pinned server buffering implementation](https://github.com/jupyter-server/jupyter_server/blob/v2.17.0/jupyter_server/services/kernels/kernelmanager.py)

## Debugger integration

The pinned Emacs Jupyter checkout contains no `debug_request` or `debug_event` implementation found by searching its Lisp sources. Its server IO publisher accepts an arbitrary channel, message type, content, and ID. The companion runtime probe overrides request-channel selection for `debug_request` and waits for its matching `debug_reply`, independently of kernel idle. It successfully carries the full experimental breakpoint/frame/reconnect sequence. This confirms a transport extension point without a second WebSocket stack; it does not establish a production debugger API.

Jupyter wraps DAP requests/replies in control-channel messages and debugger events in IOPub messages. Control traffic avoids queuing behind a paused execution. Jupyter adds state retrieval for later clients and code registration for submitted cells. The protocol defines an optional debugger field in `kernel_info_reply`; this environment omitted it despite successful debugger requests. Capability discovery needs a version-aware fallback. [Jupyter debugger protocol](https://github.com/jupyter/enhancement-proposals/blob/906afd95af965994a3c7cea8af4f01723daa7716/47-jupyter-debugger-protocol/jupyter-debugger-protocol.md)

ipykernel 6.31.0 implements `debugInfo`, `dumpCell`, workspace `inspectVariables`, and `richInspectVariables`, and forwards ordinary debugger operations to debugpy. This permits a shared kernel with distinct namespace and paused-frame operations. It does not imply inspection is always cheap: formatting objects and evaluating expressions can run user code or synchronize GPU work. Default tensor inspection should return metadata and a bounded explicit preview. [Pinned ipykernel debugger](https://github.com/ipython/ipykernel/blob/v6.31.0/ipykernel/debugger.py)

Source mapping must retain the exact submitted text, its buffer revision/start line, and the kernel's generated source path. Editing a buffer does not edit code already running in a paused frame. A breakpoint display must distinguish an old submitted snapshot from the current buffer. Qualify this using two versions of a submitted function, not just a file-based breakpoint.

## Dape reuse options

Dape's connection class subclasses `jsonrpc-process-connection`. Its request API sends asynchronous JSONRPC requests after conversion to DAP; stack, frame, variable, and evaluation UI uses that connection state. Ordinary configuration supports process or TCP adapters, not Jupyter WebSocket envelopes. [Pinned Dape source](https://github.com/svaante/dape/blob/a5dcf2d086a91fbe6e600b6de482d8f680f4410c/dape.el#L1397-L1583)

The most contained reuse experiment is a local DAP bridge: expose a conventional DAP endpoint to Dape, translate requests into Jupyter control messages, translate replies/events back, and explicitly handle Jupyter code registration and state recovery. This is a proposal, not an available component. A direct connection subclass may avoid a bridge process but depends more heavily on Dape internals. Neither option should own kernel shutdown implicitly.

Start with direct Jupyter debugger requests to prove the protocol behavior. Then estimate the bridge against the minimal UI needed. Preserve the interactive REPL independently of debugger attach/disconnect. Defer native GDB/CUDA transport until Python's session contract is established; it can later share user-facing session selection without pretending its lifetime semantics match Jupyter.

## Required evidence before a production spec

1. Real Emacs execution subscriptions stream results while the editor remains usable.
2. Socket loss is visible and cannot cause uncertain code to be resent.
3. Reattach restores the original kernel and frontend identities; a kernel restart invalidates inspection/frame references.
4. While paused, control requests return frame locals and evaluate in the selected frame without ordinary shell execution.
5. `debugInfo` restores stopped state after reconnect, followed by refreshed threads/frames.
6. Region edits generate distinct source snapshots with correct breakpoint locations.
7. Dape bridge/subclass feasibility is demonstrated separately if UI reuse is selected.

The core session adapter should own identity, submission records, lifecycle state, and evaluation target. Backends should own their execution and inspection semantics. Existing Emacs Jupyter can remain the evaluation backend only after the hazards above are bounded by observed tests.
