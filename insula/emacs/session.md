# Persistent GPU REPL sessions

`bash insula/emacs/run.sh` loads the project session transport alongside Doom's Emacs Jupyter package. Keep the GPU Jupyter server running independently (`bash insula/jupyter_rootfs/run_server.sh` if absent). The repository remains writable under `/workspace/torch_dojo` in both insulas.

Use `M-x torch-dojo-run-server-repl` to start a `torch-dojo-gpu` kernel, or `M-x torch-dojo-connect-server-repl` to attach to an existing kernel by ID. These commands accept the same optional REPL arguments as their Emacs Jupyter counterparts. The server selection/authentication UI comes from Emacs Jupyter. To avoid entering the local token interactively, evaluate:

```elisp
(let* ((token (string-trim
               (with-temp-buffer
                 (insert-file-contents "/workspace/torch_dojo/target/insula/shared/token")
                 (buffer-string))))
       (auth `(("Authorization" . ,(concat "token " token))))
       (jupyter-api-authentication-method auth)
       (server (jupyter-server :url "http://127.0.0.1:8899" :auth auth)))
  (torch-dojo-run-server-repl server "torch-dojo-gpu" "Dojo" t nil t))
```

Once the source buffer is associated, use normal Emacs Jupyter evaluation commands: `jupyter-eval-buffer`, `jupyter-eval-region`, `jupyter-eval-defun`, and the REPL. Evaluation and rendering remain in Emacs Jupyter. In Lisp, `jupyter-eval-buffer` requires a buffer argument, for example `(jupyter-eval-buffer (current-buffer))`.

For an Emacs environment outside the insula, install/load Emacs Jupyter first, then load `insula/emacs/torch-dojo-session.el` from your checkout. The same commands work against a server reached through a tunnel; actual tunnel failure behavior remains to be qualified.

## Reconnection and outcomes

- `torch-dojo-session-disconnect` closes only the frontend connection. The kernel keeps its Python objects and running code.
- `torch-dojo-session-reconnect` reopens the transport with the same frontend ID. Every reopening path sends a fresh kernel-info query, never another copy of previous code. The command returns that query's request object.
- `torch-dojo-session-status` shows connection state, server kernel ID, frontend ID, and kernel generation separately. `checking` means the socket is open but generation verification is pending; new code is rejected until verification completes.
- A changed generation sets `state-lost`, even if the server kernel ID stayed the same. `torch-dojo-session-acknowledge-restart` explicitly accepts the new empty workspace. It does not restore or reevaluate anything.

For programmatic use, wait for the reconnect query's `kernel_info_reply` before submitting more code:

```elisp
(let ((request (torch-dojo-session-reconnect jupyter-current-client)))
  (jupyter-wait-until request "kernel_info_reply" (lambda (_message) t) 15))
```

Generation checking currently uses the shell channel. If the kernel is still executing or paused, verification can wait behind that execution. Do not mistake this for state loss or resend the original code. Paused-frame control-channel integration is a follow-up ticket.

`torch-dojo-session-request-state` takes a client and original execute request ID. It returns `pending`, `unknown`, `complete`, `error`, `state-lost`, or nil for an untracked ID. Completion requires both the matching execute reply and idle. On transport loss, incomplete requests become unknown. Matching buffered messages can reconcile them later. Restart may deliver a known interruption error for an old execution before the new generation appears; retain that known outcome. The newest 128 terminal outcomes are retained by default (`torch-dojo-session-terminal-history-limit`); evicted IDs return nil. Incomplete outcomes remain retained. Disconnected or state-lost submissions are rejected and never queued for later execution.

The server's output buffer is in memory, tied to frontend identity, and not a durable journal. Failure to recover a completion remains unknown. The transport never retries a send, including a send that fails after delivery may have happened. An explicit subsequent eval is a new user action and can repeat side effects.

Kernel shutdown is separate and explicit (`jupyter-shutdown-kernel` for the current client). Do not shut down an attached shared kernel merely to detach.

## Verification and current scope

```bash
bash insula/emacs/verify_startup.sh
bash insula/emacs/verify_session.sh
```

The terminal startup smoke test exercises real Doom initialization (batch Emacs skips it). The capsule provides a non-overwriting `~/doom` symlink to `/opt/dojo/doom` because image-generated Doom startup files abbreviated that source path under the build HOME. Local mutable state remains under `.emacs.d`.

The live ERT suite creates/deletes only its own kernels against the existing loopback server and uses batch Emacs in the separate Emacs insula. It checks offline output and original completion recovery, retained CUDA object identities, a counter incrementing exactly once, rejection without later queueing, restart detection without replay through public and inherited lifecycle operations, selected-kernelspec launch, bounded outcome history, and an injected send failure after real delivery. The original recovery test failed on the unmodified frontend before the project transport was added.

This implementation targets Emacs 29.3 and Emacs Jupyter commit `242fdc709ce0faa3b9ee81dcc48cfd791898e6b8`, Jupyter Server 2.17.0 and ipykernel 6.31.0. It does not globally advise or modify Emacs Jupyter. Its subclass supplies only the Dojo clients' transport.

Session identity/outcomes currently live in the Emacs process. Recovery after Emacs process death, durable metadata, large-output buffering, real SSH failures, interactive responsiveness, and debugger UI remain follow-ups. Authentication refresh requires explicit reconnect after updating the server credentials; this transport does not auto-reconnect on reauthentication. The socket handshake has a bounded ten-second wait. Readiness requires the matching fresh kernel-info reply; older buffered replies cannot release the generation gate. Existing rich-output code is reused but its binary-buffer variants are not qualified by these tests.

The baseline plan and tickets live in `.scratch/persistent-python-session/`. Research evidence is under `docs/research/`; the compiled-language explorations stay independent from this first Python implementation slice.
