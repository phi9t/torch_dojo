# Emacs persistent GPU session: runtime evidence

## Result

Batch Emacs in its own insula successfully evaluated a whole buffer and a
selected region against one persistent Python kernel in the GPU insula. A
CUDA model and tensor retained object identity after idle disconnection,
disconnection during execution, and attachment through the public
`jupyter-connect-server-repl` command. Submitted code was never replayed.

The same Emacs-origin experiment also submits a function region, registers
its exact source with `dumpCell`, hits a breakpoint inside it, reads frame
locals, evaluates a local CUDA tensor, disconnects/reconnects while paused,
steps and continues, then edits the source and hits its shifted breakpoint.
This uses research-only debugger request helpers, not a production debugger
interface.

The existing client does **not** recover the completion/output of an
execution interrupted by a real WebSocket disconnect in this experiment.
The kernel finishes the operation once, but the original Emacs request stays
incomplete. Workspace survival and request-result recovery are separate.

## Reproduction and versions

With the repository-owned Jupyter Server already running on loopback 8899:

```bash
bash experiments/persistent-session/run_emacs_probe.sh
python3 -m json.tool target/insula/research/emacs/evidence.json
```

The probe reads the token from `target/insula/shared/token` inside the
Emacs insula; it does not print it. It creates one kernel, attaches further
clients only to that kernel ID, restarts its own kernel once, and deletes
the kernel during cleanup. Logs and structured evidence are ignored
artifacts under `target/insula/research/emacs/`.

Observed versions: Emacs 29.3; emacs-jupyter commit
`242fdc709ce0faa3b9ee81dcc48cfd791898e6b8`; Python 3.12.3; PyTorch
2.13.0+cu132; CUDA runtime 13.2; ipykernel 6.31.0; Jupyter Server 2.17.0.
The kernel reports NVIDIA B200 and `/workspace/torch_dojo` as its working
directory. Emacs has no direct GPU projection; all CUDA execution occurs
in the server kernel.

## Experiments

| Scenario | Observed result |
|---|---|
| `jupyter-eval-buffer` creates `Linear(16,8).cuda()` and a `(4,16)` CUDA tensor | Execution completes successfully |
| `jupyter-eval-region` excludes deliberately failing surrounding lines, increments counter and runs the model | Counter 1; output `(4,8)` on CUDA; original tensor identity retained |
| Close/reopen transport while idle | Counter 1; model and tensor identities unchanged |
| Submit a three-second sleep and increment; close transport before completion; stay offline four seconds | Request not idle before or after offline interval |
| Reopen transport and wait ten seconds for original request | Still not idle; request only contains `status` and `execute_input`, no final stream/reply/idle |
| Inspect namespace using a new, explicitly submitted expression | Counter 2; original model/tensor identities unchanged: interrupted code completed exactly once |
| `jupyter-connect-server-repl` with existing kernel ID from a Python source buffer | Creates named REPL; source buffer associates with its client; public region evaluation asserts retained identities and count |
| Further model execution through the attached REPL client | Counter 2; output `(4,8)`; same original objects |
| Restart same server kernel ID and attach again | PID changes; tensor, model and counter globals are absent |
| Research-only control-channel `debug_request` from Emacs | `debugInfo` returns success, temporary source prefix, breakpoint/stopped-thread lists and rich-rendering capability |
| Submit function with public `jupyter-eval-region`, breakpoint at line 3 | Exact `dumpCell` source path matches stopped frame; locals include `x` and `local_tensor`; frame evaluation returns `((4,16), 'cuda:0', True)` |
| Close/reopen transport while paused | `debugInfo` retains stopped threads and breakpoint; refreshed frame evaluation returns `torch.Size([4,16])` |
| Step/continue | Poll until frame reaches line 4; continued function produces correct tensor sum and preserves model/tensor identities |
| Edit source to use `x+3` and insert a line | New temporary source path; breakpoint hits line 4 in edited source; frame expression confirms edited local tensor |

The recorded kernel PID changes on restart while
the server kernel ID remains unchanged. PID is local evidence here, not
a proposed portable generation identifier. The probe intentionally submits
fresh inspection expressions after reconnect; it never resubmits the
original mutation.

In the paused-reconnect case the original execute request **does** become
idle after continuing: execution remained suspended throughout the offline
interval, so its completion messages were produced after the connection
returned. That differs from the sleep experiment, whose completion occurred
offline and stayed unavailable to the original request.

## Concrete client limitations

The pinned websocket action implementation reverses the public action
names: `disconnect` closes and immediately recreates the socket, whereas
`connect` closes and clears it. A first experimental pass using those
commands appeared to preserve output because it had not stayed offline.
The final probe uses the existing I/O publisher's explicit `stop` and
`start` events and checks the unfinished request on both sides of the
offline interval. This is a research workaround, not a production fix.
[Pinned websocket action implementation](https://github.com/emacs-jupyter/jupyter/blob/242fdc709ce0faa3b9ee81dcc48cfd791898e6b8/jupyter-server-kernel.el#L311-L319).

Every call to `jupyter-api-kernel-websocket` creates a fresh
`jupyter-session` and places its ID in the URL. Reopening therefore changes
the client's server buffering key. This source behavior explains the
observed failure to restore buffered completion messages; the protocol
probe in the main research report independently tests server buffering.
[Pinned websocket creation](https://github.com/emacs-jupyter/jupyter/blob/242fdc709ce0faa3b9ee81dcc48cfd791898e6b8/jupyter-rest-api.el#L868-L887).

`jupyter-connect-server-repl` already supports attaching by existing kernel
ID and creates/associates a new frontend successfully. It is distinct from
resuming pending request tracking from a disconnected frontend.
[Public attachment command](https://github.com/emacs-jupyter/jupyter/blob/242fdc709ce0faa3b9ee81dcc48cfd791898e6b8/jupyter-server.el#L369-L400).

The probe adds a temporary routing override so `debug_request` uses the
control channel. The pinned client otherwise routes unrecognized requests
to shell. The successful experiment proves the Emacs message machinery can
carry debugger requests and evaluate Emacs-submitted source at breakpoints;
it does **not** provide breakpoint markers, stopped-line highlighting,
interactive stack/variable panes or a selected-frame evaluation command.
[Pinned channel routing](https://github.com/emacs-jupyter/jupyter/blob/242fdc709ce0faa3b9ee81dcc48cfd791898e6b8/jupyter-base.el#L494-L499).
The actual `kernel_info` reply contains no debugger capability field,
despite the successful `debugInfo` request, so absence of that field cannot
be the sole discovery check for this pinned environment.

Debugger replies need their own completion rule: the research helper waits
for `debug_reply` directly. The ordinary `jupyter-reply` helper waits for a
request to become idle and returned no usable debugger reply while source
execution was paused in an early experimental pass. Also, a successful
`next` reply acknowledges the step request before the stopped frame changes;
the final probe polls the source path and expected line instead of treating
that acknowledgment as completed execution. Production code should track
continued/stopped events, with state queries supporting reconnect recovery.

One nonfatal callback diagnostic occurs during restart/cleanup:
`Kernel connection could not be established`. The probe still exits zero
after its state assertions pass. Closure callbacks need explicit lifecycle
handling before production use. Initial probe failures due to a mistaken
message accessor were corrected before the final successful run.

## Minimum requirements suggested by this evidence

- Remember server endpoint, kernel ID, a separate kernel generation
  identity, and a stable frontend buffering/session identity.
- Preserve request IDs and pending request state when reopening a socket;
  distinguish unknown outcomes from verified completion.
- Attach to an existing kernel explicitly; never replay mutations to
  restore a frontend.
- Inspect namespace state with separately submitted commands and make
  that distinction visible to the user.
- Add explicit control-channel debugger request/event handling while
  retaining public region/buffer evaluation and REPL association.
- Detect restart independently of the server kernel ID and clear old
  workspace/frame/source assumptions.

The experiment validates batch invocation of real public commands,
buffer association, debugger frame evaluation and source registration.
It does not validate interactive keyboard behavior,
tunnel packet loss, frontend process death, notebook output rendering or
large-volume output. Those remain separate tests.
