# Persistent session research probes

These are disposable experiments, not a production session client. Findings:
[runtime research](../../docs/research/persistent-python-gpu-sessions.md),
[Emacs evidence](../../docs/research/emacs-session-runtime.md), and
[integration recommendation](../../docs/research/emacs-session-integration.md),
[experiment design and validation](../../docs/research/session-experiment-design.md), and
[native orchestration](../../docs/research/native-orchestration-experiments.md).

Prerequisites: the selected Torch Dojo GPU rootfs, accessible NVIDIA GPU, and
the repository's Jupyter Server running at `127.0.0.1:8899`. Start the server
with `bash insula/jupyter_rootfs/run_server.sh` if it is absent.

From the repository root:

```bash
bash experiments/persistent-session/run_protocol_probe.sh
bash experiments/persistent-session/run_emacs_probe.sh
bash experiments/persistent-session/run_once_probe.sh
bash experiments/persistent-session/run_emacs_once_probe.sh
bash experiments/persistent-session/run_native_probe.sh
```

The first probe runs inside the GPU insula; the second runs batch Emacs in
the separate Emacs insula. Both create their own kernel, use the existing
token file without printing it, and delete their kernel on ordinary exit.
Neither stops the shared server or touches another kernel.

Evidence is written under `target/insula/research/{protocol,emacs}/`.
Repeated runs replace evidence in those directories. Do not run multiple
copies of the same probe concurrently. A forced termination can bypass
cleanup; consult the last recorded kernel ID before removing an orphan.

The protocol probe asserts successful behavior. The Emacs probe also records
known limitations, including an incomplete original request after reconnect.
An exit status of zero means the experiment completed with its expected
observations; it does not mean the current frontend is production ready.

The cache probes test linear workout reevaluation, successful-result reuse,
explicit invalidation, and resource lifecycle. `once_lab.py` is an experiment
helper, not a production API. The native probe needs host GCC/G++ and rustc;
it compiles tiny versioned libraries into target, then calls their CPU C ABI
from Python while retaining a CUDA model. Neither native CUDA nor an interpreter
kernel is qualified by that experiment. Additional evidence directories are
`target/insula/research/{once,emacs-once,native}/`.
