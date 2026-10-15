# Python orchestrating retained C++ and Rust modules

Observed on 2026-10-07. This is a disposable experiment, not a production binding or a benchmark.

## Result

An owned GPU Jupyter kernel retained a CUDA model, CUDA tensor, CPU tensor, C++ object, and Rust object across separate evaluations and frontend reconnection. Both native modules loaded a separate project shared-library dependency and modified the same CPU tensor storage in place. Python then copied that tensor to CUDA and ran the retained model. Five scenarios passed.

This supports Python as the interactive orchestration layer while compiled modules own useful state underneath. It does **not** establish native C++/Rust GPU tensor integration: the tested native calls operated on CPU `float32` storage. The GPU work was PyTorch execution after an explicit CPU-to-GPU copy.

## Reproduce and evidence

From the repository root, with the existing server on loopback port 8899:

```bash
bash experiments/persistent-session/run_native_probe.sh
```

The script uses host compilers, writes all libraries and build evidence under `target/insula/research/native`, then enters the GPU Jupyter insula and creates an owned `torch-dojo-gpu` kernel. It reads the existing server token without printing it, issues each code submission once, and deletes only its owned kernel. It never restarts the shared server. Timeout: 120 seconds for the kernel probe.

- Source: `experiments/persistent-session/native_probe.py`, `native_probe.cpp`, `native_probe.rs`, and `native_probe_dependency.c`.
- Exact build commands: `experiments/persistent-session/run_native_probe.sh`.
- Compiler versions: `target/insula/research/native/build.json`.
- Assertions and observed output: `target/insula/research/native/results.json`.
- Raw Jupyter messages: `target/insula/research/native/messages.json`.
- Initial run output: `target/insula/research/native_run.log`.

These ignored target artifacts are local evidence; rerunning replaces them.

| Component | Observed version or availability |
|---|---|
| Host GCC / G++ | Debian 12.2.0-14+deb12u1 |
| Host Rust | rustc 1.95.0 (59807616e, 2026-04-14) |
| Insula Python | 3.12.3 |
| Insula Torch | 2.13.0+cu132 |
| CUDA runtime reported by Torch | 13.2 |
| GPU | NVIDIA B200 |
| Compilers on the insula PATH | `g++`, `rustc`, and `nvcc` absent |
| Insula libc | Ubuntu GLIBC 2.39-0ubuntu8.7 |

The tiny host-built libraries loaded successfully in this rootfs. This result does not qualify arbitrary host-built extensions or their C++/libc ABI compatibility.

## Experimental design and observations

| Claim and experiment | Pass criterion | Observed result |
|---|---|---|
| Initialize CUDA workspace and both native objects | CUDA model/tensor and non-null opaque native handles coexist in one kernel | Passed; Torch CUDA execution on B200 |
| Run the same short workout twice | Native counters advance once per explicit submission; model identity, CPU/GPU data pointers, and native handles remain unchanged | Passed; counters 1 then 2; CPU values 2 then 4; CUDA result shape 4×4 |
| Abruptly drop frontend TCP, reconnect, evaluate again | Same objects remain; third explicit workout produces counters 3 and CPU value 6 | Passed; no code was automatically resubmitted |
| Load separately named version-2 artifacts beside version 1 | Old objects retain old version/counters; newly created objects use new behavior | Passed; versions [1,1] and [2,2] coexist, old counters remain 3, new counters become 1, CPU value becomes 10 |
| Close native objects explicitly | Double close is harmless at Python wrapper level; later calls are rejected; GPU objects retain identity | Passed for all four objects |

Both C++ and Rust use the same separately compiled `libdojo_dependency.so` through normal dynamic linking. The module artifacts use `$ORIGIN` library search paths. This is a real dependency-loading test for these small libraries, not a dependency ecosystem qualification.

## Binding boundary

The Python wrapper uses `ctypes.CDLL`, explicit `argtypes`/`restype`, opaque handles, and an explicit destroy function. This is the standard C ABI path supported by Python's foreign-function interface. The native functions require valid pointers, correct sizes, and live objects; Python's documentation explicitly notes that invalid foreign calls can crash the process. The wrapper checks CPU device, contiguous layout, and `float32` before exposing a tensor data pointer. [Python ctypes documentation](https://docs.python.org/3.12/library/ctypes.html)

Rust is compiled as a `cdylib`, with exported `extern "C"` functions and unmangled names. Those are the documented mechanisms for producing a system shared library for another language. Handles are allocated with `Box::into_raw` and released through the Rust destroy function; callers never free them through C++. [Rust linkage reference](https://doc.rust-lang.org/reference/linkage.html), [Rust FFI guidance](https://doc.rust-lang.org/nomicon/ffi.html?highlight=C)

The experiment deliberately uses neither pybind11 nor PyO3, does not release Python's ownership of the tensor storage, and keeps all native calls synchronous. It does not test autograd, asynchronous native workers, callbacks, exceptions/panics across the ABI, native-memory safety, or teardown during an active call. Counter mutations make repeated evaluations observable; they are intentional workout operations rather than replay suppression.

Loading a distinct artifact version leaves existing handles bound to their original library. This observed coexistence offers a practical research pattern: rebuild under a new artifact name, explicitly create new objects, then explicitly dispose of old objects. In-place `.so` replacement, module unloading, and ABI migration of retained objects were **not tested**. In particular, this result is not evidence that existing native objects adopt edited source automatically.

## Subsequent experiments, with concrete acceptance criteria

These remain queued and should not block the persistent Python milestone.

| Direction | Disposable experiment | Required evidence |
|---|---|---|
| pybind11 / PyO3 bindings | Bind the same stateful fixture through the intended project binding, keep it in the GPU kernel, rerun workout regions, raise native errors, close explicitly | Retained identity, correct type conversion and error propagation, demonstrated ownership; separate versioned artifacts for edits |
| xeus-cpp / xeus-cling | Load this project dependency with native headers, retain a C++ object, repeat calls, edit function/class definitions, reconnect | Exact supported edits and rejection behavior, symbol/include/library resolution, old object behavior after edits; do not assume arbitrary redefinition |
| LibTorch | Build a tiny versioned operator with the deployed Torch headers/libraries; pass a tensor, retain an object, edit/rebuild, repeat | Torch ABI/toolchain versions, CPU and CUDA behavior, tensor ownership, old/new artifact behavior; execute on the actual GPU |
| C++/CUDA debugger lane | Pause an owned process in CPU code, then CUDA code, inspect local/tensor state, disconnect frontend and reconnect without a detach request | Stable process/stopped state where supported; distinguish frontend transport loss, debugger detach, and target termination |
| Wasm/Emscripten | Build the dependency for Wasm, keep a host runtime alive, repeatedly call exports, replace only the frontend, then terminate/recreate the host | Exported-call state retention versus actual language evaluation, Wasm dependency linking, external Emacs bridge, loss on host replacement |
| Rust interactive runtime | Evaluate retained state and a dependency in Evcxr, edit/redefine, interrupt a long-running expression, reconnect | Native Rust evaluation semantics and interruption/lifetime constraints; ctypes success does not establish Evcxr behavior |

xeus-cling documents include/library search paths and explicit `#pragma cling load` for native libraries; that makes the dependency experiment concrete, but it does not establish compatibility with LibTorch or repeated edits. [xeus-cling build options](https://xeus-cling.readthedocs.io/en/latest/build_options.html)

xeus-cpp documents native and Wasm builds; qualify them separately. Emscripten's main/side module linking is a Wasm build path with its own restrictions, not permission to load native `.so` dependencies in a Wasm host. [xeus-cpp documentation](https://xeus-cpp.readthedocs.io/en/latest/), [Emscripten dynamic linking](https://emscripten.org/docs/compiling/Dynamic-Linking.html)

Torch's C++/CUDA extension documentation requires an appropriate compiler and, for CUDA extensions, a separately installed CUDA toolkit including `nvcc`. The current runtime insula's absent compiler commands are a concrete provisioning gap for that experiment; no toolkit or compiler was installed during this probe. [Torch extension API](https://docs.pytorch.org/docs/stable/cpp_extension.html), [custom C++/CUDA operator tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
