"""CUDA workout for 0002 softmax: load softmax.cu, compile it, launch it on torch tensors.

Run in the GPU insula:
    python experiments/softmax_gpu/workout.py [--compiler nvcc|nvrtc] [--rows 4096 --cols 8192] [--bench]
nvcc (default when installed) builds a cubin ahead of time; nvrtc compiles the source in-process
to PTX, which the driver JITs on load. Edit softmax.cu, rerun, and watch the checks.
"""
import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from torch.cuda._utils import _cuda_load_module, _nvrtc_compile

from softmax_cuda import SOURCE_PATH, THREADS, ensure_cuda_home


def _compile_nvrtc(source, names):
    ensure_cuda_home()
    kernels = {}
    for name in names:
        ptx, mangled = _nvrtc_compile(source, name)  # targets the current device's sm_XY
        kernels[name] = _cuda_load_module(ptx, [mangled])[mangled]
        print(f"      nvrtc {name}: {len(ptx)} bytes of PTX")
    return kernels


def _compile_nvcc(names):
    arch = "sm_" + "".join(map(str, torch.cuda.get_device_capability()))
    with tempfile.TemporaryDirectory() as tmp:
        cubin = Path(tmp) / "softmax.cubin"
        command = ["nvcc", f"-arch={arch}", "-cubin", str(SOURCE_PATH), "-o", str(cubin)]
        print("      " + " ".join(command))
        subprocess.run(command, check=True)
        image = cubin.read_bytes()
    print(f"      nvcc cubin: {len(image)} bytes")
    return _cuda_load_module(image, list(names))  # extern "C" keeps the kernel names unmangled


def compile_kernels(compiler="nvcc", names=("softmax_fwd", "softmax_bwd")):
    """Source text -> device code (nvcc cubin or NVRTC PTX) -> loaded module -> launchable kernels."""
    source = SOURCE_PATH.read_text()
    print(f"[1/5] loaded {SOURCE_PATH.name}: {len(source.splitlines())} lines")
    start = time.perf_counter()
    kernels = _compile_nvcc(names) if compiler == "nvcc" else _compile_nvrtc(source, names)
    print(f"[2/5] {compiler} compile + module load: {(time.perf_counter() - start) * 1e3:.0f} ms "
          f"on {torch.cuda.get_device_name()} (sm_{''.join(map(str, torch.cuda.get_device_capability()))})")
    return kernels


def launch_forward(kernels, x):
    """x: contiguous float32 CUDA tensor [rows, cols]. Tensors pass as device pointers, ints as int32."""
    y = torch.empty_like(x)
    rows, cols = x.shape
    kernels["softmax_fwd"](grid=(rows, 1, 1), block=(THREADS, 1, 1), args=[x, y, cols])
    return y


def launch_backward(kernels, y, g):
    dx = torch.empty_like(y)
    rows, cols = y.shape
    kernels["softmax_bwd"](grid=(rows, 1, 1), block=(THREADS, 1, 1), args=[y, g, dx, cols])
    return dx


def cuda_ms(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=8192)
    parser.add_argument("--compiler", choices=("nvcc", "nvrtc"),
                        default="nvcc" if shutil.which("nvcc") else "nvrtc")
    parser.add_argument("--bench", action="store_true", help="time against torch.softmax")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; start the GPU Jupyter insula")

    kernels = compile_kernels(args.compiler)
    x = torch.randn(args.rows, args.cols, device="cuda") * 30
    g = torch.randn_like(x)

    y = launch_forward(kernels, x)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, torch.softmax(x, dim=-1), atol=1e-6, rtol=1e-5)
    print(f"[3/5] forward [{args.rows}x{args.cols}] matches torch.softmax; "
          f"row sums in [{y.sum(-1).min().item():.6f}, {y.sum(-1).max().item():.6f}]")

    dx = launch_backward(kernels, y, g)
    ref = x.clone().requires_grad_()
    expected = torch.autograd.grad(torch.softmax(ref, -1), ref, g)[0]
    torch.testing.assert_close(dx, expected, atol=1e-5, rtol=1e-4)
    print("[4/5] backward matches autograd (dx = y * (g - <g, y>))")

    extreme = launch_forward(kernels, torch.tensor([[1000., 1001., -1000., 0.]], device="cuda"))
    torch.testing.assert_close(extreme, torch.softmax(torch.tensor([[1000., 1001., -1000., 0.]], device="cuda"), -1))
    print(f"[5/5] extreme logits stay finite: {extreme.tolist()[0]}")

    if args.bench:
        mine = cuda_ms(lambda: launch_forward(kernels, x))
        ref_ms = cuda_ms(lambda: torch.softmax(x, dim=-1))
        gb = 2 * x.numel() * 4 / 1e9
        print(f"forward: mine {mine:.3f} ms ({gb / mine * 1e3:.0f} GB/s) | torch.softmax {ref_ms:.3f} ms ({gb / ref_ms * 1e3:.0f} GB/s)")


if __name__ == "__main__":
    main()
