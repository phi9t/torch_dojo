"""CUDA C++ reference for 0002 softmax (kernels live in softmax.cu): one 256-thread block per row.

Each thread folds a strided slice of the row into a running (max, sum) pair;
a shared-memory tree merges the pairs with the same rescaling rule as the
online normalizer; a second strided pass writes exp(x - m) / l. Backward
reduces <g, y> the same way, then writes y * (g - <g, y>).

The kernels are float32 and compiled with NVRTC through torch's private
torch.cuda._utils helpers, because the insula ships no nvcc. Other float
dtypes are widened to float32 around the kernels. Needs a CUDA device.
"""
from pathlib import Path

import torch
from torch.utils import cpp_extension

THREADS = 256

SOURCE_PATH = Path(__file__).with_name("softmax.cu")

_kernels = {}


def ensure_cuda_home():
    """torch's NVRTC helper wants CUDA_HOME for headers; the insula only has the pip layout."""
    if cpp_extension.CUDA_HOME and (Path(cpp_extension.CUDA_HOME) / "include" / "cuda_runtime.h").exists():
        return
    candidates = [Path(torch.__file__).parents[1] / "nvidia" / "cu13", Path("/usr/local/cuda")]
    for candidate in candidates:
        if (candidate / "include").is_dir():
            cpp_extension.CUDA_HOME = str(candidate)
            return
    raise RuntimeError("no CUDA headers found; set CUDA_HOME")


def _load():
    """Compile once per process; PTX targets the current device."""
    if not _kernels:
        ensure_cuda_home()
        from torch.cuda._utils import _cuda_load_module, _nvrtc_compile
        for name in ("softmax_fwd", "softmax_bwd"):
            ptx, mangled = _nvrtc_compile(SOURCE_PATH.read_text(), name)
            _kernels[name] = _cuda_load_module(ptx, [mangled])[mangled]
    return _kernels


def _rows(t, dim):
    moved = t.movedim(dim, -1)
    return moved, moved.reshape(-1, moved.shape[-1]).contiguous().float()


class _CudaSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        moved, x2d = _rows(x, dim)
        y2d = torch.empty_like(x2d)
        rows, cols = x2d.shape
        _load()["softmax_fwd"](grid=(rows, 1, 1), block=(THREADS, 1, 1), args=[x2d, y2d, cols])
        y = y2d.reshape(moved.shape).to(x.dtype).movedim(-1, dim)
        ctx.dim = dim
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        moved, y2d = _rows(y, ctx.dim)
        g2d = g.movedim(ctx.dim, -1).reshape(y2d.shape).contiguous().float()
        dx2d = torch.empty_like(y2d)
        rows, cols = y2d.shape
        _load()["softmax_bwd"](grid=(rows, 1, 1), block=(THREADS, 1, 1), args=[y2d, g2d, dx2d, cols])
        return dx2d.reshape(moved.shape).to(y.dtype).movedim(-1, ctx.dim), None


def softmax_cuda(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("softmax_cuda needs a CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"softmax_cuda supports float16/bfloat16/float32; got {x.dtype}")
    if x.dim() == 0:
        return torch.ones_like(x)
    return _CudaSoftmax.apply(x, dim)
