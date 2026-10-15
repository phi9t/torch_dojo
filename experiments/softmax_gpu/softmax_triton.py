"""Triton reference for 0002 softmax: forward and backward kernels, one program per row.

Forward uses an online normalizer (running max m, running sum l rescaled by
exp(m_old - m_new) whenever the max grows), then a second pass writes
exp(x - m) / l. Backward is dx = y * (g - <g, y>) with the dot product
accumulated in float32. Supports float16/bfloat16/float32, any dim, and rows
longer than one block via a loop. Needs a CUDA device and triton.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_fwd_kernel(x_ptr, y_ptr, n_cols, x_row_stride, y_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * x_row_stride
    y_row = y_ptr + row * y_row_stride
    offs = tl.arange(0, BLOCK)
    m = tl.full((), float("-inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    # Pass 1: online max / normalizer, rescaling l whenever the max grows.
    for start in range(0, n_cols, BLOCK):
        x = tl.load(x_row + start + offs, mask=start + offs < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    # Pass 2: write normalized probabilities.
    for start in range(0, n_cols, BLOCK):
        mask = start + offs < n_cols
        x = tl.load(x_row + start + offs, mask=mask, other=float("-inf")).to(tl.float32)
        tl.store(y_row + start + offs, (tl.exp(x - m) / l).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softmax_bwd_kernel(y_ptr, g_ptr, dx_ptr, n_cols, row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    y_row = y_ptr + row * row_stride
    g_row = g_ptr + row * row_stride
    dx_row = dx_ptr + row * row_stride
    offs = tl.arange(0, BLOCK)
    d = tl.zeros((), tl.float32)  # <g, y>
    for start in range(0, n_cols, BLOCK):
        mask = start + offs < n_cols
        y = tl.load(y_row + start + offs, mask=mask, other=0.).to(tl.float32)
        g = tl.load(g_row + start + offs, mask=mask, other=0.).to(tl.float32)
        d += tl.sum(y * g, axis=0)
    for start in range(0, n_cols, BLOCK):
        mask = start + offs < n_cols
        y = tl.load(y_row + start + offs, mask=mask, other=0.).to(tl.float32)
        g = tl.load(g_row + start + offs, mask=mask, other=0.).to(tl.float32)
        tl.store(dx_row + start + offs, (y * (g - d)).to(dx_ptr.dtype.element_ty), mask=mask)


def _block(n_cols):
    return min(triton.next_power_of_2(n_cols), 4096)


class _TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        moved = x.movedim(dim, -1)
        x2d = moved.reshape(-1, moved.shape[-1]).contiguous()
        y2d = torch.empty_like(x2d)
        rows, cols = x2d.shape
        _softmax_fwd_kernel[(rows,)](x2d, y2d, cols, x2d.stride(0), y2d.stride(0), BLOCK=_block(cols))
        y = y2d.reshape(moved.shape).movedim(-1, dim)
        ctx.dim = dim
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        y2d = y.movedim(ctx.dim, -1).reshape(-1, y.shape[ctx.dim]).contiguous()
        g2d = g.movedim(ctx.dim, -1).reshape(y2d.shape).contiguous()
        dx2d = torch.empty_like(y2d)
        rows, cols = y2d.shape
        _softmax_bwd_kernel[(rows,)](y2d, g2d, dx2d, cols, y2d.stride(0), BLOCK=_block(cols))
        return dx2d.reshape(y.movedim(ctx.dim, -1).shape).movedim(-1, ctx.dim), None


def softmax_triton(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("softmax_triton needs a CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"softmax_triton accumulates in float32; got {x.dtype}")
    if x.dim() == 0:
        return torch.ones_like(x)
    return _TritonSoftmax.apply(x, dim)
