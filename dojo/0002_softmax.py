# %%
# 0002 · Softmax: normalize along a chosen axis without overflow
# Keep one answer cell and revise it as the constraints grow.
# Rule: do not use torch.softmax, F.softmax, or nn.Softmax in the answer.

# %%
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# Baseline: finite floating Tensor with at least one element along dim.
# Return exp(x - max(x)) / sum(exp(x - max(x))) with reductions along dim.
# Predict the probabilities for [0, 0], then inspect the overflow probe.
scores = torch.tensor([1000., 1001.])
print("raw exponential:", scores.exp())
print("shifted exponential:", (scores - scores.max()).exp())

# %% [answer]
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    cast_type = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_max_n1 = x.max(dim=dim, keepdim=True).values
    x_shifted_nd = x - x_max_n1
    x_exp_nd = x_shifted_nd.exp() if cast_type == x.dtype else x_shifted_nd.to(cast_type).exp()
    x_exp_sum_n1 = x_exp_nd.sum(dim=dim, keepdim=True)
    proba = x_exp_nd / x_exp_sum_n1
    if cast_type != x.dtype:
        proba = proba.to(x.dtype)
    return proba


# %%
check("0002_softmax_eval.py", stage="baseline")

# %%
# Follow-up: support positive or negative dim on batched tensors, including
# noncontiguous views. Preserve shape and dtype; each slice along dim sums to
# one. Row-wise constant shifts must not change probabilities.
batch = torch.arange(12., dtype=torch.float64).reshape(3, 4).T
print("noncontiguous shape/stride:", batch.shape, batch.stride())
check("0002_softmax_eval.py", stage="followup")

# %%
# Production: maintain autograd and input immutability. On a selected slice,
# the Jacobian-vector product obeys dL/dx = p * (g - sum(g*p)). The answer
# should never materialize the full Jacobian. Preserve dtype/device and match
# the framework reference even for logits around +/-1000.
check("0002_softmax_eval.py", stage="production")

# %%
# Inspect the actual eval fixtures, then run your current answer on one.
# These exploration cells never run during --candidate evaluation.
eval_data = get_eval_data("0002_softmax_eval.py")
case = eval_data["extreme_logits"]
case.purpose, case.inputs, case.expected

# %%
actual = my_softmax(**case.inputs)
result = compare(case, actual)
result.passed, result.diagnostics, result.actual, result.expected

# %%
# Edit the answer, rerun its cell, then replay. Every replay gets fresh inputs.
results = replay("0002_softmax_eval.py", my_softmax,
                 names=["extreme_logits", "noncontiguous_axis"])
failures = {name: result for name, result in results.items() if not result.passed}
failures

# %%
# Production extension: 70000 equal float16 logits. The shifted exponentials
# are all one, but their half-precision sum can overflow. Predict the answer,
# then inspect long_half_reduction and revise your accumulation dtype.
# Compare the separate softmax_shifted / softmax_wide cells in the eval script.
# Run explore_long_half_reduction() explicitly there; benchmarks are opt in.

# %%
# Backward derivation. With y = softmax(x) and z = sum(exp(x)):
#   y_i = exp(x_i) / z
#   dy_i/dx_j = y_i * (delta_ij - y_j)        # J = diag(y) - y y^T, symmetric
# Given upstream g = dL/dy, the chain rule sums over outputs i:
#   dL/dx_j = sum_i g_i * y_i * (delta_ij - y_j)
#           = y_j * g_j - y_j * sum_i g_i * y_i
#           = y_j * (g_j - <g, y>)
# So backward needs only y, g and one row-wise dot product <g, y>. The
# (n x n) Jacobian never exists: O(n) work and storage per slice, not O(n^2).
#
# Why this is the prelude to flash-attention: for O = P V with P = softmax(S),
#   dP = dO V^T,   and   <dP_i, P_i> = sum_j P_ij * (dO_i . V_j) = dO_i . O_i
# so the row-wise dot product collapses to D_i = rowsum(dO * O). It needs
# neither P nor dP, which is what lets flash-attn recompute P in tiles:
#   dS = P * (dP - D)

# %%
# Exercise: implement the backward pass by hand. Take the saved output y and
# upstream gradient g (same shape, any dim), return dL/dx. Rules: no autograd,
# no torch.softmax, no Jacobian matrix; one reduction along dim. Then, as a
# stretch, keep the reduction in float32 for float16/bfloat16 inputs.
def softmax_backward(y: torch.Tensor, g: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Compute the row-wise dot product <g, y> along dim
    dot_product = (g * y).sum(dim=dim, keepdim=True)
    # Compute the gradient dL/dx using the formula: dx = y * (g - <g, y>)
    dx = y * (g - dot_product)
    return dx

# %%
# Self-check against autograd on a noncontiguous view with extreme logits.
# Not part of --candidate evaluation; rerun after editing the exercise above.
def check_softmax_backward():
    x = (torch.randn(4, 6, dtype=torch.float64) * 500).T.requires_grad_()
    g = torch.randn_like(x)
    for dim in (0, 1, -1):
        ref = torch.autograd.grad(torch.softmax(x, dim=dim), x, g)[0]
        y = torch.softmax(x.detach(), dim=dim)
        torch.testing.assert_close(softmax_backward(y, g, dim), ref)
    return "softmax_backward matches autograd"

check_softmax_backward()

# %%
# GPU workout (needs a CUDA device; run it in the GPU Jupyter insula). Port the
# same two kernels to one program/block per row:
#   forward : online normalizer. Keep a running max m and sum l; when a new
#             element raises the max, rescale l *= exp(m_old - m_new). A second
#             pass writes exp(x - m) / l. Rows longer than one block need a loop.
#   backward: d = <g, y> as a row reduction, then dx = y * (g - d).
# Try Triton first (tl.load/tl.store with masks, tl.max, tl.sum), then CUDA C++
# (one block per row, strided per-thread loop, shared-memory tree reduction;
# the insula has no nvcc, so compile with NVRTC via torch.cuda._utils).
# Worked references: experiments/softmax_gpu/softmax_triton.py and softmax_cuda.py.
# Implement softmax_gpu below with either one, then run check_softmax_gpu.
def softmax_gpu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    raise NotImplementedError

# %%
def check_softmax_gpu(fn=None):
    if not torch.cuda.is_available():
        return "skipped: no CUDA device (start the GPU Jupyter insula)"
    fn = fn or softmax_gpu
    for dtype, tol in ((torch.float32, 1e-6), (torch.float16, 2e-3)):
        for shape, dim in (((5, 7), -1), ((3, 4, 9), 1), ((3, 70000), -1)):
            x = (torch.randn(shape, device="cuda") * 30).to(dtype).requires_grad_()
            ref = x.detach().clone().requires_grad_()
            g = torch.randn_like(x)
            y, y_ref = fn(x, dim), torch.softmax(ref.float(), dim).to(dtype)
            (y * g).sum().backward()
            (y_ref * g).sum().backward()
            torch.testing.assert_close(y, y_ref, atol=tol, rtol=tol)
            torch.testing.assert_close(x.grad.float(), ref.grad.float(), atol=10 * tol, rtol=10 * tol)
    return "softmax_gpu matches torch.softmax forward and backward"

check_softmax_gpu()
