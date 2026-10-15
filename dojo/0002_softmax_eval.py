"""Reference and staged checks for numerically stable softmax."""

# %%
import torch

from functools import partial
from torch_judge import EvalCase, compare_values

# %%
# Reference 1: shifted reductions, the equation written in three steps.
# Contract: finite floating inputs, a nonempty reduction dimension. Qualifies
# baseline and followup; float32/float64 gradients also agree. Float16 long
# reductions can overflow their normalization mass. O(elements) time/storage.
def softmax_shifted(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    numerator = torch.exp(shifted)
    mass = numerator.sum(dim=dim, keepdim=True)
    return numerator / mass

# %%
# Reference 2: accumulate in float32 for float16/bfloat16, then restore dtype.
# Subtract in the wider dtype too: finite half logits can have a difference
# beyond the half range. Extra work/storage trades bandwidth for reliable
# long reductions. Complete algorithm; it does not call reference 1.
def softmax_wide(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    shifted = work - work.max(dim=dim, keepdim=True).values
    numerator = torch.exp(shifted)
    mass = numerator.sum(dim=dim, keepdim=True)
    return (numerator / mass).to(x.dtype)

# %%
# Reference 3: production-style. Same math as reference 2, but with in-place
# exp/divide on the shifted temporary (no extra full-size allocations), and a
# hand-written backward dL/dx = p * (g - sum(g*p)) that saves only the output
# and never builds the Jacobian. Backward also runs in float32 for half dtypes.
# Not covered: a fused single-pass kernel, double backward, fully -inf rows.
_LOW_PRECISION = (torch.float16, torch.bfloat16)

class _SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        work = x.float() if x.dtype in _LOW_PRECISION else x
        shifted = work - work.amax(dim=dim, keepdim=True)
        y = shifted.exp_()
        y /= y.sum(dim=dim, keepdim=True)
        y = y.to(x.dtype)
        ctx.dim = dim
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        y_wide = y.float() if y.dtype in _LOW_PRECISION else y
        g_wide = g.float() if g.dtype in _LOW_PRECISION else g
        grad = y_wide * (g_wide - (g_wide * y_wide).sum(dim=ctx.dim, keepdim=True))
        return grad.to(y.dtype), None

# Backward derivation (y = softmax(x), g = dL/dy, <.,.> sums along dim):
#   dy_i/dx_j = y_i * (delta_ij - y_j)
#   dL/dx_j   = sum_i g_i * y_i * (delta_ij - y_j) = y_j * (g_j - <g, y>)
# Only y, g and one row-wise dot product are needed; no (n x n) Jacobian.
# Flash-attn tie-in: for O = P V, <dP_i, P_i> = dO_i . O_i, so dS = P * (dP - D)
# with D = rowsum(dO * O), without ever holding a full row of P.
def softmax_production(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if not x.is_floating_point():
        raise TypeError(f"softmax expects a floating tensor, got {x.dtype}")
    if x.dim() == 0:
        return torch.ones_like(x)
    return _SoftmaxFunction.apply(x, dim)

# %%
# Opt in: run this function to see why max subtraction alone is insufficient.
# This is a limitation demonstration of the first reference's declared domain.
def explore_long_half_reduction():
    x = torch.zeros(70000, dtype=torch.float16)
    return {
        "shape": tuple(x.shape), "dtype": x.dtype, "device": x.device,
        "shifted_mass": softmax_shifted(x).float().sum().item(),
        "wide_mass": softmax_wide(x).float().sum().item(),
        "framework_mass": torch.softmax(x, dim=-1).float().sum().item(),
    }

# %%
# Opt in: compare complete versions only after correctness checks pass.
# CPU wall-clock timing includes Python overhead, with warmup and repeated
# measurements. No promise that widening is faster; its benefit is stability.
def benchmark_softmax(shape=(128, 2048), dtype=torch.float32, device="cpu"):
    from torch_judge.measurement import measure
    x = torch.randn(shape, dtype=dtype, device=device)
    expected = torch.softmax(x, dim=-1)
    implementations = {
        "shifted": softmax_shifted, "wide": softmax_wide,
        "production": softmax_production,
        "framework (learner helper forbidden)": lambda x: torch.softmax(x, dim=-1),
    }
    reports = {}
    for name, implementation in implementations.items():
        torch.testing.assert_close(implementation(x), expected)
        reports[name] = measure(lambda: implementation(x), device=device,
                                workload={"shape": shape, "dtype": str(dtype)})
    return reports

# %%
my_softmax = softmax_wide

# %% [verifiers]
def make_eval_data():
    # Independent framework oracle; the teaching references are not loaded.
    generator = torch.Generator().manual_seed(2026)
    examples = [
        ("equal_logits", "baseline", "Two equal logits split the mass.", torch.tensor([0., 0.]), -1, None),
        ("extreme_logits", "baseline", "Raw exponentials overflow; subtract the maximum.", torch.tensor([1000., 1001., 1002.]), -1, None),
        ("one_element", "baseline", "A one-element distribution is exactly one.", torch.tensor([9.]), -1, None),
        ("noncontiguous_axis", "followup", "Normalize a positive axis on a transposed view.", torch.randn(2, 3, 5, dtype=torch.float64, generator=generator).transpose(0, 2), 1, {"seed": 2026, "shape": (2, 3, 5), "transpose": (0, 2)}),
        ("gradient_logits", "production", "Use this leaf tensor for a weighted backward experiment.", torch.tensor([[2., -1., 3.], [-1000., -999., -1001.]], dtype=torch.float64, requires_grad=True), 1, None),
        ("long_half_reduction", "production", "70000 equal half logits need a wider normalization mass.", torch.zeros(70000, dtype=torch.float16), -1, {"length": 70000, "dtype": "float16"}),
    ]
    return {name: EvalCase(name=name, stage=stage, purpose=purpose,
                          inputs={"x": x, "dim": dim}, expected=torch.softmax(x.detach(), dim=dim),
                          comparison=partial(compare_values, atol=1e-7 if x.dtype == torch.float16 else 1e-6,
                                             rtol=1e-3 if x.dtype == torch.float16 else 1e-5),
                          generation=generation)
            for name, stage, purpose, x, dim, generation in examples}


def verify_baseline_values(my_softmax):
    for name in ("equal_logits", "extreme_logits"):
        case = make_eval_data()[name]
        torch.testing.assert_close(my_softmax(**case.inputs), case.expected)


def verify_baseline_one_element(my_softmax):
    case = make_eval_data()["one_element"]
    assert torch.equal(my_softmax(**case.inputs), case.expected)


def verify_followup_dimensions_and_views(my_softmax):
    x = make_eval_data()["noncontiguous_axis"].inputs["x"]
    for dim in (0, 1, -1, -2):
        y = my_softmax(x, dim=dim)
        ref = torch.softmax(x, dim=dim)
        assert y.shape == x.shape and y.dtype == x.dtype
        assert torch.allclose(y, ref, atol=1e-12, rtol=1e-12)
        assert torch.allclose(y.sum(dim=dim), torch.ones_like(y.sum(dim=dim)))


def verify_followup_shift_invariant(my_softmax):
    x = torch.tensor([[1., -2., 3.], [4., 5., -6.]], dtype=torch.float64)
    assert torch.allclose(my_softmax(x), my_softmax(x + torch.tensor([[1000.], [-1000.]])), atol=1e-12)


def verify_production_gradient_and_immutability(my_softmax):
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        x = make_eval_data()["gradient_logits"].inputs["x"].detach().to(device).requires_grad_()
        original = x.detach().clone()
        weights = torch.tensor([[1., 2., -2.], [3., 0., 1.]], dtype=x.dtype, device=device)
        y = my_softmax(x, dim=1)
        ref_x = x.detach().clone().requires_grad_()
        ref = torch.softmax(ref_x, dim=1)
        (y * weights).sum().backward()
        (ref * weights).sum().backward()
        assert y.dtype == x.dtype and y.device == x.device
        assert torch.allclose(y, ref, atol=1e-12, rtol=1e-12)
        assert torch.allclose(x.grad, ref_x.grad, atol=1e-12, rtol=1e-12)
        assert torch.equal(x.detach(), original)


def verify_production_long_half_reduction(my_softmax):
    case = make_eval_data()["long_half_reduction"]
    torch.testing.assert_close(my_softmax(**case.inputs), case.expected, atol=1e-7, rtol=1e-3)


# %%
from torch_judge import evaluate
if not evaluate("0002_softmax_eval.py", my_softmax):
    raise SystemExit(1)
