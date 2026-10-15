"""Reference and cumulative checks for a linear vector-Jacobian product."""

# %%
import torch

# %%
def linear_vjp(x: torch.Tensor, w: torch.Tensor,
               grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = (x, w, grad_out)
    if (any(not isinstance(t, torch.Tensor) or t.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(t).all().item() for t in tensors)):
        raise ValueError("need finite float32 or float64 tensors")
    if (len({t.dtype for t in tensors}) != 1 or len({t.device for t in tensors}) != 1
            or x.ndim < 2 or w.ndim != 2 or grad_out.ndim != x.ndim
            or w.shape[0] < 1 or w.shape[1] < 1 or x.shape[-1] != w.shape[0]
            or grad_out.shape != (*x.shape[:-1], w.shape[1])):
        raise ValueError("incompatible linear VJP shapes, dtype, or device")
    grad_x = grad_out @ w.T
    grad_w = x.reshape(-1, x.shape[-1]).T @ grad_out.reshape(-1, grad_out.shape[-1])
    return grad_x, grad_w


# %% [verifiers]
def verify_baseline_hand_computed(linear_vjp):
    x = torch.tensor([[1., 2.], [3., 4.]])
    w = torch.tensor([[5., 6.], [7., 8.]])
    dy = torch.tensor([[1., 2.], [3., 4.]])
    dx, dw = linear_vjp(x, w, dy)
    assert torch.equal(dx, torch.tensor([[17., 23.], [39., 53.]]))
    assert torch.equal(dw, torch.tensor([[10., 14.], [14., 20.]]))


def verify_followup_reduce_all_leading_axes(linear_vjp):
    x = torch.arange(12., dtype=torch.float64).reshape(2, 3, 2)
    w = torch.tensor([[1., -1.], [2., 3.]], dtype=x.dtype)
    dy = torch.ones(2, 3, 2, dtype=x.dtype)
    dx, dw = linear_vjp(x, w, dy)
    assert dx.shape == x.shape and dw.shape == w.shape
    assert torch.equal(dx, torch.tensor([[[0., 5.]] * 3] * 2, dtype=x.dtype))
    assert torch.equal(dw, torch.tensor([[30., 30.], [36., 36.]], dtype=x.dtype))


def verify_followup_empty_leading_axis(linear_vjp):
    x, w, dy = torch.empty(0, 3), torch.ones(3, 2), torch.empty(0, 2)
    dx, dw = linear_vjp(x, w, dy)
    assert dx.shape == (0, 3) and torch.equal(dw, torch.zeros_like(w))


def verify_production_noncontiguous_matches_autograd(linear_vjp):
    x = torch.arange(24., dtype=torch.float64).reshape(2, 3, 4).transpose(0, 1)
    w = torch.arange(20., dtype=x.dtype).reshape(5, 4).T
    dy = torch.arange(30., dtype=x.dtype).reshape(5, 2, 3).permute(2, 1, 0)
    x, w = x.detach().requires_grad_(), w.detach().requires_grad_()
    # dy must match [3,2,5].
    before_x, before_w, before_dy = x.detach().clone(), w.detach().clone(), dy.clone()
    (x @ w).backward(dy)
    dx, dw = linear_vjp(x.detach(), w.detach(), dy)
    assert torch.allclose(dx, x.grad) and torch.allclose(dw, w.grad)
    assert torch.equal(x.detach(), before_x) and torch.equal(w.detach(), before_w)
    assert torch.equal(dy, before_dy)


def verify_production_reject_invalid_contract(linear_vjp):
    x, w, dy = torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 4)
    invalid = ((x, w, torch.ones(2, 3)), (x, torch.ones(5, 4), dy),
               (x.double(), w, dy), (x, w, dy.unsqueeze(0)),
               (torch.full((2, 3), float("inf")), w, dy),
               (torch.empty(2, 0), torch.empty(0, 4), dy))
    for args in invalid:
        try:
            linear_vjp(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid linear VJP contract")


# %%
from torch_judge import evaluate
if not evaluate("0049_linear_vjp_eval.py", linear_vjp):
    raise SystemExit(1)
