"""2D cross-correlation reference and staged checks."""
# %%
import torch
from torch.nn import functional as F

# %%
def my_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
              stride: int = 1, padding: int = 0) -> torch.Tensor:
    if x.ndim != 4 or weight.ndim != 4 or x.shape[1] != weight.shape[1]:
        raise ValueError("need BCHW input and OCHW matching-channel weights")
    if not isinstance(stride, int) or stride < 1 or not isinstance(padding, int) or padding < 0:
        raise ValueError("stride must be positive and padding nonnegative")
    if weight.shape[0] < 1 or weight.shape[2] < 1 or weight.shape[3] < 1 or x.shape[2] + 2 * padding < weight.shape[2] or x.shape[3] + 2 * padding < weight.shape[3]:
        raise ValueError("invalid kernel/output geometry")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("bias must have one value per output channel")
    if x.dtype != weight.dtype or x.device != weight.device or bias is not None and (bias.dtype != x.dtype or bias.device != x.device):
        raise ValueError("dtype/device mismatch")
    padded = F.pad(x, (padding,) * 4) if padding else x
    kh, kw = weight.shape[-2:]
    patches = padded.unfold(2, kh, stride).unfold(3, kw, stride)
    out = torch.einsum("bihwjk,oijk->bohw", patches, weight)
    return out if bias is None else out + bias[None, :, None, None]

# %% [verifiers]
def verify_baseline_known_values(my_conv2d):
    x = torch.arange(9.).reshape(1, 1, 3, 3)
    w = torch.ones(1, 1, 2, 2)
    assert torch.equal(my_conv2d(x, w), torch.tensor([[[[8., 12.], [20., 24.]]]]))
    assert torch.equal(my_conv2d(x, w, torch.tensor([2.])), torch.tensor([[[[10., 14.], [22., 26.]]]]))


def verify_followup_layout_parity_gradients(my_conv2d):
    torch.manual_seed(22)
    base = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
    x = base.transpose(2, 3)
    w = torch.randn(4, 3, 2, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, dtype=torch.float64, requires_grad=True)
    got = my_conv2d(x, w, b, stride=2, padding=1)
    expected = F.conv2d(x, w, b, stride=2, padding=1)
    assert got.shape == expected.shape and torch.allclose(got, expected, atol=1e-11)
    got.square().sum().backward(retain_graph=True)
    grads = (base.grad.clone(), w.grad.clone(), b.grad.clone())
    base.grad.zero_(); w.grad.zero_(); b.grad.zero_()
    expected.square().sum().backward()
    assert all(torch.allclose(a, c, atol=1e-10) for a, c in zip(grads, (base.grad, w.grad, b.grad)))


def verify_production_invalid_geometry(my_conv2d):
    x, w = torch.ones(1, 2, 3, 3), torch.ones(4, 2, 2, 2)
    for args in ((x, w, None, 0, 0), (x, w, None, 1, -1), (x, torch.ones(4, 1, 2, 2)),
                 (x, torch.ones(4, 2, 4, 2)), (x, w, torch.ones(3))):
        try:
            my_conv2d(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid convolution input accepted")
    try:
        my_conv2d(x, w.double())
    except ValueError:
        pass
    else:
        raise AssertionError("mixed input and weight dtypes accepted")

# %%
from torch_judge import evaluate
if not evaluate("0022_conv2d_eval.py", my_conv2d):
    raise SystemExit(1)
