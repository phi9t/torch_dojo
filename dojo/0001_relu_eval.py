"""Reference and staged checks for an elementwise ReLU."""

# %%
import torch

# %%
def relu(x: torch.Tensor) -> torch.Tensor:
    return x * (x > 0).to(dtype=x.dtype)

# %% [verifiers]
def verify_baseline_values(relu):
    x = torch.tensor([-2., -1., 0., 1., 2.])
    assert torch.equal(relu(x), torch.tensor([0., 0., 0., 1., 2.]))


def verify_baseline_shape(relu):
    x = torch.tensor([[-3., 4.], [2., -1.]])
    y = relu(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.equal(y, torch.tensor([[0., 4.], [2., 0.]]))


def verify_followup_scalar_and_noncontiguous(relu):
    scalar = torch.tensor(-2., dtype=torch.float64)
    assert relu(scalar).shape == scalar.shape and relu(scalar).item() == 0
    x = torch.arange(-12., 12.).reshape(4, 6).T
    y = relu(x)
    assert y.shape == x.shape and torch.equal(y, torch.maximum(x, torch.zeros_like(x)))


def verify_followup_derivative(relu):
    x = torch.tensor([-2., 0., 3.], requires_grad=True)
    relu(x).sum().backward()
    assert torch.equal(x.grad, torch.tensor([0., 0., 1.]))


def verify_production_dtype_device_and_immutability(relu):
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for dtype in (torch.float16, torch.float32, torch.float64):
            x = torch.tensor([[-3., 0., 5.]], dtype=dtype, device=device, requires_grad=True)
            original = x.detach().clone()
            y = relu(x)
            assert y.shape == x.shape and y.dtype == dtype and y.device == x.device
            assert torch.equal(y, torch.tensor([[0., 0., 5.]], dtype=dtype, device=device))
            (y * torch.tensor([[2., 3., 4.]], dtype=dtype, device=device)).sum().backward()
            assert torch.equal(x.grad, torch.tensor([[0., 0., 4.]], dtype=dtype, device=device))
            assert torch.equal(x.detach(), original)


# %%
from torch_judge import evaluate
if not evaluate("0001_relu_eval.py", relu):
    raise SystemExit(1)
