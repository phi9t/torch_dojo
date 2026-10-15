"""Reference and staged checks for a trainable linear layer."""

# %%
import math
import torch

# %%
class SimpleLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, *, dtype=None, device=None):
        if (not isinstance(in_features, int) or isinstance(in_features, bool)
                or not isinstance(out_features, int) or isinstance(out_features, bool)
                or in_features <= 0 or out_features <= 0):
            raise ValueError("feature counts must be positive integers")
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, dtype=dtype, device=device) / math.sqrt(in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.weight.shape[1]:
            raise ValueError("input final dimension must equal in_features")
        return x @ self.weight.T + self.bias

# %% [verifiers]
def verify_baseline_parameters(SimpleLinear):
    torch.manual_seed(349)
    expected_weight = torch.randn(2, 3) / math.sqrt(3)
    torch.manual_seed(349)
    layer = SimpleLinear(3, 2)
    assert layer.weight.shape == (2, 3) and layer.bias.shape == (2,)
    assert layer.weight.is_leaf and layer.bias.is_leaf
    assert layer.weight.requires_grad and layer.bias.requires_grad
    assert torch.equal(layer.bias, torch.zeros(2))
    assert torch.allclose(layer.weight, expected_weight)


def verify_baseline_affine_map(SimpleLinear):
    layer = SimpleLinear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1., 2., 3.], [-1., 0., 1.]]))
        layer.bias.copy_(torch.tensor([4., -2.]))
    x = torch.tensor([[1., 2., 3.], [0., 0., 1.]])
    assert torch.equal(layer.forward(x), torch.tensor([[18., 0.], [7., -1.]]))


def verify_baseline_registered_parameters(SimpleLinear):
    layer = SimpleLinear(3, 2)
    assert isinstance(layer, torch.nn.Module), "SimpleLinear must subclass torch.nn.Module"
    assert isinstance(layer.weight, torch.nn.Parameter) and isinstance(layer.bias, torch.nn.Parameter)
    params = list(layer.parameters())
    assert len(params) == 2 and any(p is layer.weight for p in params) and any(p is layer.bias for p in params), \
        "weight and bias must be registered (use torch.nn.Parameter)"
    assert set(layer.state_dict()) == {"weight", "bias"}
    assert sum(p.numel() for p in params) == 3 * 2 + 2


def verify_followup_optimizer_updates(SimpleLinear):
    layer = SimpleLinear(3, 2)
    before = [p.detach().clone() for p in layer.parameters()]
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    layer(torch.randn(4, 3)).square().sum().backward()
    opt.step()
    assert all(not torch.equal(b, p) for b, p in zip(before, layer.parameters())), "optimizer did not update parameters"


def verify_followup_vector_and_prefix(SimpleLinear):
    layer = SimpleLinear(4, 3)
    for x in (torch.randn(4), torch.randn(2, 3, 4), torch.randn(5, 2, 4).transpose(0, 1)):
        y = layer.forward(x)
        assert y.shape == x.shape[:-1] + (3,)
        assert torch.allclose(y, x @ layer.weight.T + layer.bias)


def verify_followup_gradients(SimpleLinear):
    layer = SimpleLinear(3, 2)
    x = torch.randn(4, 3, requires_grad=True)
    layer.forward(x).square().sum().backward()
    assert all(t is not None and torch.isfinite(t).all() for t in (x.grad, layer.weight.grad, layer.bias.grad))


def verify_production_validation(SimpleLinear):
    for dims in ((0, 2), (2, 0), (-1, 2)):
        try:
            SimpleLinear(*dims)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid features {dims}")
    layer = SimpleLinear(3, 2)
    try:
        layer.forward(torch.randn(2, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted wrong input feature width")


def verify_production_dtype_device_and_immutability(SimpleLinear):
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        layer = SimpleLinear(3, 2, dtype=torch.float64, device=device)
        assert layer.weight.dtype == layer.bias.dtype == torch.float64
        assert layer.weight.device.type == layer.bias.device.type == device
        x = torch.randn(2, 3, dtype=torch.float64, device=device)
        original = x.clone()
        y = layer.forward(x)
        assert y.dtype == x.dtype and y.device == x.device
        assert torch.allclose(y, x @ layer.weight.T + layer.bias)
        assert torch.equal(x, original)


# %%
from torch_judge import evaluate
if not evaluate("0003_linear_eval.py", SimpleLinear):
    raise SystemExit(1)
