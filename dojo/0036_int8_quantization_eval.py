"""Three-stage reference for a per-channel INT8 storage linear layer."""
# %%
import torch
import torch.nn as nn

# %%
class Int8Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 2 or min(weight.shape) < 1
                or weight.dtype != torch.float32 or not torch.isfinite(weight).all()):
            raise ValueError("weight must be finite float32 [O,I] with positive dimensions")
        if bias is not None and (not isinstance(bias, torch.Tensor) or bias.shape != (weight.shape[0],)
                                 or bias.dtype != weight.dtype or bias.device != weight.device
                                 or not torch.isfinite(bias).all()):
            raise ValueError("bias must be finite float32 [O] on the weight device")
        scale = weight.abs().amax(dim=1, keepdim=True) / 127
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        self.register_buffer('weight_int8', quantized)
        self.register_buffer('scale', scale)
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[-1] != self.weight_int8.shape[1]
                or x.dtype != self.scale.dtype or x.device != self.scale.device or not torch.isfinite(x).all()):
            raise ValueError("x must be finite float32 [...,I] on the module device")
        output = x @ (self.weight_int8.to(self.scale.dtype) * self.scale).T
        return output if self.bias is None else output + self.bias

# %% [verifiers]
def verify_baseline_quantized_buffers(Int8Linear):
    weight = torch.tensor([[0., 1., -1.], [0., 0., 0.]])
    layer = Int8Linear(weight)
    assert isinstance(layer, nn.Module)
    assert layer.weight_int8.dtype == torch.int8
    assert torch.equal(layer.weight_int8[0], torch.tensor([0, 127, -127], dtype=torch.int8))
    assert torch.equal(layer.weight_int8[1], torch.zeros(3, dtype=torch.int8))
    assert {'weight_int8', 'scale'} <= dict(layer.named_buffers()).keys()


def verify_baseline_forward(Int8Linear):
    layer = Int8Linear(torch.tensor([[1., -1.], [0., 0.]]))
    output = layer(torch.tensor([[2., 3.], [4., -1.]]))
    assert output.shape == (2, 2) and torch.allclose(output, torch.tensor([[-1., 0.], [5., 0.]]), atol=1e-5)


def verify_followup_bias_and_leading_dims(Int8Linear):
    layer = Int8Linear(torch.tensor([[1., 0.], [0., 2.]]), torch.tensor([3., -1.]))
    assert torch.allclose(layer(torch.tensor([2., 4.])), torch.tensor([5., 7.]), atol=.02)
    x = torch.ones(2, 3, 2)
    assert layer(x).shape == (2, 3, 2)
    assert isinstance(layer.bias, nn.Parameter)


def verify_followup_per_row_error_bound(Int8Linear):
    weight = torch.tensor([[.01, -.006, .002], [100., -37., 8.]])
    layer = Int8Linear(weight)
    recovered = layer.weight_int8.float() * layer.scale
    assert layer.scale[1] > layer.scale[0] * 100
    assert torch.all((weight - recovered).abs() <= layer.scale / 2 + 1e-5)


def verify_production_validation_and_immutability(Int8Linear):
    for weight in (torch.empty(0, 2), torch.zeros(2), torch.tensor([[float('nan')]]), torch.ones(2, 2, dtype=torch.float64)):
        try:
            Int8Linear(weight)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted weight {weight}")
    weight, bias = torch.randn(3, 2), torch.randn(3)
    copies = weight.clone(), bias.clone()
    layer = Int8Linear(weight, bias)
    assert torch.equal(weight, copies[0]) and torch.equal(bias, copies[1])
    for x in (torch.ones(2, 3), torch.ones(2, 2, dtype=torch.float64), torch.tensor([[float('inf'), 0.]])):
        try:
            layer(x)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted x {x}")
    try:
        Int8Linear(weight, torch.zeros(2))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted malformed bias")


def verify_production_state_and_gradient(Int8Linear):
    weight = torch.tensor([[1., -2.], [.3, 0.]])
    layer = Int8Linear(weight, torch.tensor([.1, -.2]))
    state = layer.state_dict()
    copy = Int8Linear(torch.zeros_like(weight), torch.zeros(2))
    copy.load_state_dict(state)
    x = torch.tensor([[2., 1.]], requires_grad=True)
    assert torch.allclose(copy(x), layer(x))
    copy(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert copy.bias.grad is not None and torch.isfinite(copy.bias.grad).all()
    assert 'weight_int8' not in dict(copy.named_parameters())
    assert 'scale' not in dict(copy.named_parameters())

# %%
from torch_judge import evaluate
if not evaluate("0036_int8_quantization_eval.py", Int8Linear):
    raise SystemExit(1)
