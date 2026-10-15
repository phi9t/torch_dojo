"""LoRA reference and staged checks."""
# %%
import math
import torch
from torch import nn
from torch.nn import functional as F

# %%
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        if any(not isinstance(v, int) or v <= 0 for v in (in_features, out_features, rank)) or not math.isfinite(alpha):
            raise ValueError("positive dimensions and finite alpha required")
        self.linear = nn.Linear(in_features, out_features)
        self.linear.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.linear.in_features:
            raise ValueError("last dimension must equal in_features")
        return self.linear(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

# %% [verifiers]
def verify_baseline_frozen_zero_update(LoRALinear):
    layer = LoRALinear(8, 5, 3)
    assert isinstance(layer, nn.Module) and isinstance(layer.linear, nn.Linear)
    assert isinstance(layer.lora_A, nn.Parameter) and layer.lora_A.shape == (3, 8)
    assert isinstance(layer.lora_B, nn.Parameter) and layer.lora_B.shape == (5, 3)
    assert not any(p.requires_grad for p in layer.linear.parameters())
    assert torch.count_nonzero(layer.lora_B) == 0
    x = torch.randn(2, 8)
    assert torch.allclose(layer(x), layer.linear(x))


def verify_followup_gradients(LoRALinear):
    layer = LoRALinear(4, 3, 2)
    x = torch.randn(2, 5, 4, requires_grad=True)
    layer(x).square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.lora_B.grad is not None and torch.count_nonzero(layer.lora_B.grad) > 0
    assert layer.lora_A.grad is not None
    assert all(p.grad is None for p in layer.linear.parameters())
    assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == 2 * (4 + 3)
    with torch.no_grad():
        layer.lora_B -= 0.1 * layer.lora_B.grad
    layer.zero_grad(set_to_none=True)
    layer(x).square().sum().backward()
    assert layer.lora_A.grad is not None and torch.count_nonzero(layer.lora_A.grad) > 0


def verify_production_parity_and_validation(LoRALinear):
    layer = LoRALinear(4, 3, 2, alpha=4).double()
    with torch.no_grad():
        layer.lora_B.fill_(0.2)
    base = torch.randn(3, 2, 4, dtype=torch.float64)
    x = base.transpose(0, 1)
    expected = layer.linear(x) + 2 * (x @ layer.lora_A.T @ layer.lora_B.T)
    assert torch.allclose(layer(x), expected, atol=1e-12)
    for args in ((4, 3, 0), (0, 3, 2), (4, 3, 2, float('inf'))):
        try:
            LoRALinear(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid LoRA configuration accepted")
    try:
        layer(torch.randn(2, 5, dtype=torch.float64))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong input width accepted")

# %%
from torch_judge import evaluate
if not evaluate("0026_lora_eval.py", LoRALinear):
    raise SystemExit(1)
