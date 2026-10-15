"""Reference and verifiers for the authored gated residual module."""

# %%
import torch
from torch import nn

# %%
class GatedResidual(nn.Module):
    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        if dimension < 1 or hidden < 1:
            raise ValueError("positive dimension and hidden width required")
        self.dimension = dimension
        self.norm = nn.LayerNorm(dimension, eps=1e-5)
        self.fc1 = nn.Linear(dimension, hidden)
        self.fc2 = nn.Linear(hidden, dimension)
        self.gate = nn.Linear(dimension, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dimension:
            raise ValueError("wrong feature dimension")
        h = self.norm(x)
        a = self.fc2(torch.nn.functional.gelu(self.fc1(h)))
        g = torch.sigmoid(self.gate(h))
        return x + g * a


# %% [verifiers]
def verify_baseline_shape_and_parameter_count(GatedResidual):
    D, F = 8, 16
    model = GatedResidual(D, F)
    assert sum(p.numel() for p in model.parameters()) == 2 * D * F + D * D + F + 4 * D
    x = torch.randn(2, 3, D)
    assert model(x).shape == x.shape


def verify_baseline_zero_branch_is_identity(GatedResidual):
    model = GatedResidual(4, 6)
    with torch.no_grad():
        model.fc2.weight.zero_()
        model.fc2.bias.zero_()
    x = torch.randn(2, 4)
    assert torch.equal(model(x), x)


def verify_followup_half_gate_uses_same_normalized_input(GatedResidual):
    model = GatedResidual(3, 5)
    with torch.no_grad():
        model.gate.weight.zero_()
        model.gate.bias.zero_()
    x = torch.randn(2, 3, dtype=torch.float32)
    h = model.norm(x)
    branch = model.fc2(torch.nn.functional.gelu(model.fc1(h)))
    assert torch.allclose(model(x), x + 0.5 * branch, atol=1e-6)


def verify_followup_backward_and_noncontiguous_input(GatedResidual):
    model = GatedResidual(4, 7)
    base = torch.randn(3, 2, 4, requires_grad=True)
    x = base.transpose(0, 1)
    assert not x.is_contiguous()
    model(x).square().sum().backward()
    assert base.grad is not None and torch.isfinite(base.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def verify_production_reject_invalid_width(GatedResidual):
    for shape in [(0, 4), (4, 0)]:
        try:
            GatedResidual(*shape)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dimension accepted")



def verify_production_state_dict_replay_and_feature_shape(GatedResidual):
    torch.manual_seed(17)
    model = GatedResidual(4, 7)
    x = torch.randn(2, 3, 4)
    expected = model(x).detach()
    restored = GatedResidual(4, 7)
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(restored(x), expected, atol=0, rtol=0)
    assert set(model.state_dict()) == {'norm.weight', 'norm.bias',
                                      'fc1.weight', 'fc1.bias',
                                      'fc2.weight', 'fc2.bias',
                                      'gate.weight', 'gate.bias'}
    try:
        model(torch.randn(2, 3, 5))
    except ValueError:
        pass
    else:
        raise AssertionError('wrong feature width accepted')

# %%
from torch_judge import evaluate
if not evaluate("0081_gated_residual_eval.py", GatedResidual):
    raise SystemExit(1)
