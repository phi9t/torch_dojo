"""SwiGLU MLP reference and staged checks."""
# %%
import torch
from torch import nn
from torch.nn import functional as F

# %%
class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        if not isinstance(d_model, int) or not isinstance(d_ff, int) or d_model <= 0 or d_ff <= 0:
            raise ValueError("widths must be positive integers")
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.gate_proj.in_features:
            raise ValueError("last dimension must equal d_model")
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# %% [verifiers]
def verify_baseline_formula(SwiGLUMLP):
    model = SwiGLUMLP(3, 5)
    assert isinstance(model, nn.Module)
    assert all(isinstance(getattr(model, n), nn.Linear) for n in ("gate_proj", "up_proj", "down_proj"))
    x = torch.tensor([[1., -2., 3.]])
    expected = model.down_proj(F.silu(model.gate_proj(x)) * model.up_proj(x))
    assert torch.allclose(model(x), expected)
    assert sum(p.numel() for p in model.parameters()) == 2 * (3 * 5 + 5) + (5 * 3 + 3)


def verify_followup_layout_and_gradients(SwiGLUMLP):
    model = SwiGLUMLP(4, 7).double()
    base = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    x = base.transpose(0, 1)
    out = model(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    out.square().sum().backward()
    assert base.grad is not None and torch.isfinite(base.grad).all()
    assert len(list(model.parameters())) == 6
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def verify_production_validation_and_empty(SwiGLUMLP):
    for dims in ((0, 4), (4, 0), (-1, 4)):
        try:
            SwiGLUMLP(*dims)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid widths accepted")
    model = SwiGLUMLP(4, 7)
    try:
        model(torch.ones(2, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong feature width accepted")
    assert model(torch.empty(2, 0, 4)).shape == (2, 0, 4)

# %%
from torch_judge import evaluate
if not evaluate("0015_mlp_eval.py", SwiGLUMLP):
    raise SystemExit(1)
