"""Top-k MoE reference and staged checks."""
# %%
import torch
from torch import nn

# %%
class MixtureOfExperts(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2):
        super().__init__()
        if any(not isinstance(v, int) or v <= 0 for v in (d_model, d_ff, num_experts, top_k)) or top_k > num_experts:
            raise ValueError("positive widths and 1 <= top_k <= num_experts required")
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)) for _ in range(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.router.in_features:
            raise ValueError("need x[B,S,d_model]")
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] == 0:
            return x.clone()
        logits = self.router(flat)
        # Stable sort preserves original expert-index order among tied logits.
        indices = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :self.top_k]
        selected = logits.gather(1, indices)
        weights = selected.softmax(dim=-1)
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == expert_id)
            if rows.numel():
                values = expert(flat[rows]) * weights[rows, slots, None]
                out = out.index_add(0, rows, values)
        return out.reshape_as(x)

# %% [verifiers]
def verify_baseline_registered_and_shape(MixtureOfExperts):
    model = MixtureOfExperts(4, 8, 3, 2)
    assert isinstance(model.router, nn.Linear) and isinstance(model.experts, nn.ModuleList)
    assert len(model.experts) == 3 and model.router.weight.shape == (3, 4)
    assert model(torch.randn(2, 5, 4)).shape == (2, 5, 4)


def verify_followup_weighted_routing_and_gradients(MixtureOfExperts):
    model = MixtureOfExperts(2, 3, 3, 2).double()
    with torch.no_grad():
        model.router.weight.zero_(); model.router.bias.copy_(torch.tensor([0., 2., 1.]))
        for idx, expert in enumerate(model.experts):
            expert[0].weight.zero_(); expert[0].bias.fill_(1.)
            expert[2].weight.zero_(); expert[2].bias.fill_(float(idx + 1))
    x = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    expected_scalar = (torch.tensor([2., 1.]).softmax(0) * torch.tensor([2., 3.])).sum()
    assert torch.allclose(model(x), torch.full_like(x, expected_scalar))
    # Random weights avoid a constant-output path in the gradient probe.
    fresh = MixtureOfExperts(2, 3, 3, 2).double()
    fresh.router.bias.data.copy_(torch.tensor([0., 2., 1.]))
    fresh(x).square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert fresh.router.weight.grad is not None and torch.isfinite(fresh.router.weight.grad).all()
    assert any(p.grad is not None for p in fresh.experts[1].parameters())
    assert fresh(torch.randn(3, 2, 2, dtype=torch.float64).transpose(0, 1)).shape == (2, 3, 2)


def verify_production_ties_empty_and_invalid(MixtureOfExperts):
    model = MixtureOfExperts(2, 3, 3, 1)
    with torch.no_grad():
        model.router.weight.zero_(); model.router.bias.zero_()
        for idx, expert in enumerate(model.experts):
            expert[0].weight.zero_(); expert[0].bias.fill_(1.)
            expert[2].weight.zero_(); expert[2].bias.fill_(float(idx + 1))
    assert torch.equal(model(torch.ones(1, 2, 2)), torch.ones(1, 2, 2))
    assert model(torch.empty(1, 0, 2)).shape == (1, 0, 2)
    for args in ((0, 2, 3), (2, 2, 3, 0), (2, 2, 3, 4)):
        try:
            MixtureOfExperts(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid MoE configuration accepted")
    try:
        model(torch.randn(1, 2, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong input width accepted")

# %%
from torch_judge import evaluate
if not evaluate("0028_moe_eval.py", MixtureOfExperts):
    raise SystemExit(1)
