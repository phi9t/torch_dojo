# %%
# 0004 · LayerNorm over the final feature dimension
# One answer, improved in place. No F.layer_norm or nn.LayerNorm in answer.

# %%
import torch
from torch_judge import check

# %%
# Baseline: x [..., D], gamma [D], beta [D], D>=1, eps>0. Use the
# population variance (divide by D), with eps INSIDE sqrt. Normalize each
# final-axis row independently, then scale and shift. Keep shape/dtype.
probe = torch.tensor([[1., 3.], [5., 7.]])
print("per-row means:", probe.mean(dim=-1, keepdim=True))
print("global mean differs:", probe.mean())

# Derive the normalized first row for [1,3] with gamma=1, beta=0
# in the limit as positive eps approaches zero.

# %% [answer]
def my_layer_norm(x: torch.Tensor, gamma: torch.Tensor,
                  beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    pass  # Replace this; keep improving this one definition.

# %%
check("0004_layernorm_eval.py", stage="baseline")

# %%
# Follow-up: support singleton feature rows (variance zero), noncontiguous
# higher-rank views, and nontrivial gamma/beta. For gamma=1, beta=0 a constant
# row becomes zero. Why is the default unbiased sample variance wrong at D=1?
print("singleton sample variance:", torch.tensor([1.]).var())
check("0004_layernorm_eval.py", stage="followup")

# %%
# Production: preserve dtype/device, leave all inputs unchanged, and match
# framework forward/backward for finite float64 inputs. Reject nonfinite or
# nonpositive eps, mismatched gamma/beta sizes, and nonfloating tensors with
# clear ValueError.
# Reductions use only O(number of rows) extra state, not a D-by-D Jacobian.
check("0004_layernorm_eval.py", stage="production")
