# %%
# 0102 · One honest MoE training step
# Routing decides which experts learn. A pleasing forward output is not
# enough: the router and the selected experts must actually get gradients.
# Source: https://arxiv.org/html/2412.19437#S2.SS1.SSS2
# The auxiliary balance term below is a small authored training exercise,
# not a claim about an unpublished model training loop.

# %%
import torch
from torch import nn
from torch_judge import check

# %%
# Suppose top-2 router logits are [0, 2, 1]. Experts 1 and 2 answer
# [2, 3] for one class. What is their weighted class logit? Which
# expert will receive zero gradient from that token's task loss?
scores = torch.tensor([0., 2., 1.])
selected = scores.argsort(descending=True, stable=True)[:2]
print("chosen IDs:", selected.tolist(), "weights:", scores[selected].softmax(0).tolist())

# With K=1, selected-softmax always returns weight 1. The task loss has
# no smooth path back to the router between route changes. Check this
# small graph before training; the selected logit's derivative is zero.
probe_logits = torch.tensor([[.2, .1, -1.]], requires_grad=True)
probe_id = probe_logits.argsort(dim=1, descending=True, stable=True)[:, :1]
probe_weight = probe_logits.gather(1, probe_id).softmax(dim=1)
probe_weight.sum().backward()
print("top-1 weight:", probe_weight.item(), "router gradient:", probe_logits.grad.tolist())

# %% [answer]
class TinyMoE(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_experts: int,
                 top_k: int, num_classes: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return class logits [T,C], route IDs [T,K], requested counts [E]."""
        pass

    def train_step(self, x: torch.Tensor, targets: torch.Tensor,
                   optimizer: torch.optim.Optimizer, balance_weight: float = 0.0
                   ) -> dict:
        """One update; report pre-update loss, CE, auxiliary loss, and counts."""
        pass

# %%
# Baseline — Router Linear(D,E), E registered two-layer Tanh experts
# Linear(D,H)->Tanh->Linear(H,C). Select top-k by descending router
# logit; ties choose lower expert ID. Softmax only selected logits;
# sum their weighted [C] answers. Return hard request counts [E].
check("0102_tiny_moe_update_eval.py", stage="baseline")

# %%
# Follow-up — Train on integer class targets with mean cross-entropy.
# Exactly one zero_grad/backward/step. Return pre-update diagnostics:
# loss, ce, aux (floats), counts (detached CPU integer tensor [E]).
# Add optional balance_weight * aux to the loss, where
# aux = E * sum_e(mean_t softmax(all router logits)[t,e]
#                 * detached_count[e] / (T*K)).
# This auxiliary term changes router gradients but not the hard count.
check("0102_tiny_moe_update_eval.py", stage="followup")

# %%
# Production — T may be zero in forward. Training needs T>=1. Validate
# positive widths, 1<=K<=E, input shape/dtype, integer target range,
# finite balance weight >=0, and matching device. No input mutation.
# Inspect route counts before and after updates: an expert with no routes
# receives no task gradient, even if the total loss falls. For a forced
# top-1 batch, compare zero balance weight against positive balance
# weight. Task-only router gradients are zero; the auxiliary term can
# move router logits enough to change the next batch's hard routes.
check("0102_tiny_moe_update_eval.py", stage="production")
