# %%
# 0028 · Top-k mixture of experts
# Keep one editable class; each stage strengthens the same forward method.

# %%
import torch
from torch import nn

# %%
# Baseline — Router logits [0,2,1] choose experts 1 and 2 for k=2.
# Softmax ONLY the chosen logits so their weights sum to one.
logits = torch.tensor([0., 2., 1.])
print("selected:", logits.topk(2).indices, "weights:", logits.topk(2).values.softmax(0))
# router: Linear(D,E). experts: E registered Linear(D,F)->ReLU->Linear(F,D).
# x[B,S,D] -> weighted sum [B,S,D]. Unchosen experts contribute zero.

# Predict whether the expert with logit 0 contributes to this k=2 output.

# %% [answer]
class MixtureOfExperts(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

# %%
from torch_judge import check
check("0028_moe_eval.py", stage="baseline")

# %%
# Follow-up — Verify routing numerically with constant experts. Chosen
# router/expert parameters and input need gradients. Support non-contiguous
# [B,S,D] and a different batch size without changing learned parameters.
check("0028_moe_eval.py", stage="followup")

# %%
# Production — Reject invalid dimensions, k outside [1,E], or wrong input
# width. For equal router logits, select the LOWEST expert indices; make
# the tie policy explicit rather than relying on topk's unspecified tie order.
# Handle empty B or S. Sparse dispatch may loop over experts; avoid building
# [B,S,E,D] expert outputs just to throw most of them away.
check("0028_moe_eval.py", stage="production")
