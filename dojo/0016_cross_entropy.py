# %%
# 0016 · Cross-entropy from stable logits to ignored labels

# %%
import torch

# %%
# Worked probe: rows [2,5,1] and [4,0,3] choose classes 1 and 0.
# Direct exp(1000) overflows; logsumexp is stable.
probe = torch.tensor([[2., 5., 1.], [4., 0., 3.]])
labels = torch.tensor([1, 0])
print("selected logits:", probe[torch.arange(2), labels])

# Derive the loss for equal logits [0,0] and target 1 before coding.

# %% [answer]
def cross_entropy_loss(logits, targets, ignore_index=-100):
    # logits [B,C], integer targets [B], C>=2; replace this body.
    pass

# %%
# Baseline: mean of logsumexp(row)-logit[target] over B>=1.
# Return a scalar tensor; support autograd. No cross-entropy helper.
from torch_judge import check
check("0016_cross_entropy_eval.py", stage="baseline")

# %%
# Follow-up: ignore_index labels contribute no loss. Reduce by the
# number of valid rows, not batch length. If none are valid (including
# B=0), return a differentiable scalar zero. Validate before indexing.
check("0016_cross_entropy_eval.py", stage="followup")

# %%
# Production: finite floating logits, integral target IDs, shape/device
# compatibility and range validation. Preserve dtype/device and
# gradients under extreme per-row shifts; use O(BC) workspace.
check("0016_cross_entropy_eval.py", stage="production")
