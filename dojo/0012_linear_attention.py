# %%
# 0012 · Linear self-attention
# One answer, three increasingly useful contracts. Run cells in Emacs/Jupyter.

# %%
import torch
import torch.nn.functional as F

# %%
# Baseline: Q,K [B,S,Dk], V [B,S,Dv]. Apply phi(x)=ELU(x)+1 and
# compute phi(Q) @ (phi(K).T @ V) divided by phi(Q) @ sum(phi(K)).
# No S-by-S score tensor is needed. Predict the two summary shapes.
Q = torch.ones(1, 3, 2)
K = torch.ones(1, 3, 2)
V = torch.arange(3.0).reshape(1, 3, 1)
print("summary shapes:", (K.transpose(1, 2) @ V).shape, K.sum(1).shape)
print("equal-key probe: every query should return", V.mean(1).item())

# %% [answer]
def linear_attention(Q, K, V):
    """Return [B,Sq,Dv] using positive ELU features; revise this function in place."""
    pass

# %%
from torch_judge import check
check("0012_linear_attention_eval.py", stage="baseline")

# %%
# Follow-up: permit Sq != Sk, including a single query and an empty query.
# The key and value lengths must still agree. Which dimension do you sum?
check("0012_linear_attention_eval.py", stage="followup")

# %%
# Production: reject malformed shapes, dtype/device mismatches and Sk=0
# when Sq>0. Preserve autograd and non-contiguous inputs. Keep peak
# intermediates O(B*(Sq+Sk)*D + B*Dk*Dv), never O(Sq*Sk).
check("0012_linear_attention_eval.py", stage="production")
