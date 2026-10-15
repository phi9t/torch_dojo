# %%
# 0005 · Scaled dot-product attention
# One answer, revised through three stages.

# %%
import math
import torch

# %%
# Baseline: Q [B,Tq,D], K [B,Tk,D], V [B,Tk,Dv] -> [B,Tq,Dv].
# Inputs are floating tensors, Tk>=1; query and key lengths may differ.
# Predict the score shape before running this probe:
q_probe = torch.zeros(2, 3, 4)
k_probe = torch.zeros(2, 5, 4)
print("score shape:", (q_probe @ k_probe.transpose(-2, -1)).shape)


# %%
# Inputs conforming to the signature below; evaluate these, then run the body line by line.
torch.manual_seed(0)
Q: torch.Tensor = torch.randn(2, 3, 4)      # [B,Tq,D]
K: torch.Tensor = torch.randn(2, 5, 4)      # [B,Tk,D]
V: torch.Tensor = torch.randn(2, 5, 6)      # [B,Tk,Dv]
allowed: torch.Tensor | None = None         # bool, broadcastable to [B,Tq,Tk]; True = may attend


# %% [answer]
def scaled_dot_product_attention(
    Q: torch.Tensor,                        # [B,Tq,D]  float
    K: torch.Tensor,                        # [B,Tk,D]  same dtype/device as Q
    V: torch.Tensor,                        # [B,Tk,Dv] same dtype/device as Q
    allowed: torch.Tensor | None = None,    # bool, broadcastable to [B,Tq,Tk]
) -> torch.Tensor:                          # [B,Tq,Dv]
    """Return attention; allowed=True means a key may be attended to."""
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("Q, K, V must be rank 3")
    batch, tq, d_head = Q.shape
    tk = K.shape[1]
    if (d_head == 0 or K.shape[0] != batch or V.shape[0] != batch
            or K.shape[2] != d_head or V.shape[1] != tk or tk == 0):
        raise ValueError("incompatible attention shapes")
    if (not Q.is_floating_point() or Q.dtype != K.dtype or Q.dtype != V.dtype
            or Q.device != K.device or Q.device != V.device):
        raise ValueError("inputs need same floating dtype and device")
    if allowed is not None:
        if allowed.dtype != torch.bool or allowed.device != Q.device:
            raise ValueError("allowed must be bool on the input device")
        try:
            allowed = torch.broadcast_to(allowed, (batch, tq, tk))
        except RuntimeError as error:
            raise ValueError("mask is not broadcastable") from error
    query_bsd, key_btd, value_btd = Q, K, V
    qk_prod_bst = query_bsd @ key_btd.transpose(-2, -1) / math.sqrt(d_head)  # [B,Tq,Tk]
    if allowed is not None:
        has_key_bs1 = allowed.any(dim=-1, keepdim=True)
        masked = qk_prod_bst.masked_fill(~allowed, -math.inf)  # out-of-place: keeps autograd clean
        # All-masked rows would softmax over all -inf (NaN); feed zeros, then zero the weights.
        qk_prod_bst = torch.where(has_key_bs1, masked, torch.zeros_like(masked))
    attn_weights_bst = torch.softmax(qk_prod_bst, dim=-1)  # [B,Tq,Tk]
    if allowed is not None:
        attn_weights_bst = attn_weights_bst * allowed.to(attn_weights_bst.dtype)
    output_bsd = attn_weights_bst @ value_btd  # [B,Tq,Dv]
    return output_bsd

# %%
from torch_judge import check
check("0005_attention_eval.py", stage="baseline")

# %%
# Follow-up: optional bool mask broadcastable to [B,Tq,Tk]; True = allowed.
# An all-masked query outputs exact zero. What happens if only key 2 is allowed?
print("single allowed key:", torch.tensor([False, False, True]))
check("0005_attention_eval.py", stage="followup")

# %%
# Production: validate shapes, floating dtype and device; keep autograd,
# batch isolation and finite output at extreme scores. Non-contiguous inputs
# should work. Dense scores take O(B*Tq*Tk) workspace.
check("0005_attention_eval.py", stage="production")
