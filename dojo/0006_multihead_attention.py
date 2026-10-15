# %%
# 0006 · Multi-head attention, from first pass to reusable layer
# Start with Q/K/V projections and head shapes. Keep improving the SAME class
# as you take the follow-up and production challenges.

# %%
import math
import torch
from torch import nn

# %%
# Baseline — Predict the shapes before running this cell.
# Q[B,Tq,D] and K,V[B,Tk,D] become [B,H,Tq,D/H] and [B,H,Tk,D/H].
# Scores have shape [B,H,Tq,Tk]; output is [B,Tq,D].
probe = torch.arange(24.).reshape(2, 3, 4)
print("three positions, two heads:", probe.reshape(2, 3, 2, 2).transpose(1, 2).shape)
# Implement four nn.Linear(D,D) projections: W_q, W_k, W_v, W_o.
# Use scaled dot-product attention. Cross-attention (Tq != Tk) must work.
# Do not use nn.MultiheadAttention or scaled_dot_product_attention.

# %%
torch.manual_seed(0)
d_model: int = 12
num_heads: int = 3
Q: torch.Tensor = torch.randn(2, 3, 12)  # [B,Tq,D]
K: torch.Tensor = torch.randn(2, 5, 12)  # [B,Tk,D]
V: torch.Tensor = torch.randn(2, 5, 12)  # [B,Tk,D]
allow: torch.Tensor | None = None  # bool, broadcastable to [B,H,Tq,Tk]; True = allowed

# %% [answer]
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if (not isinstance(d_model, int) or not isinstance(num_heads, int)
                or d_model < 1 or num_heads < 1 or d_model % num_heads):
            raise ValueError("d_model must be a positive multiple of num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                allow: torch.Tensor | None = None) -> torch.Tensor:
        # Q [B,Tq,D], K,V [B,Tk,D], allow bool ~[B,H,Tq,Tk] -> out [B,Tq,D]
        if (Q.ndim != 3 or K.ndim != 3 or V.ndim != 3 or Q.shape[0] != K.shape[0]
                or K.shape[:2] != V.shape[:2] or K.shape[1] == 0
                or any(x.shape[-1] != self.d_model for x in (Q, K, V))):
            raise ValueError("need Q[B,Tq,D], K,V[B,Tk,D], Tk >= 1")
        if Q.dtype != K.dtype or Q.dtype != V.dtype or Q.device != K.device or Q.device != V.device:
            raise ValueError("Q, K, V need matching dtype and device")
        B, Tq, _ = Q.shape
        Tk = K.shape[1]

        def split(x: torch.Tensor, length: int) -> torch.Tensor:  # [B,L,D] -> [B,H,L,d_k]
            return x.reshape(B, length, self.num_heads, self.d_k).transpose(1, 2)

        q = split(self.W_q(Q), Tq)
        k = split(self.W_k(K), Tk)
        v = split(self.W_v(V), Tk)
        work = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
        scores = q.to(work) @ k.to(work).transpose(-2, -1) / math.sqrt(self.d_k)  # [B,H,Tq,Tk]
        query_has_key = None
        if allow is not None:
            if allow.dtype != torch.bool or allow.device != Q.device:
                raise ValueError("allow must be bool on the input device")
            try:
                mask = torch.broadcast_to(allow, scores.shape)
            except RuntimeError as exc:
                raise ValueError("allow is not broadcastable to attention scores") from exc
            has_key = mask.any(dim=-1, keepdim=True)
            # Fully masked rows get zero scores BEFORE softmax so no NaN enters autograd.
            safe = torch.where(has_key, scores.masked_fill(~mask, -torch.inf), torch.zeros_like(scores))
            weights = torch.where(mask, torch.softmax(safe, dim=-1), torch.zeros_like(scores))
            query_has_key = has_key.any(dim=1).squeeze(-1)  # [B,Tq]
        else:
            weights = torch.softmax(scores, dim=-1)
        attended = (weights @ v.to(work)).to(Q.dtype)  # [B,H,Tq,d_k]
        out = self.W_o(attended.transpose(1, 2).reshape(B, Tq, self.d_model))
        if query_has_key is not None:
            out = out.masked_fill(~query_has_key[..., None], 0)
        return out

# %%
# Baseline check — What shape do you expect for Q[2,3,12], K,V[2,5,12]?
from torch_judge import check
check("0006_multihead_attention_eval.py", stage="baseline")

# %%
# Follow-up — Register the projection weights in the module. They must appear
# in parameters(), work after .to(dtype=torch.float64), and receive gradients.
# Reject non-divisible head counts and incompatible Q/K/V dimensions. A
# transposed (non-contiguous) input must still work. Add tests of your own.
check("0006_multihead_attention_eval.py", stage="followup")

# %%
# Production challenge — Optional allow is a BOOL mask broadcastable to
# [B,H,Tq,Tk]. True means allowed. Mask before softmax. For a query with no
# allowed key in ANY head, output exactly zero with finite forward/backward.
# No dropout in this exercise. Preserve dtype/device; reject mismatches.
# Think about why replacing NaNs only after softmax is too late.
check("0006_multihead_attention_eval.py", stage="production")
