# %%
# 0122 · Flash attention with a recomputing backward
# 0025 tiles the forward pass, but PyTorch autograd still keeps every tile's
# scores for backward. Production kernels save one number per query row and
# rebuild the tiles. Authored as a continuation of 0025_flash_attention.py.

# %%
import math
import torch
from torch_judge import check

# %%
# Baseline — scores s for one query row; L = logsumexp(s) is all the forward
# needs to remember. Predict what the next line prints before running it.
s = torch.tensor([1., 2., 3.])
L = torch.logsumexp(s, dim=0)
print(torch.exp(s - L).sum().item())

# %% [drill]
# Backward needs P (softmax probabilities) and the term D = rowsum(dO * O).
# Why does D equal rowsum(P * dP), and why is it cheaper to get from O?
# Derive dS = P * (dP - D) for one row, then dQ, dK, dV from dS, Q, K, P, dO.

# %%
# Contract: class FlashAttentionFn(torch.autograd.Function) with
# forward(ctx, Q, K, V, block_size=32) and backward(ctx, grad_out).
# Q [B,Sq,D], K [B,Sk,D], V [B,Sk,Dv]; returns O [B,Sq,Dv] (scaled dot-product
# attention, no mask). Visit KV tiles with a running max, mass and numerator as
# in 0025. Save only Q, K, V, O and L [B,Sq,1] with ctx.save_for_backward.
# backward returns (dQ, dK, dV, None), rebuilding each P tile as
# exp(Q_i K_j^T / sqrt(D) - L_i) instead of reading a stored score matrix.

# %%
# Inputs conforming to the signatures below. Call FlashAttentionFn.apply(Q, K, V, block_size).
torch.manual_seed(0)
Q: torch.Tensor = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)   # [B,Sq,D]
K: torch.Tensor = torch.randn(2, 7, 4, dtype=torch.float64, requires_grad=True)   # [B,Sk,D]
V: torch.Tensor = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)   # [B,Sk,Dv]
block_size: int = 3                                                               # KV tile width, > 0
grad_out: torch.Tensor = torch.randn(2, 5, 3, dtype=torch.float64)                # [B,Sq,Dv]


# %% [answer]
class FlashAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: torch.Tensor,                    # [B,Sq,D]
        K: torch.Tensor,                    # [B,Sk,D]
        V: torch.Tensor,                    # [B,Sk,Dv]
        block_size: int = 32,
    ) -> torch.Tensor:                      # O [B,Sq,Dv]
        pass

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,             # dO [B,Sq,Dv]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:   # dQ [B,Sq,D], dK [B,Sk,D], dV [B,Sk,Dv], None
        pass

# %%
# Baseline: the forward matches dense attention and joins the autograd graph.
check("0122_flash_attention_backward_eval.py", stage="baseline")

# %%
# Follow-up: gradients match dense autograd for unequal query/key lengths, tail
# tiles and any block size, and stay finite when scores are in the hundreds.
check("0122_flash_attention_backward_eval.py", stage="followup")

# %%
# Production: gradcheck in float64; no saved tensor and no operation in forward
# or backward may reach the size of the [B,Sq,Sk] score matrix; strided inputs
# and a non-contiguous upstream gradient work; reject nonpositive block sizes,
# rank/shape/dtype mismatches and queries without keys; empty queries give
# empty outputs and zero-shaped gradients. Passing a dense implementation that
# stores P and gets correct gradients must still fail the memory checks.
check("0122_flash_attention_backward_eval.py", stage="production")
