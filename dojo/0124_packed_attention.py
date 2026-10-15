# %%
# 0124 · Packed variable-length attention
# Serving and training stacks pack several sequences end to end and pass
# cu_seqlens instead of padding every sequence to the longest. Each token may
# attend only within its own sequence. Authored; compare with 0005 (dense) and
# 0009 (causal).

# %%
import math
import torch
from torch_judge import check

# %%
# Baseline — three sequences of lengths 3, 2 and 4 packed into N=9 tokens.
# Predict cu_seqlens, then how many True entries the block-diagonal mask has
# out of N*N, and how many keys token 4 (last of sequence 1) may use.
lengths = torch.tensor([3, 2, 4])
cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
print(cu_seqlens.tolist(), (lengths ** 2).sum().item(), "of", lengths.sum().item() ** 2)

# %% [drill]
# Padding to the longest sequence costs B * Lmax^2 score entries and a single
# dense N x N masked matrix costs N^2. What does attending per sequence cost?
# For lengths [16]*63 + [1000], compare the three.

# %%
# Contract: packed_attention(q, k, v, cu_seqlens, causal=False).
# q, k [N,D] and v [N,Dv] hold all sequences back to back. cu_seqlens is a 1-D
# int32/int64 tensor on the same device with cu_seqlens[0]=0 and cu_seqlens[-1]=N,
# never decreasing; sequence i occupies rows cu_seqlens[i]:cu_seqlens[i+1].
# Return [N,Dv]: scaled dot-product attention restricted to each sequence.

# %%
# Inputs conforming to the signature below: lengths 3, 2, 4 packed into N=9 tokens.
torch.manual_seed(0)
q: torch.Tensor = torch.randn(9, 4, dtype=torch.float64)          # [N,D]
k: torch.Tensor = torch.randn(9, 4, dtype=torch.float64)          # [N,D]
v: torch.Tensor = torch.randn(9, 3, dtype=torch.float64)          # [N,Dv]
cu_seqlens: torch.Tensor = torch.tensor([0, 3, 5, 9])             # [S+1] int32/int64, starts 0, ends N
causal: bool = False


# %% [answer]
def packed_attention(
    q: torch.Tensor,                        # [N,D]
    k: torch.Tensor,                        # [N,D]
    v: torch.Tensor,                        # [N,Dv]
    cu_seqlens: torch.Tensor,               # [S+1] int32/int64
    causal: bool = False,
) -> torch.Tensor:                          # [N,Dv]
    pass

# %%
# Baseline: matches attending to each sequence on its own, and changing one
# sequence's keys and values cannot change another sequence's output.
check("0124_packed_attention_eval.py", stage="baseline")

# %%
# Follow-up: causal=True masks future tokens, and position is relative to the
# start of the sequence, not the pack (the first token of sequence 2 sees only
# itself). Empty and single-token sequences are legal; N=0 returns [0,Dv].
check("0124_packed_attention_eval.py", stage="followup")

# %%
# Production: reject a cu_seqlens that does not start at 0, does not end at N,
# decreases, is not 1-D integer on the input device, plus rank, width, dtype and
# non-bool causal, all with ValueError. Gradients match per-sequence autograd.
# No operation may build a tensor of N^2/2 elements when the lengths are
# [16]*63 + [1000]. Non-contiguous inputs work. (Note: reading cu_seqlens with
# .tolist() syncs a GPU; real kernels keep it on device. Why accept that here?)
check("0124_packed_attention_eval.py", stage="production")
