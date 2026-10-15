# %%
# 0025 · Tiled online-softmax attention
# This Python exercise teaches the exact blockwise recurrence, not a CUDA kernel.

# %%
import math
import torch

# %%
# Baseline: Q,K [B,S,D], V [B,S,Dv]. For each query block, visit KV
# blocks while tracking row maximum m, normalizer l and numerator acc.
# When maximum changes, rescale old l and acc by exp(old_m-new_m).
a = torch.tensor([1., 2.]); b = torch.tensor([3., 4.])
print("two block maxima:", a.max().item(), b.max().item())

# Derive the updated normalizer when the next tile has a larger maximum.

# %% [answer]
def flash_attention(Q, K, V, block_size=32):
    """Return scaled dot-product attention without materializing [S,S]."""
    pass

# %%
from torch_judge import check
check("0025_flash_attention_eval.py", stage="baseline")

# %%
# Follow-up: incomplete tail blocks and differing query/key lengths must
# match a dense stable softmax. Test different block sizes and large logits.
check("0025_flash_attention_eval.py", stage="followup")

# %%
# Production: reject nonpositive block sizes and incompatible tensors.
# Handle non-contiguous views and gradients. For forward inference under
# torch.no_grad(), excluding Q/K/V storage, a blockwise implementation can
# use O(B*Sq*Dv + B*block_size^2 + B*block_size*(Dv+1)) peak workspace.
# Ordinary PyTorch autograd retains intermediates from many tiles for backward;
# this Python implementation does not promise that bound during training.
# A memory-efficient backward needs a custom backward or recomputation.
# The judge checks numerical behavior and gradients on small inputs. Review
# the forward allocation pattern by hand; passing the judge does not prove
# the memory bound. Why does retaining scores from every tile change it?
check("0025_flash_attention_eval.py", stage="production")
