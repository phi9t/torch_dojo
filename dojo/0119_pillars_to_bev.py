# %%
# 0119 · Pillars to image
# Goal: reshape [B*Y*X, C] pillar rows into a [B,C,Y,X] image layout without
# copying when the input allows it.
# The last step of 0108 BEV scatter. Authored practice.

# %%
import torch
from torch_judge import check

# %%
# Baseline — B=1, Y=2, X=3, C=2: six pillar rows with two channels each.
flat = torch.arange(12.).reshape(6, 2)

# %% [drill]
# View then permute
# Predict the shape, stride and contiguity of the channels-first image.
bev = flat.view(1, 2, 3, 2).permute(0, 3, 1, 2)
print(bev.shape, bev.stride(), bev.is_contiguous())

# %% [drill]
# View again?
# Predict what `bev.view(6, 2)` does, and whether `bev.reshape(6, 2)` copies.
try:
    bev.view(6, 2)
except RuntimeError:
    print("view failed")
print(bev.reshape(6, 2).data_ptr() == flat.data_ptr())

# %% [drill]
# Shared storage
# Predict whether `bev` and `flat` share memory, and what `flat[0, 0] = 99` does to `bev[0, 0, 0, 0]`.
flat[0, 0] = 99.
print(bev.data_ptr() == flat.data_ptr(), bev[0, 0, 0, 0])

# %%
# Contract: flat [B*Y*X, C] (any dtype), batch_size, grid_y, grid_x positive
# ints. Return [B,C,Y,X] with out[b,c,y,x] == flat[(b*Y+y)*X+x, c]. When flat
# is contiguous the result must be a view of it (same storage). A non-contiguous
# flat still works (a copy is allowed). Gradients flow back to flat.

# %% [answer]
def pillars_to_bev(flat: torch.Tensor, batch_size: int, grid_y: int, grid_x: int) -> torch.Tensor:
    pass

# %%
check('0119_pillars_to_bev_eval.py', stage='baseline')

# %%
# Follow-up — Batches in the right order, shared storage, a loop oracle.
check('0119_pillars_to_bev_eval.py', stage='followup')

# %%
# Production — Reject the wrong row count and non-int sizes; handle a
# transposed (non-contiguous) flat and C = 0; gradients reach every pillar row.
check('0119_pillars_to_bev_eval.py', stage='production')
