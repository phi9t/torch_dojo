# %%
# 0017 · Inverted dropout with train/eval state

# %%
import torch
import torch.nn as nn

# %%
# Worked probe: p=.25 means surviving values scale by 1/.75.
# torch.manual_seed can replay the sequence of masks during practice.
torch.manual_seed(7)
print("kept fraction:", (torch.rand(1000) >= .25).float().mean().item())

# Predict the expected output for input 3 when p=.25 during training.

# %% [answer]
class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        # Replace this body.

    def forward(self, x):
        pass

# %%
# Baseline: in training, mask each element independently with drop
# probability p and scale survivors by 1/(1-p). No built-in dropout.
from torch_judge import check
check("0017_dropout_eval.py", stage="baseline")

# %%
# Follow-up: .eval() returns input unchanged. Handle p=0 as identity
# and p=1 as all zeros, with zero gradient through that result.
check("0017_dropout_eval.py", stage="followup")

# %%
# Production: reject p outside [0,1] and nonfinite p. Preserve input
# shape/dtype/device, avoid modifying input, and use torch's RNG so
# manual_seed reproduces masks. Gradients equal mask times scale.
check("0017_dropout_eval.py", stage="production")
