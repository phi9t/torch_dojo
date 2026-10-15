# %%
# 0081 · Turn a module specification into working PyTorch
# RX03 is an authored module inspired by a reported paper-to-code format.
# It is not a recovered paper or company-specific module.

# %%
import torch
from torch import nn
from torch_judge import check

# %%
# Draw the branches before coding: h=LayerNorm(x); a=fc2(GELU(fc1(h)));
# g=sigmoid(gate(h)); y=x+g*a. Both branches see the SAME normalized h.
# For D=8,F=16 the trainable count is 2DF+D^2+F+4D.
print('Expected parameter count:', 2*8*16+8*8+16+4*8)

# %%
# Contract: x [...,D], positive D/F; learned LayerNorm eps=1e-5,
# Linear(D,F)->GELU->Linear(F,D), gate Linear(D,D)->sigmoid.
# Preserve all leading dimensions and dtype/device; no dropout.
# Name layers norm, fc1, fc2, gate so the probes can inspect state.

# %% [answer]
class GatedResidual(nn.Module):
    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

# %%
# Baseline: shape and parameter count. Set fc2 to zero: y must equal x.
check('0081_gated_residual_eval.py', stage='baseline')

# %%
# Follow-up: set gate to zero: sigmoid(0)=0.5. Check that branch and gate
# use the SAME norm(x), and that a backward pass reaches all parameters.
check('0081_gated_residual_eval.py', stage='followup')

# %%
# Production: reject invalid widths/feature dimensions, support
# noncontiguous input, and preserve the module's saved parameter state.
check('0081_gated_residual_eval.py', stage='production')
