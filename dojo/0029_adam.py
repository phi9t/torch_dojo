# %%
# 0029 · Adam optimizer
# Run cells in Emacs/Jupyter. Edit only the answer cell and rerun a check.
# The reported algorithm is Adam; skipped gradients and validation are authored follow-ups.

# %%
import torch

# %%
# Baseline — for one scalar p=1 and gradient 2, m_hat=2 and v_hat=4
# on the first step, so lr=0.1 moves p near 0.9. Bias correction is
# essential even though both moment buffers start at zero.
print("first-step bias factors:", 1 - 0.9, 1 - 0.999)
# Contract: class MyAdam(params, lr=1e-3, betas=(.9,.999), eps=1e-8)
# supports step() and zero_grad(). Parameters are leaf floating tensors.
# step changes them in place without recording an autograd operation.

# Derive the approximate first update when p=1, g=2, lr=.1, eps=1e-8.

# %% [answer]
class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass

# %%
from torch_judge import check
check("0029_adam_eval.py", stage="baseline")

# %%
# Follow-up — parameters with grad=None do not advance their moments or
# individual step counters. This matters when one branch is unused for a
# training step. Compare two parameters: update only A, then both A and B.
check("0029_adam_eval.py", stage="followup")

# %%
# Production challenge — validate hyperparameters and parameter ownership;
# keep state separate for each parameter and on its device/dtype. Repeated
# steps should agree with torch.optim.Adam for dense gradients. zero_grad()
# clears present gradients without changing parameter values. State memory
# is O(number of parameter elements); each step visits only present grads.
check("0029_adam_eval.py", stage="production")
