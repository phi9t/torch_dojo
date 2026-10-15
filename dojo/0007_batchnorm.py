# %%
# 0007 · Batch normalization with explicit train/eval state
# Keep one answer cell. Run each check after editing it in Emacs/Jupyter.

# %%
import torch

# %%
# Worked probe: two features, two samples. Reductions run over samples (dim=0).
probe = torch.tensor([[1., 10.], [3., 20.]])
print("batch mean:", probe.mean(0), "biased variance:", probe.var(0, unbiased=False))
# With gamma=1, beta=0, eps≈0, the first column normalizes to [-1, +1].
# The buffers running_mean and running_var start at 0 and 1 respectively.

# Derive the second column's normalized values from the two-sample probe.

# %% [answer]
def my_batch_norm(x, gamma, beta, running_mean, running_var,
                  eps=1e-5, momentum=0.1, training=True):
    # x [N,D], gamma/beta/buffers [D]; replace this body.
    pass

# %%
# Baseline: finite floating [N,D], N>=1, D>=1. In training, use each
# feature's current batch mean and *biased* variance (divide by N).
# Return [N,D]; support gradients through x, gamma, beta. No built-in BN.
from torch_judge import check
check("0007_batchnorm_eval.py", stage="baseline")

# %%
# Follow-up: update the two buffers in place under no_grad using
# old*(1-momentum)+batch_stat*momentum. In eval, use buffers only and do
# not mutate them. A single training sample has zero biased variance.
check("0007_batchnorm_eval.py", stage="followup")

# %%
# Production: validate shape/device/dtype, finite floating inputs,
# eps>0 and 0<=momentum<=1 before mutating buffers. Use no [N,N] matrix.
# Preserve dtype/device and avoid attaching running buffers to autograd.
# How would large offsets affect the variance calculation?
check("0007_batchnorm_eval.py", stage="production")
