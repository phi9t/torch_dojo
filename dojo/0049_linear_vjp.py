# %%
# 0049 · Linear vector-Jacobian product
# Adapted from ferric_continuum/tnsr's linear backward path.
# y=x@w; grad_out is dL/dy. Derive both input and shared-weight gradients.

# %%
import torch

# %%
# Worked probe: dy=[1,0] votes only for the first output column.
x = torch.tensor([[1., 2.], [3., 4.]])
w = torch.tensor([[5., 6.], [7., 8.]])
print("y:", x @ w)
print("one-row dx:", torch.tensor([[1., 0.]]) @ w.T)

# %% [answer]
def linear_vjp(x: torch.Tensor, w: torch.Tensor,
               grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (dL/dx, dL/dw) for y=x@w without autograd."""
    pass

# %%
# Baseline: x [N,Din], w [Din,Dout], grad_out [N,Dout].
# Predict dx=grad_out@w.T and dw=x.T@grad_out on a tiny example.
from torch_judge import check
check("0049_linear_vjp_eval.py", stage="baseline")

# %%
# Follow-up: x [...,Din] may have several leading axes; ALL positions
# contribute to one [Din,Dout] weight gradient. Handle empty leading axes
# with zero dw. Outputs retain input and weight shapes.
check("0049_linear_vjp_eval.py", stage="followup")

# %%
# Production: accept noncontiguous finite float32/float64 tensors on the
# same device/dtype, leave inputs unchanged, and reject incompatible shapes,
# dtype or device with ValueError. Require Din,Dout >= 1. Compare against autograd on a nonuniform
# upstream gradient. Avoid a [...,Din,Dout] outer-product allocation.
check("0049_linear_vjp_eval.py", stage="production")
