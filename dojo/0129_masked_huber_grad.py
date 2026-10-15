# %%
# 0129 · Masked Huber loss and manual gradient in NumPy (practice set 5/7, 10 min)
# Goal: implement `huber_loss_and_grad` and verify it with float64 finite differences.

# %%
import numpy as np

# %%
# Baseline — prediction[N,D], target[N,D], valid_mask bool [N], delta > 0.
# residual = prediction - target. Per element:
#   0.5 * r^2                     if |r| <= delta
#   delta * (|r| - 0.5 * delta)   otherwise
# Average over ALL D coordinates of VALID rows only. Return (loss, grad[N,D]);
# invalid rows have zero gradient; all-invalid/empty input -> (0.0, zeros).
# Hand-checkable: residuals [[-2, 0.5]], delta=1 -> loss 0.8125, grad [[-0.5, 0.25]].

# %% [drill]
# Predict the worked example
# Per element: 1*(2 - 0.5) = 1.5 and 0.5*0.5^2 = 0.125. Mean over 2 coordinates?
print((1.5 + 0.125) / 2)

# %% [drill]
# Gradient shape of Huber
# d/dr Huber = clip(r, -delta, delta). Predict it for r = -2, 0.5 and delta = 1, then divide by the count.
print(np.clip([-2.0, 0.5], -1, 1) / 2)

# %% [drill]
# Mask by multiplying?
# Predict `np.nan * 0.0`. What does that mean for an invalid row that holds NaN?
print(np.nan * 0.0)

# %%
# Inputs conforming to the answer's signature.
rng = np.random.default_rng(0)
prediction: np.ndarray = 2 * rng.normal(size=(4, 3))   # [N,D] float64
target: np.ndarray = rng.normal(size=(4, 3))           # [N,D] float64
valid_mask: np.ndarray = np.array([True, True, False, True])  # bool [N]
delta: float = 0.9

# %% [answer]
def huber_loss_and_grad(prediction: np.ndarray, target: np.ndarray,
                        valid_mask: np.ndarray, delta: float) -> tuple[float, np.ndarray]:
    # prediction, target [N,D]; valid_mask bool [N]; delta > 0
    # returns (scalar loss, grad_prediction [N,D])
    pass

# %%
from torch_judge import check
check("0129_masked_huber_grad_eval.py", stage="baseline")

# %%
# Follow-up — Both sides of the threshold (|r| just below, at and above delta),
# masked rows equal to dropping them, and all-invalid input.
check("0129_masked_huber_grad_eval.py", stage="followup")

# %%
# Production — Your gradient matches float64 central finite differences.
# Invalid rows may hold NaN/inf and must not leak into loss or gradient (select,
# don't multiply). Reject delta <= 0, shape mismatches and a non-bool mask.
check("0129_masked_huber_grad_eval.py", stage="production")
