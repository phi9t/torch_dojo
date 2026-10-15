# %%
# 0126 · Symmetric contrastive loss in NumPy (practice set 2/7, 20 min)
# Goal: implement `contrastive_loss` with a stable logsumexp. No cross-entropy
# helper. State time and peak intermediate-memory complexity when done.

# %%
import numpy as np

# %%
# Baseline — X[N,D], Y[N,D]: row i of X matches row i of Y.
# Drop invalid pairs from BOTH sets, normalize each row by max(||row||, eps),
# logits = Xn @ Yn.T / temperature  [n,n], cross-entropy with diagonal
# positives, return the mean of the X->Y and Y->X losses. No valid pairs -> 0.
# Hand-checkable: X=Y=I(2), temperature=1 -> 0.3132616875.

# %% [drill]
# Predict the loss
# X=Y=I(2), t=1: each row is -log(e / (e + 1)). Predict the value.
print(np.log(np.e + 1) - 1)

# %% [drill]
# Why subtract the row max?
# Predict what naive exp does at temperature 1e-3, where logits reach 1000.
with np.errstate(over="ignore"):
    print(np.exp(1000.0), np.log(np.exp(1000.0)))

# %% [drill]
# Identical embeddings
# All n valid embeddings equal and nonzero: every logit is the same. Predict the loss.
print(np.log(5))

# %%
# Inputs conforming to the answer's signature.
rng = np.random.default_rng(0)
X: np.ndarray = rng.normal(size=(5, 4))                       # [N,D] float64
Y: np.ndarray = rng.normal(size=(5, 4))                       # [N,D] float64
temperature: float = 0.5                                      # > 0
valid_mask: np.ndarray = np.array([True, True, False, True, True])  # bool [N]
eps: float = 1e-12

# %% [answer]
def contrastive_loss(X: np.ndarray, Y: np.ndarray, temperature: float,
                     valid_mask: np.ndarray, eps: float = 1e-12) -> float:
    # X, Y [N,D]; valid_mask bool [N]; returns a scalar float
    pass

# %%
from torch_judge import check
check("0126_contrastive_loss_eval.py", stage="baseline")

# %%
# Follow-up — Invalid rows leave the query AND candidate sets. All-zero
# embeddings stay finite. Permuting X, Y and the mask together leaves the loss
# unchanged; swapping X and Y does too.
check("0126_contrastive_loss_eval.py", stage="followup")

# %%
# Production — Stable at tiny temperature, invariant to per-row scale, garbage
# (NaN/inf) in INVALID rows is ignored, bad temperature/shape/mask raises
# ValueError, inputs are not mutated.
# Extension (not graded): return d loss / d logits, then explain how row
# normalization projects the embedding gradient orthogonally to each row.
check("0126_contrastive_loss_eval.py", stage="production")
