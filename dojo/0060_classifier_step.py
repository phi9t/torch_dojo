# %%
# 0060 · One honest classifier update
# Goal: Take one full-batch SGD step for a linear softmax classifier.
# Use the stable `logits` loss from 0059. Compute both parameter gradients from
# ONE parameter snapshot, then update `W` and `b` without mutating the inputs.

# %%
import numpy as np

# %%
# Baseline — One full-batch update from the current parameters.
X_probe = np.array([[-1.], [1.]])
y_probe = np.array([0, 1])
W_probe = np.zeros((1, 2)); b_probe = np.zeros(2)
print("initial logits:", X_probe @ W_probe + b_probe)
# Predict: The first step from zero parameters. What is the loss, and which
# way do the two columns of `W` move?
#
# For $Z = XW + b$, the parameter gradients are
# $\nabla_W L = X^{\mathsf{T}} \nabla_Z L$ and $\nabla_b L = \sum_n (\nabla_Z L)_{n,:}$,
# that is `dW = X.T @ dZ` and `db = dZ.sum(axis=0)`.
# Return `(new_W, new_b, loss)` where `loss` is the PRE-UPDATE mean loss.

# %% [drill]
# Which axis does db sum over?
# `dZ` has one row per example and one column per class. Predict both sums.
import numpy as np
probs = np.array([[.5, .5], [.5, .5], [.5, .5]])
onehot = np.array([[1., 0.], [0., 1.], [0., 1.]])
dZ = (probs - onehot) / 3
print(dZ.sum(axis=0), dZ.sum(axis=1))

# %% [drill]
# Did the caller's weights change?
# Predict what `W` holds after the call, and whether `new_W` is `W`.
import numpy as np
def step_in_place(W, dW):
    W -= 0.5 * dW
    return W
W = np.zeros(2)
new_W = step_in_place(W, np.ones(2))
print(W, new_W is W)

# %% [drill]
# Mean of microbatch means
# One microbatch has one example, the other three. Predict both averages.
import numpy as np
small = np.array([4.])
large = np.array([0., 0., 0.])
print((small.mean() + large.mean()) / 2, np.concatenate([small, large]).mean())

# %% [answer]
def classifier_step(X: np.ndarray, targets: np.ndarray, W: np.ndarray,
                    b: np.ndarray, learning_rate: float,
                    ignore_index: int = -100) -> tuple[np.ndarray, np.ndarray, float]:
    pass

# %%
# Baseline check — All examples valid; one full-batch SGD update.
from torch_judge import check
check("0060_classifier_step_eval.py", stage="baseline")

# %%
# Follow-up — Ignore target `-100` before indexing classes. Only valid examples
# enter the mean loss and gradient. An empty or all-ignored batch returns
# copies of `W` and `b` and loss 0. A zero learning rate still returns fresh arrays.
# Watch out: `W -= lr * dW` writes into the caller's array. Build new arrays
# with `W - lr * dW` instead.
check("0060_classifier_step_eval.py", stage="followup")

# %%
# Production challenge — Validate finite `float64` `X`, `W`, `b` and compatible
# shapes, valid integer targets, and a nonnegative finite learning rate.
# Unequal microbatches must contribute in proportion to valid-example count.
# Watch out: Averaging each microbatch's mean gradient gives a one-example
# microbatch the same weight as a large one.
# Predict: Run a small separable dataset for several steps. Why should the
# loss decrease?
check("0060_classifier_step_eval.py", stage="production")
