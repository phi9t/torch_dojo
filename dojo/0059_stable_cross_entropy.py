# %%
# 0059 · Stable cross-entropy, from two logits to ignored batches
# Goal: Implement the mean softmax cross-entropy loss and its manual gradient
# in NumPy. Keep one answer and strengthen it at each stage; no autograd or
# cross-entropy helper.

# %%
import numpy as np

# %%
# Short indexing workout: predict shapes AND values before running.
# Integer targets name one class per example, not a distribution of classes.
index_logits = np.array([[10., 11., 12.], [20., 21., 22.]])
index_targets = np.array([2, 2])
rows = np.arange(len(index_logits))
print("paired:", index_logits[rows, index_targets])
print("Cartesian grid:", index_logits[rows[:, None], index_targets[None, :]])
# Predict: Why does repeating class 2 still select the distinct coordinates
# $(0,2)$ and $(1,2)$? Then try targets `[2,0]` and a three-row, two-class matrix.

# %%
# Implement a per-example loss vector here before reducing it in your answer.
# In this tiny example the first row is uncertain; the other targets are unlikely.
shape_logits = np.array([[0., 0.], [0., 2.], [3., 1.]])
shape_targets = np.array([1, 0, 1])
q_probe = shape_logits - shape_logits.max(axis=1, keepdims=True)
normalizers = np.log(np.exp(q_probe).sum(axis=1, keepdims=True))
selected = q_probe[np.arange(3), shape_targets]
# Predict: What are `normalizers.shape`, `selected.shape` and the shape of their difference?
# `[:, None]` turns $(N,)$ into $(N,1)$, and `keepdims` preserves a reduced axis.
per_example = None  # Replace with an expression whose shape is exactly (3,).

# %%
# Adversarial self-check: run after implementing the expression above.
assert per_example.shape == (3,), "one loss per example is required"
assert np.allclose(per_example, [np.log(2), np.log1p(np.exp(2)),
                                 np.log1p(np.exp(2))])
wrong_grid = normalizers - selected
assert wrong_grid.shape == (3, 3)
assert np.isclose(wrong_grid.mean(), per_example.mean())
# Why: The means match because each normalizer and selected logit occurs $N$
# times in the grid:
# $$\operatorname{mean}(a_i - b_j) = \bar a - \bar b = \operatorname{mean}(a_i - b_i)$$
# Checking only the scalar mean cannot reveal this mistake.

# %% [drill]
# Paired or grid?
# Predict both printed arrays before running: one value per example, or a grid?
import numpy as np
drill_logits = np.array([[10., 11., 12.], [20., 21., 22.]])
drill_targets = np.array([2, 0])
print(drill_logits[np.arange(2), drill_targets])
print(drill_logits[:, drill_targets])

# %% [drill]
# Softmax of large logits
# Predict what the naive softmax prints for logits $[1000, 1001]$, and what the
# shifted one prints.
import numpy as np
big = np.array([1000., 1001.])
with np.errstate(over="ignore", invalid="ignore"):
    print(np.exp(big) / np.exp(big).sum())
shifted = big - big.max()
print(np.exp(shifted) / np.exp(shifted).sum())

# %% [drill]
# Which rows count in the mean?
# Row 3 is ignored. Predict both means.
import numpy as np
per_row = np.array([0.5, 2.0, 0.0])
valid = np.array([True, True, False])
print(per_row.mean(), per_row[valid].mean())

# %%
# Baseline — Why is this naive computation fragile?
logits_probe = np.array([[1000., 1001.]])
with np.errstate(over="ignore"):
    print("naive exp:", np.exp(logits_probe))
# Work with $q = z - \max_c z_c$. For a valid target $y$ the loss is
# $L_i = \log \sum_c e^{q_{ic}} - q_{i,y_i}$, and the mean logit gradient is
# $\nabla_z L = (\operatorname{softmax}(q) - \operatorname{onehot}(y)) / N$.
# Inputs are finite float64 logits $[N, C]$, integer targets $[N]$, $C \ge 2$.
# Return `(mean_loss, dlogits)` without mutating inputs.
#
# Predict: The loss and gradient for logits $[0, 0]$ with target class 1.
# Differentiate $\log \sum_c e^{z_c}$ with respect to $z_j$, then the selected
# target logit. Where does the averaging factor enter? Work it out before
# opening the reference eval file.

# %% [answer]
def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray,
                          ignore_index: int = -100) -> tuple[float, np.ndarray]:
    pass

# %%
# Baseline check — Try equal logits for two classes: loss $\log 2$, gradient
# $[0.5, -0.5]$ when the target is class 1.
from torch_judge import check
check("0059_stable_cross_entropy_eval.py", stage="baseline")

# %%
# Follow-up — A target equal to `ignore_index` contributes neither loss nor
# gradient. Average over VALID rows, not the total batch. Empty or
# all-ignored input returns zero loss and a gradient shaped like logits,
# filled with zeros.
# Watch out: Validate other target IDs before indexing, since NumPy treats
# $-1$ as the last class. Remove ignored rows BEFORE gathering targets.
check("0059_stable_cross_entropy_eval.py", stage="followup")

# %%
# Production challenge — Per-row logit shifts must leave loss and gradient
# unchanged, and each valid gradient row sums to zero. Use float64 finite
# differences, extreme logits, input immutability, and a bounded $O(NC)$
# workspace.
# Predict: Why can exp followed by log underflow for a wrong class?
check("0059_stable_cross_entropy_eval.py", stage="production")

# %%
# Optional hints (not the solution): keep each row's scalar aligned with its
# own selected logit. Only valid rows belong in the denominator. Subtract the
# target contribution once per full (row, class) coordinate.
# The separate eval file explains the derivation and shape answers.
# Continue to 0061_kmeans.py, then the geometry path linked in README.md.
