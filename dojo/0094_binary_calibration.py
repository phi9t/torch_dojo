# %%
# 0094 · Fit binary probability calibration
# Goal: Fit a map from raw model scores to probabilities on calibration
# labels, by Platt scaling or isotonic regression, and apply it to separate
# query scores.
#
# Original dojo exercise prompted by `https://www.deep-ml.com/problems/1466`
# `0069_calibration_metrics.py` measures calibration after prediction.

# %%
import numpy as np
from torch_judge import check

# %%
# A model reports raw scores `[-1, 1]` for outcomes `[0, 1]`.
# Predict: What probabilities should a fitted map give scores -1, 0, and 1?
calibration_scores = np.array([-1., 1.])
calibration_labels = np.array([0, 1])
query_scores = np.array([-1., 0., 1.])
print('raw scores:', query_scores, 'labels used only for fitting:', calibration_labels)
# Fit only on `calibration_scores` and `calibration_labels`; return a $(Q,)$
# floating array of probabilities for separate `query_scores`. All three
# arrays are one-dimensional; calibration length $N \ge 1$, query length $Q \ge 0$.
# Inputs must be finite real scores and binary integer labels of length $N$.
# `method` is exactly `'platt'` or `'isotonic'`; invalid inputs raise `ValueError`.
# Do not modify any input. No query labels enter the function.

# %% [drill]
# Squash once
# Scores $-3, 0, 3$ are logits. Predict their sigmoid, and the sigmoid applied again to those probabilities.
import numpy as np
scores = np.array([-3., 0., 3.])
probs = 1 / (1 + np.exp(-scores))
print(probs.round(3), (1 / (1 + np.exp(-probs))).round(3))

# %% [drill]
# Why penalize the slope?
# Scores $-1, 1$ with labels $0, 1$ are perfectly separated. Predict the logistic loss at slopes 1, 10, 100, without and with the $0.0005 a^2$ penalty.
import numpy as np
for a in (1., 10., 100.):
    loss = np.log1p(np.exp(-a))
    print(a, round(loss, 6), round(loss + .0005 * a * a, 6))

# %% [drill]
# Tied scores, two orders
# Scores `[0, 0, 1]` sorted, with the two tied labels in either order. Predict the pooled levels for labels `[0, 1, 0]` and `[1, 0, 0]`.
import numpy as np
def pav(labels):
    blocks = []
    for y in labels:
        blocks.append([y, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    return [round(t / c, 3) for t, c in blocks for _ in range(c)]
print(pav([0, 1, 0]), pav([1, 0, 0]))

# %% [answer]
def calibrate_binary(scores: np.ndarray, labels: np.ndarray,
                     query_scores: np.ndarray, method: str = 'platt') -> np.ndarray:
    pass

# %%
# Baseline — fit $\sigma(a s + b)$ by minimizing
# $$\frac{1}{N} \sum_i \ell(y_i, \sigma(a s_i + b)) + \frac{10^{-3}}{2}(a^2 + b^2)$$
# with $\ell$ the binary logistic loss.
# Watch out: penalize BOTH parameters. Use a numerically stable loss and
# return fitted probabilities, not raw scores.
check('0094_binary_calibration_eval.py', stage='baseline')

# %%
# Follow-up — for `'isotonic'`, first combine identical scores into one
# weighted label mean. Pool adjacent fitted levels whenever monotonicity
# would fall. Interpolate linearly between distinct fitted score knots;
# clamp queries outside the score range. A single knot gives a constant.
check('0094_binary_calibration_eval.py', stage='followup')

# %%
# Production — constant scores and one-class labels must remain finite.
# Check shapes, dtypes, binary labels, methods, and nonfinite values.
# Empty queries return an empty float array. Keep calibration and query
# arrays unchanged; fitting must use no held-out outcome information.
check('0094_binary_calibration_eval.py', stage='production')
