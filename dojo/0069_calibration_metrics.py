# %%
# 0069 · Multiclass calibration metrics
# Goal: Compute expected calibration error (ECE), the multiclass Brier score
# and negative log-likelihood (NLL) for a batch of predicted class
# probabilities, all in NumPy.
# Authored practice variant; see practice_bank/drills.tsv for related drills.
# Revise the same answer cell as you advance through all three stages.

# %%
import numpy as np

# %%
# Baseline — Three per-example metrics, each averaged over the batch.
# For `p=[[.8,.2],[.6,.4]]` and `labels=[0,1]`, the confidences are .8 and
# .6, while correctness is 1 and 0.
# Predict: ECE with two bins.
probe_p = np.array([[.8, .2], [.6, .4]])
probe_y = np.array([0, 1])
print("confidence, correctness:", probe_p.max(axis=1), probe_p.argmax(axis=1) == probe_y)
# Input: finite probabilities $[N,C]$ with $N \ge 1$, $C \ge 2$; integer labels $[N]$.
# Rows sum to one. Return `(ECE, multiclass Brier, NLL)`, each a mean over $N$.
# Brier sums squared class errors within a row, then averages rows:
# $$\text{Brier} = \frac{1}{N} \sum_i \sum_c (p_{ic} - \mathbf{1}[c = y_i])^2$$
# Watch out: Confidence is the row's largest probability, not the probability
# of the true class. The two differ exactly on the wrong predictions.

# %% [drill]
# Which bin does 0.5 land in?
# With two bins, predict the bin index of each confidence under both rules.
import numpy as np
confidence = np.array([0.2, 0.5, 1.0])
bins = 2
print(np.minimum((confidence * bins).astype(int), bins - 1))
print(np.clip(np.ceil(confidence * bins).astype(int) - 1, 0, bins - 1))

# %% [drill]
# Sum, then average?
# Predict the row-summed Brier score and the mean over every entry.
import numpy as np
p = np.array([[.8, .2], [.6, .4]])
y = np.array([0, 1])
squared = (p - np.eye(2)[y]) ** 2
print(round(squared.sum(axis=1).mean(), 4), round(squared.mean(), 4))

# %% [drill]
# Confidence or true-class probability?
# Predict each row's confidence, its true-class probability, and correctness.
import numpy as np
p = np.array([[.8, .2], [.6, .4]])
y = np.array([0, 1])
print(p.max(axis=1), p[np.arange(2), y], p.argmax(axis=1) == y)

# %% [answer]
def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray,
                        bins: int = 10) -> tuple[float, float, float]:
    pass

# %%
from torch_judge import check
check("0069_calibration_metrics_eval.py", stage="baseline")

# %%
# Follow-up — Equal-width confidence bins are $[0,1/B], (1/B,2/B], \ldots$.
# Empty bins contribute zero. ECE weights each bin $b$ by its share of examples:
# $$\text{ECE} = \sum_b \frac{|b|}{N} \, \lvert \operatorname{acc}(b) - \operatorname{conf}(b) \rvert$$
# Watch out: An exact interior edge belongs to the bin on its LEFT, and
# confidence 1 belongs to the last bin. `floor(conf * B)` puts both one bin
# too far right.
#
# For the probe, the first prediction is correct and the second is wrong:
# ECE $= .2$, Brier $= .4$, NLL $= -(\log .8 + \log .4)/2$.
check("0069_calibration_metrics_eval.py", stage="followup")

# %%
# Production — Reject bad shapes, labels, bins, non-finite values, entries
# outside $[0,1]$, and rows that do not sum to one (absolute tolerance `1e-10`).
# Argmax ties choose the lowest class index. Keep NLL finite when a true
# class has probability zero by clipping only that log input to float64 tiny.
# Do not mutate inputs. Memory should be $O(NC+B)$, with no `[N,C,C]` tensor.
check("0069_calibration_metrics_eval.py", stage="production")
