# %%
# 0075 · Choose K-means landmarks
# Goal: Pick $k$ K-means++ starting centers from the rows of `X`, drawing each
# new center with probability proportional to its squared distance from the
# nearest center already chosen.
#
# A reported clustering follow-up asked for better initialization. The
# squared-distance rule and its edge policies here are authored extensions.

# %%
import numpy as np

# %%
# Baseline — Suppose 0 is selected first from `[0,1,9]`. The next draw has
# weights 0, 1, and 81.
# Predict: Which remaining point is more likely?
X_probe = np.array([[0.], [1.], [9.]])
print("points:", X_probe.ravel(), "weights after choosing 0:", [0, 1, 81])
# Contract: finite numeric `X` $(N, D)$, $1 \le k \le N$, caller-provided `Generator`.
# Draw the first row index uniformly. Draw each later unused row index $i$ with
# probability $$P(i) = \frac{D(x_i)^2}{\sum_j D(x_j)^2}$$ where $D(x)$ is the
# distance from $x$ to its nearest chosen row.
# Return a copied $(k, D)$ array of centers, preserving selection order.

# %% [drill]
# Distance or squared distance?
# Center 0 is chosen from points 0, 1, 9. Predict the next-draw probabilities when weights are distances and when they are squared.
import numpy as np
d = np.array([0., 1., 9.])
print("distance:", (d / d.sum()).round(3), " squared:", (d**2 / (d**2).sum()).round(3))

# %% [drill]
# Same seed, same draws?
# Predict the first line (two fresh Generators, both seeded 7) and the second (one Generator drawn from twice).
import numpy as np
print(np.random.default_rng(7).integers(10, size=3), np.random.default_rng(7).integers(10, size=3))
rng = np.random.default_rng(7)
print(rng.integers(10, size=3), rng.integers(10, size=3))

# %% [drill]
# Nearest center, or latest?
# Points 0, 1, 9, 10; centers 0 then 10 are chosen. Predict the weights from distance to the latest center only, and to the nearest one.
import numpy as np
x = np.array([0., 1., 9., 10.])
d_first = (x - 0.) ** 2
d_second = (x - 10.) ** 2
print("latest only:", d_second, " nearest:", np.minimum(d_first, d_second))

# %% [answer]
# Revise this same answer through all three stages.
def kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    pass

# %%
from torch_judge import check
check("0075_kmeans_plus_plus_eval.py", stage="baseline")

# %%
# Follow-up — Distinct ROW INDICES can hold identical coordinates. If all
# remaining squared-distance weights are zero, draw uniformly from unused
# indices. Try three identical points with `k=3`.
# Predict: Why would a direct division by `sum(weights)` fail here?
check("0075_kmeans_plus_plus_eval.py", stage="followup")

# %%
# Production — Repeated runs with freshly seeded Generators must match.
# Reject invalid dimensions, nonfinite values, invalid `k`, and a missing RNG.
# Do not mutate `X` or hold an $(N, N, D)$ pairwise tensor. Keep an $O(N)$ nearest
# squared-distance vector and update it after each choice.
# Watch out: normalize weights by their maximum before summing to reduce
# overflow risk; document what happens if even one squared distance overflows.
check("0075_kmeans_plus_plus_eval.py", stage="production")
