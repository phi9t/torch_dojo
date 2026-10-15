# %%
# 0068 · Union of Half-Open Intervals
# Goal: Return the total length covered by a set of half-open intervals
# $[s, e)$, given as an $(N, 2)$ array with $e \ge s$, as a float.
#
# Watch out: touching endpoints add no length, and overlaps count once:
# $[0, 2) \cup [1, 3)$ covers 3, not 4.

# %%
import numpy as np

# %%
# Predict: What does summing the durations print, and how far is it from the union length 5?
intervals_probe = np.array([[0., 2.], [1., 3.], [3., 5.]])
print("naive sum:", (intervals_probe[:, 1] - intervals_probe[:, 0]).sum())
print("union length: 5.0")

# %% [drill]
# Sorted by what?
# The sweep merges intervals in order. Predict the starts after sorting by start and after sorting by end.
import numpy as np
x = np.array([[5., 9.], [0., 10.], [2., 3.]])
print(x[np.argsort(x[:, 0])][:, 0])
print(x[np.argsort(x[:, 1])][:, 0])

# %% [drill]
# Touching or overlapping?
# Predict the union length of $[0, 1)$ and $[1, 2)$, and of $[0, 1)$ and $[1.5, 2)$.
import numpy as np
def union(intervals):
    ordered = intervals[np.argsort(intervals[:, 0], kind="stable")]
    total, (start, end) = 0.0, ordered[0]
    for lo, hi in ordered[1:]:
        if lo > end:
            total, start, end = total + end - start, lo, hi
        else:
            end = max(end, hi)
    return total + end - start
print(union(np.array([[0., 1.], [1., 2.]])), union(np.array([[0., 1.], [1.5, 2.]])))

# %% [drill]
# The nested interval
# A short interval sits inside a long one. Predict the merged end if the sweep sets `end = hi` instead of `end = max(end, hi)`.
import numpy as np
ordered = np.array([[0., 10.], [2., 3.]])
end = ordered[0, 1]
print("end = hi:", ordered[1, 1], " end = max(end, hi):", max(end, ordered[1, 1]))

# %% [answer]
def interval_union(intervals: np.ndarray) -> float:
    pass

# %%
# Baseline — Sort by start and track the current merged end. An empty
# $(0, 2)$ array returns 0. Nested and touching intervals obey half-open endpoints.
from torch_judge import check
check("0068_interval_union_eval.py", stage="baseline")

# %%
# Follow-up — Duplicate and zero-length intervals add no coverage. Coordinates
# may be negative.
# Watch out: reject reversed, nonfinite, or incorrectly shaped input with
# `ValueError` before sorting.
check("0068_interval_union_eval.py", stage="followup")

# %%
# Production — Sorting costs $O(N \log N)$, then one $O(N)$ sweep; extra
# storage is $O(N)$ for the sorted copy. Do not mutate the input.
# Predict: How can a stream already sorted by start be consumed in $O(1)$ extra space?
check("0068_interval_union_eval.py", stage="production")
