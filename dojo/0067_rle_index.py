# %%
# 0067 · Run-Length Random Access
# Goal: Return the character at a zero-based decoded index of run-length data
# without expanding the runs. Runs are (single-character string, positive
# integer count) tuples. Edit the answer cell and rerun it in Jupyter.
#
# Watch out: an index outside $[0, \text{decoded length})$ raises `IndexError`.

# %%
# Predict: where does index 3 fall? Each end is exclusive.
runs_probe = [("B", 1), ("A", 2), ("E", 3)]
print("run ends:", [1, 3, 6], "index 3 should be E")

# %% [drill]
# Exclusive run ends
# The runs `B×1, A×2, E×3` end at 1, 3 and 6. Predict the decoded characters for indices 0–5 when the scan stops at the first run with `index < end`, and with `index <= end`.
import numpy as np
runs = [("B", 1), ("A", 2), ("E", 3)]
ends = np.cumsum([count for _, count in runs])
print("ends:", ends)
print("index < end: ", [next(c for (c, _), e in zip(runs, ends) if i < e) for i in range(6)])
print("index <= end:", [next(c for (c, _), e in zip(runs, ends) if i <= e) for i in range(6)])

# %% [drill]
# Binary search over prefix ends
# With prefix ends `[1, 3, 6]`, predict which run `np.searchsorted` picks for indices 0–5 with `side="right"` and with `side="left"`.
import numpy as np
ends = np.array([1, 3, 6])
idx = np.arange(6)
print("side='right':", np.searchsorted(ends, idx, side="right"))
print("side='left': ", np.searchsorted(ends, idx, side="left"))

# %% [drill]
# Is `True` a count?
# Predict whether a count of `True` passes `isinstance(count, int)` and `count > 0`.
import numpy as np
count = True
print(isinstance(count, int), count > 0, isinstance(count, bool))

# %% [answer]
def rle_index(runs: list[tuple[str, int]], index: int) -> str:
    pass

# %%
# Baseline — Accumulate counts and stop at the first end greater than index.
# Check exact run boundaries, empty input, and negative/out-of-range indices.
from torch_judge import check
check("0067_rle_index_eval.py", stage="baseline")

# %%
# Follow-up — Counts can exceed $10^{15}$. Never construct the decoded string.
# Watch out: reject malformed runs with `ValueError`, including zero/negative/bool
# counts, multi-character symbols, and malformed tuples. Validate every run,
# even if the requested character appears before a malformed later run.
check("0067_rle_index_eval.py", stage="followup")

# %%
# Production — Retain $O(\text{number of runs})$ time and $O(1)$ extra space, even
# with huge counts. Do not mutate runs.
# Predict: for many queries over immutable runs, how do prefix ends plus binary
# search reduce each lookup to $O(\log R)$?
check("0067_rle_index_eval.py", stage="production")
