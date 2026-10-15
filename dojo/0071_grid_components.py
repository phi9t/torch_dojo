# %%
# 0071 · Grid Connected Components
# Goal: Count the `True` regions in a boolean grid of shape $(H, W)$. One
# answer, three checks.
#
# Watch out: only up/down/left/right cells connect; diagonal contact does not.

# %%
import numpy as np

# %%
# Predict: two diagonal True cells are two components under 4-connectivity.
grid_probe = np.array([[1, 0], [0, 1]], dtype=bool)
print(grid_probe, "expected components: 2")

# %% [drill]
# Four neighbours or eight?
# An X of five `True` cells touches only at corners. Predict the component count with 4-connectivity and with 8-connectivity.
import numpy as np
grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
def count(grid, steps):
    seen = np.zeros_like(grid); n = 0
    for start in zip(*np.nonzero(grid)):
        if seen[start]:
            continue
        n += 1; seen[start] = True; stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if 0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1] and grid[q] and not seen[q]:
                    seen[q] = True; stack.append(q)
    return n
four = [(-1, 0), (1, 0), (0, -1), (0, 1)]
eight = four + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
print("4-connected:", count(grid, four), " 8-connected:", count(grid, eight))

# %% [drill]
# Mark on push or on pop?
# Flood a $3 \times 3$ all-`True` grid from a corner. Predict how many stack pushes happen when cells are marked as they are pushed, and when they are marked only as they are popped.
import numpy as np
def pushes(grid, mark_on_push):
    seen = np.zeros_like(grid); stack = [(0, 0)]; total = 1
    if mark_on_push:
        seen[0, 0] = True
    while stack:
        r, c = stack.pop()
        if not mark_on_push:
            if seen[r, c]:
                continue
            seen[r, c] = True
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1] and grid[q] and not seen[q]:
                if mark_on_push:
                    seen[q] = True
                stack.append(q); total += 1
    return total
print("pushes, mark on push:", pushes(np.ones((3, 3), dtype=bool), True),
      " mark on pop:", pushes(np.ones((3, 3), dtype=bool), False))

# %% [drill]
# How deep can recursion go?
# Predict Python's default recursion limit and the number of cells in a $1 \times 3000$ chain.
import sys
import numpy as np
print(sys.getrecursionlimit(), np.ones((1, 3000), bool).sum())

# %% [answer]
def count_grid_components(grid: np.ndarray) -> int:
    pass

# %%
# Baseline — Scan every cell. When a new `True` cell appears, grow one component
# with a stack or queue and mark visited cells as soon as you enqueue them.
from torch_judge import check
check("0071_grid_components_eval.py", stage="baseline")

# %%
# Follow-up — Use iterative traversal; a chain of 3000 cells exceeds usual
# recursion depth. Empty shapes are valid.
# Watch out: reject non-2D or non-bool input with `ValueError`; silently
# treating integers as masks changes the contract.
check("0071_grid_components_eval.py", stage="followup")

# %%
# Production — $O(HW)$ time and visited storage. A checkerboard has many tiny
# components. Do not mutate the learner's grid; transpose preserves count.
check("0071_grid_components_eval.py", stage="production")
