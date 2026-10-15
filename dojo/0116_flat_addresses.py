# %%
# 0116 · Flat addresses
# Goal: turn multi-axis coordinates into one row-major address and back, without
# letting an out-of-range coordinate land on a real cell.
# Fundamentals behind pillar ids in 0107/0108. Authored practice.

# %%
import torch
from torch_judge import check

# %%
# Baseline — A [batch=2, y=2, x=3] grid is stored as 12 flat rows.
lin = torch.arange(12).reshape(2, 2, 3)

# %% [drill]
# Row-major
# Predict the flat address of (b=1, y=0, x=2) by reading `lin`.
print(lin[1, 0, 2])

# %% [drill]
# One past the end
# Predict the flat address of (b=0, y=0, x=3), where x is out of range (x < 3), and compare it with (0, 1, 0).
print((0 * 2 + 0) * 3 + 3, (0 * 2 + 1) * 3 + 0)

# %% [drill]
# Same key, different cells?
# Predict whether a flat address alone can tell those two coordinates apart.

# %%
# Contract: coords int64 [K,D], shape a tuple of D positive ints. Return int64
# [K] row-major addresses (last axis fastest). Check EVERY coordinate against
# its own axis bound, not just the final address. prod(shape) must fit in int64.
# Empty K works. The eval brings its own inverse to test round trips.

# %% [answer]
def ravel_index(coords: torch.Tensor, shape: tuple) -> torch.Tensor:
    pass

# %%
check('0116_flat_addresses_eval.py', stage='baseline')

# %%
# Follow-up — Round trips through an inverse, the stride oracle, empty input.
check('0116_flat_addresses_eval.py', stage='followup')

# %%
# Production — Reject out-of-range coordinates per axis, bad shapes, wrong
# dtypes, and shapes whose product overflows int64.
check('0116_flat_addresses_eval.py', stage='production')
