# %%
# 0030 · Cosine learning rate with linear warmup
# Goal: Return the learning rate at an optimizer `step`: a linear ramp from 0
# to `max_lr` over `warmup_steps`, then a cosine decay to `min_lr` at
# `total_steps`.
#
# Run cells interactively; revise the single answer as stages grow.

# %%
import math

# %%
# Baseline — steps are optimizer updates, starting at zero. With total=10,
# warmup=2, max=1, min=0: step 0 -> 0, step 2 -> 1, step 6 -> 0.5,
# step 10 -> 0.
# Predict: Why does step 6 lie halfway through decay?
print("cosine midpoint:", (1 + math.cos(math.pi / 2)) / 2)
# After warmup, with $p = (s - w) / (T - w)$, the rate is
# $$\eta = \eta_{min} + \tfrac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos \pi p)$$
# Watch out: return a Python float. The warmup endpoint belongs to the cosine branch.

# Derive the cosine value at step 6 after the two warmup steps.

# %% [drill]
# Where does warmup start?
# With `warmup_steps=2` and `max_lr=1`, predict the ramp at steps 0, 1, 2 for `step / warmup` and for `(step + 1) / warmup`.
for step in range(3):
    print(step, 1.0 * step / 2, 1.0 * (step + 1) / 2)

# %% [drill]
# Halfway through what?
# Total 10, warmup 2. Predict the rate at step 6 when progress divides by the decay length $T - w$, and when it divides by $T$.
import math
print((1 + math.cos(math.pi * (6 - 2) / (10 - 2))) / 2, (1 + math.cos(math.pi * (6 - 2) / 10)) / 2)

# %% [drill]
# Past the end
# Without a clamp, predict the cosine formula's rate at steps 10, 14 and 18 for total 10, warmup 2.
import math
for step in (10, 14, 18):
    print(step, (1 + math.cos(math.pi * (step - 2) / (10 - 2))) / 2)

# %% [answer]
def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    pass

# %%
from torch_judge import check
check("0030_cosine_lr_eval.py", stage="baseline")

# %%
# Follow-up — zero warmup starts at `max_lr`. After `total_steps`, hold `min_lr`.
# A full warmup (`warmup_steps == total_steps`) ramps to `max_lr` at its final
# warmup step, then returns `min_lr` at `total_steps`. State this discontinuity.
check("0030_cosine_lr_eval.py", stage="followup")

# %%
# Production challenge — reject invalid counts/rates and non-integer steps.
# Require `total_steps > 0`, `0 <= warmup_steps <= total_steps`, `step >= 0`,
# and finite `0 <= min_lr <= max_lr`. Calls are pure and deterministic.
# Time $O(1)$, memory $O(1)$, even for a large step index.
check("0030_cosine_lr_eval.py", stage="production")
