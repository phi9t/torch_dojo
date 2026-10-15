"""Cosine schedule reference and cumulative checks."""
# %%
import math

# Drill 1: `step / warmup` prints `0.0 0.5 1.0` for steps 0, 1, 2, while
# `(step + 1) / warmup` prints `0.5 1.0 1.5`. Step 0 is the first update and
# must get rate 0; the shifted ramp starts at half the peak and overshoots
# `max_lr` at step 2. The ramp reaches `max_lr` exactly at `step == warmup`,
# which is where the cosine branch takes over at progress 0.

# Drill 2: It prints `0.5 0.6545084971874737`. Progress must run from 0 to 1
# over the decay steps only, $p = (s - w) / (T - w) = 4/8$, so step 6 is the
# cosine midpoint. Dividing by $T$ gives $p = 0.4$ and the schedule never
# reaches `min_lr` at step 10.

# Drill 3: Steps 10, 14, 18 print `0.0`, `0.4999999999999999` and `1.0`.
# Cosine is periodic: past $p = 1$ the curve climbs back toward `max_lr`. That
# is why steps at or after `total_steps` return `min_lr` directly instead of
# evaluating the formula.

# %%
def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    if any(type(value) is not int for value in (step, total_steps, warmup_steps)):
        raise ValueError("step counts must be integers")
    if total_steps <= 0 or step < 0 or not 0 <= warmup_steps <= total_steps:
        raise ValueError("invalid step range")
    if any(not isinstance(rate, (int, float)) or not math.isfinite(rate) for rate in (min_lr, max_lr)):
        raise ValueError("rates must be finite")
    if not 0 <= min_lr <= max_lr:
        raise ValueError("require 0 <= min_lr <= max_lr")
    if step < warmup_steps:
        return float(max_lr * step / warmup_steps)
    if step >= total_steps:
        return float(min_lr)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return float(min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress)))

# %% [verifiers]
def verify_baseline_landmarks(cosine_lr_schedule):
    samples = {0: 0.0, 1: 0.5, 2: 1.0, 6: 0.5, 10: 0.0}
    for step, expected in samples.items():
        result = cosine_lr_schedule(step, 10, 2, 1.0)
        assert isinstance(result, float) and abs(result - expected) < 1e-12


def verify_baseline_range_and_shape(cosine_lr_schedule):
    values = [cosine_lr_schedule(s, 100, 10, 0.01, 0.001) for s in range(101)]
    assert all(0.0 <= v <= 0.01 + 1e-15 for v in values[:10])
    assert all(0.001 - 1e-15 <= v <= 0.01 + 1e-15 for v in values[10:])
    assert all(a <= b + 1e-15 for a, b in zip(values[:10], values[1:11]))
    assert all(a >= b - 1e-15 for a, b in zip(values[10:100], values[11:101]))


def verify_followup_no_warmup_and_clamped_end(cosine_lr_schedule):
    assert cosine_lr_schedule(0, 8, 0, 0.2, 0.02) == 0.2
    assert cosine_lr_schedule(10**9, 8, 0, 0.2, 0.02) == 0.02


def verify_followup_full_warmup(cosine_lr_schedule):
    assert abs(cosine_lr_schedule(3, 4, 4, 1.0) - 0.75) < 1e-12
    assert cosine_lr_schedule(4, 4, 4, 1.0, 0.1) == 0.1


def verify_production_invalid_contract(cosine_lr_schedule):
    inputs = ((-1, 10, 2, 1.0, 0.0), (0, 0, 0, 1.0, 0.0),
              (0, 10, 11, 1.0, 0.0), (0, 10, 2, -1.0, 0.0),
              (0, 10, 2, 0.5, 0.6), (0.5, 10, 2, 1.0, 0.0),
              (0, 10, 2, float("nan"), 0.0))
    for args in inputs:
        try:
            cosine_lr_schedule(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject {args}")


def verify_production_reproducible_large_index(cosine_lr_schedule):
    args = (10**30, 1000, 12, 0.004, 0.0002)
    assert cosine_lr_schedule(*args) == 0.0002
    assert cosine_lr_schedule(*args) == cosine_lr_schedule(*args)

# %%
from torch_judge import evaluate
if not evaluate("0030_cosine_lr_eval.py", cosine_lr_schedule):
    raise SystemExit(1)
