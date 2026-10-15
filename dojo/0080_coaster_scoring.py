# %%
# 0080 · Three coasters, one scoring rule
# Goal: Score a roller coaster as $\text{scale} \cdot \text{comfort} \cdot \text{speed}$,
# with the scale and comfort rule chosen by its coaster and lift type.
#
# RX01 adapts a historical Zoox C++ exercise. Formulas are reported;
# validation and this Python function interface are authored.

# %%
import math
from torch_judge import check

# %%
# Predict: Steel speed 20 and bumps/second 2 has comfort
# $\min(1, 1/2, 5/20) = 1/4$, so score $= 2 \cdot (1/4) \cdot 20 = 10$.
print('Expected Steel score:', 10.)

# %%
# Inputs: `coaster_type` Wooden/Steel/Suspended; `lift_type`
# Cable/Chain/Launched; positive finite `speed` and `bumps_per_second`.
# $\text{base} = \min(1, 1/\text{bumps}, 1/\text{speed})$. Wooden scale 1, comfort base.
# Steel scale 2, comfort $\min(1, 1/\text{bumps}, 5/\text{speed})$.
# Suspended scale 0.5, comfort base $+ 0.5$ for Cable, else base.
# Watch out: return `scale * comfort * speed` as a float; every ratio is a true division.

# %% [drill]
# True or floor division?
# The C++ original might divide integers. Predict Steel comfort at speed 20, bumps 2 with `/` and with `//`.
speed, bumps = 20, 2
print(min(1, 1 / bumps, 5 / speed), min(1, 1 // bumps, 5 // speed))

# %% [drill]
# The Steel cap
# Bumps 1. Predict the Steel score at speeds 20, 100, 1000 with comfort $5/\text{speed}$, and with the base $1/\text{speed}$.
for speed in (20., 100., 1000.):
    print(speed, 2 * min(1., 1., 5. / speed) * speed, 2 * min(1., 1., 1. / speed) * speed)

# %% [answer]
def coaster_score(coaster_type: str, speed: float, bumps_per_second: float, lift_type: str) -> float:
    pass

# %%
# Baseline — calculate all three policies and the four reported examples.
check('0080_coaster_scoring_eval.py', stage='baseline')

# %%
# Follow-up — Steel speed cap and Cable bonus at other speeds.
check('0080_coaster_scoring_eval.py', stage='followup')

# %%
# Production — bad names, nonfinite and nonpositive values raise `ValueError`.
# State a Python adaptation limit: the original C++ design/ownership
# discussion is not exercised by this scalar function.
check('0080_coaster_scoring_eval.py', stage='production')
