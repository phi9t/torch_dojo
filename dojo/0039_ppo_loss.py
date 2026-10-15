# %%
# 0039 · PPO clipped surrogate
# Revise the same answer as the contract grows. This is only the policy
# surrogate; a full PPO update also chooses advantage and value estimators.

# %%
import torch
from torch import Tensor

# %%
# Baseline: finite [B] log-probs and advantages. Ratio = exp(new-old).
# Return -mean(min(ratio*advantage, clamp(ratio,1-c,1+c)*advantage)).
# Predict which branch wins for ratio=2 with advantage +1 versus -1.
ratio = torch.tensor([2., 2.])
adv = torch.tensor([1., -1.])
print('unclipped:', ratio*adv, 'clipped:', ratio.clamp(.8, 1.2)*adv)

# %% [answer]
def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,
             clip_ratio: float = 0.2) -> Tensor:
    pass

# %%
from torch_judge import check
check('0039_ppo_loss_eval.py', stage='baseline')

# %%
# Follow-up: old log-probs and advantages are fixed, so detach them.
# Clip direction depends on the advantage sign; a zero advantage contributes
# no gradient. Mean reduction is over the entire batch.
check('0039_ppo_loss_eval.py', stage='followup')

# %%
# Production: large positive ratio for positive advantage and large negative
# ratio for negative advantage are safely clipped. Avoid exp(1000) on those
# inactive branches while retaining the unclipped value when it wins.
# Empty batch yields graph-connected zero; reject invalid shapes and c outside
# [0,1). An active extreme ratio may legitimately overflow this objective.
check('0039_ppo_loss_eval.py', stage='production')
