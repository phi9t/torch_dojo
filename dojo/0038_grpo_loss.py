# %%
# 0038 · Group-relative policy loss
# Work in top-level cells and improve one answer through three stages.
# This simplified objective is an authored coding exercise, not a complete
# GRPO training recipe with importance ratios and KL controls.

# %%
import torch
from torch import Tensor

# %%
# Baseline: logps/rewards/group_ids are [B]. For each group, subtract its
# reward mean and divide by population std (unbiased=False) + eps. Return
# -mean(detached_advantage * logps). Equal-sized groups are a first probe.
probe_rewards = torch.tensor([1., 3., 10., 12.])
for group in (probe_rewards[:2], probe_rewards[2:]):
    print('group advantages:', (group-group.mean()) / (group.std(unbiased=False)+1e-5))

# Derive the advantage of reward 3 in the first two-item group.

# %% [answer]
def grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor,
              eps: float = 1e-5) -> Tensor:
    pass

# %%
from torch_judge import check
check('0038_grpo_loss_eval.py', stage='baseline')

# %%
# Follow-up: group ids need not be adjacent or nonnegative. A singleton has
# zero advantage, and reward gradients must be stopped. Check unequal group
# sizes: the final mean weights examples, not groups, equally.
check('0038_grpo_loss_eval.py', stage='followup')

# %%
# Production: constant-reward groups must remain finite. An empty batch
# returns a graph-connected zero. Validate shapes, finite data, integer ids,
# and eps > 0 before computing statistics. A loop over groups is fine;
# avoid a per-example Python loop.
check('0038_grpo_loss_eval.py', stage='production')
