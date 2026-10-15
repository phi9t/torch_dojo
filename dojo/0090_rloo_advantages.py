# %%
# 0090 · Leave one completion out of its prompt baseline
# Original dojo exercise prompted by https://www.deep-ml.com/problems/1266
# 0084_reinforce_loss.py and 0087_grpo_tokens.py are useful warm-ups.

# %%
import torch
from torch_judge import check

# %%
# Three responses to prompt A earned [1, 4, 7]; prompt B earned [10, 14].
# For A's first response, the other two average 5.5, so its advantage is
# -4.5. Predict all five advantages before editing the answer.
toy_rewards = torch.tensor([1., 10., 4., 14., 7.])
toy_groups = torch.tensor([3, 8, 3, 8, 3])
print('rewards:', toy_rewards, 'prompt groups:', toy_groups)

# %%
# Inputs: current_logps, rewards, and group_ids are one-dimensional [N]
# tensors. Each log-probability is for one entire sampled completion.
# Return (advantages [N], scalar loss), with loss equal to
# -mean(current_logps * advantages) across ALL N completions.
# Baseline for a completion is the mean reward of the OTHER completions
# with its group ID. A singleton has advantage zero.

# %% [answer]
def rloo_advantages(current_logps: torch.Tensor, rewards: torch.Tensor,
                    group_ids: torch.Tensor,
                    rollout_logps: torch.Tensor | None = None,
                    reference_logps: torch.Tensor | None = None,
                    kl_coef: float = 0.) -> tuple[torch.Tensor, torch.Tensor]:
    """Return leave-one-out advantages and a mean policy-gradient loss."""
    pass

# %%
# Baseline: use only a response's own prompt group and leave that response
# out of the baseline. Keep rewards fixed when differentiating the loss.
check('0090_rloo_advantages_eval.py', stage='baseline')

# %%
# Follow-up: group IDs can be interleaved and groups can have unequal
# sizes. Optional fixed reward shaping uses
# shaped_reward = reward - kl_coef * (rollout_logp - reference_logp).
# A positive coefficient requires both frozen vectors; the current policy
# log-probability never enters this reward shaping term.
check('0090_rloo_advantages_eval.py', stage='followup')

# %%
# Production: require finite floating [N] log-probabilities and rewards
# with matching dtype/device, integer [N] IDs on that device, and finite
# nonnegative kl_coef. Optional frozen vectors, when supplied, must be
# finite floating [N] with matching dtype/device. Raise ValueError for
# invalid inputs. Empty N returns empty advantages and graph-connected
# zero loss. Do not mutate inputs; only current_logps gets gradients.
check('0090_rloo_advantages_eval.py', stage='production')
