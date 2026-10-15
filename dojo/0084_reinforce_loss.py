# %%
# 0084 · Give earlier choices credit for later rewards
# Original dojo exercise inspired by Deep-ML's REINFORCE topic:
# https://www.deep-ml.com/problems/122

# %%
import torch
from torch_judge import check

# %%
# A policy produced log probabilities for actions actually taken.
# At time t, reward-to-go G[t] includes reward[t] plus discounted later
# rewards in that episode. Predict G for [1,2,4] when gamma=0.5.
toy_rewards = torch.tensor([[1., 2., 4.]])
print('Immediate rewards:', toy_rewards)

# %%
# Return -mean_over_episodes(sum_over_valid_time(log_prob[t] * advantage[t])).
# At baseline, advantage=reward-to-go. This is a POLICY loss: gradient
# must flow into log_probs; observed rewards are treated as fixed data.
# Keep [B,T] axes visible, even for one episode.

# %% [answer]
def reinforce_loss(log_probs: torch.Tensor, rewards: torch.Tensor,
                   gamma: float = 0.99, baseline: torch.Tensor | None = None,
                   valid: torch.Tensor | None = None) -> torch.Tensor:
    """Return a scalar episodic policy loss with reward-to-go credit."""
    pass

# %%
# Baseline: calculate discounted returns backward and check the gradient
# with respect to each chosen-action log probability.
check('0084_reinforce_loss_eval.py', stage='baseline')

# %%
# Follow-up: some episodes end early. valid[B,T] is a prefix mask:
# True for actual steps, then False for padding. Padding contributes no
# return or gradient. Subtract an optional [B,T] baseline, detached from
# this policy loss so it can have its own training objective.
check('0084_reinforce_loss_eval.py', stage='followup')

# %%
# Production contract: finite matching floating [B,T] tensors on one
# device/dtype, B,T >= 1, 0 <= gamma <= 1, boolean prefix mask, finite
# baseline of the same shape/device/dtype. Raise ValueError on bad input.
# A wholly padded batch returns graph-connected zero; noncontiguous
# log-probability tensors and policy-parameter gradients must work.
check('0084_reinforce_loss_eval.py', stage='production')
