# %%
# 0087 · A group of answers, one clipped update
# Original dojo exercise based on the group-relative objective in
# https://arxiv.org/abs/2402.03300
# The small 0038_grpo_loss.py and 0039_ppo_loss.py are useful warm-ups.

# %%
import torch
from torch_judge import check

# %%
# Several completions belong to the same prompt, identified by group_id.
# Compare their rewards within that prompt: (reward-mean)/(population_std+eps).
# A singleton or constant-reward group has zero advantage. Predict the
# signs for rewards [1,3] before running this probe.
toy_rewards = torch.tensor([1., 3.])
print('relative rewards:', (toy_rewards - toy_rewards.mean()) /
      (toy_rewards.std(unbiased=False) + 1e-5))

# %%
# For each valid completion token, compare current and frozen old log-probs
# with a PPO-style clipped ratio. Add kl_coef * (exp(log_ref-log_current)
# - (log_ref-log_current) - 1), with the reference fixed. First average
# valid tokens INSIDE each completion, then average completions. That
# denominator choice matters when answers have different lengths.

# %% [answer]
def grpo_tokens(current_logps: torch.Tensor, old_logps: torch.Tensor,
                reference_logps: torch.Tensor, rewards: torch.Tensor,
                group_ids: torch.Tensor, completion_mask: torch.Tensor,
                clip_ratio: float = 0.2, kl_coef: float = 0.01,
                eps: float = 1e-5) -> torch.Tensor:
    """Return a completion-normalized clipped group-relative token loss."""
    pass

# %%
# Baseline: two equal-length answers to one prompt. Verify which log-prob
# receives a positive gradient; then clip improvements beyond the range.
check('0087_grpo_tokens_eval.py', stage='baseline')

# %%
# Follow-up: nonadjacent group IDs, unequal lengths, and frozen reward,
# old-policy, and reference data. Explain why global token averaging gives
# a different answer when one completion has twice as many tokens.
check('0087_grpo_tokens_eval.py', stage='followup')

# %%
# Production: finite matching [B,T] log-probs, finite [B] rewards, integer
# [B] group IDs, boolean [B,T] mask with >=1 token per nonempty row;
# 0<=clip_ratio<1, kl_coef>=0, eps>0. Raise ValueError on bad inputs.
# Empty batch yields graph-connected zero; masked extreme values must not
# overflow the objective. An active extreme KL can still be unbounded.
check('0087_grpo_tokens_eval.py', stage='production')
