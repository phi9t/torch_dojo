# %%
# 0086 · The credit ledger for PPO
# Original dojo exercise based on generalized advantage estimation:
# https://arxiv.org/abs/1506.02438
# Pair it with 0039_ppo_loss.py, which consumes the advantages produced here.

# %%
import torch
from torch_judge import check

# %%
# A rollout stores reward[t], value[t], and next_value[t] for each step.
# The TD residual is reward + gamma*(not done)*next_value - value.
# Work backward: advantage[t] = residual[t] + gamma*lam*(not done)*
# advantage[t+1], stopping at a terminal state or the end of valid data.
# Predict the last step's advantage before running the two-step check.
toy_reward = torch.tensor([[1., 2.]])
toy_value = torch.tensor([[0.5, 1.]])
toy_next = torch.tensor([[1., 0.]])
print('last-step TD residual:', (toy_reward[0, 1] - toy_value[0, 1]).item())

# %% [answer]
def ppo_gae(rewards: torch.Tensor, values: torch.Tensor,
            next_values: torch.Tensor, dones: torch.Tensor,
            valid: torch.Tensor, gamma: float = 0.99,
            lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed [B,T] advantages and value targets for a rollout."""
    pass

# %%
# Baseline: a two-step episode. With lambda=0, the estimator becomes a
# one-step TD residual. Draw the backward arrows before checking.
check('0086_ppo_gae_eval.py', stage='baseline')

# %%
# Follow-up: done=True marks an actual terminal and cuts the trace.
# A time-limit truncation is done=False at the last valid step: use its
# next_value to bootstrap, but do not carry advantage from padded steps.
# valid[B,T] is a True-prefix mask for variable-length rollouts.
check('0086_ppo_gae_eval.py', stage='followup')

# %%
# Production: finite matching floating [B,T] tensors, boolean done/valid,
# B,T>=1, and gamma/lam in [0,1]. Invalid positions return zero. Treat
# rollout rewards and old critic predictions as fixed data: no gradient
# through either output. Reject bad inputs with ValueError; mutate nothing.
check('0086_ppo_gae_eval.py', stage='production')
