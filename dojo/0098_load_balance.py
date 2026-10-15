# %%
# 0098 · Can a lower load loss hide a worse traffic jam?
# An MoE balance surrogate combines hard route counts with soft scores.
# This original small-tensor exercise follows the diagnostic in the
# Olmo-core 3 report, not its distributed training implementation:
# https://allenai.org/papers/olmocore3
# For context on Qwen3's global-batch balancing, see:
# https://arxiv.org/html/2505.09388#S2

# %%
import torch
from torch_judge import check

# %%
# For one batch, let f[e] = requests_to_e/(T*K), and P[e] be the mean
# normalized score across tokens. The surrogate is E * dot(f,P), while
# requested max/mean imbalance is E * max(f). Notice that f is hard:
# its counts must not carry a gradient. P still must.
example_scores = torch.tensor([[1., 0.], [1., 0.]] + [[.4, .6]] * 4)
example_ids = torch.tensor([[0], [0], [1], [1], [1], [1]])
print('hard requests:', torch.bincount(example_ids.reshape(-1), minlength=2).tolist())
# Before coding, predict whether this scalar is above or below 1.

# %% [answer]
def route_balance(score_batches: list[torch.Tensor],
                  route_batches: list[torch.Tensor],
                  mode: str = 'global') -> tuple[torch.Tensor, float]:
    """Return a differentiable balance scalar and hard max/mean imbalance."""
    pass

# %%
# Baseline: one batch in lists of length one. scores[T,E] are finite,
# nonnegative probabilities whose rows sum to one. ids[T,K] are selected
# expert IDs, distinct within each token. Return loss E*dot(f,P) and
# imbalance E*max(f). Use floating scores in the loss computation.
check('0098_load_balance_eval.py', stage='baseline')

# %%
# Follow-up: reproduce the six-token trap above. Requested counts (2,4)
# give f=(1/3,2/3); P=(3/5,2/5). The scalar is 14/15 but imbalance is
# 4/3. A balanced route pattern with the same P gives scalar 1.
# Explain why a scalar below 1 is not proof of balanced actual routes.
check('0098_load_balance_eval.py', stage='followup')

# %%
# Production: support a list of microbatches and mode:
#   global: concatenate conceptually; one hard frequency and one P.
#   local: unweighted mean of nonempty microbatch losses; imbalance is
#          the unweighted mean of their individual hard imbalances.
#   prefix: at batch j use the hard frequency of batches 1..j with
#           batch j's P; weight each loss by T_j/sum(T). Its imbalance
#           is the final combined hard imbalance.
# Exact global and prefix are generally different objectives. Empty
# batches are ignored in local/prefix averages; if every batch is empty,
# return a differentiable zero scalar and 0.0 imbalance. Require E>=2,
# 1<=K<=E, matching devices/dtypes, valid probabilities and IDs; raise
# ValueError otherwise. Can you find two microbatches whose combined
# routes are balanced even though each is maximally imbalanced?
check('0098_load_balance_eval.py', stage='production')
