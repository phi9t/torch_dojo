# %%
# 0097 · Where did the routes go?
# An MoE router requests work; capacity rules may admit only some of it.
# Keep both ledgers or a hot expert can look healthy after drops.
# Inspired by the route accounting in the Olmo-core 3 report:
# https://allenai.org/papers/olmocore3
# This is an original, local PyTorch exercise, not its distributed code.

# %%
import torch
from torch_judge import check

# %%
# Each entry in route_ids[T,K] is one (token, expert) request. Experts
# have owners expert_to_rank[E], indexed in [0,R). A keep_mask[T,K]
# says which requests actually ran. Try predicting both rank histograms:
route_ids = torch.tensor([[0, 2], [2, 1], [0, 2]])
expert_to_rank = torch.tensor([0, 1, 0])
keep_mask = torch.tensor([[True, False], [False, True], [True, True]])
print('requests:', route_ids.tolist(), 'kept:', keep_mask.tolist())
# Experts 0 and 2 share rank 0. Does dropping two routes to expert 2
# make the original pressure on rank 0 disappear?

# %% [answer]
def route_ledger(route_ids: torch.Tensor, expert_to_rank: torch.Tensor,
                 num_ranks: int, keep_mask: torch.Tensor | None = None) -> dict:
    """Count requested, kept, dropped routes and max/mean load imbalance."""
    pass

# %%
# Baseline: return long tensors requested_expert[E] and requested_rank[R]
# in a dict. With no keep_mask, all requests are kept. Rank IDs may have
# gaps: still return all R bins. Avoid Python loops over individual routes.
check('0097_route_ledger_eval.py', stage='baseline')

# %%
# Follow-up: include kept_expert, kept_rank, dropped_expert, dropped_rank.
# The kept and dropped counts must add to requested counts in both views.
# Count a routed token twice when it legitimately chose two experts.
check('0097_route_ledger_eval.py', stage='followup')

# %%
# Production: add imbalance, a dict with keys requested_expert,
# kept_expert, requested_rank, kept_rank. For each nonempty histogram,
# imbalance = max(count) / mean(count), including zero-load experts/ranks.
# Define every zero-total imbalance as 0.0. Reject invalid IDs, owners,
# masks, and duplicate expert choices within a token with ValueError.
# route_ids must be long [T,K], owners long [E] with E>=1, R>=1;
# keep_mask must be bool [T,K] on the same device. T or K may be zero.
# Are the requested and kept imbalance values always ordered? Find a
# counterexample before trusting a single scalar dashboard.
check('0097_route_ledger_eval.py', stage='production')
