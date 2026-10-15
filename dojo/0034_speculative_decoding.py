# %%
# 0034 · Speculative acceptance and residual sampling
# This exercise handles one drafted block; no extra target-model token.

# %%
import torch

# %%
# Baseline: target_probs and draft_probs are [K,V] probability rows;
# draft_tokens is [K]. For each proposed token t, accept with probability
# min(1, p[t]/q[t]); stop at the first rejection. Return accepted tokens
# followed by ONE replacement from normalize(max(p-q,0)). Predict the
# acceptance probability when p[t]=.3, q[t]=.6.
print("acceptance chance:", .3 / .6)

# %% [answer]
def speculative_decode(target_probs, draft_probs, draft_tokens):
    pass

# %%
from torch_judge import check
check("0034_speculative_decoding_eval.py", stage="baseline")

# %%
# Follow-up: if p=q, every draft token is accepted. K=0 returns [].
# For a rejected token where p[t]=0 and q[t]=1, residual sampling must
# select only positive p-q entries. Why is sampling from p itself wrong?
check("0034_speculative_decoding_eval.py", stage="followup")

# %%
# Production: reject malformed shape, token IDs, negative/nonfinite values,
# non-normalized rows, or a draft token with q[t]=0. Do this before drawing
# randomness. Do not mutate inputs; seeded torch RNG replays outcomes.
# Iterating K rows is natural; use only O(V) working memory per rejection.
check("0034_speculative_decoding_eval.py", stage="production")
