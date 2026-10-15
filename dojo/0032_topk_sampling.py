# %%
# 0032 · Top-k / top-p sampling
# Revise one function through three stages. All filters act on a fresh copy.

# %%
import torch

# %%
# Baseline: finite logits [V], V >= 1. temperature > 0. With top_k=0 keep
# all tokens; otherwise keep exactly min(top_k,V), resolving equal logits by
# smaller vocabulary index. Return a Python int. Invalid parameters raise
# ValueError. Predict: with logits [0, 4, 4], top_k=1 must return index 1.
probe = torch.tensor([0., 4., 4.])
print("largest value:", probe.max().item(), "tie indices:", [1, 2])

# %% [answer]
def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    pass

# %%
from torch_judge import check
check("0032_topk_sampling_eval.py", stage="baseline")

# %%
# Follow-up: 0 < top_p <= 1. Sort survivors by descending scaled logit,
# breaking ties by token index. Retain the shortest prefix whose cumulative
# probability reaches top_p; always keep at least one. Compute top-p AFTER
# top-k and renormalize after both filters. What survives for [3,2,1] at
# top_p=.8? Compare the cumulative mass before sampling.
check("0032_topk_sampling_eval.py", stage="followup")

# %%
# Production: reject NaN/+inf/all -inf and malformed shapes; allow -inf to
# mask individual tokens. Stable softmax handles logits near 1e4. Do not
# mutate the caller's tensor; seeded torch RNG must replay the same draws.
# Sorting O(V log V) time and O(V) extra storage is acceptable.
check("0032_topk_sampling_eval.py", stage="production")
