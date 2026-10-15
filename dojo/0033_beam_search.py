# %%
# 0033 · Beam search decoding
# A single editable function; run each check after revising the same answer.

# %%
import torch

# %%
# Baseline: log_prob_fn receives a 1D long tensor of the current sequence
# and returns log probabilities [V]. The start token counts toward max_len.
# Return the highest-scoring list[int]; beam_width >= 1 and max_len >= 1.
# Predict the width-1 result when each next token has one clear winner.
print("greedy path score:", -0.1 - 0.2)

# %% [answer]
def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    pass

# %%
from torch_judge import check
check("0033_beam_search_eval.py", stage="baseline")

# %%
# Follow-up: completed EOS beams stay available but are never expanded.
# Continue while any live beam might win; do not stop merely because the
# current best ends with EOS. Rank by total log probability, then prefer
# lexicographically smaller token sequences on exact ties. What happens to
# a completed path when every other beam adds a negative log probability?
check("0033_beam_search_eval.py", stage="followup")

# %%
# Production: reject malformed dimensions, NaN/+inf, invalid token IDs, and
# invalid limits with ValueError. -inf means impossible; never select it.
# Support V < beam_width and max_len=1. Stop without another model call
# once all retained beams are complete. O(max_len * beam_width * V) model
# score inspection is acceptable; do not build an exponential tree.
check("0033_beam_search_eval.py", stage="production")
