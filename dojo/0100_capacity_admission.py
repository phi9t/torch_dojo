# %%
# 0100 · Who gets the last expert slot?
# Goal: Return a boolean keep mask over routed (token, slot) requests that
# admits the highest scores first until each expert's, or each rank's, slots run out.
#
# A routed expert can be full even while its neighbor on the same rank is idle.
# This is a local admission policy exercise, not a distributed throughput model.
# Source: `https://allenai.org/papers/olmocore3`

# %%
import numpy as np
from torch_judge import check

# %%
# Three tokens each request one of two experts on rank zero. With one slot
# per expert, requests to the busy expert compete. Sketch the keep mask
# before running code.
# Predict: Which request would move in a shared rank pool?
example_ids = np.array([[0], [0], [1]])
example_scores = np.array([[.9], [.8], [.1]])
print("requests:", example_ids.ravel(), "scores:", example_scores.ravel())

# Try a lopsided batch next: three requests to expert 0, none to expert 1.
# Both experts belong to rank 0 and each owns one private slot. Its rank
# therefore has two pooled slots. Before coding, write requested, kept,
# and dropped counts for each expert and for rank 0 under both policies.
# Private: expert 0 can keep one; expert 1's slot sits idle. Pooled: rank 0
# can keep the two highest-scoring requests to expert 0. The gap is one
# dropped route in this example, not just a different choice of route.
hot_ids = np.array([[0], [0], [0]])
hot_scores = np.array([[.9], [.8], [.7]])
hot_owner = np.array([0, 0])
print("requested per expert:", np.bincount(hot_ids.ravel(), minlength=2))

# %% [drill]
# Who wins a tie?
# Routes 0 and 2 tie at 0.5. Predict the admission order from a stable sort of `-scores`, and from a stable ascending sort reversed.
import numpy as np
s = np.array([.5, .9, .5])
print(np.argsort(-s, kind="stable"), np.argsort(s, kind="stable")[::-1])

# %% [drill]
# How big is a rank's pool?
# Experts 0 and 1 belong to rank 7, expert 2 to rank 2, with `capacity=2`. Predict the pooled budgets, and the budgets if each rank got just `capacity`.
import numpy as np
owner = np.array([7, 7, 2])
unique, counts = np.unique(owner, return_counts=True)
print({int(r): int(c) * 2 for r, c in zip(unique, counts)}, {int(r): 2 for r in unique})

# %% [answer]
def capacity_admission(expert_ids: np.ndarray, scores: np.ndarray,
                       owner: np.ndarray, capacity: int,
                       policy: str = "expert") -> np.ndarray:
    """Boolean keep mask [T,K] for routes requested by expert_ids[T,K]."""
    pass

# %%
# Baseline — Each expert has `capacity` slots. Admit highest scores first.
# Watch out: resolve exact score ties by lower flattened route position
# (token-major, then slot). No rerouting: a rejected request remains rejected.
check("0100_capacity_admission_eval.py", stage="baseline")

# %%
# Follow-up — `policy='rank'` pools the slots of all experts owned by a
# rank. A rank with two experts has `2*capacity` slots; one hot expert may
# borrow the idle expert's slot. Count routes, not distinct tokens. For
# `hot_ids`, compare dropped-route totals: private 2, pooled 1. Report
# the per-expert and per-rank ledger before drawing a utilization claim.
check("0100_capacity_admission_eval.py", stage="followup")

# %%
# Production — $T$ may be zero; $K$ and $E$ are positive. Expert IDs are in
# $[0, E)$; `owner[E]` holds nonnegative integer rank IDs. Scores are finite.
# Capacity is a nonnegative integer; reject invalid inputs with `ValueError`.
# Do not mutate inputs. An admitted mask always conserves the rank/expert
# budget.
# Predict: Which policy can starve one expert while helping another?
check("0100_capacity_admission_eval.py", stage="production")
