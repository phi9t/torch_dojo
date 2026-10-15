# %%
# 0125 · Masked multi-head self-attention in NumPy (practice set 1/7, 25 min)
# Goal: implement `self_attention` with float64 NumPy. No attention helper, no
# loops over batches, heads or token pairs. Say the time and the peak
# intermediate-memory complexity out loud when you finish.

# %%
import numpy as np

# %%
# Baseline — Predict the shapes before running anything.
# X[B,T,D] @ Wq[D,D] -> [B,T,D] -> split heads -> [B,H,T,D/H].
# Scores are [B,H,T,T], scaled by sqrt(D/H). Returns (out[B,T,D], weights[B,H,T,T]).
# Projection convention: Q = X @ Wq (likewise K, V); no biases.
# Contract: valid_mask is bool [B,T]. Invalid tokens are never attended to.
# Invalid QUERY rows return zero output and zero weights. If causal, query i
# cannot attend to j > i. Any entirely masked row returns zeros, never NaN.
# Hand-checkable: Wq=Wk=0, Wv=Wo=I, one head, X=[[1,0],[0,2]] -> both outputs
# [0.5, 1.0] unmasked; with causal=True -> [[1,0],[0.5,1]].

# %% [drill]
# Which rows are empty?
# Predict `allowed` for valid=[T,F,T], causal=True, then which query rows have no key.
valid = np.array([True, False, True])
allowed = valid[:, None] & valid[None, :] & np.tril(np.ones((3, 3), bool))
print(allowed.astype(int), allowed.any(-1))

# %% [drill]
# Why a plain softmax breaks
# Predict the result: softmax over a row of all -inf.
row = np.full(3, -np.inf)
with np.errstate(invalid="ignore"):
    print(np.exp(row - row.max()) / np.exp(row - row.max()).sum())

# %%
# Inputs conforming to the answer's signature (evaluate these, then step into the body).
rng = np.random.default_rng(0)
B, T, D, H = 2, 4, 6, 2
X: np.ndarray = rng.normal(size=(B, T, D))          # [B,T,D] float64
Wq: np.ndarray = rng.normal(size=(D, D))            # [D,D]
Wk: np.ndarray = rng.normal(size=(D, D))            # [D,D]
Wv: np.ndarray = rng.normal(size=(D, D))            # [D,D]
Wo: np.ndarray = rng.normal(size=(D, D))            # [D,D]
num_heads: int = H                                  # divides D
valid_mask: np.ndarray = np.array([[True, True, False, True],
                                   [False, False, False, False]])  # bool [B,T]
causal: bool = True

# %% [answer]
def self_attention(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                   Wo: np.ndarray, num_heads: int, valid_mask: np.ndarray,
                   causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    # X [B,T,D]; W* [D,D]; valid_mask bool [B,T]
    # returns (out [B,T,D], weights [B,H,T,T]), both float64
    if X.ndim != 3:
        raise ValueError("X must be [B,T,D]")
    batch_size, seq_len = X.shape[0], X.shape[1] # B, T
    dim_model = X.shape[-1] # D
    if num_heads < 1 or dim_model % num_heads:
        raise ValueError("num_heads must be a positive divisor of D")
    if any(w.shape != (dim_model, dim_model) for w in (Wq, Wk, Wv, Wo)):
        raise ValueError("weights must be [D,D]")
    if valid_mask.dtype != np.bool_ or valid_mask.shape != (batch_size, seq_len):
        raise ValueError("valid_mask must be bool [B,T]")
    dim_query = dim_model // num_heads  # Q

    query_bhtq = (X @ Wq).reshape(batch_size, seq_len, num_heads, dim_query).transpose(0, 2, 1, 3)  # [B,H,T,Q]
    key_bhtk = (X @ Wk).reshape(batch_size, seq_len, num_heads, dim_query).transpose(0, 2, 1, 3)  # [B,H,T,K]
    value_bhtv = (X @ Wv).reshape(batch_size, seq_len, num_heads, dim_query).transpose(0, 2, 1, 3)  # [B,H,T,V]
    full_dense_score_bhtt = query_bhtq @ key_bhtk.transpose(0, 1, 3, 2) / np.sqrt(dim_query)  # [B,H,T,T]
    # --- Masks. Convention: True = allowed. Score entry [b,h,i,j] is "query i looks at key j".
    # valid_mask[:, None, :, None]  -> [B,1,T,1] marks valid QUERIES (rows i); an invalid query attends to nothing.
    # valid_mask[:, None, None, :]  -> [B,1,1,T] marks valid KEYS (columns j); nobody attends to an invalid key.
    # Their AND is [B,1,T,T]; the size-1 head axis broadcasts, since every head shares one padding mask.
    allowed_b1tt = valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
    if causal:
        # tril keeps j <= i (the diagonal stays: a token sees itself); [None, None] broadcasts over B and H.
        causal_mask_tt = np.tril(np.ones((seq_len, seq_len), bool))
        allowed_b1tt = allowed_b1tt & causal_mask_tt[None, None, :, :]
    # Disallowed positions get -inf, which softmax maps to exactly 0 weight.
    scores_bhtt = np.where(allowed_b1tt, full_dense_score_bhtt, -np.inf)
    # A row can be EMPTY (no allowed key): an invalid query, or under causal a row whose visible prefix is all invalid.
    # Its max is -inf, and -inf - (-inf) = NaN, so such rows get scores 0 BEFORE the max/exp. This is only a
    # placeholder: the weights below are re-zeroed by `allowed_bhtt`, so the fake scores never reach the output.
    has_key_b1t1 = allowed_b1tt.any(-1, keepdims=True)  # [B,1,T,1]: does this query have any key?
    safe_scores_bhtt = np.where(has_key_b1t1, scores_bhtt, 0)
    # Select with np.where (not * mask), and subtract the row max so exp cannot overflow on large scores.
    weights_bhtt = np.where(allowed_b1tt, np.exp(safe_scores_bhtt - safe_scores_bhtt.max(-1, keepdims=True)), 0)
    # Normalize; an empty row sums to 0, so divide by 1 there instead of 0 (0/0 would be NaN): weights stay all zero.
    weights_bhtt = weights_bhtt / np.where(has_key_b1t1, weights_bhtt.sum(-1, keepdims=True), 1.0)  # empty row: 0/1 = 0
    attended_bhtd = weights_bhtt @ value_bhtv  # [B,H,T,Q]
    out_btd = attended_bhtd.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, dim_model) @ Wo  # [B,T,D]
    return out_btd, weights_bhtt

# %%
from torch_judge import check
check("0125_masked_self_attention_eval.py", stage="baseline")

# %%
# Follow-up — All-invalid batch elements give zeros (and no NaN). Changing an
# invalid token, or a FUTURE token under causal=True, must not move the outputs
# it is not allowed to see. Add a loop-based reference of your own to compare.
check("0125_masked_self_attention_eval.py", stage="followup")

# %%
# Production — Large activations must not overflow the softmax. Do not mutate
# inputs. Reject num_heads that does not divide D, wrong weight shapes and a
# non-bool or wrongly shaped valid_mask with ValueError.
# Extension (not graded): accept separate query and key/value sequences of
# different lengths. Which of your reshapes hard-code Tq == Tk?
check("0125_masked_self_attention_eval.py", stage="production")
