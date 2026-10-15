# %%
# 0120 · The backward of a gather
# Goal: write a row gather as a custom autograd Function whose backward is the
# collision-safe scatter-add from 0114.
# Why gather and scatter swap roles under backward (0101, 0107, 0108). Authored.

# %%
import torch
from torch_judge import check

# %%
# Baseline — table [3,2]; ids [2, 0, 2] read row 2 twice.
table = torch.arange(6.).reshape(3, 2).requires_grad_()
ids = torch.tensor([2, 0, 2])

# %% [drill]
# Forward
# Predict `table[ids]`.
print(table[ids].tolist())

# %% [drill]
# Gradient of a gather
# Predict `table.grad` after `table[ids].sum().backward()`.
table[ids].sum().backward()
print(table.grad.tolist())

# %% [drill]
# Who accumulates?
# Which rows of `table` got a nonzero gradient, and why is row 2's value not 1?

# %%
# Contract: class GatherRows(torch.autograd.Function) with
# forward(ctx, table, ids) -> table[ids] and a matching backward. table float
# [R,C], ids int64 [K] in [0, R). Backward returns (grad_table [R,C], None):
# duplicates accumulate, unread rows get 0. Out-of-range, negative or non-int64
# ids raise ValueError in forward. Empty ids give zero gradients. Do not
# mutate the inputs, and do not call table[ids] inside backward.

# %% [answer]
class GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, ids):
        pass

    @staticmethod
    def backward(ctx, grad_out):
        pass

# %%
check('0120_gather_rows_eval.py', stage='baseline')

# %%
# Follow-up — Compare with autograd on `table[ids]`, including duplicates.
check('0120_gather_rows_eval.py', stage='followup')

# %%
# Production — gradcheck in float64, non-contiguous upstream gradients,
# validation, empty ids, and a table used twice in one graph.
check('0120_gather_rows_eval.py', stage='production')
