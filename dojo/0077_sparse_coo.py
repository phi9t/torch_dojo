# %%
# 0077 · A matrix made of coordinates
# Goal: Store a sparse matrix as coordinate (COO) triples and compute both
# $Ax$ and $y^\top A$ from them.
#
# SE02. Both products come from the visible topic. COO and duplicate
# summation are authored Python practice rules.

# %%
import numpy as np
from torch_judge import check

# %%
# Predict: (0,1,2) and (0,1,3) represent A[0,1]=5.
# A has shape [2,3], plus (1,2,4). For x=[10,20,30], Ax=[100,120].
# For y=[2,3], y^T A=[0,10,12]. An ordinary indexed assignment loses 2.
print('Expected Ax:', [100., 120.], 'and y^T A:', [0., 10., 12.])

# %% [drill]
# Last write wins
# Two entries land on row 0 with values 2 and 3. Predict the output of a fancy-indexed `out[row] += v` and of `np.add.at`.
import numpy as np
row = np.array([0, 0]); vals = np.array([2., 3.])
scatter = np.zeros(2); scatter[row] += vals
accum = np.zeros(2); np.add.at(accum, row, vals)
print("out[row] += v:", scatter, " np.add.at:", accum)

# %% [drill]
# Rows or columns?
# $A$ is $2 \times 2$ with a single entry $A_{0,1} = 5$ and $x = [1, 10]$. Predict $Ax$, and what you get by scattering into `col` with `x[row]` instead.
import numpy as np
row, col, val = np.array([0]), np.array([1]), np.array([5.])
x = np.array([1., 10.])
right = np.zeros(2); np.add.at(right, row, val * x[col])
swapped = np.zeros(2); np.add.at(swapped, col, val * x[row])
print("A @ x:", right, " swapped indices:", swapped)

# %% [drill]
# Fractional indices
# Predict the dtype kind of `np.array([0.5, 1.9])` and what `astype(np.int64)` turns it into.
import numpy as np
row = np.array([0.5, 1.9])
print(row.dtype.kind, row.astype(np.int64))

# %%
# Contract: shape=(rows,cols), 1-D equally sized row/col/value arrays.
# Duplicate coordinates add, regardless of order. matvec takes [cols]
# and returns [rows]; leftvec takes [rows] and returns [cols].

# %% [answer]
class SparseCOO:
    def __init__(self, shape: tuple[int, int], row, col, value):
        pass

    def matvec(self, x: np.ndarray) -> np.ndarray:
        pass

    def leftvec(self, y: np.ndarray) -> np.ndarray:
        pass

# %%
# Baseline — Both products on a small matrix with unique coordinates.
check('0077_sparse_coo_eval.py', stage='baseline')

# %%
# Follow-up — Unsorted duplicates and empty rows/columns.
# Watch out: use accumulation, e.g. `np.add.at`, not a last-write-wins
# scatter. Check both products.
check('0077_sparse_coo_eval.py', stage='followup')

# %%
# Production — Reject negative/out-of-range/noninteger indices, malformed
# shapes, nonfinite values, and wrong vector lengths. Empty matrices work.
# COO takes O(nnz+rows+cols) storage; CSR helps repeated Ax products.
check('0077_sparse_coo_eval.py', stage='production')
