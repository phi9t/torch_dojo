"""Reference and verifiers for sparse products in both directions."""

# %%
from hypothesis import given, settings, strategies as st
import numpy as np

# Drill 1: It prints `out[row] += v: [3. 0.]  np.add.at: [5. 0.]`. Fancy-indexed
# `+=` reads `out[row]`, adds, then writes back element by element, so the
# second write to index 0 overwrites the first and the 2 is lost. `np.add.at`
# is unbuffered and applies every addition, which is how duplicate COO
# coordinates must sum.

# Drill 2: It prints `A @ x: [50.  0.]  swapped indices: [0. 5.]`. In $Ax$ each
# entry reads `x` at its column and adds into its row; swapping the roles
# computes $A^\top x$ instead. For $y^\top A$ the swap is the right one: read
# `y` at the row and add into the column.

# Drill 3: It prints `f [0 1]`. Float indices cast to integers truncate
# silently, so 0.5 becomes row 0 and 1.9 row 1 with no error. Check the dtype
# kind is integer before casting, rather than trusting `astype`.

# %%
class SparseCOO:
    def __init__(self, shape: tuple[int, int], row, col, value):
        if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 0 for n in shape):
            raise ValueError("shape must contain two nonnegative dimensions")
        self.shape = tuple(shape)
        raw_row, raw_col = np.asarray(row), np.asarray(col)
        self.value = np.asarray(value)
        if raw_row.ndim != 1 or raw_col.ndim != 1 or self.value.ndim != 1 or not (len(raw_row) == len(raw_col) == len(self.value)):
            raise ValueError("COO arrays must be equally sized vectors")
        if (raw_row.size and raw_row.dtype.kind not in "iu") or (raw_col.size and raw_col.dtype.kind not in "iu") or self.value.dtype.kind not in "iuf" or not np.isfinite(self.value).all():
            raise ValueError("invalid COO indices or values")
        self.row = raw_row.astype(np.int64, copy=True)
        self.col = raw_col.astype(np.int64, copy=True)
        self.value = self.value.astype(float, copy=True)
        if np.any((self.row < 0) | (self.row >= shape[0])) or np.any((self.col < 0) | (self.col >= shape[1])):
            raise ValueError("COO coordinate outside matrix")

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.shape != (self.shape[1],) or not np.isfinite(x).all():
            raise ValueError("x must have one value per column")
        out = np.zeros(self.shape[0], dtype=float)
        np.add.at(out, self.row, self.value * x[self.col])
        return out

    def leftvec(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.shape != (self.shape[0],) or not np.isfinite(y).all():
            raise ValueError("y must have one value per row")
        out = np.zeros(self.shape[1], dtype=float)
        np.add.at(out, self.col, self.value * y[self.row])
        return out


# %% [verifiers]
def verify_followup_duplicate_entries_accumulate_in_both_directions(SparseCOO):
    A = SparseCOO((2, 3), [0, 0, 1], [1, 1, 2], [2., 3., 4.])
    assert np.array_equal(A.matvec(np.array([10., 20., 30.])), [100., 120.])
    assert np.array_equal(A.leftvec(np.array([2., 3.])), [0., 10., 12.])


def verify_baseline_both_products_against_dense_matrix(SparseCOO):
    row = np.array([2, 0, 1, 2])
    col = np.array([1, 3, 0, 0])
    val = np.array([3., 4., -2., 5.])
    dense = np.zeros((3, 4))
    np.add.at(dense, (row, col), val)
    A = SparseCOO((3, 4), row, col, val)
    x, y = np.arange(1., 5.), np.array([2., -1., 3.])
    assert np.allclose(A.matvec(x), dense @ x)
    assert np.allclose(A.leftvec(y), y @ dense)


def verify_followup_empty_rows_columns_and_no_entries(SparseCOO):
    A = SparseCOO((2, 3), [], [], [])
    assert np.array_equal(A.matvec(np.ones(3)), np.zeros(2))
    assert np.array_equal(A.leftvec(np.ones(2)), np.zeros(3))


def verify_production_reject_bad_coordinate_and_vector(SparseCOO):
    for row, col in [([2], [0]), ([0.5], [0]), ([0], [-1])]:
        try:
            SparseCOO((2, 2), row, col, [1.])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid COO coordinate accepted")
    A = SparseCOO((2, 2), [0], [1], [1.])
    try:
        A.matvec(np.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong vector length accepted")


def verify_production_generated_duplicates_against_dense(SparseCOO):
    entry = st.tuples(st.integers(0, 2), st.integers(0, 3), st.integers(-5, 5))

    @settings(max_examples=30, deadline=None, derandomize=True)
    @given(st.lists(entry, max_size=12))
    def property_check(entries):
        row = [r for r, _, _ in entries]
        col = [c for _, c, _ in entries]
        val = [v for _, _, v in entries]
        dense = np.zeros((3, 4))
        for r, c, v in entries:
            dense[r, c] += v
        A = SparseCOO((3, 4), row, col, val)
        x = np.array([1., -2., 3., 4.])
        y = np.array([2., 0., -1.])
        assert np.allclose(A.matvec(x), dense @ x)
        assert np.allclose(A.leftvec(y), y @ dense)

    property_check()



def verify_production_invalid_values_shape_and_no_input_mutation(SparseCOO):
    row, col, val = np.array([0, 1]), np.array([1, 0]), np.array([2., 3.])
    A = SparseCOO((2, 2), row, col, val)
    row[0], col[0], val[0] = 1, 0, 99.
    assert np.array_equal(A.matvec(np.ones(2)), [2., 3.])
    for shape, r, c, v in [((2,), [], [], []), ((-1, 2), [], [], []),
                           ((2, 2), [0], [0], [float('nan')]),
                           ((2, 2), [0], [], [1])]:
        try:
            SparseCOO(shape, r, c, v)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid COO data accepted')
    for vector in (np.array([float('inf'), 0.]), np.ones(3)):
        for product in (A.matvec, A.leftvec):
            try:
                product(vector)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid dense vector accepted')

# %%
from torch_judge import evaluate
if not evaluate("0077_sparse_coo_eval.py", SparseCOO):
    raise SystemExit(1)
