"""Reference and verifiers for grouping coordinate rows."""

# %%
import torch
from hypothesis import given, settings, strategies as st

# Drill 1: It prints `[[0, 2], [0, 5], [1, 0]]`. Rows sort by column 0, then
# column 1 breaks ties, like dictionary order.

# Drill 2: It prints `[2, 1, 2, 0] [1, 1, 2]`. Input rows 0 and 2 are (1,0), the
# third distinct row, so both map to 2; the other two map to 1 and 0. Counts
# follow the sorted unique order.

# Drill 3: It prints `True`. `inverse` is the index back into `unique`, so
# gathering `unique[inverse]` rebuilds the input. That identity is a free oracle.

# %%
def group_rows(keys: torch.Tensor):
    if not isinstance(keys, torch.Tensor) or keys.dtype != torch.int64 or keys.ndim != 2 or keys.shape[1] < 1:
        raise ValueError("keys must be int64 [K,D] with D >= 1")
    if keys.shape[0] == 0:
        empty = keys.new_zeros(0)
        return keys.clone(), empty, empty.clone()
    return torch.unique(keys, dim=0, sorted=True, return_inverse=True, return_counts=True)


# %% [verifiers]
def verify_baseline_hand_groups(group_rows):
    unique, inverse, counts = group_rows(torch.tensor([[1, 0], [0, 5], [1, 0], [0, 2]]))
    assert unique.tolist() == [[0, 2], [0, 5], [1, 0]]
    assert inverse.tolist() == [2, 1, 2, 0] and counts.tolist() == [1, 1, 2]


def verify_followup_negatives_and_empty(group_rows):
    unique, inverse, counts = group_rows(torch.tensor([[0, -1], [-2, 3], [0, -1], [0, -5]]))
    assert unique.tolist() == [[-2, 3], [0, -5], [0, -1]] and inverse.tolist() == [2, 0, 2, 1] and counts.tolist() == [1, 1, 2]
    unique, inverse, counts = group_rows(torch.zeros(0, 3, dtype=torch.int64))
    assert unique.shape == (0, 3) and inverse.shape == (0,) and counts.shape == (0,)
    assert inverse.dtype == counts.dtype == torch.int64


def verify_followup_permutation_and_rebuild(group_rows):
    @settings(max_examples=25, deadline=None, derandomize=True)
    @given(st.lists(st.tuples(st.integers(-2, 2), st.integers(-2, 2), st.integers(-2, 2)), min_size=1, max_size=12),
           st.randoms(use_true_random=False))
    def trial(rows, rng):
        keys = torch.tensor(rows)
        unique, inverse, counts = group_rows(keys)
        assert torch.equal(unique[inverse], keys) and int(counts.sum()) == len(rows)
        order = list(range(len(rows))); rng.shuffle(order); perm = torch.tensor(order)
        unique2, inverse2, counts2 = group_rows(keys[perm])
        assert torch.equal(unique2, unique) and torch.equal(counts2, counts)
        assert torch.equal(inverse2, inverse[perm])
    trial()


def verify_production_no_packed_key_and_validation(group_rows):
    big = 2 ** 62
    keys = torch.tensor([[big, 1], [big, 2], [big, 1], [-big, 2]])
    unique, inverse, counts = group_rows(keys)
    assert unique.tolist() == [[-big, 2], [big, 1], [big, 2]] and counts.tolist() == [1, 2, 1]
    assert torch.equal(unique[inverse], keys)
    near = torch.tensor([[2 ** 62 + 1], [2 ** 62], [2 ** 62 + 1]])
    assert group_rows(near)[2].tolist() == [1, 2]
    for bad in (torch.zeros(3), torch.zeros(2, 3), torch.zeros(2, 2, 2, dtype=torch.int64),
                torch.zeros(2, 0, dtype=torch.int64), torch.zeros(2, 2, dtype=torch.int32)):
        try:
            group_rows(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid keys accepted')


def verify_production_no_mutation(group_rows):
    keys = torch.tensor([[1, 0], [0, 5], [1, 0]]); before = keys.clone()
    group_rows(keys)
    assert torch.equal(keys, before)

# %%
from torch_judge import evaluate
if not evaluate("0118_group_rows_eval.py", group_rows):
    raise SystemExit(1)
