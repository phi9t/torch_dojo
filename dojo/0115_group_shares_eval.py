"""Reference and verifiers for per-group shares."""

# %%
import torch
from hypothesis import given, settings, strategies as st

# Drill 1: It prints `tensor(5.) tensor(3.5000)`. Each voxel mean stands for a
# different number of points, so the plain mean 5 is wrong; weighting by count
# gives (2*3 + 8*1) / 4 = 3.5.

# Drill 2: It prints `tensor([0.7500, 0.2500]) tensor(3.5000)`. Shares are
# count/total and sum to 1, so a share-weighted sum IS the weighted mean, with no
# final divide.

# Drill 3: It prints `tensor([4., 10., 4.])`. Indexing a per-group vector with
# the row's group id gathers each row's own total. A gather reads, so repeated
# ids are harmless; the scatter that built `totals` is the step that needs care.

# %%
def group_shares(group_ids: torch.Tensor, values: torch.Tensor, num_groups: int) -> torch.Tensor:
    if not isinstance(group_ids, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise ValueError("tensors required")
    if isinstance(num_groups, bool) or not isinstance(num_groups, int) or num_groups < 0:
        raise ValueError("num_groups must be a nonnegative int")
    if group_ids.dtype != torch.int64 or group_ids.ndim != 1 or values.shape != group_ids.shape:
        raise ValueError("group_ids must be int64 [K] and values [K]")
    if values.dtype not in (torch.float32, torch.float64, torch.int64) or values.device != group_ids.device:
        raise ValueError("values must be float32/float64/int64 on the same device")
    if group_ids.numel() and (int(group_ids.min()) < 0 or int(group_ids.max()) >= num_groups):
        raise ValueError("group id outside [0, num_groups)")
    work = values.double() if values.dtype != torch.float64 else values
    if not torch.isfinite(work).all() or (work < 0).any():
        raise ValueError("values must be finite and nonnegative")
    if values.dtype == torch.int64:
        exact = work.new_zeros(num_groups).index_add(0, group_ids, work)
        if (exact >= 2.0 ** 63).any():
            raise ValueError("group total overflows int64")
    # Scale each group by its largest value so every term is <= 1 before summing.
    biggest = work.new_zeros(num_groups).scatter_reduce(0, group_ids, work, reduce='amax', include_self=True)
    unit = torch.where(biggest[group_ids] > 0, work / biggest[group_ids].clamp_min(torch.finfo(work.dtype).tiny), work)
    totals = work.new_zeros(num_groups).index_add(0, group_ids, unit)
    if group_ids.numel() and (totals[group_ids] <= 0).any():
        raise ValueError("a group has zero total")
    return (unit / totals[group_ids]).to(torch.float64 if values.dtype == torch.int64 else values.dtype)


# %% [verifiers]
def verify_baseline_count_weighted_mean(group_shares):
    shares = group_shares(torch.tensor([0, 0]), torch.tensor([3., 1.]), 1)
    assert torch.allclose(shares, torch.tensor([.75, .25])), shares
    assert abs(float((shares * torch.tensor([2., 8.])).sum()) - 3.5) < 1e-6


def verify_followup_groups_sum_to_one_in_any_order(group_shares):
    @settings(max_examples=25, deadline=None, derandomize=True)
    @given(st.lists(st.tuples(st.integers(0, 3), st.integers(1, 9)), min_size=1, max_size=12))
    def trial(rows):
        ids = torch.tensor([g for g, _ in rows]); vals = torch.tensor([float(v) for _, v in rows], dtype=torch.float64)
        shares = group_shares(ids, vals, 5)
        sums = torch.zeros(5, dtype=torch.float64).index_add(0, ids, shares)
        used = torch.zeros(5, dtype=torch.bool); used[ids] = True
        assert torch.allclose(sums[used], torch.ones(int(used.sum()), dtype=torch.float64))
        assert not sums[~used].any()
        flipped = group_shares(ids.flip(0), vals.flip(0), 5)
        assert torch.allclose(flipped.flip(0), shares)
    trial()


def verify_followup_integer_counts_and_empty_input(group_shares):
    shares = group_shares(torch.tensor([1, 1, 0]), torch.tensor([1, 3, 5]), 3)
    assert shares.dtype == torch.float64 and torch.allclose(shares, torch.tensor([.25, .75, 1.], dtype=torch.float64))
    assert group_shares(torch.zeros(0, dtype=torch.int64), torch.zeros(0), 3).shape == (0,)


def verify_production_validation(group_shares):
    ids = torch.tensor([0, 0])
    bad = [(ids, torch.tensor([-1., 2.]), 1), (ids, torch.tensor([float('nan'), 1.]), 1),
           (ids, torch.tensor([0., 0.]), 1), (torch.tensor([0, 1]), torch.ones(2), 1),
           (ids.float(), torch.ones(2), 1), (ids, torch.ones(3), 1), (ids, torch.ones(2), -1),
           (ids, torch.tensor([2 ** 62, 2 ** 62], dtype=torch.int64), 1)]
    for group_ids, values, n in bad:
        try:
            group_shares(group_ids, values, n)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input accepted')


def verify_production_extreme_values_stay_finite(group_shares):
    huge = torch.full((3,), 1e308, dtype=torch.float64)
    assert torch.allclose(group_shares(torch.zeros(3, dtype=torch.int64), huge, 1), torch.full((3,), 1 / 3, dtype=torch.float64))
    wide = torch.full((3,), 3e38, dtype=torch.float32)
    out = group_shares(torch.zeros(3, dtype=torch.int64), wide, 1)
    assert out.dtype == torch.float32 and torch.isfinite(out).all() and torch.allclose(out, torch.full((3,), 1 / 3))


def verify_production_no_mutation(group_shares):
    ids, vals = torch.tensor([0, 0, 1]), torch.tensor([1., 3., 2.])
    ids0, vals0 = ids.clone(), vals.clone()
    group_shares(ids, vals, 2)
    assert torch.equal(ids, ids0) and torch.equal(vals, vals0)

# %%
from torch_judge import evaluate
if not evaluate("0115_group_shares_eval.py", group_shares):
    raise SystemExit(1)
