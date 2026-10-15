"""Reference and staged checks for the length of half-open interval union."""

# %%
import numpy as np
from hypothesis import given, settings, strategies as st

# Drill 1: Sorted by start, the starts print `[0. 2. 5.]`; sorted by end they
# print `[2. 5. 0.]`, so $[0, 10)$ comes last and the sweep would close
# $[2, 3)$ and $[5, 9)$ before seeing the interval that covers them both. Only
# sorting by start guarantees each new interval either extends the current run
# or starts after it.

# Drill 2: It prints `2.0 1.5`. Half-open intervals that touch, $[0, 1)$ and
# $[1, 2)$, share no point but leave no gap, so they merge into one run of
# length 2; a real gap from 1 to 1.5 closes the first run. That is why the
# sweep merges when `lo > end` is false, including `lo == end`.

# Drill 3: `end = hi` gives 3.0 but the union really reaches 10.0: a nested
# interval ends before the current run, so overwriting `end` would shrink the
# run and lose 7 units of coverage. Taking the max keeps the furthest end seen.

# %%
def interval_union(intervals: np.ndarray) -> float:
    intervals = np.asarray(intervals)
    if (intervals.ndim != 2 or intervals.shape[1] != 2
            or intervals.dtype.kind not in 'iuf' or not np.isfinite(intervals).all()
            or np.any(intervals[:, 1] < intervals[:, 0])):
        raise ValueError("need finite numeric [N,2] intervals with end>=start")
    if len(intervals) == 0:
        return 0.0
    ordered = intervals[np.argsort(intervals[:, 0], kind="stable")]
    start, end = ordered[0]
    total = 0.0
    for lo, hi in ordered[1:]:
        if lo > end:
            total += end - start
            start, end = lo, hi
        else:
            end = max(end, hi)
    return float(total + end - start)


# %% [verifiers]
def verify_baseline_overlap_and_touching(interval_union):
    x = np.array([[0., 2.], [1., 3.], [3., 5.]])
    assert interval_union(x) == 5.0
    assert interval_union(np.array([[0., 1.], [1., 2.]])) == 2.0


def verify_baseline_empty_and_nested(interval_union):
    assert interval_union(np.empty((0, 2))) == 0.0
    assert interval_union(np.array([[2., 3.], [0., 5.], [1., 4.]])) == 5.0


def verify_followup_duplicate_zero_and_negative_coordinates(interval_union):
    x = np.array([[1., 3.], [1., 3.], [5., 5.], [-3., -1.]])
    assert interval_union(x) == 4.0
    assert interval_union(np.array([[2., 2.], [2., 2.]])) == 0.0


def verify_followup_reject_invalid_intervals(interval_union):
    for x in (np.array([[2., 1.]]), np.array([[0., np.inf]]),
              np.array([[np.nan, 1.]]), np.array([1., 2.]),
              np.array([[0., 1., 2.]])):
        try:
            interval_union(x)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid intervals accepted")


def verify_production_order_invariance_and_immutability(interval_union):
    x = np.array([[9., 10.], [-2., 1.], [0., 8.], [1., 1.]])
    before = x.copy()
    assert interval_union(x) == interval_union(x[::-1]) == 11.0
    assert np.array_equal(x, before)


def verify_production_generated_integer_coverage(interval_union):
    @settings(max_examples=60, deadline=None, derandomize=True)
    @given(st.lists(st.tuples(st.integers(-5, 5), st.integers(-5, 5)), max_size=12))
    def property_check(pairs):
        intervals = [(min(a, b), max(a, b)) for a, b in pairs]
        cells = {x for lo, hi in intervals for x in range(lo, hi)}
        array = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
        assert interval_union(array) == len(cells)
    property_check()


# %%
from torch_judge import evaluate
if not evaluate("0068_interval_union_eval.py", interval_union):
    raise SystemExit(1)
