"""Reference and staged checks for random access into run-length data."""

# %%
from hypothesis import given, settings, strategies as st

# Drill 1: The ends print `[1 3 6]`; `index < end` decodes
# `['B', 'A', 'A', 'E', 'E', 'E']` but `index <= end` gives
# `['B', 'B', 'A', 'A', 'E', 'E']`. A run's end is exclusive: run B covers only
# index 0, and index 1 is the first index of the next run. With `<=` every
# boundary index is claimed by the run before it.

# Drill 2: `side="right"` prints `[0 1 1 2 2 2]`, the correct runs, while
# `side="left"` prints `[0 0 1 1 2 2]`. The right side counts the ends that are
# less than or equal to the index, which is exactly the number of runs already
# finished; the left side treats an index equal to an end as still inside the
# earlier run, the same off-by-one as `<=`.

# Drill 3: It prints `True True True`. `bool` is a subclass of `int`, so
# `isinstance(True, int)` holds and `True > 0`; the only way to reject it is an
# explicit `isinstance(count, bool)` check before the integer check.

# %%
def rle_index(runs: list[tuple[str, int]], index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise IndexError(index)
    end = 0
    answer = None
    for run in runs:
        if not isinstance(run, tuple) or len(run) != 2:
            raise ValueError("each run must be a (character, count) tuple")
        char, count = run
        if (not isinstance(char, str) or len(char) != 1 or isinstance(count, bool)
                or not isinstance(count, int) or count <= 0):
            raise ValueError("runs need single characters and positive integer counts")
        end += count
        if answer is None and index < end:
            answer = char
    if answer is None:
        raise IndexError(index)
    return answer


# %% [verifiers]
def verify_baseline_interior_and_boundary(rle_index):
    runs = [("B", 1), ("A", 2), ("E", 3)]
    assert [rle_index(runs, i) for i in range(6)] == list("BAAEEE")


def verify_baseline_outside_range(rle_index):
    for runs, index in (([], 0), ([("x", 2)], 2), ([("x", 2)], -1)):
        try:
            rle_index(runs, index)
        except IndexError:
            pass
        else:
            raise AssertionError("out-of-range index accepted")


def verify_followup_huge_counts_without_expansion(rle_index):
    runs = [("a", 10**15), ("b", 2), ("c", 10**15)]
    assert rle_index(runs, 10**15 - 1) == "a"
    assert rle_index(runs, 10**15) == "b"
    assert rle_index(runs, 10**15 + 2) == "c"


def verify_followup_malformed_runs(rle_index):
    bad = ([('a', 0)], [('a', -1)], [('ab', 2)], [('a', True)], [('a', 1), ('b', 0)], [('a',)])
    for runs in bad:
        try:
            rle_index(runs, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed runs accepted: {runs}")


def verify_production_no_full_decode_or_mutation(rle_index):
    runs = [(str(i % 10), 10**12) for i in range(100)]
    original = runs.copy()
    assert rle_index(runs, 97 * 10**12 + 123) == "7"
    assert runs == original
    try:
        rle_index(runs, 100 * 10**12)
    except IndexError:
        pass
    else:
        raise AssertionError("exclusive total length accepted")


def verify_production_generated_against_small_decode(rle_index):
    @settings(max_examples=50, deadline=None, derandomize=True)
    @given(st.lists(st.tuples(st.sampled_from(list("abc")), st.integers(1, 8)), max_size=12))
    def property_check(runs):
        decoded = "".join(char * count for char, count in runs)
        for index, char in enumerate(decoded):
            assert rle_index(runs, index) == char
    property_check()


# %%
from torch_judge import evaluate
if not evaluate("0067_rle_index_eval.py", rle_index):
    raise SystemExit(1)
