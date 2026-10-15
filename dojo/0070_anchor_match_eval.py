"""Three-stage reference for deterministic anchor-to-GT matching."""

# %%
import numpy as np
from hypothesis import given, settings, strategies as st

# %%
# Drill 1: It prints `[1 0 0] [2 0]`. The IoU grid is $(A, G)$ with anchors on
# rows, so `argmax(axis=1)` reduces over GTs and returns one GT index per
# anchor, shape $(3,)$: the labelling step. `argmax(axis=0)` reduces over
# anchors and returns one anchor index per GT, shape $(2,)$: the forced-match
# step. Both are integer vectors, but they live in different index spaces, so
# swapping them silently labels anchors with anchor numbers.
#
# Drill 2: It prints `[-2  0 -1]`. Negative is strictly below 0.3, so 0.3
# itself is not negative and stays ignored ($-2$). Positive is 0.5 or more, so
# 0.5 is matched to GT 0. 0.29 is below 0.3 and becomes negative. Writing `<=`
# or `>` instead flips exactly the anchors that sit on a threshold.
#
# Drill 3: It prints `[0 0] [0 1] [0.9 0.2]`. Both GTs' best anchor is anchor
# 0, and an anchor holds one label, so GT 0 (the lower index) wins it. GT 1 does
# NOT fall back to anchor 1, even though anchor 1's own best GT is 1: forced
# matching only hands each GT its single best anchor. Anchor 1 is then
# labelled by thresholds alone, and its best IoU 0.2 is below 0.3, so the
# result is `[0, -1]`.

# %%
def match_anchors(iou: np.ndarray, negative_threshold: float = .3,
                  positive_threshold: float = .5) -> np.ndarray:
    iou = np.asarray(iou)
    if (iou.ndim != 2 or iou.dtype.kind not in 'f' or not np.isfinite(iou).all()
            or np.any((iou < 0) | (iou > 1))
            or not np.isscalar(negative_threshold) or not np.isscalar(positive_threshold)
            or not np.isfinite(negative_threshold) or not np.isfinite(positive_threshold)
            or not 0 <= negative_threshold <= positive_threshold <= 1):
        raise ValueError('invalid IoU or thresholds')
    anchors, ground_truths = iou.shape
    if ground_truths == 0:
        return np.full(anchors, -1, dtype=int)
    if anchors == 0:
        return np.empty(0, dtype=int)
    best_gt = iou.argmax(axis=1)
    best_value = iou[np.arange(anchors), best_gt]
    result = np.full(anchors, -2, dtype=int)
    result[best_value < negative_threshold] = -1
    result[best_value >= positive_threshold] = best_gt[best_value >= positive_threshold]
    claimed = set()
    for gt in range(ground_truths):
        anchor = int(iou[:, gt].argmax())
        if anchor not in claimed:
            result[anchor] = gt
            claimed.add(anchor)
    return result


# %% [verifiers]
def verify_baseline_thresholds(match_anchors):
    x = np.array([[.2, .1], [.4, .1], [.6, .2], [.1, .8]])
    assert np.array_equal(match_anchors(x), [-1, -2, 0, 1])


def verify_baseline_boundary(match_anchors):
    x = np.array([[.3], [.5], [.29]])
    assert np.array_equal(match_anchors(x), [-2, 0, -1])


def verify_followup_forced_low_iou_and_collision(match_anchors):
    assert np.array_equal(match_anchors(np.array([[.1, .02], [.03, .2]])), [0, 1])
    # Both GTs prefer anchor 0; lower GT wins. Losing GT cannot use anchor 1.
    x = np.array([[.9, .8], [.1, .2]])
    assert np.array_equal(match_anchors(x), [0, -1])


def verify_followup_ties_and_empty_dimensions(match_anchors):
    x = np.array([[.2, .1], [.2, .7], [.1, .7]])
    assert np.array_equal(match_anchors(x), [0, 1, 1])
    assert np.array_equal(match_anchors(np.zeros((2, 0))), [-1, -1])
    assert match_anchors(np.zeros((0, 2))).shape == (0,)


def verify_production_validation_and_no_mutation(match_anchors):
    x = np.array([[.4, .6]])
    original = x.copy()
    match_anchors(x)
    assert np.array_equal(x, original)
    invalid = [
        (np.array([.5]), .3, .5),
        (np.array([[np.nan]]), .3, .5),
        (np.array([[-.01]]), .3, .5),
        (np.array([[1.01]]), .3, .5),
        (x, .6, .5),
        (x, -.1, .5),
        (x, .3, 1.1),
    ]
    for matrix, low, high in invalid:
        try:
            match_anchors(matrix, low, high)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid IoU/threshold accepted')


def verify_production_generated_row_permutation_without_forcing(match_anchors):
    @settings(max_examples=20, deadline=None, derandomize=True)
    @given(st.lists(st.floats(.6, 1., allow_nan=False), min_size=2, max_size=12))
    def check(values):
        # Each row has a clear positive; every GT has an unambiguous best row.
        x = np.zeros((len(values), 1))
        x[:, 0] = values
        got = match_anchors(x)
        assert np.array_equal(got, np.zeros(len(values), dtype=int))
    check()


# %%
from torch_judge import evaluate
if not evaluate('0070_anchor_match_eval.py', match_anchors):
    raise SystemExit(1)
