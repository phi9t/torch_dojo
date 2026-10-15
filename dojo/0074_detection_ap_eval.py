"""Three-stage reference for one-class all-point interpolated detection AP."""

# %%
import numpy as np
from hypothesis import given, settings, strategies as st

# %%
# Drill 1: Keyed by `(image, gt)`, it prints `[1, 0, 1]`: the second detection
# re-claims object 0 in image 0 and is a false positive, but the third claims
# object 0 in image 1, a different object. Keyed by GT ID alone it prints
# `[1, 0, 0]`, wrongly turning a real hit into a false positive, because GT IDs
# are numbered within each image and repeat across images.
#
# Drill 2: Dividing by `total_gt` prints `[0.25 0.25 0.5 ]`: the two objects
# no detection found keep recall at one half. Dividing by the TPs found prints
# `[0.5 0.5 1. ]`, claiming every object was found, so AP would ignore every
# miss. Recall measures coverage of all GT, detected or not.
#
# Drill 3: Raw precision is `[0.    0.5   0.667]` and its right-to-left
# envelope is `[0.667 0.667 0.667]`. Recall rises by 0.5 at ranks 2 and 3, so
# AP integrates precision there: raw gives $0.5 \cdot 0.5 + 0.5 \cdot 0.667 =$
# `0.583`, the envelope gives `0.667`. The envelope reports the best precision
# reachable at that recall or beyond, so a dip that later recovers is not
# penalised; the leading false positive still costs AP, which would be 1 without it.

# %%
def detection_ap(detections: list[tuple[int, float, int]], total_gt: int) -> float:
    if isinstance(total_gt, (bool, np.bool_)) or not isinstance(total_gt, (int, np.integer)) or total_gt < 0:
        raise ValueError('total_gt must be a nonnegative integer')
    for detection in detections:
        if (not isinstance(detection, (tuple, list)) or len(detection) != 3
                or not isinstance(detection[0], (int, np.integer))
                or detection[0] < 0
                or not isinstance(detection[1], (int, float, np.integer, np.floating))
                or not np.isfinite(detection[1])
                or not isinstance(detection[2], (int, np.integer))
                or detection[2] < -1):
            raise ValueError('invalid detection tuple')
    if total_gt == 0:
        if any(gt >= 0 for _image, _score, gt in detections):
            raise ValueError('matched GT cannot exceed total_gt')
        return 0.0
    ordered = sorted(enumerate(detections), key=lambda pair: (-pair[1][1], pair[0]))
    claimed = set()
    tp = np.empty(len(ordered), dtype=np.int64)
    for j, (_, (image, _score, gt)) in enumerate(ordered):
        key = (image, gt)
        tp[j] = int(gt >= 0 and key not in claimed)
        if tp[j]:
            claimed.add(key)
    if len(claimed) > total_gt:
        raise ValueError('more distinct matched GT than total_gt')
    true_positives = np.cumsum(tp)
    precision = true_positives / np.arange(1, len(tp)+1)
    recall = true_positives / total_gt
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0., recall]) * envelope))


# %% [verifiers]
def verify_baseline_perfect_and_empty(detection_ap):
    assert detection_ap([(0, .9, 0), (0, .8, 1)], 2) == 1.
    assert detection_ap([], 2) == 0.
    assert detection_ap([(0, .5, -1)], 0) == 0.


def verify_baseline_missed_gt_denominator(detection_ap):
    assert detection_ap([(0, .7, 0)], 2) == .5


def verify_followup_duplicate_and_precision_envelope(detection_ap):
    example = [(0, .9, 0), (0, .8, 0), (0, .7, 1)]
    assert np.isclose(detection_ap(example, 2), 5/6)
    assert detection_ap([(0, .9, -1), (0, .8, 0)], 1) == .5


def verify_followup_ties_and_image_scoping(detection_ap):
    assert detection_ap([(0, .5, 0), (1, .5, 0)], 2) == 1.
    # Input order breaks ties, so a false positive before the TP costs AP.
    assert detection_ap([(0, .5, -1), (0, .5, 0)], 1) == .5
    assert detection_ap([(0, .5, 0), (0, .5, -1)], 1) == 1.


def verify_production_invalid_and_immutable(detection_ap):
    detections = [(0, .8, 0), (0, .4, -1)]
    original = detections.copy()
    detection_ap(detections, 1)
    assert detections == original
    invalid = [
        (detections, -1),
        (detections, 1.5),
        ([(-1, .5, 0)], 1),
        ([(0, np.nan, 0)], 1),
        ([(0, .5, -2)], 1),
        ([(0, .5)], 1),
        ([(0, .5, 0), (0, .4, 1)], 1),
        ([(0, .5, 0)], 0),
        ([3], 1),
        ([(0, 'bad', 0)], 1),
    ]
    for rows, total in invalid:
        try:
            detection_ap(rows, total)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid AP input accepted')


def verify_production_generated_false_positive_tail(detection_ap):
    @settings(max_examples=20, deadline=None, derandomize=True)
    @given(st.lists(st.floats(-20, .5, allow_nan=False), min_size=0, max_size=15))
    def check(scores):
        # Adding lower-scored false positives cannot change interpolated AP.
        original = [(0, 1., 0), (0, .75, -1), (0, .5, 1)]
        tail = [(0, score, -1) for score in scores]
        assert np.isclose(detection_ap(original+tail, 2), 5/6)
    check()


# %%
from torch_judge import evaluate
if not evaluate('0074_detection_ap_eval.py', detection_ap):
    raise SystemExit(1)
