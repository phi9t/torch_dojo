"""Reference and staged checks for the pairwise IoU grid (workout contract)."""
# Drill answers: for disjoint boxes 1 - IoU is exactly 1 however far apart they
# are, so the loss is flat and IoU's gradient w.r.t. position is zero there.

# %%
import numpy as np


# %%
def pairwise_iou(boxes_a, boxes_b):
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])  # [N,M,2]
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.maximum(hi - lo, 0.0), axis=-1)  # clamp each side BEFORE the product
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0.0), axis=-1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0.0), axis=-1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


# %% [verifiers]
def verify_baseline_hand_checked_cases(pairwise_iou):
    a = np.array([[0., 0., 2., 2.]])
    assert np.isclose(pairwise_iou(a, a)[0, 0], 1.0)
    assert np.isclose(pairwise_iou(a, np.array([[1., 1., 3., 3.]]))[0, 0], 1 / 7)
    far = np.array([[5., 5., 6., 6.], [2., 0., 3., 2.], [2., 2., 3., 3.]])  # apart, edge, corner
    assert np.all(pairwise_iou(a, far) == 0)


def verify_baseline_zero_area_and_swap(pairwise_iou):
    a = np.array([[0., 0., 2., 2.], [1., 1., 1., 1.]])
    b = np.array([[1., 1., 3., 3.], [1., 1., 1., 1.], [0., 0., 4., 1.]])
    r = pairwise_iou(a, b)
    assert r.shape == (2, 3) and np.all(r[1] == 0) and np.all(r[:, 1] == 0)
    assert np.isfinite(r).all() and np.allclose(r, pairwise_iou(b, a).T)


def verify_followup_empty_inputs_keep_shape(pairwise_iou):
    some = np.array([[0., 0., 1., 1.], [0., 0., 2., 2.]])
    assert pairwise_iou(np.zeros((0, 4)), some).shape == (0, 2)
    assert pairwise_iou(some, np.zeros((0, 4))).shape == (2, 0)
    assert pairwise_iou(np.zeros((0, 4)), np.zeros((0, 4))).shape == (0, 0)


def verify_followup_matches_scalar_reference_on_random_boxes(pairwise_iou):
    rng = np.random.default_rng(0)
    def boxes(n):
        p = rng.uniform(-3, 3, size=(n, 2)); s = rng.uniform(0, 3, size=(n, 2))
        return np.concatenate([p, p + s], axis=1)
    a, b = boxes(7), boxes(5)
    r = pairwise_iou(a, b)
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            w = max(0, min(p[2], q[2]) - max(p[0], q[0])); h = max(0, min(p[3], q[3]) - max(p[1], q[1]))
            u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - w * h
            assert np.isclose(r[i, j], w * h / u if u > 0 else 0.0)
    assert (r >= 0).all() and (r <= 1 + 1e-12).all()


def verify_production_far_apart_boxes_and_purity(pairwise_iou):
    a = np.array([[0., 0., 1., 1.]]); b = np.array([[2., 2., 3., 3.], [-9., -9., -8., -8.]])
    a0, b0 = a.copy(), b.copy()
    assert np.all(pairwise_iou(a, b) == 0)  # two negative sides must NOT multiply to a positive overlap
    assert np.array_equal(a, a0) and np.array_equal(b, b0)
    assert pairwise_iou(np.array([[0, 0, 2, 2]]), np.array([[1, 1, 3, 3]])).dtype == np.float64


def verify_production_vectorized_on_a_large_grid(pairwise_iou):
    import time
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 50, size=(1500, 2)); a = np.concatenate([p, p + rng.uniform(0.5, 3, size=(1500, 2))], 1)
    q = rng.uniform(0, 50, size=(1500, 2)); b = np.concatenate([q, q + rng.uniform(0.5, 3, size=(1500, 2))], 1)
    start = time.perf_counter()
    r = pairwise_iou(a, b)
    assert r.shape == (1500, 1500) and time.perf_counter() - start < 5.0, "looping over pairs is too slow"


# %%
from torch_judge import evaluate
if not evaluate("0128_box_iou_grid_eval.py", pairwise_iou):
    raise SystemExit(1)
