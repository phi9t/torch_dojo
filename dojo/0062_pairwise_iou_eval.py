"""Reference and cases for Pairwise Box IoU."""

# %%
from hypothesis import given, settings, strategies as st
import numpy as np

# %%
# Drill 1: The probe prints `-1.0 -1.0 1.0 0.0`. Each overlap side is
# $\min(\text{maxes}) - \max(\text{mins})$, which goes negative when the boxes
# are apart along that axis: here both sides are $-1$. Their product is $+1$,
# so an unclamped formula reports a full unit of overlap (and IoU 1) for boxes
# that never touch. Clamping each side at zero first gives the true area 0.
#
# Drill 2: Continuous widths give $1 / (4 + 4 - 1) = 0.1429$. The pixel habit
# treats `[0, 2]` as the three integer cells 0, 1, 2, so each box becomes
# $3 \times 3 = 9$ and the overlap $2 \times 2 = 4$: $4 / (9 + 9 - 4) = 0.2857$,
# twice the right answer. The BEV plane is measured in metres, not pixel
# indices, so a side is just $x_{max} - x_{min}$; the +1 bias is largest for
# small boxes.
#
# Drill 3: `a[:, None, 0]` is $(2, 1)$ and `b[None, :, 0]` is $(1, 3)$, so
# broadcasting pairs every left box with every right box and prints `(2, 3)`:
# rows index `left`, columns index `right`. Without the new axes NumPy tries
# to pair the boxes element by element and raises `ValueError: operands could
# not be broadcast together with shapes (2,) (3,)`. When $N = M$ the same
# mistake runs silently and returns only the diagonal of the grid.

# %%
def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a), np.asarray(b)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != 4 or b.shape[1] != 4
            or a.dtype.kind not in 'fiu' or b.dtype.kind not in 'fiu'
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a[:, 2:] < a[:, :2]) or np.any(b[:, 2:] < b[:, :2])):
        raise ValueError("need valid finite xyxy boxes")
    a, b = a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
    result = np.empty((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    # Block over pairs: output is O(NM), but workspace is O(block^2),
    # rather than a full [N,M,2] corner tensor.
    block = 256
    for i in range(0, len(a), block):
        aa = a[i:i + block]
        for j in range(0, len(b), block):
            bb = b[j:j + block]
            width = np.maximum(0., np.minimum(aa[:, None, 2], bb[None, :, 2])
                               - np.maximum(aa[:, None, 0], bb[None, :, 0]))
            height = np.maximum(0., np.minimum(aa[:, None, 3], bb[None, :, 3])
                                - np.maximum(aa[:, None, 1], bb[None, :, 1]))
            intersection = width * height
            union = area_a[i:i + block, None] + area_b[None, j:j + block] - intersection
            result[i:i + block, j:j + block] = np.divide(
                intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return result


# %% [verifiers]
def verify_baseline_identity_overlap_touch(pairwise_iou):
    a = np.array([[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 3.0, 1.0]])
    b = np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    r = pairwise_iou(a, b)
    assert r.shape == (2, 2) and np.isclose(r[0, 0], 1.0) and np.isclose(r[0, 1], 1 / 7) and (r[1, 0] == 0)


def verify_followup_empty_and_zero_area(pairwise_iou):
    a = np.zeros((0, 4))
    b = np.array([[1.0, 1.0, 1.0, 1.0]])
    assert pairwise_iou(a, b).shape == (0, 1) and pairwise_iou(b, b)[0, 0] == 0


def verify_followup_transpose_symmetry(pairwise_iou):
    a = np.array([[0.0, 0.0, 2.0, 3.0]])
    b = np.array([[1.0, 1.0, 4.0, 5.0]])
    assert np.allclose(pairwise_iou(a, b), pairwise_iou(b, a).T)


def verify_production_invalid_boxes(pairwise_iou):
    for a, b in (
        (np.array([[2., 0., 1., 1.]]), np.zeros((0, 4))),
        (np.array([[0., 0., np.nan, 1.]]), np.zeros((0, 4))),
        (np.zeros((1, 3)), np.zeros((0, 4))),
        (np.zeros((1, 4)), np.zeros((0, 3))),
    ):
        try:
            pairwise_iou(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid boxes accepted')


def verify_production_generated_box_pairs(pairwise_iou):
    """Compare small grids against an independent scalar overlap calculation."""
    box = st.tuples(st.integers(-4, 4), st.integers(-4, 4),
                    st.integers(-4, 4), st.integers(-4, 4))

    @settings(max_examples=45, deadline=None, derandomize=True)
    @given(st.lists(box, max_size=4), st.lists(box, max_size=4))
    def property_check(left, right):
        def normalize(raw):
            x0, x1 = sorted((raw[0], raw[2]))
            y0, y1 = sorted((raw[1], raw[3]))
            return (x0, y0, x1, y1)

        a = np.asarray([normalize(item) for item in left], dtype=float).reshape(-1, 4)
        b = np.asarray([normalize(item) for item in right], dtype=float).reshape(-1, 4)
        result = pairwise_iou(a, b)
        assert result.shape == (len(a), len(b))
        for i, aa in enumerate(a):
            for j, bb in enumerate(b):
                overlap = max(0, min(aa[2], bb[2]) - max(aa[0], bb[0])) * \
                          max(0, min(aa[3], bb[3]) - max(aa[1], bb[1]))
                area_a = (aa[2] - aa[0]) * (aa[3] - aa[1])
                area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
                union = area_a + area_b - overlap
                assert np.isclose(result[i, j], overlap / union if union else 0)

    property_check()


def verify_production_invariance_and_large_output(pairwise_iou):
    a = np.array([[0., 0., 2., 3.], [1., 2., 4., 6.]])
    b = np.array([[1., 1., 3., 4.], [5., 5., 7., 7.]])
    original_a, original_b = a.copy(), b.copy()
    base = pairwise_iou(a, b)
    shift = np.array([7., -5., 7., -5.])
    assert np.allclose(pairwise_iou(a + shift, b + shift), base)
    assert np.allclose(pairwise_iou(3 * a, 3 * b), base)
    assert np.array_equal(a, original_a) and np.array_equal(b, original_b)
    many_a = np.repeat(a[:1], 513, axis=0)
    many_b = np.repeat(b[:1], 257, axis=0)
    result = pairwise_iou(many_a, many_b)
    assert result.shape == (513, 257) and np.allclose(result, base[0, 0])


# %%
from torch_judge import evaluate
if not evaluate("0062_pairwise_iou_eval.py", pairwise_iou):
    raise SystemExit(1)
