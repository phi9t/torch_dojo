"""Reference and verifiers for max-pooling values and original coordinates."""

# %%
from hypothesis import given, settings, strategies as st
import numpy as np

# Drill 1: The local indices print 3, 2, 1, 0 for windows (0, 0), (0, 1),
# (1, 0), (1, 1), yet every window maps back to original `(1, 1)`. The flat
# `argmax` only says where the winner sits inside its own window; adding the
# window's top-left corner `(row * sh, col * sw)` recovers the one pixel that
# a backward pass must receive all four gradients at.

# Drill 2: It prints `index: 4  divmod by kw: (1, 1)  divmod by kh: (2, 0)`.
# A row-major flat index advances by the row length, which is the kernel
# width `kw`, so `index // kw, index % kw` gives the true `(1, 1)`; dividing by
# `kh` gives row 2, which is not even inside a two-row window. Square kernels
# hide this bug.

# Drill 3: It prints `row-major: (0, 1)  column-major: (1, 0)`. `argmax`
# returns the first maximum in scan order, so the tie rule depends on how the
# window is flattened. The contract asks for row-major, which is what
# `reshape` of a C-ordered slice gives.

# %%
def max_pool2d_with_coordinates(
    x: np.ndarray, kernel: tuple[int, int] = (2, 2), stride: tuple[int, int] = (2, 2)
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    if x.ndim != 4 or not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("need finite NCHW input")
    try:
        kh, kw = kernel
        sh, sw = stride
    except (TypeError, ValueError) as exc:
        raise ValueError("kernel and stride must be pairs") from exc
    if any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
           for n in (kh, kw, sh, sw)):
        raise ValueError("kernel and stride must be integer pairs")
    if min(kh, kw, sh, sw) < 1 or x.shape[2] < kh or x.shape[3] < kw:
        raise ValueError("need positive sizes and a valid window")
    B, C, H, W = x.shape
    OH, OW = (H - kh) // sh + 1, (W - kw) // sw + 1
    values = np.empty((B, C, OH, OW), dtype=x.dtype)
    coords = np.empty((B, C, OH, OW, 2), dtype=np.int64)
    for row in range(OH):
        for col in range(OW):
            patch = x[:, :, row * sh:row * sh + kh, col * sw:col * sw + kw]
            flat = patch.reshape(B, C, kh * kw)
            index = flat.argmax(axis=-1)
            values[:, :, row, col] = np.take_along_axis(flat, index[..., None], axis=-1)[..., 0]
            coords[:, :, row, col, 0] = row * sh + index // kw
            coords[:, :, row, col, 1] = col * sw + index % kw
    return values, coords


# %% [verifiers]
def verify_baseline_values_and_original_coordinates(max_pool2d_with_coordinates):
    x = np.array([[[[1., 9., 2., 3.], [4., 5., 6., 7.],
                    [8., 0., 3., 2.], [1., 2., 4., 5.]]]])
    values, coords = max_pool2d_with_coordinates(x, (2, 2), (2, 2))
    assert values.shape == (1, 1, 2, 2) and coords.shape == (1, 1, 2, 2, 2)
    assert np.array_equal(values[0, 0], [[9., 7.], [8., 5.]])
    assert np.array_equal(coords[0, 0], [[[0, 1], [1, 3]], [[2, 0], [3, 3]]])


def verify_followup_first_row_major_tie_and_negative_values(max_pool2d_with_coordinates):
    x = np.array([[[[-2., -2.], [-3., -4.]]]])
    values, coords = max_pool2d_with_coordinates(x)
    assert values[0, 0, 0, 0] == -2 and tuple(coords[0, 0, 0, 0]) == (0, 0)


def verify_followup_multiple_batches_channels_and_odd_edge(max_pool2d_with_coordinates):
    x = np.arange(2 * 3 * 3 * 5., dtype=float).reshape(2, 3, 3, 5)
    values, coords = max_pool2d_with_coordinates(x, (2, 2), (2, 2))
    assert values.shape == (2, 3, 1, 2)
    for b in range(2):
        for c in range(3):
            for j in range(2):
                r, q = coords[b, c, 0, j]
                assert values[b, c, 0, j] == x[b, c, r, q]
                assert (r, q) == (1, 2 * j + 1)


def verify_production_reject_no_window(max_pool2d_with_coordinates):
    try:
        max_pool2d_with_coordinates(np.ones((1, 1, 1, 2)), (2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("oversized kernel accepted")


def verify_production_generated_windows(max_pool2d_with_coordinates):
    @settings(max_examples=30, deadline=None, derandomize=True)
    @given(st.lists(st.integers(-8, 8), min_size=12, max_size=12))
    def property_check(values):
        x = np.asarray(values, dtype=float).reshape(1, 1, 3, 4)
        pooled, coords = max_pool2d_with_coordinates(x, (2, 2), (1, 1))
        for row in range(2):
            for col in range(3):
                patch = x[0, 0, row:row + 2, col:col + 2]
                local = np.unravel_index(np.argmax(patch), patch.shape)
                expected = (row + local[0], col + local[1])
                assert pooled[0, 0, row, col] == x[0, 0, expected[0], expected[1]]
                assert tuple(coords[0, 0, row, col]) == expected

    property_check()



def verify_followup_overlapping_winner_coordinate(max_pool2d_with_coordinates):
    x = np.array([[[[1., 2., 3.], [4., 9., 6.], [7., 8., 5.]]]])
    values, coords = max_pool2d_with_coordinates(x, (2, 2), (1, 1))
    assert np.array_equal(values, np.full((1, 1, 2, 2), 9.))
    assert np.array_equal(coords, np.broadcast_to([1, 1], (1, 1, 2, 2, 2)))


def verify_production_invalid_contract_and_unchanged_input(max_pool2d_with_coordinates):
    x = np.arange(16., dtype=float).reshape(1, 1, 4, 4)
    saved = x.copy()
    for bad, kernel, stride in [
        (x[0], (2, 2), (1, 1)), (x, (0, 2), (1, 1)),
        (x, (2, 2), (0, 1)), (x, (2, 2), (1.5, 1)),
        (np.full((1, 1, 2, 2), np.nan), (2, 2), (1, 1)),
    ]:
        try:
            max_pool2d_with_coordinates(bad, kernel, stride)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid pooling input accepted')
    max_pool2d_with_coordinates(x)
    assert np.array_equal(x, saved)

# %%
from torch_judge import evaluate
if not evaluate("0076_max_pool_coordinates_eval.py", max_pool2d_with_coordinates):
    raise SystemExit(1)
